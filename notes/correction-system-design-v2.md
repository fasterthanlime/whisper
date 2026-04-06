# Dictation Correction System: Design Memo v2

**Date:** 2026-04-05

## Premise

The next judge should be trained and evaluated first as an offline sparse contextual
linear model before resuming aggressive online updates, because we need to separate
feature-space gains from update-mechanic instability.

Everything below serves that goal.

---

## 1. ASR-Derived Signals: What We Have, What We Can Extract

The decoder currently computes logits at every step (`generate.rs:47`) and
immediately discards everything except the argmax token. No beam search, no
sampling, no alternatives.

### Signals available now (zero effort)

| Signal | Source | Already exposed? |
|--------|--------|-----------------|
| Word-level timestamps | Forced aligner (`ForcedAlignItem`) | Yes — `AlignedWord { word, start, end }` |
| Punctuation / capitalization | In decoded token text | Yes — Qwen3 ASR produces cased+punctuated output |
| Mel spectrogram per span | `MelExtractor` + aligner timestamps for frame boundaries | In memory during inference, just needs slicing |

### Signals requiring small work (< 2 days)

| Signal | What | How to expose | Latency cost | Leverage |
|--------|------|--------------|-------------|----------|
| **Per-token logprob** | log P(chosen\_token) at each decode step | After `model.step()` returns logits, compute `log_softmax`, extract value at argmax index. Return `Vec<(i32, f32)>` from `prefill_and_decode`. | ~0 (softmax on the argmax slice is negligible) | **HIGH** |
| **Top1–top2 margin** | Gap between best and second-best logprob | Same logits, partial sort for top-2 | Negligible | **HIGH** |
| **Top-k alternatives** | Top 5–10 token IDs + logprobs per step | `topk` op on 151936-dim logits vector | ~0 extra | HIGH (enables alternative transcript fragments later) |
| **Streaming revision count** | How many times a token was revised before commit | Counter array in `Session`, bump on each rollback | Negligible | MEDIUM |

### Signals requiring medium work (2–5 days)

| Signal | What | How to expose | Latency cost | Leverage |
|--------|------|--------------|-------------|----------|
| **Constrained re-decode** | Force-decode candidate surface tokens from a KV-cache snapshot at span start; compare summed logprob to original span | Clone KV cache (`Vec<Option<Array>>` per layer), force-feed candidate tokens, sum logprobs. Normalize by token count + small continuation window. | N\_tokens × M\_candidates extra decode steps. For 3-token span, 5 candidates: ~15 steps, ~7ms | **VERY HIGH** |
| **Entropy per step** | H = −Σ p log p over vocab | Full softmax over 151936 vocab per step | ~0.3–0.5ms per token on Apple Silicon | MEDIUM (only compute for spans that pass retrieval) |
| **Hidden states (last layer)** | 1024-dim vector per token from final decoder layer | Return `hidden` before `lm_head` in `model.rs:87` | 4KB per token stored | MEDIUM (needs projection for linear judge) |
| **Audio encoder features per span** | 1024-dim vectors from encoder output, one per ~80ms | Already computed via `encode_audio()`; retain per span using aligner timestamps | Already in memory | MEDIUM |

### Signals requiring large work (> 1 week)

| Signal | Effort | Leverage | Recommendation |
|--------|--------|----------|----------------|
| N-best hypotheses (beam search) | ~200–400 lines, KV cache forking, 2–5× memory | LOW — per-token signals are cheaper and give most of the same info | Skip |
| Full attention weights | Propagate from all 28 layers × 16 heads, ~90MB for L=50 | LOW — too expensive, noisy | Skip |
| Encoder-probe phone extraction | Linear probe on frozen encoder features, needs phone-labeled data | MEDIUM — bounded experiment | Defer to Step 7 |

---

## 2. Top 5 Highest-Leverage Signals (Ranked)

### 1. Per-token logprobs + margin

**Cost:** ~20 lines in `generate.rs`.

A span where ASR was confident (logprob > −0.3, margin > 2.0) is unlikely to be a
misrecognition. A span where ASR was uncertain (logprob < −2.0, margin < 0.5) is
exactly where correction should be aggressive.

Features for the judge: `span_mean_logprob`, `span_min_logprob`, `span_mean_margin`,
`span_min_margin`.

**Expected impact:** 10–20% reduction in false positives (corrections on spans the ASR
already got right).

**Falsification:** Compute logprob distributions for gold-correct vs. gold-incorrect
spans in the eval set. If AUC < 0.60, logprobs from this model are not informative.

### 2. Sparse hashed context features

**Cost:** ~2–3 days (new sparse FTRL implementation + feature wiring).

The judge currently has zero sentence context. Literal sparse lexical features —
hashed tokens, bigrams, and candidate×context crosses — will let the model learn
term-specific context patterns. "The serde crate" → accept. "The sir day of the
week" → reject.

**Expected impact:** 5–10% improvement in judge precision, concentrated on the
false-positive cases that are most annoying.

**Falsification:** Inspect current eval errors. If none are cases where surrounding
words would disambiguate, context features won't help yet and you need more diverse
eval data.

### 3. Memory counters (accept/reject/recency per term)

**Cost:** ~1 day (HashMap + persistence).

The judge currently has no memory of past decisions about a term. Even simple counts
(how many times was "serde" accepted, how many times rejected) give the judge a strong
prior.

**Expected impact:** 3–5% precision improvement, mostly on terms the user has
interacted with before.

**Falsification:** If the eval set doesn't re-encounter the same terms across
sessions, memory features have nothing to learn from.

### 4. Constrained re-decode (candidate logprob comparison)

**Cost:** 3–5 days (KV cache cloning, forced decode, logprob comparison).

Asks: "if I force the decoder to output 'reqwest' instead of 'request', what logprob
does the model assign?" If the candidate scores higher than the original, it's
acoustically plausible. This is the strongest *acoustic* verification signal we can
get from the existing model.

**Caution:** Compare *normalized* logprob over the forced span, possibly plus a small
(2–3 token) continuation window. Do NOT compare total-sequence scores — tokenization
length differences and decoder LM effects will swamp the comparison.

**Expected impact:** 5–15% improvement on ambiguous cases where two candidates have
similar phonetic scores.

**Falsification:** Re-decode known-wrong candidates. If they score similarly to correct
candidates (decoder LM dominates over acoustics), this signal collapses.

### 5. Streaming revision count

**Cost:** ~20 lines in `Session`.

Tokens revised multiple times during streaming before committing are tokens the model
was temporally unstable about. Orthogonal to logprobs.

**Expected impact:** 2–3% incremental improvement in span proposal.

**Falsification:** If `rollback_tokens=5` produces too little variation (most tokens
revised 0 or 1 times), the signal has no entropy.

---

## 3. Judge Feature Schema

### Architecture change: replace linfa-ftrl with custom sparse FTRL

`linfa-ftrl` only accepts dense `Array2<f64>`. To support a hashed sparse feature
space, we need a custom FTRL implementation. The algorithm is simple — ~80 lines of
Rust:

```
For each feature i with non-zero value x_i:
    σ_i = (sqrt(n_i + g_i²) - sqrt(n_i)) / α
    z_i += g_i - σ_i * w_i
    n_i += g_i²

    if |z_i| ≤ λ₁:
        w_i = 0
    else:
        w_i = -(z_i - sign(z_i) * λ₁) / ((β + sqrt(n_i)) / α + λ₂)
```

Where `z`, `n` are per-feature accumulators stored in `HashMap<u64, (f64, f64)>`.
Only entries with non-zero updates are stored. This is what Vowpal Wabbit and every
production ad system uses.

**Hash space:** 2^14 = 16384 buckets. Hash function: FNV-1a on the feature name
string, mod 2^14. One-hot: only the active bucket has value 1.0.

**Weight storage:** `HashMap<u64, (f64, f64, f64)>` for `(z_i, n_i, w_i)`.
Typical model size after training: ~500–2000 active features × 24 bytes = 12–48KB.

### Feature groups

#### A. Dense phonetic/structural features (28 features, unchanged)

Keep all existing features from `FEATURE_NAMES`. These are dense (every candidate
has values for all of them). Pass them through to the sparse FTRL as feature indices
0–27 with their float values.

#### B. ASR uncertainty features (4 dense features)

| Index | Feature | Encoding |
|-------|---------|----------|
| 28 | `span_mean_logprob` | raw / 5.0 |
| 29 | `span_min_logprob` | raw / 5.0 |
| 30 | `span_mean_margin` | raw / 5.0 |
| 31 | `span_min_margin` | raw / 5.0 |

#### C. Sparse context features (hashed into 2^14 space)

Each feature is a string hashed to a bucket index in range [1000, 1000 + 2^14).
Offset of 1000 avoids collision with the dense features.

**Literal token features:**

| Feature name string | Example |
|--------------------|---------|
| `L1={lowercased left-1 token}` | `L1=the` |
| `R1={lowercased right-1 token}` | `R1=crate` |
| `L2={left-2 token}_{left-1 token}` | `L2=import_the` |
| `R2={right-1 token}_{right-2 token}` | `R2=crate_handles` |

**Candidate identity:**

| Feature name string | Example |
|--------------------|---------|
| `TERM={candidate term}` | `TERM=serde` |

**Candidate × context crosses:**

| Feature name string | Example |
|--------------------|---------|
| `TERM={term}\|L1={left-1}` | `TERM=serde\|L1=the` |
| `TERM={term}\|R1={right-1}` | `TERM=serde\|R1=crate` |
| `TERM={term}\|L2={left bigram}` | `TERM=serde\|L2=import_the` |
| `TERM={term}\|R2={right bigram}` | `TERM=serde\|R2=crate_handles` |

**Context type flags (also hashed, but predictable bucket):**

| Feature name string | When active |
|--------------------|------------|
| `CTX=code` | `()`, `{}`, `::`, `.`, `_` within ±10 chars |
| `CTX=prose` | No code markers |
| `CTX=list` | Line starts with `-`, `*`, number |
| `CTX=sent_start` | Span at sentence start |

**App/source flag:**

| Feature name string | Example |
|--------------------|---------|
| `APP={app identifier}` | `APP=xcode`, `APP=slack` |
| `TERM={term}\|APP={app}` | `TERM=serde\|APP=terminal` |

All sparse features have value 1.0 when active, 0.0 when absent. Total active
sparse features per candidate: ~12–18.

#### D. Memory features (6 dense features)

| Index | Feature | Encoding |
|-------|---------|----------|
| 32 | `candidate_prior_accept_count` | log(1 + count) / 3.0 |
| 33 | `candidate_prior_reject_count` | log(1 + count) / 3.0 |
| 34 | `candidate_total_count` | log(1 + accept + reject) / 3.0 |
| 35 | `candidate_recent_accept_count` | accepts in last 7 days, log(1 + count) / 3.0 |
| 36 | `candidate_session_recency` | 1.0 if last 5 min, 0.5 if last 30 min, 0.0 else |
| 37 | `span_text_prior_correct_count` | log(1 + count) / 3.0 |

No ratio features. Pass raw counts, let the model learn.

#### E. Re-decode features (2 dense features, added in Step 5)

| Index | Feature | Encoding |
|-------|---------|----------|
| 38 | `redecode_logprob_delta` | (candidate\_lp − original\_lp) / 5.0, normalized by token count |
| 39 | `redecode_candidate_logprob` | candidate forced logprob / 5.0, normalized by token count |

### Total

- 28 dense phonetic/structural (existing)
- 4 dense ASR uncertainty
- 6 dense memory
- 2 dense re-decode (later)
- ~12–18 active sparse hashed features per candidate (from a 2^14 space)

### Data flow change

`build_examples()` currently receives `(TranscriptSpan, Vec<(CandidateFeatureRow, IdentifierFlags)>)`.

New interface:

```rust
struct SpanContext {
    left_tokens: Vec<String>,    // 1–2 words left of span, lowercased
    right_tokens: Vec<String>,   // 1–2 words right of span, lowercased
    code_like: bool,
    prose_like: bool,
    list_like: bool,
    sentence_start: bool,
    app_id: Option<String>,
    sentence_text: String,
}

struct SpanUncertainty {
    mean_logprob: f32,
    min_logprob: f32,
    mean_margin: f32,
    min_margin: f32,
}

struct CandidateMemory {
    accept_count: u32,
    reject_count: u32,
    recent_accept_count: u32,  // last 7 days
    last_used: Option<Instant>,
}

// Optional, added in Step 5
struct RedecodeResult {
    logprob_delta: f32,       // candidate_lp - original_lp, per-token normalized
    candidate_logprob: f32,   // absolute, per-token normalized
}
```

---

## 4. IPA-from-Audio Path: Assessment

### What it would buy

Audio-derived phones capture what the speaker actually said, before ASR word-level
decisions distorted the phonetics. This matters when:

- **ASR word-choice drift:** ASR picked "sir day" for /sɜːrdeɪ/. eSpeak("sir day")
  gives different IPA than the actual speech. Audio phones would match "serde" better.
- **Multi-word splits:** ASR split "reqwest" into "req west." Audio phones give a
  continuous sequence without the artificial boundary.
- **Technical OOV:** ASR maps unknown sounds to known words. Audio phones preserve the
  original sounds.

### Where it does NOT help

- Acronyms/spelling (ASR usually gets individual letters right)
- Cases where ASR is already correct
- Near-neighbor disambiguation where both candidates are phonetically close (still
  need context/judge)

### Where in the pipeline

| Stage | Help? | Magnitude |
|-------|-------|-----------|
| Retrieval | Yes — query index with truer phones | Significant (main value) |
| Verification | Yes — score against truer phones | Moderate |
| Span proposal | Yes — divergence between audio and transcript phones flags errors | Moderate |
| Judge | Small — incremental feature over existing scores | Small |
| Rendering | None | — |

### Engineering options

**Option A: Full CTC phone recognizer.** Fine-tune wav2vec2-base on TIMIT +
CommonVoice. Port to MLX. ~2–3 weeks, ~90MB model.

**Option B: Linear probe on Qwen3 encoder features.** The audio encoder already
produces 1024-dim per-frame features that encode phonetic information. Train a small
linear layer (1024 → ~80 phones) on frozen encoder features using TIMIT + forced
alignment. ~3–5 days, no additional model at inference.

### Recommendation

**Not mainline. Bounded experiment after Steps 1–5.**

Constrained re-decode (Step 5) captures most of the same value — it asks "does the
candidate fit the audio?" using the existing model. If re-decode proves insufficient,
try the encoder probe (Option B) as the cheapest next experiment.

**Falsification for the whole path:** If retrieval failures in the eval set are
primarily due to vocabulary coverage (term not in index) rather than phonetic distance
(term in index but IPA too far), audio-derived phones won't help.

---

## 5. User Vocabulary Onboarding

### Minimal workflow (ship first)

1. User types or pastes the term (e.g., "Bevy")
2. System auto-generates:
   - Spoken form via identifier splitting
   - IPA via eSpeak
   - Identifier flags
3. System shows spoken form + phonetic transcription. User can tap to hear TTS
   playback. If wrong, user types a correction.
4. **One question:** "Say a sentence where you'd use this word." User speaks one
   sentence. System:
   - Transcribes with ASR
   - Identifies span where term should appear
   - Records ASR output for that span as a confusion surface
   - Records sentence as a positive context example
5. Auto-generate phonetic neighbor negatives: eSpeak finds real English words close to
   the new term. Create synthetic weak-negative entries (these words should NOT be
   corrected to this term by default).
6. Term + confusion surface enter the index immediately.
7. Memory initialized: `accept_count=1, reject_count=0, session_recency=now`.

**User time: ~15 seconds.**

### Richer workflow (opt-in "Improve recognition" button)

Everything in minimal, plus:

1. **Pronunciation review:** Show IPA, play TTS. User confirms or re-records.
2. **Two additional prompted sentences:**
   - "Say a sentence using [term] in a technical context"
   - "Say the term in a list with other terms"
3. **Category tag** (optional): "Programming," "Science," "Brand name," "Acronym,"
   "Other." Becomes the `APP=` feature.
4. **LLM-generated context exemplars** (background, not blocking): Generate 5–10
   synthetic sentences for offline feature mining.

**User time: ~60 seconds.**

### What NOT to do

- Do not require a "negative sentence" at onboarding. Too cognitively expensive.
- Do not force a training wizard before the term is usable.
- Do not require pronunciation recording (TTS + eSpeak is good enough for most terms).
- Do not generate LLM examples synchronously at add time.

### Tradeoffs

| Aspect | Minimal | Richer |
|--------|---------|--------|
| Friction | ~15s | ~60s |
| False positive risk | Higher — one confusion surface | Lower — multiple surfaces + negatives |
| Judge cold-start | Relies on phonetic scores + auto negatives | 3+ real examples |
| Implementation cost | Small | Medium (UI for prompts, background LLM) |

---

## 6. Training Data from User Actions

### Events to log

| Event | Data captured |
|-------|--------------|
| `correction_accepted` | span, candidate, all candidate feature rows, judge scores, ASR logprobs, left/right context, sentence, timestamp |
| `correction_rejected` | Same as above |
| `manual_correction` | original span, corrected text, sentence, ASR logprobs, timestamp |
| `term_added` | term, spoken form, IPA, source, timestamp |
| `session_start/end` | app context, duration, correction count |

### Events → training examples

**`correction_accepted` (user picks candidate X):**
- Positive: feature row for X, target=true
- Hard negatives: feature rows for every other candidate + keep-original, target=false

**`correction_rejected` (user keeps original):**
- Positive: keep-original, target=true
- Hard negatives: all candidate feature rows, target=false

**`manual_correction` (user fixes something we didn't suggest):**
1. Run retrieval on original span for the corrected term
2. If found: treat as `correction_accepted` for that candidate
3. If not found: the original span text becomes a confusion surface for the corrected
   term; prompt user to add the term if not in vocabulary

### Avoiding data poisoning

1. **Minimum example buffer:** Don't update model weights from a single event. Buffer
   last 3–5 events per candidate term. Only train when buffer reaches 3+ consistent
   examples. One weird correction doesn't shift weights.
2. **Confidence-weighted learning rate:** Lower α (0.1 vs. 0.5) for events where the
   judge was already highly confident (> 0.8). Contradicting a confident prediction is
   more likely noise.
3. **Memory is separate from weights:** Counters update immediately but are just
   inputs. A single event shifts one feature slightly, not the decision boundary.
4. **Decay:** All counters × 0.99 at session start. Events older than 30 days get
   half weight in offline replay.
5. **Per-user isolation:** All state is local. No cross-user contamination.

### Update timing

| Signal | When | How |
|--------|------|-----|
| Memory counters | Immediately on every event | HashMap increment, persist every 10 events |
| Confusion surfaces | Immediately on manual correction | Insert into phonetic index |
| Vocabulary index | Immediately on term\_added | Rebuild affected entries |
| Model weights | **Deferred** — batch after 3+ consistent events per term | Sparse FTRL `teach_choice` on buffered examples |

### Event log format

Append-only JSONL, one file per user. Cap at 10,000 events (~2–5MB). Full feature
rows stored for offline replay. This is the source of truth for any future offline
retraining.

---

## 7. Implementation Order

### Step 1: Expose logprobs + margin from decoder

Modify `prefill_and_decode()` in `generate.rs` to return token logprobs and margin
alongside token IDs. Propagate through `Session` to attach to committed tokens. Add
`mean_logprob`, `min_logprob`, `mean_margin`, `min_margin` to `TranscriptSpan`.

- **Why first:** Cheapest change, highest leverage, unblocks Steps 2 and 4.
- **Metric:** AUC of logprob as a classifier for "is this span correct?" on the eval
  set. Need AUC > 0.60 to proceed.
- **Falsification:** If AUC ≈ 0.50, logprobs from this model are uninformative and we
  skip ASR uncertainty features.
- **Effort:** 1–2 days.

### Step 2: Implement sparse FTRL judge with context features

Replace `linfa-ftrl` with a custom sparse FTRL (~80 lines). Implement the full
feature schema from Section 3: 32 dense features (28 existing + 4 ASR uncertainty) +
hashed sparse context features in a 2^14 space.

Context features to implement:
- `L1=`, `R1=`, `L2=`, `R2=` (literal token/bigram features)
- `TERM=` (candidate identity)
- `TERM=|L1=`, `TERM=|R1=`, `TERM=|L2=`, `TERM=|R2=` (crosses)
- `CTX=code`, `CTX=prose`, `CTX=list`, `CTX=sent_start`
- `APP=` if available

- **Why second:** The judge's biggest gap is zero context. Sparse hashed features are
  the right way to add context to a linear model.
- **Metric:** Eval precision and recall with the new feature set. Train offline first
  on accumulated eval examples before enabling online updates.
- **Falsification:** If offline judge accuracy doesn't improve over the 28-feature
  baseline, context features aren't helping and the problem is elsewhere.
- **Effort:** 3–4 days (custom FTRL + feature wiring + offline eval).

### Step 3: Add memory counters + event logging

Implement per-term counters (`accept_count`, `reject_count`, `recent_accept_count`,
`last_used`), per-span-text counters, session recency tracking. Add 6 memory features
to the judge. Implement `CorrectionEvent` append-only JSONL logging.

- **Why third:** Memory is the immediate learning path. Counters update instantly;
  weight updates can wait.
- **Metric:** After N user corrections, does the judge improve for the corrected
  terms? Simulate with event replay on the eval set.
- **Falsification:** If memory features don't improve judge accuracy beyond the
  context-only model, skip them.
- **Effort:** 2–3 days.

### Step 4: Offline judge baseline with expanded features

Train the sparse FTRL judge offline on all available eval data + any accumulated
correction events. Evaluate systematically. This is the checkpoint before re-enabling
online updates.

**This step is critical.** We need to know what the expanded feature space buys us
in a controlled setting before letting online updates loose.

- **Why fourth:** Separates feature-space gains from update-mechanic instability.
- **Metric:** Precision/recall/F1 on held-out eval cases compared to current
  28-feature baseline.
- **Falsification:** If the expanded model doesn't beat the baseline offline, online
  updates won't magically fix it.
- **Effort:** 1–2 days (mostly eval infrastructure).

### Step 5: Re-enable cautious online updates

With the expanded feature space validated offline, re-enable `teach_choice` with:
- Buffered updates (3+ events per term before training)
- Lower learning rate for high-confidence contradictions
- Weight snapshot every 50 updates with rollback if eval degrades
- Counter decay on session start

- **Why fifth:** Online learning is the product differentiator, but it must be stable.
- **Metric:** Judge accuracy after 100 simulated user corrections, compared to the
  offline baseline. Should not degrade.
- **Falsification:** If online updates cause accuracy oscillation (alternating
  improvement and degradation), the update mechanics need further damping.
- **Effort:** 1–2 days.

### Step 6: Constrained re-decode experiment

Implement KV cache cloning at span-start. Force-decode candidate tokens, sum per-token
logprobs, normalize by length. Add 2 re-decode features to the judge. Evaluate
offline.

Compare *per-token normalized* logprobs over the forced span plus a 2–3 token
continuation window. Do not compare total-sequence scores.

- **Why sixth:** Most powerful acoustic signal, but most complex. Only worth the
  investment after cheaper signals are in and validated.
- **Metric:** On cases where two candidates have similar phonetic scores, does
  re-decode logprob reliably prefer the correct one?
- **Falsification:** If re-decode scores correlate with English word frequency rather
  than acoustic fit (test by zeroing audio embeddings and checking if scores barely
  change), the decoder LM dominates and this signal is not acoustic.
- **Effort:** 3–5 days.

### Step 7: Vocabulary onboarding flow

Build the minimal "say a sentence" onboarding flow. Wire up confusion surface
extraction, auto-generated phonetic neighbor negatives, memory initialization, event
logging.

- **Why seventh:** Onboarding is most valuable once the judge can consume context
  and memory properly.
- **Metric:** For newly added terms, correct on first real use after onboarding.
- **Falsification:** If the single sentence doesn't produce a usable confusion surface
  (ASR transcribes the term correctly, no error to learn from), the flow needs
  redesign.
- **Effort:** 3–5 days.

### Step 8: Encoder-probe phone extraction (bounded experiment)

Extract Qwen3 encoder features for eval spans. Train linear probe (1024 → ~80 phones)
on TIMIT. Compare probe-derived phones to eSpeak-derived phones for retrieval.

- **Why last:** Speculative. Only worth doing if Steps 1–6 haven't closed the gap.
- **Metric:** Retrieval recall for cases that currently fail due to ASR word-choice
  drift.
- **Falsification:** Probe phone accuracy < 70% → encoder features are too entangled.
- **Effort:** 3–5 days for experiment; 1–2 weeks for integration if successful.

---

## 8. Open Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Qwen3 logprobs don't correlate with correctness (model confidently wrong on OOV) | High | Test empirically in Step 1 before building on this. Margin may be more robust than raw logprob. |
| Constrained re-decode dominated by decoder LM, not acoustics | Medium | Test by comparing scores with/without audio embeddings. Defer to Step 6 so we can evaluate whether it adds value on top of logprob+context. |
| Custom sparse FTRL introduces bugs vs. tested linfa-ftrl | Medium | Write thorough unit tests. Validate offline that custom FTRL reproduces linfa-ftrl results on the 28-feature dense case before expanding. |
| FTRL with 2^14 sparse space converges too slowly from few corrections | Medium | Base weights (dense features 0–31) carry most load. Sparse features start at 0, activate gradually. L1 zeros out noise. Monitor convergence. |
| Memory counter feedback loops (auto-corrections boost counts, leading to more auto-corrections) | Medium | Decay counters. Require 3+ consistent events. On rejection of previously-accepted correction, reset `accept_count` to max(count−3, 0). |
| Onboarding sentence doesn't produce useful confusion surface | Medium | If ASR gets the term right on first try during onboarding, that's actually good — store it as a positive example and note that no confusion surface was needed. The auto-generated phonetic neighbors provide the negative signal. |
| Per-user event log grows unbounded | Low | Cap at 10,000 events. Oldest events dropped. ~2–5MB per user. |
