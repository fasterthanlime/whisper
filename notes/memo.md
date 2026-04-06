Now I have a thorough understanding of the entire system. Let me write the memo.

---

# Dictation Correction System: Design & Research Memo

## 1. Inventory of Unused ASR-Derived Signals from Qwen3 ASR / MLX

The current pipeline extracts exactly one thing from the ASR decoder: the argmax token at each step (`generate.rs:212-216`). Every other signal is discarded. Here is what exists or could exist:

| Signal | What it is | Can we get it now? | Engineering effort | Stable for product? | Where it helps | Latency/memory cost | Leverage |
|--------|-----------|-------------------|-------------------|---------------------|---------------|---------------------|----------|
| **Per-token logprobs** | log P(chosen_token) at each decode step | Logits exist at `generate.rs:47`, just need log_softmax + extract | **Small**: ~20 lines in generate.rs, return `Vec<f32>` alongside tokens | Yes, deterministic given model state | Judge: ASR confidence → abstention; span proposal: flag low-confidence regions | ~0 latency (softmax is already implicit in argmax comparison), 4 bytes per token | **HIGH** |
| **Top-k token alternatives** | Top 5-10 token IDs + logprobs at each step | Same logits, apply topk after softmax | **Small**: topk op on existing logits array, ~30 lines | Yes | Judge: "was this a close call?" margin; retrieval: generate alternative transcript fragments | ~0 extra compute (topk on 151936-dim is cheap), ~50 bytes/token for top-10 | **HIGH** |
| **Entropy per step** | H = -sum(p * log p) over vocab at each position | Requires full softmax over 151936 vocab | **Small** in code, **medium** in cost: full softmax is ~600KB per step | Noisy for individual tokens, stable when averaged over spans | Judge: span-level uncertainty → flag suspicious regions; abstention | Full softmax adds ~0.5ms per token on Apple Silicon | **MEDIUM** |
| **Margin (top1 - top2 logprob)** | Gap between best and second-best token | Logits + partial sort (top-2) | **Tiny**: 2 extra lines | Yes, very stable | Judge: low margin = model was unsure = better candidate for correction | Negligible | **HIGH** |
| **N-best hypotheses** | Alternative full transcripts | Requires beam search in generate.rs, currently greedy-only | **Large**: beam search with KV cache forking, ~200-400 lines. MLX cache is layer-by-layer `Vec<Option<Array>>` — would need per-beam copies | Medium — beam hypotheses are correlated, diversity is limited | Retrieval: try correction on each hypothesis; judge: agreement across hypotheses | 2-5x memory and latency per beam (KV cache per beam) | **LOW** for our use case — the per-token signals give most of the same information much cheaper |
| **Local constrained re-decode** | Force-decode a span with a candidate term's tokens, compare logprob to original | Have `model.step()` with cache; would re-decode from a cached prefix | **Medium**: need to snapshot KV cache at span start, re-decode with forced tokens, compare total logprob | Yes — this is just conditional likelihood | Judge verification: "does the candidate term fit this acoustic context better than what ASR said?" | Re-decode N tokens × M candidates. For 3-token span, 5 candidates: 15 extra steps, ~7ms | **VERY HIGH** — this is the single most informative signal for context-sensitive acceptance |
| **Hidden states (last layer)** | 1024-dim vector per token from decoder layer 28 output | Available at `model.rs:87` (the `hidden` variable before lm_head) | **Small**: return alongside logits, ~15 lines | Yes, but high-dimensional — needs projection for feature use | Judge: dense context embedding; span proposal: anomaly detection via cosine distance to neighbors | 4KB per token (1024 × f32). Storing for a sentence of 20 tokens: 80KB | **MEDIUM** — useful but needs a dimensionality reduction step before the linear judge can use it |
| **Attention weights** | Softmax attention scores from each decoder layer | Computed at `decoder.rs:191` but discarded immediately | **Medium**: need to return from `forward_attn`, propagate through layers, increases memory significantly | Noisy per-head, more stable when averaged | Span proposal: attention diffusion = model confused about source; alignment verification | 28 layers × 16 heads × L² per layer. For L=50: ~90MB. Impractical to store all. | **LOW** — too expensive and noisy to be practical. The information is better captured by logprobs. |
| **Word-level timestamps** | Start/end time per word from forced aligner | Already implemented in `forced_aligner.rs`, used in `Session` commit-and-rotate | **Zero**: already exposed as `AlignedWord { word, start, end }` | Yes, after LIS smoothing | Span proposal: timing anomalies (abnormally short/long words suggest error); surface rendering | Already computed | **LOW** for judge (timing is weakly correlated with errors), **MEDIUM** for span gating |
| **Audio encoder features** | 1024-dim vectors from encoder output, one per ~80ms audio frame | Available at `model.rs:61-63` via `encode_audio()` | **Small**: already computed, just need to retain per-span | Yes | Retrieval: acoustic embedding similarity between span audio and candidate pronunciation; phone-level analysis | Already computed during inference. Storing for a 1s span: ~12 vectors × 4KB = 48KB | **MEDIUM** — useful if we build acoustic matching, but not a quick win |
| **Mel spectrogram per span** | Raw 128-bin × N-frame mel for a time-aligned span | Already computed in `mel.rs`, and timestamps from aligner give us frame boundaries | **Small**: slice mel array by frame indices from aligner timestamps | Yes | IPA-from-audio path; acoustic analysis | Already in memory during decode | **LOW** on its own (needs a consumer model) |
| **Tokenization boundaries** | Which characters map to which BPE tokens; token ID sequence | Available from the tokenizer (HF `tokenizers` crate) | **Tiny**: tokenizer already loaded in Engine | Yes | Code detection: unusual token density = identifier-like text; span proposal: align correction boundaries to BPE boundaries | Negligible | **LOW** — already partially captured by identifier flags |
| **Punctuation/capitalization predictions** | Model's predicted punctuation and casing in token sequence | Already in the decoded tokens (Qwen3 ASR produces cased, punctuated text) | **Zero**: just inspect decoded text | Yes | Context flags: sentence boundaries, code-vs-prose | Already available | **MEDIUM** — cheap source of local context type |
| **Streaming revision count** | How many times a token was revised during streaming before committing | `Session` has `rollback_tokens` mechanism; could count per-position revisions | **Small**: counter array, ~20 lines in `Session` | Yes | Span proposal: frequently-revised tokens are uncertain | Negligible | **MEDIUM** |
| **Chunk boundaries** | Which mel frames / tokens belong to which streaming chunk | `Session` tracks `chunk_count` and `committed_audio_offset` | **Small**: tag tokens with chunk index | Yes | Span proposal: corrections near chunk boundaries are more likely due to context split | Negligible | **LOW** |

### Summary of availability

**Available now (zero effort):** word-level timestamps, punctuation/capitalization, mel spectrogram (in memory during inference)

**Small effort (< 1 day):** per-token logprobs, top-k alternatives, margin, streaming revision count, chunk boundary tags

**Medium effort (1-3 days):** entropy, constrained re-decode, hidden state extraction, audio encoder features per span

**Large effort (> 1 week):** beam search / N-best, full attention weight extraction

---

## 2. Top 5 Highest-Leverage Signals to Add Next

Ranked by expected impact on the actual remaining problem (context-sensitive accept/reject, reducing false positives, improving abstention):

### 1. Per-token logprobs + margin (top1 - top2)

**Why first:** This is the cheapest, most informative signal we're not using. A span where ASR was confident (logprob > -0.3, margin > 2.0) is unlikely to be a misrecognition — the model knew what it heard. A span where ASR was uncertain (logprob < -2.0, margin < 0.5) is exactly where correction should be aggressive.

**What it buys:**
- **Abstention**: High-confidence ASR spans → strongly bias toward keep-original. This directly cuts false positives.
- **Span proposal**: Low-confidence regions become higher-priority correction targets, reducing wasted retrieval on confident spans.
- **Judge calibration**: The judge currently has no information about whether the ASR "meant it." Adding mean_logprob and min_margin for a span gives the judge the single dimension it's missing most.

**Expected metric impact**: 10-20% reduction in false positives (judge accepting corrections on spans the ASR actually got right).

**Falsification**: If ASR logprobs do not correlate with whether the transcript word is correct (i.e., the model is confidently wrong as often as it's confidently right), this signal won't help. Test by computing logprob distributions for gold-correct vs. gold-incorrect spans in your eval set.

### 2. Constrained re-decode (candidate logprob comparison)

**Why second:** This is the "does the candidate actually fit the audio?" signal. Currently the judge decides based only on phonetic similarity. With constrained re-decode, we can ask: "if I force the decoder to output 'reqwest' instead of 'request', what total logprob does the model assign?" If the candidate scores higher than the original, it's acoustically plausible.

**What it buys:**
- **Context-sensitive verification**: Two candidates might have identical phonetic scores but wildly different acoustic fit. "Serde" vs. "surely" — the acoustic model knows which one it heard.
- **Near-neighbor disambiguation**: This directly addresses "distinguishing similar-sounding terms" because the acoustic model has information the phonetic pipeline doesn't.
- **Surface-mismatch cases**: When ASR produced a close-but-wrong surface form, the candidate's re-decode logprob will be higher, confirming the correction.

**Engineering approach:** Snapshot KV cache at span-start position. For each candidate, tokenize its surface form, force-decode those tokens, sum logprobs. Compare to original span's summed logprobs. The difference (candidate_logprob - original_logprob) is a single float feature for the judge.

**Expected metric impact**: 5-15% improvement in judge accuracy on ambiguous cases.

**Falsification**: If the ASR model assigns similar logprobs to acoustically different words (because its LM head dominates over acoustics by this point in the decoder), the signal collapses. Test by re-decoding known-wrong candidates and checking that they score lower.

### 3. Left/right context tokens as judge features

**Why third:** The judge currently scores each candidate in isolation — it doesn't know what words surround the span. "The serde crate" is strong context for accepting "serde"; "the sir day of the week" is strong context for rejecting it. This is the feature the judge is most obviously missing.

**What it buys:**
- **Context-sensitive acceptance**: Cross-features like (candidate × left_token) let the judge learn that "crate" after "serde" is a strong accept signal.
- **False positive reduction**: Common English phrases that happen to phonetically match a tech term get rejected because the surrounding context is wrong.

**Engineering approach:** Hash-trick sparse features (detailed in Section 3 below).

**Expected metric impact**: 5-10% improvement in judge precision, especially on the false-positive cases that are currently most annoying.

**Falsification**: If the eval set doesn't have enough examples where context disambiguates (i.e., all cases are either clearly right or clearly wrong on phonetics alone), context features won't move the needle. Check by looking at cases where the current judge is wrong and asking "would knowing the surrounding words have helped?"

### 4. Span-level ASR entropy (average across span tokens)

**Why fourth:** Complements logprobs with a different uncertainty measure. High entropy means the model was spread across many alternatives, not just torn between two. This catches the "model has no idea" case that margin alone misses (margin can be low even when entropy is low, if the top-2 are both reasonable).

**What it buys:**
- **Better abstention calibration**: Combined with logprobs, gives a 2D uncertainty space for the judge.
- **Span proposal filtering**: Skip high-entropy spans where the ASR was so confused that correction is unlikely to help.

**Cost concern:** Full softmax over 151936 vocab per token. On Apple Silicon M-series, this is ~0.3-0.5ms per token. For a 20-token sentence, adds ~10ms total. Acceptable, but only compute it for spans that pass initial retrieval (not all tokens).

**Expected metric impact**: 3-5% incremental improvement in abstention precision beyond logprobs alone.

**Falsification**: If entropy and logprob are perfectly correlated in practice (they often are for well-calibrated models), entropy adds no new information. Test correlation on your eval set.

### 5. Streaming revision count per token

**Why fifth:** Free signal. Tokens that were revised multiple times during streaming before committing are tokens the model was unstable about. This is orthogonal to logprobs (logprobs measure final-step confidence; revision count measures temporal instability across chunks).

**What it buys:**
- **Cheap uncertainty signal** that doesn't require any model changes.
- **Chunk-boundary artifact detection**: Tokens near chunk boundaries that got revised are likely split-context artifacts.

**Expected metric impact**: 2-3% incremental improvement in span proposal quality.

**Falsification**: If the rollback_tokens=5 window is too small to produce meaningful variation (most tokens are revised 0 or 1 times), the signal has no entropy. Check the distribution.

---

## 3. Concrete Judge Feature Schema with Context

### Current state

28 features, all phonetic/structural. No sentence context. FTRL logistic regression (online linear model).

### Proposed feature schema

I propose expanding to ~60-80 features, staying within a linear model but using hashed sparse features for context.

#### A. Existing features (keep all 28)

No changes. These are the backbone.

#### B. ASR uncertainty features (4 new dense features)

| Feature | Encoding | Source |
|---------|----------|--------|
| `span_mean_logprob` | float, raw (typically -5.0 to 0.0, scale by /5.0) | Per-token logprobs from decoder, averaged over span |
| `span_min_logprob` | float, raw / 5.0 | Minimum logprob in span (worst token) |
| `span_mean_margin` | float, raw / 5.0 | Average (top1 - top2 logprob) over span |
| `span_min_margin` | float, raw / 5.0 | Minimum margin in span (most uncertain token) |

#### C. Context token features (16 new sparse features via hashing)

Use the **hashing trick** to encode left/right context without an unbounded feature space.

**Encoding scheme:**
- Take 1 token left of span and 1 token right of span (the immediately adjacent words)
- Hash each token string to a bucket: `bucket = hash(token) % N_CONTEXT_BUCKETS`
- Use `N_CONTEXT_BUCKETS = 4` (yes, four — collisions are fine for a linear model, we're just breaking symmetry)

| Feature | Encoding |
|---------|----------|
| `left1_bucket_0` | 1.0 if left token hashes to bucket 0, else 0.0 |
| `left1_bucket_1` | 1.0 if left token hashes to bucket 1, else 0.0 |
| `left1_bucket_2` | 1.0 if left token hashes to bucket 2, else 0.0 |
| `left1_bucket_3` | 1.0 if left token hashes to bucket 3, else 0.0 |
| `right1_bucket_0` | ... |
| `right1_bucket_1` | ... |
| `right1_bucket_2` | ... |
| `right1_bucket_3` | ... |

8 features for basic left/right context.

**Cross features (candidate × context):**

| Feature | Encoding |
|---------|----------|
| `candidate_x_left1_bucket_0` | 1.0 if (hash(candidate_term + left_token) % 4 == 0) |
| `candidate_x_left1_bucket_1` | ... |
| `candidate_x_left1_bucket_2` | ... |
| `candidate_x_left1_bucket_3` | ... |
| `candidate_x_right1_bucket_0` | ... |
| `candidate_x_right1_bucket_1` | ... |
| `candidate_x_right1_bucket_2` | ... |
| `candidate_x_right1_bucket_3` | ... |

8 cross features. These let the model learn term-specific context patterns.

**Why 4 buckets:** With a linear model and online learning, we need features that can accumulate signal over many examples. 4 buckets means each bucket sees ~25% of all tokens — enough to learn meaningful weights quickly. More buckets = sparser updates = slower learning. You can increase to 8 or 16 later if data volume supports it.

**Hash function:** FNV-1a or xxhash on the lowercased token string. Deterministic, fast, no allocation.

#### D. Context type flags (4 new boolean features)

| Feature | Encoding | How to compute |
|---------|----------|----------------|
| `context_code_like` | 1.0 if surrounding text looks code-like | Heuristic: presence of `()`, `{}`, `::`, `.`, `_` within +-10 chars of span |
| `context_prose_like` | 1.0 if surrounding text looks like prose | Heuristic: no code markers, normal punctuation |
| `context_list_like` | 1.0 if span is in a list/enumeration | Heuristic: line starts with `-`, `*`, number |
| `context_sentence_start` | 1.0 if span is at sentence start | Heuristic: span_token_start == 0 or preceded by `.!?` |

#### E. Memory features (6 new dense features)

| Feature | Encoding | Source |
|---------|----------|--------|
| `candidate_prior_accept_count` | log(1 + count) / 3.0 | Per-(candidate_term) counter, incremented on user acceptance |
| `candidate_prior_reject_count` | log(1 + count) / 3.0 | Per-(candidate_term) counter, incremented on user rejection |
| `candidate_accept_ratio` | accepts / (accepts + rejects + 1) | Smoothed acceptance rate |
| `candidate_session_recency` | 1.0 if used in last 5 minutes, 0.5 if last 30 min, 0.0 otherwise | Session-local timestamp tracking |
| `span_text_prior_correct_count` | log(1 + count) / 3.0 | How many times this exact span text was previously corrected to anything |
| `span_text_prior_keep_count` | log(1 + count) / 3.0 | How many times this exact span text was previously kept as-is |

#### F. Re-decode features (2 new dense features, if signal #2 is implemented)

| Feature | Encoding | Source |
|---------|----------|--------|
| `redecode_logprob_delta` | (candidate_logprob - original_logprob) / 5.0 | Constrained re-decode comparison |
| `redecode_candidate_logprob` | candidate's forced logprob / 5.0 | Absolute acoustic plausibility of candidate |

### Total feature count: 28 + 4 + 16 + 4 + 6 + 2 = **60 features**

### Model recommendation

**Keep FTRL logistic regression.** Reasons:

1. Online learning is the core value proposition — users teaching the system. FTRL is designed for exactly this.
2. 60 features is well within linear model capacity.
3. The hashed context features give the linear model nonlinear-like expressiveness for the most important interactions.
4. No hyperparameter tuning hell. No gradient instability.
5. You already have it working and tested.

**When to consider upgrading:** If you find that the interaction between ASR uncertainty and context tokens requires more than the cross features provide (i.e., you need three-way interactions like candidate × context × uncertainty), consider a shallow factorization machine or a small 2-layer MLP with 60→32→1 architecture. But try the linear model first.

### Online update mechanics

**Immediate updates (memory counters):**
- `candidate_prior_accept_count`, `candidate_prior_reject_count`, `candidate_accept_ratio`, `session_recency`, `span_text_prior_*_count`
- These are lookup tables, not model weights. Update on every user action. Store as `HashMap<String, (u32, u32, Instant)>`.
- Persist across sessions in a simple JSON/bincode file.

**Slow updates (model weights via FTRL):**
- All 60 features train together through `teach_choice()`.
- Keep the current 4-epoch scheme from `judge.rs:254`.
- FTRL's L1 regularization (0.001) will zero out features that don't help, so adding features is safe.

**Guard against instability:**
- Cap the total weight update magnitude per teach_choice call: if any weight moves more than 0.5 in a single call, clip it. This prevents a single weird correction from destabilizing the model.
- Every 50 teach_choice calls, snapshot the weights. If eval performance drops below a threshold, roll back to the snapshot.
- Memory counters use exponential decay: multiply all counts by 0.99 at each session start. This prevents ancient history from dominating.

### What to pass to the judge (data flow change)

Currently `build_examples()` in `judge.rs:282` receives `(TranscriptSpan, Vec<(CandidateFeatureRow, IdentifierFlags)>)`.

Needs to also receive:
```rust
struct SpanContext {
    // ASR uncertainty
    mean_logprob: f32,
    min_logprob: f32,
    mean_margin: f32,
    min_margin: f32,
    // Sentence context
    left_token: Option<String>,   // word immediately left of span
    right_token: Option<String>,  // word immediately right of span
    // Context type
    code_like: bool,
    prose_like: bool,
    list_like: bool,
    sentence_start: bool,
    // Full sentence text (for hashing)
    sentence_text: String,
}

struct CandidateMemory {
    accept_count: u32,
    reject_count: u32,
    last_used: Option<Instant>,
}
```

This is a modest change to the interface. `TranscriptSpan` already has `token_start`/`token_end` — you just need to pass the surrounding sentence tokens alongside.

---

## 4. Assessment of IPA-from-Audio Path

### What the path would be

A separate acoustic model that takes raw audio (mel spectrogram for a time-aligned span) and produces phonetic output, bypassing the ASR text entirely.

### What output formats are possible

| Output type | Description | Engineering cost | Utility |
|-------------|-------------|-----------------|---------|
| **Phone sequence** | Linear sequence of IPA phones | Needs a CTC or attention phone recognizer model. Training data: TIMIT, CommonVoice+G2P. Fine-tuning a small model: 1-2 weeks. | Medium — gives one hypothesis, similar to transcript-derived IPA |
| **Phone posteriorgram** | Per-frame probability distribution over phone inventory (~40-80 phones) | Same model architecture, but keep the frame-level posteriors instead of decoding. Actually easier than getting the sequence. | High — preserves ambiguity, allows soft matching |
| **Phone lattice** | Compact graph of alternative phone sequences with scores | Requires CTC decoding with beam search + lattice construction. More complex. | High — but lattice matching against candidates is algorithmically complex |
| **Aligned phonetic spans** | Phone sequence with time boundaries per phone | CTC model gives this for free (alignment from CTC spike positions) | Medium — mainly useful if we also need phone-level timing |

### What it would buy over transcript-derived IPA

**The core issue:** Currently, the IPA for a span comes from eSpeak G2P applied to the ASR transcript text. This means:

1. If ASR produced "sir day" instead of "serde," eSpeak gives IPA for "sir day" — which is phonetically distant from the actual audio (the speaker said something like /sɜːrdeɪ/).
2. The phonetic distance between eSpeak("sir day") and the vocab entry for "serde" is the basis for retrieval and scoring.
3. But eSpeak("sir day") ≠ what the speaker actually said. The ASR already made a word-choice decision that distorted the phonetics.

**An audio-derived phone sequence would give us the actual phones the speaker produced**, before ASR word-level decisions distorted them. This is most valuable when:

- **ASR word-choice drift**: ASR picked a real English word that sounds similar but isn't phonetically identical to what was said. Audio IPA would match the vocab entry better than transcript IPA.
- **Technical OOV terms**: ASR has never seen "serde" in training, so it maps the sounds to known words. Audio phones preserve the actual sounds.
- **Multi-word splits**: ASR split one word into multiple ("req west" for "reqwest"). Audio phones would give a continuous phone sequence without the artificial word boundary.

**Where it would NOT help much:**

- **Acronyms/spelling**: "LLVM" spoken as "L-L-V-M" — ASR usually gets individual letters right, and audio phones won't improve on this.
- **Cases where ASR is correct**: If the transcript already has the right word, audio-derived IPA adds nothing.
- **Near-neighbor resolution where both candidates are phonetically close**: If "serde" and "surely" are both close to the audio phones, you still need the judge/context to pick.

### Where in the pipeline it would help

| Pipeline stage | How it helps | Magnitude of help |
|----------------|-------------|-------------------|
| **Retrieval** | Query the phonetic index with audio-derived phones instead of transcript-derived phones. When ASR drifted, audio phones are closer to the correct term's phones, improving recall. | **Significant** — this is the main value. Retrieval currently fails when transcript-derived IPA is too far from the target due to ASR word choice. |
| **Verification** | Compare audio phones to candidate phones directly. Token similarity and feature similarity would be computed on truer phonetic representations. | **Moderate** — improves scoring accuracy for the cases where ASR drifted. |
| **Span proposal** | Identify spans where audio phones diverge from transcript phones — these are likely ASR errors. | **Moderate** — gives a new span-proposal signal. |
| **Judge** | Audio-phone similarity as an additional feature alongside transcript-phone similarity. | **Small** — the judge already has transcript-phone scores; adding audio-phone scores is incremental. |
| **Rendering** | No help — rendering is about surface text, not phones. | None |

### Engineering cost assessment

**Option A: Fine-tune a small CTC phone recognizer**

- Base model: wav2vec2-base or HuBERT-base, fine-tuned on phone labels
- Training data: TIMIT (6300 utterances) + CommonVoice with forced alignment + G2P
- Training time: ~4 hours on a GPU
- Inference: ~50ms per second of audio on Apple Silicon (small transformer encoder)
- MLX port: wav2vec2 architecture is simpler than Qwen3, ~3-5 days of porting work
- Model size: ~90MB (base) or ~30MB (distilled)
- Total effort: **2-3 weeks** for a working prototype including MLX port

**Option B: Extract phone-like information from Qwen3 ASR encoder**

- The audio encoder already produces 1024-dim features per ~80ms frame
- These features encode phonetic information (they have to, for the decoder to work)
- Could train a small linear probe (1024 → num_phones) on top of frozen encoder features
- Training: ~1 day with TIMIT + forced alignment
- No additional model at inference time — reuse encoder features
- Total effort: **3-5 days** for the probe, assuming you have phone-labeled data for a few hours

**Option B is dramatically cheaper and should be tried first.**

### Recommendation

**Not mainline strategy right now. Worthwhile as a bounded experiment after signals #1-3 are in.**

Reasoning:
1. The constrained re-decode signal (#2 in Section 2) gives most of the same benefit — it asks "does the candidate fit the audio?" using the existing model, without a separate phone recognizer.
2. The cases where audio-derived IPA helps most (ASR word-choice drift) are also the cases where constrained re-decode helps most.
3. If constrained re-decode proves insufficient (test this!), then the encoder-probe approach (Option B) is a cheap experiment.
4. A full separate phone recognizer (Option A) is only worth it if (a) constrained re-decode doesn't work AND (b) the encoder probe doesn't produce clean enough phones.

**Falsification for the whole path:** If your eval set's retrieval failures are primarily due to the target term not being in the vocabulary (rather than phonetic distance from transcript IPA to vocab IPA), then audio-derived IPA won't help — the problem is vocabulary coverage, not phonetic representation.

---

## 5. User Vocabulary Onboarding / Teaching Workflow

### What happens when a user adds a new term (current state)

Per the codebase: user provides term text → system generates spoken forms via identifier alias expansion (`phonetic_lexicon.rs`) → eSpeak G2P generates IPA → term + aliases enter the phonetic index. No pronunciation guidance, no example sentences, no negative examples.

### Problem

Just adding the word is not enough because:
1. The judge has never seen this candidate before — it has no learned context for when to accept it.
2. The phonetic index will return it whenever anything sounds vaguely similar — high false positive risk for new terms.
3. No confusion surfaces exist yet — the system doesn't know what ASR typically produces when the user says this word.

### Minimal workflow (ship first)

**On add:**
1. User types or pastes the term (e.g., "Bevy")
2. System auto-generates:
   - Spoken form via identifier splitting ("bevy")
   - IPA via eSpeak (/ˈbɛvi/)
   - Identifier flags (camel, snake, digits, etc.)
3. System shows the spoken form and a phonetic transcription. User can tap to hear TTS playback (system TTS, not a separate model). If wrong, user can type a correction ("it's pronounced bev-ee, not beev-ee").
4. **System asks one question**: "Say a sentence where you'd use this word." User speaks one sentence. System:
   - Transcribes it with ASR
   - Identifies the span where the term should appear
   - Records the actual ASR output for that span as a confusion surface (e.g., ASR said "Debbie" → confusion surface "Debbie" for term "Bevy")
   - Records the sentence as a positive context example
5. Term enters the index immediately. Confusion surface enters the index immediately.
6. Judge gets a synthetic positive training example: features computed from the confusion surface matching to the term, target = true. Also a synthetic negative: features from the confusion surface matching to "keep original", target = false. Run one teach_choice call.

**Memory features initialized:**
- `accept_count = 1` (from the setup sentence)
- `reject_count = 0`
- `session_recency = now`

**Total user time: ~15 seconds.** Type the word, say one sentence, done.

**What you don't get:**
- Negative examples (when NOT to correct)
- Multiple confusion surfaces
- Context diversity

### Richer workflow (opt-in, progressive)

**On add (same as minimal), plus:**

1. **Pronunciation review**: System shows IPA, plays TTS. User confirms or re-records pronunciation. If re-recorded, system extracts a confusion surface from the new ASR output.

2. **Three prompted sentences**: System generates sentence prompts using templates:
   - "Say a sentence using [term] in a technical context" → e.g., "I'm building a game with Bevy"
   - "Say a sentence where a similar-sounding word should NOT be replaced" → e.g., "The bevy of options was overwhelming" (if applicable — system detects if the term has common English homophones)
   - "Say the term in a list with other terms" → e.g., "We're using Bevy, Tokio, and Axum"

   Each sentence: ASR transcribes → system extracts confusion surfaces → positive/negative training examples.

3. **Category/domain tag** (optional): User picks from a short list: "Programming," "Science," "Brand name," "Acronym," "Other." This becomes a feature the judge can use.

4. **Auto-generated negative examples**: System uses eSpeak to find real English words phonetically close to the new term (e.g., for "Bevy": "heavy," "levy," "every"). For each, creates a synthetic negative example: these words should NOT be corrected to "Bevy" unless context is right. These don't require user input.

5. **LLM-assisted example generation** (if available): Generate 5-10 synthetic sentences containing the term in various contexts. Use these as context exemplars for the cross features, not as training data directly.

**Total user time: ~90 seconds.** Mostly speaking natural sentences.

### Tradeoffs

| Aspect | Minimal | Richer |
|--------|---------|--------|
| User friction | Very low (~15s) | Moderate (~90s) |
| False positive risk for new term | Higher — one confusion surface, no negatives | Lower — multiple surfaces, auto-generated negatives |
| Time to useful corrections | Immediate (first sentence gives a confusion surface) | Immediate + better over time |
| Judge cold-start quality | 1 synthetic example — judge mostly relies on phonetic scores | 3-6 real examples — judge starts with meaningful context signal |
| Pronunciation accuracy | eSpeak guess, may be wrong for unusual terms | User-verified, more reliable |
| Implementation cost | Small — leverages existing infrastructure | Medium — needs sentence prompting UI, auto-negative generation |

### Recommendation

Ship the minimal workflow first. Add the "say a sentence" step even in minimal — it's the single highest-value onboarding action because it produces a real confusion surface. The richer workflow should be an optional "Improve recognition" button that appears after the user has added a term.

**Do NOT:**
- Force users through a training wizard before they can use a term
- Require pronunciation recording (TTS + eSpeak is good enough for most terms)
- Generate example sentences at add time using an LLM (save this for background processing)

**DO:**
- Auto-generate the negative phonetic neighbors at add time — this is cheap and prevents the most obvious false positives
- Save the raw audio from the onboarding sentence for potential future audio-IPA work
- Log everything (see Section 6)

---

## 6. Training Data Generation from User Actions

### Events to log

| Event | When | Data captured |
|-------|------|--------------|
| `correction_accepted` | User accepts a suggested replacement | span_text, candidate_term, all candidate feature rows, judge scores, ASR logprobs (if available), left/right context, sentence text, timestamp |
| `correction_rejected` | User rejects a suggestion (keeps original) | Same as above |
| `manual_correction` | User manually corrects a span (not from our suggestion) | original_span_text, corrected_text, sentence_text, ASR logprobs, timestamp |
| `term_added` | User adds a new vocabulary term | term, spoken_form, ipa, source (manual/import), timestamp |
| `term_used` | A term from vocabulary appears in final transcript (accepted or auto-corrected) | term, span_text, confidence, context, timestamp |
| `session_start` / `session_end` | Dictation session boundaries | app context, duration, total corrections count |

### Turning events into training examples

**From `correction_accepted`:**
```
Positive example: (chosen_candidate_features, target=true)
Hard negatives: (every_other_candidate_features, target=false)
Hard negative: (keep_original_features, target=false)
```
This is exactly what `teach_choice()` already does. The new part: also store the raw event for offline replay.

**From `correction_rejected`:**
```
Positive example: (keep_original_features, target=true)  // implicit via teach_choice(None)
Hard negatives: (all_candidate_features, target=false)
```

**From `manual_correction`:**
This is the richest signal — the user corrected something we didn't even suggest.
```
1. Run retrieval on the original span to find if the corrected term was a candidate
2. If yes: treat as correction_accepted for that candidate (we missed it, learn from it)
3. If no: this may indicate a missing vocabulary entry — prompt user to add it
4. Either way: the original span text becomes a confusion surface for the corrected term
```

**From `term_used` (no user action):**
```
Weak positive: increment candidate_prior_accept_count for auto-corrections
Do NOT create a training example — auto-corrections haven't been user-validated
```

### Avoiding data poisoning

**Problem:** One weird correction (user fat-fingered, changed their mind, was testing the system) can poison the model if treated as gospel.

**Mitigations:**

1. **Minimum example threshold for weight updates**: Don't update FTRL weights from a single event. Buffer the last 3-5 events per candidate term. Only run teach_choice when the buffer reaches 3+ consistent examples for the same candidate. This prevents one-off errors from shifting weights.

2. **Confidence-weighted learning rate**: Use a lower FTRL alpha (0.1 instead of 0.5) for events where the judge's original confidence was high (> 0.8). High confidence means the model was already sure — a contradicting signal is more likely to be noise.

3. **Separate memory from weights**: Memory counters (accept/reject counts) update immediately but are just features — they don't directly change the model's behavior, they're inputs the model can learn to weight. This means a single weird event only shifts one feature by a small amount.

4. **Decay and forget**: All memory counters decay by 0.99× per session start. Events older than 30 days get half weight in offline replay. This prevents ancient mistakes from persisting.

5. **Per-user isolation**: All state (model weights, memory counters, event log) is per-user, stored locally. No cross-user contamination. A user's weird correction only affects their own model.

### When to update what

| Signal | Update timing | Mechanism |
|--------|--------------|-----------|
| Memory counters (accept/reject/recency) | **Immediate**, on every event | HashMap increment, persist to disk every 10 events |
| Judge weights (FTRL) | **Batched**, every 3-5 events for the same candidate | `teach_choice()` with accumulated examples |
| Confusion surfaces | **Immediate**, on manual correction or vocabulary onboarding | Insert into phonetic index |
| Vocabulary index | **Immediate**, on term_added | Rebuild affected index entries |
| Offline model retraining | **Never** for the linear judge — FTRL handles it online. Only if you switch to a more complex model. | Replay from event log |

### Event log format

```rust
struct CorrectionEvent {
    timestamp: Instant,
    event_type: EventType,  // Accepted, Rejected, ManualCorrection
    sentence_text: String,
    span: SpanSnapshot {
        text: String,
        token_start: usize,
        token_end: usize,
        ipa_tokens: Vec<String>,
        mean_logprob: Option<f32>,
        mean_margin: Option<f32>,
    },
    candidates: Vec<CandidateSnapshot {
        alias_id: u32,
        term: String,
        features: Vec<f64>,  // full 60-feature vector
        judge_probability: f32,
    }>,
    chosen: Option<u32>,  // alias_id of chosen, or None for keep-original
    left_context: Option<String>,
    right_context: Option<String>,
}
```

Store as append-only JSONL file. One file per user. Replay for offline analysis or model retraining.

### Hard negatives

For each positive correction event (user accepted "serde" for "sir day"):
1. **In-batch negatives**: All other candidates that were proposed but not chosen (already done by teach_choice)
2. **Memory-derived negatives**: If "surely" was a candidate and was rejected, future occurrences of "sir day" with "surely" as a candidate carry a strong negative prior
3. **Auto-generated negatives from phonetic neighbors**: At term-add time, generate negative examples for common English words that sound similar (e.g., "surely" is a phonetic neighbor of "serde" — "sir day" should NOT correct to "surely")

---

## 7. Recommended Implementation Order

### Step 1: Expose per-token logprobs and margin from decoder

**Why first:** Smallest change, highest leverage, unblocks everything else.

**What to do:**
- Modify `generate.rs` `prefill_and_decode()` to return `Vec<(i32, f32, f32)>` (token_id, logprob, margin) instead of just `Vec<i32>`
- After computing logits at each step, compute `log_softmax(logits)`, extract top-1 logprob and (top1 - top2) margin
- Propagate through `Session` in `bee-transcribe` to attach logprobs to committed tokens
- Add `mean_logprob: Option<f32>`, `min_logprob: Option<f32>`, `mean_margin: Option<f32>`, `min_margin: Option<f32>` to `TranscriptSpan`

**Metric to check:** Compute logprob distributions for gold-correct vs. gold-incorrect spans in eval. If the distributions are separated (AUC > 0.65), this signal has value.

**Falsification:** If the distributions overlap completely (AUC ~0.50), logprobs from this model are not informative for our task.

**Estimated effort:** 1-2 days.

### Step 2: Add ASR uncertainty features to the judge

**Why second:** Direct consumer of Step 1. Immediate eval impact.

**What to do:**
- Add the 4 ASR uncertainty features to `FEATURE_NAMES` and `build_examples()` in `judge.rs`
- Pass `SpanContext` (or just the logprob fields) through the scoring pipeline
- Update seed training examples to include representative logprob values
- Update NUM_FEATURES from 28 to 32

**Metric to check:** Eval false positive rate (corrections on already-correct spans). Should drop measurably.

**Falsification:** If false positive rate doesn't change, logprobs aren't helping the judge decide.

**Estimated effort:** 0.5-1 day.

### Step 3: Add left/right context features and cross features to the judge

**Why third:** This is the "teach the judge about context" step. Independent of ASR signals.

**What to do:**
- Implement hashed context features (16 features: 8 context + 8 cross)
- Add context type flags (4 features)
- Modify `TranscriptSpan` or `build_examples()` to receive surrounding token info
- This requires the caller (in `beeml/src/main.rs` evaluation code) to pass sentence context alongside spans

**Metric to check:** Eval accuracy on cases where context disambiguates. Identify these cases first by manual inspection.

**Falsification:** If your eval cases are all unambiguous on phonetics alone (no false positives from lack of context), context features won't help yet — you need more diverse eval data first.

**Estimated effort:** 2-3 days.

### Step 4: Add memory counters and event logging

**Why fourth:** Enables the learning loop. Independent of model changes.

**What to do:**
- Implement `CorrectionEvent` logging (append-only JSONL)
- Implement per-term memory counters (accept/reject/recency)
- Add 6 memory features to the judge
- Wire up `teach_choice()` to log events and update counters
- Implement counter decay on session start

**Metric to check:** After N user corrections, does the judge get better for the corrected terms? Simulate with replay.

**Falsification:** If FTRL with 38 features (32 + 6 memory) doesn't learn faster than with 32, memory features aren't pulling weight.

**Estimated effort:** 2-3 days.

### Step 5: Implement constrained re-decode for candidate verification

**Why fifth:** This is the most powerful signal but also the most complex to implement correctly.

**What to do:**
- Add KV cache snapshot capability (clone the cache at span-start position)
- For each candidate that passes verification, tokenize its surface form
- Force-decode those tokens through the model from the snapshot, accumulating logprobs
- Compare to original span's logprobs
- Add 2 re-decode features to the judge

**Metric to check:** On ambiguous cases (two candidates with similar phonetic scores), does re-decode logprob reliably prefer the correct one?

**Falsification:** If the decoder's LM prior dominates (common English words always score higher regardless of audio), re-decode doesn't add acoustic information. Test by re-decoding nonsense words — they should score lower than real words, but the gap between phonetically-similar real words should still be informative.

**Estimated effort:** 3-5 days. The KV cache cloning is the tricky part — need to handle the `Vec<Option<Array>>` structure per layer.

### Step 6: Vocabulary onboarding with the minimal "say a sentence" flow

**Why sixth:** By now the judge has uncertainty features and context features, so new vocabulary entries can benefit from them immediately.

**What to do:**
- Build the onboarding UI (or at minimum a CLI flow for testing)
- Implement the single-sentence confusion surface extraction
- Wire up auto-generated phonetic neighbor negatives
- Connect to event logging

**Metric to check:** For newly added terms, how many dictation sessions does it take before the system reliably corrects them? Target: correct on first real use after onboarding.

**Falsification:** If the single onboarding sentence doesn't produce a usable confusion surface (because ASR doesn't misrecognize the term predictably), the flow needs multiple sentences or manual confusion entry.

**Estimated effort:** 3-5 days (mostly UI/UX work).

### Step 7: Evaluate encoder-probe phone extraction (bounded experiment)

**Why last:** This is speculative and only worth doing if Steps 1-5 haven't closed the remaining gap.

**What to do:**
- Extract encoder features for spans in the eval set
- Train a linear probe (1024 → ~80 phones) on TIMIT or equivalent
- Compare probe-derived phones to eSpeak-derived phones for retrieval quality
- If the probe produces cleaner phones for ASR-drifted cases, integrate as an alternative retrieval query

**Metric to check:** Retrieval recall for cases that currently fail due to ASR word-choice drift.

**Falsification:** If encoder features at the frame level don't separate phones cleanly (probe accuracy < 70%), the encoder representations are too entangled for a linear probe to extract useful phone information.

**Estimated effort:** 3-5 days for the experiment. Integration if successful: 1-2 weeks.

---

## 8. Open Risks / Unknowns

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Qwen3 ASR logprobs may not correlate with correctness.** The model may be confidently wrong on OOV terms because its LM prior is strong. | High | Test empirically in Step 1 before building on this assumption. The margin signal may be more robust than raw logprob. |
| **Constrained re-decode may be dominated by the decoder's language model.** If the LM head contributes more to logits than the audio conditioning, re-decoding "serde" vs. "surely" will just reflect English word frequency, not acoustic fit. | Medium | Test by comparing re-decode scores with and without audio features (zero out the audio embeddings). If scores barely change, the audio signal is too weak at the decoder output. |
| **Hashed context features with 4 buckets may be too coarse.** Many different contexts collapse into the same bucket, washing out the signal. | Low | Easy to increase to 8 or 16 buckets later. Start with 4 to ensure fast learning from small data. |
| **FTRL with 60 features may converge too slowly from user corrections.** Each correction only teaches one example — 60 weights is a lot to learn from. | Medium | The base weights from seed training carry most of the load. Context/memory features start at 0 and slowly activate. L1 regularization zeros out unhelpful features. Monitor convergence by tracking eval accuracy after N teach_choice calls. |
| **Memory counters can create feedback loops.** If the system corrects "sir day" → "serde" and the user accepts (even if wrong), accept_count goes up, making future corrections more aggressive, leading to more (possibly wrong) corrections. | Medium | Decay counters. Require 3+ consistent events before memory strongly influences the judge. If the user ever rejects a previously-accepted correction, reset the accept_count for that term to max(count-3, 0). |
| **On-device storage of event logs grows unbounded.** | Low | Cap event log at 10,000 events per user. Oldest events are dropped. This is ~2-5MB in JSONL. |
| **The encoder probe experiment may require phone-labeled training data we don't have.** TIMIT is small (5 hours). CommonVoice + forced alignment is noisy. | Medium | Start with TIMIT for the probe. If accuracy is marginal, augment with Librispeech + Montreal Forced Aligner for phone labels. This is standard practice. |
| **Users may not provide useful onboarding sentences.** They might say the term in isolation or in a trivial sentence that doesn't produce a realistic confusion surface. | Medium | Prompt specifically: "Use [term] in a sentence you'd actually say while working." Show a template if they're stuck. Accept whatever they give — even a bad sentence produces some signal. |
