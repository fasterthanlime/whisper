# 007: Gate + Ranker Status Report (2026-04-08)

## Architecture

Two-stage correction pipeline:

1. **Gate** (Stage A) — span-level binary classifier: "should we attempt
   correction on this span?" Operates on the transcript span + aggregate
   candidate stats. If the gate fires, candidates pass to the ranker.

2. **Ranker** (Stage B) — candidate-level binary classifier: "is this
   candidate the right replacement?" Each candidate gets its own feature
   vector; the highest-probability candidate above threshold wins.

Both models are **SparseFtrl** (FTRL-Proximal with lazy L1 regularization),
trained offline on recorded audio examples, then frozen for production.

## Model: SparseFtrl

- FTRL-Proximal optimizer (McMahan et al., 2013)
- Sparse: only stores weights for features that received updates
- HashMap<u64, (f64, f64)> accumulators (z, n per feature)
- Production uses `loaded_weights: HashMap<u64, f64>` (pre-computed, frozen)
- Binary format: u32 LE count, then (u64 LE index, f64 LE weight) pairs
- Hyperparameters: alpha=0.5, beta=1.0, l1=0.1, l2=0.01

## Gate Features (Stage A)

14 dense features + sparse hashed context features.

### Dense features (indices 0-13)

| Idx | Name | Normalization | Description |
|-----|------|---------------|-------------|
| 0 | bias | 1.0 | Always-on bias term |
| 1 | span_token_count | /4.0 | Number of whitespace tokens in span |
| 2 | span_phone_count | /12.0 | Number of IPA phones in span |
| 3 | span_low_content | binary | Span contains only low-content words (a, the, is...) |
| 4 | span_mean_logprob | /5.0 | Mean ASR log-probability across span words |
| 5 | span_min_logprob | /5.0 | Min ASR log-probability across span words |
| 6 | span_mean_margin | /5.0 | Mean ASR margin (gap between top-1 and top-2 token) |
| 7 | span_min_margin | /5.0 | Min ASR margin across span words |
| 8 | span_correct_count | ln(1+n)/3 | How many times this span text was previously corrected (TermMemory) |
| 9 | max_acceptance_score | raw | Best acceptance_score among all candidates |
| 10 | max_phonetic_score | raw | Best phonetic_score among all candidates |
| 11 | any_verified | binary | Any candidate is a verified match |
| 12 | any_acceptance_floor | binary | Any candidate passed the acceptance floor |
| 13 | candidate_count | ln(1+n)/3 | Number of retrieval candidates for this span |

### Sparse context features (hashed into offset 1000 + 16384 buckets)

| Feature key | Description |
|-------------|-------------|
| `L1={word}` | Left-1 context word (lowercased) |
| `L2={w1}_{w2}` | Left bigram |
| `R1={word}` | Right-1 context word (lowercased) |
| `R2={w1}_{w2}` | Right bigram |
| `CTX=code` | Code-like context (nearby `()`, `::`, `->`, `fn `, etc.) |
| `CTX=prose` | Prose-like context (no code markers) |
| `CTX=list` | Line starts with list marker (`-`, `*`, digit) |
| `CTX=sent_start` | Span is at sentence start |
| `APP={id}` | Application ID (not populated in offline eval) |

## Ranker Features (Stage B)

36 dense features + sparse context features (per-candidate).

### Dense features (indices 0-35)

| Idx | Name | Description |
|-----|------|-------------|
| 0 | bias | Always-on |
| 1 | acceptance_score | Combined retrieval score |
| 2 | phonetic_score | Phonetic similarity score |
| 3 | coarse_score | Coarse-grained similarity |
| 4 | token_score | Token-level alignment score |
| 5 | feature_score | Feature-level alignment score |
| 6 | feature_bonus | Bonus from feature similarity |
| 7 | best_view_score | Best score across phonetic index views |
| 8 | cross_view_support | Support across multiple index views (/6.0) |
| 9 | qgram_overlap | Q-gram overlap with query (/10.0) |
| 10 | total_qgram_overlap | Total q-gram overlap (/20.0) |
| 11 | token_count_match | Binary: candidate has same token count as span |
| 12 | phone_closeness | 1/(1+abs(phone_count_delta)) |
| 13 | alias_source_spoken | Binary: alias came from spoken form |
| 14 | alias_source_identifier | Binary: alias came from identifier form |
| 15 | alias_source_confusion | Binary: alias came from confusion form |
| 16 | identifier_acronym | Binary: term looks like an acronym |
| 17 | identifier_digits | Binary: term contains digits |
| 18 | identifier_snake | Binary: term is snake_case |
| 19 | identifier_camel | Binary: term is CamelCase |
| 20 | identifier_symbol | Binary: term contains symbols |
| 21 | short_guard_passed | Binary: passed short-query guard |
| 22 | low_content_guard_passed | Binary: passed low-content guard |
| 23 | acceptance_floor_passed | Binary: passed acceptance floor |
| 24 | verified | Binary: candidate is a verified match |
| 25 | rank_in_span | 1/(1+rank) — rank by acceptance_score |
| 26 | margin_to_next | Score gap to adjacent candidate |
| 27 | is_best_verified | Binary: best verified candidate |
| 28 | is_only_verified | Binary: only verified candidate |
| 29 | normalized_acceptance | acceptance / max_acceptance in span |
| 30 | normalized_phonetic | phonetic / max_phonetic in span |
| 31 | accept_count | ln(1+n)/3 — TermMemory accept count |
| 32 | reject_count | ln(1+n)/3 — TermMemory reject count |
| 33 | total_count | ln(1+n)/3 — total interactions |
| 34 | recent_accept_count | ln(1+n)/3 — recent accepts |
| 35 | session_recency | 1.0 if <5min, 0.5 if <30min, 0.0 otherwise |

### Sparse context features (per-candidate, in ranker)

Same as gate, plus term-crossed variants:

| Feature key | Description |
|-------------|-------------|
| `L1={word}` | Left-1 context word |
| `TERM={term}\|L1={word}` | Left-1 crossed with candidate term |
| `L2={bigram}` | Left bigram |
| `TERM={term}\|L2={bigram}` | Left bigram crossed with candidate term |
| `R1={word}` | Right-1 context word |
| `TERM={term}\|R1={word}` | Right-1 crossed with candidate term |
| `R2={bigram}` | Right bigram |
| `TERM={term}\|R2={bigram}` | Right bigram crossed with candidate term |
| `TERM={term}` | Candidate term identity |
| `CTX=code`, `CTX=prose`, `CTX=list`, `CTX=sent_start`, `APP={id}` | Same as gate |

## Training Data

### Canonical recordings (106 examples)

Authored sentences containing terms from the phonetic index. Each has:
- Audio file (`.ogg`)
- Source text (ground truth)
- Transcript (ASR output)
- Per-word alignment with logprob/margin from ASR

105/106 have gold candidate retrieved (99.1%).
98/106 have gold candidate verified (92.5%).

### Counterexample recordings (47 examples, curated)

Sentences where the surface form is phonetically similar to an indexed term,
but should NOT be corrected. Each has the same fields as canonical.

Curated to keep only **hard negatives** — entries where the surface form has
genuine phonetic similarity to the target term:

| Term | Surface forms | Count |
|------|---------------|-------|
| arborium | Aquarium | 2 |
| fasterthanlime | faster than light | 6 |
| MachO | Marco | 1 |
| qwen | Quinn | 3 |
| reqwest | request | 2 |
| rustc | Russia, Rusty, rusty, C | 13 |
| serde | third day | 5 |
| tokio | Tokyo | 3 |
| u32 | You thirty two | 2 |
| u8 | You ate, You eight, You eat | 9 |
| Vec | Vic | 1 |

Removed 66 entries that were either:
- Phonetically unrelated ("Hmm" for MIR, "Okay" for repr, "They" for serde)
- ASR hallucination loops (400+ words of repeated phrases)

### Training procedure

- 5-fold cross-validation for eval
- 4 epochs per fold
- Gate: trained on ALL spans per case. Canonical gold span = positive,
  counterexample spans = negative, non-gold canonical spans skipped (ambiguous).
- Ranker: trained only on verified gold spans. Gold candidate = positive,
  non-gold candidates = negative.
- Weight export: trained on ALL data (no holdout), 4 epochs.

## Benchmark Interpretation

The 106 canonical + 47 curated hard negatives set is a **gate stress-test**,
not the same thing as a broader end-to-end product benchmark.

The counterexample set was deliberately curated to keep only phonetically
confusable near-neighbors ("rusty" for rustc, "Quinn" for qwen, "Tokyo"
for tokio). The failure rates on this set measure **hard-negative
susceptibility** — how often the gate is tricked by adversarial near-miss
inputs — not the deployed-user false positive rate.

In production, most transcript spans will not be phonetic near-neighbors
of indexed terms, so the real false positive rate will be much lower than
what the hard-negative benchmark shows.

We maintain two separate operating points:

- **Product-conservative point**: minimize real false positives in production.
- **Hard-negative stress-test point**: track how often the gate is tricked
  by curated near-neighbors.

Both matter, but they answer different questions.

## Current Eval Results

### Upstream opportunity set

| Metric | Score |
|--------|-------|
| Gold retrieved | 105/106 (99.1%) — 1 lost at retrieval |
| Gold verified | 98/106 (92.5%) — 7 lost at verification |

### Ranker performance

- **Top-1 accuracy: 96.9%** (95/98) — when the gate opens on a gold span,
  the ranker almost always picks the right candidate.

### Gate hard-negative stress-test (at GT=0.4, RT=0.1)

| Metric | Score |
|--------|-------|
| Canonical gate open rate | 74/98 (75.5%) |
| Hard-negative rejection rate | 21/47 (44.7%) |
| Hard-negative susceptibility | 26/47 (55.3%) |

### Gate threshold sweep

| GT | Canonical open | Counterex closed | Balanced |
|----|----------------|------------------|----------|
| 0.1 | 95.9% | 2.1% | 49.0% |
| 0.2 | 85.7% | 10.6% | 48.2% |
| 0.3 | 80.6% | 31.9% | 56.3% |
| **0.4** | **75.5%** | **44.7%** | **60.1%** |
| 0.5 | 63.3% | 55.3% | 59.3% |
| 0.6 | 44.9% | 66.0% | 55.4% |
| 0.7 | 26.5% | 70.2% | 48.4% |

### Gate probability distributions

| Set | min | p25 | p50 | p75 | max |
|-----|-----|-----|-----|-----|-----|
| Canonical (should open) | 0.017 | 0.403 | 0.572 | 0.713 | 0.981 |
| Counterex (should close) | 0.097 | 0.273 | 0.451 | 0.779 | 0.956 |

The distributions overlap significantly. Median canonical gate prob is 0.572,
median counterexample is 0.451. The IQRs overlap from 0.273 to 0.713.

### Feature ablation

| Feature set | Best threshold | Canonical | Counterex | Balanced |
|-------------|---------------|-----------|-----------|----------|
| phonetic_only | T=0.7 | 1.0% | 100.0% | 50.5% |
| +asr | T=0.6 | 4.1% | 100.0% | 52.0% |
| +context | T=0.8 | 3.1% | 100.0% | 51.5% |
| all | T=0.8 | 3.1% | 100.0% | 51.5% |

Feature ablation uses the single-stage case_balanced formulation (not the
two-stage gate+ranker), which performs poorly overall. The features don't
separate well in that formulation.

## Diagnosis

The system is now clearly shaped like this:

- **Retrieval/verification/ranker are strong enough.** 99.1% retrieval,
  92.5% verification, 96.9% ranker top-1.
- **The gate is the limiting factor.** 60.1% balanced accuracy at best,
  heavy overlap between canonical and hard-negative probability distributions.
- **The gate's current feature set and data volume are not enough for
  adversarial hard negatives.**

The strongest evidence:

- Gate balanced only 60.1% at best operating point
- Ranker top-1 96.9% — already doing its job
- Gate distributions overlap heavily (canonical median 0.572, hard-neg median 0.451)
- Sparse context features don't help much in the single-stage ablation
- Only 47 hard negatives means the gate is trying to learn difficult
  contextual distinctions from very little data

The bottleneck is gate discrimination. The architecture is right.
This is a data + gate features problem.

## Next Steps

### 1. Expand hard-negative set term-by-term (requires recording)

Many canonical terms have no counterexamples at all. The gate is being
asked to generalize contextual rejection from very few examples.

Add 2-5 hard negatives per important term, prioritizing highest-risk:

| Priority | Term | Current hard-neg count | Why high-risk |
|----------|------|----------------------|---------------|
| High | u8, u32 | 9, 2 | Common English ("you ate", "you thirty-two") |
| High | Vec | 1 | Common English ("Vic", "that") |
| High | tokio | 3 | City name ("Tokyo") |
| High | reqwest | 2 | Common word ("request") |
| High | serde | 5 | Similar phrases ("third day") |
| High | rustc | 13 | Common word ("rusty", "Russia") |
| Medium | QEMU | 0 | Homophones possible |
| Medium | miri | 0 | Name collision ("Miriam", "me") |
| Medium | repr | 0 | Common word ("rapper", "wrapper") |
| Medium | SQLite | 0 | Phonetic near-miss ("sequel light") |
| Medium | x86_64 | 0 | Spoken form confusion |
| Medium | AArch64 | 0 | Spoken form confusion |
| Lower | MIR | 0 | "mirror", "mere" |
| Lower | ripgrep | 0 | "crap" etc. |
| Lower | regalloc | 0 | No obvious confusable |
| Lower | wasm | 0 | "awesome" |

### 2. Add term/family-aware gate summary features

The gate is currently span-level and anonymous — it doesn't know which
term family is tempting it. Without collapsing back into candidate-level
logic, add a few candidate-summary identity features:

- Top candidate term hash / family hash
- Top candidate alias source (spoken, identifier, confusion)
- Top candidate identifier class (acronym, digits, snake, camel, symbol)
- Top candidate is short code term (u8, u32, Vec, etc.)
- Top candidate is numeric typed term
- Top candidate is tool/crate/architecture family

This gives a term-sensitive gate without the old broken formulation.

### 3. Add dense handcrafted context features

Start simpler than semantic embeddings. Add cheap, interpretable features:

- Nearby token contains code punctuation (`()`, `::`, `->`)
- Nearby token is in a small programming lexicon
- Sentence contains multiple technical tokens
- Nearby token looks like identifier / path / type / crate / architecture
- Numeric context nearby
- Sentence looks prose-like vs code-like vs configuration-like

These are likely more data-efficient than sparse hashes on a tiny
hard-negative set.

### 4. Leave the ranker alone

At 96.9% top-1 on reachable gold spans, the ranker is doing its job.
No investment needed here.

### 5. Rerun hard-negative gate benchmark after each change

Track the gate stress-test numbers after each improvement. Keep the
product-conservative and stress-test operating points separate.

## What we would NOT do yet

- Neural context embeddings
- A bigger model
- Architecture changes to the two-stage split
- Removing the gate/ranker separation

The architecture is right. This is a data + features problem.
