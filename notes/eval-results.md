# Phase 4 Offline Judge Eval — Results

Run date: 2026-04-06. 5-fold CV, 4 epochs, stratified by term.

## Dataset

| Stat | Count |
|------|-------|
| Canonical cases | 106 |
| Gold retrieved | 104 (98.1%) |
| Gold verified (reachable) | 86 (81.1%) |
| Gold verified AND in judge decision set | 25 (23.6%) |
| Counterexample cases | 113 |
| Counterexamples with candidates | 113 (100%) |
| Terms in vocabulary | 26 |

"Reachable" means the gold candidate survived composition, pruning, and
deduplication to appear in the judge's final decision set *and* passed
verification. 81 cases (106 − 25) have gold retrieved but not in the
judge-visible decision set — retrieval works, but the judge never sees
the gold option.

## Feature activation diagnostics

Total examples (candidate rows across all folds): 51,689.

### Dense features

| Idx | Feature | Activation % | Avg magnitude |
|-----|---------|-------------|---------------|
| 0 | bias | 100.0% | 1.0000 |
| 1 | acceptance_score | 100.0% | 0.3032 |
| 2 | phonetic_score | 100.0% | 0.3032 |
| 3 | coarse_score | 99.3% | 0.2754 |
| 4 | token_score | 83.3% | 0.1426 |
| 5 | feature_score | 100.0% | 0.4876 |
| 6 | feature_bonus | 99.9% | 0.3693 |
| 7 | best_view_score | 100.0% | 0.1450 |
| 8 | cross_view_support | 100.0% | 0.5191 |
| 9 | qgram_overlap | 100.0% | 0.1383 |
| 10 | total_qgram_overlap | 100.0% | 0.2306 |
| 11 | token_count_match | 32.3% | 1.0000 |
| 12 | phone_closeness | 100.0% | 0.3759 |
| 13 | alias_source_spoken | 32.7% | 1.0000 |
| 14 | alias_source_identifier | 34.6% | 1.0000 |
| 16 | identifier_acronym | 3.8% | 1.0000 |
| 17 | identifier_digits | 13.9% | 1.0000 |
| 18 | identifier_snake | 3.1% | 1.0000 |
| 19 | identifier_camel | 12.0% | 1.0000 |
| 20 | identifier_symbol | 15.2% | 1.0000 |
| 21 | short_guard_passed | 94.0% | 1.0000 |
| 22 | low_content_guard_passed | 95.3% | 1.0000 |
| 23 | acceptance_floor_passed | 3.7% | 1.0000 |
| 24 | verified | 3.4% | 1.0000 |
| 25 | span_token_count | 100.0% | 0.5470 |
| 26 | span_phone_count | 100.0% | 0.6908 |
| 27 | span_low_content | 4.7% | 1.0000 |
| 28 | span_mean_logprob | 58.7% | 0.0164 |
| 29 | span_min_logprob | 58.7% | 0.0398 |
| 30 | span_mean_margin | 65.2% | 1.8566 |
| 31 | span_min_margin | 65.2% | 1.2139 |

Feature 15 (`alias_source_confusion`) is not shown — 0% activation in this dataset.

**Key observations:**
- `verified` (idx 24) fires on only 3.4% of candidate rows. This is the strongest gold-vs-non-gold discriminator in reachable cases, but it's so rare that gradient updates to other features overwhelm it.
- `acceptance_floor_passed` (idx 23) is similarly rare at 3.7%.
- Scores (idx 1–10, 12) are always active and shared between gold and non-gold candidates — gradient updates to these features affect all candidates indiscriminately.
- ASR features (idx 28–31) activate on ~60% of examples. They're span-level (same for all candidates in a span), so they encode "is this span worth correcting?" rather than "is this candidate good?" This explains why +ASR is the only ablation that helps.
- Guards (idx 21–22) are nearly always on (94–95%), providing almost no discriminative signal.

### Sparse features

| Stat | Value |
|------|-------|
| Avg nonzero sparse per example | 9.1 |
| Unique sparse feature buckets | 13,751 (of 16,384) |

Top 5 sparse buckets by frequency:

| Bucket | Activations |
|--------|-------------|
| 4669 | 36,889 |
| 1349 | 14,835 |
| 9856 | 6,993 |
| 16108 | 6,650 |
| 4741 | 4,926 |

Sparse features ARE active — 9.1 nonzero per example, 84% of hash buckets
used. They encode context (neighboring tokens, term × context interactions).
Like ASR features they're span-level, so they share the same destructive
interference problem when used for candidate-level discrimination.

## Eval 1: Baselines

### Deterministic (acceptance_score ranking)

| Threshold | Can. Acc | Cx. Acc | Balanced | Can. Repl% | Cx. Repl% |
|-----------|----------|---------|----------|------------|-----------|
| 0.3 | 25/104 (24.0%) | 52/113 (46.0%) | 35.0% | 28.8% | 54.0% |
| 0.4 | 25/104 (24.0%) | 52/113 (46.0%) | 35.0% | 28.8% | 54.0% |
| 0.5 | 25/104 (24.0%) | 52/113 (46.0%) | 35.0% | 28.8% | 54.0% |
| 0.6 | 14/104 (13.5%) | 73/113 (64.6%) | 39.0% | 14.4% | 35.4% |
| 0.7 | 4/104 (3.8%) | 91/113 (80.5%) | 42.2% | 4.8% | 19.5% |
| 0.8 | 2/104 (1.9%) | 98/113 (86.7%) | 44.3% | 2.9% | 13.3% |

Best balanced: 44.3% at T=0.8. Deterministic uses ALL 104 canonical cases
(not just reachable) because it doesn't depend on the judge's decision set.

### Seed-only (FTRL with seed weights, no training)

| Threshold | Can. Acc | Cx. Acc | Balanced | Can. Repl% | Cx. Repl% |
|-----------|----------|---------|----------|------------|-----------|
| 0.1 | 7/25 (28.0%) | 0/113 (0.0%) | 14.0% | 100.0% | 100.0% |
| 0.2 | 7/25 (28.0%) | 0/113 (0.0%) | 14.0% | 96.0% | 100.0% |
| 0.3 | 7/25 (28.0%) | 0/113 (0.0%) | 14.0% | 92.0% | 100.0% |
| 0.4 | 7/25 (28.0%) | 0/113 (0.0%) | 14.0% | 84.0% | 100.0% |
| 0.5 | 7/25 (28.0%) | 0/113 (0.0%) | 14.0% | 80.0% | 100.0% |
| 0.6 | 7/25 (28.0%) | 1/113 (0.9%) | 14.4% | 68.0% | 99.1% |
| 0.7 | 6/25 (24.0%) | 9/113 (8.0%) | 16.0% | 60.0% | 92.0% |
| 0.8 | 4/25 (16.0%) | 17/113 (15.0%) | 15.5% | 48.0% | 85.0% |
| 0.9 | 0/25 (0.0%) | 87/113 (77.0%) | 38.5% | 16.0% | 23.0% |

Seed-only replaces aggressively at all thresholds below 0.9. At T=0.1–0.5,
it replaces 80–100% of everything (cx included) but gets 28% canonical
right (7/25). This means **the seed model has moderate ranking quality but
no abstention ability** — it scores most candidates above threshold.

### Taught (teach_choice replay, current system)

| Threshold | Can. Acc | Cx. Acc | Balanced | Can. Repl% | Cx. Repl% |
|-----------|----------|---------|----------|------------|-----------|
| 0.1 | 7/25 (28.0%) | 20/113 (17.7%) | 22.8% | 48.0% | 82.3% |
| 0.2 | 4/25 (16.0%) | 49/113 (43.4%) | 29.7% | 20.0% | 56.6% |
| 0.3 | 4/25 (16.0%) | 83/113 (73.5%) | 44.7% | 20.0% | 26.5% |
| 0.4 | 4/25 (16.0%) | 97/113 (85.8%) | 50.9% | 16.0% | 14.2% |
| 0.5 | 2/25 (8.0%) | 103/113 (91.2%) | 49.6% | 8.0% | 8.8% |
| 0.6 | 2/25 (8.0%) | 110/113 (97.3%) | 52.7% | 8.0% | 2.7% |
| 0.7 | 1/25 (4.0%) | 113/113 (100.0%) | 52.0% | 4.0% | 0.0% |
| 0.8 | 1/25 (4.0%) | 113/113 (100.0%) | 52.0% | 4.0% | 0.0% |
| 0.9 | 1/25 (4.0%) | 113/113 (100.0%) | 52.0% | 4.0% | 0.0% |

Best balanced: 52.7% at T=0.6. Training improved cx_acc (from 0% → 97.3%)
but destroyed canonical accuracy (from 28% → 8%). **Training is learning to
reject everything.**

## Eval 2: Case-balanced FTRL

| Threshold | Can. Acc | Cx. Acc | Balanced | Can. Repl% | Cx. Repl% |
|-----------|----------|---------|----------|------------|-----------|
| 0.1 | 8/25 (32.0%) | 5/113 (4.4%) | 18.2% | 60.0% | 95.6% |
| 0.2 | 5/25 (20.0%) | 40/113 (35.4%) | 27.7% | 28.0% | 64.6% |
| 0.3 | 3/25 (12.0%) | 82/113 (72.6%) | 42.3% | 12.0% | 27.4% |
| 0.4 | 2/25 (8.0%) | 104/113 (92.0%) | 50.0% | 8.0% | 8.0% |
| 0.5 | 2/25 (8.0%) | 112/113 (99.1%) | 53.6% | 8.0% | 0.9% |
| 0.6 | 0/25 (0.0%) | 113/113 (100.0%) | 50.0% | 0.0% | 0.0% |
| 0.7+ | 0/25 (0.0%) | 113/113 (100.0%) | 50.0% | 0.0% | 0.0% |

Best balanced: 53.6% at T=0.5. Marginally better than taught (52.7%) but
the same fundamental problem: training suppresses all probabilities.

## Eval 4: Feature ablation (case-balanced, best threshold per slice)

| Slice | Best T | Can. Acc | Cx. Acc | Balanced | Can. Repl% | Cx. Repl% |
|-------|--------|----------|---------|----------|------------|-----------|
| phonetic_only (idx 0–27) | 0.4 | 0/25 (0.0%) | 113/113 (100.0%) | 50.0% | 0.0% | 0.0% |
| +asr (idx 0–31) | 0.2 | 8/25 (32.0%) | 106/113 (93.8%) | **62.9%** | 44.0% | 6.2% |
| +context (idx 0–31 + sparse) | 0.5 | 2/25 (8.0%) | 112/113 (99.1%) | 53.6% | 8.0% | 0.9% |
| all (idx 0–37 + sparse) | 0.5 | 2/25 (8.0%) | 112/113 (99.1%) | 53.6% | 8.0% | 0.9% |

**+ASR is the only slice that helps** (62.9% balanced), and adding context
or memory *hurts* — suggesting that sparse context features and memory
features introduce noise that overwhelms the ASR signal during training.

Phonetic-only collapses completely — the model learns to never replace.
This confirms that phonetic scores alone, after training updates, provide
no usable candidate discrimination.

## Eval 5: Reachable-only (case-balanced)

Identical to Eval 2. This is correct: the regular eval already uses only
the 25 reachable canonical cases (gold verified AND in judge decision set).
Reachable-only applies the same filter, so the numbers match.

## Eval 6: Formulation comparison (best threshold each)

| Formulation | Best T | Can. Acc | Cx. Acc | Balanced | Can. Repl% | Cx. Repl% |
|-------------|--------|----------|---------|----------|------------|-----------|
| independent_binary (taught) | 0.6 | 2/25 (8.0%) | 110/113 (97.3%) | 52.7% | 8.0% | 2.7% |
| case_balanced | 0.5 | 2/25 (8.0%) | 112/113 (99.1%) | 53.6% | 8.0% | 0.9% |
| freeze_dense | 0.9 | 0/25 (0.0%) | 111/113 (98.2%) | 49.1% | 8.0% | 1.8% |
| casewise_softmax | 0.8 | 0/25 (0.0%) | 113/113 (100.0%) | 50.0% | 0.0% | 0.0% |

All formulations converge to "reject everything." Softmax is worst (zero
replacement). Freeze-dense (train only sparse + ASR, preserve seed phonetic
weights) also fails — sparse features alone don't carry enough signal.

## Probability distributions (case-balanced model)

| Population | n | min | p25 | p50 | p75 | max |
|------------|---|-----|-----|-----|-----|-----|
| Gold prob (canonical, gold = best candidate) | 11 | 0.063 | 0.124 | 0.176 | 0.319 | 0.522 |
| Best-non-gold prob (canonical, gold ≠ best) | 14 | 0.025 | 0.070 | 0.109 | 0.161 | 0.240 |
| Top negative prob (counterexample) | 113 | 0.034 | 0.177 | 0.241 | 0.313 | 0.556 |

**In 14 of 25 canonical cases (56%), gold is NOT the best-scoring candidate.**
The model's ranking is worse than random for the accept/reject decision.

Gold median prob: 0.176. Cx top median prob: 0.241. **Counterexample
candidates score higher than gold candidates.** No threshold can separate
them — the distributions are interleaved.

## One-case training trace

Starting from seed weights, train on exactly 1 canonical case (term: wasm)
then 1 counterexample case (term: qwen). Monitor probability changes.

### teach_choice

| State | Gold prob | Cx best prob |
|-------|-----------|-------------|
| Before training | 0.4076 | 0.1602 |
| After 1 canonical | 0.7604 (+0.353) | 0.1885 (+0.028) |
| After 1 canonical + 1 cx | 0.5817 (−0.179) | 0.0092 (−0.179) |

Weight L2 norm: 4.33. Active features: 100.

Canonical training lifts gold substantially (+0.35) but also lifts the
unrelated cx candidate (+0.03) due to shared features. Counterexample
training then pushes down BOTH gold (−0.18) and cx (−0.18) by the same
amount. **The cx gradient is indiscriminate.**

### case_balanced

| State | Gold prob | Cx best prob |
|-------|-----------|-------------|
| Before training | 0.4076 | 0.1602 |
| After 1 canonical | 0.4162 (+0.009) | 0.1222 (−0.038) |
| After 1 canonical + 1 cx | 0.3825 (−0.034) | 0.0686 (−0.054) |

Weight L2 norm: 4.00. Active features: 61.

Case-balanced makes smaller updates (good) but the direction is still wrong:
cx training lowers gold prob (−0.034) nearly as much as cx prob (−0.054).
The hard-negative-only strategy doesn't fix feature overlap.

## Root cause analysis

### Why training destroys the model

The FTRL model has 38 dense features. Of these:

- **10 always-active continuous scores** (idx 1–10, 12): phonetic similarity,
  q-gram overlap, token scores, etc. These are active on 100% of candidates —
  both gold and non-gold, both canonical and counterexample spans. When
  counterexample training pushes down a negative candidate, the gradient flows
  through these shared features and suppresses ALL candidate scores globally.

- **2 discriminative binary features** (idx 23–24): `acceptance_floor_passed`
  and `verified`. These fire on only 3.4–3.7% of candidate rows. In reachable
  cases, `verified=1` perfectly identifies gold — but it's so sparse that its
  gradient contribution is dwarfed by the 10 always-on features.

- **4 span-level features** (idx 28–31): ASR uncertainty. These don't
  distinguish candidates within a span, but DO distinguish "spans worth
  correcting" from "spans that are fine." This is why +ASR is the only
  ablation that improves balanced accuracy.

### The fundamental mismatch

The model is asked to solve two problems simultaneously:
1. **Should this span be corrected?** (span-level decision)
2. **Which candidate is best?** (candidate-level ranking)

But features are mixed: span-level features (ASR, context) and
candidate-level features (scores, verified) are in the same linear model.
Training for problem 2 (push down bad candidates) corrupts the weights for
problem 1 (and vice versa).

### What the seed model gets right

The seed model (no training) achieves 28% canonical accuracy at all thresholds
≤ 0.5, with 80–100% replacement rate. It replaces too aggressively (100% cx
replacement at T ≤ 0.5) but at least it DOES replace, and gets the right
answer 28% of the time. Every trained variant does worse on canonical accuracy.

## Possible directions

1. **Interaction features** — e.g., `verified × acceptance_score`,
   `verified × phonetic_score` — so the model can learn "high-scoring
   verified candidates are good" without that gradient flowing through
   bare `acceptance_score`.

2. **Two-stage architecture** — Stage 1: "should this span be corrected?"
   (using ASR + context features only, span-level binary classifier).
   Stage 2: "which candidate?" (using candidate-level features only,
   ranking within spans classified as "yes" by Stage 1).

3. **Stronger verified seed weight** — Since `verified=1` is nearly a
   perfect gold indicator for reachable cases, increase its seed weight
   from 0.30 to 2.0+ so training can't easily dilute it.

4. **Candidate-relative features** — Rank of this candidate among peers
   (is it the best? 2nd best?), score margin over 2nd-best, etc. These
   would be candidate-specific without sharing values across candidates.
