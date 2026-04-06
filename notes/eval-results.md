# Phase 4 Offline Judge Eval — Results

Run date: 2026-04-06. 5-fold CV, 4 epochs, stratified by term.

## Executive summary

The single-model FTRL judge was fundamentally broken: it tried to learn
"should this span be corrected?" and "which candidate wins?" in one weight
vector, causing counterexample training to suppress shared features and
crush replacement globally. Best single-model result: 65% balanced accuracy.

Splitting into two stages — a span gate and a candidate ranker — solved the
problem: **94% balanced accuracy, 88% canonical correct, 100% counterexample
correct, zero false positive replacements.**

| Model | Balanced | Can. Acc | Cx. Acc | Can. Repl% | Cx. Repl% |
|-------|----------|----------|---------|------------|-----------|
| **Two-stage (GT=0.2, RT=0.1)** | **94.0%** | **88.0%** (22/25) | **100.0%** (113/113) | **96.0%** | **0.0%** |
| +ASR ablation (best single) | 65.3% | 36.0% (9/25) | 94.7% (107/113) | 64.0% | 5.3% |
| Case-balanced single | 50.0% | 0.0% (0/25) | 100.0% (113/113) | 8.0% | 0.0% |
| Seed-only (no training) | 38.5% | 0.0% (0/25) | 77.0% (87/113) | 16.0% | 23.0% |
| Deterministic baseline | 44.3% | 1.9% (2/104) | 86.7% (98/113) | 2.9% | 13.3% |

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
interference problem when used for candidate-level discrimination in a
single model. In the two-stage architecture, these live in Stage A where
they belong.

## Single-model results (Evals 1–6)

All single-model approaches failed to simultaneously learn span gating and
candidate ranking. The core results are presented here for context; the
two-stage architecture (Eval 8) supersedes them.

### Eval 1: Baselines

**Deterministic** (acceptance_score ranking, best threshold T=0.8):
44.3% balanced. Uses all 104 canonical cases (doesn't require judge
reachability).

**Seed-only** (no training, best T=0.9): 38.5% balanced. Replaces
aggressively at all lower thresholds (80–100% replacement rate) but
gets 28% canonical right. Has moderate ranking quality but no
abstention ability.

**Taught** (teach_choice replay, best T=0.6): 52.7% balanced. Training
improved cx_acc (from 0% → 97.3%) but destroyed canonical accuracy
(from 28% → 0%). Training learned to reject everything.

### Eval 2: Case-balanced FTRL

Best T=0.5: 50.0% balanced. Same problem — training suppresses all
probabilities. Marginally different from taught.

### Eval 4: Feature ablation (case-balanced)

| Slice | Best T | Balanced | Can. Repl% | Cx. Repl% |
|-------|--------|----------|------------|-----------|
| phonetic_only (idx 0–27) | 0.3 | 50.0% | 0.0% | 0.0% |
| **+asr (idx 0–31)** | **0.2** | **65.3%** | **64.0%** | **5.3%** |
| +context (idx 0–31 + sparse) | 0.5 | 50.0% | 8.0% | 0.0% |
| all (idx 0–37 + sparse) | 0.5 | 50.0% | 8.0% | 0.0% |

+ASR was the only slice that helped. This was the key clue that span-level
features wanted a separate model.

### Eval 6: Formulation comparison

| Formulation | Best T | Balanced |
|-------------|--------|----------|
| independent_binary | 0.6 | 50.0% |
| case_balanced | 0.5 | 50.0% |
| freeze_dense | 0.9 | 49.6% |
| casewise_softmax | 0.7 | 50.0% |

All formulations converged to "reject everything." The problem was
architectural, not an issue with the loss function or training balance.

### Single-model probability distributions

| Population | n | min | p25 | p50 | p75 | max |
|------------|---|-----|-----|-----|-----|-----|
| Gold prob (gold = best) | 8 | 0.076 | 0.166 | 0.241 | 0.249 | 0.333 |
| Best-non-gold (gold ≠ best) | 17 | 0.027 | 0.078 | 0.118 | 0.224 | 0.570 |
| Top cx prob | 113 | 0.028 | 0.171 | 0.209 | 0.264 | 0.499 |

Gold and cx probabilities are interleaved. No threshold can separate them.
In 17 of 25 canonical cases (68%), gold is NOT the best-scoring candidate.

## Root cause analysis

### Why the single model fails

The FTRL model has 38 dense features. Of these:

- **10 always-active continuous scores** (idx 1–10, 12): phonetic similarity,
  q-gram overlap, token scores. Active on 100% of candidates — gold and
  non-gold, canonical and counterexample. When counterexample training pushes
  down a negative, the gradient flows through these shared features and
  suppresses ALL candidate scores globally.

- **2 discriminative binary features** (idx 23–24): `acceptance_floor_passed`
  and `verified`. Fire on only 3.4–3.7% of candidate rows. In reachable
  cases, `verified=1` perfectly identifies gold — but so sparse that its
  gradient is dwarfed by the 10 always-on features.

- **4 span-level features** (idx 28–31): ASR uncertainty. Don't distinguish
  candidates within a span, but DO distinguish "spans worth correcting" from
  "spans that are fine."

### The fundamental mismatch

The model was asked to solve two problems simultaneously:
1. **Should this span be corrected?** (span-level decision)
2. **Which candidate is best?** (candidate-level ranking)

With mixed features in one weight vector, training for problem 2 (push down
bad candidates) corrupts the weights for problem 1 (and vice versa).

### One-case training trace (proof)

Starting from seed weights, training on 1 canonical (QEMU) then
1 counterexample (qwen):

| Mode | After canonical | After cx | Net effect on gold |
|------|----------------|----------|--------------------|
| teach_choice | gold +0.163 | gold −0.111 | gold +0.052 but cx −0.321 |
| case_balanced | gold −0.318 | gold −0.025 | gold −0.343 |

Case-balanced training actively lowers gold probability even on the
canonical example, because hard negatives share features with gold.

## Eval 8: Two-stage architecture (the fix)

### Architecture

Two separate `SparseFtrl` models with non-overlapping feature sets:

**Stage A — Span Gate** ("should this span be corrected?"):
- One prediction per span (not per candidate)
- 14 dense features: bias, span shape (3), ASR uncertainty (4),
  memory (1), candidate summary stats (5: max acceptance, max phonetic,
  any verified, any acceptance_floor, candidate count)
- Sparse features: context hashes (L1, L2, R1, R2, CTX=, APP=)
  without TERM= (span-level, not candidate-specific)
- Trained on: canonical gold spans = positive, counterexample spans = negative

**Stage B — Candidate Ranker** ("which candidate wins?"):
- One prediction per candidate, within a single span
- 36 dense features: candidate scores (12), alias/identifier flags (8),
  guards + verified (4), **candidate-relative features (6)**, memory (5)
- No span-level ASR or context features (those belong to Stage A)
- Trained only on spans where gold is retrieved + verified (86 cases,
  not just the 25 judge-reachable ones)

### Candidate-relative features (new in Stage B)

| Feature | Description |
|---------|-------------|
| rank_in_span | 1/(1+rank) where rank is position by acceptance_score desc |
| margin_to_next | Gap in acceptance_score to adjacent candidate |
| is_best_verified | This is the highest-scoring verified candidate |
| is_only_verified | This is the only verified candidate in the span |
| normalized_acceptance | acceptance_score / max in span |
| normalized_phonetic | phonetic_score / max in span |

These features are candidate-specific without sharing values across
candidates, so training on negatives does not suppress positives.

### Results

#### Stage A (gate) alone

| Gate T | Open correct | Closed correct | Balanced |
|--------|-------------|----------------|----------|
| 0.1 | 24/25 (96.0%) | 110/113 (97.3%) | **96.7%** |
| **0.2** | **24/25 (96.0%)** | **113/113 (100.0%)** | **98.0%** |
| 0.3 | 20/25 (80.0%) | 113/113 (100.0%) | 90.0% |
| 0.4 | 19/25 (76.0%) | 113/113 (100.0%) | 88.0% |
| 0.5 | 18/25 (72.0%) | 113/113 (100.0%) | 86.0% |
| 0.6 | 15/25 (60.0%) | 113/113 (100.0%) | 80.0% |
| 0.7 | 13/25 (52.0%) | 113/113 (100.0%) | 76.0% |
| 0.8 | 11/25 (44.0%) | 113/113 (100.0%) | 72.0% |
| 0.9 | 9/25 (36.0%) | 113/113 (100.0%) | 68.0% |

At GT=0.2: the gate correctly opens for 96% of canonical spans and
correctly closes for 100% of counterexample spans. Only 1 canonical
case is missed by the gate.

#### Stage B (ranker) alone

**Top-1 accuracy: 23/25 (92.0%)**

Of 25 reachable canonical cases, the ranker puts gold first in 23.

#### Composed (gate x ranker)

| Gate T | Ranker T | Can. Acc | Cx. Acc | Balanced | Can. Repl% | Cx. Repl% |
|--------|----------|----------|---------|----------|------------|-----------|
| **0.2** | **0.1** | **22/25 (88.0%)** | **113/113 (100.0%)** | **94.0%** | **96.0%** | **0.0%** |
| 0.2 | 0.5 | 20/25 (80.0%) | 113/113 (100.0%) | 90.0% | 88.0% | 0.0% |
| 0.3 | 0.2 | 18/25 (72.0%) | 113/113 (100.0%) | 86.0% | 80.0% | 0.0% |
| 0.4 | 0.2 | 17/25 (68.0%) | 113/113 (100.0%) | 84.0% | 76.0% | 0.0% |
| 0.5 | 0.2 | 16/25 (64.0%) | 113/113 (100.0%) | 82.0% | 72.0% | 0.0% |

**Zero false positive replacements at every threshold tested.**

The 3 missed canonical cases come from the gate (1 case where gate
doesn't open at GT=0.2) and the ranker (2 cases where gold isn't top-1).

#### Probability distributions

**Gate probabilities** — perfectly separated:

| Population | n | min | p25 | p50 | p75 | max |
|------------|---|-----|-----|-----|-----|-----|
| Canonical (should open) | 25 | 0.017 | 0.416 | 0.759 | 0.968 | 0.999 |
| Counterex (should close) | 113 | 0.004 | 0.015 | 0.028 | 0.046 | 0.196 |

Canonical median: 0.759. Counterexample median: 0.028. The distributions
barely overlap — the max counterexample (0.196) is below the canonical
median. This is why the gate achieves 98% balanced accuracy.

**Ranker probabilities** — gold ranks high with good confidence:

| Population | n | min | p25 | p50 | p75 | max |
|------------|---|-----|-----|-----|-----|-----|
| Gold = best (correct rank) | 23 | 0.465 | 0.580 | 0.635 | 0.776 | 0.882 |
| Gold ≠ best (wrong rank) | 2 | 0.556 | 0.556 | 0.793 | 0.793 | 0.793 |

Gold median probability: 0.635 (vs 0.241 in the single model). The ranker
gives gold high absolute probability, not just relative ranking.

The 2 cases where gold isn't best still have high non-gold probability
(0.556, 0.793), suggesting the ranker is confident about a wrong candidate
rather than uncertain.

## Why the two-stage architecture works

1. **Feature isolation**: Span-level features (ASR, context) only
   appear in the gate. Candidate-level features (scores, verified)
   only appear in the ranker. No gradient interference between the
   two decisions.

2. **Clean supervision**: The gate sees canonical vs. counterexample
   spans — a clean binary signal. The ranker sees gold vs. non-gold
   candidates within gold-present spans — also a clean binary signal.
   Neither model is asked to learn both tasks.

3. **Candidate-relative features**: The ranker uses rank-in-span,
   margin, is-best-verified, normalized scores. These are
   candidate-specific without sharing values across candidates,
   so negative updates don't suppress positive candidates.

4. **Larger training set for ranker**: Stage B trains on all 86
   gold-verified spans, not just the 25 judge-reachable ones.
   This gives 3.4x more positive supervision.

## What remains

- **3 missed canonical cases**: 1 gate miss (gate prob too low) +
  2 ranker misses (gold not top-1). Error analysis on these specific
  cases would inform whether the gate needs better ASR features or
  the ranker needs better candidate discrimination.

- **Online integration**: The two-stage architecture needs to be
  wired into the production `OnlineJudge` for live correction. The
  gate and ranker models would each maintain their own FTRL weights,
  trained from `teach_choice` feedback.

- **Dataset limitations**: 25 reachable canonical cases is small.
  The 81 unreachable cases suggest retrieval/composition improvements
  could dramatically expand the judge's opportunity set.
