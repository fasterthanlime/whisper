# Phase 4 Offline Judge Eval — Results

Run date: 2026-04-06. 5-fold CV, 4 epochs, stratified by term.

## Executive summary

The single-model FTRL judge was fundamentally broken: it tried to learn
"should this span be corrected?" and "which candidate wins?" in one weight
vector, causing counterexample training to suppress shared features and
crush replacement globally. Best single-model result: 64.5% balanced accuracy.

Splitting into two stages — a span gate and a candidate ranker — solved the
problem: **92.1% balanced accuracy over 86 reachable canonical cases,
with 5.3% false positive replacement rate.**

| Model | Balanced | Can. Acc | Cx. Acc | Can. Repl% | Cx. Repl% |
|-------|----------|----------|---------|------------|-----------|
| **Two-stage (GT=0.3, RT=0.1)** | **92.1%** | **89.5%** (77/86) | **94.7%** (107/113) | **97.7%** | **5.3%** |
| Two-stage (GT=0.4, RT=0.2) | 91.6% | 84.9% (73/86) | 98.2% (111/113) | 93.0% | 1.8% |
| Two-stage (GT=0.5, RT=0.2) | 91.3% | 82.6% (71/86) | 100.0% (113/113) | 90.7% | 0.0% |
| +ASR ablation (best single) | 64.5% | 29.1% (25/86) | 100.0% (113/113) | 45.3% | 0.0% |
| Case-balanced single | 53.3% | 9.3% (8/86) | 97.3% (110/113) | 17.4% | 2.7% |
| Deterministic baseline | 63.4% | 80.8% (84/104) | 46.0% (52/113) | 84.6% | 54.0% |

The two-stage architecture offers a smooth precision/recall tradeoff: at
GT=0.5, RT=0.2, it achieves zero false positive replacements while still
correcting 82.6% of canonical cases.

## Dataset

| Stat | Count |
|------|-------|
| Canonical cases | 106 |
| Gold retrieved | 104 (98.1%) |
| Gold verified (reachable) | 86 (81.1%) |
| Counterexample cases | 113 |
| Counterexamples with candidates | 113 (100%) |
| Terms in vocabulary | 26 |

"Reachable" means the gold candidate was retrieved AND passed verification
in at least one span for that case. 20 canonical cases (106 − 86) have gold
either not retrieved (2) or retrieved but not verified (18).

**Bug fix (2026-04-06):** An earlier version of the eval reported only 25
reachable canonical cases. This was caused by `find(|ps| ps.gold_alias_id.is_some())`
selecting the first span with gold, which often had an unverified gold
candidate even when a different span for the same case had verified gold.
Fixed by preferring spans where gold is verified (`find_best_gold_span`).

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

**Deterministic** (acceptance_score ranking, best threshold T=0.3):
63.4% balanced (84/104 canonical, 52/113 cx). Uses all 104 canonical
cases (doesn't require verification). High canonical accuracy but
replaces 54% of counterexamples — no abstention ability.

**Seed-only** (no training, best T=0.9): 40.2% balanced. Replaces
aggressively at lower thresholds (84–100% replacement rate). Has
moderate ranking quality but no abstention ability.

**Taught** (teach_choice replay, best T=0.5): 53.7% balanced. Training
improved cx_acc but destroyed canonical accuracy. Training learned to
reject almost everything.

### Eval 2: Case-balanced FTRL

Best T=0.5: 53.3% balanced. Same problem — training suppresses all
probabilities. 8/86 canonical correct.

### Eval 4: Feature ablation (case-balanced)

| Slice | Best T | Balanced | Can. Acc | Cx. Acc |
|-------|--------|----------|----------|---------|
| phonetic_only (idx 0–27) | 0.5 | 50.5% | 15.1% | 85.8% |
| **+asr (idx 0–31)** | **0.5** | **64.5%** | **29.1%** | **100.0%** |
| +context (idx 0–31 + sparse) | 0.5 | 53.3% | 9.3% | 97.3% |
| all (idx 0–37 + sparse) | 0.5 | 53.3% | 9.3% | 97.3% |

+ASR was the only slice that helped. This was the key clue that span-level
features wanted a separate model.

### Eval 6: Formulation comparison

| Formulation | Best T | Balanced |
|-------------|--------|----------|
| independent_binary | 0.5 | 53.7% |
| case_balanced | 0.5 | 53.3% |
| freeze_dense | 0.9 | 50.0% |
| casewise_softmax | 0.9 | 47.8% |

All formulations converged to "reject everything." The problem was
architectural, not an issue with the loss function or training balance.

### Single-model probability distributions

| Population | n | min | p25 | p50 | p75 | max |
|------------|---|-----|-----|-----|-----|-----|
| Gold prob (gold = best) | 48 | 0.039 | 0.159 | 0.253 | 0.424 | 0.769 |
| Best-non-gold (gold ≠ best) | 38 | 0.048 | 0.083 | 0.263 | 0.434 | 0.743 |
| Top cx prob | 113 | 0.023 | 0.188 | 0.267 | 0.351 | 0.562 |

Gold and cx probabilities are interleaved. No threshold can separate them.
In 38 of 86 canonical cases (44%), gold is NOT the best-scoring candidate.

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

Starting from seed weights, training on 1 canonical (tokio) then
1 counterexample (reqwest):

| Mode | After canonical | After cx | Net effect on gold |
|------|----------------|----------|--------------------|
| teach_choice | gold −0.755 | gold −0.159 | gold −0.914 |
| case_balanced | gold −0.865 | gold −0.017 | gold −0.882 |

Both modes actively lower gold probability even on the canonical example,
because hard negatives share features with gold.

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
- Trained only on spans where gold is retrieved + verified (86 cases)

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
candidates, so training on negatives does not suppress positive candidates.

### Results

#### Stage A (gate) alone

| Gate T | Open correct | Closed correct | Balanced |
|--------|-------------|----------------|----------|
| 0.1 | 86/86 (100.0%) | 70/113 (61.9%) | 81.0% |
| 0.2 | 86/86 (100.0%) | 92/113 (81.4%) | 90.7% |
| **0.3** | **84/86 (97.7%)** | **107/113 (94.7%)** | **96.2%** |
| 0.4 | 80/86 (93.0%) | 111/113 (98.2%) | 95.6% |
| 0.5 | 78/86 (90.7%) | 113/113 (100.0%) | 95.3% |
| 0.6 | 76/86 (88.4%) | 113/113 (100.0%) | 94.2% |
| 0.7 | 65/86 (75.6%) | 113/113 (100.0%) | 87.8% |
| 0.8 | 55/86 (64.0%) | 113/113 (100.0%) | 82.0% |
| 0.9 | 44/86 (51.2%) | 113/113 (100.0%) | 75.6% |

At GT=0.3: the gate correctly opens for 97.7% of canonical spans and
correctly closes for 94.7% of counterexample spans. Best balanced
accuracy: 96.2%.

#### Stage B (ranker) alone

**Top-1 accuracy: 78/86 (90.7%)**

Of 86 reachable canonical cases, the ranker puts gold first in 78.

#### Composed (gate x ranker)

| Gate T | Ranker T | Can. Acc | Cx. Acc | Balanced | Can. Repl% | Cx. Repl% |
|--------|----------|----------|---------|----------|------------|-----------|
| **0.3** | **0.1** | **77/86 (89.5%)** | **107/113 (94.7%)** | **92.1%** | **97.7%** | **5.3%** |
| 0.4 | 0.2 | 73/86 (84.9%) | 111/113 (98.2%) | 91.6% | 93.0% | 1.8% |
| 0.5 | 0.2 | 71/86 (82.6%) | 113/113 (100.0%) | 91.3% | 90.7% | 0.0% |
| 0.2 | 0.2 | 78/86 (90.7%) | 92/113 (81.4%) | 86.1% | 100.0% | 18.6% |

The architecture provides a smooth precision/recall tradeoff:
- **Aggressive** (GT=0.3, RT=0.1): 89.5% can, 94.7% cx, 5.3% false positive rate
- **Balanced** (GT=0.4, RT=0.2): 84.9% can, 98.2% cx, 1.8% false positive rate
- **Conservative** (GT=0.5, RT=0.2): 82.6% can, 100% cx, 0% false positive rate

#### Probability distributions

**Gate probabilities** — well separated:

| Population | n | min | p25 | p50 | p75 | max |
|------------|---|-----|-----|-----|-----|-----|
| Canonical (should open) | 86 | 0.219 | 0.710 | 0.903 | 0.979 | 1.000 |
| Counterex (should close) | 113 | 0.002 | 0.013 | 0.057 | 0.167 | 0.441 |

Canonical median: 0.903. Counterexample median: 0.057. The distributions
barely overlap — the max counterexample (0.441) is below the canonical p25
(0.710). This is why the gate achieves 96.2% balanced accuracy.

**Ranker probabilities** — gold ranks high with good confidence:

| Population | n | min | p25 | p50 | p75 | max |
|------------|---|-----|-----|-----|-----|-----|
| Gold = best (correct rank) | 78 | 0.566 | 0.747 | 0.793 | 0.851 | 0.949 |
| Gold ≠ best (wrong rank) | 8 | 0.589 | 0.641 | 0.719 | 0.757 | 0.895 |

Gold median probability: 0.793 (vs 0.253 in the single model). The ranker
gives gold high absolute probability, not just relative ranking.

The 8 cases where gold isn't best have moderate non-gold probability
(median 0.719), suggesting the ranker is confident about a wrong candidate
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

4. **Full training set**: Stage B trains on all 86 gold-verified spans,
   not limited by downstream pipeline reachability.

## What remains

- **9 missed canonical cases**: 2 gate misses (gate prob too low at GT=0.3) +
  8 ranker misses (gold not top-1) − 1 overlap = 9 total. Error analysis on
  these specific cases would inform whether the gate needs better ASR
  features or the ranker needs better candidate discrimination.

- **Online integration**: The two-stage architecture needs to be wired into
  the production `OnlineJudge` for live correction. The gate and ranker
  models would each maintain their own FTRL weights, trained from
  `teach_choice` feedback.

- **End-to-end vs judge-stage**: The 92.1% balanced accuracy is a
  judge-stage metric — it measures how well the judge performs on the
  86 cases where gold reaches it. End-to-end accuracy (78/106 = 73.6%
  at GT=0.3) is lower because 20 cases never reach the judge due to
  retrieval or verification failures. Improving the retrieval and
  verification pipeline would expand the judge's opportunity set.
