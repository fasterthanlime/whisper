# Phase 4 Offline Judge Eval Plan

## Current state

The offline eval (`--offline-eval`) runs k-fold CV using `teach_choice` replay.
Results: **canonical 2/104 (1.9%), counterexample 108/113 (95.6%)**.

The judge almost never accepts a replacement. Root cause: `teach_choice(None)`
(counterexample = keep original) trains *every* candidate as a negative example.
With ~113 counterexamples × multiple candidates each, reject signals massively
outnumber accept signals.

## Metrics reported everywhere

Every eval reports these per model/config:

| Metric | Definition |
|--------|------------|
| **canonical_acc** | % of canonical cases where judge picks gold |
| **cx_acc** | % of counterexample cases where judge abstains |
| **balanced_acc** | (canonical_acc + cx_acc) / 2 |
| **canonical_replace_rate** | % of canonical cases where *any* replacement was made |
| **cx_replace_rate** | % of counterexample cases where *any* replacement was made |

Replacement rate makes "reject everything" visible immediately. A model can get
decent cx_acc just by barely ever firing, so replace_rate is the sanity check.

## Eval 1: Baselines (seed-only, deterministic, taught)

Three scoring modes on the **same candidate decision set** per case, varying
only the scoring rule. All three see the same `ProbedSpan` candidates for each
test case — no different composition or thresholding surfaces.

| Mode | Description |
|------|-------------|
| **Deterministic** | Rank by `acceptance_score > phonetic_score > coarse_score` (current `compare_hits` ordering). Threshold: `verified && acceptance_score >= T` (sweep T). Uses the same candidate list as the judge modes. |
| **Seed-only** | Fresh `OnlineJudge::default()` (seed weights, no teaching). Score via `score_candidates` on the same candidates. |
| **Taught** | Current k-fold teach_choice replay (what we have now), same candidates. |

Purpose: does learning help at all? If seed-only beats taught, the update
regime or loss is broken.

### Implementation

In `run_offline_judge_eval`, for each fold, score the test set three ways
using the same `ProbedSpan` candidates:
1. Deterministic: rank candidates by `acceptance_score > phonetic_score > coarse_score`,
   accept top candidate if `verified && acceptance_score >= T`.
2. Seed-only: `OnlineJudge::default()`, no training, just `score_candidates`.
3. Taught: existing code path.

Report all three per fold + aggregate.

## Eval 2: Case-balanced offline FTRL

**Highest priority.** Replace `teach_choice` replay with balanced training
where each case — canonical or counterexample — has roughly equal influence.

- **Canonical case**: 1 positive (gold alias) + up to k=3 hard negatives
  (highest-scoring non-gold candidates from the same span).
- **Counterexample case**: 1 unit of weight total. Take the **single
  highest-scoring candidate** (the hardest false positive) and train it as
  one negative. Not averaged, not smeared across all candidates.

The point is to stop "all candidates negative forever." One hard negative
per counterexample case is the default first pass.

### Implementation

Add a `train_balanced` method (or free function taking `&mut SparseFtrl`):

```
fn train_balanced(
    model: &mut SparseFtrl,
    span: &TranscriptSpan,
    candidates: &[(CandidateFeatureRow, IdentifierFlags)],
    gold_alias_id: Option<u32>,  // None = keep original
    ctx: &SpanContext,
    hard_neg_cap: usize,         // e.g. 3
)
```

For canonical (gold_alias_id = Some):
- Build features for gold candidate, call `model.update(features, true)`
- Sort remaining candidates by score desc, take top `hard_neg_cap`
- For each: `model.update(features, false)`

For counterexample (gold_alias_id = None):
- Take the single top-scoring candidate (by current model score)
- `model.update(features, false)` once

## Eval 3: Threshold / abstention sweep

For each trained model (seed-only, taught, case-balanced), sweep:
- `accept_threshold` in `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`
- Optionally: margin threshold (gap between top candidate prob and keep_original)

Report per threshold: canonical_acc, cx_acc, balanced_acc, canonical_replace_rate,
cx_replace_rate.

This answers: "does the model have usable ranking but bad thresholding?"

### Implementation

`score_candidates` already returns probabilities. Sweep the threshold when
evaluating test cases instead of hardcoding 0.5. Report a table:

```
threshold  can_acc  cx_acc  balanced  can_repl%  cx_repl%
0.1        85.0%    20.0%   52.5%     95.0%      80.0%
0.3        60.0%    70.0%   65.0%     70.0%      30.0%
0.5         1.9%    95.6%   48.8%      2.0%       4.4%  ← current
```

## Eval 4: Feature ablation ladder

Run on **case-balanced binary** (Eval 2), not the broken independent-binary
replay. Otherwise you measure the objective bug more than the feature value.

Train case-balanced FTRL with these feature subsets (at best threshold from Eval 3):

| Slice | Features included |
|-------|-------------------|
| **Phonetic only** | indices 0-27 (bias, scores, guards, span shape) |
| + ASR uncertainty | indices 0-31 |
| + Sparse context | indices 0-31 + hashed context features |
| + Memory | all features (0-37 + sparse) |

For each slice, report all metrics. If softmax wins in Eval 6, re-run
ablations on softmax too.

### Implementation

Add a `feature_mask` parameter to `build_examples` (or a wrapper). Filter
the feature vec to only include features in the allowed set before training
and scoring. Sparse features (index >= SPARSE_OFFSET) are on/off as a group.

## Eval 5: Reachable-only judge eval

Filter test cases to only those where gold is reachable on the **judge-visible
decision surface** — not just locally present in some `ProbedSpan`.

Reachable means: the gold candidate survived all the way through composition,
pruning, and deduplication to appear in the final decision set that the judge
actually scores. `ProbedSpan.gold_alias_id` only tells you the gold term
appeared somewhere locally in one span's shortlist — that's necessary but not
sufficient.

For counterexamples: at least one candidate made it to the final decision set
(non-trivial decision).

Measure only: did the judge pick gold (canonical) or abstain (counterexample)?

This isolates judge quality from retrieval/composition noise. Should be a
**primary model-development metric**.

### Implementation

After the judge scores candidates for a test case, check whether gold was
among the options the judge actually saw. If not, skip the case. This means
reachability is checked *after* `score_candidates` runs on the full composed
decision set, not at the per-span level.

Report separately: `reachable_canonical_acc`, `reachable_cx_acc`,
`reachable_canonical_replace_rate`, `reachable_cx_replace_rate`.

## Eval 6: Formulation comparison

Compare three training objectives:

### A. Independent binary (current)
Each candidate scored independently: P(accept this candidate).
Threshold determines accept/reject. This is what we have.

### B. Case-balanced binary (Eval 2)
Same formulation, but with balanced training weights per case.

### C. Casewise softmax (new)
For each case, build features for all candidates + a "keep_original"
candidate. Train as a multi-class problem:
- Compute logits for all options
- Apply softmax
- Gradient = `softmax_prob - 1` for the gold, `softmax_prob - 0` for the rest

This makes keep_original a real candidate with its own features rather than
the implicit "nothing exceeded threshold" default.

**keep_original features (experiment placeholder):** For the initial experiment,
keep_original gets: bias=1, all candidate-specific scores=0, verified=0, but
*does* get context features (L1/R1/etc), ASR uncertainty features, and memory
features — it just lacks candidate-specific term features.

> **Note:** This all-zero-scores row is a temporary placeholder for the
> experiment. If softmax wins, the final design should give keep_original
> proper context/ASR/memory features. It just lacks candidate-specific
> phonetic scores (because there is no candidate). Don't mistake the
> placeholder for the final design.

### Implementation

Add `ScoringFormulation` enum: `IndependentBinary`, `CaseBalancedBinary`, `CasewiseSoftmax`.

For softmax, add to `SparseFtrl`:
```
fn update_softmax(&mut self, candidates: &[Vec<Feature>], gold_index: usize)
```

Compute logit for each candidate, softmax, then gradient update.

## Eval 7: Hyperparameter sweeps (after formulation is settled)

Only after Evals 2 + 6 pick a formulation:
- epochs: 1, 2, 4, 8
- alpha: 0.1, 0.5, 1.0, 2.0
- L1: 0, 0.0001, 0.001, 0.01
- L2: 0, 0.001, 0.01
- hard_neg_cap: 1, 3, 5, all

## Output format

All evals print a structured summary to stdout. Example:

```
=== Phase 4 Offline Judge Eval ===

--- Eval 1: Baselines (5-fold, best threshold) ---
  deterministic:  can 45/80 (56.3%)  cx 90/113 (79.6%)  bal 68.0%  repl: can 60.0% cx 20.4%
  seed_only:      can 30/80 (37.5%)  cx 95/113 (84.1%)  bal 60.8%  repl: can 40.0% cx 15.9%
  taught:         can  2/80 ( 2.5%)  cx 108/113 (95.6%) bal 49.0%  repl: can  2.5% cx  4.4%

--- Eval 2: Case-balanced FTRL ---
  balanced:       can 55/80 (68.8%)  cx 85/113 (75.2%)  bal 72.0%  repl: can 75.0% cx 24.8%

--- Eval 3: Threshold sweep (case-balanced model) ---
  threshold  can_acc  cx_acc  balanced  can_repl%  cx_repl%
  0.1        85.0%    20.0%   52.5%     95.0%      80.0%
  ...
  0.5        68.8%    75.2%   72.0%     75.0%      24.8%

--- Eval 4: Feature ablation (case-balanced, threshold=0.4) ---
  phonetic_only:  bal 65.0%  repl: can 70.0% cx 30.0%
  +asr:           bal 67.0%  repl: can 72.0% cx 28.0%
  +context:       bal 71.0%  repl: can 74.0% cx 26.0%
  +memory:        bal 72.0%  repl: can 75.0% cx 24.8%

--- Eval 5: Reachable-only (case-balanced, threshold=0.4) ---
  can 55/60 (91.7%)  cx 85/100 (85.0%)  repl: can 93.3% cx 15.0%

--- Eval 6: Formulation comparison ---
  independent_binary:   bal 49.0%  repl: can  2.5% cx  4.4%
  case_balanced_binary: bal 72.0%  repl: can 75.0% cx 24.8%
  casewise_softmax:     bal 74.0%  repl: can 78.0% cx 22.0%
```

## Key files

- `rust/beeml/src/main.rs` — `run_offline_judge_eval`, `probe_case_spans`, `ProbedSpan`, `ProbedCase`
- `rust/beeml/src/judge.rs` — `OnlineJudge`, `build_examples`, `score_examples`, `teach_choice`, `seed_model`
- `rust/beeml/src/sparse_ftrl.rs` — `SparseFtrl`, `Feature`, `update`, `predict_prob`
- `rust/beeml/src/rpc.rs` — `OfflineJudgeEvalRequest`, `OfflineJudgeEvalResult`
- `data/phonetic-seed/` — `recording_examples.jsonl`, `counterexample_recordings.jsonl`, `vocab.jsonl`

## Implementation order

1. Eval 1 (baselines) — fast sanity check, baseline numbers first
2. Eval 2 (case-balanced training) — fix the known structural bug
3. Eval 3 (threshold sweep) — find usable operating point
4. Eval 5 (reachable-only) — judge quality on the real decision surface
5. Eval 4 (feature ablation) — now that the formulation is sane
6. Eval 6 (formulation comparison) — casewise softmax experiment
7. Eval 7 (hyperparameters) — only after formulation settled
