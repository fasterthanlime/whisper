# 002: Two-stage judge as the production judge in bee-correct

## Goal

Make the two-stage architecture (span gate + candidate ranker) the default
judge in bee-correct, replacing the single-model OnlineJudge for scoring.

## Current state

- `OnlineJudge` wraps a single `SparseFtrl` model
- Two-stage gate + ranker exist as standalone functions (`build_gate_features`,
  `build_ranker_features`, `seed_gate_model`, `seed_ranker_model`) but are
  only used in the offline eval harness
- No struct encapsulates the two-stage architecture

## Design

New struct in bee-correct:

```rust
pub struct TwoStageJudge {
    gate: SparseFtrl,       // Stage A: should this span be corrected?
    ranker: SparseFtrl,     // Stage B: which candidate wins?
    memory: TermMemory,     // shared memory across both stages
    event_log: Vec<CorrectionEvent>,
    gate_threshold: f32,    // default 0.5 (conservative)
    ranker_threshold: f32,  // default 0.2 (conservative)
}
```

Public API:

```rust
impl TwoStageJudge {
    pub fn new(gate_threshold: f32, ranker_threshold: f32) -> Self;
    pub fn score_span(&self, span, candidates, ctx) -> SpanDecision;
    pub fn teach_choice(&mut self, span, candidates, chosen_alias_id, ctx);
    pub fn gate_prob(&self, span, candidates, ctx) -> f32;
    pub fn ranker_scores(&self, span, candidates) -> Vec<(u32, f32)>;
}

pub struct SpanDecision {
    pub gate_open: bool,
    pub gate_prob: f32,
    pub chosen: Option<CandidateChoice>,  // None if gate closed
    pub options: Vec<RankedCandidate>,
}
```

## Operating points

Default to conservative (GT=0.5, RT=0.2) for production.
Expose thresholds as configuration for experimentation.

## Keep OnlineJudge

Don't delete the old single-model judge yet — keep it as a regression
baseline and for the eval harness comparison. But the production code
path should use TwoStageJudge.

## Depends on

- 001 (bee-correct crate exists)
