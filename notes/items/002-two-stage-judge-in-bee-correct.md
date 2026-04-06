# 002: Two-stage judge as the production judge in bee-correct

## Goal

Make the two-stage architecture (span gate + candidate ranker) the default
judge in bee-correct, replacing the single-model OnlineJudge for scoring.

## Depends on

- 001 (bee-correct crate exists)

## Design

Separate concerns: judge owns weights + memory, logging is external.

```rust
/// Core judge: weights + memory, no event storage.
pub struct TwoStageJudge {
    gate: SparseFtrl,
    ranker: SparseFtrl,
    memory: TermMemory,
    gate_threshold: f32,    // default 0.5 (conservative)
    ranker_threshold: f32,  // default 0.2 (conservative)
}

/// Decision output with full trace for product/debug use.
pub struct SpanDecision {
    pub gate_open: bool,
    pub gate_prob: f32,
    pub chosen: Option<CandidateChoice>,
    pub options: Vec<RankedCandidate>,  // all candidates with scores
}

pub struct CandidateChoice {
    pub alias_id: u32,
    pub term: String,
    pub replacement_text: String,
    pub ranker_prob: f32,
}
```

### Public API

```rust
impl TwoStageJudge {
    pub fn new(gate_threshold: f32, ranker_threshold: f32) -> Self;

    /// Score a span: gate decision + ranked candidates.
    /// Returns enough trace for product UI and debug.
    pub fn score_span(&self, span, candidates, ctx) -> SpanDecision;

    /// Update weights from user feedback.
    pub fn teach_span(&mut self, span, candidates, chosen_alias_id, ctx);

    /// Direct access for diagnostics.
    pub fn gate_prob(&self, span, candidates, ctx) -> f32;
    pub fn ranker_scores(&self, span, candidates) -> Vec<(u32, f32)>;

    // Persistence
    pub fn save_weights(&self, path: &Path) -> Result<()>;
    pub fn load_weights(&mut self, path: &Path) -> Result<()>;
    pub fn save_memory(&self, path: &Path) -> Result<()>;
    pub fn load_memory(&mut self, path: &Path) -> Result<()>;
}
```

### Event logging is separate

```rust
/// External to TwoStageJudge — caller decides storage.
pub trait CorrectionEventSink {
    fn log_event(&mut self, event: &CorrectionEvent);
}

// Implementations: Vec<CorrectionEvent>, File, no-op, etc.
```

This avoids the judge owning a growing Vec that's awkward for FFI
ownership, persistence, and long sessions.

## Operating points

Default to conservative (GT=0.5, RT=0.2) for production.
Expose thresholds as configuration.

## Keep OnlineJudge

Don't delete the old single-model judge — keep as regression baseline
and for eval harness comparison. Production code path uses TwoStageJudge.

## Depends on

- 001 (bee-correct crate exists)
