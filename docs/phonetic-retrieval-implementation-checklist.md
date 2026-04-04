# Phonetic Retrieval Implementation Checklist

Related system doc:

- [beeml-correction-system.md](/Users/amos/bearcove/bee/docs/beeml-correction-system.md)

This document turns the roadmap into concrete backend work for this repo.

Primary reference:

- [beeml-correction-system.md](/Users/amos/bearcove/bee/docs/beeml-correction-system.md)

## Current Status

Implemented now:

- `phonetic_lexicon.rs`
  - canonical / spoken / confusion alias expansion
  - reduced IPA tokens
  - identifier flags
- `phonetic_index.rs`
  - boundary-aware 2-gram / 3-gram postings
  - raw and reduced IPA views
  - shortlist query API
- `region_proposal.rs`
  - transcript tokenization
  - contiguous span enumeration
  - eSpeak-backed span IPA generation hook
- `phonetic_verify.rs`
  - shortlist verification with phonetic scoring
- backend debug route
  - `POST /api/correct-prototype/retrieval-debug`
- frontend playground
  - local `beehive` Retrieval tab for probing spans/candidates interactively

Not integrated into main correction yet:

- indexed retrieval is not yet the proposal path in `prototype.rs`
- no retrieval benchmark route yet
- no span pruning beyond simple max-span enumeration
- no candidate provenance beyond current matched-view / alias-source fields

## Immediate Next Steps

The next steps we have actually discussed are:

1. add early span filters before verification
   - suppress obviously low-value spans like leading function-word spans
   - add cheap token-pattern and length heuristics before shortlist verification

2. improve retrieval provenance
   - return which q-gram view contributed most
   - return overlap counts per view
   - make it obvious why a candidate survived

3. add a backend retrieval benchmark route
   - `POST /api/correct-prototype/retrieval-benchmark`
   - fixed human cases / transcripts
   - top-1 / top-3 / top-10 target retrieval
   - span count, shortlist count, verify count, latency

4. wire indexed retrieval into `prototype.rs` behind a mode switch
   - `BruteForce`
   - `IndexedPhonetic`
   - run eval with the new path before making it default

5. promote the frontend retrieval playground from inspection-only to debugging tool
   - allow copying a span directly into the correction flow
   - add candidate/source badges
   - add score sorting and span filtering controls

## Scope

Target first implementation:

- transcript source: `Qwen/MLX`
- timings: forced aligner
- phonetic representation: `eSpeak`
- retrieval: q-gram inverted index over IPA
- verification: feature-aware phonetic distance
- reranking: existing local/per-region reranker, fed from a better shortlist

Out of scope for first implementation:

- ZIPA
- G2P N-best expansion
- trie/TST short-query sidecar
- WFST-based retrieval

## Deliverable 1: Canonical Retrieval Data Types

Add a new module:

- `/Users/amos/bearcove/hark/asr-synth/crates/synth-dashboard/src/phonetic_lexicon.rs`

Define these types:

```rust
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AliasSource {
    Canonical,
    Spoken,
    Confusion,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentifierFlags {
    pub acronym_like: bool,
    pub has_digits: bool,
    pub snake_like: bool,
    pub camel_like: bool,
    pub symbol_like: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LexiconAlias {
    pub alias_id: u32,
    pub term: String,
    pub alias_text: String,
    pub alias_source: AliasSource,
    pub ipa_tokens: Vec<String>,
    pub reduced_ipa_tokens: Vec<String>,
    pub token_count: u8,
    pub phone_count: u8,
    pub identifier_flags: IdentifierFlags,
}
```

Required helpers:

- `build_phonetic_lexicon(...) -> Vec<LexiconAlias>`
- `reduce_ipa_tokens(...) -> Vec<String>`
- `derive_identifier_flags(...) -> IdentifierFlags`

First lexicon sources:

- canonical vocab term
- spoken override / spoken auto form
- reviewed confusion surfaces

Do not add G2P here yet.

## Deliverable 2: Boundary-Aware Q-Gram Index

Add a new module:

- `/Users/amos/bearcove/hark/asr-synth/crates/synth-dashboard/src/phonetic_index.rs`

Define core types:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IndexView {
    RawIpa2,
    RawIpa3,
    ReducedIpa2,
    ReducedIpa3,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Posting {
    pub alias_id: u32,
    pub count: u16,
}

#[derive(Debug, Clone)]
pub struct PhoneticIndex {
    pub aliases: Vec<LexiconAlias>,
    pub postings: HashMap<(IndexView, String), Vec<Posting>>,
    pub by_phone_len: BTreeMap<u8, Vec<u32>>,
    pub by_token_count: HashMap<u8, Vec<u32>>,
}
```

Required helpers:

- `with_boundaries(tokens: &[String]) -> Vec<String>`
- `qgrams(tokens: &[String], q: usize) -> Vec<String>`
- `build_index(aliases: Vec<LexiconAlias>) -> PhoneticIndex`

Implementation notes:

- gram keys should include token boundaries
- phone-length and token-count buckets should be cheap filters
- keep the structure in memory; no DB persistence for the first version

## Deliverable 3: Query + Shortlist API

Still in `phonetic_index.rs`, define:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalQuery {
    pub text: String,
    pub ipa_tokens: Vec<String>,
    pub reduced_ipa_tokens: Vec<String>,
    pub token_count: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalCandidate {
    pub alias_id: u32,
    pub term: String,
    pub alias_text: String,
    pub alias_source: AliasSource,
    pub matched_view: IndexView,
    pub qgram_overlap: u16,
    pub token_count_match: bool,
    pub phone_count_delta: i16,
    pub coarse_score: f32,
}
```

Required APIs:

- `query_index(index: &PhoneticIndex, query: &RetrievalQuery, limit: usize) -> Vec<RetrievalCandidate>`
- `candidate_shortlist(...)`

First-pass coarse scoring should combine:

- q-gram overlap count
- token-count compatibility
- phone-length compatibility
- boundary gram matches

No phonetic DP here.

## Deliverable 4: Span Representation

Add a new module:

- `/Users/amos/bearcove/hark/asr-synth/crates/synth-dashboard/src/region_proposal.rs`

Define:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptSpan {
    pub token_start: usize,
    pub token_end: usize,
    pub char_start: usize,
    pub char_end: usize,
    pub start_sec: Option<f64>,
    pub end_sec: Option<f64>,
    pub text: String,
    pub ipa_tokens: Vec<String>,
    pub reduced_ipa_tokens: Vec<String>,
}
```

Required helpers:

- `enumerate_transcript_spans(...) -> Vec<TranscriptSpan>`
- `span_from_tokens(...) -> TranscriptSpan`

First version:

- enumerate contiguous spans of 1..=4 words
- derive IPA with `eSpeak`
- do not attempt sophisticated suspicious-region pruning yet

## Deliverable 5: Verification Layer

Add a new module:

- `/Users/amos/bearcove/hark/asr-synth/crates/synth-dashboard/src/phonetic_verify.rs`

Define:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiedCandidate {
    pub alias_id: u32,
    pub term: String,
    pub alias_text: String,
    pub alias_source: AliasSource,
    pub matched_view: IndexView,
    pub coarse_score: f32,
    pub phonetic_score: f32,
    pub weighted_edit_distance: f32,
}
```

Required API:

- `verify_shortlist(span: &TranscriptSpan, shortlist: &[RetrievalCandidate], index: &PhoneticIndex, limit: usize) -> Vec<VerifiedCandidate>`

Use:

- `rspanphon` or the current improved phonetic scorer

Keep all debug/provenance fields that explain why a candidate survived.

## Deliverable 6: Integration with Existing Prototype Pipeline

Modify:

- `/Users/amos/bearcove/hark/asr-synth/crates/synth-dashboard/src/prototype.rs`

Replace the expensive brute-force span scoring path with:

1. enumerate transcript spans
2. query phonetic index
3. verify shortlist
4. convert verified candidates into the existing proposal/edit structures

Do not rewrite the whole reranker first.

Integrate in stages:

### Stage 6A

Add an alternate proposal path behind a feature flag or mode:

- `PrototypeRetrievalMode::BruteForce`
- `PrototypeRetrievalMode::IndexedPhonetic`

### Stage 6B

Use indexed retrieval only for eval experiments first.

### Stage 6C

Promote indexed retrieval to default once metrics are acceptable.

## Deliverable 7: Debug Output

Add explicit debug payloads so failures are tunable.

Every proposal should expose:

- source span text
- span IPA
- matched alias text
- alias source
- matched index view
- coarse overlap score
- final phonetic verification score

Suggested response additions:

```rust
pub struct RetrievalDebug {
    pub span_text: String,
    pub span_ipa: Vec<String>,
    pub shortlist_count: usize,
    pub verified_count: usize,
    pub matched_view: IndexView,
    pub alias_source: AliasSource,
    pub coarse_score: f32,
    pub phonetic_score: f32,
}
```

This is not optional. Without it, tuning the retrieval layer will be blind.

## Deliverable 8: Retrieval Benchmark

Add a backend-only benchmark route or helper.

Suggested route:

- `POST /api/correct-prototype/retrieval-benchmark`

Input:

- fixed recording ids or case ids
- mode selector

Output:

- target retrieved in top 1 / top 3 / top 10
- spans enumerated
- shortlist size per span
- verified candidates per span
- latency:
  - span enumeration
  - retrieval
  - verification

This should be evaluated separately from reranker behavior.

## Deliverable 9: End-to-End Eval Mode Toggle

Expose a mode switch in the backend:

- `bruteforce`
- `indexed_phonetic`

This should be configurable on:

- live correction route
- human bakeoff route

Goal:

- compare current proposal generation vs indexed retrieval without changing the rest of the stack at the same time

## Deliverable 10: Confidence Hook

Prepare for future `Qwen/MLX` token confidence, even if it is not implemented yet.

Define optional fields on transcript tokens/spans:

```rust
pub struct TranscriptToken {
    pub text: String,
    pub start_sec: Option<f64>,
    pub end_sec: Option<f64>,
    pub confidence: Option<f32>,
}
```

Do not block retrieval work on this.

But keep the span API ready so that later we can:

- skip high-confidence regions
- prioritize low-confidence regions

## Suggested File Ownership

Concrete first-pass write set:

- `/Users/amos/bearcove/hark/asr-synth/crates/synth-dashboard/src/phonetic_lexicon.rs`
- `/Users/amos/bearcove/hark/asr-synth/crates/synth-dashboard/src/phonetic_index.rs`
- `/Users/amos/bearcove/hark/asr-synth/crates/synth-dashboard/src/phonetic_verify.rs`
- `/Users/amos/bearcove/hark/asr-synth/crates/synth-dashboard/src/region_proposal.rs`
- `/Users/amos/bearcove/hark/asr-synth/crates/synth-dashboard/src/prototype.rs`

Optional route wiring later:

- `/Users/amos/bearcove/hark/asr-synth/crates/synth-dashboard/src/jobs.rs`
- `/Users/amos/bearcove/hark/asr-synth/crates/synth-dashboard/src/main.rs`

## Recommended Build Order

Implement in this exact order:

1. lexicon types
2. q-gram index build
3. query + shortlist API
4. span enumeration
5. verifier
6. retrieval benchmark
7. prototype integration
8. eval mode toggle

This order keeps the new retrieval stack testable before it is entangled with reranking.

## Minimal Acceptance Criteria

Before switching anything by default, require:

- retrieval benchmark exists
- target top-10 recall is materially better than current span heuristics
- indexed retrieval latency is predictable
- proposal provenance is inspectable
- end-to-end eval does not regress catastrophically on exact recovery

## Immediate Next Coding Task

Implement:

- `LexiconAlias`
- `AliasSource`
- `build_phonetic_lexicon(...)`
- `PhoneticIndex`
- boundary-aware 2-gram / 3-gram posting construction

Nothing else should block that first slice.
