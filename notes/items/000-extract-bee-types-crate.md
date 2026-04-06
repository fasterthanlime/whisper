# 000: Extract bee-types crate

## Goal

Create a tiny shared types crate with no runtime dependencies, so that
bee-correct, bee-ffi, and beeml can all speak the same data model without
dragging in ASR/transcription machinery.

## Why

bee-correct needs transcript/span/alignment types but should NOT depend
on bee-transcribe (which pulls in ASR runtime, MLX, audio processing).
Without bee-types, bee-correct either:
- depends on bee-transcribe (defeats the extraction), or
- redefines the same types (drift risk)

## What goes in bee-types

Lightweight structs with facet derives, no behavior:

```rust
// Transcript types
pub struct Transcript { pub text: String, pub words: Vec<AlignedWord> }
pub struct AlignedWord { pub word: String, pub start: f32, pub end: f32 }
pub struct TranscriptSpan {
    pub text: String, pub char_start: usize, pub char_end: usize,
    pub token_start: usize, pub token_end: usize,
    pub ipa_tokens: Vec<String>, pub reduced_ipa_tokens: Vec<String>,
}

// Correction types
pub struct CorrectionEvent { ... }
pub struct SpanContext { ... }
pub struct CandidateFeatureRow { ... }
pub struct IdentifierFlags { ... }

// Decision set types
pub struct DecisionSet { pub choices: Vec<SentenceChoice> }
pub struct SentenceChoice { pub text: String, pub edits: Vec<AppliedEdit>, pub score: f32 }
pub struct AppliedEdit {
    pub span_start: usize, pub span_end: usize,
    pub original_text: String, pub replacement_text: String,
    pub alias_id: u32, pub term: String,
}

// Teaching event model (see 003 for details)
pub struct TeachingEvent { ... }
```

## Dependencies

```toml
[dependencies]
facet-core = { workspace = true }
facet-json = { workspace = true }
```

That's it. No runtime, no ASR, no phonetics. Uses facet for derive, not serde.

## Who depends on bee-types

- `bee-types`: zero deps (just serde)
- `bee-transcribe`: depends on bee-types (re-exports or converts)
- `bee-correct`: depends on bee-types + bee-phonetic
- `bee-ffi`: depends on bee-types + bee-correct + bee-transcribe
- `beeml`: depends on all of the above + vox/tokio/rayon

## Validation

- `cargo build -p bee-types` compiles with only facet
- No runtime code in bee-types (no functions, just struct definitions + facet derives)
- bee-correct does NOT depend on bee-transcribe
