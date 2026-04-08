//! Shared types for the bee correction pipeline.
//!
//! This crate contains lightweight data types used across bee-phonetic,
//! bee-correct, bee-ffi, and beeml. It has no runtime dependencies beyond
//! facet for derive macros.

use facet::Facet;

// ── Transcript types ─────────────────────────────────────────────────

/// A single word with its time boundaries from forced alignment.
#[derive(Debug, Clone, Facet)]
pub struct AlignedWord {
    /// The word as a string, ie. "platypus"
    pub word: String,

    /// The start of the word in seconds
    pub start: f64,

    /// The end of the word in seconds
    pub end: f64,

    /// How confident the ASR is about hearing this word
    pub confidence: Confidence,
}

/// Statistics for a word.
#[derive(Debug, Clone, PartialEq, Facet)]
pub struct Confidence {
    /// Mean logprob
    pub mean_lp: f32,

    /// Minimum logprob
    pub min_lp: f32,

    /// Mean margin
    pub mean_m: f32,

    /// Minimum margin
    pub min_m: f32,
}

/// A span of transcript text with phonetic representations and ASR uncertainty.
#[derive(Debug, Clone, Default, Facet)]
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
    pub mean_logprob: Option<f32>,
    pub min_logprob: Option<f32>,
    pub mean_margin: Option<f32>,
    pub min_margin: Option<f32>,
}

/// ASR alignment token with timing and uncertainty.
#[derive(Debug, Clone, Facet)]
pub struct TranscriptAlignmentToken {
    pub start_time: f64,
    pub end_time: f64,
    pub confidence: Confidence,
}

/// A word token with character offsets in the transcript.
#[derive(Debug, Clone, PartialEq, Eq, Facet)]
pub struct SentenceWordToken {
    pub char_start: usize,
    pub char_end: usize,
    pub text: String,
}

// ── Phonetic / retrieval types ───────────────────────────────────────

/// How an alias was derived from the canonical term.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Facet)]
pub enum AliasSource {
    Canonical,
    Spoken,
    Identifier,
    Confusion,
}

/// Which phonetic index view produced a match.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Facet)]
pub enum IndexView {
    RawIpa2,
    RawIpa3,
    ReducedIpa2,
    ReducedIpa3,
    Feature2,
    Feature3,
    ShortQueryFallback,
}

/// Structural flags derived from identifier analysis.
#[derive(Debug, Clone, Default, PartialEq, Eq, Facet)]
pub struct IdentifierFlags {
    pub acronym_like: bool,
    pub has_digits: bool,
    pub snake_like: bool,
    pub camel_like: bool,
    pub symbol_like: bool,
}

// ── Correction types ─────────────────────────────────────────────────

/// Context around a transcript span for judge features.
#[derive(Clone, Debug, Default, Facet)]
pub struct SpanContext {
    pub left_tokens: Vec<String>,
    pub right_tokens: Vec<String>,
    pub code_like: bool,
    pub prose_like: bool,
    pub list_like: bool,
    pub sentence_start: bool,
    pub app_id: Option<String>,
}

/// A correction event logged for training/replay.
#[derive(Clone, Debug, Facet)]
pub struct CorrectionEvent {
    pub timestamp_secs: u64,
    pub span_text: String,
    pub chosen_term: String,
    pub all_candidate_terms: Vec<String>,
    pub chosen_alias_id: Option<u32>,
}

/// An edit applied in a correction decision set.
#[derive(Clone, Debug, Facet)]
pub struct AppliedEdit {
    pub token_start: u32,
    pub token_end: u32,
    pub char_start: u32,
    pub char_end: u32,
    pub original_text: String,
    pub replacement_text: String,
    pub term: String,
    pub alias_id: u32,
    pub score: f32,
    pub phonetic_score: f32,
}
