//! Canonical utterance-global types for the next rollback model.
//!
//! Intent:
//! - one canonical token sequence
//! - one utterance-global coordinate system
//! - one cut in token space
//! - no window-relative or carry-relative coordinates
//! - one tokenizer slot initialized once at process startup
//!
//! Non-goals:
//! - this module does not describe the current `bee-kv` implementation
//! - this module does not preserve the current bridge/carry bookkeeping shape
//! - this module does not store decoded text
//!
//! Invariants:
//! - every token index is relative to the beginning of the utterance
//! - every time is relative to the beginning of the utterance
//! - `TimeRange` values are utterance-global, never window-local
//! - `TokenTrace.tokens` is the single source of truth for token order
//! - `TimedToken.starts_word` is present only on the first token of a word
//! - if `TimedToken.starts_word` is `Some(len)`, then the next `len` tokens in the
//!   same `TokenTrace` belong to that word
//! - `Cut::At(index)` refers to an utterance-global token boundary, not a local
//!   offset inside a chunk

use std::fmt;
use std::path::Path;
use std::sync::OnceLock;

use anyhow::{Result, bail};
use bee_qwen3_asr::tokenizers::Tokenizer;

static TOKENIZER: OnceLock<Tokenizer> = OnceLock::new();

/// Loads the tokenizer from `path` and installs it into the process-global slot.
///
/// Invariants:
/// - initialization happens exactly once
/// - every decode helper in this module reads from the same tokenizer instance
pub(crate) fn init_tokenizer(path: &Path) -> &'static Tokenizer {
    let loaded =
        Tokenizer::from_file(path).unwrap_or_else(|e| panic!("loading {}: {e}", path.display()));
    TOKENIZER
        .set(loaded)
        .unwrap_or_else(|_| panic!("types2 tokenizer already initialized"));
    tokenizer()
}

/// Returns the process-global tokenizer slot.
///
/// Invariant:
/// - callers must initialize the slot through [`init_tokenizer`] before use
pub(crate) fn tokenizer() -> &'static Tokenizer {
    TOKENIZER
        .get()
        .unwrap_or_else(|| panic!("types2 tokenizer not initialized"))
}

/// A model token ID.
///
/// Invariant:
/// - opaque identifier only; no text is stored here
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct TokenId(u32);

impl TokenId {
    pub(crate) fn new(id: u32) -> Self {
        Self(id)
    }

    pub(crate) fn as_u32(self) -> u32 {
        self.0
    }

    /// Decodes this single token through the process-global tokenizer.
    pub(crate) fn decode(self) -> Result<String> {
        decode_token_ids(&[self])
    }
}

/// An utterance-global token index.
///
/// Invariant:
/// - zero is the first token in the utterance
/// - values are absolute within the utterance, never rebased per chunk/window
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct TokenIndex(usize);

impl TokenIndex {
    pub(crate) fn new(index: usize) -> Self {
        Self(index)
    }

    pub(crate) fn as_usize(self) -> usize {
        self.0
    }
}

/// A count of tokens.
///
/// Invariant:
/// - this is a length, not a position
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct TokenCount(usize);

impl TokenCount {
    pub(crate) fn new(count: usize) -> Self {
        Self(count)
    }

    pub(crate) fn as_usize(self) -> usize {
        self.0
    }
}

/// A half-open token range in utterance-global token coordinates.
///
/// Invariant:
/// - `start <= end`
/// - both coordinates are utterance-global
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct UtteranceTokenRange {
    /// Inclusive start of the half-open token range in utterance-global coordinates.
    pub(crate) start: TokenIndex,
    /// Exclusive end of the half-open token range in utterance-global coordinates.
    pub(crate) end: TokenIndex,
}

impl UtteranceTokenRange {
    pub(crate) fn new(start: TokenIndex, end: TokenIndex) -> Self {
        assert!(start <= end, "utterance token ranges must be ordered");
        Self { start, end }
    }

    pub(crate) fn len(self) -> TokenCount {
        TokenCount::new(self.end.as_usize().saturating_sub(self.start.as_usize()))
    }
}

/// A timestamp in seconds relative to utterance start.
///
/// Invariant:
/// - this is utterance-global time, not wall-clock time
/// - this is never chunk-relative, window-relative, or carry-relative
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub(crate) struct UtteranceTime(f64);

impl UtteranceTime {
    pub(crate) fn from_secs(secs: f64) -> Self {
        Self(secs)
    }

    pub(crate) fn as_secs(self) -> f64 {
        self.0
    }
}

/// A half-open utterance-global time span.
///
/// Invariant:
/// - `start <= end`
/// - both endpoints are relative to utterance start
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct TimeRange {
    /// Inclusive start of the half-open time range in utterance-global coordinates.
    pub(crate) start: UtteranceTime,
    /// Exclusive end of the half-open time range in utterance-global coordinates.
    pub(crate) end: UtteranceTime,
}

impl TimeRange {
    pub(crate) fn new(start: UtteranceTime, end: UtteranceTime) -> Self {
        assert!(start <= end, "time ranges must be ordered");
        Self { start, end }
    }
}

/// One token in the canonical utterance-global sequence.
///
/// Intent:
/// - this is the unit we cut on
/// - text is derived by decoding token slices, not stored here
///
/// Invariants:
/// - `index` is utterance-global
/// - `span` is utterance-global
/// - `starts_word_len` is only populated on the first token of a word
/// - `starts_word_len`, when present, is the number of tokens in that word
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct TimedToken {
    /// Utterance-global token index for this token.
    pub(crate) index: TokenIndex,
    /// Opaque model token ID.
    pub(crate) token: TokenId,
    /// Utterance-global audio span associated with this token.
    pub(crate) span: TimeRange,
    /// Present only on the first token of a word; gives the word length in tokens.
    pub(crate) starts_word_len: Option<TokenCount>,
}

/// The canonical token sequence for some utterance or utterance slice.
///
/// Invariants:
/// - `tokens` is ordered by increasing `TimedToken.index`
/// - indices are monotonically increasing without rebase
/// - this vector is the single source of truth for token order
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TokenTrace {
    /// Canonical utterance-global token sequence in strictly increasing index order.
    pub(crate) tokens: Vec<TimedToken>,
}

impl TokenTrace {
    pub(crate) fn new(tokens: Vec<TimedToken>) -> Self {
        for pair in tokens.windows(2) {
            assert!(
                pair[0].index < pair[1].index,
                "token trace indices must increase strictly"
            );
        }
        Self { tokens }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Returns the utterance-global token span covered by this trace.
    pub(crate) fn token_range(&self) -> Option<UtteranceTokenRange> {
        let start = self.tokens.first()?.index;
        let end = TokenIndex::new(self.tokens.last()?.index.as_usize() + 1);
        Some(UtteranceTokenRange::new(start, end))
    }

    /// Returns the utterance-global time span covered by this trace.
    pub(crate) fn time_range(&self) -> Option<TimeRange> {
        Some(TimeRange::new(
            self.tokens.first()?.span.start,
            self.tokens.last()?.span.end,
        ))
    }

    /// Decodes the entire trace on demand through the process-global tokenizer.
    pub(crate) fn decode_text(&self) -> Result<String> {
        decode_timed_tokens(&self.tokens)
    }

    /// Decodes a sub-range of this trace on demand through the process-global tokenizer.
    ///
    /// Invariants:
    /// - `range` must lie within this trace's utterance-global token span
    pub(crate) fn decode_range(&self, range: UtteranceTokenRange) -> Result<String> {
        let trace_range = self
            .token_range()
            .ok_or_else(|| anyhow::anyhow!("cannot decode range from empty token trace"))?;
        if range.start < trace_range.start || range.end > trace_range.end {
            bail!(
                "decode range {}..{} lies outside trace range {}..{}",
                range.start,
                range.end,
                trace_range.start,
                trace_range.end
            );
        }
        let local_start = range.start.as_usize() - trace_range.start.as_usize();
        let local_end = range.end.as_usize() - trace_range.start.as_usize();
        decode_timed_tokens(&self.tokens[local_start..local_end])
    }
}

/// A cut in utterance-global token space.
///
/// Intent:
/// - `NoCut` means the chunk grows; nothing is committed yet
/// - `At(index)` means commit everything strictly before `index`
///
/// Invariant:
/// - `index` is an utterance-global token boundary
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Cut {
    /// No commit/carry boundary was found; the chunk should grow.
    NoCut,
    /// Commit everything strictly before this utterance-global token boundary.
    At(TokenIndex),
}

impl Cut {
    pub(crate) fn token_index(self) -> Option<TokenIndex> {
        match self {
            Self::NoCut => None,
            Self::At(index) => Some(index),
        }
    }
}

impl fmt::Display for TokenIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl fmt::Display for TokenCount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

fn decode_token_ids(token_ids: &[TokenId]) -> Result<String> {
    let ids: Vec<u32> = token_ids.iter().map(|id| id.as_u32()).collect();
    tokenizer()
        .decode(&ids, true)
        .map_err(|e| anyhow::anyhow!("decoding token ids: {e}"))
}

fn decode_timed_tokens(tokens: &[TimedToken]) -> Result<String> {
    let ids: Vec<TokenId> = tokens.iter().map(|token| token.token).collect();
    decode_token_ids(&ids)
}
