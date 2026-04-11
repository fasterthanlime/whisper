use std::fmt;

use anyhow::{Result, bail};

use crate::types2::{SampleRange, decode_token_ids};

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
    /// Utterance-global sample span associated with this token.
    pub(crate) span: SampleRange,
    /// Present only on the first token of a word; gives the word length in tokens.
    pub(crate) starts_word_len: Option<TokenCount>,
}

/// The canonical aligned token sequence for a decodeable utterance slice.
///
/// Invariants:
/// - `tokens` is ordered by increasing `TimedToken.index`
/// - indices are monotonically increasing without rebase
/// - this vector is the single source of truth for token order
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ChunkInfo {
    /// Canonical utterance-global token sequence in strictly increasing index order.
    pub(crate) tokens: Vec<TimedToken>,
}

impl ChunkInfo {
    pub(crate) fn new(tokens: Vec<TimedToken>) -> Self {
        for pair in tokens.windows(2) {
            assert!(
                pair[0].index < pair[1].index,
                "chunk token indices must increase strictly"
            );
        }
        Self { tokens }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Returns the utterance-global token span covered by this chunk.
    pub(crate) fn token_range(&self) -> Option<UtteranceTokenRange> {
        let start = self.tokens.first()?.index;
        let end = TokenIndex::new(self.tokens.last()?.index.as_usize() + 1);
        Some(UtteranceTokenRange::new(start, end))
    }

    /// Returns the utterance-global time span covered by this chunk.
    pub(crate) fn time_range(&self) -> Option<TimeRange> {
        Some(TimeRange::new(
            self.tokens.first()?.span.start.to_time(),
            self.tokens.last()?.span.end.to_time(),
        ))
    }

    /// Decodes the entire chunk on demand through the process-global tokenizer.
    pub(crate) fn decode_text(&self) -> Result<String> {
        decode_timed_tokens(&self.tokens)
    }

    /// Decodes a sub-range of this chunk on demand through the process-global tokenizer.
    ///
    /// Invariants:
    /// - `range` must lie within this chunk's utterance-global token span
    pub(crate) fn decode_range(&self, range: UtteranceTokenRange) -> Result<String> {
        let trace_range = self
            .token_range()
            .ok_or_else(|| anyhow::anyhow!("cannot decode range from empty chunk"))?;
        if range.start < trace_range.start || range.end > trace_range.end {
            bail!(
                "decode range {}..{} lies outside chunk range {}..{}",
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

fn decode_timed_tokens(tokens: &[TimedToken]) -> Result<String> {
    let ids: Vec<TokenId> = tokens.iter().map(|token| token.token).collect();
    decode_token_ids(&ids)
}
