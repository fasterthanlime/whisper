use std::fmt;

use anyhow::Result;
use compact_str::CompactString;

use crate::{SampleRange, decode_token_ids};

/// A model token ID.
///
/// Invariant:
/// - opaque identifier only; no text is stored here
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct TokenId(u32);

impl TokenId {
    pub(crate) fn new(id: u32) -> Self {
        Self(id)
    }

    pub fn as_u32(self) -> u32 {
        self.0
    }

    /// Decodes this single token through the process-global tokenizer.
    pub fn decode(self) -> Result<String> {
        decode_token_ids(&[self])
    }
}

/// An utterance-global token index.
///
/// Invariant:
/// - zero is the first token in the utterance
/// - values are absolute within the utterance, never rebased per chunk/window
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct TokenIndex(usize);

impl TokenIndex {
    pub(crate) fn new(index: usize) -> Self {
        Self(index)
    }

    pub fn as_usize(self) -> usize {
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
pub struct UtteranceTime(f64);

impl UtteranceTime {
    pub(crate) fn from_secs(secs: f64) -> Self {
        Self(secs)
    }

    pub fn as_secs(self) -> f64 {
        self.0
    }
}

/// A half-open utterance-global time span.
///
/// Invariant:
/// - `start <= end`
/// - both endpoints are relative to utterance start
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimeRange {
    /// Inclusive start of the half-open time range in utterance-global coordinates.
    pub start: UtteranceTime,
    /// Exclusive end of the half-open time range in utterance-global coordinates.
    pub end: UtteranceTime,
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
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimedToken {
    index: TokenIndex,
    token: TokenId,
    span: SampleRange,
}

impl TimedToken {
    pub(crate) fn new(index: TokenIndex, token: TokenId, span: SampleRange) -> Self {
        Self { index, token, span }
    }

    pub fn index(self) -> TokenIndex {
        self.index
    }

    pub fn token(self) -> TokenId {
        self.token
    }

    pub fn time_range(self) -> TimeRange {
        self.span.to_time_range()
    }
}

/// One ASR alternative for a decoded token.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AsrTokenAlternative {
    /// Candidate token ID proposed by the ASR decoder for this position.
    token: TokenId,
    /// Raw decoder logit for this candidate.
    logit: f32,
}

impl AsrTokenAlternative {
    pub(crate) fn new(token: TokenId, logit: f32) -> Self {
        Self { token, logit }
    }

    pub fn token(self) -> TokenId {
        self.token
    }

    pub fn logit(self) -> f32 {
        self.logit
    }
}

/// Token-level ASR confidence and candidate data.
#[derive(Clone, Debug, PartialEq)]
pub struct AsrTokenConfidence {
    /// Winner-vs-pack confidence signal derived from the ASR top-k logits.
    concentration: f32,
    /// Difference between the top-1 and top-2 logits.
    margin: f32,
    /// Ranked decoder alternatives for this token position.
    alternatives: Vec<AsrTokenAlternative>,
}

impl AsrTokenConfidence {
    pub(crate) fn new(
        concentration: f32,
        margin: f32,
        alternatives: Vec<AsrTokenAlternative>,
    ) -> Self {
        Self {
            concentration,
            margin,
            alternatives,
        }
    }

    pub fn concentration(&self) -> f32 {
        self.concentration
    }

    pub fn margin(&self) -> f32 {
        self.margin
    }

    pub fn alternatives(&self) -> &[AsrTokenAlternative] {
        &self.alternatives
    }
}

/// G2P-derived IPA for one decoded token.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct G2pTokenIpa {
    /// IPA string derived from the decoded token text through G2P.
    ipa: CompactString,
}

impl G2pTokenIpa {
    pub(crate) fn new(ipa: CompactString) -> Self {
        Self { ipa }
    }

    pub fn as_str(&self) -> &str {
        self.ipa.as_str()
    }
}

/// One ZIPA phone span aligned back onto a decoded token.
#[derive(Clone, Debug, PartialEq)]
pub struct ZipaPhoneSpan {
    /// Acoustic phone label produced by ZIPA for this span.
    phone: CompactString,
    /// Utterance-global start time of this phone span.
    start: UtteranceTime,
    /// Utterance-global end time of this phone span.
    end: UtteranceTime,
}

impl ZipaPhoneSpan {
    pub(crate) fn new(phone: CompactString, start: UtteranceTime, end: UtteranceTime) -> Self {
        Self { phone, start, end }
    }

    pub fn phone(&self) -> &str {
        self.phone.as_str()
    }

    pub fn time_range(&self) -> TimeRange {
        TimeRange::new(self.start, self.end)
    }
}

/// One normalized comparison phone attached to a token.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComparisonPhone {
    /// Normalized phone symbol used for transcript-vs-ZIPA comparison.
    phone: CompactString,
}

impl ComparisonPhone {
    pub(crate) fn new(phone: CompactString) -> Self {
        Self { phone }
    }

    pub fn as_str(&self) -> &str {
        self.phone.as_str()
    }
}

/// Token-level ZIPA timing status.
///
/// Intent:
/// - represent the actual alignment outcome explicitly
/// - preserve deleted/projected states instead of collapsing them into `None`
#[derive(Clone, Debug, PartialEq)]
pub enum ZipaTiming {
    /// ZIPA aligned this token onto a concrete utterance-global time span.
    Aligned(TimeRange),
    /// The token was projected into ZIPA space, but the projection collapsed to
    /// a zero-width deletion point.
    Deleted {
        /// Normalized ZIPA comparison index where the token collapsed.
        projected_at: usize,
    },
    /// The token projected into ZIPA comparison space, but no timed phone span
    /// could be recovered for that projected normalized range.
    Projected {
        /// Half-open normalized ZIPA comparison range.
        normalized_start: usize,
        /// Half-open normalized ZIPA comparison range.
        normalized_end: usize,
    },
    /// The token did not produce a valid comparison range.
    Invalid,
}

/// Token-aligned output bundle returned from one feed step.
///
/// Intent:
/// - one record per token
/// - ASR and IPA side data stay aligned to the same token anchor
/// - no parallel public slices
#[derive(Clone, Debug, PartialEq)]
pub struct OutputToken {
    /// Canonical utterance-global token and timing anchor.
    timed_token: TimedToken,
    /// Token-level ASR confidence and ranked alternatives for this token, when available.
    asr_confidence: Option<AsrTokenConfidence>,
    /// G2P-derived IPA for this token, when available.
    g2p_ipa: Option<G2pTokenIpa>,
    /// Normalized comparison phones derived from the transcript-side G2P view.
    transcript_phones: Vec<ComparisonPhone>,
    /// One or more ZIPA phone spans aligned back onto this token.
    zipa_phone_spans: Vec<ZipaPhoneSpan>,
    /// ZIPA timing status for this token, when audio-side alignment has run.
    zipa_timing: ZipaTiming,
}

impl OutputToken {
    pub(crate) fn new(
        timed_token: TimedToken,
        asr_confidence: Option<AsrTokenConfidence>,
        g2p_ipa: Option<G2pTokenIpa>,
        transcript_phones: Vec<ComparisonPhone>,
        zipa_phone_spans: Vec<ZipaPhoneSpan>,
        zipa_timing: ZipaTiming,
    ) -> Self {
        Self {
            timed_token,
            asr_confidence,
            g2p_ipa,
            transcript_phones,
            zipa_phone_spans,
            zipa_timing,
        }
    }

    pub fn timed_token(&self) -> TimedToken {
        self.timed_token
    }

    pub fn asr_confidence(&self) -> Option<&AsrTokenConfidence> {
        self.asr_confidence.as_ref()
    }

    pub fn g2p_ipa(&self) -> Option<&G2pTokenIpa> {
        self.g2p_ipa.as_ref()
    }

    pub fn transcript_phones(&self) -> &[ComparisonPhone] {
        &self.transcript_phones
    }

    pub fn zipa_phone_spans(&self) -> &[ZipaPhoneSpan] {
        &self.zipa_phone_spans
    }

    pub fn zipa_timing(&self) -> &ZipaTiming {
        &self.zipa_timing
    }

    pub(crate) fn set_g2p_ipa(&mut self, ipa: Option<CompactString>) {
        self.g2p_ipa = ipa.map(G2pTokenIpa::new);
    }

    pub(crate) fn set_transcript_phones(&mut self, phones: Vec<ComparisonPhone>) {
        self.transcript_phones = phones;
    }

    pub(crate) fn set_zipa_phone_spans(&mut self, phone_spans: Vec<ZipaPhoneSpan>) {
        self.zipa_phone_spans = phone_spans;
    }

    pub(crate) fn set_zipa_timing(&mut self, timing: ZipaTiming) {
        self.zipa_timing = timing;
    }
}

/// Borrowed view of the current utterance output after one feed step.
#[derive(Clone, Copy, Debug)]
pub struct FeedOutput<'a> {
    /// Full current token-aligned output tape after the feed step.
    tokens: &'a [OutputToken],
    /// Detected language for the current utterance state, when available.
    detected_language: Option<&'a str>,
}

impl<'a> FeedOutput<'a> {
    pub(crate) fn new(tokens: &'a [OutputToken], detected_language: Option<&'a str>) -> Self {
        Self {
            tokens,
            detected_language,
        }
    }

    pub fn tokens(self) -> &'a [OutputToken] {
        self.tokens
    }

    pub fn detected_language(self) -> Option<&'a str> {
        self.detected_language
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
pub enum Cut {
    /// No commit/carry boundary was found; the chunk should grow.
    NoCut,
    /// Commit everything strictly before this utterance-global token boundary.
    At(TokenIndex),
}

impl Cut {
    pub fn token_index(self) -> Option<TokenIndex> {
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

pub(crate) fn decode_timed_tokens(tokens: &[TimedToken]) -> Result<String> {
    let ids: Vec<TokenId> = tokens.iter().map(|token| token.token).collect();
    decode_token_ids(&ids)
}
