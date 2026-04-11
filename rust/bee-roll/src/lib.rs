//! Canonical utterance-global types for the next rollback model.
//!
//! Intent:
//! - one canonical token sequence
//! - one utterance-global coordinate system
//! - one cut in token space
//! - no window-relative or carry-relative coordinates
//! - one tokenizer slot initialized once at process startup
//! - sample space is canonical for audio; time is a derived view used for visualization
//! - `Utterance` is the owning boundary for synchronized rollback state
//!
//! Non-goals:
//! - this module tree does not describe the current `bee-kv` implementation
//! - this module tree does not preserve the current bridge/carry bookkeeping shape
//! - this module tree does not store decoded text as canonical state
//!
//! Invariants:
//! - every token index is relative to the beginning of the utterance
//! - every time is relative to the beginning of the utterance
//! - `TimeRange` values are utterance-global, never window-local
//! - audio buffers store utterance-global sample start plus owned samples
//! - `Cut::At(index)` refers to an utterance-global token boundary, not a local
//!   offset inside a token slice

mod audio;
mod tokenizer;
mod tokens;
mod transcript;
mod utterance;

/// Audio sample rate in Hz expected by the ASR tokenizer/audio helpers.
pub(crate) const SAMPLE_RATE: u32 = 16_000;

pub use tokens::{
    AsrTokenAlternative, AsrTokenConfidence, Cut, FeedOutput, G2pTokenIpa, OutputToken, TimeRange,
    TimedToken, TokenId, TokenIndex, UtteranceTime, ZipaPhoneSpan,
};
pub use utterance::{Cutter, Listener, Utterance};

pub(crate) use audio::*;
pub(crate) use tokenizer::*;
