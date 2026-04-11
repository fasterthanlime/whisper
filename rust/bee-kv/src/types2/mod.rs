//! Canonical utterance-global types for the next rollback model.
//!
//! Intent:
//! - one canonical token sequence
//! - one utterance-global coordinate system
//! - one cut in token space
//! - no window-relative or carry-relative coordinates
//! - one tokenizer slot initialized once at process startup
//! - sample space is canonical for audio; time is derived from samples
//! - transcript token state and KV state can be owned under one truncation boundary
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
//! - `ChunkInfo.tokens` is the single source of truth for chunk token order
//! - `TimedToken.starts_word_len` is present only on the first token of a word
//! - if `TimedToken.starts_word_len` is `Some(len)`, then the next `len` tokens in the
//!   same `ChunkInfo` belong to that word
//! - `Cut::At(index)` refers to an utterance-global token boundary, not a local
//!   offset inside a chunk

mod audio;
mod tokenizer;
mod tokens;
mod transcript;
mod utterance;

pub(crate) use audio::*;
pub(crate) use tokenizer::*;
pub(crate) use tokens::*;
#[allow(unused_imports)]
pub(crate) use transcript::*;
#[allow(unused_imports)]
pub(crate) use utterance::*;
