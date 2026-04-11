//! `Tape` is the synchronization point between transcript tokens and decoder KV.
//!
//! Desired state from the README:
//! - token boundaries remain the canonical cut/replay coordinates
//! - `stable` may stay alive in KV across feeds
//! - `carry` is replayed in prompt token IDs, not kept as independent KV state
//! - `preview` is truncated and regenerated
//!
//! That means `Tape` must remember both:
//! - utterance token boundaries
//! - decoder-visible positions for those boundaries
//!
//! The two are related but not interchangeable.

use bee_qwen3_asr::decoder::KVCache;
use bee_qwen3_asr::generate::{self, ConfidenceMode, DecodeStopReason, TokenConfidence};
use bee_qwen3_asr::mlx_rs::Array;
use bee_qwen3_asr::mlx_rs::error::Exception;
use bee_qwen3_asr::model::Qwen3ASRModel;
use compact_str::CompactString;

use crate::tokens::{TokenCount, UtteranceTokenRange, decode_timed_tokens};
use crate::{OutputToken, TokenIndex};

/// Canonical utterance-global token storage with a 1:1 mapping between
/// vector position and utterance token index.
///
/// Intent:
/// - token position is canonical
/// - callers never manipulate the backing vector directly
/// - truncation and append operations preserve the index invariant
///
/// Invariants:
/// - `tokens[i].index == TokenIndex::new(i)` for every element
/// - token order is utterance-global and never rebased
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TokenTape {
    tokens: Vec<OutputToken>,
}

impl TokenTape {
    /// Creates an empty token tape.
    pub(crate) fn new() -> Self {
        Self { tokens: Vec::new() }
    }

    /// Returns the number of tokens stored in this tape.
    pub(crate) fn len(&self) -> TokenCount {
        TokenCount::new(self.tokens.len())
    }

    /// Returns whether this tape is empty.
    pub(crate) fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Returns the token boundary immediately after the last stored token.
    pub(crate) fn end(&self) -> TokenIndex {
        TokenIndex::new(self.tokens.len())
    }

    /// Returns the full tape as a slice.
    pub(crate) fn as_slice(&self) -> &[OutputToken] {
        &self.tokens
    }

    /// Returns a borrowed token slice over an utterance-global token range.
    ///
    /// Invariants:
    /// - `range` must lie within this tape
    pub(crate) fn slice(&self, range: UtteranceTokenRange) -> &[OutputToken] {
        assert!(
            range.start <= range.end && range.end <= self.end(),
            "token tape slice must lie within the tape"
        );
        let start = range.start.as_usize();
        let end = range.end.as_usize();
        &self.tokens[start..end]
    }

    /// Appends already-indexed tokens to the end of the tape.
    ///
    /// Invariants:
    /// - appended tokens must begin exactly at the current end index
    /// - appended token indices must continue contiguously
    pub(crate) fn append(&mut self, tokens: Vec<OutputToken>) {
        let expected_start = self.end();
        for (offset, token) in tokens.iter().enumerate() {
            let expected = TokenIndex::new(expected_start.as_usize() + offset);
            assert!(
                token.timed_token().index() == expected,
                "token tape append requires token index {}, got {}",
                expected,
                token.timed_token().index()
            );
        }
        self.tokens.extend(tokens);
    }

    /// Truncates the tape to `end`, keeping tokens strictly before that boundary.
    ///
    /// Invariants:
    /// - `end` must lie within this tape
    pub(crate) fn truncate_to(&mut self, end: TokenIndex) {
        assert!(
            end <= self.end(),
            "token tape truncate must lie within the tape"
        );
        self.tokens.truncate(end.as_usize());
    }

    /// Decodes all stored tokens on demand.
    pub(crate) fn decode_text(&self) -> anyhow::Result<String> {
        let timed: Vec<_> = self
            .tokens
            .iter()
            .map(|token| token.timed_token())
            .collect();
        decode_timed_tokens(&timed)
    }
}

/// KV cache state synchronized to a token tape.
///
/// Intent:
/// - keep KV truncation under the same ownership boundary as transcript truncation
/// - represent the empty decode state as an empty cache at token boundary 0
///
/// Invariants:
/// - `decoder_position` is the number of decoder-visible prompt/generated tokens
///   currently represented in the cache
/// - the cache is truncated/extended consistently with `decoder_position`
pub(crate) struct KvTape {
    cache: Option<KVCache>,
    decoder_position: usize,
}

impl KvTape {
    /// Creates an empty KV tape at utterance token boundary 0.
    pub(crate) fn new(num_layers: usize) -> Self {
        Self {
            cache: Some(KVCache::new(num_layers)),
            decoder_position: 0,
        }
    }

    /// Returns the decoder-visible position represented by the current cache state.
    pub(crate) fn decoder_position(&self) -> usize {
        self.decoder_position
    }

    /// Advances the cached decoder position after a successful decode/prefill step.
    ///
    /// Invariants:
    /// - `decoder_position` must not move backward
    pub(crate) fn advance_to(&mut self, decoder_position: usize) {
        tracing::trace!(
            from = self.decoder_position,
            to = decoder_position,
            "bee_roll.kv_tape.advance_to"
        );
        assert!(
            decoder_position >= self.decoder_position,
            "KV tape cannot advance backward"
        );
        self.decoder_position = decoder_position;
    }

    /// Truncates the KV cache to `decoder_position`, keeping state strictly
    /// before that boundary.
    ///
    /// Invariants:
    /// - `decoder_position` must lie within the current cache boundary
    /// - the cache is truncated in lockstep with `self.decoder_position`
    pub(crate) fn truncate_to(&mut self, decoder_position: usize) {
        tracing::trace!(
            from = self.decoder_position,
            to = decoder_position,
            "bee_roll.kv_tape.truncate_to"
        );
        assert!(
            decoder_position <= self.decoder_position,
            "KV tape truncate must lie within the cache: \
             requested {decoder_position}, current {}",
            self.decoder_position,
        );
        if let Some(cache) = self.cache.as_mut() {
            cache.truncate(decoder_position);
        }
        self.decoder_position = decoder_position;
    }
}

/// The synchronized token/KV state that moves forward and rolls back together.
///
/// Intent:
/// - token-space rollback is chosen at the utterance layer
/// - transcript rollback and KV rollback are applied here as one operation
/// - higher-level utterance code must not truncate tokens and KV separately
///
/// Invariants:
/// - token truncation and KV truncation happen through one owner
/// - token-space boundaries and decoder-space boundaries are tracked separately
/// - do not assume "rewind N transcript tokens" implies "rewind KV by N"
pub(crate) struct Tape {
    /// Canonical utterance-global token sequence.
    tokens: TokenTape,
    /// KV cache state synchronized to the token sequence.
    kv: KvTape,
    /// Decoder-visible cache position for each token boundary in the current state.
    ///
    /// Invariant:
    /// - length is always `tokens.len() + 1`
    /// - entry `i` is the decoder position that keeps tokens `[0, i)`
    ///
    /// Strategic note:
    /// - this mapping is what lets `Utterance` choose replay/cut boundaries in
    ///   token space without lying to itself about KV geometry
    decoder_boundaries: Vec<usize>,
    detected_language: Option<CompactString>,
}

impl Tape {
    /// Creates an empty transcript at utterance token boundary 0.
    pub(crate) fn new(num_layers: usize) -> Self {
        Self {
            tokens: TokenTape::new(),
            kv: KvTape::new(num_layers),
            decoder_boundaries: vec![0],
            detected_language: None,
        }
    }

    /// Returns the token boundary immediately after the last stored token.
    pub(crate) fn end(&self) -> TokenIndex {
        self.tokens.end()
    }

    /// Returns a borrowed token slice over an utterance-global token range.
    pub(crate) fn slice(&self, range: UtteranceTokenRange) -> &[OutputToken] {
        self.tokens.slice(range)
    }

    /// Appends already-indexed decoded tokens to the token tape.
    ///
    /// Invariants:
    /// - the appended tokens must continue contiguously from the current end
    /// - callers must separately keep decoder position in sync with the cache
    pub(crate) fn append(&mut self, tokens: Vec<OutputToken>) {
        let appended = tokens.len();
        self.tokens.append(tokens);
        self.decoder_boundaries
            .extend(std::iter::repeat_n(self.kv.decoder_position(), appended));
    }

    /// Truncates transcript tokens and KV state to the chosen rewind points.
    ///
    /// Invariants:
    /// - `token_end` must lie within the current transcript
    pub(crate) fn truncate_to(&mut self, token_end: TokenIndex, decoder_position: usize) {
        tracing::trace!(
            token_end = token_end.as_usize(),
            decoder_position,
            current_kv = self.kv.decoder_position(),
            current_tokens = self.tokens.len().as_usize(),
            "bee_roll.tape.truncate_to"
        );
        self.tokens.truncate_to(token_end);
        self.decoder_boundaries.truncate(token_end.as_usize() + 1);
        self.kv.truncate_to(decoder_position);
        if let Some(last) = self.decoder_boundaries.last_mut() {
            *last = decoder_position;
        }
    }

    /// Rebind decoder boundaries for the carry region after prompt replay.
    ///
    /// After a decode pass replays carry tokens `[carry_start, carry_end)` in the
    /// prompt, their decoder boundaries must reflect their new positions in the
    /// current KV cache.  Without this, a later cut that promotes carried tokens
    /// to stable would read stale decoder positions from a previous decode cycle.
    pub(crate) fn rebind_carry_boundaries(
        &mut self,
        carry_start: TokenIndex,
        carry_end: TokenIndex,
        prompt_end_position: usize,
    ) {
        let carry_len = carry_end.as_usize() - carry_start.as_usize();
        if carry_len == 0 {
            return;
        }
        tracing::trace!(
            carry_start = carry_start.as_usize(),
            carry_end = carry_end.as_usize(),
            carry_len,
            prompt_end_position,
            old_first = self.decoder_boundaries[carry_start.as_usize() + 1],
            old_last = self.decoder_boundaries[carry_end.as_usize()],
            "bee_roll.tape.rebind_carry_boundaries"
        );
        for k in 1..=carry_len {
            let idx = carry_start.as_usize() + k;
            self.decoder_boundaries[idx] = prompt_end_position - carry_len + k;
        }
    }

    pub(crate) fn tokens(&self) -> &[OutputToken] {
        self.tokens.as_slice()
    }

    pub(crate) fn token_mut(&mut self, index: usize) -> Option<&mut OutputToken> {
        self.tokens.tokens.get_mut(index)
    }

    pub(crate) fn detected_language(&self) -> Option<&str> {
        self.detected_language.as_deref()
    }

    pub(crate) fn set_detected_language(&mut self, detected_language: Option<CompactString>) {
        self.detected_language = detected_language;
    }

    pub(crate) fn decoder_position(&self) -> usize {
        self.kv.decoder_position()
    }

    pub(crate) fn advance_decoder_to(&mut self, decoder_position: usize) {
        self.kv.advance_to(decoder_position);
        if let Some(last) = self.decoder_boundaries.last_mut() {
            *last = decoder_position;
        }
    }

    pub(crate) fn decoder_position_for_boundary(&self, token_boundary: TokenIndex) -> usize {
        self.decoder_boundaries[token_boundary.as_usize()]
    }

    pub(crate) fn prefill_and_decode(
        &mut self,
        model: &Qwen3ASRModel,
        prompt_tokens: &[i32],
        audio_features: &Array,
        max_new_tokens: usize,
        confidence_mode: ConfidenceMode,
    ) -> Result<(Vec<i32>, Vec<TokenConfidence>, usize, DecodeStopReason), Exception> {
        // Strategic note:
        // - this function only knows about decoder-visible prompt/generated tokens
        // - it must not infer transcript replay policy on its own
        // - the utterance layer decides which token boundary stays in KV and
        //   which tokens are replayed as carry in the prompt
        let start_position = self.kv.decoder_position();
        let (generated, confidences, next_position, stop_reason) = generate::prefill_and_decode(
            model,
            prompt_tokens,
            audio_features,
            &mut self.kv.cache,
            start_position,
            max_new_tokens,
            confidence_mode,
        )?;
        self.kv.advance_to(next_position);
        Ok((generated, confidences, next_position, stop_reason))
    }

    pub(crate) fn append_decoded(
        &mut self,
        tokens: Vec<OutputToken>,
        prompt_end_position: usize,
        next_decoder_position: usize,
    ) {
        let generated = tokens.len();
        tracing::trace!(
            generated,
            prompt_end_position,
            next_decoder_position,
            current_kv = self.kv.decoder_position(),
            current_tokens = self.tokens.len().as_usize(),
            "bee_roll.tape.append_decoded"
        );
        let expected_end = prompt_end_position.saturating_add(generated);
        assert!(
            expected_end == next_decoder_position,
            "decoded token count must match decoder-position advance"
        );
        self.tokens.append(tokens);
        self.decoder_boundaries.reserve(generated);
        let mut position = prompt_end_position;
        for _ in 0..generated {
            position += 1;
            self.decoder_boundaries.push(position);
        }
        self.kv.advance_to(next_decoder_position);
    }
}
