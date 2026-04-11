use bee_qwen3_asr::decoder::KVCache;

use crate::{ChunkInfo, TimedToken, TokenCount, TokenId, TokenIndex, UtteranceTokenRange};

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
    tokens: Vec<TimedToken>,
}

impl TokenTape {
    /// Creates an empty token tape.
    pub(crate) fn new() -> Self {
        Self { tokens: Vec::new() }
    }

    /// Creates a token tape from a fully validated token vector.
    pub(crate) fn from_tokens(tokens: Vec<TimedToken>) -> Self {
        for (i, token) in tokens.iter().enumerate() {
            assert!(
                token.index == TokenIndex::new(i),
                "token tape requires token index {i}, got {}",
                token.index
            );
        }
        Self { tokens }
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
    pub(crate) fn as_slice(&self) -> &[TimedToken] {
        &self.tokens
    }

    /// Returns a copied chunk view over an utterance-global token range.
    ///
    /// Invariants:
    /// - `range` must lie within this tape
    pub(crate) fn slice(&self, range: UtteranceTokenRange) -> ChunkInfo {
        assert!(
            range.end <= self.end(),
            "token tape slice must lie within the tape"
        );
        let start = range.start.as_usize();
        let end = range.end.as_usize();
        ChunkInfo::new(self.tokens[start..end].to_vec())
    }

    /// Appends already-indexed tokens to the end of the tape.
    ///
    /// Invariants:
    /// - appended tokens must begin exactly at the current end index
    /// - appended token indices must continue contiguously
    pub(crate) fn append(&mut self, tokens: Vec<TimedToken>) {
        let expected_start = self.end();
        for (offset, token) in tokens.iter().enumerate() {
            let expected = TokenIndex::new(expected_start.as_usize() + offset);
            assert!(
                token.index == expected,
                "token tape append requires token index {}, got {}",
                expected,
                token.index
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
        let ids: Vec<TokenId> = self.tokens.iter().map(|token| token.token).collect();
        crate::decode_token_ids(&ids)
    }
}

/// KV cache state synchronized to a token tape.
///
/// Intent:
/// - keep KV truncation under the same ownership boundary as transcript truncation
/// - hide raw `Option<KVCache>` from higher-level rollback code
///
/// Invariants:
/// - `end` is the cache/token boundary represented by the current cache state
/// - if `cache` is `Some`, it has been truncated/extended consistently with `end`
pub(crate) struct KvTape {
    cache: Option<KVCache>,
    end: TokenIndex,
}

impl KvTape {
    /// Creates an empty KV tape at utterance token boundary 0.
    pub(crate) fn new() -> Self {
        Self {
            cache: None,
            end: TokenIndex::new(0),
        }
    }

    /// Returns the token boundary represented by the current cache state.
    pub(crate) fn end(&self) -> TokenIndex {
        self.end
    }

    /// Returns the raw cache so decode code can prefill/decode into it.
    pub(crate) fn cache_mut(&mut self) -> &mut Option<KVCache> {
        &mut self.cache
    }

    /// Advances the cached token boundary after a successful decode/prefill step.
    ///
    /// Invariants:
    /// - `end` must not move backward
    pub(crate) fn advance_to(&mut self, end: TokenIndex) {
        assert!(end >= self.end, "KV tape cannot advance backward");
        self.end = end;
    }

    /// Truncates the KV cache to `end`, keeping state strictly before that boundary.
    ///
    /// Invariants:
    /// - `end` must lie within the current cache boundary
    /// - if a cache exists, it is truncated in lockstep with `self.end`
    pub(crate) fn truncate_to(&mut self, end: TokenIndex) {
        assert!(end <= self.end, "KV tape truncate must lie within the tape");
        if let Some(cache) = self.cache.as_mut() {
            cache.truncate(end.as_usize());
        }
        self.end = end;
    }
}

/// The synchronized token/KV state that moves forward and rolls back together.
///
/// Intent:
/// - all token-space cuts happen here
/// - transcript rollback and KV rollback share one operation
/// - higher-level utterance code should not truncate tokens and KV separately
///
/// Invariants:
/// - `tokens.end() == kv.end()`
pub(crate) struct Transcript {
    /// Canonical utterance-global token sequence.
    pub(crate) tokens: TokenTape,
    /// KV cache state synchronized to the token sequence.
    pub(crate) kv: KvTape,
}

impl Transcript {
    /// Creates an empty transcript at utterance token boundary 0.
    pub(crate) fn new() -> Self {
        Self {
            tokens: TokenTape::new(),
            kv: KvTape::new(),
        }
    }

    /// Returns the token boundary immediately after the last committed token.
    pub(crate) fn end(&self) -> TokenIndex {
        let tokens_end = self.tokens.end();
        let kv_end = self.kv.end();
        assert!(
            tokens_end == kv_end,
            "transcript token/KV boundaries must stay synchronized"
        );
        tokens_end
    }

    /// Returns a copied chunk view over an utterance-global token range.
    pub(crate) fn slice(&self, range: UtteranceTokenRange) -> ChunkInfo {
        self.tokens.slice(range)
    }

    /// Appends already-indexed tokens and advances the KV boundary to match.
    ///
    /// Invariants:
    /// - the appended tokens must continue contiguously from the current end
    /// - callers must only use this after the raw KV cache has already been
    ///   advanced by the matching decode step
    pub(crate) fn append(&mut self, tokens: Vec<TimedToken>) {
        self.tokens.append(tokens);
        self.kv.advance_to(self.tokens.end());
    }

    /// Truncates transcript tokens and KV state to the same token boundary.
    ///
    /// Invariants:
    /// - `end` must lie within the current transcript
    pub(crate) fn truncate_to(&mut self, end: TokenIndex) {
        self.tokens.truncate_to(end);
        self.kv.truncate_to(end);
    }
}
