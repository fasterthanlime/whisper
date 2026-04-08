//! Token-indexed text buffer with optional word alignment.
//!
//! The `TextBuffer` is the core data structure for the transcription pipeline.
//! It stores a flat sequence of ASR tokens, each optionally marking a word
//! boundary. Text is decoded on demand from the tokenizer — tokens are canonical.
//!
//! A committed boundary (token index) divides the buffer into a stable prefix
//! (aligned, won't change) and a volatile tail (rewritten each feed() call).
//!
//! **Invariant:** Stage 2 (correction) may only make decisions against the
//! committed prefix. The live tail is for display/stitching only.

use crate::audio_buffer::{AudioBuffer, TimeRange};

// ── Token types ────────────────────────────────────────────────────────

/// Numeric identifier for a token.
pub type TokenId = u32;

/// A single ASR token with confidence information. Replaces all parallel
/// `(Vec<TokenId>, Vec<TokenLogprob>)` pairs.
#[derive(Debug, Clone, Copy)]
pub struct AsrToken {
    pub id: TokenId,
    /// Log-probability of this token.
    pub logprob: f32,
    /// Gap between top-1 and top-2 log-probabilities (>= 0).
    pub margin: f32,
}

/// A count of tokens. Not a sample count, not a byte offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct TokenCount(pub usize);

impl TokenCount {
    pub fn saturating_sub(self, rhs: TokenCount) -> TokenCount {
        TokenCount(self.0.saturating_sub(rhs.0))
    }
}

impl std::ops::Add for TokenCount {
    type Output = TokenCount;
    fn add(self, rhs: TokenCount) -> TokenCount {
        TokenCount(self.0 + rhs.0)
    }
}

impl std::ops::AddAssign for TokenCount {
    fn add_assign(&mut self, rhs: TokenCount) {
        self.0 += rhs.0;
    }
}

/// An index into a token sequence. Not a byte offset, not a sample count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct TokenIndex(pub usize);

// ── Word alignment ─────────────────────────────────────────────────────

/// Timing and audio for an aligned word. Time and audio are always set
/// together — you can't have one without the other.
#[derive(Debug, Clone)]
pub struct WordAlignment {
    /// Time range in the absolute session timeline.
    pub time: TimeRange,
    /// Audio samples for this word (copied from the decode session's buffer).
    pub audio: AudioBuffer,
}

/// Marker: this token starts a new word.
#[derive(Debug, Clone)]
pub struct WordStart {
    /// Alignment info. `None` for pending (unaligned) words,
    /// `Some` for committed (aligned) words.
    pub alignment: Option<WordAlignment>,
}

// ── Token entry ────────────────────────────────────────────────────────

/// A single entry in the text buffer: a token, optionally marking a word start.
#[derive(Debug, Clone)]
pub struct TokenEntry {
    pub token: AsrToken,
    /// `Some` if this token starts a new word.
    pub word: Option<WordStart>,
}

// ── TextBuffer ─────────────────────────────────────────────────────────

/// Token-indexed text buffer.
///
/// Stores a flat sequence of `TokenEntry` values. A committed boundary divides
/// the buffer into a stable prefix (entries before the boundary have alignment)
/// and a volatile tail (rewritten on each feed() call).
///
/// Text is decoded on demand from the tokenizer — tokens are the source of truth.
pub struct TextBuffer {
    entries: Vec<TokenEntry>,
    /// Token index: entries[..committed.0] are stable and aligned.
    committed: TokenIndex,
}

impl TextBuffer {
    /// Create an empty text buffer.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            committed: TokenIndex(0),
        }
    }

    /// All entries in the buffer.
    pub fn entries(&self) -> &[TokenEntry] {
        &self.entries
    }

    /// The committed prefix (stable, aligned).
    pub fn committed_entries(&self) -> &[TokenEntry] {
        &self.entries[..self.committed.0]
    }

    /// The pending tail (volatile, may lack alignment).
    pub fn pending_entries(&self) -> &[TokenEntry] {
        &self.entries[self.committed.0..]
    }

    /// Where the committed boundary is.
    pub fn committed_index(&self) -> TokenIndex {
        self.committed
    }

    /// Total number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of committed entries.
    pub fn committed_len(&self) -> TokenCount {
        TokenCount(self.committed.0)
    }

    /// Number of pending entries.
    pub fn pending_len(&self) -> TokenCount {
        TokenCount(self.entries.len() - self.committed.0)
    }

    /// Replace the pending tail with new entries.
    /// Committed prefix is untouched.
    pub fn replace_pending(&mut self, entries: impl IntoIterator<Item = TokenEntry>) {
        self.entries.truncate(self.committed.0);
        self.entries.extend(entries);
    }

    /// Advance the committed boundary by `n` tokens.
    /// Panics if this would go past the end of the buffer.
    pub fn advance_committed(&mut self, n: TokenCount) {
        let new_idx = self.committed.0 + n.0;
        assert!(
            new_idx <= self.entries.len(),
            "advance_committed({n:?}) would go past buffer end (len={})",
            self.entries.len(),
        );
        self.committed = TokenIndex(new_idx);
    }

    /// Fill in alignment for a range of entries.
    /// The range must be within the committed prefix.
    pub fn set_alignments(
        &mut self,
        start: TokenIndex,
        alignments: impl IntoIterator<Item = (usize, WordAlignment)>,
    ) {
        for (offset, alignment) in alignments {
            let idx = start.0 + offset;
            assert!(
                idx < self.committed.0,
                "set_alignments: index {idx} is past committed boundary {}",
                self.committed.0,
            );
            if let Some(ref mut word) = self.entries[idx].word {
                word.alignment = Some(alignment);
            }
        }
    }

    /// Iterate over words in the committed prefix.
    /// Each item is (word_start_index, &[TokenEntry]) covering all tokens in that word.
    pub fn committed_words(&self) -> WordIter<'_> {
        WordIter {
            entries: self.committed_entries(),
            pos: 0,
        }
    }

    /// Iterate over words in the pending tail.
    pub fn pending_words(&self) -> WordIter<'_> {
        WordIter {
            entries: self.pending_entries(),
            pos: 0,
        }
    }

    /// Collect the token IDs from a range of entries.
    pub fn token_ids(&self, range: std::ops::Range<usize>) -> Vec<TokenId> {
        self.entries[range].iter().map(|e| e.token.id).collect()
    }

    /// Collect the token IDs for the committed prefix.
    pub fn committed_token_ids(&self) -> Vec<TokenId> {
        self.token_ids(0..self.committed.0)
    }

    /// Collect the token IDs for the pending tail.
    pub fn pending_token_ids(&self) -> Vec<TokenId> {
        self.token_ids(self.committed.0..self.entries.len())
    }
}

impl Default for TextBuffer {
    fn default() -> Self {
        Self::new()
    }
}

// ── Word iterator ──────────────────────────────────────────────────────

/// Iterator over words in a slice of token entries.
/// Each item is the slice of entries for one word (first entry has `word: Some`).
pub struct WordIter<'a> {
    entries: &'a [TokenEntry],
    pos: usize,
}

impl<'a> Iterator for WordIter<'a> {
    /// (start_offset_within_slice, entries_for_this_word)
    type Item = (usize, &'a [TokenEntry]);

    fn next(&mut self) -> Option<Self::Item> {
        // Skip to the next word start
        while self.pos < self.entries.len() && self.entries[self.pos].word.is_none() {
            self.pos += 1;
        }
        if self.pos >= self.entries.len() {
            return None;
        }
        let word_start = self.pos;
        self.pos += 1;
        // Find end: next word start or end of slice
        while self.pos < self.entries.len() && self.entries[self.pos].word.is_none() {
            self.pos += 1;
        }
        Some((word_start, &self.entries[word_start..self.pos]))
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_buffer::{SampleRate, Seconds};

    fn tok(id: TokenId, word: bool) -> TokenEntry {
        TokenEntry {
            token: AsrToken {
                id,
                logprob: -0.5,
                margin: 0.3,
            },
            word: if word {
                Some(WordStart { alignment: None })
            } else {
                None
            },
        }
    }

    #[test]
    fn empty_buffer() {
        let buf = TextBuffer::new();
        assert!(buf.is_empty());
        assert_eq!(buf.committed_len(), TokenCount(0));
        assert_eq!(buf.pending_len(), TokenCount(0));
    }

    #[test]
    fn replace_pending_on_empty() {
        let mut buf = TextBuffer::new();
        buf.replace_pending(vec![tok(1, true), tok(2, false), tok(3, true)]);
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.committed_len(), TokenCount(0));
        assert_eq!(buf.pending_len(), TokenCount(3));
    }

    #[test]
    fn advance_committed() {
        let mut buf = TextBuffer::new();
        buf.replace_pending(vec![
            tok(1, true),
            tok(2, false),
            tok(3, true),
            tok(4, false),
            tok(5, true),
        ]);
        buf.advance_committed(TokenCount(3)); // commit first 3 tokens
        assert_eq!(buf.committed_len(), TokenCount(3));
        assert_eq!(buf.pending_len(), TokenCount(2));
        assert_eq!(buf.committed_token_ids(), vec![1, 2, 3]);
        assert_eq!(buf.pending_token_ids(), vec![4, 5]);
    }

    #[test]
    fn replace_pending_preserves_committed() {
        let mut buf = TextBuffer::new();
        buf.replace_pending(vec![tok(1, true), tok(2, false), tok(3, true), tok(4, false)]);
        buf.advance_committed(TokenCount(2));
        // Replace pending with new tokens
        buf.replace_pending(vec![tok(10, true), tok(11, false), tok(12, true)]);
        assert_eq!(buf.committed_token_ids(), vec![1, 2]);
        assert_eq!(buf.pending_token_ids(), vec![10, 11, 12]);
        assert_eq!(buf.len(), 5);
    }

    #[test]
    fn word_iteration() {
        let mut buf = TextBuffer::new();
        buf.replace_pending(vec![
            tok(1, true),   // word 1 start
            tok(2, false),  // word 1 cont
            tok(3, true),   // word 2 start
            tok(4, true),   // word 3 start
            tok(5, false),  // word 3 cont
            tok(6, false),  // word 3 cont
        ]);
        buf.advance_committed(TokenCount(6));

        let words: Vec<_> = buf.committed_words().collect();
        assert_eq!(words.len(), 3);

        // Word 1: tokens [1, 2]
        assert_eq!(words[0].0, 0);
        assert_eq!(words[0].1.len(), 2);
        assert_eq!(words[0].1[0].token.id, 1);
        assert_eq!(words[0].1[1].token.id, 2);

        // Word 2: token [3]
        assert_eq!(words[1].0, 2);
        assert_eq!(words[1].1.len(), 1);

        // Word 3: tokens [4, 5, 6]
        assert_eq!(words[2].0, 3);
        assert_eq!(words[2].1.len(), 3);
    }

    #[test]
    fn set_alignments() {
        let mut buf = TextBuffer::new();
        buf.replace_pending(vec![
            tok(1, true),
            tok(2, false),
            tok(3, true),
        ]);
        buf.advance_committed(TokenCount(3));

        let alignment = WordAlignment {
            time: TimeRange::new(Seconds(0.0), Seconds(0.5)),
            audio: AudioBuffer::new(vec![0.0; 8000], SampleRate::HZ_16000),
        };
        buf.set_alignments(TokenIndex(0), [(0, alignment)]);

        // Token 0 (word start) should now have alignment
        let word0 = &buf.committed_entries()[0];
        assert!(word0.word.as_ref().unwrap().alignment.is_some());

        // Token 2 (word start) should still have no alignment
        let word1 = &buf.committed_entries()[2];
        assert!(word1.word.as_ref().unwrap().alignment.is_none());
    }

    #[test]
    #[should_panic(expected = "would go past buffer end")]
    fn advance_committed_past_end() {
        let mut buf = TextBuffer::new();
        buf.replace_pending(vec![tok(1, true)]);
        buf.advance_committed(TokenCount(5));
    }

    #[test]
    fn pending_words_iteration() {
        let mut buf = TextBuffer::new();
        buf.replace_pending(vec![
            tok(1, true),   // committed word
            tok(2, false),
            tok(3, true),   // pending word 1
            tok(4, true),   // pending word 2
        ]);
        buf.advance_committed(TokenCount(2));

        let pending: Vec<_> = buf.pending_words().collect();
        assert_eq!(pending.len(), 2);
        assert_eq!(pending[0].1[0].token.id, 3);
        assert_eq!(pending[1].1[0].token.id, 4);
    }
}
