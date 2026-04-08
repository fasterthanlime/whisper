//! Token-indexed text buffer with optional word alignment.
//!
//! A `TextBuffer` is a flat sequence of `TokenEntry` values. It supports
//! splitting and merging — no index math needed. Alignment is a pure
//! function: takes a buffer, returns a buffer with alignment filled in.

use crate::audio_buffer::{AudioBuffer, Seconds, TimeRange};

// ── Token types ────────────────────────────────────────────────────────

/// Numeric identifier for a token.
pub type TokenId = u32;

/// A single ASR token with confidence information.
#[derive(Debug, Clone, Copy)]
pub struct AsrToken {
    pub id: TokenId,
    pub logprob: f32,
    pub margin: f32,
}

/// A count of tokens.
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

// ── Word alignment ─────────────────────────────────────────────────────

/// Timing and audio for an aligned word.
#[derive(Debug, Clone)]
pub struct WordAlignment {
    pub time: TimeRange,
    pub audio: AudioBuffer,
}

/// Marker: this token starts a new word.
#[derive(Debug, Clone)]
pub struct WordStart {
    pub alignment: Option<WordAlignment>,
}

// ── Token entry ────────────────────────────────────────────────────────

/// A single entry in a text buffer: a token, optionally marking a word start.
#[derive(Debug, Clone)]
pub struct TokenEntry {
    pub token: AsrToken,
    pub word: Option<WordStart>,
}

// ── TextBuffer ─────────────────────────────────────────────────────────

/// A flat sequence of token entries. Supports split and append.
/// No committed boundary — that's the caller's concern.
pub struct TextBuffer {
    entries: Vec<TokenEntry>,
}

impl TextBuffer {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn from_entries(entries: Vec<TokenEntry>) -> Self {
        Self { entries }
    }

    pub fn entries(&self) -> &[TokenEntry] {
        &self.entries
    }

    pub fn len(&self) -> TokenCount {
        TokenCount(self.entries.len())
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Take the first `n` entries out, returning them as a new TextBuffer.
    /// Self keeps the rest.
    pub fn split_off_front(&mut self, n: TokenCount) -> TextBuffer {
        let n = n.0.min(self.entries.len());
        let rest = self.entries.split_off(n);
        let front = std::mem::replace(&mut self.entries, rest);
        TextBuffer { entries: front }
    }

    /// Append another buffer's entries to the end of this one.
    pub fn append(&mut self, other: TextBuffer) {
        self.entries.extend(other.entries);
    }

    /// Replace all entries.
    pub fn replace(&mut self, entries: Vec<TokenEntry>) {
        self.entries = entries;
    }

    /// Collect token IDs.
    pub fn token_ids(&self) -> Vec<TokenId> {
        self.entries.iter().map(|e| e.token.id).collect()
    }

    /// Snap a token count back to a word boundary.
    ///
    /// Given a desired split point `n`, returns the largest `m <= n` such
    /// that splitting at `m` does not cut through a word. Returns zero if
    /// no complete word fits within the first `n` tokens.
    pub fn snap_to_word_boundary(&self, n: TokenCount) -> TokenCount {
        let limit = n.0.min(self.entries.len());
        // Walk backward from `limit` to find a position that's either
        // at the buffer end or on a word-start (safe to split before).
        let mut split = limit;
        while split > 0 && split < self.entries.len() && self.entries[split].word.is_none() {
            split -= 1;
        }
        TokenCount(split)
    }

    /// Iterate over words. Each item is the slice of entries for one word.
    pub fn words(&self) -> WordIter<'_> {
        WordIter {
            entries: &self.entries,
            pos: 0,
        }
    }
}

impl Default for TextBuffer {
    fn default() -> Self {
        Self::new()
    }
}

// ── Alignment (pure function) ─────────────────────────────────────────

/// Result from a forced aligner — one per word.
pub struct AlignmentItem {
    pub word: String,
    pub start: Seconds,
    pub end: Seconds,
}

/// Fill in word alignment on a TextBuffer. Pure function: takes a buffer,
/// returns it with `WordStart.alignment` set on each word-start entry.
///
/// `items` are the forced aligner results (one per word, zero-based).
/// `audio` is the audio the aligner ran on (for slicing per-word audio).
/// `audio_offset` is the absolute time offset for this audio in the session.
pub fn align(
    mut buf: TextBuffer,
    items: &[AlignmentItem],
    audio: &AudioBuffer,
    audio_offset: Seconds,
) -> TextBuffer {
    let mut word_idx = 0;
    for entry in &mut buf.entries {
        if entry.word.is_some() && word_idx < items.len() {
            let item = &items[word_idx];
            let time = TimeRange::new(
                item.start + audio_offset,
                item.end + audio_offset,
            );
            let word_audio = audio.slice(TimeRange::new(item.start, item.end));
            entry.word = Some(WordStart {
                alignment: Some(WordAlignment {
                    time,
                    audio: word_audio,
                }),
            });
            word_idx += 1;
        }
    }
    buf
}

// ── Confidence from token entries ──────────────────────────────────────

/// Compute word-level confidence from a slice of token entries.
pub fn confidence(entries: &[TokenEntry]) -> bee_types::Confidence {
    let n = entries.len() as f32;
    bee_types::Confidence {
        mean_lp: entries.iter().map(|e| e.token.logprob).sum::<f32>() / n,
        min_lp: entries.iter().map(|e| e.token.logprob).fold(f32::INFINITY, f32::min),
        mean_m: entries.iter().map(|e| e.token.margin).sum::<f32>() / n,
        min_m: entries.iter().map(|e| e.token.margin).fold(f32::INFINITY, f32::min),
    }
}

// ── Word iterator ──────────────────────────────────────────────────────

/// Iterator over words in a slice of token entries.
pub struct WordIter<'a> {
    entries: &'a [TokenEntry],
    pos: usize,
}

impl<'a> Iterator for WordIter<'a> {
    type Item = &'a [TokenEntry];

    fn next(&mut self) -> Option<Self::Item> {
        // Skip to next word start
        while self.pos < self.entries.len() && self.entries[self.pos].word.is_none() {
            self.pos += 1;
        }
        if self.pos >= self.entries.len() {
            return None;
        }
        let word_start = self.pos;
        self.pos += 1;
        while self.pos < self.entries.len() && self.entries[self.pos].word.is_none() {
            self.pos += 1;
        }
        Some(&self.entries[word_start..self.pos])
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_buffer::SampleRate;

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
        assert_eq!(buf.len(), TokenCount(0));
    }

    #[test]
    fn split_off_front() {
        let mut buf = TextBuffer::from_entries(vec![
            tok(1, true),
            tok(2, false),
            tok(3, true),
            tok(4, false),
            tok(5, true),
        ]);
        let front = buf.split_off_front(TokenCount(3));
        assert_eq!(front.token_ids(), vec![1, 2, 3]);
        assert_eq!(buf.token_ids(), vec![4, 5]);
    }

    #[test]
    fn split_off_front_all() {
        let mut buf = TextBuffer::from_entries(vec![tok(1, true), tok(2, false)]);
        let front = buf.split_off_front(TokenCount(5)); // more than len
        assert_eq!(front.token_ids(), vec![1, 2]);
        assert!(buf.is_empty());
    }

    #[test]
    fn append_buffers() {
        let mut a = TextBuffer::from_entries(vec![tok(1, true), tok(2, false)]);
        let b = TextBuffer::from_entries(vec![tok(3, true), tok(4, false)]);
        a.append(b);
        assert_eq!(a.token_ids(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn word_iteration() {
        let buf = TextBuffer::from_entries(vec![
            tok(1, true),   // word 1
            tok(2, false),
            tok(3, true),   // word 2
            tok(4, true),   // word 3
            tok(5, false),
            tok(6, false),
        ]);
        let words: Vec<_> = buf.words().collect();
        assert_eq!(words.len(), 3);
        assert_eq!(words[0].len(), 2); // [1, 2]
        assert_eq!(words[1].len(), 1); // [3]
        assert_eq!(words[2].len(), 3); // [4, 5, 6]
    }

    #[test]
    fn align_fills_word_starts() {
        let buf = TextBuffer::from_entries(vec![
            tok(1, true),   // word 0
            tok(2, false),
            tok(3, true),   // word 1
        ]);
        let audio = AudioBuffer::new(vec![0.0; 16000], SampleRate::HZ_16000);
        let items = vec![
            AlignmentItem {
                word: "hello".into(),
                start: Seconds(0.0),
                end: Seconds(0.3),
            },
            AlignmentItem {
                word: "world".into(),
                start: Seconds(0.3),
                end: Seconds(0.6),
            },
        ];

        let aligned = align(buf, &items, &audio, Seconds(5.0));

        let words: Vec<_> = aligned.words().collect();
        assert_eq!(words.len(), 2);

        // Word 0 aligned with absolute offset
        let w0 = words[0][0].word.as_ref().unwrap().alignment.as_ref().unwrap();
        assert_eq!(w0.time.start, Seconds(5.0));
        assert_eq!(w0.time.end, Seconds(5.3));

        // Word 1
        let w1 = words[1][0].word.as_ref().unwrap().alignment.as_ref().unwrap();
        assert_eq!(w1.time.start, Seconds(5.3));
        assert_eq!(w1.time.end, Seconds(5.6));
    }
}
