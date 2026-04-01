use serde::{Deserialize, Serialize};

use crate::phonetic_lexicon::reduce_ipa_tokens;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptAlignmentToken {
    pub start_time: f64,
    pub end_time: f64,
}

pub trait TranscriptAlignmentTiming {
    fn start_time(&self) -> f64;
    fn end_time(&self) -> f64;
}

impl TranscriptAlignmentTiming for TranscriptAlignmentToken {
    fn start_time(&self) -> f64 {
        self.start_time
    }

    fn end_time(&self) -> f64 {
        self.end_time
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TranscriptToken {
    char_start: usize,
    char_end: usize,
    text: String,
}

pub fn enumerate_transcript_spans_with<F, A>(
    transcript: &str,
    max_span_words: usize,
    alignments: Option<&[A]>,
    mut ipa_for_text: F,
) -> Vec<TranscriptSpan>
where
    F: FnMut(&str) -> Option<Vec<String>>,
    A: TranscriptAlignmentTiming,
{
    let tokens = tokenize_transcript(transcript);
    if tokens.is_empty() {
        return Vec::new();
    }

    let mut spans = Vec::new();
    for token_start in 0..tokens.len() {
        let max_end = (token_start + max_span_words).min(tokens.len());
        for token_end in (token_start + 1)..=max_end {
            let Some(span) = span_from_tokens(
                transcript,
                &tokens,
                token_start,
                token_end,
                alignments,
                &mut ipa_for_text,
            ) else {
                continue;
            };
            spans.push(span);
        }
    }
    spans
}

fn span_from_tokens<F, A>(
    transcript: &str,
    tokens: &[TranscriptToken],
    token_start: usize,
    token_end: usize,
    alignments: Option<&[A]>,
    ipa_for_text: &mut F,
) -> Option<TranscriptSpan>
where
    F: FnMut(&str) -> Option<Vec<String>>,
    A: TranscriptAlignmentTiming,
{
    let first = tokens.get(token_start)?;
    let last = tokens.get(token_end.checked_sub(1)?)?;
    let text = transcript
        .get(first.char_start..last.char_end)?
        .trim()
        .to_string();
    if text.is_empty() {
        return None;
    }
    let ipa_tokens = ipa_for_text(&text)?;
    if ipa_tokens.is_empty() {
        return None;
    }

    let (start_sec, end_sec) = if let Some(alignments) = alignments {
        if token_end <= alignments.len() {
            (
                Some(alignments[token_start].start_time()),
                Some(alignments[token_end - 1].end_time()),
            )
        } else {
            (None, None)
        }
    } else {
        (None, None)
    };

    Some(TranscriptSpan {
        token_start,
        token_end,
        char_start: first.char_start,
        char_end: last.char_end,
        start_sec,
        end_sec,
        reduced_ipa_tokens: reduce_ipa_tokens(&ipa_tokens),
        ipa_tokens,
        text,
    })
}

fn tokenize_transcript(transcript: &str) -> Vec<TranscriptToken> {
    let mut tokens = Vec::new();
    let mut current_start = None;

    for (idx, ch) in transcript.char_indices() {
        if is_word_char(ch) {
            current_start.get_or_insert(idx);
            continue;
        }
        if let Some(start) = current_start.take() {
            tokens.push(TranscriptToken {
                char_start: start,
                char_end: idx,
                text: transcript[start..idx].to_string(),
            });
        }
    }

    if let Some(start) = current_start {
        tokens.push(TranscriptToken {
            char_start: start,
            char_end: transcript.len(),
            text: transcript[start..].to_string(),
        });
    }

    tokens
}

fn is_word_char(ch: char) -> bool {
    ch.is_alphanumeric() || matches!(ch, '_' | '\'' | '-')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enumerate_transcript_spans_tracks_char_ranges() {
        let spans = enumerate_transcript_spans_with::<_, TranscriptAlignmentToken>(
            "for arc sixty four",
            4,
            None,
            |text| Some(text.split_whitespace().map(|s| s.to_string()).collect()),
        );
        let whole = spans
            .iter()
            .find(|span| span.token_start == 0 && span.token_end == 4)
            .expect("whole-span query");
        assert_eq!(whole.text, "for arc sixty four");
        assert_eq!(whole.char_start, 0);
        assert_eq!(whole.char_end, "for arc sixty four".len());
    }

    #[test]
    fn enumerate_transcript_spans_skips_empty_ipa() {
        let spans = enumerate_transcript_spans_with::<_, TranscriptAlignmentToken>(
            "serde_json",
            2,
            None,
            |_text| None,
        );
        assert!(spans.is_empty());
    }

    #[test]
    fn uses_alignment_timing() {
        let alignments = [
            TranscriptAlignmentToken {
                start_time: 0.2,
                end_time: 0.4,
            },
            TranscriptAlignmentToken {
                start_time: 0.4,
                end_time: 0.6,
            },
        ];
        let spans =
            enumerate_transcript_spans_with("arc sixty", 2, Some(&alignments[..]), |text| {
                Some(vec![text.to_string()])
            });
        assert_eq!(spans.len(), 3);
        assert_eq!(spans[0].start_sec, Some(0.2));
        assert_eq!(spans[0].end_sec, Some(0.4));
        assert_eq!(spans[1].start_sec, Some(0.2));
        assert_eq!(spans[1].end_sec, Some(0.6));
    }
}
