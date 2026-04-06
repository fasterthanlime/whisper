use crate::phonetic_lexicon::reduce_ipa_tokens;
use crate::word_split::sentence_word_tokens;
use crate::word_split::SentenceWordToken;

pub use bee_types::TranscriptAlignmentToken;

pub trait TranscriptAlignmentTiming {
    fn start_time(&self) -> f64;
    fn end_time(&self) -> f64;
    fn mean_logprob(&self) -> Option<f32> { None }
    fn min_logprob(&self) -> Option<f32> { None }
    fn mean_margin(&self) -> Option<f32> { None }
    fn min_margin(&self) -> Option<f32> { None }
}

impl TranscriptAlignmentTiming for TranscriptAlignmentToken {
    fn start_time(&self) -> f64 {
        self.start_time
    }

    fn end_time(&self) -> f64 {
        self.end_time
    }

    fn mean_logprob(&self) -> Option<f32> {
        self.mean_logprob
    }

    fn min_logprob(&self) -> Option<f32> {
        self.min_logprob
    }

    fn mean_margin(&self) -> Option<f32> {
        self.mean_margin
    }

    fn min_margin(&self) -> Option<f32> {
        self.min_margin
    }
}

pub use bee_types::TranscriptSpan;

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
    let tokens = sentence_word_tokens(transcript);
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
    tokens: &[SentenceWordToken],
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

    let (start_sec, end_sec, mean_logprob, min_logprob, mean_margin, min_margin) =
        if let Some(alignments) = alignments {
            if token_end <= alignments.len() {
                let span_alignments = &alignments[token_start..token_end];

                // Aggregate per-word logprob stats across the span
                let lps: Vec<f32> = span_alignments
                    .iter()
                    .filter_map(|a| a.mean_logprob())
                    .collect();
                let margins: Vec<f32> = span_alignments
                    .iter()
                    .filter_map(|a| a.mean_margin())
                    .collect();
                let min_lps: Vec<f32> = span_alignments
                    .iter()
                    .filter_map(|a| a.min_logprob())
                    .collect();
                let min_ms: Vec<f32> = span_alignments
                    .iter()
                    .filter_map(|a| a.min_margin())
                    .collect();

                let mean_lp = if lps.is_empty() {
                    None
                } else {
                    Some(lps.iter().sum::<f32>() / lps.len() as f32)
                };
                let min_lp = if min_lps.is_empty() {
                    None
                } else {
                    Some(min_lps.iter().copied().fold(f32::INFINITY, f32::min))
                };
                let mean_m = if margins.is_empty() {
                    None
                } else {
                    Some(margins.iter().sum::<f32>() / margins.len() as f32)
                };
                let min_m = if min_ms.is_empty() {
                    None
                } else {
                    Some(min_ms.iter().copied().fold(f32::INFINITY, f32::min))
                };

                (
                    Some(span_alignments[0].start_time()),
                    Some(span_alignments[span_alignments.len() - 1].end_time()),
                    mean_lp,
                    min_lp,
                    mean_m,
                    min_m,
                )
            } else {
                (None, None, None, None, None, None)
            }
        } else {
            (None, None, None, None, None, None)
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
        mean_logprob,
        min_logprob,
        mean_margin,
        min_margin,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::word_split::split_sentence_words;

    #[test]
    fn enumerate_transcript_spans_tracks_char_ranges() {
        let spans = enumerate_transcript_spans_with::<_, TranscriptAlignmentToken>(
            "for arc sixty four",
            4,
            None,
            |text| Some(split_sentence_words(text)),
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
                ..Default::default()
            },
            TranscriptAlignmentToken {
                start_time: 0.4,
                end_time: 0.6,
                ..Default::default()
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
