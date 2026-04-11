use std::env;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{Context, Result, anyhow};
use bee_g2p_charsiu_mlx::engine::G2pEngine;
use bee_g2p_charsiu_mlx::ownership::{ByteSpan, OwnershipSpan};
use bee_phonetic::{
    ComparisonToken, normalize_ipa_for_comparison, normalize_ipa_for_comparison_with_spans,
    parse_reviewed_ipa,
};
use bee_qwen3_asr::tokenizers::Tokenizer;
use facet::Facet;
use regex::Regex;

#[derive(Debug, Clone, Facet)]
pub struct TranscriptWord {
    pub word: String,
    pub char_start: usize,
    pub char_end: usize,
}

#[derive(Debug, Clone, Facet)]
pub struct TextWordIpa {
    pub word: String,
    pub ipa: String,
    pub char_start: usize,
    pub char_end: usize,
}

#[derive(Debug, Clone, Facet)]
pub struct TokenPieceIpaSpan {
    pub word_index: Option<usize>,
    pub word_surface: Option<String>,
    pub token_index: usize,
    pub token: String,
    pub token_surface: String,
    pub token_char_start: usize,
    pub token_char_end: usize,
    pub ipa_byte_start: usize,
    pub ipa_byte_end: usize,
    pub ipa_text: String,
    pub ownership_score: f32,
}

#[derive(Debug, Clone, Facet)]
pub struct TokenPiecePhones {
    pub word_index: Option<usize>,
    pub word_surface: Option<String>,
    pub token_index: usize,
    pub token: String,
    pub token_surface: String,
    pub token_char_start: usize,
    pub token_char_end: usize,
    pub ipa_text: String,
    pub ipa_tokens: Vec<String>,
    pub normalized_phones: Vec<String>,
    pub ownership_score: f32,
}

#[derive(Debug, Clone, Facet)]
pub struct TokenPieceComparisonToken {
    pub word_index: Option<usize>,
    pub word_surface: Option<String>,
    pub token_index: usize,
    pub token: String,
    pub token_surface: String,
    pub token_char_start: usize,
    pub token_char_end: usize,
    pub ipa_text: String,
    pub comparison_token: String,
    pub ipa_source_start: usize,
    pub ipa_source_end: usize,
    pub ownership_score: f32,
}

#[derive(Debug, Clone, Facet)]
pub struct TranscriptComparisonToken {
    pub comparison_index: usize,
    pub comparison_token: String,
    pub comparison_source_start: usize,
    pub comparison_source_end: usize,
    pub word_index: Option<usize>,
    pub word_surface: Option<String>,
    pub token_index: usize,
    pub token: String,
    pub token_surface: String,
    pub token_char_start: usize,
    pub token_char_end: usize,
    pub ipa_text: String,
    pub ipa_source_start: usize,
    pub ipa_source_end: usize,
    pub ownership_score: f32,
}

#[derive(Debug, Clone, Facet)]
pub struct TranscriptComparisonProvenance {
    pub comparison_index: usize,
    pub word_index: Option<usize>,
    pub word_surface: Option<String>,
    pub token_index: usize,
    pub token: String,
    pub token_surface: String,
    pub token_char_start: usize,
    pub token_char_end: usize,
    pub ipa_text: String,
    pub ipa_source_start: usize,
    pub ipa_source_end: usize,
    pub ownership_score: f32,
}

#[derive(Debug, Clone, Facet)]
pub struct TranscriptComparisonSequence {
    pub tokens: Vec<ComparisonToken>,
    pub provenance: Vec<TranscriptComparisonProvenance>,
}

#[derive(Debug, Clone, Facet)]
pub struct TranscriptWordComparisonRange {
    pub word_index: usize,
    pub word_surface: String,
    pub char_start: usize,
    pub char_end: usize,
    pub comparison_start: usize,
    pub comparison_end: usize,
}

#[derive(Debug, Clone, Facet)]
pub struct TranscriptTokenPieceComparisonRange {
    pub token_index: usize,
    pub token: String,
    pub token_surface: String,
    pub token_char_start: usize,
    pub token_char_end: usize,
    pub word_index: Option<usize>,
    pub word_surface: Option<String>,
    pub comparison_start: usize,
    pub comparison_end: usize,
}

#[derive(Debug, Clone, Facet)]
pub struct TranscriptAlignmentInput {
    pub normalized: Vec<String>,
    pub sequence: TranscriptComparisonSequence,
    pub words: Vec<TranscriptWordComparisonRange>,
    pub token_pieces: Vec<TranscriptTokenPieceComparisonRange>,
}

#[derive(Debug, Clone, Facet)]
pub struct TextAnalysis {
    pub text: String,
    pub word_ipas: Vec<TextWordIpa>,
    pub token_piece_spans: Vec<TokenPieceIpaSpan>,
}

#[derive(Debug, Clone)]
struct SentenceTokenPiece {
    index: usize,
    token: String,
    char_start: usize,
    char_end: usize,
    surface: String,
}

pub struct BeeG2p {
    engine: G2pEngine,
    tokenizer: Tokenizer,
}

impl BeeG2p {
    pub fn load(model_dir: &Path, tokenizer_path: &Path) -> Result<Self> {
        let engine = G2pEngine::load(model_dir)
            .with_context(|| format!("loading Charsiu G2P model from {}", model_dir.display()))?;
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow!("loading tokenizer {}: {e}", tokenizer_path.display()))?;
        Ok(Self { engine, tokenizer })
    }

    pub fn load_default() -> Result<Self> {
        Self::load(&default_model_dir()?, &default_tokenizer_path()?)
    }

    pub fn g2p_words(&mut self, words: &[&str], lang_code: &str) -> Result<Vec<String>> {
        self.engine.g2p_batch(words, lang_code)
    }

    pub fn analyze_text(&mut self, text: &str, lang_code: &str) -> Result<TextAnalysis> {
        let words = transcript_words(text);
        let word_refs = words
            .iter()
            .map(|word| word.word.as_str())
            .collect::<Vec<_>>();
        let ipas = self.g2p_words(&word_refs, lang_code)?;
        let word_ipas = words
            .iter()
            .zip(ipas.iter())
            .map(|(word, ipa)| TextWordIpa {
                word: word.word.clone(),
                ipa: ipa.clone(),
                char_start: word.char_start,
                char_end: word.char_end,
            })
            .collect::<Vec<_>>();

        let sentence_tokens = sentence_token_pieces(&self.tokenizer, text)?;
        let mut token_piece_spans = Vec::new();

        for (word_index, (word, ipa)) in words.iter().zip(ipas.iter()).enumerate() {
            let (token_start, token_end) =
                token_span_for_char_range(word.char_start..word.char_end, &sentence_tokens);
            if token_start >= token_end {
                continue;
            }
            if token_end - token_start == 1 {
                let token = sentence_tokens
                    .get(token_start)
                    .ok_or_else(|| anyhow!("missing token piece {}", token_start))?;
                token_piece_spans.push(TokenPieceIpaSpan {
                    word_index: Some(word_index),
                    word_surface: Some(word.word.clone()),
                    token_index: token.index,
                    token: token.token.clone(),
                    token_surface: token.surface.clone(),
                    token_char_start: token.char_start,
                    token_char_end: token.char_end,
                    ipa_byte_start: 0,
                    ipa_byte_end: ipa.len(),
                    ipa_text: ipa.clone(),
                    ownership_score: 1.0,
                });
                continue;
            }

            let local_spans = sentence_tokens[token_start..token_end]
                .iter()
                .map(|token| ByteSpan {
                    label: token.token.clone(),
                    byte_start: token.char_start.max(word.char_start) - word.char_start,
                    byte_end: token.char_end.min(word.char_end) - word.char_start,
                })
                .collect::<Vec<_>>();
            let probe = self.engine.probe(&word.word, lang_code, &local_spans)?;
            token_piece_spans.extend(ownership_spans_to_token_piece_spans(
                &probe.ownership,
                &sentence_tokens[token_start..token_end],
                word_index,
                &word.word,
            ));
        }

        retune_token_piece_boundaries(&mut token_piece_spans);

        Ok(TextAnalysis {
            text: text.to_owned(),
            word_ipas,
            token_piece_spans,
        })
    }
}

pub fn token_piece_phones(analysis: &TextAnalysis) -> Vec<TokenPiecePhones> {
    analysis
        .token_piece_spans
        .iter()
        .cloned()
        .map(|span| {
            let ipa_tokens = parse_reviewed_ipa(&span.ipa_text);
            let normalized_phones = normalize_ipa_for_comparison(&ipa_tokens);
            TokenPiecePhones {
                word_index: span.word_index,
                word_surface: span.word_surface,
                token_index: span.token_index,
                token: span.token,
                token_surface: span.token_surface,
                token_char_start: span.token_char_start,
                token_char_end: span.token_char_end,
                ipa_text: span.ipa_text,
                ipa_tokens,
                normalized_phones,
                ownership_score: span.ownership_score,
            }
        })
        .collect()
}

pub fn token_piece_comparison_tokens(analysis: &TextAnalysis) -> Vec<TokenPieceComparisonToken> {
    token_piece_phones(analysis)
        .into_iter()
        .flat_map(|span| {
            normalize_ipa_for_comparison_with_spans(&span.ipa_tokens)
                .into_iter()
                .map(move |token| TokenPieceComparisonToken {
                    word_index: span.word_index,
                    word_surface: span.word_surface.clone(),
                    token_index: span.token_index,
                    token: span.token.clone(),
                    token_surface: span.token_surface.clone(),
                    token_char_start: span.token_char_start,
                    token_char_end: span.token_char_end,
                    ipa_text: span.ipa_text.clone(),
                    comparison_token: token.token,
                    ipa_source_start: token.source_start,
                    ipa_source_end: token.source_end,
                    ownership_score: span.ownership_score,
                })
        })
        .collect()
}

pub fn transcript_comparison_tokens(analysis: &TextAnalysis) -> Vec<TranscriptComparisonToken> {
    token_piece_comparison_tokens(analysis)
        .into_iter()
        .enumerate()
        .map(|(comparison_index, token)| TranscriptComparisonToken {
            comparison_index,
            comparison_token: token.comparison_token,
            comparison_source_start: comparison_index,
            comparison_source_end: comparison_index + 1,
            word_index: token.word_index,
            word_surface: token.word_surface,
            token_index: token.token_index,
            token: token.token,
            token_surface: token.token_surface,
            token_char_start: token.token_char_start,
            token_char_end: token.token_char_end,
            ipa_text: token.ipa_text,
            ipa_source_start: token.ipa_source_start,
            ipa_source_end: token.ipa_source_end,
            ownership_score: token.ownership_score,
        })
        .collect()
}

pub fn transcript_comparison_sequence(analysis: &TextAnalysis) -> TranscriptComparisonSequence {
    let rows = transcript_comparison_tokens(analysis);
    let tokens = rows
        .iter()
        .map(|token| ComparisonToken {
            token: token.comparison_token.clone(),
            source_start: token.comparison_source_start,
            source_end: token.comparison_source_end,
        })
        .collect();
    let provenance = rows
        .into_iter()
        .map(|token| TranscriptComparisonProvenance {
            comparison_index: token.comparison_index,
            word_index: token.word_index,
            word_surface: token.word_surface,
            token_index: token.token_index,
            token: token.token,
            token_surface: token.token_surface,
            token_char_start: token.token_char_start,
            token_char_end: token.token_char_end,
            ipa_text: token.ipa_text,
            ipa_source_start: token.ipa_source_start,
            ipa_source_end: token.ipa_source_end,
            ownership_score: token.ownership_score,
        })
        .collect();
    TranscriptComparisonSequence { tokens, provenance }
}

pub fn transcript_alignment_input(analysis: &TextAnalysis) -> TranscriptAlignmentInput {
    let sequence = transcript_comparison_sequence(analysis);
    let normalized = sequence
        .tokens
        .iter()
        .map(|token| token.token.clone())
        .collect::<Vec<_>>();

    let mut token_pieces = Vec::new();
    let mut comparison_index = 0usize;
    while comparison_index < sequence.provenance.len() {
        let current = &sequence.provenance[comparison_index];
        let mut next = comparison_index + 1;
        while next < sequence.provenance.len() {
            let candidate = &sequence.provenance[next];
            if candidate.token_index != current.token_index {
                break;
            }
            next += 1;
        }
        token_pieces.push(TranscriptTokenPieceComparisonRange {
            token_index: current.token_index,
            token: current.token.clone(),
            token_surface: current.token_surface.clone(),
            token_char_start: current.token_char_start,
            token_char_end: current.token_char_end,
            word_index: current.word_index,
            word_surface: current.word_surface.clone(),
            comparison_start: comparison_index,
            comparison_end: next,
        });
        comparison_index = next;
    }

    let mut words = Vec::new();
    let mut current_word_start = 0usize;
    for (word_index, word) in analysis.word_ipas.iter().enumerate() {
        let word_end = sequence
            .provenance
            .iter()
            .rposition(|row| row.word_index == Some(word_index))
            .map(|index| index + 1)
            .unwrap_or(current_word_start);
        words.push(TranscriptWordComparisonRange {
            word_index,
            word_surface: word.word.clone(),
            char_start: word.char_start,
            char_end: word.char_end,
            comparison_start: current_word_start,
            comparison_end: word_end,
        });
        current_word_start = word_end;
    }

    TranscriptAlignmentInput {
        normalized,
        sequence,
        words,
        token_pieces,
    }
}

pub fn transcript_words(text: &str) -> Vec<TranscriptWord> {
    word_re()
        .find_iter(text)
        .map(|m| TranscriptWord {
            word: m.as_str().to_owned(),
            char_start: m.start(),
            char_end: m.end(),
        })
        .collect()
}

fn ownership_spans_to_token_piece_spans(
    ownership: &[OwnershipSpan],
    local_tokens: &[SentenceTokenPiece],
    word_index: usize,
    word: &str,
) -> Vec<TokenPieceIpaSpan> {
    ownership
        .iter()
        .filter_map(|span| {
            let token = local_tokens.get(span.span_index)?;
            Some(TokenPieceIpaSpan {
                word_index: Some(word_index),
                word_surface: Some(word.to_owned()),
                token_index: token.index,
                token: token.token.clone(),
                token_surface: token.surface.clone(),
                token_char_start: token.char_start,
                token_char_end: token.char_end,
                ipa_byte_start: span.ipa_byte_start,
                ipa_byte_end: span.ipa_byte_end,
                ipa_text: span.ipa_text.clone(),
                ownership_score: span.avg_score,
            })
        })
        .collect()
}

fn retune_token_piece_boundaries(spans: &mut [TokenPieceIpaSpan]) {
    for i in 1..spans.len() {
        let (left, right) = spans.split_at_mut(i);
        let prev = &mut left[i - 1];
        let next = &mut right[0];
        if prev.word_index != next.word_index {
            continue;
        }
        if !starts_with_vowel_grapheme(&next.token_surface) {
            continue;
        }

        let mut prev_tokens = parse_reviewed_ipa(&prev.ipa_text);
        let mut next_tokens = parse_reviewed_ipa(&next.ipa_text);
        if next_tokens.len() < 2 {
            continue;
        }

        let move_count = leading_consonant_count_before_first_vowel(&next_tokens);
        if move_count == 0 || move_count >= next_tokens.len() {
            continue;
        }

        prev_tokens.extend(next_tokens.drain(..move_count));
        prev.ipa_text = prev_tokens.join("");
        next.ipa_text = next_tokens.join("");
    }
}

fn starts_with_vowel_grapheme(surface: &str) -> bool {
    surface
        .trim_start()
        .chars()
        .next()
        .is_some_and(|ch| matches!(ch.to_ascii_lowercase(), 'a' | 'e' | 'i' | 'o' | 'u' | 'y'))
}

fn leading_consonant_count_before_first_vowel(tokens: &[String]) -> usize {
    let normalized = normalize_ipa_for_comparison(tokens);
    let mut count = 0usize;
    for token in normalized {
        if is_vowel_phone(&token) {
            break;
        }
        count += 1;
    }
    count
}

fn is_vowel_phone(token: &str) -> bool {
    matches!(
        token,
        "a" | "e"
            | "i"
            | "o"
            | "u"
            | "y"
            | "ə"
            | "ɛ"
            | "ɪ"
            | "ʊ"
            | "ɔ"
            | "æ"
            | "ɑ"
            | "ɒ"
            | "ʌ"
            | "ɚ"
            | "ɝ"
            | "œ"
            | "ø"
            | "ɨ"
            | "ʉ"
            | "ɯ"
            | "ɜ"
            | "ɞ"
            | "ɐ"
    )
}

fn sentence_token_pieces(tokenizer: &Tokenizer, text: &str) -> Result<Vec<SentenceTokenPiece>> {
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|e| anyhow!("encoding text: {e}"))?;
    Ok(encoding
        .get_tokens()
        .iter()
        .zip(encoding.get_offsets())
        .enumerate()
        .map(
            |(index, (token, (char_start, char_end)))| SentenceTokenPiece {
                index,
                token: token.clone(),
                char_start: *char_start,
                char_end: *char_end,
                surface: text
                    .get(*char_start..*char_end)
                    .unwrap_or_default()
                    .to_owned(),
            },
        )
        .collect())
}

fn token_span_for_char_range(
    char_range: Range<usize>,
    tokens: &[SentenceTokenPiece],
) -> (usize, usize) {
    let indices = tokens
        .iter()
        .enumerate()
        .filter_map(|(index, token)| {
            (token.char_start < char_range.end && token.char_end > char_range.start)
                .then_some(index)
        })
        .collect::<Vec<_>>();
    match (indices.first(), indices.last()) {
        (Some(start), Some(end)) => (*start, end + 1),
        _ => (0, 0),
    }
}

fn default_model_dir() -> Result<PathBuf> {
    if let Ok(path) = env::var("BEE_G2P_CHARSIU_MODEL_DIR") {
        return Ok(PathBuf::from(path));
    }
    let fallback = PathBuf::from("/tmp/charsiu-g2p");
    if fallback.join("model.safetensors").exists() {
        return Ok(fallback);
    }
    Err(anyhow!(
        "missing Charsiu G2P model dir; set BEE_G2P_CHARSIU_MODEL_DIR or install model at /tmp/charsiu-g2p"
    ))
}

fn default_tokenizer_path() -> Result<PathBuf> {
    env::var("BEE_TOKENIZER_PATH")
        .map(PathBuf::from)
        .context("missing BEE_TOKENIZER_PATH; run with direnv")
}

fn word_re() -> &'static Regex {
    static WORD_RE: OnceLock<Regex> = OnceLock::new();
    WORD_RE.get_or_init(|| Regex::new(r"[^\W_]+(?:['’-][^\W_]+)*").expect("valid word regex"))
}

#[cfg(test)]
mod tests {
    use super::{
        TextAnalysis, TextWordIpa, TokenPieceIpaSpan, retune_token_piece_boundaries,
        transcript_alignment_input, transcript_words,
    };

    #[test]
    fn transcript_words_keeps_char_spans() {
        let words = transcript_words("For Jason, this Thursday, use Facet.");
        let rendered: Vec<_> = words
            .into_iter()
            .map(|word| (word.word, word.char_start, word.char_end))
            .collect();
        assert_eq!(
            rendered,
            vec![
                ("For".to_string(), 0, 3),
                ("Jason".to_string(), 4, 9),
                ("this".to_string(), 11, 15),
                ("Thursday".to_string(), 16, 24),
                ("use".to_string(), 26, 29),
                ("Facet".to_string(), 30, 35),
            ]
        );
    }

    #[test]
    fn transcript_alignment_input_groups_piece_ranges() {
        let analysis = TextAnalysis {
            text: "use Facet".to_owned(),
            word_ipas: vec![
                TextWordIpa {
                    word: "use".to_owned(),
                    ipa: "juz".to_owned(),
                    char_start: 0,
                    char_end: 3,
                },
                TextWordIpa {
                    word: "Facet".to_owned(),
                    ipa: "feɪsət".to_owned(),
                    char_start: 4,
                    char_end: 9,
                },
            ],
            token_piece_spans: vec![
                TokenPieceIpaSpan {
                    word_index: Some(0),
                    word_surface: Some("use".to_owned()),
                    token_index: 0,
                    token: "use".to_owned(),
                    token_surface: "use".to_owned(),
                    token_char_start: 0,
                    token_char_end: 3,
                    ipa_byte_start: 0,
                    ipa_byte_end: 3,
                    ipa_text: "juz".to_owned(),
                    ownership_score: 1.0,
                },
                TokenPieceIpaSpan {
                    word_index: Some(1),
                    word_surface: Some("Facet".to_owned()),
                    token_index: 1,
                    token: "ĠFac".to_owned(),
                    token_surface: " Fac".to_owned(),
                    token_char_start: 3,
                    token_char_end: 7,
                    ipa_byte_start: 0,
                    ipa_byte_end: 5,
                    ipa_text: "feɪs".to_owned(),
                    ownership_score: 0.9,
                },
                TokenPieceIpaSpan {
                    word_index: Some(1),
                    word_surface: Some("Facet".to_owned()),
                    token_index: 2,
                    token: "et".to_owned(),
                    token_surface: "et".to_owned(),
                    token_char_start: 7,
                    token_char_end: 9,
                    ipa_byte_start: 5,
                    ipa_byte_end: 7,
                    ipa_text: "ət".to_owned(),
                    ownership_score: 0.8,
                },
            ],
        };
        let input = transcript_alignment_input(&analysis);
        assert_eq!(input.words.len(), 2);
        assert_eq!(input.token_pieces.len(), 3);
        assert_eq!(input.words[0].comparison_start, 0);
        assert_eq!(input.words[0].comparison_end, 3);
        assert_eq!(input.words[1].comparison_start, 3);
        assert_eq!(input.words[1].comparison_end, 9);
        assert_eq!(input.token_pieces[1].comparison_start, 3);
        assert_eq!(input.token_pieces[1].comparison_end, 7);
        assert_eq!(input.token_pieces[2].comparison_start, 7);
        assert_eq!(input.token_pieces[2].comparison_end, 9);
    }

    #[test]
    fn retune_token_piece_boundaries_pulls_onset_back_to_vowel_piece() {
        let mut spans = vec![
            TokenPieceIpaSpan {
                word_index: Some(0),
                word_surface: Some("Facet".to_owned()),
                token_index: 0,
                token: "Fac".to_owned(),
                token_surface: "Fac".to_owned(),
                token_char_start: 0,
                token_char_end: 3,
                ipa_byte_start: 0,
                ipa_byte_end: 3,
                ipa_text: "feɪ".to_owned(),
                ownership_score: 0.9,
            },
            TokenPieceIpaSpan {
                word_index: Some(0),
                word_surface: Some("Facet".to_owned()),
                token_index: 1,
                token: "et".to_owned(),
                token_surface: "et".to_owned(),
                token_char_start: 3,
                token_char_end: 5,
                ipa_byte_start: 3,
                ipa_byte_end: 6,
                ipa_text: "sət".to_owned(),
                ownership_score: 0.8,
            },
        ];

        retune_token_piece_boundaries(&mut spans);

        assert_eq!(spans[0].ipa_text, "feɪs");
        assert_eq!(spans[1].ipa_text, "ət");
    }
}
