use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock};

use serde::Serialize;
use synth_corrupt::{
    cmudict::CmuDict,
    corrupt::phoneme_edit_distance,
    g2p::{ipa_to_arpabet, G2p},
};

use crate::db::VocabRow;

#[derive(Debug, Clone, Serialize)]
pub struct PrototypeCandidate {
    pub term: String,
    pub via: String,
    pub matched_form: String,
    pub matched_form_phonemes: Option<String>,
    pub term_preview: Option<String>,
    pub term_preview_phonemes: Option<String>,
    pub score: f32,
    pub lexical_score: Option<f32>,
    pub dice: Option<f32>,
    pub prefix_ratio: Option<f32>,
    pub length_ratio: Option<f32>,
    pub phonetic_score: Option<f32>,
    pub observed_acoustic_score: Option<f32>,
    pub acoustic_score: Option<f32>,
    pub acoustic_delta: Option<f32>,
    pub phonemes: Option<String>,
    pub exact_words: bool,
    pub exact_compact: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct PrototypeSpanProposal {
    pub token_start: usize,
    pub token_end: usize,
    pub char_start: usize,
    pub char_end: usize,
    pub raw_text: String,
    pub normalized: String,
    pub phonemes: Option<String>,
    pub acoustic_phonemes: Option<String>,
    pub observed_acoustic_score: Option<f32>,
    pub acoustic_trustworthy: bool,
    pub acoustic_window_start_sec: Option<f64>,
    pub acoustic_window_end_sec: Option<f64>,
    pub candidates: Vec<PrototypeCandidate>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AcceptedProposal {
    pub token_start: usize,
    pub token_end: usize,
    pub char_start: usize,
    pub char_end: usize,
    pub from: String,
    pub matched_form: String,
    pub from_phonemes: Option<String>,
    pub to: String,
    pub to_phonemes: Option<String>,
    pub via: String,
    pub score: f32,
    pub acoustic_score: Option<f32>,
    pub acoustic_delta: Option<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SentenceCandidate {
    pub label: String,
    pub text: String,
    pub edits: Vec<AcceptedProposal>,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct PrototypeCorrectionResult {
    pub original: String,
    pub corrected: String,
    pub accepted: Vec<AcceptedProposal>,
    pub proposals: Vec<PrototypeSpanProposal>,
    pub sentence_candidates: Vec<SentenceCandidate>,
}

#[derive(Debug, Clone, Copy)]
pub struct PrototypeConfig {
    pub max_span_tokens: usize,
    pub max_span_proposals: usize,
    pub max_candidates_per_span: usize,
}

impl Default for PrototypeConfig {
    fn default() -> Self {
        Self {
            max_span_tokens: 4,
            max_span_proposals: 24,
            max_candidates_per_span: 3,
        }
    }
}

#[derive(Debug, Clone)]
struct SpanToken {
    raw: String,
    normalized: String,
    start: usize,
    end: usize,
}

#[derive(Debug, Clone)]
struct LexiconTerm {
    term: String,
    term_compact: String,
    term_preview: String,
    term_preview_phonemes: Vec<String>,
    forms: Vec<FormEntry>,
}

#[derive(Debug, Clone)]
struct FormEntry {
    raw: String,
    words: String,
    word_count: usize,
    compact: String,
    via: &'static str,
    phonemes: Vec<String>,
}

#[derive(Debug, Clone)]
struct ScoredCandidate {
    term: String,
    via: &'static str,
    matched_form: String,
    matched_form_phonemes: Vec<String>,
    term_preview: String,
    term_preview_phonemes: Vec<String>,
    score: f32,
    lexical_score: Option<f32>,
    dice: Option<f32>,
    prefix_ratio: Option<f32>,
    length_ratio: Option<f32>,
    phonetic_score: Option<f32>,
    observed_acoustic_score: Option<f32>,
    acoustic_score: Option<f32>,
    acoustic_delta: Option<f32>,
    phonemes: Vec<String>,
    exact_words: bool,
    exact_compact: bool,
}

#[derive(Debug, Clone)]
pub struct AcousticSegment {
    pub phone: String,
    pub start_sec: f64,
    pub end_sec: f64,
}

pub struct AcousticContext<'a> {
    pub qwen_alignment: &'a [qwen3_asr::ForcedAlignItem],
    pub zipa_segments: &'a [AcousticSegment],
    pub zipa_by_qwen: &'a [Vec<String>],
}

struct PhoneticResources {
    g2p: G2p,
    dict: CmuDict,
}

static PHONETIC_RESOURCES: OnceLock<Option<Mutex<PhoneticResources>>> = OnceLock::new();

pub fn phonetic_preview(text: &str) -> Option<String> {
    phoneme_string(&phonemize_phrase(text))
}

pub fn prototype_correct(
    input: &str,
    vocab: &[VocabRow],
    alt_spellings: &HashMap<String, Vec<String>>,
    confusion_forms: &HashMap<String, Vec<String>>,
    config: PrototypeConfig,
) -> PrototypeCorrectionResult {
    prototype_correct_with_acoustics(input, vocab, alt_spellings, confusion_forms, config, None)
}

pub fn prototype_correct_with_acoustics(
    input: &str,
    vocab: &[VocabRow],
    alt_spellings: &HashMap<String, Vec<String>>,
    confusion_forms: &HashMap<String, Vec<String>>,
    config: PrototypeConfig,
    acoustic: Option<&AcousticContext<'_>>,
) -> PrototypeCorrectionResult {
    let lexicon = build_lexicon(vocab, alt_spellings, confusion_forms);
    let tokens = tokenize_with_offsets(input);
    let mut span_proposals = enumerate_span_proposals(input, &tokens, &lexicon, config, acoustic);
    let edit_pool = build_edit_pool(&span_proposals);
    let sentence_candidates = build_sentence_candidates(input, &edit_pool);
    let best = sentence_candidates
        .iter()
        .max_by(|a, b| a.score.total_cmp(&b.score))
        .cloned()
        .unwrap_or(SentenceCandidate {
            label: "original".to_string(),
            text: input.to_string(),
            edits: Vec::new(),
            score: 0.0,
        });
    span_proposals.truncate(config.max_span_proposals);
    PrototypeCorrectionResult {
        original: input.to_string(),
        corrected: best.text.clone(),
        accepted: best.edits.clone(),
        proposals: span_proposals,
        sentence_candidates,
    }
}

fn build_lexicon(
    vocab: &[VocabRow],
    alt_spellings: &HashMap<String, Vec<String>>,
    confusion_forms: &HashMap<String, Vec<String>>,
) -> Vec<LexiconTerm> {
    let mut out = Vec::new();
    for row in vocab {
        let term_compact = compact_key(&normalize_words(&row.term));
        let mut seen = HashSet::new();
        let mut forms = Vec::new();
        let mut add_form = |raw: &str, via: &'static str| {
            let words = normalize_words(raw);
            if words.is_empty() {
                return;
            }
            let compact = compact_key(&words);
            let word_count = raw.split_whitespace().count();
            if via == "confusion"
                && is_low_signal_confusion_alias(&term_compact, &words, word_count, &compact)
            {
                return;
            }
            if compact.is_empty() || !seen.insert((compact.clone(), via)) {
                return;
            }
            forms.push(FormEntry {
                raw: raw.trim().to_string(),
                words,
                word_count,
                compact,
                via,
                phonemes: phonemize_phrase(raw),
            });
        };

        add_form(&row.term, "canonical");
        add_form(row.spoken(), "spoken");
        if let Some(alts) = alt_spellings.get(&row.term.to_ascii_lowercase()) {
            for alt in alts {
                add_form(alt, "alt");
            }
        }
        if let Some(confusions) = confusion_forms.get(&row.term.to_ascii_lowercase()) {
            for heard in confusions {
                add_form(heard, "confusion");
            }
        }
        if forms.is_empty() {
            continue;
        }
        out.push(LexiconTerm {
            term: row.term.clone(),
            term_compact,
            term_preview: row.spoken().trim().to_string(),
            term_preview_phonemes: phonemize_phrase(row.spoken()),
            forms,
        });
    }
    out
}

fn tokenize_with_offsets(text: &str) -> Vec<SpanToken> {
    let mut tokens = Vec::new();
    let mut start = None;
    for (idx, ch) in text.char_indices() {
        if ch.is_whitespace() {
            if let Some(token_start) = start.take() {
                let raw = &text[token_start..idx];
                let normalized = normalize_token(raw);
                if !normalized.is_empty() {
                    let (start, end) = trim_token_bounds(raw, token_start);
                    tokens.push(SpanToken {
                        raw: raw.to_string(),
                        normalized,
                        start,
                        end,
                    });
                }
            }
        } else if start.is_none() {
            start = Some(idx);
        }
    }
    if let Some(token_start) = start {
        let raw = &text[token_start..];
        let normalized = normalize_token(raw);
        if !normalized.is_empty() {
            let (start, end) = trim_token_bounds(raw, token_start);
            tokens.push(SpanToken {
                raw: raw.to_string(),
                normalized,
                start,
                end,
            });
        }
    }
    tokens
}

fn enumerate_span_proposals(
    input: &str,
    tokens: &[SpanToken],
    lexicon: &[LexiconTerm],
    config: PrototypeConfig,
    acoustic: Option<&AcousticContext<'_>>,
) -> Vec<PrototypeSpanProposal> {
    let max_span = config.max_span_tokens.max(1);
    let mut proposals = Vec::new();
    for start in 0..tokens.len() {
        let end_limit = (start + max_span).min(tokens.len());
        for end in start + 1..=end_limit {
            let char_start = tokens[start].start;
            let char_end = tokens[end - 1].end;
            let raw_text = input[char_start..char_end].to_string();
            let normalized = tokens[start..end]
                .iter()
                .map(|t| t.normalized.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            let span_compact = compact_key(&normalized);
            let span_phonemes = phonemize_phrase(&raw_text);
            if span_compact.len() < 2 {
                continue;
            }
            let acoustic_window = acoustic.and_then(|ctx| acoustic_window_for_span(ctx, start, end));
            let acoustic_phones = acoustic.and_then(|ctx| acoustic_phones_for_span(ctx, start, end));
            let observed_acoustic_score =
                acoustic_phones.as_deref().and_then(|phones| phoneme_similarity(phones, &span_phonemes));
            let acoustic_trustworthy = observed_acoustic_score
                .map(|score| score >= 0.45)
                .unwrap_or(false);
            let mut candidates = lexicon
                .iter()
                .filter_map(|term| {
                    score_term(
                        &normalized,
                        &span_compact,
                        &span_phonemes,
                        acoustic_phones.as_deref(),
                        term,
                    )
                })
                .collect::<Vec<_>>();
            if candidates.is_empty() {
                continue;
            }
            candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
            candidates.truncate(config.max_candidates_per_span.max(1));
            proposals.push(PrototypeSpanProposal {
                token_start: start,
                token_end: end,
                char_start,
                char_end,
                raw_text,
                normalized,
                phonemes: phoneme_string(&span_phonemes),
                acoustic_phonemes: acoustic_phones.as_ref().and_then(|phones| phoneme_string(phones)),
                observed_acoustic_score,
                acoustic_trustworthy,
                acoustic_window_start_sec: acoustic_window.map(|(start, _)| start),
                acoustic_window_end_sec: acoustic_window.map(|(_, end)| end),
                candidates: candidates
                    .into_iter()
                    .map(|candidate| PrototypeCandidate {
                        term: candidate.term,
                        via: candidate.via.to_string(),
                        matched_form: candidate.matched_form,
                        matched_form_phonemes: phoneme_string(&candidate.matched_form_phonemes),
                        term_preview: (!candidate.term_preview.trim().is_empty())
                            .then_some(candidate.term_preview),
                        term_preview_phonemes: phoneme_string(&candidate.term_preview_phonemes),
                        score: candidate.score,
                        lexical_score: candidate.lexical_score,
                        dice: candidate.dice,
                        prefix_ratio: candidate.prefix_ratio,
                        length_ratio: candidate.length_ratio,
                        phonetic_score: candidate.phonetic_score,
                        observed_acoustic_score: candidate.observed_acoustic_score,
                        acoustic_score: candidate.acoustic_score,
                        acoustic_delta: candidate.acoustic_delta,
                        phonemes: phoneme_string(&candidate.phonemes),
                        exact_words: candidate.exact_words,
                        exact_compact: candidate.exact_compact,
                    })
                    .collect(),
            });
        }
    }
    proposals.sort_by(|a, b| {
        let a_score = a.candidates.first().map(|c| c.score).unwrap_or_default();
        let b_score = b.candidates.first().map(|c| c.score).unwrap_or_default();
        b_score
            .total_cmp(&a_score)
            .then_with(|| (b.token_end - b.token_start).cmp(&(a.token_end - a.token_start)))
    });
    proposals
}

fn score_term(
    span_words: &str,
    span_compact: &str,
    span_phonemes: &[String],
    acoustic_phones: Option<&[String]>,
    term: &LexiconTerm,
) -> Option<ScoredCandidate> {
    let mut best: Option<ScoredCandidate> = None;
    let span_word_count = span_words.split_whitespace().count();
    let has_noncanonical_forms = term.forms.iter().any(|form| form.via != "canonical");
    for form in &term.forms {
        let exact_words = span_words == form.words;
        let exact_compact = span_compact == form.compact;
        let phonetic_score = phoneme_similarity(span_phonemes, &form.phonemes);
        let observed_acoustic_score =
            acoustic_phones.and_then(|phones| phoneme_similarity(phones, span_phonemes));
        let acoustic_is_trustworthy = observed_acoustic_score
            .map(|score| score >= 0.45)
            .unwrap_or(false);
        let raw_acoustic_score =
            acoustic_phones.and_then(|phones| phoneme_similarity(phones, &form.phonemes));
        let acoustic_score = acoustic_is_trustworthy.then_some(raw_acoustic_score).flatten();
        let acoustic_delta = if acoustic_is_trustworthy {
            acoustic_delta(raw_acoustic_score, observed_acoustic_score)
        } else {
            None
        };
        let (score, lexical_score, dice, prefix_ratio, length_ratio) = if exact_words {
            (1.30 + via_bonus(form.via), Some(1.0), Some(1.0), Some(1.0), Some(1.0))
        } else if exact_compact {
            (1.22 + via_bonus(form.via), Some(1.0), Some(1.0), Some(1.0), Some(1.0))
        } else {
            if span_compact.len() <= 3 || form.compact.len() <= 3 {
                continue;
            }
            if form.via == "canonical" && has_noncanonical_forms {
                continue;
            }
            let dice = bigram_dice(span_compact, &form.compact);
            let prefix = prefix_ratio(span_compact, &form.compact);
            let len_ratio = length_ratio(span_compact, &form.compact);
            let phon = phonetic_score.unwrap_or(0.0);
            let acoustic = acoustic_score.unwrap_or(0.0);
            if form.via == "confusion" && form.word_count != span_word_count {
                continue;
            }
            if form.via == "confusion" && phon < 0.78 {
                continue;
            }
            if dice < 0.45 && prefix < 0.50 && phon < 0.62 {
                continue;
            }

            let lexical =
                0.40 * dice + 0.16 * prefix + 0.08 * len_ratio + 0.24 * phon + via_bonus(form.via);
            if lexical < 0.44 && phon < 0.72 {
                continue;
            }
            if let Some(delta) = acoustic_delta {
                if delta <= -0.12 && lexical < 0.88 {
                    continue;
                }
            }

            let mut score = lexical;
            if phon >= 0.82 {
                score += 0.10;
            }
            if lexical >= 0.56 {
                if let Some(delta) = acoustic_delta {
                    score += 0.22 * delta;
                    if delta >= 0.10 {
                        score += 0.05;
                    } else if delta <= -0.08 {
                        score -= 0.18;
                    }
                } else if acoustic_score.is_some() {
                    score += 0.03 * (acoustic - 0.5);
                }
                if acoustic >= 0.80 {
                    score += 0.04;
                } else if acoustic <= 0.36 {
                    score -= 0.08;
                }
            }
            (score, Some(lexical), Some(dice), Some(prefix), Some(len_ratio))
        };
        if best.as_ref().map(|b| score > b.score).unwrap_or(true) {
            best = Some(ScoredCandidate {
                term: term.term.clone(),
                via: form.via,
                matched_form: form.raw.clone(),
                matched_form_phonemes: form.phonemes.clone(),
                term_preview: term.term_preview.clone(),
                term_preview_phonemes: term.term_preview_phonemes.clone(),
                score,
                lexical_score,
                dice,
                prefix_ratio,
                length_ratio,
                phonetic_score,
                observed_acoustic_score,
                acoustic_score,
                acoustic_delta,
                phonemes: form.phonemes.clone(),
                exact_words,
                exact_compact,
            });
        }
    }
    best
}

fn build_edit_pool(spans: &[PrototypeSpanProposal]) -> Vec<AcceptedProposal> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for span in spans {
        let Some(top_score) = span.candidates.first().map(|c| c.score) else {
            continue;
        };
        let mut kept_for_span = 0usize;
        for candidate in &span.candidates {
            if kept_for_span >= 2 {
                break;
            }
            if candidate.score + 0.08 < top_score {
                break;
            }
            if is_noop_surface_rewrite(&span.raw_text, &candidate.term) {
                continue;
            }
            let acoustically_supported = candidate
                .acoustic_delta
                .map(|delta| delta >= 0.03)
                .unwrap_or(false);
            let phonetic_support = candidate.phonetic_score.unwrap_or(0.0);
            let via_threshold = match candidate.via.as_str() {
                "spoken" | "confusion" => 0.58,
                "alt" => 0.60,
                "canonical" => 0.64,
                _ => 0.68,
            };
            let phonetic_backed = matches!(candidate.via.as_str(), "spoken" | "confusion" | "alt")
                && phonetic_support >= 0.68;
            let strong_prefix_match = matches!(candidate.via.as_str(), "spoken" | "confusion" | "alt")
                && candidate.score >= 0.62
                && candidate.prefix_ratio.unwrap_or(0.0) >= 0.95
                && candidate.dice.unwrap_or(0.0) >= 0.72;
            let via_supported =
                candidate.score >= via_threshold && (phonetic_backed || acoustically_supported);
            if !(candidate.exact_words
                || candidate.exact_compact
                || candidate.score >= 0.68
                || via_supported
                || strong_prefix_match
                || (candidate.score >= 0.64 && acoustically_supported))
            {
                continue;
            }
            let key = (span.token_start, span.token_end, candidate.term.clone());
            if !seen.insert(key) {
                continue;
            }
            out.push(AcceptedProposal {
                token_start: span.token_start,
                token_end: span.token_end,
                char_start: span.char_start,
                char_end: span.char_end,
                from: span.raw_text.clone(),
                matched_form: candidate.matched_form.clone(),
                from_phonemes: span.phonemes.clone(),
                to: candidate.term.clone(),
                to_phonemes: candidate.term_preview_phonemes.clone(),
                via: candidate.via.clone(),
                score: candidate.score,
                acoustic_score: candidate.acoustic_score,
                acoustic_delta: candidate.acoustic_delta,
            });
            kept_for_span += 1;
        }
    }
    out = prune_redundant_edits(out);
    out.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| (b.token_end - b.token_start).cmp(&(a.token_end - a.token_start)))
    });
    out.truncate(10);
    out
}

fn build_sentence_candidates(
    input: &str,
    edits: &[AcceptedProposal],
) -> Vec<SentenceCandidate> {
    let mut candidates = Vec::new();
    candidates.push(SentenceCandidate {
        label: "original".to_string(),
        text: input.to_string(),
        edits: Vec::new(),
        score: 0.0,
    });
    let mut seen = HashSet::from([input.to_string()]);

    for edit in edits.iter().take(8).cloned() {
        let text = apply_edits(input, std::slice::from_ref(&edit));
        if seen.insert(text.clone()) {
            let score = score_sentence_candidate(std::slice::from_ref(&edit));
            candidates.push(SentenceCandidate {
                label: format!("single:{}->{}", edit.from, edit.to),
                text,
                edits: vec![edit],
                score,
            });
        }
    }

    for i in 0..edits.len().min(6) {
        for j in i + 1..edits.len().min(6) {
            let pair = [edits[i].clone(), edits[j].clone()];
            if overlaps(&pair[0], &pair[1]) {
                continue;
            }
            let mut ordered = pair.to_vec();
            ordered.sort_by_key(|edit| edit.char_start);
            let text = apply_edits(input, &ordered);
            if seen.insert(text.clone()) {
                let score = score_sentence_candidate(&ordered);
                candidates.push(SentenceCandidate {
                    label: format!("pair:{}+{}", ordered[0].to, ordered[1].to),
                    text,
                    edits: ordered,
                    score,
                });
            }
        }
    }

    let exactish = edits
        .iter()
        .filter(|edit| edit.score >= 1.20 || edit.via == "alt" || edit.via == "spoken")
        .cloned()
        .collect::<Vec<_>>();
    if !exactish.is_empty() {
        let ordered = select_non_overlapping_combo(&exactish, |_| true, 4);
        let text = apply_edits(input, &ordered);
        if seen.insert(text.clone()) {
            let score = score_sentence_candidate(&ordered);
            candidates.push(SentenceCandidate {
                label: "exactish".to_string(),
                text,
                edits: ordered,
                score,
            });
        }
    }

    let strong_combo = select_non_overlapping_combo(
        edits,
        |edit| edit.score >= 0.85 || edit.via == "alt" || edit.via == "spoken",
        4,
    );
    if !strong_combo.is_empty() {
        let text = apply_edits(input, &strong_combo);
        if seen.insert(text.clone()) {
            let score = score_sentence_candidate(&strong_combo) + 0.12 * strong_combo.len() as f32;
            candidates.push(SentenceCandidate {
                label: "strong-combo".to_string(),
                text,
                edits: strong_combo,
                score,
            });
        }
    }

    for combo in top_non_overlapping_combos(edits, 4, 12) {
        let text = apply_edits(input, &combo);
        if seen.insert(text.clone()) {
            let score = score_sentence_candidate(&combo) + combo_bonus(&combo);
            let label = format!(
                "combo:{}",
                combo
                    .iter()
                    .map(|edit| edit.to.as_str())
                    .collect::<Vec<_>>()
                    .join("+")
            );
            candidates.push(SentenceCandidate {
                label,
                text,
                edits: combo,
                score,
            });
        }
    }
    candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
    candidates
}

fn overlaps(a: &AcceptedProposal, b: &AcceptedProposal) -> bool {
    a.token_start < b.token_end && b.token_start < a.token_end
}

fn select_non_overlapping_combo(
    edits: &[AcceptedProposal],
    include: impl Fn(&AcceptedProposal) -> bool,
    max_edits: usize,
) -> Vec<AcceptedProposal> {
    let mut chosen = Vec::new();
    for edit in edits.iter().filter(|edit| include(edit)) {
        if chosen.iter().any(|existing| overlaps(existing, edit)) {
            continue;
        }
        chosen.push(edit.clone());
        if chosen.len() >= max_edits {
            break;
        }
    }
    chosen.sort_by_key(|edit| edit.char_start);
    chosen
}

fn score_sentence_candidate(edits: &[AcceptedProposal]) -> f32 {
    let mut score = 0.0;
    for edit in edits {
        let span_len = (edit.token_end - edit.token_start) as f32;
        let vocab_repair_prior = match edit.via.as_str() {
            "spoken" => 0.12,
            "confusion" => 0.11,
            "alt" => 0.10,
            "canonical" => 0.08,
            _ => 0.08,
        };
        let confidence_adjust = (edit.score - 0.68).max(-0.08);
        let span_penalty = 0.03 * (span_len - 1.0).max(0.0);
        score += vocab_repair_prior + confidence_adjust - span_penalty;
        if let Some(delta) = edit.acoustic_delta {
            score += delta * 0.30;
        } else if let Some(acoustic) = edit.acoustic_score {
            score += (acoustic - 0.55) * 0.08;
        }
    }
    score
}

fn acoustic_delta(candidate: Option<f32>, observed: Option<f32>) -> Option<f32> {
    Some(candidate? - observed?)
}

fn acoustic_phones_for_span(
    ctx: &AcousticContext<'_>,
    token_start: usize,
    token_end: usize,
) -> Option<Vec<String>> {
    if token_start < token_end && token_end <= ctx.zipa_by_qwen.len() {
        let grouped = ctx.zipa_by_qwen[token_start..token_end]
            .iter()
            .flatten()
            .cloned()
            .collect::<Vec<_>>();
        if !grouped.is_empty() {
            return Some(grouped);
        }
    }
    let (start, end) = acoustic_window_for_span(ctx, token_start, token_end)?;
    let raw_window = ctx
        .zipa_segments
        .iter()
        .filter(|seg| seg.end_sec > start && seg.start_sec < end)
        .flat_map(|seg| zipa_phone_to_arpabet(seg.phone.as_str()))
        .collect::<Vec<_>>();
    (!raw_window.is_empty()).then_some(raw_window)
}

fn acoustic_window_for_span(
    ctx: &AcousticContext<'_>,
    token_start: usize,
    token_end: usize,
) -> Option<(f64, f64)> {
    if token_end == 0
        || token_start >= ctx.qwen_alignment.len()
        || token_end > ctx.qwen_alignment.len()
        || token_start >= token_end
    {
        return None;
    }
    let start = (ctx.qwen_alignment[token_start].start_time - 0.05).max(0.0);
    let end = ctx.qwen_alignment[token_end - 1].end_time + 0.05;
    Some((start, end))
}

pub fn zipa_grouped_by_alignment_json(
    alignment: &[qwen3_asr::ForcedAlignItem],
    zipa_segments: &[AcousticSegment],
) -> serde_json::Value {
    serde_json::Value::Array(
        alignment
            .iter()
            .map(|item| {
                let phones = zipa_segments
                    .iter()
                    .filter(|seg| seg.end_sec > item.start_time && seg.start_sec < item.end_time)
                    .map(|seg| seg.phone.clone())
                    .filter(|phone| !phone.trim().is_empty() && phone != "▁")
                    .collect::<Vec<_>>();
                let grouped = if phones.is_empty() {
                    "∅".to_string()
                } else {
                    phones.join(" ")
                };
                serde_json::json!({
                    "w": grouped,
                    "s": item.start_time,
                    "e": item.end_time,
                    "t": format!(
                        "{}: {} ({:.3}s–{:.3}s)",
                        item.word,
                        if phones.is_empty() { "∅".to_string() } else { phones.join(" ") },
                        item.start_time,
                        item.end_time
                    ),
                })
            })
            .collect(),
    )
}

pub fn zipa_grouped_arpabet_by_alignment(
    alignment: &[qwen3_asr::ForcedAlignItem],
    zipa_segments: &[AcousticSegment],
) -> Vec<Vec<String>> {
    alignment
        .iter()
        .map(|item| {
            zipa_segments
                .iter()
                .filter(|seg| seg.end_sec > item.start_time && seg.start_sec < item.end_time)
                .flat_map(|seg| zipa_phone_to_arpabet(seg.phone.as_str()))
                .collect::<Vec<_>>()
        })
        .collect()
}

fn zipa_phone_to_arpabet(phone: &str) -> Vec<String> {
    let trimmed = phone.trim();
    if trimmed.is_empty() || trimmed == "▁" {
        return Vec::new();
    }
    ipa_to_arpabet(trimmed)
}

fn combo_bonus(edits: &[AcceptedProposal]) -> f32 {
    if edits.is_empty() {
        return 0.0;
    }
    let synergy = 0.10 * edits.len().saturating_sub(1) as f32;
    let exactish = edits.iter().filter(|edit| edit.score >= 1.20).count() as f32 * 0.03;
    synergy + exactish
}

fn top_non_overlapping_combos(
    edits: &[AcceptedProposal],
    max_edits: usize,
    limit: usize,
) -> Vec<Vec<AcceptedProposal>> {
    let pool = edits.iter().take(10).cloned().collect::<Vec<_>>();
    let mut combos = Vec::new();
    let mut current = Vec::new();
    enumerate_combos(&pool, 0, max_edits.max(1), &mut current, &mut combos);
    combos.sort_by(|a, b| {
        let a_score = score_sentence_candidate(a) + combo_bonus(a);
        let b_score = score_sentence_candidate(b) + combo_bonus(b);
        b_score
            .total_cmp(&a_score)
            .then_with(|| b.len().cmp(&a.len()))
    });
    combos.truncate(limit);
    combos
}

fn enumerate_combos(
    edits: &[AcceptedProposal],
    index: usize,
    max_edits: usize,
    current: &mut Vec<AcceptedProposal>,
    out: &mut Vec<Vec<AcceptedProposal>>,
) {
    if current.len() >= 2 {
        let mut candidate = current.clone();
        candidate.sort_by_key(|edit| edit.char_start);
        out.push(candidate);
    }
    if index >= edits.len() || current.len() >= max_edits {
        return;
    }
    for next in index..edits.len() {
        let edit = &edits[next];
        if current.iter().any(|existing| overlaps(existing, edit)) {
            continue;
        }
        current.push(edit.clone());
        enumerate_combos(edits, next + 1, max_edits, current, out);
        current.pop();
    }
}

fn apply_edits(input: &str, edits: &[AcceptedProposal]) -> String {
    if edits.is_empty() {
        return input.to_string();
    }
    let mut out = String::with_capacity(input.len());
    let mut cursor = 0;
    for edit in edits {
        if edit.char_start > cursor {
            out.push_str(&input[cursor..edit.char_start]);
        }
        let raw_slice = &input[edit.char_start..edit.char_end];
        out.push_str(&apply_edit_to_slice(raw_slice, &edit.matched_form, &edit.to));
        cursor = edit.char_end;
    }
    if cursor < input.len() {
        out.push_str(&input[cursor..]);
    }
    cleanup_replacement_artifacts(&out)
}

fn apply_edit_to_slice(raw_slice: &str, matched_form: &str, replacement: &str) -> String {
    let raw_tokens = tokenize_slice_words(raw_slice);
    let matched_tokens = tokenize_slice_words(matched_form);
    if matched_tokens.is_empty() {
        return replacement.to_string();
    }
    if let Some((start, end)) = find_token_subsequence(&raw_tokens, &matched_tokens) {
        let mut replace_end = raw_tokens[end - 1].2;
        if matched_form
            .chars()
            .last()
            .map(|ch| !ch.is_ascii_alphanumeric() && ch != '_')
            .unwrap_or(false)
        {
            while replace_end < raw_slice.len() {
                let mut iter = raw_slice[replace_end..].chars();
                let Some(ch) = iter.next() else { break };
                if ch.is_ascii_whitespace() || ch.is_ascii_alphanumeric() || ch == '_' {
                    break;
                }
                replace_end += ch.len_utf8();
            }
        }
        return format!(
            "{}{}{}",
            &raw_slice[..raw_tokens[start].1],
            replacement,
            &raw_slice[replace_end..]
        );
    }
    replacement.to_string()
}

fn cleanup_replacement_artifacts(text: &str) -> String {
    let mut out = text.to_string();
    while let Some(idx) = out.find(".-") {
        if idx > 0
            && out[..idx]
                .chars()
                .next_back()
                .map(|ch| ch.is_ascii_alphanumeric() || ch == '_')
                .unwrap_or(false)
        {
            out.replace_range(idx..idx + 2, "-");
        } else {
            break;
        }
    }
    out
}

fn tokenize_slice_words(text: &str) -> Vec<(String, usize, usize)> {
    let mut out = Vec::new();
    let mut current = None;
    for (idx, ch) in text.char_indices() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            if current.is_none() {
                current = Some(idx);
            }
        } else if let Some(start) = current.take() {
            out.push((text[start..idx].to_ascii_lowercase(), start, idx));
        }
    }
    if let Some(start) = current {
        out.push((text[start..].to_ascii_lowercase(), start, text.len()));
    }
    out
}

fn find_token_subsequence(
    haystack: &[(String, usize, usize)],
    needle: &[(String, usize, usize)],
) -> Option<(usize, usize)> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    for start in 0..=haystack.len() - needle.len() {
        let mut ok = true;
        for offset in 0..needle.len() {
            if haystack[start + offset].0 != needle[offset].0 {
                ok = false;
                break;
            }
        }
        if ok {
            return Some((start, start + needle.len()));
        }
    }
    None
}

fn is_noop_surface_rewrite(from: &str, to: &str) -> bool {
    fn normalize_surface(text: &str) -> String {
        text.split_whitespace()
            .map(|part| {
                part.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                    .to_ascii_lowercase()
            })
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>()
            .join(" ")
    }

    normalize_surface(from) == normalize_surface(to)
}

fn trim_token_bounds(raw: &str, absolute_start: usize) -> (usize, usize) {
    let mut core_start = None;
    let mut core_end = absolute_start + raw.len();
    for (offset, ch) in raw.char_indices() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            if core_start.is_none() {
                core_start = Some(absolute_start + offset);
            }
            core_end = absolute_start + offset + ch.len_utf8();
        }
    }
    match core_start {
        Some(start) => (start, core_end),
        None => (absolute_start, absolute_start + raw.len()),
    }
}

fn prune_redundant_edits(edits: Vec<AcceptedProposal>) -> Vec<AcceptedProposal> {
    let mut keep = vec![true; edits.len()];
    for i in 0..edits.len() {
        if !keep[i] {
            continue;
        }
        for j in 0..edits.len() {
            if i == j || !keep[j] {
                continue;
            }
            let a = &edits[i];
            let b = &edits[j];
            if a.to != b.to || !overlaps(a, b) {
                continue;
            }
            if prefers_edit(a, b) {
                keep[j] = false;
            }
        }
    }
    edits
        .into_iter()
        .zip(keep)
        .filter_map(|(edit, keep)| keep.then_some(edit))
        .collect()
}

fn prefers_edit(best: &AcceptedProposal, other: &AcceptedProposal) -> bool {
    let best_span_len = best.token_end - best.token_start;
    let other_span_len = other.token_end - other.token_start;
    if best.score > other.score + 0.02 {
        return true;
    }
    if best.score + 0.02 < other.score {
        return false;
    }
    if best_span_len < other_span_len {
        return true;
    }
    if best_span_len > other_span_len {
        return false;
    }
    best.char_start >= other.char_start && best.char_end <= other.char_end
}

fn normalize_token(token: &str) -> String {
    normalize_words(token)
}

fn normalize_words(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut prev_space = true;
    for ch in text.chars() {
        let mapped = if ch.is_ascii_alphanumeric() || ch == '_' {
            Some(ch.to_ascii_lowercase())
        } else if matches!(ch, '-' | '/' | '.' | ',' | ':' | ';' | '\'' | '"' | '(' | ')' | '[' | ']') {
            None
        } else {
            Some(' ')
        };
        match mapped {
            Some(' ') => {
                if !prev_space {
                    out.push(' ');
                    prev_space = true;
                }
            }
            Some(ch) => {
                out.push(ch);
                prev_space = false;
            }
            None => {}
        }
    }
    out.trim().to_string()
}

fn is_low_signal_confusion_alias(
    term_compact: &str,
    form_words: &str,
    word_count: usize,
    form_compact: &str,
) -> bool {
    if word_count != 1 {
        return false;
    }
    if form_compact.len() <= 2 {
        return true;
    }
    if matches!(
        form_words,
        "a"
            | "an"
            | "and"
            | "are"
            | "as"
            | "at"
            | "be"
            | "but"
            | "by"
            | "for"
            | "from"
            | "he"
            | "her"
            | "him"
            | "i"
            | "i'm"
            | "im"
            | "if"
            | "in"
            | "is"
            | "it"
            | "it's"
            | "its"
            | "me"
            | "my"
            | "of"
            | "on"
            | "or"
            | "our"
            | "she"
            | "that"
            | "the"
            | "their"
            | "them"
            | "there"
            | "they"
            | "this"
            | "to"
            | "us"
            | "we"
            | "were"
            | "what"
            | "when"
            | "where"
            | "who"
            | "with"
            | "you"
            | "your"
    ) {
        return true;
    }
    if form_compact.len() <= 3
        && bigram_dice(term_compact, form_compact) < 0.34
        && prefix_ratio(term_compact, form_compact) < 0.34
    {
        return true;
    }
    false
}

fn compact_key(text: &str) -> String {
    text.chars().filter(|c| !c.is_whitespace()).collect()
}

fn via_bonus(via: &str) -> f32 {
    match via {
        "confusion" => 0.02,
        "spoken" => 0.05,
        "alt" => 0.03,
        _ => 0.0,
    }
}

fn phonetic_resources() -> Option<&'static Mutex<PhoneticResources>> {
    PHONETIC_RESOURCES
        .get_or_init(|| {
            let dict = synth_corrupt::cmudict::load("data/cmudict.txt").ok()?;
            let g2p = G2p::load("models/g2p.fst", None).ok()?;
            Some(Mutex::new(PhoneticResources { g2p, dict }))
        })
        .as_ref()
}

fn phonemize_phrase(text: &str) -> Vec<String> {
    let normalized = normalize_words(text);
    if normalized.is_empty() {
        return Vec::new();
    }
    let Some(resources) = phonetic_resources() else {
        return Vec::new();
    };
    let guard = resources.lock().unwrap();
    normalized
        .split_whitespace()
        .flat_map(|word| guard.g2p.phonemize(word, &guard.dict))
        .collect()
}

fn phoneme_similarity(a: &[String], b: &[String]) -> Option<f32> {
    if a.is_empty() || b.is_empty() {
        return None;
    }
    let dist = phoneme_edit_distance(a, b) as f32 / 100.0;
    let max_len = a.len().max(b.len()) as f32;
    Some((1.0 - dist / max_len).clamp(0.0, 1.0))
}

fn phoneme_string(phonemes: &[String]) -> Option<String> {
    (!phonemes.is_empty()).then(|| phonemes.join(" "))
}

fn prefix_ratio(a: &str, b: &str) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let count = a
        .chars()
        .zip(b.chars())
        .take_while(|(lhs, rhs)| lhs == rhs)
        .count();
    count as f32 / a.len().min(b.len()) as f32
}

fn length_ratio(a: &str, b: &str) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    a.len().min(b.len()) as f32 / a.len().max(b.len()) as f32
}

fn bigram_dice(a: &str, b: &str) -> f32 {
    if a == b {
        return 1.0;
    }
    let a_bigrams = bigrams(a);
    let b_bigrams = bigrams(b);
    if a_bigrams.is_empty() || b_bigrams.is_empty() {
        return 0.0;
    }
    let b_set = b_bigrams.iter().collect::<HashSet<_>>();
    let overlap = a_bigrams.iter().filter(|gram| b_set.contains(gram)).count();
    (2 * overlap) as f32 / (a_bigrams.len() + b_bigrams.len()) as f32
}

fn bigrams(text: &str) -> Vec<(char, char)> {
    let chars = text.chars().collect::<Vec<_>>();
    chars.windows(2).map(|pair| (pair[0], pair[1])).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(term: &str, spoken: &str) -> VocabRow {
        VocabRow {
            id: 1,
            term: term.to_string(),
            spoken_auto: spoken.to_string(),
            spoken_override: Some(spoken.to_string()),
            reviewed: true,
            description: None,
        }
    }

    #[test]
    fn normalizes_hyphenated_surface() {
        assert_eq!(normalize_words("You thirty-two"), "you thirtytwo");
        assert_eq!(normalize_words("Yeah."), "yeah");
    }

    #[test]
    fn applies_exact_alt_spelling_replacement() {
        let vocab = vec![row("tokio", "tokio")];
        let mut alts = HashMap::new();
        alts.insert("tokio".to_string(), vec!["tokyo".to_string()]);
        let result = prototype_correct(
            "I like tokyo tasks.",
            &vocab,
            &alts,
            &HashMap::new(),
            PrototypeConfig::default(),
        );
        assert_eq!(result.corrected, "I like tokio tasks.");
        assert_eq!(result.accepted.len(), 1);
    }

    #[test]
    fn keeps_spoken_match_proposals_in_sentence_candidates() {
        let vocab = vec![row("serde", "sir day")];
        let result = prototype_correct(
            "I used sir day for this.",
            &vocab,
            &HashMap::new(),
            &HashMap::new(),
            PrototypeConfig::default(),
        );
        assert!(result
            .sentence_candidates
            .iter()
            .any(|candidate| candidate.text.contains("serde")));
    }

    #[test]
    fn preserves_edge_punctuation_in_replacement_spans() {
        let vocab = vec![row("bearcove", "bear cove")];
        let result = prototype_correct(
            "my company's called Bear Cove, and we're shipping",
            &vocab,
            &HashMap::new(),
            &HashMap::new(),
            PrototypeConfig::default(),
        );
        assert!(
            result
                .sentence_candidates
                .iter()
                .any(|candidate| candidate.text.contains("bearcove, and")),
            "{:#?}",
            result
                .sentence_candidates
                .iter()
                .map(|candidate| candidate.text.clone())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn prunes_longer_overlapping_spans_for_same_target() {
        let edits = vec![
            AcceptedProposal {
                token_start: 11,
                token_end: 13,
                char_start: 58,
                char_end: 67,
                from: "Bear Cove".to_string(),
                matched_form: "Bear Cove".to_string(),
                from_phonemes: None,
                to: "bearcove".to_string(),
                to_phonemes: None,
                via: "spoken".to_string(),
                score: 1.35,
                acoustic_score: None,
                acoustic_delta: None,
            },
            AcceptedProposal {
                token_start: 11,
                token_end: 14,
                char_start: 58,
                char_end: 71,
                from: "Bear Cove and".to_string(),
                matched_form: "Bear Cove".to_string(),
                from_phonemes: None,
                to: "bearcove".to_string(),
                to_phonemes: None,
                via: "spoken".to_string(),
                score: 0.98,
                acoustic_score: None,
                acoustic_delta: None,
            },
        ];
        let pruned = prune_redundant_edits(edits);
        assert_eq!(pruned.len(), 1);
        assert_eq!(pruned[0].from, "Bear Cove");
    }

    #[test]
    fn keeps_spacing_corrections_that_share_compact_form() {
        let vocab = vec![row("rustc", "rust see")];
        let result = prototype_correct(
            "the Rust C compiler",
            &vocab,
            &HashMap::new(),
            &HashMap::new(),
            PrototypeConfig::default(),
        );
        assert!(result
            .sentence_candidates
            .iter()
            .any(|candidate| candidate.text.contains("rustc compiler")));
    }

    #[test]
    fn confusion_forms_drive_candidate_retrieval() {
        let vocab = vec![row("serde", "sir day")];
        let mut confusions = HashMap::new();
        confusions.insert("serde".to_string(), vec!["certification".to_string()]);
        let result = prototype_correct(
            "Certification should be fine for this payload.",
            &vocab,
            &HashMap::new(),
            &confusions,
            PrototypeConfig::default(),
        );
        assert!(
            result
                .sentence_candidates
                .iter()
                .any(|candidate| candidate.text.contains("serde should be fine")),
            "{:#?}",
            result
                .sentence_candidates
                .iter()
                .map(|candidate| candidate.text.clone())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn fuzzy_confusion_aliases_do_not_dominate_irrelevant_spans() {
        let vocab = vec![row("rustc", "rust sea")];
        let mut confusions = HashMap::new();
        confusions.insert("rustc".to_string(), vec!["rusty".to_string()]);
        let result = prototype_correct(
            "it is just a config blob",
            &vocab,
            &HashMap::new(),
            &confusions,
            PrototypeConfig::default(),
        );
        assert!(
            result.proposals.iter().all(|proposal| {
                proposal
                    .candidates
                    .first()
                    .map(|candidate| candidate.term.as_str() != "rustc")
                    .unwrap_or(true)
            }),
            "{:#?}",
            result.proposals
        );
    }

    #[test]
    fn low_signal_confusion_aliases_are_ignored_but_real_ones_remain() {
        let vocab = vec![row("tokio", "tokyo")];
        let mut confusions = HashMap::new();
        confusions.insert(
            "tokio".to_string(),
            vec![
                "we".to_string(),
                "and".to_string(),
                "but".to_string(),
                "Tokyo".to_string(),
            ],
        );
        let result = prototype_correct(
            "We should consider using Tokyo for the async runtime.",
            &vocab,
            &HashMap::new(),
            &confusions,
            PrototypeConfig::default(),
        );
        assert!(
            result
                .proposals
                .iter()
                .filter(|proposal| proposal.normalized == "we")
                .all(|proposal| proposal
                    .candidates
                    .iter()
                    .all(|candidate| candidate.term.as_str() != "tokio")),
            "{:#?}",
            result.proposals
        );
        assert!(
            result
                .sentence_candidates
                .iter()
                .any(|candidate| candidate.text.contains("using tokio for the async runtime")),
            "{:#?}",
            result
                .sentence_candidates
                .iter()
                .map(|candidate| candidate.text.clone())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn builds_multi_fix_combo_candidates() {
        let vocab = vec![
            row("fasterthanlime", "faster than lime"),
            row("bearcove", "bear cove"),
            row("rustc", "rust c"),
        ];
        let result = prototype_correct(
            "Hey, my name is Faster Than Lime, and my company's called Bear Cove, and we're using the Rust C compiler.",
            &vocab,
            &HashMap::new(),
            &HashMap::new(),
            PrototypeConfig::default(),
        );
        assert!(
            result.sentence_candidates.iter().any(|candidate| {
                candidate.text.contains("fasterthanlime")
                    && candidate.text.contains("bearcove")
                    && candidate.text.contains("rustc")
            }),
            "{:#?}",
            result
                .sentence_candidates
                .iter()
                .map(|candidate| candidate.text.clone())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn acoustic_delta_is_relative_to_observed_surface() {
        let observed = Some(0.84);
        let candidate = Some(0.58);
        let down = acoustic_delta(candidate, observed).unwrap();
        let up = acoustic_delta(observed, candidate).unwrap();
        assert!((down + 0.26).abs() < 1e-5, "{down}");
        assert!((up - 0.26).abs() < 1e-5, "{up}");
        assert_eq!(acoustic_delta(None, observed), None);
        assert_eq!(acoustic_delta(candidate, None), None);
    }

    #[test]
    fn plausible_spoken_vocab_repair_beats_original_sentence() {
        let edit = AcceptedProposal {
            token_start: 3,
            token_end: 4,
            char_start: 20,
            char_end: 29,
            from: "serdejson".to_string(),
            matched_form: "serdejson".to_string(),
            from_phonemes: Some("S ER D EH JH S AH N".to_string()),
            to: "serde_json".to_string(),
            to_phonemes: Some("S ER D EH JH EY Z AH N".to_string()),
            via: "spoken".to_string(),
            score: 0.69,
            acoustic_score: Some(0.65),
            acoustic_delta: Some(0.04),
        };
        let score = score_sentence_candidate(&[edit]);
        assert!(score > 0.0, "{score}");
    }

    #[test]
    fn apply_edit_to_slice_preserves_hyphenated_suffix() {
        let out = apply_edit_to_slice(
            "A Arch sixty-four.-compatible",
            "A arch sixty-four",
            "AArch64",
        );
        assert_eq!(cleanup_replacement_artifacts(&out), "AArch64-compatible");
    }

    #[test]
    fn apply_edit_to_slice_can_consume_trailing_punctuation_from_match() {
        let out = apply_edit_to_slice("Two, eight.", "Two, eight.", "u8");
        assert_eq!(out, "u8");
    }

    #[test]
    fn build_edit_pool_admits_spoken_phonetic_repair_below_flat_threshold() {
        let spans = vec![PrototypeSpanProposal {
            token_start: 4,
            token_end: 5,
            char_start: 23,
            char_end: 32,
            raw_text: "serdejson".to_string(),
            normalized: "serdejson".to_string(),
            phonemes: Some("S EH R D EH JH S AO N".to_string()),
            acoustic_phonemes: None,
            observed_acoustic_score: None,
            acoustic_trustworthy: false,
            acoustic_window_start_sec: None,
            acoustic_window_end_sec: None,
            candidates: vec![PrototypeCandidate {
                term: "serde_json".to_string(),
                via: "spoken".to_string(),
                matched_form: "ser-dee jay-son".to_string(),
                matched_form_phonemes: Some("S EH R D EH JH EY Z AH N".to_string()),
                term_preview: Some("ser-dee jay-son".to_string()),
                term_preview_phonemes: Some("S EH R D EH JH EY Z AH N".to_string()),
                score: 0.60,
                lexical_score: Some(0.56),
                dice: Some(0.51),
                prefix_ratio: Some(0.60),
                length_ratio: Some(1.0),
                phonetic_score: Some(0.78),
                observed_acoustic_score: Some(0.61),
                acoustic_score: Some(0.65),
                acoustic_delta: Some(0.04),
                phonemes: Some("S EH R D EH JH EY Z AH N".to_string()),
                exact_words: false,
                exact_compact: false,
            }],
        }];
        let edits = build_edit_pool(&spans);
        assert_eq!(edits.len(), 1, "{:#?}", edits);
        assert_eq!(edits[0].to, "serde_json");
    }

    #[test]
    fn build_edit_pool_keeps_multiple_strong_same_span_candidates() {
        let spans = vec![PrototypeSpanProposal {
            token_start: 2,
            token_end: 3,
            char_start: 10,
            char_end: 15,
            raw_text: "Right".to_string(),
            normalized: "right".to_string(),
            phonemes: Some("R AY T".to_string()),
            acoustic_phonemes: None,
            observed_acoustic_score: None,
            acoustic_trustworthy: false,
            acoustic_window_start_sec: None,
            acoustic_window_end_sec: None,
            candidates: vec![
                PrototypeCandidate {
                    term: "regalloc".to_string(),
                    via: "confusion".to_string(),
                    matched_form: "Right.".to_string(),
                    matched_form_phonemes: Some("R AY T".to_string()),
                    term_preview: Some("regg-alloc".to_string()),
                    term_preview_phonemes: Some("R EH G G AE L L AO K".to_string()),
                    score: 1.32,
                    lexical_score: Some(1.0),
                    dice: Some(1.0),
                    prefix_ratio: Some(1.0),
                    length_ratio: Some(1.0),
                    phonetic_score: Some(1.0),
                    observed_acoustic_score: None,
                    acoustic_score: None,
                    acoustic_delta: None,
                    phonemes: Some("R EH G G AE L L AO K".to_string()),
                    exact_words: true,
                    exact_compact: true,
                },
                PrototypeCandidate {
                    term: "reqwest".to_string(),
                    via: "confusion".to_string(),
                    matched_form: "Right.".to_string(),
                    matched_form_phonemes: Some("R AY T".to_string()),
                    term_preview: Some("request".to_string()),
                    term_preview_phonemes: Some("R IH K W EH S T".to_string()),
                    score: 1.32,
                    lexical_score: Some(1.0),
                    dice: Some(1.0),
                    prefix_ratio: Some(1.0),
                    length_ratio: Some(1.0),
                    phonetic_score: Some(1.0),
                    observed_acoustic_score: None,
                    acoustic_score: None,
                    acoustic_delta: None,
                    phonemes: Some("R IH K W EH S T".to_string()),
                    exact_words: true,
                    exact_compact: true,
                },
            ],
        }];
        let edits = build_edit_pool(&spans);
        assert_eq!(edits.len(), 2, "{:#?}", edits);
        assert!(edits.iter().any(|edit| edit.to == "regalloc"));
        assert!(edits.iter().any(|edit| edit.to == "reqwest"));
    }

    #[test]
    fn build_edit_pool_admits_prefix_heavy_spoken_near_miss() {
        let spans = vec![PrototypeSpanProposal {
            token_start: 4,
            token_end: 7,
            char_start: 32,
            char_end: 62,
            raw_text: "A Arch sixty-four.-compatible".to_string(),
            normalized: "a arch sixtyfourcompatible".to_string(),
            phonemes: Some("AH AA R CH S IH K S T Y F AO AH R K AO M P AE T IH B L EH".to_string()),
            acoustic_phonemes: None,
            observed_acoustic_score: None,
            acoustic_trustworthy: false,
            acoustic_window_start_sec: None,
            acoustic_window_end_sec: None,
            candidates: vec![PrototypeCandidate {
                term: "AArch64".to_string(),
                via: "spoken".to_string(),
                matched_form: "A arch sixty-four".to_string(),
                matched_form_phonemes: Some("AH AA R CH S IH K S T Y F AO AH R".to_string()),
                term_preview: Some("A arch sixty-four".to_string()),
                term_preview_phonemes: Some("AH AA R CH S IH K S T Y F AO AH R".to_string()),
                score: 0.6277778,
                lexical_score: Some(0.7077778),
                dice: Some(0.7777778),
                prefix_ratio: Some(1.0),
                length_ratio: Some(0.5833333),
                phonetic_score: Some(0.5833334),
                observed_acoustic_score: None,
                acoustic_score: None,
                acoustic_delta: None,
                phonemes: Some("AH AA R CH S IH K S T Y F AO AH R".to_string()),
                exact_words: false,
                exact_compact: false,
            }],
        }];
        let edits = build_edit_pool(&spans);
        assert_eq!(edits.len(), 1, "{:#?}", edits);
        assert_eq!(edits[0].to, "AArch64");
    }
}
