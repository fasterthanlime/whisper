use std::hash::{Hash, Hasher};

use bee_phonetic::{AliasSource, CandidateFeatureRow, IdentifierFlags, TranscriptSpan};
use bee_phonetic::{SentenceWordToken, sentence_word_tokens};

use crate::sparse_ftrl::{Feature, SparseFtrl};

const LOW_CONTENT: &[&str] = &[
    "a", "an", "and", "the", "then", "that", "this", "these", "those", "if", "you", "we",
    "they", "he", "she", "it", "i", "me", "my", "your", "our", "their", "him", "her", "them",
    "about", "not", "sure", "what", "yeah", "well", "oh", "hmm", "uh", "um", "want", "some",
    "there", "here",
];

// ── Dense feature layout ────────────────────────────────────────────

// Each candidate is scored independently: "should this replacement happen?"
// All candidates share the same feature space so learning transfers across cases.
const FEATURE_NAMES: &[&str] = &[
    "bias",
    "acceptance_score",
    "phonetic_score",
    "coarse_score",
    "token_score",
    "feature_score",
    "feature_bonus",
    "best_view_score",
    "cross_view_support",
    "qgram_overlap",
    "total_qgram_overlap",
    "token_count_match",
    "phone_closeness",
    "alias_source_spoken",
    "alias_source_identifier",
    "alias_source_confusion",
    "identifier_acronym",
    "identifier_digits",
    "identifier_snake",
    "identifier_camel",
    "identifier_symbol",
    "short_guard_passed",
    "low_content_guard_passed",
    "acceptance_floor_passed",
    "verified",
    "span_token_count",
    "span_phone_count",
    "span_low_content",
    // ASR uncertainty features (indices 28-31)
    "span_mean_logprob",
    "span_min_logprob",
    "span_mean_margin",
    "span_min_margin",
];

const NUM_DENSE: usize = 32;

/// Offset for sparse hashed features so they don't collide with dense indices 0..31.
const SPARSE_OFFSET: u64 = 1000;
/// Hash space for sparse context features.
const SPARSE_BUCKETS: u64 = 1 << 14; // 16384

const BASE_WEIGHTS: &[(u64, f64)] = &[
    (0, 0.0),    // bias
    (1, 1.8),    // acceptance_score
    (2, 1.2),    // phonetic_score
    (3, 0.8),    // coarse_score
    (4, 0.6),    // token_score
    (5, 0.5),    // feature_score
    (6, 0.3),    // feature_bonus
    (7, 0.3),    // best_view_score
    (8, 0.15),   // cross_view_support
    (9, 0.12),   // qgram_overlap
    (10, 0.05),  // total_qgram_overlap
    (11, 0.15),  // token_count_match
    (12, 0.25),  // phone_closeness
    (13, 0.08),  // alias_source_spoken
    (14, 0.06),  // alias_source_identifier
    (15, -0.04), // alias_source_confusion
    (16, 0.04),  // identifier_acronym
    (17, 0.04),  // identifier_digits
    (18, 0.02),  // identifier_snake
    (19, 0.02),  // identifier_camel
    (20, 0.02),  // identifier_symbol
    (21, 0.10),  // short_guard_passed
    (22, 0.10),  // low_content_guard_passed
    (23, 0.20),  // acceptance_floor_passed
    (24, 0.30),  // verified
    (25, 0.02),  // span_token_count
    (26, 0.02),  // span_phone_count
    (27, -0.30), // span_low_content
    // ASR uncertainty: no base weights (start at 0, learn from data)
];

/// Threshold for the judge to accept a candidate replacement.
/// If no candidate's probability exceeds this, keep original.
const ACCEPT_THRESHOLD: f32 = 0.5;

// ── Context ─────────────────────────────────────────────────────────

/// Sentence-level context for the span being evaluated.
/// Extracted at the call site from the full transcript.
#[derive(Clone, Debug, Default)]
pub struct SpanContext {
    /// 1-2 lowercased words to the left of the span.
    pub left_tokens: Vec<String>,
    /// 1-2 lowercased words to the right of the span.
    pub right_tokens: Vec<String>,
    /// Span appears to be in code-like context.
    pub code_like: bool,
    /// Span appears to be in prose.
    pub prose_like: bool,
    /// Line starts with a list marker.
    pub list_like: bool,
    /// Span is at the start of a sentence.
    pub sentence_start: bool,
    /// Application context identifier (e.g., "terminal", "xcode").
    pub app_id: Option<String>,
}

// ── Hashing ─────────────────────────────────────────────────────────

/// FNV-1a hash of a feature name string, mapped into the sparse bucket space.
fn hash_feature(name: &str) -> u64 {
    let mut hasher = fnv::FnvHasher::default();
    name.hash(&mut hasher);
    SPARSE_OFFSET + (hasher.finish() % SPARSE_BUCKETS)
}

/// Build sparse context features for one candidate.
fn context_features(ctx: &SpanContext, candidate_term: &str) -> Vec<Feature> {
    let mut features = Vec::with_capacity(20);
    let term_lower = candidate_term.to_ascii_lowercase();

    // Left context tokens
    if let Some(l1) = ctx.left_tokens.first() {
        features.push(Feature { index: hash_feature(&format!("L1={l1}")), value: 1.0 });
        features.push(Feature { index: hash_feature(&format!("TERM={term_lower}|L1={l1}")), value: 1.0 });
    }
    if ctx.left_tokens.len() >= 2 {
        let bigram = format!("{}_{}", ctx.left_tokens[0], ctx.left_tokens[1]);
        features.push(Feature { index: hash_feature(&format!("L2={bigram}")), value: 1.0 });
        features.push(Feature { index: hash_feature(&format!("TERM={term_lower}|L2={bigram}")), value: 1.0 });
    }

    // Right context tokens
    if let Some(r1) = ctx.right_tokens.first() {
        features.push(Feature { index: hash_feature(&format!("R1={r1}")), value: 1.0 });
        features.push(Feature { index: hash_feature(&format!("TERM={term_lower}|R1={r1}")), value: 1.0 });
    }
    if ctx.right_tokens.len() >= 2 {
        let bigram = format!("{}_{}", ctx.right_tokens[0], ctx.right_tokens[1]);
        features.push(Feature { index: hash_feature(&format!("R2={bigram}")), value: 1.0 });
        features.push(Feature { index: hash_feature(&format!("TERM={term_lower}|R2={bigram}")), value: 1.0 });
    }

    // Candidate identity
    features.push(Feature { index: hash_feature(&format!("TERM={term_lower}")), value: 1.0 });

    // Context type flags
    if ctx.code_like {
        features.push(Feature { index: hash_feature("CTX=code"), value: 1.0 });
    }
    if ctx.prose_like {
        features.push(Feature { index: hash_feature("CTX=prose"), value: 1.0 });
    }
    if ctx.list_like {
        features.push(Feature { index: hash_feature("CTX=list"), value: 1.0 });
    }
    if ctx.sentence_start {
        features.push(Feature { index: hash_feature("CTX=sent_start"), value: 1.0 });
    }

    // App context
    if let Some(app) = &ctx.app_id {
        features.push(Feature { index: hash_feature(&format!("APP={app}")), value: 1.0 });
        features.push(Feature { index: hash_feature(&format!("TERM={term_lower}|APP={app}")), value: 1.0 });
    }

    features
}

// ── Judge ────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct OnlineJudge {
    model: SparseFtrl,
    update_count: u32,
}

#[derive(Clone, Debug)]
pub struct JudgeOption {
    pub alias_id: Option<u32>,
    pub term: String,
    pub is_keep_original: bool,
    pub score: f32,
    pub probability: f32,
    pub chosen: bool,
}

#[derive(Clone, Debug)]
struct JudgeExample {
    alias_id: u32,
    term: String,
    features: Vec<Feature>,
}

impl Default for OnlineJudge {
    fn default() -> Self {
        let mut model = SparseFtrl::new(1.0, 1.0, 0.0001, 0.001);
        seed_model(&mut model);
        let judge = Self {
            model,
            update_count: 0,
        };
        tracing::info!(
            num_active = judge.model.num_active(),
            "judge initialized with seed weights"
        );
        judge
    }
}

/// Seed the FTRL model by training on synthetic examples that span the
/// quality range. This teaches the model: high scores → replace, low scores → keep.
fn seed_model(model: &mut SparseFtrl) {
    // Synthetic candidates at various quality levels.
    // (accept, phonetic, coarse, token, feature, verified, target)
    let levels: &[(f64, f64, f64, f64, f64, f64, bool)] = &[
        // Strong matches → should replace
        (0.90, 0.88, 0.85, 0.86, 0.88, 1.0, true),
        (0.82, 0.80, 0.75, 0.78, 0.82, 1.0, true),
        (0.75, 0.72, 0.68, 0.70, 0.74, 1.0, true),
        (0.68, 0.65, 0.60, 0.64, 0.66, 1.0, true),
        (0.60, 0.58, 0.55, 0.56, 0.60, 1.0, true),
        // Borderline → should replace (above threshold)
        (0.55, 0.52, 0.50, 0.50, 0.54, 1.0, true),
        // Weak matches → should NOT replace
        (0.45, 0.42, 0.40, 0.40, 0.44, 1.0, false),
        (0.38, 0.35, 0.32, 0.34, 0.36, 0.0, false),
        (0.30, 0.28, 0.25, 0.26, 0.30, 0.0, false),
        (0.20, 0.18, 0.15, 0.16, 0.20, 0.0, false),
        (0.12, 0.10, 0.08, 0.10, 0.12, 0.0, false),
        (0.05, 0.04, 0.03, 0.04, 0.05, 0.0, false),
    ];

    for epoch in 0..20 {
        for &(accept, phonetic, coarse, token, feature, verified, target) in levels {
            let features = dense_features_from_synthetic(
                accept, phonetic, coarse, token, feature, verified,
            );
            model.update(&features, target);
        }
        // Early epochs use higher alpha for faster convergence
        if epoch == 5 {
            model.alpha = 0.5;
        }
    }
    // Reset to default learning rate for online updates
    model.alpha = 0.5;
}

fn dense_features_from_synthetic(
    accept: f64, phonetic: f64, coarse: f64, token: f64, feature: f64, verified: f64,
) -> Vec<Feature> {
    let values = [
        1.0,                                                        // bias
        accept,                                                     // acceptance_score
        phonetic,                                                   // phonetic_score
        coarse,                                                     // coarse_score
        token,                                                      // token_score
        feature,                                                    // feature_score
        (feature - token).max(0.0),                                 // feature_bonus
        coarse * 0.9,                                               // best_view_score
        0.33,                                                       // cross_view_support
        coarse * 0.5,                                               // qgram_overlap
        coarse * 0.8,                                               // total_qgram_overlap
        if accept > 0.5 { 1.0 } else { 0.0 },                      // token_count_match
        if accept > 0.5 { 0.80 } else { 0.40 },                    // phone_closeness
        0.0,                                                        // alias_source_spoken
        0.0,                                                        // alias_source_identifier
        0.0,                                                        // alias_source_confusion
        0.0, 0.0, 0.0, 0.0, 0.0,                                   // identifier flags
        if verified > 0.5 { 1.0 } else { 0.0 },                    // short_guard_passed
        1.0,                                                        // low_content_guard_passed
        if accept > 0.35 { 1.0 } else { 0.0 },                     // acceptance_floor_passed
        verified,                                                   // verified
        0.25,                                                       // span_token_count
        0.40,                                                       // span_phone_count
        0.0,                                                        // span_low_content
        // ASR uncertainty: not available in synthetic data
        0.0, 0.0, 0.0, 0.0,
    ];
    values
        .iter()
        .enumerate()
        .map(|(i, &v)| Feature { index: i as u64, value: v })
        .collect()
}

impl OnlineJudge {
    pub fn feature_names(&self) -> Vec<String> {
        FEATURE_NAMES.iter().map(|name| (*name).to_string()).collect()
    }

    /// Get dense feature weights (indices 0..NUM_DENSE) for debugging.
    pub fn weights(&self) -> Vec<f32> {
        (0..NUM_DENSE)
            .map(|i| self.model.weight_at(i as u64) as f32)
            .collect()
    }

    pub fn update_count(&self) -> u32 {
        self.update_count
    }

    pub fn score_candidates(
        &self,
        span: &TranscriptSpan,
        candidates: &[(CandidateFeatureRow, IdentifierFlags)],
        ctx: &SpanContext,
    ) -> Vec<JudgeOption> {
        let examples = build_examples(span, candidates, ctx);
        score_examples(&self.model, &examples)
    }

    pub fn teach_choice(
        &mut self,
        span: &TranscriptSpan,
        candidates: &[(CandidateFeatureRow, IdentifierFlags)],
        chosen_alias_id: Option<u32>,
        ctx: &SpanContext,
    ) -> Vec<JudgeOption> {
        let examples = build_examples(span, candidates, ctx);
        if examples.is_empty() {
            return vec![keep_original_option()];
        }

        // chosen_alias_id == None means "keep original" => all candidates are false
        for _ in 0..4 {
            for example in &examples {
                let label = Some(example.alias_id) == chosen_alias_id;
                self.model.update(&example.features, label);
            }
        }
        self.update_count += 1;

        tracing::debug!(
            update_count = self.update_count,
            chosen = ?chosen_alias_id,
            num_candidates = examples.len(),
            num_active = self.model.num_active(),
            "judge taught"
        );

        score_examples(&self.model, &examples)
    }
}

fn build_examples(
    span: &TranscriptSpan,
    candidates: &[(CandidateFeatureRow, IdentifierFlags)],
    ctx: &SpanContext,
) -> Vec<JudgeExample> {
    let span_token_count = (span.token_end - span.token_start) as f64;
    let span_phone_count = span.ipa_tokens.len() as f64;
    let span_low_content = is_low_content_span(&span.text) as u8 as f64;

    // ASR uncertainty (normalized by /5.0 as per design doc)
    let mean_lp = span.mean_logprob.unwrap_or(0.0) as f64 / 5.0;
    let min_lp = span.min_logprob.unwrap_or(0.0) as f64 / 5.0;
    let mean_m = span.mean_margin.unwrap_or(0.0) as f64 / 5.0;
    let min_m = span.min_margin.unwrap_or(0.0) as f64 / 5.0;

    candidates
        .iter()
        .map(|(candidate, flags)| {
            // Dense features (indices 0..31)
            let mut features: Vec<Feature> = vec![
                Feature { index: 0, value: 1.0 },                                              // bias
                Feature { index: 1, value: candidate.acceptance_score as f64 },                 // acceptance_score
                Feature { index: 2, value: candidate.phonetic_score as f64 },                   // phonetic_score
                Feature { index: 3, value: candidate.coarse_score as f64 },                     // coarse_score
                Feature { index: 4, value: candidate.token_score as f64 },                      // token_score
                Feature { index: 5, value: candidate.feature_score as f64 },                    // feature_score
                Feature { index: 6, value: candidate.feature_bonus as f64 },                    // feature_bonus
                Feature { index: 7, value: candidate.best_view_score as f64 },                  // best_view_score
                Feature { index: 8, value: candidate.cross_view_support as f64 / 6.0 },         // cross_view_support
                Feature { index: 9, value: candidate.qgram_overlap as f64 / 10.0 },             // qgram_overlap
                Feature { index: 10, value: candidate.total_qgram_overlap as f64 / 20.0 },      // total_qgram_overlap
                Feature { index: 11, value: candidate.token_count_match as u8 as f64 },          // token_count_match
                Feature { index: 12, value: 1.0 / (1.0 + candidate.phone_count_delta.abs() as f64) }, // phone_closeness
                Feature { index: 13, value: (candidate.alias_source == AliasSource::Spoken) as u8 as f64 },
                Feature { index: 14, value: (candidate.alias_source == AliasSource::Identifier) as u8 as f64 },
                Feature { index: 15, value: (candidate.alias_source == AliasSource::Confusion) as u8 as f64 },
                Feature { index: 16, value: flags.acronym_like as u8 as f64 },
                Feature { index: 17, value: flags.has_digits as u8 as f64 },
                Feature { index: 18, value: flags.snake_like as u8 as f64 },
                Feature { index: 19, value: flags.camel_like as u8 as f64 },
                Feature { index: 20, value: flags.symbol_like as u8 as f64 },
                Feature { index: 21, value: candidate.short_guard_passed as u8 as f64 },
                Feature { index: 22, value: candidate.low_content_guard_passed as u8 as f64 },
                Feature { index: 23, value: candidate.acceptance_floor_passed as u8 as f64 },
                Feature { index: 24, value: candidate.verified as u8 as f64 },
                Feature { index: 25, value: span_token_count / 4.0 },
                Feature { index: 26, value: span_phone_count / 12.0 },
                Feature { index: 27, value: span_low_content },
                // ASR uncertainty
                Feature { index: 28, value: mean_lp },
                Feature { index: 29, value: min_lp },
                Feature { index: 30, value: mean_m },
                Feature { index: 31, value: min_m },
            ];

            // Sparse context features (hashed into offset range)
            features.extend(context_features(ctx, &candidate.term));

            JudgeExample {
                alias_id: candidate.alias_id,
                term: candidate.term.clone(),
                features,
            }
        })
        .collect()
}

fn score_examples(model: &SparseFtrl, examples: &[JudgeExample]) -> Vec<JudgeOption> {
    let mut options: Vec<JudgeOption> = examples
        .iter()
        .map(|example| {
            let score = model.predict(&example.features) as f32;
            let probability = model.predict_prob(&example.features) as f32;
            JudgeOption {
                alias_id: Some(example.alias_id),
                term: example.term.clone(),
                is_keep_original: false,
                score,
                probability,
                chosen: false,
            }
        })
        .collect();

    // Sort by probability descending
    options.sort_by(|lhs, rhs| {
        rhs.probability
            .total_cmp(&lhs.probability)
            .then_with(|| rhs.score.total_cmp(&lhs.score))
    });

    // Best candidate wins only if it exceeds the threshold; otherwise keep original
    let best_exceeds_threshold = options
        .first()
        .is_some_and(|best| best.probability >= ACCEPT_THRESHOLD);

    if best_exceeds_threshold {
        if let Some(first) = options.first_mut() {
            first.chosen = true;
        }
    }

    // Always include a keep_original option so callers can see it
    let mut keep = keep_original_option();
    if !best_exceeds_threshold {
        keep.chosen = true;
    }
    // keep_original probability = 1 - best candidate probability
    keep.probability = 1.0 - options.first().map(|o| o.probability).unwrap_or(0.0);
    keep.score = -options.first().map(|o| o.score).unwrap_or(0.0);
    options.push(keep);

    options
}

fn keep_original_option() -> JudgeOption {
    JudgeOption {
        alias_id: None,
        term: "keep_original".to_string(),
        is_keep_original: true,
        score: 0.0,
        probability: 1.0,
        chosen: false,
    }
}

fn is_low_content_span(text: &str) -> bool {
    let tokens: Vec<SentenceWordToken> = sentence_word_tokens(text);
    !tokens.is_empty()
        && tokens.iter().all(|token| {
            let lower = token.text.to_ascii_lowercase();
            LOW_CONTENT.iter().any(|entry| *entry == lower)
        })
}

/// Extract SpanContext from a transcript and span boundaries.
pub fn extract_span_context(
    transcript: &str,
    char_start: usize,
    char_end: usize,
) -> SpanContext {
    let before = &transcript[..char_start];
    let after = &transcript[char_end..];

    // Left tokens: split on whitespace, take last 2
    let left_words: Vec<String> = before
        .split_whitespace()
        .rev()
        .take(2)
        .map(|w| w.to_ascii_lowercase())
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();

    // Right tokens: split on whitespace, take first 2
    let right_words: Vec<String> = after
        .split_whitespace()
        .take(2)
        .map(|w| w.to_ascii_lowercase())
        .collect();

    // Code-like: check ±10 chars for code markers
    let window_start = char_start.saturating_sub(10);
    let window_end = (char_end + 10).min(transcript.len());
    let window = &transcript[window_start..window_end];
    let code_markers = ["()", "{}", "::", ".", "_", "->", "=>", "fn ", "let "];
    let code_like = code_markers.iter().any(|m| window.contains(m));

    // List-like: line starts with a list marker
    let line_start = before.rfind('\n').map(|i| i + 1).unwrap_or(0);
    let line_prefix = transcript[line_start..char_start].trim_start();
    let list_like = line_prefix.starts_with('-')
        || line_prefix.starts_with('*')
        || line_prefix.chars().next().is_some_and(|c| c.is_ascii_digit());

    // Sentence start: span is at the beginning or after sentence-ending punctuation
    let sentence_start = before.is_empty()
        || before
            .trim_end()
            .ends_with(|c: char| c == '.' || c == '!' || c == '?' || c == '\n');

    SpanContext {
        left_tokens: left_words,
        right_tokens: right_words,
        code_like,
        prose_like: !code_like,
        list_like,
        sentence_start,
        app_id: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bee_phonetic::{AliasSource, IdentifierFlags, IndexView, TranscriptSpan};

    fn span(text: &str) -> TranscriptSpan {
        TranscriptSpan {
            token_start: 0,
            token_end: 2,
            char_start: 0,
            char_end: text.len(),
            start_sec: None,
            end_sec: None,
            text: text.to_string(),
            ipa_tokens: vec!["ɹ".into(), "ɛ".into(), "k".into()],
            reduced_ipa_tokens: vec!["r".into(), "e".into(), "k".into()],
            ..Default::default()
        }
    }

    fn ctx() -> SpanContext {
        SpanContext::default()
    }

    fn candidate(
        alias_id: u32,
        term: &str,
        acceptance: f32,
        phonetic: f32,
        coarse: f32,
        verified: bool,
    ) -> CandidateFeatureRow {
        CandidateFeatureRow {
            alias_id,
            term: term.to_string(),
            alias_text: term.to_string(),
            alias_source: AliasSource::Canonical,
            matched_view: IndexView::RawIpa2,
            qgram_overlap: 4,
            total_qgram_overlap: 7,
            best_view_score: coarse,
            cross_view_support: 3,
            token_count_match: true,
            phone_count_delta: 0,
            token_bonus: 0.0,
            phone_bonus: 0.0,
            extra_length_penalty: 0.0,
            structure_bonus: 0.0,
            coarse_score: coarse,
            token_distance: 1,
            token_weighted_distance: 1.0,
            token_boundary_penalty: 0.0,
            token_max_len: 3,
            token_score: phonetic,
            token_ops: Vec::new(),
            feature_distance: 1.0,
            feature_weighted_distance: 1.0,
            feature_boundary_penalty: 0.0,
            feature_max_len: 3,
            feature_score: phonetic,
            feature_ops: Vec::new(),
            feature_bonus: 0.0,
            feature_gate_token_ok: true,
            feature_gate_coarse_ok: true,
            feature_gate_phone_ok: true,
            short_guard_applied: false,
            short_guard_onset_match: false,
            short_guard_passed: true,
            low_content_guard_applied: false,
            low_content_guard_passed: true,
            acceptance_floor_passed: true,
            used_feature_bonus: false,
            phonetic_score: phonetic,
            acceptance_score: acceptance,
            verified,
        }
    }

    #[test]
    fn score_candidates_includes_keep_original() {
        let judge = OnlineJudge::default();
        let options = judge.score_candidates(
            &span("req west"),
            &[(
                candidate(7, "reqwest", 0.85, 0.82, 0.78, true),
                IdentifierFlags::default(),
            )],
            &ctx(),
        );
        assert!(options.iter().any(|option| option.is_keep_original));
        assert!(options.iter().any(|option| option.alias_id == Some(7)));
    }

    #[test]
    fn teach_choice_promotes_selected_candidate() {
        let mut judge = OnlineJudge::default();
        let span = span("req west");
        let candidates = vec![
            (
                candidate(7, "reqwest", 0.75, 0.74, 0.70, true),
                IdentifierFlags::default(),
            ),
            (
                candidate(8, "request", 0.40, 0.38, 0.35, false),
                IdentifierFlags::default(),
            ),
        ];

        // Teach 5 times so the model has enough signal
        for _ in 0..5 {
            judge.teach_choice(&span, &candidates, Some(7), &ctx());
        }

        let after = judge.score_candidates(&span, &candidates, &ctx());
        let after_reqwest = after
            .iter()
            .find(|option| option.alias_id == Some(7))
            .expect("reqwest option should exist")
            .probability;

        assert_eq!(judge.update_count(), 5);
        // reqwest should still be the chosen candidate and above threshold
        let chosen = after.iter().find(|o| o.chosen).expect("should have a chosen option");
        assert_eq!(chosen.alias_id, Some(7), "reqwest should be chosen");
        assert!(
            after_reqwest > ACCEPT_THRESHOLD,
            "reqwest should exceed threshold: prob={after_reqwest}"
        );
        // reqwest should beat request
        let after_request = after
            .iter()
            .find(|o| o.alias_id == Some(8))
            .expect("request option should exist")
            .probability;
        assert!(
            after_reqwest > after_request,
            "reqwest ({after_reqwest}) should beat request ({after_request})"
        );
    }

    #[test]
    fn teach_keep_original_lowers_candidate_scores() {
        let mut judge = OnlineJudge::default();
        let span = span("req west");
        let candidates = vec![
            (
                candidate(7, "reqwest", 0.75, 0.74, 0.70, true),
                IdentifierFlags::default(),
            ),
        ];

        let before = judge.score_candidates(&span, &candidates, &ctx());
        let before_prob = before
            .iter()
            .find(|o| o.alias_id == Some(7))
            .unwrap()
            .probability;

        // Teach keep_original (chosen_alias_id = None) 5 times
        for _ in 0..5 {
            judge.teach_choice(&span, &candidates, None, &ctx());
        }

        let after = judge.score_candidates(&span, &candidates, &ctx());
        let after_prob = after
            .iter()
            .find(|o| o.alias_id == Some(7))
            .unwrap()
            .probability;

        assert!(
            after_prob < before_prob,
            "teaching keep_original should lower candidate probability: before={before_prob} after={after_prob}"
        );
    }

    #[test]
    fn teaching_transfers_to_similar_candidates() {
        let mut judge = OnlineJudge::default();
        let span1 = span("sir day");
        let candidates1 = vec![
            (
                candidate(1, "serde", 0.80, 0.78, 0.75, true),
                IdentifierFlags::default(),
            ),
        ];

        // Teach: "serde" is correct for "sir day"
        for _ in 0..10 {
            judge.teach_choice(&span1, &candidates1, Some(1), &ctx());
        }

        // Now score a DIFFERENT case with similar features
        let span2 = span("toe key oh");
        let candidates2 = vec![
            (
                candidate(2, "tokio", 0.78, 0.76, 0.72, true),
                IdentifierFlags::default(),
            ),
        ];

        let result = judge.score_candidates(&span2, &candidates2, &ctx());
        let tokio_prob = result
            .iter()
            .find(|o| o.alias_id == Some(2))
            .unwrap()
            .probability;

        // The model should have learned that high acceptance/phonetic/coarse => accept
        // so tokio (with similar scores) should also get a high probability
        assert!(
            tokio_prob > 0.5,
            "learning should transfer: tokio_prob={tokio_prob}"
        );
    }

    #[test]
    fn teach_choice_actually_changes_weights() {
        let mut judge = OnlineJudge::default();
        let weights_before = judge.weights();
        let span = span("req west");
        let candidates = vec![
            (
                candidate(7, "reqwest", 0.75, 0.74, 0.70, true),
                IdentifierFlags::default(),
            ),
            (
                candidate(8, "request", 0.40, 0.38, 0.35, false),
                IdentifierFlags::default(),
            ),
        ];

        for _ in 0..10 {
            judge.teach_choice(&span, &candidates, Some(7), &ctx());
        }

        let weights_after = judge.weights();
        assert_eq!(judge.update_count(), 10);

        let max_diff = weights_before
            .iter()
            .zip(&weights_after)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff > 0.01,
            "weights should change meaningfully after 10 teaches, max_diff={max_diff}"
        );
    }

    #[test]
    fn context_features_are_generated() {
        let ctx = SpanContext {
            left_tokens: vec!["the".into()],
            right_tokens: vec!["crate".into()],
            code_like: false,
            prose_like: true,
            list_like: false,
            sentence_start: false,
            app_id: None,
        };
        let features = context_features(&ctx, "serde");
        // Should have: L1=the, TERM=serde|L1=the, R1=crate, TERM=serde|R1=crate,
        //              TERM=serde, CTX=prose = 6 features
        assert!(
            features.len() >= 5,
            "should generate context features, got {}: {features:?}",
            features.len()
        );
        // All should be in the sparse range
        assert!(
            features.iter().all(|f| f.index >= SPARSE_OFFSET),
            "context features should be in sparse range"
        );
    }

    #[test]
    fn extract_context_from_transcript() {
        let ctx = extract_span_context("I'm talking about the serde crate today", 22, 27);
        assert_eq!(ctx.left_tokens, vec!["about", "the"]);
        assert_eq!(ctx.right_tokens, vec!["crate", "today"]);
        assert!(!ctx.sentence_start);
    }
}
