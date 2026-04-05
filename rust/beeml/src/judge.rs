use bee_phonetic::{AliasSource, CandidateFeatureRow, IdentifierFlags, TranscriptSpan};
use bee_phonetic::{SentenceWordToken, sentence_word_tokens};
use linfa::traits::FitWith;
use linfa::Dataset;
use linfa_ftrl::Ftrl;
use ndarray::{Array1, Array2};

const LOW_CONTENT: &[&str] = &[
    "a", "an", "and", "the", "then", "that", "this", "these", "those", "if", "you", "we",
    "they", "he", "she", "it", "i", "me", "my", "your", "our", "their", "him", "her", "them",
    "about", "not", "sure", "what", "yeah", "well", "oh", "hmm", "uh", "um", "want", "some",
    "there", "here",
];

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
];

const NUM_FEATURES: usize = 28;

const BASE_WEIGHTS: &[f64] = &[
    0.0,   // bias
    1.8,   // acceptance_score
    1.2,   // phonetic_score
    0.8,   // coarse_score
    0.6,   // token_score
    0.5,   // feature_score
    0.3,   // feature_bonus
    0.3,   // best_view_score
    0.15,  // cross_view_support
    0.12,  // qgram_overlap
    0.05,  // total_qgram_overlap
    0.15,  // token_count_match
    0.25,  // phone_closeness
    0.08,  // alias_source_spoken
    0.06,  // alias_source_identifier
    -0.04, // alias_source_confusion
    0.04,  // identifier_acronym
    0.04,  // identifier_digits
    0.02,  // identifier_snake
    0.02,  // identifier_camel
    0.02,  // identifier_symbol
    0.10,  // short_guard_passed
    0.10,  // low_content_guard_passed
    0.20,  // acceptance_floor_passed
    0.30,  // verified
    0.02,  // span_token_count
    0.02,  // span_phone_count
    -0.30, // span_low_content (penalty: low-content spans rarely need replacement)
];

/// Threshold for the judge to accept a candidate replacement.
/// If no candidate's probability exceeds this, keep original.
const ACCEPT_THRESHOLD: f32 = 0.5;

#[derive(Clone, Debug)]
pub struct OnlineJudge {
    model: Option<Ftrl<f64>>,
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
    features: Vec<f64>,
}

impl Default for OnlineJudge {
    fn default() -> Self {
        let judge = Self {
            model: seed_model(),
            update_count: 0,
        };
        tracing::info!(
            weights = ?judge.weights().iter().map(|w| format!("{w:.3}")).collect::<Vec<_>>(),
            "judge initialized with seed weights"
        );
        judge
    }
}

/// Seed the FTRL model by training on synthetic examples that span the
/// quality range. This teaches the model: high scores → replace, low scores → keep.
fn seed_model() -> Option<Ftrl<f64>> {
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

    let mut all_features = Vec::new();
    let mut all_targets = Vec::new();
    for &(accept, phonetic, coarse, token, feature, verified, target) in levels {
        let features = vec![
            1.0,                    // bias
            accept,                 // acceptance_score
            phonetic,               // phonetic_score
            coarse,                 // coarse_score
            token,                  // token_score
            feature,                // feature_score
            (feature - token).max(0.0), // feature_bonus
            coarse * 0.9,           // best_view_score
            0.33,                   // cross_view_support
            coarse * 0.5,           // qgram_overlap
            coarse * 0.8,           // total_qgram_overlap
            if accept > 0.5 { 1.0 } else { 0.0 }, // token_count_match
            if accept > 0.5 { 0.80 } else { 0.40 }, // phone_closeness
            0.0,                    // alias_source_spoken
            0.0,                    // alias_source_identifier
            0.0,                    // alias_source_confusion
            0.0, 0.0, 0.0, 0.0, 0.0, // identifier flags
            if verified > 0.5 { 1.0 } else { 0.0 }, // short_guard_passed
            1.0,                    // low_content_guard_passed
            if accept > 0.35 { 1.0 } else { 0.0 }, // acceptance_floor_passed
            verified,               // verified
            0.25,                   // span_token_count
            0.40,                   // span_phone_count
            0.0,                    // span_low_content
        ];
        debug_assert_eq!(features.len(), NUM_FEATURES);
        all_features.extend(features);
        all_targets.push(target);
    }

    let n = all_targets.len();
    let records = Array2::from_shape_vec((n, NUM_FEATURES), all_features).ok()?;
    let targets = Array1::from_vec(all_targets);
    let dataset = Dataset::new(records, targets);
    let params = Ftrl::<f64>::params()
        .alpha(1.0)
        .beta(1.0)
        .l1_ratio(0.0001)
        .l2_ratio(0.001);

    // Train multiple epochs so the model actually converges
    let mut model: Option<Ftrl<f64>> = None;
    for _ in 0..20 {
        let prior = model.clone();
        model = match params.fit_with(model, &dataset) {
            Ok(m) => Some(m),
            Err(_) => prior,
        };
    }
    model
}

impl OnlineJudge {
    pub fn feature_names(&self) -> Vec<String> {
        FEATURE_NAMES.iter().map(|name| (*name).to_string()).collect()
    }

    pub fn weights(&self) -> Vec<f32> {
        if let Some(model) = &self.model {
            model.get_weights().iter().map(|weight| *weight as f32).collect()
        } else {
            BASE_WEIGHTS.iter().map(|weight| *weight as f32).collect()
        }
    }

    pub fn update_count(&self) -> u32 {
        self.update_count
    }

    pub fn score_candidates(
        &self,
        span: &TranscriptSpan,
        candidates: &[(CandidateFeatureRow, IdentifierFlags)],
    ) -> Vec<JudgeOption> {
        let examples = build_examples(span, candidates);
        score_examples(self.model.as_ref(), &examples)
    }

    pub fn teach_choice(
        &mut self,
        span: &TranscriptSpan,
        candidates: &[(CandidateFeatureRow, IdentifierFlags)],
        chosen_alias_id: Option<u32>,
    ) -> Vec<JudgeOption> {
        let examples = build_examples(span, candidates);
        if examples.is_empty() {
            return vec![keep_original_option()];
        }

        // chosen_alias_id == None means "keep original" => all candidates are false
        let targets = Array1::from_vec(
            examples
                .iter()
                .map(|example| Some(example.alias_id) == chosen_alias_id)
                .collect::<Vec<_>>(),
        );
        let records = records_for_examples(&examples);
        let dataset = Dataset::new(records, targets);
        let params = Ftrl::<f64>::params()
            .alpha(0.5)
            .beta(1.0)
            .l1_ratio(0.001)
            .l2_ratio(0.01);

        let mut live_model = self.model.take();
        for _ in 0..4 {
            let prior_model = live_model.clone();
            live_model = match params.fit_with(live_model, &dataset) {
                Ok(model) => Some(model),
                Err(_) => prior_model,
            };
        }
        if let Some(model) = live_model {
            self.model = Some(model);
            self.update_count += 1;
            let w = self.weights();
            tracing::debug!(
                update_count = self.update_count,
                chosen = ?chosen_alias_id,
                num_candidates = examples.len(),
                w_bias = w[0],
                w_accept = w[1],
                w_phonetic = w[2],
                w_coarse = w[3],
                w_verified = w[24],
                "judge taught"
            );
        }

        score_examples(self.model.as_ref(), &examples)
    }
}

fn build_examples(
    span: &TranscriptSpan,
    candidates: &[(CandidateFeatureRow, IdentifierFlags)],
) -> Vec<JudgeExample> {
    let span_token_count = (span.token_end - span.token_start) as f64;
    let span_phone_count = span.ipa_tokens.len() as f64;
    let span_low_content = is_low_content_span(&span.text) as u8 as f64;

    candidates
        .iter()
        .map(|(candidate, flags)| {
            let features = vec![
                1.0,                                                           // bias
                candidate.acceptance_score as f64,                             // acceptance_score
                candidate.phonetic_score as f64,                               // phonetic_score
                candidate.coarse_score as f64,                                 // coarse_score
                candidate.token_score as f64,                                  // token_score
                candidate.feature_score as f64,                                // feature_score
                candidate.feature_bonus as f64,                                // feature_bonus
                candidate.best_view_score as f64,                              // best_view_score
                candidate.cross_view_support as f64 / 6.0,                    // cross_view_support
                candidate.qgram_overlap as f64 / 10.0,                        // qgram_overlap
                candidate.total_qgram_overlap as f64 / 20.0,                  // total_qgram_overlap
                candidate.token_count_match as u8 as f64,                      // token_count_match
                1.0 / (1.0 + candidate.phone_count_delta.abs() as f64),        // phone_closeness
                (candidate.alias_source == AliasSource::Spoken) as u8 as f64,  // alias_source_spoken
                (candidate.alias_source == AliasSource::Identifier) as u8 as f64, // alias_source_identifier
                (candidate.alias_source == AliasSource::Confusion) as u8 as f64,  // alias_source_confusion
                flags.acronym_like as u8 as f64,                               // identifier_acronym
                flags.has_digits as u8 as f64,                                 // identifier_digits
                flags.snake_like as u8 as f64,                                 // identifier_snake
                flags.camel_like as u8 as f64,                                 // identifier_camel
                flags.symbol_like as u8 as f64,                                // identifier_symbol
                candidate.short_guard_passed as u8 as f64,                     // short_guard_passed
                candidate.low_content_guard_passed as u8 as f64,               // low_content_guard_passed
                candidate.acceptance_floor_passed as u8 as f64,                // acceptance_floor_passed
                candidate.verified as u8 as f64,                               // verified
                span_token_count / 4.0,                                        // span_token_count
                span_phone_count / 12.0,                                       // span_phone_count
                span_low_content,                                              // span_low_content
            ];
            debug_assert_eq!(features.len(), NUM_FEATURES);
            JudgeExample {
                alias_id: candidate.alias_id,
                term: candidate.term.clone(),
                features,
            }
        })
        .collect()
}

fn records_for_examples(examples: &[JudgeExample]) -> Array2<f64> {
    Array2::from_shape_vec(
        (examples.len(), NUM_FEATURES),
        examples
            .iter()
            .flat_map(|example| example.features.iter().copied())
            .collect::<Vec<_>>(),
    )
    .expect("judge examples should have stable width")
}

fn score_examples(model: Option<&Ftrl<f64>>, examples: &[JudgeExample]) -> Vec<JudgeOption> {
    let live_weights = model
        .map(|model| model.get_weights().iter().copied().collect::<Vec<_>>())
        .unwrap_or_else(|| BASE_WEIGHTS.to_vec());

    let mut options: Vec<JudgeOption> = examples
        .iter()
        .map(|example| {
            let score = dot(&live_weights, &example.features) as f32;
            let probability = sigmoid(score as f64) as f32;
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

fn dot(weights: &[f64], features: &[f64]) -> f64 {
    weights.iter().zip(features).map(|(w, x)| w * x).sum()
}

fn sigmoid(value: f64) -> f64 {
    if value < 0.0 {
        let exp = value.exp();
        exp / (1.0 + exp)
    } else {
        1.0 / (1.0 + (-value).exp())
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

        let before = judge.score_candidates(&span, &candidates);
        let before_reqwest = before
            .iter()
            .find(|option| option.alias_id == Some(7))
            .expect("reqwest option should exist")
            .probability;

        // Teach 5 times so the model has enough signal
        for _ in 0..5 {
            judge.teach_choice(&span, &candidates, Some(7));
        }

        let after = judge.score_candidates(&span, &candidates);
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

        let before = judge.score_candidates(&span, &candidates);
        let before_prob = before
            .iter()
            .find(|o| o.alias_id == Some(7))
            .unwrap()
            .probability;

        // Teach keep_original (chosen_alias_id = None) 5 times
        for _ in 0..5 {
            judge.teach_choice(&span, &candidates, None);
        }

        let after = judge.score_candidates(&span, &candidates);
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
            judge.teach_choice(&span1, &candidates1, Some(1));
        }

        // Now score a DIFFERENT case with similar features
        let span2 = span("toe key oh");
        let candidates2 = vec![
            (
                candidate(2, "tokio", 0.78, 0.76, 0.72, true),
                IdentifierFlags::default(),
            ),
        ];

        let result = judge.score_candidates(&span2, &candidates2);
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
            judge.teach_choice(&span, &candidates, Some(7));
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
}
