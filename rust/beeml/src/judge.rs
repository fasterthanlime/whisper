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
    "keep_original_bias",
    "keep_original_span_low_content",
    "keep_original_best_acceptance",
    "keep_original_best_phonetic",
    "keep_original_candidate_count",
];

const BASE_WEIGHTS: &[f64] = &[
    0.0, 1.8, 1.2, 0.8, 0.6, 0.5, 0.3, 0.3, 0.15, 0.12, 0.05, 0.15, 0.25, 0.08,
    0.06, -0.04, 0.04, 0.04, 0.02, 0.02, 0.02, 0.10, 0.10, 0.20, 0.30, 0.02, 0.02,
    0.55, 0.40, -0.90, -0.65, 0.05,
];

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
    alias_id: Option<u32>,
    term: String,
    is_keep_original: bool,
    features: Vec<f64>,
}

impl Default for OnlineJudge {
    fn default() -> Self {
        Self {
            model: seed_model(),
            update_count: 0,
        }
    }
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
        score_examples(self.model.as_ref(), examples)
    }

    pub fn teach_choice(
        &mut self,
        span: &TranscriptSpan,
        candidates: &[(CandidateFeatureRow, IdentifierFlags)],
        chosen_alias_id: Option<u32>,
    ) -> Vec<JudgeOption> {
        let examples = build_examples(span, candidates);
        if examples.is_empty() {
            return Vec::new();
        }
        if !examples.iter().any(|example| example.alias_id == chosen_alias_id) {
            return score_examples(self.model.as_ref(), examples);
        }

        let records = records_for_examples(&examples);
        let targets = Array1::from_vec(
            examples
                .iter()
                .map(|example| example.alias_id == chosen_alias_id)
                .collect::<Vec<_>>(),
        );
        let dataset = Dataset::new(records, targets);
        let params = Ftrl::<f64>::params()
            .alpha(0.25)
            .beta(1.0)
            .l1_ratio(0.001)
            .l2_ratio(0.001);

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
        }

        score_examples(self.model.as_ref(), examples)
    }
}

fn seed_model() -> Option<Ftrl<f64>> {
    let records = Array2::from_shape_vec(
        (6, FEATURE_NAMES.len()),
        vec![
            1.0, 0.88, 0.84, 0.81, 0.82, 0.79, 0.06, 0.76, 0.66, 0.70, 0.74, 1.0, 0.90, 0.0,
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.50, 0.60, 0.0, 0.0, 0.0,
            0.0, 0.0,
            1.0, 0.15, 0.20, 0.18, 0.22, 0.20, 0.00, 0.19, 0.12, 0.10, 0.14, 0.0, 0.40, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.30, 0.0, 0.0, 0.0,
            0.0, 0.0,
            1.0, 0.55, 0.52, 0.51, 0.52, 0.50, 0.01, 0.48, 0.33, 0.38, 0.40, 1.0, 0.80, 1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.25, 0.35, 0.0, 0.0, 0.0,
            0.0, 0.0,
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 1.0, 1.0, 0.20, 0.25,
            0.25,
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 1.0, 0.0, 0.85, 0.82,
            0.25,
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 1.0, 0.0, 0.40, 0.35,
            0.50,
        ],
    )
    .ok()?;
    let targets = Array1::from_vec(vec![true, false, true, true, false, false]);
    let dataset = Dataset::new(records, targets);
    let params = Ftrl::<f64>::params()
        .alpha(0.25)
        .beta(1.0)
        .l1_ratio(0.001)
        .l2_ratio(0.001);
    params.fit_with(None, &dataset).ok()
}

fn build_examples(
    span: &TranscriptSpan,
    candidates: &[(CandidateFeatureRow, IdentifierFlags)],
) -> Vec<JudgeExample> {
    let span_token_count = (span.token_end - span.token_start) as f64;
    let span_phone_count = span.ipa_tokens.len() as f64;
    let span_low_content = is_low_content_span(&span.text) as u8 as f64;
    let best_acceptance = candidates
        .iter()
        .map(|(candidate, _)| candidate.acceptance_score as f64)
        .fold(0.0, f64::max);
    let best_phonetic = candidates
        .iter()
        .map(|(candidate, _)| candidate.phonetic_score as f64)
        .fold(0.0, f64::max);
    let candidate_count = candidates.len() as f64;

    let mut examples = candidates
        .iter()
        .map(|(candidate, flags)| JudgeExample {
            alias_id: Some(candidate.alias_id),
            term: candidate.term.clone(),
            is_keep_original: false,
            features: vec![
                1.0,
                candidate.acceptance_score as f64,
                candidate.phonetic_score as f64,
                candidate.coarse_score as f64,
                candidate.token_score as f64,
                candidate.feature_score as f64,
                candidate.feature_bonus as f64,
                candidate.best_view_score as f64,
                candidate.cross_view_support as f64 / 6.0,
                candidate.qgram_overlap as f64 / 10.0,
                candidate.total_qgram_overlap as f64 / 20.0,
                candidate.token_count_match as u8 as f64,
                1.0 / (1.0 + candidate.phone_count_delta.abs() as f64),
                (candidate.alias_source == AliasSource::Spoken) as u8 as f64,
                (candidate.alias_source == AliasSource::Identifier) as u8 as f64,
                (candidate.alias_source == AliasSource::Confusion) as u8 as f64,
                flags.acronym_like as u8 as f64,
                flags.has_digits as u8 as f64,
                flags.snake_like as u8 as f64,
                flags.camel_like as u8 as f64,
                flags.symbol_like as u8 as f64,
                candidate.short_guard_passed as u8 as f64,
                candidate.low_content_guard_passed as u8 as f64,
                candidate.acceptance_floor_passed as u8 as f64,
                candidate.verified as u8 as f64,
                span_token_count / 4.0,
                span_phone_count / 12.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        })
        .collect::<Vec<_>>();

    examples.push(JudgeExample {
        alias_id: None,
        term: "keep_original".to_string(),
        is_keep_original: true,
        features: vec![
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            span_token_count / 4.0,
            span_phone_count / 12.0,
            1.0,
            span_low_content,
            best_acceptance,
            best_phonetic,
            candidate_count / 8.0,
        ],
    });

    examples
}

fn records_for_examples(examples: &[JudgeExample]) -> Array2<f64> {
    Array2::from_shape_vec(
        (examples.len(), FEATURE_NAMES.len()),
        examples
            .iter()
            .flat_map(|example| example.features.iter().copied())
            .collect::<Vec<_>>(),
    )
    .expect("judge examples should have stable width")
}

fn score_examples(model: Option<&Ftrl<f64>>, examples: Vec<JudgeExample>) -> Vec<JudgeOption> {
    let live_weights = model
        .map(|model| model.get_weights().iter().copied().collect::<Vec<_>>())
        .unwrap_or_else(|| BASE_WEIGHTS.to_vec());

    let mut options = examples
        .into_iter()
        .map(|example| {
            let score = dot(&live_weights, &example.features) as f32;
            let probability = sigmoid(score as f64) as f32;
            JudgeOption {
                alias_id: example.alias_id,
                term: example.term,
                is_keep_original: example.is_keep_original,
                score,
                probability,
                chosen: false,
            }
        })
        .collect::<Vec<_>>();

    let best_index = options
        .iter()
        .enumerate()
        .max_by(|(_, lhs), (_, rhs)| {
            lhs.probability
                .total_cmp(&rhs.probability)
                .then_with(|| lhs.score.total_cmp(&rhs.score))
        })
        .map(|(index, _)| index);
    if let Some(best_index) = best_index {
        options[best_index].chosen = true;
    }
    options.sort_by(|lhs, rhs| {
        rhs.probability
            .total_cmp(&lhs.probability)
            .then_with(|| rhs.score.total_cmp(&lhs.score))
    });
    options
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

        let after = judge.teach_choice(&span, &candidates, Some(7));
        let after_reqwest = after
            .iter()
            .find(|option| option.alias_id == Some(7))
            .expect("reqwest option should exist")
            .probability;
        let chosen = after.iter().find(|option| option.chosen).expect("winner exists");

        assert_eq!(judge.update_count(), 1);
        assert!(after_reqwest >= before_reqwest);
        assert_eq!(chosen.alias_id, Some(7));
    }
}
