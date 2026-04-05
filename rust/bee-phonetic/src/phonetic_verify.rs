use serde::{Deserialize, Serialize};

use crate::phonetic_index::{PhoneticIndex, RetrievalCandidate};
use crate::phonetic_lexicon::AliasSource;
use crate::region_proposal::TranscriptSpan;
use crate::word_split::sentence_word_tokens;

pub use crate::feature_view::FeatureEditOp;
pub use crate::prototype::TokenEditOp;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateFeatureRow {
    pub alias_id: u32,
    pub term: String,
    pub alias_text: String,
    pub alias_source: AliasSource,
    pub matched_view: crate::phonetic_index::IndexView,
    pub qgram_overlap: u16,
    pub total_qgram_overlap: u16,
    pub best_view_score: f32,
    pub cross_view_support: u8,
    pub token_count_match: bool,
    pub phone_count_delta: i16,
    pub token_bonus: f32,
    pub phone_bonus: f32,
    pub extra_length_penalty: f32,
    pub structure_bonus: f32,
    pub coarse_score: f32,
    pub token_distance: u16,
    pub token_weighted_distance: f32,
    pub token_boundary_penalty: f32,
    pub token_max_len: u16,
    pub token_score: f32,
    pub token_ops: Vec<TokenEditOp>,
    pub feature_distance: f32,
    pub feature_weighted_distance: f32,
    pub feature_boundary_penalty: f32,
    pub feature_max_len: u16,
    pub feature_score: f32,
    pub feature_ops: Vec<FeatureEditOp>,
    pub feature_bonus: f32,
    pub feature_gate_token_ok: bool,
    pub feature_gate_coarse_ok: bool,
    pub feature_gate_phone_ok: bool,
    pub short_guard_applied: bool,
    pub short_guard_onset_match: bool,
    pub short_guard_passed: bool,
    pub low_content_guard_applied: bool,
    pub low_content_guard_passed: bool,
    pub acceptance_floor_passed: bool,
    pub used_feature_bonus: bool,
    pub phonetic_score: f32,
    pub acceptance_score: f32,
    pub verified: bool,
}

pub type VerifiedCandidate = CandidateFeatureRow;

pub fn score_shortlist(
    span: &TranscriptSpan,
    shortlist: &[RetrievalCandidate],
    index: &PhoneticIndex,
) -> Vec<CandidateFeatureRow> {
    let span_feature_vectors = crate::feature_view::feature_vectors_for_ipa(&span.ipa_tokens);
    let mut out = shortlist
        .iter()
        .filter_map(|candidate| score_candidate(span, candidate, index, &span_feature_vectors))
        .collect::<Vec<_>>();

    out.sort_by(|a, b| {
        b.acceptance_score
            .total_cmp(&a.acceptance_score)
            .then_with(|| b.phonetic_score.total_cmp(&a.phonetic_score))
            .then_with(|| b.coarse_score.total_cmp(&a.coarse_score))
            .then_with(|| b.verified.cmp(&a.verified))
            .then_with(|| a.phone_count_delta.abs().cmp(&b.phone_count_delta.abs()))
    });
    out
}

pub fn verify_shortlist(
    span: &TranscriptSpan,
    shortlist: &[RetrievalCandidate],
    index: &PhoneticIndex,
    limit: usize,
) -> Vec<VerifiedCandidate> {
    let mut out = score_shortlist(span, shortlist, index)
        .iter()
        .filter(|candidate| candidate.verified)
        .cloned()
        .collect::<Vec<_>>();
    out.truncate(limit);
    out
}

fn score_candidate(
    span: &TranscriptSpan,
    candidate: &RetrievalCandidate,
    index: &PhoneticIndex,
    span_feature_vectors: &[Vec<f32>],
) -> Option<CandidateFeatureRow> {
    let alias = index.aliases.get(candidate.alias_id as usize)?;
    let token_details =
        crate::prototype::phoneme_similarity_details(&span.ipa_tokens, &alias.ipa_tokens)?;
    let feature_details = crate::feature_view::feature_similarity_details_with_labels(
        &span.ipa_tokens,
        span_feature_vectors,
        &alias.ipa_tokens,
        index
            .alias_feature_vectors
            .get(candidate.alias_id as usize)?,
        span.ipa_tokens.len().max(alias.ipa_tokens.len()),
    )
    .unwrap_or(crate::feature_view::FeatureSimilarityDetails {
        distance: token_details.distance as f32,
        weighted_distance: token_details.weighted_distance,
        boundary_penalty: token_details.boundary_penalty,
        max_len: token_details.max_len,
        similarity: token_details.similarity,
        ops: Vec::new(),
    });
    // Feature gate flags are kept for the judge's feature vector but no longer
    // control whether feature similarity is computed — it's always computed.
    let feature_gate_token_ok = token_details.similarity >= 0.45;
    let feature_gate_coarse_ok = candidate.coarse_score >= 0.45;
    let feature_gate_phone_ok = candidate.phone_count_delta.abs() <= 2;
    // Phonetic score combines both signals: feature similarity (articulatory) is
    // more nuanced but can be too permissive alone; token similarity (IPA Levenshtein)
    // is cruder but a good sanity check. We average them so both must be reasonable.
    let feature_bonus = (feature_details.similarity - token_details.similarity).max(0.0);
    let phonetic_score = (token_details.similarity + feature_details.similarity) * 0.5;
    let short_guard_applied = (span.token_end - span.token_start) == 1
        && span.ipa_tokens.len() <= 4
        && alias.token_count == 1
        && alias.ipa_tokens.len() <= 5;
    let short_guard_onset_match = span
        .reduced_ipa_tokens
        .first()
        .zip(alias.reduced_ipa_tokens.first())
        .is_some_and(|(lhs, rhs)| lhs == rhs);
    let short_guard_passed = !short_guard_applied
        || token_details.similarity >= 0.75
        || (short_guard_onset_match && feature_details.similarity >= 0.65);
    let low_content_guard_applied = is_low_content_span(&span.text);
    let low_content_guard_passed = !low_content_guard_applied
        || token_details.similarity >= 0.75
        || feature_details.similarity >= 0.85;
    let acceptance_score = phonetic_score;
    let acceptance_floor_passed = acceptance_score >= 0.35
        && phonetic_score >= 0.50
        && !(candidate.coarse_score < 0.20 && phonetic_score < 0.50)
        && !(token_details.similarity < 0.25 && feature_details.similarity < 0.45);
    let verified = low_content_guard_passed && short_guard_passed && acceptance_floor_passed;

    Some(CandidateFeatureRow {
        alias_id: candidate.alias_id,
        term: candidate.term.clone(),
        alias_text: candidate.alias_text.clone(),
        alias_source: candidate.alias_source,
        matched_view: candidate.matched_view,
        qgram_overlap: candidate.qgram_overlap,
        total_qgram_overlap: candidate.total_qgram_overlap,
        best_view_score: candidate.best_view_score,
        cross_view_support: candidate.cross_view_support,
        token_count_match: candidate.token_count_match,
        phone_count_delta: candidate.phone_count_delta,
        token_bonus: candidate.token_bonus,
        phone_bonus: candidate.phone_bonus,
        extra_length_penalty: candidate.extra_length_penalty,
        structure_bonus: candidate.structure_bonus,
        coarse_score: candidate.coarse_score,
        token_distance: token_details.distance as u16,
        token_weighted_distance: token_details.weighted_distance,
        token_boundary_penalty: token_details.boundary_penalty,
        token_max_len: token_details.max_len as u16,
        token_score: token_details.similarity,
        token_ops: token_details.ops,
        feature_distance: feature_details.distance,
        feature_weighted_distance: feature_details.weighted_distance,
        feature_boundary_penalty: feature_details.boundary_penalty,
        feature_max_len: feature_details.max_len as u16,
        feature_score: feature_details.similarity,
        feature_ops: feature_details.ops,
        feature_bonus,
        feature_gate_token_ok,
        feature_gate_coarse_ok,
        feature_gate_phone_ok,
        short_guard_applied,
        short_guard_onset_match,
        short_guard_passed,
        low_content_guard_applied,
        low_content_guard_passed,
        acceptance_floor_passed,
        used_feature_bonus: feature_bonus > 0.0,
        phonetic_score,
        acceptance_score,
        verified,
    })
}

fn is_low_content_span(text: &str) -> bool {
    const LOW_CONTENT: &[&str] = &[
        "a", "an", "and", "the", "then", "that", "this", "these", "those", "if", "you", "we",
        "they", "he", "she", "it", "i", "me", "my", "your", "our", "their", "him", "her", "them",
        "about", "not", "sure", "what", "yeah", "well", "oh", "hmm", "uh", "um", "want", "some",
        "there", "here",
    ];

    let tokens = sentence_word_tokens(text);
    !tokens.is_empty()
        && tokens.iter().all(|token| {
            let lower = token.text.to_ascii_lowercase();
            LOW_CONTENT.iter().any(|entry| *entry == lower)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phonetic_index::{build_index, query_index, RetrievalQuery};
    use crate::phonetic_lexicon::LexiconAlias;
    use crate::word_split::count_sentence_words;

    fn alias(alias_id: u32, term: &str, alias_text: &str, ipa: &str) -> LexiconAlias {
        let ipa_tokens = crate::prototype::parse_reviewed_ipa(ipa);
        LexiconAlias {
            alias_id,
            term: term.to_string(),
            alias_text: alias_text.to_string(),
            alias_source: AliasSource::Canonical,
            reduced_ipa_tokens: crate::phonetic_lexicon::reduce_ipa_tokens(&ipa_tokens),
            feature_tokens: crate::feature_view::feature_tokens_for_ipa(&ipa_tokens),
            ipa_tokens: ipa_tokens.clone(),
            token_count: count_sentence_words(alias_text) as u8,
            phone_count: ipa_tokens.len() as u8,
            identifier_flags: Default::default(),
        }
    }

    #[test]
    fn verify_shortlist_prefers_better_phone_match() {
        let index = build_index(vec![
            alias(0, "AArch64", "AArch64", "eɪ ɑː tʃ s ɪ k s t ɪ f ə"),
            alias(1, "SQLite", "SQLite", "s i k w l aɪ t"),
        ]);
        let span = TranscriptSpan {
            token_start: 0,
            token_end: 3,
            char_start: 0,
            char_end: 14,
            start_sec: None,
            end_sec: None,
            text: "arc sixty four".to_string(),
            ipa_tokens: crate::prototype::parse_reviewed_ipa("ɑːɹ s ɪ k s t i f ɔ ɹ"),
            reduced_ipa_tokens: crate::phonetic_lexicon::reduce_ipa_tokens(
                &crate::prototype::parse_reviewed_ipa("ɑːɹ s ɪ k s t i f ɔ ɹ"),
            ),
            ..Default::default()
        };
        let shortlist = query_index(
            &index,
            &RetrievalQuery {
                text: span.text.clone(),
                ipa_tokens: span.ipa_tokens.clone(),
                reduced_ipa_tokens: span.reduced_ipa_tokens.clone(),
                feature_tokens: crate::feature_view::feature_tokens_for_ipa(&span.ipa_tokens),
                token_count: 3,
            },
            5,
        );
        let verified = verify_shortlist(&span, &shortlist, &index, 5);
        assert_eq!(verified.first().map(|c| c.term.as_str()), Some("AArch64"));
    }

    #[test]
    fn verify_shortlist_rejects_weak_ripgrep_match_for_crap() {
        let index = build_index(vec![alias(0, "ripgrep", "ripgrep", "r ɪ p ɡ ɹ ɛ p")]);
        let span = TranscriptSpan {
            token_start: 0,
            token_end: 1,
            char_start: 0,
            char_end: 5,
            start_sec: None,
            end_sec: None,
            text: "crap".to_string(),
            ipa_tokens: crate::prototype::parse_reviewed_ipa("k ɹ a p"),
            reduced_ipa_tokens: crate::phonetic_lexicon::reduce_ipa_tokens(
                &crate::prototype::parse_reviewed_ipa("k ɹ a p"),
            ),
            ..Default::default()
        };
        let shortlist = query_index(
            &index,
            &RetrievalQuery {
                text: span.text.clone(),
                ipa_tokens: span.ipa_tokens.clone(),
                reduced_ipa_tokens: span.reduced_ipa_tokens.clone(),
                feature_tokens: crate::feature_view::feature_tokens_for_ipa(&span.ipa_tokens),
                token_count: 1,
            },
            5,
        );
        assert!(!shortlist.is_empty(), "{shortlist:#?}");
        let verified = verify_shortlist(&span, &shortlist, &index, 5);
        assert!(verified.is_empty(), "{verified:#?}");
    }

    #[test]
    fn score_shortlist_marks_rejected_candidates() {
        let index = build_index(vec![alias(0, "ripgrep", "ripgrep", "r ɪ p ɡ ɹ ɛ p")]);
        let span = TranscriptSpan {
            token_start: 0,
            token_end: 1,
            char_start: 0,
            char_end: 5,
            start_sec: None,
            end_sec: None,
            text: "crap".to_string(),
            ipa_tokens: crate::prototype::parse_reviewed_ipa("k ɹ a p"),
            reduced_ipa_tokens: crate::phonetic_lexicon::reduce_ipa_tokens(
                &crate::prototype::parse_reviewed_ipa("k ɹ a p"),
            ),
            ..Default::default()
        };
        let shortlist = query_index(
            &index,
            &RetrievalQuery {
                text: span.text.clone(),
                ipa_tokens: span.ipa_tokens.clone(),
                reduced_ipa_tokens: span.reduced_ipa_tokens.clone(),
                feature_tokens: crate::feature_view::feature_tokens_for_ipa(&span.ipa_tokens),
                token_count: 1,
            },
            5,
        );
        let scored = score_shortlist(&span, &shortlist, &index);
        assert_eq!(scored.len(), shortlist.len());
        assert!(
            scored.iter().all(|candidate| !candidate.verified),
            "{scored:#?}"
        );
    }

    #[test]
    fn verify_rejects_low_phonetic_score_despite_structure_bonus() {
        // "with U8" (w ɪ ð j uː eɪ t) vs SQLite (s iː k w l aɪ t):
        // phonetic similarity ~0.28 is garbage. The candidate should not be
        // verified even if structure_bonus inflates acceptance_score above the floor.
        // We bypass index query and feed the candidate directly to score_shortlist.
        let index = build_index(vec![
            alias(0, "SQLite", "SQLite", "s i k w l aɪ t"),
        ]);
        let span = TranscriptSpan {
            token_start: 0,
            token_end: 2,
            char_start: 0,
            char_end: 7,
            start_sec: None,
            end_sec: None,
            text: "with U8".to_string(),
            ipa_tokens: crate::prototype::parse_reviewed_ipa("w ɪ ð j uː eɪ t"),
            reduced_ipa_tokens: crate::phonetic_lexicon::reduce_ipa_tokens(
                &crate::prototype::parse_reviewed_ipa("w ɪ ð j uː eɪ t"),
            ),
            ..Default::default()
        };
        // Simulate a candidate that made the shortlist with a high coarse score
        let fake_shortlist = vec![crate::phonetic_index::RetrievalCandidate {
            alias_id: 0,
            term: "SQLite".to_string(),
            alias_text: "SQLite".to_string(),
            alias_source: AliasSource::Canonical,
            matched_view: crate::phonetic_index::IndexView::RawIpa2,
            qgram_overlap: 4,
            total_qgram_overlap: 6,
            best_view_score: 0.5,
            cross_view_support: 2,
            token_count_match: true,
            phone_count_delta: 0,
            token_bonus: 0.15,
            phone_bonus: 0.15,
            extra_length_penalty: 0.0,
            structure_bonus: 0.10,
            coarse_score: 0.63,
        }];
        let scored = score_shortlist(&span, &fake_shortlist, &index);
        assert!(
            scored.iter().all(|c| !c.verified),
            "SQLite with phonetic={:.2} should not verify for 'with U8': {scored:#?}",
            scored.first().map(|c| c.phonetic_score).unwrap_or(0.0)
        );
    }
}
