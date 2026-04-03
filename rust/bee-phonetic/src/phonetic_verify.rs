use serde::{Deserialize, Serialize};

use crate::phonetic_index::{PhoneticIndex, RetrievalCandidate};
use crate::phonetic_lexicon::AliasSource;
use crate::region_proposal::TranscriptSpan;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiedCandidate {
    pub alias_id: u32,
    pub term: String,
    pub alias_text: String,
    pub alias_source: AliasSource,
    pub matched_view: crate::phonetic_index::IndexView,
    pub qgram_overlap: u16,
    pub total_qgram_overlap: u16,
    pub best_view_score: f32,
    pub cross_view_support: u8,
    pub phone_count_delta: i16,
    pub token_bonus: f32,
    pub phone_bonus: f32,
    pub extra_length_penalty: f32,
    pub coarse_score: f32,
    pub phonetic_score: f32,
}

pub fn verify_shortlist(
    span: &TranscriptSpan,
    shortlist: &[RetrievalCandidate],
    index: &PhoneticIndex,
    limit: usize,
) -> Vec<VerifiedCandidate> {
    let mut out = shortlist
        .iter()
        .filter_map(|candidate| {
            let alias = index.aliases.get(candidate.alias_id as usize)?;
            let phonetic_score =
                crate::prototype::phoneme_similarity(&span.ipa_tokens, &alias.ipa_tokens)?;
            Some(VerifiedCandidate {
                alias_id: candidate.alias_id,
                term: candidate.term.clone(),
                alias_text: candidate.alias_text.clone(),
                alias_source: candidate.alias_source,
                matched_view: candidate.matched_view,
                qgram_overlap: candidate.qgram_overlap,
                total_qgram_overlap: candidate.total_qgram_overlap,
                best_view_score: candidate.best_view_score,
                cross_view_support: candidate.cross_view_support,
                phone_count_delta: candidate.phone_count_delta,
                token_bonus: candidate.token_bonus,
                phone_bonus: candidate.phone_bonus,
                extra_length_penalty: candidate.extra_length_penalty,
                coarse_score: candidate.coarse_score,
                phonetic_score,
            })
        })
        .collect::<Vec<_>>();

    out.sort_by(|a, b| {
        b.phonetic_score
            .total_cmp(&a.phonetic_score)
            .then_with(|| b.coarse_score.total_cmp(&a.coarse_score))
            .then_with(|| a.phone_count_delta.abs().cmp(&b.phone_count_delta.abs()))
    });
    out.truncate(limit);
    out
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
        };
        let shortlist = query_index(
            &index,
            &RetrievalQuery {
                text: span.text.clone(),
                ipa_tokens: span.ipa_tokens.clone(),
                reduced_ipa_tokens: span.reduced_ipa_tokens.clone(),
                token_count: 3,
            },
            5,
        );
        let verified = verify_shortlist(&span, &shortlist, &index, 5);
        assert_eq!(verified.first().map(|c| c.term.as_str()), Some("AArch64"));
    }
}
