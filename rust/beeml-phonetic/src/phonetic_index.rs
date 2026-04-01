use std::collections::{BTreeMap, HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::phonetic_lexicon::{AliasSource, LexiconAlias};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IndexView {
    RawIpa2,
    RawIpa3,
    ReducedIpa2,
    ReducedIpa3,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Posting {
    pub alias_id: u32,
}

#[derive(Debug, Clone)]
pub struct PhoneticIndex {
    pub aliases: Vec<LexiconAlias>,
    pub postings: HashMap<(IndexView, String), Vec<Posting>>,
    pub by_phone_len: BTreeMap<u8, Vec<u32>>,
    pub by_token_count: HashMap<u8, Vec<u32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalQuery {
    pub text: String,
    pub ipa_tokens: Vec<String>,
    pub reduced_ipa_tokens: Vec<String>,
    pub token_count: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalCandidate {
    pub alias_id: u32,
    pub term: String,
    pub alias_text: String,
    pub alias_source: AliasSource,
    pub matched_view: IndexView,
    pub qgram_overlap: u16,
    pub token_count_match: bool,
    pub phone_count_delta: i16,
    pub coarse_score: f32,
}

#[derive(Debug, Default, Clone)]
struct CandidateAccumulator {
    overlap: u16,
    matched_view: Option<IndexView>,
    best_score: f32,
}

pub fn build_index(aliases: Vec<LexiconAlias>) -> PhoneticIndex {
    let mut postings: HashMap<(IndexView, String), Vec<Posting>> = HashMap::new();
    let mut by_phone_len: BTreeMap<u8, Vec<u32>> = BTreeMap::new();
    let mut by_token_count: HashMap<u8, Vec<u32>> = HashMap::new();

    for alias in &aliases {
        by_phone_len
            .entry(alias.phone_count)
            .or_default()
            .push(alias.alias_id);
        by_token_count
            .entry(alias.token_count)
            .or_default()
            .push(alias.alias_id);

        for (view, grams) in [
            (IndexView::RawIpa2, qgrams(&alias.ipa_tokens, 2)),
            (IndexView::RawIpa3, qgrams(&alias.ipa_tokens, 3)),
            (IndexView::ReducedIpa2, qgrams(&alias.reduced_ipa_tokens, 2)),
            (IndexView::ReducedIpa3, qgrams(&alias.reduced_ipa_tokens, 3)),
        ] {
            let mut seen_grams = HashSet::new();
            for gram in grams {
                if !seen_grams.insert(gram.clone()) {
                    continue;
                }
                postings
                    .entry((view, gram))
                    .or_default()
                    .push(Posting { alias_id: alias.alias_id });
            }
        }
    }

    PhoneticIndex {
        aliases,
        postings,
        by_phone_len,
        by_token_count,
    }
}

pub fn query_index(
    index: &PhoneticIndex,
    query: &RetrievalQuery,
    limit: usize,
) -> Vec<RetrievalCandidate> {
    let query_phone_count = query.ipa_tokens.len().min(u8::MAX as usize) as u8;
    let mut accum: HashMap<u32, CandidateAccumulator> = HashMap::new();

    for (view, grams, alias_tokens) in [
        (IndexView::RawIpa2, qgrams(&query.ipa_tokens, 2), &query.ipa_tokens),
        (IndexView::RawIpa3, qgrams(&query.ipa_tokens, 3), &query.ipa_tokens),
        (
            IndexView::ReducedIpa2,
            qgrams(&query.reduced_ipa_tokens, 2),
            &query.reduced_ipa_tokens,
        ),
        (
            IndexView::ReducedIpa3,
            qgrams(&query.reduced_ipa_tokens, 3),
            &query.reduced_ipa_tokens,
        ),
    ] {
        let mut unique_query_grams = HashSet::new();
        for gram in grams {
            if !unique_query_grams.insert(gram.clone()) {
                continue;
            }
            let Some(postings) = index.postings.get(&(view, gram.clone())) else {
                continue;
            };
            for posting in postings {
                let alias = &index.aliases[posting.alias_id as usize];
                if !phone_count_compatible(query_phone_count, alias.phone_count) {
                    continue;
                }
                let overlap = accum.entry(posting.alias_id).or_default();
                overlap.overlap = overlap.overlap.saturating_add(1);
                let denom = qgrams(alias_tokens, q_from_view(view)).len().max(1) as f32;
                let score = overlap.overlap as f32 / denom;
                if score >= overlap.best_score {
                    overlap.best_score = score;
                    overlap.matched_view = Some(view);
                }
            }
        }
    }

    let mut out = accum
        .into_iter()
        .filter_map(|(alias_id, hit)| {
            let alias = index.aliases.get(alias_id as usize)?;
            let matched_view = hit.matched_view?;
            Some(RetrievalCandidate {
                alias_id,
                term: alias.term.clone(),
                alias_text: alias.alias_text.clone(),
                alias_source: alias.alias_source,
                matched_view,
                qgram_overlap: hit.overlap,
                token_count_match: alias.token_count == query.token_count,
                phone_count_delta: alias.phone_count as i16 - query_phone_count as i16,
                coarse_score: coarse_score(hit.best_score, alias, query_phone_count, query.token_count),
            })
        })
        .collect::<Vec<_>>();

    out.sort_by(|a, b| {
        b.coarse_score
            .total_cmp(&a.coarse_score)
            .then_with(|| b.qgram_overlap.cmp(&a.qgram_overlap))
            .then_with(|| a.phone_count_delta.abs().cmp(&b.phone_count_delta.abs()))
    });
    out.truncate(limit);
    out
}

pub fn with_boundaries(tokens: &[String]) -> Vec<String> {
    let mut out = Vec::with_capacity(tokens.len() + 2);
    out.push("^".to_string());
    out.extend(tokens.iter().cloned());
    out.push("$".to_string());
    out
}

pub fn qgrams(tokens: &[String], q: usize) -> Vec<String> {
    if tokens.is_empty() || q == 0 {
        return Vec::new();
    }
    let bounded = with_boundaries(tokens);
    if bounded.len() < q {
        return vec![bounded.join(" ")];
    }
    bounded
        .windows(q)
        .map(|window| window.join(" "))
        .collect()
}

fn q_from_view(view: IndexView) -> usize {
    match view {
        IndexView::RawIpa2 | IndexView::ReducedIpa2 => 2,
        IndexView::RawIpa3 | IndexView::ReducedIpa3 => 3,
    }
}

fn phone_count_compatible(query: u8, candidate: u8) -> bool {
    let delta = query.abs_diff(candidate);
    delta <= 3 || delta * 2 <= query.max(candidate)
}

fn coarse_score(
    normalized_overlap: f32,
    alias: &LexiconAlias,
    query_phone_count: u8,
    query_token_count: u8,
) -> f32 {
    let token_bonus = if alias.token_count == query_token_count {
        0.15
    } else if alias.token_count.abs_diff(query_token_count) <= 1 {
        0.05
    } else {
        -0.08
    };
    let phone_bonus = if alias.phone_count.abs_diff(query_phone_count) <= 1 {
        0.12
    } else if alias.phone_count.abs_diff(query_phone_count) <= 3 {
        0.04
    } else {
        -0.10
    };
    normalized_overlap + token_bonus + phone_bonus
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::types::{ReviewedConfusionSurfaceRow, VocabRow};
    use crate::phonetic_lexicon::build_phonetic_lexicon;

    fn row_with_reviewed_ipa(term: &str, spoken: &str, reviewed_ipa: &str) -> VocabRow {
        VocabRow {
            id: 1,
            term: term.to_string(),
            spoken_auto: spoken.to_string(),
            spoken_override: Some(spoken.to_string()),
            reviewed_ipa: Some(reviewed_ipa.to_string()),
            reviewed: true,
            description: None,
        }
    }

    fn confusion(term: &str, surface_form: &str, reviewed_ipa: &str) -> ReviewedConfusionSurfaceRow {
        ReviewedConfusionSurfaceRow {
            id: 1,
            term: term.to_string(),
            surface_form: surface_form.to_string(),
            reviewed_ipa: Some(reviewed_ipa.to_string()),
            status: "reviewed".to_string(),
            source: Some("test".to_string()),
            created_at: String::new(),
            updated_at: String::new(),
        }
    }

    #[test]
    fn builds_boundary_aware_qgrams() {
        let grams = qgrams(&["eɪ".to_string(), "ɑ".to_string(), "tʃ".to_string()], 2);
        assert_eq!(
            grams,
            vec![
                "^ eɪ".to_string(),
                "eɪ ɑ".to_string(),
                "ɑ tʃ".to_string(),
                "tʃ $".to_string()
            ]
        );
    }

    #[test]
    fn query_index_returns_aarch64_shortlist() {
        let vocab = vec![
            row_with_reviewed_ipa("AArch64", "A arch sixty-four", "eɪ ɑː tʃ s ɪ k s t ɪ f ə"),
            row_with_reviewed_ipa("SQLite", "sequel light", "s i k w l aɪ t"),
            row_with_reviewed_ipa("repr", "reppur", "r e p p u r"),
        ];
        let confusion_forms = HashMap::from([(
            "AArch64".to_string(),
            vec![confusion("AArch64", "ARC sixty four", "ɑːɹ s ɪ k s t i f ɔ ɹ")],
        )]);
        let aliases = build_phonetic_lexicon(&vocab, &confusion_forms);
        let index = build_index(aliases);

        let query = RetrievalQuery {
            text: "ARC sixty four".to_string(),
            ipa_tokens: crate::prototype::parse_reviewed_ipa("ɑːɹ s ɪ k s t i f ɔ ɹ"),
            reduced_ipa_tokens: crate::phonetic_lexicon::reduce_ipa_tokens(
                &crate::prototype::parse_reviewed_ipa("ɑːɹ s ɪ k s t i f ɔ ɹ"),
            ),
            token_count: 3,
        };

        let shortlist = query_index(&index, &query, 5);
        assert!(!shortlist.is_empty(), "{shortlist:#?}");
        assert_eq!(shortlist[0].term, "AArch64", "{shortlist:#?}");
    }

    #[test]
    fn query_index_prefers_matching_token_count() {
        let aliases = vec![
            LexiconAlias {
                alias_id: 0,
                term: "reqwest".to_string(),
                alias_text: "request".to_string(),
                alias_source: AliasSource::Spoken,
                ipa_tokens: crate::prototype::parse_reviewed_ipa("r ɪ k w ɛ s t"),
                reduced_ipa_tokens: crate::phonetic_lexicon::reduce_ipa_tokens(
                    &crate::prototype::parse_reviewed_ipa("r ɪ k w ɛ s t"),
                ),
                token_count: 1,
                phone_count: 7,
                identifier_flags: Default::default(),
            },
            LexiconAlias {
                alias_id: 1,
                term: "ripgrep".to_string(),
                alias_text: "rip grep".to_string(),
                alias_source: AliasSource::Spoken,
                ipa_tokens: crate::prototype::parse_reviewed_ipa("r ɪ p ɡ ɹ ɛ p"),
                reduced_ipa_tokens: crate::phonetic_lexicon::reduce_ipa_tokens(
                    &crate::prototype::parse_reviewed_ipa("r ɪ p ɡ ɹ ɛ p"),
                ),
                token_count: 2,
                phone_count: 7,
                identifier_flags: Default::default(),
            },
        ];
        let index = build_index(aliases);
        let query = RetrievalQuery {
            text: "request".to_string(),
            ipa_tokens: crate::prototype::parse_reviewed_ipa("r ɪ k w ɛ s t"),
            reduced_ipa_tokens: crate::phonetic_lexicon::reduce_ipa_tokens(
                &crate::prototype::parse_reviewed_ipa("r ɪ k w ɛ s t"),
            ),
            token_count: 1,
        };

        let shortlist = query_index(&index, &query, 5);
        assert_eq!(shortlist[0].term, "reqwest", "{shortlist:#?}");
    }
}
