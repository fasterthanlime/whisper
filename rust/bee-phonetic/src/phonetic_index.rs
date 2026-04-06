use std::collections::{BTreeMap, HashMap, HashSet};

use facet::Facet;

use crate::phonetic_lexicon::{
    derive_identifier_flags, is_identifier_like, looks_like_name, AliasSource, LexiconAlias,
};
use crate::word_split::sentence_word_tokens;

pub use bee_types::IndexView;

#[derive(Debug, Clone, Facet, PartialEq, Eq)]
pub struct Posting {
    pub alias_id: u32,
}

#[derive(Debug, Clone)]
pub struct PhoneticIndex {
    pub aliases: Vec<LexiconAlias>,
    pub alias_feature_vectors: Vec<Vec<Vec<f32>>>,
    pub postings: HashMap<(IndexView, String), Vec<Posting>>,
    pub short_query_postings: HashMap<String, Vec<Posting>>,
    pub by_phone_len: BTreeMap<u8, Vec<u32>>,
    pub by_token_count: HashMap<u8, Vec<u32>>,
}

#[derive(Debug, Clone, Facet)]
pub struct RetrievalQuery {
    pub text: String,
    pub ipa_tokens: Vec<String>,
    pub reduced_ipa_tokens: Vec<String>,
    pub feature_tokens: Vec<String>,
    pub token_count: u8,
}

#[derive(Debug, Clone, Facet)]
pub struct RetrievalCandidate {
    pub alias_id: u32,
    pub term: String,
    pub alias_text: String,
    pub alias_source: AliasSource,
    pub matched_view: IndexView,
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
}

#[derive(Debug, Default, Clone)]
struct CandidateAccumulator {
    total_overlap: u16,
    overlaps_by_view: HashMap<IndexView, u16>,
}

pub fn build_index(aliases: Vec<LexiconAlias>) -> PhoneticIndex {
    let mut postings: HashMap<(IndexView, String), Vec<Posting>> = HashMap::new();
    let mut short_query_postings: HashMap<String, Vec<Posting>> = HashMap::new();
    let mut by_phone_len: BTreeMap<u8, Vec<u32>> = BTreeMap::new();
    let mut by_token_count: HashMap<u8, Vec<u32>> = HashMap::new();
    let mut alias_feature_vectors = Vec::with_capacity(aliases.len());

    for alias in &aliases {
        alias_feature_vectors.push(crate::feature_view::feature_vectors_for_ipa(
            &alias.ipa_tokens,
        ));
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
            (IndexView::Feature2, qgrams(&alias.feature_tokens, 2)),
            (IndexView::Feature3, qgrams(&alias.feature_tokens, 3)),
        ] {
            let mut seen_grams = HashSet::new();
            for gram in grams {
                if !seen_grams.insert(gram.clone()) {
                    continue;
                }
                postings.entry((view, gram)).or_default().push(Posting {
                    alias_id: alias.alias_id,
                });
            }
        }

        if alias.reduced_ipa_tokens.len() <= 5 {
            let mut seen_delete_keys = HashSet::new();
            for key in short_query_keys(&alias.reduced_ipa_tokens) {
                if !seen_delete_keys.insert(key.clone()) {
                    continue;
                }
                short_query_postings.entry(key).or_default().push(Posting {
                    alias_id: alias.alias_id,
                });
            }
        }
    }

    PhoneticIndex {
        aliases,
        alias_feature_vectors,
        postings,
        short_query_postings,
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
    let query_feature_tokens = if query.feature_tokens.is_empty() {
        crate::feature_view::feature_tokens_for_ipa(&query.ipa_tokens)
    } else {
        query.feature_tokens.clone()
    };
    let query_grams_by_view = [
        (IndexView::RawIpa2, qgrams(&query.ipa_tokens, 2)),
        (IndexView::RawIpa3, qgrams(&query.ipa_tokens, 3)),
        (IndexView::ReducedIpa2, qgrams(&query.reduced_ipa_tokens, 2)),
        (IndexView::ReducedIpa3, qgrams(&query.reduced_ipa_tokens, 3)),
        (IndexView::Feature2, qgrams(&query_feature_tokens, 2)),
        (IndexView::Feature3, qgrams(&query_feature_tokens, 3)),
    ];
    let query_identifier_flags = derive_identifier_flags(&query.text);
    let query_identifier_like = is_identifier_like(&query_identifier_flags);
    let query_name_like = looks_like_name(&query.text);
    let query_has_name_like_token = sentence_word_tokens(&query.text)
        .iter()
        .any(|token| looks_like_name(&token.text));
    let mut accum: HashMap<u32, CandidateAccumulator> = HashMap::new();

    for (view, grams) in &query_grams_by_view {
        let mut unique_query_grams = HashSet::new();
        for gram in grams {
            if !unique_query_grams.insert(gram.clone()) {
                continue;
            }
            let Some(postings) = index.postings.get(&(*view, gram.clone())) else {
                continue;
            };
            for posting in postings {
                let alias = &index.aliases[posting.alias_id as usize];
                if !phone_count_compatible(query_phone_count, alias.phone_count) {
                    continue;
                }
                let overlap = accum.entry(posting.alias_id).or_default();
                overlap.total_overlap = overlap.total_overlap.saturating_add(1);
                *overlap.overlaps_by_view.entry(*view).or_default() += 1;
            }
        }
    }

    if query.reduced_ipa_tokens.len() <= 5 {
        let mut seen_short_keys = HashSet::new();
        for key in short_query_keys(&query.reduced_ipa_tokens) {
            if !seen_short_keys.insert(key.clone()) {
                continue;
            }
            let Some(postings) = index.short_query_postings.get(&key) else {
                continue;
            };
            for posting in postings {
                let alias = &index.aliases[posting.alias_id as usize];
                if !phone_count_compatible(query_phone_count, alias.phone_count) {
                    continue;
                }
                let overlap = accum.entry(posting.alias_id).or_default();
                overlap.total_overlap = overlap.total_overlap.saturating_add(1);
                *overlap
                    .overlaps_by_view
                    .entry(IndexView::ShortQueryFallback)
                    .or_default() += 1;
            }
        }
    }

    let mut out = accum
        .into_iter()
        .filter_map(|(alias_id, hit)| {
            let alias = index.aliases.get(alias_id as usize)?;
            let (matched_view, best_view_overlap) = hit
                .overlaps_by_view
                .iter()
                .max_by(|a, b| {
                    let a_score = normalized_view_overlap(a.1, &query_grams_by_view, a.0);
                    let b_score = normalized_view_overlap(b.1, &query_grams_by_view, b.0);
                    a_score.total_cmp(&b_score).then_with(|| a.1.cmp(b.1))
                })
                .map(|(view, overlap)| (*view, *overlap))?;
            let best_view_score =
                normalized_view_overlap(&best_view_overlap, &query_grams_by_view, &matched_view);
            let cross_view_support = hit.overlaps_by_view.len() as u8;
            let token_bonus = token_bonus(alias.token_count, query.token_count);
            let phone_bonus = phone_bonus(alias.phone_count, query_phone_count);
            let extra_length_penalty = extra_length_penalty(alias.phone_count, query_phone_count);
            let structure_bonus = structure_bonus(
                alias,
                query_identifier_like,
                query_name_like,
                query_has_name_like_token,
                query.token_count,
            );
            Some(RetrievalCandidate {
                alias_id,
                term: alias.term.clone(),
                alias_text: alias.alias_text.clone(),
                alias_source: alias.alias_source,
                matched_view,
                qgram_overlap: best_view_overlap,
                total_qgram_overlap: hit.total_overlap,
                best_view_score,
                cross_view_support,
                token_count_match: alias.token_count == query.token_count,
                phone_count_delta: alias.phone_count as i16 - query_phone_count as i16,
                token_bonus,
                phone_bonus,
                extra_length_penalty,
                structure_bonus,
                coarse_score: coarse_score(
                    best_view_score,
                    cross_view_support,
                    token_bonus,
                    phone_bonus,
                    extra_length_penalty,
                ),
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
    bounded.windows(q).map(|window| window.join(" ")).collect()
}

fn phone_count_compatible(query: u8, candidate: u8) -> bool {
    let delta = query.abs_diff(candidate);
    delta <= 3 || delta * 2 <= query.max(candidate)
}

fn normalized_view_overlap(
    overlap: &u16,
    query_grams_by_view: &[(IndexView, Vec<String>)],
    view: &IndexView,
) -> f32 {
    if *view == IndexView::ShortQueryFallback {
        return short_query_view_score(*overlap) * view_weight(view);
    }
    let denom = query_grams_by_view
        .iter()
        .find(|(candidate_view, _)| candidate_view == view)
        .map(|(_, grams)| grams.len().max(1) as f32)
        .unwrap_or(1.0);
    (*overlap as f32 / denom) * view_weight(view)
}

fn short_query_keys(tokens: &[String]) -> Vec<String> {
    if tokens.is_empty() {
        return Vec::new();
    }

    let mut keys = Vec::with_capacity(tokens.len() + 1);
    keys.push(short_query_key(tokens));
    if tokens.len() >= 2 {
        for idx in 0..tokens.len() {
            let mut reduced = Vec::with_capacity(tokens.len() - 1);
            reduced.extend_from_slice(&tokens[..idx]);
            reduced.extend_from_slice(&tokens[idx + 1..]);
            keys.push(short_query_key(&reduced));
        }
    }
    keys
}

fn short_query_key(tokens: &[String]) -> String {
    tokens.join("|")
}

fn short_query_view_score(overlap: u16) -> f32 {
    match overlap {
        0 => 0.0,
        1 => 0.6,
        2 => 0.85,
        _ => 1.0,
    }
}

fn token_bonus(alias_token_count: u8, query_token_count: u8) -> f32 {
    if alias_token_count == query_token_count {
        0.15
    } else if alias_token_count.abs_diff(query_token_count) <= 1 {
        0.05
    } else {
        -0.08
    }
}

fn phone_bonus(alias_phone_count: u8, query_phone_count: u8) -> f32 {
    match alias_phone_count.abs_diff(query_phone_count) {
        0 => 0.15,
        1 => 0.08,
        2 => 0.02,
        3 => -0.04,
        _ => -0.12,
    }
}

fn extra_length_penalty(alias_phone_count: u8, query_phone_count: u8) -> f32 {
    if alias_phone_count > query_phone_count {
        -0.08 * (alias_phone_count - query_phone_count) as f32
    } else if alias_phone_count < query_phone_count {
        -0.03 * (query_phone_count - alias_phone_count) as f32
    } else {
        0.0
    }
}

fn coarse_score(
    best_view_score: f32,
    cross_view_support: u8,
    token_bonus: f32,
    phone_bonus: f32,
    extra_length_penalty: f32,
) -> f32 {
    let support_bonus = 0.05 * cross_view_support.saturating_sub(1) as f32;
    best_view_score + support_bonus + token_bonus + phone_bonus + extra_length_penalty
}

fn structure_bonus(
    alias: &LexiconAlias,
    query_identifier_like: bool,
    query_name_like: bool,
    query_has_name_like_token: bool,
    query_token_count: u8,
) -> f32 {
    let alias_identifier_like = is_identifier_like(&alias.identifier_flags)
        || alias.alias_source == AliasSource::Identifier;
    let mut bonus = 0.0;

    if alias_identifier_like && query_identifier_like {
        bonus += 0.10;
    } else if alias_identifier_like && !query_identifier_like {
        bonus -= 0.10;
    }

    if alias.alias_source == AliasSource::Identifier && !query_identifier_like {
        bonus -= 0.05;
    }

    if alias.identifier_flags.acronym_like && query_token_count <= 2 && !query_identifier_like {
        bonus -= 0.06;
    }

    if query_name_like {
        if alias_identifier_like {
            bonus -= 0.22;
        } else if alias.token_count == 1
            && alias.alias_text.chars().all(|ch| ch.is_ascii_lowercase())
            && alias.alias_text.chars().any(|ch| ch.is_ascii_alphabetic())
        {
            bonus -= 0.18;
        }
    }

    if query_has_name_like_token && !query_identifier_like {
        if alias_identifier_like {
            bonus -= 0.12;
        } else if alias.token_count == 1
            && alias.alias_text.chars().all(|ch| ch.is_ascii_lowercase())
            && alias.alias_text.chars().any(|ch| ch.is_ascii_alphabetic())
        {
            bonus -= 0.12;
        }
    }

    bonus
}

fn view_weight(view: &IndexView) -> f32 {
    match view {
        IndexView::Feature2 => 0.90,
        IndexView::Feature3 => 1.0,
        IndexView::RawIpa2 | IndexView::ReducedIpa2 => 0.70,
        IndexView::RawIpa3 | IndexView::ReducedIpa3 => 0.80,
        IndexView::ShortQueryFallback => 0.9,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::phonetic_lexicon::build_phonetic_lexicon;
    use crate::types::{ReviewedConfusionSurfaceRow, VocabRow};

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

    fn confusion(
        term: &str,
        surface_form: &str,
        reviewed_ipa: &str,
    ) -> ReviewedConfusionSurfaceRow {
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
            vec![confusion(
                "AArch64",
                "ARC sixty four",
                "ɑːɹ s ɪ k s t i f ɔ ɹ",
            )],
        )]);
        let aliases = build_phonetic_lexicon(&vocab, &confusion_forms);
        let index = build_index(aliases);

        let query = RetrievalQuery {
            text: "ARC sixty four".to_string(),
            ipa_tokens: crate::prototype::parse_reviewed_ipa("ɑːɹ s ɪ k s t i f ɔ ɹ"),
            reduced_ipa_tokens: crate::phonetic_lexicon::reduce_ipa_tokens(
                &crate::prototype::parse_reviewed_ipa("ɑːɹ s ɪ k s t i f ɔ ɹ"),
            ),
            feature_tokens: crate::feature_view::feature_tokens_for_ipa(
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
                feature_tokens: crate::feature_view::feature_tokens_for_ipa(
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
                feature_tokens: crate::feature_view::feature_tokens_for_ipa(
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
            feature_tokens: crate::feature_view::feature_tokens_for_ipa(
                &crate::prototype::parse_reviewed_ipa("r ɪ k w ɛ s t"),
            ),
            token_count: 1,
        };

        let shortlist = query_index(&index, &query, 5);
        assert_eq!(shortlist[0].term, "reqwest", "{shortlist:#?}");
    }

    #[test]
    fn query_index_prefers_serde_over_serde_json_for_sirday() {
        let vocab = vec![
            row_with_reviewed_ipa("serde", "sirday", "sˈɜːdeɪ"),
            row_with_reviewed_ipa("serde_json", "sirday jason", "sˈɜːdeɪ dʒˈeɪsən"),
        ];
        let aliases = build_phonetic_lexicon(&vocab, &HashMap::new());
        let index = build_index(aliases);

        let query = RetrievalQuery {
            text: "sirday".to_string(),
            ipa_tokens: crate::prototype::parse_reviewed_ipa("s ɜː d e ɪ"),
            reduced_ipa_tokens: crate::phonetic_lexicon::reduce_ipa_tokens(
                &crate::prototype::parse_reviewed_ipa("s ɜː d e ɪ"),
            ),
            feature_tokens: crate::feature_view::feature_tokens_for_ipa(
                &crate::prototype::parse_reviewed_ipa("s ɜː d e ɪ"),
            ),
            token_count: 1,
        };

        let shortlist = query_index(&index, &query, 5);
        assert_eq!(shortlist[0].term, "serde", "{shortlist:#?}");
        let serde_json = shortlist
            .iter()
            .find(|candidate| candidate.term == "serde_json")
            .expect("serde_json candidate");
        assert!(shortlist[0].coarse_score > serde_json.coarse_score);
    }

    #[test]
    fn short_query_fallback_can_match_small_phone_strings() {
        let vocab = vec![
            row_with_reviewed_ipa("MIR", "meer", "mˈiə"),
            row_with_reviewed_ipa("miri", "miri", "mˈiəɹi"),
        ];
        let aliases = build_phonetic_lexicon(&vocab, &HashMap::new());
        let index = build_index(aliases);

        let query = RetrievalQuery {
            text: "meer".to_string(),
            ipa_tokens: crate::prototype::parse_reviewed_ipa("m i ə"),
            reduced_ipa_tokens: crate::phonetic_lexicon::reduce_ipa_tokens(
                &crate::prototype::parse_reviewed_ipa("m i ə"),
            ),
            feature_tokens: crate::feature_view::feature_tokens_for_ipa(
                &crate::prototype::parse_reviewed_ipa("m i ə"),
            ),
            token_count: 1,
        };

        let shortlist = query_index(&index, &query, 5);
        assert_eq!(shortlist[0].term, "MIR", "{shortlist:#?}");
        assert!(shortlist.iter().any(|candidate| candidate.term == "MIR"));
    }
}
