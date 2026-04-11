use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use facet::Facet;

use rspanphon::featuretable::FeatureTable;

static FEATURE_TABLE: OnceLock<FeatureTable> = OnceLock::new();
static FEATURE_VECTOR_CACHE: OnceLock<Mutex<HashMap<String, Option<Vec<f32>>>>> = OnceLock::new();
static TOKEN_SIMILARITY_CACHE: OnceLock<Mutex<HashMap<(String, String), Option<f32>>>> =
    OnceLock::new();

pub fn feature_tokens_for_ipa(ipa_tokens: &[String]) -> Vec<String> {
    let table = FEATURE_TABLE.get_or_init(FeatureTable::new);
    ipa_tokens
        .iter()
        .map(|token| match table.ft.get(token) {
            Some(features) => encode_feature_vector(features),
            None => format!("unk:{token}"),
        })
        .collect()
}

pub fn feature_similarity(a: &[String], b: &[String]) -> Option<f32> {
    if a.is_empty() || b.is_empty() {
        return None;
    }

    if a.len() == 1 && b.len() == 1 {
        return feature_similarity_for_tokens(&a[0], &b[0]);
    }

    let a_vecs = feature_vectors_for_ipa(a);
    let b_vecs = feature_vectors_for_ipa(b);
    feature_similarity_from_vectors(&a_vecs, &b_vecs, a.len().max(b.len()))
}

pub fn feature_similarity_for_tokens(left: &str, right: &str) -> Option<f32> {
    let key = if left <= right {
        (left.to_string(), right.to_string())
    } else {
        (right.to_string(), left.to_string())
    };
    let cache = TOKEN_SIMILARITY_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    {
        let guard = cache.lock().expect("token similarity cache poisoned");
        if let Some(cached) = guard.get(&key) {
            return *cached;
        }
    }

    let left_vec = feature_vector_for_token(left)?;
    let right_vec = feature_vector_for_token(right)?;
    let similarity = feature_similarity_from_vectors(
        std::slice::from_ref(&left_vec),
        std::slice::from_ref(&right_vec),
        1,
    );
    cache
        .lock()
        .expect("token similarity cache poisoned")
        .insert(key, similarity);
    similarity
}

pub fn feature_vectors_for_ipa(ipa_tokens: &[String]) -> Vec<Vec<f32>> {
    ipa_tokens
        .iter()
        .filter_map(|token| feature_vector_for_token(token))
        .collect()
}

pub fn feature_similarity_from_vectors(
    a: &[Vec<f32>],
    b: &[Vec<f32>],
    max_token_len: usize,
) -> Option<f32> {
    let details = feature_similarity_details_from_vectors(a, b, max_token_len)?;
    Some(details.similarity)
}

#[derive(Debug, Clone)]
pub struct FeatureSimilarityDetails {
    pub distance: f32,
    pub weighted_distance: f32,
    pub boundary_penalty: f32,
    pub max_len: usize,
    pub similarity: f32,
    pub ops: Vec<FeatureEditOp>,
}

#[derive(Debug, Clone, Facet)]
#[repr(u8)]
pub enum FeatureEditKind {
    Match,
    Substitute,
    Insert,
    Delete,
}

#[derive(Debug, Clone, Facet)]
pub struct FeatureEditOp {
    pub kind: FeatureEditKind,
    pub left: Option<String>,
    pub right: Option<String>,
    pub cost: f32,
    pub boundary_penalty: f32,
}

pub fn feature_similarity_details_from_vectors(
    a: &[Vec<f32>],
    b: &[Vec<f32>],
    max_token_len: usize,
) -> Option<FeatureSimilarityDetails> {
    let left = (0..a.len())
        .map(|idx| format!("#{idx}"))
        .collect::<Vec<_>>();
    let right = (0..b.len())
        .map(|idx| format!("#{idx}"))
        .collect::<Vec<_>>();
    feature_similarity_details_with_labels(left.as_slice(), a, right.as_slice(), b, max_token_len)
}

pub fn feature_similarity_details_with_labels(
    a_tokens: &[String],
    a: &[Vec<f32>],
    b_tokens: &[String],
    b: &[Vec<f32>],
    max_token_len: usize,
) -> Option<FeatureSimilarityDetails> {
    let (distance, weighted_distance, boundary_penalty, ops) =
        feature_edit_distance_details(a_tokens, a, b_tokens, b)?;
    let max_len = max_token_len.max(1);
    let similarity = (1.0 - (weighted_distance / max_len as f32)).clamp(0.0, 1.0);
    Some(FeatureSimilarityDetails {
        distance,
        weighted_distance,
        boundary_penalty,
        max_len,
        similarity,
        ops,
    })
}

fn encode_feature_vector(features: &[i8]) -> String {
    let mut out = String::with_capacity(features.len());
    for feature in features {
        out.push(match feature {
            -1 => '-',
            0 => '0',
            1 => '+',
            _ => '?',
        });
    }
    out
}

fn feature_edit_distance_details(
    a_tokens: &[String],
    a: &[Vec<f32>],
    b_tokens: &[String],
    b: &[Vec<f32>],
) -> Option<(f32, f32, f32, Vec<FeatureEditOp>)> {
    if a.is_empty() || b.is_empty() {
        return None;
    }

    #[derive(Clone, Copy)]
    enum Step {
        Match,
        Substitute,
        Insert,
        Delete,
    }

    let rows = a.len();
    let cols = b.len();
    let mut dp = vec![vec![0.0f32; cols + 1]; rows + 1];
    let mut steps = vec![vec![Step::Match; cols + 1]; rows + 1];

    for i in 1..=rows {
        dp[i][0] = dp[i - 1][0] + insertion_deletion_cost(&a[i - 1]);
        steps[i][0] = Step::Delete;
    }
    for j in 1..=cols {
        dp[0][j] = dp[0][j - 1] + insertion_deletion_cost(&b[j - 1]);
        steps[0][j] = Step::Insert;
    }

    for i in 1..=rows {
        for j in 1..=cols {
            let sub_cost = substitution_cost(&a[i - 1], &b[j - 1]);
            let del = dp[i - 1][j] + insertion_deletion_cost(&a[i - 1]);
            let ins = dp[i][j - 1] + insertion_deletion_cost(&b[j - 1]);
            let sub = dp[i - 1][j - 1] + sub_cost;

            let (best, step) = if sub <= del && sub <= ins {
                (
                    sub,
                    if sub_cost == 0.0 {
                        Step::Match
                    } else {
                        Step::Substitute
                    },
                )
            } else if del <= ins {
                (del, Step::Delete)
            } else {
                (ins, Step::Insert)
            };
            dp[i][j] = best;
            steps[i][j] = step;
        }
    }

    let mut ops = Vec::new();
    let mut i = rows;
    let mut j = cols;
    while i > 0 || j > 0 {
        let step = if i == 0 {
            Step::Insert
        } else if j == 0 {
            Step::Delete
        } else {
            steps[i][j]
        };
        match step {
            Step::Match => {
                ops.push(FeatureEditOp {
                    kind: FeatureEditKind::Match,
                    left: Some(a_tokens[i - 1].clone()),
                    right: Some(b_tokens[j - 1].clone()),
                    cost: 0.0,
                    boundary_penalty: 0.0,
                });
                i -= 1;
                j -= 1;
            }
            Step::Substitute => {
                let penalty = feature_boundary_penalty(i - 1, rows, j - 1, cols);
                ops.push(FeatureEditOp {
                    kind: FeatureEditKind::Substitute,
                    left: Some(a_tokens[i - 1].clone()),
                    right: Some(b_tokens[j - 1].clone()),
                    cost: substitution_cost(&a[i - 1], &b[j - 1]),
                    boundary_penalty: penalty,
                });
                i -= 1;
                j -= 1;
            }
            Step::Delete => {
                let penalty = feature_boundary_penalty(i - 1, rows, j.saturating_sub(1), cols);
                ops.push(FeatureEditOp {
                    kind: FeatureEditKind::Delete,
                    left: Some(a_tokens[i - 1].clone()),
                    right: None,
                    cost: insertion_deletion_cost(&a[i - 1]),
                    boundary_penalty: penalty,
                });
                i -= 1;
            }
            Step::Insert => {
                let penalty = feature_boundary_penalty(i.saturating_sub(1), rows, j - 1, cols);
                ops.push(FeatureEditOp {
                    kind: FeatureEditKind::Insert,
                    left: None,
                    right: Some(b_tokens[j - 1].clone()),
                    cost: insertion_deletion_cost(&b[j - 1]),
                    boundary_penalty: penalty,
                });
                j -= 1;
            }
        }
    }
    ops.reverse();

    let boundary_penalty = ops.iter().map(|op| op.boundary_penalty).sum::<f32>();
    let weighted_distance = dp[rows][cols] + boundary_penalty;
    Some((dp[rows][cols], weighted_distance, boundary_penalty, ops))
}

fn feature_boundary_penalty(
    left_idx: usize,
    left_len: usize,
    right_idx: usize,
    right_len: usize,
) -> f32 {
    let left_boundary = left_len > 0 && (left_idx == 0 || left_idx + 1 == left_len);
    let right_boundary = right_len > 0 && (right_idx == 0 || right_idx + 1 == right_len);
    if left_boundary || right_boundary {
        0.10
    } else {
        0.0
    }
}

// Panphon feature salience weights (Mortensen et al. 2016).
// Order matches ipa_all.csv columns: syl, son, cons, cont, delrel, lat, nas,
// strid, voi, sg, cg, ant, cor, distr, lab, hi, lo, back, round, velaric, tense, long, hitone, hireg
const FEATURE_WEIGHTS: &[f32] = &[
    1.0, 1.0, 1.0, 0.5, 0.25, 0.25, 0.25, 0.125, 0.125, 0.125, 0.125, 0.25, 0.25, 0.125, 0.25,
    0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.125, 0.0, 0.0,
];
const FEATURE_WEIGHT_SUM: f32 = 7.25; // sum of weights (hitone/hireg are 0.0)

fn substitution_cost(a: &[f32], b: &[f32]) -> f32 {
    if a == b {
        return 0.0;
    }

    let weighted_diff: f32 = a
        .iter()
        .zip(b)
        .enumerate()
        .map(|(i, (lhs, rhs))| {
            let w = FEATURE_WEIGHTS.get(i).copied().unwrap_or(0.125);
            (lhs - rhs).abs() * w
        })
        .sum();
    (weighted_diff / FEATURE_WEIGHT_SUM).clamp(0.0, 1.0)
}

fn insertion_deletion_cost(_vec: &[f32]) -> f32 {
    // Panphon: inserting or deleting any segment costs sum(weights) / sum(weights) = 1.0.
    // This makes deletion expensive — you can't cheaply delete your way to a match.
    1.0
}

pub fn feature_vector_for_token(token: &str) -> Option<Vec<f32>> {
    let cache = FEATURE_VECTOR_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    {
        let guard = cache.lock().expect("feature vector cache poisoned");
        if let Some(cached) = guard.get(token) {
            return cached.clone();
        }
    }

    let table = FEATURE_TABLE.get_or_init(FeatureTable::new);
    let computed = compute_feature_vector(table, token);
    cache
        .lock()
        .expect("feature vector cache poisoned")
        .insert(token.to_string(), computed.clone());
    computed
}

fn compute_feature_vector(table: &FeatureTable, token: &str) -> Option<Vec<f32>> {
    if let Some(vec) = table.ft.get(token) {
        return Some(vec.iter().map(|value| *value as f32).collect());
    }

    let simplified = strip_modifiers(token);
    if simplified != token {
        if let Some(vec) = table.ft.get(&simplified) {
            return Some(vec.iter().map(|value| *value as f32).collect());
        }
    }

    let phonemes = table.phonemes(token);
    if !phonemes.is_empty() {
        return average_feature_vectors(table, &phonemes);
    }

    let simplified_phonemes = table.phonemes(&simplified);
    if !simplified_phonemes.is_empty() {
        return average_feature_vectors(table, &simplified_phonemes);
    }

    None
}

fn average_feature_vectors(table: &FeatureTable, phonemes: &[String]) -> Option<Vec<f32>> {
    let vectors = phonemes
        .iter()
        .filter_map(|phoneme| table.ft.get(phoneme))
        .collect::<Vec<_>>();
    let first = vectors.first()?;
    let mut out = vec![0.0f32; first.len()];
    for vec in &vectors {
        for (idx, value) in vec.iter().enumerate() {
            out[idx] += *value as f32;
        }
    }
    let denom = vectors.len() as f32;
    for value in &mut out {
        *value /= denom;
    }
    Some(out)
}

fn strip_modifiers(token: &str) -> String {
    token
        .chars()
        .filter(|ch| !matches!(ch, 'ː' | '˞' | 'ʰ' | 'ʲ' | 'ʷ' | '̃' | '̩' | '̯'))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_feature_tokens_for_known_ipa_segments() {
        let tokens = feature_tokens_for_ipa(&["m".to_string(), "i".to_string(), "ə".to_string()]);
        assert_eq!(tokens.len(), 3);
        assert!(
            tokens.iter().all(|token| !token.starts_with("unk:")),
            "{tokens:#?}"
        );
    }

    #[test]
    fn feature_similarity_prefers_closer_segments() {
        let exact = feature_similarity(
            &["m".to_string(), "i".to_string()],
            &["m".to_string(), "i".to_string()],
        )
        .unwrap();
        let close = feature_similarity(
            &["m".to_string(), "i".to_string()],
            &["m".to_string(), "ɪ".to_string()],
        )
        .unwrap();
        let far = feature_similarity(
            &["m".to_string(), "i".to_string()],
            &["k".to_string(), "u".to_string()],
        )
        .unwrap();
        assert!(exact > close);
        assert!(close > far);
    }

    #[test]
    fn feature_similarity_handles_modifier_and_diphthongish_tokens() {
        let token = feature_similarity(
            &["ɜː".to_string(), "k".to_string(), "ə".to_string()],
            &["eə".to_string(), "k".to_string(), "əʊ".to_string()],
        )
        .unwrap();
        assert!(token > 0.45, "{token}");
    }
}
