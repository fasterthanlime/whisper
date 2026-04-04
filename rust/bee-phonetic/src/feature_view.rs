use std::sync::OnceLock;

use rspanphon::featuretable::FeatureTable;

static FEATURE_TABLE: OnceLock<FeatureTable> = OnceLock::new();

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

    let table = FEATURE_TABLE.get_or_init(FeatureTable::new);
    let a_vecs = ipa_tokens_to_feature_vectors(table, a);
    let b_vecs = ipa_tokens_to_feature_vectors(table, b);
    if a_vecs.is_empty() || b_vecs.is_empty() {
        return None;
    }

    let distance = FeatureTable::fd(a_vecs, b_vecs) as f32;
    Some((1.0 / (1.0 + distance)).clamp(0.0, 1.0))
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

fn ipa_tokens_to_feature_vectors(table: &FeatureTable, ipa_tokens: &[String]) -> Vec<Vec<i8>> {
    ipa_tokens
        .iter()
        .filter_map(|token| table.ft.get(token).cloned())
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
}
