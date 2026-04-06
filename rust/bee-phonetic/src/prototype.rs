use facet::Facet;

pub fn parse_reviewed_ipa(ipa_text: &str) -> Vec<String> {
    const MULTI_SYMBOL_PHONES: &[&str] = &[
        "t͡ʃ", "d͡ʒ", "tʃ", "dʒ", "eɪ", "aɪ", "ɔɪ", "aʊ", "oʊ", "əʊ", "ɪə", "eə", "ʊə",
    ];

    let mut out = Vec::new();
    let mut i = 0;

    while i < ipa_text.len() {
        let rest = &ipa_text[i..];
        let ch = rest
            .chars()
            .next()
            .expect("slice at char boundary should have a char");

        if is_ipa_separator(ch) {
            i += ch.len_utf8();
            continue;
        }

        if let Some(seq) = MULTI_SYMBOL_PHONES
            .iter()
            .find(|seq| rest.starts_with(**seq))
        {
            out.push((*seq).to_string());
            i += seq.len();
            continue;
        }

        let mut token = ch.to_string();
        i += ch.len_utf8();

        while i < ipa_text.len() {
            let next = ipa_text[i..]
                .chars()
                .next()
                .expect("slice at char boundary should have a char");
            if !is_ipa_modifier(next) {
                break;
            }
            token.push(next);
            i += next.len_utf8();
        }

        out.push(token);
    }

    out
}

pub fn phoneme_similarity(a: &[String], b: &[String]) -> Option<f32> {
    let details = phoneme_similarity_details(a, b)?;
    Some(details.similarity)
}

#[derive(Debug, Clone, Facet)]
#[repr(u8)]
pub enum TokenEditKind {
    Match,
    Substitute,
    Insert,
    Delete,
}

#[derive(Debug, Clone, Facet)]
pub struct TokenEditOp {
    pub kind: TokenEditKind,
    pub left: Option<String>,
    pub right: Option<String>,
    pub cost: f32,
    pub boundary_penalty: f32,
}

#[derive(Debug, Clone, Facet)]
pub struct PhonemeSimilarityDetails {
    pub distance: usize,
    pub weighted_distance: f32,
    pub boundary_penalty: f32,
    pub max_len: usize,
    pub similarity: f32,
    pub ops: Vec<TokenEditOp>,
}

pub fn phoneme_similarity_details(a: &[String], b: &[String]) -> Option<PhonemeSimilarityDetails> {
    if a.is_empty() || b.is_empty() {
        return None;
    }

    let (distance, weighted_distance, boundary_penalty, ops) = levenshtein_details(a, b);
    let max_len = a.len().max(b.len());
    let normalized = 1.0 - (weighted_distance / max_len as f32);
    Some(PhonemeSimilarityDetails {
        distance,
        weighted_distance,
        boundary_penalty,
        max_len,
        similarity: normalized.clamp(0.0, 1.0),
        ops,
    })
}

fn levenshtein_details(a: &[String], b: &[String]) -> (usize, f32, f32, Vec<TokenEditOp>) {
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
        dp[i][0] = dp[i - 1][0] + 1.0;
        steps[i][0] = Step::Delete;
    }
    for j in 1..=cols {
        dp[0][j] = dp[0][j - 1] + 1.0;
        steps[0][j] = Step::Insert;
    }

    for i in 1..=rows {
        for j in 1..=cols {
            let subst_cost = if a[i - 1] == b[j - 1] { 0.0 } else { 1.0 };
            let del = dp[i - 1][j] + 1.0;
            let ins = dp[i][j - 1] + 1.0;
            let sub = dp[i - 1][j - 1] + subst_cost;

            let (best, step) = if sub <= del && sub <= ins {
                (
                    sub,
                    if subst_cost == 0.0 {
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
                ops.push(TokenEditOp {
                    kind: TokenEditKind::Match,
                    left: Some(a[i - 1].clone()),
                    right: Some(b[j - 1].clone()),
                    cost: 0.0,
                    boundary_penalty: 0.0,
                });
                i -= 1;
                j -= 1;
            }
            Step::Substitute => {
                let penalty = token_boundary_penalty(i - 1, rows, j - 1, cols);
                ops.push(TokenEditOp {
                    kind: TokenEditKind::Substitute,
                    left: Some(a[i - 1].clone()),
                    right: Some(b[j - 1].clone()),
                    cost: 1.0,
                    boundary_penalty: penalty,
                });
                i -= 1;
                j -= 1;
            }
            Step::Delete => {
                let penalty = token_boundary_penalty(i - 1, rows, j.saturating_sub(1), cols);
                ops.push(TokenEditOp {
                    kind: TokenEditKind::Delete,
                    left: Some(a[i - 1].clone()),
                    right: None,
                    cost: 1.0,
                    boundary_penalty: penalty,
                });
                i -= 1;
            }
            Step::Insert => {
                let penalty = token_boundary_penalty(i.saturating_sub(1), rows, j - 1, cols);
                ops.push(TokenEditOp {
                    kind: TokenEditKind::Insert,
                    left: None,
                    right: Some(b[j - 1].clone()),
                    cost: 1.0,
                    boundary_penalty: penalty,
                });
                j -= 1;
            }
        }
    }
    ops.reverse();

    let distance = ops
        .iter()
        .filter(|op| !matches!(op.kind, TokenEditKind::Match))
        .count();
    let boundary_penalty = ops.iter().map(|op| op.boundary_penalty).sum::<f32>();
    let weighted_distance = dp[rows][cols] + boundary_penalty;
    (distance, weighted_distance, boundary_penalty, ops)
}

fn token_boundary_penalty(
    left_idx: usize,
    left_len: usize,
    right_idx: usize,
    right_len: usize,
) -> f32 {
    let left_boundary = left_len > 0 && (left_idx == 0 || left_idx + 1 == left_len);
    let right_boundary = right_len > 0 && (right_idx == 0 || right_idx + 1 == right_len);
    if left_boundary || right_boundary {
        0.20
    } else {
        0.0
    }
}

fn is_ipa_separator(ch: char) -> bool {
    ch.is_whitespace() || matches!(ch, 'ˈ' | 'ˌ' | '.')
}

fn is_ipa_modifier(ch: char) -> bool {
    matches!(ch, 'ː' | '˞' | 'ʰ' | 'ʲ' | 'ʷ' | '̃' | '̩' | '̯')
}

#[cfg(test)]
mod tests {
    use super::parse_reviewed_ipa;

    #[test]
    fn tokenizes_compact_reviewed_ipa_into_phones() {
        assert_eq!(parse_reviewed_ipa("sˈɜːdeɪ"), vec!["s", "ɜː", "d", "eɪ"]);
    }

    #[test]
    fn tokenizes_affricates_and_long_vowels() {
        assert_eq!(
            parse_reviewed_ipa("ˈeɪ ˈɑːtʃ sˈɪkstɪfə"),
            vec!["eɪ", "ɑː", "tʃ", "s", "ɪ", "k", "s", "t", "ɪ", "f", "ə"]
        );
    }

    #[test]
    fn tokenizes_espeak_probe_output_consistently() {
        assert_eq!(
            parse_reviewed_ipa("sˌɜː dˈe ɪ"),
            vec!["s", "ɜː", "d", "e", "ɪ"]
        );
    }
}
