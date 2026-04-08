use std::ops::Range;

use facet::Facet;

use crate::feature_view::feature_similarity;

#[derive(Debug, Clone, Facet, PartialEq, Eq)]
pub struct ComparisonToken {
    pub token: String,
    pub source_start: usize,
    pub source_end: usize,
}

#[derive(Debug, Clone, Facet, PartialEq)]
#[repr(u8)]
pub enum AlignmentOpKind {
    Match,
    Substitute,
    Insert,
    Delete,
}

#[derive(Debug, Clone, Facet, PartialEq)]
pub struct AlignmentOp {
    pub kind: AlignmentOpKind,
    pub left_index: Option<u32>,
    pub right_index: Option<u32>,
    pub left_token: Option<String>,
    pub right_token: Option<String>,
    pub cost: f32,
}

#[derive(Debug, Clone, Facet, PartialEq)]
pub struct TokenAlignment {
    pub ops: Vec<AlignmentOp>,
}

impl TokenAlignment {
    pub fn project_left_range(&self, left_range: Range<usize>) -> Option<Range<usize>> {
        if left_range.start >= left_range.end {
            return None;
        }

        let mut matched = Vec::new();
        let mut right_cursor = 0usize;
        let mut right_before = None;
        let mut right_after = None;

        for op in &self.ops {
            let left_idx = op.left_index.map(|idx| idx as usize);
            let right_idx = op.right_index.map(|idx| idx as usize);

            let (Some(left), Some(right)) = (op.left_index, op.right_index) else {
                if right_idx.is_some() {
                    right_cursor += 1;
                }
                continue;
            };
            let left = left as usize;
            let right = right as usize;
            if left_range.start <= left && left < left_range.end {
                matched.push(right);
            }

            if let Some(left_idx) = left_idx {
                if left_idx < left_range.start {
                    right_before = Some(right_idx.map_or(right_cursor, |idx| idx + 1));
                } else if left_idx >= left_range.end && right_after.is_none() {
                    right_after = Some(right_idx.unwrap_or(right_cursor));
                }
            }

            if right_idx.is_some() {
                right_cursor += 1;
            }
        }
        if !matched.is_empty() {
            let start = *matched.iter().min().expect("non-empty matches");
            let end = matched.iter().max().expect("non-empty matches") + 1;
            let expanded_start = right_before.map_or(start, |anchor| anchor.min(start));
            let expanded_end = right_after.map_or(end, |anchor| anchor.max(end));
            if expanded_start <= expanded_end {
                return Some(expanded_start..expanded_end);
            }
            return Some(start..end);
        }

        match (right_before, right_after) {
            (Some(start), Some(end)) if start <= end => Some(start..end),
            (Some(start), None) => Some(start..right_cursor),
            (None, Some(end)) => Some(0..end),
            (None, None) if right_cursor > 0 => Some(0..right_cursor),
            _ => None,
        }
    }
}

pub fn align_token_sequences(left: &[String], right: &[String]) -> TokenAlignment {
    #[derive(Clone, Copy)]
    enum Step {
        Match,
        Substitute,
        Insert,
        Delete,
    }

    let rows = left.len();
    let cols = right.len();
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
            let subst_cost = substitution_cost(&left[i - 1], &right[j - 1]);
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
                ops.push(AlignmentOp {
                    kind: AlignmentOpKind::Match,
                    left_index: Some((i - 1) as u32),
                    right_index: Some((j - 1) as u32),
                    left_token: Some(left[i - 1].clone()),
                    right_token: Some(right[j - 1].clone()),
                    cost: 0.0,
                });
                i -= 1;
                j -= 1;
            }
            Step::Substitute => {
                ops.push(AlignmentOp {
                    kind: AlignmentOpKind::Substitute,
                    left_index: Some((i - 1) as u32),
                    right_index: Some((j - 1) as u32),
                    left_token: Some(left[i - 1].clone()),
                    right_token: Some(right[j - 1].clone()),
                    cost: substitution_cost(&left[i - 1], &right[j - 1]),
                });
                i -= 1;
                j -= 1;
            }
            Step::Delete => {
                ops.push(AlignmentOp {
                    kind: AlignmentOpKind::Delete,
                    left_index: Some((i - 1) as u32),
                    right_index: None,
                    left_token: Some(left[i - 1].clone()),
                    right_token: None,
                    cost: 1.0,
                });
                i -= 1;
            }
            Step::Insert => {
                ops.push(AlignmentOp {
                    kind: AlignmentOpKind::Insert,
                    left_index: None,
                    right_index: Some((j - 1) as u32),
                    left_token: None,
                    right_token: Some(right[j - 1].clone()),
                    cost: 1.0,
                });
                j -= 1;
            }
        }
    }
    ops.reverse();

    TokenAlignment { ops }
}

fn substitution_cost(left: &str, right: &str) -> f32 {
    if left == right {
        return 0.0;
    }
    feature_similarity(&[left.to_string()], &[right.to_string()])
        .map(|similarity| (1.0 - similarity).clamp(0.0, 1.0))
        .unwrap_or(1.0)
}

#[cfg(test)]
mod tests {
    use super::{
        AlignmentOp, AlignmentOpKind, ComparisonToken, TokenAlignment, align_token_sequences,
    };

    #[test]
    fn projects_direct_matches() {
        let left = vec!["m", "ɪ", "ɹ", "ɪ"]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        let right = vec!["x", "m", "ɪ", "ɹ", "ɪ", "y"]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        let alignment = align_token_sequences(&left, &right);
        assert_eq!(alignment.project_left_range(0..4), Some(1..5));
        assert_eq!(alignment.project_left_range(1..3), Some(2..4));
    }

    #[test]
    fn projects_deleted_left_tokens_between_neighbors() {
        let left = vec!["a", "b", "c"]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        let right = vec!["a", "c"]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        let alignment = align_token_sequences(&left, &right);
        assert_eq!(alignment.project_left_range(1..2), Some(1..1));
    }

    #[test]
    fn comparison_token_keeps_source_range() {
        let token = ComparisonToken {
            token: "ə".to_string(),
            source_start: 3,
            source_end: 5,
        };
        assert_eq!(token.source_start, 3);
        assert_eq!(token.source_end, 5);
    }

    #[test]
    fn projects_sparse_match_to_surrounding_anchor_window() {
        let alignment = TokenAlignment {
            ops: vec![
                AlignmentOp {
                    kind: AlignmentOpKind::Match,
                    left_index: Some(0),
                    right_index: Some(0),
                    left_token: Some("pre".into()),
                    right_token: Some("pre".into()),
                    cost: 0.0,
                },
                AlignmentOp {
                    kind: AlignmentOpKind::Insert,
                    left_index: None,
                    right_index: Some(1),
                    left_token: None,
                    right_token: Some("r1".into()),
                    cost: 1.0,
                },
                AlignmentOp {
                    kind: AlignmentOpKind::Delete,
                    left_index: Some(1),
                    right_index: None,
                    left_token: Some("l1".into()),
                    right_token: None,
                    cost: 1.0,
                },
                AlignmentOp {
                    kind: AlignmentOpKind::Delete,
                    left_index: Some(2),
                    right_index: None,
                    left_token: Some("l2".into()),
                    right_token: None,
                    cost: 1.0,
                },
                AlignmentOp {
                    kind: AlignmentOpKind::Match,
                    left_index: Some(3),
                    right_index: Some(2),
                    left_token: Some("mid".into()),
                    right_token: Some("mid".into()),
                    cost: 0.0,
                },
                AlignmentOp {
                    kind: AlignmentOpKind::Insert,
                    left_index: None,
                    right_index: Some(3),
                    left_token: None,
                    right_token: Some("r2".into()),
                    cost: 1.0,
                },
                AlignmentOp {
                    kind: AlignmentOpKind::Insert,
                    left_index: None,
                    right_index: Some(4),
                    left_token: None,
                    right_token: Some("r3".into()),
                    cost: 1.0,
                },
                AlignmentOp {
                    kind: AlignmentOpKind::Delete,
                    left_index: Some(4),
                    right_index: None,
                    left_token: Some("l3".into()),
                    right_token: None,
                    cost: 1.0,
                },
                AlignmentOp {
                    kind: AlignmentOpKind::Match,
                    left_index: Some(5),
                    right_index: Some(5),
                    left_token: Some("post".into()),
                    right_token: Some("post".into()),
                    cost: 0.0,
                },
            ],
        };

        assert_eq!(alignment.project_left_range(1..5), Some(1..5));
    }
}
