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

#[derive(Debug, Clone, Facet, PartialEq)]
pub struct AlignmentWindowCandidate {
    pub right_start: u32,
    pub right_end: u32,
    pub score: f32,
    pub mean_similarity: f32,
    pub length_delta: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LeftBoundaryContext {
    is_word_start: bool,
    is_word_end: bool,
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
    align_token_sequences_with_context(
        left,
        right,
        &vec![
            LeftBoundaryContext {
                is_word_start: false,
                is_word_end: false,
            };
            left.len()
        ],
    )
}

pub fn align_token_sequences_with_left_word_boundaries(
    left: &[String],
    right: &[String],
    left_word_ids: &[usize],
) -> TokenAlignment {
    assert_eq!(
        left.len(),
        left_word_ids.len(),
        "left_word_ids must match left token count"
    );
    let contexts = left_word_ids
        .iter()
        .enumerate()
        .map(|(index, &word_id)| LeftBoundaryContext {
            is_word_start: index == 0 || left_word_ids[index - 1] != word_id,
            is_word_end: index + 1 == left_word_ids.len() || left_word_ids[index + 1] != word_id,
        })
        .collect::<Vec<_>>();
    align_token_sequences_with_context(left, right, &contexts)
}

fn align_token_sequences_with_context(
    left: &[String],
    right: &[String],
    contexts: &[LeftBoundaryContext],
) -> TokenAlignment {
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
        dp[i][0] = dp[i - 1][0] + delete_cost(&left[i - 1], contexts[i - 1]);
        steps[i][0] = Step::Delete;
    }
    for j in 1..=cols {
        dp[0][j] = dp[0][j - 1] + insert_cost(left, right, contexts, 0, j - 1);
        steps[0][j] = Step::Insert;
    }

    for i in 1..=rows {
        for j in 1..=cols {
            let subst_cost =
                substitution_cost_with_context(&left[i - 1], &right[j - 1], contexts[i - 1]);
            let del = dp[i - 1][j] + delete_cost(&left[i - 1], contexts[i - 1]);
            let ins = dp[i][j - 1] + insert_cost(left, right, contexts, i, j - 1);
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
                    cost: delete_cost(&left[i - 1], contexts[i - 1]),
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
                    cost: insert_cost(left, right, contexts, i, j - 1),
                });
                j -= 1;
            }
        }
    }
    ops.reverse();

    TokenAlignment { ops }
}

fn substitution_cost_with_context(left: &str, right: &str, context: LeftBoundaryContext) -> f32 {
    let base = substitution_cost(left, right);
    if base == 0.0 {
        return 0.0;
    }

    let mut extra = 0.0;
    if context.is_word_start && !is_weak_vowel(left) {
        extra += 0.35;
    }
    if context.is_word_end {
        extra += if is_weak_vowel(left) { 0.2 } else { 0.3 };
    }
    (base + extra).min(2.5)
}

fn delete_cost(token: &str, context: LeftBoundaryContext) -> f32 {
    let mut cost = 1.0;
    if context.is_word_start && !is_weak_vowel(token) {
        cost += 0.35;
    }
    if context.is_word_end {
        cost += if is_weak_vowel(token) { 0.25 } else { 0.45 };
    }
    cost
}

fn insert_cost(
    left: &[String],
    right: &[String],
    contexts: &[LeftBoundaryContext],
    left_consumed: usize,
    right_index: usize,
) -> f32 {
    let mut cost = 1.0;
    let Some(token) = right.get(right_index) else {
        return cost;
    };
    if let Some((next_left, next_context)) =
        left.get(left_consumed).zip(contexts.get(left_consumed))
    {
        if next_context.is_word_start {
            let similarity = token_similarity(next_left, token);
            if !is_weak_vowel(next_left) {
                cost -= 0.45 * similarity;
            } else {
                cost -= 0.2 * similarity;
            }
        }
    }
    cost.max(0.25)
}

pub fn top_right_anchor_windows(
    left: &[String],
    right: &[String],
    max_results: usize,
) -> Vec<AlignmentWindowCandidate> {
    if left.is_empty() || right.is_empty() || max_results == 0 {
        return Vec::new();
    }

    let min_window = left.len().saturating_sub(2).max(1);
    let max_window = (left.len() + 2).min(right.len());
    let mut candidates = Vec::new();

    for window_len in min_window..=max_window {
        for start in 0..=right.len().saturating_sub(window_len) {
            let window = &right[start..start + window_len];
            let overlap = left.len().min(window.len());
            if overlap == 0 {
                continue;
            }

            let length_delta = window_len as i32 - left.len() as i32;
            let (score, mean_similarity) = if left.len() < 5 {
                short_window_alignment_score(left, window)
            } else {
                diagonal_window_score(left, window)
            };
            let score = score - (length_delta.abs() as f32) * 0.05;

            candidates.push(AlignmentWindowCandidate {
                right_start: start as u32,
                right_end: (start + window_len) as u32,
                score,
                mean_similarity,
                length_delta,
            });
        }
    }

    candidates.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| b.mean_similarity.total_cmp(&a.mean_similarity))
            .then_with(|| a.length_delta.abs().cmp(&b.length_delta.abs()))
            .then_with(|| a.right_start.cmp(&b.right_start))
    });

    let mut selected = Vec::new();
    for candidate in candidates {
        let overlaps_existing = selected.iter().any(|existing: &AlignmentWindowCandidate| {
            ranges_overlap(
                candidate.right_start as usize..candidate.right_end as usize,
                existing.right_start as usize..existing.right_end as usize,
            )
        });
        if overlaps_existing {
            continue;
        }
        selected.push(candidate);
        if selected.len() >= max_results {
            break;
        }
    }
    selected
}

fn substitution_cost(left: &str, right: &str) -> f32 {
    if left == right {
        return 0.0;
    }
    feature_similarity(&[left.to_string()], &[right.to_string()])
        .map(|similarity| (1.0 - similarity).clamp(0.0, 1.0))
        .unwrap_or(1.0)
}

fn token_similarity(left: &str, right: &str) -> f32 {
    1.0 - substitution_cost(left, right)
}

fn diagonal_window_score(left: &[String], window: &[String]) -> (f32, f32) {
    let overlap = left.len().min(window.len());
    let mut similarity_sum = 0.0f32;
    let mut exact_matches = 0usize;
    let mut weighted_sum = 0.0f32;
    let mut total_weight = 0.0f32;
    for index in 0..overlap {
        let similarity = token_similarity(&left[index], &window[index]);
        let weight = token_anchor_weight(&left[index], &window[index]);
        if similarity >= 0.999 {
            exact_matches += 1;
        }
        similarity_sum += similarity;
        weighted_sum += similarity * weight;
        total_weight += weight;
    }

    let mean_similarity = similarity_sum / overlap as f32;
    let coverage = overlap as f32 / left.len() as f32;
    let weighted_mean = if total_weight > 0.0 {
        weighted_sum / total_weight
    } else {
        mean_similarity
    };
    let score = weighted_mean * coverage + (exact_matches as f32 / overlap as f32) * 0.15;
    (score, mean_similarity)
}

fn short_window_alignment_score(left: &[String], window: &[String]) -> (f32, f32) {
    let rows = left.len();
    let cols = window.len();
    if rows == 0 || cols == 0 {
        return (0.0, 0.0);
    }

    let mut dp = vec![vec![f32::NEG_INFINITY; cols + 1]; rows + 1];
    dp[0].fill(0.0);

    for i in 1..=rows {
        dp[i][0] = dp[i - 1][0] - deletion_penalty(&left[i - 1]);
    }

    for i in 1..=rows {
        for j in 1..=cols {
            let match_score = dp[i - 1][j - 1] + pair_match_score(&left[i - 1], &window[j - 1]);
            let delete_score = dp[i - 1][j] - deletion_penalty(&left[i - 1]);
            let insert_score = dp[i][j - 1] - insertion_penalty(&window[j - 1]);
            dp[i][j] = match_score.max(delete_score).max(insert_score);
        }
    }

    let best = dp[rows].iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let normalized = (best / rows.max(cols) as f32).clamp(-1.0, 1.0);
    let score = ((normalized + 1.0) / 2.0).clamp(0.0, 1.0);

    let mut similarity_sum = 0.0f32;
    let mut overlap = 0usize;
    for index in 0..rows.min(cols) {
        similarity_sum += token_similarity(&left[index], &window[index]);
        overlap += 1;
    }
    let mean_similarity = if overlap > 0 {
        similarity_sum / overlap as f32
    } else {
        0.0
    };
    (score, mean_similarity)
}

fn pair_match_score(left: &str, right: &str) -> f32 {
    let similarity = token_similarity(left, right);
    let weight = token_anchor_weight(left, right);
    similarity * weight - (1.0 - similarity) * 0.65
}

fn deletion_penalty(token: &str) -> f32 {
    if is_weak_vowel(token) {
        0.45
    } else if is_vowel(token) {
        0.75
    } else {
        1.1
    }
}

fn insertion_penalty(token: &str) -> f32 {
    if is_weak_vowel(token) {
        0.4
    } else if is_vowel(token) {
        0.7
    } else {
        1.0
    }
}

fn token_anchor_weight(left: &str, right: &str) -> f32 {
    token_kind_weight(left).max(token_kind_weight(right))
}

fn token_kind_weight(token: &str) -> f32 {
    if is_weak_vowel(token) {
        0.45
    } else if is_vowel(token) {
        0.75
    } else {
        1.15
    }
}

fn is_weak_vowel(token: &str) -> bool {
    matches!(token, "ə" | "ɪ" | "ʊ")
}

fn is_vowel(token: &str) -> bool {
    matches!(
        token,
        "a" | "æ" | "ɑ" | "ɔ" | "ɛ" | "ə" | "ɪ" | "ʊ" | "i" | "u"
    )
}

fn ranges_overlap(left: Range<usize>, right: Range<usize>) -> bool {
    left.start < right.end && right.start < left.end
}

#[cfg(test)]
mod tests {
    use super::{
        AlignmentOp, AlignmentOpKind, ComparisonToken, TokenAlignment, align_token_sequences,
        align_token_sequences_with_left_word_boundaries, top_right_anchor_windows,
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

    #[test]
    fn top_anchor_windows_prefers_exact_repeated_occurrence() {
        let left = vec!["m", "ɪ", "ɹ", "i"]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        let right = vec!["x", "m", "ɪ", "ɹ", "i", "m", "ɛ", "ɹ", "ɪ"]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();

        let windows = top_right_anchor_windows(&left, &right, 2);
        assert_eq!(windows[0].right_start, 1);
        assert_eq!(windows[0].right_end, 5);
    }

    #[test]
    fn top_anchor_windows_keeps_multiple_repeated_candidates() {
        let left = vec!["m", "a", "k"]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        let right = vec!["m", "a", "k", "x", "m", "a", "k"]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();

        let windows = top_right_anchor_windows(&left, &right, 2);
        assert_eq!(windows.len(), 2);
        assert_eq!(windows[0].right_start, 0);
        assert_eq!(windows[1].right_start, 4);
    }

    #[test]
    fn boundary_aware_alignment_protects_word_tails_and_next_onset() {
        let left = vec!["a", "k", "t", "ʃ", "ʊ", "l", "ɪ", "m", "ɛ", "ɹ", "ɪ"]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        let right = vec!["ɛ", "k", "t", "ʃ", "ə", "w", "ə", "l", "ɪ", "m", "ɪ", "ɪ"]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        let word_ids = vec![0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1];

        let alignment = align_token_sequences_with_left_word_boundaries(&left, &right, &word_ids);
        let ops = alignment.ops;

        let l_index = ops
            .iter()
            .position(|op| op.left_token.as_deref() == Some("l"))
            .expect("l op present");
        let final_i_index = ops
            .iter()
            .rposition(|op| op.left_token.as_deref() == Some("ɪ"))
            .expect("final i op present");
        let first_second_word_index = ops
            .iter()
            .position(|op| op.left_index.is_some_and(|index| index as usize >= 7))
            .expect("second-word op present");

        assert!(l_index < first_second_word_index);
        assert!(final_i_index < first_second_word_index);
    }
}
