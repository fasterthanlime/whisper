use std::ops::Range;

use crate::g2p::CachedEspeakG2p;
use bee_phonetic::{
    AlignmentOp, TokenAlignment, align_token_sequences, feature_similarity,
    normalize_ipa_for_comparison, normalize_ipa_for_comparison_with_spans, phoneme_similarity,
    sentence_word_tokens, top_right_anchor_windows,
};
use bee_zipa_mlx::infer::PhoneSpan;

#[derive(Debug, Clone)]
pub struct SpanAlignmentSelection {
    pub range: Range<usize>,
    pub alignment: TokenAlignment,
    pub zipa_normalized: Vec<String>,
    pub projected_alignment_score: Option<f32>,
    pub chosen_alignment_score: f32,
    pub second_best_alignment_score: Option<f32>,
    pub alignment_score_gap: Option<f32>,
    pub alignment_source: &'static str,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TimedZipaRange {
    pub normalized_range: Range<usize>,
    pub raw_phone_range: Range<usize>,
    pub start_time_secs: f64,
    pub end_time_secs: f64,
}

#[derive(Debug, Clone)]
pub struct WordAlignmentWindow {
    pub zipa_norm_range: Range<usize>,
    pub ops: Vec<AlignmentOp>,
}

#[derive(Clone)]
struct WordSegmentCandidate {
    zipa_norm_range: Range<usize>,
    alignment: TokenAlignment,
    local_score: f32,
}

pub fn transcript_word_raw_ranges(
    g2p: &mut CachedEspeakG2p,
    transcript: &str,
) -> Result<Vec<(Range<usize>, Vec<String>)>, String> {
    let words = sentence_word_tokens(transcript);
    let raw_groups = g2p
        .ipa_word_tokens_in_utterance(transcript)
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("espeak produced no tokens for '{transcript}'"))?;

    if raw_groups.len() != words.len() {
        return Err(format!(
            "espeak word-group count mismatch for '{transcript}': transcript has {} words, espeak produced {} groups",
            words.len(),
            raw_groups.len()
        ));
    }

    Ok(words
        .into_iter()
        .zip(raw_groups)
        .map(|(word, raw_tokens)| (word.char_start..word.char_end, raw_tokens))
        .collect())
}

pub fn transcript_token_range_for_span(
    word_ranges: &[(Range<usize>, Vec<String>)],
    token_start: usize,
    token_end: usize,
) -> Range<usize> {
    let start = word_ranges
        .iter()
        .take(token_start)
        .map(|(_, tokens)| tokens.len())
        .sum::<usize>();
    let len = word_ranges[token_start..token_end]
        .iter()
        .map(|(_, tokens)| tokens.len())
        .sum::<usize>();
    start..(start + len)
}

pub fn transcript_normalized_for_span(
    word_ranges: &[(Range<usize>, Vec<String>)],
    token_start: usize,
    token_end: usize,
) -> Vec<String> {
    normalize_ipa_for_comparison(
        &word_ranges[token_start..token_end]
            .iter()
            .flat_map(|(_, tokens)| tokens.iter().cloned())
            .collect::<Vec<_>>(),
    )
}

pub fn raw_slice_for_normalized_range(
    raw_tokens: &[String],
    normalized: &[bee_phonetic::ComparisonToken],
    normalized_range: Range<usize>,
) -> Vec<String> {
    let Some(first) = normalized.get(normalized_range.start) else {
        return Vec::new();
    };
    let Some(last) = normalized.get(normalized_range.end.saturating_sub(1)) else {
        return Vec::new();
    };
    raw_tokens
        .get(first.source_start..last.source_end)
        .unwrap_or(&[])
        .to_vec()
}

pub fn raw_phone_range_for_normalized_range(
    normalized: &[bee_phonetic::ComparisonToken],
    normalized_range: Range<usize>,
) -> Option<Range<usize>> {
    let first = normalized.get(normalized_range.start)?;
    let last = normalized.get(normalized_range.end.checked_sub(1)?)?;
    Some(first.source_start..last.source_end)
}

pub fn timed_range_for_raw_phone_range(
    phone_spans: &[PhoneSpan],
    raw_phone_range: Range<usize>,
) -> Option<TimedZipaRange> {
    if raw_phone_range.start >= raw_phone_range.end {
        return None;
    }
    let start = phone_spans.get(raw_phone_range.start)?;
    let end = phone_spans.get(raw_phone_range.end.checked_sub(1)?)?;
    Some(TimedZipaRange {
        normalized_range: 0..0,
        raw_phone_range,
        start_time_secs: start.start_time_secs,
        end_time_secs: end.end_time_secs,
    })
}

pub fn timed_range_for_normalized_range(
    normalized: &[bee_phonetic::ComparisonToken],
    phone_spans: &[PhoneSpan],
    normalized_range: Range<usize>,
) -> Option<TimedZipaRange> {
    let raw_phone_range =
        raw_phone_range_for_normalized_range(normalized, normalized_range.clone())?;
    let mut timed = timed_range_for_raw_phone_range(phone_spans, raw_phone_range)?;
    timed.normalized_range = normalized_range;
    Some(timed)
}

pub fn select_span_alignment_range(
    transcript_normalized: &[String],
    utterance_zipa_normalized: &[String],
    projected_range: Option<Range<usize>>,
) -> Option<SpanAlignmentSelection> {
    let mut candidate_ranges = Vec::<Range<usize>>::new();
    if let Some(range) = &projected_range {
        candidate_ranges.push(range.clone());
    }
    candidate_ranges.extend(
        top_right_anchor_windows(transcript_normalized, utterance_zipa_normalized, 3)
            .into_iter()
            .map(|window| window.right_start as usize..window.right_end as usize),
    );
    normalize_candidate_ranges(&mut candidate_ranges, utterance_zipa_normalized.len());

    let mut scored = Vec::new();
    for (candidate_index, range) in candidate_ranges.into_iter().enumerate() {
        let zipa_normalized = utterance_zipa_normalized
            .get(range.clone())
            .unwrap_or(&[])
            .to_vec();
        if zipa_normalized.is_empty() {
            continue;
        }
        let alignment = align_token_sequences(transcript_normalized, &zipa_normalized);
        let score = alignment_quality_score(
            &alignment.ops,
            transcript_normalized.len(),
            zipa_normalized.len(),
        );
        scored.push((candidate_index, range, score, alignment, zipa_normalized));
    }

    scored.sort_by(|a, b| b.2.total_cmp(&a.2));
    let projected_alignment_score = scored
        .iter()
        .find(|(candidate_index, _, _, _, _)| projected_range.is_some() && *candidate_index == 0)
        .map(|(_, _, score, _, _)| *score);
    let second_best_alignment_score = scored.get(1).map(|(_, _, score, _, _)| *score);
    let (candidate_index, range, chosen_alignment_score, alignment, zipa_normalized) =
        scored.into_iter().next()?;
    let alignment_score_gap =
        second_best_alignment_score.map(|score| chosen_alignment_score - score);
    let alignment_source = if projected_range.is_some() && candidate_index == 0 {
        "projected"
    } else {
        "anchored"
    };

    Some(SpanAlignmentSelection {
        range,
        alignment,
        zipa_normalized,
        projected_alignment_score,
        chosen_alignment_score,
        second_best_alignment_score,
        alignment_score_gap,
        alignment_source,
    })
}

pub fn select_segmental_word_windows(
    transcript_word_tokens: &[Vec<String>],
    transcript_token_ranges: &[Range<usize>],
    utterance_zipa_normalized: &[String],
    utterance_alignment: &TokenAlignment,
) -> Vec<Option<WordAlignmentWindow>> {
    if transcript_word_tokens.is_empty() {
        return Vec::new();
    }

    let candidates_per_word = transcript_word_tokens
        .iter()
        .enumerate()
        .map(|(word_index, transcript_tokens)| {
            let candidates = build_word_segment_candidates(
                transcript_tokens,
                transcript_token_ranges.get(word_index).cloned(),
                utterance_zipa_normalized,
                utterance_alignment,
            );
            tracing::debug!(
                word_index,
                transcript_tokens = %transcript_tokens.join(" "),
                token_range = ?transcript_token_ranges.get(word_index),
                candidates = candidates.len(),
                "select_segmental_word_windows: word candidates"
            );
            candidates
        })
        .collect::<Vec<_>>();

    if candidates_per_word
        .iter()
        .any(|candidates| candidates.is_empty())
    {
        tracing::debug!(
            words_with_no_candidates = candidates_per_word.iter().filter(|c| c.is_empty()).count(),
            "select_segmental_word_windows: some words have no candidates, returning all None"
        );
        return std::iter::repeat_with(|| None)
            .take(transcript_word_tokens.len())
            .collect();
    }

    let mut dp = candidates_per_word
        .iter()
        .map(|candidates| vec![f32::NEG_INFINITY; candidates.len()])
        .collect::<Vec<_>>();
    let mut backpointers = candidates_per_word
        .iter()
        .map(|candidates| vec![None; candidates.len()])
        .collect::<Vec<_>>();

    for (candidate_index, candidate) in candidates_per_word[0].iter().enumerate() {
        dp[0][candidate_index] = candidate.local_score
            - gap_penalty(&utterance_zipa_normalized[..candidate.zipa_norm_range.start]);
    }

    for word_index in 1..candidates_per_word.len() {
        for (candidate_index, candidate) in candidates_per_word[word_index].iter().enumerate() {
            let mut best_score = f32::NEG_INFINITY;
            let mut best_prev = None;
            for (prev_index, prev) in candidates_per_word[word_index - 1].iter().enumerate() {
                if prev.zipa_norm_range.end > candidate.zipa_norm_range.start {
                    continue;
                }
                let transition_score = dp[word_index - 1][prev_index]
                    - boundary_gap_penalty(
                        &utterance_zipa_normalized
                            [prev.zipa_norm_range.end..candidate.zipa_norm_range.start],
                        prev,
                        candidate,
                        transcript_word_tokens[word_index - 1].as_slice(),
                        transcript_word_tokens[word_index].as_slice(),
                    )
                    + candidate.local_score;
                if transition_score > best_score {
                    best_score = transition_score;
                    best_prev = Some(prev_index);
                }
            }
            dp[word_index][candidate_index] = best_score;
            backpointers[word_index][candidate_index] = best_prev;
        }
    }

    // Log best DP score per word to see where the DP fails
    for (word_index, candidates) in candidates_per_word.iter().enumerate() {
        let best = dp[word_index]
            .iter()
            .zip(candidates.iter())
            .filter(|(s, _)| s.is_finite())
            .max_by(|(a, _), (b, _)| a.total_cmp(b));
        match best {
            Some((score, cand)) => tracing::debug!(
                word_index,
                score,
                range = ?cand.zipa_norm_range,
                "DP: best finite score for word"
            ),
            None => tracing::debug!(word_index, "DP: all candidates NEG_INFINITY for word"),
        }
    }

    // Find the last word index that has at least one finite DP score.
    // Words after it couldn't be placed (audio too short) and stay None.
    let traceback_start = (0..candidates_per_word.len())
        .rev()
        .find(|&wi| dp[wi].iter().any(|s| s.is_finite()));

    let Some(traceback_start) = traceback_start else {
        return std::iter::repeat_with(|| None)
            .take(transcript_word_tokens.len())
            .collect();
    };

    if traceback_start < candidates_per_word.len() - 1 {
        tracing::debug!(
            traceback_start,
            n_unreachable = candidates_per_word.len() - 1 - traceback_start,
            "select_segmental_word_windows: last words unreachable (audio too short), partial alignment"
        );
    }

    let mut best_last = None;
    let mut best_last_score = f32::NEG_INFINITY;
    for (candidate_index, candidate) in candidates_per_word[traceback_start].iter().enumerate() {
        let total_score = dp[traceback_start][candidate_index]
            - gap_penalty(&utterance_zipa_normalized[candidate.zipa_norm_range.end..]);
        if total_score > best_last_score {
            best_last_score = total_score;
            best_last = Some(candidate_index);
        }
    }

    let Some(mut candidate_index) = best_last else {
        return std::iter::repeat_with(|| None)
            .take(transcript_word_tokens.len())
            .collect();
    };

    let mut chosen = vec![None; transcript_word_tokens.len()];
    for word_index in (0..=traceback_start).rev() {
        let candidate = candidates_per_word[word_index][candidate_index].clone();
        chosen[word_index] = Some(WordAlignmentWindow {
            zipa_norm_range: candidate.zipa_norm_range,
            ops: candidate.alignment.ops,
        });
        if let Some(prev_index) = backpointers[word_index][candidate_index] {
            candidate_index = prev_index;
        } else if word_index != 0 {
            break;
        }
    }

    chosen
}

pub fn strict_project_left_range(
    ops: &[AlignmentOp],
    left_range: Range<usize>,
) -> Option<Range<usize>> {
    if left_range.start >= left_range.end {
        return None;
    }

    let mut matched_right = Vec::new();
    let mut right_before = None;
    let mut right_after = None;

    for op in ops {
        let left_index = op.left_index.map(|index| index as usize);
        let right_index = op.right_index.map(|index| index as usize);

        if let Some(left_index) = left_index {
            if left_range.contains(&left_index) {
                if let Some(right_index) = right_index {
                    matched_right.push(right_index);
                }
                continue;
            }

            if left_index < left_range.start {
                if let Some(right_index) = right_index {
                    right_before = Some(right_index);
                }
            } else if left_index >= left_range.end && right_after.is_none() {
                if let Some(right_index) = right_index {
                    right_after = Some(right_index);
                }
            }
        }
    }

    if let (Some(start), Some(end)) = (matched_right.first(), matched_right.last()) {
        return Some(*start..(*end + 1));
    }

    match (right_before, right_after) {
        (Some(start), Some(end)) if start < end => Some((start + 1)..end),
        (Some(start), Some(end)) => Some(start..(end + 1)),
        _ => None,
    }
}

pub fn partition_word_alignment_windows(
    ops: &[AlignmentOp],
    left_ranges: &[Range<usize>],
    transcript_word_tokens: &[Vec<String>],
) -> Vec<Option<WordAlignmentWindow>> {
    if left_ranges.is_empty() {
        return Vec::new();
    }

    let footprints = left_ranges
        .iter()
        .map(|left_range| {
            let positions = ops
                .iter()
                .enumerate()
                .filter_map(|(position, op)| {
                    let left_index = op.left_index.map(|index| index as usize)?;
                    left_range.contains(&left_index).then_some(position)
                })
                .collect::<Vec<_>>();
            positions
                .first()
                .zip(positions.last())
                .map(|(first, last)| *first..(*last + 1))
        })
        .collect::<Vec<_>>();

    let mut op_ranges = footprints.clone();
    let existing = footprints
        .iter()
        .enumerate()
        .filter_map(|(word_index, range)| range.as_ref().map(|range| (word_index, range.clone())))
        .collect::<Vec<_>>();

    if existing.is_empty() {
        return std::iter::repeat_with(|| None)
            .take(left_ranges.len())
            .collect();
    }

    let (first_index, first_range) = &existing[0];
    if first_range.start > 0 {
        op_ranges[*first_index] = Some(0..first_range.end);
    }

    let (last_index, last_range) = &existing[existing.len() - 1];
    if last_range.end < ops.len() {
        op_ranges[*last_index] = Some(last_range.start..ops.len());
    }

    for pair in existing.windows(2) {
        let (left_word_index, left_range) = &pair[0];
        let (right_word_index, right_range) = &pair[1];
        let boundary = choose_word_boundary(
            ops,
            left_range.end,
            right_range.start,
            transcript_word_tokens
                .get(*left_word_index)
                .map(Vec::as_slice)
                .unwrap_or(&[]),
            transcript_word_tokens
                .get(*right_word_index)
                .map(Vec::as_slice)
                .unwrap_or(&[]),
        );

        let left_start = op_ranges[*left_word_index]
            .as_ref()
            .map(|range| range.start)
            .unwrap_or(left_range.start);
        let right_end = op_ranges[*right_word_index]
            .as_ref()
            .map(|range| range.end)
            .unwrap_or(right_range.end);

        op_ranges[*left_word_index] = Some(left_start..boundary);
        op_ranges[*right_word_index] = Some(boundary..right_end);
    }

    op_ranges
        .into_iter()
        .map(|op_range| {
            let op_range = op_range?;
            let word_ops = ops.get(op_range)?.to_vec();
            let mut right_indices = word_ops
                .iter()
                .filter_map(|op| op.right_index.map(|index| index as usize))
                .collect::<Vec<_>>();
            if right_indices.is_empty() {
                return None;
            }
            right_indices.sort_unstable();
            let zipa_norm_range = right_indices[0]..(right_indices[right_indices.len() - 1] + 1);
            Some(WordAlignmentWindow {
                zipa_norm_range,
                ops: word_ops,
            })
        })
        .collect()
}

fn build_word_segment_candidates(
    transcript_tokens: &[String],
    transcript_token_range: Option<Range<usize>>,
    utterance_zipa_normalized: &[String],
    utterance_alignment: &TokenAlignment,
) -> Vec<WordSegmentCandidate> {
    if transcript_tokens.is_empty() || utterance_zipa_normalized.is_empty() {
        return Vec::new();
    }

    let mut candidate_ranges = Vec::<Range<usize>>::new();
    if let Some(transcript_token_range) = transcript_token_range.clone() {
        let projected = utterance_alignment.project_left_range(transcript_token_range.clone());
        tracing::debug!(
            tokens = %transcript_tokens.join(" "),
            token_range = ?transcript_token_range,
            projected_range = ?projected,
            "build_word_segment_candidates: projection"
        );
        if let Some(projected_range) = projected {
            candidate_ranges.push(projected_range);
        }
    }
    let anchor_windows = top_right_anchor_windows(transcript_tokens, utterance_zipa_normalized, 4);
    tracing::debug!(
        tokens = %transcript_tokens.join(" "),
        anchor_count = anchor_windows.len(),
        anchors = ?anchor_windows.iter().map(|w| w.right_start as usize..w.right_end as usize).collect::<Vec<_>>(),
        "build_word_segment_candidates: anchor windows"
    );
    candidate_ranges.extend(
        anchor_windows
            .into_iter()
            .map(|window| window.right_start as usize..window.right_end as usize),
    );
    let expanded = candidate_ranges
        .iter()
        .flat_map(|range| {
            let mut variants = vec![range.clone()];
            for shift in 1..=3 {
                variants.push(range.start.saturating_sub(shift)..range.end);
                variants
                    .push(range.start..(range.end + shift).min(utterance_zipa_normalized.len()));
                variants.push(
                    range.start.saturating_sub(shift)
                        ..(range.end + shift).min(utterance_zipa_normalized.len()),
                );
                if range.start + shift < range.end {
                    variants.push((range.start + shift)..range.end);
                }
                if range.end > range.start + shift {
                    variants.push(range.start..(range.end - shift));
                }
                if range.start + shift < range.end && range.end > range.start + shift {
                    variants.push(
                        (range.start + shift)..(range.end - shift).max(range.start + shift + 1),
                    );
                }
            }
            variants
        })
        .collect::<Vec<_>>();
    candidate_ranges.extend(expanded);
    dedup_candidate_ranges(&mut candidate_ranges, utterance_zipa_normalized.len());

    candidate_ranges
        .into_iter()
        .filter_map(|range| {
            let zipa_normalized = utterance_zipa_normalized.get(range.clone())?.to_vec();
            if zipa_normalized.is_empty() {
                return None;
            }
            let alignment = align_token_sequences(transcript_tokens, &zipa_normalized);
            let local_score =
                segment_local_alignment_score(transcript_tokens, &zipa_normalized, &alignment.ops);
            Some(WordSegmentCandidate {
                zipa_norm_range: range,
                alignment,
                local_score,
            })
        })
        .collect()
}

fn segment_local_alignment_score(
    transcript_tokens: &[String],
    zipa_tokens: &[String],
    ops: &[AlignmentOp],
) -> f32 {
    let mut score = alignment_quality_score(ops, transcript_tokens.len(), zipa_tokens.len());
    let leading_deletes = ops
        .iter()
        .take_while(|op| op.left_index.is_some() && op.right_index.is_none())
        .count();
    let trailing_deletes = ops
        .iter()
        .rev()
        .take_while(|op| op.left_index.is_some() && op.right_index.is_none())
        .count();
    let leading_inserts = ops
        .iter()
        .take_while(|op| op.left_index.is_none() && op.right_index.is_some())
        .count();
    let trailing_inserts = ops
        .iter()
        .rev()
        .take_while(|op| op.left_index.is_none() && op.right_index.is_some())
        .count();
    score -= leading_deletes as f32 * 0.22;
    score -= trailing_deletes as f32 * 0.22;
    score -= insert_run_penalty(
        ops.iter()
            .take(leading_inserts)
            .filter_map(|op| op.right_token.as_deref()),
    );
    score -= insert_run_penalty(
        ops.iter()
            .rev()
            .take(trailing_inserts)
            .filter_map(|op| op.right_token.as_deref()),
    );

    if let Some(first) = transcript_tokens.first() {
        if let Some(first_right) = ops.iter().find_map(|op| op.right_token.as_deref()) {
            score += token_affinity(first, first_right) * 0.18;
        }
    }
    if let Some(last) = transcript_tokens.last() {
        if let Some(last_right) = ops.iter().rev().find_map(|op| op.right_token.as_deref()) {
            score += token_affinity(last, last_right) * 0.18;
        }
    }

    if let Some((first_left, first_right)) = first_aligned_pair(ops) {
        score += token_affinity(first_left, first_right) * 0.15;
    }
    if let Some((last_left, last_right)) = last_aligned_pair(ops) {
        score += token_affinity(last_left, last_right) * 0.15;
    }

    score
}

fn boundary_gap_penalty(
    gap_tokens: &[String],
    prev: &WordSegmentCandidate,
    next: &WordSegmentCandidate,
    prev_transcript: &[String],
    next_transcript: &[String],
) -> f32 {
    let mut penalty = gap_penalty(gap_tokens) * 1.35;
    if let (Some(last_gap), Some(prev_last)) = (gap_tokens.last(), prev_transcript.last()) {
        penalty += token_affinity(last_gap, prev_last) * 0.45;
    }
    if let (Some(first_gap), Some(next_first)) = (gap_tokens.first(), next_transcript.first()) {
        penalty += token_affinity(first_gap, next_first) * 0.75;
    }
    if gap_tokens.len() >= 2 {
        let prefix_affinity = gap_prefix_affinity(gap_tokens, next_transcript);
        let suffix_affinity = gap_suffix_affinity(gap_tokens, prev_transcript);
        penalty += prefix_affinity * 0.55;
        penalty += suffix_affinity * 0.35;
    }
    penalty += range_compression_penalty(prev, next);
    penalty
}

fn range_compression_penalty(prev: &WordSegmentCandidate, next: &WordSegmentCandidate) -> f32 {
    let gap = next
        .zipa_norm_range
        .start
        .saturating_sub(prev.zipa_norm_range.end);
    if gap == 0 { 0.0 } else { (gap as f32) * 0.03 }
}

fn gap_penalty(tokens: &[String]) -> f32 {
    tokens
        .iter()
        .map(|token| {
            if is_weak_vowelish(token) {
                0.12
            } else if is_vowelish(token) {
                0.25
            } else {
                0.55
            }
        })
        .sum()
}

fn insert_run_penalty<'a>(tokens: impl Iterator<Item = &'a str>) -> f32 {
    tokens
        .map(|token| {
            if is_weak_vowelish(token) {
                0.08
            } else if is_vowelish(token) {
                0.22
            } else {
                0.5
            }
        })
        .sum()
}

fn token_affinity(left: &str, right: &str) -> f32 {
    feature_similarity(&[left.to_string()], &[right.to_string()])
        .or_else(|| phoneme_similarity(&[left.to_string()], &[right.to_string()]))
        .unwrap_or(0.0)
        .max(0.0)
}

fn first_aligned_pair<'a>(ops: &'a [AlignmentOp]) -> Option<(&'a str, &'a str)> {
    ops.iter()
        .find_map(|op| Some((op.left_token.as_deref()?, op.right_token.as_deref()?)))
}

fn last_aligned_pair<'a>(ops: &'a [AlignmentOp]) -> Option<(&'a str, &'a str)> {
    ops.iter()
        .rev()
        .find_map(|op| Some((op.left_token.as_deref()?, op.right_token.as_deref()?)))
}

fn gap_prefix_affinity(gap_tokens: &[String], next_transcript: &[String]) -> f32 {
    let take = gap_tokens.len().min(next_transcript.len()).min(2);
    if take == 0 {
        return 0.0;
    }
    feature_similarity(&gap_tokens[..take], &next_transcript[..take])
        .or_else(|| phoneme_similarity(&gap_tokens[..take], &next_transcript[..take]))
        .unwrap_or(0.0)
        .max(0.0)
}

fn gap_suffix_affinity(gap_tokens: &[String], prev_transcript: &[String]) -> f32 {
    let take = gap_tokens.len().min(prev_transcript.len()).min(2);
    if take == 0 {
        return 0.0;
    }
    let gap_slice = &gap_tokens[gap_tokens.len() - take..];
    let prev_slice = &prev_transcript[prev_transcript.len() - take..];
    feature_similarity(gap_slice, prev_slice)
        .or_else(|| phoneme_similarity(gap_slice, prev_slice))
        .unwrap_or(0.0)
        .max(0.0)
}

fn is_weak_vowelish(token: &str) -> bool {
    matches!(token, "ə" | "ɚ" | "ɝ")
}

fn is_vowelish(token: &str) -> bool {
    matches!(
        token,
        "a" | "ɑ" | "ɔ" | "ɛ" | "ə" | "ɪ" | "ʊ" | "i" | "u" | "e" | "o" | "æ" | "ʌ" | "ɚ" | "ɝ"
    )
}

fn choose_word_boundary(
    ops: &[AlignmentOp],
    left_end: usize,
    right_start: usize,
    prev_word_tokens: &[String],
    next_word_tokens: &[String],
) -> usize {
    if left_end >= right_start {
        return right_start;
    }

    let segment_start = left_end;
    let segment = &ops[segment_start..right_start];
    if segment
        .iter()
        .all(|op| op.left_index.is_none() && op.right_index.is_some())
    {
        return segment_start;
    }

    let mut best_split = 0usize;
    let mut best_score = f32::NEG_INFINITY;

    for split in 0..=segment.len() {
        let left_score = boundary_side_affinity(&segment[..split], prev_word_tokens, false);
        let right_score = boundary_side_affinity(&segment[split..], next_word_tokens, true);
        let score = left_score + right_score;
        if score > best_score || (score == best_score && split < best_split) {
            best_score = score;
            best_split = split;
        }
    }

    segment_start + best_split
}

fn boundary_side_affinity(
    segment: &[AlignmentOp],
    neighbor_tokens: &[String],
    prefix: bool,
) -> f32 {
    let right_tokens = segment
        .iter()
        .filter_map(|op| op.right_token.clone())
        .collect::<Vec<_>>();
    if right_tokens.is_empty() || neighbor_tokens.is_empty() {
        return 0.0;
    }

    let take = right_tokens.len().min(neighbor_tokens.len());
    let neighbor_slice = if prefix {
        &neighbor_tokens[..take]
    } else {
        &neighbor_tokens[neighbor_tokens.len() - take..]
    };
    let right_slice = if prefix {
        &right_tokens[..take]
    } else {
        &right_tokens[right_tokens.len() - take..]
    };

    feature_similarity(right_slice, neighbor_slice)
        .or_else(|| phoneme_similarity(right_slice, neighbor_slice))
        .unwrap_or(0.0)
        .max(0.0)
}

fn normalize_candidate_ranges(ranges: &mut Vec<Range<usize>>, utterance_len: usize) {
    for range in ranges.iter_mut() {
        let start = range.start.saturating_sub(1);
        let end = (range.end + 1).min(utterance_len);
        *range = start..end;
    }
    ranges.sort_by(|a, b| a.start.cmp(&b.start).then_with(|| a.end.cmp(&b.end)));
    ranges.dedup_by(|a, b| a.start == b.start && a.end == b.end);
}

fn dedup_candidate_ranges(ranges: &mut Vec<Range<usize>>, utterance_len: usize) {
    for range in ranges.iter_mut() {
        range.start = range.start.min(utterance_len);
        range.end = range.end.min(utterance_len);
        if range.end < range.start {
            range.end = range.start;
        }
    }
    ranges.retain(|range| range.start < range.end);
    ranges.sort_by(|a, b| a.start.cmp(&b.start).then_with(|| a.end.cmp(&b.end)));
    ranges.dedup_by(|a, b| a.start == b.start && a.end == b.end);
}

pub fn alignment_quality_score(ops: &[AlignmentOp], left_len: usize, right_len: usize) -> f32 {
    let denom = left_len.max(right_len).max(1) as f32;
    let total_cost = ops.iter().map(|op| op.cost).sum::<f32>();
    let compression_penalty =
        (left_len.saturating_sub(right_len) as f32 / left_len.max(1) as f32).max(0.0) * 0.35;
    (1.0 - (total_cost / denom) - compression_penalty).clamp(0.0, 1.0)
}

pub fn normalize_zipa_raw_for_alignment(
    raw_tokens: &[String],
) -> (Vec<String>, Vec<bee_phonetic::ComparisonToken>) {
    let with_spans = normalize_ipa_for_comparison_with_spans(raw_tokens);
    let normalized = with_spans.iter().map(|token| token.token.clone()).collect();
    (normalized, with_spans)
}

#[cfg(test)]
mod tests {
    use super::{
        partition_word_alignment_windows, select_segmental_word_windows,
        timed_range_for_normalized_range,
    };
    use bee_phonetic::{
        AlignmentOp, AlignmentOpKind, ComparisonToken,
        align_token_sequences_with_left_word_boundaries,
    };
    use bee_zipa_mlx::infer::PhoneSpan;

    fn op(
        kind: AlignmentOpKind,
        left_index: Option<u32>,
        right_index: Option<u32>,
        left_token: Option<&str>,
        right_token: Option<&str>,
    ) -> AlignmentOp {
        AlignmentOp {
            kind,
            left_index,
            right_index,
            left_token: left_token.map(ToOwned::to_owned),
            right_token: right_token.map(ToOwned::to_owned),
            cost: 0.0,
        }
    }

    #[test]
    fn word_alignment_windows_partition_insertions_without_overlap() {
        let ops = vec![
            op(
                AlignmentOpKind::Match,
                Some(0),
                Some(0),
                Some("m"),
                Some("m"),
            ),
            op(AlignmentOpKind::Insert, None, Some(1), None, Some("ɪ")),
            op(
                AlignmentOpKind::Substitute,
                Some(1),
                Some(2),
                Some("ɹ"),
                Some("ə"),
            ),
            op(
                AlignmentOpKind::Match,
                Some(2),
                Some(3),
                Some("ɪ"),
                Some("ɪ"),
            ),
            op(AlignmentOpKind::Insert, None, Some(4), None, Some("n")),
            op(
                AlignmentOpKind::Match,
                Some(3),
                Some(5),
                Some("t"),
                Some("t"),
            ),
        ];

        let windows = partition_word_alignment_windows(
            &ops,
            &[0..2, 2..4],
            &[
                vec!["m".to_string(), "ɹ".to_string()],
                vec!["ɪ".to_string(), "t".to_string()],
            ],
        );
        let first = windows[0].as_ref().unwrap();
        let second = windows[1].as_ref().unwrap();

        assert_eq!(first.zipa_norm_range, 0..3);
        assert_eq!(second.zipa_norm_range, 3..6);
        assert_eq!(first.ops.len(), 3);
        assert_eq!(second.ops.len(), 3);
    }

    #[test]
    fn segmental_word_windows_choose_non_overlapping_ranges() {
        let transcript = vec![
            vec!["m".to_string(), "ɹ".to_string()],
            vec!["ɪ".to_string(), "t".to_string()],
        ];
        let transcript_ranges = vec![0..2, 2..4];
        let transcript_flat = transcript.iter().flatten().cloned().collect::<Vec<_>>();
        let left_word_ids = vec![0usize, 0, 1, 1];
        let utterance = vec![
            "m".to_string(),
            "ɪ".to_string(),
            "ə".to_string(),
            "ɪ".to_string(),
            "n".to_string(),
            "t".to_string(),
        ];
        let alignment = align_token_sequences_with_left_word_boundaries(
            &transcript_flat,
            &utterance,
            &left_word_ids,
        );
        let windows =
            select_segmental_word_windows(&transcript, &transcript_ranges, &utterance, &alignment);
        let first = windows[0].as_ref().unwrap();
        let second = windows[1].as_ref().unwrap();
        assert!(first.zipa_norm_range.end <= second.zipa_norm_range.start);
    }

    #[test]
    fn maps_normalized_range_to_phone_timestamps() {
        let normalized = vec![
            ComparisonToken {
                token: "t".to_string(),
                source_start: 0,
                source_end: 1,
            },
            ComparisonToken {
                token: "ʃ".to_string(),
                source_start: 1,
                source_end: 2,
            },
            ComparisonToken {
                token: "a".to_string(),
                source_start: 2,
                source_end: 3,
            },
            ComparisonToken {
                token: "ɪ".to_string(),
                source_start: 2,
                source_end: 3,
            },
        ];
        let phone_spans = vec![
            PhoneSpan {
                token_id: 1,
                token: "t".to_string(),
                start_frame: 10,
                end_frame: 12,
                start_time_secs: 0.10,
                end_time_secs: 0.12,
            },
            PhoneSpan {
                token_id: 2,
                token: "ʃ".to_string(),
                start_frame: 12,
                end_frame: 14,
                start_time_secs: 0.12,
                end_time_secs: 0.14,
            },
            PhoneSpan {
                token_id: 3,
                token: "aɪ".to_string(),
                start_frame: 14,
                end_frame: 15,
                start_time_secs: 0.14,
                end_time_secs: 0.15,
            },
        ];

        let timed = timed_range_for_normalized_range(&normalized, &phone_spans, 1..4).unwrap();
        assert_eq!(timed.normalized_range, 1..4);
        assert_eq!(timed.raw_phone_range, 1..3);
        assert!((timed.start_time_secs - 0.12).abs() < 1e-6);
        assert!((timed.end_time_secs - 0.15).abs() < 1e-6);
    }
}
