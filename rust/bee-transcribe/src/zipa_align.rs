use std::collections::HashMap;
use std::ops::Range;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use crate::g2p::CachedEspeakG2p;
use crate::timing::{log_phase_chunk, phase_start};
use bee_phonetic::{
    AlignmentOp, ComparisonToken, TokenAlignment, align_token_sequences,
    align_token_sequences_with_left_word_boundaries, feature_similarity_for_tokens,
    normalize_ipa_for_comparison, normalize_ipa_for_comparison_with_spans, phoneme_similarity,
    sentence_word_tokens, top_right_anchor_windows,
};
use bee_zipa_mlx::audio::AudioBuffer as ZipaAudioBuffer;
use bee_zipa_mlx::infer::{PhoneSpan, ZipaInference};

#[derive(Clone, Debug)]
pub struct CachedZipaOutput {
    raw_token_count: usize,
    zipa_raw: Vec<String>,
    zipa_norm_with_spans: Vec<ComparisonToken>,
    phone_spans: Vec<PhoneSpan>,
}

impl CachedZipaOutput {
    pub fn raw_token_count(&self) -> usize {
        self.raw_token_count
    }

    pub fn append(&mut self, tail: CachedZipaOutput) {
        self.raw_token_count += tail.raw_token_count;
        self.zipa_raw.extend(tail.zipa_raw);
        self.zipa_norm_with_spans.extend(tail.zipa_norm_with_spans);
        self.phone_spans.extend(tail.phone_spans);
    }

    pub fn trim_front(&mut self, cut_secs: f64) {
        let drop_count = self
            .phone_spans
            .partition_point(|span| span.start_time_secs < cut_secs);
        if drop_count == 0 {
            return;
        }

        self.raw_token_count = self.raw_token_count.saturating_sub(drop_count);
        self.zipa_raw.drain(..drop_count);
        self.phone_spans.drain(..drop_count);
        for span in &mut self.phone_spans {
            span.start_time_secs -= cut_secs;
            span.end_time_secs -= cut_secs;
        }
        self.zipa_norm_with_spans = normalize_ipa_for_comparison_with_spans(&self.zipa_raw);
    }
}

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

#[derive(Debug, Clone, PartialEq)]
pub enum ComparisonRangeTiming {
    Invalid,
    Deleted { projected_at: usize },
    NoTiming { projected_range: Range<usize> },
    Aligned(TimedZipaRange),
}

#[derive(Debug, Clone)]
pub struct WordAlignmentWindow {
    pub zipa_norm_range: Range<usize>,
    pub ops: Vec<AlignmentOp>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TokenPieceTiming {
    pub token_index: usize,
    pub token: String,
    pub token_surface: String,
    pub token_char_start: usize,
    pub token_char_end: usize,
    pub word_index: Option<usize>,
    pub word_surface: Option<String>,
    pub timing: ComparisonRangeTiming,
}

#[derive(Debug, Clone)]
pub struct TranscriptComparisonInput {
    pub transcript: String,
    pub word_char_ranges: Vec<Range<usize>>,
    pub word_normalized_ranges: Vec<Range<usize>>,
    pub word_tokens: Vec<Vec<String>>,
    pub transcript_normalized: Vec<String>,
}

#[derive(Clone, Debug)]
struct WordSegmentCandidate {
    zipa_norm_range: Range<usize>,
    alignment: TokenAlignment,
    local_score: f32,
}

#[derive(Debug, Default)]
struct SegmentalWindowStats {
    candidate_counts_before: Vec<usize>,
    candidate_counts_after: Vec<usize>,
    dp_states_explored: usize,
    token_affinity_ms: f64,
}

#[derive(Debug, Default)]
struct CandidateBuildStats {
    raw_candidate_count: usize,
    pruned_candidate_count: usize,
    prune_ms: f64,
}

#[derive(Debug, Default)]
struct CandidateBuildResult {
    stats: CandidateBuildStats,
    candidates: Vec<WordSegmentCandidate>,
}

static SMALL_FEATURE_SIMILARITY_CACHE: OnceLock<
    Mutex<HashMap<(Vec<String>, Vec<String>), Option<f32>>>,
> = OnceLock::new();

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

pub fn transcript_comparison_input_from_word_raw_ranges(
    transcript: &str,
    word_raw_ranges: &[(Range<usize>, Vec<String>)],
) -> TranscriptComparisonInput {
    let word_tokens: Vec<Vec<String>> = word_raw_ranges
        .iter()
        .map(|(_, raw)| normalize_ipa_for_comparison(raw))
        .collect();
    let transcript_normalized: Vec<String> = word_tokens
        .iter()
        .flat_map(|tokens| tokens.iter().cloned())
        .collect();
    let mut cursor = 0usize;
    let word_normalized_ranges = word_tokens
        .iter()
        .map(|tokens| {
            let start = cursor;
            cursor += tokens.len();
            start..cursor
        })
        .collect();
    let word_char_ranges = word_raw_ranges
        .iter()
        .map(|(range, _)| range.clone())
        .collect();
    TranscriptComparisonInput {
        transcript: transcript.to_owned(),
        word_char_ranges,
        word_normalized_ranges,
        word_tokens,
        transcript_normalized,
    }
}

pub fn transcript_comparison_input_from_g2p(
    transcript: &str,
    input: &bee_g2p::TranscriptAlignmentInput,
) -> TranscriptComparisonInput {
    let word_char_ranges = input
        .words
        .iter()
        .map(|word| word.char_start..word.char_end)
        .collect::<Vec<_>>();
    let word_normalized_ranges = input
        .words
        .iter()
        .map(|word| word.comparison_start..word.comparison_end)
        .collect::<Vec<_>>();
    let word_tokens = input
        .words
        .iter()
        .map(|word| input.normalized[word.comparison_start..word.comparison_end].to_vec())
        .collect::<Vec<_>>();
    TranscriptComparisonInput {
        transcript: transcript.to_owned(),
        word_char_ranges,
        word_normalized_ranges,
        word_tokens,
        transcript_normalized: input.normalized.clone(),
    }
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
    let mut stats = SegmentalWindowStats::default();
    if transcript_word_tokens.is_empty() {
        return Vec::new();
    }

    let build_start = phase_start();
    let mut candidate_build_ms = 0.0f64;
    let mut candidate_prune_ms = 0.0f64;
    let candidates_per_word = transcript_word_tokens
        .iter()
        .enumerate()
        .map(|(word_index, transcript_tokens)| {
            let candidate_start = Instant::now();
            let candidates = build_word_segment_candidates(
                transcript_tokens,
                transcript_token_ranges.get(word_index).cloned(),
                utterance_zipa_normalized,
                utterance_alignment,
            );
            candidate_build_ms += candidate_start.elapsed().as_secs_f64() * 1000.0;
            candidate_prune_ms += candidates.stats.prune_ms;
            stats
                .candidate_counts_before
                .push(candidates.stats.raw_candidate_count);
            stats
                .candidate_counts_after
                .push(candidates.stats.pruned_candidate_count);
            tracing::debug!(
                word_index,
                transcript_tokens = %transcript_tokens.join(" "),
                token_range = ?transcript_token_ranges.get(word_index),
                candidates_before = candidates.stats.raw_candidate_count,
                candidates_after = candidates.stats.pruned_candidate_count,
                "select_segmental_word_windows: word candidates"
            );
            candidates.candidates
        })
        .collect::<Vec<_>>();
    log_phase_chunk(
        "zipa_align",
        "select_segmental_word_windows_candidates",
        candidates_per_word.len(),
        build_start,
    );

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

    let dp_start = phase_start();
    for word_index in 1..candidates_per_word.len() {
        for (candidate_index, candidate) in candidates_per_word[word_index].iter().enumerate() {
            let mut best_score = f32::NEG_INFINITY;
            let mut best_prev = None;
            for (prev_index, prev) in candidates_per_word[word_index - 1].iter().enumerate() {
                stats.dp_states_explored += 1;
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
                        &mut stats,
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
    log_phase_chunk(
        "zipa_align",
        "select_segmental_word_windows_dp",
        candidates_per_word.len(),
        dp_start,
    );
    tracing::debug!(
        candidate_counts_before = ?stats.candidate_counts_before,
        candidate_counts_after = ?stats.candidate_counts_after,
        candidate_build_ms,
        candidate_prune_ms,
        dp_states_explored = stats.dp_states_explored,
        token_affinity_ms = stats.token_affinity_ms,
        "select_segmental_word_windows: summary"
    );

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
) -> CandidateBuildResult {
    if transcript_tokens.is_empty() || utterance_zipa_normalized.is_empty() {
        return CandidateBuildResult::default();
    }

    let mut candidate_ranges = Vec::<Range<usize>>::new();
    candidate_ranges.reserve(32);
    let projected_range = transcript_token_range
        .clone()
        .and_then(|transcript_token_range| {
            utterance_alignment
                .project_left_range(transcript_token_range)
                .map(|range| {
                    expand_degenerate_projected_range(
                        range,
                        transcript_tokens.len(),
                        utterance_zipa_normalized.len(),
                    )
                })
        });
    if let Some(transcript_token_range) = transcript_token_range.clone() {
        tracing::debug!(
            tokens = %transcript_tokens.join(" "),
            token_range = ?transcript_token_range,
            projected_range = ?projected_range,
            "build_word_segment_candidates: projection"
        );
        if let Some(projected_range) = projected_range.clone() {
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
    for range in candidate_ranges.clone() {
        for shift in 1..=3 {
            candidate_ranges.push(range.start.saturating_sub(shift)..range.end);
            candidate_ranges
                .push(range.start..(range.end + shift).min(utterance_zipa_normalized.len()));
            candidate_ranges.push(
                range.start.saturating_sub(shift)
                    ..(range.end + shift).min(utterance_zipa_normalized.len()),
            );
            if range.start + shift < range.end {
                candidate_ranges.push((range.start + shift)..range.end);
            }
            if range.end > range.start + shift {
                candidate_ranges.push(range.start..(range.end - shift));
            }
            if range.start + shift < range.end {
                candidate_ranges
                    .push((range.start + shift)..(range.end - shift).max(range.start + shift + 1));
            }
        }
    }
    dedup_candidate_ranges(&mut candidate_ranges, utterance_zipa_normalized.len());
    let raw_candidate_count = candidate_ranges.len();
    let prune_start = Instant::now();
    let candidate_ranges = prune_candidate_ranges(
        candidate_ranges,
        transcript_tokens,
        projected_range.as_ref(),
        utterance_zipa_normalized,
    );
    let prune_ms = prune_start.elapsed().as_secs_f64() * 1000.0;

    let candidates = candidate_ranges
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
        .collect::<Vec<_>>();

    CandidateBuildResult {
        stats: CandidateBuildStats {
            raw_candidate_count,
            pruned_candidate_count: candidates.len(),
            prune_ms,
        },
        candidates,
    }
}

fn expand_degenerate_projected_range(
    range: Range<usize>,
    transcript_len: usize,
    utterance_len: usize,
) -> Range<usize> {
    if range.start != range.end {
        return range;
    }

    let width = transcript_len.clamp(1, 4);
    let start = range.start.min(utterance_len);
    let end = (start + width).min(utterance_len);
    if end > start {
        start..end
    } else {
        start.saturating_sub(width.min(start))..start
    }
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
            score += token_affinity_cached(first, first_right) * 0.18;
        }
    }
    if let Some(last) = transcript_tokens.last() {
        if let Some(last_right) = ops.iter().rev().find_map(|op| op.right_token.as_deref()) {
            score += token_affinity_cached(last, last_right) * 0.18;
        }
    }

    if let Some((first_left, first_right)) = first_aligned_pair(ops) {
        score += token_affinity_cached(first_left, first_right) * 0.15;
    }
    if let Some((last_left, last_right)) = last_aligned_pair(ops) {
        score += token_affinity_cached(last_left, last_right) * 0.15;
    }

    score
}

fn boundary_gap_penalty(
    gap_tokens: &[String],
    prev: &WordSegmentCandidate,
    next: &WordSegmentCandidate,
    prev_transcript: &[String],
    next_transcript: &[String],
    stats: &mut SegmentalWindowStats,
) -> f32 {
    let mut penalty = gap_penalty(gap_tokens) * 1.35;
    if let (Some(last_gap), Some(prev_last)) = (gap_tokens.last(), prev_transcript.last()) {
        penalty += token_affinity(last_gap, prev_last, stats) * 0.45;
    }
    if let (Some(first_gap), Some(next_first)) = (gap_tokens.first(), next_transcript.first()) {
        penalty += token_affinity(first_gap, next_first, stats) * 0.75;
    }
    if gap_tokens.len() >= 2 {
        let prefix_affinity = gap_prefix_affinity(gap_tokens, next_transcript, stats);
        let suffix_affinity = gap_suffix_affinity(gap_tokens, prev_transcript, stats);
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

fn token_affinity(left: &str, right: &str, stats: &mut SegmentalWindowStats) -> f32 {
    let started = Instant::now();
    let score = token_affinity_cached(left, right);
    stats.token_affinity_ms += started.elapsed().as_secs_f64() * 1000.0;
    score
}

fn token_affinity_cached(left: &str, right: &str) -> f32 {
    feature_similarity_for_tokens(left, right)
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

fn gap_prefix_affinity(
    gap_tokens: &[String],
    next_transcript: &[String],
    stats: &mut SegmentalWindowStats,
) -> f32 {
    let take = gap_tokens.len().min(next_transcript.len()).min(2);
    if take == 0 {
        return 0.0;
    }
    small_feature_similarity_timed(&gap_tokens[..take], &next_transcript[..take], stats)
        .or_else(|| phoneme_similarity(&gap_tokens[..take], &next_transcript[..take]))
        .unwrap_or(0.0)
        .max(0.0)
}

fn gap_suffix_affinity(
    gap_tokens: &[String],
    prev_transcript: &[String],
    stats: &mut SegmentalWindowStats,
) -> f32 {
    let take = gap_tokens.len().min(prev_transcript.len()).min(2);
    if take == 0 {
        return 0.0;
    }
    let gap_slice = &gap_tokens[gap_tokens.len() - take..];
    let prev_slice = &prev_transcript[prev_transcript.len() - take..];
    small_feature_similarity_timed(gap_slice, prev_slice, stats)
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

    small_feature_similarity_cached(right_slice, neighbor_slice)
        .or_else(|| phoneme_similarity(right_slice, neighbor_slice))
        .unwrap_or(0.0)
        .max(0.0)
}

fn small_feature_similarity_timed(
    a: &[String],
    b: &[String],
    stats: &mut SegmentalWindowStats,
) -> Option<f32> {
    let started = Instant::now();
    let result = small_feature_similarity_cached(a, b);
    stats.token_affinity_ms += started.elapsed().as_secs_f64() * 1000.0;
    result
}

fn small_feature_similarity_cached(a: &[String], b: &[String]) -> Option<f32> {
    if a.is_empty() || b.is_empty() {
        return None;
    }

    if a.len() == 1 && b.len() == 1 {
        feature_similarity_for_tokens(&a[0], &b[0])
    } else {
        let key = if a <= b {
            (a.to_vec(), b.to_vec())
        } else {
            (b.to_vec(), a.to_vec())
        };
        let cache = SMALL_FEATURE_SIMILARITY_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        {
            let guard = cache
                .lock()
                .expect("small feature similarity cache poisoned");
            if let Some(cached) = guard.get(&key) {
                return *cached;
            }
        }
        let similarity = bee_phonetic::feature_similarity(a, b);
        cache
            .lock()
            .expect("small feature similarity cache poisoned")
            .insert(key, similarity);
        similarity
    }
}

fn prune_candidate_ranges(
    candidate_ranges: Vec<Range<usize>>,
    transcript_tokens: &[String],
    projected_range: Option<&Range<usize>>,
    utterance_zipa_normalized: &[String],
) -> Vec<Range<usize>> {
    if candidate_ranges.len() <= candidate_cap_for_word(transcript_tokens.len()) {
        return candidate_ranges;
    }

    let mut scored = candidate_ranges
        .into_iter()
        .map(|range| {
            let score = candidate_range_pre_score(
                transcript_tokens,
                utterance_zipa_normalized,
                &range,
                projected_range,
            );
            (range, score)
        })
        .collect::<Vec<_>>();
    scored.sort_by(|a, b| {
        b.1.total_cmp(&a.1)
            .then_with(|| a.0.start.cmp(&b.0.start))
            .then_with(|| a.0.end.cmp(&b.0.end))
    });
    let cap = candidate_cap_for_word(transcript_tokens.len()).min(scored.len());
    scored.truncate(cap);
    scored
        .into_iter()
        .map(|(range, _)| range)
        .collect::<Vec<_>>()
}

fn candidate_cap_for_word(word_len: usize) -> usize {
    match word_len {
        0..=2 => 6,
        3..=4 => 8,
        5..=7 => 10,
        _ => 12,
    }
}

fn candidate_range_pre_score(
    transcript_tokens: &[String],
    utterance_zipa_normalized: &[String],
    range: &Range<usize>,
    projected_range: Option<&Range<usize>>,
) -> f32 {
    let candidate = &utterance_zipa_normalized[range.clone()];
    let mut score = 0.0;

    let length_delta = (candidate.len() as isize - transcript_tokens.len() as isize).abs() as f32;
    score -= length_delta * 0.4;
    score -= (range.start as f32) * 0.01;

    if let Some(projected_range) = projected_range {
        let overlap_start = range.start.max(projected_range.start);
        let overlap_end = range.end.min(projected_range.end);
        if overlap_start < overlap_end {
            score += (overlap_end - overlap_start) as f32 * 0.8;
        }
        let range_center = (range.start + range.end) as f32 * 0.5;
        let projected_center = (projected_range.start + projected_range.end) as f32 * 0.5;
        score -= (range_center - projected_center).abs() * 0.12;
        if range.start == projected_range.start && range.end == projected_range.end {
            score += 8.0;
        }
    }

    if let Some(first) = transcript_tokens.first() {
        if let Some(candidate_first) = candidate.first() {
            score += token_affinity_cached(first, candidate_first) * 0.45;
        }
    }
    if let Some(last) = transcript_tokens.last() {
        if let Some(candidate_last) = candidate.last() {
            score += token_affinity_cached(last, candidate_last) * 0.45;
        }
    }

    let take = transcript_tokens.len().min(candidate.len()).min(2);
    if take > 1 {
        let transcript_prefix = &transcript_tokens[..take];
        let candidate_prefix = &candidate[..take];
        let transcript_suffix = &transcript_tokens[transcript_tokens.len() - take..];
        let candidate_suffix = &candidate[candidate.len() - take..];
        score += small_feature_similarity_cached(transcript_prefix, candidate_prefix)
            .unwrap_or(0.0)
            * 0.25;
        score += small_feature_similarity_cached(transcript_suffix, candidate_suffix)
            .unwrap_or(0.0)
            * 0.25;
    }

    score
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

// ---------------------------------------------------------------------------
// High-level alignment API
// ---------------------------------------------------------------------------

/// Error returned by [`TranscriptAlignment::build`].
#[derive(Debug)]
pub enum AlignmentError {
    G2p(String),
    Zipa(String),
}

impl std::fmt::Display for AlignmentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlignmentError::G2p(e) => write!(f, "G2P error: {e}"),
            AlignmentError::Zipa(e) => write!(f, "ZIPA inference error: {e}"),
        }
    }
}

/// Per-word timing result.
#[derive(Debug, Clone)]
pub enum AlignmentQuality {
    Aligned {
        start_secs: f64,
        end_secs: f64,
    },
    /// DP alignment produced no window for this word (audio too short or severe phoneme mismatch).
    NoWindow,
    /// Window found but the raw-phone range resolved to empty (degenerate normalization).
    NoTiming,
}

/// A word paired with its alignment quality.
#[derive(Debug)]
pub struct WordTiming<'a> {
    pub word: &'a str,
    pub quality: AlignmentQuality,
}

/// Result of [`TranscriptAlignment::span_timing`].
#[derive(Debug, Clone, PartialEq)]
pub enum SpanTiming {
    Aligned {
        start_secs: f64,
        end_secs: f64,
    },
    /// No word in the requested range has a resolved alignment window.
    NoAlignedWords,
    /// Some but not all words in the range have windows; timing covers only the aligned subset.
    PartialGap {
        start_secs: f64,
        end_secs: f64,
    },
    /// All windows found but the merged raw-phone range resolved to empty.
    NoTiming,
}

/// Pre-computed ZIPA alignment for a transcript + audio pair.
///
/// Construct once with [`TranscriptAlignment::build`]; query word or span
/// timings without touching ZIPA or G2P again.
pub struct TranscriptAlignment {
    transcript: String,
    /// Byte-range of each word inside `transcript` (same order as `word_windows`).
    word_char_ranges: Vec<Range<usize>>,
    transcript_alignment: TokenAlignment,
    transcript_normalized_len: usize,
    word_windows: Vec<Option<WordAlignmentWindow>>,
    zipa_norm_with_spans: Vec<ComparisonToken>,
    phone_spans: Vec<PhoneSpan>,
}

impl TranscriptAlignment {
    pub fn build_from_comparison_input_and_zipa(
        input: TranscriptComparisonInput,
        zipa_norm_with_spans: Vec<ComparisonToken>,
        phone_spans: Vec<PhoneSpan>,
    ) -> Self {
        let zipa_norm: Vec<String> = zipa_norm_with_spans
            .iter()
            .map(|t| t.token.clone())
            .collect();

        let word_ids: Vec<usize> = input
            .word_tokens
            .iter()
            .enumerate()
            .flat_map(|(wi, tokens)| std::iter::repeat_n(wi, tokens.len()))
            .collect();

        let alignment = align_token_sequences_with_left_word_boundaries(
            &input.transcript_normalized,
            &zipa_norm,
            &word_ids,
        );

        let word_windows = select_segmental_word_windows(
            &input.word_tokens,
            &input.word_normalized_ranges,
            &zipa_norm,
            &alignment,
        );

        Self {
            transcript: input.transcript,
            word_char_ranges: input.word_char_ranges,
            transcript_alignment: alignment,
            transcript_normalized_len: input.transcript_normalized.len(),
            word_windows,
            zipa_norm_with_spans,
            phone_spans,
        }
    }

    pub fn build_from_cached_zipa(
        input: TranscriptComparisonInput,
        cached_zipa: CachedZipaOutput,
    ) -> Self {
        Self::build_from_comparison_input_and_zipa(
            input,
            cached_zipa.zipa_norm_with_spans,
            cached_zipa.phone_spans,
        )
    }

    pub fn build_from_comparison_input(
        input: TranscriptComparisonInput,
        audio: &ZipaAudioBuffer,
        zipa: &ZipaInference,
    ) -> Result<Self, AlignmentError> {
        let cached = infer_cached_zipa_output(audio, zipa, 0, 0.0)?;
        Ok(Self::build_from_cached_zipa(input, cached))
    }

    /// Run ZIPA inference + G2P + DP alignment for `transcript` over `audio`.
    ///
    /// Returns the opaque handle you can query with [`word_timings`] /
    /// [`span_timing`].  Fails only on ZIPA inference errors or G2P errors;
    /// partial alignment (some words unreachable) is not an error.
    pub fn build(
        transcript: &str,
        audio: &ZipaAudioBuffer,
        g2p: &mut CachedEspeakG2p,
        zipa: &ZipaInference,
    ) -> Result<Self, AlignmentError> {
        let word_raw_ranges =
            transcript_word_raw_ranges(g2p, transcript).map_err(AlignmentError::G2p)?;
        let input = transcript_comparison_input_from_word_raw_ranges(transcript, &word_raw_ranges);
        Self::build_from_comparison_input(input, audio, zipa)
    }

    /// Number of words in the transcript.
    pub fn word_count(&self) -> usize {
        self.word_char_ranges.len()
    }

    /// Per-word timing.  Every word is present; check [`AlignmentQuality`] to
    /// distinguish fully-aligned words from those that could not be placed.
    pub fn word_timings(&self) -> Vec<WordTiming<'_>> {
        self.word_char_ranges
            .iter()
            .zip(self.word_windows.iter())
            .map(|(char_range, window)| {
                let word = &self.transcript[char_range.clone()];
                let quality = match window {
                    None => AlignmentQuality::NoWindow,
                    Some(w) => {
                        match timed_range_for_normalized_range(
                            &self.zipa_norm_with_spans,
                            &self.phone_spans,
                            w.zipa_norm_range.clone(),
                        ) {
                            Some(t) => AlignmentQuality::Aligned {
                                start_secs: t.start_time_secs,
                                end_secs: t.end_time_secs,
                            },
                            None => AlignmentQuality::NoTiming,
                        }
                    }
                };
                WordTiming { word, quality }
            })
            .collect()
    }

    /// Timed audio span covering transcript normalized phones `[start, end)`.
    pub fn comparison_range_timing(&self, comparison_range: Range<usize>) -> ComparisonRangeTiming {
        if comparison_range.start >= comparison_range.end
            || comparison_range.end > self.transcript_normalized_len
        {
            return ComparisonRangeTiming::Invalid;
        }
        let Some(projected) = self
            .transcript_alignment
            .project_left_range(comparison_range)
        else {
            return ComparisonRangeTiming::Invalid;
        };
        if projected.start >= projected.end {
            return ComparisonRangeTiming::Deleted {
                projected_at: projected.start,
            };
        }
        match timed_range_for_normalized_range(
            &self.zipa_norm_with_spans,
            &self.phone_spans,
            projected.clone(),
        ) {
            Some(timed) => ComparisonRangeTiming::Aligned(timed),
            None => ComparisonRangeTiming::NoTiming {
                projected_range: projected,
            },
        }
    }

    pub fn projected_comparison_range(
        &self,
        comparison_range: Range<usize>,
    ) -> Option<Range<usize>> {
        if comparison_range.start >= comparison_range.end
            || comparison_range.end > self.transcript_normalized_len
        {
            return None;
        }
        self.transcript_alignment
            .project_left_range(comparison_range)
    }

    pub fn token_piece_timings(
        &self,
        token_pieces: &[bee_g2p::TranscriptTokenPieceComparisonRange],
    ) -> Vec<TokenPieceTiming> {
        token_pieces
            .iter()
            .map(|token| TokenPieceTiming {
                token_index: token.token_index,
                token: token.token.clone(),
                token_surface: token.token_surface.clone(),
                token_char_start: token.token_char_start,
                token_char_end: token.token_char_end,
                word_index: token.word_index,
                word_surface: token.word_surface.clone(),
                timing: self.comparison_range_timing(token.comparison_start..token.comparison_end),
            })
            .collect()
    }

    /// Timed audio span covering words `[word_start, word_end)`.
    ///
    /// When only some words have alignment windows the returned timing covers
    /// the aligned subset and is tagged [`SpanTiming::PartialGap`] so callers
    /// can decide how to handle the gap.
    pub fn span_timing(&self, word_start: usize, word_end: usize) -> SpanTiming {
        let word_end = word_end.min(self.word_windows.len());
        let word_start = word_start.min(word_end);
        let windows = &self.word_windows[word_start..word_end];

        // Collect zipa_norm_ranges from windows that resolved.
        let resolved: Vec<Range<usize>> = windows
            .iter()
            .filter_map(|w| w.as_ref().map(|w| w.zipa_norm_range.clone()))
            .collect();

        if resolved.is_empty() {
            return SpanTiming::NoAlignedWords;
        }

        let is_partial = resolved.len() < windows.len();

        // Merge: span from the first resolved window's start to the last's end.
        let merged_start = resolved.iter().map(|r| r.start).min().unwrap();
        let merged_end = resolved.iter().map(|r| r.end).max().unwrap();

        match timed_range_for_normalized_range(
            &self.zipa_norm_with_spans,
            &self.phone_spans,
            merged_start..merged_end,
        ) {
            Some(t) if is_partial => SpanTiming::PartialGap {
                start_secs: t.start_time_secs,
                end_secs: t.end_time_secs,
            },
            Some(t) => SpanTiming::Aligned {
                start_secs: t.start_time_secs,
                end_secs: t.end_time_secs,
            },
            None => SpanTiming::NoTiming,
        }
    }
}

pub fn infer_cached_zipa_output(
    audio: &ZipaAudioBuffer,
    zipa: &ZipaInference,
    raw_token_offset: usize,
    time_offset_secs: f64,
) -> Result<CachedZipaOutput, AlignmentError> {
    let start = Instant::now();
    let infer_start = Instant::now();
    let utterance = zipa
        .infer_audio_greedy(audio)
        .map_err(|e| AlignmentError::Zipa(e.to_string()))?;
    let infer_ms = infer_start.elapsed().as_secs_f64() * 1000.0;

    let duration = audio.samples.len() as f64 / audio.sample_rate_hz as f64;

    let spans_start = Instant::now();
    let phone_spans: Vec<PhoneSpan> = utterance
        .derive_phone_spans(&zipa.tokens, duration, 0)
        .into_iter()
        .filter(|s| s.token != "▁")
        .map(|span| PhoneSpan {
            start_time_secs: span.start_time_secs + time_offset_secs,
            end_time_secs: span.end_time_secs + time_offset_secs,
            ..span
        })
        .collect();
    let spans_ms = spans_start.elapsed().as_secs_f64() * 1000.0;

    let normalize_start = Instant::now();
    let zipa_raw: Vec<String> = utterance.tokens.into_iter().filter(|t| t != "▁").collect();
    let zipa_norm_with_spans = normalize_ipa_for_comparison_with_spans(&zipa_raw)
        .into_iter()
        .map(|mut token| {
            token.source_start += raw_token_offset;
            token.source_end += raw_token_offset;
            token
        })
        .collect();
    let normalize_ms = normalize_start.elapsed().as_secs_f64() * 1000.0;
    tracing::trace!(
        target: "bee_phase",
        component = "bee_transcribe",
        phase = "infer_cached_zipa_output",
        infer_ms = infer_ms,
        spans_ms = spans_ms,
        normalize_ms = normalize_ms,
        ms = start.elapsed().as_secs_f64() * 1000.0,
        "phase timing"
    );

    Ok(CachedZipaOutput {
        raw_token_count: zipa_raw.len(),
        zipa_raw,
        zipa_norm_with_spans,
        phone_spans,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        CachedZipaOutput, ComparisonRangeTiming, TranscriptAlignment, TranscriptComparisonInput,
        expand_degenerate_projected_range, partition_word_alignment_windows,
        select_segmental_word_windows, timed_range_for_normalized_range,
        transcript_comparison_input_from_g2p, transcript_comparison_input_from_word_raw_ranges,
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
    fn cached_zipa_output_trims_prefix_before_appending_tail() {
        let mut cache = CachedZipaOutput {
            raw_token_count: 3,
            zipa_raw: vec!["a".to_string(), "b".to_string(), "c".to_string()],
            zipa_norm_with_spans: vec![
                ComparisonToken {
                    token: "a".to_string(),
                    source_start: 0,
                    source_end: 1,
                },
                ComparisonToken {
                    token: "b".to_string(),
                    source_start: 1,
                    source_end: 2,
                },
                ComparisonToken {
                    token: "c".to_string(),
                    source_start: 2,
                    source_end: 3,
                },
            ],
            phone_spans: vec![
                PhoneSpan {
                    token_id: 1,
                    token: "a".to_string(),
                    start_frame: 0,
                    end_frame: 10,
                    start_time_secs: 0.0,
                    end_time_secs: 0.5,
                },
                PhoneSpan {
                    token_id: 2,
                    token: "b".to_string(),
                    start_frame: 10,
                    end_frame: 20,
                    start_time_secs: 0.5,
                    end_time_secs: 1.0,
                },
                PhoneSpan {
                    token_id: 3,
                    token: "c".to_string(),
                    start_frame: 20,
                    end_frame: 30,
                    start_time_secs: 1.0,
                    end_time_secs: 1.5,
                },
            ],
        };

        cache.trim_front(1.0);

        assert_eq!(cache.raw_token_count(), 1);
        assert_eq!(cache.zipa_raw, vec!["c".to_string()]);
        assert_eq!(cache.zipa_norm_with_spans.len(), 1);
        assert_eq!(cache.zipa_norm_with_spans[0].source_start, 0);
        assert_eq!(cache.zipa_norm_with_spans[0].source_end, 1);
        assert_eq!(cache.phone_spans.len(), 1);
        assert_eq!(cache.phone_spans[0].token, "c");
        assert_eq!(cache.phone_spans[0].start_time_secs, 0.0);
        assert_eq!(cache.phone_spans[0].end_time_secs, 0.5);
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
    fn degenerate_projected_range_expands_forward() {
        assert_eq!(expand_degenerate_projected_range(0..0, 3, 10), 0..3);
        assert_eq!(expand_degenerate_projected_range(5..5, 2, 10), 5..7);
    }

    #[test]
    fn degenerate_projected_range_expands_backward_at_end() {
        assert_eq!(expand_degenerate_projected_range(10..10, 4, 10), 6..10);
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

    #[test]
    fn transcript_comparison_input_flattens_word_ranges() {
        let input = transcript_comparison_input_from_word_raw_ranges(
            "use Facet",
            &[
                (
                    0..3,
                    vec!["j".to_string(), "u".to_string(), "z".to_string()],
                ),
                (
                    4..9,
                    vec![
                        "f".to_string(),
                        "eɪ".to_string(),
                        "s".to_string(),
                        "ə".to_string(),
                        "t".to_string(),
                    ],
                ),
            ],
        );

        assert_eq!(input.word_char_ranges, vec![0..3, 4..9]);
        assert_eq!(input.word_normalized_ranges, vec![0..3, 3..9]);
        assert_eq!(input.word_tokens[0], vec!["j", "ʊ", "z"]);
        assert_eq!(input.word_tokens[1], vec!["f", "ɛ", "ɪ", "s", "ə", "t"]);
        assert_eq!(
            input.transcript_normalized,
            vec!["j", "ʊ", "z", "f", "ɛ", "ɪ", "s", "ə", "t"]
        );
    }

    #[test]
    fn transcript_comparison_input_from_g2p_keeps_word_ranges() {
        let input = bee_g2p::TranscriptAlignmentInput {
            normalized: vec![
                "j".to_string(),
                "ʊ".to_string(),
                "z".to_string(),
                "f".to_string(),
                "ɛ".to_string(),
                "ɪ".to_string(),
                "s".to_string(),
                "ə".to_string(),
                "t".to_string(),
            ],
            sequence: bee_g2p::TranscriptComparisonSequence {
                tokens: vec![],
                provenance: vec![],
            },
            words: vec![
                bee_g2p::TranscriptWordComparisonRange {
                    word_index: 0,
                    word_surface: "use".to_string(),
                    char_start: 0,
                    char_end: 3,
                    comparison_start: 0,
                    comparison_end: 3,
                },
                bee_g2p::TranscriptWordComparisonRange {
                    word_index: 1,
                    word_surface: "Facet".to_string(),
                    char_start: 4,
                    char_end: 9,
                    comparison_start: 3,
                    comparison_end: 9,
                },
            ],
            token_pieces: vec![],
        };

        let converted = transcript_comparison_input_from_g2p("use Facet", &input);
        assert_eq!(converted.word_char_ranges, vec![0..3, 4..9]);
        assert_eq!(converted.word_normalized_ranges, vec![0..3, 3..9]);
        assert_eq!(converted.word_tokens[0], vec!["j", "ʊ", "z"]);
        assert_eq!(converted.word_tokens[1], vec!["f", "ɛ", "ɪ", "s", "ə", "t"]);
        assert_eq!(converted.transcript_normalized, input.normalized);
    }

    // Build a minimal TranscriptAlignment with three words where the middle
    // word has no alignment window, then assert all SpanTiming variants.
    #[test]
    fn span_timing_variants() {
        use super::{AlignmentQuality, SpanTiming, WordAlignmentWindow};

        // transcript: "hello world foo"
        // words:       0..5   6..11  12..15
        let transcript = "hello world foo".to_string();
        let word_char_ranges = vec![0..5, 6..11, 12..15];

        // ZIPA normalized sequence: 4 tokens, each mapping 1:1 to a raw phone.
        let zipa_norm_with_spans = vec![
            ComparisonToken {
                token: "h".into(),
                source_start: 0,
                source_end: 1,
            },
            ComparisonToken {
                token: "ɛ".into(),
                source_start: 1,
                source_end: 2,
            },
            ComparisonToken {
                token: "w".into(),
                source_start: 2,
                source_end: 3,
            },
            ComparisonToken {
                token: "f".into(),
                source_start: 3,
                source_end: 4,
            },
        ];

        // Raw phone spans with known timings.
        let phone_spans = vec![
            PhoneSpan {
                token_id: 1,
                token: "h".into(),
                start_frame: 0,
                end_frame: 10,
                start_time_secs: 0.0,
                end_time_secs: 0.1,
            },
            PhoneSpan {
                token_id: 2,
                token: "ɛ".into(),
                start_frame: 10,
                end_frame: 20,
                start_time_secs: 0.1,
                end_time_secs: 0.2,
            },
            PhoneSpan {
                token_id: 3,
                token: "w".into(),
                start_frame: 20,
                end_frame: 30,
                start_time_secs: 0.2,
                end_time_secs: 0.3,
            },
            PhoneSpan {
                token_id: 4,
                token: "f".into(),
                start_frame: 30,
                end_frame: 40,
                start_time_secs: 0.3,
                end_time_secs: 0.4,
            },
        ];

        // word 0 → norm range 0..2, word 1 → None, word 2 → norm range 2..4
        let word_windows = vec![
            Some(WordAlignmentWindow {
                zipa_norm_range: 0..2,
                ops: vec![],
            }),
            None,
            Some(WordAlignmentWindow {
                zipa_norm_range: 2..4,
                ops: vec![],
            }),
        ];

        let ta = TranscriptAlignment {
            transcript,
            word_char_ranges,
            transcript_alignment: align_token_sequences_with_left_word_boundaries(
                &[
                    "h".to_string(),
                    "ɛ".to_string(),
                    "w".to_string(),
                    "f".to_string(),
                ],
                &[
                    "h".to_string(),
                    "ɛ".to_string(),
                    "w".to_string(),
                    "f".to_string(),
                ],
                &[0usize, 0, 2, 2],
            ),
            transcript_normalized_len: 4,
            word_windows,
            zipa_norm_with_spans,
            phone_spans,
        };

        // Single aligned word.
        assert_eq!(
            ta.span_timing(0, 1),
            SpanTiming::Aligned {
                start_secs: 0.0,
                end_secs: 0.2
            }
        );

        // Single unaligned word.
        assert_eq!(ta.span_timing(1, 2), SpanTiming::NoAlignedWords);

        // Range where one word has a window and one doesn't → PartialGap.
        assert_eq!(
            ta.span_timing(0, 2),
            SpanTiming::PartialGap {
                start_secs: 0.0,
                end_secs: 0.2
            }
        );

        // All three words: words 0 and 2 aligned, word 1 missing → PartialGap
        // covering the merged range 0..4.
        assert_eq!(
            ta.span_timing(0, 3),
            SpanTiming::PartialGap {
                start_secs: 0.0,
                end_secs: 0.4
            }
        );

        // word_timings: check quality variants.
        let timings = ta.word_timings();
        assert!(matches!(
            timings[0].quality,
            AlignmentQuality::Aligned { .. }
        ));
        assert!(matches!(timings[1].quality, AlignmentQuality::NoWindow));
        assert!(matches!(
            timings[2].quality,
            AlignmentQuality::Aligned { .. }
        ));
        assert_eq!(timings[0].word, "hello");
        assert_eq!(timings[1].word, "world");
        assert_eq!(timings[2].word, "foo");
    }

    #[test]
    fn comparison_range_timing_recovers_token_piece_times() {
        let input = TranscriptComparisonInput {
            transcript: "use Facet".to_string(),
            word_char_ranges: vec![0..3, 4..9],
            word_normalized_ranges: vec![0..3, 3..9],
            word_tokens: vec![
                vec!["j".to_string(), "ʊ".to_string(), "z".to_string()],
                vec![
                    "f".to_string(),
                    "ɛ".to_string(),
                    "ɪ".to_string(),
                    "s".to_string(),
                    "ə".to_string(),
                    "t".to_string(),
                ],
            ],
            transcript_normalized: vec![
                "j".to_string(),
                "ʊ".to_string(),
                "z".to_string(),
                "f".to_string(),
                "ɛ".to_string(),
                "ɪ".to_string(),
                "s".to_string(),
                "ə".to_string(),
                "t".to_string(),
            ],
        };
        let zipa_norm_with_spans = vec![
            ComparisonToken {
                token: "j".into(),
                source_start: 0,
                source_end: 1,
            },
            ComparisonToken {
                token: "ʊ".into(),
                source_start: 1,
                source_end: 2,
            },
            ComparisonToken {
                token: "z".into(),
                source_start: 2,
                source_end: 3,
            },
            ComparisonToken {
                token: "f".into(),
                source_start: 3,
                source_end: 4,
            },
            ComparisonToken {
                token: "ɛ".into(),
                source_start: 4,
                source_end: 5,
            },
            ComparisonToken {
                token: "ɪ".into(),
                source_start: 4,
                source_end: 5,
            },
            ComparisonToken {
                token: "s".into(),
                source_start: 5,
                source_end: 6,
            },
            ComparisonToken {
                token: "ə".into(),
                source_start: 6,
                source_end: 7,
            },
            ComparisonToken {
                token: "t".into(),
                source_start: 7,
                source_end: 8,
            },
        ];
        let phone_spans = vec![
            PhoneSpan {
                token_id: 1,
                token: "j".into(),
                start_frame: 0,
                end_frame: 10,
                start_time_secs: 0.00,
                end_time_secs: 0.10,
            },
            PhoneSpan {
                token_id: 2,
                token: "ʊ".into(),
                start_frame: 10,
                end_frame: 20,
                start_time_secs: 0.10,
                end_time_secs: 0.20,
            },
            PhoneSpan {
                token_id: 3,
                token: "z".into(),
                start_frame: 20,
                end_frame: 30,
                start_time_secs: 0.20,
                end_time_secs: 0.30,
            },
            PhoneSpan {
                token_id: 4,
                token: "f".into(),
                start_frame: 30,
                end_frame: 40,
                start_time_secs: 0.30,
                end_time_secs: 0.40,
            },
            PhoneSpan {
                token_id: 5,
                token: "eɪ".into(),
                start_frame: 40,
                end_frame: 50,
                start_time_secs: 0.40,
                end_time_secs: 0.50,
            },
            PhoneSpan {
                token_id: 6,
                token: "s".into(),
                start_frame: 50,
                end_frame: 60,
                start_time_secs: 0.50,
                end_time_secs: 0.60,
            },
            PhoneSpan {
                token_id: 7,
                token: "ə".into(),
                start_frame: 60,
                end_frame: 70,
                start_time_secs: 0.60,
                end_time_secs: 0.70,
            },
            PhoneSpan {
                token_id: 8,
                token: "t".into(),
                start_frame: 70,
                end_frame: 80,
                start_time_secs: 0.70,
                end_time_secs: 0.80,
            },
        ];

        let ta = TranscriptAlignment::build_from_comparison_input_and_zipa(
            input,
            zipa_norm_with_spans,
            phone_spans,
        );

        let ComparisonRangeTiming::Aligned(fac) = ta.comparison_range_timing(3..7) else {
            panic!("Fac timing should align");
        };
        assert_eq!(fac.raw_phone_range, 3..6);
        assert!((fac.start_time_secs - 0.30).abs() < 1e-6);
        assert!((fac.end_time_secs - 0.60).abs() < 1e-6);

        let ComparisonRangeTiming::Aligned(et) = ta.comparison_range_timing(7..9) else {
            panic!("et timing should align");
        };
        assert_eq!(et.raw_phone_range, 6..8);
        assert!((et.start_time_secs - 0.60).abs() < 1e-6);
        assert!((et.end_time_secs - 0.80).abs() < 1e-6);
    }

    #[test]
    fn token_piece_timings_wrap_comparison_ranges() {
        let input = bee_g2p::TranscriptAlignmentInput {
            normalized: vec![
                "f".to_string(),
                "ɛ".to_string(),
                "ɪ".to_string(),
                "s".to_string(),
                "ə".to_string(),
                "t".to_string(),
            ],
            sequence: bee_g2p::TranscriptComparisonSequence {
                tokens: vec![],
                provenance: vec![],
            },
            words: vec![bee_g2p::TranscriptWordComparisonRange {
                word_index: 0,
                word_surface: "Facet".to_string(),
                char_start: 0,
                char_end: 5,
                comparison_start: 0,
                comparison_end: 6,
            }],
            token_pieces: vec![
                bee_g2p::TranscriptTokenPieceComparisonRange {
                    token_index: 0,
                    token: "Fac".to_string(),
                    token_surface: "Fac".to_string(),
                    token_char_start: 0,
                    token_char_end: 3,
                    word_index: Some(0),
                    word_surface: Some("Facet".to_string()),
                    comparison_start: 0,
                    comparison_end: 4,
                },
                bee_g2p::TranscriptTokenPieceComparisonRange {
                    token_index: 1,
                    token: "et".to_string(),
                    token_surface: "et".to_string(),
                    token_char_start: 3,
                    token_char_end: 5,
                    word_index: Some(0),
                    word_surface: Some("Facet".to_string()),
                    comparison_start: 4,
                    comparison_end: 6,
                },
            ],
        };
        let ta = TranscriptAlignment::build_from_comparison_input_and_zipa(
            transcript_comparison_input_from_g2p("Facet", &input),
            vec![
                ComparisonToken {
                    token: "f".into(),
                    source_start: 0,
                    source_end: 1,
                },
                ComparisonToken {
                    token: "ɛ".into(),
                    source_start: 1,
                    source_end: 2,
                },
                ComparisonToken {
                    token: "ɪ".into(),
                    source_start: 1,
                    source_end: 2,
                },
                ComparisonToken {
                    token: "s".into(),
                    source_start: 2,
                    source_end: 3,
                },
                ComparisonToken {
                    token: "ə".into(),
                    source_start: 3,
                    source_end: 4,
                },
                ComparisonToken {
                    token: "t".into(),
                    source_start: 4,
                    source_end: 5,
                },
            ],
            vec![
                PhoneSpan {
                    token_id: 1,
                    token: "f".into(),
                    start_frame: 0,
                    end_frame: 10,
                    start_time_secs: 0.0,
                    end_time_secs: 0.1,
                },
                PhoneSpan {
                    token_id: 2,
                    token: "eɪ".into(),
                    start_frame: 10,
                    end_frame: 20,
                    start_time_secs: 0.1,
                    end_time_secs: 0.2,
                },
                PhoneSpan {
                    token_id: 3,
                    token: "s".into(),
                    start_frame: 20,
                    end_frame: 30,
                    start_time_secs: 0.2,
                    end_time_secs: 0.3,
                },
                PhoneSpan {
                    token_id: 4,
                    token: "ə".into(),
                    start_frame: 30,
                    end_frame: 40,
                    start_time_secs: 0.3,
                    end_time_secs: 0.4,
                },
                PhoneSpan {
                    token_id: 5,
                    token: "t".into(),
                    start_frame: 40,
                    end_frame: 50,
                    start_time_secs: 0.4,
                    end_time_secs: 0.5,
                },
            ],
        );

        let timings = ta.token_piece_timings(&input.token_pieces);
        assert_eq!(timings.len(), 2);
        let ComparisonRangeTiming::Aligned(fac) = &timings[0].timing else {
            panic!("Fac should align");
        };
        assert!((fac.start_time_secs - 0.0).abs() < 1e-6);
        assert!((fac.end_time_secs - 0.3).abs() < 1e-6);
        let ComparisonRangeTiming::Aligned(et) = &timings[1].timing else {
            panic!("et should align");
        };
        assert!((et.start_time_secs - 0.3).abs() < 1e-6);
        assert!((et.end_time_secs - 0.5).abs() < 1e-6);
    }
}
