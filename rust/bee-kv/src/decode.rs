use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use bee_phonetic::sentence_word_tokens;
use bee_qwen3_asr::generate::{
    ConfidenceMode, build_followup_prompt, build_initial_prompt, prefill_and_decode,
};
use bee_qwen3_asr::mel::MelExtractor;
use bee_qwen3_asr::mlx_rs::Array;
use bee_qwen3_asr::mlx_rs::module::Module;
use bee_qwen3_asr::mlx_rs::ops;
use bee_qwen3_asr::model::Qwen3ASRModel;
use bee_qwen3_asr::tokenizers::Tokenizer;

use crate::alignment::{
    AlignmentContext, adjust_keep_boundary_secs, build_transcript_alignment, suffix_after_prefix,
    timed_aligned_words_for_alignment, timed_generated_bridge_for_cuts,
    timed_generated_prefix_for_cut,
};
use crate::html;
use crate::print::{annotate_chunk_runs, clone_chunk_run, print_finalizing_banner};
use crate::tui::{ExerciseTui, append_display_delta, append_exact, update_exercise_progress};
use crate::types::*;
use crate::{
    BOUNDARY_SWEEP_OFFSETS, DEFAULT_START_POSITION_FOR_FRESH_FOLLOWUP, HOP_LENGTH, N_FFT,
    SAMPLE_RATE,
};

/// Decodes audio using an initial prompt with language and context.
pub(crate) fn decode_with_initial_prompt(
    model: &Qwen3ASRModel,
    tokenizer: &Tokenizer,
    audio_features: &Array,
    n_audio_tokens: usize,
    language: &str,
    context: &str,
    max_new_tokens: usize,
) -> Result<ExperimentResult> {
    let prompt_tokens = build_initial_prompt(n_audio_tokens, language, context, tokenizer);
    decode_prompt(
        model,
        tokenizer,
        audio_features,
        prompt_tokens,
        max_new_tokens,
        0,
        format!("initial context={context:?}"),
    )
}

/// Decodes audio using a followup prompt without prior context tokens.
pub(crate) fn decode_with_followup_prompt(
    model: &Qwen3ASRModel,
    tokenizer: &Tokenizer,
    audio_features: &Array,
    n_audio_tokens: usize,
    language: &str,
    max_new_tokens: usize,
) -> Result<ExperimentResult> {
    let prompt_tokens = build_followup_prompt(n_audio_tokens, language, tokenizer);
    decode_prompt(
        model,
        tokenizer,
        audio_features,
        prompt_tokens,
        max_new_tokens,
        DEFAULT_START_POSITION_FOR_FRESH_FOLLOWUP,
        "followup-fresh".to_string(),
    )
}

/// Runs prefill-and-decode on pre-built prompt tokens and returns the transcription result.
pub(crate) fn decode_prompt(
    model: &Qwen3ASRModel,
    tokenizer: &Tokenizer,
    audio_features: &Array,
    prompt_tokens: Vec<i32>,
    max_new_tokens: usize,
    start_position: usize,
    label: String,
) -> Result<ExperimentResult> {
    let mut cache = None;
    let (generated, _confidence, _next_position, _stop_reason) = prefill_and_decode(
        model,
        &prompt_tokens,
        audio_features,
        &mut cache,
        start_position,
        max_new_tokens,
        ConfidenceMode::Streaming,
    )
    .with_context(|| format!("running decode for {label}"))?;

    let token_ids: Vec<u32> = generated
        .iter()
        .map(|&id| u32::try_from(id).context("generated negative token id"))
        .collect::<Result<_>>()?;
    let transcript = tokenizer
        .decode(&token_ids, true)
        .map_err(|e| anyhow::anyhow!("decoding transcript: {e}"))?;

    Ok(ExperimentResult {
        label,
        prompt_tokens: prompt_tokens.len(),
        generated_tokens: generated.len(),
        transcript,
    })
}

/// Decodes audio in fixed-size chunks using sequential followup prompts.
pub(crate) fn decode_chunked_followup(
    model: &Qwen3ASRModel,
    tokenizer: &Tokenizer,
    samples: &[f32],
    thinker: &bee_qwen3_asr::config::ThinkerConfig,
    language: &str,
    context: &str,
    max_new_tokens: usize,
    chunk_ms: usize,
) -> Result<ChunkedExperimentResult> {
    let chunk_size_samples = ms_to_samples(chunk_ms)?;

    let chunk_runs = run_chunked_followup_sequence(
        model,
        tokenizer,
        samples,
        thinker,
        language,
        context,
        max_new_tokens,
        &build_chunk_plan(samples.len(), chunk_size_samples, chunk_size_samples)?,
        None,
    )?;

    Ok(ChunkedExperimentResult {
        chunk_ms,
        chunk_results: chunk_runs.iter().map(ChunkResult::from_run).collect(),
        combined_transcript: combine_transcripts(&chunk_runs),
    })
}

/// Decodes progressively longer audio prefixes to compare transcription stability.
pub(crate) fn decode_prefix_rerun(
    model: &Qwen3ASRModel,
    tokenizer: &Tokenizer,
    samples: &[f32],
    thinker: &bee_qwen3_asr::config::ThinkerConfig,
    language: &str,
    context: &str,
    max_new_tokens: usize,
    chunk_ms: usize,
) -> Result<PrefixRerunExperimentResult> {
    let chunk_size_samples = (chunk_ms * SAMPLE_RATE as usize) / 1000;
    if chunk_size_samples == 0 {
        bail!("chunk size is zero; chunk_ms={chunk_ms}");
    }

    let mel_extractor = MelExtractor::new(
        N_FFT,
        HOP_LENGTH,
        thinker.audio_config.num_mel_bins,
        SAMPLE_RATE,
    );
    let probe_ids = Array::from_slice(&[0_i32], &[1, 1]);
    let embed_dtype = model.model.embed_tokens.forward(&probe_ids)?.dtype();

    let mut prefix_results = Vec::new();
    for prefix_end in (chunk_size_samples..samples.len()).step_by(chunk_size_samples) {
        let prefix_samples = &samples[..prefix_end];
        prefix_results.push(decode_prefix_window(
            model,
            tokenizer,
            &mel_extractor,
            embed_dtype,
            prefix_samples,
            language,
            context,
            max_new_tokens,
            format!("prefix 0..{}ms", (prefix_end * 1000) / SAMPLE_RATE as usize),
        )?);
    }

    if prefix_results
        .last()
        .is_none_or(|result| result.sample_count != samples.len())
    {
        prefix_results.push(decode_prefix_window(
            model,
            tokenizer,
            &mel_extractor,
            embed_dtype,
            samples,
            language,
            context,
            max_new_tokens,
            format!(
                "prefix 0..{}ms",
                (samples.len() * 1000) / SAMPLE_RATE as usize
            ),
        )?);
    }

    Ok(PrefixRerunExperimentResult {
        chunk_ms,
        prefix_results,
    })
}

/// Decodes chunks, then truncates tokens at a chosen chunk and replays to test rollback policies.
pub(crate) fn decode_truncate_replay(
    model: &Qwen3ASRModel,
    tokenizer: &Tokenizer,
    samples: &[f32],
    thinker: &bee_qwen3_asr::config::ThinkerConfig,
    language: &str,
    context: &str,
    max_new_tokens: usize,
    chunk_ms: usize,
    rollback_policy: RollbackPolicy,
    replay_chunk_index: usize,
    truncate_tokens: usize,
) -> Result<TruncateReplayExperimentResult> {
    let chunk_size_samples = ms_to_samples(chunk_ms)?;
    let chunk_plan = build_chunk_plan(samples.len(), chunk_size_samples, chunk_size_samples)?;

    let baseline_runs = run_chunked_followup_sequence(
        model,
        tokenizer,
        samples,
        thinker,
        language,
        context,
        max_new_tokens,
        &chunk_plan,
        None,
    )?;
    if replay_chunk_index == 0 || replay_chunk_index >= baseline_runs.len() {
        bail!(
            "replay chunk index out of range: {} (valid followup chunks: 1..{})",
            replay_chunk_index,
            baseline_runs.len().saturating_sub(1)
        );
    }

    let replay_from_chunk_index = match rollback_policy {
        RollbackPolicy::TextSuffix => replay_chunk_index + 1,
        RollbackPolicy::ChunkSegment => replay_chunk_index,
    };
    if replay_from_chunk_index >= baseline_runs.len() {
        bail!(
            "rollback policy {} needs a later chunk to replay, but chunk {} is the last chunk",
            rollback_policy.as_str(),
            replay_chunk_index
        );
    }

    let baseline_target = &baseline_runs[replay_chunk_index];
    let truncated_tokens = truncate_tokens.min(baseline_target.generated_tokens);
    if rollback_policy == RollbackPolicy::TextSuffix && truncated_tokens == 0 {
        bail!(
            "text-suffix rollback needs at least one generated token in chunk {}",
            replay_chunk_index
        );
    }

    let rollback_position = match rollback_policy {
        RollbackPolicy::TextSuffix => baseline_target
            .end_position
            .saturating_sub(truncated_tokens),
        RollbackPolicy::ChunkSegment => baseline_target.start_position,
    };

    let replay_runs = run_chunked_followup_sequence(
        model,
        tokenizer,
        samples,
        thinker,
        language,
        context,
        max_new_tokens,
        &chunk_plan,
        Some(ReplayPlan {
            rollback_policy,
            rollback_after_chunk_index: replay_chunk_index,
            rollback_position,
            replay_from_chunk_index,
        }),
    )?;

    Ok(TruncateReplayExperimentResult {
        chunk_ms,
        rollback_policy,
        replay_chunk_index,
        replay_from_chunk_index,
        rollback_position,
        requested_truncate_tokens: truncate_tokens,
        applied_truncate_tokens: truncated_tokens,
        baseline_runs,
        replay_runs,
    })
}

/// Runs two independent chunked decode lanes with different first-chunk sizes for comparison.
pub(crate) fn decode_dual_lane_followup(
    model: &Qwen3ASRModel,
    tokenizer: &Tokenizer,
    samples: &[f32],
    thinker: &bee_qwen3_asr::config::ThinkerConfig,
    language: &str,
    context: &str,
    max_new_tokens: usize,
    chunk_ms: usize,
    lane_b_first_chunk_ms: usize,
) -> Result<DualLaneFollowupExperimentResult> {
    let chunk_size_samples = ms_to_samples(chunk_ms)?;
    let lane_b_first_chunk_samples = ms_to_samples(lane_b_first_chunk_ms)?;
    if lane_b_first_chunk_samples <= chunk_size_samples {
        bail!(
            "lane-b first chunk must be larger than lane-a chunk: lane_b_first_chunk_ms={} chunk_ms={}",
            lane_b_first_chunk_ms,
            chunk_ms
        );
    }

    let lane_a_runs = run_chunked_followup_sequence(
        model,
        tokenizer,
        samples,
        thinker,
        language,
        context,
        max_new_tokens,
        &build_chunk_plan(samples.len(), chunk_size_samples, chunk_size_samples)?,
        None,
    )?;
    let lane_b_runs = run_chunked_followup_sequence(
        model,
        tokenizer,
        samples,
        thinker,
        language,
        context,
        max_new_tokens,
        &build_chunk_plan(
            samples.len(),
            lane_b_first_chunk_samples,
            chunk_size_samples,
        )?,
        None,
    )?;

    Ok(DualLaneFollowupExperimentResult {
        chunk_ms,
        lane_b_first_chunk_ms,
        lane_a_runs,
        lane_b_runs,
    })
}

/// Runs sliding-window decoding with a fixed time-based rollback.
pub(crate) fn decode_sliding_window_timed_rollback(
    model: &Qwen3ASRModel,
    tokenizer: &Tokenizer,
    samples: &[f32],
    thinker: &bee_qwen3_asr::config::ThinkerConfig,
    language: &str,
    context: &str,
    max_new_tokens: usize,
    chunk_ms: usize,
    rollback_ms: usize,
    wav_path: &Path,
) -> Result<SlidingWindowTimedRollbackExperimentResult> {
    let window_samples = ms_to_samples(chunk_ms)?;
    let rollback_samples = ms_to_samples(rollback_ms)?;
    if rollback_samples >= window_samples {
        bail!(
            "rollback must be smaller than window: rollback_ms={} chunk_ms={}",
            rollback_ms,
            chunk_ms
        );
    }

    let stride_samples = window_samples - rollback_samples;
    let window_plan = build_overlapping_window_plan(samples.len(), window_samples, stride_samples)?;
    let mel_extractor = MelExtractor::new(
        N_FFT,
        HOP_LENGTH,
        thinker.audio_config.num_mel_bins,
        SAMPLE_RATE,
    );
    let probe_ids = Array::from_slice(&[0_i32], &[1, 1]);
    let embed_dtype = model.model.embed_tokens.forward(&probe_ids)?.dtype();

    let mut cache = None;
    let mut start_position = 0usize;
    let mut window_runs = Vec::new();
    let mut align_ctx = AlignmentContext::new()?;

    for (window_index, window) in window_plan.iter().enumerate() {
        let chunk_samples = &samples[window.start_sample..window.end_sample];
        let chunk_run = decode_chunk_followup_step(
            model,
            tokenizer,
            &mel_extractor,
            embed_dtype,
            chunk_samples,
            window_index,
            language,
            context,
            max_new_tokens,
            &mut cache,
            start_position,
            window.start_sample,
            window.end_sample,
            None,
        )?;

        let rollback = if window_index + 1 < window_plan.len() {
            let keep_until_sample = window.end_sample.saturating_sub(rollback_samples);
            let keep_until_secs =
                (keep_until_sample.saturating_sub(window.start_sample)) as f64 / SAMPLE_RATE as f64;
            let keep = timed_generated_prefix_for_cut(
                &mut align_ctx,
                tokenizer,
                &chunk_run,
                chunk_samples,
                keep_until_secs,
            )?;
            let kept_token_ids = kept_generated_token_ids(&chunk_run, keep.kept_token_count);
            let kept_text = decode_token_ids(tokenizer, &kept_token_ids)?;
            let rollback_position =
                chunk_run.start_position + chunk_run.prompt_tokens + keep.kept_token_count;
            truncate_cache(&mut cache, rollback_position)?;
            start_position = rollback_position;
            Some(WindowRollbackDecision {
                keep_boundary_policy: KeepBoundaryPolicy::Fixed,
                target_keep_until_secs: keep_until_secs,
                keep_until_secs: Some(keep_until_secs),
                replay_until_secs: None,
                kept_word_count: keep.kept_word_count,
                kept_token_count: keep.kept_token_count,
                kept_token_ids,
                kept_text,
                bridge_token_ids: Vec::new(),
                bridge_text: None,
                rollback_position,
                keep_boundary_debug: KeepBoundaryDebug {
                    earliest_candidate_secs: keep_until_secs,
                    min_keep_secs: keep_until_secs,
                    snapped: false,
                    chosen_word: None,
                },
            })
        } else {
            start_position = chunk_run.end_position;
            None
        };

        window_runs.push(SlidingWindowRun {
            chunk_run,
            rollback,
            replayed_prefix: None,
        });
    }

    let html_path = html::write_sliding_window_timed_rollback_html(
        "sliding-window-timed-rollback",
        &window_runs,
        samples,
        wav_path,
    )?;
    let committed_timeline_path = Some(html::write_committed_timeline_html(
        "sliding-window-timed-rollback-committed",
        &window_runs,
        samples,
        wav_path,
    )?);

    Ok(SlidingWindowTimedRollbackExperimentResult {
        mode_label: "sliding-window-timed-rollback",
        chunk_ms,
        rollback_ms,
        stride_ms: (stride_samples * 1000) / SAMPLE_RATE as usize,
        window_runs,
        html_path,
        committed_timeline_path,
        interrupted_early: false,
    })
}

/// Runs sliding-window decoding where each window fully replays the prior context.
pub(crate) fn decode_sliding_window_full_replay(
    model: &Qwen3ASRModel,
    tokenizer: &Tokenizer,
    samples: &[f32],
    thinker: &bee_qwen3_asr::config::ThinkerConfig,
    language: &str,
    context: &str,
    max_new_tokens: usize,
    chunk_ms: usize,
    rollback_ms: usize,
    wav_path: &Path,
) -> Result<SlidingWindowTimedRollbackExperimentResult> {
    let window_samples = ms_to_samples(chunk_ms)?;
    let rollback_samples = ms_to_samples(rollback_ms)?;
    if rollback_samples >= window_samples {
        bail!(
            "rollback must be smaller than window: rollback_ms={} chunk_ms={}",
            rollback_ms,
            chunk_ms
        );
    }

    let stride_samples = window_samples - rollback_samples;
    let window_plan = build_overlapping_window_plan(samples.len(), window_samples, stride_samples)?;
    let mel_extractor = MelExtractor::new(
        N_FFT,
        HOP_LENGTH,
        thinker.audio_config.num_mel_bins,
        SAMPLE_RATE,
    );
    let probe_ids = Array::from_slice(&[0_i32], &[1, 1]);
    let embed_dtype = model.model.embed_tokens.forward(&probe_ids)?.dtype();

    let mut cache = None;
    let mut start_position = 0usize;
    let mut window_runs = Vec::new();
    let mut replay_prefix_for_next: Option<CarriedBridge> = None;
    let mut align_ctx = AlignmentContext::new()?;
    let mut tui = ExerciseTui::new();
    let mut committed_text = String::new();
    let mut draft_text = String::new();

    for (window_index, window) in window_plan.iter().enumerate() {
        let chunk_samples = &samples[window.start_sample..window.end_sample];
        let replayed_prefix = replay_prefix_for_next.clone();
        update_exercise_progress(
            &mut tui,
            "Decoding",
            window_index,
            &committed_text,
            &draft_text,
        );
        let replayed_prefix_word_count = replayed_prefix
            .as_ref()
            .map(|prefix| {
                if prefix.words.is_empty() {
                    prefix
                        .token_ids
                        .len()
                        .max(sentence_word_tokens(&prefix.text).len())
                } else {
                    prefix.words.len()
                }
            })
            .unwrap_or(0);
        tui.log(format!(
            "decoding chunk {window_index}: audio={}..{}ms samples={} replayed_prefix_words={} start_position={}",
            (window.start_sample * 1000) / SAMPLE_RATE as usize,
            (window.end_sample * 1000) / SAMPLE_RATE as usize,
            chunk_samples.len(),
            replayed_prefix_word_count,
            start_position,
        ));
        let chunk_run = decode_chunk_followup_step(
            model,
            tokenizer,
            &mel_extractor,
            embed_dtype,
            chunk_samples,
            window_index,
            language,
            context,
            max_new_tokens,
            &mut cache,
            start_position,
            window.start_sample,
            window.end_sample,
            replayed_prefix.as_ref(),
        )?;

        let rollback = if window_index + 1 < window_plan.len() {
            let keep_until_sample = window.end_sample.saturating_sub(rollback_samples);
            let keep_until_secs =
                (keep_until_sample.saturating_sub(window.start_sample)) as f64 / SAMPLE_RATE as f64;
            let keep = timed_generated_prefix_for_cut(
                &mut align_ctx,
                tokenizer,
                &chunk_run,
                chunk_samples,
                keep_until_secs,
            )?;
            let kept_token_ids = kept_generated_token_ids(&chunk_run, keep.kept_token_count);
            let kept_text = decode_token_ids(tokenizer, &kept_token_ids)?;
            let carried_token_count =
                kept_carried_token_count(replayed_prefix.as_ref(), Some(keep_until_secs), None);
            let replay_prefix_token_count = replayed_prefix
                .as_ref()
                .map(|prefix| prefix.token_ids.len())
                .unwrap_or(0);
            let base_prompt_tokens = chunk_run
                .prompt_tokens
                .saturating_sub(replay_prefix_token_count);
            let rollback_position = chunk_run.start_position
                + base_prompt_tokens
                + carried_token_count
                + keep.kept_token_count;
            truncate_cache(&mut cache, rollback_position)?;
            start_position = rollback_position;
            Some(WindowRollbackDecision {
                keep_boundary_policy: KeepBoundaryPolicy::Fixed,
                target_keep_until_secs: keep_until_secs,
                keep_until_secs: Some(keep_until_secs),
                replay_until_secs: None,
                kept_word_count: keep.kept_word_count,
                kept_token_count: keep.kept_token_count,
                kept_token_ids,
                kept_text,
                bridge_token_ids: Vec::new(),
                bridge_text: None,
                rollback_position,
                keep_boundary_debug: KeepBoundaryDebug {
                    earliest_candidate_secs: keep_until_secs,
                    min_keep_secs: keep_until_secs,
                    snapped: false,
                    chosen_word: None,
                },
            })
        } else {
            start_position = chunk_run.end_position;
            None
        };
        draft_text.clear();
        draft_text.push_str(chunk_run.transcript.as_str());
        if let Some(rollback) = rollback.as_ref() {
            append_exact(
                &mut committed_text,
                decode_token_ids(tokenizer, &rollback.kept_token_ids)?.as_str(),
            );
            draft_text.clear();
        } else {
            append_exact(&mut committed_text, chunk_run.transcript.as_str());
            draft_text.clear();
        }
        tui.log(format!(
            "decoded chunk {window_index}: decode_ms={:.1} start_position={} end_position={} generated_tokens={} stop_reason={}",
            chunk_run.decode_ms,
            chunk_run.start_position,
            chunk_run.end_position,
            chunk_run.generated_tokens,
            chunk_run.stop_reason.as_str(),
        ));

        update_exercise_progress(
            &mut tui,
            if window_index + 1 < window_plan.len() {
                "Rolling"
            } else {
                "Finalizing"
            },
            window_index,
            &committed_text,
            &draft_text,
        );

        replay_prefix_for_next =
            (!chunk_run.generated_token_ids.is_empty()).then(|| CarriedBridge {
                token_ids: chunk_run.generated_token_ids.clone(),
                text: chunk_run.transcript.clone(),
                words: Vec::new(),
            });
        window_runs.push(SlidingWindowRun {
            chunk_run,
            rollback,
            replayed_prefix,
        });
        if window_index + 1 == window_plan.len() {
            tui.clear();
            print_finalizing_banner();
        }
    }

    tui.clear();

    let html_path = html::write_sliding_window_timed_rollback_html(
        "sliding-window-full-replay",
        &window_runs,
        samples,
        wav_path,
    )?;
    let committed_timeline_path = Some(html::write_committed_timeline_html(
        "sliding-window-full-replay-committed",
        &window_runs,
        samples,
        wav_path,
    )?);

    Ok(SlidingWindowTimedRollbackExperimentResult {
        mode_label: "sliding-window-full-replay",
        chunk_ms,
        rollback_ms,
        stride_ms: (stride_samples * 1000) / SAMPLE_RATE as usize,
        window_runs,
        html_path,
        committed_timeline_path,
        interrupted_early: false,
    })
}

/// Runs sliding-window decoding with a replayable bridge segment between windows.
pub(crate) fn decode_sliding_window_bridge_replay(
    model: &Qwen3ASRModel,
    tokenizer: &Tokenizer,
    samples: &[f32],
    thinker: &bee_qwen3_asr::config::ThinkerConfig,
    language: &str,
    context: &str,
    max_new_tokens: usize,
    chunk_ms: usize,
    bridge_ms: usize,
    max_bridge_windows: usize,
    stride_ms: Option<usize>,
    keep_boundary_policy: KeepBoundaryPolicy,
    rollback_ms: usize,
    wav_path: &Path,
    interrupted: &Arc<AtomicBool>,
) -> Result<SlidingWindowTimedRollbackExperimentResult> {
    let window_samples = ms_to_samples(chunk_ms)?;
    let rollback_samples = ms_to_samples(rollback_ms)?;
    let (committed_samples, bridge_samples) = if let Some(stride_ms) = stride_ms {
        let committed_samples = ms_to_samples(stride_ms)?;
        let bridge_samples = window_samples
            .checked_sub(committed_samples + rollback_samples)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "invalid bridge geometry: stride_ms={} rollback_ms={} chunk_ms={}",
                    stride_ms,
                    rollback_ms,
                    chunk_ms
                )
            })?;
        if bridge_samples == 0 {
            bail!(
                "bridge segment must be non-zero: stride_ms={} rollback_ms={} chunk_ms={}",
                stride_ms,
                rollback_ms,
                chunk_ms
            );
        }
        (committed_samples, bridge_samples)
    } else {
        let committed_samples = window_samples
            .checked_sub(ms_to_samples(bridge_ms)? + rollback_samples)
            .ok_or_else(|| anyhow::anyhow!("committed segment underflow"))?;
        let bridge_samples = ms_to_samples(bridge_ms)?;
        if committed_samples == 0 {
            bail!(
                "committed segment must be non-zero: bridge_ms={} rollback_ms={} chunk_ms={}",
                bridge_ms,
                rollback_ms,
                chunk_ms
            );
        }
        (committed_samples, bridge_samples)
    };

    let stride_samples = committed_samples;
    let mel_extractor = MelExtractor::new(
        N_FFT,
        HOP_LENGTH,
        thinker.audio_config.num_mel_bins,
        SAMPLE_RATE,
    );
    let probe_ids = Array::from_slice(&[0_i32], &[1, 1]);
    let embed_dtype = model.model.embed_tokens.forward(&probe_ids)?.dtype();

    let mut cache = None;
    let mut start_position = 0usize;
    let mut window_runs = Vec::new();
    let mut replay_prefix_for_next: Option<CarriedBridge> = None;
    let mut next_window_start = 0usize;
    let mut window_index = 0usize;
    let mut unresolved_keep_samples = committed_samples;
    let mut align_ctx = AlignmentContext::new()?;
    let mut interrupted_early = false;
    let mut tui = ExerciseTui::new();
    let mut committed_token_ids: Vec<u32> = Vec::new();
    let mut draft_token_ids: Vec<u32> = Vec::new();
    let mut committed_text = String::new();
    let mut draft_text = String::new();

    while next_window_start < samples.len() {
        if interrupted.load(Ordering::SeqCst) {
            interrupted_early = true;
            tui.clear();
            eprintln!(
                "interrupt received: stopping before chunk {window_index} and writing reports"
            );
            break;
        }
        if window_index >= max_bridge_windows {
            bail!(
                "bridge replay reached {} windows, exceeding limit of {}; increase stride or chunk size",
                window_index,
                max_bridge_windows
            );
        }

        let current_window_samples = unresolved_keep_samples + bridge_samples + rollback_samples;
        let window = ChunkPlanEntry {
            chunk_index: window_index,
            start_sample: next_window_start,
            end_sample: (next_window_start + current_window_samples).min(samples.len()),
        };
        let chunk_samples = &samples[window.start_sample..window.end_sample];
        let replayed_prefix = replay_prefix_for_next.clone();
        let replayed_prefix_text = carried_bridge_text(tokenizer, replayed_prefix.as_ref())?;
        update_exercise_progress(
            &mut tui,
            "Decoding",
            window_index,
            &committed_text,
            &draft_text,
        );
        tui.log(format!(
            "decoding chunk {window_index}: audio={}..{}ms samples={} replayed_prefix_words={} start_position={}",
            (window.start_sample * 1000) / SAMPLE_RATE as usize,
            (window.end_sample * 1000) / SAMPLE_RATE as usize,
            chunk_samples.len(),
            sentence_word_tokens(&replayed_prefix_text).len(),
            start_position,
        ));
        let chunk_run = decode_chunk_followup_step(
            model,
            tokenizer,
            &mel_extractor,
            embed_dtype,
            chunk_samples,
            window_index,
            language,
            context,
            max_new_tokens,
            &mut cache,
            start_position,
            window.start_sample,
            window.end_sample,
            replayed_prefix.as_ref(),
        )?;
        tui.log(format!(
            "decoded chunk {window_index}: decode_ms={:.1} start_position={} end_position={} generated_tokens={} stop_reason={}",
            chunk_run.decode_ms,
            chunk_run.start_position,
            chunk_run.end_position,
            chunk_run.generated_tokens,
            chunk_run.stop_reason.as_str(),
        ));
        draft_token_ids.clear();
        if let Some(prefix) = replayed_prefix.as_ref() {
            draft_token_ids.extend_from_slice(&prefix.token_ids);
        }
        draft_token_ids.extend_from_slice(&chunk_run.generated_token_ids);
        draft_text.clear();
        append_display_delta(&mut draft_text, replayed_prefix_text.as_str());
        append_display_delta(&mut draft_text, chunk_run.transcript.as_str());
        update_exercise_progress(
            &mut tui,
            if window.end_sample < samples.len() {
                "Rolling"
            } else {
                "Finalizing"
            },
            window_index,
            &committed_text,
            &draft_text,
        );

        let has_next_window = window.end_sample < samples.len();
        let rollback = if has_next_window {
            let generated_transcript = normalized_transcript(&chunk_run.transcript);
            let normalized_replayed_prefix = normalized_transcript(&replayed_prefix_text);
            let combined_transcript =
                match Some(normalized_replayed_prefix).filter(|text| !text.is_empty()) {
                    Some(prefix) if !prefix.is_empty() && !generated_transcript.is_empty() => {
                        format!("{prefix} {generated_transcript}")
                    }
                    Some(prefix) if !prefix.is_empty() => prefix.to_string(),
                    _ => generated_transcript.to_string(),
                };
            let alignment =
                build_transcript_alignment(&mut align_ctx, &combined_transcript, chunk_samples)?;
            let target_keep_until_secs = unresolved_keep_samples as f64 / SAMPLE_RATE as f64;
            let replay_until_secs =
                (unresolved_keep_samples + bridge_samples) as f64 / SAMPLE_RATE as f64;
            let (candidate_keep_until_secs, keep_boundary_debug) = adjust_keep_boundary_secs(
                keep_boundary_policy,
                &alignment,
                target_keep_until_secs,
                replay_until_secs,
            )?;
            let timed_words = timed_aligned_words_for_alignment(&combined_transcript, &alignment)?;
            let found_boundary =
                keep_boundary_policy == KeepBoundaryPolicy::Fixed || keep_boundary_debug.snapped;
            let keep_until_secs = if found_boundary {
                Some(candidate_keep_until_secs)
            } else {
                None
            };
            let split = timed_generated_bridge_for_cuts(
                tokenizer,
                &combined_transcript,
                replayed_prefix.as_ref(),
                &timed_words,
                keep_until_secs.unwrap_or(0.0),
                replay_until_secs,
                keep_boundary_debug.chosen_word.as_ref(),
            )?;
            let carried_token_count = kept_carried_token_count(
                replayed_prefix.as_ref(),
                keep_until_secs,
                keep_boundary_debug.chosen_word.as_ref(),
            );
            let mut kept_token_ids =
                kept_carried_token_ids(replayed_prefix.as_ref(), carried_token_count);
            kept_token_ids.extend(kept_generated_token_ids(&chunk_run, split.kept_token_count));
            let kept_text = split.kept_text.clone();
            if keep_until_secs.is_some() && !kept_text.is_empty() {
                tui.log(format!("kept words: {}", kept_text));
            }
            let replay_prefix_token_count = replayed_prefix
                .as_ref()
                .map(|prefix| prefix.token_ids.len())
                .unwrap_or(0);
            let base_prompt_tokens = chunk_run
                .prompt_tokens
                .saturating_sub(replay_prefix_token_count);
            let rollback_position = if keep_until_secs.is_some()
                && (carried_token_count > 0 || split.kept_token_count > 0)
            {
                chunk_run.start_position
                    + base_prompt_tokens
                    + carried_token_count
                    + split.kept_token_count
            } else {
                chunk_run.start_position
            };
            truncate_cache(&mut cache, rollback_position)?;
            start_position = rollback_position;
            replay_prefix_for_next =
                (!split.bridge.text.is_empty()).then_some(split.bridge.clone());
            let bridge_text = if split.bridge.token_ids.is_empty() {
                None
            } else {
                Some(decode_token_ids(tokenizer, &split.bridge.token_ids)?)
            };
            let rollback = WindowRollbackDecision {
                keep_boundary_policy,
                target_keep_until_secs,
                keep_until_secs,
                replay_until_secs: Some(replay_until_secs),
                kept_word_count: sentence_word_tokens(&kept_text).len(),
                kept_token_count: kept_token_ids.len(),
                kept_token_ids,
                kept_text,
                bridge_token_ids: split.bridge.token_ids.clone(),
                bridge_text,
                rollback_position,
                keep_boundary_debug,
            };
            committed_token_ids.extend_from_slice(&rollback.kept_token_ids);
            draft_token_ids.clear();
            let committed_delta = if rollback.keep_until_secs.is_some() {
                suffix_after_prefix(
                    Some(replayed_prefix_text.as_str()),
                    rollback.kept_text.as_str(),
                )
            } else {
                ""
            };
            append_display_delta(&mut committed_text, committed_delta);
            draft_text.clear();
            if let Some(prefix) = replay_prefix_for_next.as_ref() {
                draft_token_ids.extend_from_slice(&prefix.token_ids);
                append_display_delta(
                    &mut draft_text,
                    carried_bridge_text(tokenizer, Some(prefix))?.as_str(),
                );
            }
            Some(rollback)
        } else {
            start_position = chunk_run.end_position;
            replay_prefix_for_next = None;
            if let Some(prefix) = replayed_prefix.as_ref() {
                committed_token_ids.extend_from_slice(&prefix.token_ids);
                append_display_delta(&mut committed_text, replayed_prefix_text.as_str());
            }
            committed_token_ids.extend_from_slice(&chunk_run.generated_token_ids);
            append_display_delta(&mut committed_text, chunk_run.transcript.as_str());
            draft_token_ids.clear();
            draft_text.clear();
            None
        };

        window_runs.push(SlidingWindowRun {
            chunk_run,
            rollback,
            replayed_prefix,
        });
        if !has_next_window {
            tui.clear();
            print_finalizing_banner();
            break;
        }
        next_window_start =
            if let Some(rollback) = window_runs.last().and_then(|run| run.rollback.as_ref()) {
                let next_start = if let Some(keep_until_secs) = rollback.keep_until_secs {
                    window.start_sample + ((keep_until_secs * SAMPLE_RATE as f64).round() as usize)
                } else {
                    window.start_sample
                };
                // Preserve the full replay tail after the chosen keep cut, then add one new
                // committed segment so windows never shrink after snapping the boundary earlier.
                let replay_tail_samples = window.end_sample.saturating_sub(next_start);
                unresolved_keep_samples = replay_tail_samples
                    .saturating_add(committed_samples)
                    .saturating_sub(bridge_samples + rollback_samples);
                next_start
            } else {
                next_window_start.saturating_add(stride_samples)
            };
        window_index += 1;
    }

    tui.clear();

    let html_path = html::write_sliding_window_timed_rollback_html(
        "sliding-window-bridge-replay",
        &window_runs,
        samples,
        wav_path,
    )?;
    let committed_timeline_path = Some(html::write_committed_timeline_html(
        "sliding-window-bridge-replay-committed",
        &window_runs,
        samples,
        wav_path,
    )?);

    Ok(SlidingWindowTimedRollbackExperimentResult {
        mode_label: "sliding-window-bridge-replay",
        chunk_ms,
        rollback_ms,
        stride_ms: (stride_samples * 1000) / SAMPLE_RATE as usize,
        window_runs,
        html_path,
        committed_timeline_path,
        interrupted_early,
    })
}

/// Runs the chunk-segment merge rollback experiment on the first three chunks.
pub(crate) fn decode_chunk_segment_merge_rollback(
    model: &Qwen3ASRModel,
    tokenizer: &Tokenizer,
    samples: &[f32],
    thinker: &bee_qwen3_asr::config::ThinkerConfig,
    language: &str,
    context: &str,
    max_new_tokens: usize,
    chunk_ms: usize,
) -> Result<ChunkSegmentMergeRollbackExperimentResult> {
    let chunk_size_samples = ms_to_samples(chunk_ms)?;
    let chunk_plan = build_chunk_plan(samples.len(), chunk_size_samples, chunk_size_samples)?;
    if chunk_plan.len() < 3 {
        bail!(
            "chunk-segment-merge-rollback needs at least 3 chunks; got {}",
            chunk_plan.len()
        );
    }

    let baseline_runs = run_chunked_followup_sequence(
        model,
        tokenizer,
        samples,
        thinker,
        language,
        context,
        max_new_tokens,
        &chunk_plan[..3],
        None,
    )?;

    let mel_extractor = MelExtractor::new(
        N_FFT,
        HOP_LENGTH,
        thinker.audio_config.num_mel_bins,
        SAMPLE_RATE,
    );
    let probe_ids = Array::from_slice(&[0_i32], &[1, 1]);
    let embed_dtype = model.model.embed_tokens.forward(&probe_ids)?.dtype();

    let mut cache = None;
    let chunk0 = &chunk_plan[0];
    let chunk1 = &chunk_plan[1];
    let chunk2 = &chunk_plan[2];

    let replay_chunk0 = decode_chunk_followup_step(
        model,
        tokenizer,
        &mel_extractor,
        embed_dtype,
        &samples[chunk0.start_sample..chunk0.end_sample],
        0,
        language,
        context,
        max_new_tokens,
        &mut cache,
        0,
        chunk0.start_sample,
        chunk0.end_sample,
        None,
    )?;

    let _original_chunk1 = decode_chunk_followup_step(
        model,
        tokenizer,
        &mel_extractor,
        embed_dtype,
        &samples[chunk1.start_sample..chunk1.end_sample],
        1,
        language,
        context,
        max_new_tokens,
        &mut cache,
        replay_chunk0.end_position,
        chunk1.start_sample,
        chunk1.end_sample,
        None,
    )?;

    truncate_cache(&mut cache, replay_chunk0.end_position)?;

    let merged_chunk = decode_chunk_followup_step(
        model,
        tokenizer,
        &mel_extractor,
        embed_dtype,
        &samples[chunk1.start_sample..chunk2.end_sample],
        1,
        language,
        context,
        max_new_tokens,
        &mut cache,
        replay_chunk0.end_position,
        chunk1.start_sample,
        chunk2.end_sample,
        None,
    )?;

    let baseline_annotated = annotate_chunk_runs(&baseline_runs, samples)?;
    let replay_annotated = annotate_chunk_runs(
        &[
            clone_chunk_run(&replay_chunk0),
            clone_chunk_run(&merged_chunk),
        ],
        samples,
    )?;
    let html_path = html::write_chunk_segment_merge_rollback_html(
        &baseline_runs,
        &[
            clone_chunk_run(&replay_chunk0),
            clone_chunk_run(&merged_chunk),
        ],
        samples,
    )?;

    Ok(ChunkSegmentMergeRollbackExperimentResult {
        chunk_ms,
        baseline_runs,
        replay_chunk0,
        merged_chunk,
        baseline_annotated,
        replay_annotated,
        html_path,
    })
}

/// Sweeps boundary offsets to compare chunk merge rollback behavior.
pub(crate) fn decode_chunk_segment_merge_boundary_sweep(
    model: &Qwen3ASRModel,
    tokenizer: &Tokenizer,
    samples: &[f32],
    thinker: &bee_qwen3_asr::config::ThinkerConfig,
    language: &str,
    context: &str,
    max_new_tokens: usize,
    chunk_ms: usize,
) -> Result<ChunkSegmentMergeBoundarySweepExperimentResult> {
    let chunk_size_samples = ms_to_samples(chunk_ms)?;
    let chunk_plan = build_chunk_plan(samples.len(), chunk_size_samples, chunk_size_samples)?;
    if chunk_plan.len() < 3 {
        bail!(
            "chunk-segment-merge-boundary-sweep needs at least 3 chunks; got {}",
            chunk_plan.len()
        );
    }

    let baseline_runs = run_chunked_followup_sequence(
        model,
        tokenizer,
        samples,
        thinker,
        language,
        context,
        max_new_tokens,
        &chunk_plan[..3],
        None,
    )?;

    let mel_extractor = MelExtractor::new(
        N_FFT,
        HOP_LENGTH,
        thinker.audio_config.num_mel_bins,
        SAMPLE_RATE,
    );
    let probe_ids = Array::from_slice(&[0_i32], &[1, 1]);
    let embed_dtype = model.model.embed_tokens.forward(&probe_ids)?.dtype();

    let chunk0 = &chunk_plan[0];
    let chunk1 = &chunk_plan[1];
    let chunk2 = &chunk_plan[2];

    let mut sweep_results = Vec::new();
    let exact_boundary = baseline_runs[0].end_position;
    for offset in BOUNDARY_SWEEP_OFFSETS {
        let mut cache = None;

        let replay_chunk0 = decode_chunk_followup_step(
            model,
            tokenizer,
            &mel_extractor,
            embed_dtype,
            &samples[chunk0.start_sample..chunk0.end_sample],
            0,
            language,
            context,
            max_new_tokens,
            &mut cache,
            0,
            chunk0.start_sample,
            chunk0.end_sample,
            None,
        )?;

        let _chunk1 = decode_chunk_followup_step(
            model,
            tokenizer,
            &mel_extractor,
            embed_dtype,
            &samples[chunk1.start_sample..chunk1.end_sample],
            1,
            language,
            context,
            max_new_tokens,
            &mut cache,
            replay_chunk0.end_position,
            chunk1.start_sample,
            chunk1.end_sample,
            None,
        )?;

        let rollback_position = if offset.is_negative() {
            exact_boundary.saturating_sub(offset.unsigned_abs())
        } else {
            exact_boundary.saturating_add(offset as usize)
        };
        truncate_cache(&mut cache, rollback_position)?;

        let merged_chunk = decode_chunk_followup_step(
            model,
            tokenizer,
            &mel_extractor,
            embed_dtype,
            &samples[chunk1.start_sample..chunk2.end_sample],
            1,
            language,
            context,
            max_new_tokens,
            &mut cache,
            rollback_position,
            chunk1.start_sample,
            chunk2.end_sample,
            None,
        )?;

        sweep_results.push(BoundarySweepResult {
            offset,
            rollback_position,
            merged_chunk,
        });
    }

    Ok(ChunkSegmentMergeBoundarySweepExperimentResult {
        chunk_ms,
        exact_boundary,
        baseline_runs,
        sweep_results,
    })
}

/// Decodes a single prefix window for prefix-rerun comparisons.
pub(crate) fn decode_prefix_window(
    model: &Qwen3ASRModel,
    tokenizer: &Tokenizer,
    mel_extractor: &MelExtractor,
    embed_dtype: bee_qwen3_asr::mlx_rs::Dtype,
    prefix_samples: &[f32],
    language: &str,
    context: &str,
    max_new_tokens: usize,
    label: String,
) -> Result<PrefixResult> {
    let (mel_data, n_mels, n_frames) = mel_extractor
        .extract(prefix_samples)
        .with_context(|| format!("extracting log-mel for {label}"))?;
    let mel = Array::from_slice(&mel_data, &[n_mels as i32, n_frames as i32]);
    let audio_features = model
        .encode_audio(&mel)
        .with_context(|| format!("encoding audio for {label}"))?;
    let n_audio_tokens = audio_features.shape()[0] as usize;
    let audio_features = ops::expand_dims(&audio_features, 0)?;
    let audio_features = audio_features.as_dtype(embed_dtype)?;

    let prompt_tokens = build_initial_prompt(n_audio_tokens, language, context, tokenizer);
    let experiment = decode_prompt(
        model,
        tokenizer,
        &audio_features,
        prompt_tokens,
        max_new_tokens,
        0,
        label,
    )?;

    Ok(PrefixResult {
        label: experiment.label,
        prompt_tokens: experiment.prompt_tokens,
        generated_tokens: experiment.generated_tokens,
        transcript: experiment.transcript,
        sample_count: prefix_samples.len(),
    })
}

/// Runs chunked follow-up decoding over a plan of chunk boundaries.
pub(crate) fn run_chunked_followup_sequence(
    model: &Qwen3ASRModel,
    tokenizer: &Tokenizer,
    samples: &[f32],
    thinker: &bee_qwen3_asr::config::ThinkerConfig,
    language: &str,
    context: &str,
    max_new_tokens: usize,
    chunk_plan: &[ChunkPlanEntry],
    replay_plan: Option<ReplayPlan>,
) -> Result<Vec<ChunkRun>> {
    let mel_extractor = MelExtractor::new(
        N_FFT,
        HOP_LENGTH,
        thinker.audio_config.num_mel_bins,
        SAMPLE_RATE,
    );
    let probe_ids = Array::from_slice(&[0_i32], &[1, 1]);
    let embed_dtype = model.model.embed_tokens.forward(&probe_ids)?.dtype();

    let mut cache = None;
    let mut start_position = 0usize;
    let mut chunk_runs = Vec::new();

    for chunk in chunk_plan {
        let chunk_index = chunk.chunk_index;
        let chunk_samples = &samples[chunk.start_sample..chunk.end_sample];
        if let Some(plan) = replay_plan {
            if chunk_index == plan.replay_from_chunk_index {
                truncate_cache(&mut cache, plan.rollback_position).with_context(|| {
                    format!(
                        "truncating cache for policy {} before replay chunk {}",
                        plan.rollback_policy.as_str(),
                        chunk_index
                    )
                })?;
                start_position = plan.rollback_position;
            }
            if chunk_index > plan.rollback_after_chunk_index
                && chunk_index < plan.replay_from_chunk_index
            {
                continue;
            }
        }

        let chunk_run = decode_chunk_followup_step(
            model,
            tokenizer,
            &mel_extractor,
            embed_dtype,
            chunk_samples,
            chunk_index,
            language,
            context,
            max_new_tokens,
            &mut cache,
            start_position,
            chunk.start_sample,
            chunk.end_sample,
            None,
        )?;
        start_position = chunk_run.end_position;
        chunk_runs.push(chunk_run);
    }

    Ok(chunk_runs)
}

/// Decodes one chunk with the correct prompt, cache state, and replay prefix.
pub(crate) fn decode_chunk_followup_step(
    model: &Qwen3ASRModel,
    tokenizer: &Tokenizer,
    mel_extractor: &MelExtractor,
    embed_dtype: bee_qwen3_asr::mlx_rs::Dtype,
    chunk_samples: &[f32],
    chunk_index: usize,
    language: &str,
    context: &str,
    max_new_tokens: usize,
    cache: &mut Option<bee_qwen3_asr::decoder::KVCache>,
    start_position: usize,
    start_sample: usize,
    end_sample: usize,
    replay_prefix: Option<&CarriedBridge>,
) -> Result<ChunkRun> {
    let decode_start = Instant::now();
    let (mel_data, n_mels, n_frames) = mel_extractor
        .extract(chunk_samples)
        .with_context(|| format!("extracting log-mel for chunk {chunk_index}"))?;
    let mel = Array::from_slice(&mel_data, &[n_mels as i32, n_frames as i32]);
    let audio_features = model
        .encode_audio(&mel)
        .with_context(|| format!("encoding chunk {chunk_index}"))?;
    let n_audio_tokens = audio_features.shape()[0] as usize;
    let audio_features = ops::expand_dims(&audio_features, 0)?;
    let audio_features = audio_features.as_dtype(embed_dtype)?;

    let (label, mut prompt_tokens) = if chunk_index == 0 {
        (
            format!("chunk {chunk_index} initial"),
            build_initial_prompt(n_audio_tokens, language, context, tokenizer),
        )
    } else {
        (
            format!("chunk {chunk_index} followup"),
            build_followup_prompt(n_audio_tokens, language, tokenizer),
        )
    };
    if let Some(prefix) = replay_prefix.filter(|prefix| !prefix.token_ids.is_empty()) {
        prompt_tokens.extend(prompt_tokens_from_token_ids(&prefix.token_ids)?);
    }

    let prompt_len = prompt_tokens.len();
    let (generated, _confidence, end_position, stop_reason) = prefill_and_decode(
        model,
        &prompt_tokens,
        &audio_features,
        cache,
        start_position,
        max_new_tokens,
        ConfidenceMode::Streaming,
    )
    .with_context(|| format!("decoding {label}"))?;

    let token_ids: Vec<u32> = generated
        .iter()
        .map(|&id| u32::try_from(id).context("generated negative token id"))
        .collect::<Result<_>>()?;
    let transcript = tokenizer
        .decode(&token_ids, true)
        .map_err(|e| anyhow::anyhow!("decoding transcript: {e}"))?;

    Ok(ChunkRun {
        label,
        prompt_tokens: prompt_len,
        generated_tokens: generated.len(),
        generated_token_ids: token_ids,
        transcript,
        sample_count: chunk_samples.len(),
        decode_ms: decode_start.elapsed().as_secs_f64() * 1000.0,
        stop_reason,
        start_position,
        end_position,
        start_sample,
        end_sample,
    })
}

/// Truncates the KV cache to a target rollback position.
pub(crate) fn truncate_cache(
    cache: &mut Option<bee_qwen3_asr::decoder::KVCache>,
    rollback_position: usize,
) -> Result<()> {
    let cache = cache
        .as_mut()
        .ok_or_else(|| anyhow::anyhow!("cache missing before truncate"))?;
    cache.truncate(rollback_position);
    Ok(())
}

/// Trims leading and trailing whitespace from a transcript.
pub(crate) fn normalized_transcript(text: &str) -> &str {
    text.trim()
}

/// Decodes token IDs into a transcript string.
pub(crate) fn decode_token_ids(tokenizer: &Tokenizer, token_ids: &[u32]) -> Result<String> {
    tokenizer
        .decode(token_ids, true)
        .map_err(|e| anyhow::anyhow!("decoding transcript tokens: {e}"))
}

/// Converts unsigned token IDs into signed prompt token IDs.
pub(crate) fn prompt_tokens_from_token_ids(token_ids: &[u32]) -> Result<Vec<i32>> {
    token_ids
        .iter()
        .map(|&id| i32::try_from(id).context("prompt token id overflow"))
        .collect()
}

/// Tokenizes prompt text into token IDs without adding special tokens.
pub(crate) fn tokenize_token_ids(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    Ok(tokenizer
        .encode_fast(text, false)
        .map_err(|e| anyhow::anyhow!("encoding prompt text: {e}"))?
        .get_ids()
        .to_vec())
}

/// Returns the leading generated token IDs that should be kept.
pub(crate) fn kept_generated_token_ids(chunk_run: &ChunkRun, kept_token_count: usize) -> Vec<u32> {
    chunk_run.generated_token_ids[..kept_token_count.min(chunk_run.generated_token_ids.len())]
        .to_vec()
}

/// Returns the leading carried-bridge token IDs that should be kept.
pub(crate) fn kept_carried_token_ids(
    prefix: Option<&CarriedBridge>,
    kept_token_count: usize,
) -> Vec<u32> {
    let Some(prefix) = prefix else {
        return Vec::new();
    };
    prefix.token_ids[..kept_token_count.min(prefix.token_ids.len())].to_vec()
}

/// Counts how many carried-bridge tokens should survive a rollback cut.
pub(crate) fn kept_carried_token_count(
    prefix: Option<&CarriedBridge>,
    keep_until_secs: Option<f64>,
    chosen_word: Option<&BoundaryWordDebug>,
) -> usize {
    let Some(prefix) = prefix else {
        return 0;
    };
    if let Some(chosen_word) = chosen_word {
        if chosen_word.word_index < prefix.words.len() {
            return prefix.words[chosen_word.word_index].token_range.end;
        }
    }
    let Some(keep_until_secs) = keep_until_secs else {
        return 0;
    };
    prefix
        .words
        .iter()
        .take_while(|word| word.end_secs <= keep_until_secs)
        .last()
        .map(|word| word.token_range.end)
        .unwrap_or(0)
}

/// Returns the decoded text for a carried bridge, if any.
pub(crate) fn carried_bridge_text(
    tokenizer: &Tokenizer,
    bridge: Option<&CarriedBridge>,
) -> Result<String> {
    match bridge {
        Some(bridge) if !bridge.token_ids.is_empty() => {
            decode_token_ids(tokenizer, &bridge.token_ids)
        }
        Some(bridge) => Ok(bridge.text.clone()),
        None => Ok(String::new()),
    }
}

/// Concatenates chunk transcripts without inserting separators.
pub(crate) fn combine_transcripts(chunks: &[ChunkRun]) -> String {
    let mut combined = String::new();
    for chunk in chunks {
        if chunk.transcript.is_empty() {
            continue;
        }
        combined.push_str(&chunk.transcript);
    }
    combined
}

/// Builds a chunk plan from the total sample count and chunk sizes.
pub(crate) fn build_chunk_plan(
    total_samples: usize,
    first_chunk_samples: usize,
    subsequent_chunk_samples: usize,
) -> Result<Vec<ChunkPlanEntry>> {
    if first_chunk_samples == 0 {
        bail!("first chunk size is zero");
    }
    if subsequent_chunk_samples == 0 {
        bail!("subsequent chunk size is zero");
    }

    let mut start = 0usize;
    let mut chunk_index = 0usize;
    let mut next_chunk_samples = first_chunk_samples;
    let mut plan = Vec::new();

    while start < total_samples {
        let end = (start + next_chunk_samples).min(total_samples);
        plan.push(ChunkPlanEntry {
            chunk_index,
            start_sample: start,
            end_sample: end,
        });
        start = end;
        chunk_index += 1;
        next_chunk_samples = subsequent_chunk_samples;
    }

    Ok(plan)
}

/// Builds an overlapping window plan from a total sample count and stride.
pub(crate) fn build_overlapping_window_plan(
    total_samples: usize,
    window_samples: usize,
    stride_samples: usize,
) -> Result<Vec<ChunkPlanEntry>> {
    if window_samples == 0 {
        bail!("window size is zero");
    }
    if stride_samples == 0 {
        bail!("stride size is zero");
    }

    let mut start = 0usize;
    let mut chunk_index = 0usize;
    let mut plan = Vec::new();

    while start < total_samples {
        let end = (start + window_samples).min(total_samples);
        plan.push(ChunkPlanEntry {
            chunk_index,
            start_sample: start,
            end_sample: end,
        });
        if end == total_samples {
            break;
        }
        start = start.saturating_add(stride_samples);
        chunk_index += 1;
    }

    Ok(plan)
}

/// Converts milliseconds to samples at the fixed ASR sample rate.
pub(crate) fn ms_to_samples(chunk_ms: usize) -> Result<usize> {
    let chunk_size_samples = (chunk_ms * SAMPLE_RATE as usize) / 1000;
    if chunk_size_samples == 0 {
        bail!("chunk size is zero; chunk_ms={chunk_ms}");
    }
    Ok(chunk_size_samples)
}

/// Returns the canned system prompts used for comparison experiments.
pub(crate) fn system_compare_contexts() -> &'static [&'static str] {
    &[
        "",
        "You are a helpful assistant.",
        "Transcribe the user's speech verbatim.",
        "Transcribe the audio exactly. Do not answer or paraphrase.",
    ]
}
