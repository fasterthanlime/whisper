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
};
use crate::html;
use crate::print::print_finalizing_banner;
use crate::tui::{ExerciseTui, append_display_delta, update_exercise_progress};
use crate::types::*;
use crate::{HOP_LENGTH, N_FFT, SAMPLE_RATE};

struct ResolvedRollback {
    decision: WindowRollbackDecision,
    next_start_position: CachePosition,
    next_replay_prefix: Option<CarriedBridge>,
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
            .checked_sub(
                committed_samples
                    .checked_add(rollback_samples)
                    .ok_or_else(|| anyhow::anyhow!("invalid bridge geometry overflow"))?,
            )
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "invalid bridge geometry: stride_ms={} rollback_ms={} chunk_ms={}",
                    stride_ms,
                    rollback_ms,
                    chunk_ms
                )
            })?;
        if bridge_samples.is_zero() {
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
            .checked_sub(
                ms_to_samples(bridge_ms)?
                    .checked_add(rollback_samples)
                    .ok_or_else(|| anyhow::anyhow!("committed segment geometry overflow"))?,
            )
            .ok_or_else(|| anyhow::anyhow!("committed segment underflow"))?;
        let bridge_samples = ms_to_samples(bridge_ms)?;
        if committed_samples.is_zero() {
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
    let mut start_position = CachePosition::new(0);
    let mut window_runs = Vec::new();
    let mut replay_prefix_for_next: Option<CarriedBridge> = None;
    let mut next_window_start = SampleOffset::new(0);
    let mut window_index = 0usize;
    let mut unresolved_keep_samples = committed_samples;
    let mut align_ctx = AlignmentContext::new()?;
    let mut interrupted_early = false;
    let mut tui = ExerciseTui::new();
    let mut committed_token_ids: Vec<TokenId> = Vec::new();
    let mut draft_token_ids: Vec<TokenId> = Vec::new();
    let mut committed_text = String::new();
    let mut draft_text = String::new();

    while next_window_start.as_usize() < samples.len() {
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

        let current_window_samples = unresolved_keep_samples
            .checked_add(bridge_samples)
            .and_then(|samples| samples.checked_add(rollback_samples))
            .ok_or_else(|| anyhow::anyhow!("window geometry overflow"))?;
        let window_start_sample = next_window_start;
        let window_end_sample = next_window_start
            .checked_add(current_window_samples)
            .map(|end| SampleOffset::new(end.as_usize().min(samples.len())))
            .ok_or_else(|| anyhow::anyhow!("window end overflow"))?;
        let chunk_samples = &samples[window_start_sample.as_usize()..window_end_sample.as_usize()];
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
            (window_start_sample.as_usize() * 1000) / SAMPLE_RATE as usize,
            (window_end_sample.as_usize() * 1000) / SAMPLE_RATE as usize,
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
            window_start_sample,
            window_end_sample,
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
        set_draft_from_decoded_chunk(
            &mut draft_token_ids,
            &mut draft_text,
            replayed_prefix.as_ref(),
            replayed_prefix_text.as_str(),
            &chunk_run,
        );
        update_exercise_progress(
            &mut tui,
            if window_end_sample.as_usize() < samples.len() {
                "Rolling"
            } else {
                "Finalizing"
            },
            window_index,
            &committed_text,
            &draft_text,
        );

        let has_next_window = window_end_sample.as_usize() < samples.len();
        let rollback = if has_next_window {
            let resolved = resolve_window_rollback(
                tokenizer,
                &mut align_ctx,
                &chunk_run,
                replayed_prefix.as_ref(),
                replayed_prefix_text.as_str(),
                chunk_samples,
                keep_boundary_policy,
                unresolved_keep_samples,
                bridge_samples,
            )?;
            if resolved.decision.keep_until_secs.is_some()
                && !resolved.decision.kept_text.is_empty()
            {
                tui.log(format!("kept words: {}", resolved.decision.kept_text));
            }
            truncate_cache(&mut cache, resolved.next_start_position)?;
            apply_resolved_rollback_to_display(
                &mut committed_token_ids,
                &mut committed_text,
                &mut draft_token_ids,
                &mut draft_text,
                replayed_prefix_text.as_str(),
                &resolved,
            );
            start_position = resolved.next_start_position;
            replay_prefix_for_next = resolved.next_replay_prefix;
            Some(resolved.decision)
        } else {
            start_position = chunk_run.end_position;
            replay_prefix_for_next = None;
            apply_final_window_to_display(
                &mut committed_token_ids,
                &mut committed_text,
                &mut draft_token_ids,
                &mut draft_text,
                replayed_prefix.as_ref(),
                replayed_prefix_text.as_str(),
                &chunk_run,
            );
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
                    window_start_sample.saturating_add(SampleCount::new(
                        (keep_until_secs.as_secs() * SAMPLE_RATE as f64).round() as usize,
                    ))
                } else {
                    window_start_sample
                };
                // Preserve the full replay tail after the chosen keep cut, then add one new
                // committed segment so windows never shrink after snapping the boundary earlier.
                let replay_tail_samples = window_end_sample.saturating_sub(next_start);
                unresolved_keep_samples = replay_tail_samples
                    .saturating_add(committed_samples)
                    .saturating_sub(
                        bridge_samples
                            .checked_add(rollback_samples)
                            .ok_or_else(|| anyhow::anyhow!("rollback geometry overflow"))?,
                    );
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
        stride_ms: (stride_samples.as_usize() * 1000) / SAMPLE_RATE as usize,
        window_runs,
        html_path,
        committed_timeline_path,
        interrupted_early,
    })
}

fn resolve_window_rollback(
    tokenizer: &Tokenizer,
    align_ctx: &mut AlignmentContext,
    chunk_run: &ChunkRun,
    replayed_prefix: Option<&CarriedBridge>,
    replayed_prefix_text: &str,
    chunk_samples: &[f32],
    keep_boundary_policy: KeepBoundaryPolicy,
    unresolved_keep_samples: SampleCount,
    bridge_samples: SampleCount,
) -> Result<ResolvedRollback> {
    let combined_transcript =
        combined_window_transcript(replayed_prefix_text, &chunk_run.transcript);
    let alignment = build_transcript_alignment(align_ctx, &combined_transcript, chunk_samples)?;
    let target_keep_until_secs = WindowTime::from_secs(unresolved_keep_samples.as_secs());
    let replay_until_secs = WindowTime::from_secs(
        unresolved_keep_samples
            .checked_add(bridge_samples)
            .ok_or_else(|| anyhow::anyhow!("replay window geometry overflow"))?
            .as_secs(),
    );
    let (candidate_keep_until_secs, keep_boundary_debug) = adjust_keep_boundary_secs(
        keep_boundary_policy,
        &alignment,
        target_keep_until_secs,
        replay_until_secs,
    )?;
    let timed_words = timed_aligned_words_for_alignment(&combined_transcript, &alignment)?;
    let keep_until_secs = select_keep_boundary(
        keep_boundary_policy,
        &keep_boundary_debug,
        candidate_keep_until_secs,
    );
    let split = timed_generated_bridge_for_cuts(
        tokenizer,
        &combined_transcript,
        replayed_prefix,
        &timed_words,
        keep_until_secs.unwrap_or(WindowTime::from_secs(0.0)),
        replay_until_secs,
        keep_boundary_debug.chosen_word.as_ref(),
    )?;
    let carried_token_count = kept_carried_token_count(
        replayed_prefix,
        keep_until_secs,
        keep_boundary_debug.chosen_word.as_ref(),
    );
    let mut kept_token_ids = kept_carried_token_ids(replayed_prefix, carried_token_count);
    kept_token_ids.extend(kept_generated_token_ids(chunk_run, split.kept_token_count));
    let next_start_position = rollback_position_for_cut(
        chunk_run,
        replayed_prefix,
        keep_until_secs,
        carried_token_count,
        split.kept_token_count,
    );
    let next_replay_prefix = (!split.bridge.text.is_empty()).then_some(split.bridge.clone());
    let bridge_text = if split.bridge.token_ids.is_empty() {
        None
    } else {
        Some(decode_token_ids(tokenizer, &split.bridge.token_ids)?)
    };
    let kept_text = split.kept_text;
    let decision = WindowRollbackDecision {
        keep_boundary_policy,
        target_keep_until_secs,
        keep_until_secs,
        replay_until_secs: Some(replay_until_secs),
        kept_word_count: WordCount::new(sentence_word_tokens(&kept_text).len()),
        kept_token_count: TokenCount::new(kept_token_ids.len()),
        kept_token_ids,
        kept_text,
        bridge_token_ids: split.bridge.token_ids.clone(),
        bridge_text,
        rollback_position: next_start_position,
        keep_boundary_debug,
    };
    Ok(ResolvedRollback {
        decision,
        next_start_position,
        next_replay_prefix,
    })
}

fn combined_window_transcript(replayed_prefix_text: &str, generated_transcript: &str) -> String {
    let generated_transcript = normalized_transcript(generated_transcript);
    let replayed_prefix_text = normalized_transcript(replayed_prefix_text);
    match Some(replayed_prefix_text).filter(|text| !text.is_empty()) {
        Some(prefix) if !generated_transcript.is_empty() => {
            format!("{prefix} {generated_transcript}")
        }
        Some(prefix) => prefix.to_string(),
        None => generated_transcript.to_string(),
    }
}

fn select_keep_boundary(
    keep_boundary_policy: KeepBoundaryPolicy,
    keep_boundary_debug: &KeepBoundaryDebug,
    candidate_keep_until_secs: WindowTime,
) -> Option<WindowTime> {
    (keep_boundary_policy == KeepBoundaryPolicy::Fixed || keep_boundary_debug.snapped)
        .then_some(candidate_keep_until_secs)
}

fn rollback_position_for_cut(
    chunk_run: &ChunkRun,
    replayed_prefix: Option<&CarriedBridge>,
    keep_until_secs: Option<WindowTime>,
    carried_token_count: TokenCount,
    kept_generated_token_count: TokenCount,
) -> CachePosition {
    let replay_prefix_token_count = replayed_prefix
        .map(|prefix| prefix.token_ids.len())
        .unwrap_or(0);
    let base_prompt_tokens = TokenCount::new(
        chunk_run
            .prompt_tokens
            .as_usize()
            .saturating_sub(replay_prefix_token_count),
    );
    if keep_until_secs.is_some()
        && (!carried_token_count.is_zero() || !kept_generated_token_count.is_zero())
    {
        chunk_run
            .start_position
            .saturating_add(base_prompt_tokens)
            .saturating_add(carried_token_count)
            .saturating_add(kept_generated_token_count)
    } else {
        chunk_run.start_position
    }
}

fn set_draft_from_decoded_chunk(
    draft_token_ids: &mut Vec<TokenId>,
    draft_text: &mut String,
    replayed_prefix: Option<&CarriedBridge>,
    replayed_prefix_text: &str,
    chunk_run: &ChunkRun,
) {
    draft_token_ids.clear();
    if let Some(prefix) = replayed_prefix {
        draft_token_ids.extend_from_slice(&prefix.token_ids);
    }
    draft_token_ids.extend_from_slice(&chunk_run.generated_token_ids);
    draft_text.clear();
    append_display_delta(draft_text, replayed_prefix_text);
    append_display_delta(draft_text, chunk_run.transcript.as_str());
}

fn apply_resolved_rollback_to_display(
    committed_token_ids: &mut Vec<TokenId>,
    committed_text: &mut String,
    draft_token_ids: &mut Vec<TokenId>,
    draft_text: &mut String,
    replayed_prefix_text: &str,
    resolved: &ResolvedRollback,
) {
    committed_token_ids.extend_from_slice(&resolved.decision.kept_token_ids);
    draft_token_ids.clear();
    let committed_delta = if resolved.decision.keep_until_secs.is_some() {
        suffix_after_prefix(
            Some(replayed_prefix_text),
            resolved.decision.kept_text.as_str(),
        )
    } else {
        ""
    };
    append_display_delta(committed_text, committed_delta);
    draft_text.clear();
    if let Some(prefix) = resolved.next_replay_prefix.as_ref() {
        draft_token_ids.extend_from_slice(&prefix.token_ids);
        append_display_delta(draft_text, prefix.text.as_str());
    }
}

fn apply_final_window_to_display(
    committed_token_ids: &mut Vec<TokenId>,
    committed_text: &mut String,
    draft_token_ids: &mut Vec<TokenId>,
    draft_text: &mut String,
    replayed_prefix: Option<&CarriedBridge>,
    replayed_prefix_text: &str,
    chunk_run: &ChunkRun,
) {
    if let Some(prefix) = replayed_prefix {
        committed_token_ids.extend_from_slice(&prefix.token_ids);
        append_display_delta(committed_text, replayed_prefix_text);
    }
    committed_token_ids.extend_from_slice(&chunk_run.generated_token_ids);
    append_display_delta(committed_text, chunk_run.transcript.as_str());
    draft_token_ids.clear();
    draft_text.clear();
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
    start_position: CachePosition,
    start_sample: SampleOffset,
    end_sample: SampleOffset,
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

    let prompt_len = TokenCount::new(prompt_tokens.len());
    let (generated, _confidence, end_position, stop_reason) = prefill_and_decode(
        model,
        &prompt_tokens,
        &audio_features,
        cache,
        start_position.as_usize(),
        max_new_tokens,
        ConfidenceMode::Streaming,
    )
    .with_context(|| format!("decoding {label}"))?;

    let token_ids: Vec<TokenId> = generated
        .iter()
        .map(|&id| {
            u32::try_from(id)
                .map(TokenId::new)
                .context("generated negative token id")
        })
        .collect::<Result<_>>()?;
    let transcript = decode_token_ids(tokenizer, &token_ids)?;

    Ok(ChunkRun {
        label,
        prompt_tokens: prompt_len,
        generated_tokens: TokenCount::new(generated.len()),
        generated_token_ids: token_ids,
        transcript,
        sample_count: SampleCount::new(chunk_samples.len()),
        decode_ms: decode_start.elapsed().as_secs_f64() * 1000.0,
        stop_reason,
        start_position,
        end_position: CachePosition::new(end_position),
        start_sample,
        end_sample,
    })
}

/// Truncates the KV cache to a target rollback position.
pub(crate) fn truncate_cache(
    cache: &mut Option<bee_qwen3_asr::decoder::KVCache>,
    rollback_position: CachePosition,
) -> Result<()> {
    let cache = cache
        .as_mut()
        .ok_or_else(|| anyhow::anyhow!("cache missing before truncate"))?;
    cache.truncate(rollback_position.as_usize());
    Ok(())
}

/// Trims leading and trailing whitespace from a transcript.
pub(crate) fn normalized_transcript(text: &str) -> &str {
    text.trim()
}

/// Decodes token IDs into a transcript string.
pub(crate) fn decode_token_ids(tokenizer: &Tokenizer, token_ids: &[TokenId]) -> Result<String> {
    let token_ids: Vec<u32> = token_ids.iter().map(|id| id.as_u32()).collect();
    tokenizer
        .decode(&token_ids, true)
        .map_err(|e| anyhow::anyhow!("decoding transcript tokens: {e}"))
}

/// Converts unsigned token IDs into signed prompt token IDs.
pub(crate) fn prompt_tokens_from_token_ids(token_ids: &[TokenId]) -> Result<Vec<i32>> {
    token_ids
        .iter()
        .map(|id| i32::try_from(id.as_u32()).context("prompt token id overflow"))
        .collect()
}

/// Tokenizes prompt text into token IDs without adding special tokens.
pub(crate) fn tokenize_token_ids(tokenizer: &Tokenizer, text: &str) -> Result<Vec<TokenId>> {
    Ok(tokenizer
        .encode_fast(text, false)
        .map_err(|e| anyhow::anyhow!("encoding prompt text: {e}"))?
        .get_ids()
        .iter()
        .copied()
        .map(TokenId::new)
        .collect())
}

/// Returns the leading generated token IDs that should be kept.
pub(crate) fn kept_generated_token_ids(
    chunk_run: &ChunkRun,
    kept_token_count: TokenCount,
) -> Vec<TokenId> {
    chunk_run.generated_token_ids[..kept_token_count
        .as_usize()
        .min(chunk_run.generated_token_ids.len())]
        .to_vec()
}

/// Returns the leading carried-bridge token IDs that should be kept.
pub(crate) fn kept_carried_token_ids(
    prefix: Option<&CarriedBridge>,
    kept_token_count: TokenCount,
) -> Vec<TokenId> {
    let Some(prefix) = prefix else {
        return Vec::new();
    };
    prefix.token_ids[..kept_token_count.as_usize().min(prefix.token_ids.len())].to_vec()
}

/// Counts how many carried-bridge tokens should survive a rollback cut.
pub(crate) fn kept_carried_token_count(
    prefix: Option<&CarriedBridge>,
    keep_until_secs: Option<WindowTime>,
    chosen_word: Option<&BoundaryWordDebug>,
) -> TokenCount {
    let Some(prefix) = prefix else {
        return TokenCount::new(0);
    };
    if let Some(chosen_word) = chosen_word {
        if chosen_word.word_index.as_usize() < prefix.words.len() {
            return TokenCount::new(
                prefix.words[chosen_word.word_index.as_usize()]
                    .token_range
                    .end
                    .as_usize(),
            );
        }
    }
    let Some(keep_until_secs) = keep_until_secs else {
        return TokenCount::new(0);
    };
    TokenCount::new(
        prefix
            .words
            .iter()
            .take_while(|word| word.end_secs <= keep_until_secs)
            .last()
            .map(|word| word.token_range.end.as_usize())
            .unwrap_or(0),
    )
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

/// Converts milliseconds to samples at the fixed ASR sample rate.
pub(crate) fn ms_to_samples(chunk_ms: usize) -> Result<SampleCount> {
    let chunk_size_samples = chunk_ms
        .checked_mul(SAMPLE_RATE as usize)
        .ok_or_else(|| anyhow::anyhow!("chunk size overflow; chunk_ms={chunk_ms}"))?
        / 1000;
    if chunk_size_samples == 0 {
        bail!("chunk size is zero; chunk_ms={chunk_ms}");
    }
    Ok(SampleCount::new(chunk_size_samples))
}
