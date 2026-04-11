use std::env;
use std::fs;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use std::{collections::VecDeque, io};

use anyhow::{Context, Result, bail};
use bee_phonetic::sentence_word_tokens;
use bee_qwen3_asr::config::AsrConfig;
use bee_qwen3_asr::generate::{
    ConfidenceMode, DecodeStopReason, build_followup_prompt, build_initial_prompt,
    prefill_and_decode,
};
use bee_qwen3_asr::load;
use bee_qwen3_asr::mel::{MelExtractor, load_audio};
use bee_qwen3_asr::mlx_rs::Array;
use bee_qwen3_asr::mlx_rs::module::{Module, ModuleParametersExt};
use bee_qwen3_asr::mlx_rs::ops;
use bee_qwen3_asr::model::Qwen3ASRModel;
use bee_qwen3_asr::tokenizers::Tokenizer;
use bee_transcribe::g2p::CachedEspeakG2p;
use bee_transcribe::zipa_align::{SpanTiming, TranscriptAlignment};
use bee_zipa_mlx::audio::AudioBuffer as ZipaAudioBuffer;
use bee_zipa_mlx::infer::ZipaInference;
use crossterm::cursor::{Hide, Show};
use crossterm::execute;
use crossterm::terminal::{EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph, Wrap};

const DEFAULT_LANGUAGE: &str = "English";
const DEFAULT_MAX_NEW_TOKENS: usize = 256;
const SAMPLE_RATE: u32 = 16_000;
const N_FFT: usize = 400;
const HOP_LENGTH: usize = 160;
const DEFAULT_WAV_RELATIVE_TO_CRATE: &str = "../../.artifacts/repros/frozen/EB54CF36.wav";
const DEFAULT_START_POSITION_FOR_FRESH_FOLLOWUP: usize = 0;
const DEFAULT_CHUNK_MS: usize = 2_000;
const DEFAULT_ROLLBACK_MS: usize = 1_000;
const DEFAULT_BRIDGE_MS: usize = 1_000;
const DEFAULT_STRIDE_MS: Option<usize> = None;
const DEFAULT_KEEP_BOUNDARY_POLICY: KeepBoundaryPolicy = KeepBoundaryPolicy::Fixed;
const DEFAULT_LANE_B_FIRST_CHUNK_MS: usize = 3_000;
const DEFAULT_REPLAY_CHUNK_INDEX: usize = 1;
const DEFAULT_TRUNCATE_TOKENS: usize = 4;
const DEFAULT_MLX_CACHE_LIMIT_MB: Option<usize> = None;
const KEEP_BOUNDARY_MIN_KEPT_SECS: f64 = 0.200;
const MAX_BRIDGE_WINDOWS: usize = 50;
const ANSI_BLUE: &str = "\x1b[34m";
const ANSI_BOLD: &str = "\x1b[1m";
const ANSI_RESET: &str = "\x1b[0m";
const BOUNDARY_SWEEP_OFFSETS: [isize; 13] =
    [-48, -44, -40, -36, -32, -28, -24, -20, -16, -12, -8, -4, 0];

fn main() -> Result<()> {
    let interrupted = Arc::new(AtomicBool::new(false));
    let interrupted_for_handler = Arc::clone(&interrupted);
    ctrlc::set_handler(move || {
        interrupted_for_handler.store(true, Ordering::SeqCst);
    })
    .context("installing Ctrl-C handler")?;

    let args = Args::parse()?;
    if let Some(limit_mb) = args.mlx_cache_limit_mb {
        let limit_bytes = limit_mb
            .checked_mul(1024 * 1024)
            .ok_or_else(|| anyhow::anyhow!("mlx cache limit too large: {limit_mb}MB"))?;
        let previous_bytes = bee_transcribe::set_mlx_cache_limit(limit_bytes)
            .map_err(|e| anyhow::anyhow!("setting MLX cache limit: {e}"))?;
        eprintln!(
            "mlx cache limit: {}MB (previous {:.1}MB)",
            limit_mb,
            previous_bytes as f64 / (1024.0 * 1024.0)
        );
    }

    let config_path = args.model_dir.join("config.json");
    let config = AsrConfig::from_file(&config_path)
        .with_context(|| format!("loading {}", config_path.display()))?;
    let thinker = config.thinker_config.clone();

    let tokenizer = Tokenizer::from_file(&args.tokenizer_path)
        .map_err(|e| anyhow::anyhow!("loading {}: {e}", args.tokenizer_path.display()))?;

    let mut model = Qwen3ASRModel::new(&thinker).context("constructing qwen3-asr model")?;
    let load_stats = load::load_weights(&mut model, &args.model_dir)
        .with_context(|| format!("loading weights from {}", args.model_dir.display()))?;
    model.eval().context("switching model to eval mode")?;

    let wav_path_str = args.wav_path.to_string_lossy().into_owned();
    let samples = load_audio(&wav_path_str, SAMPLE_RATE)
        .with_context(|| format!("loading {}", args.wav_path.display()))?;

    let mel_extractor = MelExtractor::new(
        N_FFT,
        HOP_LENGTH,
        thinker.audio_config.num_mel_bins,
        SAMPLE_RATE,
    );
    let (mel_data, n_mels, n_frames) = mel_extractor
        .extract(&samples)
        .context("extracting log-mel features")?;
    let mel = Array::from_slice(&mel_data, &[n_mels as i32, n_frames as i32]);

    let audio_features = model
        .encode_audio(&mel)
        .context("encoding audio into ASR features")?;
    let n_audio_tokens = audio_features.shape()[0] as usize;
    let audio_features = ops::expand_dims(&audio_features, 0)?;

    let probe_ids = Array::from_slice(&[0_i32], &[1, 1]);
    let embed_dtype = model.model.embed_tokens.forward(&probe_ids)?.dtype();
    let audio_features = audio_features.as_dtype(embed_dtype)?;

    let summary = RunSummary {
        wav_path: &args.wav_path,
        model_dir: &args.model_dir,
        tokenizer_path: &args.tokenizer_path,
        language: &args.language,
        load_stats: &load_stats,
        sample_count: samples.len(),
        mel_frames: n_frames,
        audio_tokens: n_audio_tokens,
    };

    match args.mode {
        Mode::Initial => {
            let experiment = decode_with_initial_prompt(
                &model,
                &tokenizer,
                &audio_features,
                n_audio_tokens,
                &args.language,
                &args.context,
                args.max_new_tokens,
            )?;
            print_summary(&summary);
            print_experiment(&experiment);
        }
        Mode::FollowupFresh => {
            let experiment = decode_with_followup_prompt(
                &model,
                &tokenizer,
                &audio_features,
                n_audio_tokens,
                &args.language,
                args.max_new_tokens,
            )?;
            print_summary(&summary);
            print_experiment(&experiment);
        }
        Mode::SystemCompare => {
            print_summary(&summary);
            for context in system_compare_contexts() {
                let experiment = decode_with_initial_prompt(
                    &model,
                    &tokenizer,
                    &audio_features,
                    n_audio_tokens,
                    &args.language,
                    context,
                    args.max_new_tokens,
                )?;
                print_experiment(&experiment);
            }
            let experiment = decode_with_followup_prompt(
                &model,
                &tokenizer,
                &audio_features,
                n_audio_tokens,
                &args.language,
                args.max_new_tokens,
            )?;
            print_experiment(&experiment);
        }
        Mode::ChunkedFollowup => {
            let experiment = decode_chunked_followup(
                &model,
                &tokenizer,
                &samples,
                &thinker,
                &args.language,
                &args.context,
                args.max_new_tokens,
                args.chunk_ms,
            )?;
            print_summary(&summary);
            print_chunked_experiment(&experiment);
        }
        Mode::PrefixRerun => {
            let experiment = decode_prefix_rerun(
                &model,
                &tokenizer,
                &samples,
                &thinker,
                &args.language,
                &args.context,
                args.max_new_tokens,
                args.chunk_ms,
            )?;
            print_summary(&summary);
            print_prefix_rerun_experiment(&experiment);
        }
        Mode::TruncateReplay => {
            let experiment = decode_truncate_replay(
                &model,
                &tokenizer,
                &samples,
                &thinker,
                &args.language,
                &args.context,
                args.max_new_tokens,
                args.chunk_ms,
                args.rollback_policy,
                args.replay_chunk_index,
                args.truncate_tokens,
            )?;
            print_summary(&summary);
            print_truncate_replay_experiment(&experiment);
        }
        Mode::SlidingWindowTimedRollback => {
            let experiment = decode_sliding_window_timed_rollback(
                &model,
                &tokenizer,
                &samples,
                &thinker,
                &args.language,
                &args.context,
                args.max_new_tokens,
                args.chunk_ms,
                args.rollback_ms,
                &args.wav_path,
            )?;
            print_summary(&summary);
            print_sliding_window_timed_rollback_experiment(&experiment);
        }
        Mode::SlidingWindowFullReplay => {
            let experiment = decode_sliding_window_full_replay(
                &model,
                &tokenizer,
                &samples,
                &thinker,
                &args.language,
                &args.context,
                args.max_new_tokens,
                args.chunk_ms,
                args.rollback_ms,
                &args.wav_path,
            )?;
            print_summary(&summary);
            print_sliding_window_timed_rollback_experiment(&experiment);
        }
        Mode::SlidingWindowBridgeReplay => {
            let experiment = decode_sliding_window_bridge_replay(
                &model,
                &tokenizer,
                &samples,
                &thinker,
                &args.language,
                &args.context,
                args.max_new_tokens,
                args.chunk_ms,
                args.bridge_ms,
                args.max_bridge_windows,
                args.stride_ms,
                args.keep_boundary_policy,
                args.rollback_ms,
                &args.wav_path,
                &interrupted,
            )?;
            print_summary(&summary);
            print_sliding_window_timed_rollback_experiment(&experiment);
        }
        Mode::DualLaneFollowup => {
            let experiment = decode_dual_lane_followup(
                &model,
                &tokenizer,
                &samples,
                &thinker,
                &args.language,
                &args.context,
                args.max_new_tokens,
                args.chunk_ms,
                args.lane_b_first_chunk_ms,
            )?;
            print_summary(&summary);
            print_dual_lane_followup_experiment(&experiment);
        }
        Mode::ChunkSegmentMergeRollback => {
            let experiment = decode_chunk_segment_merge_rollback(
                &model,
                &tokenizer,
                &samples,
                &thinker,
                &args.language,
                &args.context,
                args.max_new_tokens,
                args.chunk_ms,
            )?;
            print_summary(&summary);
            print_chunk_segment_merge_rollback_experiment(&experiment);
        }
        Mode::ChunkSegmentMergeBoundarySweep => {
            let experiment = decode_chunk_segment_merge_boundary_sweep(
                &model,
                &tokenizer,
                &samples,
                &thinker,
                &args.language,
                &args.context,
                args.max_new_tokens,
                args.chunk_ms,
            )?;
            print_summary(&summary);
            print_chunk_segment_merge_boundary_sweep_experiment(&experiment);
        }
    }

    Ok(())
}

fn decode_with_initial_prompt(
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

fn decode_with_followup_prompt(
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

fn decode_prompt(
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

fn decode_chunked_followup(
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

fn decode_prefix_rerun(
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

fn decode_truncate_replay(
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

fn decode_dual_lane_followup(
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

fn decode_sliding_window_timed_rollback(
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

    let html_path = write_sliding_window_timed_rollback_html(
        "sliding-window-timed-rollback",
        &window_runs,
        samples,
        wav_path,
    )?;
    let committed_timeline_path = Some(write_committed_timeline_html(
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

fn decode_sliding_window_full_replay(
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
        tui.log(format!(
            "decoding chunk {window_index}: audio={}..{}ms samples={} replayed_prefix_words={} start_position={}",
            (window.start_sample * 1000) / SAMPLE_RATE as usize,
            (window.end_sample * 1000) / SAMPLE_RATE as usize,
            chunk_samples.len(),
            replayed_prefix
                .as_ref()
                .map(|prefix| sentence_word_tokens(&prefix.text).len())
                .unwrap_or(0),
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

    let html_path = write_sliding_window_timed_rollback_html(
        "sliding-window-full-replay",
        &window_runs,
        samples,
        wav_path,
    )?;
    let committed_timeline_path = Some(write_committed_timeline_html(
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

fn decode_sliding_window_bridge_replay(
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
        draft_text.clear();
        append_exact(&mut draft_text, replayed_prefix_text.as_str());
        append_exact(&mut draft_text, chunk_run.transcript.as_str());
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
            )?;
            if keep_until_secs.is_some() && !split.kept_text.is_empty() {
                tui.log(format!("kept words: {}", split.kept_text));
            }
            let kept_token_ids = kept_generated_token_ids(&chunk_run, split.kept_token_count);
            let kept_text = decode_token_ids(tokenizer, &kept_token_ids)?;
            let rollback_position = if keep_until_secs.is_some() && split.kept_token_count > 0 {
                chunk_run.start_position + chunk_run.prompt_tokens + split.kept_token_count
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
                kept_word_count: split.kept_word_count,
                kept_token_count: split.kept_token_count,
                kept_token_ids,
                kept_text,
                bridge_token_ids: split.bridge.token_ids.clone(),
                bridge_text,
                rollback_position,
                keep_boundary_debug,
            };
            let kept_exact_text = decode_token_ids(tokenizer, &rollback.kept_token_ids)?;
            let delta = suffix_after_prefix(
                Some(replayed_prefix_text.as_str()),
                kept_exact_text.as_str(),
            );
            append_exact(&mut committed_text, delta);
            draft_text.clear();
            if let Some(prefix) = replay_prefix_for_next.as_ref() {
                append_exact(
                    &mut draft_text,
                    carried_bridge_text(tokenizer, Some(prefix))?.as_str(),
                );
            }
            Some(rollback)
        } else {
            start_position = chunk_run.end_position;
            replay_prefix_for_next = None;
            let delta =
                suffix_after_prefix(Some(replayed_prefix_text.as_str()), draft_text.as_str());
            append_exact(&mut committed_text, delta);
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

    let html_path = write_sliding_window_timed_rollback_html(
        "sliding-window-bridge-replay",
        &window_runs,
        samples,
        wav_path,
    )?;
    let committed_timeline_path = Some(write_committed_timeline_html(
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

fn decode_chunk_segment_merge_rollback(
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
    let html_path = write_chunk_segment_merge_rollback_html(
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

fn decode_chunk_segment_merge_boundary_sweep(
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

fn decode_prefix_window(
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

fn run_chunked_followup_sequence(
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

fn decode_chunk_followup_step(
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

fn truncate_cache(
    cache: &mut Option<bee_qwen3_asr::decoder::KVCache>,
    rollback_position: usize,
) -> Result<()> {
    let cache = cache
        .as_mut()
        .ok_or_else(|| anyhow::anyhow!("cache missing before truncate"))?;
    cache.truncate(rollback_position);
    Ok(())
}

fn normalized_transcript(text: &str) -> &str {
    text.trim()
}

fn decode_token_ids(tokenizer: &Tokenizer, token_ids: &[u32]) -> Result<String> {
    tokenizer
        .decode(token_ids, true)
        .map_err(|e| anyhow::anyhow!("decoding transcript tokens: {e}"))
}

fn prompt_tokens_from_token_ids(token_ids: &[u32]) -> Result<Vec<i32>> {
    token_ids
        .iter()
        .map(|&id| i32::try_from(id).context("prompt token id overflow"))
        .collect()
}

fn tokenize_token_ids(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    Ok(tokenizer
        .encode_fast(text, false)
        .map_err(|e| anyhow::anyhow!("encoding prompt text: {e}"))?
        .get_ids()
        .to_vec())
}

fn kept_generated_token_ids(chunk_run: &ChunkRun, kept_token_count: usize) -> Vec<u32> {
    chunk_run.generated_token_ids[..kept_token_count.min(chunk_run.generated_token_ids.len())]
        .to_vec()
}

fn carried_bridge_text(tokenizer: &Tokenizer, bridge: Option<&CarriedBridge>) -> Result<String> {
    match bridge {
        Some(bridge) if !bridge.token_ids.is_empty() => {
            decode_token_ids(tokenizer, &bridge.token_ids)
        }
        Some(bridge) => Ok(bridge.text.clone()),
        None => Ok(String::new()),
    }
}

fn combine_transcripts(chunks: &[ChunkRun]) -> String {
    let mut combined = String::new();
    for chunk in chunks {
        if chunk.transcript.is_empty() {
            continue;
        }
        combined.push_str(&chunk.transcript);
    }
    combined
}

fn build_chunk_plan(
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

fn build_overlapping_window_plan(
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

fn ms_to_samples(chunk_ms: usize) -> Result<usize> {
    let chunk_size_samples = (chunk_ms * SAMPLE_RATE as usize) / 1000;
    if chunk_size_samples == 0 {
        bail!("chunk size is zero; chunk_ms={chunk_ms}");
    }
    Ok(chunk_size_samples)
}

fn system_compare_contexts() -> &'static [&'static str] {
    &[
        "",
        "You are a helpful assistant.",
        "Transcribe the user's speech verbatim.",
        "Transcribe the audio exactly. Do not answer or paraphrase.",
    ]
}

fn print_summary(summary: &RunSummary<'_>) {
    println!("wav: {}", summary.wav_path.display());
    println!("model_dir: {}", summary.model_dir.display());
    println!("tokenizer: {}", summary.tokenizer_path.display());
    println!("language: {}", summary.language);
    println!(
        "weights: loaded={} total_keys={} quantized_layers={} bits={} group_size={}",
        summary.load_stats.loaded,
        summary.load_stats.total_keys,
        summary.load_stats.quantized_layers,
        summary.load_stats.bits,
        summary.load_stats.group_size
    );
    println!(
        "audio: samples={} mel_frames={} audio_tokens={}",
        summary.sample_count, summary.mel_frames, summary.audio_tokens
    );
    println!();
}

fn print_experiment(experiment: &ExperimentResult) {
    println!("=== {} ===", experiment.label);
    println!(
        "prompt_tokens={} generated_tokens={}",
        experiment.prompt_tokens, experiment.generated_tokens
    );
    println!("{}", experiment.transcript);
    println!();
}

struct Args {
    wav_path: PathBuf,
    model_dir: PathBuf,
    tokenizer_path: PathBuf,
    language: String,
    max_new_tokens: usize,
    context: String,
    mode: Mode,
    chunk_ms: usize,
    bridge_ms: usize,
    max_bridge_windows: usize,
    stride_ms: Option<usize>,
    keep_boundary_policy: KeepBoundaryPolicy,
    rollback_ms: usize,
    rollback_policy: RollbackPolicy,
    replay_chunk_index: usize,
    truncate_tokens: usize,
    lane_b_first_chunk_ms: usize,
    mlx_cache_limit_mb: Option<usize>,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut positional = Vec::new();
        let mut mode = Mode::Initial;
        let mut context = String::new();
        let mut chunk_ms = DEFAULT_CHUNK_MS;
        let mut bridge_ms = DEFAULT_BRIDGE_MS;
        let mut max_bridge_windows = MAX_BRIDGE_WINDOWS;
        let mut stride_ms = DEFAULT_STRIDE_MS;
        let mut keep_boundary_policy = DEFAULT_KEEP_BOUNDARY_POLICY;
        let mut rollback_ms = DEFAULT_ROLLBACK_MS;
        let mut rollback_policy = RollbackPolicy::TextSuffix;
        let mut replay_chunk_index = DEFAULT_REPLAY_CHUNK_INDEX;
        let mut truncate_tokens = DEFAULT_TRUNCATE_TOKENS;
        let mut lane_b_first_chunk_ms = DEFAULT_LANE_B_FIRST_CHUNK_MS;
        let mut mlx_cache_limit_mb = DEFAULT_MLX_CACHE_LIMIT_MB;

        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            if arg == "-h" || arg == "--help" {
                print_usage();
                std::process::exit(0);
            }
            if let Some(value) = arg.strip_prefix("--mode=") {
                mode = Mode::parse(value)?;
                continue;
            }
            if arg == "--mode" {
                let value = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--mode requires a value"))?;
                mode = Mode::parse(&value)?;
                continue;
            }
            if let Some(value) = arg.strip_prefix("--context=") {
                context = value.to_string();
                continue;
            }
            if arg == "--context" {
                context = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--context requires a value"))?;
                continue;
            }
            if let Some(value) = arg.strip_prefix("--chunk-ms=") {
                chunk_ms = value
                    .parse::<usize>()
                    .with_context(|| format!("parsing --chunk-ms from '{value}'"))?;
                continue;
            }
            if arg == "--chunk-ms" {
                let value = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--chunk-ms requires a value"))?;
                chunk_ms = value
                    .parse::<usize>()
                    .with_context(|| format!("parsing --chunk-ms from '{value}'"))?;
                continue;
            }
            if let Some(value) = arg.strip_prefix("--bridge-ms=") {
                bridge_ms = value
                    .parse::<usize>()
                    .with_context(|| format!("parsing --bridge-ms from '{value}'"))?;
                continue;
            }
            if arg == "--bridge-ms" {
                let value = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--bridge-ms requires a value"))?;
                bridge_ms = value
                    .parse::<usize>()
                    .with_context(|| format!("parsing --bridge-ms from '{value}'"))?;
                continue;
            }
            if let Some(value) = arg.strip_prefix("--max-bridge-windows=") {
                max_bridge_windows = value
                    .parse::<usize>()
                    .with_context(|| format!("parsing --max-bridge-windows from '{value}'"))?;
                continue;
            }
            if arg == "--max-bridge-windows" {
                let value = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--max-bridge-windows requires a value"))?;
                max_bridge_windows = value
                    .parse::<usize>()
                    .with_context(|| format!("parsing --max-bridge-windows from '{value}'"))?;
                continue;
            }
            if let Some(value) = arg.strip_prefix("--stride-ms=") {
                stride_ms = Some(
                    value
                        .parse::<usize>()
                        .with_context(|| format!("parsing --stride-ms from '{value}'"))?,
                );
                continue;
            }
            if arg == "--stride-ms" {
                let value = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--stride-ms requires a value"))?;
                stride_ms = Some(
                    value
                        .parse::<usize>()
                        .with_context(|| format!("parsing --stride-ms from '{value}'"))?,
                );
                continue;
            }
            if let Some(value) = arg.strip_prefix("--keep-boundary=") {
                keep_boundary_policy = KeepBoundaryPolicy::parse(value)?;
                continue;
            }
            if arg == "--keep-boundary" {
                let value = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--keep-boundary requires a value"))?;
                keep_boundary_policy = KeepBoundaryPolicy::parse(&value)?;
                continue;
            }
            if let Some(value) = arg.strip_prefix("--rollback-ms=") {
                rollback_ms = value
                    .parse::<usize>()
                    .with_context(|| format!("parsing --rollback-ms from '{value}'"))?;
                continue;
            }
            if arg == "--rollback-ms" {
                let value = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--rollback-ms requires a value"))?;
                rollback_ms = value
                    .parse::<usize>()
                    .with_context(|| format!("parsing --rollback-ms from '{value}'"))?;
                continue;
            }
            if let Some(value) = arg.strip_prefix("--mlx-cache-limit-mb=") {
                mlx_cache_limit_mb = Some(
                    value
                        .parse::<usize>()
                        .with_context(|| format!("parsing --mlx-cache-limit-mb from '{value}'"))?,
                );
                continue;
            }
            if arg == "--mlx-cache-limit-mb" {
                let value = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--mlx-cache-limit-mb requires a value"))?;
                mlx_cache_limit_mb = Some(
                    value
                        .parse::<usize>()
                        .with_context(|| format!("parsing --mlx-cache-limit-mb from '{value}'"))?,
                );
                continue;
            }
            if let Some(value) = arg.strip_prefix("--rollback-policy=") {
                rollback_policy = RollbackPolicy::parse(value)?;
                continue;
            }
            if arg == "--rollback-policy" {
                let value = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--rollback-policy requires a value"))?;
                rollback_policy = RollbackPolicy::parse(&value)?;
                continue;
            }
            if let Some(value) = arg.strip_prefix("--replay-chunk-index=") {
                replay_chunk_index = value
                    .parse::<usize>()
                    .with_context(|| format!("parsing --replay-chunk-index from '{value}'"))?;
                continue;
            }
            if arg == "--replay-chunk-index" {
                let value = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--replay-chunk-index requires a value"))?;
                replay_chunk_index = value
                    .parse::<usize>()
                    .with_context(|| format!("parsing --replay-chunk-index from '{value}'"))?;
                continue;
            }
            if let Some(value) = arg.strip_prefix("--truncate-tokens=") {
                truncate_tokens = value
                    .parse::<usize>()
                    .with_context(|| format!("parsing --truncate-tokens from '{value}'"))?;
                continue;
            }
            if arg == "--truncate-tokens" {
                let value = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--truncate-tokens requires a value"))?;
                truncate_tokens = value
                    .parse::<usize>()
                    .with_context(|| format!("parsing --truncate-tokens from '{value}'"))?;
                continue;
            }
            if let Some(value) = arg.strip_prefix("--lane-b-first-chunk-ms=") {
                lane_b_first_chunk_ms = value
                    .parse::<usize>()
                    .with_context(|| format!("parsing --lane-b-first-chunk-ms from '{value}'"))?;
                continue;
            }
            if arg == "--lane-b-first-chunk-ms" {
                let value = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--lane-b-first-chunk-ms requires a value"))?;
                lane_b_first_chunk_ms = value
                    .parse::<usize>()
                    .with_context(|| format!("parsing --lane-b-first-chunk-ms from '{value}'"))?;
                continue;
            }
            positional.push(arg);
        }

        let wav_path = positional
            .first()
            .map(PathBuf::from)
            .unwrap_or_else(default_wav_path);
        let model_dir = env_path("BEE_ASR_MODEL_DIR")?;
        let tokenizer_path =
            env_path("BEE_TOKENIZER_PATH").unwrap_or_else(|_| model_dir.join("tokenizer.json"));
        let language = positional
            .get(1)
            .cloned()
            .unwrap_or_else(|| DEFAULT_LANGUAGE.to_string());
        let max_new_tokens = positional
            .get(2)
            .map(|s| {
                s.parse::<usize>()
                    .with_context(|| format!("parsing max_new_tokens from '{s}'"))
            })
            .transpose()?
            .unwrap_or(DEFAULT_MAX_NEW_TOKENS);

        if !wav_path.is_file() {
            bail!("wav file not found: {}", wav_path.display());
        }
        if !model_dir.is_dir() {
            bail!("model dir not found: {}", model_dir.display());
        }
        if !tokenizer_path.is_file() {
            bail!("tokenizer not found: {}", tokenizer_path.display());
        }

        Ok(Self {
            wav_path,
            model_dir,
            tokenizer_path,
            language,
            max_new_tokens,
            context,
            mode,
            chunk_ms,
            bridge_ms,
            max_bridge_windows,
            stride_ms,
            keep_boundary_policy,
            rollback_ms,
            rollback_policy,
            replay_chunk_index,
            truncate_tokens,
            lane_b_first_chunk_ms,
            mlx_cache_limit_mb,
        })
    }
}

#[derive(Clone, Copy)]
enum Mode {
    Initial,
    FollowupFresh,
    SystemCompare,
    ChunkedFollowup,
    PrefixRerun,
    TruncateReplay,
    SlidingWindowTimedRollback,
    SlidingWindowFullReplay,
    SlidingWindowBridgeReplay,
    DualLaneFollowup,
    ChunkSegmentMergeRollback,
    ChunkSegmentMergeBoundarySweep,
}

impl Mode {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "initial" => Ok(Self::Initial),
            "followup-fresh" => Ok(Self::FollowupFresh),
            "system-compare" => Ok(Self::SystemCompare),
            "chunked-followup" => Ok(Self::ChunkedFollowup),
            "prefix-rerun" => Ok(Self::PrefixRerun),
            "truncate-replay" => Ok(Self::TruncateReplay),
            "sliding-window-timed-rollback" => Ok(Self::SlidingWindowTimedRollback),
            "sliding-window-full-replay" => Ok(Self::SlidingWindowFullReplay),
            "sliding-window-bridge-replay" => Ok(Self::SlidingWindowBridgeReplay),
            "dual-lane-followup" => Ok(Self::DualLaneFollowup),
            "chunk-segment-merge-rollback" => Ok(Self::ChunkSegmentMergeRollback),
            "chunk-segment-merge-boundary-sweep" => Ok(Self::ChunkSegmentMergeBoundarySweep),
            _ => bail!("unknown mode: {value}"),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RollbackPolicy {
    TextSuffix,
    ChunkSegment,
}

impl RollbackPolicy {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "text-suffix" => Ok(Self::TextSuffix),
            "chunk-segment" => Ok(Self::ChunkSegment),
            _ => bail!("unknown rollback policy: {value}"),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::TextSuffix => "text-suffix",
            Self::ChunkSegment => "chunk-segment",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum KeepBoundaryPolicy {
    Fixed,
    NearestWordEnd,
}

impl KeepBoundaryPolicy {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "fixed" => Ok(Self::Fixed),
            "nearest-word-end" => Ok(Self::NearestWordEnd),
            _ => bail!("unknown keep boundary policy: {value}"),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::NearestWordEnd => "nearest-word-end",
        }
    }
}

struct ExperimentResult {
    label: String,
    prompt_tokens: usize,
    generated_tokens: usize,
    transcript: String,
}

struct RunSummary<'a> {
    wav_path: &'a Path,
    model_dir: &'a Path,
    tokenizer_path: &'a Path,
    language: &'a str,
    load_stats: &'a load::LoadStats,
    sample_count: usize,
    mel_frames: usize,
    audio_tokens: usize,
}

struct ChunkResult {
    label: String,
    prompt_tokens: usize,
    generated_tokens: usize,
    transcript: String,
    sample_count: usize,
}

struct ChunkedExperimentResult {
    chunk_ms: usize,
    chunk_results: Vec<ChunkResult>,
    combined_transcript: String,
}

struct ChunkRun {
    label: String,
    prompt_tokens: usize,
    generated_tokens: usize,
    generated_token_ids: Vec<u32>,
    transcript: String,
    sample_count: usize,
    decode_ms: f64,
    stop_reason: DecodeStopReason,
    start_position: usize,
    end_position: usize,
    start_sample: usize,
    end_sample: usize,
}

impl ChunkResult {
    fn from_run(run: &ChunkRun) -> Self {
        Self {
            label: run.label.clone(),
            prompt_tokens: run.prompt_tokens,
            generated_tokens: run.generated_tokens,
            transcript: run.transcript.clone(),
            sample_count: run.sample_count,
        }
    }
}

struct PrefixResult {
    label: String,
    prompt_tokens: usize,
    generated_tokens: usize,
    transcript: String,
    sample_count: usize,
}

struct PrefixRerunExperimentResult {
    chunk_ms: usize,
    prefix_results: Vec<PrefixResult>,
}

struct ChunkPlanEntry {
    chunk_index: usize,
    start_sample: usize,
    end_sample: usize,
}

#[derive(Clone, Copy)]
struct ReplayPlan {
    rollback_policy: RollbackPolicy,
    rollback_after_chunk_index: usize,
    rollback_position: usize,
    replay_from_chunk_index: usize,
}

struct TruncateReplayExperimentResult {
    chunk_ms: usize,
    rollback_policy: RollbackPolicy,
    replay_chunk_index: usize,
    replay_from_chunk_index: usize,
    rollback_position: usize,
    requested_truncate_tokens: usize,
    applied_truncate_tokens: usize,
    baseline_runs: Vec<ChunkRun>,
    replay_runs: Vec<ChunkRun>,
}

struct WindowRollbackDecision {
    keep_boundary_policy: KeepBoundaryPolicy,
    target_keep_until_secs: f64,
    keep_until_secs: Option<f64>,
    replay_until_secs: Option<f64>,
    kept_word_count: usize,
    kept_token_count: usize,
    kept_token_ids: Vec<u32>,
    kept_text: String,
    bridge_token_ids: Vec<u32>,
    bridge_text: Option<String>,
    rollback_position: usize,
    keep_boundary_debug: KeepBoundaryDebug,
}

#[derive(Clone)]
struct CarriedBridgeWord {
    text: String,
    start_secs: f64,
    end_secs: f64,
}

#[derive(Clone)]
struct CarriedBridge {
    token_ids: Vec<u32>,
    text: String,
    words: Vec<CarriedBridgeWord>,
}

struct SlidingWindowRun {
    chunk_run: ChunkRun,
    rollback: Option<WindowRollbackDecision>,
    replayed_prefix: Option<CarriedBridge>,
}

struct SlidingWindowTimedRollbackExperimentResult {
    mode_label: &'static str,
    chunk_ms: usize,
    rollback_ms: usize,
    stride_ms: usize,
    window_runs: Vec<SlidingWindowRun>,
    html_path: PathBuf,
    committed_timeline_path: Option<PathBuf>,
    interrupted_early: bool,
}

#[derive(Clone)]
struct BoundaryWordDebug {
    text: String,
    start_secs: f64,
    end_secs: f64,
}

#[derive(Clone)]
struct KeepBoundaryDebug {
    earliest_candidate_secs: f64,
    min_keep_secs: f64,
    snapped: bool,
    chosen_word: Option<BoundaryWordDebug>,
}

struct DualLaneFollowupExperimentResult {
    chunk_ms: usize,
    lane_b_first_chunk_ms: usize,
    lane_a_runs: Vec<ChunkRun>,
    lane_b_runs: Vec<ChunkRun>,
}

struct ChunkSegmentMergeRollbackExperimentResult {
    chunk_ms: usize,
    baseline_runs: Vec<ChunkRun>,
    replay_chunk0: ChunkRun,
    merged_chunk: ChunkRun,
    baseline_annotated: String,
    replay_annotated: String,
    html_path: PathBuf,
}

struct BoundarySweepResult {
    offset: isize,
    rollback_position: usize,
    merged_chunk: ChunkRun,
}

struct ChunkSegmentMergeBoundarySweepExperimentResult {
    chunk_ms: usize,
    exact_boundary: usize,
    baseline_runs: Vec<ChunkRun>,
    sweep_results: Vec<BoundarySweepResult>,
}

fn env_path(name: &str) -> Result<PathBuf> {
    let value = env::var(name).with_context(|| format!("{name} is not set"))?;
    Ok(PathBuf::from(value))
}

fn default_wav_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(DEFAULT_WAV_RELATIVE_TO_CRATE)
}

fn print_usage() {
    eprintln!(
        "usage: bee-kv [--mode MODE] [--context TEXT] [--chunk-ms N] [--bridge-ms N] [--rollback-ms N] [wav-path] [language] [max-new-tokens]\n\
         defaults:\n\
           wav-path = {}\n\
           language = {DEFAULT_LANGUAGE}\n\
           max-new-tokens = {DEFAULT_MAX_NEW_TOKENS}\n\
           mode = initial\n\
           chunk-ms = {DEFAULT_CHUNK_MS}\n\
           bridge-ms = {DEFAULT_BRIDGE_MS}\n\
           max-bridge-windows = {MAX_BRIDGE_WINDOWS}\n\
           rollback-ms = {DEFAULT_ROLLBACK_MS}\n\
         modes:\n\
           initial\n\
           followup-fresh\n\
           system-compare\n\
           chunked-followup\n\
           prefix-rerun\n\
           truncate-replay\n\
           sliding-window-timed-rollback\n\
           sliding-window-full-replay\n\
           sliding-window-bridge-replay\n\
           dual-lane-followup\n\
           chunk-segment-merge-rollback\n\
           chunk-segment-merge-boundary-sweep\n\
         rollback policies for truncate-replay:\n\
           text-suffix\n\
           chunk-segment\n\
         environment:\n\
           BEE_ASR_MODEL_DIR\n\
           BEE_TOKENIZER_PATH (optional; defaults to $BEE_ASR_MODEL_DIR/tokenizer.json)",
        default_wav_path().display()
    );
}

fn print_chunked_experiment(experiment: &ChunkedExperimentResult) {
    println!("=== chunked-followup ===");
    println!("chunk_ms={}", experiment.chunk_ms);
    println!();
    for chunk in &experiment.chunk_results {
        println!("--- {} ---", chunk.label);
        println!(
            "samples={} prompt_tokens={} generated_tokens={}",
            chunk.sample_count, chunk.prompt_tokens, chunk.generated_tokens
        );
        println!("{}", chunk.transcript);
        println!();
    }
    println!("=== combined ===");
    println!("{}", experiment.combined_transcript);
    println!();
}

fn print_prefix_rerun_experiment(experiment: &PrefixRerunExperimentResult) {
    println!("=== prefix-rerun ===");
    println!("chunk_ms={}", experiment.chunk_ms);
    println!();
    for prefix in &experiment.prefix_results {
        println!("--- {} ---", prefix.label);
        println!(
            "samples={} prompt_tokens={} generated_tokens={}",
            prefix.sample_count, prefix.prompt_tokens, prefix.generated_tokens
        );
        println!("{}", prefix.transcript);
        println!();
    }
}

fn print_truncate_replay_experiment(experiment: &TruncateReplayExperimentResult) {
    println!("=== truncate-replay ===");
    println!("chunk_ms={}", experiment.chunk_ms);
    println!("rollback_policy={}", experiment.rollback_policy.as_str());
    println!("replay_chunk_index={}", experiment.replay_chunk_index);
    println!(
        "replay_from_chunk_index={}",
        experiment.replay_from_chunk_index
    );
    println!("rollback_position={}", experiment.rollback_position);
    println!(
        "requested_truncate_tokens={} applied_truncate_tokens={}",
        experiment.requested_truncate_tokens, experiment.applied_truncate_tokens
    );
    println!();

    println!("=== baseline ===");
    for chunk in &experiment.baseline_runs {
        print_chunk_run(chunk);
    }
    println!("=== baseline combined ===");
    println!("{}", combine_transcripts(&experiment.baseline_runs));
    println!();

    println!("=== replay path ===");
    for chunk in &experiment.replay_runs {
        print_chunk_run(chunk);
    }
    println!("=== replay combined ===");
    println!("{}", combine_transcripts(&experiment.replay_runs));
    println!();
}

fn print_sliding_window_timed_rollback_experiment(
    experiment: &SlidingWindowTimedRollbackExperimentResult,
) {
    println!("=== {} ===", experiment.mode_label);
    println!("chunk_ms={}", experiment.chunk_ms);
    println!("rollback_ms={}", experiment.rollback_ms);
    println!("stride_ms={}", experiment.stride_ms);
    println!("interrupted_early={}", experiment.interrupted_early);
    println!();

    for run in &experiment.window_runs {
        print_chunk_run(&run.chunk_run);
        if let Some(prefix) = &run.replayed_prefix {
            println!("replayed_prefix_text={}", prefix.text);
        }
        if let Some(rollback) = &run.rollback {
            println!(
                "rollback keep_until={:?} kept_words={} kept_tokens={} bridge_tokens={} rollback_position={}",
                rollback.keep_until_secs,
                rollback.kept_word_count,
                rollback.kept_token_count,
                rollback.bridge_token_ids.len(),
                rollback.rollback_position
            );
            println!("kept_text={}", rollback.kept_text);
            if rollback.keep_until_secs.is_some() && !rollback.kept_text.is_empty() {
                println!(
                    "{ANSI_BLUE}{ANSI_BOLD}==================== KEPT WORDS ===================={ANSI_RESET}"
                );
                println!("{ANSI_BLUE}{ANSI_BOLD}{}{}", rollback.kept_text, ANSI_RESET);
                println!(
                    "{ANSI_BLUE}{ANSI_BOLD}===================================================={ANSI_RESET}"
                );
            }
            if let Some(replay_until_secs) = rollback.replay_until_secs {
                println!("replay_until={:.3}s", replay_until_secs);
            }
            if let Some(bridge_text) = &rollback.bridge_text {
                println!("bridge_text={}", bridge_text);
            }
        } else {
            println!("rollback final-window");
        }
        println!();
    }
    println!("html_path={}", experiment.html_path.display());
    if let Some(committed_timeline_path) = &experiment.committed_timeline_path {
        println!(
            "committed_timeline_path={}",
            committed_timeline_path.display()
        );
    }
    println!();
}

fn print_dual_lane_followup_experiment(experiment: &DualLaneFollowupExperimentResult) {
    println!("=== dual-lane-followup ===");
    println!("lane_a_chunk_ms={}", experiment.chunk_ms);
    println!("lane_b_first_chunk_ms={}", experiment.lane_b_first_chunk_ms);
    println!();

    println!("=== lane-a row ===");
    print_lane_row("A", &experiment.lane_a_runs);
    println!("=== lane-b row ===");
    print_lane_row("B", &experiment.lane_b_runs);
    println!();

    println!("=== lane-a chunks ===");
    for chunk in &experiment.lane_a_runs {
        print_chunk_run(chunk);
    }
    println!("=== lane-b chunks ===");
    for chunk in &experiment.lane_b_runs {
        print_chunk_run(chunk);
    }
}

fn print_chunk_segment_merge_rollback_experiment(
    experiment: &ChunkSegmentMergeRollbackExperimentResult,
) {
    println!("=== chunk-segment-merge-rollback ===");
    println!("chunk_ms={}", experiment.chunk_ms);
    println!();

    println!("=== baseline first three ===");
    for chunk in &experiment.baseline_runs {
        print_chunk_run(chunk);
    }
    println!(
        "baseline_bracketed={}",
        bracketed_chunks(&experiment.baseline_runs)
    );
    println!("baseline_annotated={}", experiment.baseline_annotated);
    println!();

    println!("=== replay after rollback ===");
    print_chunk_run(&experiment.replay_chunk0);
    print_chunk_run(&experiment.merged_chunk);
    println!(
        "replay_bracketed={}",
        bracketed_chunks(&[
            clone_chunk_run(&experiment.replay_chunk0),
            clone_chunk_run(&experiment.merged_chunk),
        ])
    );
    println!("replay_annotated={}", experiment.replay_annotated);
    println!("html_path={}", experiment.html_path.display());
    println!();
}

fn print_chunk_segment_merge_boundary_sweep_experiment(
    experiment: &ChunkSegmentMergeBoundarySweepExperimentResult,
) {
    println!("=== chunk-segment-merge-boundary-sweep ===");
    println!("chunk_ms={}", experiment.chunk_ms);
    println!("exact_boundary={}", experiment.exact_boundary);
    println!(
        "baseline_bracketed={}",
        bracketed_chunks(&experiment.baseline_runs)
    );
    println!();
    for result in &experiment.sweep_results {
        println!(
            "offset={:+} rollback_position={} replay_bracketed=[{}] [{}]",
            result.offset,
            result.rollback_position,
            experiment.baseline_runs[0].transcript,
            result.merged_chunk.transcript
        );
    }
    println!();
}

fn clone_chunk_run(chunk: &ChunkRun) -> ChunkRun {
    ChunkRun {
        label: chunk.label.clone(),
        prompt_tokens: chunk.prompt_tokens,
        generated_tokens: chunk.generated_tokens,
        generated_token_ids: chunk.generated_token_ids.clone(),
        transcript: chunk.transcript.clone(),
        sample_count: chunk.sample_count,
        decode_ms: chunk.decode_ms,
        stop_reason: chunk.stop_reason,
        start_position: chunk.start_position,
        end_position: chunk.end_position,
        start_sample: chunk.start_sample,
        end_sample: chunk.end_sample,
    }
}

fn bracketed_chunks(chunks: &[ChunkRun]) -> String {
    chunks
        .iter()
        .map(|chunk| format!("[{}]", chunk.transcript))
        .collect::<Vec<_>>()
        .join(" ")
}

fn annotate_chunk_runs(chunks: &[ChunkRun], samples: &[f32]) -> Result<String> {
    let combined_transcript = combine_transcripts(chunks);
    let mut align_ctx = AlignmentContext::new()?;
    let alignment = build_transcript_alignment(&mut align_ctx, &combined_transcript, samples)?;
    let mut word_start = 0usize;
    let mut annotated = Vec::with_capacity(chunks.len());

    for chunk in chunks {
        let word_count = sentence_word_tokens(&chunk.transcript).len();
        let word_end = word_start + word_count;
        annotated.push(format!(
            "[{}]{}",
            chunk.transcript,
            format_span_timing(alignment.span_timing(word_start, word_end))
        ));
        word_start = word_end;
    }

    Ok(annotated.join(" "))
}

#[derive(Clone)]
struct WordPlacement {
    text: String,
    chunk_index: usize,
    start_secs: Option<f64>,
    end_secs: Option<f64>,
    quality_label: &'static str,
}

struct SlidingWordPlacement {
    text: String,
    start_secs: Option<f64>,
    end_secs: Option<f64>,
    quality_label: &'static str,
    carried: bool,
    kept: bool,
    bridge: bool,
    cut_word: bool,
}

struct CommittedWordPlacement {
    text: String,
    start_secs: Option<f64>,
    end_secs: Option<f64>,
    quality_label: &'static str,
    second_bin: usize,
}

fn committed_words_equivalent(
    left: &CommittedWordPlacement,
    right: &CommittedWordPlacement,
) -> bool {
    if left.text != right.text {
        return false;
    }
    match (
        left.start_secs,
        left.end_secs,
        right.start_secs,
        right.end_secs,
    ) {
        (Some(left_start), Some(left_end), Some(right_start), Some(right_end)) => {
            (left_start - right_start).abs() <= 0.12 && (left_end - right_end).abs() <= 0.12
        }
        _ => true,
    }
}

fn build_word_placements(
    align_ctx: &mut AlignmentContext,
    chunks: &[ChunkRun],
    samples: &[f32],
) -> Result<Vec<WordPlacement>> {
    let combined_transcript = combine_transcripts(chunks);
    let alignment = build_transcript_alignment(align_ctx, &combined_transcript, samples)?;
    let word_timings = alignment.word_timings();
    let mut next_word = 0usize;
    let mut placements = Vec::new();

    for (chunk_index, chunk) in chunks.iter().enumerate() {
        let chunk_words = sentence_word_tokens(&chunk.transcript);
        for _ in chunk_words {
            let word_timing = word_timings
                .get(next_word)
                .ok_or_else(|| anyhow::anyhow!("missing word timing at index {next_word}"))?;
            let (start_secs, end_secs, quality_label) = match &word_timing.quality {
                bee_transcribe::zipa_align::AlignmentQuality::Aligned {
                    start_secs,
                    end_secs,
                } => (Some(*start_secs), Some(*end_secs), "aligned"),
                bee_transcribe::zipa_align::AlignmentQuality::NoWindow => (None, None, "no-window"),
                bee_transcribe::zipa_align::AlignmentQuality::NoTiming => (None, None, "no-timing"),
            };
            placements.push(WordPlacement {
                text: word_timing.word.to_string(),
                chunk_index,
                start_secs,
                end_secs,
                quality_label,
            });
            next_word += 1;
        }
    }

    Ok(placements)
}

fn write_sliding_window_timed_rollback_html(
    mode_label: &str,
    window_runs: &[SlidingWindowRun],
    samples: &[f32],
    wav_path: &Path,
) -> Result<PathBuf> {
    let duration_secs = samples.len() as f64 / SAMPLE_RATE as f64;
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.artifacts/bee-kv");
    fs::create_dir_all(&out_dir).with_context(|| format!("creating {}", out_dir.display()))?;
    let out_path = out_dir.join(format!("{mode_label}.html"));
    let audio_src = file_url_for_path(wav_path)?;
    let mut align_ctx = AlignmentContext::new()?;
    let html = render_sliding_window_timed_rollback_html(
        &mut align_ctx,
        mode_label,
        window_runs,
        samples,
        duration_secs,
        &audio_src,
    )?;
    fs::write(&out_path, html).with_context(|| format!("writing {}", out_path.display()))?;
    Ok(out_path)
}

fn write_committed_timeline_html(
    mode_label: &str,
    window_runs: &[SlidingWindowRun],
    samples: &[f32],
    wav_path: &Path,
) -> Result<PathBuf> {
    let duration_secs = samples.len() as f64 / SAMPLE_RATE as f64;
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.artifacts/bee-kv");
    fs::create_dir_all(&out_dir).with_context(|| format!("creating {}", out_dir.display()))?;
    let out_path = out_dir.join(format!("{mode_label}.html"));
    let audio_src = file_url_for_path(wav_path)?;
    let mut align_ctx = AlignmentContext::new()?;
    let words = collect_committed_word_placements(&mut align_ctx, window_runs, samples)?;
    let html = render_committed_timeline_html(mode_label, duration_secs, &words, &audio_src);
    fs::write(&out_path, html).with_context(|| format!("writing {}", out_path.display()))?;
    Ok(out_path)
}

fn render_sliding_window_timed_rollback_html(
    align_ctx: &mut AlignmentContext,
    mode_label: &str,
    window_runs: &[SlidingWindowRun],
    samples: &[f32],
    total_duration_secs: f64,
    audio_src: &str,
) -> Result<String> {
    let width_px = 1100.0;
    let row_height_px = 110.0;
    let mut rows = String::new();

    for (row_index, run) in window_runs.iter().enumerate() {
        let chunk = &run.chunk_run;
        let chunk_samples = &samples[chunk.start_sample..chunk.end_sample];
        let words = build_window_word_placements(align_ctx, run, chunk_samples)?;
        rows.push_str(&render_sliding_window_row(
            width_px,
            row_height_px,
            row_index,
            chunk,
            run.rollback.as_ref(),
            run.replayed_prefix.as_ref(),
            &words,
            total_duration_secs,
            audio_src,
        ));
    }

    Ok(format!(
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>bee-kv {mode_label}</title><style>\
body{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#111315;color:#ece7dc;padding:24px;color-scheme:dark;}}\
.legend{{margin-bottom:18px;font-size:13px;color:#b9b3a7;}}\
.audio-panel{{margin:0 0 20px 0;padding:12px 14px;border:1px solid #4b4f56;background:#181c20;width:{width_px}px;box-shadow:0 10px 30px rgba(0,0,0,0.22);}}\
.audio-title{{font-weight:700;margin:0 0 8px 0;}}\
.audio-player{{width:100%;margin:6px 0 0 0;color-scheme:dark;accent-color:#d97706;}}\
.row{{margin-bottom:28px;}}\
.row-title{{font-weight:700;margin:0 0 6px 0;}}\
.row-meta{{margin:0 0 8px 0;font-size:13px;color:#b0aa9d;}}\
.row-audio{{margin:0 0 8px 0;display:flex;align-items:center;gap:10px;}}\
.row-audio audio{{width:420px;max-width:100%;color-scheme:dark;accent-color:#d97706;}}\
.transcript-line{{width:{width_px}px;margin:0 0 8px 0;font-size:13px;line-height:1.5;color:#ded7ca;}}\
.timeline{{width:{width_px}px;border:1px solid #4b4f56;background:#181c20;position:relative;padding:12px 0;margin-bottom:8px;box-shadow:0 10px 30px rgba(0,0,0,0.22);}}\
.track{{position:relative;width:{width_px}px;height:{row_height_px}px;border-top:1px solid #343940;border-bottom:1px solid #343940;background:linear-gradient(180deg,#1b2024,#14181c);overflow:hidden;}}\
.word{{text-align:center;vertical-align:middle;position:absolute;height:2em;padding:4px 2px;border-radius:4px;border:1px solid #7a6d56;white-space:nowrap;overflow:visible;font-size:12px;box-sizing:border-box;cursor:default;font-family:\"SF Pro\", serif;box-shadow:0 3px 10px rgba(0,0,0,0.28);}}\
.word.carried{{background:#544775;border-color:#aa9cf0;color:#f0ebff;}}\
.word.kept{{background:#183b2a;border-color:#62d596;color:#dffff0;font-weight:700;box-shadow:0 0 0 2px rgba(98,213,150,0.24),0 3px 10px rgba(0,0,0,0.28);}}\
.word.bridge{{background:#4b3c12;border-color:#e4bd4f;color:#fff1bf;}}\
.word.rolled{{background:#4d2528;border-color:#db7f88;color:#ffe0e3;}}\
.word.cut-word{{outline:3px solid rgba(117,173,255,0.6);outline-offset:1px;border-style:dashed;}}\
.word.no-window,.word.no-timing{{background:#3b352c;border-color:#908672;color:#f1e8d5;height:22px;font-size:11px;}}\
.search-range{{position:absolute;height:12px;border-bottom:2px solid #7e8793;border-left:2px solid #7e8793;border-right:2px solid #7e8793;border-bottom-left-radius:8px;border-bottom-right-radius:8px;background:rgba(126,135,147,0.10);pointer-events:none;}}\
.search-range-label{{position:absolute;transform:translateX(-50%);font-size:11px;color:#9ba4af;font-weight:700;pointer-events:none;}}\
.cut{{position:absolute;top:0;width:2px;height:{row_height_px}px;background:#5ba2ff;}}\
.cut-label{{position:absolute;transform:translateX(6px);font-size:11px;color:#7fb6ff;font-weight:700;white-space:nowrap;}}\
.cut-label.cut-label-cut{{top:2px;}}\
.cut-label.cut-label-target{{top:18px;}}\
.cut-label.cut-label-bridge{{top:2px;}}\
.window-start,.window-end{{position:absolute;top:0;width:1px;height:{row_height_px}px;background:#5b616b;}}\
.window-label{{position:absolute;bottom:2px;transform:translateX(4px);font-size:11px;color:#9299a4;}}\
.playhead{{position:absolute;top:0;width:2px;height:{row_height_px}px;background:#d97706;pointer-events:none;display:none;}}\
.axis{{display:flex;justify-content:space-between;font-size:12px;color:#9299a4;margin-top:6px;}}\
.word:hover::after{{content:attr(data-full-word);position:absolute;left:0;top:-28px;background:#f6f1e5;color:#111315;padding:2px 6px;border-radius:4px;white-space:nowrap;z-index:10;font-size:11px;line-height:16px;box-shadow:0 2px 6px rgba(0,0,0,0.35);}}\
</style></head><body><h1>bee-kv {mode_label}</h1><div class=\"audio-panel\"><div class=\"audio-title\">Full Recording</div><div>Source: {audio_src}</div><audio id=\"master-audio\" class=\"audio-player\" controls preload=\"metadata\" src=\"{audio_src}\"></audio></div><p class=\"legend\">Each row is one decode window. Purple words are prompt text replayed into that row. Green words are KV-kept generated words. Yellow words are bridge-region generated words. Red words are re-decoded tail words. Blue line marks the keep cut. Brown line marks the bridge cut. Orange line is the current playhead.</p>{rows}<script>\
const masterAudio = document.getElementById('master-audio');\
const chunkAudios = Array.from(document.querySelectorAll('audio[data-window-start]'));\
const allAudios = [masterAudio, ...chunkAudios].filter(Boolean);\
let rafId = null;\
let activeAudio = null;\
function pauseOthers(active){{ for (const audio of allAudios) {{ if (audio !== active) audio.pause(); }} }}\
function updatePlayheads(time){{\
  document.querySelectorAll('[data-row-start]').forEach((row) => {{\
    const start = Number(row.dataset.rowStart);\
    const end = Number(row.dataset.rowEnd);\
    const duration = end - start;\
    const playhead = row.querySelector('.playhead');\
    if (!playhead) return;\
    if (time < start || time > end || duration <= 0) {{ playhead.style.display = 'none'; return; }}\
    const frac = (time - start) / duration;\
    playhead.style.display = 'block';\
    playhead.style.left = `${{Math.max(0, Math.min(1, frac)) * 100}}%`;\
  }});\
}}\
function hidePlayheads(){{\
  document.querySelectorAll('[data-row-start] .playhead').forEach((playhead) => {{\
    playhead.style.display = 'none';\
  }});\
}}\
function syncPlayheads(audio){{ updatePlayheads(audio?.currentTime || 0); }}\
function stopTracking(){{\
  if (rafId !== null) cancelAnimationFrame(rafId);\
  rafId = null;\
  activeAudio = null;\
  hidePlayheads();\
}}\
function tick(){{\
  if (!activeAudio || activeAudio.paused || activeAudio.ended) {{ stopTracking(); return; }}\
  if (chunkAudios.includes(activeAudio)) {{\
    const end = Number(activeAudio.dataset.windowEnd);\
    if ((activeAudio.currentTime || 0) >= end) {{\
      activeAudio.currentTime = end;\
      activeAudio.pause();\
      stopTracking();\
      return;\
    }}\
  }}\
  syncPlayheads(activeAudio);\
  rafId = requestAnimationFrame(tick);\
}}\
function startTracking(audio){{\
  pauseOthers(audio);\
  activeAudio = audio;\
  if (rafId !== null) cancelAnimationFrame(rafId);\
  syncPlayheads(audio);\
  rafId = requestAnimationFrame(tick);\
}}\
masterAudio.addEventListener('play', () => startTracking(masterAudio));\
masterAudio.addEventListener('pause', () => {{ if (activeAudio === masterAudio) stopTracking(); }});\
masterAudio.addEventListener('ended', () => {{ if (activeAudio === masterAudio) stopTracking(); }});\
masterAudio.addEventListener('seeking', () => syncPlayheads(masterAudio));\
chunkAudios.forEach((audio) => {{\
  const start = Number(audio.dataset.windowStart);\
  const end = Number(audio.dataset.windowEnd);\
  audio.addEventListener('play', () => {{\
    if (audio.currentTime < start || audio.currentTime >= end) audio.currentTime = start;\
    startTracking(audio);\
  }});\
  audio.addEventListener('seeking', () => {{\
    if (audio.currentTime < start) audio.currentTime = start;\
    if (audio.currentTime > end) audio.currentTime = end;\
    if (activeAudio === audio) syncPlayheads(audio);\
  }});\
  audio.addEventListener('pause', () => {{ if (activeAudio === audio) stopTracking(); }});\
  audio.addEventListener('ended', () => {{ if (activeAudio === audio) stopTracking(); }});\
}});\
updatePlayheads(0);\
</script></body></html>"
    ))
}

fn collect_committed_word_placements(
    align_ctx: &mut AlignmentContext,
    window_runs: &[SlidingWindowRun],
    samples: &[f32],
) -> Result<Vec<CommittedWordPlacement>> {
    let mut placements: Vec<CommittedWordPlacement> = Vec::new();

    for run in window_runs {
        let chunk = &run.chunk_run;
        let chunk_samples = &samples[chunk.start_sample..chunk.end_sample];
        let words = build_window_word_placements(align_ctx, run, chunk_samples)?;
        let window_start_secs = chunk.start_sample as f64 / SAMPLE_RATE as f64;
        let keep_word_count = if let Some(rollback) = &run.rollback {
            sentence_word_tokens(&rollback.kept_text).len()
        } else {
            words.len()
        };
        if keep_word_count == 0 {
            continue;
        }
        let mut candidate = Vec::new();
        for word in words.into_iter().take(keep_word_count) {
            let start_secs = word.start_secs.map(|start| window_start_secs + start);
            let end_secs = word.end_secs.map(|end| window_start_secs + end);
            let second_bin = start_secs
                .map(|start| start.floor() as usize)
                .unwrap_or_else(|| window_start_secs.floor() as usize);
            candidate.push(CommittedWordPlacement {
                text: word.text,
                start_secs,
                end_secs,
                quality_label: word.quality_label,
                second_bin,
            });
        }
        let max_overlap = placements.len().min(candidate.len());
        let overlap = (0..=max_overlap)
            .rev()
            .find(|&count| {
                placements[placements.len().saturating_sub(count)..]
                    .iter()
                    .zip(candidate.iter().take(count))
                    .all(|(left, right)| committed_words_equivalent(left, right))
            })
            .unwrap_or(0);
        placements.extend(candidate.into_iter().skip(overlap));
    }

    Ok(placements)
}

fn render_committed_timeline_html(
    mode_label: &str,
    duration_secs: f64,
    words: &[CommittedWordPlacement],
    audio_src: &str,
) -> String {
    let px_per_sec = 100.0_f64;
    let width_px = (duration_secs * px_per_sec).ceil();
    let row_height_px = 132.0;
    let transcript_line = words
        .iter()
        .map(|word| html_escape(&word.text))
        .collect::<Vec<_>>()
        .join(" ");
    let total_seconds = duration_secs.ceil() as usize;
    let mut second_bands = String::new();
    let mut second_markers = String::new();
    for second in 0..=total_seconds {
        let x = ((second as f64) / duration_secs.min(duration_secs.max(1.0))) * width_px;
        if second < total_seconds {
            let left = (second as f64 / duration_secs) * width_px;
            let right = (((second + 1) as f64).min(duration_secs) / duration_secs) * width_px;
            let width = (right - left).max(0.0);
            let hue = (second * 47) % 360;
            second_bands.push_str(&format!(
                "<div class=\"second-band\" style=\"left:{left:.1}px;width:{width:.1}px;background:hsl({hue} 55% 90% / 0.55)\"></div>"
            ));
        }
        if second as f64 <= duration_secs {
            second_markers.push_str(&format!(
                "<div class=\"second-marker\" style=\"left:{x:.1}px\"></div><div class=\"second-label\" style=\"left:{x:.1}px\">{second}s</div>"
            ));
        }
    }

    let mut word_divs = String::new();
    let mut fallback_x = 0.0;
    let mut lane_end_x = [0.0_f64; 3];
    let lane_tops = [20.0_f64, 54.0_f64, 88.0_f64];
    for word in words {
        let hue = (word.second_bin * 47) % 360;
        let class = format!("word {}", word.quality_label);
        let (left, width, lane_index) = match (word.start_secs, word.end_secs) {
            (Some(start), Some(end)) => {
                let left = (start / duration_secs) * width_px;
                let width = ((end - start).max(0.08) / duration_secs) * width_px;
                let mut lane_index = 0usize;
                while lane_index + 1 < lane_end_x.len() && lane_end_x[lane_index] > left {
                    lane_index += 1;
                }
                if lane_end_x[lane_index] > left && lane_index == lane_end_x.len() - 1 {
                    lane_index = lane_end_x
                        .iter()
                        .enumerate()
                        .min_by(|a, b| a.1.total_cmp(b.1))
                        .map(|(idx, _)| idx)
                        .unwrap_or(0);
                }
                lane_end_x[lane_index] = left + width + 6.0;
                (left, width, lane_index)
            }
            _ => {
                let left = fallback_x;
                fallback_x += 90.0;
                (left, 84.0, 2)
            }
        };
        let top = lane_tops[lane_index.min(lane_tops.len() - 1)];
        let text = html_escape(&word.text);
        word_divs.push_str(&format!(
            "<div class=\"{class}\" style=\"left:{left:.1}px;top:{top:.1}px;width:{width:.1}px;background:hsl({hue} 58% 82%);border-color:hsl({hue} 38% 42%);\" title=\"committed @ {start:.2}-{end:.2}s\" data-full-word=\"{text}\">{text}</div>",
            start = word.start_secs.unwrap_or(0.0),
            end = word.end_secs.unwrap_or(0.0),
        ));
    }

    let word_timings_js = words
        .iter()
        .filter_map(|w| {
            let s = w.start_secs?;
            let e = w.end_secs?;
            let t = w.text.replace('\\', "\\\\").replace('"', "\\\"");
            Some(format!("{{t:\"{t}\",s:{s:.3},e:{e:.3}}}"))
        })
        .collect::<Vec<_>>()
        .join(",");

    format!(
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>bee-kv {mode_label}</title><style>\
body{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#111315;color:#ece7dc;padding:24px;color-scheme:dark;}}\
.legend{{margin-bottom:18px;font-size:13px;color:#b9b3a7;}}\
.audio-panel{{margin:0 0 20px 0;padding:12px 14px;border:1px solid #4b4f56;background:#181c20;max-width:900px;box-shadow:0 10px 30px rgba(0,0,0,0.22);}}\
.audio-title{{font-weight:700;margin:0 0 8px 0;}}\
.audio-player{{width:100%;margin:6px 0 0 0;color-scheme:dark;accent-color:#d97706;}}\
#current-word-display{{font-family:\"SF Pro Display\",\"SF Pro\",ui-sans-serif,-apple-system,sans-serif;font-size:80px;font-weight:700;height:110px;display:flex;align-items:center;justify-content:center;color:#f6f1e5;margin:20px 0;letter-spacing:-0.01em;flex-shrink:0;}}\
.timeline-scroll{{overflow-x:auto;margin-bottom:8px;}}\
.transcript-line{{width:{width_px}px;margin:0 0 8px 0;font-size:13px;line-height:1.5;color:#ded7ca;}}\
.timeline{{width:{width_px}px;border:1px solid #4b4f56;background:#181c20;position:relative;padding:12px 0;box-shadow:0 10px 30px rgba(0,0,0,0.22);}}\
.track{{position:relative;width:{width_px}px;height:{row_height_px}px;border-top:1px solid #343940;border-bottom:1px solid #343940;background:linear-gradient(180deg,#1b2024,#14181c);overflow:hidden;}}\
.second-band{{position:absolute;top:0;height:{row_height_px}px;pointer-events:none;mix-blend-mode:screen;}}\
.second-marker{{position:absolute;top:0;width:1px;height:{row_height_px}px;background:#5b616b;}}\
.second-label{{position:absolute;top:2px;transform:translateX(4px);font-size:11px;color:#9299a4;}}\
.playhead{{position:absolute;top:0;width:2px;height:{row_height_px}px;background:#d97706;pointer-events:none;display:none;}}\
.word{{text-align:center;vertical-align:middle;position:absolute;height:2em;padding:4px 2px;border-radius:4px;border:1px solid #7a6d56;white-space:nowrap;overflow:visible;font-size:12px;box-sizing:border-box;cursor:default;font-family:\"SF Pro\", serif;color:#101214;box-shadow:0 3px 10px rgba(0,0,0,0.28);}}\
.word.no-window,.word.no-timing{{background:#3b352c !important;border-color:#908672 !important;color:#f1e8d5 !important;height:22px;font-size:11px;}}\
.axis{{display:flex;justify-content:space-between;font-size:12px;color:#9299a4;margin-top:6px;}}\
.word:hover::after{{content:attr(data-full-word);position:absolute;left:0;top:-28px;background:#f6f1e5;color:#111315;padding:2px 6px;border-radius:4px;white-space:nowrap;z-index:10;font-size:11px;line-height:16px;box-shadow:0 2px 6px rgba(0,0,0,0.35);}}\
</style></head><body>\
<h1>bee-kv {mode_label}</h1>\
<div class=\"audio-panel\"><div class=\"audio-title\">Full Recording</div><div>Source: {audio_src}</div><audio id=\"master-audio\" class=\"audio-player\" controls preload=\"metadata\" src=\"{audio_src}\"></audio></div>\
<p class=\"legend\">Committed words only. Each word keeps the exact timing it had when it was marked green in its source row. Background bands and box colors change every one-second interval; no extra re-alignment is performed.</p>\
<div id=\"current-word-display\"></div>\
<div class=\"timeline-scroll\">\
<div class=\"transcript-line\">{transcript_line}</div>\
<div class=\"timeline\" data-row-start=\"0\" data-row-end=\"{duration_secs:.6}\">\
<div class=\"track\">{second_bands}{second_markers}<div class=\"playhead\"></div>{word_divs}</div>\
<div class=\"axis\"><span>0.00s</span><span>{duration_secs:.2}s total</span></div>\
</div></div>\
<script>\
const wordTimings=[{word_timings_js}];\
const trackWidth={width_px};\
const audio=document.getElementById('master-audio');\
const row=document.querySelector('[data-row-start]');\
const playhead=row?.querySelector('.playhead');\
const currentWordEl=document.getElementById('current-word-display');\
const scrollEl=document.querySelector('.timeline-scroll');\
let rafId=null;\
function findWord(t){{for(let i=0;i<wordTimings.length;i++){{const w=wordTimings[i];if(t>=w.s&&t<=w.e)return w.t;}}return '';}}\
function updatePlayhead(time){{\
  if(!row||!playhead)return;\
  const start=Number(row.dataset.rowStart);\
  const end=Number(row.dataset.rowEnd);\
  const dur=end-start;\
  if(time<start||time>end||dur<=0){{playhead.style.display='none';return;}}\
  const frac=(time-start)/dur;\
  const px=Math.max(0,Math.min(1,frac))*trackWidth;\
  playhead.style.display='block';\
  playhead.style.left=`${{px}}px`;\
  if(scrollEl){{const cw=scrollEl.offsetWidth;scrollEl.scrollLeft=Math.max(0,px-cw/2);}}\
  if(currentWordEl)currentWordEl.textContent=findWord(time);\
}}\
function hidePlayhead(){{if(playhead)playhead.style.display='none';}}\
function syncPlayhead(){{updatePlayhead(audio.currentTime||0);}}\
function stopTracking(){{if(rafId!==null)cancelAnimationFrame(rafId);rafId=null;hidePlayhead();if(currentWordEl)currentWordEl.textContent='';}}\
function tick(){{if(audio.paused||audio.ended){{stopTracking();return;}}syncPlayhead();rafId=requestAnimationFrame(tick);}}\
audio.addEventListener('play',()=>{{if(rafId!==null)cancelAnimationFrame(rafId);syncPlayhead();rafId=requestAnimationFrame(tick);}});\
audio.addEventListener('pause',stopTracking);\
audio.addEventListener('ended',stopTracking);\
audio.addEventListener('seeking',()=>updatePlayhead(audio.currentTime||0));\
updatePlayhead(0);\
</script></body></html>",
    )
}

fn render_sliding_window_row(
    width_px: f64,
    row_height_px: f64,
    row_index: usize,
    chunk: &ChunkRun,
    rollback: Option<&WindowRollbackDecision>,
    replayed_prefix: Option<&CarriedBridge>,
    words: &[SlidingWordPlacement],
    total_duration_secs: f64,
    audio_src: &str,
) -> String {
    let transcript_line = html_escape(&chunk.transcript);
    let mut markers = String::new();
    let window_start_secs = chunk.start_sample as f64 / SAMPLE_RATE as f64;
    let window_end_secs = chunk.end_sample as f64 / SAMPLE_RATE as f64;
    let window_duration_secs = (chunk.end_sample - chunk.start_sample) as f64 / SAMPLE_RATE as f64;

    markers.push_str(&format!(
        "<div class=\"window-start\" style=\"left:0px\"></div><div class=\"window-end\" style=\"left:{:.1}px\"></div><div class=\"window-label\" style=\"left:0px\">{:.2}s</div><div class=\"window-label\" style=\"left:{:.1}px\">{:.2}s</div>",
        width_px,
        window_start_secs,
        width_px,
        window_end_secs
    ));
    markers.push_str("<div class=\"playhead\"></div>");
    if let Some(rollback) = rollback {
        let search_start_x = (rollback.keep_boundary_debug.earliest_candidate_secs
            / window_duration_secs)
            * width_px;
        let search_end_x = (rollback.target_keep_until_secs / window_duration_secs) * width_px;
        let search_width = (search_end_x - search_start_x).max(0.0);
        let search_label_x = search_start_x + (search_width / 2.0);
        markers.push_str(&format!(
            "<div class=\"search-range\" style=\"left:{search_start_x:.1}px;top:{:.1}px;width:{search_width:.1}px\"></div><div class=\"search-range-label\" style=\"left:{search_label_x:.1}px;top:{:.1}px\">search</div>",
            row_height_px - 14.0,
            row_height_px - 28.0,
        ));
        if let Some(keep_until_secs) = rollback.keep_until_secs {
            let cut_x = (keep_until_secs / window_duration_secs) * width_px;
            markers.push_str(&format!(
                "<div class=\"cut\" style=\"left:{cut_x:.1}px\"></div><div class=\"cut-label cut-label-cut\" style=\"left:{cut_x:.1}px\">cut @{:.2}s</div>",
                window_start_secs + keep_until_secs
            ));
        }
        let target_cut_x = (rollback.target_keep_until_secs / window_duration_secs) * width_px;
        markers.push_str(&format!(
            "<div class=\"cut\" style=\"left:{target_cut_x:.1}px;background:#5a5a5a;opacity:0.55\"></div><div class=\"cut-label cut-label-target\" style=\"left:{target_cut_x:.1}px;color:#5a5a5a\">target @{:.2}s</div>",
            window_start_secs + rollback.target_keep_until_secs
        ));
        if let Some(replay_until_secs) = rollback.replay_until_secs {
            let replay_x = (replay_until_secs / window_duration_secs) * width_px;
            markers.push_str(&format!(
                "<div class=\"cut\" style=\"left:{replay_x:.1}px;background:#8b5e1a\"></div><div class=\"cut-label cut-label-bridge\" style=\"left:{replay_x:.1}px;color:#8b5e1a\">bridge @{:.2}s</div>",
                window_start_secs + replay_until_secs
            ));
        }
    }

    let mut word_divs = String::new();
    let mut fallback_x = 0.0;
    let mut lane_end_x = [0.0_f64; 3];
    let lane_tops = [16.0_f64, 46.0_f64, 76.0_f64];
    for word in words {
        let segment_class = if word.carried {
            "carried"
        } else if word.kept {
            "kept"
        } else if word.bridge {
            "bridge"
        } else {
            "rolled"
        };
        let cut_word_class = if word.cut_word { " cut-word" } else { "" };
        let class = format!(
            "word {segment_class} {}{cut_word_class}",
            word.quality_label
        );
        let (left, width, lane_index) = match (word.start_secs, word.end_secs) {
            (Some(start), Some(end)) => {
                let left = (start / window_duration_secs) * width_px;
                let width = ((end - start).max(0.08) / window_duration_secs) * width_px;
                let mut lane_index = 0usize;
                while lane_index + 1 < lane_end_x.len() && lane_end_x[lane_index] > left {
                    lane_index += 1;
                }
                if lane_end_x[lane_index] > left && lane_index == lane_end_x.len() - 1 {
                    lane_index = lane_end_x
                        .iter()
                        .enumerate()
                        .min_by(|a, b| a.1.total_cmp(b.1))
                        .map(|(idx, _)| idx)
                        .unwrap_or(0);
                }
                lane_end_x[lane_index] = left + width + 6.0;
                (left, width, lane_index)
            }
            _ => {
                let left = fallback_x;
                fallback_x += 90.0;
                (left, 84.0, 2)
            }
        };
        let top = lane_tops[lane_index.min(lane_tops.len() - 1)];
        word_divs.push_str(&format!(
            "<div class=\"{class}\" style=\"left:{left:.1}px;top:{top:.1}px;width:{width:.1}px\" title=\"window {start:.2}-{end:.2}s: {text}\" data-full-word=\"{text}\">{text}</div>",
            start = window_start_secs,
            end = window_end_secs,
            text = html_escape(&word.text)
        ));
    }

    let meta = if let Some(rollback) = rollback {
        let chosen_boundary = rollback
            .keep_boundary_debug
            .chosen_word
            .as_ref()
            .map(|word| {
                format!(
                    "{} [{:.2}-{:.2}s]",
                    html_escape(&word.text),
                    window_start_secs + word.start_secs,
                    window_start_secs + word.end_secs
                )
            })
            .unwrap_or_else(|| "none".to_string());
        format!(
            "audio {:.2}s..{:.2}s | replayed_prefix={} | kept_text={} | bridge_text={} | kept_tokens={} | bridge_tokens={} | rollback_position={} | keep_policy={} | keep_target={:.2}s | keep_cut={:.2}s | keep_search=[{:.2}s..{:.2}s] | min_keep={:.2}s | snapped={} | keep_word={}",
            window_start_secs,
            window_end_secs,
            html_escape(replayed_prefix.map(|p| p.text.as_str()).unwrap_or("none")),
            html_escape(&rollback.kept_text),
            html_escape(rollback.bridge_text.as_deref().unwrap_or("none")),
            rollback.kept_token_count,
            rollback.bridge_token_ids.len(),
            rollback.rollback_position,
            rollback.keep_boundary_policy.as_str(),
            window_start_secs + rollback.target_keep_until_secs,
            window_start_secs + rollback.keep_until_secs.unwrap_or(0.0),
            window_start_secs + rollback.keep_boundary_debug.earliest_candidate_secs,
            window_start_secs + rollback.target_keep_until_secs,
            window_start_secs + rollback.keep_boundary_debug.min_keep_secs,
            if rollback.keep_boundary_debug.snapped {
                "yes"
            } else {
                "no"
            },
            chosen_boundary
        )
    } else {
        format!(
            "audio {:.2}s..{:.2}s | final window",
            window_start_secs, window_end_secs
        )
    };

    format!(
        "<section class=\"row\" data-row-start=\"{:.6}\" data-row-end=\"{:.6}\"><div class=\"row-title\">{}</div><div class=\"row-meta\">{}</div><div class=\"row-audio\"><span>Chunk Audio</span><audio id=\"chunk-audio-{}\" controls preload=\"metadata\" src=\"{}\" data-window-start=\"{:.6}\" data-window-end=\"{:.6}\"></audio></div><div class=\"transcript-line\">{}</div><div class=\"timeline\"><div class=\"track\">{}{}</div><div class=\"axis\"><span>0.00s</span><span>{:.2}s window</span><span>{:.2}s total</span></div></div></section>",
        window_start_secs,
        window_end_secs,
        html_escape(&chunk.label),
        meta,
        row_index,
        audio_src,
        window_start_secs,
        window_end_secs,
        transcript_line,
        markers,
        word_divs,
        window_duration_secs,
        total_duration_secs
    )
}

fn build_window_word_placements(
    align_ctx: &mut AlignmentContext,
    run: &SlidingWindowRun,
    chunk_samples: &[f32],
) -> Result<Vec<SlidingWordPlacement>> {
    let generated_transcript = normalized_transcript(&run.chunk_run.transcript);
    let replayed_prefix = run
        .replayed_prefix
        .as_ref()
        .map(|prefix| normalized_transcript(&prefix.text))
        .filter(|text| !text.is_empty());
    if generated_transcript.is_empty() && replayed_prefix.is_none() {
        return Ok(Vec::new());
    }

    let combined_transcript = match (replayed_prefix, generated_transcript.is_empty()) {
        (Some(prefix), false) => format!("{prefix} {generated_transcript}"),
        (Some(prefix), true) => prefix.to_string(),
        (None, false) => generated_transcript.to_string(),
        (None, true) => String::new(),
    };
    let alignment = build_transcript_alignment(align_ctx, &combined_transcript, chunk_samples)?;
    let word_timings = alignment.word_timings();
    let carried_word_count = replayed_prefix
        .map(sentence_word_tokens)
        .map(|words| words.len())
        .unwrap_or(0);
    let kept_word_count = run
        .rollback
        .as_ref()
        .map_or(word_timings.len(), |r| r.kept_word_count);
    let replay_word_count = run.rollback.as_ref().map_or(word_timings.len(), |r| {
        let bridge_words = r
            .bridge_text
            .as_ref()
            .map(|text| sentence_word_tokens(text).len())
            .unwrap_or(0);
        r.kept_word_count + bridge_words
    });
    let chosen_cut = run
        .rollback
        .as_ref()
        .and_then(|r| r.keep_boundary_debug.chosen_word.as_ref());

    Ok(word_timings
        .into_iter()
        .enumerate()
        .map(|(index, word_timing)| {
            let (start_secs, end_secs, quality_label) = match word_timing.quality {
                bee_transcribe::zipa_align::AlignmentQuality::Aligned {
                    start_secs,
                    end_secs,
                } => (Some(start_secs), Some(end_secs), "aligned"),
                bee_transcribe::zipa_align::AlignmentQuality::NoWindow => (None, None, "no-window"),
                bee_transcribe::zipa_align::AlignmentQuality::NoTiming => (None, None, "no-timing"),
            };
            let generated_index = index.saturating_sub(carried_word_count);
            let carried = index < carried_word_count;
            let cut_word = match (&chosen_cut, start_secs, end_secs) {
                (Some(chosen), Some(start), Some(end)) => {
                    word_timing.word == chosen.text
                        && (start - chosen.start_secs).abs() < 0.000_1
                        && (end - chosen.end_secs).abs() < 0.000_1
                }
                _ => false,
            };
            SlidingWordPlacement {
                text: word_timing.word.to_string(),
                start_secs,
                end_secs,
                quality_label,
                carried,
                kept: !carried && generated_index < kept_word_count,
                bridge: !carried
                    && generated_index >= kept_word_count
                    && generated_index < replay_word_count,
                cut_word,
            }
        })
        .collect())
}

fn write_chunk_segment_merge_rollback_html(
    baseline_runs: &[ChunkRun],
    replay_runs: &[ChunkRun],
    samples: &[f32],
) -> Result<PathBuf> {
    let mut align_ctx = AlignmentContext::new()?;
    let baseline_words = build_word_placements(&mut align_ctx, baseline_runs, samples)?;
    let replay_words = build_word_placements(&mut align_ctx, replay_runs, samples)?;
    let duration_secs = samples.len() as f64 / SAMPLE_RATE as f64;
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.artifacts/bee-kv");
    fs::create_dir_all(&out_dir).with_context(|| format!("creating {}", out_dir.display()))?;
    let out_path = out_dir.join("chunk-segment-merge-rollback.html");
    let html = render_word_timeline_html(
        duration_secs,
        baseline_runs,
        replay_runs,
        &baseline_words,
        &replay_words,
    );
    fs::write(&out_path, html).with_context(|| format!("writing {}", out_path.display()))?;
    Ok(out_path)
}

fn render_word_timeline_html(
    duration_secs: f64,
    baseline_runs: &[ChunkRun],
    replay_runs: &[ChunkRun],
    baseline_words: &[WordPlacement],
    replay_words: &[WordPlacement],
) -> String {
    let width_px = 1400.0;
    let row_height_px = 132.0;
    let baseline_row = render_word_row(
        "Baseline",
        width_px,
        row_height_px,
        duration_secs,
        baseline_runs,
        baseline_words,
    );
    let replay_row = render_word_row(
        "Replay",
        width_px,
        row_height_px,
        duration_secs,
        replay_runs,
        replay_words,
    );
    format!(
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>bee-kv rollback word timeline</title><style>\
body{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#f7f4ec;color:#1d1b19;padding:24px;}}\
.timeline{{width:{width_px}px;border:1px solid #b9b09f;background:#fffdf8;position:relative;padding:12px 0;margin-bottom:28px;}}\
.row-title{{font-weight:700;margin:0 0 8px 0;}}\
.transcript-line{{width:{width_px}px;margin:0 0 8px 0;font-size:13px;line-height:1.5;color:#3b352d;}}\
.chunk-divider{{color:#8a7f6a;padding:0 6px;}}\
.track{{position:relative;width:{width_px}px;height:{row_height_px}px;border-top:1px solid #d6cfbf;border-bottom:1px solid #d6cfbf;background:linear-gradient(180deg,#fffdf8,#f4efe3);overflow:hidden;}}\
.word{{text-align:center;vertical-align:middle;position:absolute;height:2em;padding:4px 2px;border-radius:4px;border:1px solid #7a6d56;background:#efe2b8;white-space:nowrap;overflow:visible;font-size:12px;box-sizing:border-box;cursor:default;font-family:\"SF Pro\", serif;}}\
.word.chunk-0{{background:#e7d9a8;}} .word.chunk-1{{background:#cfdcc8;}} .word.chunk-2{{background:#d7d0ea;}}\
.word.no-window,.word.no-timing{{background:#e7c2c2;border-color:#9f5d5d;height:22px;font-size:11px;}}\
.boundary{{position:absolute;top:0;width:1px;height:{row_height_px}px;background:#9d9483;}}\
.boundary-label{{position:absolute;top:2px;transform:translateX(4px);font-size:11px;color:#6d6457;}}\
.axis{{display:flex;justify-content:space-between;font-size:12px;color:#6d6457;margin-top:6px;}}\
.legend{{margin-bottom:16px;font-size:13px;color:#514a41;}}\
.word:hover::after{{content:attr(data-full-word);position:absolute;left:0;top:-28px;background:#1d1b19;color:#fffdf8;padding:2px 6px;border-radius:4px;white-space:nowrap;z-index:10;font-size:11px;line-height:16px;box-shadow:0 2px 6px rgba(0,0,0,0.18);}}\
</style></head><body><h1>bee-kv rollback word timeline</h1><p class=\"legend\">Word boxes are positioned by ZIPA-derived word timings. Vertical lines mark chunk ends.</p>{baseline_row}{replay_row}</body></html>"
    )
}

fn render_word_row(
    title: &str,
    width_px: f64,
    _row_height_px: f64,
    duration_secs: f64,
    runs: &[ChunkRun],
    words: &[WordPlacement],
) -> String {
    let transcript_line = runs
        .iter()
        .map(|chunk| html_escape(&chunk.transcript))
        .collect::<Vec<_>>()
        .join("<span class=\"chunk-divider\">|</span>");
    let mut boundaries = String::new();
    for run in runs {
        let x = ((run.end_sample as f64 / SAMPLE_RATE as f64) / duration_secs) * width_px;
        let ms = (run.end_sample * 1000) / SAMPLE_RATE as usize;
        boundaries.push_str(&format!(
            "<div class=\"boundary\" style=\"left:{x:.1}px\"></div><div class=\"boundary-label\" style=\"left:{x:.1}px\">{ms}ms</div>"
        ));
    }

    let mut word_divs = String::new();
    let mut fallback_x = 0.0;
    let mut lane_end_x = [0.0_f64; 3];
    let lane_tops = [20.0_f64, 54.0_f64, 88.0_f64];
    for word in words {
        let class = format!(
            "word chunk-{} {}",
            word.chunk_index.min(2),
            word.quality_label
        );
        let (left, width, lane_index) = match (word.start_secs, word.end_secs) {
            (Some(start), Some(end)) => {
                let left = (start / duration_secs) * width_px;
                let width = ((end - start).max(0.08) / duration_secs) * width_px;
                let mut lane_index = 0usize;
                while lane_index + 1 < lane_end_x.len() && lane_end_x[lane_index] > left {
                    lane_index += 1;
                }
                if lane_end_x[lane_index] > left && lane_index == lane_end_x.len() - 1 {
                    let min_lane = lane_end_x
                        .iter()
                        .enumerate()
                        .min_by(|a, b| a.1.total_cmp(b.1))
                        .map(|(idx, _)| idx)
                        .unwrap_or(0);
                    lane_index = min_lane;
                }
                lane_end_x[lane_index] = left + width + 6.0;
                (left, width, lane_index)
            }
            _ => {
                let left = fallback_x;
                fallback_x += 90.0;
                (left, 84.0, 2)
            }
        };
        let top = lane_tops[lane_index.min(lane_tops.len() - 1)];
        word_divs.push_str(&format!(
            "<div class=\"{class}\" style=\"left:{left:.1}px;top:{top:.1}px;width:{width:.1}px\" title=\"{title}: {text}\" data-full-word=\"{text}\">{text}</div>",
            text = html_escape(&word.text)
        ));
    }

    format!(
        "<section><div class=\"row-title\">{}</div><div class=\"transcript-line\">{}</div><div class=\"timeline\"><div class=\"track\">{}{}</div><div class=\"axis\"><span>0.00s</span><span>{:.2}s</span></div></div></section>",
        html_escape(title),
        transcript_line,
        boundaries,
        word_divs,
        duration_secs
    )
}

fn html_escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('\"', "&quot;")
}

fn file_url_for_path(path: &Path) -> Result<String> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::fs::canonicalize(path).with_context(|| format!("canonicalizing {}", path.display()))?
    };
    Ok(format!("file://{}", absolute.display()))
}

struct AlignmentContext {
    g2p: CachedEspeakG2p,
    zipa: ZipaInference,
}

impl AlignmentContext {
    fn new() -> Result<Self> {
        Ok(Self {
            g2p: CachedEspeakG2p::english(&g2p_base_dir()).context("initializing g2p engine")?,
            zipa: ZipaInference::load_quantized_bundle_dir(&zipa_bundle_dir()?)
                .context("loading ZIPA bundle")?,
        })
    }
}

fn build_transcript_alignment(
    align_ctx: &mut AlignmentContext,
    transcript: &str,
    samples: &[f32],
) -> Result<TranscriptAlignment> {
    let zipa_audio = ZipaAudioBuffer {
        samples: samples.to_vec(),
        sample_rate_hz: SAMPLE_RATE,
    };

    TranscriptAlignment::build(transcript, &zipa_audio, &mut align_ctx.g2p, &align_ctx.zipa)
        .map_err(|error| anyhow::anyhow!(error.to_string()))
}

fn format_span_timing(span_timing: SpanTiming) -> String {
    match span_timing {
        SpanTiming::Aligned {
            start_secs,
            end_secs,
        } => format!("{{{start_secs:.2}-{end_secs:.2}s}}"),
        SpanTiming::PartialGap {
            start_secs,
            end_secs,
        } => format!("{{partial {start_secs:.2}-{end_secs:.2}s}}"),
        SpanTiming::NoAlignedWords => "{no-aligned-words}".to_string(),
        SpanTiming::NoTiming => "{no-timing}".to_string(),
    }
}

fn g2p_base_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target")
}

fn zipa_bundle_dir() -> Result<PathBuf> {
    if let Ok(path) = env::var("BEE_ZIPA_BUNDLE_DIR") {
        return Ok(PathBuf::from(path));
    }

    let home = env::var("HOME").context("HOME is not set for ZIPA fallback path")?;
    Ok(PathBuf::from(home).join("bearcove/zipa-mlx-hf"))
}

fn print_lane_row(label: &str, chunks: &[ChunkRun]) {
    for chunk in chunks {
        let seam_ms = (chunk.end_sample * 1000) / SAMPLE_RATE as usize;
        println!(
            "{} seam@{}ms [{} tok]: {}",
            label, seam_ms, chunk.generated_tokens, chunk.transcript
        );
    }
    println!();
}

fn print_chunk_run(chunk: &ChunkRun) {
    println!("--- {} ---", chunk.label);
    println!(
        "samples={} audio={}..{}ms prompt_tokens={} generated_tokens={} decode_ms={:.1} stop_reason={} start_position={} end_position={}",
        chunk.sample_count,
        (chunk.start_sample * 1000) / SAMPLE_RATE as usize,
        (chunk.end_sample * 1000) / SAMPLE_RATE as usize,
        chunk.prompt_tokens,
        chunk.generated_tokens,
        chunk.decode_ms,
        chunk.stop_reason.as_str(),
        chunk.start_position,
        chunk.end_position
    );
    println!("{}", chunk.transcript);
    println!();
}

fn print_finalizing_banner() {
    println!(
        "{ANSI_BLUE}{ANSI_BOLD}===================== FINALIZING ====================={ANSI_RESET}"
    );
}

struct ExerciseTui {
    enabled: bool,
    terminal: Option<Terminal<CrosstermBackend<io::Stdout>>>,
    phase: String,
    chunk_index: usize,
    committed: String,
    draft: String,
    logs: VecDeque<String>,
}

impl ExerciseTui {
    fn new() -> Self {
        let enabled = std::io::stdout().is_terminal();
        let terminal = if enabled {
            let mut stdout = std::io::stdout();
            execute!(stdout, EnterAlternateScreen, Hide).ok();
            let backend = CrosstermBackend::new(stdout);
            Terminal::new(backend).ok()
        } else {
            None
        };
        let mut tui = Self {
            enabled,
            terminal,
            phase: "Starting".to_string(),
            chunk_index: 0,
            committed: String::new(),
            draft: String::new(),
            logs: VecDeque::new(),
        };
        tui.render();
        tui
    }

    fn clear(&mut self) {
        if !self.enabled {
            return;
        }
        if let Some(terminal) = self.terminal.as_mut() {
            let _ = terminal.clear();
        }
        let _ = self.terminal.take();
        let mut stdout = std::io::stdout();
        let _ = execute!(stdout, Show, LeaveAlternateScreen);
        self.enabled = false;
    }

    fn log(&mut self, message: impl Into<String>) {
        if !self.enabled {
            return;
        }
        self.logs.push_back(message.into());
        while self.logs.len() > 8 {
            self.logs.pop_front();
        }
        self.render();
    }

    fn update(&mut self, phase: &str, chunk_index: usize, committed: &str, draft: &str) {
        if !self.enabled {
            return;
        }
        self.phase.clear();
        self.phase.push_str(phase);
        self.chunk_index = chunk_index;
        self.committed.clear();
        self.committed.push_str(committed);
        self.draft.clear();
        self.draft.push_str(draft);
        self.render();
    }

    fn render(&mut self) {
        if !self.enabled {
            return;
        }
        let phase = self.phase.clone();
        let chunk_index = self.chunk_index;
        let committed = if self.committed.trim().is_empty() {
            "[empty]".to_string()
        } else {
            self.committed.clone()
        };
        let draft = if self.draft.trim().is_empty() {
            "[empty]".to_string()
        } else {
            self.draft.clone()
        };
        let logs = self.logs.iter().cloned().collect::<Vec<_>>();
        if let Some(terminal) = self.terminal.as_mut() {
            let _ = terminal.draw(|frame| {
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Length(3),
                        Constraint::Min(6),
                        Constraint::Min(6),
                        Constraint::Length(10),
                    ])
                    .split(frame.area());

                let header = Paragraph::new(Line::from(vec![
                    Span::styled(
                        "Bee Exercise",
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw("  "),
                    Span::styled(
                        format!("{phase}"),
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw("  chunk "),
                    Span::styled(format!("{chunk_index}"), Style::default().fg(Color::Green)),
                ]))
                .block(Block::default().borders(Borders::ALL).title("Status"));

                let committed_panel = Paragraph::new(committed)
                    .style(Style::default().fg(Color::White))
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .title("Committed")
                            .border_style(Style::default().fg(Color::Green)),
                    )
                    .wrap(Wrap { trim: false });

                let draft_panel = Paragraph::new(draft)
                    .style(Style::default().fg(Color::Yellow))
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .title("Draft")
                            .border_style(Style::default().fg(Color::Yellow)),
                    )
                    .wrap(Wrap { trim: false });

                let log_items = if logs.is_empty() {
                    vec![ListItem::new(Line::from(Span::styled(
                        "No events yet",
                        Style::default().fg(Color::DarkGray),
                    )))]
                } else {
                    logs.into_iter()
                        .map(|entry| {
                            ListItem::new(Line::from(Span::styled(
                                entry,
                                Style::default().fg(Color::LightYellow),
                            )))
                        })
                        .collect()
                };
                let event_log = List::new(log_items).block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title("Event Log")
                        .border_style(Style::default().fg(Color::DarkGray)),
                );

                frame.render_widget(header, chunks[0]);
                frame.render_widget(committed_panel, chunks[1]);
                frame.render_widget(draft_panel, chunks[2]);
                frame.render_widget(event_log, chunks[3]);
            });
        }
    }
}

fn update_exercise_progress(
    tui: &mut ExerciseTui,
    phase: &str,
    chunk_index: usize,
    committed: &str,
    draft: &str,
) {
    tui.update(phase, chunk_index, committed, draft);
}

fn append_exact(target: &mut String, text: &str) {
    if !text.is_empty() {
        target.push_str(text);
    }
}

fn suffix_after_prefix<'a>(prefix: Option<&str>, text: &'a str) -> &'a str {
    let Some(prefix) = prefix else {
        return text;
    };
    if let Some(suffix) = text.strip_prefix(prefix) {
        suffix
    } else {
        text
    }
}

struct TimedGeneratedPrefix {
    kept_word_count: usize,
    kept_token_count: usize,
}

#[derive(Clone, Debug)]
struct TimedWord {
    text: String,
    char_range: std::ops::Range<usize>,
    start_secs: f64,
    end_secs: f64,
}

struct TimedGeneratedBridge {
    kept_word_count: usize,
    kept_token_count: usize,
    kept_text: String,
    bridge: CarriedBridge,
}

fn timed_aligned_words_for_alignment(
    transcript: &str,
    alignment: &TranscriptAlignment,
) -> Result<Vec<TimedWord>> {
    let word_ranges = sentence_word_tokens(transcript);
    let word_timings = alignment.word_timings();
    if word_ranges.len() != word_timings.len() {
        bail!(
            "alignment word count mismatch: transcript has {} words, alignment has {}",
            word_ranges.len(),
            word_timings.len()
        );
    }

    let mut timed_words = Vec::with_capacity(word_ranges.len());
    for (word_range, word_timing) in word_ranges.iter().zip(word_timings.iter()) {
        let bee_transcribe::zipa_align::AlignmentQuality::Aligned {
            start_secs,
            end_secs,
        } = word_timing.quality
        else {
            break;
        };

        timed_words.push(TimedWord {
            text: word_timing.word.to_string(),
            char_range: word_range.char_start..word_range.char_end,
            start_secs,
            end_secs,
        });
    }

    Ok(timed_words)
}

fn timed_generated_prefix_for_cut(
    align_ctx: &mut AlignmentContext,
    tokenizer: &Tokenizer,
    chunk_run: &ChunkRun,
    chunk_samples: &[f32],
    keep_until_secs: f64,
) -> Result<TimedGeneratedPrefix> {
    let transcript = normalized_transcript(&chunk_run.transcript);
    if transcript.is_empty() {
        return Ok(TimedGeneratedPrefix {
            kept_word_count: 0,
            kept_token_count: 0,
        });
    }

    let alignment = build_transcript_alignment(align_ctx, transcript, chunk_samples)?;
    let timed_words = timed_aligned_words_for_alignment(transcript, &alignment)?;
    let kept_word_count = timed_words
        .iter()
        .take_while(|word| word.end_secs <= keep_until_secs)
        .count();

    let kept_text = if kept_word_count == 0 {
        String::new()
    } else {
        let end = timed_words
            .get(kept_word_count - 1)
            .map(|word| word.char_range.end)
            .ok_or_else(|| anyhow::anyhow!("missing word range for kept prefix"))?;
        transcript[..end].to_string()
    };

    let kept_token_count = if kept_text.is_empty() {
        0
    } else {
        tokenizer
            .encode_fast(kept_text.as_str(), false)
            .map_err(|e| anyhow::anyhow!("encoding kept prefix: {e}"))?
            .len()
    };

    Ok(TimedGeneratedPrefix {
        kept_word_count,
        kept_token_count,
    })
}

fn timed_generated_bridge_for_cuts(
    tokenizer: &Tokenizer,
    combined_transcript: &str,
    replayed_prefix: Option<&CarriedBridge>,
    timed_words: &[TimedWord],
    keep_until_secs: f64,
    replay_until_secs: f64,
) -> Result<TimedGeneratedBridge> {
    if timed_words.is_empty() {
        let bridge = replayed_prefix.cloned().unwrap_or_else(|| CarriedBridge {
            token_ids: Vec::new(),
            text: String::new(),
            words: Vec::new(),
        });
        return Ok(TimedGeneratedBridge {
            kept_word_count: 0,
            kept_token_count: 0,
            kept_text: String::new(),
            bridge,
        });
    }

    let carried_word_count = replayed_prefix
        .map(|prefix| {
            if prefix.words.is_empty() {
                sentence_word_tokens(&prefix.text).len()
            } else {
                prefix.words.len()
            }
        })
        .unwrap_or(0);
    let generated_words = &timed_words[carried_word_count.min(timed_words.len())..];
    let kept_word_count = generated_words
        .iter()
        .take_while(|word| word.end_secs <= keep_until_secs)
        .count();
    let bridge_start_index = timed_words.partition_point(|word| word.start_secs < keep_until_secs);
    let bridge_end_index = timed_words.partition_point(|word| word.start_secs < replay_until_secs);
    let bridge_words_slice = if bridge_start_index <= bridge_end_index {
        &timed_words[bridge_start_index..bridge_end_index]
    } else {
        &timed_words[0..0]
    };

    let kept_text = if let Some(end_word) = timed_words
        .iter()
        .filter(|word| word.end_secs <= keep_until_secs)
        .last()
    {
        combined_transcript[..end_word.char_range.end].to_string()
    } else {
        String::new()
    };

    let bridge_words = bridge_words_slice
        .iter()
        .map(|word| CarriedBridgeWord {
            text: word.text.clone(),
            start_secs: (word.start_secs - keep_until_secs).max(0.0),
            end_secs: (word.end_secs - keep_until_secs).max(0.0),
        })
        .collect();

    let bridge = if let (Some(first_bridge_word), Some(last_bridge_word)) =
        (bridge_words_slice.first(), bridge_words_slice.last())
    {
        let text = combined_transcript
            [first_bridge_word.char_range.start..last_bridge_word.char_range.end]
            .to_string();
        CarriedBridge {
            token_ids: tokenize_token_ids(tokenizer, &text)?,
            text,
            words: bridge_words,
        }
    } else if bridge_words_slice.is_empty() {
        CarriedBridge {
            token_ids: Vec::new(),
            text: String::new(),
            words: Vec::new(),
        }
    } else {
        unreachable!()
    };

    let generated_kept_text = if kept_word_count == 0 {
        String::new()
    } else {
        let generated_start = generated_words
            .first()
            .map(|word| word.char_range.start)
            .ok_or_else(|| anyhow::anyhow!("missing generated word start"))?;
        let generated_end = generated_words
            .get(kept_word_count - 1)
            .map(|word| word.char_range.end)
            .ok_or_else(|| anyhow::anyhow!("missing generated kept word range"))?;
        combined_transcript[generated_start..generated_end].to_string()
    };

    let kept_token_count = if generated_kept_text.is_empty() {
        0
    } else {
        tokenizer
            .encode_fast(generated_kept_text.as_str(), false)
            .map_err(|e| anyhow::anyhow!("encoding kept bridge prefix: {e}"))?
            .len()
    };

    Ok(TimedGeneratedBridge {
        kept_word_count,
        kept_token_count,
        kept_text,
        bridge,
    })
}

fn adjust_keep_boundary_secs(
    policy: KeepBoundaryPolicy,
    alignment: &TranscriptAlignment,
    target_keep_until_secs: f64,
    replay_until_secs: f64,
) -> Result<(f64, KeepBoundaryDebug)> {
    let fixed_debug = KeepBoundaryDebug {
        earliest_candidate_secs: target_keep_until_secs,
        min_keep_secs: target_keep_until_secs,
        snapped: false,
        chosen_word: None,
    };
    if policy == KeepBoundaryPolicy::Fixed {
        return Ok((target_keep_until_secs, fixed_debug));
    }

    let min_keep_secs = KEEP_BOUNDARY_MIN_KEPT_SECS.min(target_keep_until_secs);
    let earliest_candidate_secs = min_keep_secs;
    let mut best_candidate = None;
    let mut best_distance = f64::INFINITY;

    for word_timing in alignment.word_timings() {
        if let bee_transcribe::zipa_align::AlignmentQuality::Aligned {
            start_secs,
            end_secs,
        } = word_timing.quality
        {
            if end_secs <= 0.0 || end_secs >= replay_until_secs {
                continue;
            }
            if end_secs > target_keep_until_secs || end_secs < earliest_candidate_secs {
                continue;
            }
            let distance = (end_secs - target_keep_until_secs).abs();
            if distance < best_distance {
                best_distance = distance;
                best_candidate = Some(BoundaryWordDebug {
                    text: word_timing.word.to_string(),
                    start_secs,
                    end_secs,
                });
            }
        }
    }

    let chosen_word = best_candidate;
    let keep_until_secs = chosen_word
        .as_ref()
        .map(|word| word.end_secs)
        .unwrap_or(target_keep_until_secs);
    let debug = KeepBoundaryDebug {
        earliest_candidate_secs,
        min_keep_secs,
        snapped: chosen_word.is_some(),
        chosen_word,
    };

    Ok((keep_until_secs, debug))
}
