use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};
use bee_qwen3_asr::config::AsrConfig;
use bee_qwen3_asr::load;
use bee_qwen3_asr::mel::{MelExtractor, load_audio};
use bee_qwen3_asr::mlx_rs::Array;
use bee_qwen3_asr::mlx_rs::module::{Module, ModuleParametersExt};
use bee_qwen3_asr::mlx_rs::ops;
use bee_qwen3_asr::model::Qwen3ASRModel;
use bee_qwen3_asr::tokenizers::Tokenizer;

mod alignment;
mod decode;
mod html;
mod print;
mod tui;
mod types;

use decode::*;
use print::*;
use types::*;

/// Default transcription language passed to the ASR model.
const DEFAULT_LANGUAGE: &str = "English";
/// Maximum number of tokens the model may generate per decode call.
const DEFAULT_MAX_NEW_TOKENS: usize = 256;
/// Audio sample rate in Hz expected by the mel extractor and alignment engine.
const SAMPLE_RATE: u32 = 16_000;
/// FFT window size for mel-spectrogram computation.
const N_FFT: usize = 400;
/// Hop length (in samples) between successive mel frames.
const HOP_LENGTH: usize = 160;
/// Fallback WAV file path relative to the crate root.
const DEFAULT_WAV_RELATIVE_TO_CRATE: &str = "../../.artifacts/repros/frozen/EB54CF36.wav";
/// Audio start position (in samples) for the fresh follow-up decode mode.
const DEFAULT_START_POSITION_FOR_FRESH_FOLLOWUP: usize = 0;
/// Default audio chunk duration in milliseconds for chunked decode modes.
const DEFAULT_CHUNK_MS: usize = 2_000;
/// Default rollback duration in milliseconds for sliding-window modes.
const DEFAULT_ROLLBACK_MS: usize = 1_000;
/// Default bridge window duration in milliseconds for bridge-replay mode.
const DEFAULT_BRIDGE_MS: usize = 1_000;
/// Optional explicit stride between chunks; `None` means stride = chunk size - rollback.
const DEFAULT_STRIDE_MS: Option<usize> = None;
/// Default policy for snapping the keep boundary to a word edge.
const DEFAULT_KEEP_BOUNDARY_POLICY: KeepBoundaryPolicy = KeepBoundaryPolicy::Fixed;
/// Default first-chunk duration in milliseconds for lane B in dual-lane mode.
const DEFAULT_LANE_B_FIRST_CHUNK_MS: usize = 3_000;
/// Default chunk index at which truncate-replay begins replaying.
const DEFAULT_REPLAY_CHUNK_INDEX: usize = 1;
/// Default number of tokens to truncate when replaying from a rollback point.
const DEFAULT_TRUNCATE_TOKENS: usize = 4;
/// Optional MLX memory cache limit in megabytes; `None` means unlimited.
const DEFAULT_MLX_CACHE_LIMIT_MB: Option<usize> = None;
/// Minimum audio duration (in seconds) that must be kept when adjusting chunk boundaries.
const KEEP_BOUNDARY_MIN_KEPT_SECS: f64 = 0.200;
/// Maximum number of bridge windows before forcing a full decode reset.
const MAX_BRIDGE_WINDOWS: usize = 50;
/// ANSI escape code for blue text.
const ANSI_BLUE: &str = "\x1b[34m";
/// ANSI escape code for bold text.
const ANSI_BOLD: &str = "\x1b[1m";
/// ANSI escape code to reset text formatting.
const ANSI_RESET: &str = "\x1b[0m";
/// Token-level offsets to probe when sweeping for optimal chunk boundaries.
const BOUNDARY_SWEEP_OFFSETS: [isize; 13] =
    [-48, -44, -40, -36, -32, -28, -24, -20, -16, -12, -8, -4, 0];

/// Loads the ASR model and audio, then runs the decode mode specified by CLI arguments.
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
