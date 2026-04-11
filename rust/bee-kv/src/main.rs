use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};
use bee_qwen3_asr::config::AsrConfig;
use bee_qwen3_asr::load;
use bee_qwen3_asr::mel::{MelExtractor, load_audio};
use bee_qwen3_asr::mlx_rs::module::ModuleParametersExt;
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
/// Default audio chunk duration in milliseconds for the bridge replay mode.
const DEFAULT_CHUNK_MS: usize = 2_000;
/// Default rollback duration in milliseconds for the bridge replay mode.
const DEFAULT_ROLLBACK_MS: usize = 1_000;
/// Default bridge window duration in milliseconds for bridge-replay mode.
const DEFAULT_BRIDGE_MS: usize = 1_000;
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
    let (_, _, n_frames) = mel_extractor
        .extract(&samples)
        .context("extracting log-mel features")?;

    let summary = RunSummary {
        wav_path: &args.wav_path,
        model_dir: &args.model_dir,
        tokenizer_path: &args.tokenizer_path,
        language: &args.language,
        load_stats: &load_stats,
        sample_count: samples.len(),
        mel_frames: n_frames,
        audio_tokens: 0,
    };

    let experiment = match args.mode {
        Mode::SlidingWindowBridgeReplay => decode_sliding_window_bridge_replay(
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
        )?,
    };
    print_summary(&summary);
    print_sliding_window_timed_rollback_experiment(&experiment);

    Ok(())
}
