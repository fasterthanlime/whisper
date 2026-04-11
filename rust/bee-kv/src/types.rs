use std::env;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use bee_qwen3_asr::generate::DecodeStopReason;
use bee_qwen3_asr::load;
use clap::{Parser, ValueEnum};

use crate::{
    DEFAULT_BRIDGE_MS, DEFAULT_CHUNK_MS, DEFAULT_LANGUAGE, DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_ROLLBACK_MS, DEFAULT_WAV_RELATIVE_TO_CRATE, MAX_BRIDGE_WINDOWS,
};

/// CLI arguments parsed from `clap`.
#[derive(Debug, Parser)]
#[command(
    name = "bee-kv",
    version,
    about = "Sliding-window bridge replay for bee-kv",
    after_help = "Environment:\n  BEE_ASR_MODEL_DIR\n  BEE_TOKENIZER_PATH (optional; defaults to $BEE_ASR_MODEL_DIR/tokenizer.json)"
)]
pub(crate) struct Args {
    /// Decoding mode.
    #[arg(long, value_enum, default_value_t = Mode::SlidingWindowBridgeReplay)]
    pub(crate) mode: Mode,
    /// Optional context string injected into the initial prompt.
    #[arg(long, default_value_t = String::new())]
    pub(crate) context: String,
    /// Audio chunk duration in milliseconds.
    #[arg(long = "chunk-ms", default_value_t = DEFAULT_CHUNK_MS)]
    pub(crate) chunk_ms: usize,
    /// Bridge window duration in milliseconds for bridge-replay mode.
    #[arg(long = "bridge-ms", default_value_t = DEFAULT_BRIDGE_MS)]
    pub(crate) bridge_ms: usize,
    /// Maximum number of bridge windows before forcing a full reset.
    #[arg(long = "max-bridge-windows", default_value_t = MAX_BRIDGE_WINDOWS)]
    pub(crate) max_bridge_windows: usize,
    /// Optional explicit stride between chunks in milliseconds.
    #[arg(long = "stride-ms")]
    pub(crate) stride_ms: Option<usize>,
    /// Policy for snapping the keep boundary to a word edge.
    #[arg(long = "keep-boundary", value_enum, default_value_t = KeepBoundaryPolicy::Fixed)]
    pub(crate) keep_boundary_policy: KeepBoundaryPolicy,
    /// Rollback duration in milliseconds for sliding-window mode.
    #[arg(long = "rollback-ms", default_value_t = DEFAULT_ROLLBACK_MS)]
    pub(crate) rollback_ms: usize,
    /// Optional MLX memory cache limit in megabytes.
    #[arg(long = "mlx-cache-limit-mb")]
    pub(crate) mlx_cache_limit_mb: Option<usize>,
    /// Path to the input WAV file.
    #[arg(index = 1, value_name = "WAV_PATH", default_value_os_t = default_wav_path())]
    pub(crate) wav_path: PathBuf,
    /// Language hint passed to the ASR model prompt.
    #[arg(index = 2, value_name = "LANGUAGE", default_value_t = String::from(DEFAULT_LANGUAGE))]
    pub(crate) language: String,
    /// Maximum number of tokens the model may generate per decode call.
    #[arg(index = 3, value_name = "MAX_NEW_TOKENS", default_value_t = DEFAULT_MAX_NEW_TOKENS)]
    pub(crate) max_new_tokens: usize,
    /// Directory containing the ASR model weights and config.
    #[arg(skip = PathBuf::new())]
    pub(crate) model_dir: PathBuf,
    /// Path to the tokenizer JSON file.
    #[arg(skip = PathBuf::new())]
    pub(crate) tokenizer_path: PathBuf,
}

impl Args {
    /// Parses CLI arguments from `env::args()` into an `Args` struct.
    pub(crate) fn parse() -> Result<Self> {
        let mut args = <Self as Parser>::parse();
        args.model_dir = env_path("BEE_ASR_MODEL_DIR")?;
        args.tokenizer_path = env_path("BEE_TOKENIZER_PATH")
            .unwrap_or_else(|_| args.model_dir.join("tokenizer.json"));

        if !args.wav_path.is_file() {
            bail!("wav file not found: {}", args.wav_path.display());
        }
        if !args.model_dir.is_dir() {
            bail!("model dir not found: {}", args.model_dir.display());
        }
        if !args.tokenizer_path.is_file() {
            bail!("tokenizer not found: {}", args.tokenizer_path.display());
        }
        Ok(args)
    }
}

/// Available CLI modes.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub(crate) enum Mode {
    /// Sliding-window bridge replay mode.
    SlidingWindowBridgeReplay,
}

/// Policy for snapping the keep boundary to a word edge.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub(crate) enum KeepBoundaryPolicy {
    /// Keep boundary at the exact time-based position (no snapping).
    Fixed,
    /// Snap the keep boundary to the nearest word end.
    NearestWordEnd,
}

impl KeepBoundaryPolicy {
    /// Returns the CLI string representation.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::NearestWordEnd => "nearest-word-end",
        }
    }
}

/// Summary of model and audio info printed before experiment results.
pub(crate) struct RunSummary<'a> {
    /// Path to the input WAV file.
    pub(crate) wav_path: &'a Path,
    /// Directory containing the ASR model.
    pub(crate) model_dir: &'a Path,
    /// Path to the tokenizer file.
    pub(crate) tokenizer_path: &'a Path,
    /// Language used for the ASR prompt.
    pub(crate) language: &'a str,
    /// Statistics from loading the model weights.
    pub(crate) load_stats: &'a load::LoadStats,
    /// Total number of audio samples in the WAV file.
    pub(crate) sample_count: usize,
    /// Number of mel-spectrogram frames extracted.
    pub(crate) mel_frames: usize,
    /// Number of audio tokens produced by the audio encoder.
    pub(crate) audio_tokens: usize,
}

/// Detailed per-chunk decode state including token IDs and timing.
pub(crate) struct ChunkRun {
    /// Human-readable label for this chunk.
    pub(crate) label: String,
    /// Number of prompt tokens fed to the model.
    pub(crate) prompt_tokens: usize,
    /// Number of tokens generated by the model.
    pub(crate) generated_tokens: usize,
    /// Raw generated token IDs (used for KV cache replay).
    pub(crate) generated_token_ids: Vec<u32>,
    /// Decoded transcript text.
    pub(crate) transcript: String,
    /// Number of audio samples in this chunk.
    pub(crate) sample_count: usize,
    /// Wall-clock decode time in milliseconds.
    pub(crate) decode_ms: f64,
    /// Why the model stopped generating (EOS, max tokens, etc.).
    pub(crate) stop_reason: DecodeStopReason,
    /// KV cache start position for this chunk.
    pub(crate) start_position: usize,
    /// KV cache end position after this chunk.
    pub(crate) end_position: usize,
    /// Start sample index in the full audio.
    pub(crate) start_sample: usize,
    /// End sample index in the full audio.
    pub(crate) end_sample: usize,
}

/// Decision about where to keep/rollback in a sliding window.
pub(crate) struct WindowRollbackDecision {
    /// Policy used to snap the boundary.
    pub(crate) keep_boundary_policy: KeepBoundaryPolicy,
    /// Target time (in seconds within window) to keep up to.
    pub(crate) target_keep_until_secs: f64,
    /// Actual keep boundary time after snapping (if any).
    pub(crate) keep_until_secs: Option<f64>,
    /// Time (in seconds within window) up to which bridge tokens are replayed.
    pub(crate) replay_until_secs: Option<f64>,
    /// Number of generated words kept from this window.
    pub(crate) kept_word_count: usize,
    /// Number of generated tokens kept in the KV cache.
    pub(crate) kept_token_count: usize,
    /// Token IDs of the kept portion.
    pub(crate) kept_token_ids: Vec<u32>,
    /// Text of the kept portion.
    pub(crate) kept_text: String,
    /// Token IDs of the bridge region (between keep and rollback).
    pub(crate) bridge_token_ids: Vec<u32>,
    /// Text of the bridge region, if any.
    pub(crate) bridge_text: Option<String>,
    /// KV cache position to roll back to for the next window.
    pub(crate) rollback_position: usize,
    /// Debug info about how the keep boundary was chosen.
    pub(crate) keep_boundary_debug: KeepBoundaryDebug,
}

/// A single word within a carried bridge, with its token range and timing.
#[derive(Clone)]
pub(crate) struct CarriedBridgeWord {
    /// Range of token indices this word spans within the bridge token IDs.
    pub(crate) token_range: std::ops::Range<usize>,
    /// End time in seconds (relative to the window start).
    pub(crate) end_secs: f64,
}

/// Bridge tokens and text carried from a previous window into the next decode.
#[derive(Clone)]
pub(crate) struct CarriedBridge {
    /// Token IDs of the bridge portion to replay.
    pub(crate) token_ids: Vec<u32>,
    /// Decoded text of the bridge tokens.
    pub(crate) text: String,
    /// Per-word breakdown with timing and token ranges.
    pub(crate) words: Vec<CarriedBridgeWord>,
}

/// One window's decode output in a sliding-window experiment.
pub(crate) struct SlidingWindowRun {
    /// The chunk decode result for this window.
    pub(crate) chunk_run: ChunkRun,
    /// Rollback decision made after this window (if not the last).
    pub(crate) rollback: Option<WindowRollbackDecision>,
    /// Bridge prefix replayed from the previous window, if any.
    pub(crate) replayed_prefix: Option<CarriedBridge>,
}

/// Result of a sliding-window timed-rollback experiment.
pub(crate) struct SlidingWindowTimedRollbackExperimentResult {
    /// Label identifying the sliding-window variant.
    pub(crate) mode_label: &'static str,
    /// Chunk duration in milliseconds.
    pub(crate) chunk_ms: usize,
    /// Rollback duration in milliseconds.
    pub(crate) rollback_ms: usize,
    /// Stride between windows in milliseconds.
    pub(crate) stride_ms: usize,
    /// Per-window decode results.
    pub(crate) window_runs: Vec<SlidingWindowRun>,
    /// Path to the generated HTML visualization.
    pub(crate) html_path: PathBuf,
    /// Path to the committed-words timeline HTML, if generated.
    pub(crate) committed_timeline_path: Option<PathBuf>,
    /// Whether the experiment was interrupted by Ctrl-C.
    pub(crate) interrupted_early: bool,
}

/// Debug info about a word that was considered as a keep boundary.
#[derive(Clone)]
pub(crate) struct BoundaryWordDebug {
    /// Index of this word in the window's word list.
    pub(crate) word_index: usize,
    /// The word text.
    pub(crate) text: String,
    /// Start time in seconds (relative to window start).
    pub(crate) start_secs: f64,
    /// End time in seconds (relative to window start).
    pub(crate) end_secs: f64,
}

/// Debug info about how the keep boundary was selected.
#[derive(Clone)]
pub(crate) struct KeepBoundaryDebug {
    /// Earliest candidate time that was considered for the boundary.
    pub(crate) earliest_candidate_secs: f64,
    /// Minimum keep duration enforced.
    pub(crate) min_keep_secs: f64,
    /// Whether the boundary was snapped to a word edge.
    pub(crate) snapped: bool,
    /// The word chosen as the boundary, if any.
    pub(crate) chosen_word: Option<BoundaryWordDebug>,
}

/// Reads a required environment variable as a `PathBuf`.
pub(crate) fn env_path(name: &str) -> Result<PathBuf> {
    let value = env::var(name).with_context(|| format!("{name} is not set"))?;
    Ok(PathBuf::from(value))
}

/// Returns the default WAV file path relative to the crate manifest directory.
pub(crate) fn default_wav_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(DEFAULT_WAV_RELATIVE_TO_CRATE)
}
