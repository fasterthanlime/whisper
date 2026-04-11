use std::env;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use bee_qwen3_asr::generate::DecodeStopReason;
use bee_qwen3_asr::load;

use crate::print::print_usage;
use crate::{
    DEFAULT_BRIDGE_MS, DEFAULT_CHUNK_MS, DEFAULT_KEEP_BOUNDARY_POLICY,
    DEFAULT_LANE_B_FIRST_CHUNK_MS, DEFAULT_LANGUAGE, DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MLX_CACHE_LIMIT_MB, DEFAULT_REPLAY_CHUNK_INDEX, DEFAULT_ROLLBACK_MS, DEFAULT_STRIDE_MS,
    DEFAULT_TRUNCATE_TOKENS, DEFAULT_WAV_RELATIVE_TO_CRATE, MAX_BRIDGE_WINDOWS,
};

/// CLI arguments parsed from `env::args()`.
pub(crate) struct Args {
    /// Path to the input WAV file.
    pub(crate) wav_path: PathBuf,
    /// Directory containing the ASR model weights and config.
    pub(crate) model_dir: PathBuf,
    /// Path to the tokenizer JSON file.
    pub(crate) tokenizer_path: PathBuf,
    /// Language hint passed to the ASR model prompt.
    pub(crate) language: String,
    /// Maximum number of tokens the model may generate per decode call.
    pub(crate) max_new_tokens: usize,
    /// Optional context string injected into the initial prompt.
    pub(crate) context: String,
    /// Which decode experiment to run.
    pub(crate) mode: Mode,
    /// Audio chunk duration in milliseconds.
    pub(crate) chunk_ms: usize,
    /// Bridge window duration in milliseconds for bridge-replay mode.
    pub(crate) bridge_ms: usize,
    /// Maximum number of bridge windows before forcing a full reset.
    pub(crate) max_bridge_windows: usize,
    /// Optional explicit stride between chunks in milliseconds.
    pub(crate) stride_ms: Option<usize>,
    /// Policy for snapping the keep boundary to a word edge.
    pub(crate) keep_boundary_policy: KeepBoundaryPolicy,
    /// Rollback duration in milliseconds for sliding-window modes.
    pub(crate) rollback_ms: usize,
    /// Policy for computing the rollback position.
    pub(crate) rollback_policy: RollbackPolicy,
    /// Chunk index at which truncate-replay begins replaying.
    pub(crate) replay_chunk_index: usize,
    /// Number of tokens to truncate when replaying from a rollback point.
    pub(crate) truncate_tokens: usize,
    /// First-chunk duration in milliseconds for lane B in dual-lane mode.
    pub(crate) lane_b_first_chunk_ms: usize,
    /// Optional MLX memory cache limit in megabytes.
    pub(crate) mlx_cache_limit_mb: Option<usize>,
}

impl Args {
    /// Parses CLI arguments from `env::args()` into an `Args` struct.
    pub(crate) fn parse() -> Result<Self> {
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

/// Which decode experiment to run.
#[derive(Clone, Copy)]
pub(crate) enum Mode {
    /// Single full-audio decode with an initial system prompt.
    Initial,
    /// Single full-audio decode with a follow-up (continuation) prompt.
    FollowupFresh,
    /// Decode with multiple system prompts and compare outputs.
    SystemCompare,
    /// Decode audio in fixed-size chunks with follow-up prompts.
    ChunkedFollowup,
    /// Re-run each chunk with the previous chunk's transcript as prefix context.
    PrefixRerun,
    /// Replay from a rollback point after truncating KV cache tokens.
    TruncateReplay,
    /// Sliding window with time-based rollback and KV cache reuse.
    SlidingWindowTimedRollback,
    /// Sliding window where each window fully replays all prior context.
    SlidingWindowFullReplay,
    /// Sliding window with bridge-region replay for KV cache continuity.
    SlidingWindowBridgeReplay,
    /// Two parallel decode lanes with different chunk boundaries for comparison.
    DualLaneFollowup,
    /// Merge chunk segments with rollback to test boundary robustness.
    ChunkSegmentMergeRollback,
    /// Sweep chunk boundary positions to find optimal merge points.
    ChunkSegmentMergeBoundarySweep,
}

impl Mode {
    /// Parses a CLI mode string into a `Mode` enum variant.
    pub(crate) fn parse(value: &str) -> Result<Self> {
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

/// How to compute the rollback position when replaying a chunk.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum RollbackPolicy {
    /// Roll back to the longest matching text suffix between chunks.
    TextSuffix,
    /// Roll back to the exact chunk segment boundary in the KV cache.
    ChunkSegment,
}

impl RollbackPolicy {
    /// Parses a CLI string into a `RollbackPolicy`.
    pub(crate) fn parse(value: &str) -> Result<Self> {
        match value {
            "text-suffix" => Ok(Self::TextSuffix),
            "chunk-segment" => Ok(Self::ChunkSegment),
            _ => bail!("unknown rollback policy: {value}"),
        }
    }

    /// Returns the CLI string representation.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::TextSuffix => "text-suffix",
            Self::ChunkSegment => "chunk-segment",
        }
    }
}

/// Policy for snapping the keep boundary to a word edge.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum KeepBoundaryPolicy {
    /// Keep boundary at the exact time-based position (no snapping).
    Fixed,
    /// Snap the keep boundary to the nearest word end.
    NearestWordEnd,
}

impl KeepBoundaryPolicy {
    /// Parses a CLI string into a `KeepBoundaryPolicy`.
    pub(crate) fn parse(value: &str) -> Result<Self> {
        match value {
            "fixed" => Ok(Self::Fixed),
            "nearest-word-end" => Ok(Self::NearestWordEnd),
            _ => bail!("unknown keep boundary policy: {value}"),
        }
    }

    /// Returns the CLI string representation.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::NearestWordEnd => "nearest-word-end",
        }
    }
}

/// Result of a single full-audio decode experiment.
pub(crate) struct ExperimentResult {
    /// Human-readable label describing this experiment.
    pub(crate) label: String,
    /// Number of prompt tokens fed to the model.
    pub(crate) prompt_tokens: usize,
    /// Number of tokens generated by the model.
    pub(crate) generated_tokens: usize,
    /// Decoded transcript text.
    pub(crate) transcript: String,
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

/// Result of decoding a single audio chunk.
pub(crate) struct ChunkResult {
    /// Human-readable label for this chunk.
    pub(crate) label: String,
    /// Number of prompt tokens fed to the model for this chunk.
    pub(crate) prompt_tokens: usize,
    /// Number of tokens generated for this chunk.
    pub(crate) generated_tokens: usize,
    /// Decoded transcript for this chunk.
    pub(crate) transcript: String,
    /// Number of audio samples in this chunk.
    pub(crate) sample_count: usize,
}

/// Result of a chunked follow-up decode experiment.
pub(crate) struct ChunkedExperimentResult {
    /// Chunk duration in milliseconds used for this experiment.
    pub(crate) chunk_ms: usize,
    /// Per-chunk decode results.
    pub(crate) chunk_results: Vec<ChunkResult>,
    /// Combined transcript from all chunks.
    pub(crate) combined_transcript: String,
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

impl ChunkResult {
    /// Converts a `ChunkRun` into a `ChunkResult`, dropping token-level detail.
    pub(crate) fn from_run(run: &ChunkRun) -> Self {
        Self {
            label: run.label.clone(),
            prompt_tokens: run.prompt_tokens,
            generated_tokens: run.generated_tokens,
            transcript: run.transcript.clone(),
            sample_count: run.sample_count,
        }
    }
}

/// Result of decoding a chunk with the previous chunk's transcript as prefix.
pub(crate) struct PrefixResult {
    /// Human-readable label for this prefix-rerun chunk.
    pub(crate) label: String,
    /// Number of prompt tokens (including prefix).
    pub(crate) prompt_tokens: usize,
    /// Number of tokens generated.
    pub(crate) generated_tokens: usize,
    /// Decoded transcript.
    pub(crate) transcript: String,
    /// Number of audio samples in this chunk.
    pub(crate) sample_count: usize,
}

/// Result of a prefix-rerun experiment across all chunks.
pub(crate) struct PrefixRerunExperimentResult {
    /// Chunk duration in milliseconds.
    pub(crate) chunk_ms: usize,
    /// Per-chunk results with prefix context.
    pub(crate) prefix_results: Vec<PrefixResult>,
}

/// A single entry in a chunk plan specifying sample boundaries.
pub(crate) struct ChunkPlanEntry {
    /// Zero-based index of this chunk.
    pub(crate) chunk_index: usize,
    /// Start sample index (inclusive) in the full audio.
    pub(crate) start_sample: usize,
    /// End sample index (exclusive) in the full audio.
    pub(crate) end_sample: usize,
}

/// Plan for replaying from a specific rollback point.
#[derive(Clone, Copy)]
pub(crate) struct ReplayPlan {
    /// Which rollback strategy to use.
    pub(crate) rollback_policy: RollbackPolicy,
    /// Chunk index after which to roll back.
    pub(crate) rollback_after_chunk_index: usize,
    /// KV cache position to truncate to.
    pub(crate) rollback_position: usize,
    /// Chunk index from which to start replaying.
    pub(crate) replay_from_chunk_index: usize,
}

/// Result of a truncate-replay experiment comparing baseline vs replayed chunks.
pub(crate) struct TruncateReplayExperimentResult {
    /// Chunk duration in milliseconds.
    pub(crate) chunk_ms: usize,
    /// Rollback policy used.
    pub(crate) rollback_policy: RollbackPolicy,
    /// Chunk index that was replayed.
    pub(crate) replay_chunk_index: usize,
    /// Chunk index from which replay started.
    pub(crate) replay_from_chunk_index: usize,
    /// KV cache position that was rolled back to.
    pub(crate) rollback_position: usize,
    /// Number of tokens requested to truncate.
    pub(crate) requested_truncate_tokens: usize,
    /// Number of tokens actually truncated.
    pub(crate) applied_truncate_tokens: usize,
    /// Baseline (no replay) chunk runs.
    pub(crate) baseline_runs: Vec<ChunkRun>,
    /// Chunk runs after replaying from the rollback point.
    pub(crate) replay_runs: Vec<ChunkRun>,
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

/// Result of a dual-lane follow-up experiment.
pub(crate) struct DualLaneFollowupExperimentResult {
    /// Chunk duration in milliseconds.
    pub(crate) chunk_ms: usize,
    /// First-chunk duration in milliseconds for lane B.
    pub(crate) lane_b_first_chunk_ms: usize,
    /// Chunk runs from lane A (standard chunking).
    pub(crate) lane_a_runs: Vec<ChunkRun>,
    /// Chunk runs from lane B (offset chunking).
    pub(crate) lane_b_runs: Vec<ChunkRun>,
}

/// Result of a chunk-segment merge rollback experiment.
pub(crate) struct ChunkSegmentMergeRollbackExperimentResult {
    /// Chunk duration in milliseconds.
    pub(crate) chunk_ms: usize,
    /// Baseline chunk runs (no merge).
    pub(crate) baseline_runs: Vec<ChunkRun>,
    /// Replayed first chunk after rollback.
    pub(crate) replay_chunk0: ChunkRun,
    /// Merged chunk that spans the boundary.
    pub(crate) merged_chunk: ChunkRun,
    /// Annotated baseline transcript with word timings.
    pub(crate) baseline_annotated: String,
    /// Annotated replay transcript with word timings.
    pub(crate) replay_annotated: String,
    /// Path to the generated HTML visualization.
    pub(crate) html_path: PathBuf,
}

/// Result of sweeping one boundary offset position.
pub(crate) struct BoundarySweepResult {
    /// Token offset from the exact chunk boundary.
    pub(crate) offset: isize,
    /// KV cache rollback position for this offset.
    pub(crate) rollback_position: usize,
    /// Merged chunk decoded at this boundary offset.
    pub(crate) merged_chunk: ChunkRun,
}

/// Result of a boundary sweep experiment across multiple offsets.
pub(crate) struct ChunkSegmentMergeBoundarySweepExperimentResult {
    /// Chunk duration in milliseconds.
    pub(crate) chunk_ms: usize,
    /// Exact chunk boundary position (in KV cache tokens).
    pub(crate) exact_boundary: usize,
    /// Baseline chunk runs for reference.
    pub(crate) baseline_runs: Vec<ChunkRun>,
    /// Results for each swept boundary offset.
    pub(crate) sweep_results: Vec<BoundarySweepResult>,
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
