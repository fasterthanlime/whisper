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

pub(crate) struct Args {
    pub(crate) wav_path: PathBuf,
    pub(crate) model_dir: PathBuf,
    pub(crate) tokenizer_path: PathBuf,
    pub(crate) language: String,
    pub(crate) max_new_tokens: usize,
    pub(crate) context: String,
    pub(crate) mode: Mode,
    pub(crate) chunk_ms: usize,
    pub(crate) bridge_ms: usize,
    pub(crate) max_bridge_windows: usize,
    pub(crate) stride_ms: Option<usize>,
    pub(crate) keep_boundary_policy: KeepBoundaryPolicy,
    pub(crate) rollback_ms: usize,
    pub(crate) rollback_policy: RollbackPolicy,
    pub(crate) replay_chunk_index: usize,
    pub(crate) truncate_tokens: usize,
    pub(crate) lane_b_first_chunk_ms: usize,
    pub(crate) mlx_cache_limit_mb: Option<usize>,
}

impl Args {
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

#[derive(Clone, Copy)]
pub(crate) enum Mode {
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

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum RollbackPolicy {
    TextSuffix,
    ChunkSegment,
}

impl RollbackPolicy {
    pub(crate) fn parse(value: &str) -> Result<Self> {
        match value {
            "text-suffix" => Ok(Self::TextSuffix),
            "chunk-segment" => Ok(Self::ChunkSegment),
            _ => bail!("unknown rollback policy: {value}"),
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::TextSuffix => "text-suffix",
            Self::ChunkSegment => "chunk-segment",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum KeepBoundaryPolicy {
    Fixed,
    NearestWordEnd,
}

impl KeepBoundaryPolicy {
    pub(crate) fn parse(value: &str) -> Result<Self> {
        match value {
            "fixed" => Ok(Self::Fixed),
            "nearest-word-end" => Ok(Self::NearestWordEnd),
            _ => bail!("unknown keep boundary policy: {value}"),
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::NearestWordEnd => "nearest-word-end",
        }
    }
}

pub(crate) struct ExperimentResult {
    pub(crate) label: String,
    pub(crate) prompt_tokens: usize,
    pub(crate) generated_tokens: usize,
    pub(crate) transcript: String,
}

pub(crate) struct RunSummary<'a> {
    pub(crate) wav_path: &'a Path,
    pub(crate) model_dir: &'a Path,
    pub(crate) tokenizer_path: &'a Path,
    pub(crate) language: &'a str,
    pub(crate) load_stats: &'a load::LoadStats,
    pub(crate) sample_count: usize,
    pub(crate) mel_frames: usize,
    pub(crate) audio_tokens: usize,
}

pub(crate) struct ChunkResult {
    pub(crate) label: String,
    pub(crate) prompt_tokens: usize,
    pub(crate) generated_tokens: usize,
    pub(crate) transcript: String,
    pub(crate) sample_count: usize,
}

pub(crate) struct ChunkedExperimentResult {
    pub(crate) chunk_ms: usize,
    pub(crate) chunk_results: Vec<ChunkResult>,
    pub(crate) combined_transcript: String,
}

pub(crate) struct ChunkRun {
    pub(crate) label: String,
    pub(crate) prompt_tokens: usize,
    pub(crate) generated_tokens: usize,
    pub(crate) generated_token_ids: Vec<u32>,
    pub(crate) transcript: String,
    pub(crate) sample_count: usize,
    pub(crate) decode_ms: f64,
    pub(crate) stop_reason: DecodeStopReason,
    pub(crate) start_position: usize,
    pub(crate) end_position: usize,
    pub(crate) start_sample: usize,
    pub(crate) end_sample: usize,
}

impl ChunkResult {
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

pub(crate) struct PrefixResult {
    pub(crate) label: String,
    pub(crate) prompt_tokens: usize,
    pub(crate) generated_tokens: usize,
    pub(crate) transcript: String,
    pub(crate) sample_count: usize,
}

pub(crate) struct PrefixRerunExperimentResult {
    pub(crate) chunk_ms: usize,
    pub(crate) prefix_results: Vec<PrefixResult>,
}

pub(crate) struct ChunkPlanEntry {
    pub(crate) chunk_index: usize,
    pub(crate) start_sample: usize,
    pub(crate) end_sample: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct ReplayPlan {
    pub(crate) rollback_policy: RollbackPolicy,
    pub(crate) rollback_after_chunk_index: usize,
    pub(crate) rollback_position: usize,
    pub(crate) replay_from_chunk_index: usize,
}

pub(crate) struct TruncateReplayExperimentResult {
    pub(crate) chunk_ms: usize,
    pub(crate) rollback_policy: RollbackPolicy,
    pub(crate) replay_chunk_index: usize,
    pub(crate) replay_from_chunk_index: usize,
    pub(crate) rollback_position: usize,
    pub(crate) requested_truncate_tokens: usize,
    pub(crate) applied_truncate_tokens: usize,
    pub(crate) baseline_runs: Vec<ChunkRun>,
    pub(crate) replay_runs: Vec<ChunkRun>,
}

pub(crate) struct WindowRollbackDecision {
    pub(crate) keep_boundary_policy: KeepBoundaryPolicy,
    pub(crate) target_keep_until_secs: f64,
    pub(crate) keep_until_secs: Option<f64>,
    pub(crate) replay_until_secs: Option<f64>,
    pub(crate) kept_word_count: usize,
    pub(crate) kept_token_count: usize,
    pub(crate) kept_token_ids: Vec<u32>,
    pub(crate) kept_text: String,
    pub(crate) bridge_token_ids: Vec<u32>,
    pub(crate) bridge_text: Option<String>,
    pub(crate) rollback_position: usize,
    pub(crate) keep_boundary_debug: KeepBoundaryDebug,
}

#[derive(Clone)]
pub(crate) struct CarriedBridgeWord {
    pub(crate) text: String,
    pub(crate) token_range: std::ops::Range<usize>,
    pub(crate) start_secs: f64,
    pub(crate) end_secs: f64,
}

#[derive(Clone)]
pub(crate) struct CarriedBridge {
    pub(crate) token_ids: Vec<u32>,
    pub(crate) text: String,
    pub(crate) words: Vec<CarriedBridgeWord>,
}

pub(crate) struct SlidingWindowRun {
    pub(crate) chunk_run: ChunkRun,
    pub(crate) rollback: Option<WindowRollbackDecision>,
    pub(crate) replayed_prefix: Option<CarriedBridge>,
}

pub(crate) struct SlidingWindowTimedRollbackExperimentResult {
    pub(crate) mode_label: &'static str,
    pub(crate) chunk_ms: usize,
    pub(crate) rollback_ms: usize,
    pub(crate) stride_ms: usize,
    pub(crate) window_runs: Vec<SlidingWindowRun>,
    pub(crate) html_path: PathBuf,
    pub(crate) committed_timeline_path: Option<PathBuf>,
    pub(crate) interrupted_early: bool,
}

#[derive(Clone)]
pub(crate) struct BoundaryWordDebug {
    pub(crate) word_index: usize,
    pub(crate) text: String,
    pub(crate) start_secs: f64,
    pub(crate) end_secs: f64,
}

#[derive(Clone)]
pub(crate) struct KeepBoundaryDebug {
    pub(crate) earliest_candidate_secs: f64,
    pub(crate) min_keep_secs: f64,
    pub(crate) snapped: bool,
    pub(crate) chosen_word: Option<BoundaryWordDebug>,
}

pub(crate) struct DualLaneFollowupExperimentResult {
    pub(crate) chunk_ms: usize,
    pub(crate) lane_b_first_chunk_ms: usize,
    pub(crate) lane_a_runs: Vec<ChunkRun>,
    pub(crate) lane_b_runs: Vec<ChunkRun>,
}

pub(crate) struct ChunkSegmentMergeRollbackExperimentResult {
    pub(crate) chunk_ms: usize,
    pub(crate) baseline_runs: Vec<ChunkRun>,
    pub(crate) replay_chunk0: ChunkRun,
    pub(crate) merged_chunk: ChunkRun,
    pub(crate) baseline_annotated: String,
    pub(crate) replay_annotated: String,
    pub(crate) html_path: PathBuf,
}

pub(crate) struct BoundarySweepResult {
    pub(crate) offset: isize,
    pub(crate) rollback_position: usize,
    pub(crate) merged_chunk: ChunkRun,
}

pub(crate) struct ChunkSegmentMergeBoundarySweepExperimentResult {
    pub(crate) chunk_ms: usize,
    pub(crate) exact_boundary: usize,
    pub(crate) baseline_runs: Vec<ChunkRun>,
    pub(crate) sweep_results: Vec<BoundarySweepResult>,
}

pub(crate) fn env_path(name: &str) -> Result<PathBuf> {
    let value = env::var(name).with_context(|| format!("{name} is not set"))?;
    Ok(PathBuf::from(value))
}

pub(crate) fn default_wav_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(DEFAULT_WAV_RELATIVE_TO_CRATE)
}
