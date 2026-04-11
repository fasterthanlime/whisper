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
    token_range: std::ops::Range<usize>,
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
    word_index: usize,
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
