use std::env;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use bee_qwen3_asr::config::AsrConfig;
use bee_qwen3_asr::generate::{
    ConfidenceMode, build_followup_prompt, build_initial_prompt, prefill_and_decode,
};
use bee_qwen3_asr::load;
use bee_qwen3_asr::mel::{MelExtractor, load_audio};
use bee_qwen3_asr::mlx_rs::Array;
use bee_qwen3_asr::mlx_rs::module::{Module, ModuleParametersExt};
use bee_qwen3_asr::mlx_rs::ops;
use bee_qwen3_asr::model::Qwen3ASRModel;
use bee_qwen3_asr::tokenizers::Tokenizer;

const DEFAULT_LANGUAGE: &str = "English";
const DEFAULT_MAX_NEW_TOKENS: usize = 256;
const SAMPLE_RATE: u32 = 16_000;
const N_FFT: usize = 400;
const HOP_LENGTH: usize = 160;
const DEFAULT_WAV_RELATIVE_TO_CRATE: &str = "../../.artifacts/repros/frozen/EB54CF36.wav";
const DEFAULT_START_POSITION_FOR_FRESH_FOLLOWUP: usize = 0;
const DEFAULT_CHUNK_MS: usize = 2_000;
const DEFAULT_LANE_B_FIRST_CHUNK_MS: usize = 3_000;
const DEFAULT_REPLAY_CHUNK_INDEX: usize = 1;
const DEFAULT_TRUNCATE_TOKENS: usize = 4;

fn main() -> Result<()> {
    let args = Args::parse()?;

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
    let (generated, _confidence, _next_position) = prefill_and_decode(
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
        transcript: transcript.trim().to_string(),
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
    )?;

    Ok(ChunkSegmentMergeRollbackExperimentResult {
        chunk_ms,
        baseline_runs,
        replay_chunk0,
        merged_chunk,
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
) -> Result<ChunkRun> {
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

    let (label, prompt_tokens) = if chunk_index == 0 {
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

    let prompt_len = prompt_tokens.len();
    let (generated, _confidence, end_position) = prefill_and_decode(
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
        .map_err(|e| anyhow::anyhow!("decoding transcript: {e}"))?
        .trim()
        .to_string();

    Ok(ChunkRun {
        label,
        prompt_tokens: prompt_len,
        generated_tokens: generated.len(),
        transcript,
        sample_count: chunk_samples.len(),
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

fn combine_transcripts(chunks: &[ChunkRun]) -> String {
    let mut combined = String::new();
    for chunk in chunks {
        if chunk.transcript.is_empty() {
            continue;
        }
        if !combined.is_empty() {
            combined.push(' ');
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
    rollback_policy: RollbackPolicy,
    replay_chunk_index: usize,
    truncate_tokens: usize,
    lane_b_first_chunk_ms: usize,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut positional = Vec::new();
        let mut mode = Mode::Initial;
        let mut context = String::new();
        let mut chunk_ms = DEFAULT_CHUNK_MS;
        let mut rollback_policy = RollbackPolicy::TextSuffix;
        let mut replay_chunk_index = DEFAULT_REPLAY_CHUNK_INDEX;
        let mut truncate_tokens = DEFAULT_TRUNCATE_TOKENS;
        let mut lane_b_first_chunk_ms = DEFAULT_LANE_B_FIRST_CHUNK_MS;

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
            rollback_policy,
            replay_chunk_index,
            truncate_tokens,
            lane_b_first_chunk_ms,
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
    DualLaneFollowup,
    ChunkSegmentMergeRollback,
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
            "dual-lane-followup" => Ok(Self::DualLaneFollowup),
            "chunk-segment-merge-rollback" => Ok(Self::ChunkSegmentMergeRollback),
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
    transcript: String,
    sample_count: usize,
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
        "usage: bee-kv [--mode MODE] [--context TEXT] [--chunk-ms N] [wav-path] [language] [max-new-tokens]\n\
         defaults:\n\
           wav-path = {}\n\
           language = {DEFAULT_LANGUAGE}\n\
           max-new-tokens = {DEFAULT_MAX_NEW_TOKENS}\n\
           mode = initial\n\
           chunk-ms = {DEFAULT_CHUNK_MS}\n\
         modes:\n\
           initial\n\
           followup-fresh\n\
           system-compare\n\
           chunked-followup\n\
           prefix-rerun\n\
           truncate-replay\n\
           dual-lane-followup\n\
           chunk-segment-merge-rollback\n\
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
    println!();
}

fn clone_chunk_run(chunk: &ChunkRun) -> ChunkRun {
    ChunkRun {
        label: chunk.label.clone(),
        prompt_tokens: chunk.prompt_tokens,
        generated_tokens: chunk.generated_tokens,
        transcript: chunk.transcript.clone(),
        sample_count: chunk.sample_count,
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
        "samples={} audio={}..{}ms prompt_tokens={} generated_tokens={} start_position={} end_position={}",
        chunk.sample_count,
        (chunk.start_sample * 1000) / SAMPLE_RATE as usize,
        (chunk.end_sample * 1000) / SAMPLE_RATE as usize,
        chunk.prompt_tokens,
        chunk.generated_tokens,
        chunk.start_position,
        chunk.end_position
    );
    println!("{}", chunk.transcript);
    println!();
}
