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

    let mut cache = None;
    let mut start_position = 0usize;
    let mut chunk_results = Vec::new();
    let mut combined_transcript = String::new();

    for (chunk_index, chunk_samples) in samples.chunks(chunk_size_samples).enumerate() {
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

        let (generated, _confidence, next_position) = prefill_and_decode(
            model,
            &prompt_tokens,
            &audio_features,
            &mut cache,
            start_position,
            max_new_tokens,
            ConfidenceMode::Streaming,
        )
        .with_context(|| format!("decoding {label}"))?;
        start_position = next_position;

        let token_ids: Vec<u32> = generated
            .iter()
            .map(|&id| u32::try_from(id).context("generated negative token id"))
            .collect::<Result<_>>()?;
        let transcript = tokenizer
            .decode(&token_ids, true)
            .map_err(|e| anyhow::anyhow!("decoding transcript: {e}"))?
            .trim()
            .to_string();

        if !transcript.is_empty() {
            if !combined_transcript.is_empty() {
                combined_transcript.push(' ');
            }
            combined_transcript.push_str(&transcript);
        }

        chunk_results.push(ChunkResult {
            label,
            prompt_tokens: prompt_tokens.len(),
            generated_tokens: generated.len(),
            transcript,
            sample_count: chunk_samples.len(),
        });
    }

    Ok(ChunkedExperimentResult {
        chunk_ms,
        chunk_results,
        combined_transcript,
    })
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
}

impl Args {
    fn parse() -> Result<Self> {
        let mut positional = Vec::new();
        let mut mode = Mode::Initial;
        let mut context = String::new();
        let mut chunk_ms = DEFAULT_CHUNK_MS;

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
        })
    }
}

#[derive(Clone, Copy)]
enum Mode {
    Initial,
    FollowupFresh,
    SystemCompare,
    ChunkedFollowup,
}

impl Mode {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "initial" => Ok(Self::Initial),
            "followup-fresh" => Ok(Self::FollowupFresh),
            "system-compare" => Ok(Self::SystemCompare),
            "chunked-followup" => Ok(Self::ChunkedFollowup),
            _ => bail!("unknown mode: {value}"),
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
