use std::env;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use bee_qwen3_asr::config::AsrConfig;
use bee_qwen3_asr::generate::{ConfidenceMode, build_initial_prompt, prefill_and_decode};
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

    // `prefill()` injects audio features into token embeddings with `where`, so
    // cast audio features to the embedding dtype ahead of time.
    let probe_ids = Array::from_slice(&[0_i32], &[1, 1]);
    let embed_dtype = model.model.embed_tokens.forward(&probe_ids)?.dtype();
    let audio_features = audio_features.as_dtype(embed_dtype)?;

    let prompt_tokens = build_initial_prompt(n_audio_tokens, &args.language, "", &tokenizer);
    let mut cache = None;
    let (generated, _confidence, _next_position) = prefill_and_decode(
        &model,
        &prompt_tokens,
        &audio_features,
        &mut cache,
        0,
        args.max_new_tokens,
        ConfidenceMode::Streaming,
    )
    .context("running one-shot decode")?;

    let token_ids: Vec<u32> = generated
        .iter()
        .map(|&id| u32::try_from(id).context("generated negative token id"))
        .collect::<Result<_>>()?;
    let transcript = tokenizer
        .decode(&token_ids, true)
        .map_err(|e| anyhow::anyhow!("decoding transcript: {e}"))?;

    println!("wav: {}", args.wav_path.display());
    println!("model_dir: {}", args.model_dir.display());
    println!("tokenizer: {}", args.tokenizer_path.display());
    println!("language: {}", args.language);
    println!(
        "weights: loaded={} total_keys={} quantized_layers={} bits={} group_size={}",
        load_stats.loaded,
        load_stats.total_keys,
        load_stats.quantized_layers,
        load_stats.bits,
        load_stats.group_size
    );
    println!(
        "audio: samples={} mel_frames={} audio_tokens={} generated_tokens={}",
        samples.len(),
        n_frames,
        n_audio_tokens,
        generated.len()
    );
    println!();
    println!("{}", transcript.trim());

    Ok(())
}

struct Args {
    wav_path: PathBuf,
    model_dir: PathBuf,
    tokenizer_path: PathBuf,
    language: String,
    max_new_tokens: usize,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut positional = Vec::new();
        for arg in env::args().skip(1) {
            if arg == "-h" || arg == "--help" {
                print_usage();
                std::process::exit(0);
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
        })
    }
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
        "usage: bee-kv [wav-path] [language] [max-new-tokens]\n\
         defaults:\n\
           wav-path = {}\n\
           language = {DEFAULT_LANGUAGE}\n\
           max-new-tokens = {DEFAULT_MAX_NEW_TOKENS}\n\
         environment:\n\
           BEE_ASR_MODEL_DIR\n\
           BEE_TOKENIZER_PATH (optional; defaults to $BEE_ASR_MODEL_DIR/tokenizer.json)",
        default_wav_path().display()
    );
}
