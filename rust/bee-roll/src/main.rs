use std::env;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context as _;
use bee_qwen3_asr::config::AsrConfig;
use bee_qwen3_asr::load;
use bee_qwen3_asr::mlx_rs::module::ModuleParametersExt;
use bee_qwen3_asr::model::Qwen3ASRModel;
use bee_roll::{Cutting, Utterance};
use bee_transcribe::set_mlx_cache_limit;
use bee_zipa_mlx::audio::load_wav_mono_f32;
use tokenizers::Tokenizer;
use tracing_subscriber::EnvFilter;

const CHUNK_SAMPLES: usize = 3_200;
const MLX_CACHE_LIMIT_MB: usize = 200;

fn main() -> anyhow::Result<()> {
    init_tracing();

    let mut cutting = Cutting::Never;
    let mut wavs = Vec::new();

    for arg in env::args_os().skip(1) {
        if arg == "--auto" {
            cutting = Cutting::Auto;
            continue;
        }
        if arg == "--never" {
            cutting = Cutting::Never;
            continue;
        }
        wavs.push(PathBuf::from(arg));
    }

    if wavs.len() != 1 {
        anyhow::bail!("usage: cargo run -p bee-roll -- [--auto|--never] <wav>");
    }

    let model_dir = PathBuf::from(env::var("BEE_ASR_MODEL_DIR")?);
    let tokenizer_path = PathBuf::from(env::var("BEE_TOKENIZER_PATH")?);
    let g2p_model_dir = g2p_model_dir()?;
    let zipa_bundle_dir = zipa_bundle_dir()?;

    let previous_cache_bytes = set_mlx_cache_limit(MLX_CACHE_LIMIT_MB * 1024 * 1024)
        .map_err(|e| anyhow::anyhow!("setting MLX cache limit: {e}"))?;
    println!(
        "mlx cache limit: {MLX_CACHE_LIMIT_MB}MB (previous {:.1}MB)",
        previous_cache_bytes as f64 / (1024.0 * 1024.0)
    );

    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("loading tokenizer {}: {e}", tokenizer_path.display()))?;
    let asr_config = AsrConfig::from_file(&model_dir.join("config.json"))?;
    let mut model = Qwen3ASRModel::new(&asr_config.thinker_config)?;
    load::load_weights(&mut model, &model_dir)?;
    model.eval()?;
    let model = Arc::new(model);

    let wav_path = &wavs[0];
    let samples = load_wav_mono_f32(wav_path)
        .with_context(|| format!("loading wav {}", wav_path.display()))?;

    let mut utterance = Utterance::new(
        asr_config.thinker_config.text_config.num_hidden_layers,
        cutting,
    );
    utterance.attach_qwen_asr(
        Arc::clone(&model),
        &tokenizer_path,
        asr_config.thinker_config.audio_config.num_mel_bins,
        "en",
    );
    utterance.attach_phonetics(&g2p_model_dir, &tokenizer_path, &zipa_bundle_dir, "eng-us")?;

    println!(
        "cutting: {}",
        match cutting {
            Cutting::Never => "Never",
            Cutting::Auto => "Auto",
        }
    );
    println!("wav: {}", wav_path.display());
    println!("chunk_samples: {CHUNK_SAMPLES}");
    println!();

    for (feed_index, chunk) in samples.samples.chunks(CHUNK_SAMPLES).enumerate() {
        let output = utterance.feed(chunk.to_vec());
        let ids = output
            .tokens()
            .iter()
            .map(|token| token.timed_token().token().as_u32())
            .collect::<Vec<_>>();

        let transcript = tokenizer.decode(&ids, true).map_err(|e| {
            anyhow::anyhow!(
                "decoding transcript for {} feed {}: {e}",
                wav_path.display(),
                feed_index + 1
            )
        })?;

        let secs = ((feed_index + 1) * CHUNK_SAMPLES) as f64 / 16_000.0;
        println!("feed {:02} secs={secs:.3}", feed_index + 1);
        println!("text: {transcript}");
        println!();
    }

    Ok(())
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,tokenizers=warn"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .without_time()
        .try_init();
}

fn g2p_model_dir() -> anyhow::Result<PathBuf> {
    if let Ok(path) = env::var("BEE_G2P_CHARSIU_MODEL_DIR") {
        return Ok(PathBuf::from(path));
    }
    let fallback = PathBuf::from("/tmp/charsiu-g2p");
    if fallback.join("model.safetensors").exists() {
        return Ok(fallback);
    }
    anyhow::bail!("missing Charsiu model dir")
}

fn zipa_bundle_dir() -> anyhow::Result<PathBuf> {
    if let Ok(path) = env::var("BEE_ZIPA_BUNDLE_DIR") {
        return Ok(PathBuf::from(path));
    }
    let fallback = PathBuf::from(env::var("HOME")?).join("bearcove/zipa-mlx-hf");
    if fallback.join("config.styx").exists() {
        return Ok(fallback);
    }
    anyhow::bail!("missing ZIPA bundle dir")
}
