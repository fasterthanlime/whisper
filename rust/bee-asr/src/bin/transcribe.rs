use std::path::Path;
use std::time::Instant;

use mlx_rs::module::ModuleParametersExt;
use mlx_rs::ops;
use mlx_rs::Array;

use bee_asr::config::AsrConfig;
use bee_asr::generate;
use bee_asr::load;
use bee_asr::mel::{load_audio_wav, MelExtractor};
use bee_asr::model::{Qwen3ASRModel, AUDIO_END_TOKEN_ID, AUDIO_PAD_TOKEN_ID, AUDIO_START_TOKEN_ID};
use bee_asr::streaming::{self, StreamingMode, StreamingOptions, StreamingState};

// Chat template token IDs
const TOK_IM_START: i32 = 151644;
const TOK_IM_END: i32 = 151645;
const TOK_SYSTEM: i32 = 8948;
const TOK_USER: i32 = 872;
const TOK_ASSISTANT: i32 = 77091;
const TOK_NEWLINE: i32 = 198;

fn find_tokenizer(model_dir: &Path) -> Option<tokenizers::Tokenizer> {
    let paths = [
        model_dir.join("tokenizer.json"),
        dirs::home_dir()?.join("Library/Caches/qwen3-asr/Qwen--Qwen3-ASR-1.7B/tokenizer.json"),
        dirs::home_dir()?.join("Library/Caches/qwen3-asr/Qwen--Qwen3-ASR-0.6B/tokenizer.json"),
    ];
    for p in &paths {
        if p.exists() {
            if let Ok(t) = tokenizers::Tokenizer::from_file(p) {
                return Some(t);
            }
        }
    }
    None
}

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    let align_mode = args.iter().any(|a| a == "--align");
    let streaming_mode = args.iter().find_map(|a| {
        if a == "--streaming" || a == "--streaming=accumulate" {
            Some(StreamingMode::Accumulate)
        } else if a == "--streaming=overlap" {
            Some(StreamingMode::Overlap)
        } else if a == "--streaming=rotate" {
            Some(StreamingMode::Rotate)
        } else if a == "--streaming=rotate-cached" {
            Some(StreamingMode::RotateCached)
        } else {
            None
        }
    });
    let non_flag_args: Vec<&String> = args[1..].iter().filter(|a| !a.starts_with("--")).collect();

    if non_flag_args.len() < 2 {
        eprintln!(
            "Usage: transcribe [--streaming[=accumulate|overlap|rotate]] <model_dir> <audio.wav>"
        );
        std::process::exit(1);
    }

    let model_dir = Path::new(non_flag_args[0]);
    let audio_path = non_flag_args[1];

    // 1. Load config
    let t0 = Instant::now();
    let config_path = model_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)?;
    let config: AsrConfig = serde_json::from_str(&config_str)?;
    let thinker = &config.thinker_config;
    println!("Config loaded in {:.0}ms", t0.elapsed().as_millis());

    // 2. Create model
    let t0 = Instant::now();
    let mut model = Qwen3ASRModel::new(thinker)?;
    println!("Model created in {:.0}ms", t0.elapsed().as_millis());

    // 3. Load weights
    let t0 = Instant::now();
    let stats = load::load_weights(&mut model, model_dir)?;
    model.eval()?;
    println!(
        "Weights loaded in {:.0}ms: {}/{} keys, {} quantized layers ({}bit)",
        t0.elapsed().as_millis(),
        stats.loaded,
        stats.total_keys,
        stats.quantized_layers,
        stats.bits,
    );

    // 4. Load audio
    let t0 = Instant::now();
    let samples = load_audio_wav(audio_path, 16000)?;
    println!(
        "Audio: {} samples ({:.1}s) in {:.0}ms",
        samples.len(),
        samples.len() as f64 / 16000.0,
        t0.elapsed().as_millis()
    );

    // Report memory after model load
    {
        let (active, peak, cache) = bee_asr::streaming::mlx_memory_stats();
        println!(
            "Memory after load: active={:.1}MB peak={:.1}MB cache={:.1}MB",
            active as f64 / 1e6,
            peak as f64 / 1e6,
            cache as f64 / 1e6,
        );
    }

    let tokenizer =
        find_tokenizer(model_dir).ok_or_else(|| anyhow::anyhow!("no tokenizer.json found"))?;

    if align_mode {
        run_align(&samples, &tokenizer, model_dir)?;
    } else if let Some(mode) = streaming_mode {
        run_streaming(&mut model, &samples, tokenizer, mode)?;
    } else {
        run_batch(&mut model, &samples, &tokenizer)?;
    }

    Ok(())
}

fn run_streaming(
    model: &mut Qwen3ASRModel,
    samples: &[f32],
    tokenizer: tokenizers::Tokenizer,
    mode: StreamingMode,
) -> anyhow::Result<()> {
    let mut opts = StreamingOptions::default().with_mode(mode);
    if let Ok(v) = std::env::var("COMMIT_TOKENS") {
        opts.commit_token_count = v.parse().unwrap();
    }
    if let Ok(v) = std::env::var("COMMIT_STABLE") {
        opts.commit_after_stable = v.parse().unwrap();
    }
    if let Ok(v) = std::env::var("CHUNK_SEC") {
        opts.chunk_size_sec = v.parse().unwrap();
    }
    let chunk_samples = (opts.chunk_size_sec * 16000.0) as usize;

    // Load forced aligner for rotate mode
    let aligner = if mode == StreamingMode::Rotate || mode == StreamingMode::RotateCached {
        let aligner_dir = dirs::home_dir()
            .unwrap()
            .join("Library/Caches/qwen3-asr/Qwen--Qwen3-ForcedAligner-0.6B");
        if aligner_dir.exists() {
            println!("Loading forced aligner for rotate mode...");
            let t0 = Instant::now();
            let a = bee_asr::forced_aligner::ForcedAligner::load(&aligner_dir, tokenizer.clone())?;
            println!("Aligner loaded in {:.0}ms", t0.elapsed().as_millis());
            Some(a)
        } else {
            println!("Warning: forced aligner not found, using proportional estimate");
            None
        }
    } else {
        None
    };

    let mut state = StreamingState::new(opts, tokenizer, aligner);

    println!(
        "\n--- Streaming mode={:?} (chunk={}s) ---",
        mode, state.options.chunk_size_sec
    );

    let t_total = Instant::now();
    let mut chunk_idx = 0;

    // Feed audio in chunk-sized pieces
    let mut offset = 0;
    while offset < samples.len() {
        let end = (offset + chunk_samples).min(samples.len());
        let chunk = &samples[offset..end];
        offset = end;

        let t0 = Instant::now();
        let result = streaming::feed_audio(model, &mut state, chunk)?;
        let ms = t0.elapsed().as_millis();

        chunk_idx += 1;
        if let Some(text) = result {
            println!("  chunk {}: {:.0}ms — {}", chunk_idx, ms, text);
        } else {
            println!("  chunk {}: {:.0}ms — (buffering)", chunk_idx, ms);
        }
    }

    // Finish
    let t0 = Instant::now();
    let final_text = streaming::finish_streaming(model, &mut state)?;
    let finish_ms = t0.elapsed().as_millis();
    let total_ms = t_total.elapsed().as_millis();

    println!("\nFinish: {:.0}ms", finish_ms);
    println!("Total streaming: {:.0}ms", total_ms);
    println!("\nTranscription: {}", final_text);

    Ok(())
}

fn run_align(
    samples: &[f32],
    tokenizer: &tokenizers::Tokenizer,
    _asr_model_dir: &Path,
) -> anyhow::Result<()> {
    // For alignment, we need the aligner model, not the ASR model
    let aligner_dir = dirs::home_dir()
        .unwrap()
        .join("Library/Caches/qwen3-asr/Qwen--Qwen3-ForcedAligner-0.6B");
    if !aligner_dir.exists() {
        anyhow::bail!("Aligner model not found at {}", aligner_dir.display());
    }

    println!("\nLoading forced aligner...");
    let t0 = Instant::now();
    let mut aligner =
        bee_asr::forced_aligner::ForcedAligner::load(&aligner_dir, tokenizer.clone())?;
    println!("Aligner loaded in {:.0}ms", t0.elapsed().as_millis());

    // Use a known transcription for testing
    let text = "The quick brown fox jumps over the lazy dog.";
    println!("Aligning: {:?}", text);

    let t0 = Instant::now();
    let items = aligner.align(samples, text)?;
    println!("Aligned in {:.0}ms\n", t0.elapsed().as_millis());

    for item in &items {
        println!(
            "  [{:.3}s - {:.3}s] {}",
            item.start_time, item.end_time, item.word
        );
    }

    Ok(())
}

fn run_batch(
    mut model: &mut Qwen3ASRModel,
    samples: &[f32],
    tokenizer: &tokenizers::Tokenizer,
) -> anyhow::Result<()> {
    let mel_extractor = MelExtractor::new(400, 160, 128, 16000);
    let (mel_data, n_mels, n_frames) = mel_extractor.extract(samples)?;
    let mel = Array::from_slice(&mel_data, &[n_mels as i32, n_frames as i32]);

    let t0 = Instant::now();
    let audio_features = model.encode_audio(&mel)?;
    audio_features.eval()?;
    let n_audio_tokens = audio_features.shape()[0] as usize;
    println!(
        "Encoded: {} audio tokens in {:.0}ms",
        n_audio_tokens,
        t0.elapsed().as_millis()
    );

    let audio_features = mlx_rs::ops::expand_dims(&audio_features, 0)?;

    let mut prompt_tokens: Vec<i32> = vec![
        TOK_IM_START,
        TOK_SYSTEM,
        TOK_NEWLINE,
        TOK_IM_END,
        TOK_NEWLINE,
        TOK_IM_START,
        TOK_USER,
        TOK_NEWLINE,
        AUDIO_START_TOKEN_ID,
    ];
    prompt_tokens.extend(std::iter::repeat_n(AUDIO_PAD_TOKEN_ID, n_audio_tokens));
    prompt_tokens.extend_from_slice(&[
        AUDIO_END_TOKEN_ID,
        TOK_IM_END,
        TOK_NEWLINE,
        TOK_IM_START,
        TOK_ASSISTANT,
        TOK_NEWLINE,
    ]);

    let seq_len = prompt_tokens.len();
    let input_ids = Array::from_slice(&prompt_tokens, &[1, seq_len as i32]);
    let positions: Vec<i32> = (0..seq_len as i32).collect();
    let pos_arr = Array::from_slice(&positions, &[1, 1, seq_len as i32]);
    let position_ids = ops::broadcast_to(&pos_arr, &[1, 3, seq_len as i32])?;

    for run in 0..3 {
        let t0 = Instant::now();
        let af = model.encode_audio(&mel)?;
        af.eval()?;
        let enc_ms = t0.elapsed().as_millis();
        let af = mlx_rs::ops::expand_dims(&af, 0)?;

        let output_tokens = generate::generate(&mut model, &input_ids, &af, &position_ids, 512)?;
        let total_ms = t0.elapsed().as_millis();
        let gen_ms = total_ms - enc_ms;
        println!(
            "Run {}: encode {:.0}ms + generate {} tokens in {:.0}ms ({:.1} tok/s) = {:.0}ms total",
            run + 1,
            enc_ms,
            output_tokens.len(),
            gen_ms,
            output_tokens.len() as f64 / (gen_ms as f64 / 1000.0),
            total_ms,
        );
    }

    let output_tokens =
        generate::generate(&mut model, &input_ids, &audio_features, &position_ids, 512)?;
    let ids: Vec<u32> = output_tokens.iter().map(|&t| t as u32).collect();
    let text = tokenizer
        .decode(&ids, true)
        .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
    println!("\nTranscription: {}", text);

    Ok(())
}
