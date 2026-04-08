use std::path::{Path, PathBuf};
use std::time::Instant;

use bee_transcribe::{EngineConfig, SessionOptions, Update};
use tracing_subscriber::EnvFilter;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let args: Vec<String> = std::env::args().collect();
    let use_v2 = args.iter().any(|a| a == "--v2");
    let positional: Vec<&str> = args[1..]
        .iter()
        .filter(|a| !a.starts_with("--"))
        .map(|s| s.as_str())
        .collect();
    if positional.is_empty() {
        eprintln!("Usage: transcribe [--v2] <audio.wav>");
        std::process::exit(1);
    }
    let audio_path = positional[0];

    let model_dir = std::env::var("BEE_ASR_MODEL_DIR")
        .map_err(|_| anyhow::anyhow!("BEE_ASR_MODEL_DIR not set"))?;
    let tokenizer_dir = std::env::var("BEE_TOKENIZER_DIR").unwrap_or_else(|_| model_dir.clone());
    let aligner_dir =
        std::env::var("BEE_ALIGNER_DIR").map_err(|_| anyhow::anyhow!("BEE_ALIGNER_DIR not set"))?;
    let vad_dir =
        std::env::var("BEE_VAD_DIR").map_err(|_| anyhow::anyhow!("BEE_VAD_DIR not set"))?;

    // Correction engine: look in group container (same as install-bee.sh)
    let group_container: PathBuf = dirs::home_dir()
        .unwrap()
        .join("Library/Group Containers/B2N6FSRTPV.group.fasterthanlime.bee");
    let correction_dir_path = group_container.join("phonetic-seed");
    let correction_dir: Option<&Path> = if correction_dir_path.exists() {
        println!("Correction dataset: {}", correction_dir_path.display());
        Some(&correction_dir_path)
    } else {
        println!(
            "Correction dataset not found at {}",
            correction_dir_path.display()
        );
        None
    };
    let correction_events_path = correction_dir.map(|d| d.join("events.jsonl"));

    // Load engine
    let t0 = Instant::now();
    let engine = bee_transcribe::Engine::load(&EngineConfig {
        model_dir: Path::new(&model_dir),
        tokenizer_dir: Path::new(&tokenizer_dir),
        aligner_dir: Path::new(&aligner_dir),
        silero_dir: Path::new(&vad_dir),
        correction_dir,
        correction_events_path,
    })?;
    println!("Engine loaded in {:.0}ms", t0.elapsed().as_millis());

    // Load audio
    let t0 = Instant::now();
    let samples = bee_qwen3_asr::mel::load_audio_wav(audio_path, 16000)?;
    let duration = samples.len() as f64 / 16000.0;
    println!(
        "Audio: {:.1}s ({} samples) loaded in {:.0}ms",
        duration,
        samples.len(),
        t0.elapsed().as_millis()
    );

    // Create session with env var overrides
    let mut options = SessionOptions::default();
    if let Ok(v) = std::env::var("BEE_CHUNK_DURATION") {
        options.chunk_duration = v.parse().unwrap();
    }
    if let Ok(v) = std::env::var("BEE_VAD_THRESHOLD") {
        options.vad_threshold = v.parse().unwrap();
    }
    if let Ok(v) = std::env::var("BEE_ROLLBACK_TOKENS") {
        options.rollback_tokens = v.parse().unwrap();
    }
    if let Ok(v) = std::env::var("BEE_COMMIT_TOKENS") {
        options.commit_token_count = v.parse().unwrap();
    }
    if let Ok(v) = std::env::var("BEE_MAX_TOKENS_STREAMING") {
        options.max_tokens_streaming = v.parse().unwrap();
    }
    if let Ok(v) = std::env::var("BEE_MAX_TOKENS_FINAL") {
        options.max_tokens_final = v.parse().unwrap();
    }
    let chunk_samples = (options.chunk_duration * 16000.0) as usize;

    println!(
        "\n--- Streaming{} (chunk={:.0}ms) ---\n",
        if use_v2 { " [v2]" } else { "" },
        chunk_samples as f64 / 16.0
    );

    if use_v2 {
        run_v2(&engine, options, &samples, chunk_samples)
    } else {
        run_v1(&engine, options, &samples, chunk_samples)
    }
}

fn run_v1(
    engine: &bee_transcribe::Engine,
    options: SessionOptions,
    samples: &[f32],
    chunk_samples: usize,
) -> anyhow::Result<()> {
    let mut session = engine.session(options)?;

    let t_total = Instant::now();
    let mut chunk_idx = 0;
    let mut offset = 0;
    let mut last_text = String::new();

    while offset < samples.len() {
        let end = (offset + chunk_samples).min(samples.len());
        let chunk = &samples[offset..end];
        offset = end;
        chunk_idx += 1;

        let t0 = Instant::now();
        let result = session.feed(chunk)?;
        let ms = t0.elapsed().as_millis();

        match result {
            Some(update) => {
                if update.text != last_text {
                    print_update(chunk_idx, ms, &update);
                    last_text = update.text.clone();
                } else {
                    println!("  chunk {chunk_idx}: {ms:.0}ms (unchanged)");
                }
            }
            None => {
                println!("  chunk {chunk_idx}: {ms:.0}ms (silence/buffering)");
            }
        }
    }

    let t0 = Instant::now();
    let result = session.finish()?;
    let final_update = result.update;
    print_final(&final_update, t0, t_total);
    Ok(())
}

fn run_v2(
    engine: &bee_transcribe::Engine,
    options: SessionOptions,
    samples: &[f32],
    chunk_samples: usize,
) -> anyhow::Result<()> {
    let mut session = engine.session_v2(options)?;

    let t_total = Instant::now();
    let mut chunk_idx = 0;
    let mut offset = 0;
    let mut last_text = String::new();

    while offset < samples.len() {
        let end = (offset + chunk_samples).min(samples.len());
        let chunk = &samples[offset..end];
        offset = end;
        chunk_idx += 1;

        let t0 = Instant::now();
        let result = session.feed(chunk)?;
        let ms = t0.elapsed().as_millis();

        match result {
            Some(update) => {
                if update.text != last_text {
                    print_update(chunk_idx, ms, &update);
                    last_text = update.text.clone();
                } else {
                    println!("  chunk {chunk_idx}: {ms:.0}ms (unchanged)");
                }
            }
            None => {
                println!("  chunk {chunk_idx}: {ms:.0}ms (silence/buffering)");
            }
        }
    }

    let t0 = Instant::now();
    let result = session.finish()?;
    print_final(&result.update, t0, t_total);
    Ok(())
}

fn print_final(update: &Update, t0: Instant, t_total: Instant) {
    let finish_ms = t0.elapsed().as_millis();
    println!(
        "\n--- Final ({finish_ms:.0}ms, total {:.0}ms) ---",
        t_total.elapsed().as_millis()
    );
    println!("  text: {:?}", update.text);

    if !update.alignments.is_empty() {
        println!("\nAlignments:");
        for w in &update.alignments {
            println!("  [{:.3}s - {:.3}s] {}", w.start, w.end, w.word);
        }
    }
}

fn print_update(chunk: usize, ms: u128, update: &Update) {
    println!("  chunk {chunk}: {ms:.0}ms | {}", update.text);
}
