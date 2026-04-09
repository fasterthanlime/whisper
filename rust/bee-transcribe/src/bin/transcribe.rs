use std::path::{Path, PathBuf};
use std::time::Instant;

use bee_qwen3_asr::generate::TOP_K;
use bee_transcribe::text_buffer::TokenEntry;
use bee_transcribe::{EngineConfig, SessionOptions, SessionSnapshot};
use tokenizers::Tokenizer;
use tracing_subscriber::EnvFilter;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let args: Vec<String> = std::env::args().collect();
    let show_alts = args.iter().any(|a| a == "--alternatives" || a == "--alts");
    let positional: Vec<&str> = args[1..]
        .iter()
        .filter(|a| !a.starts_with("--"))
        .map(|s| s.as_str())
        .collect();
    if positional.is_empty() {
        eprintln!("Usage: transcribe [--alternatives] <audio.wav>");
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
    let disable_correction = std::env::var("BEE_DISABLE_CORRECTION")
        .map(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false);
    let group_container: PathBuf = dirs::home_dir()
        .unwrap()
        .join("Library/Group Containers/B2N6FSRTPV.group.fasterthanlime.bee");
    let correction_dir_path = group_container.join("phonetic-seed");
    let correction_dir: Option<&Path> = if disable_correction {
        println!("Correction disabled via BEE_DISABLE_CORRECTION");
        None
    } else if correction_dir_path.exists() {
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
        "\n--- Streaming (chunk={:.0}ms) ---\n",
        chunk_samples as f64 / 16.0
    );

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
            Some(snapshot) => {
                if snapshot.full_text != last_text {
                    print_update(chunk_idx, ms, &snapshot);
                    last_text = snapshot.full_text.clone();
                } else {
                    println!("  chunk {chunk_idx}: {ms:.0}ms (unchanged)");
                }
                if show_alts {
                    print_alternatives(session.tokenizer(), session.pending_entries());
                }
            }
            None => {
                println!("  chunk {chunk_idx}: {ms:.0}ms (silence/buffering)");
            }
        }
    }

    let t0 = Instant::now();
    let result = session.finish()?;
    let finish_ms = t0.elapsed().as_millis();
    println!(
        "\n--- Final ({finish_ms:.0}ms, total {:.0}ms) ---",
        t_total.elapsed().as_millis()
    );
    println!("  text: {:?}", result.snapshot.full_text);

    if !result.snapshot.committed_words.is_empty() {
        println!("\nAlignments:");
        for w in &result.snapshot.committed_words {
            println!("  [{:.3}s - {:.3}s] {}", w.start, w.end, w.word);
        }
    }

    Ok(())
}

fn print_update(chunk: usize, ms: u128, snapshot: &SessionSnapshot) {
    println!(
        "  chunk {chunk}: {ms:.0}ms | rev={} committed={} pending={} volatile={} | {}",
        snapshot.revision,
        snapshot.committed_token_count,
        snapshot.pending_token_count,
        snapshot.ambiguity.volatile_token_count,
        snapshot.full_text,
    );
}

/// Print per-word alternatives with confidence info.
///
/// Groups tokens by word boundaries (WordStart markers), decodes each
/// alternative token, and shows concentration/margin for the chosen token.
fn print_alternatives(tokenizer: &Tokenizer, entries: &[TokenEntry]) {
    if entries.is_empty() {
        return;
    }

    // Group entries into words
    let mut words: Vec<&[TokenEntry]> = Vec::new();
    let mut word_start = None;
    for (i, entry) in entries.iter().enumerate() {
        if entry.word.is_some() {
            if let Some(start) = word_start {
                words.push(&entries[start..i]);
            }
            word_start = Some(i);
        }
    }
    if let Some(start) = word_start {
        words.push(&entries[start..]);
    }

    println!("    ┌─ alternatives ─────────────────────────────────");
    for word_entries in &words {
        // Decode the chosen word
        let word_ids: Vec<u32> = word_entries.iter().map(|e| e.token.id).collect();
        let word_text = tokenizer.decode(&word_ids, true).unwrap_or_default();

        // Average confidence across tokens in this word
        let n = word_entries.len() as f32;
        let avg_conc: f32 = word_entries
            .iter()
            .map(|e| e.token.concentration)
            .sum::<f32>()
            / n;
        let avg_margin: f32 = word_entries.iter().map(|e| e.token.margin).sum::<f32>() / n;

        // Collect alternatives for each token position
        let mut alt_columns: Vec<Vec<String>> = vec![Vec::new(); TOP_K];
        for entry in *word_entries {
            for k in 0..TOP_K {
                let alt_text = tokenizer
                    .decode(&[entry.token.top_ids[k]], true)
                    .unwrap_or_default();
                alt_columns[k].push(alt_text);
            }
        }

        // Format: chosen word, then alternatives
        let alts: Vec<String> = (1..TOP_K)
            .map(|k| {
                let alt_word: String = alt_columns[k].join("");
                let alt_word = alt_word.trim();
                if alt_word.is_empty() || alt_word == word_text.trim() {
                    return String::new();
                }
                // Show the logit delta from top-1
                let avg_delta: f32 = word_entries
                    .iter()
                    .map(|e| e.token.top_logits[0] - e.token.top_logits[k])
                    .sum::<f32>()
                    / n;
                format!("{alt_word}(-{avg_delta:.1})")
            })
            .filter(|s| !s.is_empty())
            .collect();

        let alts_str = if alts.is_empty() {
            String::new()
        } else {
            format!("  alts: {}", alts.join(", "))
        };

        println!(
            "    │ {:>20}  conc={:5.1} margin={:5.1}{alts_str}",
            word_text.trim(),
            avg_conc,
            avg_margin,
        );
    }
    println!("    └──────────────────────────────────────────────");
}
