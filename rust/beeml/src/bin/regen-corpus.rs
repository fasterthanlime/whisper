//! Regenerate the phonetic seed corpus by re-transcribing audio files.
//! Produces per-word alignments with ASR logprob data.
//!
//! Usage: regen-corpus [--dry-run]
//!
//! Expects BEE_ASR_MODEL_DIR, BEE_TOKENIZER_PATH, BEE_ALIGNER_DIR env vars.

use std::path::{Path, PathBuf};
use std::time::Instant;

use bee_phonetic::dataset::{RecordingExampleRow, RecordingWordAlignment, SeedDataset};
use bee_transcribe::{Engine, EngineConfig, SessionOptions};

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let dry_run = std::env::args().any(|a| a == "--dry-run");

    let model_dir = std::env::var("BEE_ASR_MODEL_DIR")
        .map_err(|_| anyhow::anyhow!("BEE_ASR_MODEL_DIR not set"))?;
    let tokenizer_dir = std::env::var("BEE_TOKENIZER_DIR")
        .unwrap_or_else(|_| model_dir.clone());
    let aligner_dir =
        std::env::var("BEE_ALIGNER_DIR").map_err(|_| anyhow::anyhow!("BEE_ALIGNER_DIR not set"))?;

    let dataset = SeedDataset::load_canonical()?;
    let corpus_root = SeedDataset::canonical_root();

    println!(
        "Corpus: {} recording examples in {}",
        dataset.recording_examples.len(),
        corpus_root.display()
    );

    // Load engine
    let t0 = Instant::now();
    let engine = Engine::load(&EngineConfig {
        model_dir: Path::new(&model_dir),
        tokenizer_dir: Path::new(&tokenizer_dir),
        aligner_dir: Path::new(&aligner_dir),
    })?;
    println!("Engine loaded in {:.0}ms", t0.elapsed().as_millis());

    let mut updated_rows: Vec<RecordingExampleRow> = Vec::new();
    let mut errors = 0u32;

    for (idx, row) in dataset.recording_examples.iter().enumerate() {
        let audio_path = corpus_root.join(&row.audio_path);
        let audio_path_str = audio_path.to_string_lossy();

        print!("[{}/{}] {} ({})... ", idx + 1, dataset.recording_examples.len(), row.term, row.audio_path);

        let samples = match bee_qwen3_asr::mel::load_audio(&audio_path_str, 16000) {
            Ok(s) => s,
            Err(e) => {
                println!("SKIP (audio load error: {e})");
                errors += 1;
                updated_rows.push(row.clone());
                continue;
            }
        };

        let options = SessionOptions::default();
        let chunk_samples = (options.chunk_duration * 16000.0) as usize;
        let mut session = engine.session(options);

        // Feed audio in chunks
        let mut offset = 0;
        while offset < samples.len() {
            let end = (offset + chunk_samples).min(samples.len());
            let chunk = &samples[offset..end];
            offset = end;
            let _ = session.feed(chunk);
        }

        let final_update = match session.finish() {
            Ok(u) => u,
            Err(e) => {
                println!("SKIP (transcription error: {e})");
                errors += 1;
                updated_rows.push(row.clone());
                continue;
            }
        };

        let words: Vec<RecordingWordAlignment> = final_update
            .alignments
            .iter()
            .map(|w| RecordingWordAlignment {
                word: w.word.clone(),
                start: w.start,
                end: w.end,
                mean_logprob: w.mean_logprob,
                min_logprob: w.min_logprob,
                mean_margin: w.mean_margin,
                min_margin: w.min_margin,
            })
            .collect();

        let new_transcript = final_update.text.clone();
        let changed = new_transcript != row.transcript;

        println!(
            "{} words, logprobs: {}/{}{}",
            words.len(),
            words.iter().filter(|w| w.mean_logprob.is_some()).count(),
            words.len(),
            if changed {
                format!(" TRANSCRIPT CHANGED: {:?} -> {:?}", row.transcript, new_transcript)
            } else {
                String::new()
            }
        );

        updated_rows.push(RecordingExampleRow {
            term: row.term.clone(),
            text: row.text.clone(),
            take: row.take,
            audio_path: row.audio_path.clone(),
            transcript: new_transcript,
            words,
        });
    }

    println!("\n--- Summary ---");
    println!("Processed: {}", updated_rows.len());
    println!("Errors: {}", errors);
    let with_logprobs = updated_rows.iter().filter(|r| !r.words.is_empty()).count();
    println!("With word alignments: {}", with_logprobs);

    if dry_run {
        println!("Dry run — not writing files.");
    } else {
        let output_path = corpus_root.join("recording_examples.jsonl");
        write_jsonl(&output_path, &updated_rows)?;
        println!("Wrote {}", output_path.display());
    }

    Ok(())
}

fn write_jsonl(path: &Path, rows: &[RecordingExampleRow]) -> anyhow::Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create(path)?;
    for row in rows {
        write!(file, "{}", facet_json::to_string(row).map_err(|e| anyhow::anyhow!("{e:?}"))?)?;
        writeln!(file)?;
    }
    Ok(())
}
