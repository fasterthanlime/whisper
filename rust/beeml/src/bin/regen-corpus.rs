//! Regenerate the phonetic seed corpus by re-transcribing audio files.
//! Produces per-word alignments with ASR logprob data.
//!
//! Usage: regen-corpus [--dry-run] [--only-canonical] [--only-counterexamples]
//!
//! Expects BEE_ASR_MODEL_DIR, BEE_TOKENIZER_PATH, BEE_ALIGNER_DIR env vars.

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use bee_phonetic::dataset::{
    CounterexampleRecordingRow, RecordingExampleRow, RecordingWordAlignment, SeedDataset,
};
use bee_transcribe::{Engine, EngineConfig, SessionOptions};
use facet::Facet;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let args: Vec<String> = std::env::args().collect();
    let dry_run = args.iter().any(|a| a == "--dry-run");
    let only_canonical = args.iter().any(|a| a == "--only-canonical");
    let only_counterexamples = args.iter().any(|a| a == "--only-counterexamples");

    if only_canonical && only_counterexamples {
        anyhow::bail!("--only-canonical and --only-counterexamples are mutually exclusive");
    }

    let model_dir = std::env::var("BEE_ASR_MODEL_DIR")
        .map_err(|_| anyhow::anyhow!("BEE_ASR_MODEL_DIR not set"))?;
    let tokenizer_dir =
        std::env::var("BEE_TOKENIZER_DIR").unwrap_or_else(|_| model_dir.clone());
    let aligner_dir =
        std::env::var("BEE_ALIGNER_DIR").map_err(|_| anyhow::anyhow!("BEE_ALIGNER_DIR not set"))?;
    let silero_dir =
        std::env::var("BEE_VAD_DIR").map_err(|_| anyhow::anyhow!("BEE_VAD_DIR not set"))?;

    let corpus_root = SeedDataset::canonical_root();

    let t0 = Instant::now();
    print!("Loading ASR engine... ");
    std::io::stdout().flush()?;
    let engine = Engine::load(&EngineConfig {
        model_dir: Path::new(&model_dir),
        tokenizer_dir: Path::new(&tokenizer_dir),
        aligner_dir: Path::new(&aligner_dir),
        silero_dir: Path::new(&silero_dir),
    })?;
    println!("done ({:.0}ms)", t0.elapsed().as_millis());

    // ── Canonical ──
    if !only_counterexamples {
        let dataset = SeedDataset::load_canonical()?;
        println!(
            "\n=== Canonical: {} recordings ===",
            dataset.recording_examples.len()
        );

        let mut updated: Vec<RecordingExampleRow> = Vec::new();
        let mut errors = 0u32;

        for (idx, row) in dataset.recording_examples.iter().enumerate() {
            print!(
                "[{}/{}] {} ... ",
                idx + 1,
                dataset.recording_examples.len(),
                row.term,
            );
            std::io::stdout().flush()?;

            let audio_path = corpus_root.join(&row.audio_path);
            match transcribe_file(&engine, &audio_path) {
                Ok((transcript, words)) => {
                    let changed = transcript != row.transcript;
                    println!(
                        "{} words, logprobs: {}/{}{}",
                        words.len(),
                        words.iter().filter(|w| w.mean_logprob.is_some()).count(),
                        words.len(),
                        if changed {
                            format!(" CHANGED: {:?} -> {:?}", row.transcript, transcript)
                        } else {
                            String::new()
                        }
                    );
                    updated.push(RecordingExampleRow {
                        term: row.term.clone(),
                        text: row.text.clone(),
                        take: row.take,
                        audio_path: row.audio_path.clone(),
                        transcript,
                        words,
                    });
                }
                Err(e) => {
                    println!("SKIP ({e})");
                    errors += 1;
                    updated.push(row.clone());
                }
            }
        }

        println!("\nProcessed: {}, Errors: {}", updated.len(), errors);
        if dry_run {
            println!("Dry run — not writing.");
        } else {
            let path = corpus_root.join("recording_examples.jsonl");
            write_jsonl(&path, &updated)?;
            println!("Wrote {}", path.display());
        }
    }

    // ── Counterexamples ──
    if !only_canonical {
        let ce_path = corpus_root.join("counterexample_recordings.jsonl");
        let ce_rows = load_counterexamples(&ce_path)?;
        println!("\n=== Counterexamples: {} recordings ===", ce_rows.len());

        let mut updated: Vec<CounterexampleRecordingRow> = Vec::new();
        let mut errors = 0u32;

        for (idx, row) in ce_rows.iter().enumerate() {
            print!(
                "[{}/{}] {} ({}) ... ",
                idx + 1,
                ce_rows.len(),
                row.term,
                row.surface_form,
            );
            std::io::stdout().flush()?;

            let audio_path = corpus_root.join(&row.audio_path);
            match transcribe_file(&engine, &audio_path) {
                Ok((transcript, words)) => {
                    let changed = transcript != row.transcript;
                    println!(
                        "{} words, logprobs: {}/{}{}",
                        words.len(),
                        words.iter().filter(|w| w.mean_logprob.is_some()).count(),
                        words.len(),
                        if changed {
                            format!(" CHANGED: {:?} -> {:?}", row.transcript, transcript)
                        } else {
                            String::new()
                        }
                    );
                    updated.push(CounterexampleRecordingRow {
                        term: row.term.clone(),
                        text: row.text.clone(),
                        take: row.take,
                        audio_path: row.audio_path.clone(),
                        transcript,
                        surface_form: row.surface_form.clone(),
                        words,
                    });
                }
                Err(e) => {
                    println!("SKIP ({e})");
                    errors += 1;
                    updated.push(row.clone());
                }
            }
        }

        println!("\nProcessed: {}, Errors: {}", updated.len(), errors);
        if dry_run {
            println!("Dry run — not writing.");
        } else {
            write_jsonl(&ce_path, &updated)?;
            println!("Wrote {}", ce_path.display());
        }
    }

    Ok(())
}

fn transcribe_file(
    engine: &Engine,
    audio_path: &Path,
) -> anyhow::Result<(String, Vec<RecordingWordAlignment>)> {
    let samples = bee_qwen3_asr::mel::load_audio(&audio_path.to_string_lossy(), 16000)
        .map_err(|e| anyhow::anyhow!("audio load: {e}"))?;

    let options = SessionOptions::default();
    let chunk_samples = (options.chunk_duration * 16000.0) as usize;
    let mut session = engine.session(options)?;

    let mut offset = 0;
    while offset < samples.len() {
        let end = (offset + chunk_samples).min(samples.len());
        let _ = session.feed(&samples[offset..end]);
        offset = end;
    }

    let update = session.finish().map_err(|e| anyhow::anyhow!("transcription: {e}"))?;

    let words = update
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

    Ok((update.text, words))
}

fn write_jsonl<'a, T: Facet<'a>>(path: &Path, rows: &[T]) -> anyhow::Result<()> {
    let mut file = std::fs::File::create(path)?;
    for row in rows {
        write!(
            file,
            "{}",
            facet_json::to_string(row).map_err(|e| anyhow::anyhow!("{e:?}"))?
        )?;
        writeln!(file)?;
    }
    Ok(())
}

fn load_counterexamples(path: &Path) -> anyhow::Result<Vec<CounterexampleRecordingRow>> {
    let text = std::fs::read_to_string(path)?;
    let mut rows = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let row = facet_json::from_str::<CounterexampleRecordingRow>(line)
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        rows.push(row);
    }
    Ok(rows)
}
