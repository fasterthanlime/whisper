use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use bee_correct::g2p::CachedEspeakG2p;
use bee_phonetic::{
    feature_similarity, normalize_ipa_for_comparison, phoneme_similarity, reduce_ipa_tokens,
};
use bee_zipa_mlx::infer::ZipaInference;
use serde_json::Value;

struct Args {
    bundle_dir: Option<PathBuf>,
    quantized_checkpoint: Option<PathBuf>,
    wav_dir: PathBuf,
    limit: Option<usize>,
    term: Option<String>,
    use_transcript: bool,
}

struct ComparisonRow {
    wav: PathBuf,
    term: String,
    text: String,
    zipa_raw: Vec<String>,
    zipa_reduced: Vec<String>,
    espeak_raw: Vec<String>,
    espeak_reduced: Vec<String>,
    zipa_normalized: Vec<String>,
    espeak_normalized: Vec<String>,
    raw_similarity: Option<f32>,
    reduced_similarity: Option<f32>,
    normalized_similarity: Option<f32>,
    normalized_feature_similarity: Option<f32>,
    feature_similarity: Option<f32>,
    reduced_exact: bool,
    normalized_exact: bool,
}

struct RecordingExample {
    term: String,
    text: String,
    audio_path: String,
    transcript: String,
}

fn main() -> Result<()> {
    let args = parse_args()?;

    let inference = match (&args.bundle_dir, &args.quantized_checkpoint) {
        (Some(path), None) => ZipaInference::load_quantized_bundle_dir(path)?,
        (None, Some(path)) => ZipaInference::load_quantized_safetensors(path)?,
        (None, None) => ZipaInference::load_reference_small_no_diacritics()?,
        (Some(_), Some(_)) => unreachable!(),
    };

    let recordings = load_recording_examples(&recording_examples_path())?;
    let mut g2p = CachedEspeakG2p::english(&g2p_base_dir())?;

    let mut rows = Vec::new();
    for row in &recordings {
        if let Some(term) = &args.term {
            if !row.term.eq_ignore_ascii_case(term) {
                continue;
            }
        }
        if args.limit.is_some_and(|limit| rows.len() >= limit) {
            break;
        }

        let wav = wav_path_for(&args.wav_dir, &row.audio_path);
        let zipa = inference.infer_wav(&wav)?;
        let zipa_raw = zipa
            .tokens
            .into_iter()
            .filter(|token| token != "▁")
            .collect::<Vec<_>>();
        let zipa_reduced = reduce_ipa_tokens(&zipa_raw);

        let text = if args.use_transcript {
            row.transcript.clone()
        } else {
            row.text.clone()
        };
        let espeak_raw = g2p
            .ipa_tokens(&text)?
            .ok_or_else(|| anyhow!("espeak produced no tokens for '{}'", text))?;
        let espeak_reduced = reduce_ipa_tokens(&espeak_raw);
        let zipa_normalized = normalize_ipa_for_comparison(&zipa_raw);
        let espeak_normalized = normalize_ipa_for_comparison(&espeak_raw);

        rows.push(ComparisonRow {
            wav,
            term: row.term.clone(),
            text,
            raw_similarity: phoneme_similarity(&zipa_raw, &espeak_raw),
            reduced_similarity: phoneme_similarity(&zipa_reduced, &espeak_reduced),
            normalized_similarity: phoneme_similarity(&zipa_normalized, &espeak_normalized),
            normalized_feature_similarity: feature_similarity(&zipa_normalized, &espeak_normalized),
            feature_similarity: feature_similarity(&zipa_raw, &espeak_raw),
            reduced_exact: zipa_reduced == espeak_reduced,
            normalized_exact: zipa_normalized == espeak_normalized,
            zipa_raw,
            zipa_reduced,
            zipa_normalized,
            espeak_raw,
            espeak_reduced,
            espeak_normalized,
        });
    }

    if rows.is_empty() {
        return Err(anyhow!("no matching recording examples"));
    }

    let mut raw_total = 0.0f32;
    let mut raw_count = 0usize;
    let mut reduced_total = 0.0f32;
    let mut reduced_count = 0usize;
    let mut normalized_total = 0.0f32;
    let mut normalized_count = 0usize;
    let mut feature_total = 0.0f32;
    let mut feature_count = 0usize;
    let mut normalized_feature_total = 0.0f32;
    let mut normalized_feature_count = 0usize;
    let mut exact_count = 0usize;
    let mut normalized_exact_count = 0usize;

    for row in &rows {
        if let Some(value) = row.raw_similarity {
            raw_total += value;
            raw_count += 1;
        }
        if let Some(value) = row.reduced_similarity {
            reduced_total += value;
            reduced_count += 1;
        }
        if let Some(value) = row.feature_similarity {
            feature_total += value;
            feature_count += 1;
        }
        if let Some(value) = row.normalized_similarity {
            normalized_total += value;
            normalized_count += 1;
        }
        if let Some(value) = row.normalized_feature_similarity {
            normalized_feature_total += value;
            normalized_feature_count += 1;
        }
        if row.reduced_exact {
            exact_count += 1;
        }
        if row.normalized_exact {
            normalized_exact_count += 1;
        }

        println!("wav: {}", row.wav.display());
        println!("term: {}", row.term);
        println!("text: {}", row.text);
        println!("zipa_raw: {}", row.zipa_raw.join(" "));
        println!("espeak_raw: {}", row.espeak_raw.join(" "));
        println!("zipa_reduced: {}", row.zipa_reduced.join(" "));
        println!("espeak_reduced: {}", row.espeak_reduced.join(" "));
        println!("zipa_normalized: {}", row.zipa_normalized.join(" "));
        println!("espeak_normalized: {}", row.espeak_normalized.join(" "));
        println!("raw_similarity: {}", format_optional(row.raw_similarity));
        println!(
            "reduced_similarity: {}",
            format_optional(row.reduced_similarity)
        );
        println!(
            "normalized_similarity: {}",
            format_optional(row.normalized_similarity)
        );
        println!(
            "feature_similarity: {}",
            format_optional(row.feature_similarity)
        );
        println!(
            "normalized_feature_similarity: {}",
            format_optional(row.normalized_feature_similarity)
        );
        println!("reduced_exact: {}", row.reduced_exact);
        println!("normalized_exact: {}", row.normalized_exact);
        println!();
    }

    println!("summary.rows: {}", rows.len());
    println!("summary.reduced_exact: {}/{}", exact_count, rows.len());
    println!(
        "summary.normalized_exact: {}/{}",
        normalized_exact_count,
        rows.len()
    );
    println!(
        "summary.raw_similarity_mean: {}",
        format_mean(raw_total, raw_count)
    );
    println!(
        "summary.reduced_similarity_mean: {}",
        format_mean(reduced_total, reduced_count)
    );
    println!(
        "summary.normalized_similarity_mean: {}",
        format_mean(normalized_total, normalized_count)
    );
    println!(
        "summary.feature_similarity_mean: {}",
        format_mean(feature_total, feature_count)
    );
    println!(
        "summary.normalized_feature_similarity_mean: {}",
        format_mean(normalized_feature_total, normalized_feature_count)
    );

    Ok(())
}

fn parse_args() -> Result<Args> {
    let mut args = env::args_os();
    let _program = args.next();
    let mut bundle_dir = None;
    let mut quantized_checkpoint = None;
    let mut wav_dir = default_wav_dir();
    let mut limit = None;
    let mut term = None;
    let mut use_transcript = false;

    while let Some(arg) = args.next() {
        match arg.to_string_lossy().as_ref() {
            "--bundle-dir" => {
                let value = args
                    .next()
                    .ok_or_else(|| anyhow!("--bundle-dir requires a value"))?;
                bundle_dir = Some(PathBuf::from(value));
            }
            "--quantized-checkpoint" => {
                let value = args
                    .next()
                    .ok_or_else(|| anyhow!("--quantized-checkpoint requires a value"))?;
                quantized_checkpoint = Some(PathBuf::from(value));
            }
            "--wav-dir" => {
                let value = args
                    .next()
                    .ok_or_else(|| anyhow!("--wav-dir requires a value"))?;
                wav_dir = PathBuf::from(value);
            }
            "--limit" => {
                let value = args
                    .next()
                    .ok_or_else(|| anyhow!("--limit requires a value"))?;
                limit = Some(value.to_string_lossy().parse::<usize>()?);
            }
            "--term" => {
                let value = args
                    .next()
                    .ok_or_else(|| anyhow!("--term requires a value"))?;
                term = Some(value.to_string_lossy().into_owned());
            }
            "--use-transcript" => use_transcript = true,
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(anyhow!("unexpected argument: {other}")),
        }
    }

    if bundle_dir.is_some() && quantized_checkpoint.is_some() {
        return Err(anyhow!(
            "--bundle-dir and --quantized-checkpoint cannot be used together"
        ));
    }

    Ok(Args {
        bundle_dir,
        quantized_checkpoint,
        wav_dir,
        limit,
        term,
        use_transcript,
    })
}

fn print_usage() {
    eprintln!(
        "usage: zipa-compare-espeak [--bundle-dir DIR | --quantized-checkpoint PATH] [--wav-dir DIR] [--limit N] [--term TERM] [--use-transcript]"
    );
}

fn default_wav_dir() -> PathBuf {
    let local = PathBuf::from("/Users/amos/bearcove/bee/data/phonetic-seed/audio-wav");
    if local.is_dir() {
        return local;
    }

    let worktree =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/phonetic-seed/audio-wav");
    if worktree.is_dir() {
        return worktree;
    }

    local
}

fn wav_path_for(wav_dir: &Path, audio_path: &str) -> PathBuf {
    let stem = Path::new(audio_path)
        .file_stem()
        .expect("recording audio path has a stem");
    wav_dir.join(format!("{}.wav", stem.to_string_lossy()))
}

fn recording_examples_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../data/phonetic-seed/recording_examples.jsonl")
}

fn g2p_base_dir() -> PathBuf {
    std::env::temp_dir().join("bee-zipa-mlx-espeak")
}

fn load_recording_examples(path: &Path) -> Result<Vec<RecordingExample>> {
    let file = File::open(path)?;
    let mut rows = Vec::new();
    for (line_idx, line) in BufReader::new(file).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(&line)
            .map_err(|error| anyhow!("parse {} line {}: {error}", path.display(), line_idx + 1))?;
        rows.push(RecordingExample {
            term: value
                .get("term")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("missing term in {} line {}", path.display(), line_idx + 1))?
                .to_string(),
            text: value
                .get("text")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("missing text in {} line {}", path.display(), line_idx + 1))?
                .to_string(),
            audio_path: value
                .get("audio_path")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    anyhow!(
                        "missing audio_path in {} line {}",
                        path.display(),
                        line_idx + 1
                    )
                })?
                .to_string(),
            transcript: value
                .get("transcript")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    anyhow!(
                        "missing transcript in {} line {}",
                        path.display(),
                        line_idx + 1
                    )
                })?
                .to_string(),
        });
    }
    Ok(rows)
}

fn format_optional(value: Option<f32>) -> String {
    value
        .map(|value| format!("{value:.4}"))
        .unwrap_or_else(|| "n/a".to_string())
}

fn format_mean(total: f32, count: usize) -> String {
    if count == 0 {
        "n/a".to_string()
    } else {
        format!("{:.4}", total / count as f32)
    }
}
