use std::env;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use bee_phonetic::SeedDataset;
use bee_transcribe::{Engine, EngineConfig};
use bee_zipa_mlx::infer::ZipaInference;
use beeml::g2p::CachedEspeakG2p;
use beeml::judge::OnlineJudge;

#[path = "../offline_eval.rs"]
mod offline_eval;
#[path = "../rapid_fire.rs"]
mod rapid_fire;
#[path = "../rpc_impl.rs"]
mod rpc_impl;
#[path = "../service.rs"]
mod service;
#[path = "../util.rs"]
mod util;

use service::{BeeMlService, BeemlServiceInner};
use util::load_counterexample_recordings;

fn main() -> Result<()> {
    let limit = parse_flag_u32("--limit").unwrap_or(20);
    let show = parse_flag_u32("--show").unwrap_or(8);
    let bucket = parse_flag_string("--bucket");

    let service = load_service()?;
    let result = service
        .eval_corpus_alignment(limit as usize, bucket.as_deref())
        .map_err(|error| anyhow::anyhow!("running corpus alignment eval: {error}"))?;

    println!("rows: {}", result.rows.len());
    for summary in &result.bucket_summaries {
        println!(
            "bucket={} rows={} uf_mean={} raw_mean={}",
            summary.bucket,
            summary.rows,
            fmt(summary.utterance_feature_similarity_mean),
            fmt(summary.utterance_similarity_mean),
        );
    }
    println!();

    for row in result.rows.iter().take(show as usize) {
        println!(
            "#{:03} [{}] {}  uf={} raw={} spans={} worst={} bestΔ={}",
            row.ordinal,
            row.bucket,
            row.prompt_text,
            fmt(row.utterance_feature_similarity),
            fmt(row.utterance_similarity),
            row.positive_span_count,
            fmt(row.worst_span_feature_similarity),
            fmt(row.best_span_delta),
        );
        println!("term: {}", row.term);
        println!("wav: {}", row.wav_path);
        println!("asr: {}", row.asr_transcript);
        if let Some(notes) = &row.prompt_notes {
            println!("notes: {}", notes);
        }
        if let Some(trace) = &row.trace {
            if let Some(span) = worst_span(trace) {
                println!("worst span: {:?}", span.span_text);
                println!(
                    "utterance window around ZIPA {}..{}:",
                    span.zipa_norm_start, span.zipa_norm_end
                );
                let utterance_window = crop_utterance_ops(
                    &trace.utterance_alignment,
                    span.zipa_norm_start as usize,
                    span.zipa_norm_end as usize,
                    6,
                );
                print_ops("Transcript", "ZIPA", &utterance_window);
                println!("span-local alignment:");
                print_ops("Transcript", "ZIPA", &span.alignment);
            } else {
                println!("utterance:");
                print_ops("Transcript", "ZIPA", &trace.utterance_alignment);
            }
        } else if let Some(error) = &row.error {
            println!("error: {}", error);
        }
        println!("{}", "-".repeat(80));
    }

    Ok(())
}

fn parse_flag_u32(flag: &str) -> Option<u32> {
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == flag {
            return args.next().and_then(|value| value.parse().ok());
        }
    }
    None
}

fn parse_flag_string(flag: &str) -> Option<String> {
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == flag {
            return args.next();
        }
    }
    None
}

fn fmt(value: Option<f32>) -> String {
    value
        .map(|v| format!("{v:.4}"))
        .unwrap_or_else(|| "n/a".to_string())
}

fn print_ops(
    left_label: &str,
    right_label: &str,
    ops: &[beeml::rpc::TranscribePhoneticAlignmentOp],
) {
    let left = ops
        .iter()
        .map(|op| {
            op.transcript_token
                .clone()
                .unwrap_or_else(|| "∅".to_string())
        })
        .collect::<Vec<_>>();
    let right = ops
        .iter()
        .map(|op| op.zipa_token.clone().unwrap_or_else(|| "∅".to_string()))
        .collect::<Vec<_>>();
    println!("{left_label}: {}", left.join(" "));
    println!("{right_label}: {}", right.join(" "));
}

fn crop_utterance_ops(
    ops: &[beeml::rpc::TranscribePhoneticAlignmentOp],
    zipa_start: usize,
    zipa_end: usize,
    context: usize,
) -> Vec<beeml::rpc::TranscribePhoneticAlignmentOp> {
    if ops.is_empty() {
        return Vec::new();
    }
    let first = ops.iter().position(|op| {
        op.zipa_index
            .map(|idx| (idx as usize) >= zipa_start && (idx as usize) < zipa_end)
            .unwrap_or(false)
    });
    let last = ops.iter().rposition(|op| {
        op.zipa_index
            .map(|idx| (idx as usize) >= zipa_start && (idx as usize) < zipa_end)
            .unwrap_or(false)
    });
    match (first, last) {
        (Some(first), Some(last)) => {
            let start = first.saturating_sub(context);
            let end = (last + context + 1).min(ops.len());
            ops[start..end].to_vec()
        }
        _ => ops
            .iter()
            .take((context * 2 + 1).min(ops.len()))
            .cloned()
            .collect(),
    }
}

fn worst_span(
    trace: &beeml::rpc::TranscribePhoneticTrace,
) -> Option<&beeml::rpc::TranscribePhoneticSpan> {
    trace.spans.iter().min_by(|a, b| {
        a.transcript_feature_similarity
            .unwrap_or(f32::INFINITY)
            .total_cmp(&b.transcript_feature_similarity.unwrap_or(f32::INFINITY))
    })
}

fn load_service() -> Result<BeeMlService> {
    let model_dir = env::var("BEE_ASR_MODEL_DIR")
        .map(PathBuf::from)
        .context("BEE_ASR_MODEL_DIR must be set")?;
    let tokenizer_dir = env::var("BEE_TOKENIZER_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| model_dir.clone());
    let aligner_dir = env::var("BEE_ALIGNER_DIR")
        .map(PathBuf::from)
        .context("BEE_ALIGNER_DIR must be set")?;
    let silero_dir = env::var("BEE_VAD_DIR")
        .map(PathBuf::from)
        .context("BEE_VAD_DIR must be set")?;

    let engine = Engine::load(&EngineConfig {
        model_dir: &model_dir,
        tokenizer_dir: &tokenizer_dir,
        aligner_dir: &aligner_dir,
        silero_dir: &silero_dir,
        correction_dir: None,
        correction_events_path: None,
    })
    .context("loading engine")?;

    let dataset = SeedDataset::load_canonical().context("loading canonical dataset")?;
    dataset.validate().context("validating dataset")?;
    let index = dataset.phonetic_index();
    let counterexamples =
        load_counterexample_recordings().context("loading counterexample recordings")?;

    let zipa_bundle_dir = env::var("BEE_ZIPA_BUNDLE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("bearcove/zipa-mlx-hf")
        });
    let zipa_wav_dir = env::var("BEE_PHONETIC_WAV_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("bearcove/bee/data/phonetic-seed/audio-wav")
        });
    let corpus_dir = env::var("BEE_ZIPA_CORPUS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("bearcove/bee/data/zipa-corpus")
        });
    let event_log_path = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".beeml")
        .join("events.jsonl");

    let zipa = ZipaInference::load_quantized_bundle_dir(&zipa_bundle_dir)
        .with_context(|| format!("loading ZIPA bundle {}", zipa_bundle_dir.display()))?;

    Ok(BeeMlService {
        inner: Arc::new(BeemlServiceInner {
            engine,
            index,
            dataset,
            counterexamples,
            g2p: Mutex::new(
                CachedEspeakG2p::english(
                    Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join("../../target")
                        .as_ref(),
                )
                .context("initializing g2p engine")?,
            ),
            zipa: Mutex::new(zipa),
            zipa_wav_dir,
            corpus_dir,
            corpus_eval_jobs: Mutex::new(std::collections::HashMap::new()),
            next_corpus_eval_job_id: std::sync::atomic::AtomicU64::new(1),
            judge: Mutex::new(OnlineJudge::default()),
            event_log_path,
        }),
    })
}
