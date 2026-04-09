use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use bee_phonetic::{SeedDataset, top_right_anchor_windows};
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
    let snapshot_out = parse_flag_string("--snapshot-out");

    let service = load_service()?;
    let result = service
        .eval_corpus_alignment(limit as usize, bucket.as_deref())
        .map_err(|error| anyhow::anyhow!("running corpus alignment eval: {error}"))?;

    if let Some(path) = snapshot_out.as_deref() {
        write_snapshot_jsonl(path, &result.rows)?;
    }

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
            "#{:03} [{}] {}  uf={} raw={} spans={} contentful={} eligible={} volatile={} ready={} worst={} bestΔ={}",
            row.ordinal,
            row.bucket,
            row.prompt_text,
            fmt(row.utterance_feature_similarity),
            fmt(row.utterance_similarity),
            row.positive_span_count,
            row.contentful_span_count,
            row.rescue_eligible_span_count,
            row.tail_volatile_token_count,
            row.row_rescue_ready,
            fmt(row.worst_span_feature_similarity),
            fmt(row.best_span_delta),
        );
        println!("term: {}", row.term);
        println!("wav: {}", row.wav_path);
        println!("asr: {}", row.asr_transcript);
        if let Some(role) = &row.selected_span_role {
            println!(
                "selected span: {} {:?} base={} bestΔ={}",
                format_selected_span_role(role),
                row.selected_span_text,
                fmt(row.selected_span_feature_similarity),
                fmt(row.selected_span_best_delta),
            );
        }
        if let Some(notes) = &row.prompt_notes {
            println!("notes: {}", notes);
        }
        if let Some(trace) = &row.trace {
            println!("snapshot revision: {}", trace.snapshot_revision);
            println!("aligned transcript: {}", trace.aligned_transcript);
            if !trace.pending_text.is_empty() {
                println!("pending tail: {}", trace.pending_text);
            }
            println!(
                "tail ambiguity: pending={} volatile={} low_conc={} low_margin={} mean_conc={:.2} mean_margin={:.2}",
                trace.tail_ambiguity.pending_token_count,
                trace.tail_ambiguity.volatile_token_count,
                trace.tail_ambiguity.low_concentration_count,
                trace.tail_ambiguity.low_margin_count,
                trace.tail_ambiguity.mean_concentration,
                trace.tail_ambiguity.mean_margin,
            );
            println!(
                "utterance transcript norm: {}",
                trace.utterance_transcript_normalized.join(" ")
            );
            println!("utterance ZIPA raw: {}", trace.utterance_zipa_raw.join(" "));
            println!(
                "utterance ZIPA norm: {}",
                trace.utterance_zipa_normalized.join(" ")
            );
            if !trace.word_alignments.is_empty() {
                println!("word alignments:");
                for word in &trace.word_alignments {
                    println!(
                        "  toks {}:{} {:?} -> ZIPA {}:{} | transcript={} | zipa={}",
                        word.token_start,
                        word.token_end,
                        word.word_text,
                        word.zipa_norm_start,
                        word.zipa_norm_end,
                        word.transcript_normalized.join(" "),
                        word.zipa_normalized.join(" "),
                    );
                }
            }
            if let Some(span) = selected_span(trace) {
                println!("worst span: {:?}", span.span_text);
                println!(
                    "alignment source={} confidence={} class={} usefulness={} eligible={} phones {}->{} proj={} chosen={} second={} gap={}",
                    span.alignment_source,
                    format_confidence(&span.anchor_confidence),
                    format_span_class(&span.span_class),
                    format_usefulness(&span.span_usefulness),
                    span.zipa_rescue_eligible,
                    span.transcript_phone_count,
                    span.chosen_zipa_phone_count,
                    fmt(span.projected_alignment_score),
                    fmt(span.chosen_alignment_score),
                    fmt(span.second_best_alignment_score),
                    fmt(span.alignment_score_gap),
                );
                let top_windows = top_right_anchor_windows(
                    &span.transcript_normalized,
                    &trace.utterance_zipa_normalized,
                    3,
                );
                if !top_windows.is_empty() {
                    println!("top coarse windows:");
                    for window in top_windows {
                        let slice = trace
                            .utterance_zipa_normalized
                            .get(window.right_start as usize..window.right_end as usize)
                            .unwrap_or(&[]);
                        println!(
                            "  ZIPA {}..{} score={:.4} mean={:.4} Δlen={}: {}",
                            window.right_start,
                            window.right_end,
                            window.score,
                            window.mean_similarity,
                            window.length_delta,
                            slice.join(" ")
                        );
                    }
                }
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

fn format_confidence(value: &beeml::rpc::TranscribePhoneticAnchorConfidence) -> &'static str {
    match value {
        beeml::rpc::TranscribePhoneticAnchorConfidence::Low => "low",
        beeml::rpc::TranscribePhoneticAnchorConfidence::Medium => "medium",
        beeml::rpc::TranscribePhoneticAnchorConfidence::High => "high",
    }
}

fn format_usefulness(value: &beeml::rpc::TranscribePhoneticSpanUsefulness) -> &'static str {
    match value {
        beeml::rpc::TranscribePhoneticSpanUsefulness::Low => "low",
        beeml::rpc::TranscribePhoneticSpanUsefulness::Medium => "medium",
        beeml::rpc::TranscribePhoneticSpanUsefulness::High => "high",
    }
}

fn format_span_class(value: &beeml::rpc::TranscribePhoneticSpanClass) -> &'static str {
    match value {
        beeml::rpc::TranscribePhoneticSpanClass::Repeat => "repeat",
        beeml::rpc::TranscribePhoneticSpanClass::ShortCodeTerm => "short_code_term",
        beeml::rpc::TranscribePhoneticSpanClass::VowelHeavy => "vowel_heavy",
        beeml::rpc::TranscribePhoneticSpanClass::ProperNoun => "proper_noun",
        beeml::rpc::TranscribePhoneticSpanClass::FunctionWord => "function_word",
        beeml::rpc::TranscribePhoneticSpanClass::Ordinary => "ordinary",
    }
}

fn format_selected_span_role(value: &beeml::rpc::CorpusAlignmentSelectedSpanRole) -> &'static str {
    match value {
        beeml::rpc::CorpusAlignmentSelectedSpanRole::BestRescue => "best_rescue",
        beeml::rpc::CorpusAlignmentSelectedSpanRole::WorstContentful => "worst_contentful",
        beeml::rpc::CorpusAlignmentSelectedSpanRole::WorstRaw => "worst_raw",
    }
}

#[derive(facet::Facet)]
struct SnapshotRow {
    ordinal: u32,
    bucket: String,
    prompt_id: String,
    term: String,
    prompt_text: String,
    utterance_feature_similarity: Option<f32>,
    utterance_similarity: Option<f32>,
    tail_volatile_token_count: u32,
    row_rescue_ready: bool,
    positive_span_count: u32,
    contentful_span_count: u32,
    rescue_eligible_span_count: u32,
    selected_span_role: Option<String>,
    selected_span_text: Option<String>,
    selected_span_feature_similarity: Option<f32>,
    selected_span_best_delta: Option<f32>,
    worst_span_text: Option<String>,
    alignment_source: Option<String>,
    anchor_confidence: Option<String>,
    span_class: Option<String>,
    span_usefulness: Option<String>,
    zipa_rescue_eligible: Option<bool>,
    transcript_phone_count: Option<u32>,
    chosen_zipa_phone_count: Option<u32>,
    projected_alignment_score: Option<f32>,
    chosen_alignment_score: Option<f32>,
    second_best_alignment_score: Option<f32>,
    alignment_score_gap: Option<f32>,
}

fn write_snapshot_jsonl(path: &str, rows: &[beeml::rpc::CorpusAlignmentEvalRow]) -> Result<()> {
    let file = File::create(path).with_context(|| format!("creating snapshot file {path}"))?;
    let mut writer = BufWriter::new(file);
    for row in rows {
        let worst_span = row.trace.as_ref().and_then(selected_span);
        let snapshot = SnapshotRow {
            ordinal: row.ordinal,
            bucket: row.bucket.clone(),
            prompt_id: row.prompt_id.clone(),
            term: row.term.clone(),
            prompt_text: row.prompt_text.clone(),
            utterance_feature_similarity: row.utterance_feature_similarity,
            utterance_similarity: row.utterance_similarity,
            tail_volatile_token_count: row.tail_volatile_token_count,
            row_rescue_ready: row.row_rescue_ready,
            positive_span_count: row.positive_span_count,
            contentful_span_count: row.contentful_span_count,
            rescue_eligible_span_count: row.rescue_eligible_span_count,
            selected_span_role: row
                .selected_span_role
                .as_ref()
                .map(|role| format_selected_span_role(role).to_string()),
            selected_span_text: row.selected_span_text.clone(),
            selected_span_feature_similarity: row.selected_span_feature_similarity,
            selected_span_best_delta: row.selected_span_best_delta,
            worst_span_text: worst_span.map(|span| span.span_text.clone()),
            alignment_source: worst_span.map(|span| span.alignment_source.clone()),
            anchor_confidence: worst_span
                .map(|span| format_confidence(&span.anchor_confidence).to_string()),
            span_class: worst_span.map(|span| format_span_class(&span.span_class).to_string()),
            span_usefulness: worst_span
                .map(|span| format_usefulness(&span.span_usefulness).to_string()),
            zipa_rescue_eligible: worst_span.map(|span| span.zipa_rescue_eligible),
            transcript_phone_count: worst_span.map(|span| span.transcript_phone_count),
            chosen_zipa_phone_count: worst_span.map(|span| span.chosen_zipa_phone_count),
            projected_alignment_score: worst_span.and_then(|span| span.projected_alignment_score),
            chosen_alignment_score: worst_span.and_then(|span| span.chosen_alignment_score),
            second_best_alignment_score: worst_span
                .and_then(|span| span.second_best_alignment_score),
            alignment_score_gap: worst_span.and_then(|span| span.alignment_score_gap),
        };
        let json = facet_json::to_string(&snapshot).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        writer.write_all(json.as_bytes())?;
        writer.write_all(b"\n")?;
    }
    Ok(())
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

fn selected_span(
    trace: &beeml::rpc::TranscribePhoneticTrace,
) -> Option<&beeml::rpc::TranscribePhoneticSpan> {
    trace
        .best_rescue_span_index
        .or(trace.worst_contentful_span_index)
        .or(trace.worst_raw_span_index)
        .and_then(|index| trace.spans.get(index as usize))
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
