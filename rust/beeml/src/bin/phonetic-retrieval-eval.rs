use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use bee_phonetic::{
    enumerate_transcript_spans_with, query_index, verify_shortlist, RetrievalQuery, SeedDataset,
    TranscriptAlignmentToken, TranscriptSpan, VerifiedCandidate,
};
use beeml::g2p::CachedEspeakG2p;
use rayon::prelude::*;
use serde::Deserialize;

#[derive(Debug, Clone)]
struct EvalConfig {
    max_span_words: usize,
    shortlist_limit: usize,
    verify_limit: usize,
    sample_limit: Option<usize>,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            max_span_words: 4,
            shortlist_limit: 10,
            verify_limit: 10,
            sample_limit: None,
        }
    }
}

#[derive(Debug, Clone)]
struct RankedTermHit {
    term: String,
    span_text: String,
    candidate: VerifiedCandidate,
}

#[derive(Debug, Clone)]
struct PreparedRecording {
    term: String,
    transcript: String,
    spans: Vec<TranscriptSpan>,
}

#[derive(Debug, Clone, Deserialize)]
struct CounterexampleRecordingRow {
    term: String,
    text: String,
    take: i64,
    audio_path: String,
    transcript: String,
    surface_form: String,
}

#[derive(Debug, Default)]
struct EvalSummary {
    total: usize,
    hit_top1: usize,
    hit_top3: usize,
    hit_top10: usize,
    miss: usize,
}

#[derive(Debug, Default)]
struct CounterexampleSummary {
    total: usize,
    false_positive_top1: usize,
    false_positive_top3: usize,
    false_positive_top10: usize,
}

#[derive(Debug, Default)]
struct EvalTimings {
    recordings: usize,
    spans: usize,
    g2p_calls: usize,
    g2p_cache_hits: usize,
    g2p_cache_misses: usize,
    enumerate_total: Duration,
    g2p_total: Duration,
    query_total: Duration,
    verify_total: Duration,
    recording_total: Duration,
}

#[derive(Debug, Default)]
struct RecordingTimings {
    spans: usize,
    g2p_calls: usize,
    g2p_cache_hits: usize,
    g2p_cache_misses: usize,
    enumerate_total: Duration,
    g2p_total: Duration,
    query_total: Duration,
    verify_total: Duration,
    recording_total: Duration,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args()?;
    let dataset = SeedDataset::load_canonical()?;
    dataset.validate()?;

    let index = dataset.phonetic_index();
    let mut g2p = CachedEspeakG2p::english_with_persist_path(Some(eval_g2p_cache_path()))?;
    let mut summary = EvalSummary::default();
    let mut counter_summary = CounterexampleSummary::default();
    let mut timings = EvalTimings::default();
    let mut misses = Vec::new();
    let mut counter_hits = Vec::new();

    let mut prepared = Vec::new();
    for row in dataset
        .recording_examples
        .iter()
        .take(config.sample_limit.unwrap_or(usize::MAX))
    {
        summary.total += 1;
        let (prepared_recording, recording_timings) = prepare_recording(&mut g2p, row, &config)?;
        timings.recordings += 1;
        timings.spans += recording_timings.spans;
        timings.g2p_calls += recording_timings.g2p_calls;
        timings.g2p_cache_hits += recording_timings.g2p_cache_hits;
        timings.g2p_cache_misses += recording_timings.g2p_cache_misses;
        timings.enumerate_total += recording_timings.enumerate_total;
        timings.g2p_total += recording_timings.g2p_total;
        timings.recording_total += recording_timings.recording_total;
        prepared.push(prepared_recording);
    }

    let evaluated = prepared
        .par_iter()
        .map(|recording| evaluate_prepared_recording(&index, recording, &config))
        .collect::<Vec<_>>();

    for (recording, ranked, recording_timings) in evaluated {
        timings.query_total += recording_timings.query_total;
        timings.verify_total += recording_timings.verify_total;
        timings.recording_total += recording_timings.recording_total;
        let target_rank = ranked
            .iter()
            .position(|hit| hit.term.eq_ignore_ascii_case(&recording.term))
            .map(|idx| idx + 1);

        match target_rank {
            Some(rank) => {
                if rank <= 1 {
                    summary.hit_top1 += 1;
                }
                if rank <= 3 {
                    summary.hit_top3 += 1;
                }
                if rank <= 10 {
                    summary.hit_top10 += 1;
                }
            }
            None => {
                summary.miss += 1;
                misses.push((
                    recording.term,
                    recording.transcript,
                    ranked.into_iter().take(5).collect::<Vec<_>>(),
                ));
            }
        }
    }

    let counterexample_rows = load_counterexample_recordings()?;
    let mut counter_prepared = Vec::new();
    for row in counterexample_rows
        .iter()
        .take(config.sample_limit.unwrap_or(usize::MAX))
    {
        counter_summary.total += 1;
        let (prepared_recording, recording_timings) =
            prepare_counterexample_recording(&mut g2p, row, &config)?;
        timings.recordings += 1;
        timings.spans += recording_timings.spans;
        timings.g2p_calls += recording_timings.g2p_calls;
        timings.g2p_cache_hits += recording_timings.g2p_cache_hits;
        timings.g2p_cache_misses += recording_timings.g2p_cache_misses;
        timings.enumerate_total += recording_timings.enumerate_total;
        timings.g2p_total += recording_timings.g2p_total;
        timings.recording_total += recording_timings.recording_total;
        counter_prepared.push(prepared_recording);
    }

    let counter_evaluated = counter_prepared
        .par_iter()
        .map(|recording| evaluate_prepared_recording(&index, recording, &config))
        .collect::<Vec<_>>();

    for (recording, ranked, recording_timings) in counter_evaluated {
        timings.query_total += recording_timings.query_total;
        timings.verify_total += recording_timings.verify_total;
        timings.recording_total += recording_timings.recording_total;
        let target_rank = ranked
            .iter()
            .position(|hit| hit.term.eq_ignore_ascii_case(&recording.term))
            .map(|idx| idx + 1);
        if let Some(rank) = target_rank {
            if rank <= 1 {
                counter_summary.false_positive_top1 += 1;
            }
            if rank <= 3 {
                counter_summary.false_positive_top3 += 1;
            }
            if rank <= 10 {
                counter_summary.false_positive_top10 += 1;
            }
            counter_hits.push((recording.term, recording.transcript, ranked, rank));
        }
    }

    println!("recordings={}", summary.total);
    println!("top1={}", summary.hit_top1);
    println!("top3={}", summary.hit_top3);
    println!("top10={}", summary.hit_top10);
    println!("miss={}", summary.miss);
    println!();
    println!("counterexamples={}", counter_summary.total);
    println!(
        "counterexample_fp_top1={}",
        counter_summary.false_positive_top1
    );
    println!(
        "counterexample_fp_top3={}",
        counter_summary.false_positive_top3
    );
    println!(
        "counterexample_fp_top10={}",
        counter_summary.false_positive_top10
    );
    println!();
    println!("timings_ms");
    println!(
        "recording_total={:.3}",
        timings.recording_total.as_secs_f64() * 1000.0
    );
    println!(
        "recording_avg={:.3}",
        millis_per(timings.recording_total, timings.recordings)
    );
    println!(
        "enumerate_total={:.3}",
        timings.enumerate_total.as_secs_f64() * 1000.0
    );
    println!(
        "enumerate_avg={:.3}",
        millis_per(timings.enumerate_total, timings.recordings)
    );
    println!("g2p_total={:.3}", timings.g2p_total.as_secs_f64() * 1000.0);
    println!(
        "g2p_avg={:.3}",
        millis_per(timings.g2p_total, timings.g2p_calls)
    );
    println!(
        "query_total={:.3}",
        timings.query_total.as_secs_f64() * 1000.0
    );
    println!(
        "query_avg={:.3}",
        millis_per(timings.query_total, timings.spans)
    );
    println!(
        "verify_total={:.3}",
        timings.verify_total.as_secs_f64() * 1000.0
    );
    println!(
        "verify_avg={:.3}",
        millis_per(timings.verify_total, timings.spans)
    );
    println!("g2p_calls={}", timings.g2p_calls);
    println!("g2p_cache_hits={}", timings.g2p_cache_hits);
    println!("g2p_cache_misses={}", timings.g2p_cache_misses);
    println!("spans={}", timings.spans);

    if !misses.is_empty() {
        println!();
        println!("sample_misses={}", misses.len().min(10));
        for (term, transcript, ranked) in misses.into_iter().take(10) {
            println!("---");
            println!("term={term}");
            println!("transcript={transcript}");
            for hit in ranked {
                println!(
                    "candidate={} span={} phonetic_score={:.3} coarse_score={:.3}",
                    hit.term,
                    hit.span_text,
                    hit.candidate.phonetic_score,
                    hit.candidate.coarse_score
                );
            }
        }
    }

    if !counter_hits.is_empty() {
        println!();
        println!("sample_counterexample_hits={}", counter_hits.len().min(10));
        counter_hits.sort_by_key(|(_, _, _, rank)| *rank);
        for (term, transcript, ranked, rank) in counter_hits.into_iter().take(10) {
            println!("---");
            println!("term={term}");
            println!("transcript={transcript}");
            println!("false_positive_rank={rank}");
            for hit in ranked.into_iter().take(5) {
                println!(
                    "candidate={} span={} phonetic_score={:.3} coarse_score={:.3}",
                    hit.term,
                    hit.span_text,
                    hit.candidate.phonetic_score,
                    hit.candidate.coarse_score
                );
            }
        }
    }

    Ok(())
}

fn prepare_recording(
    g2p: &mut CachedEspeakG2p,
    row: &bee_phonetic::RecordingExampleRow,
    config: &EvalConfig,
) -> Result<(PreparedRecording, RecordingTimings), Box<dyn std::error::Error>> {
    let recording_start = Instant::now();
    let mut timings = RecordingTimings::default();
    let mut g2p_error = None;
    let enumerate_start = Instant::now();
    let spans = enumerate_transcript_spans_with::<_, TranscriptAlignmentToken>(
        &row.transcript,
        config.max_span_words,
        None,
        |text| {
            let g2p_start = Instant::now();
            let result = match g2p.ipa_tokens(text) {
                Ok(tokens) => tokens,
                Err(error) => {
                    g2p_error = Some(error.to_string());
                    None
                }
            };
            timings.g2p_total += g2p_start.elapsed();
            timings.g2p_calls += 1;
            if g2p.last_lookup_hit_cache() {
                timings.g2p_cache_hits += 1;
            } else {
                timings.g2p_cache_misses += 1;
            }
            result
        },
    );
    timings.enumerate_total += enumerate_start.elapsed();
    if let Some(error) = g2p_error {
        return Err(error.into());
    }
    timings.spans = spans.len();
    timings.recording_total = recording_start.elapsed();

    Ok((
        PreparedRecording {
            term: row.term.clone(),
            transcript: row.transcript.clone(),
            spans,
        },
        timings,
    ))
}

fn prepare_counterexample_recording(
    g2p: &mut CachedEspeakG2p,
    row: &CounterexampleRecordingRow,
    config: &EvalConfig,
) -> Result<(PreparedRecording, RecordingTimings), Box<dyn std::error::Error>> {
    let recording_start = Instant::now();
    let mut timings = RecordingTimings::default();
    let mut g2p_error = None;
    let enumerate_start = Instant::now();
    let spans = enumerate_transcript_spans_with::<_, TranscriptAlignmentToken>(
        &row.transcript,
        config.max_span_words,
        None,
        |text| {
            let g2p_start = Instant::now();
            let result = match g2p.ipa_tokens(text) {
                Ok(tokens) => tokens,
                Err(error) => {
                    g2p_error = Some(error.to_string());
                    None
                }
            };
            timings.g2p_total += g2p_start.elapsed();
            timings.g2p_calls += 1;
            if g2p.last_lookup_hit_cache() {
                timings.g2p_cache_hits += 1;
            } else {
                timings.g2p_cache_misses += 1;
            }
            result
        },
    );
    timings.enumerate_total += enumerate_start.elapsed();
    if let Some(error) = g2p_error {
        return Err(error.into());
    }
    timings.spans = spans.len();
    timings.recording_total = recording_start.elapsed();

    Ok((
        PreparedRecording {
            term: row.term.clone(),
            transcript: row.transcript.clone(),
            spans,
        },
        timings,
    ))
}

fn evaluate_prepared_recording(
    index: &bee_phonetic::PhoneticIndex,
    recording: &PreparedRecording,
    config: &EvalConfig,
) -> (PreparedRecording, Vec<RankedTermHit>, RecordingTimings) {
    let recording_start = Instant::now();
    let mut best_by_term: HashMap<String, RankedTermHit> = HashMap::new();

    let mut timings = RecordingTimings::default();

    for span in &recording.spans {
        let query_start = Instant::now();
        let shortlist = query_index(
            index,
            &RetrievalQuery {
                text: span.text.clone(),
                ipa_tokens: span.ipa_tokens.clone(),
                reduced_ipa_tokens: span.reduced_ipa_tokens.clone(),
                feature_tokens: bee_phonetic::feature_tokens_for_ipa(&span.ipa_tokens),
                token_count: (span.token_end - span.token_start) as u8,
            },
            config.shortlist_limit,
        );
        timings.query_total += query_start.elapsed();
        let verify_start = Instant::now();
        let verified = verify_shortlist(&span, &shortlist, index, config.verify_limit);
        timings.verify_total += verify_start.elapsed();

        for candidate in verified {
            let hit = RankedTermHit {
                term: candidate.term.clone(),
                span_text: span.text.clone(),
                candidate,
            };
            match best_by_term.get(&hit.term) {
                Some(existing) if compare_hits(existing, &hit).is_ge() => {}
                _ => {
                    best_by_term.insert(hit.term.clone(), hit);
                }
            }
        }
    }

    let mut ranked = best_by_term.into_values().collect::<Vec<_>>();
    ranked.sort_by(compare_hits);
    timings.recording_total = recording_start.elapsed();
    (recording.clone(), ranked, timings)
}

fn compare_hits(a: &RankedTermHit, b: &RankedTermHit) -> std::cmp::Ordering {
    b.candidate
        .acceptance_score
        .total_cmp(&a.candidate.acceptance_score)
        .then_with(|| {
            b.candidate
                .phonetic_score
                .total_cmp(&a.candidate.phonetic_score)
        })
        .then_with(|| {
            b.candidate
                .coarse_score
                .total_cmp(&a.candidate.coarse_score)
        })
        .then_with(|| b.candidate.qgram_overlap.cmp(&a.candidate.qgram_overlap))
}

fn parse_args() -> Result<EvalConfig, Box<dyn std::error::Error>> {
    let mut config = EvalConfig::default();
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--max-span-words" => {
                config.max_span_words = args
                    .next()
                    .ok_or("missing value for --max-span-words")?
                    .parse()?;
            }
            "--shortlist-limit" => {
                config.shortlist_limit = args
                    .next()
                    .ok_or("missing value for --shortlist-limit")?
                    .parse()?;
            }
            "--verify-limit" => {
                config.verify_limit = args
                    .next()
                    .ok_or("missing value for --verify-limit")?
                    .parse()?;
            }
            "--limit" => {
                config.sample_limit =
                    Some(args.next().ok_or("missing value for --limit")?.parse()?);
            }
            other => {
                return Err(format!("unknown argument: {other}").into());
            }
        }
    }

    Ok(config)
}

fn millis_per(duration: Duration, count: usize) -> f64 {
    if count == 0 {
        0.0
    } else {
        duration.as_secs_f64() * 1000.0 / count as f64
    }
}

fn eval_g2p_cache_path() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/phonetic-retrieval-eval-g2p-cache.tsv")
}

fn load_counterexample_recordings(
) -> Result<Vec<CounterexampleRecordingRow>, Box<dyn std::error::Error>> {
    let path = counterexample_recordings_path();
    let text = std::fs::read_to_string(&path)?;
    let mut out = Vec::new();
    for (line_idx, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row = serde_json::from_str::<CounterexampleRecordingRow>(line)
            .map_err(|error| format!("{}:{}: {error}", path.display(), line_idx + 1))?;
        out.push(row);
    }
    Ok(out)
}

fn counterexample_recordings_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../data/phonetic-seed/counterexample_recordings.jsonl")
}
