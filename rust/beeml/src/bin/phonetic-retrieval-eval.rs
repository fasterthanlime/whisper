use std::collections::HashMap;
use std::time::{Duration, Instant};

use bee_phonetic::{
    enumerate_transcript_spans_with, query_index, verify_shortlist, RetrievalQuery, SeedDataset,
    TranscriptAlignmentToken, VerifiedCandidate,
};
use beeml::g2p::CachedEspeakG2p;

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

#[derive(Debug, Default)]
struct EvalSummary {
    total: usize,
    hit_top1: usize,
    hit_top3: usize,
    hit_top10: usize,
    miss: usize,
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
    let mut timings = EvalTimings::default();
    let mut misses = Vec::new();

    for row in dataset
        .recording_examples
        .iter()
        .take(config.sample_limit.unwrap_or(usize::MAX))
    {
        summary.total += 1;

        let (ranked, recording_timings) = evaluate_recording(&index, &mut g2p, row, &config)?;
        timings.recordings += 1;
        timings.spans += recording_timings.spans;
        timings.g2p_calls += recording_timings.g2p_calls;
        timings.g2p_cache_hits += recording_timings.g2p_cache_hits;
        timings.g2p_cache_misses += recording_timings.g2p_cache_misses;
        timings.enumerate_total += recording_timings.enumerate_total;
        timings.g2p_total += recording_timings.g2p_total;
        timings.query_total += recording_timings.query_total;
        timings.verify_total += recording_timings.verify_total;
        timings.recording_total += recording_timings.recording_total;
        let target_rank = ranked
            .iter()
            .position(|hit| hit.term.eq_ignore_ascii_case(&row.term))
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
                    row.term.clone(),
                    row.transcript.clone(),
                    ranked.into_iter().take(5).collect::<Vec<_>>(),
                ));
            }
        }
    }

    println!("recordings={}", summary.total);
    println!("top1={}", summary.hit_top1);
    println!("top3={}", summary.hit_top3);
    println!("top10={}", summary.hit_top10);
    println!("miss={}", summary.miss);
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

    Ok(())
}

fn evaluate_recording(
    index: &bee_phonetic::PhoneticIndex,
    g2p: &mut CachedEspeakG2p,
    row: &bee_phonetic::RecordingExampleRow,
    config: &EvalConfig,
) -> Result<(Vec<RankedTermHit>, RecordingTimings), Box<dyn std::error::Error>> {
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

    let mut best_by_term: HashMap<String, RankedTermHit> = HashMap::new();

    for span in spans {
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
    Ok((ranked, timings))
}

fn compare_hits(a: &RankedTermHit, b: &RankedTermHit) -> std::cmp::Ordering {
    b.candidate
        .phonetic_score
        .total_cmp(&a.candidate.phonetic_score)
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
