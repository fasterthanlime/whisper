use std::collections::HashMap;
use std::process::Command;

use bee_phonetic::{
    enumerate_transcript_spans_with, parse_reviewed_ipa, query_index, verify_shortlist,
    RetrievalQuery, SeedDataset, TranscriptAlignmentToken, VerifiedCandidate,
};

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args()?;
    let dataset = SeedDataset::load_canonical()?;
    dataset.validate()?;

    let index = dataset.phonetic_index();
    let mut g2p = EspeakG2p::default();
    let mut summary = EvalSummary::default();
    let mut misses = Vec::new();

    for row in dataset
        .recording_examples
        .iter()
        .take(config.sample_limit.unwrap_or(usize::MAX))
    {
        summary.total += 1;

        let ranked = evaluate_recording(&index, &mut g2p, row, &config)?;
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
    g2p: &mut EspeakG2p,
    row: &bee_phonetic::RecordingExampleRow,
    config: &EvalConfig,
) -> Result<Vec<RankedTermHit>, Box<dyn std::error::Error>> {
    let mut g2p_error = None;
    let spans = enumerate_transcript_spans_with::<_, TranscriptAlignmentToken>(
        &row.transcript,
        config.max_span_words,
        None,
        |text| match g2p.ipa_tokens(text) {
            Ok(tokens) => tokens,
            Err(error) => {
                g2p_error = Some(error.to_string());
                None
            }
        },
    );
    if let Some(error) = g2p_error {
        return Err(error.into());
    }

    let mut best_by_term: HashMap<String, RankedTermHit> = HashMap::new();

    for span in spans {
        let shortlist = query_index(
            index,
            &RetrievalQuery {
                text: span.text.clone(),
                ipa_tokens: span.ipa_tokens.clone(),
                reduced_ipa_tokens: span.reduced_ipa_tokens.clone(),
                token_count: (span.token_end - span.token_start) as u8,
            },
            config.shortlist_limit,
        );
        let verified = verify_shortlist(&span, &shortlist, index, config.verify_limit);

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
    Ok(ranked)
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

#[derive(Default)]
struct EspeakG2p {
    cache: HashMap<String, Vec<String>>,
}

impl EspeakG2p {
    fn ipa_tokens(
        &mut self,
        text: &str,
    ) -> Result<Option<Vec<String>>, Box<dyn std::error::Error>> {
        let key = text.trim();
        if key.is_empty() {
            return Ok(None);
        }
        if let Some(tokens) = self.cache.get(key) {
            return Ok(Some(tokens.clone()));
        }

        let output = Command::new("espeak-ng")
            .args(["-q", "--ipa=3", "--sep= "])
            .arg(key)
            .output()?;
        if !output.status.success() {
            return Err(format!("espeak-ng failed for '{key}'").into());
        }

        let ipa = String::from_utf8(output.stdout)?.trim().to_string();
        let tokens = parse_reviewed_ipa(&ipa);
        if tokens.is_empty() {
            return Ok(None);
        }

        self.cache.insert(key.to_string(), tokens.clone());
        Ok(Some(tokens))
    }
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
