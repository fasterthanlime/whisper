use std::collections::HashMap;
use std::path::PathBuf;

use bee_phonetic::{
    enumerate_transcript_spans_with, feature_tokens_for_ipa, query_index, score_shortlist,
    CandidateFeatureRow, LexiconAlias, RetrievalQuery, SeedDataset, TranscriptAlignmentToken,
    TranscriptSpan,
};
use beeml::g2p::CachedEspeakG2p;
use serde::Deserialize;

#[derive(Debug, Clone)]
struct Config {
    term: String,
    max_span_words: usize,
    shortlist_limit: usize,
    verify_limit: usize,
    recordings_limit: usize,
    counterexamples: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            term: String::new(),
            max_span_words: 4,
            shortlist_limit: 10,
            verify_limit: 10,
            recordings_limit: 3,
            counterexamples: false,
        }
    }
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

#[derive(Debug, Clone)]
struct DebugRecording {
    text: String,
    transcript: String,
    surface_form: Option<String>,
    take: Option<i64>,
    audio_path: Option<String>,
}

#[derive(Debug)]
struct SpanCaseDebug {
    span: TranscriptSpan,
    shortlist: Vec<bee_phonetic::RetrievalCandidate>,
    scored: Vec<CandidateFeatureRow>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args()?;
    if config.term.trim().is_empty() {
        return Err("--term is required".into());
    }

    let dataset = SeedDataset::load_canonical()?;
    dataset.validate()?;
    let index = dataset.phonetic_index();
    let mut g2p = CachedEspeakG2p::english_with_persist_path(Some(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/phonetic-retrieval-eval-g2p-cache.tsv"),
    ))?;

    let term_aliases = index
        .aliases
        .iter()
        .filter(|alias| alias.term.eq_ignore_ascii_case(&config.term))
        .collect::<Vec<_>>();

    println!("term={}", config.term);
    println!("target_aliases={}", term_aliases.len());
    for alias in &term_aliases {
        println!("--- alias");
        println!("text={}", alias.alias_text);
        println!("source={:?}", alias.alias_source);
        println!("ipa={}", alias.ipa_tokens.join(" "));
        println!("reduced={}", alias.reduced_ipa_tokens.join(" "));
        println!("features={}", alias.feature_tokens.join(" "));
        println!(
            "flags=acronym:{} digits:{} snake:{} camel:{} symbol:{}",
            alias.identifier_flags.acronym_like,
            alias.identifier_flags.has_digits,
            alias.identifier_flags.snake_like,
            alias.identifier_flags.camel_like,
            alias.identifier_flags.symbol_like
        );
    }

    let rows = if config.counterexamples {
        load_counterexample_recordings()?
            .into_iter()
            .filter(|row| row.term.eq_ignore_ascii_case(&config.term))
            .take(config.recordings_limit)
            .map(|row| DebugRecording {
                text: row.text,
                transcript: row.transcript,
                surface_form: Some(row.surface_form),
                take: Some(row.take),
                audio_path: Some(row.audio_path),
            })
            .collect::<Vec<_>>()
    } else {
        dataset
            .recording_examples
            .iter()
            .filter(|row| row.term.eq_ignore_ascii_case(&config.term))
            .take(config.recordings_limit)
            .map(|row| DebugRecording {
                text: row.text.clone(),
                transcript: row.transcript.clone(),
                surface_form: None,
                take: Some(row.take),
                audio_path: Some(row.audio_path.clone()),
            })
            .collect::<Vec<_>>()
    };

    println!("recordings={}", rows.len());
    println!(
        "source={}",
        if config.counterexamples {
            "counterexamples"
        } else {
            "canonical"
        }
    );

    for (idx, row) in rows.into_iter().enumerate() {
        println!();
        println!("===== recording {} =====", idx + 1);
        println!("text={}", row.text);
        println!("transcript={}", row.transcript);
        if let Some(surface_form) = &row.surface_form {
            println!("surface_form={surface_form}");
        }
        if let Some(take) = row.take {
            println!("take={take}");
        }
        if let Some(audio_path) = &row.audio_path {
            println!("audio_path={audio_path}");
        }

        let spans = enumerate_transcript_spans_with::<_, TranscriptAlignmentToken>(
            &row.transcript,
            config.max_span_words,
            None,
            |text| g2p.ipa_tokens(text).ok().flatten(),
        );

        let mut cases = spans
            .into_iter()
            .map(|span| {
                let shortlist = query_index(
                    &index,
                    &RetrievalQuery {
                        text: span.text.clone(),
                        ipa_tokens: span.ipa_tokens.clone(),
                        reduced_ipa_tokens: span.reduced_ipa_tokens.clone(),
                        feature_tokens: bee_phonetic::feature_tokens_for_ipa(&span.ipa_tokens),
                        token_count: (span.token_end - span.token_start) as u8,
                    },
                    config.shortlist_limit,
                );
                let scored = score_shortlist(&span, &shortlist, &index);
                SpanCaseDebug {
                    span,
                    shortlist,
                    scored,
                }
            })
            .collect::<Vec<_>>();

        cases.sort_by(|a, b| score_case(b, &config.term).total_cmp(&score_case(a, &config.term)));

        for case in cases.iter().take(12) {
            print_case(case, &config.term, &index.aliases);
        }
    }

    Ok(())
}

fn score_case(case: &SpanCaseDebug, term: &str) -> f32 {
    let target_verified = case
        .scored
        .iter()
        .find(|candidate| candidate.verified && candidate.term.eq_ignore_ascii_case(term))
        .map(|candidate| {
            1000.0 + candidate.acceptance_score * 100.0 + candidate.phonetic_score * 10.0
        });
    if let Some(score) = target_verified {
        return score;
    }

    let target_shortlist = case
        .shortlist
        .iter()
        .find(|candidate| candidate.term.eq_ignore_ascii_case(term))
        .map(|candidate| 500.0 + candidate.coarse_score * 10.0);
    if let Some(score) = target_shortlist {
        return score;
    }

    case.scored
        .first()
        .map(|candidate| candidate.acceptance_score * 100.0 + candidate.phonetic_score * 10.0)
        .or_else(|| {
            case.shortlist
                .first()
                .map(|candidate| candidate.coarse_score * 10.0)
        })
        .unwrap_or(0.0)
}

fn print_case(case: &SpanCaseDebug, term: &str, aliases: &[LexiconAlias]) {
    let target_in_shortlist = case
        .shortlist
        .iter()
        .any(|candidate| candidate.term.eq_ignore_ascii_case(term));
    let target_in_verified = case
        .scored
        .iter()
        .any(|candidate| candidate.verified && candidate.term.eq_ignore_ascii_case(term));

    println!("--- span");
    println!(
        "text={} tokens={}:{} target_shortlist={} target_verified={}",
        case.span.text,
        case.span.token_start,
        case.span.token_end,
        target_in_shortlist,
        target_in_verified
    );
    println!("ipa={}", case.span.ipa_tokens.join(" "));
    println!("reduced={}", case.span.reduced_ipa_tokens.join(" "));
    println!(
        "features={}",
        feature_tokens_for_ipa(&case.span.ipa_tokens).join(" ")
    );

    let scored_by_alias = case
        .scored
        .iter()
        .map(|candidate| (candidate.alias_id, candidate))
        .collect::<HashMap<_, _>>();

    for candidate in case.shortlist.iter().take(6) {
        let alias = &aliases[candidate.alias_id as usize];
        println!(
            "  candidate term={} alias={}",
            candidate.term, candidate.alias_text
        );
        println!("    alias_ipa={}", alias.ipa_tokens.join(" "));
        println!("    alias_reduced={}", alias.reduced_ipa_tokens.join(" "));
        println!("    alias_features={}", alias.feature_tokens.join(" "));
        println!(
            "    source={:?} lane={:?} q={} q_total={} coarse={:.3} best_view={:.3} support={} token_bonus={:.2} phone_bonus={:.2} length_penalty={:.2}",
            candidate.alias_source,
            candidate.matched_view,
            candidate.qgram_overlap,
            candidate.total_qgram_overlap,
            candidate.coarse_score,
            candidate.best_view_score,
            candidate.cross_view_support,
            candidate.token_bonus,
            candidate.phone_bonus,
            candidate.extra_length_penalty
        );
        if let Some(verified) = scored_by_alias.get(&candidate.alias_id) {
            println!(
                "    verify token={:.3} ({:.3}/{}) feature={:.3} ({:.3}/{}) bonus={:.3} used_feature_bonus={} phonetic={:.3} accept={:.3} verified={}",
                verified.token_score,
                verified.token_weighted_distance,
                verified.token_max_len,
                verified.feature_score,
                verified.feature_weighted_distance,
                verified.feature_max_len,
                verified.feature_bonus,
                verified.used_feature_bonus,
                verified.phonetic_score,
                verified.acceptance_score,
                verified.verified
            );
            println!(
                "    token_distance raw={} boundary_penalty={:.2}",
                verified.token_distance, verified.token_boundary_penalty
            );
            if !verified.token_ops.is_empty() {
                println!(
                    "    token_ops={}",
                    verified
                        .token_ops
                        .iter()
                        .map(format_token_op)
                        .collect::<Vec<_>>()
                        .join(" | ")
                );
            }
            println!(
                "    feature_distance raw={:.3} boundary_penalty={:.2}",
                verified.feature_distance, verified.feature_boundary_penalty
            );
            if !verified.feature_ops.is_empty() {
                println!(
                    "    feature_ops={}",
                    verified
                        .feature_ops
                        .iter()
                        .map(format_feature_op)
                        .collect::<Vec<_>>()
                        .join(" | ")
                );
            }
            println!(
                "    feature_gate token_ok={} coarse_ok={} phone_ok={}",
                verified.feature_gate_token_ok,
                verified.feature_gate_coarse_ok,
                verified.feature_gate_phone_ok
            );
            println!(
                "    short_guard applied={} onset_match={} passed={}",
                verified.short_guard_applied,
                verified.short_guard_onset_match,
                verified.short_guard_passed
            );
            println!(
                "    low_content_guard applied={} passed={}",
                verified.low_content_guard_applied, verified.low_content_guard_passed
            );
            println!(
                "    acceptance_floor passed={} structure_bonus={:.2}",
                verified.acceptance_floor_passed, verified.structure_bonus
            );
        }
    }
}

fn format_token_op(op: &bee_phonetic::TokenEditOp) -> String {
    let kind = match op.kind {
        bee_phonetic::prototype::TokenEditKind::Match => "=",
        bee_phonetic::prototype::TokenEditKind::Substitute => "~",
        bee_phonetic::prototype::TokenEditKind::Insert => "+",
        bee_phonetic::prototype::TokenEditKind::Delete => "-",
    };
    format!(
        "{kind}({}->{}) c={:.2} bp={:.2}",
        op.left.as_deref().unwrap_or("_"),
        op.right.as_deref().unwrap_or("_"),
        op.cost,
        op.boundary_penalty
    )
}

fn format_feature_op(op: &bee_phonetic::FeatureEditOp) -> String {
    let kind = match op.kind {
        bee_phonetic::feature_view::FeatureEditKind::Match => "=",
        bee_phonetic::feature_view::FeatureEditKind::Substitute => "~",
        bee_phonetic::feature_view::FeatureEditKind::Insert => "+",
        bee_phonetic::feature_view::FeatureEditKind::Delete => "-",
    };
    format!(
        "{kind}({}->{}) c={:.2} bp={:.2}",
        op.left.as_deref().unwrap_or("_"),
        op.right.as_deref().unwrap_or("_"),
        op.cost,
        op.boundary_penalty
    )
}

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let mut config = Config::default();
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--term" => {
                config.term = args.next().ok_or("missing value for --term")?;
            }
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
            "--recordings" => {
                config.recordings_limit = args
                    .next()
                    .ok_or("missing value for --recordings")?
                    .parse()?;
            }
            "--counterexamples" => {
                config.counterexamples = true;
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    Ok(config)
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
