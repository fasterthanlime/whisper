use std::collections::HashMap;
use std::env;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use bee_phonetic::{
    enumerate_transcript_spans_with, query_index, score_shortlist, PhoneticIndex, RetrievalQuery,
    SeedDataset, TranscriptAlignmentToken, TranscriptSpan,
};
use bee_transcribe::{AlignedWord, Engine, EngineConfig, SessionOptions};
use beeml::g2p::CachedEspeakG2p;
use beeml::judge::{extract_span_context, OnlineJudge};
use beeml::rpc::{
    AcceptedEdit, AliasSource, BeeMl, CandidateFeatureDebug, CorrectionDebugResult,
    CorrectionRequest, CorrectionResult, FilterDecision, IdentifierFlags, JudgeEvalFailure,
    JudgeOptionDebug, JudgeStateDebug, ModelSummary, OfflineJudgeEvalRequest,
    OfflineJudgeEvalResult, OfflineJudgeFoldResult, ProbDistribution, RapidFireChoice,
    RapidFireComponent, RapidFireComponentChoice, RapidFireDecisionSet, RapidFireEdit,
    RejectedGroupSpan, RerankerDebugTrace, RetrievalCandidateDebug, RetrievalEvalMiss,
    RetrievalEvalTermSummary, RetrievalIndexView, RetrievalPrototypeEvalProgress,
    RetrievalPrototypeEvalRequest, RetrievalPrototypeEvalResult, RetrievalPrototypeProbeRequest,
    RetrievalPrototypeProbeResult, RetrievalPrototypeTeachingCase,
    RetrievalPrototypeTeachingDeckRequest, RetrievalPrototypeTeachingDeckResult, SpanDebugTrace,
    SpanDebugView, TeachRetrievalPrototypeJudgeRequest, TermAliasView, TermInspectionRequest,
    TermInspectionResult, ThresholdRow, TimingBreakdown, TranscribeWavResult, TwoStageGridPoint,
    TwoStageResult,
};
use serde::Deserialize;
use tokio::net::TcpListener;
use tracing::{debug, error, info, warn};
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::fmt::writer::MakeWriterExt;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};
use vox::{NoopClient, Rx, Tx};

fn init_tracing() -> Result<WorkerGuard> {
    let log_dir = env::var("BML_LOG_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../logs/beeml"));
    std::fs::create_dir_all(&log_dir)
        .with_context(|| format!("creating log directory {}", log_dir.display()))?;

    let file_appender = tracing_appender::rolling::daily(&log_dir, "beeml.log");
    let (file_writer, guard) = tracing_appender::non_blocking(file_appender);
    let stderr_writer = std::io::stderr.with_max_level(tracing::Level::INFO);
    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info,beeml=debug"));

    tracing_subscriber::registry()
        .with(env_filter)
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(stderr_writer)
                .with_ansi(true),
        )
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(file_writer)
                .with_ansi(false),
        )
        .try_init()
        .context("initializing tracing subscriber")?;

    info!(log_dir = %log_dir.display(), "tracing initialized");
    Ok(guard)
}

/// Collect admitted edits from span traces using retrieval/phonetic scores only.
/// The judge is NOT consulted here — choices are ranked by acceptance_score
/// so the user sees all phonetically plausible options regardless of judge state.
fn collect_admitted_edits(spans: &[SpanDebugTrace], is_counterexample: bool) -> Vec<EditCandidate> {
    let mut edits = Vec::new();
    for span in spans {
        let mut admitted_for_span = Vec::new();
        // Deduplicate candidates by term — keep best acceptance_score per term.
        let mut best_by_term: HashMap<String, &RetrievalCandidateDebug> = HashMap::new();
        for candidate in &span.candidates {
            match best_by_term.get(&candidate.term) {
                Some(existing)
                    if existing.features.acceptance_score
                        >= candidate.features.acceptance_score => {}
                _ => {
                    best_by_term.insert(candidate.term.clone(), candidate);
                }
            }
        }

        for candidate in best_by_term.values() {
            let acceptance_score = candidate.features.acceptance_score;
            let phonetic_score = candidate.features.phonetic_score;
            let accepted = if is_counterexample {
                acceptance_score >= 0.60 && phonetic_score >= 0.60
            } else {
                acceptance_score >= 0.30 && phonetic_score >= 0.30
            };
            if !accepted {
                continue;
            }
            // Skip identity edits
            if normalize_comparable_text(&candidate.term)
                == normalize_comparable_text(&span.span.text)
            {
                continue;
            }
            admitted_for_span.push(EditCandidate {
                span_token_start: span.span.token_start,
                span_token_end: span.span.token_end,
                span_text: span.span.text.clone(),
                alias_id: candidate.alias_id,
                replacement_text: candidate.term.clone(),
                score: acceptance_score,
                probability: acceptance_score,
                acceptance_score,
                phonetic_score,
                verified: candidate.features.verified,
            });
        }

        admitted_for_span.sort_by(|lhs, rhs| {
            rhs.acceptance_score
                .total_cmp(&lhs.acceptance_score)
                .then_with(|| rhs.phonetic_score.total_cmp(&lhs.phonetic_score))
        });
        admitted_for_span.truncate(RAPID_FIRE_MAX_EDITS_PER_SPAN);
        edits.extend(admitted_for_span);
    }
    dedupe_edit_candidates(edits)
}

fn dedupe_edit_candidates(edits: Vec<EditCandidate>) -> Vec<EditCandidate> {
    let mut best = HashMap::<String, EditCandidate>::new();
    for edit in edits {
        let key = format!(
            "{}:{}:{}",
            edit.span_token_start, edit.span_token_end, edit.alias_id
        );
        match best.get(&key) {
            Some(existing) if existing.acceptance_score >= edit.acceptance_score => {}
            _ => {
                best.insert(key, edit);
            }
        }
    }
    let mut edits = best.into_values().collect::<Vec<_>>();
    edits.sort_by(|lhs, rhs| {
        lhs.span_token_start
            .cmp(&rhs.span_token_start)
            .then_with(|| lhs.span_token_end.cmp(&rhs.span_token_end))
            .then_with(|| rhs.acceptance_score.total_cmp(&lhs.acceptance_score))
    });
    edits
}

/// Group edits by replacement term. Each term gets its own component with
/// different span hypotheses. This avoids transitive overlap chains where
/// "MacO"(6:7) and "must be"(12:14) get merged through intermediate spans.
fn build_conflict_components(edits: &[EditCandidate]) -> Vec<Vec<EditCandidate>> {
    let mut by_term: HashMap<String, Vec<EditCandidate>> = HashMap::new();
    for edit in edits {
        by_term
            .entry(edit.replacement_text.clone())
            .or_default()
            .push(edit.clone());
    }
    let mut groups: Vec<Vec<EditCandidate>> = by_term.into_values().collect();
    for group in &mut groups {
        group.sort_by(|lhs, rhs| {
            rhs.acceptance_score
                .total_cmp(&lhs.acceptance_score)
                .then_with(|| lhs.span_token_start.cmp(&rhs.span_token_start))
        });
    }
    groups.sort_by(|a, b| {
        let a_best = a.first().map(|e| e.acceptance_score).unwrap_or(0.0);
        let b_best = b.first().map(|e| e.acceptance_score).unwrap_or(0.0);
        b_best.total_cmp(&a_best)
    });
    groups
}

fn build_component(
    component_id: u32,
    edits: Vec<EditCandidate>,
    span_keep_probabilities: &HashMap<(u32, u32), f32>,
) -> Component {
    let component_spans = unique_component_spans(&edits);
    let atomic_edits = edits;

    // Keep hypothesis uses the max keep_original probability across the component's spans.
    let keep_probability = component_spans
        .iter()
        .filter_map(|s| span_keep_probabilities.get(&(s.token_start, s.token_end)))
        .copied()
        .fold(0.0f32, f32::max);
    let keep_hypothesis = ComponentHypothesis {
        component_id,
        component_spans: component_spans.clone(),
        choose_keep_original: true,
        edits: Vec::new(),
        score: keep_probability,
        probability: keep_probability,
    };
    // Each edit is a separate hypothesis (different span for the same term).
    let mut edit_hypotheses: Vec<ComponentHypothesis> = atomic_edits
        .iter()
        .map(|edit| ComponentHypothesis {
            component_id,
            component_spans: component_spans.clone(),
            choose_keep_original: false,
            edits: vec![edit.clone()],
            score: edit.acceptance_score,
            probability: edit.acceptance_score,
        })
        .collect();
    edit_hypotheses.sort_by(|lhs, rhs| {
        rhs.probability
            .total_cmp(&lhs.probability)
            .then_with(|| rhs.score.total_cmp(&lhs.score))
    });
    edit_hypotheses.dedup_by(|a, b| {
        a.edits.first().map(|e| e.span_token_start) == b.edits.first().map(|e| e.span_token_start)
            && a.edits.first().map(|e| e.span_token_end)
                == b.edits.first().map(|e| e.span_token_end)
    });
    // Reserve one slot for keep_original so it never gets truncated out.
    edit_hypotheses.truncate(RAPID_FIRE_COMPONENT_HYPOTHESES - 1);
    let mut all_hypotheses = edit_hypotheses;
    all_hypotheses.push(keep_hypothesis);

    Component {
        component_id,
        component_spans,
        hypotheses: all_hypotheses,
    }
}

#[derive(Clone)]
struct Component {
    component_id: u32,
    component_spans: Vec<RejectedGroupSpan>,
    hypotheses: Vec<ComponentHypothesis>,
}

fn compose_sentence_hypotheses(
    transcript: &str,
    components: &[Component],
    total_combinations: usize,
) -> (&'static str, Vec<SentenceHypothesis>) {
    if components.is_empty() {
        return (
            "exact",
            vec![SentenceHypothesis {
                components: Vec::new(),
                sentence: transcript.to_string(),
                score: 0.0,
                probability: 1.0,
            }],
        );
    }
    if total_combinations <= RAPID_FIRE_EXACT_THRESHOLD {
        (
            "exact",
            enumerate_sentence_hypotheses(transcript, components),
        )
    } else {
        ("beam", beam_sentence_hypotheses(transcript, components))
    }
}

fn enumerate_sentence_hypotheses(
    transcript: &str,
    components: &[Component],
) -> Vec<SentenceHypothesis> {
    fn recurse(
        transcript: &str,
        components: &[Component],
        index: usize,
        chosen: &mut Vec<ComponentHypothesis>,
        out: &mut Vec<SentenceHypothesis>,
    ) {
        if index == components.len() {
            if let Some(hypothesis) = build_sentence_hypothesis(transcript, chosen.clone()) {
                out.push(hypothesis);
            }
            return;
        }
        for hypothesis in &components[index].hypotheses {
            chosen.push(hypothesis.clone());
            recurse(transcript, components, index + 1, chosen, out);
            chosen.pop();
        }
    }
    let mut out = Vec::new();
    recurse(transcript, components, 0, &mut Vec::new(), &mut out);
    out
}

fn beam_sentence_hypotheses(transcript: &str, components: &[Component]) -> Vec<SentenceHypothesis> {
    let mut beam = vec![SentenceHypothesis {
        components: Vec::new(),
        sentence: transcript.to_string(),
        score: 0.0,
        probability: 0.0,
    }];
    for component in components {
        let mut next = Vec::new();
        for partial in &beam {
            for hypothesis in &component.hypotheses {
                let mut combined = partial.components.clone();
                combined.push(hypothesis.clone());
                if let Some(hypothesis) = build_sentence_hypothesis(transcript, combined) {
                    next.push(hypothesis);
                }
            }
        }
        next.sort_by(|lhs, rhs| {
            rhs.probability
                .total_cmp(&lhs.probability)
                .then_with(|| rhs.score.total_cmp(&lhs.score))
        });
        next.truncate(RAPID_FIRE_BEAM_WIDTH);
        beam = next;
    }
    beam
}

fn build_sentence_hypothesis(
    transcript: &str,
    components: Vec<ComponentHypothesis>,
) -> Option<SentenceHypothesis> {
    let edits = components
        .iter()
        .flat_map(|component| component.edits.iter().cloned())
        .collect::<Vec<_>>();
    // Check for overlapping edits across components — skip invalid combinations.
    for (i, a) in edits.iter().enumerate() {
        for b in edits.iter().skip(i + 1) {
            if edits_overlap(a, b) {
                return None;
            }
        }
    }
    let sentence = apply_atomic_edits(transcript, &edits);
    // Score by the weakest component (min, not average). This way a sentence
    // with two good edits (0.59, 0.55) scores 0.55 — which beats a sentence
    // with one good edit + keep (0.59, 0.82 keep → min 0.59, but the keep
    // component contributes no edits so its score is just the keep probability).
    // A sentence with no edits (all keep) gets probability 1.0.
    let num_edits = components
        .iter()
        .filter(|c| !c.choose_keep_original)
        .count();
    let probability = if num_edits == 0 {
        0.0 // all-keep is lowest priority in the choice list
    } else {
        // Min of edit component scores — weakest link
        components
            .iter()
            .filter(|c| !c.choose_keep_original)
            .map(|c| c.probability)
            .fold(f32::INFINITY, f32::min)
    };
    Some(SentenceHypothesis {
        components,
        sentence,
        score: probability,
        probability,
    })
}

fn apply_atomic_edits(transcript: &str, edits: &[EditCandidate]) -> String {
    let mut tokens = transcript
        .split_whitespace()
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    let mut edits = edits.to_vec();
    edits.sort_by(|lhs, rhs| rhs.span_token_start.cmp(&lhs.span_token_start));
    for edit in edits {
        let start = edit.span_token_start as usize;
        let end = edit.span_token_end as usize;
        if start > end || end > tokens.len() {
            continue;
        }
        let replacement_tokens = edit
            .replacement_text
            .split_whitespace()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        tokens.splice(start..end, replacement_tokens);
    }
    tokens.join(" ")
}

fn component_hypothesis_to_rpc(hypothesis: &ComponentHypothesis) -> RapidFireComponentChoice {
    let primary = hypothesis.edits.first();
    RapidFireComponentChoice {
        component_id: hypothesis.component_id,
        choose_keep_original: hypothesis.choose_keep_original,
        span_token_start: primary.map(|edit| edit.span_token_start),
        span_token_end: primary.map(|edit| edit.span_token_end),
        chosen_alias_id: primary.map(|edit| edit.alias_id),
        replaced_text: if hypothesis.edits.is_empty() {
            String::new()
        } else {
            hypothesis
                .edits
                .iter()
                .map(|edit| edit.span_text.clone())
                .collect::<Vec<_>>()
                .join(" + ")
        },
        replacement_text: if hypothesis.edits.is_empty() {
            "keep_original".to_string()
        } else {
            hypothesis
                .edits
                .iter()
                .map(|edit| edit.replacement_text.clone())
                .collect::<Vec<_>>()
                .join(" + ")
        },
        score: hypothesis.score,
        probability: hypothesis.probability,
        component_spans: hypothesis.component_spans.clone(),
    }
}

fn prune_sentence_hypotheses(
    hypotheses: Vec<SentenceHypothesis>,
    is_counterexample: bool,
) -> Vec<SentenceHypothesis> {
    if !is_counterexample {
        return hypotheses;
    }
    let mut kept = Vec::new();
    let mut keep_hypothesis = None;
    for hypothesis in hypotheses {
        if hypothesis
            .components
            .iter()
            .all(|component| component.choose_keep_original)
        {
            keep_hypothesis = Some(hypothesis);
            continue;
        }
        let strongest_acceptance = hypothesis
            .components
            .iter()
            .flat_map(|component| component.edits.iter())
            .map(|edit| edit.acceptance_score)
            .fold(0.0, f32::max);
        if strongest_acceptance >= 0.80 {
            kept.push(hypothesis);
        }
    }
    if let Some(keep_hypothesis) = keep_hypothesis {
        kept.push(keep_hypothesis);
    }
    kept.sort_by(|lhs, rhs| {
        rhs.probability
            .total_cmp(&lhs.probability)
            .then_with(|| rhs.score.total_cmp(&lhs.score))
    });
    kept
}

fn average_or_zero(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().copied().sum::<f32>() / values.len() as f32
    }
}

fn unique_component_spans(edits: &[EditCandidate]) -> Vec<RejectedGroupSpan> {
    let mut spans = edits
        .iter()
        .map(|edit| RejectedGroupSpan {
            token_start: edit.span_token_start,
            token_end: edit.span_token_end,
        })
        .collect::<Vec<_>>();
    spans.sort_by(|lhs, rhs| {
        lhs.token_start
            .cmp(&rhs.token_start)
            .then_with(|| lhs.token_end.cmp(&rhs.token_end))
    });
    spans.dedup_by(|lhs, rhs| lhs.token_start == rhs.token_start && lhs.token_end == rhs.token_end);
    spans
}

fn build_overlap_groups(spans: &[SpanDebugTrace]) -> Vec<Vec<SpanDebugTrace>> {
    let mut groups: Vec<Vec<SpanDebugTrace>> = Vec::new();
    for span in spans {
        let mut overlapping = Vec::new();
        for (index, group) in groups.iter().enumerate() {
            if group.iter().any(|candidate| spans_overlap(candidate, span)) {
                overlapping.push(index);
            }
        }
        if overlapping.is_empty() {
            groups.push(vec![span.clone()]);
            continue;
        }
        let mut merged = vec![span.clone()];
        for index in overlapping.into_iter().rev() {
            merged.extend(groups.swap_remove(index));
        }
        merged.sort_by(|lhs, rhs| {
            lhs.span
                .token_start
                .cmp(&rhs.span.token_start)
                .then_with(|| lhs.span.token_end.cmp(&rhs.span.token_end))
        });
        groups.push(merged);
    }
    groups
}

fn edits_overlap(lhs: &EditCandidate, rhs: &EditCandidate) -> bool {
    lhs.span_token_start < rhs.span_token_end && rhs.span_token_start < lhs.span_token_end
}

fn spans_overlap(a: &SpanDebugTrace, b: &SpanDebugTrace) -> bool {
    a.span.token_start < b.span.token_end && b.span.token_start < a.span.token_end
}

fn normalize_comparable_text(text: &str) -> String {
    let mut normalized = String::with_capacity(text.len());
    let mut last_was_space = false;
    for ch in text.chars().flat_map(|ch| ch.to_lowercase()) {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            normalized.push(ch);
            last_was_space = false;
        } else if !last_was_space {
            normalized.push(' ');
            last_was_space = true;
        }
    }
    normalized.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let _tracing_guard = init_tracing()?;
    let listen_addr = env::var("BML_WS_ADDR").unwrap_or_else(|_| "127.0.0.1:9944".to_string());
    let model_dir = env::var("BEE_ASR_MODEL_DIR")
        .map(PathBuf::from)
        .context("BEE_ASR_MODEL_DIR must be set")?;
    let tokenizer_dir = env::var("BEE_TOKENIZER_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| model_dir.clone());
    let aligner_dir = env::var("BEE_ALIGNER_DIR")
        .map(PathBuf::from)
        .context("BEE_ALIGNER_DIR must be set")?;

    info!(model_dir = %model_dir.display(), "loading ASR engine");
    let engine = Engine::load(&EngineConfig {
        model_dir: &model_dir,
        tokenizer_dir: &tokenizer_dir,
        aligner_dir: &aligner_dir,
    })
    .context("loading engine")?;

    let dataset =
        SeedDataset::load_canonical().context("loading canonical phonetic seed dataset")?;
    dataset
        .validate()
        .context("validating canonical phonetic seed dataset")?;
    let index = dataset.phonetic_index();
    let counterexamples =
        load_counterexample_recordings().context("loading counterexample phonetic recordings")?;

    let event_log_path = dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".beeml")
        .join("events.jsonl");

    let mut judge = OnlineJudge::default();
    // Replay correction events from previous sessions
    if event_log_path.exists() {
        match load_correction_events(&event_log_path) {
            Ok(events) => {
                info!(path = %event_log_path.display(), count = events.len(), "loaded correction events");
                judge.replay_events(events);
            }
            Err(e) => {
                tracing::warn!(path = %event_log_path.display(), error = %e, "failed to load correction events, starting fresh");
            }
        }
    }

    let handler = BeeMlService {
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
            judge: Mutex::new(judge),
            event_log_path,
        }),
    };

    // --offline-eval: run full Phase 4 eval suite and exit
    if std::env::args().any(|a| a == "--offline-eval") {
        let folds = std::env::args()
            .skip_while(|a| a != "--folds")
            .nth(1)
            .and_then(|v| v.parse().ok())
            .unwrap_or(5u32);
        let epochs = std::env::args()
            .skip_while(|a| a != "--epochs")
            .nth(1)
            .and_then(|v| v.parse().ok())
            .unwrap_or(4u32);

        handler
            .run_offline_judge_eval(OfflineJudgeEvalRequest {
                folds,
                max_span_words: 3,
                shortlist_limit: 100,
                verify_limit: 20,
                train_epochs: epochs,
            })
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        return Ok(());
    }

    let listener = TcpListener::bind(&listen_addr)
        .await
        .with_context(|| format!("binding websocket listener on {listen_addr}"))?;

    info!(listen_addr, "beeml vox websocket server listening");

    loop {
        let (stream, peer_addr) = listener
            .accept()
            .await
            .context("accepting websocket socket")?;
        let handler = handler.clone();

        tokio::spawn(async move {
            let link = match vox_websocket::WsLink::server(stream).await {
                Ok(link) => link,
                Err(error) => {
                    warn!(%peer_addr, error = %error, "websocket handshake failed");
                    return;
                }
            };

            let establish = vox_core::acceptor_on(link)
                .on_connection(beeml::rpc::BeeMlDispatcher::new(handler))
                .establish::<NoopClient>()
                .await;

            match establish {
                Ok(client) => {
                    info!(%peer_addr, "client connected");
                    client.caller.closed().await;
                    info!(%peer_addr, "client disconnected");
                }
                Err(error) => {
                    error!(%peer_addr, error = %error, "vox session establish failed");
                }
            }
        });
    }
}
