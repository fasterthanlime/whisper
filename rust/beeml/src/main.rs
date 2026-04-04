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
use bee_transcribe::{Engine, EngineConfig, SessionOptions};
use beeml::g2p::CachedEspeakG2p;
use beeml::judge::OnlineJudge;
use beeml::rpc::{
    AcceptedEdit, AliasSource, BeeMl, CandidateFeatureDebug, CorrectionDebugResult,
    CorrectionRequest, CorrectionResult, FilterDecision, IdentifierFlags, RerankerDebugTrace,
    JudgeEvalFailure, JudgeOptionDebug, JudgeStateDebug, RapidFireChoice, RapidFireComponent,
    RapidFireComponentChoice, RapidFireDecisionSet, RapidFireEdit,
    RejectedGroupSpan,
    RetrievalEvalMiss, RetrievalEvalTermSummary, RetrievalPrototypeEvalProgress,
    RetrievalPrototypeTeachingCase,
    RetrievalPrototypeTeachingDeckRequest, RetrievalPrototypeTeachingDeckResult,
    RetrievalCandidateDebug, RetrievalIndexView, RetrievalPrototypeEvalRequest,
    RetrievalPrototypeEvalResult, RetrievalPrototypeProbeRequest,
    RetrievalPrototypeProbeResult, SpanDebugTrace, SpanDebugView, TeachRetrievalPrototypeJudgeRequest,
    TermAliasView, TermInspectionRequest, TermInspectionResult, TimingBreakdown,
    TranscribeWavResult,
};
use serde::Deserialize;
use tokio::net::TcpListener;
use tracing::{debug, error, info, warn};
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::fmt::writer::MakeWriterExt;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};
use vox::{NoopClient, Rx, Tx};

#[derive(Clone)]
struct BeeMlService {
    inner: Arc<BeemlServiceInner>,
}

struct BeemlServiceInner {
    engine: Engine,
    index: PhoneticIndex,
    dataset: SeedDataset,
    counterexamples: Vec<CounterexampleRecordingRow>,
    g2p: Mutex<CachedEspeakG2p>,
    judge: Mutex<OnlineJudge>,
}

#[derive(Clone, Debug, Deserialize)]
struct CounterexampleRecordingRow {
    term: String,
    text: String,
    take: i64,
    audio_path: String,
    transcript: String,
    surface_form: String,
}

#[derive(Clone)]
struct EvalCase {
    case_id: String,
    suite: &'static str,
    target_term: String,
    source_text: String,
    transcript: String,
    should_abstain: bool,
    take: Option<i64>,
    audio_path: Option<String>,
    surface_form: Option<String>,
}

fn init_tracing() -> Result<WorkerGuard> {
    let log_dir = env::var("BML_LOG_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../logs/beeml"));
    std::fs::create_dir_all(&log_dir)
        .with_context(|| format!("creating log directory {}", log_dir.display()))?;

    let file_appender = tracing_appender::rolling::daily(&log_dir, "beeml.log");
    let (file_writer, guard) = tracing_appender::non_blocking(file_appender);
    let stderr_writer = std::io::stderr.with_max_level(tracing::Level::INFO);
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,beeml=debug"));

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

impl BeeMlService {
    fn run_probe(
        &self,
        request: RetrievalPrototypeProbeRequest,
        teach: Option<TeachRetrievalPrototypeJudgeRequest>,
    ) -> Result<RetrievalPrototypeProbeResult, String> {
        let expected_source_text = request.expected_source_text.clone();
        let alignments = if request.words.is_empty() {
            None
        } else {
            Some(
                request
                    .words
                    .iter()
                    .map(|word| TranscriptAlignmentToken {
                        start_time: word.start,
                        end_time: word.end,
                    })
                    .collect::<Vec<_>>(),
            )
        };

        let spans = {
            let mut g2p = self
                .inner
                .g2p
                .lock()
                .map_err(|_| "g2p cache mutex poisoned".to_string())?;
            enumerate_transcript_spans_with(
                &request.transcript,
                request.max_span_words as usize,
                alignments.as_deref(),
                |text| g2p.ipa_tokens(text).ok().flatten(),
            )
        };

        let mut judge = self
            .inner
            .judge
            .lock()
            .map_err(|_| "judge mutex poisoned".to_string())?;
        let mut taught_span = teach.is_none();
        let selected_component_choices = teach
            .as_ref()
            .map(|teach| teach.selected_component_choices.clone())
            .unwrap_or_default();
        let rejected_group_spans = teach
            .as_ref()
            .filter(|teach| teach.reject_group)
            .map(|teach| {
                teach
                    .rejected_group_spans
                    .iter()
                    .map(|span| (span.token_start as usize, span.token_end as usize))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let traces = spans
            .into_iter()
            .map(|span| {
                let shortlist = query_index(
                    &self.inner.index,
                    &RetrievalQuery {
                        text: span.text.clone(),
                        ipa_tokens: span.ipa_tokens.clone(),
                        reduced_ipa_tokens: span.reduced_ipa_tokens.clone(),
                        feature_tokens: bee_phonetic::feature_tokens_for_ipa(&span.ipa_tokens),
                        token_count: (span.token_end - span.token_start) as u8,
                    },
                    request.shortlist_limit.max(50) as usize,
                );
                let scored_rows = score_shortlist(&span, &shortlist, &self.inner.index);
                let judge_input = scored_rows
                    .iter()
                    .map(|row| {
                        let alias = &self.inner.index.aliases[row.alias_id as usize];
                        (row.clone(), alias.identifier_flags.clone())
                    })
                    .collect::<Vec<_>>();

                let explicitly_chosen_span = teach.as_ref().is_some_and(|teach| {
                    !teach.reject_group
                        && teach.selected_component_choices.is_empty()
                        && teach.span_token_start as usize == span.token_start
                        && teach.span_token_end as usize == span.token_end
                });
                let selected_component_choice = selected_component_choices.iter().find(|choice| {
                    if choice.choose_keep_original {
                        choice.component_spans.iter().any(|component_span| {
                            component_span.token_start as usize == span.token_start
                                && component_span.token_end as usize == span.token_end
                        })
                    } else {
                        choice.span_token_start.is_some_and(|start| start as usize == span.token_start)
                            && choice.span_token_end.is_some_and(|end| end as usize == span.token_end)
                    }
                });
                let rejected_group_match = rejected_group_spans
                    .iter()
                    .any(|(start, end)| *start == span.token_start && *end == span.token_end);
                let should_teach =
                    explicitly_chosen_span || rejected_group_match || selected_component_choice.is_some();
                if should_teach {
                    taught_span = true;
                }

                let chosen_alias_id = selected_component_choice
                    .and_then(|choice| if choice.choose_keep_original { None } else { choice.chosen_alias_id })
                    .or_else(|| teach.as_ref().and_then(|teach| {
                    if explicitly_chosen_span && !teach.choose_keep_original {
                        teach.chosen_alias_id
                    } else {
                        None
                    }
                }));
                let judge_options = if should_teach {
                    judge.teach_choice(&span, &judge_input, chosen_alias_id)
                } else {
                    judge.score_candidates(&span, &judge_input)
                };

                let candidates = scored_rows
                    .iter()
                    .map(|scored| {
                        let alias = &self.inner.index.aliases[scored.alias_id as usize];
                        RetrievalCandidateDebug {
                            alias_id: scored.alias_id,
                            term: scored.term.clone(),
                            alias_text: scored.alias_text.clone(),
                            alias_source: map_alias_source(scored.alias_source),
                            alias_ipa_tokens: alias.ipa_tokens.clone(),
                            alias_reduced_ipa_tokens: alias.reduced_ipa_tokens.clone(),
                            alias_feature_tokens: alias.feature_tokens.clone(),
                            identifier_flags: map_identifier_flags(&alias.identifier_flags),
                            features: map_candidate_features(scored),
                            filter_decisions: vec![
                                FilterDecision {
                                    name: "feature_gate".to_string(),
                                    passed: scored.feature_gate_token_ok
                                        && scored.feature_gate_coarse_ok
                                        && scored.feature_gate_phone_ok,
                                    detail: format!(
                                        "token_ok={} coarse_ok={} phone_ok={}",
                                        scored.feature_gate_token_ok,
                                        scored.feature_gate_coarse_ok,
                                        scored.feature_gate_phone_ok
                                    ),
                                },
                                FilterDecision {
                                    name: "short_guard".to_string(),
                                    passed: scored.short_guard_passed,
                                    detail: format!(
                                        "applied={} onset_match={}",
                                        scored.short_guard_applied, scored.short_guard_onset_match
                                    ),
                                },
                                FilterDecision {
                                    name: "low_content_guard".to_string(),
                                    passed: scored.low_content_guard_passed,
                                    detail: format!("applied={}", scored.low_content_guard_applied),
                                },
                                FilterDecision {
                                    name: "acceptance_floor".to_string(),
                                    passed: scored.acceptance_floor_passed,
                                    detail: format!(
                                        "accept={:.3} phonetic={:.3} coarse={:.3}",
                                        scored.acceptance_score,
                                        scored.phonetic_score,
                                        scored.coarse_score
                                    ),
                                },
                                FilterDecision {
                                    name: "verified".to_string(),
                                    passed: scored.verified,
                                    detail: if scored.verified {
                                        "candidate survives verifier".to_string()
                                    } else {
                                        "candidate rejected by verifier".to_string()
                                    },
                                },
                            ],
                            reached_reranker: false,
                            accepted: scored.verified,
                        }
                    })
                    .collect::<Vec<_>>();

                SpanDebugTrace {
                    span: SpanDebugView {
                        token_start: span.token_start as u32,
                        token_end: span.token_end as u32,
                        char_start: span.char_start as u32,
                        char_end: span.char_end as u32,
                        start_sec: span.start_sec.unwrap_or(0.0),
                        end_sec: span.end_sec.unwrap_or(0.0),
                        text: span.text,
                        feature_tokens: bee_phonetic::feature_tokens_for_ipa(&span.ipa_tokens),
                        ipa_tokens: span.ipa_tokens,
                        reduced_ipa_tokens: span.reduced_ipa_tokens,
                    },
                    candidates,
                    judge_options: judge_options
                        .into_iter()
                        .map(|option| JudgeOptionDebug {
                            alias_id: option.alias_id,
                            term: option.term,
                            is_keep_original: option.is_keep_original,
                            score: option.score,
                            probability: option.probability,
                            chosen: option.chosen,
                        })
                        .collect(),
                }
            })
            .collect::<Vec<_>>();

        if let Some(teach) = &teach {
            let sentence_level_keep = teach.choose_keep_original
                && !teach.reject_group
                && teach.selected_component_choices.is_empty();
            if !taught_span && !sentence_level_keep {
                return Err("requested span was not present in the probe result".to_string());
            }
            if teach.reject_group && rejected_group_spans.is_empty() {
                return Err("reject_group requested without any rejected_group_spans".to_string());
            }
            if !teach.selected_component_choices.is_empty() && !teach.reject_group && !taught_span {
                return Err("selected_component_choices did not match any spans in the probe result".to_string());
            }
        }

        let rapid_fire = expected_source_text
            .as_deref()
            .map(|expected| build_rapid_fire_decision_set(&request.transcript, &traces, expected));

        Ok(RetrievalPrototypeProbeResult {
            transcript: request.transcript,
            spans: traces,
            timings: TimingBreakdown {
                span_enumeration_ms: 0,
                retrieval_ms: 0,
                verify_ms: 0,
                rerank_ms: 0,
                total_ms: 0,
            },
            judge_state: JudgeStateDebug {
                update_count: judge.update_count(),
                learning_rate: 0.25,
                feature_names: judge.feature_names(),
                weights: judge.weights(),
            },
            rapid_fire,
        })
    }

    fn teaching_cases(
        &self,
        limit: usize,
        include_counterexamples: bool,
    ) -> Vec<EvalCase> {
        let mut cases = self
            .inner
            .dataset
            .recording_examples
            .iter()
            .enumerate()
            .map(|(index, row)| EvalCase {
                case_id: format!("canonical:{index}"),
                suite: "canonical",
                target_term: row.term.clone(),
                source_text: row.text.clone(),
                transcript: row.transcript.clone(),
                should_abstain: false,
                take: Some(row.take),
                audio_path: Some(row.audio_path.clone()),
                surface_form: None,
            })
            .collect::<Vec<_>>();

        if include_counterexamples {
            cases.extend(
                self.inner
                    .counterexamples
                    .iter()
                    .enumerate()
                    .map(|(index, row)| EvalCase {
                        case_id: format!("counterexample:{index}"),
                        suite: "counterexample",
                        target_term: row.term.clone(),
                        source_text: row.text.clone(),
                        transcript: row.transcript.clone(),
                        should_abstain: true,
                        take: Some(row.take),
                        audio_path: Some(row.audio_path.clone()),
                        surface_form: Some(row.surface_form.clone()),
                    }),
            );
        }

        let random_seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos() as u64)
            .unwrap_or(0);
        cases.sort_by_key(|case| randomish_case_key(case, random_seed));

        if limit == 0 || limit >= cases.len() {
            cases
        } else {
            cases.into_iter().take(limit).collect()
        }
    }

    fn evaluate_case(
        &self,
        case: &EvalCase,
        request: &RetrievalPrototypeEvalRequest,
    ) -> Result<CaseEvalResult, String> {
        // Use the same pipeline as run_probe — one path for everything.
        let probe_result = self.run_probe(
            RetrievalPrototypeProbeRequest {
                transcript: case.transcript.clone(),
                words: Vec::new(),
                max_span_words: request.max_span_words,
                shortlist_limit: request.shortlist_limit,
                verify_limit: request.verify_limit,
                expected_source_text: None,
            },
            None,
        )?;

        // Extract retrieval rank: best verified candidate per term across all spans.
        let mut best_by_term: HashMap<String, f32> = HashMap::new();
        for span_trace in &probe_result.spans {
            for candidate in &span_trace.candidates {
                if !candidate.accepted {
                    continue;
                }
                let score = candidate.features.acceptance_score;
                match best_by_term.get(&candidate.term) {
                    Some(&existing) if existing >= score => {}
                    _ => {
                        best_by_term.insert(candidate.term.clone(), score);
                    }
                }
            }
        }
        let mut ranked: Vec<(String, f32)> = best_by_term.into_iter().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let target_rank = ranked
            .iter()
            .position(|(term, _)| term.eq_ignore_ascii_case(&case.target_term))
            .map(|idx| idx + 1);
        let best_span_text = ranked
            .first()
            .map(|(term, _)| term.clone())
            .unwrap_or_default();

        // Extract judge choice: best non-keep candidate across all spans.
        let mut best_judge_choice: Option<CaseJudgeChoice> = None;
        for span_trace in &probe_result.spans {
            if let Some(best) = span_trace
                .judge_options
                .iter()
                .filter(|o| !o.is_keep_original)
                .max_by(|a, b| {
                    a.probability
                        .total_cmp(&b.probability)
                        .then_with(|| a.score.total_cmp(&b.score))
                })
            {
                match &best_judge_choice {
                    Some(existing)
                        if existing.probability > best.probability
                            || (existing.probability == best.probability
                                && existing.score >= best.score) => {}
                    _ => {
                        best_judge_choice = Some(CaseJudgeChoice {
                            span_text: span_trace.span.text.clone(),
                            chosen_action: best.term.clone(),
                            probability: best.probability,
                            score: best.score,
                            is_keep_original: false,
                        });
                    }
                }
            }
        }
        // Judge chooses keep_original if no candidate exceeded threshold
        let judge_choice = match best_judge_choice {
            Some(choice) if choice.probability >= 0.5 => choice,
            _ => CaseJudgeChoice {
                span_text: String::new(),
                chosen_action: "keep_original".to_string(),
                probability: 0.0,
                score: 0.0,
                is_keep_original: true,
            },
        };

        Ok(CaseEvalResult {
            target_rank,
            best_span_text,
            judge_choice,
        })
    }
}

const RAPID_FIRE_COMPONENT_HYPOTHESES: usize = 4;
const RAPID_FIRE_SENTENCE_CHOICES: usize = 12;
const RAPID_FIRE_EXACT_THRESHOLD: usize = 64;
const RAPID_FIRE_BEAM_WIDTH: usize = 16;
const RAPID_FIRE_MAX_EDITS_PER_SPAN: usize = 4;

#[derive(Clone)]
struct EditCandidate {
    span_token_start: u32,
    span_token_end: u32,
    span_text: String,
    alias_id: u32,
    replacement_text: String,
    score: f32,
    probability: f32,
    acceptance_score: f32,
    phonetic_score: f32,
    verified: bool,
}

#[derive(Clone)]
struct ComponentHypothesis {
    component_id: u32,
    component_spans: Vec<RejectedGroupSpan>,
    edits: Vec<EditCandidate>,
    choose_keep_original: bool,
    score: f32,
    probability: f32,
}

#[derive(Clone)]
struct SentenceHypothesis {
    components: Vec<ComponentHypothesis>,
    sentence: String,
    score: f32,
    probability: f32,
}

fn build_rapid_fire_decision_set(
    transcript: &str,
    spans: &[SpanDebugTrace],
    expected_source_text: &str,
) -> RapidFireDecisionSet {
    let is_counterexample =
        normalize_comparable_text(transcript) == normalize_comparable_text(expected_source_text);
    let admitted_edits = collect_admitted_edits(spans, is_counterexample);
    let span_keep_probabilities: HashMap<(u32, u32), f32> = spans
        .iter()
        .map(|s| {
            let keep_prob = s.judge_options.iter()
                .find(|o| o.is_keep_original)
                .map(|o| o.probability)
                .unwrap_or(0.0);
            ((s.span.token_start, s.span.token_end), keep_prob)
        })
        .collect();
    let components = build_conflict_components(&admitted_edits)
        .into_iter()
        .enumerate()
        .map(|(index, edits)| build_component(index as u32, edits, &span_keep_probabilities))
        .collect::<Vec<_>>();
    let total_combinations = components
        .iter()
        .map(|component| component.hypotheses.len())
        .product::<usize>();
    debug!(
        transcript,
        expected_source_text,
        is_counterexample,
        spans = spans.len(),
        admitted_edits = admitted_edits.len(),
        components = components.len(),
        total_combinations,
        "building rapid fire decision set"
    );
    let (search_mode, mut sentence_hypotheses) =
        compose_sentence_hypotheses(transcript, &components, total_combinations);

    sentence_hypotheses.sort_by(|lhs, rhs| {
        rhs.probability
            .total_cmp(&lhs.probability)
            .then_with(|| rhs.score.total_cmp(&lhs.score))
    });
    sentence_hypotheses = prune_sentence_hypotheses(sentence_hypotheses, is_counterexample);

    let mut choices = sentence_hypotheses
        .iter()
        .take(RAPID_FIRE_SENTENCE_CHOICES)
        .cloned()
        .enumerate()
        .map(|(index, hypothesis)| {
            let is_gold =
                normalize_comparable_text(&hypothesis.sentence) == normalize_comparable_text(expected_source_text);
            let component_choices = hypothesis
                .components
                .iter()
                .map(component_hypothesis_to_rpc)
                .collect::<Vec<_>>();
            let edits = hypothesis
                .components
                .iter()
                .flat_map(|component| component.edits.iter())
                .map(|edit| RapidFireEdit {
                    span_token_start: edit.span_token_start,
                    span_token_end: edit.span_token_end,
                    replaced_text: edit.span_text.clone(),
                    replacement_text: edit.replacement_text.clone(),
                })
                .collect::<Vec<_>>();
            let primary_edit = edits.first().cloned().unwrap_or(RapidFireEdit {
                span_token_start: 0,
                span_token_end: 0,
                replaced_text: String::new(),
                replacement_text: String::new(),
            });
            RapidFireChoice {
                option_id: format!("hypothesis:{index}"),
                span_token_start: primary_edit.span_token_start,
                span_token_end: primary_edit.span_token_end,
                choose_keep_original: edits.is_empty(),
                chosen_alias_id: component_choices
                    .iter()
                    .find_map(|choice| choice.chosen_alias_id),
                sentence: hypothesis.sentence,
                replaced_text: primary_edit.replaced_text,
                replacement_text: primary_edit.replacement_text,
                score: hypothesis.score,
                probability: hypothesis.probability,
                is_gold,
                is_judge_pick: false,
                edits,
                component_choices,
            }
        })
        .collect::<Vec<_>>();

    choices.sort_by(|lhs, rhs| {
        rhs.probability
            .total_cmp(&lhs.probability)
            .then_with(|| rhs.score.total_cmp(&lhs.score))
    });

    // Ensure keep_original is always present: partition edit choices from keep,
    // truncate edits to leave room, then append the keep choice.
    let keep_choice = choices
        .iter()
        .position(|choice| choice.choose_keep_original)
        .map(|idx| choices.remove(idx))
        .or_else(|| {
            // Fallback: synthesize from the all-keep sentence hypothesis
            sentence_hypotheses
                .iter()
                .enumerate()
                .find(|(_, h)| h.components.iter().all(|c| c.choose_keep_original))
                .map(|(index, hypothesis)| RapidFireChoice {
                    option_id: format!("hypothesis:{index}"),
                    span_token_start: 0,
                    span_token_end: 0,
                    choose_keep_original: true,
                    chosen_alias_id: None,
                    sentence: hypothesis.sentence.clone(),
                    replaced_text: String::new(),
                    replacement_text: String::new(),
                    score: hypothesis.score,
                    probability: hypothesis.probability,
                    is_gold: normalize_comparable_text(&hypothesis.sentence)
                        == normalize_comparable_text(expected_source_text),
                    is_judge_pick: false,
                    edits: Vec::new(),
                    component_choices: hypothesis
                        .components
                        .iter()
                        .map(component_hypothesis_to_rpc)
                        .collect(),
                })
        })
        .unwrap_or_else(|| {
            // Last resort: synthesize directly from transcript
            RapidFireChoice {
                option_id: "keep_original".to_string(),
                span_token_start: 0,
                span_token_end: 0,
                choose_keep_original: true,
                chosen_alias_id: None,
                sentence: transcript.to_string(),
                replaced_text: String::new(),
                replacement_text: String::new(),
                score: 0.0,
                probability: 0.0,
                is_gold: normalize_comparable_text(transcript)
                    == normalize_comparable_text(expected_source_text),
                is_judge_pick: false,
                edits: Vec::new(),
                component_choices: Vec::new(),
            }
        });
    choices.truncate(RAPID_FIRE_SENTENCE_CHOICES - 1);
    choices.push(keep_choice);

    if let Some(first) = choices.first_mut() {
        first.is_judge_pick = true;
    }

    info!(
        transcript,
        expected_source_text,
        is_counterexample,
        search_mode,
        spans = spans.len(),
        admitted_edits = admitted_edits.len(),
        components = components.len(),
        total_combinations,
        visible_choices = choices.len(),
        no_exact_match = !choices.iter().any(|choice| choice.is_gold),
        top_choice_sentence = choices.first().map(|choice| choice.sentence.as_str()).unwrap_or(""),
        top_choice_keep = choices.first().map(|choice| choice.choose_keep_original).unwrap_or(true),
        "rapid fire decision set built"
    );

    RapidFireDecisionSet {
        no_exact_match: !choices.iter().any(|choice| choice.is_gold),
        rejected_group_spans: Vec::new(),
        components: components
            .iter()
            .map(|component| RapidFireComponent {
                component_id: component.component_id,
                spans: component.component_spans.clone(),
                hypotheses: component
                    .hypotheses
                    .iter()
                    .map(component_hypothesis_to_rpc)
                    .collect(),
            })
            .collect(),
        total_combinations: total_combinations as u32,
        search_mode: search_mode.to_string(),
        choices,
    }
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
                Some(existing) if existing.features.acceptance_score >= candidate.features.acceptance_score => {}
                _ => { best_by_term.insert(candidate.term.clone(), candidate); }
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
            if normalize_comparable_text(&candidate.term) == normalize_comparable_text(&span.span.text) {
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
            Some(existing)
                if existing.acceptance_score >= edit.acceptance_score => {}
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
            && a.edits.first().map(|e| e.span_token_end) == b.edits.first().map(|e| e.span_token_end)
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
        ("exact", enumerate_sentence_hypotheses(transcript, components))
    } else {
        ("beam", beam_sentence_hypotheses(transcript, components))
    }
}

fn enumerate_sentence_hypotheses(transcript: &str, components: &[Component]) -> Vec<SentenceHypothesis> {
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
    let num_edits = components.iter().filter(|c| !c.choose_keep_original).count();
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
        if hypothesis.components.iter().all(|component| component.choose_keep_original) {
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

struct CaseJudgeChoice {
    span_text: String,
    chosen_action: String,
    probability: f32,
    score: f32,
    is_keep_original: bool,
}

struct CaseEvalResult {
    target_rank: Option<usize>,
    best_span_text: String,
    judge_choice: CaseJudgeChoice,
}

impl BeeMl for BeeMlService {
    async fn transcribe_wav(&self, wav_bytes: Vec<u8>) -> Result<TranscribeWavResult, String> {
        let samples = bee_transcribe::decode_wav(&wav_bytes).map_err(|e| e.to_string())?;

        let mut session = self.inner.engine.session(SessionOptions::default());

        session.feed(&samples).map_err(|e| e.to_string())?;
        let update = session.finish().map_err(|e| e.to_string())?;

        Ok(TranscribeWavResult {
            transcript: update.text,
            words: update.alignments,
        })
    }

    async fn stream_transcribe(
        &self,
        _audio_in: Rx<Vec<f32>>,
        _updates_out: Tx<bee_transcribe::Update>,
    ) -> Result<(), String> {
        Err("stream_transcribe is temporarily unavailable while beeml migrates to the new correction RPC surface".to_string())
    }

    async fn correct_transcript(
        &self,
        request: CorrectionRequest,
    ) -> Result<CorrectionResult, String> {
        Ok(CorrectionResult {
            original_transcript: request.transcript.clone(),
            corrected_transcript: request.transcript,
            accepted_edits: Vec::<AcceptedEdit>::new(),
        })
    }

    async fn debug_correction(
        &self,
        request: CorrectionRequest,
    ) -> Result<CorrectionDebugResult, String> {
        Ok(CorrectionDebugResult {
            result: CorrectionResult {
                original_transcript: request.transcript.clone(),
                corrected_transcript: request.transcript,
                accepted_edits: Vec::new(),
            },
            spans: Vec::new(),
            reranker_regions: Vec::<RerankerDebugTrace>::new(),
            timings: TimingBreakdown {
                span_enumeration_ms: 0,
                retrieval_ms: 0,
                verify_ms: 0,
                rerank_ms: 0,
                total_ms: 0,
            },
        })
    }

    async fn probe_retrieval_prototype(
        &self,
        request: RetrievalPrototypeProbeRequest,
    ) -> Result<RetrievalPrototypeProbeResult, String> {
        self.run_probe(request, None)
    }

    async fn teach_retrieval_prototype_judge(
        &self,
        request: TeachRetrievalPrototypeJudgeRequest,
    ) -> Result<RetrievalPrototypeProbeResult, String> {
        self.run_probe(request.probe.clone(), Some(request))
    }

    async fn load_retrieval_prototype_teaching_deck(
        &self,
        request: RetrievalPrototypeTeachingDeckRequest,
    ) -> Result<RetrievalPrototypeTeachingDeckResult, String> {
        Ok(RetrievalPrototypeTeachingDeckResult {
            cases: self
                .teaching_cases(request.limit as usize, request.include_counterexamples)
                .into_iter()
                .map(|case| RetrievalPrototypeTeachingCase {
                    case_id: case.case_id,
                    suite: case.suite.to_string(),
                    target_term: case.target_term,
                    source_text: case.source_text,
                    transcript: case.transcript,
                    should_abstain: case.should_abstain,
                    take: case.take,
                    audio_path: case.audio_path,
                    surface_form: case.surface_form,
                })
                .collect(),
        })
    }

    async fn inspect_term(
        &self,
        request: TermInspectionRequest,
    ) -> Result<TermInspectionResult, String> {
        let aliases = self
            .inner
            .index
            .aliases
            .iter()
            .filter(|alias| alias.term.eq_ignore_ascii_case(&request.term))
            .map(|alias| TermAliasView {
                alias_text: alias.alias_text.clone(),
                alias_source: map_alias_source(alias.alias_source),
                ipa_tokens: alias.ipa_tokens.clone(),
                reduced_ipa_tokens: alias.reduced_ipa_tokens.clone(),
                feature_tokens: alias.feature_tokens.clone(),
                identifier_flags: map_identifier_flags(&alias.identifier_flags),
            })
            .collect();

        Ok(TermInspectionResult {
            term: request.term,
            aliases,
        })
    }

    async fn run_retrieval_prototype_eval(
        &self,
        request: RetrievalPrototypeEvalRequest,
        progress: Tx<RetrievalPrototypeEvalProgress>,
    ) -> Result<RetrievalPrototypeEvalResult, String> {
        let cases = self.teaching_cases(request.limit as usize, true);
        {
            let judge = self
                .inner
                .judge
                .lock()
                .map_err(|_| "judge mutex poisoned".to_string())?;
            info!(
                update_count = judge.update_count(),
                "running eval"
            );
        }

        let mut top1_hits = 0u32;
        let mut top3_hits = 0u32;
        let mut top10_hits = 0u32;
        let mut judge_correct = 0u32;
        let mut judge_replace_correct = 0u32;
        let mut judge_abstain_correct = 0u32;
        let mut misses = Vec::new();
        let mut judge_failures = Vec::new();
        let mut per_term = HashMap::<String, RetrievalEvalTermSummary>::new();

        let total = cases.len() as u32;
        for (recording_id, case) in cases.iter().enumerate() {
            let result = self.evaluate_case(case, &request)?;
            let entry = per_term
                .entry(case.target_term.clone())
                .or_insert(RetrievalEvalTermSummary {
                    term: case.target_term.clone(),
                    cases: 0,
                    top1_hits: 0,
                    top3_hits: 0,
                    top10_hits: 0,
                });
            entry.cases += 1;

            if let Some(rank) = result.target_rank {
                if rank <= 1 {
                    top1_hits += 1;
                    entry.top1_hits += 1;
                }
                if rank <= 3 {
                    top3_hits += 1;
                    entry.top3_hits += 1;
                }
                if rank <= 10 {
                    top10_hits += 1;
                    entry.top10_hits += 1;
                }
            } else {
                misses.push(RetrievalEvalMiss {
                    recording_id: recording_id as u32,
                    suite: case.suite.to_string(),
                    term: case.target_term.clone(),
                    transcript: case.transcript.clone(),
                    best_span_text: result.best_span_text.clone(),
                });
            }

            let judge_ok = if case.should_abstain {
                result.judge_choice.is_keep_original
            } else {
                !result.judge_choice.is_keep_original
                    && result
                        .judge_choice
                        .chosen_action
                        .eq_ignore_ascii_case(&case.target_term)
            };
            if judge_ok {
                judge_correct += 1;
                if case.should_abstain {
                    judge_abstain_correct += 1;
                } else {
                    judge_replace_correct += 1;
                }
            } else {
                judge_failures.push(JudgeEvalFailure {
                    case_id: case.case_id.clone(),
                    suite: case.suite.to_string(),
                    target_term: case.target_term.clone(),
                    transcript: case.transcript.clone(),
                    expected_action: if case.should_abstain {
                        "keep_original".to_string()
                    } else {
                        case.target_term.clone()
                    },
                    chosen_action: result.judge_choice.chosen_action.clone(),
                    chosen_span_text: result.judge_choice.span_text.clone(),
                    chosen_probability: result.judge_choice.probability,
                });
            }

            let _ = progress.send(RetrievalPrototypeEvalProgress {
                evaluated: recording_id as u32 + 1,
                total,
                judge_correct,
            }).await;
        }

        let mut per_term = per_term.into_values().collect::<Vec<_>>();
        per_term.sort_by(|a, b| a.term.cmp(&b.term));

        info!(
            evaluated = cases.len(),
            judge_correct,
            judge_replace_correct,
            judge_abstain_correct,
            top1_hits,
            judge_failures = judge_failures.len(),
            "eval complete"
        );

        Ok(RetrievalPrototypeEvalResult {
            evaluated_cases: cases.len() as u32,
            top1_hits,
            top3_hits,
            top10_hits,
            judge_correct,
            judge_replace_correct,
            judge_abstain_correct,
            misses,
            judge_failures,
            per_term,
        })
    }
}

fn randomish_case_key(case: &EvalCase, seed: u64) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    seed.hash(&mut hasher);
    case.case_id.hash(&mut hasher);
    case.target_term.hash(&mut hasher);
    case.transcript.hash(&mut hasher);
    hasher.finish()
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let _tracing_guard = init_tracing()?;
    let listen_addr = env::var("BML_WS_ADDR").unwrap_or_else(|_| "127.0.0.1:9944".to_string());
    let model_dir = env::var("BEE_ASR_MODEL_DIR")
        .map(PathBuf::from)
        .context("BEE_ASR_MODEL_DIR must be set")?;
    let tokenizer_path = env::var("BEE_TOKENIZER_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| model_dir.join("tokenizer.json"));
    let aligner_dir = env::var("BEE_ALIGNER_DIR")
        .map(PathBuf::from)
        .context("BEE_ALIGNER_DIR must be set")?;

    info!(model_dir = %model_dir.display(), "loading ASR engine");
    let engine = Engine::load(&EngineConfig {
        model_dir: &model_dir,
        tokenizer_path: &tokenizer_path,
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

    let handler = BeeMlService {
        inner: Arc::new(BeemlServiceInner {
            engine,
            index,
            dataset,
            counterexamples,
            g2p: Mutex::new(CachedEspeakG2p::english().context("initializing g2p engine")?),
            judge: Mutex::new(OnlineJudge::default()),
        }),
    };

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

fn map_alias_source(source: bee_phonetic::AliasSource) -> AliasSource {
    match source {
        bee_phonetic::AliasSource::Canonical => AliasSource::Canonical,
        bee_phonetic::AliasSource::Spoken => AliasSource::Spoken,
        bee_phonetic::AliasSource::Identifier => AliasSource::Identifier,
        bee_phonetic::AliasSource::Confusion => AliasSource::Confusion,
    }
}

fn map_index_view(view: bee_phonetic::IndexView) -> RetrievalIndexView {
    match view {
        bee_phonetic::IndexView::RawIpa2 => RetrievalIndexView::RawIpa2,
        bee_phonetic::IndexView::RawIpa3 => RetrievalIndexView::RawIpa3,
        bee_phonetic::IndexView::ReducedIpa2 => RetrievalIndexView::ReducedIpa2,
        bee_phonetic::IndexView::ReducedIpa3 => RetrievalIndexView::ReducedIpa3,
        bee_phonetic::IndexView::Feature2 => RetrievalIndexView::Feature2,
        bee_phonetic::IndexView::Feature3 => RetrievalIndexView::Feature3,
        bee_phonetic::IndexView::ShortQueryFallback => RetrievalIndexView::ShortQueryFallback,
    }
}

fn map_candidate_features(candidate: &bee_phonetic::CandidateFeatureRow) -> CandidateFeatureDebug {
    CandidateFeatureDebug {
        matched_view: map_index_view(candidate.matched_view),
        qgram_overlap: candidate.qgram_overlap,
        total_qgram_overlap: candidate.total_qgram_overlap,
        best_view_score: candidate.best_view_score,
        cross_view_support: candidate.cross_view_support,
        token_count_match: candidate.token_count_match,
        phone_count_delta: candidate.phone_count_delta,
        token_bonus: candidate.token_bonus,
        phone_bonus: candidate.phone_bonus,
        extra_length_penalty: candidate.extra_length_penalty,
        structure_bonus: candidate.structure_bonus,
        coarse_score: candidate.coarse_score,
        token_distance: candidate.token_distance,
        token_weighted_distance: candidate.token_weighted_distance,
        token_boundary_penalty: candidate.token_boundary_penalty,
        token_max_len: candidate.token_max_len,
        token_score: candidate.token_score,
        feature_distance: candidate.feature_distance,
        feature_weighted_distance: candidate.feature_weighted_distance,
        feature_boundary_penalty: candidate.feature_boundary_penalty,
        feature_max_len: candidate.feature_max_len,
        feature_score: candidate.feature_score,
        feature_bonus: candidate.feature_bonus,
        feature_gate_token_ok: candidate.feature_gate_token_ok,
        feature_gate_coarse_ok: candidate.feature_gate_coarse_ok,
        feature_gate_phone_ok: candidate.feature_gate_phone_ok,
        short_guard_applied: candidate.short_guard_applied,
        short_guard_onset_match: candidate.short_guard_onset_match,
        short_guard_passed: candidate.short_guard_passed,
        low_content_guard_applied: candidate.low_content_guard_applied,
        low_content_guard_passed: candidate.low_content_guard_passed,
        acceptance_floor_passed: candidate.acceptance_floor_passed,
        used_feature_bonus: candidate.used_feature_bonus,
        phonetic_score: candidate.phonetic_score,
        acceptance_score: candidate.acceptance_score,
        verified: candidate.verified,
    }
}

fn map_identifier_flags(flags: &bee_phonetic::IdentifierFlags) -> IdentifierFlags {
    IdentifierFlags {
        acronym_like: flags.acronym_like,
        has_digits: flags.has_digits,
        snake_like: flags.snake_like,
        camel_like: flags.camel_like,
        symbol_like: flags.symbol_like,
    }
}

fn compare_candidate_rows(
    a: &bee_phonetic::CandidateFeatureRow,
    b: &bee_phonetic::CandidateFeatureRow,
) -> std::cmp::Ordering {
    b.acceptance_score
        .total_cmp(&a.acceptance_score)
        .then_with(|| b.phonetic_score.total_cmp(&a.phonetic_score))
        .then_with(|| b.coarse_score.total_cmp(&a.coarse_score))
        .then_with(|| b.qgram_overlap.cmp(&a.qgram_overlap))
}

fn load_counterexample_recordings() -> Result<Vec<CounterexampleRecordingRow>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../data/phonetic-seed/counterexample_recordings.jsonl");
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("reading {}", path.display()))?;
    let mut rows = Vec::new();
    for (line_idx, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row = serde_json::from_str::<CounterexampleRecordingRow>(line)
            .with_context(|| format!("parsing {} line {}", path.display(), line_idx + 1))?;
        rows.push(row);
    }
    Ok(rows)
}
