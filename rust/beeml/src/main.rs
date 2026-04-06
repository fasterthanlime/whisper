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
use beeml::judge::{OnlineJudge, extract_span_context};
use beeml::rpc::{
    AcceptedEdit, AliasSource, BeeMl, CandidateFeatureDebug, CorrectionDebugResult,
    CorrectionRequest, CorrectionResult, FilterDecision, IdentifierFlags, RerankerDebugTrace,
    JudgeEvalFailure, JudgeOptionDebug, JudgeStateDebug, OfflineJudgeEvalRequest,
    OfflineJudgeEvalResult, OfflineJudgeFoldResult, RapidFireChoice, RapidFireComponent,
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
    event_log_path: std::path::PathBuf,
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
    words: Vec<AlignedWord>,
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
                        mean_logprob: word.mean_logprob,
                        min_logprob: word.min_logprob,
                        mean_margin: word.mean_margin,
                        min_margin: word.min_margin,
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
                    request.shortlist_limit as usize,
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
                let span_ctx = extract_span_context(&request.transcript, span.char_start, span.char_end);
                let judge_options = if should_teach {
                    judge.teach_choice(&span, &judge_input, chosen_alias_id, &span_ctx)
                } else {
                    judge.score_candidates(&span, &judge_input, &span_ctx)
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

            // Persist correction events to disk
            if let Err(e) = save_correction_events(&self.inner.event_log_path, judge.event_log()) {
                tracing::warn!(error = %e, "failed to save correction events");
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
                words: row.words.iter().map(|w| AlignedWord {
                    word: w.word.clone(),
                    start: w.start,
                    end: w.end,
                    mean_logprob: w.mean_logprob,
                    min_logprob: w.min_logprob,
                    mean_margin: w.mean_margin,
                    min_margin: w.min_margin,
                }).collect(),
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
                        words: Vec::new(),
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

    /// Extract spans + candidates for a case without involving the judge.
    /// Used by offline eval to separate feature extraction from judge training.
    fn probe_case_spans(
        &self,
        case: &EvalCase,
        max_span_words: u8,
        shortlist_limit: u16,
    ) -> Result<ProbedCase, String> {
        let alignments = if case.words.is_empty() {
            None
        } else {
            Some(
                case.words
                    .iter()
                    .map(|word| TranscriptAlignmentToken {
                        start_time: word.start,
                        end_time: word.end,
                        mean_logprob: word.mean_logprob,
                        min_logprob: word.min_logprob,
                        mean_margin: word.mean_margin,
                        min_margin: word.min_margin,
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
                &case.transcript,
                max_span_words as usize,
                alignments.as_deref(),
                |text| g2p.ipa_tokens(text).ok().flatten(),
            )
        };

        let probed_spans: Vec<ProbedSpan> = spans
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
                    shortlist_limit as usize,
                );
                let scored_rows = score_shortlist(&span, &shortlist, &self.inner.index);
                let candidates: Vec<_> = scored_rows
                    .iter()
                    .map(|row| {
                        let alias = &self.inner.index.aliases[row.alias_id as usize];
                        (row.clone(), alias.identifier_flags.clone())
                    })
                    .collect();
                let ctx = extract_span_context(&case.transcript, span.char_start, span.char_end);

                // Find gold alias_id: the best-scoring alias for the target term
                let gold_alias_id = if !case.should_abstain {
                    candidates
                        .iter()
                        .find(|(c, _)| c.term.eq_ignore_ascii_case(&case.target_term))
                        .map(|(c, _)| c.alias_id)
                } else {
                    None
                };

                ProbedSpan {
                    span,
                    candidates,
                    ctx,
                    gold_alias_id,
                }
            })
            .collect();

        Ok(ProbedCase {
            case: case.clone(),
            spans: probed_spans,
        })
    }

    fn evaluate_case(
        &self,
        case: &EvalCase,
        request: &RetrievalPrototypeEvalRequest,
    ) -> Result<CaseEvalResult, String> {
        // Use the same pipeline as run_probe — one path for everything.
        // Pass expected_source_text so rapid fire decision set is built with is_gold flags.
        let probe_result = self.run_probe(
            RetrievalPrototypeProbeRequest {
                transcript: case.transcript.clone(),
                words: case.words.clone(),
                max_span_words: request.max_span_words,
                shortlist_limit: request.shortlist_limit,
                verify_limit: request.verify_limit,
                expected_source_text: Some(case.source_text.clone()),
            },
            None,
        )?;

        // Stage 1: Retrieval — did any span's candidate list contain the target term?
        let mut target_best_rank: Option<usize> = None;
        for span_trace in &probe_result.spans {
            if let Some(rank) = span_trace
                .candidates
                .iter()
                .enumerate()
                .find(|(_, c)| c.term.eq_ignore_ascii_case(&case.target_term))
                .map(|(i, _)| i + 1)
            {
                match target_best_rank {
                    Some(existing) if existing <= rank => {}
                    _ => { target_best_rank = Some(rank); }
                }
            }
        }
        let target_in_shortlist = target_best_rank.is_some();

        // Stage 2: Composition — is the gold sentence in the judge-visible decision set?
        let rapid_fire = probe_result.rapid_fire.as_ref();
        let choices = rapid_fire.map(|rf| &rf.choices[..]).unwrap_or(&[]);
        let decision_set_size = choices.len();
        let replacement_choice_count = choices.iter().filter(|c| !c.choose_keep_original).count();
        let gold_choice_index = choices.iter().position(|c| c.is_gold);
        let gold_reachable = gold_choice_index.is_some();

        // Stage 3: Judge — what would the judge pick?
        // Build a lookup of judge probabilities per (span_start, span_end, alias_id)
        // and per (span_start, span_end) for keep_original, then score each sentence
        // choice by the judge's actual probabilities.
        let mut judge_edit_prob: HashMap<(u32, u32, u32), f32> = HashMap::new();
        let mut judge_keep_prob: HashMap<(u32, u32), f32> = HashMap::new();
        for span_trace in &probe_result.spans {
            let ts = span_trace.span.token_start;
            let te = span_trace.span.token_end;
            for opt in &span_trace.judge_options {
                if opt.is_keep_original {
                    judge_keep_prob.insert((ts, te), opt.probability);
                } else if let Some(alias_id) = opt.alias_id {
                    judge_edit_prob.insert((ts, te, alias_id), opt.probability);
                }
            }
        }

        // Score each choice by the judge: for each component, look up the judge
        // probability for the chosen action (keep or specific edit). The sentence
        // score is the min across components (weakest link).
        // For keep_original sentences that have no component_choices (synthesized
        // fallback), use the min keep probability across all spans with edit candidates.
        let judge_score_for_choice = |choice: &RapidFireChoice| -> f32 {
            if choice.component_choices.is_empty() {
                // Synthesized keep_original with no components — use min keep prob
                // across all spans that have edit candidates
                if judge_keep_prob.is_empty() {
                    return 1.0; // no spans with candidates, keeping is trivially correct
                }
                return judge_keep_prob.values().copied().fold(f32::MAX, f32::min);
            }
            let mut min_prob = f32::MAX;
            for cc in &choice.component_choices {
                if cc.choose_keep_original {
                    // This component keeps original — use keep prob for its spans
                    let prob = cc.component_spans.iter()
                        .filter_map(|s| judge_keep_prob.get(&(s.token_start, s.token_end)))
                        .copied()
                        .fold(0.0f32, f32::max);
                    min_prob = min_prob.min(prob);
                } else if let (Some(start), Some(end), Some(alias_id)) =
                    (cc.span_token_start, cc.span_token_end, cc.chosen_alias_id)
                {
                    let prob = judge_edit_prob
                        .get(&(start, end, alias_id))
                        .copied()
                        .unwrap_or(0.0);
                    min_prob = min_prob.min(prob);
                }
            }
            if min_prob == f32::MAX { 0.0 } else { min_prob }
        };

        // Log judge scores for the first few cases to debug
        if !case.should_abstain {
            let gold = choices.iter().find(|c| c.is_gold);
            let keep = choices.iter().find(|c| c.choose_keep_original);
            tracing::debug!(
                case_id = %case.case_id,
                target = %case.target_term,
                gold_score = gold.map(|c| judge_score_for_choice(c)),
                keep_score = keep.map(|c| judge_score_for_choice(c)),
                gold_reachable,
                num_choices = choices.len(),
                "judge eval debug"
            );
        }

        let judge_pick = choices.iter().max_by(|a, b| {
            judge_score_for_choice(a).total_cmp(&judge_score_for_choice(b))
        });
        let (chosen_kind, chosen_choice_id, chosen_sentence, chosen_edit_count, chosen_probability) =
            if let Some(pick) = judge_pick {
                (
                    if pick.choose_keep_original { EvalChoiceKind::KeepOriginal } else { EvalChoiceKind::SentenceCandidate },
                    Some(pick.option_id.clone()),
                    pick.sentence.clone(),
                    pick.edits.len(),
                    judge_score_for_choice(pick),
                )
            } else {
                (EvalChoiceKind::KeepOriginal, None, case.transcript.clone(), 0, 0.0)
            };

        let judge_correct = if case.should_abstain {
            matches!(chosen_kind, EvalChoiceKind::KeepOriginal)
        } else {
            judge_pick.is_some_and(|p| p.is_gold)
        };

        // Why is gold unreachable? (canonical cases only)
        let gold_unreachable_reason = if gold_reachable || case.should_abstain {
            None
        } else if !target_in_shortlist {
            Some(GoldUnreachableReason::TargetNotRetrieved)
        } else {
            // Target is in shortlist. Check if any choice contains the target term.
            let any_choice_has_target = choices.iter().any(|c| {
                !c.choose_keep_original
                    && c.edits.iter().any(|e| {
                        e.replacement_text.eq_ignore_ascii_case(&case.target_term)
                    })
            });
            if !any_choice_has_target {
                // Target was retrieved but never made it into any composed sentence.
                // Either other required edits were missing, or composition pruned it.
                Some(GoldUnreachableReason::MissingRequiredEdits)
            } else {
                // Target term IS in some choices, but none match the gold sentence.
                // This means the edits were applied but the surface form doesn't match.
                let closest = choices
                    .iter()
                    .filter(|c| !c.choose_keep_original)
                    .filter(|c| c.edits.iter().any(|e| e.replacement_text.eq_ignore_ascii_case(&case.target_term)))
                    .min_by_key(|c| {
                        // Simple word-level edit distance to gold
                        let got_words: Vec<&str> = c.sentence.split_whitespace().collect();
                        let want_words: Vec<&str> = case.source_text.split_whitespace().collect();
                        let diff = got_words.len().abs_diff(want_words.len())
                            + got_words.iter().zip(&want_words).filter(|(a, b)| a != b).count();
                        diff
                    });
                Some(GoldUnreachableReason::SurfaceMismatch {
                    closest_sentence: closest.map(|c| c.sentence.clone()).unwrap_or_default(),
                })
            }
        };

        // Attribution: which stage failed first? (canonical cases only)
        let first_failure = if case.should_abstain {
            if !judge_correct {
                Some(EvalFailureStage::Judge)
            } else {
                None
            }
        } else if !target_in_shortlist {
            Some(EvalFailureStage::RetrievalShortlist)
        } else if !gold_reachable {
            Some(EvalFailureStage::Composition)
        } else if !judge_correct {
            Some(EvalFailureStage::Judge)
        } else {
            None
        };

        Ok(CaseEvalResult {
            target_in_shortlist,
            target_best_rank,
            gold_reachable,
            gold_choice_rank: gold_choice_index,
            gold_unreachable_reason,
            decision_set_size,
            replacement_choice_count,
            chosen_kind,
            chosen_choice_id,
            chosen_sentence,
            chosen_edit_count,
            chosen_probability,
            judge_correct,
            first_failure,
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

/// Per-span data extracted from a probe, ready for judge training/evaluation.
/// Does not include judge scores — those are computed separately per fold.
struct ProbedSpan {
    span: TranscriptSpan,
    candidates: Vec<(bee_phonetic::CandidateFeatureRow, bee_phonetic::IdentifierFlags)>,
    ctx: beeml::judge::SpanContext,
    /// For canonical cases: the alias_id of the target term (if retrieved).
    gold_alias_id: Option<u32>,
}

/// All probed spans for one eval case, plus metadata for the offline judge eval.
struct ProbedCase {
    case: EvalCase,
    spans: Vec<ProbedSpan>,
}

// ── Offline eval helpers ───────────────────────────────────────────

#[derive(Clone, Debug, Default)]
struct EvalMetrics {
    canonical_correct: u32,
    canonical_total: u32,
    canonical_replaced: u32,
    cx_correct: u32,
    cx_total: u32,
    cx_replaced: u32,
}

impl EvalMetrics {
    fn canonical_pct(&self) -> f64 {
        if self.canonical_total == 0 { 0.0 } else { self.canonical_correct as f64 / self.canonical_total as f64 * 100.0 }
    }
    fn cx_pct(&self) -> f64 {
        if self.cx_total == 0 { 0.0 } else { self.cx_correct as f64 / self.cx_total as f64 * 100.0 }
    }
    fn balanced_pct(&self) -> f64 {
        (self.canonical_pct() + self.cx_pct()) / 2.0
    }
    fn canonical_replace_pct(&self) -> f64 {
        if self.canonical_total == 0 { 0.0 } else { self.canonical_replaced as f64 / self.canonical_total as f64 * 100.0 }
    }
    fn cx_replace_pct(&self) -> f64 {
        if self.cx_total == 0 { 0.0 } else { self.cx_replaced as f64 / self.cx_total as f64 * 100.0 }
    }
    fn merge(&mut self, other: &EvalMetrics) {
        self.canonical_correct += other.canonical_correct;
        self.canonical_total += other.canonical_total;
        self.canonical_replaced += other.canonical_replaced;
        self.cx_correct += other.cx_correct;
        self.cx_total += other.cx_total;
        self.cx_replaced += other.cx_replaced;
    }
}

#[derive(Clone, Debug)]
enum TrainMode {
    /// No training, use seed weights only.
    None,
    /// Current teach_choice replay.
    TeachChoice { epochs: usize },
    /// Case-balanced: 1 positive + hard negatives for canonical, 1 hard negative for cx.
    CaseBalanced { epochs: usize, hard_neg_cap: usize },
    /// Casewise softmax: all candidates + keep_original compete.
    CasewiseSoftmax { epochs: usize },
    /// Freeze dense seed weights, only train sparse/ASR/memory features.
    FreezeDense { epochs: usize, hard_neg_cap: usize },
}

/// Eval 1a: Deterministic baseline using the same candidate set as the judge.
fn eval_deterministic_kfold(
    probed_cases: &[ProbedCase],
    case_folds: &[usize],
    folds: usize,
    acceptance_threshold: f32,
) -> EvalMetrics {
    let mut total = EvalMetrics::default();
    for fold_k in 0..folds {
        let mut fold = EvalMetrics::default();
        for (i, pc) in probed_cases.iter().enumerate() {
            if case_folds[i] != fold_k { continue; }

            if pc.case.should_abstain {
                fold.cx_total += 1;
                // Check if any verified candidate above threshold exists
                let any_replace = pc.spans.iter().any(|ps| {
                    ps.candidates.iter().any(|(c, _)| {
                        c.verified && c.acceptance_score >= acceptance_threshold
                    })
                });
                if any_replace { fold.cx_replaced += 1; } else { fold.cx_correct += 1; }
            } else {
                // Find span with gold alias
                let gold_span = pc.spans.iter().find(|ps| ps.gold_alias_id.is_some());
                if let Some(gs) = gold_span {
                    fold.canonical_total += 1;
                    // Rank all verified candidates by acceptance_score > phonetic_score > coarse_score
                    let mut ranked: Vec<_> = gs.candidates.iter()
                        .filter(|(c, _)| c.verified && c.acceptance_score >= acceptance_threshold)
                        .collect();
                    ranked.sort_by(|(a, _), (b, _)| {
                        b.acceptance_score.total_cmp(&a.acceptance_score)
                            .then_with(|| b.phonetic_score.total_cmp(&a.phonetic_score))
                            .then_with(|| b.coarse_score.total_cmp(&a.coarse_score))
                    });
                    if !ranked.is_empty() {
                        fold.canonical_replaced += 1;
                        if ranked[0].0.alias_id == gs.gold_alias_id.unwrap() {
                            fold.canonical_correct += 1;
                        }
                    }
                }
            }
        }
        total.merge(&fold);
    }
    total
}

/// Pre-scored case: for each test case, store the probabilities so we can
/// sweep thresholds without re-training.
struct ScoredCase {
    should_abstain: bool,
    /// For canonical: the alias_id of the gold candidate.
    gold_alias_id: Option<u32>,
    /// Best candidate (alias_id, probability) from the gold span (canonical) or
    /// highest-prob candidate across all spans (counterexample).
    best_candidate: Option<(u32, f32)>,
    /// Whether gold was reachable (canonical: gold span exists; cx: any candidates exist).
    reachable: bool,
}

/// Train k-fold, score all test cases, return pre-scored results.
fn train_and_score_kfold(
    probed_cases: &[ProbedCase],
    case_folds: &[usize],
    folds: usize,
    train_mode: TrainMode,
    slice: beeml::judge::FeatureSlice,
) -> Vec<ScoredCase> {
    use beeml::judge::FeatureSlice;

    let use_ablation = !matches!(slice, FeatureSlice::All);
    let mut scored = Vec::with_capacity(probed_cases.len());
    // Initialize with None — only test-fold cases get scored.
    scored.resize_with(probed_cases.len(), || ScoredCase {
        should_abstain: false,
        gold_alias_id: None,
        best_candidate: None,
        reachable: false,
    });

    for fold_k in 0..folds {
        let mut judge = OnlineJudge::new_quiet();

        // ── Training phase ──
        let epochs = match &train_mode {
            TrainMode::None => 0,
            TrainMode::TeachChoice { epochs } => *epochs,
            TrainMode::CaseBalanced { epochs, .. } => *epochs,
            TrainMode::CasewiseSoftmax { epochs } => *epochs,
            TrainMode::FreezeDense { epochs, .. } => *epochs,
        };

        if matches!(&train_mode, TrainMode::FreezeDense { .. }) {
            // Freeze the 28 original phonetic/structural dense features (indices 0-27)
            judge.freeze_dense(28);
        }

        for _epoch in 0..epochs {
            for (i, pc) in probed_cases.iter().enumerate() {
                if case_folds[i] == fold_k { continue; }
                if use_ablation {
                    train_case_ablated(&mut judge, pc, &train_mode, slice);
                } else {
                    train_case(&mut judge, pc, &train_mode);
                }
            }
        }

        // ── Score test cases ──
        for (i, pc) in probed_cases.iter().enumerate() {
            if case_folds[i] != fold_k { continue; }

            if pc.case.should_abstain {
                let has_candidates = pc.spans.iter().any(|ps| !ps.candidates.is_empty());
                // Find max probability of any candidate across all spans
                let max_prob = pc.spans.iter()
                    .filter(|ps| !ps.candidates.is_empty())
                    .filter_map(|ps| {
                        if use_ablation {
                            score_span_ablated(&judge, ps, slice).into_iter()
                                .max_by(|a, b| a.1.total_cmp(&b.1))
                        } else {
                            let options = judge.score_candidates(&ps.span, &ps.candidates, &ps.ctx);
                            options.iter()
                                .filter(|o| !o.is_keep_original)
                                .max_by(|a, b| a.probability.total_cmp(&b.probability))
                                .map(|o| (o.alias_id.unwrap_or(0), o.probability))
                        }
                    })
                    .max_by(|a, b| a.1.total_cmp(&b.1));

                scored[i] = ScoredCase {
                    should_abstain: true,
                    gold_alias_id: None,
                    best_candidate: max_prob,
                    reachable: has_candidates,
                };
            } else {
                let gold_span = pc.spans.iter().find(|ps| ps.gold_alias_id.is_some());
                let (best_candidate, reachable) = if let Some(gs) = gold_span {
                    // Reachable = gold candidate exists AND is verified
                    let gold_verified = gs.gold_alias_id
                        .map(|gid| gs.candidates.iter().any(|(c, _)| c.alias_id == gid && c.verified))
                        .unwrap_or(false);
                    let best = if use_ablation {
                        let s = score_span_ablated(&judge, gs, slice);
                        s.into_iter().max_by(|a, b| a.1.total_cmp(&b.1))
                    } else {
                        let options = judge.score_candidates(&gs.span, &gs.candidates, &gs.ctx);
                        options.iter()
                            .filter(|o| !o.is_keep_original)
                            .max_by(|a, b| a.probability.total_cmp(&b.probability))
                            .map(|o| (o.alias_id.unwrap_or(0), o.probability))
                    };
                    (best, gold_verified)
                } else {
                    (None, false)
                };
                scored[i] = ScoredCase {
                    should_abstain: false,
                    gold_alias_id: gold_span.and_then(|gs| gs.gold_alias_id),
                    best_candidate,
                    reachable,
                };
            }
        }
    }
    scored
}

/// Evaluate pre-scored cases at a given threshold.
fn eval_at_threshold(scored: &[ScoredCase], threshold: f32, reachable_only: bool) -> EvalMetrics {
    let mut m = EvalMetrics::default();
    for sc in scored {
        if sc.should_abstain {
            if reachable_only && !sc.reachable { continue; }
            m.cx_total += 1;
            let replaced = sc.best_candidate.map(|(_, p)| p >= threshold).unwrap_or(false);
            if replaced { m.cx_replaced += 1; } else { m.cx_correct += 1; }
        } else {
            if reachable_only && !sc.reachable { continue; }
            if !sc.reachable { continue; } // skip unreachable canonical regardless
            m.canonical_total += 1;
            if let Some((alias_id, prob)) = sc.best_candidate {
                if prob >= threshold {
                    m.canonical_replaced += 1;
                    if sc.gold_alias_id == Some(alias_id) {
                        m.canonical_correct += 1;
                    }
                }
            }
        }
    }
    m
}

/// Train on a single probed case using the given mode (non-ablated).
fn train_case(judge: &mut OnlineJudge, pc: &ProbedCase, mode: &TrainMode) {
    match mode {
        TrainMode::None => {}
        TrainMode::TeachChoice { .. } => {
            if pc.case.should_abstain {
                if let Some(ps) = best_cx_span(pc) {
                    judge.teach_choice(&ps.span, &ps.candidates, Option::None, &ps.ctx);
                }
            } else if let Some(ps) = gold_span(pc) {
                judge.teach_choice(&ps.span, &ps.candidates, ps.gold_alias_id, &ps.ctx);
            }
        }
        TrainMode::CaseBalanced { hard_neg_cap, .. } => {
            if pc.case.should_abstain {
                if let Some(ps) = best_cx_span(pc) {
                    judge.train_balanced(&ps.span, &ps.candidates, Option::None, &ps.ctx, *hard_neg_cap);
                }
            } else if let Some(ps) = gold_span(pc) {
                judge.train_balanced(&ps.span, &ps.candidates, ps.gold_alias_id, &ps.ctx, *hard_neg_cap);
            }
        }
        TrainMode::CasewiseSoftmax { .. } => {
            if pc.case.should_abstain {
                if let Some(ps) = best_cx_span(pc) {
                    judge.train_softmax(&ps.span, &ps.candidates, Option::None, &ps.ctx);
                }
            } else if let Some(ps) = gold_span(pc) {
                judge.train_softmax(&ps.span, &ps.candidates, ps.gold_alias_id, &ps.ctx);
            }
        }
        TrainMode::FreezeDense { hard_neg_cap, .. } => {
            if pc.case.should_abstain {
                if let Some(ps) = best_cx_span(pc) {
                    judge.train_balanced(&ps.span, &ps.candidates, Option::None, &ps.ctx, *hard_neg_cap);
                }
            } else if let Some(ps) = gold_span(pc) {
                judge.train_balanced(&ps.span, &ps.candidates, ps.gold_alias_id, &ps.ctx, *hard_neg_cap);
            }
        }
    }
}

/// Train on a single probed case with feature ablation.
/// Uses build_examples + filter_features + direct model update.
fn train_case_ablated(
    judge: &mut OnlineJudge,
    pc: &ProbedCase,
    mode: &TrainMode,
    slice: beeml::judge::FeatureSlice,
) {
    use beeml::judge::{build_examples, filter_features};

    let ps = if pc.case.should_abstain {
        best_cx_span(pc)
    } else {
        gold_span(pc)
    };
    let Some(ps) = ps else { return };

    let examples = build_examples(&ps.span, &ps.candidates, &ps.ctx, &Default::default());
    if examples.is_empty() { return; }

    // Filter features for ablation
    let filtered: Vec<_> = examples.iter()
        .map(|e| filter_features(&e.features, slice))
        .collect();

    match mode {
        TrainMode::None => {}
        TrainMode::TeachChoice { .. } | TrainMode::CaseBalanced { .. } | TrainMode::FreezeDense { .. } => {
            // Case-balanced: same logic but with filtered features
            if let Some(gold_id) = ps.gold_alias_id {
                // Canonical
                if let Some(gold_idx) = examples.iter().position(|e| e.alias_id == gold_id) {
                    judge.model_mut().update(&filtered[gold_idx], true);
                }
                let hard_neg_cap = match mode {
                    TrainMode::CaseBalanced { hard_neg_cap, .. }
                    | TrainMode::FreezeDense { hard_neg_cap, .. } => *hard_neg_cap,
                    _ => examples.len(),
                };
                let mut neg_indices: Vec<usize> = (0..examples.len())
                    .filter(|&j| examples[j].alias_id != gold_id)
                    .collect();
                neg_indices.sort_by(|&a, &b| {
                    let sa = judge.model().predict(&filtered[a]);
                    let sb = judge.model().predict(&filtered[b]);
                    sb.total_cmp(&sa)
                });
                for &idx in neg_indices.iter().take(hard_neg_cap) {
                    judge.model_mut().update(&filtered[idx], false);
                }
            } else {
                // Counterexample: single hardest negative
                let hardest = (0..examples.len())
                    .max_by(|&a, &b| {
                        let sa = judge.model().predict(&filtered[a]);
                        let sb = judge.model().predict(&filtered[b]);
                        sa.total_cmp(&sb)
                    });
                if let Some(idx) = hardest {
                    judge.model_mut().update(&filtered[idx], false);
                }
            }
        }
        TrainMode::CasewiseSoftmax { .. } => {
            // Softmax with filtered features
            let gold_index = if let Some(gold_id) = ps.gold_alias_id {
                examples.iter().position(|e| e.alias_id == gold_id)
                    .unwrap_or(filtered.len()) // keep_original
            } else {
                filtered.len() // keep_original
            };
            // Add keep_original features (filtered)
            let keep_features = filter_features(
                &beeml::judge::build_keep_original_features(&ps.span, &ps.ctx),
                slice,
            );
            let mut all = filtered.clone();
            all.push(keep_features);
            judge.model_mut().update_softmax(&all, gold_index);
        }
    }
}

/// Score a span with ablated features, returns (alias_id, probability) pairs.
fn score_span_ablated(
    judge: &OnlineJudge,
    ps: &ProbedSpan,
    slice: beeml::judge::FeatureSlice,
) -> Vec<(u32, f32)> {
    use beeml::judge::{build_examples, filter_features};

    let examples = build_examples(&ps.span, &ps.candidates, &ps.ctx, &Default::default());
    examples.iter()
        .map(|e| {
            let filtered = filter_features(&e.features, slice);
            let prob = judge.model().predict_prob(&filtered) as f32;
            (e.alias_id, prob)
        })
        .collect()
}

/// Get the best counterexample span (most candidates = most likely false positive).
fn best_cx_span(pc: &ProbedCase) -> Option<&ProbedSpan> {
    pc.spans.iter()
        .filter(|ps| !ps.candidates.is_empty())
        .max_by_key(|ps| ps.candidates.len())
}

/// Get the span containing the gold alias.
fn gold_span(pc: &ProbedCase) -> Option<&ProbedSpan> {
    pc.spans.iter().find(|ps| ps.gold_alias_id.is_some())
}

#[derive(Clone, Debug)]
enum EvalFailureStage {
    RetrievalShortlist,
    Composition,
    Judge,
}

#[derive(Clone, Debug)]
enum GoldUnreachableReason {
    /// Target term not in shortlist at all
    TargetNotRetrieved,
    /// Target term found but other required edits are missing from candidates
    MissingRequiredEdits,
    /// All edits exist but composition/pruning didn't produce the gold combination
    CompositionDidNotProduce,
    /// Composition produced the right edits but rendered text doesn't match expected
    SurfaceMismatch { closest_sentence: String },
}

#[derive(Clone, Debug)]
enum EvalChoiceKind {
    KeepOriginal,
    SentenceCandidate,
}

struct CaseEvalResult {
    // Stage 1: Retrieval
    target_in_shortlist: bool,
    target_best_rank: Option<usize>,

    // Stage 2: Composition (judge-visible decision set)
    gold_reachable: bool,
    gold_choice_rank: Option<usize>,
    gold_unreachable_reason: Option<GoldUnreachableReason>,
    decision_set_size: usize,
    replacement_choice_count: usize,

    // Stage 3: Judge
    chosen_kind: EvalChoiceKind,
    chosen_choice_id: Option<String>,
    chosen_sentence: String,
    chosen_edit_count: usize,
    chosen_probability: f32,
    judge_correct: bool,

    // Attribution
    first_failure: Option<EvalFailureStage>,
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

        let total = cases.len() as u32;
        let eval_start = std::time::Instant::now();

        // Evaluate all cases in parallel using rayon.
        let service = self.clone();
        let eval_results: Vec<(usize, EvalCase, Result<CaseEvalResult, String>)> =
            tokio::task::block_in_place(|| {
                use rayon::prelude::*;
                cases
                    .into_par_iter()
                    .enumerate()
                    .map(|(idx, case)| {
                        let result = service.evaluate_case(&case, &request);
                        (idx, case, result)
                    })
                    .collect()
            });

        // Aggregate metrics sequentially and send progress.
        let mut canonical_cases = 0u32;
        let mut canonical_shortlist_found = 0u32;
        let mut canonical_gold_reachable = 0u32;
        let mut canonical_judge_correct = 0u32;
        let mut counterexample_cases = 0u32;
        let mut counterexample_replacement_built = 0u32;
        let mut counterexample_judge_correct = 0u32;
        let mut failures_at_retrieval = 0u32;
        let mut failures_at_composition = 0u32;
        let mut failures_at_judge = 0u32;
        let mut unreachable_not_retrieved = 0u32;
        let mut unreachable_missing_edits = 0u32;
        let mut unreachable_composition = 0u32;
        let mut unreachable_surface_mismatch = 0u32;

        // Legacy counters
        let mut judge_correct = 0u32;
        let mut judge_replace_correct = 0u32;
        let mut judge_abstain_correct = 0u32;
        let mut top1_hits = 0u32;
        let mut top3_hits = 0u32;
        let mut top10_hits = 0u32;
        let mut misses = Vec::new();
        let mut judge_failures = Vec::new();
        let mut per_term = HashMap::<String, RetrievalEvalTermSummary>::new();

        for (recording_id, case, result) in eval_results {
            let result = result?;

            // Per-term legacy tracking
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

            if case.should_abstain {
                // Counterexample
                counterexample_cases += 1;
                if result.replacement_choice_count > 0 {
                    counterexample_replacement_built += 1;
                }
                if result.judge_correct {
                    counterexample_judge_correct += 1;
                    judge_abstain_correct += 1;
                    judge_correct += 1;
                } else {
                    judge_failures.push(JudgeEvalFailure {
                        case_id: case.case_id.clone(),
                        suite: case.suite.to_string(),
                        target_term: case.target_term.clone(),
                        transcript: case.transcript.clone(),
                        expected_action: "keep_original".to_string(),
                        chosen_action: result.chosen_sentence.clone(),
                        chosen_span_text: result.chosen_choice_id.clone().unwrap_or_default(),
                        chosen_probability: result.chosen_probability,
                    });
                    // Counterexample judge failures tracked separately, not in the canonical funnel
                }
            } else {
                // Canonical
                canonical_cases += 1;
                if result.target_in_shortlist {
                    canonical_shortlist_found += 1;
                }
                if result.gold_reachable {
                    canonical_gold_reachable += 1;
                }
                if result.judge_correct {
                    canonical_judge_correct += 1;
                    judge_replace_correct += 1;
                    judge_correct += 1;
                } else {
                    judge_failures.push(JudgeEvalFailure {
                        case_id: case.case_id.clone(),
                        suite: case.suite.to_string(),
                        target_term: case.target_term.clone(),
                        transcript: case.transcript.clone(),
                        expected_action: case.target_term.clone(),
                        chosen_action: result.chosen_sentence.clone(),
                        chosen_span_text: result.chosen_choice_id.clone().unwrap_or_default(),
                        chosen_probability: result.chosen_probability,
                    });
                }

                // Failure attribution
                match &result.first_failure {
                    Some(EvalFailureStage::RetrievalShortlist) => failures_at_retrieval += 1,
                    Some(EvalFailureStage::Composition) => failures_at_composition += 1,
                    Some(EvalFailureStage::Judge) => failures_at_judge += 1,
                    None => {}
                }

                // Gold unreachable breakdown
                match &result.gold_unreachable_reason {
                    Some(GoldUnreachableReason::TargetNotRetrieved) => unreachable_not_retrieved += 1,
                    Some(GoldUnreachableReason::MissingRequiredEdits) => unreachable_missing_edits += 1,
                    Some(GoldUnreachableReason::CompositionDidNotProduce) => unreachable_composition += 1,
                    Some(GoldUnreachableReason::SurfaceMismatch { .. }) => unreachable_surface_mismatch += 1,
                    None => {}
                }

                // Legacy retrieval rank
                if let Some(rank) = result.target_best_rank {
                    if rank <= 1 { top1_hits += 1; entry.top1_hits += 1; }
                    if rank <= 3 { top3_hits += 1; entry.top3_hits += 1; }
                    if rank <= 10 { top10_hits += 1; entry.top10_hits += 1; }
                } else {
                    misses.push(RetrievalEvalMiss {
                        recording_id: recording_id as u32,
                        suite: case.suite.to_string(),
                        term: case.target_term.clone(),
                        transcript: case.transcript.clone(),
                        best_span_text: result.chosen_sentence.clone(),
                    });
                }
            }

            let _ = progress.send(RetrievalPrototypeEvalProgress {
                evaluated: recording_id as u32 + 1,
                total,
                judge_correct,
            }).await;
        }

        let mut per_term = per_term.into_values().collect::<Vec<_>>();
        per_term.sort_by(|a, b| a.term.cmp(&b.term));

        let eval_elapsed = eval_start.elapsed();
        info!(
            canonical = canonical_cases,
            shortlist = canonical_shortlist_found,
            reachable = canonical_gold_reachable,
            judge = canonical_judge_correct,
            cx = counterexample_cases,
            cx_leak = counterexample_replacement_built,
            cx_judge = counterexample_judge_correct,
            fail_retrieval = failures_at_retrieval,
            fail_composition = failures_at_composition,
            fail_judge = failures_at_judge,
            unreach_not_retrieved = unreachable_not_retrieved,
            unreach_missing_edits = unreachable_missing_edits,
            unreach_composition = unreachable_composition,
            unreach_surface = unreachable_surface_mismatch,
            total_ms = eval_elapsed.as_millis() as u64,
            "eval complete"
        );

        Ok(RetrievalPrototypeEvalResult {
            evaluated_cases: total,
            canonical_cases,
            canonical_shortlist_found,
            canonical_gold_reachable,
            canonical_judge_correct,
            counterexample_cases,
            counterexample_replacement_built,
            counterexample_judge_correct,
            failures_at_retrieval,
            failures_at_composition,
            failures_at_judge,
            unreachable_not_retrieved,
            unreachable_missing_edits,
            unreachable_composition,
            unreachable_surface_mismatch,
            judge_correct,
            judge_replace_correct,
            judge_abstain_correct,
            top1_hits,
            top3_hits,
            top10_hits,
            misses,
            judge_failures,
            per_term,
        })
    }

    async fn run_offline_judge_eval(
        &self,
        request: OfflineJudgeEvalRequest,
    ) -> Result<OfflineJudgeEvalResult, String> {
        let folds = request.folds.max(2) as usize;
        let train_epochs = request.train_epochs.max(1) as usize;
        let max_span_words = if request.max_span_words == 0 { 3 } else { request.max_span_words };
        let shortlist_limit = if request.shortlist_limit == 0 { 100 } else { request.shortlist_limit };

        // Step 1: Load all cases and probe spans (no judge involved).
        let cases = self.teaching_cases(0, true);
        info!(cases = cases.len(), folds, train_epochs, "starting offline judge eval");

        let service = self.clone();
        let probed_cases: Vec<ProbedCase> = tokio::task::block_in_place(|| {
            use rayon::prelude::*;
            cases
                .into_par_iter()
                .filter_map(|case| {
                    match service.probe_case_spans(&case, max_span_words, shortlist_limit) {
                        Ok(pc) => Some(pc),
                        Err(e) => {
                            tracing::warn!(case_id = %case.case_id, error = %e, "probe failed");
                            None
                        }
                    }
                })
                .collect()
        });

        info!(probed = probed_cases.len(), "probed all cases");

        // Step 2: Term-stratified k-fold split.
        let mut terms: Vec<String> = probed_cases
            .iter()
            .map(|pc| pc.case.target_term.to_ascii_lowercase())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        terms.sort();

        let term_to_fold: HashMap<String, usize> = terms
            .iter()
            .enumerate()
            .map(|(i, term)| (term.clone(), i % folds))
            .collect();

        let case_folds: Vec<usize> = probed_cases
            .iter()
            .map(|pc| {
                *term_to_fold
                    .get(&pc.case.target_term.to_ascii_lowercase())
                    .unwrap_or(&0)
            })
            .collect();

        // ── Dataset summary ──────────────────────────────────────────────
        println!("\n=== Phase 4 Offline Judge Eval ({folds}-fold CV, {train_epochs} epochs) ===");

        // Reachability summary
        let canonical_count = probed_cases.iter().filter(|pc| !pc.case.should_abstain).count();
        let canonical_with_gold = probed_cases.iter()
            .filter(|pc| !pc.case.should_abstain)
            .filter(|pc| pc.spans.iter().any(|ps| ps.gold_alias_id.is_some()))
            .count();
        let canonical_gold_verified = probed_cases.iter()
            .filter(|pc| !pc.case.should_abstain)
            .filter(|pc| pc.spans.iter().any(|ps| {
                if let Some(gold_id) = ps.gold_alias_id {
                    ps.candidates.iter().any(|(c, _)| c.alias_id == gold_id && c.verified)
                } else {
                    false
                }
            }))
            .count();
        let cx_count = probed_cases.iter().filter(|pc| pc.case.should_abstain).count();
        let cx_with_candidates = probed_cases.iter()
            .filter(|pc| pc.case.should_abstain)
            .filter(|pc| pc.spans.iter().any(|ps| !ps.candidates.is_empty()))
            .count();
        println!("\n  Dataset: {canonical_count} canonical ({canonical_with_gold} gold retrieved, {canonical_gold_verified} gold verified), {cx_count} counterexamples ({cx_with_candidates} with candidates)");

        // ── Feature activation diagnostics ──────────────────────────────
        {
            use beeml::judge::{build_examples, FeatureSlice, SPARSE_OFFSET, NUM_DENSE};
            println!("\n--- Feature activation diagnostics ---");

            let mut total_examples = 0u64;
            let mut dense_nonzero_sum = 0u64;
            let mut sparse_nonzero_sum = 0u64;
            let mut sparse_feature_counts: HashMap<u64, u32> = HashMap::new();
            let mut dense_abs_sums = vec![0.0f64; NUM_DENSE];
            let mut dense_nonzero_counts = vec![0u64; NUM_DENSE];

            for pc in &probed_cases {
                for ps in &pc.spans {
                    if ps.candidates.is_empty() { continue; }
                    let examples = build_examples(&ps.span, &ps.candidates, &ps.ctx, &Default::default());
                    for ex in &examples {
                        total_examples += 1;
                        for f in &ex.features {
                            if f.index < NUM_DENSE as u64 {
                                if f.value != 0.0 {
                                    dense_nonzero_sum += 1;
                                    dense_abs_sums[f.index as usize] += f.value.abs();
                                    dense_nonzero_counts[f.index as usize] += 1;
                                }
                            } else {
                                if f.value != 0.0 {
                                    sparse_nonzero_sum += 1;
                                    *sparse_feature_counts.entry(f.index).or_default() += 1;
                                }
                            }
                        }
                    }
                }
            }

            let unique_sparse = sparse_feature_counts.len();
            println!("  Total examples: {total_examples}");
            println!("  Avg dense nonzero: {:.1}", dense_nonzero_sum as f64 / total_examples.max(1) as f64);
            println!("  Avg sparse nonzero: {:.1}", sparse_nonzero_sum as f64 / total_examples.max(1) as f64);
            println!("  Unique sparse features: {unique_sparse}");

            // Top 20 most frequent sparse features
            let mut sparse_sorted: Vec<_> = sparse_feature_counts.iter().collect();
            sparse_sorted.sort_by(|a, b| b.1.cmp(a.1));
            println!("  Top 20 sparse features by frequency:");
            for (idx, count) in sparse_sorted.iter().take(20) {
                println!("    bucket {idx}: {count} activations");
            }

            // Dense feature activation rates
            println!("  Dense feature activation (nonzero rate, avg magnitude when active):");
            for (i, name) in beeml::judge::FEATURE_NAMES.iter().enumerate() {
                let rate = if total_examples > 0 { dense_nonzero_counts[i] as f64 / total_examples as f64 * 100.0 } else { 0.0 };
                let avg_mag = if dense_nonzero_counts[i] > 0 { dense_abs_sums[i] / dense_nonzero_counts[i] as f64 } else { 0.0 };
                if rate > 0.0 {
                    println!("    [{i:2}] {name:<30} {rate:5.1}%  avg={avg_mag:.4}");
                }
            }
        }

        // ── Eval 1: Baselines ──────────────────────────────────────────

        let thresholds: &[f32] = &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

        fn print_sweep(label: &str, scored: &[ScoredCase], thresholds: &[f32], reachable_only: bool) {
            println!("\n  [{label}] threshold sweep:");
            for &t in thresholds {
                let m = eval_at_threshold(scored, t, reachable_only);
                println!(
                    "    T={t:.1}  can {}/{} ({:.1}%)  cx {}/{} ({:.1}%)  bal {:.1}%  repl: can {:.1}% cx {:.1}%",
                    m.canonical_correct, m.canonical_total, m.canonical_pct(),
                    m.cx_correct, m.cx_total, m.cx_pct(),
                    m.balanced_pct(),
                    m.canonical_replace_pct(), m.cx_replace_pct(),
                );
            }
        }

        fn best_threshold(scored: &[ScoredCase], thresholds: &[f32], reachable_only: bool) -> (f32, EvalMetrics) {
            let mut best_t = 0.5f32;
            let mut best_bal = 0.0f64;
            let mut best_m = EvalMetrics::default();
            for &t in thresholds {
                let m = eval_at_threshold(scored, t, reachable_only);
                if m.balanced_pct() > best_bal {
                    best_bal = m.balanced_pct();
                    best_t = t;
                    best_m = m;
                }
            }
            (best_t, best_m)
        }

        println!("\n--- Eval 1: Baselines ---");

        // 1a. Deterministic baseline
        {
            println!("\n  [deterministic] acceptance_score threshold sweep:");
            let det_thresholds: &[f32] = &[0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
            for &t in det_thresholds {
                let m = eval_deterministic_kfold(&probed_cases, &case_folds, folds, t);
                println!(
                    "    T={t:.1}  can {}/{} ({:.1}%)  cx {}/{} ({:.1}%)  bal {:.1}%  repl: can {:.1}% cx {:.1}%",
                    m.canonical_correct, m.canonical_total, m.canonical_pct(),
                    m.cx_correct, m.cx_total, m.cx_pct(),
                    m.balanced_pct(),
                    m.canonical_replace_pct(), m.cx_replace_pct(),
                );
            }
        }

        // 1b. Seed-only baseline — train once, sweep thresholds
        let scored_seed = train_and_score_kfold(&probed_cases, &case_folds, folds, TrainMode::None, beeml::judge::FeatureSlice::All);
        print_sweep("seed_only", &scored_seed, thresholds, false);

        // 1c. Taught (current teach_choice replay)
        let scored_taught = train_and_score_kfold(&probed_cases, &case_folds, folds, TrainMode::TeachChoice { epochs: train_epochs }, beeml::judge::FeatureSlice::All);
        print_sweep("taught", &scored_taught, thresholds, false);

        // ── Eval 2+3: Case-balanced FTRL + threshold sweep ─────────────
        println!("\n--- Eval 2: Case-balanced FTRL ---");
        let scored_balanced = train_and_score_kfold(&probed_cases, &case_folds, folds, TrainMode::CaseBalanced { epochs: train_epochs, hard_neg_cap: 3 }, beeml::judge::FeatureSlice::All);
        print_sweep("case_balanced", &scored_balanced, thresholds, false);

        // ── Eval 4: Feature ablation (on case-balanced) ────────────────
        println!("\n--- Eval 4: Feature ablation (case-balanced, best threshold) ---");
        {
            use beeml::judge::FeatureSlice;
            let slices = [
                FeatureSlice::PhoneticOnly,
                FeatureSlice::PlusAsr,
                FeatureSlice::PlusContext,
                FeatureSlice::All,
            ];
            for slice in &slices {
                let scored = train_and_score_kfold(&probed_cases, &case_folds, folds, TrainMode::CaseBalanced { epochs: train_epochs, hard_neg_cap: 3 }, *slice);
                let (bt, m) = best_threshold(&scored, thresholds, false);
                println!(
                    "  {:<16} T={bt:.1}  can {}/{} ({:.1}%)  cx {}/{} ({:.1}%)  bal {:.1}%  repl: can {:.1}% cx {:.1}%",
                    slice.name(),
                    m.canonical_correct, m.canonical_total, m.canonical_pct(),
                    m.cx_correct, m.cx_total, m.cx_pct(),
                    m.balanced_pct(),
                    m.canonical_replace_pct(), m.cx_replace_pct(),
                );
            }
        }

        // ── Eval 5: Reachable-only ─────────────────────────────────────
        println!("\n--- Eval 5: Reachable-only (case-balanced) ---");
        // Reuse scored_balanced, just filter by reachable
        print_sweep("reachable_only", &scored_balanced, thresholds, true);

        // ── Eval 6: Formulation comparison ─────────────────────────────
        println!("\n--- Eval 6: Formulation comparison (best threshold each) ---");
        {
            let formulations: &[(&str, TrainMode)] = &[
                ("independent_binary", TrainMode::TeachChoice { epochs: train_epochs }),
                ("case_balanced", TrainMode::CaseBalanced { epochs: train_epochs, hard_neg_cap: 3 }),
                ("freeze_dense", TrainMode::FreezeDense { epochs: train_epochs, hard_neg_cap: 3 }),
                ("casewise_softmax", TrainMode::CasewiseSoftmax { epochs: train_epochs }),
            ];
            for (name, train_mode) in formulations {
                let scored = train_and_score_kfold(&probed_cases, &case_folds, folds, train_mode.clone(), beeml::judge::FeatureSlice::All);
                let (bt, m) = best_threshold(&scored, thresholds, false);
                println!(
                    "  {name:<24} T={bt:.1}  can {}/{} ({:.1}%)  cx {}/{} ({:.1}%)  bal {:.1}%  repl: can {:.1}% cx {:.1}%",
                    m.canonical_correct, m.canonical_total, m.canonical_pct(),
                    m.cx_correct, m.cx_total, m.cx_pct(),
                    m.balanced_pct(),
                    m.canonical_replace_pct(), m.cx_replace_pct(),
                );
            }
        }

        // ── Probability distribution diagnostics ────────────────────────
        println!("\n--- Probability distributions (case-balanced model) ---");
        {
            let mut gold_probs: Vec<f32> = Vec::new();
            let mut cx_top_probs: Vec<f32> = Vec::new();

            for sc in &scored_balanced {
                if sc.should_abstain {
                    if let Some((_, prob)) = sc.best_candidate {
                        cx_top_probs.push(prob);
                    }
                } else if sc.reachable {
                    if let Some((alias_id, prob)) = sc.best_candidate {
                        if sc.gold_alias_id == Some(alias_id) {
                            gold_probs.push(prob);
                        }
                    }
                }
            }

            gold_probs.sort_by(|a, b| a.total_cmp(b));
            cx_top_probs.sort_by(|a, b| a.total_cmp(b));

            fn percentiles(vals: &[f32]) -> String {
                if vals.is_empty() { return "N/A".to_string(); }
                let p = |pct: f64| -> f32 {
                    let idx = ((vals.len() as f64 - 1.0) * pct).round() as usize;
                    vals[idx.min(vals.len() - 1)]
                };
                format!("n={:<4} min={:.3} p25={:.3} p50={:.3} p75={:.3} max={:.3}",
                    vals.len(), p(0.0), p(0.25), p(0.5), p(0.75), p(1.0))
            }

            println!("  Gold candidate prob (canonical, gold=best):  {}", percentiles(&gold_probs));
            println!("  Top negative prob (counterexample):           {}", percentiles(&cx_top_probs));

            // Also show where gold candidate is NOT the best
            let mut gold_not_best_probs: Vec<f32> = Vec::new();
            let mut gold_not_best_best: Vec<f32> = Vec::new();
            for sc in &scored_balanced {
                if !sc.should_abstain && sc.reachable {
                    if let Some((alias_id, _prob)) = sc.best_candidate {
                        if sc.gold_alias_id != Some(alias_id) {
                            // Gold was not the top candidate — find gold prob
                            // We don't have it directly, but we know it's not the best
                            gold_not_best_best.push(_prob);
                        }
                    }
                }
            }
            if !gold_not_best_best.is_empty() {
                gold_not_best_best.sort_by(|a, b| a.total_cmp(b));
                println!("  Best-non-gold prob (canonical, gold!=best):   {}", percentiles(&gold_not_best_best));
                println!("  Cases where gold is NOT best candidate: {}", gold_not_best_best.len());
            }
        }

        // ── One-case training trace ────────────────────────────────────
        println!("\n--- One-case training trace ---");
        {
            // Pick first canonical case with gold span and first cx case with candidates
            let first_canonical = probed_cases.iter()
                .find(|pc| !pc.case.should_abstain && pc.spans.iter().any(|ps| ps.gold_alias_id.is_some()));
            let first_cx = probed_cases.iter()
                .find(|pc| pc.case.should_abstain && pc.spans.iter().any(|ps| !ps.candidates.is_empty()));

            if let (Some(can_case), Some(cx_case)) = (first_canonical, first_cx) {
                for (mode_name, mode) in &[
                    ("teach_choice", TrainMode::TeachChoice { epochs: 1 }),
                    ("case_balanced", TrainMode::CaseBalanced { epochs: 1, hard_neg_cap: 3 }),
                ] {
                    println!("\n  [{mode_name}] training trace:");
                    let mut judge = OnlineJudge::new_quiet();

                    // Score before training
                    let can_span = gold_span(can_case).unwrap();
                    let cx_span = best_cx_span(cx_case).unwrap();

                    let pre_can = judge.score_candidates(&can_span.span, &can_span.candidates, &can_span.ctx);
                    let gold_pre = pre_can.iter()
                        .find(|o| o.alias_id == can_span.gold_alias_id)
                        .map(|o| o.probability).unwrap_or(0.0);
                    let best_pre = pre_can.iter()
                        .filter(|o| !o.is_keep_original)
                        .max_by(|a, b| a.probability.total_cmp(&b.probability))
                        .map(|o| o.probability).unwrap_or(0.0);

                    let pre_cx = judge.score_candidates(&cx_span.span, &cx_span.candidates, &cx_span.ctx);
                    let cx_best_pre = pre_cx.iter()
                        .filter(|o| !o.is_keep_original)
                        .max_by(|a, b| a.probability.total_cmp(&b.probability))
                        .map(|o| o.probability).unwrap_or(0.0);

                    println!("    Before training:");
                    println!("      canonical gold prob={gold_pre:.4}, best prob={best_pre:.4} (term={})", can_case.case.target_term);
                    println!("      counterex best prob={cx_best_pre:.4} (term={})", cx_case.case.target_term);

                    // Train on canonical case
                    train_case(&mut judge, can_case, mode);
                    let post_can1 = judge.score_candidates(&can_span.span, &can_span.candidates, &can_span.ctx);
                    let gold_post1 = post_can1.iter()
                        .find(|o| o.alias_id == can_span.gold_alias_id)
                        .map(|o| o.probability).unwrap_or(0.0);
                    let post_cx1 = judge.score_candidates(&cx_span.span, &cx_span.candidates, &cx_span.ctx);
                    let cx_best_post1 = post_cx1.iter()
                        .filter(|o| !o.is_keep_original)
                        .max_by(|a, b| a.probability.total_cmp(&b.probability))
                        .map(|o| o.probability).unwrap_or(0.0);

                    println!("    After training on 1 canonical:");
                    println!("      canonical gold prob={gold_post1:.4} (delta={:+.4})", gold_post1 - gold_pre);
                    println!("      counterex best prob={cx_best_post1:.4} (delta={:+.4})", cx_best_post1 - cx_best_pre);

                    // Train on counterexample case
                    train_case(&mut judge, cx_case, mode);
                    let post_can2 = judge.score_candidates(&can_span.span, &can_span.candidates, &can_span.ctx);
                    let gold_post2 = post_can2.iter()
                        .find(|o| o.alias_id == can_span.gold_alias_id)
                        .map(|o| o.probability).unwrap_or(0.0);
                    let post_cx2 = judge.score_candidates(&cx_span.span, &cx_span.candidates, &cx_span.ctx);
                    let cx_best_post2 = post_cx2.iter()
                        .filter(|o| !o.is_keep_original)
                        .max_by(|a, b| a.probability.total_cmp(&b.probability))
                        .map(|o| o.probability).unwrap_or(0.0);

                    println!("    After training on 1 counterexample:");
                    println!("      canonical gold prob={gold_post2:.4} (delta={:+.4} from canonical-only)", gold_post2 - gold_post1);
                    println!("      counterex best prob={cx_best_post2:.4} (delta={:+.4} from canonical-only)", cx_best_post2 - cx_best_post1);

                    // Weight norm
                    let weights = judge.weights();
                    let weight_norm: f64 = weights.iter().map(|w| (*w as f64) * (*w as f64)).sum::<f64>().sqrt();
                    println!("    Weight L2 norm: {weight_norm:.4}");
                    println!("    Active features: {}", judge.model().num_active());
                }
            }
        }

        // Return a summary (the RPC result is less important than stdout now)
        Ok(OfflineJudgeEvalResult {
            canonical_correct: 0,
            canonical_total: 0,
            counterexample_correct: 0,
            counterexample_total: 0,
            fold_results: vec![],
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
            g2p: Mutex::new(CachedEspeakG2p::english().context("initializing g2p engine")?),
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

fn load_correction_events(path: &std::path::Path) -> anyhow::Result<Vec<beeml::judge::CorrectionEvent>> {
    use std::io::BufRead;
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let mut events = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str(&line) {
            Ok(event) => events.push(event),
            Err(e) => tracing::warn!(error = %e, "skipping malformed event line"),
        }
    }
    // Cap at 10,000 events (keep most recent)
    if events.len() > 10_000 {
        events = events.split_off(events.len() - 10_000);
    }
    Ok(events)
}

fn save_correction_events(path: &std::path::Path, events: &[beeml::judge::CorrectionEvent]) -> anyhow::Result<()> {
    use std::io::Write;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::File::create(path)?;
    // Cap at 10,000 events (keep most recent)
    let start = events.len().saturating_sub(10_000);
    for event in &events[start..] {
        serde_json::to_writer(&mut file, event)?;
        writeln!(file)?;
    }
    Ok(())
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
