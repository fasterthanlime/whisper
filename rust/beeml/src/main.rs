use std::collections::HashMap;
use std::env;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

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
    JudgeEvalFailure, JudgeOptionDebug, JudgeStateDebug,
    RetrievalEvalMiss, RetrievalEvalTermSummary, RetrievalPrototypeTeachingCase,
    RetrievalPrototypeTeachingDeckRequest, RetrievalPrototypeTeachingDeckResult,
    RetrievalCandidateDebug, RetrievalIndexView, RetrievalPrototypeEvalRequest,
    RetrievalPrototypeEvalResult, RetrievalPrototypeProbeRequest,
    RetrievalPrototypeProbeResult, SpanDebugTrace, SpanDebugView, TeachRetrievalPrototypeJudgeRequest,
    TermAliasView, TermInspectionRequest, TermInspectionResult, TimingBreakdown,
    TranscribeWavResult,
};
use serde::Deserialize;
use tokio::net::TcpListener;
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

impl BeeMlService {
    fn run_probe(
        &self,
        request: RetrievalPrototypeProbeRequest,
        teach: Option<TeachRetrievalPrototypeJudgeRequest>,
    ) -> Result<RetrievalPrototypeProbeResult, String> {
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
                let scored_rows = score_shortlist(&span, &shortlist, &self.inner.index)
                    .into_iter()
                    .take(request.verify_limit as usize)
                    .collect::<Vec<_>>();
                let judge_input = scored_rows
                    .iter()
                    .map(|row| {
                        let alias = &self.inner.index.aliases[row.alias_id as usize];
                        (row.clone(), alias.identifier_flags.clone())
                    })
                    .collect::<Vec<_>>();

                let should_teach = teach.as_ref().is_some_and(|teach| {
                    teach.span_token_start as usize == span.token_start
                        && teach.span_token_end as usize == span.token_end
                });
                if should_teach {
                    taught_span = true;
                }

                let chosen_alias_id = teach.as_ref().and_then(|teach| {
                    if should_teach && !teach.choose_keep_original {
                        teach.chosen_alias_id
                    } else {
                        None
                    }
                });
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

        if teach.is_some() && !taught_span {
            return Err("requested span was not present in the probe result".to_string());
        }

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

        if limit == 0 || limit >= cases.len() {
            cases
        } else {
            cases.into_iter().take(limit).collect()
        }
    }

    fn spans_for_transcript(
        &self,
        transcript: &str,
        max_span_words: usize,
    ) -> Result<Vec<TranscriptSpan>, String> {
        let mut g2p = self
            .inner
            .g2p
            .lock()
            .map_err(|_| "g2p cache mutex poisoned".to_string())?;
        Ok(enumerate_transcript_spans_with(
            transcript,
            max_span_words,
            None::<&[TranscriptAlignmentToken]>,
            |text| g2p.ipa_tokens(text).ok().flatten(),
        ))
    }

    fn evaluate_case(
        &self,
        judge: &OnlineJudge,
        case: &EvalCase,
        request: &RetrievalPrototypeEvalRequest,
    ) -> Result<CaseEvalResult, String> {
        let spans = self.spans_for_transcript(&case.transcript, request.max_span_words as usize)?;
        let mut best_by_term: HashMap<String, (String, bee_phonetic::CandidateFeatureRow)> =
            HashMap::new();
        let mut best_judge_choice: Option<CaseJudgeChoice> = None;

        for span in spans {
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
            let scored_rows = score_shortlist(&span, &shortlist, &self.inner.index)
                .into_iter()
                .take(request.verify_limit as usize)
                .collect::<Vec<_>>();

            for candidate in scored_rows.iter().filter(|candidate| candidate.verified) {
                match best_by_term.get(&candidate.term) {
                    Some((_, existing))
                        if compare_candidate_rows(existing, candidate).is_ge() => {}
                    _ => {
                        best_by_term.insert(
                            candidate.term.clone(),
                            (span.text.clone(), candidate.clone()),
                        );
                    }
                }
            }

            let judge_input = scored_rows
                .iter()
                .map(|row| {
                    let alias = &self.inner.index.aliases[row.alias_id as usize];
                    (row.clone(), alias.identifier_flags.clone())
                })
                .collect::<Vec<_>>();
            let judge_options = judge.score_candidates(&span, &judge_input);
            if let Some(chosen) = judge_options.into_iter().find(|option| option.chosen) {
                match &best_judge_choice {
                    Some(existing)
                        if existing.probability > chosen.probability
                            || (existing.probability == chosen.probability
                                && existing.score >= chosen.score) => {}
                    _ => {
                        best_judge_choice = Some(CaseJudgeChoice {
                            span_text: span.text.clone(),
                            chosen_action: if chosen.is_keep_original {
                                "keep_original".to_string()
                            } else {
                                chosen.term.clone()
                            },
                            probability: chosen.probability,
                            score: chosen.score,
                            is_keep_original: chosen.is_keep_original,
                        });
                    }
                }
            }
        }

        let mut ranked = best_by_term.into_iter().collect::<Vec<_>>();
        ranked.sort_by(|(a_term, a_hit), (b_term, b_hit)| {
            compare_candidate_rows(&b_hit.1, &a_hit.1).then_with(|| a_term.cmp(b_term))
        });
        let target_rank = ranked
            .iter()
            .position(|(term, _)| term.eq_ignore_ascii_case(&case.target_term))
            .map(|idx| idx + 1);
        let best_span_text = ranked
            .first()
            .map(|(_, hit)| hit.0.clone())
            .unwrap_or_default();
        let judge_choice = best_judge_choice.unwrap_or(CaseJudgeChoice {
            span_text: String::new(),
            chosen_action: "keep_original".to_string(),
            probability: 0.0,
            score: 0.0,
            is_keep_original: true,
        });

        Ok(CaseEvalResult {
            target_rank,
            best_span_text,
            judge_choice,
        })
    }
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
    ) -> Result<RetrievalPrototypeEvalResult, String> {
        let cases = self.teaching_cases(request.limit as usize, true);
        let judge = self
            .inner
            .judge
            .lock()
            .map_err(|_| "judge mutex poisoned".to_string())?
            .clone();

        let mut top1_hits = 0u32;
        let mut top3_hits = 0u32;
        let mut top10_hits = 0u32;
        let mut judge_correct = 0u32;
        let mut judge_replace_correct = 0u32;
        let mut judge_abstain_correct = 0u32;
        let mut misses = Vec::new();
        let mut judge_failures = Vec::new();
        let mut per_term = HashMap::<String, RetrievalEvalTermSummary>::new();

        for (recording_id, case) in cases.iter().enumerate() {
            let result = self.evaluate_case(&judge, case, &request)?;
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
        }

        let mut per_term = per_term.into_values().collect::<Vec<_>>();
        per_term.sort_by(|a, b| a.term.cmp(&b.term));

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

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
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

    eprintln!("loading ASR engine from {}", model_dir.display());
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

    eprintln!("beeml vox websocket server listening on ws://{listen_addr}");

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
                    eprintln!("websocket handshake failed for {peer_addr}: {error}");
                    return;
                }
            };

            let establish = vox_core::acceptor_on(link)
                .on_connection(beeml::rpc::BeeMlDispatcher::new(handler))
                .establish::<NoopClient>()
                .await;

            match establish {
                Ok(client) => {
                    eprintln!("client connected: {peer_addr}");
                    client.caller.closed().await;
                    eprintln!("client disconnected: {peer_addr}");
                }
                Err(error) => {
                    eprintln!("vox session establish failed for {peer_addr}: {error}");
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
