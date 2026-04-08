use std::collections::HashMap;

use bee_transcribe::SessionOptions;
use bee_zipa_mlx::audio::AudioBuffer;
use beeml::judge::OnlineJudge;
use beeml::rpc::{
    AcceptedEdit, BeeMl, CorpusCapturePlanResult, CorrectionDebugResult, CorrectionRequest,
    CorrectionResult, DeleteCorpusRecordingRequest, DeleteCorpusRecordingResult, JudgeEvalFailure,
    ModelSummary, OfflineJudgeEvalRequest, OfflineJudgeEvalResult, PhoneticComparisonRequest,
    PhoneticComparisonResult, ProbDistribution, RerankerDebugTrace, RetrievalEvalMiss,
    RetrievalEvalTermSummary, RetrievalPrototypeEvalProgress, RetrievalPrototypeEvalRequest,
    RetrievalPrototypeEvalResult, RetrievalPrototypeProbeRequest, RetrievalPrototypeProbeResult,
    RetrievalPrototypeTeachingCase, RetrievalPrototypeTeachingDeckRequest,
    RetrievalPrototypeTeachingDeckResult, SaveCorpusRecordingRequest, SaveCorpusRecordingResult,
    TeachRetrievalPrototypeJudgeRequest, TermAliasView, TermInspectionRequest,
    TermInspectionResult, ThresholdRow, TimingBreakdown, TranscribeWavResult, TwoStageGridPoint,
    TwoStageResult,
};
use tracing::info;
use vox::{Rx, Tx};

use crate::offline_eval::*;
use crate::service::{BeeMlService, EvalCase};
use crate::util::*;

impl BeeMl for BeeMlService {
    async fn transcribe_wav(&self, wav_bytes: Vec<u8>) -> Result<TranscribeWavResult, String> {
        let samples = bee_transcribe::decode_wav(&wav_bytes).map_err(|e| e.to_string())?;
        let audio = AudioBuffer {
            samples: samples.clone(),
            sample_rate_hz: 16_000,
        };

        let mut session = self
            .inner
            .engine
            .session(SessionOptions::default())
            .map_err(|e| e.to_string())?;

        session.feed(&samples).map_err(|e| e.to_string())?;
        let result = session.finish().map_err(|e| e.to_string())?;
        let update = result.update;
        let phonetic_trace = self
            .build_transcribe_phonetic_trace(&audio, &update.text, &update.alignments)
            .ok();

        Ok(TranscribeWavResult {
            transcript: update.text,
            words: update.alignments,
            phonetic_trace,
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
            info!(update_count = judge.update_count(), "running eval");
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
            let entry =
                per_term
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
                    Some(GoldUnreachableReason::TargetNotRetrieved) => {
                        unreachable_not_retrieved += 1
                    }
                    Some(GoldUnreachableReason::MissingRequiredEdits) => {
                        unreachable_missing_edits += 1
                    }
                    Some(GoldUnreachableReason::CompositionDidNotProduce) => {
                        unreachable_composition += 1
                    }
                    Some(GoldUnreachableReason::SurfaceMismatch { .. }) => {
                        unreachable_surface_mismatch += 1
                    }
                    None => {}
                }

                // Legacy retrieval rank
                if let Some(rank) = result.target_best_rank {
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
                        best_span_text: result.chosen_sentence.clone(),
                    });
                }
            }

            let _ = progress
                .send(RetrievalPrototypeEvalProgress {
                    evaluated: recording_id as u32 + 1,
                    total,
                    judge_correct,
                })
                .await;
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
        let max_span_words = if request.max_span_words == 0 {
            3
        } else {
            request.max_span_words
        };
        let shortlist_limit = if request.shortlist_limit == 0 {
            100
        } else {
            request.shortlist_limit
        };

        // Step 1: Load all cases and probe spans (no judge involved).
        let cases = self.teaching_cases(0, true);
        info!(
            cases = cases.len(),
            folds, train_epochs, "starting offline judge eval"
        );

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
        let canonical_count = probed_cases
            .iter()
            .filter(|pc| !pc.case.should_abstain)
            .count();
        let canonical_with_gold = probed_cases
            .iter()
            .filter(|pc| !pc.case.should_abstain)
            .filter(|pc| pc.spans.iter().any(|ps| ps.gold_alias_id.is_some()))
            .count();
        let canonical_gold_verified = probed_cases
            .iter()
            .filter(|pc| !pc.case.should_abstain)
            .filter(|pc| {
                pc.spans.iter().any(|ps| {
                    if let Some(gold_id) = ps.gold_alias_id {
                        ps.candidates
                            .iter()
                            .any(|(c, _)| c.alias_id == gold_id && c.verified)
                    } else {
                        false
                    }
                })
            })
            .count();
        let cx_count = probed_cases
            .iter()
            .filter(|pc| pc.case.should_abstain)
            .count();
        let cx_with_candidates = probed_cases
            .iter()
            .filter(|pc| pc.case.should_abstain)
            .filter(|pc| pc.spans.iter().any(|ps| !ps.candidates.is_empty()))
            .count();
        println!(
            "\n  Dataset: {canonical_count} canonical ({canonical_with_gold} gold retrieved, {canonical_gold_verified} gold verified), {cx_count} counterexamples ({cx_with_candidates} with candidates)"
        );

        // ── Feature activation diagnostics ──────────────────────────────
        {
            use beeml::judge::{NUM_DENSE, build_examples};
            println!("\n--- Feature activation diagnostics ---");

            let mut total_examples = 0u64;
            let mut dense_nonzero_sum = 0u64;
            let mut sparse_nonzero_sum = 0u64;
            let mut sparse_feature_counts: HashMap<u64, u32> = HashMap::new();
            let mut dense_abs_sums = vec![0.0f64; NUM_DENSE];
            let mut dense_nonzero_counts = vec![0u64; NUM_DENSE];

            for pc in &probed_cases {
                for ps in &pc.spans {
                    if ps.candidates.is_empty() {
                        continue;
                    }
                    let examples =
                        build_examples(&ps.span, &ps.candidates, &ps.ctx, &Default::default());
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
            println!(
                "  Avg dense nonzero: {:.1}",
                dense_nonzero_sum as f64 / total_examples.max(1) as f64
            );
            println!(
                "  Avg sparse nonzero: {:.1}",
                sparse_nonzero_sum as f64 / total_examples.max(1) as f64
            );
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
                let rate = if total_examples > 0 {
                    dense_nonzero_counts[i] as f64 / total_examples as f64 * 100.0
                } else {
                    0.0
                };
                let avg_mag = if dense_nonzero_counts[i] > 0 {
                    dense_abs_sums[i] / dense_nonzero_counts[i] as f64
                } else {
                    0.0
                };
                if rate > 0.0 {
                    println!("    [{i:2}] {name:<30} {rate:5.1}%  avg={avg_mag:.4}");
                }
            }
        }

        // ── Eval 1: Baselines ──────────────────────────────────────────

        let thresholds: &[f32] = &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

        fn metrics_to_row(t: f32, m: &EvalMetrics) -> ThresholdRow {
            ThresholdRow {
                threshold: t,
                canonical_correct: m.canonical_correct,
                canonical_total: m.canonical_total,
                cx_correct: m.cx_correct,
                cx_total: m.cx_total,
                balanced_pct: m.balanced_pct() as f32,
                canonical_replace_pct: m.canonical_replace_pct() as f32,
                cx_replace_pct: m.cx_replace_pct() as f32,
            }
        }

        fn metrics_to_summary(name: &str, t: f32, m: &EvalMetrics) -> ModelSummary {
            ModelSummary {
                name: name.to_string(),
                best_threshold: t,
                canonical_correct: m.canonical_correct,
                canonical_total: m.canonical_total,
                cx_correct: m.cx_correct,
                cx_total: m.cx_total,
                balanced_pct: m.balanced_pct() as f32,
                canonical_replace_pct: m.canonical_replace_pct() as f32,
                cx_replace_pct: m.cx_replace_pct() as f32,
            }
        }

        fn collect_sweep(
            scored: &[ScoredCase],
            thresholds: &[f32],
            reachable_only: bool,
        ) -> Vec<ThresholdRow> {
            thresholds
                .iter()
                .map(|&t| {
                    let m = eval_at_threshold(scored, t, reachable_only);
                    metrics_to_row(t, &m)
                })
                .collect()
        }

        fn make_prob_dist(label: &str, vals: &[f32]) -> ProbDistribution {
            let p = |pct: f64| -> f32 {
                if vals.is_empty() {
                    return 0.0;
                }
                let idx = ((vals.len() as f64 - 1.0) * pct).round() as usize;
                vals[idx.min(vals.len() - 1)]
            };
            ProbDistribution {
                label: label.to_string(),
                n: vals.len() as u32,
                min: p(0.0),
                p25: p(0.25),
                p50: p(0.5),
                p75: p(0.75),
                max: p(1.0),
            }
        }

        fn print_sweep(
            label: &str,
            scored: &[ScoredCase],
            thresholds: &[f32],
            reachable_only: bool,
        ) {
            println!("\n  [{label}] threshold sweep:");
            for &t in thresholds {
                let m = eval_at_threshold(scored, t, reachable_only);
                println!(
                    "    T={t:.1}  can {}/{} ({:.1}%)  cx {}/{} ({:.1}%)  bal {:.1}%  repl: can {:.1}% cx {:.1}%",
                    m.canonical_correct,
                    m.canonical_total,
                    m.canonical_pct(),
                    m.cx_correct,
                    m.cx_total,
                    m.cx_pct(),
                    m.balanced_pct(),
                    m.canonical_replace_pct(),
                    m.cx_replace_pct(),
                );
            }
        }

        fn best_threshold(
            scored: &[ScoredCase],
            thresholds: &[f32],
            reachable_only: bool,
        ) -> (f32, EvalMetrics) {
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
        let deterministic_sweep;
        {
            println!("\n  [deterministic] acceptance_score threshold sweep:");
            let det_thresholds: &[f32] = &[0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
            deterministic_sweep = det_thresholds.iter().map(|&t| {
                let m = eval_deterministic_kfold(&probed_cases, &case_folds, folds, t);
                println!(
                    "    T={t:.1}  can {}/{} ({:.1}%)  cx {}/{} ({:.1}%)  bal {:.1}%  repl: can {:.1}% cx {:.1}%",
                    m.canonical_correct, m.canonical_total, m.canonical_pct(),
                    m.cx_correct, m.cx_total, m.cx_pct(),
                    m.balanced_pct(),
                    m.canonical_replace_pct(), m.cx_replace_pct(),
                );
                metrics_to_row(t, &m)
            }).collect::<Vec<_>>();
        }

        // 1b. Seed-only baseline — train once, sweep thresholds
        let scored_seed = train_and_score_kfold(
            &probed_cases,
            &case_folds,
            folds,
            TrainMode::None,
            beeml::judge::FeatureSlice::All,
        );
        print_sweep("seed_only", &scored_seed, thresholds, false);
        let seed_only_sweep = collect_sweep(&scored_seed, thresholds, false);

        // 1c. Taught (current teach_choice replay)
        let scored_taught = train_and_score_kfold(
            &probed_cases,
            &case_folds,
            folds,
            TrainMode::TeachChoice {
                epochs: train_epochs,
            },
            beeml::judge::FeatureSlice::All,
        );
        print_sweep("taught", &scored_taught, thresholds, false);
        let taught_sweep = collect_sweep(&scored_taught, thresholds, false);

        // ── Eval 2+3: Case-balanced FTRL + threshold sweep ─────────────
        println!("\n--- Eval 2: Case-balanced FTRL ---");
        let scored_balanced = train_and_score_kfold(
            &probed_cases,
            &case_folds,
            folds,
            TrainMode::CaseBalanced {
                epochs: train_epochs,
                hard_neg_cap: 3,
            },
            beeml::judge::FeatureSlice::All,
        );
        print_sweep("case_balanced", &scored_balanced, thresholds, false);
        let case_balanced_sweep = collect_sweep(&scored_balanced, thresholds, false);

        // ── Eval 4: Feature ablation (on case-balanced) ────────────────
        println!("\n--- Eval 4: Feature ablation (case-balanced, best threshold) ---");
        let ablation_results;
        {
            use beeml::judge::FeatureSlice;
            let slices = [
                FeatureSlice::PhoneticOnly,
                FeatureSlice::PlusAsr,
                FeatureSlice::PlusContext,
                FeatureSlice::All,
            ];
            ablation_results = slices.iter().map(|slice| {
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
                metrics_to_summary(slice.name(), bt, &m)
            }).collect::<Vec<_>>();
        }

        // ── Eval 5: Reachable-only ─────────────────────────────────────
        println!("\n--- Eval 5: Reachable-only (case-balanced) ---");
        // Reuse scored_balanced, just filter by reachable
        print_sweep("reachable_only", &scored_balanced, thresholds, true);

        // ── Eval 6: Formulation comparison ─────────────────────────────
        println!("\n--- Eval 6: Formulation comparison (best threshold each) ---");
        let formulation_results;
        {
            let formulations: &[(&str, TrainMode)] = &[
                (
                    "independent_binary",
                    TrainMode::TeachChoice {
                        epochs: train_epochs,
                    },
                ),
                (
                    "case_balanced",
                    TrainMode::CaseBalanced {
                        epochs: train_epochs,
                        hard_neg_cap: 3,
                    },
                ),
                (
                    "freeze_dense",
                    TrainMode::FreezeDense {
                        epochs: train_epochs,
                        hard_neg_cap: 3,
                    },
                ),
                (
                    "casewise_softmax",
                    TrainMode::CasewiseSoftmax {
                        epochs: train_epochs,
                    },
                ),
            ];
            formulation_results = formulations.iter().map(|(name, train_mode)| {
                let scored = train_and_score_kfold(&probed_cases, &case_folds, folds, train_mode.clone(), beeml::judge::FeatureSlice::All);
                let (bt, m) = best_threshold(&scored, thresholds, false);
                println!(
                    "  {name:<24} T={bt:.1}  can {}/{} ({:.1}%)  cx {}/{} ({:.1}%)  bal {:.1}%  repl: can {:.1}% cx {:.1}%",
                    m.canonical_correct, m.canonical_total, m.canonical_pct(),
                    m.cx_correct, m.cx_total, m.cx_pct(),
                    m.balanced_pct(),
                    m.canonical_replace_pct(), m.cx_replace_pct(),
                );
                metrics_to_summary(name, bt, &m)
            }).collect::<Vec<_>>();
        }

        // ── Eval 8: Two-stage (span gate + candidate ranker) ──────────
        println!("\n--- Eval 8: Two-stage (span gate + candidate ranker) ---");
        let two_stage_result;
        let two_stage_scored =
            train_and_score_twostage_kfold(&probed_cases, &case_folds, folds, train_epochs, 3);
        let (mut best_gt, mut best_rt);
        {
            // Stage A alone: gate accuracy at various thresholds
            println!("\n  Stage A (gate) alone:");
            let mut gate_sweep = Vec::new();
            for &gt in thresholds {
                let mut gate_pos_total = 0u32;
                let mut gate_pos_correct = 0u32;
                let mut gate_neg_total = 0u32;
                let mut gate_neg_correct = 0u32;
                for sc in &two_stage_scored {
                    if !sc.reachable {
                        continue;
                    }
                    let gate_open = sc.gate_prob >= gt;
                    if sc.should_abstain {
                        gate_neg_total += 1;
                        if !gate_open {
                            gate_neg_correct += 1;
                        }
                    } else {
                        gate_pos_total += 1;
                        if gate_open {
                            gate_pos_correct += 1;
                        }
                    }
                }
                if gate_pos_total == 0 && gate_neg_total == 0 {
                    continue;
                }
                let pos_pct = if gate_pos_total > 0 {
                    gate_pos_correct as f64 / gate_pos_total as f64 * 100.0
                } else {
                    0.0
                };
                let neg_pct = if gate_neg_total > 0 {
                    gate_neg_correct as f64 / gate_neg_total as f64 * 100.0
                } else {
                    0.0
                };
                let bal = (pos_pct + neg_pct) / 2.0;
                println!(
                    "    GT={gt:.1}  open_correct {gate_pos_correct}/{gate_pos_total} ({pos_pct:.1}%)  closed_correct {gate_neg_correct}/{gate_neg_total} ({neg_pct:.1}%)  bal {bal:.1}%"
                );
                gate_sweep.push(ThresholdRow {
                    threshold: gt,
                    canonical_correct: gate_pos_correct,
                    canonical_total: gate_pos_total,
                    cx_correct: gate_neg_correct,
                    cx_total: gate_neg_total,
                    balanced_pct: bal as f32,
                    canonical_replace_pct: pos_pct as f32,
                    cx_replace_pct: (100.0 - neg_pct) as f32,
                });
            }

            // Stage B alone: ranker top-1 accuracy on gold-present spans
            println!("\n  Stage B (ranker) alone (gold-verified spans):");
            let mut ranker_correct = 0u32;
            let mut ranker_total = 0u32;
            {
                for sc in &two_stage_scored {
                    if sc.should_abstain || !sc.reachable {
                        continue;
                    }
                    ranker_total += 1;
                    let tm = sc
                        .gold_term
                        .as_deref()
                        .zip(sc.ranker_best_term.as_deref())
                        .is_some_and(|(g, r)| g.eq_ignore_ascii_case(r));
                    if tm {
                        ranker_correct += 1;
                    }
                }
                println!(
                    "    Top-1 accuracy: {ranker_correct}/{ranker_total} ({:.1}%)",
                    if ranker_total > 0 {
                        ranker_correct as f64 / ranker_total as f64 * 100.0
                    } else {
                        0.0
                    }
                );
            }

            // Composed: sweep gate_threshold × ranker_threshold
            println!("\n  Composed (gate × ranker) threshold sweep:");
            let gate_thresholds: &[f32] = &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
            let ranker_thresholds: &[f32] = &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
            let mut best_bal = 0.0f64;
            best_gt = 0.0f32;
            best_rt = 0.0f32;
            let mut best_m = EvalMetrics::default();
            for &gt in gate_thresholds {
                for &rt in ranker_thresholds {
                    let m = eval_twostage_at_thresholds(&two_stage_scored, gt, rt);
                    let bal = m.balanced_pct();
                    if bal > best_bal
                        || (bal == best_bal
                            && m.canonical_replace_pct() > best_m.canonical_replace_pct())
                    {
                        best_bal = bal;
                        best_gt = gt;
                        best_rt = rt;
                        best_m = m.clone();
                    }
                }
            }
            println!(
                "    Best: GT={best_gt:.1} RT={best_rt:.1}  can {}/{} ({:.1}%)  cx {}/{} ({:.1}%)  bal {:.1}%  repl: can {:.1}% cx {:.1}%",
                best_m.canonical_correct,
                best_m.canonical_total,
                best_m.canonical_pct(),
                best_m.cx_correct,
                best_m.cx_total,
                best_m.cx_pct(),
                best_m.balanced_pct(),
                best_m.canonical_replace_pct(),
                best_m.cx_replace_pct(),
            );

            fn metrics_to_grid(gt: f32, rt: f32, m: &EvalMetrics) -> TwoStageGridPoint {
                TwoStageGridPoint {
                    gate_threshold: gt,
                    ranker_threshold: rt,
                    canonical_correct: m.canonical_correct,
                    canonical_total: m.canonical_total,
                    cx_correct: m.cx_correct,
                    cx_total: m.cx_total,
                    balanced_pct: m.balanced_pct() as f32,
                    canonical_replace_pct: m.canonical_replace_pct() as f32,
                    cx_replace_pct: m.cx_replace_pct() as f32,
                }
            }

            let best_grid = metrics_to_grid(best_gt, best_rt, &best_m);

            // Show a few interesting grid points
            println!("\n    Selected grid points:");
            let mut grid_points = Vec::new();
            for &gt in &[0.2, 0.3, 0.4, 0.5] {
                for &rt in &[0.2, 0.3, 0.4, 0.5] {
                    let m = eval_twostage_at_thresholds(&two_stage_scored, gt, rt);
                    println!(
                        "      GT={gt:.1} RT={rt:.1}  can {}/{} ({:.1}%)  cx {}/{} ({:.1}%)  bal {:.1}%  repl: can {:.1}% cx {:.1}%",
                        m.canonical_correct,
                        m.canonical_total,
                        m.canonical_pct(),
                        m.cx_correct,
                        m.cx_total,
                        m.cx_pct(),
                        m.balanced_pct(),
                        m.canonical_replace_pct(),
                        m.cx_replace_pct(),
                    );
                    grid_points.push(metrics_to_grid(gt, rt, &m));
                }
            }

            // Gate probability distributions
            println!("\n  Gate probability distributions:");
            let mut gate_can_probs: Vec<f32> = two_stage_scored
                .iter()
                .filter(|sc| !sc.should_abstain && sc.reachable)
                .map(|sc| sc.gate_prob)
                .collect();
            let mut gate_cx_probs: Vec<f32> = two_stage_scored
                .iter()
                .filter(|sc| sc.should_abstain && sc.reachable)
                .map(|sc| sc.gate_prob)
                .collect();
            gate_can_probs.sort_by(|a, b| a.total_cmp(b));
            gate_cx_probs.sort_by(|a, b| a.total_cmp(b));

            fn pctl(vals: &[f32]) -> String {
                if vals.is_empty() {
                    return "N/A".to_string();
                }
                let p = |pct: f64| -> f32 {
                    let idx = ((vals.len() as f64 - 1.0) * pct).round() as usize;
                    vals[idx.min(vals.len() - 1)]
                };
                format!(
                    "n={:<4} min={:.3} p25={:.3} p50={:.3} p75={:.3} max={:.3}",
                    vals.len(),
                    p(0.0),
                    p(0.25),
                    p(0.5),
                    p(0.75),
                    p(1.0)
                )
            }
            println!("    Canonical (should open):  {}", pctl(&gate_can_probs));
            println!("    Counterex (should close): {}", pctl(&gate_cx_probs));

            // Ranker probability distributions
            println!("\n  Ranker probability distributions:");
            let mut ranker_gold_probs: Vec<f32> = Vec::new();
            let mut ranker_nongold_probs: Vec<f32> = Vec::new();
            for sc in &two_stage_scored {
                if sc.should_abstain || !sc.reachable {
                    continue;
                }
                if let Some((_alias_id, prob)) = sc.ranker_best {
                    let tm = sc
                        .gold_term
                        .as_deref()
                        .zip(sc.ranker_best_term.as_deref())
                        .is_some_and(|(g, r)| g.eq_ignore_ascii_case(r));
                    if tm {
                        ranker_gold_probs.push(prob);
                    } else {
                        ranker_nongold_probs.push(prob);
                    }
                }
            }
            ranker_gold_probs.sort_by(|a, b| a.total_cmp(b));
            ranker_nongold_probs.sort_by(|a, b| a.total_cmp(b));
            println!(
                "    Gold=best (correct rank):    {}",
                pctl(&ranker_gold_probs)
            );
            println!(
                "    Gold!=best (wrong rank):     {}",
                pctl(&ranker_nongold_probs)
            );

            two_stage_result = TwoStageResult {
                gate_sweep,
                ranker_top1_correct: ranker_correct,
                ranker_top1_total: ranker_total,
                best: best_grid,
                grid_points,
                gate_canonical_dist: make_prob_dist("canonical", &gate_can_probs),
                gate_cx_dist: make_prob_dist("counterexample", &gate_cx_probs),
                ranker_gold_best_dist: make_prob_dist("gold=best", &ranker_gold_probs),
                ranker_gold_not_best_dist: make_prob_dist("gold!=best", &ranker_nongold_probs),
            };
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
                if vals.is_empty() {
                    return "N/A".to_string();
                }
                let p = |pct: f64| -> f32 {
                    let idx = ((vals.len() as f64 - 1.0) * pct).round() as usize;
                    vals[idx.min(vals.len() - 1)]
                };
                format!(
                    "n={:<4} min={:.3} p25={:.3} p50={:.3} p75={:.3} max={:.3}",
                    vals.len(),
                    p(0.0),
                    p(0.25),
                    p(0.5),
                    p(0.75),
                    p(1.0)
                )
            }

            println!(
                "  Gold candidate prob (canonical, gold=best):  {}",
                percentiles(&gold_probs)
            );
            println!(
                "  Top negative prob (counterexample):           {}",
                percentiles(&cx_top_probs)
            );

            // Also show where gold candidate is NOT the best
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
                println!(
                    "  Best-non-gold prob (canonical, gold!=best):   {}",
                    percentiles(&gold_not_best_best)
                );
                println!(
                    "  Cases where gold is NOT best candidate: {}",
                    gold_not_best_best.len()
                );
            }
        }

        // ── One-case training trace ────────────────────────────────────
        println!("\n--- One-case training trace ---");
        {
            // Pick first canonical case with gold span and first cx case with candidates
            let first_canonical = probed_cases.iter().find(|pc| {
                !pc.case.should_abstain && pc.spans.iter().any(|ps| ps.gold_alias_id.is_some())
            });
            let first_cx = probed_cases.iter().find(|pc| {
                pc.case.should_abstain && pc.spans.iter().any(|ps| !ps.candidates.is_empty())
            });

            if let (Some(can_case), Some(cx_case)) = (first_canonical, first_cx) {
                for (mode_name, mode) in &[
                    ("teach_choice", TrainMode::TeachChoice { epochs: 1 }),
                    (
                        "case_balanced",
                        TrainMode::CaseBalanced {
                            epochs: 1,
                            hard_neg_cap: 3,
                        },
                    ),
                ] {
                    println!("\n  [{mode_name}] training trace:");
                    let mut judge = OnlineJudge::new_quiet();

                    // Score before training
                    let can_span = gold_span(can_case).unwrap();
                    let cx_span = best_cx_span(cx_case).unwrap();

                    let pre_can =
                        judge.score_candidates(&can_span.span, &can_span.candidates, &can_span.ctx);
                    let gold_pre = pre_can
                        .iter()
                        .find(|o| o.alias_id == can_span.gold_alias_id)
                        .map(|o| o.probability)
                        .unwrap_or(0.0);
                    let best_pre = pre_can
                        .iter()
                        .filter(|o| !o.is_keep_original)
                        .max_by(|a, b| a.probability.total_cmp(&b.probability))
                        .map(|o| o.probability)
                        .unwrap_or(0.0);

                    let pre_cx =
                        judge.score_candidates(&cx_span.span, &cx_span.candidates, &cx_span.ctx);
                    let cx_best_pre = pre_cx
                        .iter()
                        .filter(|o| !o.is_keep_original)
                        .max_by(|a, b| a.probability.total_cmp(&b.probability))
                        .map(|o| o.probability)
                        .unwrap_or(0.0);

                    println!("    Before training:");
                    println!(
                        "      canonical gold prob={gold_pre:.4}, best prob={best_pre:.4} (term={})",
                        can_case.case.target_term
                    );
                    println!(
                        "      counterex best prob={cx_best_pre:.4} (term={})",
                        cx_case.case.target_term
                    );

                    // Train on canonical case
                    train_case(&mut judge, can_case, mode);
                    let post_can1 =
                        judge.score_candidates(&can_span.span, &can_span.candidates, &can_span.ctx);
                    let gold_post1 = post_can1
                        .iter()
                        .find(|o| o.alias_id == can_span.gold_alias_id)
                        .map(|o| o.probability)
                        .unwrap_or(0.0);
                    let post_cx1 =
                        judge.score_candidates(&cx_span.span, &cx_span.candidates, &cx_span.ctx);
                    let cx_best_post1 = post_cx1
                        .iter()
                        .filter(|o| !o.is_keep_original)
                        .max_by(|a, b| a.probability.total_cmp(&b.probability))
                        .map(|o| o.probability)
                        .unwrap_or(0.0);

                    println!("    After training on 1 canonical:");
                    println!(
                        "      canonical gold prob={gold_post1:.4} (delta={:+.4})",
                        gold_post1 - gold_pre
                    );
                    println!(
                        "      counterex best prob={cx_best_post1:.4} (delta={:+.4})",
                        cx_best_post1 - cx_best_pre
                    );

                    // Train on counterexample case
                    train_case(&mut judge, cx_case, mode);
                    let post_can2 =
                        judge.score_candidates(&can_span.span, &can_span.candidates, &can_span.ctx);
                    let gold_post2 = post_can2
                        .iter()
                        .find(|o| o.alias_id == can_span.gold_alias_id)
                        .map(|o| o.probability)
                        .unwrap_or(0.0);
                    let post_cx2 =
                        judge.score_candidates(&cx_span.span, &cx_span.candidates, &cx_span.ctx);
                    let cx_best_post2 = post_cx2
                        .iter()
                        .filter(|o| !o.is_keep_original)
                        .max_by(|a, b| a.probability.total_cmp(&b.probability))
                        .map(|o| o.probability)
                        .unwrap_or(0.0);

                    println!("    After training on 1 counterexample:");
                    println!(
                        "      canonical gold prob={gold_post2:.4} (delta={:+.4} from canonical-only)",
                        gold_post2 - gold_post1
                    );
                    println!(
                        "      counterex best prob={cx_best_post2:.4} (delta={:+.4} from canonical-only)",
                        cx_best_post2 - cx_best_post1
                    );

                    // Weight norm
                    let weights = judge.weights();
                    let weight_norm: f64 = weights
                        .iter()
                        .map(|w| (*w as f64) * (*w as f64))
                        .sum::<f64>()
                        .sqrt();
                    println!("    Weight L2 norm: {weight_norm:.4}");
                    println!("    Active features: {}", judge.model().num_active());
                }
            }
        }

        // ── Three scoreboards (006) ─────────────────────────────────────
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║                    THREE SCOREBOARDS                        ║");
        println!("╚══════════════════════════════════════════════════════════════╝");
        {
            // Scoreboard 1: End-to-end (denominator = ALL cases)
            let mut e2e_can_correct = 0u32;
            let mut e2e_can_total = 0u32;
            let mut e2e_cx_abstained = 0u32;
            let mut e2e_cx_total = 0u32;
            let mut e2e_cx_false_pos = 0u32;
            for (i, pc) in probed_cases.iter().enumerate() {
                let sc = &two_stage_scored[i];
                if pc.case.should_abstain {
                    e2e_cx_total += 1;
                    let gate_open = sc.gate_prob >= best_gt;
                    let ranker_fires = sc.ranker_best.map_or(false, |(_, p)| p >= best_rt);
                    if gate_open && ranker_fires {
                        e2e_cx_false_pos += 1;
                    } else {
                        e2e_cx_abstained += 1;
                    }
                } else {
                    e2e_can_total += 1;
                    let gate_open = sc.gate_prob >= best_gt;
                    let ranker_fires = sc.ranker_best.map_or(false, |(_, p)| p >= best_rt);
                    let ranker_correct_id = sc
                        .gold_term
                        .as_deref()
                        .zip(sc.ranker_best_term.as_deref())
                        .is_some_and(|(g, r)| g.eq_ignore_ascii_case(r));
                    if gate_open && ranker_fires && ranker_correct_id {
                        e2e_can_correct += 1;
                    }
                }
            }
            let e2e_can_pct = if e2e_can_total > 0 {
                e2e_can_correct as f64 / e2e_can_total as f64 * 100.0
            } else {
                0.0
            };
            let e2e_cx_pct = if e2e_cx_total > 0 {
                e2e_cx_abstained as f64 / e2e_cx_total as f64 * 100.0
            } else {
                0.0
            };
            let e2e_fp_pct = if e2e_cx_total > 0 {
                e2e_cx_false_pos as f64 / e2e_cx_total as f64 * 100.0
            } else {
                0.0
            };

            println!(
                "\n┌─ 1. End-to-end (all cases, GT={best_gt:.1} RT={best_rt:.1}) ──────────────"
            );
            println!(
                "│  Canonical corrected:    {e2e_can_correct:>3}/{e2e_can_total:<3}  ({e2e_can_pct:.1}%)"
            );
            println!(
                "│  Counterex abstained:    {e2e_cx_abstained:>3}/{e2e_cx_total:<3}  ({e2e_cx_pct:.1}%)"
            );
            println!(
                "│  False positive rate:    {e2e_cx_false_pos:>3}/{e2e_cx_total:<3}  ({e2e_fp_pct:.1}%)"
            );
            println!("└──────────────────────────────────────────────────");

            // Scoreboard 2: Judge-stage (reachable only)
            // Gate accuracy at best_gt (reachable cases)
            let mut gate_pos_correct = 0u32;
            let mut gate_pos_total = 0u32;
            let mut gate_neg_correct = 0u32;
            let mut gate_neg_total = 0u32;
            for sc in &two_stage_scored {
                if !sc.reachable {
                    continue;
                }
                let gate_open = sc.gate_prob >= best_gt;
                if sc.should_abstain {
                    gate_neg_total += 1;
                    if !gate_open {
                        gate_neg_correct += 1;
                    }
                } else {
                    gate_pos_total += 1;
                    if gate_open {
                        gate_pos_correct += 1;
                    }
                }
            }
            let gate_pos_pct = if gate_pos_total > 0 {
                gate_pos_correct as f64 / gate_pos_total as f64 * 100.0
            } else {
                0.0
            };
            let gate_neg_pct = if gate_neg_total > 0 {
                gate_neg_correct as f64 / gate_neg_total as f64 * 100.0
            } else {
                0.0
            };
            let gate_bal = (gate_pos_pct + gate_neg_pct) / 2.0;

            // Ranker top-1 (reachable canonical)
            let mut rank_correct = 0u32;
            let mut rank_total = 0u32;
            for sc in &two_stage_scored {
                if sc.should_abstain || !sc.reachable {
                    continue;
                }
                rank_total += 1;
                let term_match = sc
                    .gold_term
                    .as_deref()
                    .zip(sc.ranker_best_term.as_deref())
                    .is_some_and(|(g, r)| g.eq_ignore_ascii_case(r));
                if term_match {
                    rank_correct += 1;
                }
            }
            let rank_pct = if rank_total > 0 {
                rank_correct as f64 / rank_total as f64 * 100.0
            } else {
                0.0
            };

            // Composed balanced (reachable)
            let judge_m = eval_twostage_at_thresholds(&two_stage_scored, best_gt, best_rt);
            let judge_bal = judge_m.balanced_pct();

            println!("\n┌─ 2. Judge-stage (reachable only) ─────────────────────────");
            println!(
                "│  Gate balanced accuracy: {gate_bal:.1}%  (open {gate_pos_correct}/{gate_pos_total} {gate_pos_pct:.1}%, close {gate_neg_correct}/{gate_neg_total} {gate_neg_pct:.1}%)"
            );
            println!("│  Ranker top-1 accuracy:  {rank_correct}/{rank_total}  ({rank_pct:.1}%)");
            println!(
                "│  Composed balanced:      {judge_bal:.1}%  (can {}/{}  cx {}/{})",
                judge_m.canonical_correct,
                judge_m.canonical_total,
                judge_m.cx_correct,
                judge_m.cx_total
            );
            println!("└──────────────────────────────────────────────────");

            // Scoreboard 3: Upstream opportunity set
            let retrieved_pct = if canonical_count > 0 {
                canonical_with_gold as f64 / canonical_count as f64 * 100.0
            } else {
                0.0
            };
            let verified_pct = if canonical_count > 0 {
                canonical_gold_verified as f64 / canonical_count as f64 * 100.0
            } else {
                0.0
            };
            let lost_retrieval = canonical_count - canonical_with_gold;
            let lost_verification = canonical_with_gold - canonical_gold_verified;

            println!("\n┌─ 3. Upstream opportunity set ──────────────────────────────");
            println!(
                "│  Gold retrieved:         {canonical_with_gold:>3}/{canonical_count:<3}  ({retrieved_pct:.1}%)  — {lost_retrieval} lost at retrieval"
            );
            println!(
                "│  Gold verified:          {canonical_gold_verified:>3}/{canonical_count:<3}  ({verified_pct:.1}%)  — {lost_verification} lost at verification"
            );
            println!("└──────────────────────────────────────────────────");
        }

        // ── Per-case failure report (005) ────────────────────────────────
        println!("\n=== Per-case failure report (at GT={best_gt:.1} RT={best_rt:.1}) ===");
        {
            let trunc = |s: &str, n: usize| -> String {
                if s.len() <= n {
                    s.to_string()
                } else {
                    format!("{}…", &s[..n])
                }
            };

            // Bucket 1: Not retrieved — canonical cases where gold term not in any span's shortlist
            let mut not_retrieved = Vec::new();
            // Bucket 2: Not verified — gold retrieved but no verified candidate
            let mut not_verified = Vec::new();
            // Bucket 3: Gate misses — reachable but gate_prob < best_gt
            let mut gate_misses = Vec::new();
            // Bucket 4: Ranker misses — gate opens but ranker top-1 is not gold
            let mut ranker_misses = Vec::new();

            for (i, pc) in probed_cases.iter().enumerate() {
                if pc.case.should_abstain {
                    continue; // only canonical cases
                }

                let sc = &two_stage_scored[i];

                // Check if gold term appears in any span's shortlist
                let gold_in_shortlist = pc.spans.iter().any(|ps| ps.gold_alias_id.is_some());
                if !gold_in_shortlist {
                    not_retrieved.push(format!(
                        "  [{:>3}] term={:<30} transcript={}",
                        pc.case.case_id,
                        pc.case.target_term,
                        trunc(&pc.case.transcript, 60),
                    ));
                    continue;
                }

                // Gold is in shortlist — check if verified
                let gold_verified = pc.spans.iter().any(|ps| {
                    ps.gold_alias_id.map_or(false, |gid| {
                        ps.candidates
                            .iter()
                            .any(|(c, _)| c.alias_id == gid && c.verified)
                    })
                });
                if !gold_verified {
                    // Find the gold candidate to show WHY it failed verification
                    let gold_c = pc
                        .spans
                        .iter()
                        .filter_map(|ps| {
                            ps.gold_alias_id.and_then(|gid| {
                                ps.candidates
                                    .iter()
                                    .find(|(c, _)| c.alias_id == gid)
                                    .map(|(c, _)| c)
                            })
                        })
                        .next();
                    let diag = if let Some(c) = gold_c {
                        let mut reasons = Vec::new();
                        if c.short_guard_applied && !c.short_guard_passed {
                            reasons.push(format!(
                                "SHORT_GUARD(onset={}, feat={:.2}, tok={:.2})",
                                c.short_guard_onset_match, c.feature_score, c.token_score
                            ));
                        }
                        if c.low_content_guard_applied && !c.low_content_guard_passed {
                            reasons.push(format!(
                                "LOW_CONTENT(tok={:.2}, feat={:.2})",
                                c.token_score, c.feature_score
                            ));
                        }
                        if !c.acceptance_floor_passed {
                            reasons.push(format!(
                                "ACCEPT_FLOOR(phon={:.2}, accept={:.2}, coarse={:.2})",
                                c.phonetic_score, c.acceptance_score, c.coarse_score
                            ));
                        }
                        if reasons.is_empty() {
                            format!(
                                "alias={:<20} phon={:.2} accept={:.2} (unknown reason)",
                                c.alias_text, c.phonetic_score, c.acceptance_score
                            )
                        } else {
                            format!("alias={:<20} {}", c.alias_text, reasons.join(" + "))
                        }
                    } else {
                        "gold candidate not found in shortlist".to_string()
                    };
                    not_verified.push(format!(
                        "  [{:>3}] term={:<20} {}\n        transcript={}",
                        pc.case.case_id,
                        pc.case.target_term,
                        diag,
                        trunc(&pc.case.transcript, 60),
                    ));
                    continue;
                }

                // Gold is verified (reachable). Check gate.
                if sc.gate_prob < best_gt {
                    gate_misses.push(format!(
                        "  [{:>3}] term={:<30} gate_prob={:.3}  transcript={}",
                        pc.case.case_id,
                        pc.case.target_term,
                        sc.gate_prob,
                        trunc(&pc.case.transcript, 50),
                    ));
                    continue;
                }

                // Gate opens. Check ranker.
                if let Some((_alias_id, ranker_prob)) = sc.ranker_best {
                    let term_match = sc
                        .gold_term
                        .as_deref()
                        .zip(sc.ranker_best_term.as_deref())
                        .is_some_and(|(g, r)| g.eq_ignore_ascii_case(r));
                    if !term_match {
                        let picked_name = sc.ranker_best_term.as_deref().unwrap_or("?");
                        ranker_misses.push(format!(
                            "  [{:>3}] term={:<30} ranker_picked={:<30} ranker_prob={:.3}  transcript={}",
                            pc.case.case_id,
                            pc.case.target_term,
                            picked_name,
                            ranker_prob,
                            trunc(&pc.case.transcript, 40),
                        ));
                    }
                }
            }

            println!("\n1. Not retrieved ({} cases):", not_retrieved.len());
            for line in &not_retrieved {
                println!("{line}");
            }

            println!("\n2. Not verified ({} cases):", not_verified.len());
            for line in &not_verified {
                println!("{line}");
            }

            println!("\n3. Gate misses ({} cases):", gate_misses.len());
            for line in &gate_misses {
                println!("{line}");
            }

            println!("\n4. Ranker misses ({} cases):", ranker_misses.len());
            for line in &ranker_misses {
                println!("{line}");
            }

            println!();
        }

        Ok(OfflineJudgeEvalResult {
            canonical_cases: canonical_count as u32,
            gold_retrieved: canonical_with_gold as u32,
            gold_verified: canonical_gold_verified as u32,
            gold_reachable: two_stage_result.best.canonical_total,
            counterexample_cases: cx_count as u32,
            deterministic_sweep,
            seed_only_sweep,
            taught_sweep,
            case_balanced_sweep,
            ablation_results,
            formulation_results,
            two_stage: two_stage_result,
            canonical_correct: 0,
            canonical_total: 0,
            counterexample_correct: 0,
            counterexample_total: 0,
            fold_results: vec![],
        })
    }

    async fn run_phonetic_comparison(
        &self,
        request: PhoneticComparisonRequest,
    ) -> Result<PhoneticComparisonResult, String> {
        self.run_phonetic_comparison(request)
    }

    async fn get_corpus_capture_plan(&self) -> Result<CorpusCapturePlanResult, String> {
        let prompts = self.corpus_capture_prompts();
        let prompt_ids = prompts
            .iter()
            .map(|prompt| prompt.prompt_id.clone())
            .collect::<std::collections::HashSet<_>>();
        let recordings = load_corpus_recordings(&self.inner.corpus_dir)
            .map_err(|e| e.to_string())?
            .into_iter()
            .filter(|row| prompt_ids.contains(&row.prompt_id))
            .collect();
        Ok(CorpusCapturePlanResult {
            corpus_dir: self.inner.corpus_dir.display().to_string(),
            prompts,
            recordings,
        })
    }

    async fn save_corpus_recording(
        &self,
        request: SaveCorpusRecordingRequest,
    ) -> Result<SaveCorpusRecordingResult, String> {
        let recording =
            save_corpus_recording(&self.inner.corpus_dir, &request).map_err(|e| e.to_string())?;
        let total_recordings = load_corpus_recordings(&self.inner.corpus_dir)
            .map_err(|e| e.to_string())?
            .len()
            .min(u32::MAX as usize) as u32;
        Ok(SaveCorpusRecordingResult {
            recording,
            total_recordings,
        })
    }

    async fn delete_corpus_recording(
        &self,
        request: DeleteCorpusRecordingRequest,
    ) -> Result<DeleteCorpusRecordingResult, String> {
        let deleted =
            delete_corpus_recording(&self.inner.corpus_dir, &request).map_err(|e| e.to_string())?;
        let total_recordings = load_corpus_recordings(&self.inner.corpus_dir)
            .map_err(|e| e.to_string())?
            .len()
            .min(u32::MAX as usize) as u32;
        Ok(DeleteCorpusRecordingResult {
            deleted,
            total_recordings,
        })
    }
}
