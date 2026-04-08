/// Per-span data extracted from a probe, ready for judge training/evaluation.
/// Does not include judge scores — those are computed separately per fold.
struct ProbedSpan {
    span: TranscriptSpan,
    candidates: Vec<(
        bee_phonetic::CandidateFeatureRow,
        bee_phonetic::IdentifierFlags,
    )>,
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
        if self.canonical_total == 0 {
            0.0
        } else {
            self.canonical_correct as f64 / self.canonical_total as f64 * 100.0
        }
    }
    fn cx_pct(&self) -> f64 {
        if self.cx_total == 0 {
            0.0
        } else {
            self.cx_correct as f64 / self.cx_total as f64 * 100.0
        }
    }
    fn balanced_pct(&self) -> f64 {
        (self.canonical_pct() + self.cx_pct()) / 2.0
    }
    fn canonical_replace_pct(&self) -> f64 {
        if self.canonical_total == 0 {
            0.0
        } else {
            self.canonical_replaced as f64 / self.canonical_total as f64 * 100.0
        }
    }
    fn cx_replace_pct(&self) -> f64 {
        if self.cx_total == 0 {
            0.0
        } else {
            self.cx_replaced as f64 / self.cx_total as f64 * 100.0
        }
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
            if case_folds[i] != fold_k {
                continue;
            }

            if pc.case.should_abstain {
                fold.cx_total += 1;
                // Check if any verified candidate above threshold exists
                let any_replace = pc.spans.iter().any(|ps| {
                    ps.candidates
                        .iter()
                        .any(|(c, _)| c.verified && c.acceptance_score >= acceptance_threshold)
                });
                if any_replace {
                    fold.cx_replaced += 1;
                } else {
                    fold.cx_correct += 1;
                }
            } else {
                // Find span with gold alias
                let gold_span = find_best_gold_span(&pc.spans);
                if let Some(gs) = gold_span {
                    fold.canonical_total += 1;
                    // Rank all verified candidates by acceptance_score > phonetic_score > coarse_score
                    let mut ranked: Vec<_> = gs
                        .candidates
                        .iter()
                        .filter(|(c, _)| c.verified && c.acceptance_score >= acceptance_threshold)
                        .collect();
                    ranked.sort_by(|(a, _), (b, _)| {
                        b.acceptance_score
                            .total_cmp(&a.acceptance_score)
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
                if case_folds[i] == fold_k {
                    continue;
                }
                if use_ablation {
                    train_case_ablated(&mut judge, pc, &train_mode, slice);
                } else {
                    train_case(&mut judge, pc, &train_mode);
                }
            }
        }

        // ── Score test cases ──
        for (i, pc) in probed_cases.iter().enumerate() {
            if case_folds[i] != fold_k {
                continue;
            }

            if pc.case.should_abstain {
                let has_candidates = pc.spans.iter().any(|ps| !ps.candidates.is_empty());
                // Find max probability of any candidate across all spans
                let max_prob = pc
                    .spans
                    .iter()
                    .filter(|ps| !ps.candidates.is_empty())
                    .filter_map(|ps| {
                        if use_ablation {
                            score_span_ablated(&judge, ps, slice)
                                .into_iter()
                                .max_by(|a, b| a.1.total_cmp(&b.1))
                        } else {
                            let options = judge.score_candidates(&ps.span, &ps.candidates, &ps.ctx);
                            options
                                .iter()
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
                let gold_span = find_best_gold_span(&pc.spans);
                let (best_candidate, reachable) = if let Some(gs) = gold_span {
                    // Reachable = gold candidate exists AND is verified
                    let gold_verified = gs
                        .gold_alias_id
                        .map(|gid| {
                            gs.candidates
                                .iter()
                                .any(|(c, _)| c.alias_id == gid && c.verified)
                        })
                        .unwrap_or(false);
                    let best = if use_ablation {
                        let s = score_span_ablated(&judge, gs, slice);
                        s.into_iter().max_by(|a, b| a.1.total_cmp(&b.1))
                    } else {
                        let options = judge.score_candidates(&gs.span, &gs.candidates, &gs.ctx);
                        options
                            .iter()
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
            if reachable_only && !sc.reachable {
                continue;
            }
            m.cx_total += 1;
            let replaced = sc
                .best_candidate
                .map(|(_, p)| p >= threshold)
                .unwrap_or(false);
            if replaced {
                m.cx_replaced += 1;
            } else {
                m.cx_correct += 1;
            }
        } else {
            if reachable_only && !sc.reachable {
                continue;
            }
            if !sc.reachable {
                continue;
            } // skip unreachable canonical regardless
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
                    judge.train_balanced(
                        &ps.span,
                        &ps.candidates,
                        Option::None,
                        &ps.ctx,
                        *hard_neg_cap,
                    );
                }
            } else if let Some(ps) = gold_span(pc) {
                judge.train_balanced(
                    &ps.span,
                    &ps.candidates,
                    ps.gold_alias_id,
                    &ps.ctx,
                    *hard_neg_cap,
                );
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
                    judge.train_balanced(
                        &ps.span,
                        &ps.candidates,
                        Option::None,
                        &ps.ctx,
                        *hard_neg_cap,
                    );
                }
            } else if let Some(ps) = gold_span(pc) {
                judge.train_balanced(
                    &ps.span,
                    &ps.candidates,
                    ps.gold_alias_id,
                    &ps.ctx,
                    *hard_neg_cap,
                );
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
    if examples.is_empty() {
        return;
    }

    // Filter features for ablation
    let filtered: Vec<_> = examples
        .iter()
        .map(|e| filter_features(&e.features, slice))
        .collect();

    match mode {
        TrainMode::None => {}
        TrainMode::TeachChoice { .. }
        | TrainMode::CaseBalanced { .. }
        | TrainMode::FreezeDense { .. } => {
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
                let hardest = (0..examples.len()).max_by(|&a, &b| {
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
                examples
                    .iter()
                    .position(|e| e.alias_id == gold_id)
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
    examples
        .iter()
        .map(|e| {
            let filtered = filter_features(&e.features, slice);
            let prob = judge.model().predict_prob(&filtered) as f32;
            (e.alias_id, prob)
        })
        .collect()
}

/// Get the best counterexample span (most candidates = most likely false positive).
fn best_cx_span(pc: &ProbedCase) -> Option<&ProbedSpan> {
    pc.spans
        .iter()
        .filter(|ps| !ps.candidates.is_empty())
        .max_by_key(|ps| ps.candidates.len())
}

/// Get the span containing the gold alias (prefers verified gold).
fn gold_span(pc: &ProbedCase) -> Option<&ProbedSpan> {
    find_best_gold_span(&pc.spans)
}

/// Find best gold span: prefer verified gold, fall back to any gold.
fn find_best_gold_span(spans: &[ProbedSpan]) -> Option<&ProbedSpan> {
    spans
        .iter()
        .find(|ps| {
            if let Some(gid) = ps.gold_alias_id {
                ps.candidates
                    .iter()
                    .any(|(c, _)| c.alias_id == gid && c.verified)
            } else {
                false
            }
        })
        .or_else(|| spans.iter().find(|ps| ps.gold_alias_id.is_some()))
}

/// Two-stage scored case: gate probability + ranker probability.
struct TwoStageScoredCase {
    should_abstain: bool,
    gold_alias_id: Option<u32>,
    /// Gate probability for the best span (canonical: gold span; cx: best span).
    gate_prob: f32,
    /// Best candidate (alias_id, ranker_probability) from the ranker.
    ranker_best: Option<(u32, f32)>,
    /// Whether gold was reachable (retrieved + verified).
    reachable: bool,
}

/// Train two-stage model (gate + ranker) with k-fold CV.
/// Returns pre-scored results for threshold sweeping.
fn train_and_score_twostage_kfold(
    probed_cases: &[ProbedCase],
    case_folds: &[usize],
    folds: usize,
    epochs: usize,
    hard_neg_cap: usize,
) -> Vec<TwoStageScoredCase> {
    use beeml::judge::{
        build_gate_features, build_ranker_features, seed_gate_model, seed_ranker_model,
    };

    let mut scored = Vec::with_capacity(probed_cases.len());
    scored.resize_with(probed_cases.len(), || TwoStageScoredCase {
        should_abstain: false,
        gold_alias_id: None,
        gate_prob: 0.0,
        ranker_best: None,
        reachable: false,
    });

    for fold_k in 0..folds {
        // Create separate models for gate and ranker
        let mut gate_model = beeml::sparse_ftrl::SparseFtrl::new(0.5, 1.0, 0.0001, 0.001);
        seed_gate_model(&mut gate_model);
        let mut ranker_model = beeml::sparse_ftrl::SparseFtrl::new(0.5, 1.0, 0.0001, 0.001);
        seed_ranker_model(&mut ranker_model);

        let memory: beeml::judge::TermMemory = Default::default();

        // ── Training phase ──
        for _epoch in 0..epochs {
            for (i, pc) in probed_cases.iter().enumerate() {
                if case_folds[i] == fold_k {
                    continue;
                }

                // Stage A: train gate on ALL spans
                if pc.case.should_abstain {
                    // Counterexample: all spans are negative
                    for ps in &pc.spans {
                        if ps.candidates.is_empty() {
                            continue;
                        }
                        let feats = build_gate_features(&ps.span, &ps.candidates, &ps.ctx, &memory);
                        gate_model.update(&feats, false);
                    }
                } else {
                    // Canonical: gold span = positive, skip non-gold spans (ambiguous)
                    // Prefer verified-gold span over any-gold span
                    if let Some(ps) = find_best_gold_span(&pc.spans) {
                        let feats = build_gate_features(&ps.span, &ps.candidates, &ps.ctx, &memory);
                        gate_model.update(&feats, true);
                    }
                }

                // Stage B: train ranker only on spans with verified gold
                if !pc.case.should_abstain {
                    if let Some(ps) = find_best_gold_span(&pc.spans) {
                        let gold_id = ps.gold_alias_id.unwrap();
                        let gold_verified = ps
                            .candidates
                            .iter()
                            .any(|(c, _)| c.alias_id == gold_id && c.verified);
                        if gold_verified {
                            let examples = build_ranker_features(&ps.span, &ps.candidates, &memory);
                            // Gold = positive
                            if let Some(gold_ex) = examples.iter().find(|e| e.alias_id == gold_id) {
                                ranker_model.update(&gold_ex.features, true);
                            }
                            // Hard negatives
                            let mut neg_indices: Vec<usize> = examples
                                .iter()
                                .enumerate()
                                .filter(|(_, e)| e.alias_id != gold_id)
                                .map(|(j, _)| j)
                                .collect();
                            neg_indices.sort_by(|&a, &b| {
                                let sa = ranker_model.predict(&examples[a].features);
                                let sb = ranker_model.predict(&examples[b].features);
                                sb.total_cmp(&sa)
                            });
                            for &idx in neg_indices.iter().take(hard_neg_cap) {
                                ranker_model.update(&examples[idx].features, false);
                            }
                        }
                    }
                }
            }
        }

        // ── Score test cases ──
        for (i, pc) in probed_cases.iter().enumerate() {
            if case_folds[i] != fold_k {
                continue;
            }

            if pc.case.should_abstain {
                let has_candidates = pc.spans.iter().any(|ps| !ps.candidates.is_empty());
                // Gate: max gate prob across all spans
                let best_gate = pc
                    .spans
                    .iter()
                    .filter(|ps| !ps.candidates.is_empty())
                    .map(|ps| {
                        let feats = build_gate_features(&ps.span, &ps.candidates, &ps.ctx, &memory);
                        let gp = gate_model.predict_prob(&feats) as f32;
                        // Ranker: best candidate in this span
                        let examples = build_ranker_features(&ps.span, &ps.candidates, &memory);
                        let best_cand = examples
                            .iter()
                            .map(|e| (e.alias_id, ranker_model.predict_prob(&e.features) as f32))
                            .max_by(|a, b| a.1.total_cmp(&b.1));
                        (gp, best_cand)
                    })
                    .max_by(|a, b| a.0.total_cmp(&b.0));

                let (gate_prob, ranker_best) =
                    best_gate.map(|(gp, rb)| (gp, rb)).unwrap_or((0.0, None));

                scored[i] = TwoStageScoredCase {
                    should_abstain: true,
                    gold_alias_id: None,
                    gate_prob,
                    ranker_best,
                    reachable: has_candidates,
                };
            } else {
                // Prefer the span where gold is verified; fall back to any span with gold
                let gold_span = pc
                    .spans
                    .iter()
                    .find(|ps| {
                        if let Some(gid) = ps.gold_alias_id {
                            ps.candidates
                                .iter()
                                .any(|(c, _)| c.alias_id == gid && c.verified)
                        } else {
                            false
                        }
                    })
                    .or_else(|| pc.spans.iter().find(|ps| ps.gold_alias_id.is_some()));
                if let Some(gs) = gold_span {
                    let gold_id = gs.gold_alias_id.unwrap();
                    let gold_verified = gs
                        .candidates
                        .iter()
                        .any(|(c, _)| c.alias_id == gold_id && c.verified);

                    let gate_feats =
                        build_gate_features(&gs.span, &gs.candidates, &gs.ctx, &memory);
                    let gate_prob = gate_model.predict_prob(&gate_feats) as f32;

                    let examples = build_ranker_features(&gs.span, &gs.candidates, &memory);
                    let ranker_best = examples
                        .iter()
                        .map(|e| (e.alias_id, ranker_model.predict_prob(&e.features) as f32))
                        .max_by(|a, b| a.1.total_cmp(&b.1));

                    scored[i] = TwoStageScoredCase {
                        should_abstain: false,
                        gold_alias_id: Some(gold_id),
                        gate_prob,
                        ranker_best,
                        reachable: gold_verified,
                    };
                } else {
                    scored[i] = TwoStageScoredCase {
                        should_abstain: false,
                        gold_alias_id: None,
                        gate_prob: 0.0,
                        ranker_best: None,
                        reachable: false,
                    };
                }
            }
        }
    }
    scored
}

/// Evaluate two-stage scored cases at given thresholds.
fn eval_twostage_at_thresholds(
    scored: &[TwoStageScoredCase],
    gate_threshold: f32,
    ranker_threshold: f32,
) -> EvalMetrics {
    let mut m = EvalMetrics::default();
    for sc in scored {
        if !sc.reachable {
            continue;
        }
        if sc.should_abstain {
            m.cx_total += 1;
            // Gate must open AND ranker must accept for a replacement to happen
            let gate_open = sc.gate_prob >= gate_threshold;
            let ranker_accept = sc
                .ranker_best
                .map(|(_, p)| p >= ranker_threshold)
                .unwrap_or(false);
            let replaced = gate_open && ranker_accept;
            if replaced {
                m.cx_replaced += 1;
            }
            if !replaced {
                m.cx_correct += 1;
            } // Correctly abstained
        } else {
            m.canonical_total += 1;
            let gate_open = sc.gate_prob >= gate_threshold;
            let ranker_accept = sc
                .ranker_best
                .map(|(_, p)| p >= ranker_threshold)
                .unwrap_or(false);
            let replaced = gate_open && ranker_accept;
            if replaced {
                m.canonical_replaced += 1;
                if let Some((alias_id, _)) = sc.ranker_best {
                    if sc.gold_alias_id == Some(alias_id) {
                        m.canonical_correct += 1;
                    }
                }
            }
        }
    }
    m
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
