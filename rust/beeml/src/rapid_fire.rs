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
            let keep_prob = s
                .judge_options
                .iter()
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
            let is_gold = normalize_comparable_text(&hypothesis.sentence)
                == normalize_comparable_text(expected_source_text);
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
        top_choice_sentence = choices
            .first()
            .map(|choice| choice.sentence.as_str())
            .unwrap_or(""),
        top_choice_keep = choices
            .first()
            .map(|choice| choice.choose_keep_original)
            .unwrap_or(true),
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
