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
