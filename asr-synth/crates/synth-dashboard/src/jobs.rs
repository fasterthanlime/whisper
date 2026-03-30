use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, OnceLock};

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use rand::prelude::*;
use serde::{Deserialize, Serialize};

use crate::tts;
use crate::{err, AppError, AppState};

use std::sync::atomic::Ordering;

#[derive(Clone)]
struct PreparedExample {
    prompt: String,
    completion: String,
    allow_repeat: bool,
}

#[derive(Clone)]
struct CounterexampleExample {
    sentence: String,
    weight: u32,
}

#[derive(Clone)]
struct AppliedMistakePoolRow {
    term: String,
    clean_sentence: String,
    corrupted_sentence: String,
    corruption_surface: String,
    source: String,
    weight: u32,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct AppliedMistakeEvalRow {
    term: String,
    clean_sentence: String,
    corrupted_sentence: String,
    corruption_surface: String,
    source: String,
}

#[derive(Default)]
struct AppliedMistakeBuildStats {
    train_mistakes: usize,
    eval_mistakes: usize,
    train_identity: usize,
    train_counterexamples: usize,
    skipped_missing_term: usize,
    skipped_acceptable_surface: usize,
    skipped_duplicate: usize,
}

struct AppliedMistakeBuildOutput {
    train_mistakes: Vec<AppliedMistakePoolRow>,
    eval_mistakes: Vec<AppliedMistakeEvalRow>,
    train_identity: Vec<String>,
    train_counterexamples: Vec<CounterexampleExample>,
    stats: AppliedMistakeBuildStats,
}

fn load_applied_eval_rows(path: &str) -> anyhow::Result<Vec<AppliedMistakeEvalRow>> {
    let content = std::fs::read_to_string(path)?;
    let mut rows = Vec::new();
    for line in content.lines().filter(|l| !l.trim().is_empty()) {
        rows.push(serde_json::from_str::<AppliedMistakeEvalRow>(line)?);
    }
    Ok(rows)
}

fn vocab_preview<'a>(
    vocab: &'a [crate::db::VocabRow],
    text: &'a str,
) -> (&'a str, Option<String>) {
    if let Some(row) = vocab
        .iter()
        .find(|row| row.term.eq_ignore_ascii_case(text.trim()))
    {
        let spoken = row.spoken().trim();
        return (
            spoken,
            crate::prototype::phonetic_preview(spoken)
                .or_else(|| crate::prototype::phonetic_preview(text)),
        );
    }
    (text, crate::prototype::phonetic_preview(text))
}

fn prototype_trace_excerpt(
    vocab: &[crate::db::VocabRow],
    result: &crate::prototype::PrototypeCorrectionResult,
    reranker: Option<&serde_json::Value>,
) -> serde_json::Value {
    serde_json::json!({
        "proposal_count": result.proposals.len(),
        "accepted_count": result.accepted.len(),
        "sentence_candidate_count": result.sentence_candidates.len(),
        "accepted": result.accepted.iter().map(|edit| {
            let (to_preview, to_preview_phonemes) = vocab_preview(vocab, &edit.to);
            serde_json::json!({
                "from": edit.from,
                "from_phonemes": edit.from_phonemes,
                "to": edit.to,
                "to_preview": to_preview,
                "to_preview_phonemes": to_preview_phonemes,
                "via": edit.via,
                "score": edit.score,
                "acoustic_score": edit.acoustic_score,
                "acoustic_delta": edit.acoustic_delta,
            })
        }).collect::<Vec<_>>(),
        "proposals": result.proposals.iter().take(4).map(|proposal| {
            serde_json::json!({
                "raw_text": proposal.raw_text,
                "normalized": proposal.normalized,
                "phonemes": proposal.phonemes,
                "acoustic_phonemes": proposal.acoustic_phonemes,
                "observed_acoustic_score": proposal.observed_acoustic_score,
                "acoustic_trustworthy": proposal.acoustic_trustworthy,
                "acoustic_window_start_sec": proposal.acoustic_window_start_sec,
                "acoustic_window_end_sec": proposal.acoustic_window_end_sec,
                "candidates": proposal.candidates.iter().take(4).map(|candidate| serde_json::json!({
                    "term": candidate.term,
                    "via": candidate.via,
                    "matched_form": candidate.matched_form,
                    "matched_form_phonemes": candidate.matched_form_phonemes,
                    "term_preview": candidate.term_preview,
                    "term_preview_phonemes": candidate.term_preview_phonemes,
                    "score": candidate.score,
                    "lexical_score": candidate.lexical_score,
                    "dice": candidate.dice,
                    "prefix_ratio": candidate.prefix_ratio,
                    "length_ratio": candidate.length_ratio,
                    "phonetic_score": candidate.phonetic_score,
                    "observed_acoustic_score": candidate.observed_acoustic_score,
                    "acoustic_score": candidate.acoustic_score,
                    "acoustic_delta": candidate.acoustic_delta,
                    "exact_words": candidate.exact_words,
                    "exact_compact": candidate.exact_compact,
                })).collect::<Vec<_>>(),
            })
        }).collect::<Vec<_>>(),
        "sentence_candidates": result.sentence_candidates.iter().take(6).map(|candidate| {
            serde_json::json!({
                "label": candidate.label,
                "text": candidate.text,
                "score": candidate.score,
                "edits": candidate.edits.iter().map(|edit| serde_json::json!({
                    "from": edit.from,
                    "to": edit.to,
                    "via": edit.via,
                    "score": edit.score,
                    "acoustic_score": edit.acoustic_score,
                    "acoustic_delta": edit.acoustic_delta,
                })).collect::<Vec<_>>(),
            })
        }).collect::<Vec<_>>(),
        "reranker": reranker.cloned(),
    })
}

#[derive(Default, Clone, Debug)]
struct PrototypeEvalAnalysis {
    target_proposed: bool,
    target_sentence_candidate: bool,
    target_accepted_edit: bool,
    exact_ok: bool,
    target_ok: bool,
    failure_reason: &'static str,
}

#[derive(Default)]
struct PrototypeEvalFailureBuckets {
    exact_ok: usize,
    target_only: usize,
    reranker_missed_target_candidate: usize,
    proposal_found_no_sentence_edit: usize,
    wrong_proposals_only: usize,
    no_proposal: usize,
}

impl PrototypeEvalFailureBuckets {
    fn record(&mut self, reason: &str) {
        match reason {
            "exact_ok" => self.exact_ok += 1,
            "target_only" => self.target_only += 1,
            "reranker_missed_target_candidate" => self.reranker_missed_target_candidate += 1,
            "proposal_found_no_sentence_edit" => self.proposal_found_no_sentence_edit += 1,
            "wrong_proposals_only" => self.wrong_proposals_only += 1,
            "no_proposal" => self.no_proposal += 1,
            _ => {}
        }
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "exact_ok": self.exact_ok,
            "target_only": self.target_only,
            "reranker_missed_target_candidate": self.reranker_missed_target_candidate,
            "proposal_found_no_sentence_edit": self.proposal_found_no_sentence_edit,
            "wrong_proposals_only": self.wrong_proposals_only,
            "no_proposal": self.no_proposal,
        })
    }
}

fn analyze_prototype_eval_row(
    result: &crate::prototype::PrototypeCorrectionResult,
    term: &str,
    expected_fragment: Option<&str>,
    expected_sentence: &str,
    alt_spellings: &HashMap<String, Vec<String>>,
) -> PrototypeEvalAnalysis {
    let target_term = term.trim();
    let expected_fragment = expected_fragment.unwrap_or(target_term);
    let target_proposed = result.proposals.iter().any(|proposal| {
        proposal
            .candidates
            .iter()
            .any(|candidate| candidate.term.eq_ignore_ascii_case(target_term))
    });
    let target_sentence_candidate = result.sentence_candidates.iter().any(|candidate| {
        eval_fragment_matches(alt_spellings, target_term, expected_fragment, &candidate.text)
    });
    let target_accepted_edit = result.accepted.iter().any(|edit| {
        edit.to.eq_ignore_ascii_case(target_term) || edit.to.eq_ignore_ascii_case(expected_fragment)
    });
    let exact_ok = normalized_compare_eq(&result.corrected, expected_sentence);
    let target_ok =
        eval_fragment_matches(alt_spellings, target_term, expected_fragment, &result.corrected);
    let failure_reason = if exact_ok {
        "exact_ok"
    } else if target_ok {
        "target_only"
    } else if target_sentence_candidate {
        "reranker_missed_target_candidate"
    } else if target_proposed {
        "proposal_found_no_sentence_edit"
    } else if !result.proposals.is_empty() {
        "wrong_proposals_only"
    } else {
        "no_proposal"
    };
    PrototypeEvalAnalysis {
        target_proposed,
        target_sentence_candidate,
        target_accepted_edit,
        exact_ok,
        target_ok,
        failure_reason,
    }
}

#[derive(Deserialize)]
pub struct PrototypeRerankerPrepareBody {
    pub corpus_limit: Option<usize>,
    pub human_limit: Option<usize>,
    pub max_candidates_per_row: Option<usize>,
}

fn normalize_prepare_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn normalized_token(token: &str) -> String {
    token
        .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != ':' && c != '-')
        .to_ascii_lowercase()
}

fn technicalish_token_count(text: &str) -> usize {
    text.split_whitespace()
        .filter(|token| {
            let token = token.trim_matches(|c: char| {
                !c.is_ascii_alphanumeric() && c != '_' && c != ':' && c != '-'
            });
            !token.is_empty()
                && (token.contains("::")
                    || token.contains('_')
                    || token.chars().any(|c| c.is_ascii_digit())
                    || token.chars().skip(1).any(|c| c.is_ascii_uppercase())
                    || (token.len() > 1 && token.chars().all(|c| c.is_ascii_uppercase())))
        })
        .count()
}

fn has_adjacent_repeat(text: &str) -> bool {
    let tokens: Vec<String> = text
        .split_whitespace()
        .map(normalized_token)
        .filter(|t| !t.is_empty())
        .collect();
    tokens.windows(2).any(|pair| pair[0] == pair[1])
}

fn is_prepare_example_reasonable(qwen: &str, expected: &str) -> bool {
    if qwen.is_empty() || expected.is_empty() {
        return false;
    }
    if qwen.contains('\n')
        || expected.contains('\n')
        || qwen.contains("<|")
        || expected.contains("<|")
    {
        return false;
    }

    let q_words = qwen.split_whitespace().count();
    let e_words = expected.split_whitespace().count();
    if q_words < 4 || e_words < 4 || q_words > 22 || e_words > 22 {
        return false;
    }
    if q_words.abs_diff(e_words) > 4 {
        return false;
    }
    if has_adjacent_repeat(qwen) || has_adjacent_repeat(expected) {
        return false;
    }
    if technicalish_token_count(qwen) > 3 || technicalish_token_count(expected) > 3 {
        return false;
    }

    true
}

fn sample_weighted_index(weights: &[u32], rng: &mut impl Rng) -> usize {
    let total: u64 = weights.iter().map(|w| *w as u64).sum();
    if total == 0 {
        return rng.random_range(0..weights.len());
    }
    let mut pick = rng.random_range(0..total);
    for (idx, weight) in weights.iter().enumerate() {
        let weight = *weight as u64;
        if pick < weight {
            return idx;
        }
        pick -= weight;
    }
    weights.len().saturating_sub(1)
}

fn find_ascii_ci(haystack: &str, needle: &str) -> Option<usize> {
    haystack
        .to_ascii_lowercase()
        .find(&needle.to_ascii_lowercase())
}

fn is_fragment_boundary_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_' || ch == ':'
}

fn find_exact_fragment_ascii_ci(haystack: &str, needle: &str) -> Option<usize> {
    let needle = needle.trim();
    if needle.is_empty() {
        return None;
    }
    let haystack_lower = haystack.to_ascii_lowercase();
    let needle_lower = needle.to_ascii_lowercase();
    for (pos, _) in haystack_lower.match_indices(&needle_lower) {
        let before = haystack[..pos].chars().next_back();
        let after = haystack[pos + needle.len()..].chars().next();
        let left_ok = before.map(|ch| !is_fragment_boundary_char(ch)).unwrap_or(true);
        let right_ok = after.map(|ch| !is_fragment_boundary_char(ch)).unwrap_or(true);
        if left_ok && right_ok {
            return Some(pos);
        }
    }
    None
}

fn replace_first_exact_fragment_ascii_ci(
    haystack: &str,
    needle: &str,
    replacement: &str,
) -> Option<String> {
    let needle = needle.trim();
    let pos = find_exact_fragment_ascii_ci(haystack, needle)?;
    Some(format!(
        "{}{}{}",
        &haystack[..pos],
        replacement,
        &haystack[pos + needle.len()..]
    ))
}

fn applied_holdout_sentence(term: &str, sentence: &str) -> bool {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    term.to_ascii_lowercase().hash(&mut hasher);
    normalize_prepare_text(sentence)
        .to_ascii_lowercase()
        .hash(&mut hasher);
    hasher.finish() % 10 == 0
}

fn build_applied_mistake_output(
    db: &crate::db::Db,
    alt_spellings: &HashMap<String, Vec<String>>,
) -> anyhow::Result<AppliedMistakeBuildOutput> {
    let reviewed_vocab = db.list_reviewed_vocab()?;
    let qwen_confusions = db.get_all_qwen_confusions()?;
    let corpus_pairs = db.corpus_eval_set()?;

    let mut known_surfaces: HashMap<String, HashMap<String, (u32, String)>> = HashMap::new();
    for (term, heards) in qwen_confusions {
        for heard in heards {
            let heard = normalize_prepare_text(&heard);
            if heard.is_empty() {
                continue;
            }
            let slot = known_surfaces
                .entry(term.to_ascii_lowercase())
                .or_default()
                .entry(heard.clone())
                .or_insert((0, "vocab_confusions".to_string()));
            slot.0 += 1;
        }
    }
    for item in corpus_pairs {
        if !item.is_mistake {
            continue;
        }
        let qwen = normalize_prepare_text(&item.qwen);
        if qwen.is_empty() || eval_fragment_matches(alt_spellings, &item.term, &item.original, &qwen)
        {
            continue;
        }
        let slot = known_surfaces
            .entry(item.term.to_ascii_lowercase())
            .or_default()
            .entry(qwen)
            .or_insert((0, "corpus_pairs".to_string()));
        slot.0 += item.hit_count.clamp(1, 5) as u32;
    }

    let mut train_mistakes = Vec::new();
    let mut eval_mistakes = Vec::new();
    let mut train_identity = Vec::new();
    let mut train_counterexamples = Vec::new();
    let mut seen = HashSet::<(String, String, String)>::new();
    let mut stats = AppliedMistakeBuildStats::default();

    for vocab in reviewed_vocab {
        let term = vocab.term;
        let term_key = term.to_ascii_lowercase();
        let surfaces = known_surfaces.get(&term_key).cloned().unwrap_or_default();
        let term_sentences = db.authored_sentences_for_term(&term).unwrap_or_default();
        for sentence in term_sentences {
            let sentence = normalize_prepare_text(&sentence);
            if sentence.is_empty() {
                continue;
            }
            if find_exact_fragment_ascii_ci(&sentence, &term).is_none() {
                stats.skipped_missing_term += 1;
                continue;
            }
            let is_holdout = applied_holdout_sentence(&term, &sentence);
            if is_holdout {
                for (surface, (weight, source)) in &surfaces {
                    if eval_fragment_matches(alt_spellings, &term, &term, surface) {
                        stats.skipped_acceptable_surface += 1;
                        continue;
                    }
                    let Some(corrupted) =
                        replace_first_exact_fragment_ascii_ci(&sentence, &term, surface)
                    else {
                        stats.skipped_missing_term += 1;
                        continue;
                    };
                    let corrupted = normalize_prepare_text(&corrupted);
                    if corrupted == sentence || !is_prepare_example_reasonable(&corrupted, &sentence)
                    {
                        continue;
                    }
                    if !seen.insert((term_key.clone(), sentence.clone(), corrupted.clone())) {
                        stats.skipped_duplicate += 1;
                        continue;
                    }
                    eval_mistakes.push(AppliedMistakeEvalRow {
                        term: term.clone(),
                        clean_sentence: sentence.clone(),
                        corrupted_sentence: corrupted,
                        corruption_surface: surface.clone(),
                        source: source.clone(),
                    });
                    stats.eval_mistakes += 1;
                }
            } else {
                if is_prepare_example_reasonable(&sentence, &sentence) {
                    train_identity.push(sentence.clone());
                    stats.train_identity += 1;
                }
                for (surface, (weight, source)) in &surfaces {
                    if eval_fragment_matches(alt_spellings, &term, &term, surface) {
                        stats.skipped_acceptable_surface += 1;
                        continue;
                    }
                    let Some(corrupted) =
                        replace_first_exact_fragment_ascii_ci(&sentence, &term, surface)
                    else {
                        stats.skipped_missing_term += 1;
                        continue;
                    };
                    let corrupted = normalize_prepare_text(&corrupted);
                    if corrupted == sentence || !is_prepare_example_reasonable(&corrupted, &sentence)
                    {
                        continue;
                    }
                    if !seen.insert((term_key.clone(), sentence.clone(), corrupted.clone())) {
                        stats.skipped_duplicate += 1;
                        continue;
                    }
                    train_mistakes.push(AppliedMistakePoolRow {
                        term: term.clone(),
                        clean_sentence: sentence.clone(),
                        corrupted_sentence: corrupted,
                        corruption_surface: surface.clone(),
                        source: source.clone(),
                        weight: (*weight).max(1),
                    });
                    stats.train_mistakes += 1;
                }
            }
        }

        for (sentence, surface_form) in db.authored_counterexamples_for_term(&term).unwrap_or_default() {
            let sentence = normalize_prepare_text(&sentence);
            if sentence.is_empty() || applied_holdout_sentence(&term, &sentence) {
                continue;
            }
            if !sentence
                .to_ascii_lowercase()
                .contains(&surface_form.to_ascii_lowercase())
            {
                continue;
            }
            if is_prepare_example_reasonable(&sentence, &sentence) {
                train_counterexamples.push(CounterexampleExample {
                    sentence,
                    weight: 1,
                });
                stats.train_counterexamples += 1;
            }
        }
    }

    Ok(AppliedMistakeBuildOutput {
        train_mistakes,
        eval_mistakes,
        train_identity,
        train_counterexamples,
        stats,
    })
}

fn correction_example_repeat_count(item: &crate::db::EvalItem, eval_bonus: usize) -> usize {
    let hit_bonus = item.hit_count.clamp(1, 5) as usize;
    let repeat = 1 + (hit_bonus.saturating_sub(1) / 3) + usize::from(eval_bonus > 0);
    repeat.clamp(1, 2)
}

fn counterexample_example_repeat_count(weight: u32) -> usize {
    match weight {
        0..=3 => 1,
        _ => 2,
    }
}

fn build_prototype_reranker_prompt(
    qwen: &str,
    candidate: &crate::prototype::SentenceCandidate,
) -> String {
    let mut prompt = format!(
        "<task> Decide whether the candidate sentence is the correct technical correction of the ASR sentence.\n\
<rules> Answer yes or no only. Prefer plausible repairs to reviewed vocabulary. Reject candidates that change meaning, invent terms, or apply weak edits.\n\
<qwen> {}\n\
<candidate> {}\n\
<edits>",
        qwen.trim(),
        candidate.text.trim(),
    );

    if candidate.edits.is_empty() {
        prompt.push_str(" none");
    } else {
        for edit in &candidate.edits {
            let mut line = format!(
                "\n- {} -> {} | via {} | score {:.2}",
                edit.from, edit.to, edit.via, edit.score
            );
            if let Some(delta) = edit.acoustic_delta {
                line.push_str(&format!(" | Δ {:.2}", delta));
            }
            if let Some(from) = &edit.from_phonemes {
                line.push_str(&format!(" | from {}", from));
            }
            if let Some(to) = &edit.to_phonemes {
                line.push_str(&format!(" | to {}", to));
            }
            prompt.push_str(&line);
        }
    }
    prompt.push_str("\n<answer>");
    prompt
}

fn prototype_reranker_valid_split(term: &str, qwen: &str, expected: &str) -> bool {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    term.to_ascii_lowercase().hash(&mut hasher);
    normalize_prepare_text(qwen)
        .to_ascii_lowercase()
        .hash(&mut hasher);
    normalize_prepare_text(expected)
        .to_ascii_lowercase()
        .hash(&mut hasher);
    hasher.finish() % 10 == 0
}

fn select_prototype_reranker_candidates(
    mut candidates: Vec<crate::prototype::SentenceCandidate>,
    max_candidates_per_row: usize,
) -> Vec<crate::prototype::SentenceCandidate> {
    candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
    let original = candidates.iter().find(|c| c.label == "original").cloned();
    let mut selected = candidates
        .into_iter()
        .take(max_candidates_per_row)
        .collect::<Vec<_>>();
    if let Some(original) = original {
        if !selected.iter().any(|candidate| candidate.label == "original") {
            selected.push(original);
            selected.sort_by(|a, b| b.score.total_cmp(&a.score));
        }
    }
    selected
}

fn load_human_prototype_items(
    state: &Arc<AppState>,
    db: &crate::db::Db,
    alt_spellings: &HashMap<String, Vec<String>>,
    limit: usize,
    randomize: bool,
    sample_seed: u64,
) -> anyhow::Result<Vec<PrototypeBakeoffItem>> {
    use rand::seq::SliceRandom;
    use rand::SeedableRng;

    let mut selected = Vec::new();
    for rec in db.authored_sentence_recordings_for_eval()? {
        let expected_fragment = rec
            .surface_form
            .as_deref()
            .filter(|_| rec.kind == "counterexample")
            .unwrap_or(&rec.term)
            .to_string();
        if !eval_fragment_matches(
            alt_spellings,
            &rec.term,
            &expected_fragment,
            &rec.sentence,
        ) {
            continue;
        }
        selected.push((rec, expected_fragment));
    }
    if randomize {
        let mut rng = rand::rngs::StdRng::seed_from_u64(sample_seed);
        selected.shuffle(&mut rng);
    }
    selected.truncate(limit);

    let mut items = Vec::with_capacity(selected.len());
    for (rec, expected_fragment) in selected {
        let samples_16k = load_authored_recording_16k(&rec.wav_path)?;
        let parakeet = state
            .parakeet
            .transcribe_samples(
                samples_16k,
                16000,
                1,
                Some(parakeet_rs::TimestampMode::Words),
            )?
            .text;
        items.push(PrototypeBakeoffItem {
            case_id: format!("hum-{}", rec.id),
            term: rec.term,
            qwen: parakeet,
            expected: rec.sentence,
            hit_count: 1,
            recording_id: Some(rec.id),
            wav_path: Some(rec.wav_path),
            template_sentence: None,
            qwen_fragment: None,
            expected_fragment: Some(expected_fragment),
        });
    }
    Ok(items)
}

fn prototype_bakeoff_case_id(
    source: &str,
    recording_id: Option<i64>,
    term: &str,
    qwen: &str,
    expected: &str,
) -> String {
    if source == "human" {
        if let Some(id) = recording_id {
            return format!("hum-{id}");
        }
    }
    let prefix = match source {
        "human" => "hum",
        "applied" => "app",
        "corpus" => "cor",
        other => other,
    };
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    source.hash(&mut hasher);
    term.hash(&mut hasher);
    qwen.hash(&mut hasher);
    expected.hash(&mut hasher);
    format!("{prefix}-{:016x}", hasher.finish())
}

fn shuffle_human_bakeoff_items(items: &mut [PrototypeBakeoffItem], seed: u64) {
    use rand::seq::SliceRandom;
    use rand::SeedableRng;

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    items.shuffle(&mut rng);
}

fn latest_eval_failure_term_weights(state: &AppState) -> HashMap<String, usize> {
    let db = state.db.lock().unwrap();
    let jobs = match db.list_jobs() {
        Ok(jobs) => jobs,
        Err(_) => return HashMap::new(),
    };
    drop(db);

    for job in jobs {
        if job.job_type != "eval" || job.status != "completed" {
            continue;
        }
        let Some(result) = job.result.as_deref() else {
            continue;
        };
        let Ok(value) = serde_json::from_str::<serde_json::Value>(result) else {
            continue;
        };
        let Some(entries) = value.get("entries").and_then(|v| v.as_array()) else {
            continue;
        };

        let mut weights = HashMap::new();
        for entry in entries {
            let Some(cat) = entry.get("cat").and_then(|v| v.as_str()) else {
                continue;
            };
            if !matches!(cat, "wrong" | "broken" | "blank" | "timeout") {
                continue;
            }
            let Some(term) = entry.get("term").and_then(|v| v.as_str()) else {
                continue;
            };
            *weights.entry(term.to_ascii_lowercase()).or_insert(0) += 1;
        }

        if !weights.is_empty() {
            return weights;
        }
    }

    HashMap::new()
}

const TEMPLATE_TARGET_PER_TERM: usize = 200;
const TEMPLATE_BATCH_TERMS: usize = 4;
const TEMPLATE_BATCH_SIZE: usize = 25;
const TEMPLATE_MAX_ROUNDS: usize = 8;
const MIN_CORRECTION_EXAMPLES_PER_TERM: usize = 8;
const MAX_CORRECTION_EXAMPLES_PER_TERM: usize = 32;
const COUNTEREXAMPLE_SHARE_OF_NOCHANGE: f64 = 0.5;
const BG_TEMPLATE_TARGET_PER_TERM_DEFAULT: usize = 200;
const BG_CONFUSION_TARGET_PER_TERM_DEFAULT: usize = 8;
const BG_TEMPLATE_TERM_BATCH: usize = 2;
const BG_TEMPLATE_LOCAL_ATTEMPTS_PER_TERM: usize = 4;
const BG_TEMPLATE_OPENAI_PER_TERM: usize = 4;
const BG_CONFUSION_BATCH_TERMS: usize = 12;

struct BackgroundMaintenanceSettings {
    template_target_per_term: usize,
    confusion_target_per_term: usize,
}

fn load_background_maintenance_settings(state: &Arc<AppState>) -> BackgroundMaintenanceSettings {
    let db = state.db.lock().unwrap();
    let template_target_per_term = db
        .get_setting_usize("background_template_target_per_term")
        .ok()
        .flatten()
        .unwrap_or(BG_TEMPLATE_TARGET_PER_TERM_DEFAULT)
        .clamp(1, 1000);
    let confusion_target_per_term = db
        .get_setting_usize("background_confusion_target_per_term")
        .ok()
        .flatten()
        .unwrap_or(BG_CONFUSION_TARGET_PER_TERM_DEFAULT)
        .clamp(1, 1000);
    BackgroundMaintenanceSettings {
        template_target_per_term,
        confusion_target_per_term,
    }
}

fn pick_local_tts_backend(state: &AppState) -> Option<&'static str> {
    let names = state.tts.available_backends();
    if names.contains(&"pocket-hq") {
        Some("pocket-hq")
    } else {
        None
    }
}

fn sentence_contains_term(sentence: &str, term: &str) -> bool {
    let sentence_lower = sentence.to_ascii_lowercase();
    let term_lower = term.to_ascii_lowercase();
    sentence_lower.contains(&term_lower)
}

pub async fn api_template_coverage(
    State(state): State<Arc<AppState>>,
) -> Result<Response, AppError> {
    let db = state.db.lock().unwrap();
    let vocab = db.list_reviewed_vocab().map_err(err)?;
    let authored = db.authored_sentence_term_counts().map_err(err)?;
    let templates = db.template_sentence_term_counts().map_err(err)?;

    let authored_map = authored
        .into_iter()
        .map(|(term, count)| (term.to_ascii_lowercase(), count))
        .collect::<HashMap<_, _>>();
    let template_map = templates
        .into_iter()
        .map(|(term, count)| (term.to_ascii_lowercase(), count))
        .collect::<HashMap<_, _>>();

    let mut rows = vocab
        .into_iter()
        .map(|row| {
            let key = row.term.to_ascii_lowercase();
            let authored_count = authored_map.get(&key).copied().unwrap_or(0);
            let template_count = template_map.get(&key).copied().unwrap_or(0);
            let total = authored_count + template_count;
            serde_json::json!({
                "term": row.term,
                "description": row.description,
                "authored_count": authored_count,
                "template_count": template_count,
                "total_count": total,
                "target": TEMPLATE_TARGET_PER_TERM as i64,
                "coverage_ratio": if TEMPLATE_TARGET_PER_TERM == 0 { 1.0 } else { (total as f64 / TEMPLATE_TARGET_PER_TERM as f64).min(1.0) },
            })
        })
        .collect::<Vec<_>>();

    rows.sort_by(|a, b| {
        let a_total = a.get("total_count").and_then(|v| v.as_i64()).unwrap_or(0);
        let b_total = b.get("total_count").and_then(|v| v.as_i64()).unwrap_or(0);
        a_total.cmp(&b_total).then_with(|| {
            a.get("term")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .cmp(b.get("term").and_then(|v| v.as_str()).unwrap_or(""))
        })
    });

    let ready_terms = rows
        .iter()
        .filter(|row| {
            row.get("total_count").and_then(|v| v.as_i64()).unwrap_or(0)
                >= TEMPLATE_TARGET_PER_TERM as i64
        })
        .count();

    Ok(Json(serde_json::json!({
        "target": TEMPLATE_TARGET_PER_TERM,
        "ready_terms": ready_terms,
        "total_terms": rows.len(),
        "terms": rows,
    }))
    .into_response())
}

pub async fn api_template_sentences(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(term): axum::extract::Path<String>,
) -> Result<Response, AppError> {
    let db = state.db.lock().unwrap();
    let authored = db.authored_sentences_for_term(&term).map_err(err)?;
    let cached = db.template_sentences_for_term(&term).map_err(err)?;
    Ok(Json(serde_json::json!({
        "term": term,
        "authored": authored,
        "cached": cached,
    }))
    .into_response())
}

fn is_template_sentence_reasonable(sentence: &str, term: &str) -> bool {
    let sentence = normalize_prepare_text(sentence);
    if sentence.is_empty() || sentence.contains('\n') || sentence.contains("<|") {
        return false;
    }
    if !sentence_contains_term(&sentence, term) {
        return false;
    }
    let words = sentence.split_whitespace().count();
    if !(5..=24).contains(&words) {
        return false;
    }
    if has_adjacent_repeat(&sentence) {
        return false;
    }
    if technicalish_token_count(&sentence) > 4 {
        return false;
    }
    true
}

fn merge_template_sentence(bank: &mut Vec<String>, sentence: &str) -> bool {
    let sentence = normalize_prepare_text(sentence);
    if bank
        .iter()
        .any(|existing| existing.eq_ignore_ascii_case(&sentence))
    {
        return false;
    }
    bank.push(sentence);
    true
}

async fn fetch_openai_template_batch(
    api_key: &str,
    batch: &[(String, Option<String>)],
    examples: &[String],
    per_term: usize,
) -> anyhow::Result<Vec<(String, String)>> {
    let terms_str = batch
        .iter()
        .map(|(term, desc)| match desc {
            Some(desc) if !desc.trim().is_empty() => format!("{term}: {desc}"),
            _ => term.clone(),
        })
        .collect::<Vec<_>>()
        .join("\n");

    let example_block = if examples.is_empty() {
        String::new()
    } else {
        format!(
            "\n\nStyle reference sentences:\n{}",
            examples
                .iter()
                .take(30)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n")
        )
    };

    let client = reqwest::Client::new();
    let resp = client
        .post("https://api.openai.com/v1/chat/completions")
        .header("Authorization", format!("Bearer {api_key}"))
        .json(&serde_json::json!({
            "model": "gpt-5-mini",
            "messages": [{
                "role": "system",
                "content": "You generate short, natural spoken sentences for ASR correction training. Every sentence must sound like a software developer speaking naturally. Each sentence must contain the exact target term text exactly once. Do not output lists, labels, explanations, markdown, or quoted speech. Output JSON only: {\"suggestions\": [{\"term\": \"...\", \"sentence\": \"...\"}]}."
            }, {
                "role": "user",
                "content": format!(
                    "Generate {per_term} distinct sentences for each target term below. Keep them plausible, concise, and varied.\n\nTargets:\n{terms_str}{example_block}"
                )
            }],
            "response_format": {"type": "json_object"},
        }))
        .send()
        .await?;

    let body: serde_json::Value = resp.json().await?;
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("{\"suggestions\":[]}");
    let parsed: serde_json::Value =
        serde_json::from_str(content).unwrap_or_else(|_| serde_json::json!({"suggestions": []}));

    let mut out = Vec::new();
    if let Some(items) = parsed.get("suggestions").and_then(|v| v.as_array()) {
        for item in items {
            let Some(term) = item.get("term").and_then(|v| v.as_str()) else {
                continue;
            };
            let Some(sentence) = item.get("sentence").and_then(|v| v.as_str()) else {
                continue;
            };
            out.push((term.trim().to_string(), normalize_prepare_text(sentence)));
        }
    }
    Ok(out)
}

async fn hydrate_prepare_template_bank(
    state: &Arc<AppState>,
    job_id: i64,
    terms: &[(String, Option<String>)],
) -> HashMap<String, Vec<String>> {
    let (mut bank, style_examples) = {
        let db = state.db.lock().unwrap();
        let mut bank = HashMap::<String, Vec<String>>::new();
        for (term, _) in terms {
            let mut merged = Vec::new();
            for sentence in db.authored_sentences_for_term(term).unwrap_or_default() {
                if is_template_sentence_reasonable(&sentence, term) {
                    let _ = merge_template_sentence(&mut merged, &sentence);
                }
            }
            for sentence in db.template_sentences_for_term(term).unwrap_or_default() {
                if is_template_sentence_reasonable(&sentence, term) {
                    let _ = merge_template_sentence(&mut merged, &sentence);
                }
            }
            bank.insert(term.to_ascii_lowercase(), merged);
        }
        (bank, db.all_authored_sentences().unwrap_or_default())
    };

    let Ok(api_key) = std::env::var("OPENAI_API_KEY") else {
        let db = state.db.lock().unwrap();
        let _ = db.append_job_log(
            job_id,
            "OPENAI_API_KEY not set. Using cached/authored template sentences only.",
        );
        return bank;
    };

    let mut total_inserted = 0usize;
    for round in 0..TEMPLATE_MAX_ROUNDS {
        let needing = terms
            .iter()
            .filter_map(|(term, desc)| {
                let count = bank
                    .get(&term.to_ascii_lowercase())
                    .map(|items| items.len())
                    .unwrap_or(0);
                (count < TEMPLATE_TARGET_PER_TERM).then_some((term.clone(), desc.clone(), count))
            })
            .collect::<Vec<_>>();

        if needing.is_empty() {
            break;
        }

        {
            let db = state.db.lock().unwrap();
            let _ = db.append_job_log(
                job_id,
                &format!(
                    "Hydrating template bank: round {} with {} terms still below target {}",
                    round + 1,
                    needing.len(),
                    TEMPLATE_TARGET_PER_TERM
                ),
            );
        }

        for chunk in needing.chunks(TEMPLATE_BATCH_TERMS) {
            let batch = chunk
                .iter()
                .map(|(term, desc, _)| (term.clone(), desc.clone()))
                .collect::<Vec<_>>();
            let batch_keys = batch
                .iter()
                .map(|(term, _)| term.to_ascii_lowercase())
                .collect::<Vec<_>>();
            let requested = chunk
                .iter()
                .map(|(_, _, count)| TEMPLATE_BATCH_SIZE.min(TEMPLATE_TARGET_PER_TERM - *count))
                .max()
                .unwrap_or(TEMPLATE_BATCH_SIZE);

            match fetch_openai_template_batch(&api_key, &batch, &style_examples, requested).await {
                Ok(items) => {
                    let mut inserted = 0usize;
                    let db = state.db.lock().unwrap();
                    for (term, sentence) in items {
                        let key = term.to_ascii_lowercase();
                        let resolved_key = if bank.contains_key(&key) {
                            Some(key)
                        } else {
                            batch_keys
                                .iter()
                                .find(|candidate| sentence_contains_term(&sentence, candidate))
                                .cloned()
                        };
                        let Some(resolved_key) = resolved_key else {
                            continue;
                        };
                        let resolved_term = batch
                            .iter()
                            .find(|(candidate, _)| candidate.eq_ignore_ascii_case(&resolved_key))
                            .map(|(candidate, _)| candidate.clone())
                            .unwrap_or_else(|| resolved_key.clone());
                        let Some(existing) = bank.get_mut(&resolved_key) else {
                            continue;
                        };
                        if existing.len() >= TEMPLATE_TARGET_PER_TERM {
                            continue;
                        }
                        if !is_template_sentence_reasonable(&sentence, &resolved_term) {
                            continue;
                        }
                        if merge_template_sentence(existing, &sentence) {
                            let _ = db.insert_template_sentence(
                                &resolved_term,
                                &sentence,
                                "gpt-5-mini",
                            );
                            inserted += 1;
                        }
                    }
                    total_inserted += inserted;
                    let _ = db.append_job_log(
                        job_id,
                        &format!(
                            "Template batch [{}] -> {} new sentences",
                            batch
                                .iter()
                                .map(|(term, _)| term.as_str())
                                .collect::<Vec<_>>()
                                .join(", "),
                            inserted
                        ),
                    );
                }
                Err(e) => {
                    let db = state.db.lock().unwrap();
                    let _ = db.append_job_log(job_id, &format!("Template generation failed: {e}"));
                    return bank;
                }
            }
        }
    }

    let db = state.db.lock().unwrap();
    let hydrated_terms = bank.values().filter(|items| !items.is_empty()).count();
    let _ = db.append_job_log(
        job_id,
        &format!(
            "Template bank ready: {} new cached templates, {} terms with at least one sentence",
            total_inserted, hydrated_terms
        ),
    );
    bank
}

fn background_template_candidates(
    state: &Arc<AppState>,
) -> anyhow::Result<Vec<(String, Option<String>)>> {
    let settings = load_background_maintenance_settings(state);
    let db = state.db.lock().unwrap();
    let vocab = db.list_reviewed_vocab()?;
    let authored = db.authored_sentence_term_counts()?;
    let templates = db.template_sentence_term_counts()?;
    drop(db);

    let authored_map = authored
        .into_iter()
        .map(|(term, count)| (term.to_ascii_lowercase(), count as usize))
        .collect::<HashMap<_, _>>();
    let template_map = templates
        .into_iter()
        .map(|(term, count)| (term.to_ascii_lowercase(), count as usize))
        .collect::<HashMap<_, _>>();

    let mut needing = vocab
        .into_iter()
        .filter_map(|row| {
            let key = row.term.to_ascii_lowercase();
            let total = authored_map.get(&key).copied().unwrap_or(0)
                + template_map.get(&key).copied().unwrap_or(0);
            (total < settings.template_target_per_term).then_some((
                row.term,
                row.description,
                total,
            ))
        })
        .collect::<Vec<_>>();

    needing.sort_by_key(|(_, _, total)| *total);
    Ok(needing
        .into_iter()
        .take(BG_TEMPLATE_TERM_BATCH)
        .map(|(term, desc, _)| (term, desc))
        .collect())
}

async fn background_hydrate_templates_once(state: &Arc<AppState>) -> anyhow::Result<bool> {
    let settings = load_background_maintenance_settings(state);
    let batch = background_template_candidates(state)?;
    if batch.is_empty() {
        return Ok(false);
    }

    let (style_examples, mut bank) = {
        let db = state.db.lock().unwrap();
        let mut bank = HashMap::<String, Vec<String>>::new();
        for (term, _) in &batch {
            let mut merged = Vec::new();
            for sentence in db.authored_sentences_for_term(term).unwrap_or_default() {
                if is_template_sentence_reasonable(&sentence, term) {
                    let _ = merge_template_sentence(&mut merged, &sentence);
                }
            }
            for sentence in db.template_sentences_for_term(term).unwrap_or_default() {
                if is_template_sentence_reasonable(&sentence, term) {
                    let _ = merge_template_sentence(&mut merged, &sentence);
                }
            }
            bank.insert(term.to_ascii_lowercase(), merged);
        }
        (db.all_authored_sentences().unwrap_or_default(), bank)
    };

    let mut made_progress = false;
    let mut generator =
        synth_train::SentenceGenerator::start(&synth_train::SentenceGeneratorConfig::default())
            .ok();

    for (term, desc) in &batch {
        let key = term.to_ascii_lowercase();
        let Some(existing) = bank.get_mut(&key) else {
            continue;
        };
        if existing.len() >= settings.template_target_per_term {
            continue;
        }
        if let Some(gen) = generator.as_mut() {
            for _ in 0..BG_TEMPLATE_LOCAL_ATTEMPTS_PER_TERM {
                if existing.len() >= settings.template_target_per_term {
                    break;
                }
                let Some(sentence) = gen.generate_sentence_retry(term, desc.as_deref(), 2)? else {
                    continue;
                };
                if !is_template_sentence_reasonable(&sentence, term) {
                    continue;
                }
                if merge_template_sentence(existing, &sentence) {
                    let db = state.db.lock().unwrap();
                    let _ = db.insert_template_sentence(term, &sentence, "local-sentgen");
                    made_progress = true;
                }
            }
        }
    }

    let needing_openai = batch
        .iter()
        .filter_map(|(term, desc)| {
            let count = bank
                .get(&term.to_ascii_lowercase())
                .map(|items| items.len())
                .unwrap_or(0);
            (count < settings.template_target_per_term).then_some((term.clone(), desc.clone()))
        })
        .collect::<Vec<_>>();

    if !needing_openai.is_empty() {
        if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
            if let Ok(items) = fetch_openai_template_batch(
                &api_key,
                &needing_openai,
                &style_examples,
                BG_TEMPLATE_OPENAI_PER_TERM,
            )
            .await
            {
                let db = state.db.lock().unwrap();
                for (term, sentence) in items {
                    let key = term.to_ascii_lowercase();
                    let resolved_key = if bank.contains_key(&key) {
                        Some(key)
                    } else {
                        needing_openai
                            .iter()
                            .map(|(candidate, _)| candidate.to_ascii_lowercase())
                            .find(|candidate| sentence_contains_term(&sentence, candidate))
                    };
                    let Some(resolved_key) = resolved_key else {
                        continue;
                    };
                    let resolved_term = needing_openai
                        .iter()
                        .find(|(candidate, _)| candidate.eq_ignore_ascii_case(&resolved_key))
                        .map(|(candidate, _)| candidate.clone())
                        .unwrap_or_else(|| resolved_key.clone());
                    let Some(existing) = bank.get_mut(&resolved_key) else {
                        continue;
                    };
                    if existing.len() >= settings.template_target_per_term {
                        continue;
                    }
                    if !is_template_sentence_reasonable(&sentence, &resolved_term) {
                        continue;
                    }
                    if merge_template_sentence(existing, &sentence) {
                        let _ =
                            db.insert_template_sentence(&resolved_term, &sentence, "gpt-5-mini");
                        made_progress = true;
                    }
                }
            }
        }
    }

    Ok(made_progress)
}

async fn scan_vocab_terms_incremental(
    state: &Arc<AppState>,
    batch: &[(String, Option<String>)],
    tts_backend: &str,
) -> anyhow::Result<usize> {
    if batch.is_empty() {
        return Ok(0);
    }

    let batch_spoken: Vec<String> = batch
        .iter()
        .map(|(t, ovr)| ovr.as_deref().unwrap_or(t).to_string())
        .collect();
    let tts_text = batch_spoken.join(", ");

    let mut audio = state.tts.generate(tts_backend, &tts_text).await?;
    audio.normalize();
    let samples_16k = tts::resample_to_16k(&audio.samples, audio.sample_rate)?;
    let align_items = {
        let state2 = state.clone();
        let samples = samples_16k.clone();
        let text = tts_text.clone();
        tokio::task::spawn_blocking(move || state2.aligner.align(&samples, &text)).await??
    };

    let mut scanned = 0usize;
    let mut align_idx = 0usize;
    for (ti, (term, _spoken_override)) in batch.iter().enumerate() {
        let spoken = &batch_spoken[ti];
        let spoken_words: Vec<&str> = spoken.split_whitespace().collect();
        let num_words = spoken_words.len();
        if align_idx + num_words > align_items.len() {
            continue;
        }

        let start_time = align_items[align_idx].start_time;
        let end_time = align_items[align_idx + num_words - 1].end_time;
        align_idx += num_words;

        if align_idx < align_items.len() {
            let next_word = &align_items[align_idx].word;
            if next_word == "," || next_word.starts_with(',') {
                align_idx += 1;
            }
        }

        let start_sample = (start_time * 16000.0).max(0.0) as usize;
        let end_sample = ((end_time + 0.05) * 16000.0).min(samples_16k.len() as f64) as usize;
        if start_sample >= end_sample || end_sample > samples_16k.len() {
            continue;
        }
        let segment = samples_16k[start_sample..end_sample].to_vec();

        let state_q = state.clone();
        let seg_q = segment.clone();
        let state_p = state.clone();
        let seg_p = segment;
        let qwen_task = tokio::task::spawn_blocking(move || -> String {
            state_q
                .asr
                .transcribe_samples(
                    &seg_q,
                    qwen3_asr::TranscribeOptions::default().with_language("english"),
                )
                .map(|r| r.text)
                .unwrap_or_default()
        });
        let parakeet_task = tokio::task::spawn_blocking(move || -> String {
            state_p
                .parakeet
                .transcribe_samples(seg_p, 16000, 1, None)
                .map(|r| r.text)
                .unwrap_or_default()
        });

        let (qwen, parakeet) = tokio::join!(qwen_task, parakeet_task);
        let qwen = qwen.unwrap_or_default();
        let parakeet = parakeet.unwrap_or_default();
        let term_lower = term.to_lowercase();
        let qwen_match = qwen.trim().to_lowercase() == term_lower;
        let parakeet_match = parakeet.trim().to_lowercase() == term_lower;

        let db = state.db.lock().unwrap();
        let _ = db.insert_confusion(
            term,
            qwen.trim(),
            parakeet.trim(),
            qwen_match,
            parakeet_match,
            tts_backend,
        );
        scanned += 1;
    }

    Ok(scanned)
}

async fn background_confusion_scan_once(state: &Arc<AppState>) -> anyhow::Result<bool> {
    let Some(tts_backend) = pick_local_tts_backend(state) else {
        return Ok(false);
    };
    let settings = load_background_maintenance_settings(state);

    let batch = {
        let db = state.db.lock().unwrap();
        let vocab = db.list_reviewed_vocab()?;
        let scanned_counts = db
            .vocab_scan_results()?
            .into_iter()
            .map(|(term, total, _, _)| (term.to_ascii_lowercase(), total as usize))
            .collect::<HashMap<_, _>>();
        let mut candidates = vocab
            .into_iter()
            .filter_map(|row| {
                let scans = scanned_counts
                    .get(&row.term.to_ascii_lowercase())
                    .copied()
                    .unwrap_or(0);
                (scans < settings.confusion_target_per_term).then_some((
                    row.term,
                    row.spoken_override,
                    scans,
                ))
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|a, b| a.2.cmp(&b.2).then_with(|| a.0.cmp(&b.0)));
        candidates
            .into_iter()
            .take(BG_CONFUSION_BATCH_TERMS)
            .map(|(term, spoken_override, _)| (term, spoken_override))
            .collect::<Vec<_>>()
    };

    if batch.is_empty() {
        return Ok(false);
    }

    let scanned = scan_vocab_terms_incremental(state, &batch, tts_backend).await?;
    Ok(scanned > 0)
}

pub fn spawn_background_maintenance_loop(state: Arc<AppState>, notify: Arc<tokio::sync::Notify>) {
    tokio::spawn(async move {
        loop {
            notify.notified().await;

            if state.training_exclusive() {
                continue;
            }

            loop {
                if state.training_exclusive() {
                    break;
                }
                let mut made_progress = false;

                match background_confusion_scan_once(&state).await {
                    Ok(progress) => made_progress |= progress,
                    Err(e) => eprintln!("[background] confusion scan failed: {e}"),
                }
                match background_hydrate_templates_once(&state).await {
                    Ok(progress) => made_progress |= progress,
                    Err(e) => eprintln!("[background] template hydration failed: {e}"),
                }

                if !made_progress {
                    break;
                }

                tokio::time::sleep(std::time::Duration::from_millis(250)).await;
            }
        }
    });
}

fn decode_wav_mono(wav_bytes: &[u8]) -> anyhow::Result<(Vec<f32>, u32)> {
    let cursor = std::io::Cursor::new(wav_bytes);
    let mut reader = hound::WavReader::new(cursor)?;
    let spec = reader.spec();

    let samples_f32: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().filter_map(|s| s.ok()).collect(),
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max)
                .collect()
        }
    };

    let mut mono: Vec<f32> = if spec.channels > 1 {
        samples_f32
            .chunks(spec.channels as usize)
            .map(|ch| ch.iter().sum::<f32>() / ch.len() as f32)
            .collect()
    } else {
        samples_f32
    };

    let peak = mono.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak > 0.001 {
        let gain = 0.95 / peak;
        for s in &mut mono {
            *s *= gain;
        }
    }

    Ok((mono, spec.sample_rate))
}

fn load_authored_recording_16k(path: &str) -> anyhow::Result<Vec<f32>> {
    let wav_bytes = std::fs::read(path)?;
    let (mono, sample_rate) = if path.ends_with(".ogg") {
        tts::decode_ogg_opus_mono(&wav_bytes)?
    } else {
        decode_wav_mono(&wav_bytes)?
    };
    tts::resample_to_16k(&mono, sample_rate)
}

fn load_inline_audio_16k(audio_b64: &str) -> anyhow::Result<Vec<f32>> {
    use base64::Engine as _;
    let wav = base64::engine::general_purpose::STANDARD.decode(audio_b64)?;
    let (mono, sample_rate) = decode_wav_mono(&wav)?;
    tts::resample_to_16k(&mono, sample_rate)
}

fn write_temp_wav_16k(samples: &[f32], stem: &str) -> anyhow::Result<std::path::PathBuf> {
    let mut path = std::env::temp_dir();
    let unique = format!(
        "hark_{}_{:x}_{}.wav",
        stem,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    );
    path.push(unique);
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&path, spec)?;
    for &sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        writer.write_sample((clamped * i16::MAX as f32) as i16)?;
    }
    writer.finalize()?;
    Ok(path)
}

fn decode_powsm_trace_from_16k(samples_16k: &[f32]) -> anyhow::Result<serde_json::Value> {
    let temp_wav = write_temp_wav_16k(samples_16k, "powsm")?;
    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../scripts/phone_decode_powsm.sh");
    let output = Command::new("bash")
        .arg(script)
        .arg("--json")
        .arg(&temp_wav)
        .output()?;
    let _ = std::fs::remove_file(&temp_wav);
    if !output.status.success() {
        return Err(anyhow::anyhow!(
            "phone_decode_powsm failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let stdout = String::from_utf8(output.stdout)?;
    let line = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .ok_or_else(|| anyhow::anyhow!("phone_decode_powsm produced no JSON"))?;
    Ok(serde_json::from_str(line)?)
}

fn decode_zipa_trace_from_16k(samples_16k: &[f32]) -> anyhow::Result<serde_json::Value> {
    let temp_wav = write_temp_wav_16k(samples_16k, "zipa")?;
    let result = decode_zipa_trace_from_wav_cold(&temp_wav);
    let _ = std::fs::remove_file(&temp_wav);
    result
}

fn decode_zipa_trace_from_wav_cold(path: &std::path::Path) -> anyhow::Result<serde_json::Value> {
    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../scripts/phone_decode_zipa.sh");
    let output = Command::new("bash")
        .arg(script)
        .arg("--json")
        .arg(path)
        .output()?;
    if !output.status.success() {
        return Err(anyhow::anyhow!(
            "phone_decode_zipa failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let stdout = String::from_utf8(output.stdout)?;
    let line = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .ok_or_else(|| anyhow::anyhow!("phone_decode_zipa produced no JSON"))?;
    Ok(serde_json::from_str(line)?)
}

pub struct ZipaSidecar {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl ZipaSidecar {
    pub fn start(repo_id: &str, model_name: &str) -> anyhow::Result<Self> {
        let mut child = Command::new("bash")
            .arg("scripts/phone_decode_zipa_sidecar.sh")
            .env("ZIPA_REPO_ID", repo_id)
            .env("ZIPA_MODEL_NAME", model_name)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow::anyhow!("zipa sidecar missing stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow::anyhow!("zipa sidecar missing stdout"))?;
        let mut sidecar = Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
        };
        let ready = sidecar.read_json_line()?;
        if ready.get("ready").and_then(|v| v.as_bool()) != Some(true) {
            anyhow::bail!("zipa sidecar failed to start: {ready}");
        }
        Ok(sidecar)
    }

    fn read_json_line(&mut self) -> anyhow::Result<serde_json::Value> {
        let mut line = String::new();
        let read = self.stdout.read_line(&mut line)?;
        if read == 0 {
            anyhow::bail!("zipa sidecar closed stdout");
        }
        Ok(serde_json::from_str(line.trim())?)
    }

    pub fn decode_wav_path(
        &mut self,
        path: &std::path::Path,
    ) -> anyhow::Result<serde_json::Value> {
        let req = serde_json::json!({ "audio": path });
        serde_json::to_writer(&mut self.stdin, &req)?;
        self.stdin.write_all(b"\n")?;
        self.stdin.flush()?;
        let value = self.read_json_line()?;
        if let Some(error) = value.get("error").and_then(|v| v.as_str()) {
            anyhow::bail!("zipa sidecar error: {error}");
        }
        Ok(value)
    }

    pub fn kill(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn prime_zipa_sidecar(sidecar: &mut ZipaSidecar, stem: &str) -> anyhow::Result<()> {
    let silence_16k = vec![0.0_f32; 4096];
    let temp_wav = write_temp_wav_16k(&silence_16k, stem)?;
    let result = sidecar.decode_wav_path(&temp_wav);
    let _ = std::fs::remove_file(&temp_wav);
    result.map(|_| ())
}

pub fn prewarm_zipa_sidecars(state: &AppState) -> anyhow::Result<()> {
    {
        let mut guard = state.zipa_sidecar.lock().unwrap();
        if guard.is_none() {
            *guard = Some(ZipaSidecar::start(
                "anyspeech/zipa-small-crctc-300k",
                "model.int8.onnx",
            )?);
        }
        if let Some(sidecar) = guard.as_mut() {
            prime_zipa_sidecar(sidecar, "zipa_prewarm_300k")?;
        }
    }
    {
        let mut guard = state.zipa_ns700k_sidecar.lock().unwrap();
        if guard.is_none() {
            *guard = Some(ZipaSidecar::start(
                "anyspeech/zipa-small-crctc-ns-700k",
                "model.int8.onnx",
            )?);
        }
        if let Some(sidecar) = guard.as_mut() {
            prime_zipa_sidecar(sidecar, "zipa_prewarm_ns700k")?;
        }
    }
    Ok(())
}

fn decode_zipa_trace_from_16k_warm(
    state: &AppState,
    samples_16k: &[f32],
) -> anyhow::Result<serde_json::Value> {
    decode_zipa_ns700k_trace_from_16k_warm(state, samples_16k)
}

fn decode_zipa_300k_trace_from_16k_warm(
    state: &AppState,
    samples_16k: &[f32],
) -> anyhow::Result<serde_json::Value> {
    decode_zipa_trace_from_16k_warm_with_sidecar(
        &state.zipa_sidecar,
        samples_16k,
        "zipa_300k",
        "anyspeech/zipa-small-crctc-300k",
        "model.int8.onnx",
    )
}

fn decode_zipa_ns700k_trace_from_16k_warm(
    state: &AppState,
    samples_16k: &[f32],
) -> anyhow::Result<serde_json::Value> {
    decode_zipa_trace_from_16k_warm_with_sidecar(
        &state.zipa_ns700k_sidecar,
        samples_16k,
        "zipa_ns700k",
        "anyspeech/zipa-small-crctc-ns-700k",
        "model.int8.onnx",
    )
}

fn decode_zipa_trace_from_16k_warm_with_sidecar(
    sidecar_slot: &std::sync::Mutex<Option<ZipaSidecar>>,
    samples_16k: &[f32],
    stem: &str,
    repo_id: &str,
    model_name: &str,
) -> anyhow::Result<serde_json::Value> {
    let total_started = std::time::Instant::now();
    let wav_started = std::time::Instant::now();
    let temp_wav = write_temp_wav_16k(samples_16k, stem)?;
    let temp_wav_write_ms = wav_started.elapsed().as_millis() as u64;
    let result = (|| -> anyhow::Result<serde_json::Value> {
        let mut guard = sidecar_slot.lock().unwrap();
        let mut sidecar_start_ms = 0u64;
        if guard.is_none() {
            let started = std::time::Instant::now();
            *guard = Some(ZipaSidecar::start(repo_id, model_name)?);
            sidecar_start_ms = started.elapsed().as_millis() as u64;
        }
        let sidecar = guard.as_mut().unwrap();
        let roundtrip_started = std::time::Instant::now();
        match sidecar.decode_wav_path(&temp_wav) {
            Ok(mut value) => {
                if let serde_json::Value::Object(obj) = &mut value {
                    obj.insert(
                        "host_timing_ms".to_string(),
                        serde_json::to_value(ZipaHostTiming {
                            temp_wav_write_ms,
                            sidecar_start_ms,
                            sidecar_roundtrip_ms: roundtrip_started.elapsed().as_millis() as u64,
                            total_ms: total_started.elapsed().as_millis() as u64,
                        })
                        .unwrap_or(serde_json::json!({})),
                    );
                }
                Ok(value)
            }
            Err(error) => {
                if let Some(mut server) = guard.take() {
                    server.kill();
                }
                Err(error)
            }
        }
    })();
    let _ = std::fs::remove_file(&temp_wav);
    result.or_else(|_| decode_zipa_trace_from_16k(samples_16k))
}

fn decode_allophant_trace_from_16k(samples_16k: &[f32]) -> anyhow::Result<serde_json::Value> {
    let temp_wav = write_temp_wav_16k(samples_16k, "allophant")?;
    let result = decode_allophant_trace_from_wav_cold(&temp_wav);
    let _ = std::fs::remove_file(&temp_wav);
    result
}

fn decode_allophant_trace_from_wav_cold(
    path: &std::path::Path,
) -> anyhow::Result<serde_json::Value> {
    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../scripts/phone_decode_allophant.sh");
    let output = Command::new("bash")
        .arg(script)
        .arg("--json")
        .arg(path)
        .output()?;
    if !output.status.success() {
        return Err(anyhow::anyhow!(
            "phone_decode_allophant failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let stdout = String::from_utf8(output.stdout)?;
    let line = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .ok_or_else(|| anyhow::anyhow!("phone_decode_allophant produced no JSON"))?;
    Ok(serde_json::from_str(line)?)
}

pub struct AllophantSidecar {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl AllophantSidecar {
    pub fn start() -> anyhow::Result<Self> {
        let mut child = Command::new("bash")
            .arg("scripts/phone_decode_allophant_sidecar.sh")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow::anyhow!("allophant sidecar missing stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow::anyhow!("allophant sidecar missing stdout"))?;
        let mut sidecar = Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
        };
        let ready = sidecar.read_json_line()?;
        if ready.get("ready").and_then(|v| v.as_bool()) != Some(true) {
            anyhow::bail!("allophant sidecar failed to start: {ready}");
        }
        Ok(sidecar)
    }

    fn read_json_line(&mut self) -> anyhow::Result<serde_json::Value> {
        let mut line = String::new();
        let read = self.stdout.read_line(&mut line)?;
        if read == 0 {
            anyhow::bail!("allophant sidecar closed stdout");
        }
        Ok(serde_json::from_str(line.trim())?)
    }

    pub fn decode_wav_path(
        &mut self,
        path: &std::path::Path,
    ) -> anyhow::Result<serde_json::Value> {
        let req = serde_json::json!({ "audio": path });
        serde_json::to_writer(&mut self.stdin, &req)?;
        self.stdin.write_all(b"\n")?;
        self.stdin.flush()?;
        let value = self.read_json_line()?;
        if let Some(error) = value.get("error").and_then(|v| v.as_str()) {
            anyhow::bail!("allophant sidecar error: {error}");
        }
        Ok(value)
    }

    pub fn kill(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn decode_allophant_trace_from_16k_warm(
    state: &AppState,
    samples_16k: &[f32],
) -> anyhow::Result<serde_json::Value> {
    let temp_wav = write_temp_wav_16k(samples_16k, "allophant")?;
    let result = (|| -> anyhow::Result<serde_json::Value> {
        let mut guard = state.allophant_sidecar.lock().unwrap();
        if guard.is_none() {
            *guard = Some(AllophantSidecar::start()?);
        }
        let sidecar = guard.as_mut().unwrap();
        match sidecar.decode_wav_path(&temp_wav) {
            Ok(value) => Ok(value),
            Err(error) => {
                if let Some(mut server) = guard.take() {
                    server.kill();
                }
                Err(error)
            }
        }
    })();
    let _ = std::fs::remove_file(&temp_wav);
    result.or_else(|_| decode_allophant_trace_from_16k(samples_16k))
}

pub struct CohereSidecar {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl CohereSidecar {
    pub fn start() -> anyhow::Result<Self> {
        let mut child = Command::new("bash")
            .arg("scripts/cohere_asr_sidecar.sh")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow::anyhow!("cohere sidecar missing stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow::anyhow!("cohere sidecar missing stdout"))?;
        let mut sidecar = Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
        };
        let ready = sidecar.read_json_line()?;
        if ready.get("ready").and_then(|v| v.as_bool()) != Some(true) {
            anyhow::bail!("cohere sidecar failed to start: {ready}");
        }
        Ok(sidecar)
    }

    fn read_json_line(&mut self) -> anyhow::Result<serde_json::Value> {
        let mut line = String::new();
        let read = self.stdout.read_line(&mut line)?;
        if read == 0 {
            anyhow::bail!("cohere sidecar closed stdout");
        }
        Ok(serde_json::from_str(line.trim())?)
    }

    pub fn transcribe_wav_path(
        &mut self,
        path: &std::path::Path,
    ) -> anyhow::Result<serde_json::Value> {
        let req = serde_json::json!({ "audio": path });
        serde_json::to_writer(&mut self.stdin, &req)?;
        self.stdin.write_all(b"\n")?;
        self.stdin.flush()?;
        let value = self.read_json_line()?;
        if let Some(error) = value.get("error").and_then(|v| v.as_str()) {
            anyhow::bail!("cohere sidecar error: {error}");
        }
        Ok(value)
    }

    pub fn kill(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub fn transcribe_cohere_from_16k_warm(
    state: &AppState,
    samples_16k: &[f32],
) -> anyhow::Result<serde_json::Value> {
    let temp_wav = write_temp_wav_16k(samples_16k, "cohere_asr")?;
    let result = (|| -> anyhow::Result<serde_json::Value> {
        let mut guard = state.cohere_sidecar.lock().unwrap();
        if guard.is_none() {
            *guard = Some(CohereSidecar::start()?);
        }
        let sidecar = guard.as_mut().unwrap();
        match sidecar.transcribe_wav_path(&temp_wav) {
            Ok(value) => Ok(value),
            Err(error) => {
                if let Some(mut server) = guard.take() {
                    server.kill();
                }
                Err(error)
            }
        }
    })();
    let _ = std::fs::remove_file(&temp_wav);
    result
}

fn clean_zipa_phone_string(text: &str) -> String {
    text.replace('▁', " ")
        .split_whitespace()
        .filter(|token| *token != "_" && *token != "▁")
        .collect::<Vec<_>>()
        .join(" ")
}

fn normalize_zipa_to_house_ipa(text: &str) -> String {
    let tokens: Vec<String> = clean_zipa_phone_string(text)
        .split_whitespace()
        .map(|token| match token {
            "g" => "ɡ".to_string(),
            "ɚ" => "əɹ".to_string(),
            "ɝ" => "ɜːɹ".to_string(),
            other => other.to_string(),
        })
        .collect();

    let mut out = Vec::new();
    let mut i = 0usize;
    while i < tokens.len() {
        let current = tokens[i].as_str();
        let next = tokens.get(i + 1).map(|s| s.as_str());
        let combined = match (current, next) {
            ("o", Some("ʊ")) => Some("əʊ"),
            ("a", Some("ɪ")) => Some("aɪ"),
            ("a", Some("ʊ")) => Some("aʊ"),
            ("e", Some("ɪ")) => Some("eɪ"),
            ("ɔ", Some("ɪ")) => Some("ɔɪ"),
            ("ɛ", Some("ɪ")) => Some("eə"),
            ("ɛ", Some("ɚ")) => Some("eə"),
            ("ɛ", Some("əɹ")) => Some("eə"),
            _ => None,
        };
        if let Some(token) = combined {
            out.push(token.to_string());
            i += 2;
            continue;
        }
        out.push(tokens[i].clone());
        i += 1;
    }
    out.join(" ")
}

fn with_leading_silence(samples: &[f32], sample_rate: u32, silence_ms: usize) -> Vec<f32> {
    if silence_ms == 0 {
        return samples.to_vec();
    }
    let silence_samples = ((sample_rate as usize) * silence_ms) / 1000;
    let mut out = Vec::with_capacity(silence_samples + samples.len());
    out.resize(silence_samples, 0.0);
    out.extend_from_slice(samples);
    out
}

pub async fn api_ipa_decode(
    State(state): State<Arc<AppState>>,
    body: axum::body::Bytes,
) -> Result<Response, AppError> {
    let (mono, sample_rate) = decode_wav_mono(&body).map_err(err)?;
    let samples_16k = tts::resample_to_16k(&mono, sample_rate).map_err(err)?;
    let zipa_started = std::time::Instant::now();
    let zipa_trace = decode_zipa_trace_from_16k_warm(&state, &samples_16k).ok();
    let zipa_elapsed_ms = zipa_started.elapsed().as_millis() as u64;
    let zipa_phones = zipa_trace
        .as_ref()
        .and_then(|trace| trace.get("phones"))
        .and_then(|value| value.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let zipa_raw_phones = clean_zipa_phone_string(&zipa_phones);
    let zipa_phones = normalize_zipa_to_house_ipa(&zipa_raw_phones);
    let zipa_300k_started = std::time::Instant::now();
    let zipa_300k_trace = decode_zipa_300k_trace_from_16k_warm(&state, &samples_16k).ok();
    let zipa_300k_elapsed_ms = zipa_300k_started.elapsed().as_millis() as u64;
    let zipa_300k_phones = zipa_300k_trace
        .as_ref()
        .and_then(|trace| trace.get("phones"))
        .and_then(|value| value.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let zipa_300k_raw_phones = clean_zipa_phone_string(&zipa_300k_phones);
    let zipa_300k_phones = normalize_zipa_to_house_ipa(&zipa_300k_raw_phones);
    Ok(Json(serde_json::json!({
        "phones": zipa_phones,
        "trace": zipa_trace,
        "zipa": zipa_trace.as_ref().map(|trace| serde_json::json!({
            "phones": zipa_phones,
            "raw_phones": zipa_raw_phones,
            "trace": trace,
            "elapsed_ms": zipa_elapsed_ms,
        })),
        "zipa_300k": zipa_300k_trace.as_ref().map(|trace| serde_json::json!({
            "phones": zipa_300k_phones,
            "raw_phones": zipa_300k_raw_phones,
            "trace": trace,
            "elapsed_ms": zipa_300k_elapsed_ms,
        })),
        "zipa_ns700k": zipa_trace.as_ref().map(|trace| serde_json::json!({
            "phones": zipa_phones,
            "raw_phones": zipa_raw_phones,
            "trace": trace,
            "elapsed_ms": zipa_elapsed_ms,
        })),
    }))
    .into_response())
}

#[derive(serde::Deserialize)]
pub struct IpaSuggestTtsBody {
    pub text: String,
    pub backend: Option<String>,
}

pub async fn api_ipa_suggest_tts(
    State(state): State<Arc<AppState>>,
    Json(body): Json<IpaSuggestTtsBody>,
) -> Result<Response, AppError> {
    use base64::Engine as _;
    let text = body.text.trim().to_string();
    if text.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "text must not be empty".to_string()));
    }
    let backend = body.backend.unwrap_or_else(|| "openai".to_string());
    let tts_started = std::time::Instant::now();
    let mut audio = state.tts.generate(&backend, &text).await.map_err(err)?;
    audio.normalize();
    let tts_elapsed_ms = tts_started.elapsed().as_millis() as u64;
    let tts_audio_b64 = base64::engine::general_purpose::STANDARD
        .encode(audio.to_wav().map_err(err)?);
    let zipa_input =
        with_leading_silence(&audio.samples, audio.sample_rate, 180);
    let samples_16k = tts::resample_to_16k(&zipa_input, audio.sample_rate).map_err(err)?;
    let zipa_started = std::time::Instant::now();
    let zipa_trace = decode_zipa_trace_from_16k_warm(&state, &samples_16k).map_err(err)?;
    let zipa_elapsed_ms = zipa_started.elapsed().as_millis() as u64;
    let zipa_phones = zipa_trace
        .get("phones")
        .and_then(|value| value.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let zipa_raw_phones = clean_zipa_phone_string(&zipa_phones);
    let zipa_phones = normalize_zipa_to_house_ipa(&zipa_raw_phones);
    let zipa_300k_started = std::time::Instant::now();
    let zipa_300k_trace = decode_zipa_300k_trace_from_16k_warm(&state, &samples_16k).map_err(err)?;
    let zipa_300k_elapsed_ms = zipa_300k_started.elapsed().as_millis() as u64;
    let zipa_300k_phones = zipa_300k_trace
        .get("phones")
        .and_then(|value| value.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let zipa_300k_raw_phones = clean_zipa_phone_string(&zipa_300k_phones);
    let zipa_300k_phones = normalize_zipa_to_house_ipa(&zipa_300k_raw_phones);
    Ok(Json(serde_json::json!({
        "text": text,
        "backend": backend,
        "tts_elapsed_ms": tts_elapsed_ms,
        "tts_audio_b64": tts_audio_b64,
        "zipa_input_pad_ms": 180,
        "zipa": {
            "phones": zipa_phones,
            "raw_phones": zipa_raw_phones,
            "trace": zipa_trace,
            "elapsed_ms": zipa_elapsed_ms,
        },
        "zipa_300k": {
            "phones": zipa_300k_phones,
            "raw_phones": zipa_300k_raw_phones,
            "trace": zipa_300k_trace,
            "elapsed_ms": zipa_300k_elapsed_ms,
        },
        "zipa_ns700k": {
            "phones": zipa_phones,
            "raw_phones": zipa_raw_phones,
            "trace": zipa_trace,
            "elapsed_ms": zipa_elapsed_ms,
        }
    }))
    .into_response())
}

pub async fn api_ipa_suggest_espeak(
    Json(body): Json<IpaSuggestTtsBody>,
) -> Result<Response, AppError> {
    let text = body.text.trim().to_string();
    if text.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "text must not be empty".to_string()));
    }
    let lang = body
        .backend
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("en");
    let started = std::time::Instant::now();
    let data_dir = ensure_espeak_bundled_data_dir().map_err(err)?;
    let engine = espeak_ng::EspeakNg::with_data_dir(lang, &data_dir).map_err(err)?;
    let ipa = engine.text_to_phonemes(&text).map_err(err)?;
    let elapsed_ms = started.elapsed().as_millis() as u64;
    Ok(Json(serde_json::json!({
        "text": text,
        "lang": lang,
        "ipa": ipa.trim(),
        "elapsed_ms": elapsed_ms,
    }))
    .into_response())
}

fn ensure_espeak_bundled_data_dir() -> anyhow::Result<&'static std::path::PathBuf> {
    static DIR: OnceLock<anyhow::Result<std::path::PathBuf>> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = std::env::temp_dir().join("hark-espeak-ng-data");
        std::fs::create_dir_all(&dir)?;
        if !dir.join("phontab").exists() || !dir.join("en_dict").exists() {
            espeak_ng::install_bundled_languages(&dir, &["en"])?;
        }
        Ok(dir)
    })
    .as_ref()
    .map_err(|e| anyhow::anyhow!("{e}"))
}

fn decode_allophant_traces_from_16k_batch(
    clips_16k: &[Vec<f32>],
) -> anyhow::Result<Vec<serde_json::Value>> {
    if clips_16k.is_empty() {
        return Ok(Vec::new());
    }

    let mut temp_wavs = Vec::with_capacity(clips_16k.len());
    for (idx, clip) in clips_16k.iter().enumerate() {
        temp_wavs.push(write_temp_wav_16k(clip, &format!("allophant_word_{idx}"))?);
    }

    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../scripts/phone_decode_allophant.sh");
    let mut command = Command::new("bash");
    command.arg(script).arg("--json");
    for wav in &temp_wavs {
        command.arg(wav);
    }
    let output = command.output();
    for wav in &temp_wavs {
        let _ = std::fs::remove_file(wav);
    }
    let output = output?;
    if !output.status.success() {
        return Err(anyhow::anyhow!(
            "phone_decode_allophant batch failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let stdout = String::from_utf8(output.stdout)?;
    let traces = stdout
        .lines()
        .filter_map(|line| {
            let line = line.trim_start();
            line.starts_with('{')
                .then(|| serde_json::from_str::<serde_json::Value>(line))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if traces.len() != clips_16k.len() {
        return Err(anyhow::anyhow!(
            "phone_decode_allophant batch produced {} JSON payloads for {} clips",
            traces.len(),
            clips_16k.len()
        ));
    }
    Ok(traces)
}

const AUTHORED_PHONE_GROUPING_VERSION: i64 = 4;

fn phone_segments_to_alignment(phone_trace: &serde_json::Value) -> serde_json::Value {
    let segments = phone_trace
        .get("segments")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    serde_json::json!(segments
        .into_iter()
        .filter(|seg| seg.get("within_original_audio").and_then(|v| v.as_bool()).unwrap_or(true))
        .filter_map(|seg| {
            Some(serde_json::json!({
                "w": seg.get("phone")?.as_str()?,
                "s": seg.get("start_sec")?.as_f64()?,
                "e": seg.get("end_sec")?.as_f64()?,
                "avg_logprob": seg.get("avg_logprob").and_then(|v| v.as_f64()),
            }))
        })
        .collect::<Vec<_>>())
}

fn alignment_group_ownership_ranges(
    alignment: &[qwen3_asr::ForcedAlignItem],
) -> Vec<(f64, f64)> {
    if alignment.is_empty() {
        return Vec::new();
    }

    let centers = alignment
        .iter()
        .map(|item| (item.start_time + item.end_time) * 0.5)
        .collect::<Vec<_>>();
    let mut boundaries = Vec::with_capacity(alignment.len() + 1);
    boundaries.push(alignment[0].start_time.min(centers[0]));
    for pair in centers.windows(2) {
        let midpoint = (pair[0] + pair[1]) * 0.5;
        let prev = *boundaries.last().unwrap_or(&midpoint);
        boundaries.push(midpoint.max(prev));
    }
    boundaries.push(alignment.last().unwrap().end_time.max(*boundaries.last().unwrap()));
    boundaries
        .windows(2)
        .map(|pair| (pair[0], pair[1]))
        .collect()
}

fn interval_overlap_sec(a_start: f64, a_end: f64, b_start: f64, b_end: f64) -> f64 {
    (a_end.min(b_end) - a_start.max(b_start)).max(0.0)
}

fn slice_samples_16k(samples_16k: &[f32], start_sec: f64, end_sec: f64) -> Vec<f32> {
    if end_sec <= start_sec || samples_16k.is_empty() {
        return Vec::new();
    }
    let sample_rate = 16_000.0;
    let total_len = samples_16k.len();
    let start = (start_sec.max(0.0) * sample_rate).floor() as usize;
    let end = (end_sec.max(0.0) * sample_rate).ceil() as usize;
    let start = start.min(total_len);
    let end = end.min(total_len);
    if end <= start {
        return Vec::new();
    }
    samples_16k[start..end].to_vec()
}

fn offset_trace_segments(
    trace: &serde_json::Value,
    absolute_start_sec: f64,
) -> anyhow::Result<Vec<crate::prototype::AcousticSegment>> {
    let segments = trace
        .get("segments")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow::anyhow!("allophant trace missing segments"))?;
    let mut out = Vec::with_capacity(segments.len());
    for seg in segments {
        let phone = seg
            .get("phone")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        if phone.is_empty() {
            continue;
        }
        let local_start = seg.get("start_sec").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let local_end = seg.get("end_sec").and_then(|v| v.as_f64()).unwrap_or(local_start);
        if local_end <= local_start {
            continue;
        }
        out.push(crate::prototype::AcousticSegment {
            phone,
            start_sec: absolute_start_sec + local_start,
            end_sec: absolute_start_sec + local_end,
        });
    }
    Ok(out)
}

fn group_phone_segments_by_alignment_json(
    alignment: &[qwen3_asr::ForcedAlignItem],
    segments: &[crate::prototype::AcousticSegment],
) -> serde_json::Value {
    let ownership = alignment_group_ownership_ranges(alignment);
    let mut grouped = vec![Vec::<(crate::prototype::AcousticSegment, f64)>::new(); alignment.len()];

    for seg in segments {
        if seg.end_sec <= seg.start_sec || ownership.is_empty() {
            continue;
        }

        let mut best_idx = None;
        let mut best_overlap = 0.0f64;
        let seg_mid = (seg.start_sec + seg.end_sec) * 0.5;
        let mut best_center_distance = f64::INFINITY;

        for (idx, (own_start, own_end)) in ownership.iter().copied().enumerate() {
            let overlap = interval_overlap_sec(seg.start_sec, seg.end_sec, own_start, own_end);
            if overlap <= 0.0 {
                continue;
            }
            let word_mid = (own_start + own_end) * 0.5;
            let center_distance = (seg_mid - word_mid).abs();
            let better = overlap > best_overlap + 1e-9
                || ((overlap - best_overlap).abs() <= 1e-9
                    && center_distance < best_center_distance);
            if better {
                best_idx = Some(idx);
                best_overlap = overlap;
                best_center_distance = center_distance;
            }
        }

        if let Some(idx) = best_idx {
            grouped[idx].push((seg.clone(), best_overlap));
        }
    }

    serde_json::Value::Array(
        alignment
            .iter()
            .enumerate()
            .map(|(idx, item)| {
                let owned = grouped.get(idx).cloned().unwrap_or_default();
                let phones = owned
                    .iter()
                    .map(|(seg, _)| seg.phone.as_str())
                    .filter(|phone| !phone.trim().is_empty())
                    .collect::<Vec<_>>();
                let phone_items = owned
                    .iter()
                    .map(|(seg, overlap_sec)| {
                        serde_json::json!({
                            "p": seg.phone,
                            "s": seg.start_sec,
                            "e": seg.end_sec,
                            "overlap_sec": overlap_sec,
                        })
                    })
                    .collect::<Vec<_>>();
                let (own_start, own_end) = ownership
                    .get(idx)
                    .copied()
                    .unwrap_or((item.start_time, item.end_time));
                let ipa = if phones.is_empty() {
                    "∅".to_string()
                } else {
                    phones.join(" ")
                };
                serde_json::json!({
                    "word": item.word,
                    "w": ipa,
                    "s": item.start_time,
                    "e": item.end_time,
                    "own_s": own_start,
                    "own_e": own_end,
                    "n": phone_items.len(),
                    "phones": phone_items,
                    "t": format!(
                        "{}: {} | align {:.3}s–{:.3}s | own {:.3}s–{:.3}s",
                        item.word,
                        ipa,
                        item.start_time,
                        item.end_time,
                        own_start,
                        own_end
                    ),
                })
            })
            .collect(),
    )
}

fn decode_grouped_phone_segments_by_alignment_json(
    samples_16k: &[f32],
    alignment: &[qwen3_asr::ForcedAlignItem],
    fallback_segments: &[crate::prototype::AcousticSegment],
) -> anyhow::Result<serde_json::Value> {
    let ownership = alignment_group_ownership_ranges(alignment);
    if ownership.is_empty() {
        return Ok(serde_json::Value::Array(Vec::new()));
    }

    let clips = ownership
        .iter()
        .map(|(own_start, own_end)| slice_samples_16k(samples_16k, *own_start, *own_end))
        .collect::<Vec<_>>();
    let traces = decode_allophant_traces_from_16k_batch(&clips)?;
    if traces.len() != alignment.len() {
        return Err(anyhow::anyhow!(
            "decoded {} traces for {} alignment items",
            traces.len(),
            alignment.len()
        ));
    }
    let fallback_rows = group_phone_segments_by_alignment_json(alignment, fallback_segments)
        .as_array()
        .cloned()
        .unwrap_or_default();

    Ok(serde_json::Value::Array(
        alignment
            .iter()
            .enumerate()
            .map(|(idx, item)| {
                let (own_start, own_end) = ownership
                    .get(idx)
                    .copied()
                    .unwrap_or((item.start_time, item.end_time));
                let trace = traces.get(idx).cloned().unwrap_or(serde_json::json!({}));
                let absolute_segments =
                    offset_trace_segments(&trace, own_start).unwrap_or_else(|_| Vec::new());
                let phones = absolute_segments
                    .iter()
                    .map(|seg| seg.phone.as_str())
                    .filter(|phone| !phone.trim().is_empty())
                    .collect::<Vec<_>>();
                if phones.is_empty() {
                    let mut fallback = fallback_rows.get(idx).cloned().unwrap_or_else(|| {
                        serde_json::json!({
                            "word": item.word,
                            "w": "∅",
                            "s": item.start_time,
                            "e": item.end_time,
                            "own_s": own_start,
                            "own_e": own_end,
                            "n": 0,
                            "phones": [],
                            "t": format!(
                                "{}: ∅ | align {:.3}s–{:.3}s | own {:.3}s–{:.3}s | utterance-fallback",
                                item.word, item.start_time, item.end_time, own_start, own_end
                            ),
                        })
                    });
                    if let Some(obj) = fallback.as_object_mut() {
                        obj.insert(
                            "trace_source".to_string(),
                            serde_json::Value::String("utterance_fallback".to_string()),
                        );
                        obj.insert("slice_trace".to_string(), trace);
                        obj.insert(
                            "own_s".to_string(),
                            serde_json::Value::from(own_start),
                        );
                        obj.insert(
                            "own_e".to_string(),
                            serde_json::Value::from(own_end),
                        );
                    }
                    return fallback;
                }
                let ipa = if phones.is_empty() {
                    "∅".to_string()
                } else {
                    phones.join(" ")
                };
                let phone_items = absolute_segments
                    .iter()
                    .enumerate()
                    .map(|(seg_idx, seg)| {
                        let local = trace
                            .get("segments")
                            .and_then(|v| v.as_array())
                            .and_then(|rows| rows.get(seg_idx))
                            .cloned()
                            .unwrap_or(serde_json::json!({}));
                        serde_json::json!({
                            "p": seg.phone,
                            "s": seg.start_sec,
                            "e": seg.end_sec,
                            "local_s": local.get("start_sec").and_then(|v| v.as_f64()),
                            "local_e": local.get("end_sec").and_then(|v| v.as_f64()),
                            "avg_logprob": local.get("avg_logprob").and_then(|v| v.as_f64()),
                        })
                    })
                    .collect::<Vec<_>>();
                serde_json::json!({
                    "word": item.word,
                    "w": ipa,
                    "s": item.start_time,
                    "e": item.end_time,
                    "own_s": own_start,
                    "own_e": own_end,
                    "n": phone_items.len(),
                    "phones": phone_items,
                    "trace_source": "per_word_allophant",
                    "slice_trace": trace,
                    "t": format!(
                        "{}: {} | align {:.3}s–{:.3}s | own {:.3}s–{:.3}s | per-word",
                        item.word,
                        ipa,
                        item.start_time,
                        item.end_time,
                        own_start,
                        own_end
                    ),
                })
            })
            .collect(),
    ))
}

fn persist_authored_recording_phone_trace(
    state: &Arc<AppState>,
    rec: &crate::db::AuthoredSentenceRecordingRow,
) -> anyhow::Result<()> {
    let qwen_text = rec
        .qwen_clean
        .as_deref()
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .ok_or_else(|| anyhow::anyhow!("recording {} has no qwen_clean", rec.id))?;
    let samples_16k = load_authored_recording_16k(&rec.wav_path)?;
    let trace = decode_allophant_trace_from_16k_warm(state, &samples_16k)?;
    let qwen_alignment = state.aligner.align(&samples_16k, qwen_text)?;
    let sentence_alignment = state.aligner.align(&samples_16k, &rec.sentence)?;
    let qwen_alignment_json = fmt_alignment_json(&qwen_alignment).to_string();
    let sentence_alignment_json = fmt_alignment_json(&sentence_alignment).to_string();
    let utterance_segments = zipa_segments_from_trace(&trace);
    let qwen_grouped_json = decode_grouped_phone_segments_by_alignment_json(
        &samples_16k,
        &qwen_alignment,
        &utterance_segments,
    )
    .unwrap_or_else(|_| group_phone_segments_by_alignment_json(&qwen_alignment, &utterance_segments))
    .to_string();
    let sentence_grouped_json = decode_grouped_phone_segments_by_alignment_json(
        &samples_16k,
        &sentence_alignment,
        &utterance_segments,
    )
    .unwrap_or_else(|_| {
        group_phone_segments_by_alignment_json(&sentence_alignment, &utterance_segments)
    })
    .to_string();
    let db = state.db.lock().unwrap();
    db.upsert_authored_recording_phone_trace(
        rec.id,
        "allophant",
        AUTHORED_PHONE_GROUPING_VERSION,
        trace.get("inventory_lang").and_then(|v| v.as_str()),
        Some(qwen_text),
        Some(&rec.sentence),
        &trace.to_string(),
        Some(&qwen_alignment_json),
        Some(&qwen_grouped_json),
        Some(&sentence_alignment_json),
        Some(&sentence_grouped_json),
    )?;
    Ok(())
}

pub fn spawn_authored_asr_precompute_loop(state: Arc<AppState>, notify: Arc<tokio::sync::Notify>) {
    tokio::spawn(async move {
        loop {
            notify.notified().await;

            if state.training_exclusive() {
                continue;
            }

            loop {
                if state.training_exclusive() {
                    break;
                }
                let pending = {
                    let db = state.db.lock().unwrap();
                    db.authored_recordings_needing_qwen_clean(&state.qwen_model_key, 4)
                };
                let pending = match pending {
                    Ok(rows) => rows,
                    Err(e) => {
                        eprintln!("[authored-asr] failed to query pending recordings: {e}");
                        break;
                    }
                };
                let qwen_pending_empty = pending.is_empty();

                for rec in pending {
                    let state2 = state.clone();
                    let model_key = state.qwen_model_key.clone();
                    let result = tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
                        let samples_16k = load_authored_recording_16k(&rec.wav_path)?;
                        let qwen = state2
                            .asr
                            .transcribe_samples(
                                &samples_16k,
                                qwen3_asr::TranscribeOptions::default().with_language("english"),
                            )?
                            .text;
                        let db = state2.db.lock().unwrap();
                        db.update_authored_recording_qwen_clean(rec.id, &qwen, &model_key)?;
                        Ok(qwen)
                    })
                    .await;

                    match result {
                        Ok(Ok(_)) => {
                            eprintln!("[authored-asr] cached clean qwen for recording {}", rec.id);
                        }
                        Ok(Err(e)) => {
                            eprintln!("[authored-asr] failed to cache recording {}: {e}", rec.id);
                        }
                        Err(e) => {
                            eprintln!("[authored-asr] task failed for recording {}: {e}", rec.id);
                        }
                    }
                }

                let pending_phone = {
                    let db = state.db.lock().unwrap();
                    db.authored_recordings_needing_phone_trace(
                        "allophant",
                        AUTHORED_PHONE_GROUPING_VERSION,
                        4,
                    )
                };
                let pending_phone = match pending_phone {
                    Ok(rows) => rows,
                    Err(e) => {
                        eprintln!("[authored-phones] failed to query pending recordings: {e}");
                        break;
                    }
                };

                if qwen_pending_empty && pending_phone.is_empty() {
                    break;
                }

                for rec in pending_phone {
                    let state2 = state.clone();
                    let rec_id = rec.id;
                    let result = tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
                        persist_authored_recording_phone_trace(&state2, &rec)
                    })
                    .await;

                    match result {
                        Ok(Ok(())) => {
                            eprintln!("[authored-phones] cached timed IPA for recording {}", rec_id);
                        }
                        Ok(Err(e)) => {
                            eprintln!(
                                "[authored-phones] failed to cache recording {}: {e}",
                                rec_id
                            );
                        }
                        Err(e) => {
                            eprintln!(
                                "[authored-phones] task failed for recording {}: {e}",
                                rec_id
                            );
                        }
                    }
                }
            }
        }
    });
}

// ==================== Domain Types ====================

/// The canonical written form of a term (e.g. "serde", "facet-json")
#[derive(Debug, Clone)]
struct WrittenTerm(String);

/// The spoken/pronounced form of a term (e.g. "sir day", "facet json")
#[derive(Debug, Clone)]
struct SpokenTerm(String);

/// A sentence as it should be written (contains the WrittenTerm)
#[derive(Debug, Clone)]
struct WrittenSentence(String);

/// A sentence as it should be spoken (contains the SpokenTerm for TTS)
#[derive(Debug, Clone)]
struct SpokenSentence(String);

/// Build a written sentence containing the term, and its spoken counterpart.
/// Generate a (written, spoken) sentence pair for a vocab term.
/// Tries the LLM generator first if available, falls back to Markov chain.
fn make_sentence_pair(
    generator: Option<&mut synth_train::SentenceGenerator>,
    chain: &MarkovChain,
    term: &WrittenTerm,
    spoken: &SpokenTerm,
    description: Option<&str>,
    overrides: &std::collections::HashMap<String, String>,
    rng: &mut impl Rng,
) -> (WrittenSentence, SpokenSentence) {
    // Try LLM first
    if let Some(gen) = generator {
        if let Ok(Some(sentence)) = gen.generate_sentence_retry(&term.0, description, 3) {
            let mut all_overrides = overrides.clone();
            all_overrides.insert(term.0.clone(), spoken.0.clone());
            let spoken_sentence = tts::build_spoken_form(&sentence, &all_overrides);
            return (WrittenSentence(sentence), SpokenSentence(spoken_sentence));
        }
    }

    // Fallback: Markov chain
    let sentence = chain.generate_with(&term.0, 15, rng);
    let mut all_overrides = overrides.clone();
    all_overrides.insert(term.0.clone(), spoken.0.clone());
    let spoken_sentence = tts::build_spoken_form(&sentence, &all_overrides);
    (WrittenSentence(sentence), SpokenSentence(spoken_sentence))
}

/// Result of running one corpus pass through the pipeline.
struct CorpusPassResult {
    /// The written sentence (ground truth text)
    sentence: WrittenSentence,
    /// The spoken sentence (what TTS said)
    spoken_sentence: SpokenSentence,
    /// Full Qwen ASR transcription
    qwen_full: String,
    /// Full Parakeet ASR transcription
    parakeet_full: String,
    /// Forced alignment of the written sentence against audio
    orig_alignment: Vec<qwen3_asr::ForcedAlignItem>,
    /// Forced alignment of Qwen transcription against audio
    qwen_alignment: Vec<qwen3_asr::ForcedAlignItem>,
    /// Forced alignment of Parakeet transcription against audio
    parakeet_alignment: Vec<qwen3_asr::ForcedAlignItem>,
    /// Consensus extraction result
    extraction: ConsensusResult,
    /// Whether the term was found in the original alignment
    term_found: bool,
    /// Time range of the term in the original alignment
    term_range: (f64, f64),
    /// Forced alignment of the spoken sentence against audio (for visualization)
    spoken_alignment: Vec<qwen3_asr::ForcedAlignItem>,
}

/// Run the ASR + alignment + extraction pipeline on pre-generated audio.
/// `written` is the sentence as written (for alignment ground truth).
/// `spoken` is the sentence as spoken (for TTS alignment / visualization).
/// `term` is the written form of the vocab term (for finding the time range).
async fn run_corpus_pass(
    state: &Arc<AppState>,
    full_16k: &[f32],
    written: &WrittenSentence,
    spoken: &SpokenSentence,
    term: &WrittenTerm,
    protected_terms: &std::collections::HashSet<String>,
    dual_asr: bool,
) -> anyhow::Result<CorpusPassResult> {
    tracing::debug!(term = %term.0, dual_asr, written = %written.0, "run_corpus_pass");
    // ASR — always run Qwen, optionally run Parakeet
    let state_q = state.clone();
    let samples_q = full_16k.to_vec();
    let qwen_task = tokio::task::spawn_blocking(move || -> String {
        state_q
            .asr
            .transcribe_samples(
                &samples_q,
                qwen3_asr::TranscribeOptions::default().with_language("english"),
            )
            .map(|r| r.text)
            .unwrap_or_default()
    });

    let parakeet_full = if dual_asr {
        let state_p = state.clone();
        let samples_p = full_16k.to_vec();
        let parakeet_task = tokio::task::spawn_blocking(move || -> String {
            state_p
                .parakeet
                .transcribe_samples(samples_p, 16000, 1, None)
                .map(|r| r.text)
                .unwrap_or_default()
        });
        parakeet_task.await.unwrap_or_default()
    } else {
        String::new()
    };
    let qwen_full = qwen_task.await.unwrap_or_default();

    tracing::debug!(qwen = %qwen_full, parakeet = %parakeet_full, "ASR results");

    // Alignment — written sentence is ground truth, spoken sentence for visualization
    let orig_alignment = state
        .aligner
        .align(full_16k, &written.0)
        .unwrap_or_default();
    let spoken_alignment = state.aligner.align(full_16k, &spoken.0).unwrap_or_default();

    let term_lower = term.0.to_lowercase();
    let term_found_range = find_term_time_range(&orig_alignment, &term_lower);
    let (term_start, term_end) = expanded_term_time_range(
        &orig_alignment,
        &term_lower,
        full_16k.len() as f64 / 16000.0,
    )
    .unwrap_or((0.0, full_16k.len() as f64 / 16000.0));

    let qwen_alignment = state
        .aligner
        .align(full_16k, &qwen_full)
        .unwrap_or_default();
    let parakeet_alignment = if dual_asr {
        state
            .aligner
            .align(full_16k, &parakeet_full)
            .unwrap_or_default()
    } else {
        Vec::new()
    };

    tracing::debug!(
        term_found = term_found_range.is_some(),
        term_start,
        term_end,
        "alignment done"
    );

    let extraction = extract_with_consensus(
        &orig_alignment,
        &qwen_alignment,
        &parakeet_alignment,
        term_start,
        term_end,
        protected_terms,
    );

    tracing::debug!(clean = extraction.clean, orig = %extraction.original, qwen = %extraction.qwen, para = %extraction.parakeet, "extraction");

    Ok(CorpusPassResult {
        sentence: written.clone(),
        spoken_sentence: spoken.clone(),
        qwen_full,
        parakeet_full,
        orig_alignment,
        qwen_alignment,
        parakeet_alignment,
        extraction,
        term_found: term_found_range.is_some(),
        term_range: (term_start, term_end),
        spoken_alignment,
    })
}

fn fmt_alignment(items: &[qwen3_asr::ForcedAlignItem]) -> String {
    serde_json::to_string(&items.iter().map(|a| {
        serde_json::json!({"w": a.word, "s": (a.start_time * 1000.0).round() / 1000.0, "e": (a.end_time * 1000.0).round() / 1000.0})
    }).collect::<Vec<_>>()).unwrap_or_default()
}

fn fmt_alignment_json(items: &[qwen3_asr::ForcedAlignItem]) -> serde_json::Value {
    serde_json::json!(items.iter().map(|a| {
        serde_json::json!({"w": a.word, "s": (a.start_time * 1000.0).round() / 1000.0, "e": (a.end_time * 1000.0).round() / 1000.0})
    }).collect::<Vec<_>>())
}

// ==================== Markov Chain ====================

/// Word-level bigram Markov chain for generating sentences containing a target word.
#[derive(Clone)]
struct MarkovChain {
    /// word → [(next_word, count)]
    forward: HashMap<String, Vec<(String, u32)>>,
    /// word → [(prev_word, count)]
    backward: HashMap<String, Vec<(String, u32)>>,
}

const SENTENCE_START: &str = "<S>";
const SENTENCE_END: &str = "</S>";

impl MarkovChain {
    fn build(sentences: &[String]) -> Self {
        let mut fwd_counts: HashMap<String, HashMap<String, u32>> = HashMap::new();
        let mut bwd_counts: HashMap<String, HashMap<String, u32>> = HashMap::new();

        const BANNED: &[&str] = &[
            "fuck", "fucking", "shit", "jesus", "christ", "damn", "idiot", "idk", "omg", "lol",
            "lmao", "rofl", "wtf", "stfu", "smh",
        ];

        for sentence in sentences {
            let words: Vec<&str> = sentence.split_whitespace().collect();
            if words.len() < 3 {
                continue;
            }

            // Forward: <S> → w0 → w1 → ... → </S>
            let mut prev = SENTENCE_START.to_string();
            for &w in &words {
                let lower = w.to_lowercase();
                let clean: String = lower.chars().filter(|c| c.is_alphanumeric()).collect();
                if BANNED.contains(&clean.as_str()) {
                    continue;
                }
                *fwd_counts
                    .entry(prev.clone())
                    .or_default()
                    .entry(lower.clone())
                    .or_default() += 1;
                *bwd_counts
                    .entry(lower.clone())
                    .or_default()
                    .entry(prev)
                    .or_default() += 1;
                prev = lower;
            }
            *fwd_counts
                .entry(prev)
                .or_default()
                .entry(SENTENCE_END.to_string())
                .or_default() += 1;
        }

        let to_vec =
            |m: HashMap<String, HashMap<String, u32>>| -> HashMap<String, Vec<(String, u32)>> {
                m.into_iter()
                    .map(|(k, counts)| (k, counts.into_iter().collect()))
                    .collect()
            };

        MarkovChain {
            forward: to_vec(fwd_counts),
            backward: to_vec(bwd_counts),
        }
    }

    /// Pick a random next word weighted by frequency.
    fn sample(choices: &[(String, u32)], rng: &mut impl Rng) -> Option<String> {
        if choices.is_empty() {
            return None;
        }
        let total: u32 = choices.iter().map(|(_, c)| c).sum();
        let mut pick = rng.random_range(0..total);
        for (word, count) in choices {
            if pick < *count {
                return Some(word.clone());
            }
            pick -= count;
        }
        Some(choices.last().unwrap().0.clone())
    }

    /// Generate a sentence containing `target_word`.
    /// Walks forward/backward until a natural sentence boundary (SENTENCE_END/START)
    /// or max_words. Falls back to a template if the word isn't in the chain.
    fn generate_with(&self, target_word: &str, max_words: usize, rng: &mut impl Rng) -> String {
        let target_lower = target_word.to_lowercase();

        // Build forward from target until sentence end
        let mut forward_words = Vec::new();
        let mut cur = target_lower.clone();
        for _ in 0..max_words {
            let Some(choices) = self.forward.get(&cur) else {
                break;
            };
            let Some(next) = Self::sample(choices, rng) else {
                break;
            };
            if next == SENTENCE_END {
                break;
            }
            forward_words.push(next.clone());
            cur = next;
        }

        // Build backward from target until sentence start
        let mut backward_words = Vec::new();
        cur = target_lower.clone();
        for _ in 0..max_words {
            let Some(choices) = self.backward.get(&cur) else {
                break;
            };
            let Some(prev) = Self::sample(choices, rng) else {
                break;
            };
            if prev == SENTENCE_START {
                break;
            }
            backward_words.push(prev.clone());
            cur = prev;
        }
        backward_words.reverse();

        // Ensure the target is never the first word — TTS often clips the start.
        if backward_words.is_empty() {
            const FILLERS: &[&str] = &[
                "So", "Well", "Now", "OK", "Right", "Also", "And", "But", "Then",
            ];
            backward_words.push(FILLERS[rng.random_range(0..FILLERS.len())].to_string());
        }

        // Assemble: backward + target + forward
        let mut words = backward_words;
        words.push(target_word.to_string());
        words.extend(forward_words);

        if words.len() < 3 {
            const TEMPLATES: &[&str] = &[
                "I've been looking into {} and it seems useful.",
                "The {} approach worked really well for us.",
                "We should probably use {} for this project.",
                "Have you tried {} in production yet?",
                "The problem with {} is the documentation.",
                "I think {} would be a good fit here.",
                "After switching to {} things got much better.",
                "Does anyone here have experience with {}?",
            ];
            let idx = rng.random_range(0..TEMPLATES.len());
            return TEMPLATES[idx].replace("{}", target_word);
        }

        // Capitalize first word, add period if no punctuation at end
        if let Some(first) = words.first_mut() {
            let mut chars = first.chars();
            if let Some(c) = chars.next() {
                *first = c.to_uppercase().to_string() + chars.as_str();
            }
        }
        let mut result = words.join(" ");
        if !result.ends_with('.') && !result.ends_with('?') && !result.ends_with('!') {
            result.push('.');
        }
        result
    }
}

/// Apply pronunciation overrides to all vocab words in a sentence.
fn apply_overrides_to_sentence(sentence: &str, overrides: &HashMap<String, String>) -> String {
    let mut result = sentence.to_string();
    // Sort by length descending to replace longer terms first
    let mut terms: Vec<_> = overrides.iter().collect();
    terms.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
    for (term, spoken) in terms {
        if spoken == term {
            continue;
        }
        let lower_result = result.to_lowercase();
        let lower_term = term.to_lowercase();
        if let Some(pos) = lower_result.find(&lower_term) {
            result = format!(
                "{}{}{}",
                &result[..pos],
                spoken,
                &result[pos + term.len()..]
            );
        }
    }
    result
}

fn normalize_eval_fragment(text: &str) -> String {
    text.split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'' && c != '_' && c != '-')
        })
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn normalize_eval_fragment_loose(text: &str) -> String {
    normalize_eval_fragment(text).replace('-', "_")
}

fn normalized_phrase_in_candidate(candidate_norm: &str, phrase_norm: &str) -> bool {
    let phrase_words: Vec<&str> = phrase_norm.split_whitespace().collect();
    if phrase_words.is_empty() {
        return false;
    }
    let candidate_words: Vec<&str> = candidate_norm.split_whitespace().collect();
    if candidate_words.len() < phrase_words.len() {
        return false;
    }
    candidate_words
        .windows(phrase_words.len())
        .any(|window| window == phrase_words)
}

fn eval_fragment_matches(
    alt_spellings: &HashMap<String, Vec<String>>,
    term: &str,
    expected: &str,
    candidate: &str,
) -> bool {
    let expected_norm = normalize_eval_fragment_loose(expected);
    let candidate_norm = normalize_eval_fragment_loose(candidate);
    if candidate_norm == expected_norm
        || normalized_phrase_in_candidate(&candidate_norm, &expected_norm)
    {
        return true;
    }
    alt_spellings
        .get(&term.to_lowercase())
        .map(|alts| {
            alts.iter().any(|alt| {
                let alt_norm = normalize_eval_fragment_loose(alt);
                alt_norm == candidate_norm
                    || normalized_phrase_in_candidate(&candidate_norm, &alt_norm)
            })
        })
        .unwrap_or(false)
}

fn candidate_edits_supported_by_expected(
    candidate: &crate::prototype::SentenceCandidate,
    expected: &str,
) -> bool {
    let expected_norm = normalize_eval_fragment_loose(expected);
    candidate.edits.iter().all(|edit| {
        let to_norm = normalize_eval_fragment_loose(&edit.to);
        !to_norm.is_empty() && normalized_phrase_in_candidate(&expected_norm, &to_norm)
    })
}

fn splice_fragment(sentence: &str, term: &str, fragment: &str) -> String {
    let lower = sentence.to_lowercase();
    let lower_term = term.to_lowercase();
    if let Some(pos) = lower.find(&lower_term) {
        format!(
            "{}{}{}",
            &sentence[..pos],
            fragment,
            &sentence[pos + term.len()..]
        )
    } else {
        fragment.to_string()
    }
}

fn extract_synthetic_fragment(sentence: &str, term: &str, full_text: &str) -> String {
    let lower_sentence = sentence.to_lowercase();
    let lower_term = term.to_lowercase();
    let Some(pos) = lower_sentence.find(&lower_term) else {
        return full_text.trim().to_string();
    };
    let prefix = sentence[..pos].trim_end();
    let suffix = sentence[pos + term.len()..].trim_start();
    let full_trimmed = full_text.trim();

    if prefix.is_empty() || suffix.is_empty() {
        return full_trimmed.to_string();
    }

    if let Some(start) = full_trimmed.find(prefix) {
        let frag_start = start + prefix.len();
        let rest = &full_trimmed[frag_start..];
        if let Some(end_rel) = rest.find(suffix) {
            let fragment = rest[..end_rel].trim();
            if !fragment.is_empty() {
                return fragment.to_string();
            }
        }
    }

    full_trimmed.to_string()
}

struct EvalFocus {
    expected: String,
    extracted_expected: String,
    asr: String,
    corrected: String,
    expected_alignment: serde_json::Value,
    asr_alignment: serde_json::Value,
    corrected_alignment: serde_json::Value,
    cons_range: (f64, f64),
    trim_info: TrimInfo,
    alignment_failed: bool,
    trace: serde_json::Value,
}

fn apply_eval_noise(samples: &[f32], noise_level: f32) -> Vec<f32> {
    if noise_level <= 0.0 {
        return samples.to_vec();
    }

    let mut rng = rand::rngs::StdRng::from_os_rng();
    samples
        .iter()
        .map(|&s| (s + rng.random_range(-noise_level..=noise_level)).clamp(-1.0, 1.0))
        .collect()
}

fn extract_eval_focus(
    state: &Arc<AppState>,
    audio_16k: &[f32],
    expected: &str,
    asr: &str,
    corrected: Option<&str>,
    target_fragment: &str,
) -> anyhow::Result<EvalFocus> {
    let expected_align = state.aligner.align(audio_16k, expected)?;
    let target_fragment_lower = target_fragment.to_lowercase();
    let raw_term_range = find_term_time_range(&expected_align, &target_fragment_lower);
    let term_range = expanded_term_time_range(
        &expected_align,
        &target_fragment_lower,
        audio_16k.len() as f64 / 16000.0,
    )
    .ok_or_else(|| {
        anyhow::anyhow!(
            "target '{}' not found in expected alignment",
            target_fragment
        )
    })?;
    let asr_align = state.aligner.align(audio_16k, asr)?;
    let corrected_align = if let Some(corrected) = corrected.filter(|s| !s.trim().is_empty()) {
        state.aligner.align(audio_16k, corrected)?
    } else {
        Vec::new()
    };

    let protected_terms = std::iter::once(target_fragment.to_lowercase()).collect();
    let extracted = extract_with_consensus(
        &expected_align,
        &asr_align,
        &corrected_align,
        term_range.0,
        term_range.1,
        &protected_terms,
    );

    let expected_norm = normalize_eval_fragment(target_fragment);
    let extracted_expected_norm = normalize_eval_fragment(&extracted.original);
    let extraction_failed = extracted.original.trim().is_empty();
    let expected_fragment_mismatch = extracted_expected_norm != expected_norm;
    let overlap_asr = words_inside_range(&asr_align, term_range.0, term_range.1);
    let overlap_corrected = words_inside_range(&corrected_align, term_range.0, term_range.1);
    let use_overlap_fallback = extraction_failed;
    let trace = serde_json::json!({
        "target_fragment": target_fragment,
        "raw_term_range": raw_term_range.map(|(s, e)| serde_json::json!([s, e])),
        "expanded_term_range": [term_range.0, term_range.1],
        "expected_norm": expected_norm,
        "extracted_expected_norm": extracted_expected_norm,
        "alignment_failed": extraction_failed,
        "expected_fragment_mismatch": expected_fragment_mismatch,
        "used_overlap_fallback": use_overlap_fallback,
        "overlap_asr": overlap_asr,
        "overlap_corrected": overlap_corrected,
        "consensus": extracted.debug,
    });

    Ok(EvalFocus {
        expected: target_fragment.to_string(),
        extracted_expected: extracted.original.clone(),
        asr: if use_overlap_fallback && !overlap_asr.is_empty() {
            overlap_asr
        } else {
            extracted.qwen
        },
        corrected: if use_overlap_fallback && !overlap_corrected.is_empty() {
            overlap_corrected
        } else {
            extracted.parakeet
        },
        expected_alignment: fmt_alignment_json(&expected_align),
        asr_alignment: fmt_alignment_json(&asr_align),
        corrected_alignment: fmt_alignment_json(&corrected_align),
        cons_range: if use_overlap_fallback {
            term_range
        } else {
            extracted.cons_range
        },
        trim_info: extracted.trim_info,
        alignment_failed: extraction_failed,
        trace,
    })
}

/// Find the time range of a spoken term in alignment items.
/// Handles multi-word spoken forms (e.g., "sir day" for serde).
fn find_term_time_range(
    align_items: &[qwen3_asr::ForcedAlignItem],
    spoken_term_lower: &str,
) -> Option<(f64, f64)> {
    let target_clean: String = spoken_term_lower
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == ' ')
        .collect();

    for i in 0..align_items.len() {
        let mut concat = String::new();
        for j in i..align_items.len().min(i + 5) {
            if !concat.is_empty() {
                concat.push(' ');
            }
            concat.push_str(&align_items[j].word.to_lowercase());
            let concat_clean: String = concat
                .chars()
                .filter(|c| c.is_alphanumeric() || *c == ' ')
                .collect();
            if concat_clean.trim() == target_clean.trim() {
                let mut start = align_items[i].start_time;
                let mut end = align_items[j].end_time;
                if end <= start + 0.001 {
                    if i > 0 {
                        start = start.min(align_items[i - 1].end_time);
                    }
                    if j + 1 < align_items.len() {
                        end = end.max(align_items[j + 1].start_time);
                    } else {
                        end = end.max(align_items[j].start_time);
                    }
                    if end <= start + 0.001 {
                        end = start + 0.08;
                    }
                }
                return Some((start, end));
            }
        }
    }
    None
}

/// Find the term range, then expand into neighboring silence gaps exactly like corpus extraction.
///
/// The forced aligner's word boundaries are imprecise. Short technical terms often bleed into
/// the surrounding silence gaps, and the ASR/model lanes may place their recovered word slightly
/// outside the narrow expected box. Expanding into real neighboring gaps gives eval the same
/// boundary semantics as corpus extraction instead of a stricter, mismatched slice.
fn expanded_term_time_range(
    align_items: &[qwen3_asr::ForcedAlignItem],
    spoken_term_lower: &str,
    audio_duration: f64,
) -> Option<(f64, f64)> {
    let (start, end) = find_term_time_range(align_items, spoken_term_lower)?;
    const MIN_GAP: f64 = 0.05; // only expand into gaps > 50ms

    let prev_end = align_items
        .iter()
        .filter(|a| a.end_time <= start + 0.001)
        .last()
        .map(|a| a.end_time)
        .unwrap_or(start);
    let expanded_start = if start - prev_end > MIN_GAP {
        prev_end
    } else {
        start
    };

    let next_start = align_items
        .iter()
        .find(|a| a.start_time >= end - 0.001)
        .map(|a| a.start_time)
        .unwrap_or(end);
    let expanded_end = if next_start - end > MIN_GAP {
        next_start
    } else {
        end
    };

    Some((expanded_start.max(0.0), expanded_end.min(audio_duration)))
}

fn words_overlapping_range(
    items: &[qwen3_asr::ForcedAlignItem],
    start: f64,
    end: f64,
    epsilon: f64,
) -> String {
    items
        .iter()
        .filter(|a| overlaps_range_eps(a, start, end, epsilon))
        .map(|a| {
            a.word
                .trim_matches(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
        })
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn overlaps_range(item: &qwen3_asr::ForcedAlignItem, start: f64, end: f64) -> bool {
    item.end_time > start && item.start_time < end
}

fn overlaps_range_eps(
    item: &qwen3_asr::ForcedAlignItem,
    start: f64,
    end: f64,
    epsilon: f64,
) -> bool {
    item.end_time > start - epsilon && item.start_time < end + epsilon
}

fn words_inside_range(items: &[qwen3_asr::ForcedAlignItem], start: f64, end: f64) -> String {
    items
        .iter()
        .filter(|a| overlaps_range(a, start, end))
        .map(|a| {
            a.word
                .trim_matches(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
        })
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Collect sorted, deduped boundary times from alignment items.
fn lane_boundaries(items: &[qwen3_asr::ForcedAlignItem]) -> Vec<f64> {
    let mut b = Vec::with_capacity(items.len() * 2);
    for a in items {
        b.push(a.start_time);
        b.push(a.end_time);
    }
    b.sort_by(|a, b| a.partial_cmp(b).unwrap());
    b.dedup_by(|a, b| (*a - *b).abs() < 0.002);
    b
}

/// A tri-boundary: a time where all 3 lanes have a boundary,
/// annotated with the word before/after in each lane.
struct TriBoundary {
    time: f64,
    before: (Option<String>, Option<String>, Option<String>), // orig, qwen, para
    after: (Option<String>, Option<String>, Option<String>),
}

impl TriBoundary {
    /// The words before the boundary all match across lanes.
    fn before_matches(&self) -> bool {
        match (&self.before.0, &self.before.1, &self.before.2) {
            (Some(a), Some(b), Some(c)) => {
                a.to_lowercase() == b.to_lowercase() && b.to_lowercase() == c.to_lowercase()
            }
            (None, None, None) => true, // all empty = start of audio
            _ => false,
        }
    }
    /// The words after the boundary all match across lanes.
    fn after_matches(&self) -> bool {
        match (&self.after.0, &self.after.1, &self.after.2) {
            (Some(a), Some(b), Some(c)) => {
                a.to_lowercase() == b.to_lowercase() && b.to_lowercase() == c.to_lowercase()
            }
            (None, None, None) => true, // all empty = end of audio
            _ => false,
        }
    }
}

/// Find the word ending just before `time` in an alignment.
fn word_before(items: &[qwen3_asr::ForcedAlignItem], time: f64, eps: f64) -> Option<String> {
    items
        .iter()
        .rev()
        .find(|a| a.end_time <= time + eps && a.end_time >= time - eps)
        .or_else(|| items.iter().rev().find(|a| a.end_time <= time + eps))
        .map(|a| a.word.clone())
}

/// Find the word starting just after `time` in an alignment.
fn word_after(items: &[qwen3_asr::ForcedAlignItem], time: f64, eps: f64) -> Option<String> {
    items
        .iter()
        .find(|a| a.start_time >= time - eps && a.start_time <= time + eps)
        .or_else(|| items.iter().find(|a| a.start_time >= time - eps))
        .map(|a| a.word.clone())
}

/// Compute annotated tri-boundaries.
fn compute_tri_boundaries(
    orig_b: &[f64],
    qwen_b: &[f64],
    para_b: &[f64],
    orig_align: &[qwen3_asr::ForcedAlignItem],
    qwen_align: &[qwen3_asr::ForcedAlignItem],
    para_align: &[qwen3_asr::ForcedAlignItem],
    epsilon: f64,
) -> Vec<TriBoundary> {
    let mut times = Vec::new();
    for &t in orig_b {
        let q_match = qwen_b.iter().any(|&q| (q - t).abs() <= epsilon);
        let p_match = para_b.iter().any(|&p| (p - t).abs() <= epsilon);
        if q_match && p_match {
            times.push(t);
        }
    }
    times.dedup_by(|a, b| (*a - *b).abs() < epsilon);

    times
        .into_iter()
        .map(|t| TriBoundary {
            time: t,
            before: (
                word_before(orig_align, t, epsilon),
                word_before(qwen_align, t, epsilon),
                word_before(para_align, t, epsilon),
            ),
            after: (
                word_after(orig_align, t, epsilon),
                word_after(qwen_align, t, epsilon),
                word_after(para_align, t, epsilon),
            ),
        })
        .collect()
}

/// Extract words from an alignment that START within [start, end).
fn words_in_range(items: &[qwen3_asr::ForcedAlignItem], start: f64, end: f64) -> String {
    items
        .iter()
        .filter(|a| a.start_time >= start && a.start_time < end)
        .map(|a| a.word.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

struct ConsensusResult {
    original: String,
    qwen: String,
    parakeet: String,
    cons_range: (f64, f64),
    clean: bool,
    debug: serde_json::Value,
    /// Info about what trim_matching_edges did
    trim_info: TrimInfo,
}

/// Records what the edge-trimming step did so we can visualize it.
#[derive(Clone, Debug, serde::Serialize)]
struct TrimInfo {
    /// Words before trimming, per lane
    pre_orig: Vec<String>,
    pre_qwen: Vec<String>,
    pre_para: Vec<String>,
    /// How many words trimmed from left
    trimmed_left: usize,
    /// How many words trimmed from right
    trimmed_right: usize,
    left_trace: serde_json::Value,
    right_trace: serde_json::Value,
}

/// Trim matching words from alignment item slices.
/// Returns (orig, qwen, parakeet) as trimmed word strings.
/// Stops if the original lane's word is a known vocab term (in `protected`).
fn trim_matching_edges(
    orig: &[qwen3_asr::ForcedAlignItem],
    qwen: &[qwen3_asr::ForcedAlignItem],
    para: &[qwen3_asr::ForcedAlignItem],
    start: f64,
    end: f64,
    protected: &std::collections::HashSet<String>,
) -> (String, String, String, TrimInfo) {
    let mut o: Vec<&str> = orig
        .iter()
        .filter(|a| overlaps_range(a, start, end))
        .map(|a| a.word.as_str())
        .collect();
    let mut q: Vec<&str> = qwen
        .iter()
        .filter(|a| overlaps_range(a, start, end))
        .map(|a| a.word.as_str())
        .collect();
    let mut p: Vec<&str> = para
        .iter()
        .filter(|a| overlaps_range(a, start, end))
        .map(|a| a.word.as_str())
        .collect();
    let has_para = !p.is_empty();

    let pre_orig: Vec<String> = o.iter().map(|s| s.to_string()).collect();
    let pre_qwen: Vec<String> = q.iter().map(|s| s.to_string()).collect();
    let pre_para: Vec<String> = p.iter().map(|s| s.to_string()).collect();
    let mut left_trace = Vec::new();
    let mut right_trace = Vec::new();

    let mut trimmed_left = 0;
    while o.len() > 1 && q.len() > 1 && (!has_para || p.len() > 1) {
        let ow = o[0].to_lowercase();
        let qw = q[0].to_lowercase();
        let pw = if has_para {
            Some(p[0].to_lowercase())
        } else {
            None
        };
        if protected.contains(&ow) {
            left_trace.push(serde_json::json!({"action":"stop","reason":"protected_term","orig":ow,"qwen":qw,"para":pw}));
            break;
        }
        if ow != qw {
            left_trace.push(serde_json::json!({"action":"stop","reason":"orig_qwen_mismatch","orig":ow,"qwen":qw,"para":pw}));
            break;
        }
        if has_para {
            let pw_value = pw.clone().unwrap_or_default();
            if ow != pw_value {
                left_trace.push(serde_json::json!({"action":"stop","reason":"orig_para_mismatch","orig":ow,"qwen":qw,"para":pw}));
                break;
            }
            p.remove(0);
        }
        left_trace.push(serde_json::json!({"action":"trim","reason":"all_lanes_match","orig":ow,"qwen":qw,"para":pw}));
        o.remove(0);
        q.remove(0);
        trimmed_left += 1;
    }

    let mut trimmed_right = 0;
    while o.len() > 1 && q.len() > 1 && (!has_para || p.len() > 1) {
        let ow = o.last().unwrap().to_lowercase();
        let qw = q.last().unwrap().to_lowercase();
        let pw = if has_para {
            p.last().map(|w| w.to_lowercase())
        } else {
            None
        };
        if protected.contains(&ow) {
            right_trace.push(serde_json::json!({"action":"stop","reason":"protected_term","orig":ow,"qwen":qw,"para":pw}));
            break;
        }
        if ow != qw {
            right_trace.push(serde_json::json!({"action":"stop","reason":"orig_qwen_mismatch","orig":ow,"qwen":qw,"para":pw}));
            break;
        }
        if has_para {
            let pw_value = pw.clone().unwrap_or_default();
            if ow != pw_value {
                right_trace.push(serde_json::json!({"action":"stop","reason":"orig_para_mismatch","orig":ow,"qwen":qw,"para":pw}));
                break;
            }
            p.pop();
        }
        right_trace.push(serde_json::json!({"action":"trim","reason":"all_lanes_match","orig":ow,"qwen":qw,"para":pw}));
        o.pop();
        q.pop();
        trimmed_right += 1;
    }

    let trim_info = TrimInfo {
        pre_orig,
        pre_qwen,
        pre_para,
        trimmed_left,
        trimmed_right,
        left_trace: serde_json::json!(left_trace),
        right_trace: serde_json::json!(right_trace),
    };
    let clean_join = |words: &[&str]| -> String {
        words
            .iter()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|w| !w.is_empty())
            .collect::<Vec<_>>()
            .join(" ")
    };
    (clean_join(&o), clean_join(&q), clean_join(&p), trim_info)
}

/// Extract triplets using annotated tri-boundaries.
///
/// 1. Compute tri-boundaries with before/after word annotations.
/// 2. For the LEFT boundary: walk tri-boundaries right-to-left from the term start,
///    pick the first one where `before` matches across all 3 lanes.
/// 3. For the RIGHT boundary: walk left-to-right from the term end,
///    pick the first one where `after` matches across all 3 lanes.
/// 4. Slice all 3 lanes with [left, right).
fn extract_with_consensus(
    orig_align: &[qwen3_asr::ForcedAlignItem],
    qwen_align: &[qwen3_asr::ForcedAlignItem],
    parakeet_align: &[qwen3_asr::ForcedAlignItem],
    term_start: f64,
    term_end: f64,
    protected_terms: &std::collections::HashSet<String>,
) -> ConsensusResult {
    let r2 = |v: f64| (v * 1000.0).round() / 1000.0;
    let has_parakeet = !parakeet_align.is_empty();
    let boundary_trace;

    let (start, end, clean) = if has_parakeet {
        // Tri-boundary consensus: all 3 lanes must agree
        let orig_b = lane_boundaries(orig_align);
        let qwen_b = lane_boundaries(qwen_align);
        let para_b = lane_boundaries(parakeet_align);

        let tris = compute_tri_boundaries(
            &orig_b,
            &qwen_b,
            &para_b,
            orig_align,
            qwen_align,
            parakeet_align,
            0.05,
        );

        let left = tris
            .iter()
            .rev()
            .find(|tb| tb.time <= term_start + 0.01 && tb.before_matches())
            .map(|tb| tb.time);
        let right = tris
            .iter()
            .find(|tb| tb.time >= term_end - 0.01 && tb.after_matches())
            .map(|tb| tb.time);
        boundary_trace = serde_json::json!({
            "mode": "tri_boundary",
            "candidates": tris.iter().map(|tb| serde_json::json!({
                "time": tb.time,
                "before": [tb.before.0.clone(), tb.before.1.clone(), tb.before.2.clone()],
                "after": [tb.after.0.clone(), tb.after.1.clone(), tb.after.2.clone()],
                "before_matches": tb.before_matches(),
                "after_matches": tb.after_matches(),
                "eligible_left": tb.time <= term_start + 0.01,
                "eligible_right": tb.time >= term_end - 0.01,
            })).collect::<Vec<_>>(),
            "selected_left": left,
            "selected_right": right,
        });

        (
            left.unwrap_or(term_start),
            right.unwrap_or(term_end),
            left.is_some() && right.is_some(),
        )
    } else {
        // Single-lane: use bi-boundaries (orig + qwen only)
        let orig_b = lane_boundaries(orig_align);
        let qwen_b = lane_boundaries(qwen_align);

        // Find boundaries present in both lanes within epsilon
        let epsilon = 0.05;
        let mut bi_boundaries = Vec::new();
        for &ob in &orig_b {
            if qwen_b.iter().any(|&qb| (ob - qb).abs() < epsilon) {
                bi_boundaries.push(ob);
            }
        }

        let left = bi_boundaries
            .iter()
            .rev()
            .find(|&&t| t <= term_start + 0.01)
            .copied();
        let right = bi_boundaries
            .iter()
            .find(|&&t| t >= term_end - 0.01)
            .copied();
        boundary_trace = serde_json::json!({
            "mode": "bi_boundary",
            "epsilon": epsilon,
            "candidates": bi_boundaries.iter().map(|t| serde_json::json!({
                "time": t,
                "eligible_left": *t <= term_start + 0.01,
                "eligible_right": *t >= term_end - 0.01,
            })).collect::<Vec<_>>(),
            "selected_left": left,
            "selected_right": right,
        });

        (
            left.unwrap_or(term_start),
            right.unwrap_or(term_end),
            left.is_some() && right.is_some(),
        )
    };

    // Extract words in range then trim matching edges.
    // Uses alignment items directly (same word boundaries as the aligner).
    // Stops trimming at vocab terms to avoid eating important words.
    let (original, qwen, parakeet, trim_info) = trim_matching_edges(
        orig_align,
        qwen_align,
        parakeet_align,
        start,
        end,
        protected_terms,
    );

    // If either required lane is empty after trimming, mark as not clean
    let clean = clean && !original.is_empty() && !qwen.is_empty();

    // If consensus range starts at 0 or hits the end, we failed to find a real boundary
    let clean = clean && start > 0.01;

    // Discard if the aligner produced garbage within the consensus range:
    // (a) any phantom word < 20ms, or (b) overlapping timestamps on the same lane.
    const MIN_WORD_DURATION: f64 = 0.02;
    let has_aligner_garbage = |lane: &[qwen3_asr::ForcedAlignItem]| -> (bool, serde_json::Value) {
        let in_range: Vec<_> = lane
            .iter()
            .filter(|a| a.start_time >= start - 0.001 && a.start_time < end + 0.001)
            .collect();
        let too_short = in_range
            .iter()
            .filter(|a| (a.end_time - a.start_time) < MIN_WORD_DURATION)
            .map(|a| {
                serde_json::json!({
                    "word": a.word,
                    "start": a.start_time,
                    "end": a.end_time,
                    "duration": a.end_time - a.start_time,
                })
            })
            .collect::<Vec<_>>();
        let mut overlaps = Vec::new();
        for w in in_range.windows(2) {
            if w[1].start_time < w[0].end_time - 0.005 {
                overlaps.push(serde_json::json!({
                    "left": {"word": w[0].word, "start": w[0].start_time, "end": w[0].end_time},
                    "right": {"word": w[1].word, "start": w[1].start_time, "end": w[1].end_time},
                }));
            }
        }
        let trace = serde_json::json!({
            "in_range": in_range.iter().map(|a| serde_json::json!({
                "word": a.word,
                "start": a.start_time,
                "end": a.end_time,
                "duration": a.end_time - a.start_time,
            })).collect::<Vec<_>>(),
            "too_short": too_short,
            "overlaps": overlaps,
        });
        (
            trace["too_short"]
                .as_array()
                .map(|v| !v.is_empty())
                .unwrap_or(false)
                || trace["overlaps"]
                    .as_array()
                    .map(|v| !v.is_empty())
                    .unwrap_or(false),
            trace,
        )
    };
    let (orig_garbage, orig_garbage_trace) = has_aligner_garbage(orig_align);
    let (qwen_garbage, qwen_garbage_trace) = has_aligner_garbage(qwen_align);
    let clean = clean && !orig_garbage && !qwen_garbage;

    let debug = serde_json::json!({
        "term_start": r2(term_start),
        "term_end": r2(term_end),
        "cons_start": r2(start),
        "cons_end": r2(end),
        "clean": clean,
        "has_parakeet": has_parakeet,
        "boundary_trace": boundary_trace,
        "trim": {
            "pre_orig": trim_info.pre_orig,
            "pre_qwen": trim_info.pre_qwen,
            "pre_para": trim_info.pre_para,
            "trimmed_left": trim_info.trimmed_left,
            "trimmed_right": trim_info.trimmed_right,
            "left_trace": trim_info.left_trace,
            "right_trace": trim_info.right_trace,
            "post_orig": original,
            "post_qwen": qwen,
            "post_para": parakeet,
        },
        "aligner_garbage": {
            "orig": orig_garbage_trace,
            "qwen": qwen_garbage_trace,
        },
    });

    ConsensusResult {
        original,
        qwen,
        parakeet,
        cons_range: (start, end),
        clean,
        debug,
        trim_info,
    }
}

/// Strip carrier phrase prefix variants from ASR output.
fn strip_carrier_prefix(text: &str) -> String {
    let lower = text.to_lowercase();
    // Try various forms ASR might produce for "The word is: X"
    for prefix in &["the word is ", "the word is: ", "the word is, "] {
        if let Some(rest) = lower.strip_prefix(prefix) {
            // Return with original casing from the position after prefix
            let idx = prefix.len();
            if idx < text.len() {
                return text[idx..]
                    .trim_start_matches(|c: char| c == ':' || c == ',' || c == ' ')
                    .to_string();
            }
            return rest.to_string();
        }
    }
    text.to_string()
}

// ==================== Cancel ====================

pub async fn api_stop_job(State(state): State<Arc<AppState>>) -> Result<Response, AppError> {
    state.job_cancel.store(true, Ordering::Relaxed);
    // The running job will see the flag, finish its current item, and exit gracefully
    Ok(Json(serde_json::json!({"ok": true})).into_response())
}

// ==================== Job Configs ====================

#[derive(Deserialize)]
pub struct CorpusJobBody {
    pub tts_backend: Option<String>,
    pub rounds: Option<usize>, // number of rounds (0 = endless, default: 100)
    pub dual_asr: Option<bool>, // true = Qwen + Parakeet, false = Qwen only (default: false)
}

#[derive(Deserialize)]
pub struct PrepareJobBody {
    pub total_examples: Option<usize>,
    pub error_rate: Option<f64>,
}

#[derive(Deserialize)]
pub struct TrainJobBody {
    pub data: Option<String>,
    pub adapters: Option<String>,
    pub model: Option<String>,
    pub iters: Option<usize>,
    pub batch_size: Option<usize>,
    pub num_layers: Option<usize>,
    pub patience: Option<usize>,
    pub steps_per_eval: Option<usize>,
}

// ==================== Job Guard ====================

fn check_no_running_jobs(state: &Arc<AppState>) -> Result<(), AppError> {
    let db = state.db.lock().unwrap();
    let jobs = db.list_jobs().map_err(err)?;
    for job in &jobs {
        if job.status == "running" {
            return Err(err(anyhow::anyhow!(
                "Cannot start job: job #{} ({}) is still running",
                job.id,
                job.job_type
            )));
        }
    }
    // Reset cancel flag for the new job
    state.job_cancel.store(false, Ordering::Relaxed);
    Ok(())
}

// ==================== Corpus Generation ====================

pub async fn api_start_corpus_job(
    State(state): State<Arc<AppState>>,
    Json(body): Json<CorpusJobBody>,
) -> Result<Response, AppError> {
    check_no_running_jobs(&state)?;

    let tts_backend = body.tts_backend.unwrap_or_else(|| "openai".to_string());
    let rounds = body.rounds.unwrap_or(100); // 0 = endless
    let dual_asr = body.dual_asr.unwrap_or(false);
    let config_json =
        serde_json::json!({"tts_backend": tts_backend, "rounds": rounds, "dual_asr": dual_asr})
            .to_string();

    let job_id = {
        let db = state.db.lock().unwrap();
        db.create_job("corpus", Some(&config_json)).map_err(err)?
    };

    let state2 = state.clone();
    let backend = tts_backend.clone();
    tokio::spawn(async move {
        let result = run_corpus_job(&state2, job_id, &backend, rounds, dual_asr).await;
        let db = state2.db.lock().unwrap();
        match result {
            Ok(_) => {
                let _ = db.finish_job(job_id, "completed", None);
            }
            Err(e) => {
                let _ = db.append_job_log(job_id, &format!("ERROR: {e}"));
                let _ = db.finish_job(job_id, "failed", None);
            }
        }
    });

    Ok(Json(serde_json::json!({"job_id": job_id})).into_response())
}

async fn run_corpus_job(
    state: &Arc<AppState>,
    job_id: i64,
    tts_backend: &str,
    max_rounds: usize, // 0 = endless
    dual_asr: bool,
) -> anyhow::Result<()> {
    // Collect reviewed vocab terms + sentence corpus for Markov chains
    let (vocab_terms, all_texts) = {
        let db = state.db.lock().unwrap();
        (db.list_reviewed_vocab()?, db.all_sentence_texts()?)
    };

    if vocab_terms.is_empty() {
        anyhow::bail!("No reviewed vocab terms — nothing to do");
    }

    let protected_terms: std::collections::HashSet<String> =
        vocab_terms.iter().map(|v| v.term.to_lowercase()).collect();

    {
        let db = state.db.lock().unwrap();
        db.append_job_log(
            job_id,
            &format!(
                "Building Markov chain from {} sentences...",
                all_texts.len()
            ),
        )?;
    }
    let chain = MarkovChain::build(&all_texts);

    // Start local LLM sentence generator (falls back to Markov if it fails)
    let mut generator = {
        let db = state.db.lock().unwrap();
        let _ = db.append_job_log(
            job_id,
            "Starting sentence generator (Qwen2.5-1.5B-Instruct)...",
        );
        drop(db);
        match synth_train::SentenceGenerator::start(&synth_train::SentenceGeneratorConfig::default())
        {
            Ok(g) => {
                let db = state.db.lock().unwrap();
                let _ = db.append_job_log(job_id, "Sentence generator ready.");
                Some(g)
            }
            Err(e) => {
                let db = state.db.lock().unwrap();
                let _ = db.append_job_log(
                    job_id,
                    &format!("Sentence generator failed: {e} — falling back to Markov chains"),
                );
                None
            }
        }
    };

    let endless = max_rounds == 0;
    {
        let db = state.db.lock().unwrap();
        db.append_job_log(
            job_id,
            &format!(
                "Corpus: {} terms, {} rounds, backend: {tts_backend}, sentences: {}",
                vocab_terms.len(),
                if endless {
                    "\u{221e}".to_string()
                } else {
                    max_rounds.to_string()
                },
                if generator.is_some() { "LLM" } else { "Markov" },
            ),
        )?;
    }

    let mut rng = rand::rngs::StdRng::from_os_rng();
    let mut round = 0usize;
    let mut new_mistakes = 0usize;
    let mut dup_mistakes = 0usize;
    let mut correct = 0usize;
    let mut noisy = 0usize;
    let mut errors = 0usize;
    let mut tts_ms = 0u64;
    let mut asr_ms = 0u64;
    let job_start = std::time::Instant::now();

    // Pipeline: TTS producer generates items dynamically, consumer does ASR.
    let is_network_tts = tts_backend == "openai" || tts_backend == "elevenlabs";
    let tts_concurrency: usize = if is_network_tts { 8 } else { 4 };

    struct RoundItem {
        term: String,
        sentence: String,
        spoken: String,
    }
    let (tx, mut rx) =
        tokio::sync::mpsc::channel::<(RoundItem, Result<tts::TtsAudio, anyhow::Error>, u64)>(
            tts_concurrency * 2,
        );

    // TTS producer: dynamically picks random terms, generates sentences, does TTS
    let cancel = state.job_cancel.clone();
    let tts_backend_owned = tts_backend.to_string();
    let state_tts = state.clone();
    let vocab_for_producer: Vec<(String, String)> = vocab_terms
        .iter()
        .map(|v| (v.term.clone(), v.spoken().to_string()))
        .collect();
    // Build overrides map for all vocab terms — used to pronounce every term in the sentence
    let overrides: HashMap<String, String> = vocab_terms
        .iter()
        .filter_map(|v| {
            v.spoken_override
                .as_ref()
                .map(|s| (v.term.clone(), s.clone()))
        })
        .collect();
    let chain_clone = chain.clone();
    let overrides_clone = overrides.clone();
    // Build description map for LLM context
    let descriptions: HashMap<String, String> = vocab_terms
        .iter()
        .filter_map(|v| v.description.as_ref().map(|d| (v.term.clone(), d.clone())))
        .collect();
    let producer = tokio::spawn(async move {
        use rand::seq::SliceRandom;
        let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(tts_concurrency));
        let mut rng = rand::rngs::StdRng::from_os_rng();
        let mut produced = 0usize;

        loop {
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            if !endless && produced >= max_rounds {
                break;
            }

            let permit = match semaphore.clone().acquire_owned().await {
                Ok(p) => p,
                Err(_) => break,
            };

            // Pick random term
            let (term, spoken_term) = vocab_for_producer.choose(&mut rng).unwrap().clone();
            let written = WrittenTerm(term.clone());
            let spoken_t = SpokenTerm(spoken_term);
            let desc = descriptions.get(&term).map(|s| s.as_str());
            let (ws, ss) = make_sentence_pair(
                generator.as_mut(),
                &chain_clone,
                &written,
                &spoken_t,
                desc,
                &overrides_clone,
                &mut rng,
            );

            let item = RoundItem {
                term,
                sentence: ws.0,
                spoken: ss.0,
            };
            let tx = tx.clone();
            let state_tts = state_tts.clone();
            let backend = tts_backend_owned.clone();
            let cancel = cancel.clone();

            tokio::spawn(async move {
                if cancel.load(Ordering::Relaxed) {
                    drop(permit);
                    return;
                }
                let t0 = std::time::Instant::now();
                let result = state_tts.tts.generate(&backend, &item.spoken).await;
                let ms = t0.elapsed().as_millis() as u64;
                let _ = tx.send((item, result, ms)).await;
                drop(permit);
            });

            produced += 1;
        }
    });

    // Consumer: ASR + alignment + upsert
    while let Some((item, tts_result, tts_call_ms)) = rx.recv().await {
        if state.job_cancel.load(Ordering::Relaxed) {
            let db = state.db.lock().unwrap();
            let _ = db.append_job_log(job_id, "Stopped by user.");
            break;
        }

        round += 1;
        tts_ms += tts_call_ms;

        let mut audio = match tts_result {
            Ok(mut a) => {
                a.normalize();
                a
            }
            Err(e) => {
                let db = state.db.lock().unwrap();
                let _ = db.append_job_log(job_id, &format!("TTS FAILED: {e}"));
                errors += 1;
                continue;
            }
        };

        let full_16k = match tts::resample_to_16k(&audio.samples, audio.sample_rate) {
            Ok(s) => s,
            Err(_) => {
                errors += 1;
                continue;
            }
        };

        let t0 = std::time::Instant::now();
        let written_term = WrittenTerm(item.term.clone());
        let written_sentence = WrittenSentence(item.sentence.clone());
        let spoken_sentence = SpokenSentence(item.spoken.clone());
        let result = match run_corpus_pass(
            state,
            &full_16k,
            &written_sentence,
            &spoken_sentence,
            &written_term,
            &protected_terms,
            dual_asr,
        )
        .await
        {
            Ok(r) => r,
            Err(e) => {
                tracing::error!("corpus pass failed: {e}");
                errors += 1;
                continue;
            }
        };
        asr_ms += t0.elapsed().as_millis() as u64;

        if result.extraction.clean {
            let cons = &result.extraction;
            let is_mistake = {
                let db = state.db.lock().unwrap();
                !db.is_acceptable_spelling(&item.term, &cons.original, &cons.qwen)
                    .unwrap_or(false)
            };
            let cons_time_json =
                serde_json::to_string(&[cons.cons_range.0, cons.cons_range.1]).ok();
            let trim_info_json = serde_json::to_string(&cons.trim_info).ok();

            // Encode audio as Ogg Opus for playback in the review UI
            let ogg_bytes = tts::encode_ogg_opus(&audio.samples, audio.sample_rate)
                .await
                .ok();

            let db = state.db.lock().unwrap();
            match db.upsert_corpus_pair(
                &item.term,
                &cons.original,
                &cons.qwen,
                &cons.parakeet,
                &item.sentence,
                &item.spoken,
                Some(&fmt_alignment(&result.orig_alignment)),
                Some(&fmt_alignment(&result.qwen_alignment)),
                Some(&fmt_alignment(&result.parakeet_alignment)),
                cons_time_json.as_deref(),
                trim_info_json.as_deref(),
                is_mistake,
                ogg_bytes.as_deref(),
            ) {
                Ok((pair_id, is_new)) => {
                    if is_mistake {
                        if is_new {
                            let _ = db.append_job_log(
                                job_id,
                                &format!(
                                    "NEW|{}|{}|{}|{}",
                                    pair_id, item.term, cons.original, cons.qwen
                                ),
                            );
                            new_mistakes += 1;
                        } else {
                            dup_mistakes += 1;
                        }
                    } else {
                        correct += 1;
                    }
                }
                Err(e) => {
                    tracing::error!("upsert_corpus_pair failed for '{}': {e}", item.term);
                    errors += 1;
                }
            }
        } else {
            noisy += 1;
        }

        // Progress log every 10 rounds
        if round % 10 == 0 {
            let elapsed_ms = job_start.elapsed().as_millis() as u64;
            let rps = if elapsed_ms > 0 {
                round as f64 / (elapsed_ms as f64 / 1000.0)
            } else {
                0.0
            };
            let db = state.db.lock().unwrap();
            let _ = db.append_job_log(
                job_id,
                &format!(
                    "[{}{}] {} new, {} dup, {} ok, {} noisy, {} err | {:.1}/s",
                    round,
                    if endless {
                        String::new()
                    } else {
                        format!("/{max_rounds}")
                    },
                    new_mistakes,
                    dup_mistakes,
                    correct,
                    noisy,
                    errors,
                    rps,
                ),
            );
            let _ = db.update_job_result(job_id, &serde_json::json!({
                "rounds": round,
                "new_mistakes": new_mistakes,
                "dup_mistakes": dup_mistakes,
                "correct": correct,
                "noisy": noisy,
                "errors": errors,
                "rounds_per_sec": (rps * 10.0).round() / 10.0,
                "avg_tts_ms": if round > 0 { (tts_ms as f64 / round as f64).round() as u64 } else { 0 },
                "avg_asr_ms": if round > 0 { (asr_ms as f64 / round as f64).round() as u64 } else { 0 },
            }).to_string());
        }
    }

    let _ = producer.await;

    let db = state.db.lock().unwrap();
    db.append_job_log(job_id, &format!(
        "Done: {round} rounds — {new_mistakes} new mistakes, {dup_mistakes} duplicates, {correct} correct, {noisy} noisy, {errors} errors",
    ))?;
    let _ = db.update_job_result(
        job_id,
        &serde_json::json!({
            "rounds": round,
            "new_mistakes": new_mistakes,
            "dup_mistakes": dup_mistakes,
            "correct": correct,
            "noisy": noisy,
            "errors": errors,
        })
        .to_string(),
    );

    Ok(())
}

// ==================== Prepare ====================

pub async fn api_start_prepare_job(
    State(state): State<Arc<AppState>>,
    Json(body): Json<PrepareJobBody>,
) -> Result<Response, AppError> {
    check_no_running_jobs(&state)?;

    let total_examples = body.total_examples.unwrap_or(12000);
    let error_rate = body.error_rate.unwrap_or(0.5);
    let config_json =
        serde_json::json!({"total_examples": total_examples, "error_rate": error_rate}).to_string();

    let job_id = {
        let db = state.db.lock().unwrap();
        db.create_job("prepare", Some(&config_json)).map_err(err)?
    };

    let state2 = state.clone();
    tokio::spawn(async move {
        let state3 = state2.clone();
        let handle = tokio::task::spawn_blocking(move || {
            let mut rng = rand::rngs::StdRng::from_os_rng();
            let alt_spellings = {
                let db = state3.db.lock().unwrap();
                db.get_all_alt_spellings().unwrap_or_default()
            };
            let build = {
                let db = state3.db.lock().unwrap();
                build_applied_mistake_output(&db, &alt_spellings)
            };

            let Ok(build) = build else {
                let db = state3.db.lock().unwrap();
                let _ = db.append_job_log(
                    job_id,
                    "ERROR: Failed to build applied-mistake dataset from authored sentences and known confusions.",
                );
                let _ = db.finish_job(job_id, "failed", None);
                return;
            };

            if build.train_mistakes.is_empty() || build.eval_mistakes.is_empty() {
                let db = state3.db.lock().unwrap();
                let _ = db.append_job_log(
                    job_id,
                    &format!(
                        "ERROR: Need both train and held-out applied mistakes. Got {} train mistake rows and {} held-out eval rows.",
                        build.train_mistakes.len(),
                        build.eval_mistakes.len()
                    ),
                );
                let _ = db.finish_job(job_id, "failed", None);
                return;
            }
            if build.train_identity.is_empty() {
                let db = state3.db.lock().unwrap();
                let _ = db.append_job_log(
                    job_id,
                    "ERROR: No clean authored train sentences available after filtering.",
                );
                let _ = db.finish_job(job_id, "failed", None);
                return;
            }

            let n_error = (total_examples as f64 * error_rate).round() as usize;
            let n_identity = total_examples.saturating_sub(n_error);
            let n_counterexample_target = if build.train_counterexamples.is_empty() {
                0
            } else {
                ((n_identity as f64) * COUNTEREXAMPLE_SHARE_OF_NOCHANGE).round() as usize
            };
            let n_plain_identity_target = n_identity.saturating_sub(n_counterexample_target);

            {
                let db = state3.db.lock().unwrap();
                let _ = db.append_job_log(
                    job_id,
                    &format!(
                        "Preparing from clean authored sentences + one applied known mistake: {} train mistakes, {} held-out eval mistakes, {} clean train sentences, {} counterexamples",
                        build.train_mistakes.len(),
                        build.eval_mistakes.len(),
                        build.train_identity.len(),
                        build.train_counterexamples.len(),
                    ),
                );
                let _ = db.append_job_log(
                    job_id,
                    &format!(
                        "Generating {} error + {} no-change ({} plain identity, {} counterexample target) = {} total examples",
                        n_error,
                        n_identity,
                        n_plain_identity_target,
                        n_counterexample_target,
                        total_examples,
                    ),
                );
            }

            let mut examples = Vec::with_capacity(total_examples);
            let mut seen_prompts = HashMap::<String, String>::new();
            let mut correction_count = 0usize;
            let mut identity_count = 0usize;
            let mut counterexample_count = 0usize;
            let mut duplicate_skips = 0usize;
            let mut conflict_skips = 0usize;
            let mut weighted_repeat_pushes = 0usize;

            let mut push_example =
                |example: PreparedExample| match seen_prompts.get(&example.prompt) {
                    Some(existing) if existing == &example.completion => {
                        if example.allow_repeat {
                            examples.push(serde_json::json!({
                                "prompt": example.prompt,
                                "completion": example.completion,
                            }));
                            weighted_repeat_pushes += 1;
                            true
                        } else {
                            duplicate_skips += 1;
                            false
                        }
                    }
                    Some(_) => {
                        conflict_skips += 1;
                        false
                    }
                    None => {
                        seen_prompts.insert(example.prompt.clone(), example.completion.clone());
                        examples.push(serde_json::json!({
                            "prompt": example.prompt,
                            "completion": example.completion,
                        }));
                        true
                    }
                };

            let error_weights: Vec<u32> = build
                .train_mistakes
                .iter()
                .map(|row| row.weight.max(1))
                .collect();
            let mut error_attempts = 0usize;
            let max_error_attempts = n_error.saturating_mul(30).max(2000);
            while correction_count < n_error && error_attempts < max_error_attempts {
                error_attempts += 1;
                let row = &build.train_mistakes[sample_weighted_index(&error_weights, &mut rng)];
                let prompt = synth_train::build_correction_prompt("", &row.corrupted_sentence);
                if push_example(PreparedExample {
                    prompt,
                    completion: format!(" {}<|endoftext|>", row.clean_sentence),
                    allow_repeat: true,
                }) {
                    correction_count += 1;
                }
            }

            let counterexample_weights: Vec<u32> = build
                .train_counterexamples
                .iter()
                .map(|row| row.weight.max(1))
                .collect();
            let mut counterexample_attempts = 0usize;
            let max_counterexample_attempts = n_counterexample_target.saturating_mul(20).max(200);
            while counterexample_count < n_counterexample_target
                && counterexample_attempts < max_counterexample_attempts
            {
                counterexample_attempts += 1;
                let item = &build.train_counterexamples
                    [sample_weighted_index(&counterexample_weights, &mut rng)];
                let prompt = synth_train::build_correction_prompt("", &item.sentence);
                let repeats = counterexample_example_repeat_count(item.weight);
                let mut added = 0usize;
                for _ in 0..repeats {
                    if counterexample_count + added >= n_counterexample_target {
                        break;
                    }
                    if push_example(PreparedExample {
                        prompt: prompt.clone(),
                        completion: format!(" {}<|endoftext|>", item.sentence),
                        allow_repeat: true,
                    }) {
                        added += 1;
                    }
                }
                counterexample_count += added;
            }

            let mut identity_attempts = 0usize;
            let max_identity_attempts = n_plain_identity_target.saturating_mul(20).max(1000);
            while identity_count < n_plain_identity_target
                && identity_attempts < max_identity_attempts
            {
                identity_attempts += 1;
                let sentence = &build.train_identity[rng.random_range(0..build.train_identity.len())];
                let prompt = synth_train::build_correction_prompt("", sentence);
                if push_example(PreparedExample {
                    prompt,
                    completion: format!(" {}<|endoftext|>", sentence),
                    allow_repeat: false,
                }) {
                    identity_count += 1;
                }
            }

            examples.shuffle(&mut rng);
            let n = examples.len();
            let n_train = (n as f64 * 0.9) as usize;
            let train = &examples[..n_train];
            let valid = &examples[n_train..];

            std::fs::create_dir_all("training/data").ok();
            synth_train::write_jsonl("training/data/train.jsonl", train).ok();
            synth_train::write_jsonl("training/data/valid.jsonl", valid).ok();
            let applied_eval_json = build
                .eval_mistakes
                .iter()
                .map(|row| serde_json::to_value(row).unwrap_or(serde_json::Value::Null))
                .collect::<Vec<_>>();
            synth_train::write_jsonl("training/applied-eval.jsonl", &applied_eval_json).ok();

            let stats = serde_json::json!({
                "mode": "applied-known-mistake",
                "requested_total": total_examples,
                "error_rate": error_rate,
                "correction_examples": correction_count,
                "identity_examples": identity_count + counterexample_count,
                "plain_identity_examples": identity_count,
                "counterexample_examples": counterexample_count,
                "train_pool_mistakes": build.train_mistakes.len(),
                "eval_pool_mistakes": build.eval_mistakes.len(),
                "train_identity_pool": build.train_identity.len(),
                "train_counterexample_pool": build.train_counterexamples.len(),
                "skipped_missing_term": build.stats.skipped_missing_term,
                "skipped_acceptable_surface": build.stats.skipped_acceptable_surface,
                "skipped_duplicate_rows": build.stats.skipped_duplicate,
                "duplicate_skips": duplicate_skips,
                "conflict_skips": conflict_skips,
                "weighted_repeat_pushes": weighted_repeat_pushes,
                "total": n,
                "train_count": train.len(),
                "valid_count": valid.len(),
                "applied_eval_count": build.eval_mistakes.len(),
            });

            let db = state3.db.lock().unwrap();
            let _ = db.append_job_log(
                job_id,
                &format!(
                    "Done: {} error + {} no-change ({} plain identity, {} counterexample) = {} total ({} train / {} valid)\n\
                     Held-out applied eval: {} rows\n\
                     Source pools: {} train mistakes, {} clean train sentences, {} counterexamples\n\
                     Skipped while building pools: {} missing-term, {} acceptable-surface, {} duplicate-row\n\
                     Skipped while sampling: {} duplicate, {} conflicting prompt\n\
                     Weighted repeats kept: {}",
                    correction_count,
                    identity_count + counterexample_count,
                    identity_count,
                    counterexample_count,
                    n,
                    train.len(),
                    valid.len(),
                    build.eval_mistakes.len(),
                    build.train_mistakes.len(),
                    build.train_identity.len(),
                    build.train_counterexamples.len(),
                    build.stats.skipped_missing_term,
                    build.stats.skipped_acceptable_surface,
                    build.stats.skipped_duplicate,
                    duplicate_skips,
                    conflict_skips,
                    weighted_repeat_pushes,
                ),
            );
            let _ = db.finish_job(job_id, "completed", Some(&stats.to_string()));
        });

        if let Err(e) = handle.await {
            let db = state2.db.lock().unwrap();
            let _ = db.append_job_log(job_id, &format!("ERROR: {e}"));
            let _ = db.finish_job(job_id, "failed", None);
        }
    });

    Ok(Json(serde_json::json!({"job_id": job_id})).into_response())
}

pub async fn api_start_prototype_reranker_prepare_job(
    State(state): State<Arc<AppState>>,
    Json(body): Json<PrototypeRerankerPrepareBody>,
) -> Result<Response, AppError> {
    check_no_running_jobs(&state)?;

    let corpus_limit = body.corpus_limit.unwrap_or(4000).clamp(0, 20_000);
    let human_limit = body.human_limit.unwrap_or(2000).clamp(0, 10_000);
    let max_candidates_per_row = body.max_candidates_per_row.unwrap_or(6).clamp(2, 12);
    let config_json = serde_json::json!({
        "corpus_limit": corpus_limit,
        "human_limit": human_limit,
        "max_candidates_per_row": max_candidates_per_row,
    })
    .to_string();

    let job_id = {
        let db = state.db.lock().unwrap();
        db.create_job("prototype-reranker-prepare", Some(&config_json))
            .map_err(err)?
    };

    let state2 = state.clone();
    tokio::spawn(async move {
        let state3 = state2.clone();
        let handle = tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            let (mut build, vocab, alt_spellings, confusion_forms) = {
                let db = state3.db.lock().unwrap();
                let vocab = db.list_reviewed_vocab().unwrap_or_default();
                let alt_spellings = db.get_all_alt_spellings().unwrap_or_default();
                let confusion_forms = db.get_all_reviewed_confusion_surfaces().unwrap_or_default();
                let build = build_applied_mistake_output(&db, &alt_spellings)?;
                (build, vocab, alt_spellings, confusion_forms)
            };

            build.train_mistakes.sort_by(|a, b| {
                b.weight
                    .cmp(&a.weight)
                    .then_with(|| a.term.cmp(&b.term))
                    .then_with(|| a.corrupted_sentence.cmp(&b.corrupted_sentence))
            });
            build.train_counterexamples.sort_by(|a, b| a.sentence.cmp(&b.sentence));
            build.train_identity.sort();
            if corpus_limit > 0 {
                build.train_mistakes.truncate(corpus_limit);
            }
            let keep_original_limit = human_limit.max(1);
            if build.train_identity.len() > keep_original_limit {
                build.train_identity.truncate(keep_original_limit);
            }
            if build.train_counterexamples.len() > keep_original_limit {
                build.train_counterexamples.truncate(keep_original_limit);
            }

            {
                let db = state3.db.lock().unwrap();
                let _ = db.append_job_log(
                    job_id,
                    &format!(
                        "Preparing prototype reranker dataset from applied task rows: {} known mistakes + {} identity + {} counterexamples",
                        build.train_mistakes.len(),
                        build.train_identity.len(),
                        build.train_counterexamples.len(),
                    ),
                );
            }

            let prototype_config = crate::prototype::PrototypeConfig::default();
            let mut rng = rand::rng();
            let mut source_rows = 0usize;
            let mut rows_with_positive = 0usize;
            let mut rows_without_positive = 0usize;
            let mut mistake_rows = 0usize;
            let mut identity_rows = 0usize;
            let mut counterexample_rows = 0usize;
            let mut mistake_positive_rows = 0usize;
            let mut identity_positive_rows = 0usize;
            let mut counterexample_positive_rows = 0usize;
            let mut positives = 0usize;
            let mut negatives = 0usize;

            let mut train_groups = Vec::<Vec<serde_json::Value>>::new();
            let mut valid_groups = Vec::<Vec<serde_json::Value>>::new();

            let mut push_row =
                |source_name: &str,
                 term: &str,
                 qwen: &str,
                 expected: &str,
                 extra_meta: serde_json::Value| {
                    source_rows += 1;
                    let result = crate::prototype::prototype_correct_with_acoustics(
                        qwen,
                        &vocab,
                        &alt_spellings,
                        &confusion_forms,
                        prototype_config,
                        None,
                    );
                    let mut candidates = select_prototype_reranker_candidates(
                        result.sentence_candidates,
                        max_candidates_per_row,
                    );
                    if source_name == "applied-mistake" && !term.is_empty() {
                        candidates.retain(|candidate| {
                            candidate.label == "original"
                                || normalized_compare_eq(&candidate.text, expected)
                                || (!candidate.edits.is_empty()
                                    && candidate
                                        .edits
                                        .iter()
                                        .all(|edit| edit.to.eq_ignore_ascii_case(term)))
                        });
                    } else if candidates.len() > 3 {
                        let mut compact = Vec::new();
                        if let Some(original) =
                            candidates.iter().find(|candidate| candidate.label == "original")
                        {
                            compact.push(original.clone());
                        }
                        compact.extend(
                            candidates
                                .into_iter()
                                .filter(|candidate| candidate.label != "original")
                                .take(2),
                        );
                        candidates = compact;
                    }
                    let mut row_examples = Vec::new();
                    let mut row_has_positive = false;

                    for candidate in candidates {
                        let full_ok = normalized_compare_eq(&candidate.text, expected);
                        let target_ok = if term.is_empty() {
                            full_ok
                        } else {
                            eval_fragment_matches(&alt_spellings, term, term, &candidate.text)
                        };
                        let edits_supported =
                            candidate_edits_supported_by_expected(&candidate, expected);
                        let positive = full_ok;
                        row_has_positive |= positive;
                        if positive {
                            positives += 1;
                        } else {
                            negatives += 1;
                        }
                        let prompt = build_prototype_reranker_prompt(qwen, &candidate);
                        row_examples.push(serde_json::json!({
                            "prompt": prompt,
                            "completion": if positive { " yes<|endoftext|>" } else { " no<|endoftext|>" },
                            "meta": {
                                "source": source_name,
                                "term": term,
                                "qwen": qwen,
                                "expected": expected,
                                "candidate_text": candidate.text,
                                "candidate_label": candidate.label,
                                "candidate_score": candidate.score,
                                "edits": candidate.edits,
                                "target_ok": target_ok,
                                "full_ok": full_ok,
                                "edits_supported_by_expected": edits_supported,
                                "positive": positive,
                                "task": "applied-known-mistake",
                                "row_meta": extra_meta,
                            }
                        }));
                    }

                    if row_has_positive {
                        rows_with_positive += 1;
                        match source_name {
                            "applied-mistake" => mistake_positive_rows += 1,
                            "identity" => identity_positive_rows += 1,
                            "counterexample" => counterexample_positive_rows += 1,
                            _ => {}
                        }
                    } else {
                        rows_without_positive += 1;
                    }

                    if prototype_reranker_valid_split(term, qwen, expected) {
                        valid_groups.push(row_examples);
                    } else {
                        train_groups.push(row_examples);
                    }
                };

            for item in build.train_mistakes {
                mistake_rows += 1;
                push_row(
                    "applied-mistake",
                    &item.term,
                    &item.corrupted_sentence,
                    &item.clean_sentence,
                    serde_json::json!({
                        "corruption_surface": item.corruption_surface,
                        "corruption_source": item.source,
                        "weight": item.weight,
                    }),
                );
            }

            for sentence in build.train_identity {
                identity_rows += 1;
                push_row(
                    "identity",
                    "",
                    &sentence,
                    &sentence,
                    serde_json::json!({}),
                );
            }

            for item in build.train_counterexamples {
                counterexample_rows += 1;
                push_row(
                    "counterexample",
                    "",
                    &item.sentence,
                    &item.sentence,
                    serde_json::json!({
                        "weight": item.weight,
                    }),
                );
            }

            train_groups.shuffle(&mut rng);
            valid_groups.shuffle(&mut rng);
            let mut train = train_groups.into_iter().flatten().collect::<Vec<_>>();
            let mut valid = valid_groups.into_iter().flatten().collect::<Vec<_>>();
            train.shuffle(&mut rng);
            valid.shuffle(&mut rng);
            let n = train.len() + valid.len();

            std::fs::create_dir_all("training/prototype-reranker").ok();
            synth_train::write_jsonl("training/prototype-reranker/train.jsonl", &train).ok();
            synth_train::write_jsonl("training/prototype-reranker/valid.jsonl", &valid).ok();

            let stats = serde_json::json!({
                "requested_corpus_limit": corpus_limit,
                "requested_human_limit": human_limit,
                "max_candidates_per_row": max_candidates_per_row,
                "task": "applied-known-mistake",
                "source_rows": source_rows,
                "mistake_rows": mistake_rows,
                "identity_rows": identity_rows,
                "counterexample_rows": counterexample_rows,
                "rows_with_positive": rows_with_positive,
                "rows_without_positive": rows_without_positive,
                "mistake_positive_rows": mistake_positive_rows,
                "identity_positive_rows": identity_positive_rows,
                "counterexample_positive_rows": counterexample_positive_rows,
                "positives": positives,
                "negatives": negatives,
                "total": n,
                "train_count": train.len(),
                "valid_count": valid.len(),
                "output_dir": "training/prototype-reranker",
            });

            let db = state3.db.lock().unwrap();
            let _ = db.append_job_log(
                job_id,
                &format!(
                    "Done: applied task reranker export\n\
                     Rows: {} mistakes, {} identity, {} counterexamples\n\
                     Positives by row: {} / {} / {}\n\
                     Rows with no positive candidate: {}\n\
                     Exported {} examples ({} positive / {} negative) to training/prototype-reranker ({} train / {} valid)",
                    mistake_rows,
                    identity_rows,
                    counterexample_rows,
                    mistake_positive_rows,
                    identity_positive_rows,
                    counterexample_positive_rows,
                    rows_without_positive,
                    n,
                    positives,
                    negatives,
                    train.len(),
                    valid.len(),
                ),
            );
            let _ = db.finish_job(job_id, "completed", Some(&stats.to_string()));
            Ok(())
        });

        match handle.await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                let db = state2.db.lock().unwrap();
                let _ = db.append_job_log(job_id, &format!("ERROR: {e}"));
                let _ = db.finish_job(job_id, "failed", None);
            }
            Err(e) => {
                let db = state2.db.lock().unwrap();
                let _ = db.append_job_log(job_id, &format!("ERROR: {e}"));
                let _ = db.finish_job(job_id, "failed", None);
            }
        }
    });

    Ok(Json(serde_json::json!({"job_id": job_id})).into_response())
}

// ==================== Train ====================

pub async fn api_start_train_job(
    State(state): State<Arc<AppState>>,
    Json(body): Json<TrainJobBody>,
) -> Result<Response, AppError> {
    check_no_running_jobs(&state)?;

    let config = synth_train::TrainConfig {
        data: body.data.unwrap_or_else(|| "training/data".into()),
        adapters: body.adapters.unwrap_or_else(|| "training/adapters".into()),
        model: body
            .model
            .unwrap_or_else(|| "mlx-community/Qwen3.5-2B-4bit".into()),
        iters: body.iters.unwrap_or(2000),
        batch_size: body.batch_size.unwrap_or(4),
        num_layers: body.num_layers.unwrap_or(8),
        early_stop_patience: body.patience.unwrap_or(10),
        steps_per_eval: body.steps_per_eval.unwrap_or(500),
        ..Default::default()
    };
    let config_json = serde_json::json!({
        "data": config.data,
        "adapters": config.adapters,
        "model": config.model,
        "iters": config.iters,
        "batch_size": config.batch_size,
        "num_layers": config.num_layers,
        "patience": config.early_stop_patience,
        "steps_per_eval": config.steps_per_eval,
    })
    .to_string();

    let job_id = {
        let db = state.db.lock().unwrap();
        db.create_job("train", Some(&config_json)).map_err(err)?
    };

    state.set_training_exclusive(true);
    state.unload_heavy_models();

    let state2 = state.clone();
    tokio::task::spawn_blocking(move || {
        {
            let db = state2.db.lock().unwrap();
            let _ = db.append_job_log(
                job_id,
                &format!(
                    "Starting training: data={}, adapters={}, model={}, iters={}, batch_size={}, num_layers={}, patience={}, steps_per_eval={}",
                    config.data, config.adapters, config.model, config.iters, config.batch_size, config.num_layers,
                    config.early_stop_patience, config.steps_per_eval
                ),
            );
        }

        let result = synth_train::train_streaming(
            &config,
            || state2.job_cancel.load(Ordering::Relaxed),
            |line| {
                let db = state2.db.lock().unwrap();
                let _ = db.append_job_log(job_id, line);
            },
        );

        let db = state2.db.lock().unwrap();

        // Compute adapter size on disk
        let adapter_size = std::fs::read_dir(&config.adapters)
            .ok()
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .filter_map(|e| e.metadata().ok().map(|m| m.len()))
                    .sum::<u64>()
            })
            .unwrap_or(0);
        let adapter_mb = adapter_size as f64 / (1024.0 * 1024.0);

        // Parse final validation loss from the job log
        let log = db
            .get_job(job_id)
            .ok()
            .flatten()
            .map(|j| j.log)
            .unwrap_or_default();
        let val_loss = log.lines().rev().find_map(|line| {
            // MLX-LM outputs lines like "Val loss 2.345, Val took 1.2s"
            let lower = line.to_lowercase();
            if lower.contains("val") && lower.contains("loss") {
                lower
                    .split_whitespace()
                    .filter_map(|w| w.trim_matches(',').parse::<f64>().ok())
                    .next()
            } else {
                None
            }
        });

        match result {
            Ok(status) if status.success() => {
                let _ = db.append_job_log(
                    job_id,
                    &format!(
                        "Training completed. Adapters: {adapter_mb:.1}MB{}",
                        val_loss
                            .map(|v| format!(", final val loss: {v:.4}"))
                            .unwrap_or_default()
                    ),
                );
                let _ = db.finish_job(
                    job_id,
                    "completed",
                    Some(
                        &serde_json::json!({
                            "exit_code": 0,
                            "adapter_mb": (adapter_mb * 10.0).round() / 10.0,
                            "val_loss": val_loss,
                        })
                        .to_string(),
                    ),
                );
            }
            Ok(status) => {
                let code = status.code().unwrap_or(-1);
                if state2.job_cancel.load(Ordering::Relaxed) {
                    let _ = db.append_job_log(job_id, "Stopped by user.");
                    let _ = db.finish_job(
                        job_id,
                        "stopped",
                        Some(&serde_json::json!({"exit_code": code, "stopped": true}).to_string()),
                    );
                } else {
                    let _ = db.append_job_log(job_id, &format!("Training exited with code {code}"));
                    let _ = db.finish_job(
                        job_id,
                        "failed",
                        Some(&serde_json::json!({"exit_code": code}).to_string()),
                    );
                }
            }
            Err(e) => {
                let _ = db.append_job_log(job_id, &format!("ERROR: {e}"));
                let _ = db.finish_job(job_id, "failed", None);
            }
        }

        state2.set_training_exclusive(false);
        state2.background_work_notify.notify_one();
        state2.authored_asr_notify.notify_one();
    });

    Ok(Json(serde_json::json!({"job_id": job_id})).into_response())
}

// ==================== Vocab Curation (LLM-assisted) ====================

#[derive(Deserialize)]
pub struct CurateBody {
    pub batch_size: Option<usize>,
}

pub async fn api_start_curate_job(
    State(state): State<Arc<AppState>>,
    Json(body): Json<CurateBody>,
) -> Result<Response, AppError> {
    check_no_running_jobs(&state)?;

    let api_key = std::env::var("OPENAI_API_KEY")
        .map_err(|_| err(anyhow::anyhow!("OPENAI_API_KEY not set")))?;
    let batch_size = body.batch_size.unwrap_or(100);

    let job_id = {
        let db = state.db.lock().unwrap();
        db.create_job(
            "curate",
            Some(&serde_json::json!({"batch_size": batch_size}).to_string()),
        )
        .map_err(err)?
    };

    let state2 = state.clone();
    tokio::spawn(async move {
        let result = run_curate_job(&state2, job_id, &api_key, batch_size).await;
        let db = state2.db.lock().unwrap();
        match result {
            Ok((kept, removed)) => {
                let _ = db.append_job_log(
                    job_id,
                    &format!("\n=== DONE ===\n{kept} terms kept, {removed} removed"),
                );
                let _ = db.finish_job(
                    job_id,
                    "completed",
                    Some(&serde_json::json!({"kept": kept, "removed": removed}).to_string()),
                );
            }
            Err(e) => {
                let _ = db.append_job_log(job_id, &format!("ERROR: {e}"));
                let _ = db.finish_job(job_id, "failed", None);
            }
        }
    });

    Ok(Json(serde_json::json!({"job_id": job_id})).into_response())
}

async fn run_curate_job(
    state: &Arc<AppState>,
    job_id: i64,
    api_key: &str,
    batch_size: usize,
) -> anyhow::Result<(usize, usize)> {
    let terms = {
        let db = state.db.lock().unwrap();
        // Only curate terms that haven't been curated yet (idempotent)
        db.uncurated_vocab_terms()?
    };
    let total = terms.len();
    let term_strings: Vec<&str> = terms.iter().map(|(t, _)| t.as_str()).collect();

    {
        let db = state.db.lock().unwrap();
        db.append_job_log(
            job_id,
            &format!("Curating {total} vocab terms in batches of {batch_size} using GPT..."),
        )?;
    }

    let client = reqwest::Client::new();
    let mut kept_total = 0usize;
    let mut removed_total = 0usize;
    let num_batches = (total + batch_size - 1) / batch_size;
    let concurrency = 5;

    // Process batches in groups of `concurrency`
    let batches: Vec<Vec<&str>> = term_strings
        .chunks(batch_size)
        .map(|b| b.to_vec())
        .collect();

    for chunk_start in (0..batches.len()).step_by(concurrency) {
        if state.job_cancel.load(Ordering::Relaxed) {
            let db = state.db.lock().unwrap();
            let _ = db.append_job_log(job_id, "Stopped by user.");
            break;
        }

        let chunk_end = (chunk_start + concurrency).min(batches.len());
        let chunk = &batches[chunk_start..chunk_end];

        {
            let db = state.db.lock().unwrap();
            let _ = db.append_job_log(
                job_id,
                &format!(
                    "[batches {}-{}/{}] Sending {} batches in parallel...",
                    chunk_start + 1,
                    chunk_end,
                    num_batches,
                    chunk.len()
                ),
            );
        }

        // Fire all batches in this chunk concurrently
        let mut handles = Vec::new();
        for (i, batch) in chunk.iter().enumerate() {
            let batch_idx = chunk_start + i;
            let terms_list = batch.join("\n");
            let client = client.clone();
            let api_key = api_key.to_string();

            let handle = tokio::spawn(async move {
                let prompt = format!(
                    "You are helping curate a vocabulary list for an ASR (speech recognition) error correction system. \
                    The vocabulary should contain ONLY terms that a software developer would actually SAY OUT LOUD when \
                    dictating text — proper nouns, library names, acronyms, technical jargon.\n\n\
                    REMOVE:\n\
                    - Compiler flags (-Zmacro-stats, -Zself-profile)\n\
                    - Hex constants (0x0A, 0xFF)\n\
                    - Single letters or very short meaningless tokens (0B, 0O, 0x)\n\
                    - Punctuation-heavy tokens (--doc, .asm)\n\
                    - Version numbers (v0.1.0, 1.2.3)\n\
                    - Common English words that any ASR would handle fine (the, is, and)\n\
                    - Anything nobody would ever say in a sentence\n\n\
                    KEEP:\n\
                    - Library/framework names (serde, Tokio, Axum, React)\n\
                    - Tool names (rustc, cargo, npm, webpack)\n\
                    - Acronyms people say (JSON, HTML, API, SSH, TLS, ASR)\n\
                    - Technical terms (mutex, async, WebSocket, middleware)\n\
                    - Project names, proper nouns\n\n\
                    For each term you KEEP, also suggest how it should be pronounced if it's not obvious \
                    (e.g., \"serde\" → \"sir day\", \"Axum\" → \"axum\"). If pronunciation is obvious, leave it blank.\n\n\
                    Respond with ONLY a JSON array of objects: [{{\"term\": \"...\", \"keep\": true/false, \"pronunciation\": \"...\" or null}}]\n\
                    No markdown, no explanation, just the JSON array.\n\n\
                    Terms to evaluate:\n{terms_list}"
                );

                let resp = client
                    .post("https://api.openai.com/v1/chat/completions")
                    .header("Authorization", format!("Bearer {api_key}"))
                    .json(&serde_json::json!({
                        "model": "gpt-4o-mini",
                        "temperature": 0,
                        "messages": [{"role": "user", "content": prompt}],
                    }))
                    .send()
                    .await;

                let resp = match resp {
                    Ok(r) => r,
                    Err(e) => {
                        return Err(anyhow::anyhow!(
                            "batch {}: request failed: {e}",
                            batch_idx + 1
                        ))
                    }
                };

                if !resp.status().is_success() {
                    let status = resp.status();
                    let body = resp.text().await.unwrap_or_default();
                    return Err(anyhow::anyhow!(
                        "batch {}: API error {status}: {body}",
                        batch_idx + 1
                    ));
                }

                let json: serde_json::Value = resp
                    .json()
                    .await
                    .map_err(|e| anyhow::anyhow!("batch {}: parse failed: {e}", batch_idx + 1))?;

                let content = json["choices"][0]["message"]["content"]
                    .as_str()
                    .unwrap_or("[]");
                let results: Vec<serde_json::Value> = serde_json::from_str(content)
                    .or_else(|_| {
                        let cleaned = content
                            .trim()
                            .strip_prefix("```json")
                            .or_else(|| content.trim().strip_prefix("```"))
                            .unwrap_or(content)
                            .strip_suffix("```")
                            .unwrap_or(content)
                            .trim();
                        serde_json::from_str(cleaned)
                    })
                    .unwrap_or_default();

                Ok(results)
            });
            handles.push(handle);
        }

        // Collect results
        for handle in handles {
            match handle.await {
                Ok(Ok(results)) => {
                    let mut kept = 0usize;
                    let mut removed = 0usize;
                    let db = state.db.lock().unwrap();
                    for item in &results {
                        let term = item["term"].as_str().unwrap_or("");
                        let keep = item["keep"].as_bool().unwrap_or(true);
                        let pronunciation =
                            item["pronunciation"].as_str().filter(|s| !s.is_empty());

                        if keep {
                            kept += 1;
                            let _ = db.set_vocab_curated(term, "kept");
                            if let Some(pron) = pronunciation {
                                if let Ok(Some(vocab)) = db.find_vocab_by_term(term) {
                                    if vocab.spoken_override.is_none() {
                                        let _ = db.update_vocab_override(vocab.id, Some(pron));
                                        let _ = db.append_job_log(
                                            job_id,
                                            &format!("  + {term} → \"{pron}\""),
                                        );
                                    }
                                }
                            }
                        } else {
                            removed += 1;
                            let _ = db.set_vocab_curated(term, "removed");
                        }
                    }
                    let _ = db.append_job_log(job_id, &format!("  kept {kept}, removed {removed}"));
                    kept_total += kept;
                    removed_total += removed;
                }
                Ok(Err(e)) => {
                    let db = state.db.lock().unwrap();
                    let _ = db.append_job_log(job_id, &format!("  ERROR: {e}"));
                }
                Err(e) => {
                    let db = state.db.lock().unwrap();
                    let _ = db.append_job_log(job_id, &format!("  TASK ERROR: {e}"));
                }
            }
        }
    }

    Ok((kept_total, removed_total))
}

// ==================== Vocab Scan ====================

#[derive(Deserialize)]
pub struct VocabScanBody {
    pub tts_backend: Option<String>,
    pub batch_size: Option<usize>,
    pub limit: Option<usize>,
}

pub async fn api_start_vocab_scan(
    State(state): State<Arc<AppState>>,
    Json(body): Json<VocabScanBody>,
) -> Result<Response, AppError> {
    check_no_running_jobs(&state)?;

    let tts_backend = body.tts_backend.unwrap_or_else(|| "pocket-hq".to_string());
    let batch_size = body.batch_size.unwrap_or(50);
    let limit = body.limit.unwrap_or(0);

    let job_id = {
        let db = state.db.lock().unwrap();
        db.create_job("vocab-scan", Some(&serde_json::json!({"tts_backend": tts_backend, "batch_size": batch_size, "limit": limit}).to_string())).map_err(err)?
    };

    let state2 = state.clone();
    tokio::spawn(async move {
        let result = run_vocab_scan(&state2, job_id, &tts_backend, batch_size, limit).await;
        let db = state2.db.lock().unwrap();
        match result {
            Ok((scanned, errors)) => {
                let _ = db.append_job_log(
                    job_id,
                    &format!("\n=== DONE ===\n{scanned} terms scanned, {errors} with ASR errors"),
                );
                let _ = db.finish_job(
                    job_id,
                    "completed",
                    Some(&serde_json::json!({"scanned": scanned, "errors": errors}).to_string()),
                );
            }
            Err(e) => {
                let _ = db.append_job_log(job_id, &format!("ERROR: {e}"));
                let _ = db.finish_job(job_id, "failed", None);
            }
        }
    });

    Ok(Json(serde_json::json!({"job_id": job_id})).into_response())
}

async fn run_vocab_scan(
    state: &Arc<AppState>,
    job_id: i64,
    tts_backend: &str,
    batch_size: usize,
    limit: usize,
) -> anyhow::Result<(usize, usize)> {
    let mut terms = {
        let db = state.db.lock().unwrap();
        db.all_vocab_terms()?
    };
    if limit > 0 && limit < terms.len() {
        terms.truncate(limit);
    }
    let total = terms.len();

    {
        let db = state.db.lock().unwrap();
        db.clear_confusions()?;
        db.append_job_log(
            job_id,
            &format!(
                "Scanning {total} vocab terms in batches of {batch_size}, backend: {tts_backend}"
            ),
        )?;
    }

    let mut scanned = 0usize;
    let mut errors = 0usize;

    // Process in batches
    for (batch_idx, batch) in terms.chunks(batch_size).enumerate() {
        if state.job_cancel.load(Ordering::Relaxed) {
            let db = state.db.lock().unwrap();
            let _ = db.append_job_log(job_id, "Stopped by user.");
            break;
        }
        let batch_terms: Vec<&str> = batch.iter().map(|(t, _)| t.as_str()).collect();
        // Use spoken override if available, otherwise the term itself
        let batch_spoken: Vec<String> = batch
            .iter()
            .map(|(t, ovr)| ovr.as_deref().unwrap_or(t).to_string())
            .collect();

        // Build a comma-separated list for TTS
        let tts_text = batch_spoken.join(", ");

        {
            let db = state.db.lock().unwrap();
            let _ = db.append_job_log(
                job_id,
                &format!(
                    "[batch {}/{}] {} terms: {}...",
                    batch_idx + 1,
                    (total + batch_size - 1) / batch_size,
                    batch_terms.len(),
                    &tts_text[..80.min(tts_text.len())]
                ),
            );
        }

        // TTS the batch
        let audio = match state.tts.generate(tts_backend, &tts_text).await {
            Ok(mut a) => {
                a.normalize();
                a
            }
            Err(e) => {
                let db = state.db.lock().unwrap();
                let _ = db.append_job_log(job_id, &format!("  TTS FAILED: {e}"));
                continue;
            }
        };

        let samples_16k = match tts::resample_to_16k(&audio.samples, audio.sample_rate) {
            Ok(s) => s,
            Err(e) => {
                let db = state.db.lock().unwrap();
                let _ = db.append_job_log(job_id, &format!("  Resample FAILED: {e}"));
                continue;
            }
        };

        // Align to get word boundaries
        let align_items = {
            let state3 = state.clone();
            let samples = samples_16k.clone();
            let text = tts_text.clone();
            tokio::task::spawn_blocking(move || state3.aligner.align(&samples, &text)).await??
        };

        // For each term in the batch, find its aligned segment and run ASR
        let mut term_idx = 0;
        let mut align_idx = 0;

        for (ti, (term, _spoken_override)) in batch.iter().enumerate() {
            let spoken = &batch_spoken[ti];
            let spoken_words: Vec<&str> = spoken.split_whitespace().collect();
            let num_words = spoken_words.len();

            // Consume align_items for this term's words
            if align_idx + num_words > align_items.len() {
                // Not enough alignment items — skip
                let db = state.db.lock().unwrap();
                let _ =
                    db.append_job_log(job_id, &format!("  SKIP '{}' (alignment ran out)", term));
                continue;
            }

            let start_time = align_items[align_idx].start_time;
            let end_time = align_items[align_idx + num_words - 1].end_time;
            align_idx += num_words;

            // Skip the comma separator if present
            if align_idx < align_items.len() {
                let next_word = &align_items[align_idx].word;
                if next_word == "," || next_word.starts_with(',') {
                    align_idx += 1;
                }
            }

            // Crop audio segment (16kHz)
            let start_sample = (start_time * 16000.0).max(0.0) as usize;
            let end_sample = ((end_time + 0.05) * 16000.0).min(samples_16k.len() as f64) as usize;
            if start_sample >= end_sample || end_sample > samples_16k.len() {
                continue;
            }
            let segment: Vec<f32> = samples_16k[start_sample..end_sample].to_vec();

            // Run both ASR models on the cropped segment
            let state_q = state.clone();
            let seg_q = segment.clone();
            let state_p = state.clone();
            let seg_p = segment;

            let qwen_task = tokio::task::spawn_blocking(move || -> String {
                state_q
                    .asr
                    .transcribe_samples(
                        &seg_q,
                        qwen3_asr::TranscribeOptions::default().with_language("english"),
                    )
                    .map(|r| r.text)
                    .unwrap_or_default()
            });

            let parakeet_task = tokio::task::spawn_blocking(move || -> String {
                state_p
                    .parakeet
                    .transcribe_samples(seg_p, 16000, 1, None)
                    .map(|r| r.text)
                    .unwrap_or_default()
            });

            let (qwen, parakeet) = tokio::join!(qwen_task, parakeet_task);
            let qwen = qwen.unwrap_or_default();
            let parakeet = parakeet.unwrap_or_default();

            let term_lower = term.to_lowercase();
            let qwen_match = qwen.trim().to_lowercase() == term_lower;
            let parakeet_match = parakeet.trim().to_lowercase() == term_lower;

            {
                let db = state.db.lock().unwrap();
                let _ = db.insert_confusion(
                    term,
                    qwen.trim(),
                    parakeet.trim(),
                    qwen_match,
                    parakeet_match,
                    tts_backend,
                );
            }

            scanned += 1;
            if !qwen_match || !parakeet_match {
                errors += 1;
                let db = state.db.lock().unwrap();
                let _ = db.append_job_log(
                    job_id,
                    &format!(
                        "  {} → qwen: '{}'{} parakeet: '{}'{}",
                        term,
                        qwen.trim(),
                        if qwen_match { "" } else { " \u{2717}" },
                        parakeet.trim(),
                        if parakeet_match { "" } else { " \u{2717}" },
                    ),
                );
            }
        }
    }

    Ok((scanned, errors))
}

// ==================== Evaluate ====================

#[derive(Deserialize)]
pub struct EvalJobBody {
    pub model: Option<String>,
    pub adapters: Option<String>,
    pub repeats: Option<usize>,
    pub noise_level: Option<f32>,
    pub source: Option<String>,
}

pub async fn api_start_eval_job(
    State(state): State<Arc<AppState>>,
    Json(body): Json<EvalJobBody>,
) -> Result<Response, AppError> {
    check_no_running_jobs(&state)?;

    let config = synth_train::InferenceConfig {
        model: body
            .model
            .unwrap_or_else(|| synth_train::InferenceConfig::default().model),
        adapters: body.adapters.unwrap_or_else(|| "training/adapters".into()),
        ..Default::default()
    };
    let resolved_model = synth_train::resolved_correction_base_model(&config)
        .unwrap_or_else(|_| config.model.clone());
    let source = body.source.unwrap_or_else(|| "applied".to_string());
    let repeats = body.repeats.unwrap_or(1).clamp(1, 32);
    let noise_level = body.noise_level.unwrap_or(0.0).clamp(0.0, 0.5);

    let job_id = {
        let db = state.db.lock().unwrap();
        db.create_job(
            "eval",
            Some(
                &serde_json::json!({
                    "model": resolved_model,
                    "requested_model": config.model,
                    "adapters": config.adapters,
                    "source": source,
                    "repeats": repeats,
                    "noise_level": noise_level
                })
                .to_string(),
            ),
        )
        .map_err(err)?
    };

    let alt_spellings = {
        let db = state.db.lock().unwrap();
        db.get_all_alt_spellings().map_err(err)?
    };

    // Gather eval data up front so we can validate the chosen source before spawning.
    let applied_eval_rows = if source == "applied" {
        match load_applied_eval_rows("training/applied-eval.jsonl") {
            Ok(rows) => rows,
            Err(_) => Vec::new(),
        }
    } else {
        Vec::new()
    };

    let (recordings, synthetic_items, all_texts, overrides) = {
        let db = state.db.lock().unwrap();
        let recordings = db
            .authored_sentence_recordings_for_eval()
            .map_err(err)?
            .into_iter()
            .filter(|rec| {
                let expected_fragment = rec
                    .surface_form
                    .as_deref()
                    .filter(|_| rec.kind == "counterexample")
                    .unwrap_or(&rec.term);
                eval_fragment_matches(&alt_spellings, &rec.term, expected_fragment, &rec.sentence)
            })
            .collect::<Vec<_>>();
        let synthetic_items = db.corpus_eval_set().map_err(err)?;
        let all_texts = db.all_sentence_texts().map_err(err)?;
        let overrides = db.get_spoken_overrides().map_err(err)?;
        (recordings, synthetic_items, all_texts, overrides)
    };

    if source != "human" && source != "synthetic" && source != "applied" {
        let db = state.db.lock().unwrap();
        let _ = db.finish_job(job_id, "failed", None);
        return Ok(Json(serde_json::json!({"job_id": job_id, "error": "Unknown eval source. Use `applied`, `human`, or `synthetic`."})).into_response());
    }

    if source == "human" && recordings.is_empty() {
        let db = state.db.lock().unwrap();
        let _ = db.append_job_log(job_id, "No human recordings to evaluate.");
        let _ = db.finish_job(job_id, "failed", None);
        return Ok(Json(serde_json::json!({"job_id": job_id, "error": "No human recordings. Record some takes in the Author tab first."})).into_response());
    }

    let synthetic_mistakes: Vec<_> = synthetic_items
        .into_iter()
        .filter(|item| item.is_mistake)
        .filter(|item| {
            !eval_fragment_matches(&alt_spellings, &item.term, &item.original, &item.qwen)
        })
        .collect();

    if source == "synthetic" && synthetic_mistakes.is_empty() {
        let db = state.db.lock().unwrap();
        let _ = db.append_job_log(job_id, "No synthetic mistake templates to evaluate.");
        let _ = db.finish_job(job_id, "failed", None);
        return Ok(Json(serde_json::json!({"job_id": job_id, "error": "No known mistakes in corpus_pairs. Generate or review corpus first."})).into_response());
    }

    if source == "applied" && applied_eval_rows.is_empty() {
        let db = state.db.lock().unwrap();
        let _ = db.append_job_log(job_id, "No held-out applied-mistake eval rows.");
        let _ = db.finish_job(job_id, "failed", None);
        return Ok(Json(serde_json::json!({"job_id": job_id, "error": "No applied eval set. Run Prepare first."})).into_response());
    }

    let state2 = state.clone();
    eprintln!(
        "[eval] Job {job_id} created with source={}, {} recordings, {} synthetic templates, {} overrides",
        source,
        recordings.len(),
        synthetic_mistakes.len(),
        overrides.len()
    );
    tokio::spawn(async move {
        // Ensure shared inference server is running before the loop
        {
            let mut guard = state2.inference_server.lock().unwrap();
            let needs_restart = guard
                .as_ref()
                .map(|server| !server.matches(&config))
                .unwrap_or(true);
            if needs_restart {
                if let Some(mut server) = guard.take() {
                    server.kill();
                }
                let db = state2.db.lock().unwrap();
                let _ = db.append_job_log(
                    job_id,
                    &format!(
                        "Starting inference server ({} + {})...",
                        resolved_model, config.adapters
                    ),
                );
                drop(db);
                match synth_train::InferenceServer::start(&config) {
                    Ok(s) => {
                        *guard = Some(s);
                    }
                    Err(e) => {
                        let db = state2.db.lock().unwrap();
                        let _ = db.append_job_log(
                            job_id,
                            &format!("Failed to start inference server: {e}"),
                        );
                        let _ = db.finish_job(job_id, "failed", None);
                        return;
                    }
                }
            } else {
                let db = state2.db.lock().unwrap();
                let _ = db.append_job_log(job_id, "Inference server already running.");
            }
        }

        let total = if source == "synthetic" {
            synthetic_mistakes.len()
        } else if source == "applied" {
            applied_eval_rows.len()
        } else {
            recordings.len()
        };
        let mut correct = 0usize;
        let mut wrong = 0usize;
        let mut alignment_failed_count = 0usize;
        let mut blank = 0usize;
        let mut timeouts = 0usize;
        let mut evaluated = 0usize;
        let mut scorable = 0usize;
        let mut asr_correct = 0usize;
        let mut skipped = 0usize;
        let mut entries: Vec<serde_json::Value> = Vec::new();
        let alt_spellings = alt_spellings;

        {
            let start_msg = if source == "synthetic" {
                format!(
                    "Evaluating {total} synthetic mistake templates x {repeats} variants (fresh Markov sentences, correction-only)..."
                )
            } else if source == "applied" {
                format!(
                    "Evaluating {total} held-out applied known-mistake sentences (clean authored sentence + one known corruption)..."
                )
            } else {
                format!(
                    "Evaluating {total} human recordings x {repeats} repeats (noise {noise_level:.3}, ASR \u{2192} correct, pipelined)..."
                )
            };
            let db = state2.db.lock().unwrap();
            let _ = db.append_job_log(job_id, &start_msg);
        }

        let total_attempts = total * repeats;
        let mut attempt_idx = 0usize;

        if source == "applied" {
            for item in &applied_eval_rows {
                if state2.job_cancel.load(Ordering::Relaxed) {
                    let db = state2.db.lock().unwrap();
                    let _ = db.append_job_log(job_id, "Stopped by user.");
                    break;
                }

                attempt_idx += 1;
                let prompt =
                    synth_train::build_correction_prompt("", &item.corrupted_sentence);
                let infer_started = std::time::Instant::now();
                let result = {
                    let mut guard = state2.inference_server.lock().unwrap();
                    guard.as_mut().unwrap().infer_with_stats(&prompt)
                };
                let infer_total_ms = infer_started.elapsed().as_millis() as u64;
                let infer_stats = result.as_ref().ok().map(|output| output.stats.clone());
                let raw_output = result
                    .as_ref()
                    .ok()
                    .map(|output| output.raw_text.clone())
                    .unwrap_or_default();
                evaluated += 1;

                let asr_was_correct =
                    normalized_compare_eq(&item.corrupted_sentence, &item.clean_sentence);
                if asr_was_correct {
                    asr_correct += 1;
                }

                let (category, corrected_text) = match result {
                    Ok(ref output) if output.text.trim().is_empty() => {
                        blank += 1;
                        ("blank", String::new())
                    }
                    Ok(output) => {
                        let corrected = output.text;
                        if normalized_compare_eq(&corrected, &item.clean_sentence) {
                            correct += 1;
                            if asr_was_correct {
                                ("kept", corrected)
                            } else {
                                ("fixed", corrected)
                            }
                        } else {
                            wrong += 1;
                            if asr_was_correct {
                                ("broken", corrected)
                            } else {
                                ("wrong", corrected)
                            }
                        }
                    }
                    Err(e) => {
                        let msg = e.to_string();
                        timeouts += 1;
                        if msg.contains("timeout") {
                            ("timeout", "(timeout)".into())
                        } else {
                            ("error", format!("(error: {msg})"))
                        }
                    }
                };

                let asr_accuracy = if evaluated > 0 {
                    asr_correct as f64 / evaluated as f64 * 100.0
                } else {
                    0.0
                };
                let post_accuracy = if evaluated > 0 {
                    correct as f64 / evaluated as f64 * 100.0
                } else {
                    0.0
                };

                entries.push(serde_json::json!({
                    "source": "applied",
                    "term": item.term,
                    "source_kind": item.source,
                    "sentence": item.clean_sentence,
                    "qwen": item.corrupted_sentence,
                    "output": corrected_text.trim(),
                    "raw_output": raw_output,
                    "expected": item.clean_sentence,
                    "corruption_surface": item.corruption_surface,
                    "cat": category,
                    "timing": {
                        "infer_total_ms": infer_total_ms,
                        "encode_ms": infer_stats.as_ref().map(|stats| stats.encode_ms),
                        "prefill_ms": infer_stats.as_ref().map(|stats| stats.prefill_ms),
                        "decode_ms": infer_stats.as_ref().map(|stats| stats.decode_ms),
                        "generate_ms": infer_stats.as_ref().map(|stats| stats.generate_ms),
                        "total_ms": infer_stats.as_ref().map(|stats| stats.total_ms).unwrap_or(infer_total_ms),
                        "prompt_tokens": infer_stats.as_ref().map(|stats| stats.prompt_tokens),
                        "output_tokens": infer_stats.as_ref().map(|stats| stats.output_tokens),
                    }
                }));

                let db = state2.db.lock().unwrap();
                let _ = db.update_job_result(
                    job_id,
                    &serde_json::json!({
                        "source": "applied",
                        "total": evaluated,
                        "source_rows": total,
                        "repeats": 1,
                        "noise_level": 0.0,
                        "correct": correct,
                        "wrong": wrong,
                        "blank": blank,
                        "timeouts": timeouts,
                        "asr_correct": asr_correct,
                        "asr_accuracy": (asr_accuracy * 10.0).round() / 10.0,
                        "post_accuracy": (post_accuracy * 10.0).round() / 10.0,
                        "accuracy": (post_accuracy * 10.0).round() / 10.0,
                        "word_accuracy": serde_json::Value::Null,
                        "entries": entries,
                    })
                    .to_string(),
                );
                let _ = db.append_job_log(
                    job_id,
                    &format!(
                        "[{}/{}] [{}] {} [{}]: \"{}\" → \"{}\" (expected \"{}\")",
                        attempt_idx,
                        total_attempts.max(1),
                        category,
                        item.term,
                        item.source,
                        item.corrupted_sentence,
                        corrected_text.trim(),
                        item.clean_sentence
                    ),
                );
            }

            let asr_accuracy = if evaluated > 0 {
                asr_correct as f64 / evaluated as f64 * 100.0
            } else {
                0.0
            };
            let post_accuracy = if evaluated > 0 {
                correct as f64 / evaluated as f64 * 100.0
            } else {
                0.0
            };

            let db = state2.db.lock().unwrap();
            let _ = db.append_job_log(
                job_id,
                &format!(
                    "\n=== RESULTS ({evaluated} held-out applied mistakes) ===\n\
                 Corrupted baseline:           {asr_accuracy:.1}% ({asr_correct}/{evaluated})\n\
                 Post-correction accuracy:     {post_accuracy:.1}% ({correct}/{evaluated})\n\
                 \n\
                 Correct: {correct} | Wrong: {wrong} | Blank: {blank} | Timeouts: {timeouts}"
                ),
            );
            let _ = db.finish_job(
                job_id,
                "completed",
                Some(
                    &serde_json::json!({
                        "source": "applied",
                        "total": evaluated,
                        "source_rows": total,
                        "repeats": 1,
                        "noise_level": 0.0,
                        "correct": correct,
                        "wrong": wrong,
                        "blank": blank,
                        "timeouts": timeouts,
                        "asr_correct": asr_correct,
                        "asr_accuracy": asr_accuracy,
                        "post_accuracy": post_accuracy,
                        "accuracy": post_accuracy,
                        "word_accuracy": serde_json::Value::Null,
                        "entries": entries,
                    })
                    .to_string(),
                ),
            );
            return;
        }

        if source == "synthetic" {
            let chain = MarkovChain::build(&all_texts);
            let mut rng = rand::rngs::StdRng::from_os_rng();

            for item in &synthetic_mistakes {
                if state2.job_cancel.load(Ordering::Relaxed) {
                    let db = state2.db.lock().unwrap();
                    let _ = db.append_job_log(job_id, "Stopped by user.");
                    break;
                }

                for repeat_idx in 0..repeats {
                    attempt_idx += 1;
                    if state2.job_cancel.load(Ordering::Relaxed) {
                        let db = state2.db.lock().unwrap();
                        let _ = db.append_job_log(job_id, "Stopped by user.");
                        break;
                    }

                    let template_sentence = chain.generate_with(&item.term, 15, &mut rng);
                    let full_expected =
                        splice_fragment(&template_sentence, &item.term, &item.original);
                    let full_asr = splice_fragment(&template_sentence, &item.term, &item.qwen);
                    let prompt = synth_train::build_correction_prompt("", &full_asr);
                    let infer_started = std::time::Instant::now();
                    let result = {
                        let mut guard = state2.inference_server.lock().unwrap();
                        guard.as_mut().unwrap().infer_with_stats(&prompt)
                    };
                    let infer_total_ms = infer_started.elapsed().as_millis() as u64;
                    let infer_stats = result.as_ref().ok().map(|output| output.stats.clone());
                    let raw_output = result
                        .as_ref()
                        .ok()
                        .map(|output| output.raw_text.clone())
                        .unwrap_or_default();
                    evaluated += 1;

                    let expected_norm = normalize_eval_fragment(&item.original);
                    let asr_was_correct = eval_fragment_matches(
                        &alt_spellings,
                        &item.term,
                        &item.original,
                        &item.qwen,
                    );
                    if asr_was_correct {
                        asr_correct += 1;
                    }

                    let expected_fragment =
                        extract_synthetic_fragment(&template_sentence, &item.term, &full_expected);
                    let asr_fragment =
                        extract_synthetic_fragment(&template_sentence, &item.term, &full_asr);

                    let (category, _corrected_text, corrected_fragment) = match result {
                        Ok(ref output) if output.text.trim().is_empty() => {
                            blank += 1;
                            ("blank", String::new(), String::new())
                        }
                        Ok(output) => {
                            let corrected = output.text;
                            let corrected_fragment = extract_synthetic_fragment(
                                &template_sentence,
                                &item.term,
                                &corrected,
                            );
                            if eval_fragment_matches(
                                &alt_spellings,
                                &item.term,
                                &item.original,
                                &corrected_fragment,
                            ) {
                                correct += 1;
                                if asr_was_correct {
                                    ("kept", corrected, corrected_fragment)
                                } else {
                                    ("fixed", corrected, corrected_fragment)
                                }
                            } else {
                                wrong += 1;
                                if asr_was_correct {
                                    ("broken", corrected, corrected_fragment)
                                } else {
                                    ("wrong", corrected, corrected_fragment)
                                }
                            }
                        }
                        Err(e) => {
                            timeouts += 1;
                            let msg = e.to_string();
                            if msg.contains("timeout") {
                                ("timeout", "(timeout)".into(), "(timeout)".into())
                            } else {
                                (
                                    "error",
                                    format!("(error: {msg})"),
                                    format!("(error: {msg})"),
                                )
                            }
                        }
                    };

                    let asr_accuracy = if evaluated > 0 {
                        asr_correct as f64 / evaluated as f64 * 100.0
                    } else {
                        0.0
                    };
                    let post_accuracy = if evaluated > 0 {
                        correct as f64 / evaluated as f64 * 100.0
                    } else {
                        0.0
                    };

                    entries.push(serde_json::json!({
                        "source": "synthetic",
                        "term": item.term,
                        "sentence": full_expected,
                        "template_sentence": template_sentence,
                        "variant_no": repeat_idx + 1,
                        "cat": category,
                        "qwen": asr_fragment,
                        "output": corrected_fragment.trim(),
                        "raw_output": raw_output,
                        "expected": expected_fragment,
                        "timing": {
                            "infer_total_ms": infer_total_ms,
                            "encode_ms": infer_stats.as_ref().map(|stats| stats.encode_ms),
                            "prefill_ms": infer_stats.as_ref().map(|stats| stats.prefill_ms),
                            "decode_ms": infer_stats.as_ref().map(|stats| stats.decode_ms),
                            "generate_ms": infer_stats.as_ref().map(|stats| stats.generate_ms),
                            "total_ms": infer_stats.as_ref().map(|stats| stats.total_ms).unwrap_or(infer_total_ms),
                            "prompt_tokens": infer_stats.as_ref().map(|stats| stats.prompt_tokens),
                            "output_tokens": infer_stats.as_ref().map(|stats| stats.output_tokens),
                        }
                    }));

                    let db = state2.db.lock().unwrap();
                    let _ = db.update_job_result(
                        job_id,
                        &serde_json::json!({
                            "source": "synthetic",
                            "total": evaluated,
                            "source_recordings": 0,
                            "source_templates": total,
                            "repeats": repeats,
                            "noise_level": 0.0,
                            "correct": correct, "wrong": wrong, "blank": blank, "timeouts": timeouts,
                            "asr_correct": asr_correct,
                            "asr_accuracy": (asr_accuracy * 10.0).round() / 10.0,
                            "post_accuracy": (post_accuracy * 10.0).round() / 10.0,
                            "entries": entries,
                        })
                        .to_string(),
                    );

                    let _ = db.append_job_log(
                        job_id,
                        &format!(
                            "[{}/{}] [{}] {} (synthetic {}, hit {}): \"{}\" \u{2192} \"{}\"{}",
                            attempt_idx,
                            total_attempts,
                            category,
                            item.term,
                            repeat_idx + 1,
                            item.hit_count,
                            asr_fragment,
                            corrected_fragment.trim(),
                            format!(" (expected \"{}\")", expected_fragment)
                        ),
                    );
                }
            }

            let asr_accuracy = if evaluated > 0 {
                asr_correct as f64 / evaluated as f64 * 100.0
            } else {
                0.0
            };
            let post_accuracy = if evaluated > 0 {
                correct as f64 / evaluated as f64 * 100.0
            } else {
                0.0
            };

            let db = state2.db.lock().unwrap();
            let _ = db.append_job_log(
                job_id,
                &format!(
                    "\n=== RESULTS ({evaluated} synthetic variants from {total} mistake templates) ===\n\
                 ASR baseline (template):      {asr_accuracy:.1}% ({asr_correct}/{evaluated})\n\
                 Post-correction accuracy:     {post_accuracy:.1}% ({correct}/{evaluated})\n\
                 \n\
                 Correct: {correct} | Wrong: {wrong} | Blank: {blank} | Timeouts: {timeouts}"
                ),
            );
            let _ = db.finish_job(
                job_id,
                "completed",
                Some(
                    &serde_json::json!({
                        "source": "synthetic",
                        "total": evaluated,
                        "source_recordings": 0,
                        "source_templates": total,
                        "repeats": repeats,
                        "noise_level": 0.0,
                        "correct": correct, "wrong": wrong, "blank": blank, "timeouts": timeouts,
                        "asr_correct": asr_correct,
                        "asr_accuracy": asr_accuracy,
                        "post_accuracy": post_accuracy,
                        "entries": entries,
                    })
                    .to_string(),
                ),
            );
            return;
        }

        for rec in &recordings {
            if state2.job_cancel.load(Ordering::Relaxed) {
                let db = state2.db.lock().unwrap();
                let _ = db.append_job_log(job_id, "Stopped by user.");
                break;
            }
            let expected_fragment = rec
                .surface_form
                .as_deref()
                .filter(|_| rec.kind == "counterexample")
                .unwrap_or(&rec.term)
                .to_string();

            let wav_bytes = match std::fs::read(&rec.wav_path) {
                Ok(bytes) => bytes,
                Err(e) => {
                    let db = state2.db.lock().unwrap();
                    let _ = db.append_job_log(
                        job_id,
                        &format!(
                            "Missing recording file for {} take {}: {}",
                            rec.term, rec.take_no, e
                        ),
                    );
                    skipped += repeats;
                    continue;
                }
            };

            let (mono, sample_rate) = if rec.wav_path.ends_with(".ogg") {
                match tts::decode_ogg_opus_mono(&wav_bytes) {
                    Ok(v) => v,
                    Err(e) => {
                        let db = state2.db.lock().unwrap();
                        let _ = db.append_job_log(
                            job_id,
                            &format!("Decode failed for {} take {}: {}", rec.term, rec.take_no, e),
                        );
                        skipped += repeats;
                        continue;
                    }
                }
            } else {
                match decode_wav_mono(&wav_bytes) {
                    Ok(v) => v,
                    Err(e) => {
                        let db = state2.db.lock().unwrap();
                        let _ = db.append_job_log(
                            job_id,
                            &format!("Decode failed for {} take {}: {}", rec.term, rec.take_no, e),
                        );
                        skipped += repeats;
                        continue;
                    }
                }
            };

            let clean_16k = match tts::resample_to_16k(&mono, sample_rate) {
                Ok(s) => s,
                Err(e) => {
                    let db = state2.db.lock().unwrap();
                    let _ = db.append_job_log(
                        job_id,
                        &format!(
                            "Resample failed for {} take {}: {}",
                            rec.term, rec.take_no, e
                        ),
                    );
                    skipped += repeats;
                    continue;
                }
            };
            let use_cached_clean_qwen = noise_level == 0.0 && repeats == 1;

            for repeat_idx in 0..repeats {
                attempt_idx += 1;
                if state2.job_cancel.load(Ordering::Relaxed) {
                    let db = state2.db.lock().unwrap();
                    let _ = db.append_job_log(job_id, "Stopped by user.");
                    break;
                }

                let samples = apply_eval_noise(&clean_16k, noise_level);
                let attempt_started = std::time::Instant::now();
                let asr_started = std::time::Instant::now();
                let (asr_qwen, asr_source) = if use_cached_clean_qwen
                    && rec.qwen_clean_model.as_deref() == Some(&state2.qwen_model_key)
                {
                    (rec.qwen_clean.clone().unwrap_or_default(), "cache")
                } else {
                    let state_q = state2.clone();
                    let asr_samples = samples.clone();
                    let qwen = tokio::task::spawn_blocking(move || -> String {
                        state_q
                            .asr
                            .transcribe_samples(
                                &asr_samples,
                                qwen3_asr::TranscribeOptions::default().with_language("english"),
                            )
                            .map(|r| r.text)
                            .unwrap_or_default()
                    })
                    .await
                    .unwrap_or_default();

                    if use_cached_clean_qwen && !qwen.trim().is_empty() {
                        let db = state2.db.lock().unwrap();
                        let _ = db.update_authored_recording_qwen_clean(
                            rec.id,
                            &qwen,
                            &state2.qwen_model_key,
                        );
                    }

                    (qwen, "live")
                };
                let asr_ms = asr_started.elapsed().as_millis() as u64;

                let prompt = synth_train::build_correction_prompt("", &asr_qwen);
                let infer_started = std::time::Instant::now();
                let result = {
                    let mut guard = state2.inference_server.lock().unwrap();
                    guard.as_mut().unwrap().infer_with_stats(&prompt)
                };
                let infer_total_ms = infer_started.elapsed().as_millis() as u64;
                let infer_stats = result.as_ref().ok().map(|output| output.stats.clone());
                let raw_output = result
                    .as_ref()
                    .ok()
                    .map(|output| output.raw_text.clone())
                    .unwrap_or_default();
                evaluated += 1;

                let focus_ms;
                let (category, corrected_text, expected_text, asr_focus_text, focus) = match result
                {
                    Ok(ref output) if output.text.trim().is_empty() => {
                        let focus_started = std::time::Instant::now();
                        let focus = match extract_eval_focus(
                            &state2,
                            &samples,
                            &rec.sentence,
                            &asr_qwen,
                            None,
                            &expected_fragment,
                        ) {
                            Ok(focus) => focus,
                            Err(e) => {
                                let db = state2.db.lock().unwrap();
                                let _ = db.append_job_log(
                                    job_id,
                                    &format!(
                                        "[{}/{}] Focus extraction failed for {} (take {}, run {}): {}",
                                        attempt_idx,
                                        total_attempts,
                                        rec.term,
                                        rec.take_no,
                                        repeat_idx + 1,
                                        e
                                    ),
                                );
                                skipped += 1;
                                evaluated -= 1;
                                continue;
                            }
                        };
                        focus_ms = focus_started.elapsed().as_millis() as u64;
                        let asr_was_correct = eval_fragment_matches(
                            &alt_spellings,
                            &rec.term,
                            &focus.expected,
                            &focus.asr,
                        );
                        if !focus.alignment_failed {
                            scorable += 1;
                        }
                        if !focus.alignment_failed && asr_was_correct {
                            asr_correct += 1;
                        }
                        blank += 1;
                        (
                            "blank",
                            String::new(),
                            focus.expected.clone(),
                            focus.asr.clone(),
                            focus,
                        )
                    }
                    Ok(output) => {
                        let corrected = output.text;
                        let focus_started = std::time::Instant::now();
                        let focus = match extract_eval_focus(
                            &state2,
                            &samples,
                            &rec.sentence,
                            &asr_qwen,
                            Some(&corrected),
                            &expected_fragment,
                        ) {
                            Ok(focus) => focus,
                            Err(e) => {
                                let db = state2.db.lock().unwrap();
                                let _ = db.append_job_log(
                                    job_id,
                                    &format!(
                                        "[{}/{}] Focus extraction failed for {} (take {}, run {}): {}",
                                        attempt_idx,
                                        total_attempts,
                                        rec.term,
                                        rec.take_no,
                                        repeat_idx + 1,
                                        e
                                    ),
                                );
                                skipped += 1;
                                evaluated -= 1;
                                continue;
                            }
                        };
                        focus_ms = focus_started.elapsed().as_millis() as u64;
                        let asr_was_correct = eval_fragment_matches(
                            &alt_spellings,
                            &rec.term,
                            &focus.expected,
                            &focus.asr,
                        );
                        if !focus.alignment_failed {
                            scorable += 1;
                        }
                        if !focus.alignment_failed && asr_was_correct {
                            asr_correct += 1;
                        }
                        if eval_fragment_matches(
                            &alt_spellings,
                            &rec.term,
                            &focus.expected,
                            &focus.corrected,
                        ) {
                            correct += 1;
                            if asr_was_correct {
                                (
                                    "kept",
                                    focus.corrected.clone(),
                                    focus.expected.clone(),
                                    focus.asr.clone(),
                                    focus,
                                )
                            } else {
                                (
                                    "fixed",
                                    focus.corrected.clone(),
                                    focus.expected.clone(),
                                    focus.asr.clone(),
                                    focus,
                                )
                            }
                        } else {
                            if focus.alignment_failed {
                                alignment_failed_count += 1;
                                (
                                    "align_fail",
                                    focus.corrected.clone(),
                                    focus.expected.clone(),
                                    focus.asr.clone(),
                                    focus,
                                )
                            } else {
                                wrong += 1;
                                if asr_was_correct {
                                    (
                                        "broken",
                                        focus.corrected.clone(),
                                        focus.expected.clone(),
                                        focus.asr.clone(),
                                        focus,
                                    )
                                } else {
                                    (
                                        "wrong",
                                        focus.corrected.clone(),
                                        focus.expected.clone(),
                                        focus.asr.clone(),
                                        focus,
                                    )
                                }
                            }
                        }
                    }
                    Err(e) => {
                        let focus_started = std::time::Instant::now();
                        let focus = match extract_eval_focus(
                            &state2,
                            &samples,
                            &rec.sentence,
                            &asr_qwen,
                            None,
                            &expected_fragment,
                        ) {
                            Ok(focus) => focus,
                            Err(extract_err) => {
                                let db = state2.db.lock().unwrap();
                                let _ = db.append_job_log(
                                    job_id,
                                    &format!(
                                        "[{}/{}] Focus extraction failed for {} (take {}, run {}): {}",
                                        attempt_idx,
                                        total_attempts,
                                        rec.term,
                                        rec.take_no,
                                        repeat_idx + 1,
                                        extract_err
                                    ),
                                );
                                skipped += 1;
                                evaluated -= 1;
                                continue;
                            }
                        };
                        focus_ms = focus_started.elapsed().as_millis() as u64;
                        let asr_was_correct = eval_fragment_matches(
                            &alt_spellings,
                            &rec.term,
                            &focus.expected,
                            &focus.asr,
                        );
                        if !focus.alignment_failed {
                            scorable += 1;
                        }
                        if !focus.alignment_failed && asr_was_correct {
                            asr_correct += 1;
                        }
                        let msg = e.to_string();
                        timeouts += 1;
                        if msg.contains("timeout") {
                            (
                                "timeout",
                                "(timeout)".into(),
                                focus.expected.clone(),
                                focus.asr.clone(),
                                focus,
                            )
                        } else {
                            (
                                "error",
                                format!("(error: {msg})"),
                                focus.expected.clone(),
                                focus.asr.clone(),
                                focus,
                            )
                        }
                    }
                };

                let asr_accuracy = if scorable > 0 {
                    asr_correct as f64 / scorable as f64 * 100.0
                } else {
                    0.0
                };
                let post_accuracy = if scorable > 0 {
                    correct as f64 / scorable as f64 * 100.0
                } else {
                    0.0
                };

                entries.push(serde_json::json!({
                    "recording_id": rec.id,
                    "term": rec.term,
                    "kind": rec.kind,
                    "surface_form": rec.surface_form,
                    "sentence": rec.sentence,
                    "take_no": rec.take_no,
                    "run_no": repeat_idx + 1,
                    "cat": category,
                    "qwen": asr_focus_text,
                    "output": corrected_text.trim(),
                    "raw_output": raw_output,
                    "expected": expected_text,
                    "extracted_expected": focus.extracted_expected,
                    "alignment_failed": focus.alignment_failed,
                    "cons_time": [focus.cons_range.0, focus.cons_range.1],
                    "trim_info": focus.trim_info,
                    "trace": focus.trace,
                    "timing": {
                        "asr_source": asr_source,
                        "asr_ms": asr_ms,
                        "infer_total_ms": infer_total_ms,
                        "encode_ms": infer_stats.as_ref().map(|stats| stats.encode_ms),
                        "prefill_ms": infer_stats.as_ref().map(|stats| stats.prefill_ms),
                        "decode_ms": infer_stats.as_ref().map(|stats| stats.decode_ms),
                        "generate_ms": infer_stats.as_ref().map(|stats| stats.generate_ms),
                        "prompt_tokens": infer_stats.as_ref().map(|stats| stats.prompt_tokens),
                        "output_tokens": infer_stats.as_ref().map(|stats| stats.output_tokens),
                        "focus_ms": focus_ms,
                        "total_ms": attempt_started.elapsed().as_millis() as u64,
                    },
                    "alignments": {
                        "expected": focus.expected_alignment,
                        "asr": focus.asr_alignment,
                        "corrected": focus.corrected_alignment,
                    }
                }));

                let db = state2.db.lock().unwrap();
                let _ = db.update_job_result(
                    job_id,
                    &serde_json::json!({
                        "source": "human",
                        "total": evaluated,
                        "scorable": scorable,
                        "source_recordings": total,
                        "repeats": repeats,
                        "noise_level": noise_level,
                        "correct": correct, "wrong": wrong, "alignment_failed": alignment_failed_count, "blank": blank, "timeouts": timeouts,
                        "asr_correct": asr_correct,
                        "asr_accuracy": (asr_accuracy * 10.0).round() / 10.0,
                        "post_accuracy": (post_accuracy * 10.0).round() / 10.0,
                        "entries": entries,
                    })
                    .to_string(),
                );

                let _ = db.append_job_log(
                    job_id,
                    &format!(
                        "[{}/{}] [{}] {} (take {}, run {}): \"{}\" \u{2192} \"{}\"{}",
                        attempt_idx,
                        total_attempts,
                        category,
                        rec.term,
                        rec.take_no,
                        repeat_idx + 1,
                        asr_focus_text,
                        corrected_text.trim(),
                        format!(" (expected \"{}\")", expected_text)
                    ),
                );
            }
        }

        let asr_accuracy = if scorable > 0 {
            asr_correct as f64 / scorable as f64 * 100.0
        } else {
            0.0
        };
        let post_accuracy = if scorable > 0 {
            correct as f64 / scorable as f64 * 100.0
        } else {
            0.0
        };

        let db = state2.db.lock().unwrap();
        let _ = db.append_job_log(
            job_id,
            &format!(
                "\n=== RESULTS ({evaluated} attempts from {total} recordings, {skipped} skipped, {scorable} scored) ===\n\
             ASR accuracy (baseline):     {asr_accuracy:.1}% ({asr_correct}/{scorable})\n\
             Post-correction accuracy:    {post_accuracy:.1}% ({correct}/{scorable})\n\
             \n\
             Correct: {correct} | Wrong: {wrong} | Align-failed: {alignment_failed_count} | Blank: {blank} | Timeouts: {timeouts}"
            ),
        );
        let _ = db.finish_job(
            job_id,
            "completed",
            Some(
                &serde_json::json!({
                    "source": "human",
                    "total": evaluated,
                    "scorable": scorable,
                    "source_recordings": total,
                    "repeats": repeats,
                    "noise_level": noise_level,
                    "correct": correct, "wrong": wrong, "alignment_failed": alignment_failed_count, "blank": blank, "timeouts": timeouts,
                    "asr_correct": asr_correct,
                    "asr_accuracy": asr_accuracy,
                    "post_accuracy": post_accuracy,
                    "entries": entries,
                })
                .to_string(),
            ),
        );
    });

    Ok(Json(serde_json::json!({"job_id": job_id})).into_response())
}

// ==================== Live Correction ====================

#[derive(Deserialize)]
pub struct CorrectionBody {
    pub parakeet: String,
    pub qwen: String,
    pub use_adapters: Option<bool>,
}

#[derive(Deserialize)]
pub struct PrototypeCorrectionBody {
    pub transcript: Option<String>,
    pub qwen: Option<String>,
    pub audio_wav_base64: Option<String>,
    pub max_span_tokens: Option<usize>,
    pub max_span_proposals: Option<usize>,
    pub max_candidates_per_span: Option<usize>,
    pub use_model_reranker: Option<bool>,
    pub use_adapters: Option<bool>,
    pub use_prototype_adapters: Option<bool>,
    pub reranker_mode: Option<String>,
    pub prototype_reranker_train_id: Option<i64>,
}

#[derive(Deserialize)]
pub struct PrototypeBakeoffBody {
    pub source: Option<String>,
    pub limit: Option<usize>,
    pub randomize: Option<bool>,
    pub sample_seed: Option<u64>,
    pub use_model_reranker: Option<bool>,
    pub use_current_adapters: Option<bool>,
    pub use_prototype_adapters: Option<bool>,
    pub reranker_mode: Option<String>,
    pub prototype_reranker_train_id: Option<i64>,
}

#[derive(Deserialize)]
pub struct PrototypeAlignmentDebugBody {
    pub transcript: Option<String>,
    pub qwen: Option<String>,
    pub recording_id: Option<i64>,
    pub audio_wav_base64: Option<String>,
}

#[derive(Deserialize)]
pub struct ZipaTimingDebugBody {
    pub recording_id: Option<i64>,
    pub audio_wav_base64: Option<String>,
    pub repeats: Option<usize>,
}

#[derive(Deserialize)]
pub struct PrototypeAlignmentBenchmarkBody {
    pub recording_ids: Option<Vec<i64>>,
    pub terms: Option<Vec<String>>,
    pub limit: Option<usize>,
}

#[derive(Deserialize)]
pub struct PrototypeBakeoffDetailBody {
    pub source: Option<String>,
    pub recording_id: Option<i64>,
    pub transcript: Option<String>,
    pub qwen: Option<String>,
    pub expected: String,
    pub current: String,
    pub prototype: String,
    pub use_model_reranker: Option<bool>,
    pub use_prototype_adapters: Option<bool>,
    pub reranker_mode: Option<String>,
    pub prototype_reranker_train_id: Option<i64>,
}

#[derive(Clone)]
struct PrototypeBakeoffItem {
    case_id: String,
    term: String,
    qwen: String,
    expected: String,
    hit_count: i64,
    recording_id: Option<i64>,
    wav_path: Option<String>,
    template_sentence: Option<String>,
    qwen_fragment: Option<String>,
    expected_fragment: Option<String>,
}

#[derive(Clone)]
struct AcousticInput {
    transcript_alignment: Vec<qwen3_asr::ForcedAlignItem>,
    transcript_word_confidence: Vec<Option<f32>>,
    timing_source: String,
    zipa_trace: serde_json::Value,
    zipa_segments: Vec<crate::prototype::AcousticSegment>,
    zipa_by_transcript: Vec<Vec<String>>,
    timing: AcousticTiming,
}

#[derive(Clone, Default, Serialize)]
struct AcousticTiming {
    zipa_decode_ms: u64,
    zipa_segment_extract_ms: u64,
    espeak_align_ms: u64,
    parakeet_transcribe_ms: u64,
    parakeet_map_ms: u64,
    fallback_align_ms: u64,
    zipa_group_ms: u64,
    total_ms: u64,
}

#[derive(Clone, Default, Serialize)]
struct ZipaHostTiming {
    temp_wav_write_ms: u64,
    sidecar_start_ms: u64,
    sidecar_roundtrip_ms: u64,
    total_ms: u64,
}

#[derive(Clone, Debug)]
struct TimedWord {
    text: String,
    start: f64,
    end: f64,
    confidence: Option<f32>,
}

#[derive(Clone, Copy, Debug)]
enum TimingMapStep {
    Q1P1,
    Q1P2,
    Q2P1,
    SkipQ,
    SkipP,
}

#[derive(Clone, Copy, Debug)]
enum PhoneAlignStep {
    Match,
    DeleteQ,
    InsertZ,
}

fn normalize_timing_word(token: &str) -> String {
    token
        .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .to_ascii_lowercase()
}

fn compact_timing_word(token: &str) -> String {
    normalize_timing_word(token)
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect()
}

fn tokenize_timing_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(normalize_timing_word)
        .filter(|token| !token.is_empty())
        .collect()
}

fn timing_bigram_dice(a: &str, b: &str) -> f32 {
    if a == b {
        return 1.0;
    }
    if a.len() < 2 || b.len() < 2 {
        return 0.0;
    }
    let a_bigrams = a
        .as_bytes()
        .windows(2)
        .map(|w| (w[0], w[1]))
        .collect::<Vec<_>>();
    let b_bigrams = b
        .as_bytes()
        .windows(2)
        .map(|w| (w[0], w[1]))
        .collect::<HashSet<_>>();
    let overlap = a_bigrams.iter().filter(|g| b_bigrams.contains(g)).count();
    (2 * overlap) as f32 / (a_bigrams.len() + b_bigrams.len()) as f32
}

fn timing_match_cost(q: &str, p: &str) -> f32 {
    if q.is_empty() || p.is_empty() {
        return 1.0;
    }
    if q == p {
        return 0.0;
    }
    let q_compact = compact_timing_word(q);
    let p_compact = compact_timing_word(p);
    if q_compact.is_empty() || p_compact.is_empty() {
        return 1.0;
    }
    if q_compact == p_compact {
        return 0.05;
    }
    if q_compact.starts_with(&p_compact) || p_compact.starts_with(&q_compact) {
        return 0.22;
    }
    let dice = timing_bigram_dice(&q_compact, &p_compact);
    if dice >= 0.85 {
        return 0.18;
    }
    if dice >= 0.65 {
        return 0.35;
    }
    if dice >= 0.45 {
        return 0.58;
    }
    1.0
}

fn split_window(start: f64, end: f64, left_weight: usize, right_weight: usize) -> (f64, f64, f64) {
    let duration = (end - start).max(0.0);
    let total = left_weight.max(1) + right_weight.max(1);
    let pivot = start + duration * (left_weight.max(1) as f64 / total as f64);
    (start, pivot, end)
}

fn combine_confidence(confidences: &[Option<f32>]) -> Option<f32> {
    let values = confidences
        .iter()
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<f32>() / values.len() as f32)
    }
}

fn parakeet_words_to_transcript_timing(
    transcript_text: &str,
    parakeet_words: &[TimedWord],
) -> Option<(Vec<qwen3_asr::ForcedAlignItem>, Vec<Option<f32>>)> {
    let transcript_words = tokenize_timing_words(transcript_text);
    if transcript_words.is_empty() || parakeet_words.is_empty() {
        return None;
    }
    let para_words = parakeet_words
        .iter()
        .map(|w| normalize_timing_word(&w.text))
        .collect::<Vec<_>>();
    let n = transcript_words.len();
    let m = para_words.len();
    let inf = 1e9f32;
    let mut dp = vec![vec![inf; m + 1]; n + 1];
    let mut prev = vec![vec![None; m + 1]; n + 1];
    dp[0][0] = 0.0;
    for i in 0..=n {
        for j in 0..=m {
            let cur = dp[i][j];
            if !cur.is_finite() {
                continue;
            }
            if i < n && j < m {
                let cost = cur + timing_match_cost(&transcript_words[i], &para_words[j]);
                if cost < dp[i + 1][j + 1] {
                    dp[i + 1][j + 1] = cost;
                    prev[i + 1][j + 1] = Some(TimingMapStep::Q1P1);
                }
            }
            if i < n && j + 1 < m {
                let merged = format!("{}{}", para_words[j], para_words[j + 1]);
                let cost = cur + timing_match_cost(&transcript_words[i], &merged) + 0.08;
                if cost < dp[i + 1][j + 2] {
                    dp[i + 1][j + 2] = cost;
                    prev[i + 1][j + 2] = Some(TimingMapStep::Q1P2);
                }
            }
            if i + 1 < n && j < m {
                let merged = format!("{}{}", transcript_words[i], transcript_words[i + 1]);
                let cost = cur + timing_match_cost(&merged, &para_words[j]) + 0.08;
                if cost < dp[i + 2][j + 1] {
                    dp[i + 2][j + 1] = cost;
                    prev[i + 2][j + 1] = Some(TimingMapStep::Q2P1);
                }
            }
            if i < n {
                let cost = cur + 0.85;
                if cost < dp[i + 1][j] {
                    dp[i + 1][j] = cost;
                    prev[i + 1][j] = Some(TimingMapStep::SkipQ);
                }
            }
            if j < m {
                let cost = cur + 0.85;
                if cost < dp[i][j + 1] {
                    dp[i][j + 1] = cost;
                    prev[i][j + 1] = Some(TimingMapStep::SkipP);
                }
            }
        }
    }
    if !dp[n][m].is_finite() {
        return None;
    }
    let mut assigned = vec![None::<(f64, f64)>; n];
    let mut assigned_confidence = vec![None::<f32>; n];
    let mut i = n;
    let mut j = m;
    let mut matched = 0usize;
    while i > 0 || j > 0 {
        let Some(step) = prev[i][j] else { break };
        match step {
            TimingMapStep::Q1P1 => {
                let pw = &parakeet_words[j - 1];
                assigned[i - 1] = Some((pw.start, pw.end));
                assigned_confidence[i - 1] = pw.confidence;
                matched += 1;
                i -= 1;
                j -= 1;
            }
            TimingMapStep::Q1P2 => {
                let p0 = &parakeet_words[j - 2];
                let p1 = &parakeet_words[j - 1];
                assigned[i - 1] = Some((p0.start, p1.end));
                assigned_confidence[i - 1] = combine_confidence(&[p0.confidence, p1.confidence]);
                matched += 1;
                i -= 1;
                j -= 2;
            }
            TimingMapStep::Q2P1 => {
                let pw = &parakeet_words[j - 1];
                let left = compact_timing_word(&transcript_words[i - 2]).len();
                let right = compact_timing_word(&transcript_words[i - 1]).len();
                let (start, pivot, end) = split_window(pw.start, pw.end, left, right);
                assigned[i - 2] = Some((start, pivot));
                assigned[i - 1] = Some((pivot, end));
                assigned_confidence[i - 2] = pw.confidence;
                assigned_confidence[i - 1] = pw.confidence;
                matched += 2;
                i -= 2;
                j -= 1;
            }
            TimingMapStep::SkipQ => {
                i -= 1;
            }
            TimingMapStep::SkipP => {
                j -= 1;
            }
        }
    }
    if matched * 2 < n {
        return None;
    }
    let mut out = Vec::with_capacity(n);
    let mut transcript_confidence = Vec::with_capacity(n);
    let mut last_end = 0.0f64;
    let mut next_assigned = assigned.iter().enumerate().filter_map(|(idx, range)| range.map(|r| (idx, r)));
    let mut next_item = next_assigned.next();
    for (idx, word) in transcript_words.iter().enumerate() {
        let (start, end) = if let Some((s, e)) = assigned[idx] {
            last_end = e;
            (s, e)
        } else {
            let future_start = next_item
                .filter(|(future_idx, _)| *future_idx > idx)
                .map(|(_, (s, _))| s)
                .unwrap_or(last_end + 0.12);
            let start = last_end;
            let end = future_start.max(start + 0.02);
            last_end = end;
            (start, end)
        };
        while let Some((future_idx, _)) = next_item {
            if future_idx <= idx {
                next_item = next_assigned.next();
            } else {
                break;
            }
        }
        out.push(ai(word, start, end));
        transcript_confidence.push(assigned_confidence[idx]);
    }
    Some((out, transcript_confidence))
}

fn zipa_segment_house_tokens(phone: &str) -> Vec<String> {
    let cleaned = phone
        .trim()
        .replace('~', "")
        .replace('ˈ', "")
        .replace('ˌ', "");
    if cleaned.is_empty() || cleaned == "▁" || cleaned == "_" {
        return Vec::new();
    }
    let parsed = crate::prototype::parse_house_ipa(&cleaned);
    if !parsed.is_empty() {
        return parsed;
    }
    vec![cleaned]
}

fn feature_delete_cost(phone: &str) -> u32 {
    match phone {
        "AH" | "ER" => 60,
        "HH" | "Y" | "W" => 75,
        _ => 100,
    }
}

fn feature_insert_cost(phone: &str) -> u32 {
    match phone {
        "AH" | "ER" => 60,
        "HH" | "Y" | "W" => 75,
        _ => 100,
    }
}

fn assigned_ranges_to_alignment(
    transcript_words: &[String],
    assigned: &[Option<(f64, f64)>],
) -> Option<Vec<qwen3_asr::ForcedAlignItem>> {
    if transcript_words.is_empty() || assigned.len() != transcript_words.len() {
        return None;
    }
    let matched = assigned.iter().filter(|r| r.is_some()).count();
    if matched * 2 < transcript_words.len() {
        return None;
    }
    let mut out = Vec::with_capacity(transcript_words.len());
    let mut last_end = 0.0f64;
    let mut next_assigned = assigned
        .iter()
        .enumerate()
        .filter_map(|(idx, range)| range.map(|r| (idx, r)));
    let mut next_item = next_assigned.next();
    for (idx, word) in transcript_words.iter().enumerate() {
        let (start, end) = if let Some((s, e)) = assigned[idx] {
            last_end = e;
            (s, e)
        } else {
            let future_start = next_item
                .filter(|(future_idx, _)| *future_idx > idx)
                .map(|(_, (s, _))| s)
                .unwrap_or(last_end + 0.12);
            let start = last_end;
            let end = future_start.max(start + 0.02);
            last_end = end;
            (start, end)
        };
        while let Some((future_idx, _)) = next_item {
            if future_idx <= idx {
                next_item = next_assigned.next();
            } else {
                break;
            }
        }
        out.push(ai(word, start, end));
    }
    Some(out)
}

fn transcript_ipa_words_to_zipa_timing(
    transcript_words: &[String],
    transcript_ipa_words: &[Vec<String>],
    zipa_segments: &[crate::prototype::AcousticSegment],
) -> Option<Vec<qwen3_asr::ForcedAlignItem>> {
    if transcript_words.is_empty()
        || transcript_words.len() != transcript_ipa_words.len()
        || zipa_segments.is_empty()
    {
        return None;
    }

    let mut q_features = Vec::<(String, usize)>::new();
    for (word_idx, ipa_tokens) in transcript_ipa_words.iter().enumerate() {
        for feature in crate::prototype::house_ipa_to_feature_tokens(ipa_tokens) {
            q_features.push((feature, word_idx));
        }
    }
    let mut z_features = Vec::<(String, usize)>::new();
    for (seg_idx, seg) in zipa_segments.iter().enumerate() {
        let house = zipa_segment_house_tokens(&seg.phone);
        for feature in crate::prototype::house_ipa_to_feature_tokens(&house) {
            z_features.push((feature, seg_idx));
        }
    }
    if q_features.is_empty() || z_features.is_empty() {
        return None;
    }

    let m = q_features.len();
    let n = z_features.len();
    let inf = u32::MAX / 4;
    let mut dp = vec![vec![inf; n + 1]; m + 1];
    let mut prev = vec![vec![None; n + 1]; m + 1];
    dp[0][0] = 0;
    for i in 1..=m {
        let del = feature_delete_cost(&q_features[i - 1].0);
        dp[i][0] = dp[i - 1][0].saturating_add(del);
        prev[i][0] = Some(PhoneAlignStep::DeleteQ);
    }
    for j in 1..=n {
        let ins = feature_insert_cost(&z_features[j - 1].0);
        dp[0][j] = dp[0][j - 1].saturating_add(ins);
        prev[0][j] = Some(PhoneAlignStep::InsertZ);
    }
    for i in 1..=m {
        for j in 1..=n {
            let sub = (synth_corrupt::features::substitution_cost(&q_features[i - 1].0, &z_features[j - 1].0) * 100.0) as u32;
            let mut best = dp[i - 1][j - 1].saturating_add(sub);
            let mut step = PhoneAlignStep::Match;
            let del = dp[i - 1][j].saturating_add(feature_delete_cost(&q_features[i - 1].0));
            if del < best {
                best = del;
                step = PhoneAlignStep::DeleteQ;
            }
            let ins = dp[i][j - 1].saturating_add(feature_insert_cost(&z_features[j - 1].0));
            if ins < best {
                best = ins;
                step = PhoneAlignStep::InsertZ;
            }
            dp[i][j] = best;
            prev[i][j] = Some(step);
        }
    }

    let mut matched_seg_indices = vec![Vec::<usize>::new(); transcript_words.len()];
    let mut i = m;
    let mut j = n;
    while i > 0 || j > 0 {
        let Some(step) = prev[i][j] else { break };
        match step {
            PhoneAlignStep::Match => {
                let word_idx = q_features[i - 1].1;
                let seg_idx = z_features[j - 1].1;
                if !matched_seg_indices[word_idx].contains(&seg_idx) {
                    matched_seg_indices[word_idx].push(seg_idx);
                }
                i -= 1;
                j -= 1;
            }
            PhoneAlignStep::DeleteQ => {
                i -= 1;
            }
            PhoneAlignStep::InsertZ => {
                j -= 1;
            }
        }
    }

    let anchor_ranges = matched_seg_indices
        .iter()
        .map(|indices| {
            Some((
                indices.iter().min().copied()?,
                indices.iter().max().copied()?,
            ))
        })
        .collect::<Vec<_>>();
    let anchored = anchor_ranges
        .iter()
        .enumerate()
        .filter_map(|(idx, range)| range.map(|r| (idx, r)))
        .collect::<Vec<_>>();
    let last_seg_idx = zipa_segments.len().saturating_sub(1);
    let mut assigned = vec![None; transcript_words.len()];
    for (pos, (word_idx, (first, last))) in anchored.iter().copied().enumerate() {
        let own_first = if pos == 0 {
            0
        } else {
            let (_, (_, prev_last)) = anchored[pos - 1];
            ((prev_last + first) / 2) + 1
        };
        let own_last = if pos + 1 >= anchored.len() {
            last_seg_idx
        } else {
            let (_, (next_first, _)) = anchored[pos + 1];
            (last + next_first) / 2
        };
        assigned[word_idx] = Some((
            zipa_segments[own_first.min(last_seg_idx)].start_sec,
            zipa_segments[own_last.min(last_seg_idx)].end_sec,
        ));
    }

    assigned_ranges_to_alignment(transcript_words, &assigned)
}

fn espeak_words_to_transcript_zipa_timing(
    transcript_text: &str,
    zipa_segments: &[crate::prototype::AcousticSegment],
) -> Option<Vec<qwen3_asr::ForcedAlignItem>> {
    let transcript_words = tokenize_timing_words(transcript_text);
    if transcript_words.is_empty() || zipa_segments.is_empty() {
        return None;
    }
    let data_dir = ensure_espeak_bundled_data_dir().ok()?;
    let engine = espeak_ng::EspeakNg::with_data_dir("en", &data_dir).ok()?;
    let transcript_ipa_words = transcript_words
        .iter()
        .map(|word| {
            let ipa = engine.text_to_phonemes(word).ok()?;
            let phones = crate::prototype::parse_house_ipa(&ipa);
            (!phones.is_empty()).then_some(phones)
        })
        .collect::<Option<Vec<_>>>()?;
    transcript_ipa_words_to_zipa_timing(&transcript_words, &transcript_ipa_words, zipa_segments)
}

fn build_acoustic_input_from_samples_16k(
    state: &AppState,
    aligner: &crate::LazyAligner,
    samples_16k: &[f32],
    transcript_text: &str,
) -> Option<AcousticInput> {
    let total_started = std::time::Instant::now();
    let zipa_started = std::time::Instant::now();
    let zipa_trace = decode_zipa_trace_from_16k_warm(state, samples_16k).ok()?;
    let zipa_decode_ms = zipa_started.elapsed().as_millis() as u64;

    let zipa_segments_started = std::time::Instant::now();
    let zipa_segments = zipa_segments_from_trace(&zipa_trace);
    let zipa_segment_extract_ms = zipa_segments_started.elapsed().as_millis() as u64;

    let espeak_started = std::time::Instant::now();
    let transcript_alignment_from_espeak =
        espeak_words_to_transcript_zipa_timing(transcript_text, &zipa_segments);
    let espeak_align_ms = espeak_started.elapsed().as_millis() as u64;

    let parakeet_started = std::time::Instant::now();
    let parakeet_result = state.parakeet.transcribe_samples(
        samples_16k.to_vec(),
        16000,
        1,
        Some(parakeet_rs::TimestampMode::Words),
    );
    let parakeet_transcribe_ms = parakeet_started.elapsed().as_millis() as u64;
    let parakeet_map_started = std::time::Instant::now();
    let parakeet_alignment = parakeet_result.ok().and_then(|result| {
        let words = result
            .tokens
            .into_iter()
            .map(|token| TimedWord {
                text: token.text,
                start: token.start as f64,
                end: token.end as f64,
                confidence: token.confidence,
            })
            .collect::<Vec<_>>();
        parakeet_words_to_transcript_timing(transcript_text, &words)
    });
    let parakeet_map_ms = parakeet_map_started.elapsed().as_millis() as u64;

    let transcript_word_confidence = parakeet_alignment
        .as_ref()
        .map(|(_, confidence)| confidence.clone())
        .unwrap_or_default();

    let fallback_started = std::time::Instant::now();
    let (transcript_alignment, timing_source, fallback_align_ms) =
        if let Some(alignment) = transcript_alignment_from_espeak {
        (alignment, "espeak_zipa_dp".to_string(), 0)
    } else if let Some((alignment, _)) = parakeet_alignment {
        (alignment, "parakeet_words".to_string(), 0)
    } else {
        (
            aligner.align(samples_16k, transcript_text).ok()?,
            "forced_aligner_fallback".to_string(),
            fallback_started.elapsed().as_millis() as u64,
        )
    };

    let zipa_group_started = std::time::Instant::now();
    let zipa_by_transcript = crate::prototype::zipa_grouped_arpabet_by_alignment(
        &transcript_alignment,
        &zipa_segments,
    );
    let zipa_group_ms = zipa_group_started.elapsed().as_millis() as u64;
    Some(AcousticInput {
        transcript_alignment,
        transcript_word_confidence,
        timing_source,
        zipa_trace,
        zipa_segments,
        zipa_by_transcript,
        timing: AcousticTiming {
            zipa_decode_ms,
            zipa_segment_extract_ms,
            espeak_align_ms,
            parakeet_transcribe_ms,
            parakeet_map_ms,
            fallback_align_ms,
            zipa_group_ms,
            total_ms: total_started.elapsed().as_millis() as u64,
        },
    })
}

fn zipa_phone_is_placeholder(phone: &str) -> bool {
    let trimmed = phone.trim();
    trimmed.is_empty()
        || matches!(trimmed, "▁" | "_" | "__" | "∅" | "blank" | "sil" | "silence")
}

fn first_voiced_zipa_start_sec(segments: &[crate::prototype::AcousticSegment]) -> Option<f64> {
    segments
        .iter()
        .find(|seg| !zipa_phone_is_placeholder(&seg.phone))
        .map(|seg| seg.start_sec)
}

fn acoustic_alignments_json(input: &AcousticInput) -> serde_json::Value {
    serde_json::json!({
        "timing_source": input.timing_source.clone(),
        "transcript": fmt_alignment_json(&input.transcript_alignment),
        "espeak": fmt_alignment_json(&input.transcript_alignment),
        "qwen": fmt_alignment_json(&input.transcript_alignment),
        "zipa": phone_segments_to_alignment(&input.zipa_trace),
        "zipa_transcript": crate::prototype::zipa_grouped_by_alignment_json(&input.transcript_alignment, &input.zipa_segments),
        "zipa_espeak": crate::prototype::zipa_grouped_by_alignment_json(&input.transcript_alignment, &input.zipa_segments),
        "zipa_qwen": crate::prototype::zipa_grouped_by_alignment_json(&input.transcript_alignment, &input.zipa_segments),
    })
}

fn transcript_word_debug_json(
    transcript_alignment: &[qwen3_asr::ForcedAlignItem],
    transcript_word_confidence: &[Option<f32>],
    zipa_by_transcript: &[Vec<String>],
) -> serde_json::Value {
    let espeak = ensure_espeak_bundled_data_dir()
        .ok()
        .and_then(|data_dir| espeak_ng::EspeakNg::with_data_dir("en", &data_dir).ok());
    serde_json::Value::Array(
        transcript_alignment
            .iter()
            .enumerate()
            .map(|(idx, item)| {
                let espeak_ipa = espeak
                    .as_ref()
                    .and_then(|engine| engine.text_to_phonemes(&item.word).ok())
                    .unwrap_or_default();
                let grouped = zipa_by_transcript.get(idx).cloned().unwrap_or_default();
                serde_json::json!({
                    "i": idx,
                    "word": item.word,
                    "s": item.start_time,
                    "e": item.end_time,
                    "c": transcript_word_confidence.get(idx).copied().flatten(),
                    "espeak_ipa": espeak_ipa,
                    "zipa_grouped": grouped,
                })
            })
            .collect(),
    )
}

const DEFAULT_ALIGNMENT_BENCHMARK_TERMS: &[&str] = &[
    "bearcove",
    "fasterthanlime",
    "ripgrep",
    "serde_json",
    "mir",
    "u8",
    "aarch64",
    "macho",
    "regalloc",
    "tokio",
];

fn find_term_span_in_tokenized_words(words: &[String], term: &str) -> Option<(usize, usize)> {
    let target = compact_timing_word(term);
    if target.is_empty() {
        return None;
    }
    for start in 0..words.len() {
        let mut joined = String::new();
        for end in start..words.len() {
            joined.push_str(&compact_timing_word(&words[end]));
            if joined == target {
                return Some((start, end + 1));
            }
            if !target.starts_with(&joined) && joined.len() >= target.len() {
                break;
            }
        }
    }
    None
}

fn select_alignment_benchmark_recordings(
    mut recordings: Vec<crate::db::AuthoredSentenceRecordingRow>,
    body: &PrototypeAlignmentBenchmarkBody,
) -> Vec<crate::db::AuthoredSentenceRecordingRow> {
    let limit = body.limit.unwrap_or(10).clamp(1, 50);
    recordings.sort_by_key(|rec| rec.id);

    if let Some(recording_ids) = &body.recording_ids {
        let wanted = recording_ids.iter().copied().collect::<HashSet<_>>();
        return recordings
            .into_iter()
            .filter(|rec| wanted.contains(&rec.id))
            .take(limit)
            .collect();
    }

    let wanted_terms = body
        .terms
        .as_ref()
        .map(|terms| {
            terms.iter()
                .map(|term| term.to_ascii_lowercase())
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| {
            DEFAULT_ALIGNMENT_BENCHMARK_TERMS
                .iter()
                .map(|term| term.to_string())
                .collect::<Vec<_>>()
        });

    let mut wanted_by_id = HashMap::<i64, usize>::new();
    for (idx, wanted) in wanted_terms.iter().enumerate() {
        if let Some(rec) = recordings
            .iter()
            .find(|rec| rec.term.eq_ignore_ascii_case(wanted))
        {
            wanted_by_id.entry(rec.id).or_insert(idx);
        }
    }

    let mut chosen = Vec::new();
    let mut fallback = Vec::new();
    for rec in recordings.into_iter() {
        if let Some(rank) = wanted_by_id.get(&rec.id).copied() {
            chosen.push((rank, rec));
        } else {
            fallback.push(rec);
        }
    }
    chosen.sort_by_key(|(rank, rec)| (*rank, rec.id));

    let mut out = chosen
        .into_iter()
        .map(|(_, rec)| rec)
        .take(limit)
        .collect::<Vec<_>>();
    if out.len() < limit {
        out.extend(fallback.into_iter().take(limit - out.len()));
    }
    out
}

fn zipa_segments_from_trace(trace: &serde_json::Value) -> Vec<crate::prototype::AcousticSegment> {
    trace.get("segments")
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
        .filter_map(|seg| {
            Some(crate::prototype::AcousticSegment {
                phone: seg.get("phone")?.as_str()?.to_string(),
                start_sec: seg.get("start_sec")?.as_f64()?,
                end_sec: seg.get("end_sec")?.as_f64()?,
            })
        })
        .collect()
}

/// Run a single correction inference via a temporary server.
/// For batch work, use the eval job which keeps the server running.
pub async fn api_correct(
    State(state): State<Arc<AppState>>,
    Json(body): Json<CorrectionBody>,
) -> Result<Response, AppError> {
    let state2 = state.clone();
    let result = tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
        let config = synth_train::InferenceConfig {
            attach_adapters: body.use_adapters.unwrap_or(true),
            ..Default::default()
        };
        let mut server_guard = state2.inference_server.lock().unwrap();
        // Start server on first use, reuse for subsequent calls
        if server_guard
            .as_ref()
            .map(|server| !server.matches(&config))
            .unwrap_or(true)
        {
            if let Some(mut server) = server_guard.take() {
                server.kill();
            }
            eprintln!("[correct] Starting inference server...");
            *server_guard = Some(synth_train::InferenceServer::start(&config)?);
            eprintln!("[correct] Inference server ready");
        }
        let server = server_guard.as_mut().unwrap();
        let prompt = synth_train::build_correction_prompt(&body.parakeet, &body.qwen);
        let output = server.infer_with_stats(&prompt)?;
        Ok(serde_json::json!({
            "corrected": output.text,
            "raw_output": output.raw_text,
            "timing": output.stats,
            "use_adapters": config.attach_adapters,
        })
        .to_string())
    })
    .await
    .map_err(|e| err(e))?
    .map_err(err)?;

    let parsed: serde_json::Value = serde_json::from_str(&result).map_err(err)?;
    Ok(Json(parsed).into_response())
}

pub async fn api_correct_prototype(
    State(state): State<Arc<AppState>>,
    Json(body): Json<PrototypeCorrectionBody>,
) -> Result<Response, AppError> {
    let total_started = std::time::Instant::now();
    let db_started = std::time::Instant::now();
    let transcript = body
        .transcript
        .clone()
        .or(body.qwen.clone())
        .unwrap_or_default();
    let (vocab, alt_spellings, confusion_forms) = {
        let db = state.db.lock().unwrap();
        (
            db.list_reviewed_vocab().map_err(err)?,
            db.get_all_alt_spellings().map_err(err)?,
            db.get_all_reviewed_confusion_surfaces().map_err(err)?,
        )
    };
    let db_ms = db_started.elapsed().as_millis() as u64;
    let config = crate::prototype::PrototypeConfig {
        max_span_tokens: body.max_span_tokens.unwrap_or(4).clamp(1, 6),
        max_span_proposals: body.max_span_proposals.unwrap_or(24).clamp(1, 64),
        max_candidates_per_span: body.max_candidates_per_span.unwrap_or(3).clamp(1, 8),
    };
    let audio_prepare_started = std::time::Instant::now();
    let acoustic_input = if let Some(audio_b64) = body.audio_wav_base64.clone() {
        let state2 = state.clone();
        let qwen_text = transcript.clone();
        tokio::task::spawn_blocking(move || -> Option<AcousticInput> {
            use base64::Engine as _;
            let wav = base64::engine::general_purpose::STANDARD
                .decode(audio_b64)
                .ok()?;
            let (mono, sample_rate) = decode_wav_mono(&wav).ok()?;
            let samples_16k = tts::resample_to_16k(&mono, sample_rate).ok()?;
            build_acoustic_input_from_samples_16k(&state2, &state2.aligner, &samples_16k, &qwen_text)
        })
        .await
        .map_err(err)?
    } else {
        None
    };
    let audio_prepare_ms = audio_prepare_started.elapsed().as_millis() as u64;
    let acoustic_context = acoustic_input.as_ref().map(|input| crate::prototype::AcousticContext {
        transcript_alignment: &input.transcript_alignment,
        transcript_word_confidence: &input.transcript_word_confidence,
        zipa_segments: &input.zipa_segments,
        zipa_by_transcript: &input.zipa_by_transcript,
    });
    let prototype_started = std::time::Instant::now();
    let (mut result, prototype_timing) = crate::prototype::prototype_correct_with_acoustics_timed(
        &transcript,
        &vocab,
        &alt_spellings,
        &confusion_forms,
        config,
        acoustic_context.as_ref(),
    );
    let prototype_ms = prototype_started.elapsed().as_millis() as u64;

    let use_model_reranker = body.use_model_reranker.unwrap_or(true);
    let use_prototype_adapters = body.use_prototype_adapters.unwrap_or(false);
    let prototype_reranker_train_id = body.prototype_reranker_train_id;
    let reranker_mode = resolve_prototype_reranker_mode(
        body.reranker_mode.as_deref(),
        use_model_reranker,
        use_prototype_adapters,
    );
    let reranker_started = std::time::Instant::now();
    let reranker = if reranker_mode != PrototypeRerankerMode::Off
        && should_run_prototype_reranker(&result.sentence_candidates)
    {
        let candidates = result
            .sentence_candidates
            .iter()
            .take(8)
            .cloned()
            .collect::<Vec<_>>();
        let reranker_candidates = candidates.clone();
        let state2 = state.clone();
        let rerank_value =
            tokio::task::spawn_blocking(move || -> anyhow::Result<serde_json::Value> {
                match reranker_mode {
                    PrototypeRerankerMode::Off => Ok(serde_json::json!(null)),
                    PrototypeRerankerMode::Trained => run_prototype_reranker(
                        state2.as_ref(),
                        &transcript,
                        &reranker_candidates,
                        true,
                        prototype_reranker_train_id,
                    ),
                    PrototypeRerankerMode::OfficialQwen3 => run_official_qwen3_reranker(
                        state2.as_ref(),
                        &transcript,
                        &reranker_candidates,
                    ),
                }
            })
            .await
            .map_err(err)?
            .map_err(err)?;

        let chosen_index = rerank_value.get("chosen_index").and_then(|v| v.as_u64()).map(|v| v as usize);
        if let Some(idx) = chosen_index {
            if let Some(candidate) = candidates.get(idx) {
                result.corrected = candidate.text.clone();
                result.accepted = candidate.edits.clone();
            }
        }
        Some(rerank_value)
    } else {
        None
    };
    let reranker_ms = reranker_started.elapsed().as_millis() as u64;

    Ok(Json(serde_json::json!({
        "original": result.original,
        "corrected": result.corrected,
        "accepted": result.accepted,
        "proposals": result.proposals,
        "sentence_candidates": result.sentence_candidates,
        "alignments": acoustic_input.as_ref().map(acoustic_alignments_json),
        "zipa_trace": acoustic_input.as_ref().map(|input| input.zipa_trace.clone()),
        "reranker": reranker,
        "timing": {
            "db_ms": db_ms,
            "audio_prepare_ms": audio_prepare_ms,
            "prototype_ms": prototype_ms,
            "prototype": serde_json::to_value(&prototype_timing).unwrap_or(serde_json::json!({})),
            "reranker_ms": reranker_ms,
            "total_ms": total_started.elapsed().as_millis() as u64,
            "acoustic": acoustic_input.as_ref().map(|input| serde_json::to_value(&input.timing).unwrap_or(serde_json::json!({}))),
            "proposal_count": result.proposals.len(),
            "sentence_candidate_count": result.sentence_candidates.len(),
            "accepted_count": result.accepted.len(),
        },
    }))
    .into_response())
}

pub async fn api_correct_prototype_alignment_debug(
    State(state): State<Arc<AppState>>,
    Json(body): Json<PrototypeAlignmentDebugBody>,
) -> Result<Response, AppError> {
    let state2 = state.clone();
    let value = tokio::task::spawn_blocking(move || -> anyhow::Result<serde_json::Value> {
        let total_started = std::time::Instant::now();
        let mut transcript = body.transcript.or(body.qwen).unwrap_or_default();
        let mut transcript_label = if transcript.trim().is_empty() {
            "Parakeet".to_string()
        } else {
            "Provided transcript".to_string()
        };
        let audio_prepare_started = std::time::Instant::now();
        let samples_16k = if let Some(recording_id) = body.recording_id {
            let recording = {
                let db = state2.db.lock().unwrap();
                db.get_authored_sentence_recording(recording_id)?
                    .ok_or_else(|| anyhow::anyhow!("recording not found"))?
            };
            load_authored_recording_16k(&recording.wav_path)?
        } else if let Some(audio_b64) = body.audio_wav_base64 {
            use base64::Engine as _;
            let wav = base64::engine::general_purpose::STANDARD.decode(audio_b64)?;
            let (mono, sample_rate) = decode_wav_mono(&wav)?;
            tts::resample_to_16k(&mono, sample_rate)?
        } else {
            anyhow::bail!("missing recording_id or audio_wav_base64");
        };

        let parakeet_result = state2.parakeet.transcribe_samples(
            samples_16k.clone(),
            16000,
            1,
            Some(parakeet_rs::TimestampMode::Words),
        );
        let parakeet_alignment = parakeet_result
            .as_ref()
            .map(|r| {
                r.tokens
                    .iter()
                    .map(|token| {
                        serde_json::json!({
                            "w": token.text,
                            "s": token.start,
                            "e": token.end,
                            "c": token.confidence,
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let parakeet_text = parakeet_result
            .as_ref()
            .map(|r| r.text.clone())
            .unwrap_or_default();
        if transcript.trim().is_empty() {
            transcript = parakeet_text.clone();
            transcript_label = "Parakeet".to_string();
        }

        let audio_prepare_ms = audio_prepare_started.elapsed().as_millis() as u64;
        let acoustic_input = build_acoustic_input_from_samples_16k(
            &state2,
            &state2.aligner,
            &samples_16k,
            &transcript,
        )
        .ok_or_else(|| anyhow::anyhow!("failed to build acoustic input"))?;

        let transcript_first_start = acoustic_input
            .transcript_alignment
            .first()
            .map(|item| item.start_time);
        let zipa_first_start = acoustic_input.zipa_segments.first().map(|seg| seg.start_sec);
        let zipa_first_voiced_start = first_voiced_zipa_start_sec(&acoustic_input.zipa_segments);
        let auto_offset_ms = match (transcript_first_start, zipa_first_voiced_start) {
            (Some(transcript_start), Some(zipa_start)) => {
                Some(((transcript_start - zipa_start) * 1000.0).round() as i64)
            }
            _ => None,
        };

        Ok(serde_json::json!({
            "ok": true,
            "transcript": transcript,
            "transcript_label": transcript_label,
            "parakeet_transcript": parakeet_text,
            "parakeet_error": parakeet_result.as_ref().err().map(|e| e.to_string()),
            "parakeet_alignment": parakeet_alignment,
            "alignments": acoustic_alignments_json(&acoustic_input),
            "word_debug": transcript_word_debug_json(
                &acoustic_input.transcript_alignment,
                &acoustic_input.transcript_word_confidence,
                &acoustic_input.zipa_by_transcript,
            ),
            "zipa_trace": acoustic_input.zipa_trace.clone(),
            "debug": {
                "timing_source": acoustic_input.timing_source.clone(),
                "transcript_first_start_sec": transcript_first_start,
                "zipa_first_phone_start_sec": zipa_first_start,
                "zipa_first_voiced_start_sec": zipa_first_voiced_start,
                "zipa_auto_offset_ms": auto_offset_ms,
            },
            "timing": {
                "audio_prepare_ms": audio_prepare_ms,
                "total_ms": total_started.elapsed().as_millis() as u64,
                "acoustic": serde_json::to_value(&acoustic_input.timing).unwrap_or(serde_json::json!({})),
            },
        }))
    })
    .await
    .map_err(err)?
    .map_err(err)?;

    Ok(Json(value).into_response())
}

pub async fn api_zipa_timing_debug(
    State(state): State<Arc<AppState>>,
    Json(body): Json<ZipaTimingDebugBody>,
) -> Result<Response, AppError> {
    let state2 = state.clone();
    let value = tokio::task::spawn_blocking(move || -> anyhow::Result<serde_json::Value> {
        let total_started = std::time::Instant::now();
        let samples_16k = if let Some(recording_id) = body.recording_id {
            let recording = {
                let db = state2.db.lock().unwrap();
                db.get_authored_sentence_recording(recording_id)?
                    .ok_or_else(|| anyhow::anyhow!("recording not found"))?
            };
            load_authored_recording_16k(&recording.wav_path)?
        } else if let Some(audio_b64) = body.audio_wav_base64.as_deref() {
            load_inline_audio_16k(audio_b64)?
        } else {
            anyhow::bail!("missing recording_id or audio_wav_base64");
        };

        let repeats = body.repeats.unwrap_or(1).clamp(1, 8);
        let mut runs = Vec::with_capacity(repeats);
        for idx in 0..repeats {
            let run_started = std::time::Instant::now();
            let trace = decode_zipa_trace_from_16k_warm(&state2, &samples_16k)?;
            runs.push(serde_json::json!({
                "run": idx + 1,
                "elapsed_ms": run_started.elapsed().as_millis() as u64,
                "trace": trace,
            }));
        }

        Ok(serde_json::json!({
            "ok": true,
            "kind": "zipa_timing_debug",
            "repeats": repeats,
            "elapsed_ms": total_started.elapsed().as_millis() as u64,
            "runs": runs,
        }))
    })
    .await
    .map_err(err)?
    .map_err(err)?;

    Ok(Json(value).into_response())
}

pub async fn api_correct_prototype_alignment_benchmark(
    State(state): State<Arc<AppState>>,
    Json(body): Json<PrototypeAlignmentBenchmarkBody>,
) -> Result<Response, AppError> {
    let state2 = state.clone();
    let value = tokio::task::spawn_blocking(move || -> anyhow::Result<serde_json::Value> {
        let recordings = {
            let db = state2.db.lock().unwrap();
            db.authored_sentence_recordings_for_eval()?
        };
        let selected = select_alignment_benchmark_recordings(recordings, &body);
        let mut cases = Vec::new();
        let mut words_total = 0usize;
        let mut words_with_grouped = 0usize;
        let mut target_words_total = 0usize;
        let mut target_words_with_grouped = 0usize;
        let mut cases_all_target_grouped = 0usize;
        let mut cases_missing_target_grouping = 0usize;
        let mut timing_sources = HashMap::<String, usize>::new();

        for rec in selected {
            let case_started = std::time::Instant::now();
            let samples_16k = load_authored_recording_16k(&rec.wav_path)?;
            let acoustic_input = build_acoustic_input_from_samples_16k(
                &state2,
                &state2.aligner,
                &samples_16k,
                &rec.sentence,
            )
            .ok_or_else(|| anyhow::anyhow!("failed to build acoustic input for recording {}", rec.id))?;
            let parakeet_result = state2.parakeet.transcribe_samples(
                samples_16k.clone(),
                16000,
                1,
                Some(parakeet_rs::TimestampMode::Words),
            );
            let transcript_words = tokenize_timing_words(&rec.sentence);
            let target_span = find_term_span_in_tokenized_words(&transcript_words, &rec.term);
            let grouped_counts = acoustic_input
                .zipa_by_transcript
                .iter()
                .map(|phones| !phones.is_empty())
                .collect::<Vec<_>>();
            let case_words_total = grouped_counts.len();
            let case_words_with_grouped = grouped_counts.iter().filter(|&&v| v).count();
            words_total += case_words_total;
            words_with_grouped += case_words_with_grouped;

            let (case_target_words_total, case_target_words_with_grouped, target_word_debug) =
                if let Some((start, end)) = target_span {
                    let total = end - start;
                    let with_grouped = grouped_counts[start..end]
                        .iter()
                        .filter(|&&v| v)
                        .count();
                    let debug_rows = transcript_word_debug_json(
                        &acoustic_input.transcript_alignment,
                        &acoustic_input.transcript_word_confidence,
                        &acoustic_input.zipa_by_transcript,
                    )
                    .as_array()
                    .cloned()
                    .unwrap_or_default()
                    .into_iter()
                    .skip(start)
                    .take(total)
                    .collect::<Vec<_>>();
                    (total, with_grouped, serde_json::Value::Array(debug_rows))
                } else {
                    (0usize, 0usize, serde_json::Value::Array(Vec::new()))
                };
            target_words_total += case_target_words_total;
            target_words_with_grouped += case_target_words_with_grouped;
            if case_target_words_total > 0 {
                if case_target_words_total == case_target_words_with_grouped {
                    cases_all_target_grouped += 1;
                } else {
                    cases_missing_target_grouping += 1;
                }
            }

            *timing_sources
                .entry(acoustic_input.timing_source.clone())
                .or_default() += 1;

            let transcript_first_start = acoustic_input
                .transcript_alignment
                .first()
                .map(|item| item.start_time);
            let zipa_first_start = acoustic_input.zipa_segments.first().map(|seg| seg.start_sec);
            let zipa_first_voiced_start = first_voiced_zipa_start_sec(&acoustic_input.zipa_segments);

            cases.push(serde_json::json!({
                "case_id": format!("hum-{}", rec.id),
                "recording_id": rec.id,
                "term": rec.term,
                "sentence": rec.sentence,
                "transcript_label": "Expected sentence",
                "transcript": rec.sentence,
                "parakeet_transcript": parakeet_result.as_ref().map(|r| r.text.clone()).unwrap_or_default(),
                "parakeet_error": parakeet_result.as_ref().err().map(|e| e.to_string()),
                "timing_source": acoustic_input.timing_source,
                "elapsed_ms": case_started.elapsed().as_millis() as u64,
                "words_total": case_words_total,
                "words_with_grouped": case_words_with_grouped,
                "target_span": target_span.map(|(s, e)| serde_json::json!({"start": s, "end": e})),
                "target_words_total": case_target_words_total,
                "target_words_with_grouped": case_target_words_with_grouped,
                "all_target_words_grouped": case_target_words_total > 0 && case_target_words_total == case_target_words_with_grouped,
                "alignments": acoustic_alignments_json(&acoustic_input),
                "target_word_debug": target_word_debug,
                "debug": {
                    "transcript_first_start_sec": transcript_first_start,
                    "zipa_first_phone_start_sec": zipa_first_start,
                    "zipa_first_voiced_start_sec": zipa_first_voiced_start,
                    "zipa_auto_offset_ms": match (transcript_first_start, zipa_first_voiced_start) {
                        (Some(transcript_start), Some(zipa_start)) => Some(((transcript_start - zipa_start) * 1000.0).round() as i64),
                        _ => None,
                    },
                }
            }));
        }

        let selected_count = cases.len();
        Ok(serde_json::json!({
            "ok": true,
            "kind": "alignment_benchmark",
            "selected_cases": selected_count,
            "summary": {
                "cases": selected_count,
                "words_total": words_total,
                "words_with_grouped": words_with_grouped,
                "word_group_coverage": if words_total > 0 { words_with_grouped as f64 / words_total as f64 } else { 0.0 },
                "target_words_total": target_words_total,
                "target_words_with_grouped": target_words_with_grouped,
                "target_group_coverage": if target_words_total > 0 { target_words_with_grouped as f64 / target_words_total as f64 } else { 0.0 },
                "cases_all_target_grouped": cases_all_target_grouped,
                "cases_missing_target_grouping": cases_missing_target_grouping,
                "timing_sources": timing_sources,
            },
            "cases": cases,
        }))
    })
    .await
    .map_err(err)?
    .map_err(err)?;

    Ok(Json(value).into_response())
}

pub async fn api_correct_prototype_bakeoff(
    State(state): State<Arc<AppState>>,
    Json(body): Json<PrototypeBakeoffBody>,
) -> Result<Response, AppError> {
    let limit = body.limit.unwrap_or(150).clamp(1, 250);
    let source = body
        .source
        .as_deref()
        .unwrap_or("human")
        .trim()
        .to_ascii_lowercase();
    let randomize = body.randomize.unwrap_or(source == "human");
    let sample_seed = body.sample_seed.unwrap_or_else(rand::random::<u64>);
    let use_model_reranker = body.use_model_reranker.unwrap_or(true);
    let use_current_adapters = body.use_current_adapters.unwrap_or(true);
    let use_prototype_adapters = body.use_prototype_adapters.unwrap_or(false);
    let prototype_reranker_train_id = body.prototype_reranker_train_id;
    let reranker_mode = resolve_prototype_reranker_mode(
        body.reranker_mode.as_deref(),
        use_model_reranker,
        use_prototype_adapters,
    );
    let prototype_only_eval = true;
    if matches!(source.as_str(), "applied" | "human") {
        check_no_running_jobs(&state)?;
        let job_id = {
            let db = state.db.lock().unwrap();
            db.create_job(
                "prototype-bakeoff",
                Some(
                    &serde_json::json!({
                        "source": source,
                        "limit": limit,
                        "randomize": randomize,
                        "sample_seed": sample_seed,
                        "reranker_mode": body.reranker_mode,
                    })
                    .to_string(),
                ),
            )
            .map_err(err)?
        };
        let state2 = state.clone();
        tokio::spawn(async move {
            let state3 = state2.clone();
            let outcome = tokio::task::spawn_blocking(move || -> anyhow::Result<serde_json::Value> {
                let (mut items, vocab, alt_spellings, confusion_forms) = {
                    let db = state3.db.lock().unwrap();
                    let vocab = db.list_reviewed_vocab()?;
                    let alt_spellings = db.get_all_alt_spellings()?;
                    let confusion_forms = db.get_all_reviewed_confusion_surfaces()?;
                    let items = if source == "applied" {
                        let mut items = load_applied_eval_rows("training/applied-eval.jsonl")?;
                        items.sort_by(|a, b| {
                            a.term
                                .cmp(&b.term)
                                .then_with(|| a.corrupted_sentence.cmp(&b.corrupted_sentence))
                        });
                        items
                            .into_iter()
                            .map(|item| PrototypeBakeoffItem {
                                case_id: prototype_bakeoff_case_id(
                                    "applied",
                                    None,
                                    &item.term,
                                    &item.corrupted_sentence,
                                    &item.clean_sentence,
                                ),
                                term: item.term.clone(),
                                qwen: item.corrupted_sentence,
                                expected: item.clean_sentence,
                                hit_count: 1,
                                recording_id: None,
                                wav_path: None,
                                template_sentence: None,
                                qwen_fragment: Some(item.corruption_surface),
                                expected_fragment: Some(item.term),
                            })
                            .collect::<Vec<_>>()
                    } else {
                        load_human_prototype_items(
                            &state3,
                            &db,
                            &alt_spellings,
                            limit,
                            randomize,
                            sample_seed,
                        )?
                        .into_iter()
                        .map(|mut item| {
                            item.case_id = prototype_bakeoff_case_id(
                                "human",
                                item.recording_id,
                                &item.term,
                                &item.qwen,
                                &item.expected,
                            );
                            item
                        })
                        .collect::<Vec<_>>()
                    };
                    (items, vocab, alt_spellings, confusion_forms)
                };
                items.truncate(limit);

                let total = items.len();
                let prototype_config = crate::prototype::PrototypeConfig::default();
                let mut baseline_ok = 0usize;
                let mut baseline_target_ok = 0usize;
                let mut prototype_ok = 0usize;
                let mut failure_buckets = PrototypeEvalFailureBuckets::default();
                let mut entries = Vec::with_capacity(total);

                for (idx, item) in items.iter().enumerate() {
                    let row_started = std::time::Instant::now();
                    if state3.job_cancel.load(Ordering::Relaxed) {
                        let db = state3.db.lock().unwrap();
                        let _ = db.append_job_log(job_id, "Stopped by user.");
                        break;
                    }

                    let acoustic_input = if source == "human" {
                        item.wav_path.as_deref().and_then(|wav_path| {
                            let samples_16k = load_authored_recording_16k(wav_path).ok()?;
                            build_acoustic_input_from_samples_16k(
                                &state3,
                                &state3.aligner,
                                &samples_16k,
                                &item.qwen,
                            )
                        })
                    } else {
                        None
                    };
                    let acoustic_context = acoustic_input.as_ref().map(|input| crate::prototype::AcousticContext {
                        transcript_alignment: &input.transcript_alignment,
                        transcript_word_confidence: &input.transcript_word_confidence,
                        zipa_segments: &input.zipa_segments,
                        zipa_by_transcript: &input.zipa_by_transcript,
                    });
                    let mut result = crate::prototype::prototype_correct_with_acoustics(
                        &item.qwen,
                        &vocab,
                        &alt_spellings,
                        &confusion_forms,
                        prototype_config,
                        acoustic_context.as_ref(),
                    );
                    let mut reranker_summary = None;
                    if reranker_mode != PrototypeRerankerMode::Off
                        && should_run_prototype_reranker(&result.sentence_candidates)
                    {
                        let candidates = result
                            .sentence_candidates
                            .iter()
                            .take(8)
                            .cloned()
                            .collect::<Vec<_>>();
                        let reranker = match reranker_mode {
                            PrototypeRerankerMode::Off => serde_json::json!(null),
                            PrototypeRerankerMode::Trained => {
                                run_prototype_reranker(
                                    state3.as_ref(),
                                    &item.qwen,
                                    &candidates,
                                    true,
                                    prototype_reranker_train_id,
                                )?
                            }
                            PrototypeRerankerMode::OfficialQwen3 => {
                                run_official_qwen3_reranker(state3.as_ref(), &item.qwen, &candidates)?
                            }
                        };
                        if let Some(chosen) = reranker
                            .get("chosen_index")
                            .and_then(|v| v.as_u64())
                            .map(|v| v as usize)
                        {
                            if let Some(candidate) = candidates.get(chosen) {
                                result.corrected = candidate.text.clone();
                                result.accepted = candidate.edits.clone();
                            }
                        }
                        reranker_summary = Some(reranker);
                    }

                    let prototype_text = result.corrected.clone();
                    let baseline_hit = normalized_compare_eq(&item.qwen, &item.expected);
                    let baseline_target_hit = item
                        .expected_fragment
                        .as_deref()
                        .map(|fragment| eval_fragment_matches(&alt_spellings, &item.term, fragment, &item.qwen))
                        .unwrap_or(false);
                    let prototype_hit = normalized_compare_eq(&prototype_text, &item.expected);
                    let analysis = analyze_prototype_eval_row(
                        &result,
                        &item.term,
                        item.expected_fragment.as_deref(),
                        &item.expected,
                        &alt_spellings,
                    );
                    baseline_ok += usize::from(baseline_hit);
                    baseline_target_ok += usize::from(baseline_target_hit);
                    prototype_ok += usize::from(prototype_hit);
                    failure_buckets.record(analysis.failure_reason);

                    let (expected_fragment_preview, expected_fragment_phonemes) = item
                        .expected_fragment
                        .as_deref()
                        .map(|fragment| vocab_preview(&vocab, fragment))
                        .unwrap_or(("", None));
                    let transcript_label = if source == "human" { "Parakeet" } else { "Input" };
                    entries.push(serde_json::json!({
                        "term": item.term,
                        "case_id": item.case_id,
                        "source": source,
                        "transcript_label": transcript_label,
                        "transcript": item.qwen,
                        "transcript_error": serde_json::Value::Null,
                        "expected": item.expected,
                        "qwen": item.qwen,
                        "recording_id": item.recording_id,
                        "template_sentence": item.template_sentence,
                        "expected_fragment": item.expected_fragment,
                        "expected_fragment_preview": (!expected_fragment_preview.is_empty()).then_some(expected_fragment_preview),
                        "expected_fragment_phonemes": expected_fragment_phonemes,
                        "qwen_fragment": item.qwen_fragment,
                        "qwen_fragment_phonemes": item.qwen_fragment.as_deref().and_then(crate::prototype::phonetic_preview),
                        "hit_count": item.hit_count,
                        "baseline_ok": baseline_hit,
                        "baseline_target_ok": baseline_target_hit,
                        "current": item.qwen,
                        "current_ok": baseline_hit,
                        "current_target_ok": baseline_target_hit,
                        "prototype": prototype_text,
                        "prototype_ok": prototype_hit,
                        "prototype_target_ok": analysis.target_ok,
                        "elapsed_ms": row_started.elapsed().as_millis() as u64,
                        "prototype_accepted": result.accepted.iter().map(|edit| serde_json::json!({
                            "from": edit.from,
                            "to": edit.to,
                            "score": edit.score,
                            "via": edit.via,
                            "acoustic_score": edit.acoustic_score,
                            "acoustic_delta": edit.acoustic_delta,
                            "from_phonemes": crate::prototype::phonetic_preview(&edit.from),
                            "to_phonemes": vocab_preview(&vocab, &edit.to).1,
                        })).collect::<Vec<_>>(),
                        "analysis": {
                            "failure_reason": analysis.failure_reason,
                            "target_proposed": analysis.target_proposed,
                            "target_sentence_candidate": analysis.target_sentence_candidate,
                            "target_accepted_edit": analysis.target_accepted_edit,
                            "exact_ok": analysis.exact_ok,
                            "target_ok": analysis.target_ok,
                        },
                        "prototype_trace_excerpt": prototype_trace_excerpt(&vocab, &result, reranker_summary.as_ref()),
                    }));

                    let snapshot = serde_json::json!({
                        "source": source,
                        "limit": limit,
                        "randomize": source == "human" && randomize,
                        "sample_seed": sample_seed,
                        "prototype_only_eval": true,
                        "processed": idx + 1,
                        "summary": {
                            "n": total,
                            "baseline": baseline_ok,
                            "current": baseline_ok,
                            "prototype": prototype_ok,
                            "prototype_wrong": (idx + 1).saturating_sub(prototype_ok),
                        },
                        "failure_buckets": failure_buckets.to_json(),
                        "target_summary": {
                            "n": total,
                            "baseline": baseline_target_ok,
                            "current": baseline_target_ok,
                            "prototype": entries.iter().filter(|entry| entry["prototype_target_ok"].as_bool().unwrap_or(false)).count(),
                            "prototype_wrong": (idx + 1).saturating_sub(entries.iter().filter(|entry| entry["prototype_target_ok"].as_bool().unwrap_or(false)).count()),
                        },
                        "entries": entries,
                    });
                    let db = state3.db.lock().unwrap();
                    let _ = db.update_job_result(job_id, &snapshot.to_string());
                    let _ = db.append_job_log(
                        job_id,
                        &format!(
                            "[{}/{}] {} {}",
                            idx + 1,
                            total,
                            if prototype_hit { "ok" } else { "wrong" },
                            snapshot["entries"][idx]["term"].as_str().unwrap_or("?")
                        ),
                    );
                }

                Ok(serde_json::json!({
                    "source": source,
                    "limit": limit,
                    "randomize": source == "human" && randomize,
                    "sample_seed": sample_seed,
                    "prototype_only_eval": true,
                    "processed": entries.len(),
                    "summary": {
                        "n": total,
                        "baseline": baseline_ok,
                        "current": baseline_ok,
                        "prototype": prototype_ok,
                        "prototype_wrong": total.saturating_sub(prototype_ok),
                    },
                    "failure_buckets": failure_buckets.to_json(),
                    "target_summary": {
                        "n": total,
                        "baseline": baseline_target_ok,
                        "current": baseline_target_ok,
                        "prototype": entries.iter().filter(|entry| entry["prototype_target_ok"].as_bool().unwrap_or(false)).count(),
                        "prototype_wrong": entries.len().saturating_sub(entries.iter().filter(|entry| entry["prototype_target_ok"].as_bool().unwrap_or(false)).count()),
                    },
                    "entries": entries,
                }))
            })
            .await;

            let db = state2.db.lock().unwrap();
            match outcome {
                Ok(Ok(value)) => {
                    let _ = db.finish_job(job_id, "completed", Some(&value.to_string()));
                }
                Ok(Err(e)) => {
                    let _ = db.append_job_log(job_id, &format!("ERROR: {e}"));
                    let _ = db.finish_job(job_id, "failed", None);
                }
                Err(e) => {
                    let _ = db.append_job_log(job_id, &format!("ERROR: {e}"));
                    let _ = db.finish_job(job_id, "failed", None);
                }
            }
        });

        return Ok(Json(serde_json::json!({"job_id": job_id})).into_response());
    }
    let state2 = state.clone();
    let value = tokio::task::spawn_blocking(move || -> anyhow::Result<serde_json::Value> {
        let (mut items, vocab, alt_spellings, confusion_forms) = {
            let db = state2.db.lock().unwrap();
            let vocab = db.list_reviewed_vocab()?;
            let alt_spellings = db.get_all_alt_spellings()?;
            let confusion_forms = db.get_all_reviewed_confusion_surfaces()?;
            let mut items = if source == "corpus" {
                let mut items = db.corpus_eval_set()?;
                items.retain(|item| item.is_mistake);
                items.sort_by(|a, b| {
                    b.hit_count
                        .cmp(&a.hit_count)
                        .then_with(|| a.term.cmp(&b.term))
                        .then_with(|| a.qwen.cmp(&b.qwen))
                });
                items
                    .into_iter()
                        .map(|item| PrototypeBakeoffItem {
                            case_id: prototype_bakeoff_case_id(
                                "corpus",
                                None,
                                &item.term,
                                &splice_fragment(&item.sentence, &item.term, &item.qwen),
                                &splice_fragment(&item.sentence, &item.term, &item.original),
                            ),
                            term: item.term.clone(),
                            qwen: splice_fragment(&item.sentence, &item.term, &item.qwen),
                            expected: splice_fragment(&item.sentence, &item.term, &item.original),
                        hit_count: item.hit_count,
                        recording_id: None,
                        wav_path: None,
                        template_sentence: Some(item.sentence),
                        qwen_fragment: Some(item.qwen),
                        expected_fragment: Some(item.original),
                    })
                    .collect::<Vec<_>>()
            } else if source == "applied" {
                let mut items = load_applied_eval_rows("training/applied-eval.jsonl")?;
                items.sort_by(|a, b| {
                    a.term
                        .cmp(&b.term)
                        .then_with(|| a.corrupted_sentence.cmp(&b.corrupted_sentence))
                });
                items
                    .into_iter()
                        .map(|item| PrototypeBakeoffItem {
                            case_id: prototype_bakeoff_case_id(
                                "applied",
                                None,
                                &item.term,
                                &item.corrupted_sentence,
                                &item.clean_sentence,
                            ),
                            term: item.term.clone(),
                            qwen: item.corrupted_sentence,
                            expected: item.clean_sentence,
                        hit_count: 1,
                        recording_id: None,
                        wav_path: None,
                        template_sentence: None,
                        qwen_fragment: Some(item.corruption_surface),
                        expected_fragment: Some(item.term),
                    })
                    .collect::<Vec<_>>()
            } else {
                load_human_prototype_items(
                    &state2,
                    &db,
                    &alt_spellings,
                    limit,
                    randomize,
                    sample_seed,
                )?
                    .into_iter()
                    .map(|mut item| {
                        item.case_id = prototype_bakeoff_case_id(
                            "human",
                            item.recording_id,
                            &item.term,
                            &item.qwen,
                            &item.expected,
                        );
                        item
                    })
                    .collect::<Vec<_>>()
            };
            (items, vocab, alt_spellings, confusion_forms)
        };
        items.truncate(limit);

        let current_outputs = if prototype_only_eval {
            items.iter().map(|item| item.qwen.clone()).collect::<Vec<_>>()
        } else {
            let current_config = synth_train::InferenceConfig {
                attach_adapters: use_current_adapters,
                ..Default::default()
            };
            let mut server_guard = state2.inference_server.lock().unwrap();
            if server_guard
                .as_ref()
                .map(|server| !server.matches(&current_config))
                .unwrap_or(true)
            {
                if let Some(mut server) = server_guard.take() {
                    server.kill();
                }
                *server_guard = Some(synth_train::InferenceServer::start(&current_config)?);
            }
            let server = server_guard.as_mut().unwrap();
            let mut outputs = Vec::with_capacity(items.len());
            for item in &items {
                let prompt = synth_train::build_correction_prompt("", &item.qwen);
                let output = server.infer_with_stats(&prompt)?;
                outputs.push(output.text);
            }
            outputs
        };

        let prototype_config = crate::prototype::PrototypeConfig::default();
        let mut prototype_results = Vec::with_capacity(items.len());
        if reranker_mode != PrototypeRerankerMode::Off {
            for item in &items {
                let acoustic_input = if source == "human" {
                    item.wav_path.as_deref().and_then(|wav_path| {
                        let samples_16k = load_authored_recording_16k(wav_path).ok()?;
                        build_acoustic_input_from_samples_16k(
                            &state2,
                            &state2.aligner,
                            &samples_16k,
                            &item.qwen,
                        )
                    })
                } else {
                    None
                };
                let acoustic_context = acoustic_input.as_ref().map(|input| crate::prototype::AcousticContext {
                    transcript_alignment: &input.transcript_alignment,
                    transcript_word_confidence: &input.transcript_word_confidence,
                    zipa_segments: &input.zipa_segments,
                    zipa_by_transcript: &input.zipa_by_transcript,
                });
                let mut result = crate::prototype::prototype_correct_with_acoustics(
                    &item.qwen,
                    &vocab,
                    &alt_spellings,
                    &confusion_forms,
                    prototype_config,
                    acoustic_context.as_ref(),
                );
                let mut reranker_summary = None;
                if should_run_prototype_reranker(&result.sentence_candidates) {
                    let candidates = result
                        .sentence_candidates
                        .iter()
                        .take(8)
                        .cloned()
                        .collect::<Vec<_>>();
                    let reranker = match reranker_mode {
                        PrototypeRerankerMode::Off => serde_json::json!(null),
                        PrototypeRerankerMode::Trained => run_prototype_reranker(
                            state2.as_ref(),
                            &item.qwen,
                            &candidates,
                            true,
                            prototype_reranker_train_id,
                        )?,
                        PrototypeRerankerMode::OfficialQwen3 => run_official_qwen3_reranker(
                            state2.as_ref(),
                            &item.qwen,
                            &candidates,
                        )?,
                    };
                    if let Some(idx) = reranker.get("chosen_index").and_then(|v| v.as_u64()).map(|v| v as usize) {
                        if let Some(candidate) = candidates.get(idx) {
                            result.corrected = candidate.text.clone();
                            result.accepted = candidate.edits.clone();
                        }
                    }
                    reranker_summary = Some(reranker);
                }
                prototype_results.push((result, reranker_summary));
            }
        } else {
            for item in &items {
                let acoustic_input = if source == "human" {
                    item.wav_path.as_deref().and_then(|wav_path| {
                        let samples_16k = load_authored_recording_16k(wav_path).ok()?;
                        build_acoustic_input_from_samples_16k(
                            &state2,
                            &state2.aligner,
                            &samples_16k,
                            &item.qwen,
                        )
                    })
                } else {
                    None
                };
                let acoustic_context = acoustic_input.as_ref().map(|input| crate::prototype::AcousticContext {
                    transcript_alignment: &input.transcript_alignment,
                    transcript_word_confidence: &input.transcript_word_confidence,
                    zipa_segments: &input.zipa_segments,
                    zipa_by_transcript: &input.zipa_by_transcript,
                });
                let result = crate::prototype::prototype_correct_with_acoustics(
                    &item.qwen,
                    &vocab,
                    &alt_spellings,
                    &confusion_forms,
                    prototype_config,
                    acoustic_context.as_ref(),
                );
                prototype_results.push((result, None));
            }
        }

        let mut baseline_ok = 0usize;
        let mut current_ok = 0usize;
        let mut prototype_ok = 0usize;
        let mut prototype_only = 0usize;
        let mut current_only = 0usize;
        let mut both_wrong = 0usize;
        let mut baseline_target_ok = 0usize;
        let mut current_target_ok = 0usize;
        let mut prototype_target_ok = 0usize;
        let mut prototype_only_target = 0usize;
        let mut current_only_target = 0usize;
        let mut both_wrong_target = 0usize;
        let mut failure_buckets = PrototypeEvalFailureBuckets::default();
        let mut entries = Vec::with_capacity(items.len());

        for ((item, current), (prototype, reranker_summary)) in items
            .into_iter()
            .zip(current_outputs.into_iter())
            .zip(prototype_results.into_iter())
        {
            let prototype_text = prototype.corrected.clone();
            let baseline_hit = normalized_compare_eq(&item.qwen, &item.expected);
            let current_hit = normalized_compare_eq(&current, &item.expected);
            let prototype_hit = normalized_compare_eq(&prototype_text, &item.expected);
            let baseline_target_hit = item
                .expected_fragment
                .as_deref()
                .map(|fragment| eval_fragment_matches(&alt_spellings, &item.term, fragment, &item.qwen))
                .unwrap_or(false);
            let current_target_hit = item
                .expected_fragment
                .as_deref()
                .map(|fragment| eval_fragment_matches(&alt_spellings, &item.term, fragment, &current))
                .unwrap_or(false);
            let prototype_target_hit = item
                .expected_fragment
                .as_deref()
                .map(|fragment| {
                    eval_fragment_matches(&alt_spellings, &item.term, fragment, &prototype_text)
                })
                .unwrap_or(false);
            let analysis = analyze_prototype_eval_row(
                &prototype,
                &item.term,
                item.expected_fragment.as_deref(),
                &item.expected,
                &alt_spellings,
            );
            baseline_ok += usize::from(baseline_hit);
            current_ok += usize::from(current_hit);
            prototype_ok += usize::from(prototype_hit);
            prototype_only += usize::from(prototype_hit && !current_hit);
            current_only += usize::from(current_hit && !prototype_hit);
            both_wrong += usize::from(!current_hit && !prototype_hit);
            baseline_target_ok += usize::from(baseline_target_hit);
            current_target_ok += usize::from(current_target_hit);
            prototype_target_ok += usize::from(prototype_target_hit);
            prototype_only_target += usize::from(prototype_target_hit && !current_target_hit);
            current_only_target += usize::from(current_target_hit && !prototype_target_hit);
            both_wrong_target += usize::from(!current_target_hit && !prototype_target_hit);
            failure_buckets.record(analysis.failure_reason);
            let (expected_fragment_preview, expected_fragment_phonemes) = item
                .expected_fragment
                .as_deref()
                .map(|fragment| vocab_preview(&vocab, fragment))
                .unwrap_or(("", None));
            entries.push(serde_json::json!({
                "term": item.term,
                "case_id": item.case_id,
                "source": source,
                "expected": item.expected,
                "qwen": item.qwen,
                "recording_id": item.recording_id,
                "template_sentence": item.template_sentence,
                "expected_fragment": item.expected_fragment,
                "expected_fragment_preview": (!expected_fragment_preview.is_empty()).then_some(expected_fragment_preview),
                "expected_fragment_phonemes": expected_fragment_phonemes,
                "qwen_fragment": item.qwen_fragment,
                "qwen_fragment_phonemes": item.qwen_fragment.as_deref().and_then(crate::prototype::phonetic_preview),
                "hit_count": item.hit_count,
                "baseline_ok": baseline_hit,
                "baseline_target_ok": baseline_target_hit,
                "current": current,
                "current_ok": current_hit,
                "current_target_ok": current_target_hit,
                "prototype": prototype_text,
                "prototype_ok": prototype_hit,
                "prototype_target_ok": prototype_target_hit,
                "analysis": {
                    "failure_reason": analysis.failure_reason,
                    "target_proposed": analysis.target_proposed,
                    "target_sentence_candidate": analysis.target_sentence_candidate,
                    "target_accepted_edit": analysis.target_accepted_edit,
                    "exact_ok": analysis.exact_ok,
                    "target_ok": analysis.target_ok,
                },
                "prototype_accepted": prototype.accepted.iter().map(|edit| serde_json::json!({
                    "from": edit.from,
                    "to": edit.to,
                    "score": edit.score,
                    "via": edit.via,
                    "acoustic_score": edit.acoustic_score,
                    "acoustic_delta": edit.acoustic_delta,
                    "from_phonemes": crate::prototype::phonetic_preview(&edit.from),
                    "to_phonemes": vocab_preview(&vocab, &edit.to).1,
                })).collect::<Vec<_>>(),
                "prototype_trace_excerpt": prototype_trace_excerpt(&vocab, &prototype, reranker_summary.as_ref()),
            }));
        }

        Ok(serde_json::json!({
            "source": source,
            "limit": limit,
            "randomize": source == "human" && randomize,
            "sample_seed": sample_seed,
            "prototype_only_eval": prototype_only_eval,
            "summary": {
                "n": entries.len(),
                "baseline": baseline_ok,
                "current": current_ok,
                "prototype": prototype_ok,
                "prototype_only": prototype_only,
                "current_only": current_only,
                "both_wrong": both_wrong,
                "prototype_wrong": entries.len().saturating_sub(prototype_ok),
            },
            "failure_buckets": failure_buckets.to_json(),
            "target_summary": {
                "n": entries.len(),
                "baseline": baseline_target_ok,
                "current": current_target_ok,
                "prototype": prototype_target_ok,
                "prototype_only": prototype_only_target,
                "current_only": current_only_target,
                "both_wrong": both_wrong_target,
                "prototype_wrong": entries.len().saturating_sub(prototype_target_ok),
            },
            "entries": entries,
        }))
    })
    .await
    .map_err(err)?
    .map_err(err)?;

    Ok(Json(value).into_response())
}

pub async fn api_correct_prototype_bakeoff_detail(
    State(state): State<Arc<AppState>>,
    Json(body): Json<PrototypeBakeoffDetailBody>,
) -> Result<Response, AppError> {
    let source = body
        .source
        .as_deref()
        .unwrap_or("human")
        .trim()
        .to_ascii_lowercase();
    let state2 = state.clone();
    let expected = body.expected;
    let transcript = body.transcript.or(body.qwen).unwrap_or_default();
    let current = body.current;
    let prototype = body.prototype;
    let use_model_reranker = body.use_model_reranker.unwrap_or(true);
    let use_prototype_adapters = body.use_prototype_adapters.unwrap_or(false);
    let prototype_reranker_train_id = body.prototype_reranker_train_id;
    let reranker_mode = resolve_prototype_reranker_mode(
        body.reranker_mode.as_deref(),
        use_model_reranker,
        use_prototype_adapters,
    );
    if source != "human" {
        let value = tokio::task::spawn_blocking(move || -> anyhow::Result<serde_json::Value> {
            let (vocab, alt_spellings, confusion_forms) = {
                let db = state2.db.lock().unwrap();
                (
                    db.list_reviewed_vocab()?,
                    db.get_all_alt_spellings()?,
                    db.get_all_reviewed_confusion_surfaces()?,
                )
            };
            let mut prototype_result = crate::prototype::prototype_correct_with_acoustics(
                &transcript,
                &vocab,
                &alt_spellings,
                &confusion_forms,
                crate::prototype::PrototypeConfig::default(),
                None,
            );
            let reranker = if reranker_mode != PrototypeRerankerMode::Off
                && should_run_prototype_reranker(&prototype_result.sentence_candidates)
            {
                let candidates = prototype_result
                    .sentence_candidates
                    .iter()
                    .take(8)
                    .cloned()
                    .collect::<Vec<_>>();
                let rerank_value = match reranker_mode {
                    PrototypeRerankerMode::Off => serde_json::json!(null),
                    PrototypeRerankerMode::Trained => {
                        run_prototype_reranker(
                            state.as_ref(),
                            &transcript,
                            &candidates,
                            true,
                            prototype_reranker_train_id,
                        )?
                    }
                    PrototypeRerankerMode::OfficialQwen3 => {
                        run_official_qwen3_reranker(state.as_ref(), &transcript, &candidates)?
                    }
                };
                if let Some(idx) = rerank_value
                    .get("chosen_index")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
                {
                    if let Some(candidate) = candidates.get(idx) {
                        prototype_result.corrected = candidate.text.clone();
                        prototype_result.accepted = candidate.edits.clone();
                    }
                }
                Some(rerank_value)
            } else {
                None
            };
            Ok(serde_json::json!({
                "ok": true,
                "alignments": {
                    "expected": [],
                    "transcript": [],
                    "espeak": [],
                    "qwen": [],
                    "current": [],
                    "prototype": [],
                    "zipa": [],
                    "zipa_transcript": [],
                    "zipa_espeak": [],
                    "zipa_qwen": [],
                },
                "prototype_trace": {
                    "corrected": prototype_result.corrected,
                    "accepted": prototype_result.accepted,
                    "proposals": prototype_result.proposals,
                    "sentence_candidates": prototype_result.sentence_candidates,
                    "reranker": reranker,
                },
            }))
        })
        .await
        .map_err(err)?
        .map_err(err)?;

        return Ok(Json(value).into_response());
    }
    let Some(recording_id) = body.recording_id else {
        return Ok(Json(serde_json::json!({
            "ok": false,
            "error": "Missing recording_id for human bakeoff detail."
        }))
        .into_response());
    };

    let value = tokio::task::spawn_blocking(move || -> anyhow::Result<serde_json::Value> {
        let detail_started = std::time::Instant::now();
        let db_started = std::time::Instant::now();
        let (recording, vocab, alt_spellings, confusion_forms) = {
            let db = state2.db.lock().unwrap();
            let recording = db
                .get_authored_sentence_recording(recording_id)?
                .ok_or_else(|| anyhow::anyhow!("recording not found"))?;
            let vocab = db.list_reviewed_vocab()?;
            let alt_spellings = db.get_all_alt_spellings()?;
            let confusion_forms = db.get_all_reviewed_confusion_surfaces()?;
            (recording, vocab, alt_spellings, confusion_forms)
        };
        let db_ms = db_started.elapsed().as_millis() as u64;
        let audio_prepare_started = std::time::Instant::now();
        let samples_16k = load_authored_recording_16k(&recording.wav_path)?;
        let parakeet_result = state2.parakeet.transcribe_samples(
            samples_16k.clone(),
            16000,
            1,
            Some(parakeet_rs::TimestampMode::Words),
        );
        let parakeet = parakeet_result
            .as_ref()
            .map(|r| r.text.clone())
            .unwrap_or_default();
        let parakeet_alignment_words = parakeet_result
            .as_ref()
            .map(|r| {
                r.tokens
                    .iter()
                    .map(|token| {
                        serde_json::json!({
                            "w": token.text,
                            "s": token.start,
                            "e": token.end,
                            "c": token.confidence,
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let expected_alignment = state2.aligner.align(&samples_16k, &expected)?;
        let acoustic_input = build_acoustic_input_from_samples_16k(
            &state2,
            &state2.aligner,
            &samples_16k,
            &parakeet,
        );
        let transcript_alignment = acoustic_input
            .as_ref()
            .map(|input| input.transcript_alignment.clone())
            .unwrap_or_else(|| state2.aligner.align(&samples_16k, &parakeet).unwrap_or_default());
        let current_alignment = state2.aligner.align(&samples_16k, &current)?;
        let prototype_alignment = state2.aligner.align(&samples_16k, &prototype)?;
        let zipa_trace = acoustic_input.as_ref().map(|input| input.zipa_trace.clone());
        let zipa_segments = acoustic_input
            .as_ref()
            .map(|input| input.zipa_segments.clone())
            .unwrap_or_default();
        let zipa_by_transcript = acoustic_input
            .as_ref()
            .map(|input| input.zipa_by_transcript.clone())
            .unwrap_or_else(|| crate::prototype::zipa_grouped_arpabet_by_alignment(&transcript_alignment, &zipa_segments));
        let transcript_word_confidence = acoustic_input
            .as_ref()
            .map(|input| input.transcript_word_confidence.clone())
            .unwrap_or_else(|| vec![None; transcript_alignment.len()]);
        let audio_prepare_ms = audio_prepare_started.elapsed().as_millis() as u64;
        let acoustic_context = crate::prototype::AcousticContext {
            transcript_alignment: &transcript_alignment,
            transcript_word_confidence: &transcript_word_confidence,
            zipa_segments: &zipa_segments,
            zipa_by_transcript: &zipa_by_transcript,
        };
        let prototype_started = std::time::Instant::now();
        let (mut prototype_result, prototype_timing) =
            crate::prototype::prototype_correct_with_acoustics_timed(
            &parakeet,
            &vocab,
            &alt_spellings,
            &confusion_forms,
            crate::prototype::PrototypeConfig::default(),
            Some(&acoustic_context),
        );
        let prototype_ms = prototype_started.elapsed().as_millis() as u64;
        let reranker_started = std::time::Instant::now();
        let reranker = if reranker_mode != PrototypeRerankerMode::Off
            && should_run_prototype_reranker(&prototype_result.sentence_candidates)
        {
            let candidates = prototype_result
                .sentence_candidates
                .iter()
                .take(8)
                .cloned()
                .collect::<Vec<_>>();
            let rerank_value = match reranker_mode {
                PrototypeRerankerMode::Off => serde_json::json!(null),
                PrototypeRerankerMode::Trained => {
                    run_prototype_reranker(
                        state.as_ref(),
                        &parakeet,
                        &candidates,
                        true,
                        prototype_reranker_train_id,
                    )?
                }
                PrototypeRerankerMode::OfficialQwen3 => {
                    run_official_qwen3_reranker(state.as_ref(), &parakeet, &candidates)?
                }
            };
            let chosen_index = rerank_value.get("chosen_index").and_then(|v| v.as_u64()).map(|v| v as usize);
            if let Some(idx) = chosen_index {
                if let Some(candidate) = candidates.get(idx) {
                    prototype_result.corrected = candidate.text.clone();
                    prototype_result.accepted = candidate.edits.clone();
                }
            }
            Some(rerank_value)
        } else {
            None
        };
        let reranker_ms = reranker_started.elapsed().as_millis() as u64;
        let zipa_alignment = zipa_trace
            .as_ref()
            .map(phone_segments_to_alignment)
            .unwrap_or_else(|| serde_json::json!([]));
        let parakeet_error = parakeet_result.as_ref().err().map(|e| e.to_string());
        Ok(serde_json::json!({
            "ok": true,
            "recording_id": recording_id,
            "transcript_label": "Parakeet",
            "transcript": parakeet,
            "transcript_error": parakeet_error,
            "current": parakeet,
            "qwen": parakeet,
            "parakeet": parakeet,
            "cohere": "",
            "parakeet_alignment": parakeet_alignment_words,
            "qwen_error": serde_json::Value::Null,
            "parakeet_error": parakeet_error,
            "cohere_error": serde_json::Value::Null,
            "correction_input": "parakeet",
            "elapsed_ms": detail_started.elapsed().as_millis() as u64,
            "timing": {
                "db_ms": db_ms,
                "audio_prepare_ms": audio_prepare_ms,
                "prototype_ms": prototype_ms,
                "prototype": serde_json::to_value(&prototype_timing).unwrap_or(serde_json::json!({})),
                "reranker_ms": reranker_ms,
                "total_ms": detail_started.elapsed().as_millis() as u64,
                "acoustic": acoustic_input.as_ref().map(|input| serde_json::to_value(&input.timing).unwrap_or(serde_json::json!({}))),
                "proposal_count": prototype_result.proposals.len(),
                "sentence_candidate_count": prototype_result.sentence_candidates.len(),
                "accepted_count": prototype_result.accepted.len(),
            },
            "alignments": {
                "timing_source": acoustic_input
                    .as_ref()
                    .map(|input| input.timing_source.clone())
                    .unwrap_or_else(|| {
                        if !parakeet.trim().is_empty() {
                            "no_acoustic_context".to_string()
                        } else {
                            "no_transcript".to_string()
                        }
                    }),
                "expected": fmt_alignment_json(&expected_alignment),
                "transcript": fmt_alignment_json(&transcript_alignment),
                "espeak": fmt_alignment_json(&transcript_alignment),
                "qwen": fmt_alignment_json(&transcript_alignment),
                "current": fmt_alignment_json(&current_alignment),
                "prototype": fmt_alignment_json(&prototype_alignment),
                "zipa": zipa_alignment,
                "zipa_transcript": crate::prototype::zipa_grouped_by_alignment_json(&transcript_alignment, &zipa_segments),
                "zipa_espeak": crate::prototype::zipa_grouped_by_alignment_json(&transcript_alignment, &zipa_segments),
                "zipa_qwen": crate::prototype::zipa_grouped_by_alignment_json(&transcript_alignment, &zipa_segments),
            },
            "zipa_trace": zipa_trace,
            "prototype_trace": {
                "corrected": prototype_result.corrected,
                "accepted": prototype_result.accepted,
                "proposals": prototype_result.proposals,
                "sentence_candidates": prototype_result.sentence_candidates,
                "reranker": reranker,
            },
        }))
    })
    .await
    .map_err(err)?
    .map_err(err)?;

    Ok(Json(value).into_response())
}

const PROTOTYPE_RERANKER_DEFAULT_MODEL: &str = "Qwen/Qwen2.5-0.5B-Instruct";
const PROTOTYPE_RERANKER_DEFAULT_ADAPTERS: &str = "training/prototype-reranker-adapters";
const PROTOTYPE_RERANKER_PORT: u16 = 8901;
const OFFICIAL_QWEN3_RERANKER_MODEL: &str = "Qwen/Qwen3-Reranker-0.6B";

#[derive(Debug, Clone)]
struct PrototypeRerankerTrainChoice {
    job_id: i64,
    model: String,
    adapters: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrototypeRerankerMode {
    Off,
    Trained,
    OfficialQwen3,
}

fn resolve_prototype_reranker_mode(
    reranker_mode: Option<&str>,
    use_model_reranker: bool,
    use_prototype_adapters: bool,
) -> PrototypeRerankerMode {
    if !use_model_reranker {
        return PrototypeRerankerMode::Off;
    }
    match reranker_mode
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "trained" => PrototypeRerankerMode::Trained,
        "qwen3" | "official-qwen3" | "official" => PrototypeRerankerMode::OfficialQwen3,
        "off" | "" => {
            if use_prototype_adapters {
                PrototypeRerankerMode::Trained
            } else {
                PrototypeRerankerMode::Off
            }
        }
        _ => {
            if use_prototype_adapters {
                PrototypeRerankerMode::Trained
            } else {
                PrototypeRerankerMode::Off
            }
        }
    }
}

pub struct Qwen3RerankerSidecar {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Qwen3RerankerSidecar {
    pub fn start() -> anyhow::Result<Self> {
        let mut child = Command::new("bash")
            .arg("scripts/qwen3_reranker_sidecar.sh")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow::anyhow!("qwen3 reranker sidecar missing stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow::anyhow!("qwen3 reranker sidecar missing stdout"))?;
        let mut sidecar = Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
        };
        let ready = sidecar.read_json_line()?;
        if ready.get("ready").and_then(|v| v.as_bool()) != Some(true) {
            anyhow::bail!("qwen3 reranker sidecar failed to start: {ready}");
        }
        Ok(sidecar)
    }

    fn read_json_line(&mut self) -> anyhow::Result<serde_json::Value> {
        let mut line = String::new();
        let read = self.stdout.read_line(&mut line)?;
        if read == 0 {
            anyhow::bail!("qwen3 reranker sidecar closed stdout");
        }
        Ok(serde_json::from_str(line.trim())?)
    }

    pub fn score(
        &mut self,
        instruction: &str,
        query: &str,
        documents: &[String],
    ) -> anyhow::Result<serde_json::Value> {
        let req = serde_json::json!({
            "instruction": instruction,
            "query": query,
            "documents": documents,
        });
        serde_json::to_writer(&mut self.stdin, &req)?;
        self.stdin.write_all(b"\n")?;
        self.stdin.flush()?;
        let value = self.read_json_line()?;
        if let Some(error) = value.get("error").and_then(|v| v.as_str()) {
            anyhow::bail!("qwen3 reranker sidecar error: {error}");
        }
        Ok(value)
    }

    pub fn kill(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub struct PrototypeRerankerSidecar {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    model: String,
    adapters: Option<String>,
}

impl PrototypeRerankerSidecar {
    pub fn start(model: &str, adapters: Option<&str>) -> anyhow::Result<Self> {
        let mut child = Command::new("bash")
            .arg("scripts/prototype_reranker_sidecar.sh")
            .arg(model)
            .arg(adapters.unwrap_or(""))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow::anyhow!("prototype reranker sidecar missing stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow::anyhow!("prototype reranker sidecar missing stdout"))?;
        let mut sidecar = Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            model: model.to_string(),
            adapters: adapters.map(|v| v.to_string()),
        };
        let ready = sidecar.read_json_line()?;
        if ready.get("ready").and_then(|v| v.as_bool()) != Some(true) {
            anyhow::bail!("prototype reranker sidecar failed to start: {ready}");
        }
        Ok(sidecar)
    }

    pub fn matches(&self, model: &str, adapters: Option<&str>) -> bool {
        self.model == model && self.adapters.as_deref() == adapters
    }

    fn read_json_line(&mut self) -> anyhow::Result<serde_json::Value> {
        let mut line = String::new();
        let read = self.stdout.read_line(&mut line)?;
        if read == 0 {
            anyhow::bail!("prototype reranker sidecar closed stdout");
        }
        Ok(serde_json::from_str(line.trim())?)
    }

    pub fn score(&mut self, prompts: &[String]) -> anyhow::Result<serde_json::Value> {
        let req = serde_json::json!({ "prompts": prompts });
        serde_json::to_writer(&mut self.stdin, &req)?;
        self.stdin.write_all(b"\n")?;
        self.stdin.flush()?;
        let value = self.read_json_line()?;
        if let Some(error) = value.get("error").and_then(|v| v.as_str()) {
            anyhow::bail!("prototype reranker sidecar error: {error}");
        }
        Ok(value)
    }

    pub fn kill(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn format_prototype_reranker_edits(candidate: &crate::prototype::SentenceCandidate) -> String {
    if candidate.edits.is_empty() {
        return "no edits".to_string();
    }
    candidate
        .edits
        .iter()
        .map(|edit| {
            let mut evidence = vec![format!("via {}", edit.via), format!("score {:.2}", edit.score)];
            if let Some(delta) = edit.acoustic_delta {
                evidence.push(format!("Δ {:.2}", delta));
            } else if let Some(acoustic) = edit.acoustic_score {
                evidence.push(format!("a {:.2}", acoustic));
            }
            if let Some(phones) = &edit.from_phonemes {
                evidence.push(format!("from /{phones}/"));
            }
            if let Some(phones) = &edit.to_phonemes {
                evidence.push(format!("to /{phones}/"));
            }
            format!("{} -> {} ({})", edit.from, edit.to, evidence.join(", "))
        })
        .collect::<Vec<_>>()
        .join("; ")
}

fn recent_completed_prototype_reranker_trains(
    db: &crate::db::Db,
) -> Vec<PrototypeRerankerTrainChoice> {
    let Ok(jobs) = db.list_jobs() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for job in jobs {
        if job.job_type != "train" || job.status != "completed" {
            continue;
        }
        let Some(config) = job.config.as_deref() else {
            continue;
        };
        let Ok(config_json) = serde_json::from_str::<serde_json::Value>(config) else {
            continue;
        };
        if config_json.get("data").and_then(|v| v.as_str()) != Some("training/prototype-reranker")
        {
            continue;
        }
        let model = config_json
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or(PROTOTYPE_RERANKER_DEFAULT_MODEL)
            .to_string();
        let adapters = config_json
            .get("adapters")
            .and_then(|v| v.as_str())
            .unwrap_or(PROTOTYPE_RERANKER_DEFAULT_ADAPTERS)
            .to_string();
        out.push(PrototypeRerankerTrainChoice {
            job_id: job.id,
            model,
            adapters,
        });
    }
    out
}

fn prototype_reranker_config(
    state: &AppState,
    use_adapters: bool,
    selected_train_id: Option<i64>,
) -> synth_train::InferenceConfig {
    let (model, adapters) = if use_adapters {
        let db = state.db.lock().unwrap();
        let recent = recent_completed_prototype_reranker_trains(&db);
        let selected = recent
            .iter()
            .find(|train| Some(train.job_id) == selected_train_id)
            .cloned()
            .or_else(|| recent.into_iter().next());
        selected
            .map(|train| (train.model, train.adapters))
            .unwrap_or((
                PROTOTYPE_RERANKER_DEFAULT_MODEL.to_string(),
                PROTOTYPE_RERANKER_DEFAULT_ADAPTERS.to_string(),
            ))
    } else {
        (
            PROTOTYPE_RERANKER_DEFAULT_MODEL.to_string(),
            String::new(),
        )
    };
    synth_train::InferenceConfig {
        model,
        adapters,
        attach_adapters: use_adapters,
        max_tokens: 6,
        port: PROTOTYPE_RERANKER_PORT,
    }
}

fn should_run_prototype_reranker(
    candidates: &[crate::prototype::SentenceCandidate],
) -> bool {
    candidates.len() > 1
}

fn build_official_qwen3_reranker_instruction() -> &'static str {
    "Given an ASR sentence, judge whether the Document is the correct technical correction of the ASR sentence. Prefer reviewed vocabulary repairs. Reject candidates that change meaning, invent terms, or apply weak edits."
}

fn build_official_qwen3_reranker_document(
    candidate: &crate::prototype::SentenceCandidate,
) -> String {
    let mut doc = format!("<Candidate>: {}", candidate.text.trim());
    doc.push_str("\n<Edits>: ");
    doc.push_str(&format_prototype_reranker_edits(candidate));
    doc
}

fn run_prototype_reranker(
    state: &AppState,
    asr_text: &str,
    candidates: &[crate::prototype::SentenceCandidate],
    use_adapters: bool,
    selected_train_id: Option<i64>,
) -> anyhow::Result<serde_json::Value> {
    let infer_config = prototype_reranker_config(state, use_adapters, selected_train_id);
    if use_adapters {
        let adapters = (!infer_config.adapters.is_empty()).then_some(infer_config.adapters.as_str());
        let mut server_guard = state.prototype_reranker_sidecar.lock().unwrap();
        if server_guard
            .as_ref()
            .map(|server| !server.matches(&infer_config.model, adapters))
            .unwrap_or(true)
        {
            if let Some(mut server) = server_guard.take() {
                server.kill();
            }
            *server_guard = Some(PrototypeRerankerSidecar::start(&infer_config.model, adapters)?);
        }
        let server = server_guard.as_mut().unwrap();
        let prompts = candidates
            .iter()
            .map(|candidate| build_prototype_reranker_prompt(asr_text, candidate))
            .collect::<Vec<_>>();
        let value = server.score(&prompts)?;
        let results = value
            .get("results")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let timing = value
            .get("timing_ms")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));
        let backend = value
            .get("backend")
            .cloned()
            .unwrap_or_else(|| serde_json::json!("mlx-lm"));

        let mut best_index = None;
        let mut best_yes = f32::NEG_INFINITY;
        let mut best_heuristic = f32::NEG_INFINITY;
        let mut scored_candidates = Vec::new();

        for (idx, candidate) in candidates.iter().enumerate() {
            let result = results.get(idx);
            let yes_prob = result
                .and_then(|v| v.get("yes_prob"))
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0) as f32;
            let no_prob = result
                .and_then(|v| v.get("no_prob"))
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0) as f32;
            let answer = result
                .and_then(|v| v.get("answer"))
                .and_then(|v| v.as_str())
                .unwrap_or(if yes_prob >= no_prob { "yes" } else { "no" })
                .to_string();
            let label_logprobs = result
                .and_then(|v| v.get("label_logprobs"))
                .cloned()
                .unwrap_or_else(|| serde_json::json!({}));

            if yes_prob > best_yes
                || ((yes_prob - best_yes).abs() < 1e-6 && candidate.score > best_heuristic)
            {
                best_yes = yes_prob;
                best_heuristic = candidate.score;
                best_index = Some(idx);
            }

            scored_candidates.push(serde_json::json!({
                "index": idx,
                "label": candidate.label,
                "text": candidate.text,
                "heuristic_score": candidate.score,
                "yes_prob": yes_prob,
                "no_prob": no_prob,
                "answer": answer,
                "prompt": prompts[idx],
                "label_logprobs": label_logprobs,
            }));
        }

        return Ok(serde_json::json!({
            "mode": "trained-prototype-reranker-mlx",
            "train_job_id": selected_train_id,
            "model": infer_config.model,
            "adapters": if infer_config.attach_adapters { Some(infer_config.adapters.clone()) } else { None::<String> },
            "use_adapters": infer_config.attach_adapters,
            "backend": backend,
            "candidate_count": candidates.len(),
            "chosen_index": best_index,
            "chosen_label": best_index.and_then(|idx| candidates.get(idx).map(|c| c.label.clone())),
            "chosen_text": best_index.and_then(|idx| candidates.get(idx).map(|c| c.text.clone())),
            "timing_ms": timing,
            "candidates": scored_candidates,
        }));
    }

    let mut server_guard = state.prototype_reranker_server.lock().unwrap();
    if server_guard
        .as_ref()
        .map(|server| !server.matches(&infer_config))
        .unwrap_or(true)
    {
        if let Some(mut server) = server_guard.take() {
            server.kill();
        }
        *server_guard = Some(synth_train::InferenceServer::start(&infer_config)?);
    }
    let server = server_guard.as_mut().unwrap();

    let mut best_index = None;
    let mut best_yes = f32::NEG_INFINITY;
    let mut best_heuristic = f32::NEG_INFINITY;
    let mut scored_candidates = Vec::new();

    for (idx, candidate) in candidates.iter().enumerate() {
        let prompt = build_prototype_reranker_prompt(asr_text, candidate);
        let output = server.infer_with_stats(&prompt)?;
        let answer = output.text.trim().to_ascii_lowercase();
        let yes_prob = if answer.starts_with("yes") {
            1.0
        } else if answer.starts_with("no") {
            0.0
        } else {
            0.5
        };
        let no_prob = 1.0 - yes_prob;

        if yes_prob > best_yes
            || ((yes_prob - best_yes).abs() < 1e-6 && candidate.score > best_heuristic)
        {
            best_yes = yes_prob;
            best_heuristic = candidate.score;
            best_index = Some(idx);
        }

        scored_candidates.push(serde_json::json!({
            "index": idx,
            "label": candidate.label,
            "text": candidate.text,
            "heuristic_score": candidate.score,
            "yes_prob": yes_prob,
            "no_prob": no_prob,
            "answer": output.text,
            "raw_output": output.raw_text,
            "prompt": prompt,
            "timing": output.stats,
        }));
    }

    Ok(serde_json::json!({
        "mode": if use_adapters { "trained-prototype-reranker" } else { "prototype-reranker-base" },
        "train_job_id": selected_train_id,
        "model": infer_config.model,
        "adapters": if infer_config.attach_adapters { Some(infer_config.adapters.clone()) } else { None::<String> },
        "use_adapters": infer_config.attach_adapters,
        "candidate_count": candidates.len(),
        "chosen_index": best_index,
        "chosen_label": best_index.and_then(|idx| candidates.get(idx).map(|c| c.label.clone())),
        "chosen_text": best_index.and_then(|idx| candidates.get(idx).map(|c| c.text.clone())),
        "candidates": scored_candidates,
    }))
}

fn run_official_qwen3_reranker(
    state: &AppState,
    asr_text: &str,
    candidates: &[crate::prototype::SentenceCandidate],
) -> anyhow::Result<serde_json::Value> {
    let mut server_guard = state.official_qwen3_reranker_server.lock().unwrap();
    if server_guard.is_none() {
        *server_guard = Some(Qwen3RerankerSidecar::start()?);
    }
    let server = server_guard.as_mut().unwrap();
    let documents = candidates
        .iter()
        .map(build_official_qwen3_reranker_document)
        .collect::<Vec<_>>();
    let query = asr_text.trim();
    let instruction = build_official_qwen3_reranker_instruction();
    let value = server.score(instruction, query, &documents)?;
    let scores = value
        .get("scores")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let timing = value.get("timing_ms").cloned().unwrap_or_else(|| serde_json::json!({}));
    let device = value.get("device").cloned().unwrap_or_else(|| serde_json::json!(null));
    let model = value
        .get("model")
        .cloned()
        .unwrap_or_else(|| serde_json::json!(OFFICIAL_QWEN3_RERANKER_MODEL));

    let mut best_index = None;
    let mut best_yes = f32::NEG_INFINITY;
    let mut best_heuristic = f32::NEG_INFINITY;
    let mut scored_candidates = Vec::new();

    for (idx, candidate) in candidates.iter().enumerate() {
        let yes_prob = scores
            .get(idx)
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;
        let no_prob = 1.0 - yes_prob;
        if yes_prob > best_yes
            || ((yes_prob - best_yes).abs() < 1e-6 && candidate.score > best_heuristic)
        {
            best_yes = yes_prob;
            best_heuristic = candidate.score;
            best_index = Some(idx);
        }
        scored_candidates.push(serde_json::json!({
            "index": idx,
            "label": candidate.label,
            "text": candidate.text,
            "heuristic_score": candidate.score,
            "yes_prob": yes_prob,
            "no_prob": no_prob,
            "document": documents[idx],
        }));
    }

    Ok(serde_json::json!({
        "mode": "official-qwen3-reranker",
        "model": model,
        "device": device,
        "candidate_count": candidates.len(),
        "chosen_index": best_index,
        "chosen_label": best_index.and_then(|idx| candidates.get(idx).map(|c| c.label.clone())),
        "chosen_text": best_index.and_then(|idx| candidates.get(idx).map(|c| c.text.clone())),
        "instruction": instruction,
        "query": query,
        "timing_ms": timing,
        "candidates": scored_candidates,
    }))
}

fn normalized_compare_eq(a: &str, b: &str) -> bool {
    fn normalize(text: &str) -> String {
        text.chars()
            .filter_map(|ch| {
                if ch.is_ascii_alphanumeric() || ch == '_' {
                    Some(ch.to_ascii_lowercase())
                } else {
                    None
                }
            })
            .collect()
    }
    normalize(a) == normalize(b)
}

// ==================== Scan Results ====================

pub async fn api_scan_results(State(state): State<Arc<AppState>>) -> Result<Response, AppError> {
    let db = state.db.lock().unwrap();
    let results = db.vocab_scan_results().map_err(err)?;
    let json: Vec<serde_json::Value> = results
        .iter()
        .map(|(term, total, qwen_err, parakeet_err)| {
            let confusions = db.confusions_for_term(term).unwrap_or_default();
            let qwen_heard: Vec<&str> = confusions.iter().map(|(q, _)| q.as_str()).collect();
            let parakeet_heard: Vec<&str> = confusions.iter().map(|(_, p)| p.as_str()).collect();
            serde_json::json!({
                "term": term,
                "total": total,
                "qwen_errors": qwen_err,
                "parakeet_errors": parakeet_err,
                "qwen_heard": qwen_heard,
                "parakeet_heard": parakeet_heard,
            })
        })
        .collect();
    Ok(Json(json).into_response())
}

// ==================== Test One Term ====================

#[derive(Deserialize)]
pub struct TestTermBody {
    pub term: String,
    pub tts_backend: Option<String>,
    pub dual_asr: Option<bool>,
}

/// Run one corpus pass for a single term and return the full result (without saving to DB).
pub async fn api_test_term(
    State(state): State<Arc<AppState>>,
    Json(body): Json<TestTermBody>,
) -> Result<Response, AppError> {
    let written = WrittenTerm(body.term.trim().to_string());
    let tts_backend = body.tts_backend.unwrap_or_else(|| "pocket-hq".to_string());
    let dual_asr = body.dual_asr.unwrap_or(false);

    let vocab = {
        let db = state.db.lock().unwrap();
        db.find_vocab_by_term(&written.0)
            .map_err(err)?
            .ok_or_else(|| err(format!("Term '{}' not found", written.0)))?
    };
    let spoken = SpokenTerm(vocab.spoken().to_string());

    let (all_texts, overrides) = {
        let db = state.db.lock().unwrap();
        let texts = db.all_sentence_texts().map_err(err)?;
        let vocab_list = db.list_reviewed_vocab().map_err(err)?;
        let ov: HashMap<String, String> = vocab_list
            .iter()
            .filter_map(|v| {
                v.spoken_override
                    .as_ref()
                    .map(|s| (v.term.clone(), s.clone()))
            })
            .collect();
        (texts, ov)
    };
    let chain = MarkovChain::build(&all_texts);
    let mut rng = rand::rngs::StdRng::from_os_rng();

    // Retry up to 5 times if we get a noisy extraction
    let mut result;
    let mut audio;
    let mut written_sentence;
    let mut spoken_sentence;
    let mut attempts = 0;
    loop {
        attempts += 1;
        let pair = make_sentence_pair(None, &chain, &written, &spoken, None, &overrides, &mut rng);
        written_sentence = pair.0;
        spoken_sentence = pair.1;

        audio = state
            .tts
            .generate(&tts_backend, &spoken_sentence.0)
            .await
            .map_err(|e| err(e))?;
        audio.normalize();
        let full_16k = tts::resample_to_16k(&audio.samples, audio.sample_rate).map_err(err)?;

        let protected_terms: std::collections::HashSet<String> = {
            let db = state.db.lock().unwrap();
            db.list_reviewed_vocab()
                .unwrap_or_default()
                .iter()
                .map(|v| v.term.to_lowercase())
                .collect()
        };
        result = run_corpus_pass(
            &state,
            &full_16k,
            &written_sentence,
            &spoken_sentence,
            &written,
            &protected_terms,
            dual_asr,
        )
        .await
        .map_err(err)?;

        if result.extraction.clean || attempts >= 5 {
            break;
        }
    }

    let full_16k = tts::resample_to_16k(&audio.samples, audio.sample_rate).map_err(err)?;

    // Encode audio as base64 WAV for playback
    let wav_b64 = {
        use base64::Engine;
        let wav = audio.to_wav().map_err(err)?;
        base64::engine::general_purpose::STANDARD.encode(&wav)
    };

    let ex = &result.extraction;
    Ok(Json(serde_json::json!({
        "term": written.0,
        "spoken": spoken.0,
        "sentence": result.sentence.0,
        "spoken_sentence": result.spoken_sentence.0,
        "qwen_full": result.qwen_full,
        "parakeet_full": result.parakeet_full,
        "term_found": result.term_found,
        "attempts": attempts,
        "extraction": {
            "original": ex.original,
            "qwen": ex.qwen,
            "parakeet": ex.parakeet,
            "clean": ex.clean,
            "cons_range": [ex.cons_range.0, ex.cons_range.1],
        },
        "alignments": {
            "spoken": fmt_alignment_json(&result.spoken_alignment),
            "original": fmt_alignment_json(&result.orig_alignment),
            "qwen": fmt_alignment_json(&result.qwen_alignment),
            "parakeet": fmt_alignment_json(&result.parakeet_alignment),
        },
        "term_range": [result.term_range.0, result.term_range.1],
        "wav_b64": wav_b64,
    }))
    .into_response())
}

// ==================== Pipeline Status ====================

pub async fn api_view_corpus(
    State(state): State<Arc<AppState>>,
    axum::extract::Query(params): axum::extract::Query<HashMap<String, String>>,
) -> Result<Response, AppError> {
    let filter_term = params
        .get("term")
        .map(|s| s.as_str())
        .filter(|s| !s.is_empty());
    let mistakes_only = params.get("mistakes").map(|s| s == "1").unwrap_or(false);
    let review_filter = params.get("review").map(|s| s.as_str());
    let limit: usize = params
        .get("limit")
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    let offset: usize = params
        .get("offset")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let db = state.db.lock().unwrap();
    let pairs = db
        .corpus_pairs_query(filter_term, mistakes_only, review_filter, limit, offset)
        .map_err(err)?;
    let stats = db.corpus_stats().map_err(err)?;
    Ok(Json(serde_json::json!({"pairs": pairs, "stats": stats})).into_response())
}

pub async fn api_corpus_audio(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(id): axum::extract::Path<i64>,
) -> Result<Response, AppError> {
    let db = state.db.lock().unwrap();
    let audio: Option<Vec<u8>> = db.get_corpus_audio(id).map_err(err)?;
    match audio {
        Some(bytes) => {
            Ok(([(axum::http::header::CONTENT_TYPE, "audio/ogg")], bytes).into_response())
        }
        None => Ok(axum::http::StatusCode::NOT_FOUND.into_response()),
    }
}

pub async fn api_reset_corpus(State(state): State<Arc<AppState>>) -> Result<Response, AppError> {
    let db = state.db.lock().unwrap();
    db.reset_corpus().map_err(err)?;
    let _ = std::fs::remove_file("data/corpus_dashboard.jsonl");
    Ok(Json(serde_json::json!({"ok": true})).into_response())
}

#[derive(Deserialize)]
pub struct DeleteCorpusTermBody {
    pub term: String,
}

pub async fn api_delete_corpus_term(
    State(state): State<Arc<AppState>>,
    Json(body): Json<DeleteCorpusTermBody>,
) -> Result<Response, AppError> {
    let db = state.db.lock().unwrap();
    let deleted = db.delete_corpus_pairs_for_term(&body.term).map_err(err)?;
    Ok(Json(serde_json::json!({"ok": true, "deleted": deleted})).into_response())
}

#[derive(Deserialize)]
pub struct RejectCorpusPairBody {
    pub id: i64,
}

pub async fn api_reject_corpus_pair(
    State(state): State<Arc<AppState>>,
    Json(body): Json<RejectCorpusPairBody>,
) -> Result<Response, AppError> {
    let db = state.db.lock().unwrap();
    db.reject_corpus_pair(body.id).map_err(err)?;
    Ok(Json(serde_json::json!({"ok": true})).into_response())
}

#[derive(Deserialize)]
pub struct ApproveCorpusPairBody {
    pub id: i64,
}

pub async fn api_approve_corpus_pair(
    State(state): State<Arc<AppState>>,
    Json(body): Json<ApproveCorpusPairBody>,
) -> Result<Response, AppError> {
    let db = state.db.lock().unwrap();
    db.approve_corpus_pair(body.id).map_err(err)?;
    Ok(Json(serde_json::json!({"ok": true})).into_response())
}

#[derive(Deserialize)]
pub struct AddEvalMistakeBody {
    pub term: String,
    pub expected: String,
    pub qwen: String,
    pub sentence: String,
    pub recording_id: Option<i64>,
    pub kind: Option<String>,
    pub surface_form: Option<String>,
    pub alignment_failed: bool,
    pub alignments: Option<serde_json::Value>,
    pub cons_time: Option<serde_json::Value>,
    pub trim_info: Option<serde_json::Value>,
    pub trace: Option<serde_json::Value>,
}

pub async fn api_add_eval_mistake(
    State(state): State<Arc<AppState>>,
    Json(body): Json<AddEvalMistakeBody>,
) -> Result<Response, AppError> {
    let term = body.term.trim();
    let expected = body.expected.trim();
    let qwen = body.qwen.trim();
    let sentence = body.sentence.trim();

    if body.alignment_failed {
        return Ok(Json(serde_json::json!({
            "ok": false,
            "error": "Cannot add eval rows with alignment failure."
        }))
        .into_response());
    }
    if body.kind.as_deref() == Some("counterexample") {
        return Ok(Json(serde_json::json!({
            "ok": false,
            "error": "Counterexample eval rows should not be added as mistakes."
        }))
        .into_response());
    }
    if term.is_empty() || expected.is_empty() || qwen.is_empty() || sentence.is_empty() {
        return Ok(Json(serde_json::json!({
            "ok": false,
            "error": "Missing required eval feedback fields."
        }))
        .into_response());
    }
    if normalize_eval_fragment(expected) == normalize_eval_fragment(qwen) {
        return Ok(Json(serde_json::json!({
            "ok": false,
            "error": "This eval row is not a mistake."
        }))
        .into_response());
    }

    let (orig_align, qwen_align) = if let Some(alignments) = body.alignments.as_ref() {
        (
            alignments.get("expected").cloned(),
            alignments.get("asr").cloned(),
        )
    } else {
        (None, None)
    };
    let mut trim_info = body.trim_info.unwrap_or_else(|| serde_json::json!({}));
    if !trim_info.is_object() {
        trim_info = serde_json::json!({ "raw": trim_info });
    }
    if let Some(obj) = trim_info.as_object_mut() {
        obj.insert("source".into(), serde_json::json!("eval_feedback"));
        obj.insert("recording_id".into(), serde_json::json!(body.recording_id));
        obj.insert("kind".into(), serde_json::json!(body.kind));
        obj.insert("surface_form".into(), serde_json::json!(body.surface_form));
        obj.insert(
            "trace".into(),
            body.trace.unwrap_or(serde_json::Value::Null),
        );
    }

    let db = state.db.lock().unwrap();
    let (id, is_new) = db
        .add_eval_feedback_mistake(
            term,
            expected,
            qwen,
            sentence,
            sentence,
            orig_align.as_ref().map(|v| v.to_string()).as_deref(),
            qwen_align.as_ref().map(|v| v.to_string()).as_deref(),
            body.cons_time.as_ref().map(|v| v.to_string()).as_deref(),
            Some(&trim_info.to_string()),
        )
        .map_err(err)?;

    Ok(Json(serde_json::json!({
        "ok": true,
        "id": id,
        "is_new": is_new
    }))
    .into_response())
}

#[derive(Deserialize)]
pub struct AddAltSpellingBody {
    pub term: String,
    pub alt_spelling: String,
}

pub async fn api_add_alt_spelling(
    State(state): State<Arc<AppState>>,
    Json(body): Json<AddAltSpellingBody>,
) -> Result<Response, AppError> {
    let db = state.db.lock().unwrap();
    let updated = db
        .add_alt_spelling(&body.term, &body.alt_spelling)
        .map_err(err)?;
    Ok(Json(serde_json::json!({"ok": true, "retroactive_updates": updated})).into_response())
}

// ==================== Confusions Review ====================

pub async fn api_confusions_next(State(state): State<Arc<AppState>>) -> Result<Response, AppError> {
    let db = state.db.lock().unwrap();
    let item = db.next_unreviewed_confusion().map_err(err)?;
    let progress = db.confusions_review_progress().map_err(err)?;
    let can_mark_alt = item
        .as_ref()
        .map(|i| {
            let term = i["term"].as_str().unwrap_or("");
            let original = i["original"].as_str().unwrap_or("");
            term.to_lowercase() == original.to_lowercase()
        })
        .unwrap_or(false);
    Ok(Json(serde_json::json!({
        "item": item,
        "progress": progress,
        "can_mark_alt": can_mark_alt,
    }))
    .into_response())
}

/// Preview training data: returns sample prompts from the training set so the user
/// can see exactly what the model will be trained on.
pub async fn api_preview_training(
    State(_state): State<Arc<AppState>>,
) -> Result<Response, AppError> {
    let train_path = "training/data/train.jsonl";
    if !std::path::Path::new(train_path).exists() {
        return Ok(
            Json(serde_json::json!({"error": "No training data. Run Prepare first."}))
                .into_response(),
        );
    }

    let content = std::fs::read_to_string(train_path).map_err(err)?;
    let all_lines: Vec<&str> = content.lines().filter(|l| !l.trim().is_empty()).collect();
    let total = all_lines.len();

    // Separate correction vs identity examples
    let mut corrections = Vec::new();
    let mut identities = Vec::new();
    for line in &all_lines {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
            let prompt = v["prompt"].as_str().unwrap_or("");
            let completion = v["completion"].as_str().unwrap_or("");
            let prompt_qwen = prompt
                .split_once("<qwen> ")
                .map(|(_, rest)| rest)
                .and_then(|rest| rest.split_once("\n<fixd>"))
                .map(|(qwen_text, _)| normalize_prepare_text(qwen_text))
                .unwrap_or_default();
            let completion_text = normalize_prepare_text(
                completion
                    .trim()
                    .trim_end_matches("<|endoftext|>")
                    .trim(),
            );
            let is_identity = !prompt_qwen.is_empty() && prompt_qwen == completion_text;
            let entry = serde_json::json!({"prompt": prompt, "completion": completion});
            if is_identity {
                identities.push(entry);
            } else {
                corrections.push(entry);
            }
        }
    }

    // Sample up to 10 of each type
    let mut rng = rand::rng();
    use rand::seq::SliceRandom;
    let mut corr_sample: Vec<_> = corrections.iter().collect();
    corr_sample.shuffle(&mut rng);
    let corr_sample: Vec<_> = corr_sample.into_iter().take(10).cloned().collect();

    let mut id_sample: Vec<_> = identities.iter().collect();
    id_sample.shuffle(&mut rng);
    let id_sample: Vec<_> = id_sample.into_iter().take(10).cloned().collect();

    Ok(Json(serde_json::json!({
        "total": total,
        "correction_count": corrections.len(),
        "identity_count": identities.len(),
        "correction_samples": corr_sample,
        "identity_samples": id_sample,
    }))
    .into_response())
}

/// Delete training data (train/valid/test splits) so prepare can be re-run.
pub async fn api_reset_training(State(_state): State<Arc<AppState>>) -> Result<Response, AppError> {
    let _ = std::fs::remove_file("training/data/train.jsonl");
    let _ = std::fs::remove_file("training/data/valid.jsonl");
    let _ = std::fs::remove_file("training/data/test.jsonl");
    let _ = std::fs::remove_file("training/applied-eval.jsonl");
    Ok(Json(serde_json::json!({"ok": true})).into_response())
}

pub async fn api_pipeline_status(State(state): State<Arc<AppState>>) -> Result<Response, AppError> {
    let (
        approved_count,
        vocab_reviewed,
        authored_sentences,
        authored_counterexamples,
        human_recordings,
        phone_traces,
        running_job,
        last_eval,
        last_train,
        vocab_scanned,
        recent_prototype_reranker_trains,
    ) = {
        let db = state.db.lock().unwrap();
        let (approved, _, _) = db.sentence_count_by_status().map_err(err)?;
        let (reviewed, _, _) = db.vocab_review_counts().unwrap_or((0, 0, 0));
        let authored_sentences = db.authored_sentence_count().unwrap_or(0);
        let authored_counterexamples = db.authored_counterexample_count().unwrap_or(0);
        let human = db.authored_sentence_recordings_count().map_err(err)?;
        let phone_traces = db.authored_recording_phone_trace_count().unwrap_or(0);
        let scanned = db.confusion_count().map_err(err)?;
        let jobs = db.list_jobs().map_err(err)?;
        let running = jobs.iter().find(|j| j.status == "running").cloned();
        let eval = jobs
            .iter()
            .filter(|j| j.job_type == "eval" && j.status == "completed")
            .next()
            .and_then(|j| j.result.as_ref())
            .and_then(|r| serde_json::from_str::<serde_json::Value>(r).ok());
        let train = jobs
            .iter()
            .filter(|j| {
                if j.job_type != "train" || j.status != "completed" {
                    return false;
                }
                let Some(config) = j.config.as_deref() else {
                    return false;
                };
                let Ok(config_json) = serde_json::from_str::<serde_json::Value>(config) else {
                    return false;
                };
                config_json.get("data").and_then(|v| v.as_str()) == Some("training/prototype-reranker")
            })
            .next()
            .and_then(|j| j.result.as_ref())
            .and_then(|r| serde_json::from_str::<serde_json::Value>(r).ok());
        let mut recent_trains = jobs
            .iter()
            .filter_map(|j| {
                if j.job_type != "train" || j.status != "completed" {
                    return None;
                }
                let config = j.config.as_deref()?;
                let config_json = serde_json::from_str::<serde_json::Value>(config).ok()?;
                if config_json.get("data").and_then(|v| v.as_str())
                    != Some("training/prototype-reranker")
                {
                    return None;
                }
                let result_json = j
                    .result
                    .as_deref()
                    .and_then(|r| serde_json::from_str::<serde_json::Value>(r).ok())
                    .unwrap_or_else(|| serde_json::json!({}));
                Some(serde_json::json!({
                    "id": j.id,
                    "created_at": j.created_at,
                    "finished_at": j.finished_at,
                    "model": config_json.get("model").and_then(|v| v.as_str()),
                    "adapters": config_json.get("adapters").and_then(|v| v.as_str()),
                    "data": config_json.get("data").and_then(|v| v.as_str()),
                    "batch_size": config_json.get("batch_size").and_then(|v| v.as_i64()),
                    "num_layers": config_json.get("num_layers").and_then(|v| v.as_i64()),
                    "iters": config_json.get("iters").and_then(|v| v.as_i64()),
                    "steps_per_eval": config_json.get("steps_per_eval").and_then(|v| v.as_i64()),
                    "patience": config_json.get("patience").and_then(|v| v.as_i64()),
                    "val_loss": result_json.get("val_loss").cloned().unwrap_or(serde_json::Value::Null),
                    "adapter_mb": result_json.get("adapter_mb").cloned().unwrap_or(serde_json::Value::Null),
                }))
            })
            .collect::<Vec<_>>();
        recent_trains.sort_by(|a, b| {
            b.get("id")
                .and_then(|v| v.as_i64())
                .cmp(&a.get("id").and_then(|v| v.as_i64()))
        });
        recent_trains.truncate(8);
        (
            approved,
            reviewed,
            authored_sentences,
            authored_counterexamples,
            human,
            phone_traces,
            running,
            eval,
            train,
            scanned,
            recent_trains,
        )
    };

    // Check filesystem for corpus / training data / adapters
    let corpus_lines = {
        let db = state.db.lock().unwrap();
        db.corpus_pair_count().unwrap_or(0) as usize
    };
    let corpus_exists = corpus_lines > 0;

    let training_data_exists = std::path::Path::new("training/data/train.jsonl").exists();
    let (train_count, valid_count) = if training_data_exists {
        let tc = std::fs::read_to_string("training/data/train.jsonl")
            .map(|s| s.lines().filter(|l| !l.trim().is_empty()).count())
            .unwrap_or(0);
        let vc = std::fs::read_to_string("training/data/valid.jsonl")
            .map(|s| s.lines().filter(|l| !l.trim().is_empty()).count())
            .unwrap_or(0);
        (tc, vc)
    } else {
        (0, 0)
    };

    let adapters_exist = !recent_prototype_reranker_trains.is_empty();
    let applied_eval_exists = std::path::Path::new("training/applied-eval.jsonl").exists();
    let applied_eval_count = if applied_eval_exists {
        std::fs::read_to_string("training/applied-eval.jsonl")
            .map(|s| s.lines().filter(|l| !l.trim().is_empty()).count())
            .unwrap_or(0)
    } else {
        0
    };
    let prepare_ready = authored_sentences > 0 && vocab_scanned > 0;

    let backends = state.tts.available_backends();

    let running_json = running_job.map(|j| {
        serde_json::json!({
            "id": j.id,
            "job_type": j.job_type,
            "status": j.status,
        })
    });

    Ok(Json(serde_json::json!({
        "approved_count": approved_count,
        "vocab_reviewed": vocab_reviewed,
        "authored_sentences": authored_sentences,
        "authored_counterexamples": authored_counterexamples,
        "corpus_exists": corpus_exists,
        "corpus_lines": corpus_lines,
        "prepare_ready": prepare_ready,
        "training_data_exists": training_data_exists,
        "train_count": train_count,
        "valid_count": valid_count,
        "applied_eval_exists": applied_eval_exists,
        "applied_eval_count": applied_eval_count,
        "adapters_exist": adapters_exist,
        "human_recordings": human_recordings,
        "phone_traces": phone_traces,
        "vocab_scanned": vocab_scanned,
        "backends": backends,
        "running_job": running_json,
        "last_eval": last_eval,
        "last_train": last_train,
        "recent_prototype_reranker_trains": recent_prototype_reranker_trains,
    }))
    .into_response())
}

// ==================== Algorithm Test Suite ====================

fn ai(word: &str, start: f64, end: f64) -> qwen3_asr::ForcedAlignItem {
    qwen3_asr::ForcedAlignItem {
        word: word.to_string(),
        start_time: start,
        end_time: end,
    }
}

fn fmt_align_json(items: &[qwen3_asr::ForcedAlignItem]) -> serde_json::Value {
    serde_json::Value::Array(
        items
            .iter()
            .map(|a| serde_json::json!({"w": a.word, "s": a.start_time, "e": a.end_time}))
            .collect(),
    )
}

pub async fn api_algorithm_tests() -> Result<Response, AppError> {
    let mut results = Vec::new();

    struct TestCase {
        name: &'static str,
        desc: &'static str,
        term: &'static str,
        orig: Vec<qwen3_asr::ForcedAlignItem>,
        qwen: Vec<qwen3_asr::ForcedAlignItem>,
        para: Vec<qwen3_asr::ForcedAlignItem>,
        protected: Vec<&'static str>,
        expect_orig: &'static str,
        expect_qwen: &'static str,
        expect_clean: bool,
    }

    let cases = vec![
        TestCase {
            name: "Basic single word",
            desc: "Simple term in the middle, same words on both sides",
            term: "tokio",
            orig: vec![ai("the",0.0,0.2), ai("tokio",0.3,0.7), ai("crate",0.8,1.1)],
            qwen: vec![ai("the",0.0,0.2), ai("Tokyo",0.3,0.7), ai("crate",0.8,1.1)],
            para: vec![],
            protected: vec!["tokio"],
            expect_orig: "tokio",
            expect_qwen: "Tokyo",
            expect_clean: true,
        },
        TestCase {
            name: "Edge trimming — matching context",
            desc: "\"the\" and \"crate\" match on both sides and should be trimmed",
            term: "reqwest",
            orig: vec![ai("use",0.0,0.2), ai("the",0.3,0.5), ai("reqwest",0.6,1.0), ai("crate",1.1,1.4), ai("for",1.5,1.7)],
            qwen: vec![ai("use",0.0,0.2), ai("the",0.3,0.5), ai("request",0.6,1.0), ai("crate",1.1,1.4), ai("for",1.5,1.7)],
            para: vec![],
            protected: vec!["reqwest"],
            expect_orig: "reqwest",
            expect_qwen: "request",
            expect_clean: true,
        },
        TestCase {
            name: "Protected term not trimmed",
            desc: "\"async\" is protected — should not be trimmed even though it matches",
            term: "async",
            orig: vec![ai("use",0.0,0.2), ai("async",0.3,0.7), ai("fn",0.8,1.0)],
            qwen: vec![ai("use",0.0,0.2), ai("a",0.3,0.5), ai("sync",0.5,0.7), ai("fn",0.8,1.0)],
            para: vec![],
            protected: vec!["async"],
            expect_orig: "async",
            expect_qwen: "a sync",
            expect_clean: true,
        },
        TestCase {
            name: "Gap expansion — term with gaps",
            desc: "Gaps around the term in Original; Qwen splits it into two words in the gap space",
            term: "lldb",
            orig: vec![ai("but",0.0,0.3), ai("lldb",0.5,0.8), ai("is",1.1,1.3)],
            qwen: vec![ai("but",0.0,0.3), ai("L",0.5,0.7), ai("LDB",0.8,1.0), ai("is",1.1,1.3)],
            para: vec![],
            protected: vec!["lldb"],
            expect_orig: "lldb",
            expect_qwen: "L LDB",
            expect_clean: true,
        },
        TestCase {
            name: "No gap — no expansion",
            desc: "No gap between term and next word; should NOT swallow the adjacent word",
            term: "bumpalo",
            orig: vec![ai("use",0.0,0.3), ai("bumpalo",0.4,0.8), ai("to",0.81,1.0), ai("format",1.1,1.4)],
            qwen: vec![ai("use",0.0,0.3), ai("Bumpalo",0.4,0.8), ai("to",0.81,1.0), ai("format",1.1,1.4)],
            para: vec![],
            protected: vec!["bumpalo"],
            expect_orig: "bumpalo",
            expect_qwen: "Bumpalo",
            expect_clean: true,
        },
        TestCase {
            name: "Empty Qwen lane → discard",
            desc: "Qwen has nothing in the consensus range — extraction should be noisy",
            term: "QEMU",
            orig: vec![ai("the",0.0,0.2), ai("QEMU",0.3,0.7), ai("vm",0.8,1.0)],
            qwen: vec![ai("the",0.0,0.2), ai("vm",0.8,1.0)],
            para: vec![],
            protected: vec!["qemu"],
            expect_orig: "QEMU",
            expect_qwen: "",
            expect_clean: false,
        },
        TestCase {
            name: "Phantom word in range → discard",
            desc: "A degenerate 1ms word inside the term's range — aligner garbage",
            term: "jiff",
            orig: vec![ai("the",0.30,0.45), ai("jiff",0.48,0.80), ai("crate",0.9,1.2)],
            qwen: vec![ai("the",0.30,0.45), ai("the",0.48,0.481), ai("JIF",0.50,0.80), ai("crate",0.9,1.2)],
            para: vec![],
            protected: vec!["jiff"],
            expect_orig: "",
            expect_qwen: "",
            expect_clean: false,
        },
        TestCase {
            name: "Overlapping timestamps → discard",
            desc: "Two words overlap in time on the same lane — aligner confused",
            term: "jiff",
            orig: vec![ai("the",0.30,0.45), ai("jiff",0.48,0.80), ai("crate",0.9,1.2)],
            qwen: vec![ai("the",0.30,0.45), ai("JIF",0.48,0.75), ai("F",0.60,0.80), ai("crate",0.9,1.2)],
            para: vec![],
            protected: vec!["jiff"],
            expect_orig: "",
            expect_qwen: "",
            expect_clean: false,
        },
        TestCase {
            name: "Punctuation stripping",
            desc: "Trailing punctuation on extracted words should be stripped",
            term: "rustc",
            orig: vec![ai("is",0.0,0.2), ai("rustc,",0.3,0.7), ai("which",0.8,1.1)],
            qwen: vec![ai("is",0.0,0.2), ai("Rust",0.3,0.5), ai("C,",0.5,0.7), ai("which",0.8,1.1)],
            para: vec![],
            protected: vec!["rustc"],
            expect_orig: "rustc",
            expect_qwen: "Rust C",
            expect_clean: true,
        },
        TestCase {
            name: "Two-lane trimming (no Parakeet)",
            desc: "Single-lane mode: trimming works with just orig + qwen",
            term: "serde",
            orig: vec![ai("the",0.0,0.2), ai("serde",0.3,0.7), ai("lib",0.8,1.0), ai("is",1.1,1.3)],
            qwen: vec![ai("the",0.0,0.2), ai("sturdy",0.3,0.7), ai("lib",0.8,1.0), ai("is",1.1,1.3)],
            para: vec![],
            protected: vec!["serde"],
            expect_orig: "serde",
            expect_qwen: "sturdy",
            expect_clean: true,
        },
        TestCase {
            name: "Three-lane requires all match to trim",
            desc: "Right edge: orig+qwen match but parakeet differs → no trim",
            term: "JIT",
            orig: vec![ai("the",0.0,0.2), ai("JIT",0.3,0.6), ai("for",0.7,0.9)],
            qwen: vec![ai("the",0.0,0.2), ai("jiff",0.3,0.6), ai("for",0.7,0.9)],
            para: vec![ai("the",0.0,0.2), ai("jit",0.3,0.6), ai("four",0.7,0.9)],
            protected: vec!["jit"],
            expect_orig: "JIT for",
            expect_qwen: "jiff for",
            expect_clean: true,
        },
        TestCase {
            name: "Identity match (not a mistake)",
            desc: "Qwen heard it correctly — stored as correct, not a mistake",
            term: "tokio",
            orig: vec![ai("use",0.0,0.2), ai("tokio",0.3,0.7), ai("runtime",0.8,1.2)],
            qwen: vec![ai("use",0.0,0.2), ai("tokio",0.3,0.7), ai("runtime",0.8,1.2)],
            para: vec![],
            protected: vec!["tokio"],
            expect_orig: "tokio",
            expect_qwen: "tokio",
            expect_clean: true,
        },
        // ==================== Messy / real-world cases ====================
        TestCase {
            name: "Word split into 3 tokens on Qwen",
            desc: "\"bincode\" heard as \"Bin\"+\"code\"+\"is\" — Qwen spreads it wider than Original",
            term: "bincode",
            orig: vec![ai("use",0.0,0.3), ai("bincode",0.4,0.9), ai("for",1.0,1.2), ai("that",1.3,1.5)],
            qwen: vec![ai("use",0.0,0.3), ai("Bin",0.4,0.6), ai("code",0.6,0.8), ai("is",0.85,1.0), ai("for",1.0,1.2), ai("that",1.3,1.5)],
            para: vec![],
            protected: vec!["bincode"],
            expect_orig: "bincode",
            expect_qwen: "Bin code is",
            expect_clean: true,
        },
        TestCase {
            name: "Term appears twice — only target matched",
            desc: "\"async\" appears at 0.5s and 2.5s; the term range points at the first one",
            term: "async",
            orig: vec![ai("the",0.0,0.2), ai("async",0.5,0.9), ai("fn",1.0,1.2), ai("calls",1.3,1.6), ai("async",2.5,2.9), ai("code",3.0,3.3)],
            qwen: vec![ai("the",0.0,0.2), ai("a",0.5,0.65), ai("sink",0.65,0.9), ai("fn",1.0,1.2), ai("calls",1.3,1.6), ai("a",2.5,2.65), ai("sync",2.65,2.9), ai("code",3.0,3.3)],
            para: vec![],
            protected: vec!["async"],
            expect_orig: "async",
            expect_qwen: "a sink",
            expect_clean: true,
        },
        TestCase {
            name: "Lanes completely diverge",
            desc: "Qwen heard a totally different sentence — no matching boundaries",
            term: "reqwest",
            orig: vec![ai("use",0.0,0.3), ai("reqwest",0.4,0.8), ai("for",0.9,1.1), ai("http",1.2,1.5)],
            qwen: vec![ai("the",0.0,0.4), ai("request",0.5,0.9), ai("was",1.0,1.2), ai("denied",1.3,1.8)],
            para: vec![],
            protected: vec!["reqwest"],
            // No matching boundaries → falls back to term range, but trimming may differ
            expect_orig: "reqwest",
            expect_qwen: "request",
            expect_clean: false, // no bi-boundaries found
        },
        TestCase {
            name: "Qwen token wider than Original",
            desc: "Qwen's \"Fasterthanlime\" starts before and ends after Original's \"fasterthanlime\"",
            term: "fasterthanlime",
            orig: vec![ai("by",0.0,0.2), ai("fasterthanlime",0.35,1.0), ai("is",1.1,1.3)],
            qwen: vec![ai("by",0.0,0.2), ai("Fasterthanlime",0.3,1.05), ai("is",1.1,1.3)],
            para: vec![],
            protected: vec!["fasterthanlime"],
            expect_orig: "fasterthanlime",
            expect_qwen: "Fasterthanlime",
            expect_clean: true,
        },
        TestCase {
            name: "Term at sentence end — no right context",
            desc: "Nothing after the term; extraction should still work with left context only",
            term: "repr",
            orig: vec![ai("use",0.0,0.2), ai("the",0.3,0.5), ai("repr",0.6,1.0)],
            qwen: vec![ai("use",0.0,0.2), ai("the",0.3,0.5), ai("rep",0.6,0.85), ai("r",0.85,1.0)],
            para: vec![],
            protected: vec!["repr"],
            expect_orig: "repr",
            expect_qwen: "rep r",
            expect_clean: true,
        },
        TestCase {
            name: "Dense touching boxes — no gaps anywhere",
            desc: "All words touch each other, no silence gaps; gap expansion should not trigger",
            term: "tokio",
            orig: vec![ai("use",0.0,0.20), ai("tokio",0.20,0.50), ai("spawn",0.50,0.80), ai("for",0.80,1.00)],
            qwen: vec![ai("use",0.0,0.20), ai("Tokyo",0.20,0.50), ai("spawn",0.50,0.80), ai("for",0.80,1.00)],
            para: vec![],
            protected: vec!["tokio"],
            expect_orig: "tokio",
            expect_qwen: "Tokyo",
            expect_clean: true,
        },
        TestCase {
            name: "Large gap — Qwen fills gap with extra words",
            desc: "Big gap around term; Qwen has extra words filling the silence",
            term: "lldb",
            orig: vec![ai("run",0.0,0.3), ai("lldb",0.6,1.0), ai("now",1.4,1.7)],
            qwen: vec![ai("run",0.0,0.3), ai("the",0.45,0.55), ai("L",0.6,0.75), ai("LDB",0.8,1.0), ai("tool",1.1,1.3), ai("now",1.4,1.7)],
            para: vec![],
            protected: vec!["lldb"],
            // Gap expansion: prev_end=0.3, start=0.6, gap=0.3s > 50ms → expand to 0.3
            // next_start=1.4, end=1.0, gap=0.4s > 50ms → expand to 1.4
            // So range is 0.3–1.4, captures "the L LDB tool" on Qwen, "run" trimmed, "now" trimmed
            expect_orig: "lldb",
            expect_qwen: "the L LDB tool",
            expect_clean: true,
        },
    ];

    for tc in &cases {
        let protected: std::collections::HashSet<String> =
            tc.protected.iter().map(|s| s.to_lowercase()).collect();

        // Simulate gap expansion (same logic as run_corpus_pass)
        let term_lower = tc.term.to_lowercase();
        let term_range = find_term_time_range(&tc.orig, &term_lower);
        let (term_start, term_end) = match term_range {
            Some((start, end)) => {
                const MIN_GAP: f64 = 0.05;
                let prev_end = tc
                    .orig
                    .iter()
                    .filter(|a| a.end_time < start - 0.001)
                    .last()
                    .map(|a| a.end_time)
                    .unwrap_or(start);
                let expanded_start = if start - prev_end > MIN_GAP {
                    prev_end
                } else {
                    start
                };
                let next_start = tc
                    .orig
                    .iter()
                    .find(|a| a.start_time > end + 0.001)
                    .map(|a| a.start_time)
                    .unwrap_or(end);
                let expanded_end = if next_start - end > MIN_GAP {
                    next_start
                } else {
                    end
                };
                (expanded_start, expanded_end)
            }
            None => (0.0, 10.0),
        };

        let extraction = extract_with_consensus(
            &tc.orig, &tc.qwen, &tc.para, term_start, term_end, &protected,
        );
        let is_mistake = extraction.original.to_lowercase() != extraction.qwen.to_lowercase();

        let pass = if !tc.expect_clean {
            // For "should discard" cases, only check that clean is false
            !extraction.clean
        } else {
            extraction.original == tc.expect_orig
                && extraction.qwen == tc.expect_qwen
                && extraction.clean
        };

        results.push(serde_json::json!({
            "name": tc.name,
            "desc": tc.desc,
            "term": tc.term,
            "pass": pass,
            "term_range": [term_start, term_end],
            "extraction": {
                "original": extraction.original,
                "qwen": extraction.qwen,
                "parakeet": extraction.parakeet,
                "clean": extraction.clean,
                "cons_range": [extraction.cons_range.0, extraction.cons_range.1],
            },
            "expected": {
                "original": tc.expect_orig,
                "qwen": tc.expect_qwen,
                "clean": tc.expect_clean,
            },
            "is_mistake": is_mistake,
            "trim_info": extraction.trim_info,
            "alignments": {
                "original": fmt_align_json(&tc.orig),
                "qwen": fmt_align_json(&tc.qwen),
                "parakeet": fmt_align_json(&tc.para),
            },
        }));
    }

    let all_pass = results.iter().all(|r| r["pass"].as_bool().unwrap_or(false));
    Ok(Json(serde_json::json!({
        "all_pass": all_pass,
        "total": results.len(),
        "passed": results.iter().filter(|r| r["pass"].as_bool().unwrap_or(false)).count(),
        "tests": results,
    }))
    .into_response())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prototype::{
        AcceptedProposal, AcousticSegment, PrototypeCandidate, PrototypeCorrectionResult,
        PrototypeSpanProposal, SentenceCandidate,
    };
    use qwen3_asr::ForcedAlignItem;

    fn item(word: &str, start: f64, end: f64) -> ForcedAlignItem {
        ForcedAlignItem {
            word: word.to_string(),
            start_time: start,
            end_time: end,
        }
    }

    fn protected(terms: &[&str]) -> std::collections::HashSet<String> {
        terms.iter().map(|s| s.to_lowercase()).collect()
    }

    fn seg(phone: &str, start: f64, end: f64) -> AcousticSegment {
        AcousticSegment {
            phone: phone.to_string(),
            start_sec: start,
            end_sec: end,
        }
    }

    fn grouped_words(json: serde_json::Value) -> Vec<serde_json::Value> {
        json.as_array().cloned().unwrap_or_default()
    }

    fn timed(text: &str, start: f64, end: f64) -> TimedWord {
        TimedWord {
            text: text.to_string(),
            start,
            end,
            confidence: None,
        }
    }

    fn timed_conf(text: &str, start: f64, end: f64, confidence: f32) -> TimedWord {
        TimedWord {
            text: text.to_string(),
            start,
            end,
            confidence: Some(confidence),
        }
    }

    fn empty_result() -> PrototypeCorrectionResult {
        PrototypeCorrectionResult {
            original: String::new(),
            corrected: String::new(),
            accepted: Vec::new(),
            proposals: Vec::new(),
            sentence_candidates: Vec::new(),
        }
    }

    fn target_candidate(term: &str) -> PrototypeCandidate {
        PrototypeCandidate {
            term: term.to_string(),
            via: "spoken".to_string(),
            matched_form: term.to_string(),
            matched_form_phonemes: None,
            term_preview: Some(term.to_string()),
            term_preview_phonemes: None,
            score: 0.9,
            lexical_score: None,
            dice: None,
            prefix_ratio: None,
            length_ratio: None,
            phonetic_score: Some(0.9),
            observed_acoustic_score: None,
            acoustic_score: None,
            acoustic_delta: None,
            phonemes: None,
            exact_words: false,
            exact_compact: false,
        }
    }

    fn proposal_for(term: &str) -> PrototypeSpanProposal {
        PrototypeSpanProposal {
            token_start: 0,
            token_end: 1,
            char_start: 0,
            char_end: 0,
            raw_text: "raw".to_string(),
            normalized: "raw".to_string(),
            phonemes: None,
            acoustic_phonemes: None,
            parakeet_confidence: None,
            observed_acoustic_score: None,
            acoustic_trustworthy: false,
            acoustic_window_start_sec: None,
            acoustic_window_end_sec: None,
            candidates: vec![target_candidate(term)],
        }
    }

    fn sentence_candidate(text: &str) -> SentenceCandidate {
        SentenceCandidate {
            label: "cand".to_string(),
            text: text.to_string(),
            edits: Vec::new(),
            score: 0.8,
        }
    }

    fn bakeoff_item(case_id: &str, term: &str) -> PrototypeBakeoffItem {
        PrototypeBakeoffItem {
            case_id: case_id.to_string(),
            term: term.to_string(),
            qwen: format!("{term} qwen"),
            expected: format!("{term} expected"),
            hit_count: 1,
            recording_id: None,
            wav_path: None,
            template_sentence: None,
            qwen_fragment: None,
            expected_fragment: Some(term.to_string()),
        }
    }

    #[test]
    fn ownership_ranges_handle_zero_width_alignment_items() {
        let alignment = vec![
            item("the", 1.12, 1.20),
            item("root,", 1.60, 1.60),
            item("run", 1.76, 2.00),
        ];

        let ownership = alignment_group_ownership_ranges(&alignment);
        assert_eq!(ownership.len(), 3);

        assert!(ownership[0].0 <= ownership[0].1);
        assert!(ownership[1].0 <= ownership[1].1);
        assert!(ownership[2].0 <= ownership[2].1);

        assert!(ownership[0].1 <= ownership[1].0 + 1e-9);
        assert!(ownership[1].1 <= ownership[2].0 + 1e-9);

        assert!(ownership[1].1 - ownership[1].0 > 0.0);
        assert!(ownership[1].0 <= 1.60 && ownership[1].1 >= 1.60);
    }

    #[test]
    fn grouped_phone_segments_assign_each_segment_to_exactly_one_word() {
        let alignment = vec![
            item("dash", 2.72, 2.96),
            item("dash", 2.96, 3.12),
            item("hidden", 3.12, 3.44),
        ];
        let segments = vec![
            seg("d", 2.754, 2.834),
            seg("æ", 2.834, 2.894),
            seg("ʃ", 2.894, 2.955),
            seg("d", 2.955, 3.035),
            seg("æ", 3.035, 3.095),
            seg("ʃ", 3.095, 3.196),
        ];

        let grouped = grouped_words(group_phone_segments_by_alignment_json(&alignment, &segments));
        assert_eq!(grouped.len(), 3);

        let counts = grouped
            .iter()
            .map(|row| row.get("n").and_then(|v| v.as_u64()).unwrap_or(0))
            .collect::<Vec<_>>();
        assert_eq!(counts, vec![3, 3, 0]);

        let total_assigned = counts.iter().sum::<u64>();
        assert_eq!(total_assigned, segments.len() as u64);
    }

    #[test]
    fn grouped_phone_segments_do_not_smear_boundary_phone_across_neighbors() {
        let alignment = vec![
            item("you", 0.48, 0.56),
            item("want", 0.56, 0.64),
            item("to", 0.64, 0.72),
        ];
        let segments = vec![
            seg("iɪ", 0.5226, 0.5628),
            seg("w", 0.5628, 0.6030),
            seg("ʌ", 0.6030, 0.6633),
        ];

        let grouped = grouped_words(group_phone_segments_by_alignment_json(&alignment, &segments));
        let phones_for = |idx: usize| {
            grouped[idx]
                .get("phones")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .filter_map(|row| row.get("p").and_then(|v| v.as_str()).map(str::to_string))
                .collect::<Vec<_>>()
        };

        assert_eq!(phones_for(0), vec!["iɪ".to_string()]);
        assert_eq!(phones_for(1), vec!["w".to_string(), "ʌ".to_string()]);
        assert!(phones_for(2).is_empty());
    }

    #[test]
    fn first_voiced_zipa_start_ignores_placeholder_segments() {
        let segments = vec![
            seg("__", 0.00, 0.08),
            seg("▁", 0.08, 0.11),
            seg("blank", 0.11, 0.15),
            seg("w", 0.15, 0.22),
        ];

        let start = first_voiced_zipa_start_sec(&segments).expect("voiced segment start");
        assert!((start - 0.15).abs() < 1e-9, "{start}");
    }

    #[test]
    fn find_term_span_matches_split_term_words() {
        let words = vec!["bear".to_string(), "cove".to_string(), "company".to_string()];
        assert_eq!(find_term_span_in_tokenized_words(&words, "bearcove"), Some((0, 2)));
    }

    #[test]
    fn find_term_span_matches_compacted_alphanumeric_term() {
        let words = vec!["we".to_string(), "can".to_string(), "pack".to_string(), "u8".to_string()];
        assert_eq!(find_term_span_in_tokenized_words(&words, "u8"), Some((3, 4)));

        let words = vec!["a".to_string(), "arch".to_string(), "64".to_string()];
        assert_eq!(find_term_span_in_tokenized_words(&words, "AArch64"), Some((0, 3)));
    }

    #[test]
    fn parakeet_timing_maps_one_to_one_words() {
        let (mapped, confidence) = parakeet_words_to_transcript_timing(
            "bear cove",
            &[timed("bear", 1.00, 1.24), timed("cove", 1.24, 1.52)],
        )
        .expect("expected timing map");

        assert_eq!(mapped.len(), 2);
        assert_eq!(mapped[0].word, "bear");
        assert!((mapped[0].start_time - 1.00).abs() < 1e-6);
        assert!((mapped[0].end_time - 1.24).abs() < 1e-6);
        assert_eq!(mapped[1].word, "cove");
        assert!((mapped[1].start_time - 1.24).abs() < 1e-6);
        assert!((mapped[1].end_time - 1.52).abs() < 1e-6);
        assert_eq!(confidence, vec![None, None]);
    }

    #[test]
    fn parakeet_timing_merges_two_words_into_one_qwen_token() {
        let (mapped, confidence) = parakeet_words_to_transcript_timing(
            "bearcove",
            &[timed("bear", 1.00, 1.22), timed("cove", 1.22, 1.54)],
        )
        .expect("expected timing map");

        assert_eq!(mapped.len(), 1);
        assert_eq!(mapped[0].word, "bearcove");
        assert!((mapped[0].start_time - 1.00).abs() < 1e-6);
        assert!((mapped[0].end_time - 1.54).abs() < 1e-6);
        assert_eq!(confidence, vec![None]);
    }

    #[test]
    fn parakeet_timing_splits_one_word_across_two_qwen_tokens() {
        let (mapped, confidence) =
            parakeet_words_to_transcript_timing("bear cove", &[timed("bearcove", 2.00, 2.64)])
            .expect("expected timing map");

        assert_eq!(mapped.len(), 2);
        assert_eq!(mapped[0].word, "bear");
        assert_eq!(mapped[1].word, "cove");
        assert!((mapped[0].start_time - 2.00).abs() < 1e-6);
        assert!(mapped[0].end_time > mapped[0].start_time);
        assert!((mapped[1].end_time - 2.64).abs() < 1e-6);
        assert!((mapped[0].end_time - mapped[1].start_time).abs() < 1e-6);
        assert_eq!(confidence, vec![None, None]);
    }

    #[test]
    fn parakeet_timing_preserves_confidence_on_direct_matches() {
        let (_, confidence) = parakeet_words_to_transcript_timing(
            "bear cove",
            &[
                timed_conf("bear", 1.00, 1.24, 0.91),
                timed_conf("cove", 1.24, 1.52, 0.77),
            ],
        )
        .expect("expected timing map");

        assert_eq!(confidence, vec![Some(0.91), Some(0.77)]);
    }

    #[test]
    fn parakeet_timing_averages_confidence_when_merging_words() {
        let (_, confidence) = parakeet_words_to_transcript_timing(
            "bearcove",
            &[
                timed_conf("bear", 1.00, 1.22, 0.80),
                timed_conf("cove", 1.22, 1.54, 0.60),
            ],
        )
        .expect("expected timing map");

        assert_eq!(confidence.len(), 1);
        assert!((confidence[0].unwrap_or_default() - 0.70).abs() < 1e-6, "{confidence:?}");
    }

    #[test]
    fn espeak_zipa_dp_maps_bear_cove_from_fuzzy_house_ipa() {
        let qwen_words = vec!["bear".to_string(), "cove".to_string()];
        let qwen_ipa = vec![
            crate::prototype::parse_house_ipa("bˈeə"),
            crate::prototype::parse_house_ipa("kˈəʊv"),
        ];
        let zipa_segments = vec![
            seg("b", 1.00, 1.06),
            seg("ɛ", 1.06, 1.14),
            seg("ɹ", 1.14, 1.20),
            seg("k", 1.20, 1.27),
            seg("o", 1.27, 1.35),
            seg("ʊ", 1.35, 1.42),
            seg("v", 1.42, 1.48),
        ];

        let mapped = transcript_ipa_words_to_zipa_timing(&qwen_words, &qwen_ipa, &zipa_segments)
            .expect("expected eSpeak/ZIPA timing map");

        assert_eq!(mapped.len(), 2);
        assert_eq!(mapped[0].word, "bear");
        assert_eq!(mapped[1].word, "cove");
        assert!((mapped[0].start_time - 1.00).abs() < 1e-6);
        assert!((mapped[0].end_time - 1.20).abs() < 1e-6);
        assert!((mapped[1].start_time - 1.20).abs() < 1e-6);
        assert!((mapped[1].end_time - 1.48).abs() < 1e-6);
    }

    #[test]
    fn espeak_zipa_dp_handles_sir_day_style_vowel_drift() {
        let qwen_words = vec!["sir".to_string(), "day".to_string()];
        let qwen_ipa = vec![
            crate::prototype::parse_house_ipa("sˈɜː"),
            crate::prototype::parse_house_ipa("dˈeɪ"),
        ];
        let zipa_segments = vec![
            seg("s", 0.50, 0.56),
            seg("ɚ", 0.56, 0.66),
            seg("d", 0.66, 0.73),
            seg("ɛ", 0.73, 0.81),
            seg("ɪ", 0.81, 0.90),
        ];

        let mapped = transcript_ipa_words_to_zipa_timing(&qwen_words, &qwen_ipa, &zipa_segments)
            .expect("expected fuzzy eSpeak/ZIPA timing map");

        assert_eq!(mapped.len(), 2);
        assert_eq!(mapped[0].word, "sir");
        assert_eq!(mapped[1].word, "day");
        assert!((mapped[0].start_time - 0.50).abs() < 1e-6);
        assert!((mapped[0].end_time - 0.66).abs() < 1e-6);
        assert!((mapped[1].start_time - 0.66).abs() < 1e-6);
        assert!((mapped[1].end_time - 0.90).abs() < 1e-6);
    }

    #[test]
    fn slice_samples_16k_uses_absolute_second_bounds() {
        let samples = (0..32_000).map(|n| n as f32).collect::<Vec<_>>();
        let slice = slice_samples_16k(&samples, 0.25, 0.50);
        assert_eq!(slice.len(), 4_000);
        assert_eq!(slice.first().copied(), Some(4_000.0));
        assert_eq!(slice.last().copied(), Some(7_999.0));
    }

    #[test]
    fn offset_trace_segments_maps_local_times_back_to_original_audio() {
        let trace = serde_json::json!({
            "segments": [
                {"phone": "m", "start_sec": 0.0, "end_sec": 0.05, "avg_logprob": -0.1},
                {"phone": "ɪ", "start_sec": 0.05, "end_sec": 0.10, "avg_logprob": -0.2}
            ]
        });
        let segments = offset_trace_segments(&trace, 1.25).expect("offset ok");
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].phone, "m");
        assert!((segments[0].start_sec - 1.25).abs() < 1e-6);
        assert!((segments[0].end_sec - 1.30).abs() < 1e-6);
        assert_eq!(segments[1].phone, "ɪ");
        assert!((segments[1].start_sec - 1.30).abs() < 1e-6);
        assert!((segments[1].end_sec - 1.35).abs() < 1e-6);
    }

    #[test]
    fn eval_analysis_marks_no_proposal() {
        let result = empty_result();
        let analysis = analyze_prototype_eval_row(
            &result,
            "MIR",
            Some("MIR"),
            "show up in MIR before lowering",
            &HashMap::new(),
        );
        assert_eq!(analysis.failure_reason, "no_proposal");
        assert!(!analysis.target_proposed);
        assert!(!analysis.target_sentence_candidate);
    }

    #[test]
    fn eval_analysis_marks_proposal_found_without_sentence_candidate() {
        let mut result = empty_result();
        result.proposals.push(proposal_for("regalloc"));
        let analysis = analyze_prototype_eval_row(
            &result,
            "regalloc",
            Some("regalloc"),
            "our regalloc isn't the best",
            &HashMap::new(),
        );
        assert_eq!(analysis.failure_reason, "proposal_found_no_sentence_edit");
        assert!(analysis.target_proposed);
        assert!(!analysis.target_sentence_candidate);
    }

    #[test]
    fn eval_analysis_marks_reranker_missing_target_candidate() {
        let mut result = empty_result();
        result.proposals.push(proposal_for("Bear Cove"));
        result
            .sentence_candidates
            .push(sentence_candidate("talk about the Bear Cove Company"));
        result.corrected = "talk about the Bercove Company".to_string();
        let analysis = analyze_prototype_eval_row(
            &result,
            "Bear Cove",
            Some("Bear Cove"),
            "talk about the Bear Cove Company",
            &HashMap::new(),
        );
        assert_eq!(analysis.failure_reason, "reranker_missed_target_candidate");
        assert!(analysis.target_proposed);
        assert!(analysis.target_sentence_candidate);
        assert!(!analysis.target_ok);
    }

    #[test]
    fn eval_analysis_marks_target_only_when_term_fixed_but_sentence_not_exact() {
        let mut result = empty_result();
        result.corrected = "run ripgrep with dash dash hidden".to_string();
        result.accepted.push(AcceptedProposal {
            token_start: 0,
            token_end: 1,
            char_start: 0,
            char_end: 0,
            from: "rip grip".to_string(),
            matched_form: "rip grip".to_string(),
            from_phonemes: None,
            to: "ripgrep".to_string(),
            to_phonemes: None,
            via: "spoken".to_string(),
            score: 0.8,
            acoustic_score: None,
            acoustic_delta: None,
        });
        let analysis = analyze_prototype_eval_row(
            &result,
            "ripgrep",
            Some("ripgrep"),
            "run ripgrep with --hidden",
            &HashMap::new(),
        );
        assert_eq!(analysis.failure_reason, "target_only");
        assert!(analysis.target_accepted_edit);
        assert!(analysis.target_ok);
        assert!(!analysis.exact_ok);
    }

    #[test]
    fn human_case_id_prefers_recording_id() {
        let case_id = prototype_bakeoff_case_id("human", Some(42), "MIR", "mere", "show MIR");
        assert_eq!(case_id, "hum-42");
    }

    #[test]
    fn nonhuman_case_id_is_stable() {
        let left = prototype_bakeoff_case_id("applied", None, "ripgrep", "rip grip", "ripgrep");
        let right = prototype_bakeoff_case_id("applied", None, "ripgrep", "rip grip", "ripgrep");
        assert_eq!(left, right);
        assert!(left.starts_with("app-"));
    }

    #[test]
    fn human_shuffle_is_seeded_and_repeatable() {
        let original = vec![
            bakeoff_item("a", "alpha"),
            bakeoff_item("b", "beta"),
            bakeoff_item("c", "gamma"),
            bakeoff_item("d", "delta"),
        ];

        let mut left = original.clone();
        let mut right = original.clone();
        shuffle_human_bakeoff_items(&mut left, 7);
        shuffle_human_bakeoff_items(&mut right, 7);

        let left_ids = left.iter().map(|item| item.case_id.clone()).collect::<Vec<_>>();
        let right_ids = right
            .iter()
            .map(|item| item.case_id.clone())
            .collect::<Vec<_>>();
        assert_eq!(left_ids, right_ids);
        assert_ne!(
            left_ids,
            original
                .iter()
                .map(|item| item.case_id.clone())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn trim_two_lane_strips_matching_edges() {
        // "reqwest for" vs "requests for" — "for" should be trimmed from right
        let orig = vec![item("reqwest", 1.0, 1.3), item("for", 1.3, 1.5)];
        let qwen = vec![item("requests", 1.0, 1.3), item("for", 1.3, 1.5)];
        let para = vec![]; // single-lane

        let (o, q, p, _ti) = trim_matching_edges(&orig, &qwen, &para, 0.0, 2.0, &protected(&[]));
        assert_eq!(o, "reqwest");
        assert_eq!(q, "requests");
        assert_eq!(p, "");
    }

    #[test]
    fn trim_two_lane_strips_both_edges() {
        // "the reqwest for" vs "the requests for" — trim "the" from left and "for" from right
        let orig = vec![
            item("the", 0.5, 0.7),
            item("reqwest", 1.0, 1.3),
            item("for", 1.3, 1.5),
        ];
        let qwen = vec![
            item("the", 0.5, 0.7),
            item("requests", 1.0, 1.3),
            item("for", 1.3, 1.5),
        ];
        let para = vec![];

        let (o, q, _, _ti) = trim_matching_edges(&orig, &qwen, &para, 0.0, 2.0, &protected(&[]));
        assert_eq!(o, "reqwest");
        assert_eq!(q, "requests");
    }

    #[test]
    fn trim_protects_vocab_terms() {
        // "async for" vs "a sync for" — "async" is protected, don't trim even though left doesn't match
        // Actually test: "for async" vs "for a sync" — "for" matches but "async" is protected on left
        let orig = vec![
            item("for", 0.5, 0.7),
            item("async", 1.0, 1.3),
            item("stuff", 1.3, 1.5),
        ];
        let qwen = vec![
            item("for", 0.5, 0.7),
            item("a", 1.0, 1.1),
            item("sync", 1.1, 1.3),
            item("stuff", 1.3, 1.5),
        ];
        let para = vec![];

        let (o, q, _, _ti) =
            trim_matching_edges(&orig, &qwen, &para, 0.0, 2.0, &protected(&["async"]));
        // "for" on left would match, but next word is "async" which is protected → stop
        // Wait, "for" itself is not protected, so it should be trimmed. The protection check
        // is on the word being considered for trimming.
        // "for" is not protected → trim it. Then "async" is protected → stop.
        // "stuff" matches on right → trim it.
        assert_eq!(o, "async");
        assert_eq!(q, "a sync");
    }

    #[test]
    fn trim_three_lane_requires_all_match() {
        let orig = vec![
            item("the", 0.5, 0.7),
            item("JIT", 1.0, 1.3),
            item("for", 1.3, 1.5),
        ];
        let qwen = vec![
            item("the", 0.5, 0.7),
            item("jiff", 1.0, 1.3),
            item("for", 1.3, 1.5),
        ];
        let para = vec![
            item("the", 0.5, 0.7),
            item("jit", 1.0, 1.3),
            item("four", 1.3, 1.5),
        ];

        let (o, q, pk, _ti) = trim_matching_edges(&orig, &qwen, &para, 0.0, 2.0, &protected(&[]));
        // Left: "the" matches all 3 → trimmed
        // Right: "for" vs "for" vs "four" → mismatch → not trimmed
        assert_eq!(o, "JIT for");
        assert_eq!(q, "jiff for");
        assert_eq!(pk, "jit four");
    }

    #[test]
    fn trim_keeps_at_least_one_word() {
        // All words match — should keep at least 1 word per lane
        let orig = vec![item("the", 0.5, 0.7), item("end", 0.7, 1.0)];
        let qwen = vec![item("the", 0.5, 0.7), item("end", 0.7, 1.0)];
        let para = vec![];

        let (o, q, _, _ti) = trim_matching_edges(&orig, &qwen, &para, 0.0, 2.0, &protected(&[]));
        // After left trim of "the", both have ["end"] — len is 1, stop
        assert_eq!(o, "end");
        assert_eq!(q, "end");
    }

    #[test]
    fn trim_respects_time_range() {
        // Items outside range should be excluded
        let orig = vec![
            item("before", 0.1, 0.3),
            item("JIT", 1.0, 1.3),
            item("after", 2.5, 2.8),
        ];
        let qwen = vec![
            item("before", 0.1, 0.3),
            item("jiff", 1.0, 1.3),
            item("after", 2.5, 2.8),
        ];
        let para = vec![];

        let (o, q, _, _ti) = trim_matching_edges(&orig, &qwen, &para, 0.5, 2.0, &protected(&[]));
        assert_eq!(o, "JIT");
        assert_eq!(q, "jiff");
    }

    #[test]
    fn trim_info_records_what_was_removed() {
        let orig = vec![
            item("the", 0.5, 0.7),
            item("reqwest", 1.0, 1.3),
            item("for", 1.3, 1.5),
            item("you", 1.5, 1.7),
        ];
        let qwen = vec![
            item("the", 0.5, 0.7),
            item("requests", 1.0, 1.3),
            item("for", 1.3, 1.5),
            item("you", 1.5, 1.7),
        ];
        let para = vec![];

        let (o, q, _, ti) = trim_matching_edges(&orig, &qwen, &para, 0.0, 2.0, &protected(&[]));
        assert_eq!(o, "reqwest");
        assert_eq!(q, "requests");
        assert_eq!(ti.trimmed_left, 1); // "the"
        assert_eq!(ti.trimmed_right, 2); // "for", "you"
        assert_eq!(ti.pre_orig, vec!["the", "reqwest", "for", "you"]);
        assert_eq!(ti.pre_qwen, vec!["the", "requests", "for", "you"]);
    }

    #[test]
    fn eval_match_allows_expected_phrase_inside_context() {
        let alt = HashMap::new();
        assert!(eval_fragment_matches(&alt, "kajit", "kajit", "while kajit"));
    }

    #[test]
    fn eval_match_rejects_similar_spelling_without_alt() {
        let alt = HashMap::new();
        assert!(!eval_fragment_matches(&alt, "tokio", "tokio", "while tokyo"));
    }

    #[test]
    fn eval_match_allows_alt_spelling_inside_context() {
        let mut alt = HashMap::new();
        alt.insert("tokio".to_string(), vec!["tokyo".to_string()]);
        assert!(eval_fragment_matches(&alt, "tokio", "tokio", "while tokyo"));
    }

    #[test]
    fn exact_fragment_finder_rejects_substrings_inside_larger_identifiers() {
        assert!(find_exact_fragment_ascii_ci(
            "If the schema changes, serde_json will fail.",
            "serde"
        )
        .is_none());
        assert!(find_exact_fragment_ascii_ci(
            "The failure should be easier to reproduce.",
            "repr"
        )
        .is_none());
    }

    #[test]
    fn exact_fragment_finder_allows_identifier_adjacent_punctuation() {
        assert_eq!(
            find_exact_fragment_ascii_ci(
                "The Dockerfile needs AArch64-compatible binaries.",
                "AArch64"
            ),
            Some("The Dockerfile needs ".len())
        );
        assert_eq!(
            find_exact_fragment_ascii_ci(
                "That enum should be repr(u8) for the C ABI.",
                "repr"
            ),
            Some("That enum should be ".len())
        );
    }
}
