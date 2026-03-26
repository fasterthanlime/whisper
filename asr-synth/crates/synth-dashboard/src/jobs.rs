use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;

use axum::{
    extract::State,
    response::{IntoResponse, Response},
    Json,
};
use rand::prelude::*;
use serde::Deserialize;

use crate::tts;
use crate::{err, AppError, AppState};
use parakeet_rs::Transcriber;

use std::sync::atomic::Ordering;

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
fn make_sentence_pair(
    chain: &MarkovChain,
    term: &WrittenTerm,
    spoken: &SpokenTerm,
    rng: &mut impl Rng,
) -> (WrittenSentence, SpokenSentence) {
    let sentence = chain.generate_with(&term.0, 15, rng);
    let spoken_sentence = {
        let lower = sentence.to_lowercase();
        let lower_term = term.0.to_lowercase();
        if let Some(pos) = lower.find(&lower_term) {
            format!("{}{}{}", &sentence[..pos], &spoken.0, &sentence[pos + term.0.len()..])
        } else {
            sentence.clone()
        }
    };
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
        state_q.asr.transcribe_samples(&samples_q, qwen3_asr::TranscribeOptions::default().with_language("english"))
            .map(|r| r.text)
            .unwrap_or_default()
    });

    let parakeet_full = if dual_asr {
        let state_p = state.clone();
        let samples_p = full_16k.to_vec();
        let parakeet_task = tokio::task::spawn_blocking(move || -> String {
            let mut p = state_p.parakeet.lock().unwrap();
            p.transcribe_samples(samples_p, 16000, 1, None)
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
    let orig_alignment = state.aligner.align(full_16k, &written.0).unwrap_or_default();
    let spoken_alignment = state.aligner.align(full_16k, &spoken.0).unwrap_or_default();

    // Find the WRITTEN term in the original alignment
    let term_lower = term.0.to_lowercase();
    let term_found_range = find_term_time_range(&orig_alignment, &term_lower);
    let (term_start, term_end) = term_found_range
        .unwrap_or((0.0, full_16k.len() as f64 / 16000.0));

    let qwen_alignment = state.aligner.align(full_16k, &qwen_full).unwrap_or_default();
    let parakeet_alignment = if dual_asr {
        state.aligner.align(full_16k, &parakeet_full).unwrap_or_default()
    } else {
        Vec::new()
    };

    tracing::debug!(term_found = term_found_range.is_some(), term_start, term_end, "alignment done");

    let extraction = extract_with_consensus(
        &orig_alignment, &qwen_alignment, &parakeet_alignment,
        term_start, term_end, protected_terms,
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
            "fuck", "fucking", "shit", "jesus", "christ", "damn", "idiot",
            "idk", "omg", "lol", "lmao", "rofl", "wtf", "stfu", "smh",
        ];

        for sentence in sentences {
            let words: Vec<&str> = sentence.split_whitespace().collect();
            if words.len() < 3 { continue; }

            // Forward: <S> → w0 → w1 → ... → </S>
            let mut prev = SENTENCE_START.to_string();
            for &w in &words {
                let lower = w.to_lowercase();
                let clean: String = lower.chars().filter(|c| c.is_alphanumeric()).collect();
                if BANNED.contains(&clean.as_str()) { continue; }
                *fwd_counts.entry(prev.clone()).or_default().entry(lower.clone()).or_default() += 1;
                *bwd_counts.entry(lower.clone()).or_default().entry(prev).or_default() += 1;
                prev = lower;
            }
            *fwd_counts.entry(prev).or_default().entry(SENTENCE_END.to_string()).or_default() += 1;
        }

        let to_vec = |m: HashMap<String, HashMap<String, u32>>| -> HashMap<String, Vec<(String, u32)>> {
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
        if choices.is_empty() { return None; }
        let total: u32 = choices.iter().map(|(_, c)| c).sum();
        let mut pick = rng.random_range(0..total);
        for (word, count) in choices {
            if pick < *count { return Some(word.clone()); }
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
            let Some(choices) = self.forward.get(&cur) else { break };
            let Some(next) = Self::sample(choices, rng) else { break };
            if next == SENTENCE_END { break; }
            forward_words.push(next.clone());
            cur = next;
        }

        // Build backward from target until sentence start
        let mut backward_words = Vec::new();
        cur = target_lower.clone();
        for _ in 0..max_words {
            let Some(choices) = self.backward.get(&cur) else { break };
            let Some(prev) = Self::sample(choices, rng) else { break };
            if prev == SENTENCE_START { break; }
            backward_words.push(prev.clone());
            cur = prev;
        }
        backward_words.reverse();

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
        if spoken == term { continue; }
        let lower_result = result.to_lowercase();
        let lower_term = term.to_lowercase();
        if let Some(pos) = lower_result.find(&lower_term) {
            result = format!("{}{}{}", &result[..pos], spoken, &result[pos + term.len()..]);
        }
    }
    result
}

/// Find the time range of a spoken term in alignment items.
/// Handles multi-word spoken forms (e.g., "sir day" for serde).
fn find_term_time_range(
    align_items: &[qwen3_asr::ForcedAlignItem],
    spoken_term_lower: &str,
) -> Option<(f64, f64)> {
    let target_clean: String = spoken_term_lower.chars()
        .filter(|c| c.is_alphanumeric() || *c == ' ')
        .collect();

    for i in 0..align_items.len() {
        let mut concat = String::new();
        for j in i..align_items.len().min(i + 5) {
            if !concat.is_empty() { concat.push(' '); }
            concat.push_str(&align_items[j].word.to_lowercase());
            let concat_clean: String = concat.chars()
                .filter(|c| c.is_alphanumeric() || *c == ' ')
                .collect();
            if concat_clean.trim() == target_clean.trim() {
                return Some((align_items[i].start_time, align_items[j].end_time));
            }
        }
    }
    None
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
    items.iter().rev()
        .find(|a| a.end_time <= time + eps && a.end_time >= time - eps)
        .or_else(|| items.iter().rev().find(|a| a.end_time <= time + eps))
        .map(|a| a.word.clone())
}

/// Find the word starting just after `time` in an alignment.
fn word_after(items: &[qwen3_asr::ForcedAlignItem], time: f64, eps: f64) -> Option<String> {
    items.iter()
        .find(|a| a.start_time >= time - eps && a.start_time <= time + eps)
        .or_else(|| items.iter().find(|a| a.start_time >= time - eps))
        .map(|a| a.word.clone())
}

/// Compute annotated tri-boundaries.
fn compute_tri_boundaries(
    orig_b: &[f64], qwen_b: &[f64], para_b: &[f64],
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

    times.into_iter().map(|t| {
        TriBoundary {
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
        }
    }).collect()
}

/// Extract words from an alignment that START within [start, end).
fn words_in_range(items: &[qwen3_asr::ForcedAlignItem], start: f64, end: f64) -> String {
    items.iter()
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
}

/// Trim matching words from alignment item slices.
/// Returns (orig, qwen, parakeet) as trimmed word strings.
/// Stops if the original lane's word is a known vocab term (in `protected`).
fn trim_matching_edges(
    orig: &[qwen3_asr::ForcedAlignItem],
    qwen: &[qwen3_asr::ForcedAlignItem],
    para: &[qwen3_asr::ForcedAlignItem],
    start: f64, end: f64,
    protected: &std::collections::HashSet<String>,
) -> (String, String, String, TrimInfo) {
    let mut o: Vec<&str> = orig.iter().filter(|a| a.start_time >= start && a.start_time < end).map(|a| a.word.as_str()).collect();
    let mut q: Vec<&str> = qwen.iter().filter(|a| a.start_time >= start && a.start_time < end).map(|a| a.word.as_str()).collect();
    let mut p: Vec<&str> = para.iter().filter(|a| a.start_time >= start && a.start_time < end).map(|a| a.word.as_str()).collect();
    let has_para = !p.is_empty();

    let pre_orig: Vec<String> = o.iter().map(|s| s.to_string()).collect();
    let pre_qwen: Vec<String> = q.iter().map(|s| s.to_string()).collect();
    let pre_para: Vec<String> = p.iter().map(|s| s.to_string()).collect();

    let mut trimmed_left = 0;
    while o.len() > 1 && q.len() > 1 && (!has_para || p.len() > 1) {
        let ow = o[0].to_lowercase();
        if protected.contains(&ow) { break; }
        let qw = q[0].to_lowercase();
        if ow != qw { break; }
        if has_para {
            let pw = p[0].to_lowercase();
            if ow != pw { break; }
            p.remove(0);
        }
        o.remove(0);
        q.remove(0);
        trimmed_left += 1;
    }

    let mut trimmed_right = 0;
    while o.len() > 1 && q.len() > 1 && (!has_para || p.len() > 1) {
        let ow = o.last().unwrap().to_lowercase();
        if protected.contains(&ow) { break; }
        let qw = q.last().unwrap().to_lowercase();
        if ow != qw { break; }
        if has_para {
            let pw = p.last().unwrap().to_lowercase();
            if ow != pw { break; }
            p.pop();
        }
        o.pop();
        q.pop();
        trimmed_right += 1;
    }

    let trim_info = TrimInfo { pre_orig, pre_qwen, pre_para, trimmed_left, trimmed_right };
    let strip_edge_punct = |s: String| -> String {
        s.trim_matches(|c: char| !c.is_alphanumeric()).to_string()
    };
    (strip_edge_punct(o.join(" ")), strip_edge_punct(q.join(" ")), strip_edge_punct(p.join(" ")), trim_info)
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

    let (start, end, clean) = if has_parakeet {
        // Tri-boundary consensus: all 3 lanes must agree
        let orig_b = lane_boundaries(orig_align);
        let qwen_b = lane_boundaries(qwen_align);
        let para_b = lane_boundaries(parakeet_align);

        let tris = compute_tri_boundaries(
            &orig_b, &qwen_b, &para_b,
            orig_align, qwen_align, parakeet_align, 0.05,
        );

        let left = tris.iter().rev()
            .find(|tb| tb.time <= term_start + 0.01 && tb.before_matches())
            .map(|tb| tb.time);
        let right = tris.iter()
            .find(|tb| tb.time >= term_end - 0.01 && tb.after_matches())
            .map(|tb| tb.time);

        (left.unwrap_or(term_start), right.unwrap_or(term_end), left.is_some() && right.is_some())
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

        let left = bi_boundaries.iter().rev()
            .find(|&&t| t <= term_start + 0.01)
            .copied();
        let right = bi_boundaries.iter()
            .find(|&&t| t >= term_end - 0.01)
            .copied();

        (left.unwrap_or(term_start), right.unwrap_or(term_end), left.is_some() && right.is_some())
    };

    // Extract words in range then trim matching edges.
    // Uses alignment items directly (same word boundaries as the aligner).
    // Stops trimming at vocab terms to avoid eating important words.
    let (original, qwen, parakeet, trim_info) = trim_matching_edges(
        orig_align, qwen_align, parakeet_align, start, end, protected_terms,
    );

    // If either required lane is empty after trimming, mark as not clean
    let clean = clean && !original.is_empty() && !qwen.is_empty();

    // If any word in the consensus range has suspiciously short duration (<80ms),
    // it's likely boundary noise — discard the whole extraction.
    const MIN_WORD_DURATION: f64 = 0.08;
    let has_short_word = orig_align.iter()
        .chain(qwen_align.iter())
        .filter(|a| a.start_time >= start && a.start_time < end)
        .any(|a| (a.end_time - a.start_time) < MIN_WORD_DURATION);
    let clean = clean && !has_short_word;

    let debug = serde_json::json!({
        "term_start": r2(term_start),
        "term_end": r2(term_end),
        "cons_start": r2(start),
        "cons_end": r2(end),
        "clean": clean,
        "has_parakeet": has_parakeet,
    });

    ConsensusResult { original, qwen, parakeet, cons_range: (start, end), clean, debug, trim_info }
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
                return text[idx..].trim_start_matches(|c: char| c == ':' || c == ',' || c == ' ').to_string();
            }
            return rest.to_string();
        }
    }
    text.to_string()
}

// ==================== Cancel ====================

pub async fn api_stop_job(
    State(state): State<Arc<AppState>>,
) -> Result<Response, AppError> {
    state.job_cancel.store(true, Ordering::Relaxed);
    // The running job will see the flag, finish its current item, and exit gracefully
    Ok(Json(serde_json::json!({"ok": true})).into_response())
}

// ==================== Job Configs ====================

#[derive(Deserialize)]
pub struct CorpusJobBody {
    pub tts_backend: Option<String>,
    pub rounds: Option<usize>,  // number of rounds (0 = endless, default: 100)
    pub dual_asr: Option<bool>, // true = Qwen + Parakeet, false = Qwen only (default: false)
}

#[derive(Deserialize)]
pub struct PrepareJobBody {
    pub total_examples: Option<usize>,
    pub error_rate: Option<f64>,
}

#[derive(Deserialize)]
pub struct TrainJobBody {
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
    let config_json = serde_json::json!({"tts_backend": tts_backend, "rounds": rounds, "dual_asr": dual_asr}).to_string();

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

    let protected_terms: std::collections::HashSet<String> = vocab_terms.iter()
        .map(|v| v.term.to_lowercase())
        .collect();

    {
        let db = state.db.lock().unwrap();
        db.append_job_log(job_id, &format!("Building Markov chain from {} sentences...", all_texts.len()))?;
    }
    let chain = MarkovChain::build(&all_texts);

    let endless = max_rounds == 0;
    {
        let db = state.db.lock().unwrap();
        db.append_job_log(job_id, &format!(
            "Corpus: {} terms, {} rounds, backend: {tts_backend}",
            vocab_terms.len(),
            if endless { "\u{221e}".to_string() } else { max_rounds.to_string() },
        ))?;
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
    let (tx, mut rx) = tokio::sync::mpsc::channel::<(RoundItem, Result<tts::TtsAudio, anyhow::Error>, u64)>(tts_concurrency * 2);

    // TTS producer: dynamically picks random terms, generates sentences, does TTS
    let cancel = state.job_cancel.clone();
    let tts_backend_owned = tts_backend.to_string();
    let state_tts = state.clone();
    let vocab_for_producer: Vec<(String, String)> = vocab_terms.iter()
        .map(|v| (v.term.clone(), v.spoken().to_string()))
        .collect();
    let chain_clone = chain.clone();
    let producer = tokio::spawn(async move {
        use rand::seq::SliceRandom;
        let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(tts_concurrency));
        let mut rng = rand::rngs::StdRng::from_os_rng();
        let mut produced = 0usize;

        loop {
            if cancel.load(Ordering::Relaxed) { break; }
            if !endless && produced >= max_rounds { break; }

            let permit = match semaphore.clone().acquire_owned().await {
                Ok(p) => p,
                Err(_) => break,
            };

            // Pick random term
            let (term, spoken_term) = vocab_for_producer.choose(&mut rng).unwrap().clone();
            let written = WrittenTerm(term.clone());
            let spoken_t = SpokenTerm(spoken_term);
            let (ws, ss) = make_sentence_pair(&chain_clone, &written, &spoken_t, &mut rng);

            let item = RoundItem { term, sentence: ws.0, spoken: ss.0 };
            let tx = tx.clone();
            let state_tts = state_tts.clone();
            let backend = tts_backend_owned.clone();
            let cancel = cancel.clone();

            tokio::spawn(async move {
                if cancel.load(Ordering::Relaxed) { drop(permit); return; }
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
            Ok(mut a) => { a.normalize(); a }
            Err(e) => {
                let db = state.db.lock().unwrap();
                let _ = db.append_job_log(job_id, &format!("TTS FAILED: {e}"));
                errors += 1;
                continue;
            }
        };

        let full_16k = match tts::resample_to_16k(&audio.samples, audio.sample_rate) {
            Ok(s) => s,
            Err(_) => { errors += 1; continue; }
        };

        let t0 = std::time::Instant::now();
        let written_term = WrittenTerm(item.term.clone());
        let written_sentence = WrittenSentence(item.sentence.clone());
        let spoken_sentence = SpokenSentence(item.spoken.clone());
        let result = match run_corpus_pass(state, &full_16k, &written_sentence, &spoken_sentence, &written_term, &protected_terms, dual_asr).await {
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
            let is_mistake = cons.original.to_lowercase() != cons.qwen.to_lowercase();
            let cons_time_json = serde_json::to_string(&[cons.cons_range.0, cons.cons_range.1]).ok();
            let trim_info_json = serde_json::to_string(&cons.trim_info).ok();

            // Encode audio as Ogg Opus for playback in the review UI
            let ogg_bytes = tts::encode_ogg_opus(&audio.samples, audio.sample_rate).await.ok();

            let db = state.db.lock().unwrap();
            match db.upsert_corpus_pair(
                &item.term, &cons.original, &cons.qwen, &cons.parakeet,
                &item.sentence, &item.spoken,
                Some(&fmt_alignment(&result.orig_alignment)),
                Some(&fmt_alignment(&result.qwen_alignment)),
                Some(&fmt_alignment(&result.parakeet_alignment)),
                cons_time_json.as_deref(), trim_info_json.as_deref(), is_mistake,
                ogg_bytes.as_deref(),
            ) {
                Ok((pair_id, is_new)) => {
                    if is_mistake {
                        if is_new {
                            let _ = db.append_job_log(job_id, &format!("NEW|{}|{}|{}|{}", pair_id, item.term, cons.original, cons.qwen));
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
            let rps = if elapsed_ms > 0 { round as f64 / (elapsed_ms as f64 / 1000.0) } else { 0.0 };
            let db = state.db.lock().unwrap();
            let _ = db.append_job_log(job_id, &format!(
                "[{}{}] {} new, {} dup, {} ok, {} noisy, {} err | {:.1}/s",
                round, if endless { String::new() } else { format!("/{max_rounds}") },
                new_mistakes, dup_mistakes, correct, noisy, errors, rps,
            ));
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
    let _ = db.update_job_result(job_id, &serde_json::json!({
        "rounds": round,
        "new_mistakes": new_mistakes,
        "dup_mistakes": dup_mistakes,
        "correct": correct,
        "noisy": noisy,
        "errors": errors,
    }).to_string());

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
    let config_json = serde_json::json!({"total_examples": total_examples, "error_rate": error_rate}).to_string();

    let job_id = {
        let db = state.db.lock().unwrap();
        db.create_job("prepare", Some(&config_json)).map_err(err)?
    };

    let state2 = state.clone();
    tokio::task::spawn_blocking(move || {
        use rand::seq::SliceRandom;

        // Load corpus pairs: term + fragment errors
        let corpus_pairs: Vec<(String, String, String, String)> = { // (term, original, qwen, parakeet)
            let db = state2.db.lock().unwrap();
            db.corpus_unique_triplets().unwrap_or_default()
        };

        // Load sentences for Markov chain
        let all_texts = {
            let db = state2.db.lock().unwrap();
            db.all_sentence_texts().unwrap_or_default()
        };

        if corpus_pairs.is_empty() {
            let db = state2.db.lock().unwrap();
            let _ = db.append_job_log(job_id, "ERROR: No corpus pairs. Generate corpus first.");
            let _ = db.finish_job(job_id, "failed", None);
            return;
        }

        {
            let db = state2.db.lock().unwrap();
            let _ = db.append_job_log(job_id, &format!(
                "Building Markov chain from {} sentences, {} corpus pairs...",
                all_texts.len(), corpus_pairs.len()
            ));
        }

        let chain = MarkovChain::build(&all_texts);
        let mut rng = rand::rngs::StdRng::from_os_rng();

        let n_error = (total_examples as f64 * error_rate).round() as usize;
        let n_identity = total_examples - n_error;

        {
            let db = state2.db.lock().unwrap();
            let _ = db.append_job_log(job_id, &format!(
                "Generating {} error + {} identity = {} total examples",
                n_error, n_identity, total_examples
            ));
        }

        let mut examples = Vec::with_capacity(total_examples);
        let mut correction_count = 0usize;
        let mut identity_count = 0usize;

        // Helper: splice a fragment into a Markov sentence at the term position
        let splice_fragment = |sentence: &str, term: &str, fragment: &str| -> String {
            let lower = sentence.to_lowercase();
            let lower_term = term.to_lowercase();
            if let Some(pos) = lower.find(&lower_term) {
                format!("{}{}{}", &sentence[..pos], fragment, &sentence[pos + term.len()..])
            } else {
                // Term not found in sentence — just use the fragment
                fragment.to_string()
            }
        };

        // Error examples: pick a corpus pair, generate a sentence, splice errors in
        for _ in 0..n_error {
            let (term, original_frag, qwen_frag, keet_frag) = &corpus_pairs[rng.random_range(0..corpus_pairs.len())];
            let sentence = chain.generate_with(term, 15, &mut rng);

            let full_original = splice_fragment(&sentence, term, original_frag);
            let full_qwen = splice_fragment(&sentence, term, qwen_frag);
            let full_keet = splice_fragment(&sentence, term, keet_frag);

            let prompt = synth_train::build_correction_prompt(&full_keet, &full_qwen);
            examples.push(serde_json::json!({
                "prompt": prompt,
                "completion": format!(" {}<|endoftext|>", full_original),
            }));
            correction_count += 1;
        }

        // Identity examples: generate a sentence, all lanes are the same
        let vocab_terms: Vec<&str> = corpus_pairs.iter().map(|(t, _, _, _)| t.as_str()).collect();
        for _ in 0..n_identity {
            let term = vocab_terms[rng.random_range(0..vocab_terms.len())];
            let sentence = chain.generate_with(term, 15, &mut rng);
            let prompt = synth_train::build_correction_prompt("", &sentence);
            examples.push(serde_json::json!({
                "prompt": prompt,
                "completion": format!(" {}<|endoftext|>", sentence),
            }));
            identity_count += 1;
        }

        examples.shuffle(&mut rng);

        // Fixed 90/10 train/valid split
        let n = examples.len();
        let n_train = (n as f64 * 0.9) as usize;
        let train = &examples[..n_train];
        let valid = &examples[n_train..];

        std::fs::create_dir_all("training/data").ok();
        synth_train::write_jsonl("training/data/train.jsonl", train).ok();
        synth_train::write_jsonl("training/data/valid.jsonl", valid).ok();

        let stats = serde_json::json!({
            "correction_examples": correction_count,
            "identity_examples": identity_count,
            "total": n,
            "train_count": train.len(),
            "valid_count": valid.len(),
        });

        let db = state2.db.lock().unwrap();
        let _ = db.append_job_log(
            job_id,
            &format!(
                "Done: {} error + {} identity = {} total ({} train / {} valid)",
                correction_count, identity_count, n, train.len(), valid.len(),
            ),
        );
        let _ = db.finish_job(job_id, "completed", Some(&stats.to_string()));
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
        model: body.model.unwrap_or_else(|| "Qwen/Qwen2.5-0.5B".into()),
        iters: body.iters.unwrap_or(2000),
        batch_size: body.batch_size.unwrap_or(4),
        num_layers: body.num_layers.unwrap_or(8),
        early_stop_patience: body.patience.unwrap_or(10),
        steps_per_eval: body.steps_per_eval.unwrap_or(500),
        ..Default::default()
    };
    let config_json = serde_json::json!({
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

    let state2 = state.clone();
    tokio::task::spawn_blocking(move || {
        {
            let db = state2.db.lock().unwrap();
            let _ = db.append_job_log(
                job_id,
                &format!(
                    "Starting training: model={}, iters={}, batch_size={}, num_layers={}, patience={}, steps_per_eval={}",
                    config.model, config.iters, config.batch_size, config.num_layers,
                    config.early_stop_patience, config.steps_per_eval
                ),
            );
        }

        let result = synth_train::train_streaming(&config, |line| {
            let db = state2.db.lock().unwrap();
            let _ = db.append_job_log(job_id, line);
        });

        let db = state2.db.lock().unwrap();

        // Compute adapter size on disk
        let adapter_size = std::fs::read_dir("training/adapters")
            .ok()
            .map(|entries| {
                entries.filter_map(|e| e.ok())
                    .filter_map(|e| e.metadata().ok().map(|m| m.len()))
                    .sum::<u64>()
            })
            .unwrap_or(0);
        let adapter_mb = adapter_size as f64 / (1024.0 * 1024.0);

        // Parse final validation loss from the job log
        let log = db.get_job(job_id).ok().flatten()
            .map(|j| j.log).unwrap_or_default();
        let val_loss = log.lines().rev()
            .find_map(|line| {
                // MLX-LM outputs lines like "Val loss 2.345, Val took 1.2s"
                let lower = line.to_lowercase();
                if lower.contains("val") && lower.contains("loss") {
                    lower.split_whitespace()
                        .filter_map(|w| w.trim_matches(',').parse::<f64>().ok())
                        .next()
                } else {
                    None
                }
            });

        match result {
            Ok(status) if status.success() => {
                let _ = db.append_job_log(job_id, &format!(
                    "Training completed. Adapters: {adapter_mb:.1}MB{}",
                    val_loss.map(|v| format!(", final val loss: {v:.4}")).unwrap_or_default()
                ));
                let _ = db.finish_job(job_id, "completed", Some(&serde_json::json!({
                    "exit_code": 0,
                    "adapter_mb": (adapter_mb * 10.0).round() / 10.0,
                    "val_loss": val_loss,
                }).to_string()));
            }
            Ok(status) => {
                let code = status.code().unwrap_or(-1);
                let _ = db.append_job_log(job_id, &format!("Training exited with code {code}"));
                let _ = db.finish_job(job_id, "failed", Some(&serde_json::json!({"exit_code": code}).to_string()));
            }
            Err(e) => {
                let _ = db.append_job_log(job_id, &format!("ERROR: {e}"));
                let _ = db.finish_job(job_id, "failed", None);
            }
        }
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
        db.create_job("curate", Some(&serde_json::json!({"batch_size": batch_size}).to_string())).map_err(err)?
    };

    let state2 = state.clone();
    tokio::spawn(async move {
        let result = run_curate_job(&state2, job_id, &api_key, batch_size).await;
        let db = state2.db.lock().unwrap();
        match result {
            Ok((kept, removed)) => {
                let _ = db.append_job_log(job_id, &format!("\n=== DONE ===\n{kept} terms kept, {removed} removed"));
                let _ = db.finish_job(job_id, "completed", Some(&serde_json::json!({"kept": kept, "removed": removed}).to_string()));
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
        db.append_job_log(job_id, &format!("Curating {total} vocab terms in batches of {batch_size} using GPT..."))?;
    }

    let client = reqwest::Client::new();
    let mut kept_total = 0usize;
    let mut removed_total = 0usize;
    let num_batches = (total + batch_size - 1) / batch_size;
    let concurrency = 5;

    // Process batches in groups of `concurrency`
    let batches: Vec<Vec<&str>> = term_strings.chunks(batch_size).map(|b| b.to_vec()).collect();

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
            let _ = db.append_job_log(job_id, &format!(
                "[batches {}-{}/{}] Sending {} batches in parallel...",
                chunk_start + 1, chunk_end, num_batches, chunk.len()
            ));
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

                let resp = client.post("https://api.openai.com/v1/chat/completions")
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
                    Err(e) => return Err(anyhow::anyhow!("batch {}: request failed: {e}", batch_idx + 1)),
                };

                if !resp.status().is_success() {
                    let status = resp.status();
                    let body = resp.text().await.unwrap_or_default();
                    return Err(anyhow::anyhow!("batch {}: API error {status}: {body}", batch_idx + 1));
                }

                let json: serde_json::Value = resp.json().await
                    .map_err(|e| anyhow::anyhow!("batch {}: parse failed: {e}", batch_idx + 1))?;

                let content = json["choices"][0]["message"]["content"].as_str().unwrap_or("[]");
                let results: Vec<serde_json::Value> = serde_json::from_str(content)
                    .or_else(|_| {
                        let cleaned = content.trim()
                            .strip_prefix("```json").or_else(|| content.trim().strip_prefix("```")).unwrap_or(content)
                            .strip_suffix("```").unwrap_or(content).trim();
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
                        let pronunciation = item["pronunciation"].as_str().filter(|s| !s.is_empty());

                        if keep {
                            kept += 1;
                            let _ = db.set_vocab_curated(term, "kept");
                            if let Some(pron) = pronunciation {
                                if let Ok(Some(vocab)) = db.find_vocab_by_term(term) {
                                    if vocab.spoken_override.is_none() {
                                        let _ = db.update_vocab_override(vocab.id, Some(pron));
                                        let _ = db.append_job_log(job_id, &format!("  + {term} → \"{pron}\""));
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
                let _ = db.append_job_log(job_id, &format!("\n=== DONE ===\n{scanned} terms scanned, {errors} with ASR errors"));
                let _ = db.finish_job(job_id, "completed", Some(&serde_json::json!({"scanned": scanned, "errors": errors}).to_string()));
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
        db.append_job_log(job_id, &format!("Scanning {total} vocab terms in batches of {batch_size}, backend: {tts_backend}"))?;
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
        let batch_spoken: Vec<String> = batch.iter().map(|(t, ovr)| {
            ovr.as_deref().unwrap_or(t).to_string()
        }).collect();

        // Build a comma-separated list for TTS
        let tts_text = batch_spoken.join(", ");

        {
            let db = state.db.lock().unwrap();
            let _ = db.append_job_log(job_id, &format!("[batch {}/{}] {} terms: {}...",
                batch_idx + 1, (total + batch_size - 1) / batch_size,
                batch_terms.len(),
                &tts_text[..80.min(tts_text.len())]
            ));
        }

        // TTS the batch
        let audio = match state.tts.generate(tts_backend, &tts_text).await {
            Ok(mut a) => { a.normalize(); a }
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
            tokio::task::spawn_blocking(move || {
                state3.aligner.align(&samples, &text)
            }).await??
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
                let _ = db.append_job_log(job_id, &format!("  SKIP '{}' (alignment ran out)", term));
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
                state_q.asr.transcribe_samples(&seg_q, qwen3_asr::TranscribeOptions::default().with_language("english"))
                    .map(|r| r.text)
                    .unwrap_or_default()
            });

            let parakeet_task = tokio::task::spawn_blocking(move || -> String {
                let mut p = state_p.parakeet.lock().unwrap();
                p.transcribe_samples(seg_p, 16000, 1, None)
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
                let _ = db.insert_confusion(term, qwen.trim(), parakeet.trim(), qwen_match, parakeet_match, tts_backend);
            }

            scanned += 1;
            if !qwen_match || !parakeet_match {
                errors += 1;
                let db = state.db.lock().unwrap();
                let _ = db.append_job_log(job_id, &format!("  {} → qwen: '{}'{} parakeet: '{}'{}",
                    term,
                    qwen.trim(), if qwen_match { "" } else { " \u{2717}" },
                    parakeet.trim(), if parakeet_match { "" } else { " \u{2717}" },
                ));
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
}

pub async fn api_start_eval_job(
    State(state): State<Arc<AppState>>,
    Json(body): Json<EvalJobBody>,
) -> Result<Response, AppError> {
    check_no_running_jobs(&state)?;

    let config = synth_train::InferenceConfig {
        model: body.model.unwrap_or_else(|| "Qwen/Qwen2.5-0.5B".into()),
        adapters: body.adapters.unwrap_or_else(|| "training/adapters".into()),
        ..Default::default()
    };

    let job_id = {
        let db = state.db.lock().unwrap();
        db.create_job("eval", Some(&serde_json::json!({"model": config.model, "adapters": config.adapters}).to_string())).map_err(err)?
    };

    let state2 = state.clone();
    tokio::task::spawn_blocking(move || {
        {
            let db = state2.db.lock().unwrap();
            let _ = db.append_job_log(job_id, &format!("Loading model {} with adapters {}...", config.model, config.adapters));
        }

        // Start inference server once — model stays loaded for all sentences
        let server = match synth_train::InferenceServer::start(&config) {
            Ok(s) => s,
            Err(e) => {
                let db = state2.db.lock().unwrap();
                let _ = db.append_job_log(job_id, &format!("Failed to start inference server: {e}"));
                let _ = db.finish_job(job_id, "failed", None);
                return;
            }
        };

        // Read unique corpus triplets from DB (already clean)
        let triplets = {
            let db = state2.db.lock().unwrap();
            db.corpus_unique_triplets().unwrap_or_default()
        };

        let total = triplets.len();
        {
            let db = state2.db.lock().unwrap();
            let _ = db.append_job_log(job_id, &format!("Server ready. Evaluating {total} unique clean triplets..."));
        }

        // Eval categories:
        // - fixed: ASR was wrong, model corrected it right
        // - kept:  ASR was right (or close), model left it correct
        // - broken: ASR was right, model changed it to something wrong
        // - wrong_fix: ASR was wrong, model's fix is also wrong
        // - blank: model output nothing
        // - timeout: inference timed out
        let mut fixed = 0usize;
        let mut kept = 0usize;
        let mut broken = 0usize;
        let mut wrong_fix = 0usize;
        let mut blank = 0usize;
        let mut timeouts = 0usize;
        let mut total_evaluated = 0usize;

        let mut cancelled = false;
        for (i, (term, original, qwen, parakeet)) in triplets.iter().enumerate() {
            if state2.job_cancel.load(Ordering::Relaxed) {
                cancelled = true;
                let db = state2.db.lock().unwrap();
                let _ = db.append_job_log(job_id, "Stopped by user.");
                break;
            }
            let original: &str = original;
            let qwen: &str = qwen;
            let parakeet: &str = parakeet;
            let term: &str = term;
            let prompt = synth_train::build_correction_prompt(parakeet, qwen);

            let result = server.infer(&prompt);
            total_evaluated += 1;

            let orig_lower = original.trim().to_lowercase();
            let keet_lower = parakeet.trim().to_lowercase();
            let qwen_lower = qwen.trim().to_lowercase();
            let asr_was_correct = keet_lower == orig_lower || qwen_lower == orig_lower;

            let (category, corrected_text) = match result {
                Ok(ref corrected) if corrected.trim().is_empty() => {
                    blank += 1;
                    ("blank", String::new())
                }
                Ok(corrected) => {
                    let corr_lower = corrected.trim().to_lowercase();
                    let model_correct = corr_lower == orig_lower;

                    if model_correct && asr_was_correct {
                        kept += 1;
                        ("kept", corrected)
                    } else if model_correct && !asr_was_correct {
                        fixed += 1;
                        ("fixed", corrected)
                    } else if !model_correct && asr_was_correct {
                        broken += 1;
                        ("broken", corrected)
                    } else {
                        wrong_fix += 1;
                        ("wrong_fix", corrected)
                    }
                }
                Err(e) => {
                    let msg = e.to_string();
                    if msg.contains("timeout") || msg.contains("timed out") {
                        timeouts += 1;
                        ("timeout", format!("(timeout: {msg})"))
                    } else {
                        timeouts += 1;
                        ("error", format!("(error: {msg})"))
                    }
                }
            };

            let correct = fixed + kept;
            let accuracy = if total_evaluated > 0 { correct as f64 / total_evaluated as f64 * 100.0 } else { 0.0 };

            let db = state2.db.lock().unwrap();
            let _ = db.update_job_result(job_id, &serde_json::json!({
                "total": total_evaluated,
                "fixed": fixed, "kept": kept, "broken": broken,
                "wrong_fix": wrong_fix, "blank": blank, "timeouts": timeouts,
                "accuracy": (accuracy * 10.0).round() / 10.0,
            }).to_string());

            // Log entry with category tag
            let _ = db.append_job_log(job_id, &format!(
                "[{}/{}] [{category}] {term}: <keet> {parakeet} <qwen> {qwen} => \"{}\"{}",
                i+1, total, corrected_text.trim(),
                if category == "fixed" || category == "kept" { String::new() }
                else { format!(" (expected \"{}\")", original) }
            ));
        }
        // Kill the server immediately — don't wait for graceful shutdown
        drop(server);

        let correct = fixed + kept;
        let accuracy = if total_evaluated > 0 { correct as f64 / total_evaluated as f64 * 100.0 } else { 0.0 };
        let db = state2.db.lock().unwrap();
        let _ = db.append_job_log(job_id, &format!(
            "\n=== RESULTS ({total_evaluated} evaluated) ===\n\
            Correct: {correct}/{total_evaluated} ({accuracy:.1}%)\n\
            Fixed: {fixed} | Kept: {kept} | Broken: {broken} | Wrong fix: {wrong_fix} | Blank: {blank} | Timeouts: {timeouts}"
        ));
        let _ = db.finish_job(job_id, "completed", Some(&serde_json::json!({
            "total": total_evaluated,
            "fixed": fixed, "kept": kept, "broken": broken,
            "wrong_fix": wrong_fix, "blank": blank, "timeouts": timeouts,
            "accuracy": accuracy,
        }).to_string()));
    });

    Ok(Json(serde_json::json!({"job_id": job_id})).into_response())
}

// ==================== Live Correction ====================

#[derive(Deserialize)]
pub struct CorrectionBody {
    pub parakeet: String,
    pub qwen: String,
}

/// Run a single correction inference via a temporary server.
/// For batch work, use the eval job which keeps the server running.
pub async fn api_correct(
    State(state): State<Arc<AppState>>,
    Json(body): Json<CorrectionBody>,
) -> Result<Response, AppError> {
    let state2 = state.clone();
    let result = tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
        let mut server_guard = state2.inference_server.lock().unwrap();
        // Start server on first use, reuse for subsequent calls
        if server_guard.is_none() {
            eprintln!("[correct] Starting inference server...");
            let config = synth_train::InferenceConfig::default();
            *server_guard = Some(synth_train::InferenceServer::start(&config)?);
            eprintln!("[correct] Inference server ready");
        }
        let server = server_guard.as_ref().unwrap();
        let prompt = synth_train::build_correction_prompt(&body.parakeet, &body.qwen);
        server.infer(&prompt)
    })
    .await
    .map_err(|e| err(e))?
    .map_err(err)?;

    Ok(Json(serde_json::json!({"corrected": result})).into_response())
}

// ==================== Scan Results ====================

pub async fn api_scan_results(
    State(state): State<Arc<AppState>>,
) -> Result<Response, AppError> {
    let db = state.db.lock().unwrap();
    let results = db.vocab_scan_results().map_err(err)?;
    let json: Vec<serde_json::Value> = results.iter().map(|(term, total, qwen_err, parakeet_err)| {
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
    }).collect();
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
        db.find_vocab_by_term(&written.0).map_err(err)?
            .ok_or_else(|| err(format!("Term '{}' not found", written.0)))?
    };
    let spoken = SpokenTerm(vocab.spoken().to_string());

    let all_texts = {
        let db = state.db.lock().unwrap();
        db.all_sentence_texts().map_err(err)?
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
        let pair = make_sentence_pair(&chain, &written, &spoken, &mut rng);
        written_sentence = pair.0;
        spoken_sentence = pair.1;

        audio = state.tts.generate(&tts_backend, &spoken_sentence.0).await.map_err(|e| err(e))?;
        audio.normalize();
        let full_16k = tts::resample_to_16k(&audio.samples, audio.sample_rate).map_err(err)?;

        let protected_terms: std::collections::HashSet<String> = {
            let db = state.db.lock().unwrap();
            db.list_reviewed_vocab().unwrap_or_default().iter().map(|v| v.term.to_lowercase()).collect()
        };
        result = run_corpus_pass(&state, &full_16k, &written_sentence, &spoken_sentence, &written, &protected_terms, dual_asr).await.map_err(err)?;

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
    })).into_response())
}

// ==================== Pipeline Status ====================

pub async fn api_view_corpus(
    State(state): State<Arc<AppState>>,
    axum::extract::Query(params): axum::extract::Query<HashMap<String, String>>,
) -> Result<Response, AppError> {
    let filter_term = params.get("term").map(|s| s.as_str()).filter(|s| !s.is_empty());
    let mistakes_only = params.get("mistakes").map(|s| s == "1").unwrap_or(false);
    let limit: usize = params.get("limit").and_then(|s| s.parse().ok()).unwrap_or(50);
    let offset: usize = params.get("offset").and_then(|s| s.parse().ok()).unwrap_or(0);

    let db = state.db.lock().unwrap();
    let pairs = db.corpus_pairs_query(filter_term, mistakes_only, limit, offset).map_err(err)?;
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
        Some(bytes) => Ok((
            [(axum::http::header::CONTENT_TYPE, "audio/ogg")],
            bytes,
        ).into_response()),
        None => Ok(axum::http::StatusCode::NOT_FOUND.into_response()),
    }
}

pub async fn api_reset_corpus(
    State(state): State<Arc<AppState>>,
) -> Result<Response, AppError> {
    let db = state.db.lock().unwrap();
    db.reset_corpus().map_err(err)?;
    // Also clean up old JSONL file if it exists
    let _ = std::fs::remove_file("data/corpus_dashboard.jsonl");
    Ok(Json(serde_json::json!({"ok": true})).into_response())
}

/// Preview training data: returns sample prompts from the training set so the user
/// can see exactly what the model will be trained on.
pub async fn api_preview_training(
    State(_state): State<Arc<AppState>>,
) -> Result<Response, AppError> {
    let train_path = "training/data/train.jsonl";
    if !std::path::Path::new(train_path).exists() {
        return Ok(Json(serde_json::json!({"error": "No training data. Run Prepare first."})).into_response());
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
            // An identity example has the same text in both <keet> and <qwen> slots
            let is_identity = if let Some(rest) = prompt.strip_prefix("<keet> ") {
                if let Some(idx) = rest.find("\n<qwen> ") {
                    let keet_text = &rest[..idx];
                    let qwen_text = rest[idx + 7..].strip_suffix("\n<fixd>").unwrap_or(&rest[idx + 7..]);
                    keet_text == qwen_text
                } else { false }
            } else { false };
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
    })).into_response())
}

/// Delete training data (train/valid/test splits) so prepare can be re-run.
pub async fn api_reset_training(
    State(_state): State<Arc<AppState>>,
) -> Result<Response, AppError> {
    let _ = std::fs::remove_file("training/data/train.jsonl");
    let _ = std::fs::remove_file("training/data/valid.jsonl");
    let _ = std::fs::remove_file("training/data/test.jsonl");
    Ok(Json(serde_json::json!({"ok": true})).into_response())
}

pub async fn api_pipeline_status(
    State(state): State<Arc<AppState>>,
) -> Result<Response, AppError> {
    let (approved_count, vocab_reviewed, human_recordings, running_job, last_eval, last_train, vocab_scanned) = {
        let db = state.db.lock().unwrap();
        let (approved, _, _) = db.sentence_count_by_status().map_err(err)?;
        let (reviewed, _, _) = db.vocab_review_counts().unwrap_or((0, 0, 0));
        let human = db.sentences_with_human_recording_count().map_err(err)?;
        let scanned = db.confusion_count().map_err(err)?;
        let jobs = db.list_jobs().map_err(err)?;
        let running = jobs.iter().find(|j| j.status == "running").cloned();
        let eval = jobs.iter()
            .filter(|j| j.job_type == "eval" && j.status == "completed")
            .next()
            .and_then(|j| j.result.as_ref())
            .and_then(|r| serde_json::from_str::<serde_json::Value>(r).ok());
        let train = jobs.iter()
            .filter(|j| j.job_type == "train" && j.status == "completed")
            .next()
            .and_then(|j| j.result.as_ref())
            .and_then(|r| serde_json::from_str::<serde_json::Value>(r).ok());
        (approved, reviewed, human, running, eval, train, scanned)
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
            .map(|s| s.lines().filter(|l| !l.trim().is_empty()).count()).unwrap_or(0);
        let vc = std::fs::read_to_string("training/data/valid.jsonl")
            .map(|s| s.lines().filter(|l| !l.trim().is_empty()).count()).unwrap_or(0);
        (tc, vc)
    } else {
        (0, 0)
    };

    let adapters_exist = std::path::Path::new("training/adapters").exists()
        && std::fs::read_dir("training/adapters")
            .map(|mut d| d.next().is_some())
            .unwrap_or(false);

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
        "corpus_exists": corpus_exists,
        "corpus_lines": corpus_lines,
        "training_data_exists": training_data_exists,
        "train_count": train_count,
        "valid_count": valid_count,
        "adapters_exist": adapters_exist,
        "human_recordings": human_recordings,
        "vocab_scanned": vocab_scanned,
        "backends": backends,
        "running_job": running_json,
        "last_eval": last_eval,
        "last_train": last_train,
    }))
    .into_response())
}

#[cfg(test)]
mod tests {
    use super::*;
    use qwen3_asr::ForcedAlignItem;

    fn item(word: &str, start: f64, end: f64) -> ForcedAlignItem {
        ForcedAlignItem { word: word.to_string(), start_time: start, end_time: end }
    }

    fn protected(terms: &[&str]) -> std::collections::HashSet<String> {
        terms.iter().map(|s| s.to_lowercase()).collect()
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
        let orig = vec![item("the", 0.5, 0.7), item("reqwest", 1.0, 1.3), item("for", 1.3, 1.5)];
        let qwen = vec![item("the", 0.5, 0.7), item("requests", 1.0, 1.3), item("for", 1.3, 1.5)];
        let para = vec![];

        let (o, q, _, _ti) = trim_matching_edges(&orig, &qwen, &para, 0.0, 2.0, &protected(&[]));
        assert_eq!(o, "reqwest");
        assert_eq!(q, "requests");
    }

    #[test]
    fn trim_protects_vocab_terms() {
        // "async for" vs "a sync for" — "async" is protected, don't trim even though left doesn't match
        // Actually test: "for async" vs "for a sync" — "for" matches but "async" is protected on left
        let orig = vec![item("for", 0.5, 0.7), item("async", 1.0, 1.3), item("stuff", 1.3, 1.5)];
        let qwen = vec![item("for", 0.5, 0.7), item("a", 1.0, 1.1), item("sync", 1.1, 1.3), item("stuff", 1.3, 1.5)];
        let para = vec![];

        let (o, q, _, _ti) = trim_matching_edges(&orig, &qwen, &para, 0.0, 2.0, &protected(&["async"]));
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
        let orig = vec![item("the", 0.5, 0.7), item("JIT", 1.0, 1.3), item("for", 1.3, 1.5)];
        let qwen = vec![item("the", 0.5, 0.7), item("jiff", 1.0, 1.3), item("for", 1.3, 1.5)];
        let para = vec![item("the", 0.5, 0.7), item("jit", 1.0, 1.3), item("four", 1.3, 1.5)];

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
        let orig = vec![item("before", 0.1, 0.3), item("JIT", 1.0, 1.3), item("after", 2.5, 2.8)];
        let qwen = vec![item("before", 0.1, 0.3), item("jiff", 1.0, 1.3), item("after", 2.5, 2.8)];
        let para = vec![];

        let (o, q, _, _ti) = trim_matching_edges(&orig, &qwen, &para, 0.5, 2.0, &protected(&[]));
        assert_eq!(o, "JIT");
        assert_eq!(q, "jiff");
    }

    #[test]
    fn trim_info_records_what_was_removed() {
        let orig = vec![item("the", 0.5, 0.7), item("reqwest", 1.0, 1.3), item("for", 1.3, 1.5), item("you", 1.5, 1.7)];
        let qwen = vec![item("the", 0.5, 0.7), item("requests", 1.0, 1.3), item("for", 1.3, 1.5), item("you", 1.5, 1.7)];
        let para = vec![];

        let (o, q, _, ti) = trim_matching_edges(&orig, &qwen, &para, 0.0, 2.0, &protected(&[]));
        assert_eq!(o, "reqwest");
        assert_eq!(q, "requests");
        assert_eq!(ti.trimmed_left, 1);  // "the"
        assert_eq!(ti.trimmed_right, 2); // "for", "you"
        assert_eq!(ti.pre_orig, vec!["the", "reqwest", "for", "you"]);
        assert_eq!(ti.pre_qwen, vec!["the", "requests", "for", "you"]);
    }
}
