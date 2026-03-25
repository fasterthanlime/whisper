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

// ==================== Markov Chain ====================

/// Word-level bigram Markov chain for generating sentences containing a target word.
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
) -> ConsensusResult {
    let r2 = |v: f64| (v * 1000.0).round() / 1000.0;

    let orig_b = lane_boundaries(orig_align);
    let qwen_b = lane_boundaries(qwen_align);
    let para_b = lane_boundaries(parakeet_align);

    let tris = compute_tri_boundaries(
        &orig_b, &qwen_b, &para_b,
        orig_align, qwen_align, parakeet_align, 0.05,
    );

    // Left: walk right-to-left, find first where before_matches
    let left = tris.iter().rev()
        .find(|tb| tb.time <= term_start + 0.01 && tb.before_matches())
        .map(|tb| tb.time);

    // Right: walk left-to-right, find first where after_matches
    let right = tris.iter()
        .find(|tb| tb.time >= term_end - 0.01 && tb.after_matches())
        .map(|tb| tb.time);

    let start = left.unwrap_or(term_start);
    let end = right.unwrap_or(term_end);
    let clean = left.is_some() && right.is_some();

    let original = words_in_range(orig_align, start, end);
    let qwen = words_in_range(qwen_align, start, end);
    let parakeet = words_in_range(parakeet_align, start, end);

    let rv = |v: &[f64]| -> Vec<f64> { v.iter().map(|&x| r2(x)).collect() };
    let tri_debug: Vec<serde_json::Value> = tris.iter().map(|tb| {
        serde_json::json!({
            "t": r2(tb.time),
            "before": [&tb.before.0, &tb.before.1, &tb.before.2],
            "after": [&tb.after.0, &tb.after.1, &tb.after.2],
            "bm": tb.before_matches(),
            "am": tb.after_matches(),
        })
    }).collect();

    let debug = serde_json::json!({
        "orig_bounds": rv(&orig_b),
        "qwen_bounds": rv(&qwen_b),
        "para_bounds": rv(&para_b),
        "tri_boundaries": tri_debug,
        "term_start": r2(term_start),
        "term_end": r2(term_end),
        "left_chosen": left.map(r2),
        "right_chosen": right.map(r2),
    });

    ConsensusResult { original, qwen, parakeet, cons_range: (start, end), clean, debug }
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
    pub limit: Option<usize>,  // max items to process this run (default: all)
    pub passes: Option<usize>, // TTS+ASR passes per term (default: 25)
}

#[derive(Deserialize)]
pub struct PrepareJobBody {
    pub identity_count: Option<usize>,
}

#[derive(Deserialize)]
pub struct TrainJobBody {
    pub model: Option<String>,
    pub iters: Option<usize>,
    pub batch_size: Option<usize>,
    pub num_layers: Option<usize>,
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
    let limit = match body.limit {
        Some(0) | None => usize::MAX, // 0 or absent = unlimited
        Some(n) => n,
    };
    let passes = body.passes.unwrap_or(25);
    let config_json = serde_json::json!({"tts_backend": tts_backend, "limit": limit, "passes": passes}).to_string();

    let job_id = {
        let db = state.db.lock().unwrap();
        db.create_job("corpus", Some(&config_json)).map_err(err)?
    };

    let state2 = state.clone();
    let backend = tts_backend.clone();
    tokio::spawn(async move {
        let result = run_corpus_job(&state2, job_id, &backend, limit, passes).await;
        let db = state2.db.lock().unwrap();
        match result {
            Ok(count) => {
                let _ = db.finish_job(
                    job_id,
                    "completed",
                    Some(&serde_json::json!({"sentences": count}).to_string()),
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

async fn run_corpus_job(
    state: &Arc<AppState>,
    job_id: i64,
    tts_backend: &str,
    limit: usize,
    target_passes: usize,
) -> anyhow::Result<usize> {
    // Collect reviewed vocab terms + approved sentences
    let (vocab_terms, all_texts) = {
        let db = state.db.lock().unwrap();
        (db.list_reviewed_vocab()?, db.all_sentence_texts()?)
    };

    // Build Markov chain from all sentence texts
    {
        let db = state.db.lock().unwrap();
        db.append_job_log(job_id, &format!("Building Markov chain from {} sentences...", all_texts.len()))?;
    }
    let chain = MarkovChain::build(&all_texts);

    // Build work items from vocab terms
    struct WorkItem {
        term: String,
        spoken: String,
        passes_needed: usize,
    }

    // Count existing passes per term from DB
    let (existing_passes, existing_total) = {
        let db = state.db.lock().unwrap();
        let passes = db.corpus_passes_per_term().unwrap_or_default();
        let total: usize = passes.values().sum();
        (passes, total)
    };

    let mut work: Vec<WorkItem> = Vec::new();
    let mut total_passes_needed = 0usize;
    for v in &vocab_terms {
        let done = existing_passes.get(&v.term).copied().unwrap_or(0);
        let needed = target_passes.saturating_sub(done);
        if needed > 0 {
            total_passes_needed += needed;
            work.push(WorkItem {
                term: v.term.clone(),
                spoken: v.spoken().to_string(),
                passes_needed: needed,
            });
        }
    }
    // Shuffle so each run covers different terms first (not always starting at A)
    {
        use rand::seq::SliceRandom;
        let mut rng = rand::rngs::StdRng::from_os_rng();
        work.shuffle(&mut rng);
    }

    let passes_to_run = total_passes_needed.min(limit);

    {
        let db = state.db.lock().unwrap();
        db.append_job_log(job_id, &format!(
            "Corpus: {} vocab terms × {} passes, {} pairs exist, {} needed, running {} (limit {})\nBackend: {tts_backend}",
            vocab_terms.len(), target_passes, existing_total, total_passes_needed, passes_to_run,
            if limit == usize::MAX { "∞".to_string() } else { limit.to_string() },
        ))?;
    }

    // No more JSONL file — corpus pairs go straight to SQLite

    let mut count = 0usize;
    let mut errors = 0usize;
    let mut items_done = 0usize;
    let job_start = std::time::Instant::now();
    let mut tts_ms = 0u64;
    let mut asr_ms = 0u64;
    let mut align_ms = 0u64;

    let update_stats = |state: &Arc<AppState>, job_id: i64, count: usize, items_done: usize, errors: usize,
                        tts_ms: u64, asr_ms: u64, align_ms: u64, elapsed_ms: u64| {
        let pairs_per_sec = if elapsed_ms > 0 { count as f64 / (elapsed_ms as f64 / 1000.0) } else { 0.0 };
        let db = state.db.lock().unwrap();
        let _ = db.update_job_result(job_id, &serde_json::json!({
            "pairs_written": count,
            "pairs_total": existing_total + count,
            "items_done": items_done,
            "items_total": work.len(),
            "errors": errors,
            "pairs_per_sec": (pairs_per_sec * 10.0).round() / 10.0,
            "avg_tts_ms": if count > 0 { (tts_ms as f64 / count as f64).round() as u64 } else { 0 },
            "avg_asr_ms": if count > 0 { (asr_ms as f64 / count as f64).round() as u64 } else { 0 },
            "avg_align_ms": if count > 0 { (align_ms as f64 / count as f64).round() as u64 } else { 0 },
        }).to_string());
    };

    let mut rng = rand::rngs::StdRng::from_os_rng();

    // Get spoken overrides
    let overrides = {
        let db = state.db.lock().unwrap();
        db.get_spoken_overrides().unwrap_or_default()
    };
    let _ = &overrides; // suppress unused warning for now

    // Pre-generate all (term, sentence, spoken) tuples
    struct PassItem {
        term: String,
        sentence: String,
        spoken: String,
        item_idx: usize,
        pass_idx: usize,
    }
    let mut pass_items: Vec<PassItem> = Vec::new();
    let mut item_idx = 0;
    for item in &work {
        if pass_items.len() >= passes_to_run { break; }
        let passes_this = item.passes_needed.min(passes_to_run - pass_items.len());
        for pass_idx in 0..passes_this {
            let sentence = chain.generate_with(&item.term, 15, &mut rng);
            let spoken = {
                let lower = sentence.to_lowercase();
                let lower_term = item.term.to_lowercase();
                if let Some(pos) = lower.find(&lower_term) {
                    format!("{}{}{}", &sentence[..pos], &item.spoken, &sentence[pos + item.term.len()..])
                } else {
                    sentence.clone()
                }
            };
            pass_items.push(PassItem { term: item.term.clone(), sentence, spoken, item_idx, pass_idx });
        }
        item_idx += 1;
    }
    let total_items = item_idx;

    {
        let db = state.db.lock().unwrap();
        let _ = db.append_job_log(job_id, &format!("Generated {} Markov sentences, starting pipeline...", pass_items.len()));
    }

    // Pipeline: TTS producer runs ahead, consumer does ASR+alignment as audio arrives.
    // Bounded channel with PIPELINE_AHEAD slots — TTS can run that many ahead of ASR.
    const PIPELINE_AHEAD: usize = 4;
    let pass_items = std::sync::Arc::new(pass_items);
    let (tx, mut rx) = tokio::sync::mpsc::channel::<(usize, Result<tts::TtsAudio, anyhow::Error>, u64)>(PIPELINE_AHEAD);

    // TTS producer task
    let cancel = state.job_cancel.clone();
    let tts_backend_owned = tts_backend.to_string();
    let state_tts = state.clone();
    let items_ref = pass_items.clone();
    let producer = tokio::spawn(async move {
        for (idx, pi) in items_ref.iter().enumerate() {
            if cancel.load(Ordering::Relaxed) { break; }
            let t0 = std::time::Instant::now();
            let result = state_tts.tts.generate(&tts_backend_owned, &pi.spoken).await;
            let ms = t0.elapsed().as_millis() as u64;
            if tx.send((idx, result, ms)).await.is_err() {
                break;
            }
        }
    });

    // Consumer: process audio as it arrives from TTS
    while let Some((idx, tts_result, tts_call_ms)) = rx.recv().await {
        let pi = &pass_items[idx];
        let term = &pi.term;
        let sentence = &pi.sentence;
        if state.job_cancel.load(Ordering::Relaxed) {
            let db = state.db.lock().unwrap();
            let _ = db.append_job_log(job_id, "Stopped by user.");
            break;
        }

        tts_ms += tts_call_ms;

        let mut audio = match tts_result {
            Ok(mut a) => { a.normalize(); a }
            Err(e) => {
                let db = state.db.lock().unwrap();
                let _ = db.append_job_log(job_id, &format!("TTS FAILED: {e} — {}", &sentence.chars().take(40).collect::<String>()));
                errors += 1;
                count += 1;
                continue;
            }
        };

        let full_16k = match tts::resample_to_16k(&audio.samples, audio.sample_rate) {
            Ok(s) => s,
            Err(_) => { errors += 1; count += 1; continue; }
        };

        // ASR (dual, parallel)
        let t0 = std::time::Instant::now();
        let state_q = state.clone();
        let samples_q = full_16k.clone();
        let qwen_task = tokio::task::spawn_blocking(move || -> String {
            state_q.asr
                .transcribe_samples(&samples_q, qwen3_asr::TranscribeOptions::default().with_language("english"))
                .map(|r| r.text)
                .unwrap_or_default()
        });
        let state_p = state.clone();
        let samples_p = full_16k.clone();
        let parakeet_task = tokio::task::spawn_blocking(move || -> String {
            let mut p = state_p.parakeet.lock().unwrap();
            p.transcribe_samples(samples_p, 16000, 1, None)
                .map(|r| r.text)
                .unwrap_or_default()
        });
        let (qwen_full, parakeet_full) = tokio::join!(qwen_task, parakeet_task);
        let qwen_full = qwen_full.unwrap_or_default();
        let parakeet_full = parakeet_full.unwrap_or_default();
        asr_ms += t0.elapsed().as_millis() as u64;

        // Alignment
        let t0 = std::time::Instant::now();
        let orig_align = state.aligner.align(&full_16k, &sentence).unwrap_or_default();
        let term_lower = term.to_lowercase();
        let (term_start, term_end) = find_term_time_range(&orig_align, &term_lower)
            .unwrap_or((0.0, full_16k.len() as f64 / 16000.0));
        let qwen_align = state.aligner.align(&full_16k, &qwen_full).unwrap_or_default();
        let parakeet_align = state.aligner.align(&full_16k, &parakeet_full).unwrap_or_default();
        align_ms += t0.elapsed().as_millis() as u64;

        let cons = extract_with_consensus(&orig_align, &qwen_align, &parakeet_align, term_start, term_end);

        if cons.clean {
            let fmt = |items: &[qwen3_asr::ForcedAlignItem]| -> String {
                serde_json::to_string(&items.iter().map(|a| {
                    serde_json::json!({"w": a.word, "s": (a.start_time * 1000.0).round() / 1000.0, "e": (a.end_time * 1000.0).round() / 1000.0})
                }).collect::<Vec<_>>()).unwrap_or_default()
            };
            let cons_time_json = serde_json::to_string(&[cons.cons_range.0, cons.cons_range.1]).ok();
            let db = state.db.lock().unwrap();
            let _ = db.insert_corpus_pair(
                &pi.term, &cons.original, &cons.qwen, &cons.parakeet, &pi.sentence, &pi.spoken,
                Some(&fmt(&orig_align)), Some(&fmt(&qwen_align)), Some(&fmt(&parakeet_align)),
                cons_time_json.as_deref(),
            );
        }
        count += 1;
        items_done = pi.item_idx + 1;
        if pi.pass_idx == 0 || (count % 10) == 0 {
            let db = state.db.lock().unwrap();
            let _ = db.append_job_log(job_id, &format!(
                "[{}/{}] \"{}\" pass {}: \"{}\"",
                pi.item_idx + 1, total_items, pi.term,
                pi.pass_idx + 1,
                &pi.sentence.chars().take(50).collect::<String>(),
            ));
        }

        if count % 2 == 0 {
            update_stats(state, job_id, count, items_done, errors, tts_ms, asr_ms, align_ms, job_start.elapsed().as_millis() as u64);
        }
    }

    // Wait for producer to finish
    let _ = producer.await;
    update_stats(state, job_id, count, items_done, errors, tts_ms, asr_ms, align_ms, job_start.elapsed().as_millis() as u64);

    {
        let db = state.db.lock().unwrap();
        db.append_job_log(job_id, &format!(
            "Done: {count} pairs written ({items_done} items, {errors} errors). Total corpus: {} pairs.",
            existing_total + count
        ))?;
    }

    Ok(count)
}

// ==================== Prepare ====================

pub async fn api_start_prepare_job(
    State(state): State<Arc<AppState>>,
    Json(body): Json<PrepareJobBody>,
) -> Result<Response, AppError> {
    check_no_running_jobs(&state)?;

    let identity_count = body.identity_count.unwrap_or(95000);
    let config_json = serde_json::json!({"identity_count": identity_count}).to_string();

    let job_id = {
        let db = state.db.lock().unwrap();
        db.create_job("prepare", Some(&config_json)).map_err(err)?
    };

    let state2 = state.clone();
    tokio::task::spawn_blocking(move || {
        // Export corpus from DB to JSONL for synth_train::prepare
        {
            let db = state2.db.lock().unwrap();
            std::fs::create_dir_all("data").ok();
            let pairs = db.corpus_pairs_all().unwrap_or_default();
            let mut f = std::io::BufWriter::new(std::fs::File::create("data/corpus_dashboard.jsonl").unwrap());
            for p in &pairs {
                let _ = writeln!(f, "{}", serde_json::json!({
                    "original": p["original"], "qwen": p["qwen"], "parakeet": p["parakeet"],
                }));
            }
        }

        let config = synth_train::PrepareConfig {
            input: "data/corpus_dashboard.jsonl".into(),
            identity_count,
            ..Default::default()
        };

        let result = synth_train::prepare(&config, |msg| {
            let db = state2.db.lock().unwrap();
            let _ = db.append_job_log(job_id, msg);
        });

        let db = state2.db.lock().unwrap();
        match result {
            Ok(stats) => {
                let _ = db.append_job_log(
                    job_id,
                    &format!(
                        "Done: {} corrections + {} identity = {} train / {} valid / {} test",
                        stats.correction_examples,
                        stats.identity_examples,
                        stats.train_count,
                        stats.valid_count,
                        stats.test_count,
                    ),
                );
                let _ = db.finish_job(
                    job_id,
                    "completed",
                    Some(
                        &serde_json::json!({
                            "correction_examples": stats.correction_examples,
                            "identity_examples": stats.identity_examples,
                            "train_count": stats.train_count,
                            "valid_count": stats.valid_count,
                            "test_count": stats.test_count,
                        })
                        .to_string(),
                    ),
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

// ==================== Train ====================

pub async fn api_start_train_job(
    State(state): State<Arc<AppState>>,
    Json(body): Json<TrainJobBody>,
) -> Result<Response, AppError> {
    check_no_running_jobs(&state)?;

    let config = synth_train::TrainConfig {
        model: body.model.unwrap_or_else(|| "Qwen/Qwen2.5-0.5B".into()),
        iters: body.iters.unwrap_or(1000),
        batch_size: body.batch_size.unwrap_or(1),
        num_layers: body.num_layers.unwrap_or(4),
        ..Default::default()
    };
    let config_json = serde_json::json!({
        "model": config.model,
        "iters": config.iters,
        "batch_size": config.batch_size,
        "num_layers": config.num_layers,
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
                    "Starting training: model={}, iters={}, batch_size={}, num_layers={}",
                    config.model, config.iters, config.batch_size, config.num_layers
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
                state_q.asr
                    .transcribe_samples(&seg_q, qwen3_asr::TranscribeOptions::default().with_language("english"))
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

        let mut correct = 0usize;
        let mut total_evaluated = 0usize;
        let mut total_words = 0usize;
        let mut corrected_words = 0usize;

        for (i, (original, qwen, parakeet, term)) in triplets.iter().enumerate() {
            let original: &str = original;
            let qwen: &str = qwen;
            let parakeet: &str = parakeet;
            let term: &str = term;
            let prompt = synth_train::build_correction_prompt(parakeet, qwen);
            match server.infer(&prompt) {
                Ok(corrected) => {
                    total_evaluated += 1;
                    let original_clean = original.trim().to_lowercase();
                    let corrected_clean = corrected.trim().to_lowercase();
                    let is_correct = original_clean == corrected_clean;
                    if is_correct { correct += 1; }

                    let orig_words: Vec<&str> = original.split_whitespace().collect();
                    let corr_words: Vec<&str> = corrected.split_whitespace().collect();
                    total_words += orig_words.len();
                    corrected_words += orig_words.iter().zip(corr_words.iter())
                        .filter(|(a, b)| a.to_lowercase() == b.to_lowercase())
                        .count();

                    let icon = if is_correct { "\u{2713}" } else { "\u{2717}" };
                    let accuracy = correct as f64 / total_evaluated as f64 * 100.0;
                    let word_acc = if total_words > 0 { corrected_words as f64 / total_words as f64 * 100.0 } else { 0.0 };
                    let db = state2.db.lock().unwrap();

                    // Update live stats
                    let _ = db.update_job_result(job_id, &serde_json::json!({
                        "correct": correct, "total": total_evaluated,
                        "accuracy": (accuracy * 10.0).round() / 10.0,
                        "word_accuracy": (word_acc * 10.0).round() / 10.0,
                    }).to_string());

                    if is_correct {
                        let _ = db.append_job_log(job_id, &format!("[{}/{}] {icon} {term}: P=\"{parakeet}\" Q=\"{qwen}\" => \"{corrected}\"", i+1, total));
                    } else {
                        let _ = db.append_job_log(job_id, &format!(
                            "[{}/{}] {icon} {term}: P=\"{parakeet}\" Q=\"{qwen}\" => \"{corrected}\" (expected \"{original}\")",
                            i+1, total
                        ));
                    }
                }
                Err(e) => {
                    let db = state2.db.lock().unwrap();
                    let _ = db.append_job_log(job_id, &format!("[{}/{}] ERROR: {e}", i+1, total));
                }
            }
        }
        // server is dropped here, killing the subprocess

        let accuracy = if total_evaluated > 0 { correct as f64 / total_evaluated as f64 * 100.0 } else { 0.0 };
        let word_acc = if total_words > 0 { corrected_words as f64 / total_words as f64 * 100.0 } else { 0.0 };
        let db = state2.db.lock().unwrap();
        let _ = db.append_job_log(job_id, &format!(
            "\n=== RESULTS ===\nFragment accuracy: {correct}/{total_evaluated} ({accuracy:.1}%)\nWord accuracy: {corrected_words}/{total_words} ({word_acc:.1}%)"
        ));
        let _ = db.finish_job(job_id, "completed", Some(&serde_json::json!({
            "correct": correct, "total": total_evaluated,
            "accuracy": accuracy,
            "word_accuracy": word_acc,
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

// ==================== Pipeline Status ====================

pub async fn api_view_corpus(
    State(state): State<Arc<AppState>>,
) -> Result<Response, AppError> {
    let pairs = {
        let db = state.db.lock().unwrap();
        db.corpus_pairs_all().map_err(err)?
    };
    Ok(Json(serde_json::json!({"pairs": pairs})).into_response())
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
    let train_count = if training_data_exists {
        std::fs::read_to_string("training/data/train.jsonl")
            .map(|s| s.lines().filter(|l| !l.trim().is_empty()).count())
            .unwrap_or(0)
    } else {
        0
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
