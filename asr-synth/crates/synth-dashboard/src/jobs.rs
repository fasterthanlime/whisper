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

        for sentence in sentences {
            let words: Vec<&str> = sentence.split_whitespace().collect();
            if words.len() < 3 { continue; }

            // Forward: <S> → w0 → w1 → ... → </S>
            let mut prev = SENTENCE_START.to_string();
            for &w in &words {
                let lower = w.to_lowercase();
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

    /// Generate a sentence containing `target_word` (lowercased).
    /// Returns None if the word isn't in the chain at all.
    fn generate_with(&self, target_word: &str, max_words: usize, rng: &mut impl Rng) -> Option<String> {
        let target_lower = target_word.to_lowercase();

        // Build forward from target
        let mut forward_words = Vec::new();
        let mut cur = target_lower.clone();
        for _ in 0..max_words / 2 {
            let choices = self.forward.get(&cur)?;
            let next = Self::sample(choices, rng)?;
            if next == SENTENCE_END { break; }
            forward_words.push(next.clone());
            cur = next;
        }

        // Build backward from target
        let mut backward_words = Vec::new();
        cur = target_lower.clone();
        for _ in 0..max_words / 2 {
            let choices = self.backward.get(&cur);
            let Some(choices) = choices else { break; };
            let prev = Self::sample(choices, rng)?;
            if prev == SENTENCE_START { break; }
            backward_words.push(prev.clone());
            cur = prev;
        }
        backward_words.reverse();

        // Assemble: backward + target + forward
        let mut words = backward_words;
        words.push(target_word.to_string()); // preserve original casing
        words.extend(forward_words);

        if words.len() < 3 {
            // Too short — fall back to a simple template
            return Some(format!("The {} configuration is important.", target_word));
        }

        // Capitalize first word
        if let Some(first) = words.first_mut() {
            let mut chars = first.chars();
            if let Some(c) = chars.next() {
                *first = c.to_uppercase().to_string() + chars.as_str();
            }
        }

        Some(words.join(" "))
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

/// Align an ASR transcription against audio, then extract the words
/// that overlap with a given time range.
fn extract_asr_at_time_range(
    aligner: &qwen3_asr::ForcedAligner,
    samples_16k: &[f32],
    asr_text: &str,
    start: f64,
    end: f64,
) -> String {
    if asr_text.is_empty() { return String::new(); }

    // Align the ASR output against the audio
    let asr_align = aligner.align(samples_16k, asr_text).unwrap_or_default();
    if asr_align.is_empty() { return asr_text.to_string(); }

    // Collect words whose time range overlaps with [start, end]
    let mut words = Vec::new();
    for item in &asr_align {
        // Check overlap: word overlaps if it doesn't end before start or begin after end
        if item.end_time > start + 0.02 && item.start_time < end - 0.02 {
            words.push(item.word.as_str());
        }
    }

    if words.is_empty() {
        // No overlap found — fall back to full text
        asr_text.to_string()
    } else {
        words.join(" ")
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
    let limit = body.limit.unwrap_or(usize::MAX);
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

    // Count existing passes per term in the corpus file
    std::fs::create_dir_all("data").ok();
    let corpus_path = "data/corpus_dashboard.jsonl";
    let mut existing_passes: HashMap<String, usize> = HashMap::new();
    if let Ok(content) = std::fs::read_to_string(corpus_path) {
        for line in content.lines() {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                if let Some(orig) = v["original"].as_str() {
                    *existing_passes.entry(orig.to_string()).or_default() += 1;
                }
            }
        }
    }
    let existing_total: usize = existing_passes.values().sum();

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

    let passes_to_run = total_passes_needed.min(limit);

    {
        let db = state.db.lock().unwrap();
        db.append_job_log(job_id, &format!(
            "Corpus: {} vocab terms × {} passes, {} pairs exist, {} needed, running {} (limit {})\nBackend: {tts_backend}",
            vocab_terms.len(), target_passes, existing_total, total_passes_needed, passes_to_run,
            if limit == usize::MAX { "∞".to_string() } else { limit.to_string() },
        ))?;
    }

    // Append to existing file
    let mut file = std::io::BufWriter::new(
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(corpus_path)
            .map_err(|e| anyhow::anyhow!("Failed to open corpus file: {e}"))?,
    );

    let mut count = 0usize;
    let mut errors = 0usize;
    let mut items_done = 0usize;

    let update_stats = |state: &Arc<AppState>, job_id: i64, count: usize, items_done: usize, errors: usize| {
        let db = state.db.lock().unwrap();
        let _ = db.update_job_result(job_id, &serde_json::json!({
            "pairs_written": count,
            "pairs_total": existing_total + count,
            "items_done": items_done,
            "items_total": work.len(),
            "errors": errors,
        }).to_string());
    };

    let mut rng = rand::rngs::StdRng::from_os_rng();

    // Get spoken overrides for building spoken forms of Markov sentences
    let overrides = {
        let db = state.db.lock().unwrap();
        db.get_spoken_overrides().unwrap_or_default()
    };

    'outer: for item in &work {
        if count >= passes_to_run { break; }

        let passes_this_item = item.passes_needed.min(passes_to_run - count);

        for pass in 0..passes_this_item {
            if state.job_cancel.load(Ordering::Relaxed) {
                let db = state.db.lock().unwrap();
                let _ = db.append_job_log(job_id, "Stopped by user.");
                break 'outer;
            }

            // Generate a Markov sentence containing this term
            let sentence = chain.generate_with(&item.term, 15, &mut rng)
                .unwrap_or_else(|| format!("Please check the {} setting.", item.term));

            // Build spoken form: replace the term with its spoken pronunciation
            let spoken = {
                let lower_sentence = sentence.to_lowercase();
                let lower_term = item.term.to_lowercase();
                if let Some(pos) = lower_sentence.find(&lower_term) {
                    // Replace the term occurrence with the spoken form
                    let before = &sentence[..pos];
                    let after = &sentence[pos + item.term.len()..];
                    // Apply overrides to surrounding words too
                    let spoken_term = &item.spoken;
                    format!("{before}{spoken_term}{after}")
                } else {
                    sentence.clone()
                }
            };

            // Apply any other vocab overrides to the rest of the sentence
            let spoken = apply_overrides_to_sentence(&spoken, &overrides);

            // TTS the full Markov sentence
            let audio = match state.tts.generate(tts_backend, &spoken).await {
                Ok(mut a) => { a.normalize(); a }
                Err(e) => {
                    let db = state.db.lock().unwrap();
                    let _ = db.append_job_log(job_id, &format!("TTS FAILED: {e} — {}", &sentence.chars().take(40).collect::<String>()));
                    errors += 1;
                    continue;
                }
            };

            // Resample full sentence to 16kHz
            let full_16k = match tts::resample_to_16k(&audio.samples, audio.sample_rate) {
                Ok(s) => s,
                Err(e) => {
                    let db = state.db.lock().unwrap();
                    let _ = db.append_job_log(job_id, &format!("Resample FAILED: {e}"));
                    errors += 1;
                    continue;
                }
            };

            // 1) Align the ORIGINAL text against the audio to find the term's time range
            let orig_align = state.aligner.align(&full_16k, &spoken).unwrap_or_default();
            let spoken_term_lower = item.spoken.to_lowercase();

            let (term_start, term_end) = find_term_time_range(&orig_align, &spoken_term_lower)
                .unwrap_or((0.0, full_16k.len() as f64 / 16000.0));

            // 2) Run dual ASR on the FULL audio (natural context → better transcription)
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

            // 3) Align each ASR output against the same audio, then extract
            //    the word(s) that overlap with the target term's time range
            let qwen = extract_asr_at_time_range(
                &state.aligner, &full_16k, &qwen_full, term_start, term_end,
            );
            let parakeet = extract_asr_at_time_range(
                &state.aligner, &full_16k, &parakeet_full, term_start, term_end,
            );

            // Write both ASR outputs as separate training pairs
            // original = the sentence with correct spelling
            // qwen/parakeet = what ASR heard
            writeln!(file, "{}", serde_json::json!({
                "original": sentence,
                "qwen": qwen,
                "parakeet": parakeet,
                "term": item.term,
            }))?;
            count += 1;

            // Log
            if pass == 0 || (count % 10) == 0 {
                let db = state.db.lock().unwrap();
                let _ = db.append_job_log(job_id, &format!(
                    "[{}/{}] \"{}\" pass {}/{}: \"{}\"",
                    items_done + 1, work.len(), item.term,
                    pass + 1, passes_this_item,
                    &sentence.chars().take(50).collect::<String>(),
                ));
            }

            // Update live stats
            if count % 5 == 0 {
                file.flush()?;
                update_stats(state, job_id, count, items_done, errors);
            }
        }

        items_done += 1;
    }

    file.flush()?;
    update_stats(state, job_id, count, items_done, errors);

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
        match result {
            Ok(status) if status.success() => {
                let _ = db.append_job_log(job_id, "Training completed successfully");
                let _ = db.finish_job(job_id, "completed", Some(&serde_json::json!({"exit_code": 0}).to_string()));
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

        let sentences = {
            let db = state2.db.lock().unwrap();
            db.list_approved_sentences().unwrap_or_default()
        };

        let total = sentences.len();
        {
            let db = state2.db.lock().unwrap();
            let _ = db.append_job_log(job_id, &format!("Server ready. Evaluating {total} sentences..."));
        }

        let mut correct = 0usize;
        let mut total_evaluated = 0usize;
        let mut total_words = 0usize;
        let mut corrected_words = 0usize;

        // Read corpus for ASR outputs
        let corpus: Vec<serde_json::Value> = std::fs::read_to_string("data/corpus_dashboard.jsonl")
            .unwrap_or_default()
            .lines()
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect();

        let corpus_map: std::collections::HashMap<String, (String, String)> = corpus.iter()
            .filter_map(|v| {
                let orig = v["original"].as_str()?.to_string();
                let p = v["parakeet"].as_str()?.to_string();
                let q = v["qwen"].as_str()?.to_string();
                Some((orig, (p, q)))
            })
            .collect();

        for (i, sentence) in sentences.iter().enumerate() {
            let (parakeet, qwen) = match corpus_map.get(&sentence.text) {
                Some(pair) => pair.clone(),
                None => {
                    let db = state2.db.lock().unwrap();
                    let _ = db.append_job_log(job_id, &format!("[{}/{}] SKIP (no corpus entry)", i+1, total));
                    continue;
                }
            };

            let prompt = synth_train::build_correction_prompt(&parakeet, &qwen);
            match server.infer(&prompt) {
                Ok(corrected) => {
                    total_evaluated += 1;
                    let original_clean = sentence.text.trim().to_lowercase();
                    let corrected_clean = corrected.trim().to_lowercase();
                    let is_correct = original_clean == corrected_clean;
                    if is_correct { correct += 1; }

                    let orig_words: Vec<&str> = sentence.text.split_whitespace().collect();
                    let corr_words: Vec<&str> = corrected.split_whitespace().collect();
                    total_words += orig_words.len();
                    corrected_words += orig_words.iter().zip(corr_words.iter())
                        .filter(|(a, b)| a.to_lowercase() == b.to_lowercase())
                        .count();

                    let icon = if is_correct { "\u{2713}" } else { "\u{2717}" };
                    let preview: String = sentence.text.chars().take(50).collect();
                    let db = state2.db.lock().unwrap();
                    if is_correct {
                        let _ = db.append_job_log(job_id, &format!("[{}/{}] {icon} {preview}", i+1, total));
                    } else {
                        let _ = db.append_job_log(job_id, &format!(
                            "[{}/{}] {icon} {preview}\n  expected: {}\n  got:      {corrected}",
                            i+1, total, sentence.text
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
            "\n=== RESULTS ===\nSentence accuracy: {correct}/{total_evaluated} ({accuracy:.1}%)\nWord accuracy: {corrected_words}/{total_words} ({word_acc:.1}%)"
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
    Json(body): Json<CorrectionBody>,
) -> Result<Response, AppError> {
    let result = tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
        let config = synth_train::InferenceConfig::default();
        let server = synth_train::InferenceServer::start(&config)?;
        let prompt = synth_train::build_correction_prompt(&body.parakeet, &body.qwen);
        server.infer(&prompt)
        // server dropped here, subprocess killed
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

pub async fn api_view_corpus() -> Result<Response, AppError> {
    let path = "data/corpus_dashboard.jsonl";
    let content = std::fs::read_to_string(path).unwrap_or_default();
    let pairs: Vec<serde_json::Value> = content.lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    Ok(Json(serde_json::json!({"pairs": pairs})).into_response())
}

pub async fn api_reset_corpus() -> Result<Response, AppError> {
    let path = "data/corpus_dashboard.jsonl";
    if std::path::Path::new(path).exists() {
        std::fs::remove_file(path).map_err(|e| err(anyhow::anyhow!("Failed to remove corpus: {e}")))?;
    }
    Ok(Json(serde_json::json!({"ok": true})).into_response())
}

pub async fn api_pipeline_status(
    State(state): State<Arc<AppState>>,
) -> Result<Response, AppError> {
    let (approved_count, vocab_reviewed, human_recordings, running_job, last_eval, vocab_scanned) = {
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
        (approved, reviewed, human, running, eval, scanned)
    };

    // Check filesystem for corpus / training data / adapters
    let corpus_exists = std::path::Path::new("data/corpus_dashboard.jsonl").exists();
    let corpus_lines = if corpus_exists {
        std::fs::read_to_string("data/corpus_dashboard.jsonl")
            .map(|s| s.lines().filter(|l| !l.trim().is_empty()).count())
            .unwrap_or(0)
    } else {
        0
    };

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
    }))
    .into_response())
}
