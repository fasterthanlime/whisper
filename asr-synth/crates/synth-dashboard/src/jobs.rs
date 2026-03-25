use std::io::Write;
use std::sync::Arc;

use axum::{
    extract::State,
    response::{IntoResponse, Response},
    Json,
};
use serde::Deserialize;

use crate::tts;
use crate::{err, AppError, AppState};
use parakeet_rs::Transcriber;

use std::sync::atomic::Ordering;

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
    // Collect both approved sentences AND reviewed vocab terms
    let (sentences, vocab_terms) = {
        let db = state.db.lock().unwrap();
        (db.list_approved_sentences()?, db.list_reviewed_vocab()?)
    };

    // Build unified list of (original_text, spoken_text, is_short) items
    let mut items: Vec<(String, String, bool)> = Vec::new();
    for s in &sentences {
        items.push((s.text.clone(), s.spoken.clone(), false));
    }
    for v in &vocab_terms {
        let spoken = v.spoken().to_string();
        let is_short = spoken.split_whitespace().count() <= 3;
        items.push((v.term.clone(), spoken, is_short));
    }

    // Count existing passes per original in the corpus file
    std::fs::create_dir_all("data").ok();
    let corpus_path = "data/corpus_dashboard.jsonl";
    let mut existing_passes: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
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

    // Build work list: for each item, how many more passes are needed?
    struct WorkItem {
        original: String,
        spoken: String,
        is_short: bool,
        passes_needed: usize,
    }
    let mut work: Vec<WorkItem> = Vec::new();
    let mut total_passes_needed = 0usize;
    for (orig, spoken, is_short) in &items {
        let done = existing_passes.get(orig).copied().unwrap_or(0);
        let needed = target_passes.saturating_sub(done);
        if needed > 0 {
            total_passes_needed += needed;
            work.push(WorkItem {
                original: orig.clone(),
                spoken: spoken.clone(),
                is_short: *is_short,
                passes_needed: needed,
            });
        }
    }

    let passes_to_run = total_passes_needed.min(limit);

    {
        let db = state.db.lock().unwrap();
        db.append_job_log(job_id, &format!(
            "Corpus: {} items × {} passes, {} pairs exist, {} passes needed, running {} (limit {})\nBackend: {tts_backend}",
            items.len(), target_passes, existing_total, total_passes_needed, passes_to_run,
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

    'outer: for item in &work {
        if count >= passes_to_run { break; }

        let text_preview: String = item.original.chars().take(40).collect();
        let passes_this_item = item.passes_needed.min(passes_to_run - count);

        // For short items, build carrier phrase
        let tts_text = if item.is_short {
            format!("The word is: {}.", item.spoken.replace('-', " "))
        } else {
            item.spoken.clone()
        };

        for pass in 0..passes_this_item {
            if state.job_cancel.load(Ordering::Relaxed) {
                let db = state.db.lock().unwrap();
                let _ = db.append_job_log(job_id, "Stopped by user.");
                break 'outer;
            }

            // TTS
            let mut audio = match state.tts.generate(tts_backend, &tts_text).await {
                Ok(mut a) => { a.normalize(); a }
                Err(e) => {
                    let db = state.db.lock().unwrap();
                    let _ = db.append_job_log(job_id, &format!("TTS FAILED: {e} — {text_preview}"));
                    errors += 1;
                    continue;
                }
            };

            // Carrier phrase: align + crop
            if item.is_short {
                if let Ok(full_16k) = tts::resample_to_16k(&audio.samples, audio.sample_rate) {
                    let carrier_items = state.aligner.align(&full_16k, &tts_text).unwrap_or_default();
                    let skip = 3;
                    if carrier_items.len() > skip {
                        let start_time = carrier_items[skip].start_time;
                        let end_time = carrier_items.last().map(|it| it.end_time).unwrap_or(start_time + 1.0);
                        let start_sample = ((start_time - 0.05).max(0.0) * audio.sample_rate as f64) as usize;
                        let end_sample = ((end_time + 0.1) * audio.sample_rate as f64).min(audio.samples.len() as f64) as usize;
                        if start_sample < end_sample && end_sample <= audio.samples.len() {
                            audio.samples = audio.samples[start_sample..end_sample].to_vec();
                        }
                    }
                }
            }

            // Resample
            let samples_16k = match tts::resample_to_16k(&audio.samples, audio.sample_rate) {
                Ok(s) => s,
                Err(e) => {
                    let db = state.db.lock().unwrap();
                    let _ = db.append_job_log(job_id, &format!("Resample FAILED: {e} — {text_preview}"));
                    errors += 1;
                    continue;
                }
            };

            // Dual ASR
            let state_q = state.clone();
            let samples_q = samples_16k.clone();
            let qwen_task = tokio::task::spawn_blocking(move || -> String {
                state_q.asr
                    .transcribe_samples(&samples_q, qwen3_asr::TranscribeOptions::default())
                    .map(|r| r.text)
                    .unwrap_or_default()
            });
            let state_p = state.clone();
            let samples_p = samples_16k;
            let parakeet_task = tokio::task::spawn_blocking(move || -> String {
                let mut p = state_p.parakeet.lock().unwrap();
                p.transcribe_samples(samples_p.to_vec(), 16000, 1, None)
                    .map(|r| r.text)
                    .unwrap_or_default()
            });

            let (qwen, parakeet) = tokio::join!(qwen_task, parakeet_task);
            let qwen = qwen.unwrap_or_default();
            let parakeet = parakeet.unwrap_or_default();

            writeln!(file, "{}", serde_json::json!({
                "original": item.original,
                "parakeet": parakeet,
                "qwen": qwen,
            }))?;
            count += 1;

            // Log every pass for first item, then every Nth
            if pass == 0 || (count % 10) == 0 {
                let db = state.db.lock().unwrap();
                let _ = db.append_job_log(job_id, &format!(
                    "[{}/{}] {text_preview} (pass {}/{}): Q=\"{}\" P=\"{}\"",
                    items_done + 1, work.len(), pass + 1, passes_this_item,
                    &qwen.chars().take(30).collect::<String>(),
                    &parakeet.chars().take(30).collect::<String>(),
                ));
            }

            // Update live stats periodically
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
                    .transcribe_samples(&seg_q, qwen3_asr::TranscribeOptions::default())
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
