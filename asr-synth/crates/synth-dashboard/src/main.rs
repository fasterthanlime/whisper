mod db;
mod jobs;
mod review;
mod tts;

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{Html, IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use db::Db;
use parakeet_rs::Transcriber;
use serde::Deserialize;
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

pub type AppError = (StatusCode, String);

pub struct AppState {
    db: Mutex<Db>,
    log_path: std::path::PathBuf,
    docs_root: String,
    tts: tts::TtsManager,
    asr: qwen3_asr::AsrInference,
    parakeet: std::sync::Mutex<parakeet_rs::ParakeetTDT>,
    aligner: qwen3_asr::ForcedAligner,
    review: Mutex<review::ReviewSession>,
    precompute_notify: std::sync::Arc<tokio::sync::Notify>,
    vocab_precompute_notify: std::sync::Arc<tokio::sync::Notify>,
    authored_asr_notify: std::sync::Arc<tokio::sync::Notify>,
    background_work_notify: std::sync::Arc<tokio::sync::Notify>,
    audio_dir: String,
    qwen_model_key: String,
    job_cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Shared inference server for live correction (started on first use)
    inference_server: std::sync::Mutex<Option<synth_train::InferenceServer>>,
}

#[derive(Deserialize)]
struct ListParams {
    status: Option<String>,
    limit: Option<i64>,
    offset: Option<i64>,
}

#[derive(Deserialize)]
struct VocabListParams {
    search: Option<String>,
    reviewed: Option<bool>,
    has_override: Option<bool>,
    sort: Option<String>, // "alpha" (default) or "recent"
    limit: Option<i64>,
    offset: Option<i64>,
}

#[derive(Deserialize)]
struct VocabUpdateBody {
    spoken_override: Option<String>,
    reviewed: Option<bool>,
    description: Option<String>,
}

#[derive(Deserialize)]
struct VocabAddBody {
    term: String,
    spoken_override: Option<String>,
    description: Option<String>,
}

#[derive(Deserialize)]
struct SettingsUpdateBody {
    background_template_target_per_term: Option<usize>,
    background_confusion_target_per_term: Option<usize>,
}

#[derive(Deserialize)]
struct AltSpellingBody {
    term: String,
    alt_spelling: String,
}

#[derive(Deserialize)]
struct GenerateBody {
    count: Option<usize>,
    prioritize_unknown: Option<bool>,
}

#[derive(Deserialize)]
struct SentenceUpdateBody {
    status: Option<String>,
    spoken: Option<String>,
}

#[derive(Deserialize)]
struct TtsPreviewBody {
    text: String,
    backend: Option<String>,
    /// When set, also run forced alignment on the generated audio with this text
    /// and return JSON instead of raw WAV.
    align_text: Option<String>,
}

pub fn err(e: impl std::fmt::Display) -> AppError {
    (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
}

fn normalize_suggested_term(term: &str) -> String {
    term.trim()
        .trim_matches(|c: char| {
            !c.is_alphanumeric() && c != '_' && c != '-' && c != '(' && c != ')'
        })
        .to_lowercase()
}

fn sanitize_spellcheck_issues(value: serde_json::Value) -> serde_json::Value {
    let raw = if let Some(items) = value.get("issues").and_then(|v| v.as_array()) {
        items.clone()
    } else if let Some(items) = value.as_array() {
        items.clone()
    } else {
        Vec::new()
    };

    let issues = raw
        .into_iter()
        .filter_map(|item| {
            let id = item.get("id")?.as_i64()?;
            let original = item.get("original")?.as_str()?.trim();
            let suggestion = item.get("suggestion")?.as_str()?.trim();
            let reason = item
                .get("reason")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim();

            let original_lower = original.to_lowercase();
            let suggestion_lower = suggestion.to_lowercase();
            let reason_lower = reason.to_lowercase();

            if original.is_empty()
                || suggestion.is_empty()
                || original_lower == suggestion_lower
                || original_lower.contains("no issue")
                || suggestion_lower.contains("no issue")
                || reason_lower.contains("no issue")
            {
                return None;
            }

            Some(serde_json::json!({
                "id": id,
                "original": original,
                "suggestion": suggestion,
                "reason": reason,
            }))
        })
        .collect::<Vec<_>>();

    serde_json::json!({ "issues": issues })
}

async fn index() -> Result<Response, AppError> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("static/index.html");
    let content = std::fs::read_to_string(&path).map_err(err)?;
    Ok((
        [(axum::http::header::CACHE_CONTROL, "no-store")],
        Html(content),
    )
        .into_response())
}

// ==================== STATS ====================

async fn api_stats(State(state): State<Arc<AppState>>) -> Result<Response, AppError> {
    let db = state.db.lock().unwrap();
    let stats = db.stats().map_err(err)?;
    Ok(Json(stats).into_response())
}

// ==================== VOCAB ====================

async fn api_vocab_list(
    State(state): State<Arc<AppState>>,
    Query(params): Query<VocabListParams>,
) -> Result<Response, AppError> {
    let db = state.db.lock().unwrap();
    let sort_recent = params.sort.as_deref() == Some("recent");
    let list = db
        .list_vocab(
            params.search.as_deref(),
            params.reviewed,
            params.has_override,
            sort_recent,
            params.limit.unwrap_or(100),
            params.offset.unwrap_or(0),
        )
        .map_err(err)?;
    Ok(Json(list).into_response())
}

async fn api_vocab_import(State(state): State<Arc<AppState>>) -> Result<Response, AppError> {
    let docs_root = state.docs_root.clone();
    let state2 = state.clone();

    let result = tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
        let vocab = synth_textgen::corpus::extract_vocab(&docs_root)?;
        let db = state2.db.lock().unwrap();
        let count = db.import_vocab(&vocab)?;
        Ok(serde_json::json!({
            "extracted": vocab.len(),
            "imported": count,
        }))
    })
    .await
    .map_err(|e| err(e))?
    .map_err(err)?;

    Ok(Json(result).into_response())
}

async fn api_vocab_update(
    State(state): State<Arc<AppState>>,
    Path(id): Path<i64>,
    Json(body): Json<VocabUpdateBody>,
) -> Result<Response, AppError> {
    let db = state.db.lock().unwrap();
    if let Some(ref spoken) = body.spoken_override {
        // Empty string means clear the override
        let val = if spoken.is_empty() {
            None
        } else {
            Some(spoken.as_str())
        };
        db.update_vocab_override(id, val).map_err(err)?;
    }
    if let Some(reviewed) = body.reviewed {
        db.set_vocab_reviewed(id, reviewed).map_err(err)?;
    }
    if let Some(ref desc) = body.description {
        let val = if desc.is_empty() {
            None
        } else {
            Some(desc.as_str())
        };
        db.update_vocab_description(id, val).map_err(err)?;
    }
    drop(db);
    state.background_work_notify.notify_one();
    Ok(Json(serde_json::json!({ "ok": true })).into_response())
}

async fn api_vocab_delete(
    State(state): State<Arc<AppState>>,
    Path(id): Path<i64>,
) -> Result<Response, AppError> {
    let db = state.db.lock().unwrap();
    if let Some(row) = db.get_vocab(id).map_err(err)? {
        db.delete_vocab_by_term(&row.term).map_err(err)?;
    }
    Ok(Json(serde_json::json!({"ok": true})).into_response())
}

async fn api_settings_get(State(state): State<Arc<AppState>>) -> Result<Response, AppError> {
    let db = state.db.lock().unwrap();
    let template_target = db
        .get_setting_usize("background_template_target_per_term")
        .map_err(err)?
        .unwrap_or(200);
    let confusion_target = db
        .get_setting_usize("background_confusion_target_per_term")
        .map_err(err)?
        .unwrap_or(8);
    Ok(Json(serde_json::json!({
        "background_template_target_per_term": template_target,
        "background_confusion_target_per_term": confusion_target,
    }))
    .into_response())
}

async fn api_settings_update(
    State(state): State<Arc<AppState>>,
    Json(body): Json<SettingsUpdateBody>,
) -> Result<Response, AppError> {
    {
        let db = state.db.lock().unwrap();
        if let Some(value) = body.background_template_target_per_term {
            db.set_setting(
                "background_template_target_per_term",
                &value.clamp(1, 1000).to_string(),
            )
            .map_err(err)?;
        }
        if let Some(value) = body.background_confusion_target_per_term {
            db.set_setting(
                "background_confusion_target_per_term",
                &value.clamp(1, 1000).to_string(),
            )
            .map_err(err)?;
        }
    }
    state.background_work_notify.notify_one();
    api_settings_get(State(state)).await
}

async fn api_vocab_add(
    State(state): State<Arc<AppState>>,
    Json(body): Json<VocabAddBody>,
) -> Result<Response, AppError> {
    let term = body.term.trim().to_string();
    if term.is_empty() {
        return Ok(Json(serde_json::json!({"error": "term is empty"})).into_response());
    }
    let spoken_auto = synth_textgen::corpus::to_spoken(&term);
    let db = state.db.lock().unwrap();
    db.insert_candidate_vocab(&term, &spoken_auto)
        .map_err(err)?;
    if let Ok(Some(row)) = db.find_vocab_by_term(&term) {
        // Manual add = automatically reviewed + curated
        db.set_vocab_reviewed(row.id, true).map_err(err)?;
        db.set_vocab_curated(&term, "kept").map_err(err)?;
        if let Some(ref spoken) = body.spoken_override {
            if !spoken.is_empty() {
                db.update_vocab_override(row.id, Some(spoken))
                    .map_err(err)?;
            }
        }
        if let Some(ref desc) = body.description {
            if !desc.is_empty() {
                db.update_vocab_description(row.id, Some(desc))
                    .map_err(err)?;
            }
        }
    }
    drop(db);
    state.background_work_notify.notify_one();
    Ok(Json(serde_json::json!({"ok": true})).into_response())
}

async fn api_vocab_alt_spellings(State(state): State<Arc<AppState>>) -> Result<Response, AppError> {
    let db = state.db.lock().unwrap();
    let alt_spellings = db.get_all_alt_spellings().map_err(err)?;
    Ok(Json(alt_spellings).into_response())
}

async fn api_vocab_alt_spelling_add(
    State(state): State<Arc<AppState>>,
    Json(body): Json<AltSpellingBody>,
) -> Result<Response, AppError> {
    let term = body.term.trim();
    let alt_spelling = body.alt_spelling.trim();
    if term.is_empty() || alt_spelling.is_empty() {
        return Ok(
            Json(serde_json::json!({"error": "term and alt_spelling are required"}))
                .into_response(),
        );
    }
    let db = state.db.lock().unwrap();
    let updated = db.add_alt_spelling(term, alt_spelling).map_err(err)?;
    Ok(Json(serde_json::json!({"ok": true, "retroactive_updates": updated})).into_response())
}

async fn api_vocab_alt_spelling_delete(
    State(state): State<Arc<AppState>>,
    Json(body): Json<AltSpellingBody>,
) -> Result<Response, AppError> {
    let term = body.term.trim();
    let alt_spelling = body.alt_spelling.trim();
    if term.is_empty() || alt_spelling.is_empty() {
        return Ok(
            Json(serde_json::json!({"error": "term and alt_spelling are required"}))
                .into_response(),
        );
    }
    let db = state.db.lock().unwrap();
    let deleted = db.delete_alt_spelling(term, alt_spelling).map_err(err)?;
    Ok(Json(serde_json::json!({"ok": true, "deleted": deleted})).into_response())
}

// ==================== SENTENCES ====================

async fn api_sentences_list(
    State(state): State<Arc<AppState>>,
    Query(params): Query<ListParams>,
) -> Result<Response, AppError> {
    let db = state.db.lock().unwrap();
    let limit = params.limit.unwrap_or(50);

    // When requesting pending sentences, auto-promote candidates if we're running low
    if params.status.as_deref() == Some("pending") {
        let pending_count = db.list_sentences(Some("pending"), 1, 0).map_err(err)?.len() as i64;
        if pending_count < limit {
            let needed = limit - pending_count;
            let candidates = db.pick_candidates(needed, true).map_err(err)?;
            for (text, spoken, vocab_terms, unknown_words) in &candidates {
                let _ = db.insert_sentence_from_candidate(text, spoken, vocab_terms, unknown_words);
            }
        }
    }

    let list = db
        .list_sentences(params.status.as_deref(), limit, params.offset.unwrap_or(0))
        .map_err(err)?;
    Ok(Json(list).into_response())
}

/// Scan sources incrementally, extract vocab, find sentences.
/// Reads each source file from where the last import left off,
/// stops after ~200 new vocab terms are discovered. Re-import to get the next batch.
#[derive(Deserialize)]
struct ImportParams {
    source: Option<String>, // "all" (default), "hark", "claude", "codex"
    limit: Option<usize>,   // new vocab target (default 200)
}

async fn api_candidates_import(
    State(state): State<Arc<AppState>>,
    Query(params): Query<ImportParams>,
) -> Result<Response, AppError> {
    let docs_root = state.docs_root.clone();
    let state2 = state.clone();
    let source = params.source.unwrap_or_else(|| "all".to_string());
    let vocab_limit = params.limit.unwrap_or(200);

    let result = tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
        eprintln!("[import] extracting vocab from {docs_root}...");
        let vocab = synth_textgen::corpus::extract_vocab(&docs_root)?;
        eprintln!("[import] {} vocab terms extracted", vocab.len());

        let overrides = {
            let db = state2.db.lock().unwrap();
            db.get_spoken_overrides()?
        };

        // Build term lookup for sentence matching
        let term_lookup: std::collections::HashMap<String, &synth_textgen::corpus::VocabEntry> = vocab
            .iter()
            .map(|v| (v.term.to_lowercase(), v))
            .collect();

        let hark = shellexpand::tilde("~/Library/Application Support/hark/transcription_log.jsonl").to_string();
        let claude = shellexpand::tilde("~/.claude/history.jsonl").to_string();
        let codex = shellexpand::tilde("~/.codex/history.jsonl").to_string();
        let files: Vec<(&str, String)> = match source.as_str() {
            "hark" => vec![("hark", hark)],
            "claude" => vec![("claude", claude)],
            "codex" => vec![("codex", codex)],
            _ => vec![("hark", hark), ("claude", claude), ("codex", codex)],
        };

        let mut total_new_vocab = 0usize;
        let mut total_inserted = 0usize;
        let mut total_lines = 0usize;

        for (name, path) in &files {
            let start_offset = {
                let db = state2.db.lock().unwrap();
                db.get_import_offset(name)?
            };

            let file_len = match std::fs::metadata(path) {
                Ok(m) => m.len(),
                Err(_) => { eprintln!("[import] {name}: file not found: {path}"); continue; }
            };

            if start_offset >= file_len {
                eprintln!("[import] {name}: already fully scanned ({start_offset}/{file_len} bytes)");
                continue;
            }

            eprintln!("[import] {name}: resuming from byte {start_offset}/{file_len}");

            use std::io::{BufRead, Seek};
            let mut file = std::io::BufReader::new(std::fs::File::open(path)?);
            file.seek(std::io::SeekFrom::Start(start_offset))?;

            // If resuming mid-file, skip partial first line
            if start_offset > 0 {
                let mut discard = String::new();
                file.read_line(&mut discard)?;
            }

            let mut new_vocab = 0usize;
            let mut inserted = 0usize;
            let mut lines = 0usize;
            let mut last_offset = start_offset;

            let mut line_buf = String::new();
            loop {
                line_buf.clear();
                let bytes_read = file.read_line(&mut line_buf)?;
                if bytes_read == 0 { break; } // EOF
                last_offset = file.stream_position()?;
                lines += 1;

                let line = line_buf.trim();
                let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else { continue };
                let text = v["display"].as_str().or_else(|| v["text"].as_str()).unwrap_or("");
                if text.len() < 20 || text.len() > 300
                    || text.contains("[Pasted") || text.contains("[Image")
                    || text.starts_with('/') || text.starts_with("> ")
                    || text.contains('`') || text.contains("${")
                    || text.contains("::") || text.contains("//") || text.contains("->")
                { continue; }

                // Process this text through the sentence pipeline
                let sentences = synth_textgen::templates::extract_sentences(
                    text, &term_lookup, &overrides,
                );

                let db = state2.db.lock().unwrap();
                for s in &sentences {
                    let unknown = tts::detect_unknown_words(&s.text);
                    let unknown_json = serde_json::to_string(&unknown)?;

                    for w in &unknown {
                        if db.insert_candidate_vocab(w, &w.to_lowercase())? {
                            new_vocab += 1;
                        }
                    }

                    let vocab_json = serde_json::to_string(&s.vocab_terms)?;
                    if db.insert_sentence_from_candidate(&s.text, &s.spoken, &vocab_json, &unknown_json)? {
                        inserted += 1;
                    }
                }
                drop(db);

                // Stop once we've found enough new vocab
                if total_new_vocab + new_vocab >= vocab_limit {
                    eprintln!("[import] {name}: hit vocab limit ({new_vocab} new vocab from {lines} lines)");
                    break;
                }
            }

            // Save progress
            {
                let db = state2.db.lock().unwrap();
                db.set_import_offset(name, last_offset)?;
            }

            let pct = if file_len > 0 { last_offset * 100 / file_len } else { 100 };
            eprintln!("[import] {name}: {lines} lines, {inserted} sentences, {new_vocab} vocab ({pct}% of file)");

            total_new_vocab += new_vocab;
            total_inserted += inserted;
            total_lines += lines;

            if total_new_vocab >= vocab_limit { break; }
        }

        eprintln!("[import] done. {total_inserted} sentences, {total_new_vocab} new vocab");

        Ok(serde_json::json!({
            "imported": total_inserted,
            "lines_scanned": total_lines,
            "new_vocab": total_new_vocab,
        }))
    })
    .await
    .map_err(|e| err(e))?
    .map_err(err)?;

    Ok(Json(result).into_response())
}

/// Fast: pick N random candidates and promote to sentences table
async fn api_sentences_generate(
    State(state): State<Arc<AppState>>,
    Json(body): Json<GenerateBody>,
) -> Result<Response, AppError> {
    let count = body.count.unwrap_or(20) as i64;
    let prioritize = body.prioritize_unknown.unwrap_or(true);
    let db = state.db.lock().unwrap();

    let candidates = db.pick_candidates(count, prioritize).map_err(err)?;
    if candidates.is_empty() {
        return Ok(Json(serde_json::json!({
            "picked": 0,
            "message": "No candidates available. Run 'Import Sources' first.",
        }))
        .into_response());
    }

    let mut inserted = 0;
    for (text, spoken, vocab_terms, unknown_words) in &candidates {
        db.insert_sentence_from_candidate(text, spoken, vocab_terms, unknown_words)
            .map_err(err)?;
        inserted += 1;
    }

    Ok(Json(serde_json::json!({ "picked": inserted })).into_response())
}

async fn api_sentence_update(
    State(state): State<Arc<AppState>>,
    Path(id): Path<i64>,
    Json(body): Json<SentenceUpdateBody>,
) -> Result<Response, AppError> {
    let db = state.db.lock().unwrap();
    if let Some(ref status) = body.status {
        db.update_sentence_status(id, status).map_err(err)?;
    }
    if let Some(ref spoken) = body.spoken {
        db.update_sentence_spoken(id, spoken).map_err(err)?;
    }
    Ok(Json(serde_json::json!({ "ok": true })).into_response())
}

// ==================== TTS PREVIEW ====================

async fn api_tts_preview(
    State(state): State<Arc<AppState>>,
    Json(body): Json<TtsPreviewBody>,
) -> Result<Response, AppError> {
    let text = body.text;
    let backend = body.backend.unwrap_or_else(|| "pocket-hq".to_string());
    let align_text = body.align_text;

    eprintln!(
        "TTS preview: backend={backend} text={:?}",
        &text[..text.len().min(50)]
    );
    let mut audio = state.tts.generate(&backend, &text).await.map_err(|e| {
        eprintln!("TTS error: {e}");
        err(e)
    })?;
    audio.normalize();
    let wav_bytes = audio.to_wav().map_err(err)?;

    // If align_text provided, run forced alignment and return JSON
    if let Some(align_text) = align_text {
        let samples = audio.samples.clone();
        let sample_rate = audio.sample_rate;
        let state2 = state.clone();

        let alignment =
            tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<serde_json::Value>> {
                // Resample to 16kHz for the aligner
                let samples_16k = if sample_rate != 16000 {
                    use rubato::{
                        Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType,
                        WindowFunction,
                    };
                    let params = SincInterpolationParameters {
                        sinc_len: 256,
                        f_cutoff: 0.95,
                        interpolation: SincInterpolationType::Linear,
                        oversampling_factor: 256,
                        window: WindowFunction::BlackmanHarris2,
                    };
                    let mut resampler = SincFixedIn::<f32>::new(
                        16000.0 / sample_rate as f64,
                        2.0,
                        params,
                        samples.len(),
                        1,
                    )?;
                    let output = resampler.process(&[&samples], None)?;
                    output.into_iter().next().unwrap_or_default()
                } else {
                    samples
                };

                let items = state2
                    .aligner
                    .align(&samples_16k, &align_text)
                    .map_err(|e| anyhow::anyhow!("Aligner: {e}"))?;

                Ok(items
                    .iter()
                    .map(|item| {
                        serde_json::json!({
                            "word": item.word,
                            "start": item.start_time,
                            "end": item.end_time,
                        })
                    })
                    .collect())
            })
            .await
            .map_err(|e| err(e))?
            .map_err(err)?;

        use base64::Engine;
        let audio_b64 = base64::engine::general_purpose::STANDARD.encode(&wav_bytes);

        return Ok(Json(serde_json::json!({
            "audio_b64": audio_b64,
            "alignment": alignment,
        }))
        .into_response());
    }

    Ok(([(axum::http::header::CONTENT_TYPE, "audio/wav")], wav_bytes).into_response())
}

// ==================== G2P SCAN ====================

#[derive(Deserialize)]
struct G2pScanBody {
    text: String,
}

async fn api_g2p_scan(Json(body): Json<G2pScanBody>) -> Result<Response, AppError> {
    let text = body.text;
    let unknown = tokio::task::spawn_blocking(move || tts::detect_unknown_words(&text))
        .await
        .map_err(|e| err(e))?;
    Ok(Json(serde_json::json!({ "unknown_words": unknown })).into_response())
}

async fn api_tts_backends(State(state): State<Arc<AppState>>) -> Result<Response, AppError> {
    Ok(Json(serde_json::json!({
        "backends": state.tts.available_backends(),
    }))
    .into_response())
}

// ==================== ASR ====================

async fn api_asr_transcribe(
    State(state): State<Arc<AppState>>,
    body: axum::body::Bytes,
) -> Result<Response, AppError> {
    // body is raw WAV bytes — decode to f32 samples, resample to 16kHz
    let state2 = state.clone();
    let result = tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
        let cursor = std::io::Cursor::new(body.to_vec());
        let mut reader =
            hound::WavReader::new(cursor).map_err(|e| anyhow::anyhow!("WAV decode: {e}"))?;
        let spec = reader.spec();

        // Convert to f32 samples
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

        // Convert to mono if stereo
        let mono = if spec.channels > 1 {
            samples_f32
                .chunks(spec.channels as usize)
                .map(|ch| ch.iter().sum::<f32>() / ch.len() as f32)
                .collect()
        } else {
            samples_f32
        };

        // Resample to 16kHz if needed
        let samples_16k = if spec.sample_rate != 16000 {
            use rubato::{
                Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType,
                WindowFunction,
            };
            let params = SincInterpolationParameters {
                sinc_len: 256,
                f_cutoff: 0.95,
                interpolation: SincInterpolationType::Linear,
                oversampling_factor: 256,
                window: WindowFunction::BlackmanHarris2,
            };
            let mut resampler = SincFixedIn::<f32>::new(
                16000.0 / spec.sample_rate as f64,
                2.0,
                params,
                mono.len(),
                1,
            )?;
            let output = resampler.process(&[&mono], None)?;
            output.into_iter().next().unwrap_or_default()
        } else {
            mono
        };

        let result = state2
            .asr
            .transcribe_samples(&samples_16k, qwen3_asr::TranscribeOptions::default())
            .map_err(|e| anyhow::anyhow!("ASR: {e}"))?;

        Ok(result.text)
    })
    .await
    .map_err(|e| err(e))?
    .map_err(err)?;

    Ok(Json(serde_json::json!({ "text": result })).into_response())
}

/// Run both ASR models on uploaded audio. Returns {qwen, parakeet}.
async fn api_asr_dual(
    State(state): State<Arc<AppState>>,
    body: axum::body::Bytes,
) -> Result<Response, AppError> {
    let state2 = state.clone();
    let result = tokio::task::spawn_blocking(move || -> anyhow::Result<(String, String)> {
        let cursor = std::io::Cursor::new(body.to_vec());
        let mut reader =
            hound::WavReader::new(cursor).map_err(|e| anyhow::anyhow!("WAV decode: {e}"))?;
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
        let mono: Vec<f32> = if spec.channels > 1 {
            samples_f32
                .chunks(spec.channels as usize)
                .map(|ch| ch.iter().sum::<f32>() / ch.len() as f32)
                .collect()
        } else {
            samples_f32
        };
        let samples_16k = tts::resample_to_16k(&mono, spec.sample_rate)?;

        let qwen = state2
            .asr
            .transcribe_samples(&samples_16k, qwen3_asr::TranscribeOptions::default())
            .map(|r| r.text)
            .unwrap_or_default();
        let parakeet = {
            let mut p = state2.parakeet.lock().unwrap();
            p.transcribe_samples(samples_16k.to_vec(), 16000, 1, None)
                .map(|r| r.text)
                .unwrap_or_default()
        };
        Ok((qwen, parakeet))
    })
    .await
    .map_err(|e| err(e))?
    .map_err(err)?;

    Ok(Json(serde_json::json!({"qwen": result.0, "parakeet": result.1})).into_response())
}

// ==================== FORCED ALIGNMENT ====================

async fn api_align(
    State(state): State<Arc<AppState>>,
    mut multipart: axum::extract::Multipart,
) -> Result<Response, AppError> {
    let mut audio_bytes: Option<Vec<u8>> = None;
    let mut text: Option<String> = None;

    while let Some(field) = multipart.next_field().await.map_err(|e| err(e))? {
        match field.name() {
            Some("audio") => {
                audio_bytes = Some(field.bytes().await.map_err(|e| err(e))?.to_vec());
            }
            Some("text") => {
                text = Some(field.text().await.map_err(|e| err(e))?);
            }
            _ => {}
        }
    }

    let audio_bytes = audio_bytes.ok_or_else(|| err(anyhow::anyhow!("missing 'audio' field")))?;
    let text = text.ok_or_else(|| err(anyhow::anyhow!("missing 'text' field")))?;

    let state2 = state.clone();
    let result =
        tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<qwen3_asr::ForcedAlignItem>> {
            // Decode WAV
            let cursor = std::io::Cursor::new(audio_bytes);
            let mut reader =
                hound::WavReader::new(cursor).map_err(|e| anyhow::anyhow!("WAV decode: {e}"))?;
            let spec = reader.spec();

            let samples_f32: Vec<f32> = match spec.sample_format {
                hound::SampleFormat::Float => {
                    reader.samples::<f32>().filter_map(|s| s.ok()).collect()
                }
                hound::SampleFormat::Int => {
                    let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
                    reader
                        .samples::<i32>()
                        .filter_map(|s| s.ok())
                        .map(|s| s as f32 / max)
                        .collect()
                }
            };

            let mono = if spec.channels > 1 {
                samples_f32
                    .chunks(spec.channels as usize)
                    .map(|ch| ch.iter().sum::<f32>() / ch.len() as f32)
                    .collect()
            } else {
                samples_f32
            };

            // Resample to 16kHz
            let samples_16k = if spec.sample_rate != 16000 {
                use rubato::{
                    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType,
                    WindowFunction,
                };
                let params = SincInterpolationParameters {
                    sinc_len: 256,
                    f_cutoff: 0.95,
                    interpolation: SincInterpolationType::Linear,
                    oversampling_factor: 256,
                    window: WindowFunction::BlackmanHarris2,
                };
                let mut resampler = SincFixedIn::<f32>::new(
                    16000.0 / spec.sample_rate as f64,
                    2.0,
                    params,
                    mono.len(),
                    1,
                )?;
                let output = resampler.process(&[&mono], None)?;
                output.into_iter().next().unwrap_or_default()
            } else {
                mono
            };

            state2
                .aligner
                .align(&samples_16k, &text)
                .map_err(|e| anyhow::anyhow!("Aligner: {e}"))
        })
        .await
        .map_err(|e| err(e))?
        .map_err(err)?;

    let alignment: Vec<serde_json::Value> = result
        .iter()
        .map(|item| {
            serde_json::json!({
                "word": item.word,
                "start": item.start_time,
                "end": item.end_time,
            })
        })
        .collect();

    Ok(Json(serde_json::json!({ "alignment": alignment })).into_response())
}

// ==================== HARK IMPORT ====================

async fn api_hark_import(State(state): State<Arc<AppState>>) -> Result<Response, AppError> {
    let db = state.db.lock().unwrap();
    let count = db.import_hark_log(&state.log_path).map_err(err)?;
    Ok(Json(serde_json::json!({ "imported": count })).into_response())
}

// ==================== JOBS ====================

async fn api_jobs(State(state): State<Arc<AppState>>) -> Result<Response, AppError> {
    let db = state.db.lock().unwrap();
    let jobs = db.list_jobs().map_err(err)?;
    Ok(Json(jobs).into_response())
}

async fn api_job_detail(
    State(state): State<Arc<AppState>>,
    Path(id): Path<i64>,
) -> Result<Response, AppError> {
    let db = state.db.lock().unwrap();
    match db.get_job(id).map_err(err)? {
        Some(job) => Ok(Json(job).into_response()),
        None => Ok((StatusCode::NOT_FOUND, "not found").into_response()),
    }
}

// ==================== CLI ====================

#[derive(clap::Parser)]
struct Cli {
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    #[arg(long, default_value = "3456")]
    port: u16,

    #[arg(long, default_value = "corpus.db")]
    db: String,

    #[arg(long)]
    log: Option<String>,

    #[arg(long, default_value = "~/bearcove")]
    docs_root: String,

    /// Voice reference WAV for pocket-tts
    #[arg(long, default_value = "voices/amos2_short.wav")]
    voice: String,

    /// Kokoro voice name (e.g. "am_puck", "am_adam", "af_heart")
    #[arg(long, default_value = "am_puck")]
    kokoro_voice: String,

    /// Number of parallel pocket-tts workers (share weights, separate state)
    #[arg(long, default_value = "2")]
    tts_workers: usize,

    /// Qwen3 ASR model directory (GGUF quantized)
    #[arg(
        long,
        default_value = "~/Library/Caches/qwen3-asr/Alkd--qwen3-asr-gguf--qwen3_asr_1_7b_q8_0_gguf"
    )]
    qwen_model: String,

    /// Parakeet TDT model directory
    #[arg(long, default_value = "models/parakeet-tdt")]
    parakeet_model: String,

    /// Qwen3 ForcedAligner model ID (downloaded from HuggingFace Hub)
    #[arg(long, default_value = "Qwen/Qwen3-ForcedAligner-0.6B")]
    aligner_model: String,

    /// Cache directory for HuggingFace Hub downloads
    #[arg(long, default_value = "~/Library/Caches/qwen3-asr")]
    hf_cache: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    use clap::Parser;

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "synth_dashboard=debug,info".parse().unwrap()),
        )
        .init();

    let cli = Cli::parse();

    let log_path = cli
        .log
        .map(std::path::PathBuf::from)
        .unwrap_or_else(dirs_log_path);
    let docs_root = shellexpand::tilde(&cli.docs_root).to_string();

    // Load CMUdict for unknown word detection
    tts::init_cmudict();

    // Open DB and run migrations
    let db = Db::open(std::path::Path::new(&cli.db))?;

    // Clean up orphaned running jobs from previous process
    db.fail_orphaned_jobs()?;

    // Seed pronunciation overrides
    let seeded = db.seed_overrides()?;
    if seeded > 0 {
        eprintln!("Seeded {seeded} pronunciation overrides");
    }

    // Auto-import Hark log
    if log_path.exists() {
        match db.import_hark_log(&log_path) {
            Ok(n) if n > 0 => eprintln!("Imported {n} transcriptions from {}", log_path.display()),
            Ok(_) => {}
            Err(e) => eprintln!("Warning: could not import log: {e}"),
        }
    }

    // Load all available TTS backends
    let tts_manager = tts::init(&cli.voice, &cli.kokoro_voice, cli.tts_workers);
    eprintln!("TTS backends: {:?}", tts_manager.available_backends());

    // Load Parakeet TDT
    eprintln!("Loading Parakeet TDT...");
    let parakeet = parakeet_rs::ParakeetTDT::from_pretrained(&cli.parakeet_model, None)?;
    eprintln!("Parakeet ready");

    // Load Qwen3 ASR model
    let qwen_model_dir = shellexpand::tilde(&cli.qwen_model).to_string();
    eprintln!("Loading Qwen3 ASR from {qwen_model_dir}...");
    let asr = qwen3_asr::AsrInference::load(
        std::path::Path::new(&qwen_model_dir),
        qwen3_asr::best_device(),
    )?;
    eprintln!("Qwen3 ASR ready");

    // Load ForcedAligner (downloads from HF Hub if needed)
    let hf_cache = shellexpand::tilde(&cli.hf_cache).to_string();
    eprintln!("Loading ForcedAligner ({})...", cli.aligner_model);
    let aligner = qwen3_asr::ForcedAligner::from_pretrained(
        &cli.aligner_model,
        std::path::Path::new(&hf_cache),
        qwen3_asr::best_device(),
    )?;
    eprintln!("ForcedAligner ready");

    let precompute_notify = std::sync::Arc::new(tokio::sync::Notify::new());
    let vocab_precompute_notify = std::sync::Arc::new(tokio::sync::Notify::new());
    let authored_asr_notify = std::sync::Arc::new(tokio::sync::Notify::new());
    let background_work_notify = std::sync::Arc::new(tokio::sync::Notify::new());
    let audio_dir = "audio".to_string();
    std::fs::create_dir_all(&audio_dir).ok();

    let state = Arc::new(AppState {
        db: Mutex::new(db),
        log_path,
        docs_root,
        tts: tts_manager,
        asr,
        parakeet: std::sync::Mutex::new(parakeet),
        aligner,
        review: Mutex::new(review::ReviewSession::new()),
        precompute_notify: precompute_notify.clone(),
        vocab_precompute_notify: vocab_precompute_notify.clone(),
        authored_asr_notify: authored_asr_notify.clone(),
        background_work_notify: background_work_notify.clone(),
        audio_dir: audio_dir.clone(),
        qwen_model_key: qwen_model_dir.clone(),
        job_cancel: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        inference_server: std::sync::Mutex::new(None),
    });

    // Start background pre-computation loop
    review::spawn_precompute_loop(state.clone(), precompute_notify, audio_dir.clone());
    review::spawn_vocab_precompute_loop(state.clone(), vocab_precompute_notify, audio_dir);
    jobs::spawn_authored_asr_precompute_loop(state.clone(), authored_asr_notify.clone());
    jobs::spawn_background_maintenance_loop(state.clone(), background_work_notify.clone());
    authored_asr_notify.notify_one();
    background_work_notify.notify_one();

    // ==================== AUTHORING ====================

    async fn api_author_next(State(state): State<Arc<AppState>>) -> Result<Response, AppError> {
        let db = state.db.lock().unwrap();
        match db.pick_term_for_authoring().map_err(err)? {
            Some((vocab, count)) => Ok(Json(serde_json::json!({
                "term": vocab.term,
                "spoken": vocab.spoken(),
                "sentence_count": count,
            }))
            .into_response()),
            None => Ok(Json(serde_json::json!({"done": true})).into_response()),
        }
    }

    #[derive(Deserialize)]
    struct AuthorSubmitBody {
        term: String,
        sentence: String,
        kind: Option<String>,
        surface_form: Option<String>,
    }

    async fn api_author_submit(
        State(state): State<Arc<AppState>>,
        Json(body): Json<AuthorSubmitBody>,
    ) -> Result<Response, AppError> {
        let sentence = body.sentence.trim().to_string();
        let kind = body.kind.as_deref().unwrap_or("term");
        let surface_form = body
            .surface_form
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string);
        if sentence.is_empty() {
            return Ok(Json(serde_json::json!({"error": "empty sentence"})).into_response());
        }
        if kind.eq_ignore_ascii_case("counterexample") {
            let Some(surface) = surface_form.as_deref() else {
                return Ok(Json(
                    serde_json::json!({"error": "counterexample requires a surface form"}),
                )
                .into_response());
            };
            if !sentence.to_lowercase().contains(&surface.to_lowercase()) {
                return Ok(Json(
                    serde_json::json!({"error": format!("Counterexample sentence must contain '{}'", surface)}),
                )
                .into_response());
            }
            if sentence.to_lowercase().contains(&body.term.to_lowercase()) {
                return Ok(Json(
                    serde_json::json!({"error": format!("Counterexample sentence must not contain canonical term '{}'", body.term)}),
                )
                .into_response());
            }
        } else if !sentence.to_lowercase().contains(&body.term.to_lowercase()) {
            return Ok(Json(
                serde_json::json!({"error": format!("Sentence must contain '{}'", body.term)}),
            )
            .into_response());
        }
        let db = state.db.lock().unwrap();
        let id = db
            .insert_authored_sentence_with_kind(
                &body.term,
                &sentence,
                kind,
                surface_form.as_deref(),
            )
            .map_err(err)?;
        Ok(Json(serde_json::json!({"id": id})).into_response())
    }

    async fn api_author_sentences(
        State(state): State<Arc<AppState>>,
    ) -> Result<Response, AppError> {
        let db = state.db.lock().unwrap();
        let sentences = db.list_authored_sentences().map_err(err)?;
        Ok(Json(sentences).into_response())
    }

    async fn api_author_sentence_recordings(
        State(state): State<Arc<AppState>>,
        Path(id): Path<i64>,
    ) -> Result<Response, AppError> {
        let db = state.db.lock().unwrap();
        let Some((term, sentence, kind, surface_form)) =
            db.get_authored_sentence(id).map_err(err)?
        else {
            return Ok(Json(serde_json::json!({"error": "sentence not found"})).into_response());
        };
        let recordings = db
            .authored_sentence_recordings_for_sentence(&sentence)
            .map_err(err)?;
        use base64::Engine as _;
        let recordings = recordings
            .into_iter()
            .map(|r| {
                let audio_b64 = std::fs::read(&r.wav_path)
                    .ok()
                    .map(|bytes| base64::engine::general_purpose::STANDARD.encode(bytes));
                let audio_mime = if r.wav_path.ends_with(".ogg") {
                    "audio/ogg"
                } else {
                    "audio/wav"
                };
                serde_json::json!({
                    "id": r.id,
                    "term": r.term,
                    "sentence": r.sentence,
                    "kind": r.kind,
                    "surface_form": r.surface_form,
                    "take_no": r.take_no,
                    "created_at": r.created_at,
                    "audio_b64": audio_b64,
                    "audio_mime": audio_mime,
                })
            })
            .collect::<Vec<_>>();
        Ok(Json(serde_json::json!({
            "sentence": sentence,
            "term": term,
            "kind": kind,
            "surface_form": surface_form,
            "recordings": recordings,
        }))
        .into_response())
    }

    async fn api_author_sentence_recording_upload(
        State(state): State<Arc<AppState>>,
        Path(id): Path<i64>,
        body: axum::body::Bytes,
    ) -> Result<Response, AppError> {
        let (term, sentence) = {
            let db = state.db.lock().unwrap();
            db.get_authored_sentence(id)
                .map_err(err)?
                .map(|(term, sentence, _kind, _surface_form)| (term, sentence))
                .ok_or_else(|| err("sentence not found"))?
        };

        let (term, sentence, mono, sample_rate) =
            tokio::task::spawn_blocking(move || -> Result<_, AppError> {
                let cursor = std::io::Cursor::new(body.to_vec());
                let mut reader = hound::WavReader::new(cursor).map_err(err)?;
                let spec = reader.spec();

                let samples_f32: Vec<f32> = match spec.sample_format {
                    hound::SampleFormat::Float => {
                        reader.samples::<f32>().filter_map(|s| s.ok()).collect()
                    }
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

                Ok((term, sentence, mono, spec.sample_rate))
            })
            .await
            .map_err(|e| err(e))??;

        let ogg_bytes = tts::encode_ogg_opus(&mono, sample_rate)
            .await
            .map_err(err)?;

        let audio_dir = state.audio_dir.clone();
        std::fs::create_dir_all(&audio_dir).ok();
        let (recording_id, take_no) = {
            let db = state.db.lock().unwrap();
            let take_no = db
                .next_authored_sentence_recording_take_no(&sentence)
                .map_err(err)?;
            let ogg_path = format!("{}/authored_{}_take_{}.ogg", audio_dir, id, take_no);
            std::fs::write(&ogg_path, &ogg_bytes).map_err(err)?;
            let recording_id = db
                .insert_authored_sentence_recording(&term, &sentence, take_no, &ogg_path)
                .map_err(err)?;
            (recording_id, take_no)
        };
        state.authored_asr_notify.notify_one();

        let recordings = {
            let db = state.db.lock().unwrap();
            db.authored_sentence_recordings_for_sentence(&sentence)
                .map_err(err)?
        };
        use base64::Engine as _;

        let result = serde_json::json!({
            "id": recording_id,
            "take_no": take_no,
            "sentence": sentence,
            "term": term,
            "recordings": recordings.into_iter().map(|r| {
                let audio_b64 = std::fs::read(&r.wav_path)
                    .ok()
                    .map(|bytes| base64::engine::general_purpose::STANDARD.encode(bytes));
                let audio_mime = if r.wav_path.ends_with(".ogg") { "audio/ogg" } else { "audio/wav" };
                serde_json::json!({
                    "id": r.id,
                    "term": r.term,
                    "sentence": r.sentence,
                    "take_no": r.take_no,
                    "created_at": r.created_at,
                    "audio_b64": audio_b64,
                    "audio_mime": audio_mime,
                })
            }).collect::<Vec<_>>()
        });

        Ok(Json(result).into_response())
    }

    async fn api_author_sentence_recording_delete(
        State(state): State<Arc<AppState>>,
        Path(recording_id): Path<i64>,
    ) -> Result<Response, AppError> {
        let recording = {
            let db = state.db.lock().unwrap();
            db.get_authored_sentence_recording(recording_id)
                .map_err(err)?
        };
        let Some(recording) = recording else {
            return Ok(Json(serde_json::json!({"error": "recording not found"})).into_response());
        };
        let _ = std::fs::remove_file(&recording.wav_path);
        let db = state.db.lock().unwrap();
        db.delete_authored_sentence_recording(recording_id)
            .map_err(err)?;
        Ok(Json(serde_json::json!({"ok": true})).into_response())
    }

    async fn api_author_sentence_recording_audio(
        State(state): State<Arc<AppState>>,
        Path(recording_id): Path<i64>,
    ) -> Result<Response, AppError> {
        let recording = {
            let db = state.db.lock().unwrap();
            db.get_authored_sentence_recording(recording_id)
                .map_err(err)?
        };
        let Some(recording) = recording else {
            return Ok((StatusCode::NOT_FOUND, "recording not found").into_response());
        };
        let bytes = std::fs::read(&recording.wav_path).map_err(err)?;
        let mime = if recording.wav_path.ends_with(".ogg") {
            "audio/ogg"
        } else {
            "audio/wav"
        };
        Ok(([(axum::http::header::CONTENT_TYPE, mime)], bytes).into_response())
    }

    #[derive(Deserialize)]
    struct AuthorSentenceUpdateBody {
        sentence: String,
    }

    async fn api_author_sentence_update(
        State(state): State<Arc<AppState>>,
        Path(id): Path<i64>,
        Json(body): Json<AuthorSentenceUpdateBody>,
    ) -> Result<Response, AppError> {
        let db = state.db.lock().unwrap();
        db.update_authored_sentence(id, body.sentence.trim())
            .map_err(err)?;
        Ok(Json(serde_json::json!({"ok": true})).into_response())
    }

    async fn api_author_sentence_delete(
        State(state): State<Arc<AppState>>,
        Path(id): Path<i64>,
    ) -> Result<Response, AppError> {
        let db = state.db.lock().unwrap();
        db.delete_authored_sentence(id).map_err(err)?;
        Ok(Json(serde_json::json!({"ok": true})).into_response())
    }

    #[derive(Deserialize)]
    struct RejectSuggestionBody {
        term: String,
    }

    async fn api_author_reject_suggestion(
        State(state): State<Arc<AppState>>,
        Json(body): Json<RejectSuggestionBody>,
    ) -> Result<Response, AppError> {
        let db = state.db.lock().unwrap();
        db.reject_suggestion(&body.term).map_err(err)?;
        Ok(Json(serde_json::json!({"ok": true})).into_response())
    }

    async fn api_author_stats(State(state): State<Arc<AppState>>) -> Result<Response, AppError> {
        let db = state.db.lock().unwrap();
        let total = db.authored_sentence_count().map_err(err)?;
        let terms = db.authored_sentence_term_counts().map_err(err)?;
        let counterexamples_total = db.authored_counterexample_count().map_err(err)?;
        let counterexample_terms = db.authored_counterexample_term_counts().map_err(err)?;
        let recordings_total = db.authored_sentence_recordings_count().map_err(err)?;
        let sentences_with_recordings =
            db.authored_sentences_with_recordings_count().map_err(err)?;
        let vocab_count = db.list_reviewed_vocab().map_err(err)?.len();
        Ok(Json(serde_json::json!({
            "total_sentences": total,
            "counterexample_total": counterexamples_total,
            "vocab_count": vocab_count,
            "terms": terms,
            "counterexample_terms": counterexample_terms,
            "recordings_total": recordings_total,
            "sentences_with_recordings": sentences_with_recordings,
        }))
        .into_response())
    }

    /// Ask OpenAI to generate sentences for terms that need more coverage.
    async fn api_author_suggest_sentences(
        State(state): State<Arc<AppState>>,
    ) -> Result<Response, AppError> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| err(anyhow::anyhow!("OPENAI_API_KEY not set")))?;

        let (terms_needing, existing_sentences) = {
            let db = state.db.lock().unwrap();
            let vocab = db.list_reviewed_vocab().map_err(err)?;
            let term_counts = db.authored_sentence_term_counts().map_err(err)?;
            let count_map: std::collections::HashMap<String, i64> = term_counts
                .into_iter()
                .map(|(t, c)| (t.to_lowercase(), c))
                .collect();

            // Pick the 10 terms with fewest sentences
            let mut needing: Vec<(String, String, Option<String>, i64)> = vocab
                .iter()
                .map(|v| {
                    let count = count_map.get(&v.term.to_lowercase()).copied().unwrap_or(0);
                    (
                        v.term.clone(),
                        v.spoken().to_string(),
                        v.description.clone(),
                        count,
                    )
                })
                .collect();
            needing.sort_by_key(|(_, _, _, c)| *c);
            needing.truncate(10);

            let sents = db.all_authored_sentences().map_err(err)?;
            (needing, sents)
        };

        if terms_needing.is_empty() {
            return Ok(Json(serde_json::json!({"suggestions": []})).into_response());
        }

        let terms_str: Vec<String> = terms_needing
            .iter()
            .map(|(t, spoken, desc, c)| {
                let desc_str = desc
                    .as_deref()
                    .map(|d| format!(", {d}"))
                    .unwrap_or_default();
                format!("{t} (pronounced \"{spoken}\"{desc_str}, {c} sentences so far)")
            })
            .collect();

        let client = reqwest::Client::new();
        let resp = client.post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&serde_json::json!({
                "model": "gpt-5-mini",
                "messages": [{
                    "role": "system",
                    "content": "You help a developer build training sentences for an ASR correction model. Generate natural sentences that a software developer would actually say out loud — the kind of thing you'd say to a colleague, in a meeting, or while dictating code notes. Each sentence must contain the given technical term. Vary sentence structure, length, and context. Output JSON: {\"suggestions\": [{\"term\": \"...\", \"sentence\": \"...\"}]}"
                }, {
                    "role": "user",
                    "content": format!("Generate 3 sentences for each of these terms:\n{}\n\nExisting sentences for style reference:\n{}",
                        terms_str.join("\n"),
                        existing_sentences.iter().take(20).cloned().collect::<Vec<_>>().join("\n"))
                }],
                "response_format": {"type": "json_object"},
            }))
            .send().await.map_err(|e| err(e))?;

        let body: serde_json::Value = resp.json().await.map_err(|e| err(e))?;
        let content = body["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("{}");
        let mut parsed: serde_json::Value =
            serde_json::from_str(content).unwrap_or(serde_json::json!({"suggestions": []}));

        Ok(Json(parsed).into_response())
    }

    /// Spellcheck all authored sentences via OpenAI. Returns list of issues.
    async fn api_author_spellcheck(
        State(state): State<Arc<AppState>>,
    ) -> Result<Response, AppError> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| err(anyhow::anyhow!("OPENAI_API_KEY not set")))?;

        let sentences = {
            let db = state.db.lock().unwrap();
            db.list_authored_sentences().map_err(err)?
        };

        let lines: Vec<String> = sentences
            .iter()
            .map(|s| format!("[{}] {}", s["id"], s["sentence"].as_str().unwrap_or("")))
            .collect();

        let client = reqwest::Client::new();
        let resp = client.post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&serde_json::json!({
                "model": "gpt-5-mini",
                "messages": [{
                    "role": "system",
                    "content": "You are a proofreader. The user will give you numbered sentences about programming. Find typos, misspellings, and grammar errors. Ignore technical terms, crate names, tool names, acronyms — they're intentional. Output JSON: {\"issues\": [{\"id\": number, \"original\": \"the wrong word\", \"suggestion\": \"the fix\", \"reason\": \"brief explanation\"}]}. If no issues found, output {\"issues\": []}."
                }, {
                    "role": "user",
                    "content": lines.join("\n")
                }],
                "response_format": {"type": "json_object"},
            }))
            .send().await.map_err(|e| err(e))?;

        let body: serde_json::Value = resp.json().await.map_err(|e| err(e))?;
        let content = body["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("[]");
        let issues: serde_json::Value =
            serde_json::from_str(content).unwrap_or(serde_json::json!([]));

        Ok(Json(sanitize_spellcheck_issues(issues)).into_response())
    }

    /// Ask OpenAI to suggest new vocab terms based on existing ones + sentences.
    async fn api_author_suggest_vocab(
        State(state): State<Arc<AppState>>,
    ) -> Result<Response, AppError> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| err(anyhow::anyhow!("OPENAI_API_KEY not set")))?;

        let (existing_terms, sentences, rejected) = {
            let db = state.db.lock().unwrap();
            let vocab = db.list_reviewed_vocab().map_err(err)?;
            let terms: Vec<String> = vocab
                .iter()
                .map(|v| {
                    if let Some(desc) = &v.description {
                        format!("{} ({})", v.term, desc)
                    } else {
                        v.term.clone()
                    }
                })
                .collect();
            let sents = db.all_authored_sentences().map_err(err)?;
            let rejected = db.list_rejected_suggestions().map_err(err)?;
            (terms, sents, rejected)
        };

        let all_excluded: Vec<String> = existing_terms
            .iter()
            .cloned()
            .chain(rejected.iter().cloned())
            .collect();
        let excluded_terms: HashSet<String> = all_excluded
            .iter()
            .map(|term| normalize_suggested_term(term))
            .collect();
        let user_msg = format!("DO NOT SUGGEST any of these (already in vocab or previously rejected): {}\n\nExample sentences for style reference:\n{}",
            all_excluded.join(", "),
            sentences.iter().take(30).cloned().collect::<Vec<String>>().join("\n"));

        let client = reqwest::Client::new();
        let resp = client.post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&serde_json::json!({
                "model": "gpt-5-mini",
                "messages": [{
                    "role": "system",
                    "content": "You help build a vocabulary of technical terms that speech recognition gets wrong. Given existing vocab and example sentences, suggest 20 more terms that the user likely uses and that ASR would struggle with. Focus on: programming tools, crate names, acronyms, technical jargon, project names, non-English-word identifiers. Do NOT suggest common English words or hyphenated compound words. For each term, provide a short description (what it is), a pronunciation (how a human would say it), and a natural example sentence. Output JSON: {\"suggestions\": [{\"term\": \"...\", \"description\": \"short description of what this is\", \"pronunciation\": \"how a human would say it phonetically\", \"sentence\": \"a natural sentence using the term\"}]}"
                }, {
                    "role": "user",
                    "content": user_msg
                }],
                "response_format": {"type": "json_object"},
            }))
            .send().await.map_err(|e| err(e))?;

        let body: serde_json::Value = resp.json().await.map_err(|e| err(e))?;
        let content = body["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("{}");
        let parsed: serde_json::Value =
            serde_json::from_str(content).unwrap_or(serde_json::json!({"suggestions": []}));

        let mut seen = HashSet::new();
        let filtered = parsed
            .get("suggestions")
            .and_then(|v| v.as_array())
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| {
                        let term = item.get("term")?.as_str()?.trim();
                        if term.is_empty() {
                            return None;
                        }
                        let normalized = normalize_suggested_term(term);
                        if normalized.is_empty()
                            || excluded_terms.contains(&normalized)
                            || !seen.insert(normalized)
                        {
                            return None;
                        }
                        Some(item.clone())
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        Ok(Json(serde_json::json!({"suggestions": filtered})).into_response())
    }

    let app = Router::new()
        // UI
        .route("/", get(index))
        // Stats
        .route("/api/stats", get(api_stats))
        // Vocab
        .route("/api/vocab", get(api_vocab_list).post(api_vocab_add))
        .route("/api/vocab/alt-spellings", get(api_vocab_alt_spellings))
        .route("/api/vocab/alt-spellings", post(api_vocab_alt_spelling_add))
        .route(
            "/api/vocab/alt-spellings/delete",
            post(api_vocab_alt_spelling_delete),
        )
        .route("/api/vocab/import", post(api_vocab_import))
        .route("/api/vocab/{id}", post(api_vocab_update))
        .route("/api/vocab/{id}/delete", post(api_vocab_delete))
        .route(
            "/api/settings",
            get(api_settings_get).post(api_settings_update),
        )
        // Candidates + Sentences
        .route("/api/candidates/import", post(api_candidates_import))
        .route("/api/sentences", get(api_sentences_list))
        .route("/api/sentences/generate", post(api_sentences_generate))
        .route("/api/sentences/{id}", post(api_sentence_update))
        // TTS + G2P
        .route("/api/tts/backends", get(api_tts_backends))
        .route("/api/tts/preview", post(api_tts_preview))
        .route("/api/g2p/scan", post(api_g2p_scan))
        // ASR
        .route("/api/asr/transcribe", post(api_asr_transcribe))
        .route("/api/asr/dual", post(api_asr_dual))
        // Forced alignment
        .route("/api/align", post(api_align))
        // Review (server-side orchestrated)
        .route("/api/review/current", get(review::api_review_current))
        .route(
            "/api/review/current/approve",
            post(review::api_review_approve),
        )
        .route(
            "/api/review/current/reject",
            post(review::api_review_reject),
        )
        .route(
            "/api/review/current/pronunciation",
            post(review::api_review_pronunciation),
        )
        .route(
            "/api/review/current/text",
            post(review::api_review_edit_text),
        )
        .route(
            "/api/review/current/backend",
            post(review::api_review_backend),
        )
        .route("/api/review/current/asr", post(review::api_review_asr))
        // Vocab review
        .route(
            "/api/review/vocab/current",
            get(review::api_vocab_review_current),
        )
        .route(
            "/api/review/vocab/approve",
            post(review::api_vocab_review_approve),
        )
        .route(
            "/api/review/vocab/reject",
            post(review::api_vocab_review_reject),
        )
        .route(
            "/api/review/vocab/pronunciation",
            post(review::api_vocab_review_pronunciation),
        )
        .route(
            "/api/review/vocab/backend",
            post(review::api_vocab_review_backend),
        )
        // Hark
        .route("/api/hark/import", post(api_hark_import))
        // Jobs
        .route("/api/jobs", get(api_jobs))
        .route("/api/jobs/{id}", get(api_job_detail))
        // Pipeline jobs
        .route("/api/jobs/corpus", post(jobs::api_start_corpus_job))
        .route("/api/jobs/prepare", post(jobs::api_start_prepare_job))
        .route("/api/jobs/train", post(jobs::api_start_train_job))
        .route("/api/jobs/eval", post(jobs::api_start_eval_job))
        .route("/api/jobs/vocab-scan", post(jobs::api_start_vocab_scan))
        .route("/api/jobs/curate", post(jobs::api_start_curate_job))
        .route("/api/jobs/stop", post(jobs::api_stop_job))
        .route("/api/pipeline/status", get(jobs::api_pipeline_status))
        .route(
            "/api/pipeline/template-coverage",
            get(jobs::api_template_coverage),
        )
        .route(
            "/api/pipeline/template-coverage/{term}",
            get(jobs::api_template_sentences),
        )
        .route("/api/pipeline/corpus", get(jobs::api_view_corpus))
        .route("/api/algorithm-tests", get(jobs::api_algorithm_tests))
        .route(
            "/api/pipeline/corpus/{id}/audio",
            get(jobs::api_corpus_audio),
        )
        .route("/api/pipeline/reset-corpus", post(jobs::api_reset_corpus))
        .route(
            "/api/pipeline/delete-corpus-term",
            post(jobs::api_delete_corpus_term),
        )
        .route(
            "/api/pipeline/reject-corpus-pair",
            post(jobs::api_reject_corpus_pair),
        )
        .route(
            "/api/pipeline/approve-corpus-pair",
            post(jobs::api_approve_corpus_pair),
        )
        .route(
            "/api/pipeline/eval-add-mistake",
            post(jobs::api_add_eval_mistake),
        )
        .route(
            "/api/pipeline/alt-spelling",
            post(jobs::api_add_alt_spelling),
        )
        .route("/api/confusions/next", get(jobs::api_confusions_next))
        .route(
            "/api/pipeline/preview-training",
            get(jobs::api_preview_training),
        )
        .route(
            "/api/pipeline/reset-training",
            post(jobs::api_reset_training),
        )
        .route("/api/pipeline/scan-results", get(jobs::api_scan_results))
        .route("/api/correct", post(jobs::api_correct))
        .route("/api/test-term", post(jobs::api_test_term))
        // Authoring
        .route("/api/author/next", get(api_author_next))
        .route("/api/author/submit", post(api_author_submit))
        .route("/api/author/stats", get(api_author_stats))
        .route("/api/author/sentences", get(api_author_sentences))
        .route(
            "/api/author/sentences/{id}/recordings",
            get(api_author_sentence_recordings),
        )
        .route(
            "/api/author/sentences/{id}/recordings",
            post(api_author_sentence_recording_upload),
        )
        .route(
            "/api/author/sentences/{id}",
            post(api_author_sentence_update),
        )
        .route(
            "/api/author/sentences/{id}/delete",
            post(api_author_sentence_delete),
        )
        .route(
            "/api/author/recordings/{id}",
            post(api_author_sentence_recording_delete),
        )
        .route(
            "/api/author/recordings/{id}/audio",
            get(api_author_sentence_recording_audio),
        )
        .route("/api/author/spellcheck", post(api_author_spellcheck))
        .route("/api/author/suggest-vocab", post(api_author_suggest_vocab))
        .route(
            "/api/author/reject-suggestion",
            post(api_author_reject_suggestion),
        )
        .route(
            "/api/author/suggest-sentences",
            post(api_author_suggest_sentences),
        )
        .with_state(state);

    let addr = format!("{}:{}", cli.host, cli.port);
    eprintln!("hark ml listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

fn dirs_log_path() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    std::path::PathBuf::from(home).join("Library/Application Support/hark/transcription_log.jsonl")
}
