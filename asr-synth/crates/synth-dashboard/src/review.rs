use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use axum::{
    extract::State,
    response::{IntoResponse, Response},
    Json,
};
use serde::Deserialize;
use tokio::sync::Notify;

use crate::db::{Db, SentenceRow};
use crate::tts;
use crate::{err, AppError, AppState};
use parakeet_rs::Transcriber;

/// Clean a word for comparison: strip non-alphanumeric, lowercase.
fn clean_word(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_alphanumeric())
        .collect::<String>()
        .to_lowercase()
}

/// Align original words to aligner output using greedy concatenation.
///
/// The aligner may:
/// - Drop words entirely (punctuation, short words)
/// - Split one word into multiple tokens ("HIR" → "H", "I", "R")
/// - Transform words ("don't" → "dont")
///
/// Strategy: walk original words. For each, try to match by consuming one or more
/// consecutive aligner tokens whose concatenation equals the cleaned original word.
/// If no match, the word was dropped — fill from nearest neighbor.
fn match_alignment(
    original_words: &[&str],
    aligner_items: &[qwen3_asr::ForcedAlignItem],
) -> Vec<serde_json::Value> {
    let mut aligned: Vec<(usize, f64, f64)> = Vec::new(); // (orig_idx, start, end)
    let mut ai = 0; // current position in aligner items

    for (oi, orig_word) in original_words.iter().enumerate() {
        let orig_clean = clean_word(orig_word);
        if orig_clean.is_empty() {
            // Punctuation-only token — will be filled from neighbor
            continue;
        }

        // Try consuming 1..=N aligner tokens starting at ai
        let mut matched = false;
        'outer: for take in 1..=5.min(aligner_items.len().saturating_sub(ai)) {
            let mut concat = String::new();
            for k in 0..take {
                concat.push_str(&clean_word(&aligner_items[ai + k].word));
            }
            if concat == orig_clean {
                // Match! Use the time range spanning all consumed tokens
                let start = aligner_items[ai].start_time;
                let end = aligner_items[ai + take - 1].end_time;
                aligned.push((oi, start, end));
                ai += take;
                matched = true;
                break 'outer;
            }
        }

        if !matched {
            // Maybe the aligner skipped ahead — peek if a later aligner token matches
            // (handles aligner inserting extra tokens before this word)
            for skip in 1..=3.min(aligner_items.len().saturating_sub(ai)) {
                for take in 1..=3.min(aligner_items.len().saturating_sub(ai + skip)) {
                    let mut concat = String::new();
                    for k in 0..take {
                        concat.push_str(&clean_word(&aligner_items[ai + skip + k].word));
                    }
                    if concat == orig_clean {
                        let start = aligner_items[ai + skip].start_time;
                        let end = aligner_items[ai + skip + take - 1].end_time;
                        aligned.push((oi, start, end));
                        ai = ai + skip + take;
                        matched = true;
                        break;
                    }
                }
                if matched {
                    break;
                }
            }
        }
        if !matched {
            eprintln!(
                "[align] DROPPED word {oi}: '{}' (clean: '{}')",
                orig_word, orig_clean
            );
        }
    }

    eprintln!(
        "[align] {} original words, {} aligned, {} dropped",
        original_words.len(),
        aligned.len(),
        original_words.len() - aligned.len()
    );

    // Build the full result. For dropped words, interpolate time slots between
    // their surrounding aligned neighbors so each gets a visible column.
    let mut result: Vec<serde_json::Value> = Vec::with_capacity(original_words.len());

    // First pass: identify runs of dropped words between aligned anchors
    // and assign interpolated time ranges.
    let mut oi = 0;
    let mut ai = 0; // index into aligned[]
    while oi < original_words.len() {
        if ai < aligned.len() && aligned[ai].0 == oi {
            // This word is aligned — ensure minimum duration so findWordAt can match it
            let start = aligned[ai].1;
            let end = if aligned[ai].2 <= start {
                start + 0.05
            } else {
                aligned[ai].2
            };
            result
                .push(serde_json::json!({"word": original_words[oi], "start": start, "end": end}));
            ai += 1;
            oi += 1;
        } else {
            // Run of dropped words: collect them until next aligned word (or end)
            let run_start = oi;
            while oi < original_words.len() && !(ai < aligned.len() && aligned[ai].0 == oi) {
                oi += 1;
            }
            let run_len = oi - run_start;

            // Place dropped words in unique time slots BEFORE the next aligned word.
            // Each gets a 50ms window, counting backwards from next_start.
            let prev_end = if ai > 0 { aligned[ai - 1].2 } else { 0.0 };
            let next_start = if ai < aligned.len() {
                aligned[ai].1
            } else {
                prev_end + 0.5
            };
            let slot = 0.05;
            // Place them just before next_start, working backwards
            let block_start = next_start - slot * run_len as f64;
            for k in 0..run_len {
                let start = block_start + slot * k as f64;
                let end = block_start + slot * (k + 1) as f64;
                // Clamp so we don't go before prev_end
                let start = start.max(prev_end + 0.001 * (k as f64 + 1.0));
                result.push(serde_json::json!({
                    "word": original_words[run_start + k],
                    "start": start,
                    "end": end,
                    "dropped": true,
                }));
            }
        }
    }

    if result.len() != original_words.len() {
        eprintln!(
            "[align] BUG: result has {} entries but original has {} words!",
            result.len(),
            original_words.len()
        );
    }

    // Log dropped words with their interpolated times
    for (i, (word, entry)) in original_words.iter().zip(result.iter()).enumerate() {
        let start = entry["start"].as_f64().unwrap_or(0.0);
        let end = entry["end"].as_f64().unwrap_or(0.0);
        let is_interpolated = end - start < 0.04;
        if is_interpolated {
            eprintln!(
                "[align] interpolated [{i}] '{}' t={:.3}..{:.3}",
                word, start, end
            );
        }
    }

    result
}

// ==================== Review Session State ====================

pub struct ReviewSession {
    pub current_id: Option<i64>,
    pub queue: VecDeque<i64>,
    pub backend: String,
    pub precomputed: HashMap<i64, PrecomputedData>,
    // Vocab review
    pub vocab_queue: VecDeque<i64>, // shuffled once, stable order
    pub vocab_precomputed: HashMap<i64, PrecomputedData>,
    /// The vocab ID currently being shown to the user (already popped from queue).
    pub vocab_current_id: Option<i64>,
    /// Which vocab ID the precompute loop is currently computing (if any).
    /// Used by /current to avoid redundant concurrent computation.
    pub vocab_computing: Option<i64>,
}

pub struct PrecomputedData {
    pub audio_b64: String,
    pub alignment: Vec<serde_json::Value>, // spoken text alignment (for waveform)
    pub written_alignment: Vec<serde_json::Value>, // written text alignment (for transcript grid)
    pub qwen_alignment: Vec<serde_json::Value>,
    pub parakeet_alignment: Vec<serde_json::Value>,
    pub wav_path: String,
    pub qwen_asr: String,
    pub parakeet_asr: String,
}

impl ReviewSession {
    pub fn new() -> Self {
        Self {
            current_id: None,
            queue: VecDeque::new(),
            backend: "pocket-hq".to_string(),
            precomputed: HashMap::new(),
            vocab_queue: VecDeque::new(),
            vocab_precomputed: HashMap::new(),
            vocab_current_id: None,
            vocab_computing: None,
        }
    }
}

// ==================== Compute Review Data ====================

/// Compute TTS + alignment for a sentence. Blocking — call from spawn_blocking.
pub fn compute_for_sentence(
    state: &Arc<AppState>,
    sentence: &SentenceRow,
    backend: &str,
    audio_dir: &str,
) -> anyhow::Result<PrecomputedData> {
    // Generate TTS — replace hyphens with spaces for better pronunciation
    let spoken_owned = sentence.spoken.replace('-', " ");

    // For very short text (single words, acronyms), wrap in a carrier phrase
    // so TTS has context for pronunciation, then crop to just the target word(s)
    let word_count = spoken_owned.split_whitespace().count();
    let carrier = word_count <= 3;
    let tts_text = if carrier {
        format!("The word is: {}.", spoken_owned)
    } else {
        spoken_owned.clone()
    };

    // TTS is async for remote backends — we need a runtime handle
    let rt = tokio::runtime::Handle::current();
    let mut audio = rt.block_on(state.tts.generate(backend, &tts_text))?;
    audio.normalize();

    // If we used a carrier phrase, align the full text and crop to just the target word(s)
    if carrier {
        let full_16k = tts::resample_to_16k(&audio.samples, audio.sample_rate)?;
        let carrier_items = state
            .aligner
            .align(&full_16k, &tts_text)
            .unwrap_or_default();
        // "The word is [target]." — skip first 3 words ("The", "word", "is")
        let skip = 3;
        if carrier_items.len() > skip {
            let start_time = carrier_items[skip].start_time;
            let end_time = carrier_items
                .last()
                .map(|i| i.end_time)
                .unwrap_or(start_time + 1.0);
            // Add small padding
            let start_sample = ((start_time - 0.05).max(0.0) * audio.sample_rate as f64) as usize;
            let end_sample = ((end_time + 0.1) * audio.sample_rate as f64)
                .min(audio.samples.len() as f64) as usize;
            if start_sample < end_sample && end_sample <= audio.samples.len() {
                audio.samples = audio.samples[start_sample..end_sample].to_vec();
            }
        }
    }

    let wav_bytes = audio.to_wav()?;

    // Save WAV to disk
    std::fs::create_dir_all(audio_dir).ok();
    let wav_path = format!("{}/{}.wav", audio_dir, sentence.id);
    std::fs::write(&wav_path, &wav_bytes)?;

    // Resample to 16kHz for aligner
    let samples_16k = tts::resample_to_16k(&audio.samples, audio.sample_rate)?;
    let spoken_text = &spoken_owned;

    // Run forced alignment on spoken text (for waveform playback sync)
    let spoken_items = state
        .aligner
        .align(&samples_16k, spoken_text)
        .map_err(|e| anyhow::anyhow!("Aligner (spoken): {e}"))?;

    // alignment is built below after written_alignment, filling in dropped words

    // Run forced alignment on written text (for transcript grid display row)
    let original_words: Vec<&str> = sentence.text.split_whitespace().collect();
    let written_items = state
        .aligner
        .align(&samples_16k, &sentence.text)
        .unwrap_or_default();

    // Map aligner words back to original words, handling splits/drops/transforms
    let written_alignment = match_alignment(&original_words, &written_items);

    // Do the same for spoken alignment — fill in dropped words
    let spoken_words: Vec<&str> = spoken_text.split_whitespace().collect();
    let alignment = match_alignment(&spoken_words, &spoken_items);

    // Run ASR on the TTS audio (round-trip quality check)
    let qwen_asr = match state.asr.transcribe_samples(
        &samples_16k,
        qwen3_asr::TranscribeOptions::default().with_language("english"),
    ) {
        Ok(r) => r.text,
        Err(e) => {
            eprintln!("[review] Qwen ASR failed: {e}");
            String::new()
        }
    };

    let parakeet_asr = {
        let mut parakeet = state.parakeet.lock().unwrap();
        match parakeet.transcribe_samples(samples_16k.to_vec(), 16000, 1, None) {
            Ok(r) => r.text,
            Err(e) => {
                eprintln!("[review] Parakeet ASR failed: {e}");
                String::new()
            }
        }
    };

    // Run forced alignment on ASR outputs too (for time-based grouping)
    let align_to_json = |text: &str| -> Vec<serde_json::Value> {
        if text.is_empty() {
            return vec![];
        }
        match state.aligner.align(&samples_16k, text) {
            Ok(items) => items
                .iter()
                .map(|item| {
                    serde_json::json!({
                        "word": item.word, "start": item.start_time, "end": item.end_time,
                    })
                })
                .collect(),
            Err(e) => {
                eprintln!("[review] Aligner failed on ASR text: {e}");
                vec![]
            }
        }
    };
    let qwen_alignment = align_to_json(&qwen_asr);
    let parakeet_alignment = align_to_json(&parakeet_asr);

    // Encode audio as base64
    use base64::Engine;
    let audio_b64 = base64::engine::general_purpose::STANDARD.encode(&wav_bytes);

    // Store in DB
    let alignment_json = serde_json::to_string(&alignment)?;
    {
        let db = state.db.lock().unwrap();
        db.update_sentence_precomputed(
            sentence.id,
            &wav_path,
            &alignment_json,
            backend,
            &sentence.spoken,
        )?;
    }

    Ok(PrecomputedData {
        audio_b64,
        alignment,
        written_alignment,
        qwen_alignment,
        parakeet_alignment,
        wav_path,
        qwen_asr,
        parakeet_asr,
    })
}

/// Build the full JSON response for a review screen.
fn build_review_response(
    state: &Arc<AppState>,
    sentence: &SentenceRow,
    precomputed: &PrecomputedData,
    backend: &str,
) -> serde_json::Value {
    let backends = state.tts.available_backends();
    let unknown_words: Vec<String> =
        serde_json::from_str(&sentence.unknown_words).unwrap_or_default();

    let (approved, rejected, total) = {
        let db = state.db.lock().unwrap();
        db.sentence_count_by_status().unwrap_or((0, 0, 0))
    };
    let reviewed = approved + rejected;
    let remaining = total - reviewed;

    serde_json::json!({
        "sentence": {
            "id": sentence.id,
            "text": sentence.text,
            "spoken": sentence.spoken,
            "unknown_words": unknown_words,
            "status": sentence.status,
        },
        "audio_b64": precomputed.audio_b64,
        "alignment": precomputed.alignment,
        "written_alignment": precomputed.written_alignment,
        "asr": {
            "qwen": precomputed.qwen_asr,
            "parakeet": precomputed.parakeet_asr,
            "qwen_alignment": precomputed.qwen_alignment,
            "parakeet_alignment": precomputed.parakeet_alignment,
        },
        "backend": backend,
        "backends": backends,
        "progress": {
            "reviewed": reviewed,
            "total": total,
            "remaining": remaining,
        },
        "ready": true,
    })
}

/// Ensure the review session has a current sentence and queue is populated.
/// Returns the current sentence ID or None if nothing left.
fn ensure_current(state: &Arc<AppState>) -> Option<i64> {
    let mut review = state.review.lock().unwrap();

    // If current sentence is gone or already reviewed, clear it
    if let Some(id) = review.current_id {
        let db = state.db.lock().unwrap();
        match db.get_sentence(id) {
            Ok(Some(s)) if s.status == "pending" => return Some(id),
            _ => {
                review.current_id = None;
            }
        }
    }

    // Pop from queue until we find a valid pending sentence
    while let Some(id) = review.queue.pop_front() {
        let db = state.db.lock().unwrap();
        if let Ok(Some(s)) = db.get_sentence(id) {
            if s.status == "pending" {
                review.current_id = Some(id);
                return Some(id);
            }
        }
    }

    // Queue empty — refill from DB
    {
        let db = state.db.lock().unwrap();
        // Auto-promote candidates if needed
        let pending_count = db.pending_sentence_ids(1).map(|v| v.len()).unwrap_or(0);
        if pending_count == 0 {
            let candidates = db.pick_candidates(50, true).unwrap_or_default();
            for (text, spoken, vocab_terms, unknown_words) in &candidates {
                let _ = db.insert_sentence_from_candidate(text, spoken, vocab_terms, unknown_words);
            }
        }

        if let Ok(ids) = db.pending_sentence_ids(50) {
            for id in ids {
                review.queue.push_back(id);
            }
        }
    }

    // Try again
    if let Some(id) = review.queue.pop_front() {
        review.current_id = Some(id);
        Some(id)
    } else {
        None
    }
}

// ==================== Background Pre-computation ====================

pub fn spawn_precompute_loop(state: Arc<AppState>, notify: Arc<Notify>, audio_dir: String) {
    tokio::spawn(async move {
        loop {
            notify.notified().await;

            // Grab the next few IDs that need precomputation
            let (ids_to_compute, backend) = {
                let review = state.review.lock().unwrap();
                let backend = review.backend.clone();
                let mut ids = Vec::new();
                for id in &review.queue {
                    if !review.precomputed.contains_key(id) && ids.len() < 3 {
                        ids.push(*id);
                    }
                }
                (ids, backend)
            };

            for id in ids_to_compute {
                let sentence = {
                    let db = state.db.lock().unwrap();
                    db.get_sentence(id).ok().flatten()
                };
                let Some(sentence) = sentence else { continue };
                if sentence.status != "pending" {
                    continue;
                }

                let state2 = state.clone();
                let backend2 = backend.clone();
                let audio_dir2 = audio_dir.clone();

                // Run TTS + alignment on blocking thread
                let result = tokio::task::spawn_blocking(move || {
                    compute_for_sentence(&state2, &sentence, &backend2, &audio_dir2)
                })
                .await;

                match result {
                    Ok(Ok(data)) => {
                        eprintln!("[precompute] sentence {} ready", id);
                        let mut review = state.review.lock().unwrap();
                        review.precomputed.insert(id, data);
                    }
                    Ok(Err(e)) => {
                        eprintln!("[precompute] sentence {} failed: {e}", id);
                    }
                    Err(e) => {
                        eprintln!("[precompute] sentence {} task failed: {e}", id);
                    }
                }
            }
        }
    });
}

pub fn spawn_vocab_precompute_loop(state: Arc<AppState>, notify: Arc<Notify>, audio_dir: String) {
    tokio::spawn(async move {
        loop {
            notify.notified().await;

            let backend = {
                let review = state.review.lock().unwrap();
                review.backend.clone()
            };

            // Get next few from the stable queue that aren't precomputed yet
            ensure_vocab_queue(&state);
            let ids = {
                let review = state.review.lock().unwrap();
                review
                    .vocab_queue
                    .iter()
                    .filter(|id| !review.vocab_precomputed.contains_key(id))
                    .take(3)
                    .copied()
                    .collect::<Vec<_>>()
            };

            for id in ids {
                let vocab = {
                    let db = state.db.lock().unwrap();
                    db.get_vocab(id).ok().flatten()
                };
                let Some(vocab) = vocab else { continue };
                if vocab.reviewed {
                    continue;
                }

                // Mark this ID as being computed so /current can wait instead of racing
                {
                    let mut review = state.review.lock().unwrap();
                    review.vocab_computing = Some(id);
                }

                let sentence = vocab_to_sentence(&vocab);
                let state2 = state.clone();
                let backend2 = backend.clone();
                let audio_dir2 = audio_dir.clone();

                let result = tokio::task::spawn_blocking(move || {
                    compute_for_sentence(&state2, &sentence, &backend2, &audio_dir2)
                })
                .await;

                {
                    let mut review = state.review.lock().unwrap();
                    review.vocab_computing = None;
                    match result {
                        Ok(Ok(data)) => {
                            eprintln!("[vocab-precompute] {} ready", vocab.term);
                            review.vocab_precomputed.insert(id, data);
                        }
                        Ok(Err(e)) => eprintln!("[vocab-precompute] {} failed: {e}", vocab.term),
                        Err(e) => eprintln!("[vocab-precompute] {} task failed: {e}", vocab.term),
                    }
                }
            }
        }
    });
}

// ==================== API Endpoints ====================

pub async fn api_review_current(State(state): State<Arc<AppState>>) -> Result<Response, AppError> {
    let Some(id) = ensure_current(&state) else {
        return Ok(Json(serde_json::json!({
            "sentence": null,
            "ready": true,
        }))
        .into_response());
    };

    // Check if we have precomputed data
    let precomputed = {
        let mut review = state.review.lock().unwrap();
        review.precomputed.remove(&id)
    };

    if let Some(data) = precomputed {
        let sentence = {
            let db = state.db.lock().unwrap();
            db.get_sentence(id)
                .map_err(err)?
                .ok_or_else(|| err(anyhow::anyhow!("sentence gone")))?
        };
        let backend = state.review.lock().unwrap().backend.clone();
        let response = build_review_response(&state, &sentence, &data, &backend);

        // Trigger precomputation of next sentences
        state.precompute_notify.notify_one();

        return Ok(Json(response).into_response());
    }

    // Not precomputed — compute synchronously (cold start)
    let sentence = {
        let db = state.db.lock().unwrap();
        db.get_sentence(id)
            .map_err(err)?
            .ok_or_else(|| err(anyhow::anyhow!("sentence gone")))?
    };
    let backend = state.review.lock().unwrap().backend.clone();
    let audio_dir = state.audio_dir.clone();

    let state2 = state.clone();
    let backend2 = backend.clone();
    let data = tokio::task::spawn_blocking(move || {
        compute_for_sentence(&state2, &sentence, &backend2, &audio_dir)
    })
    .await
    .map_err(|e| err(e))?
    .map_err(err)?;

    // Re-read sentence (may have been updated by compute)
    let sentence = {
        let db = state.db.lock().unwrap();
        db.get_sentence(id)
            .map_err(err)?
            .ok_or_else(|| err(anyhow::anyhow!("sentence gone")))?
    };

    let response = build_review_response(&state, &sentence, &data, &backend);

    // Trigger precomputation of next sentences
    state.precompute_notify.notify_one();

    Ok(Json(response).into_response())
}

pub async fn api_review_approve(State(state): State<Arc<AppState>>) -> Result<Response, AppError> {
    let id = {
        let review = state.review.lock().unwrap();
        review.current_id
    };
    let Some(id) = id else {
        return Ok(Json(serde_json::json!({"error": "no current sentence"})).into_response());
    };

    {
        let db = state.db.lock().unwrap();
        db.update_sentence_status(id, "approved").map_err(err)?;
    }

    // Advance to next
    {
        let mut review = state.review.lock().unwrap();
        review.current_id = None;
        review.precomputed.remove(&id);
    }

    // Trigger precompute and return immediately — frontend fetches next separately
    state.precompute_notify.notify_one();
    Ok(Json(serde_json::json!({"ok": true})).into_response())
}

pub async fn api_review_reject(State(state): State<Arc<AppState>>) -> Result<Response, AppError> {
    let id = {
        let review = state.review.lock().unwrap();
        review.current_id
    };
    let Some(id) = id else {
        return Ok(Json(serde_json::json!({"error": "no current sentence"})).into_response());
    };

    {
        let db = state.db.lock().unwrap();
        db.update_sentence_status(id, "rejected").map_err(err)?;
    }

    // Advance to next
    {
        let mut review = state.review.lock().unwrap();
        review.current_id = None;
        review.precomputed.remove(&id);
    }

    state.precompute_notify.notify_one();
    Ok(Json(serde_json::json!({"ok": true})).into_response())
}

#[derive(Deserialize)]
pub struct PronunciationBody {
    word: String,
    spoken: String,
}

pub async fn api_review_pronunciation(
    State(state): State<Arc<AppState>>,
    Json(body): Json<PronunciationBody>,
) -> Result<Response, AppError> {
    let id = {
        let review = state.review.lock().unwrap();
        review.current_id
    };
    let Some(id) = id else {
        return Ok(Json(serde_json::json!({"error": "no current sentence"})).into_response());
    };

    // Update vocab override
    {
        let db = state.db.lock().unwrap();
        match db.find_vocab_by_term(&body.word) {
            Ok(Some(vocab)) => {
                eprintln!(
                    "[pronunciation] updating vocab '{}' (id={}) → '{}'",
                    vocab.term, vocab.id, body.spoken
                );
                db.update_vocab_override(vocab.id, Some(&body.spoken))
                    .map_err(err)?;
            }
            Ok(None) => {
                eprintln!(
                    "[pronunciation] vocab entry '{}' not found, inserting",
                    body.word
                );
                let _ = db.insert_candidate_vocab(&body.word, &body.spoken);
                // Now update the override
                if let Ok(Some(vocab)) = db.find_vocab_by_term(&body.word) {
                    db.update_vocab_override(vocab.id, Some(&body.spoken))
                        .map_err(err)?;
                }
            }
            Err(e) => eprintln!("[pronunciation] error finding vocab: {e}"),
        }
    }

    // Rebuild spoken form from scratch for current + queued sentences using ALL overrides
    let ids_to_update = {
        let review = state.review.lock().unwrap();
        let mut ids = vec![id];
        ids.extend(review.queue.iter());
        ids
    };
    {
        let db = state.db.lock().unwrap();
        let overrides = db.get_spoken_overrides().map_err(err)?;
        for sid in ids_to_update {
            if let Ok(Some(s)) = db.get_sentence(sid) {
                let new_spoken = tts::build_spoken_form(&s.text, &overrides);
                eprintln!(
                    "[pronunciation] sentence {sid}: '{}' → '{}'",
                    s.spoken, new_spoken
                );
                if new_spoken != s.spoken {
                    let _ = db.update_sentence_spoken(sid, &new_spoken);
                }
            }
        }
    }

    // Invalidate all precomputed data (spoken forms changed)
    {
        let mut review = state.review.lock().unwrap();
        review.precomputed.clear();
    }

    // Re-read updated sentence, re-compute TTS + alignment
    let sentence = {
        let db = state.db.lock().unwrap();
        db.get_sentence(id)
            .map_err(err)?
            .ok_or_else(|| err(anyhow::anyhow!("sentence gone")))?
    };
    let backend = state.review.lock().unwrap().backend.clone();
    let audio_dir = state.audio_dir.clone();
    let state2 = state.clone();
    let backend2 = backend.clone();

    let data = tokio::task::spawn_blocking(move || {
        compute_for_sentence(&state2, &sentence, &backend2, &audio_dir)
    })
    .await
    .map_err(|e| err(e))?
    .map_err(err)?;

    let sentence = {
        let db = state.db.lock().unwrap();
        db.get_sentence(id)
            .map_err(err)?
            .ok_or_else(|| err(anyhow::anyhow!("sentence gone")))?
    };

    let response = build_review_response(&state, &sentence, &data, &backend);

    // Trigger precomputation for queued sentences (with updated spoken forms)
    state.precompute_notify.notify_one();

    Ok(Json(response).into_response())
}

#[derive(Deserialize)]
pub struct EditTextBody {
    text: String,
}

/// Edit the sentence text (fix transcription errors). Rebuilds spoken form using vocab overrides, re-synths.
pub async fn api_review_edit_text(
    State(state): State<Arc<AppState>>,
    Json(body): Json<EditTextBody>,
) -> Result<Response, AppError> {
    let id = {
        let review = state.review.lock().unwrap();
        review.current_id
    };
    let Some(id) = id else {
        return Ok(Json(serde_json::json!({"error": "no current sentence"})).into_response());
    };

    // Update the text and rebuild spoken form using existing vocab overrides
    let new_text = body.text.trim().to_string();
    {
        let db = state.db.lock().unwrap();
        let overrides = db.get_spoken_overrides().map_err(err)?;
        let spoken = tts::build_spoken_form(&new_text, &overrides);

        // Update text, spoken, and unknown words
        let unknown = crate::tts::detect_unknown_words(&new_text);
        let unknown_json = serde_json::to_string(&unknown).unwrap_or_default();
        db.update_sentence_text(id, &new_text, &spoken, &unknown_json)
            .map_err(err)?;
    }

    // Invalidate precomputed data
    {
        let mut review = state.review.lock().unwrap();
        review.precomputed.clear();
    }

    // Re-compute TTS + alignment
    let sentence = {
        let db = state.db.lock().unwrap();
        db.get_sentence(id)
            .map_err(err)?
            .ok_or_else(|| err(anyhow::anyhow!("sentence gone")))?
    };
    let backend = state.review.lock().unwrap().backend.clone();
    let audio_dir = state.audio_dir.clone();
    let state2 = state.clone();
    let backend2 = backend.clone();

    let data = tokio::task::spawn_blocking(move || {
        compute_for_sentence(&state2, &sentence, &backend2, &audio_dir)
    })
    .await
    .map_err(|e| err(e))?
    .map_err(err)?;

    let sentence = {
        let db = state.db.lock().unwrap();
        db.get_sentence(id)
            .map_err(err)?
            .ok_or_else(|| err(anyhow::anyhow!("sentence gone")))?
    };

    let response = build_review_response(&state, &sentence, &data, &backend);
    state.precompute_notify.notify_one();
    Ok(Json(response).into_response())
}

#[derive(Deserialize)]
pub struct BackendBody {
    backend: String,
}

pub async fn api_review_backend(
    State(state): State<Arc<AppState>>,
    Json(body): Json<BackendBody>,
) -> Result<Response, AppError> {
    // Update backend and invalidate precomputed cache
    {
        let mut review = state.review.lock().unwrap();
        review.backend = body.backend.clone();
        review.precomputed.clear();
    }

    let id = {
        let review = state.review.lock().unwrap();
        review.current_id
    };
    let Some(id) = id else {
        return Ok(Json(serde_json::json!({"error": "no current sentence"})).into_response());
    };

    // Re-compute current sentence with new backend
    let sentence = {
        let db = state.db.lock().unwrap();
        db.get_sentence(id)
            .map_err(err)?
            .ok_or_else(|| err(anyhow::anyhow!("sentence gone")))?
    };
    let backend = body.backend.clone();
    let audio_dir = state.audio_dir.clone();
    let state2 = state.clone();
    let backend2 = backend.clone();

    let data = tokio::task::spawn_blocking(move || {
        compute_for_sentence(&state2, &sentence, &backend2, &audio_dir)
    })
    .await
    .map_err(|e| err(e))?
    .map_err(err)?;

    let sentence = {
        let db = state.db.lock().unwrap();
        db.get_sentence(id)
            .map_err(err)?
            .ok_or_else(|| err(anyhow::anyhow!("sentence gone")))?
    };

    let response = build_review_response(&state, &sentence, &data, &backend);

    // Trigger precomputation with new backend
    state.precompute_notify.notify_one();

    Ok(Json(response).into_response())
}

/// Run both ASR models on uploaded audio, then align the ASR text against the
/// TTS waveform so the transcript grid can display time-aligned human ASR results.
pub async fn api_review_asr(
    State(state): State<Arc<AppState>>,
    body: axum::body::Bytes,
) -> Result<Response, AppError> {
    let current_id = state.review.lock().unwrap().current_id;
    let audio_dir = state.audio_dir.clone();
    let state2 = state.clone();

    let result =
        tokio::task::spawn_blocking(move || -> anyhow::Result<serde_json::Value> {
            let wav_bytes = body.to_vec();

            // Decode human recording WAV
            let cursor = std::io::Cursor::new(wav_bytes);
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

            let mut mono: Vec<f32> = if spec.channels > 1 {
                samples_f32
                    .chunks(spec.channels as usize)
                    .map(|ch| ch.iter().sum::<f32>() / ch.len() as f32)
                    .collect()
            } else {
                samples_f32
            };

            // Normalize audio
            let peak = mono.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
            if peak > 0.001 {
                let gain = 0.95 / peak;
                for s in &mut mono {
                    *s *= gain;
                }
            }

            // Save normalized recording to disk for eval
            if let Some(id) = current_id {
                let human_wav_path = format!("{}/human_{}.wav", audio_dir, id);
                let mut audio = crate::tts::TtsAudio {
                    samples: mono.clone(),
                    sample_rate: spec.sample_rate,
                };
                match audio.to_wav() {
                    Ok(bytes) => {
                        if let Err(e) = std::fs::write(&human_wav_path, &bytes) {
                            eprintln!("[human-asr] Failed to save recording: {e}");
                        } else {
                            let db = state2.db.lock().unwrap();
                            let _ = db.update_sentence_human_wav(id, &human_wav_path);
                            eprintln!("[human-asr] Saved normalized recording to {human_wav_path}");
                        }
                    }
                    Err(e) => eprintln!("[human-asr] Failed to encode WAV: {e}"),
                }
            }

            let samples_16k = crate::tts::resample_to_16k(&mono, spec.sample_rate)?;

            // Run both ASR models on human recording
            let qwen = match state2.asr.transcribe_samples(
                &samples_16k,
                qwen3_asr::TranscribeOptions::default().with_language("english"),
            ) {
                Ok(r) => r.text,
                Err(e) => format!("(error: {e})"),
            };

            let parakeet = {
                let mut p = state2.parakeet.lock().unwrap();
                match p.transcribe_samples(samples_16k.to_vec(), 16000, 1, None) {
                    Ok(r) => r.text,
                    Err(e) => format!("(error: {e})"),
                }
            };

            // Run forced aligner on the HUMAN recording for all alignments
            let sentence = if let Some(id) = current_id {
                let db = state2.db.lock().unwrap();
                db.get_sentence(id).ok().flatten()
            } else {
                None
            };

            // Align spoken form against human audio
            let spoken_text = sentence
                .as_ref()
                .map(|s| s.spoken.replace('-', " "))
                .unwrap_or_default();
            let written_text = sentence
                .as_ref()
                .map(|s| s.text.clone())
                .unwrap_or_default();

            let spoken_items = if !spoken_text.is_empty() {
                state2
                    .aligner
                    .align(&samples_16k, &spoken_text)
                    .unwrap_or_default()
            } else {
                vec![]
            };

            let written_items = if !written_text.is_empty() {
                state2
                    .aligner
                    .align(&samples_16k, &written_text)
                    .unwrap_or_default()
            } else {
                vec![]
            };

            let spoken_words: Vec<&str> = spoken_text.split_whitespace().collect();
            let alignment = match_alignment(&spoken_words, &spoken_items);

            let original_words: Vec<&str> = written_text.split_whitespace().collect();
            let written_alignment = match_alignment(&original_words, &written_items);

            // Align ASR text against human audio
            let align_asr =
                |text: &str| -> Vec<serde_json::Value> {
                    if text.is_empty() {
                        return vec![];
                    }
                    match state2.aligner.align(&samples_16k, text) {
                Ok(items) => items.iter().map(|item| serde_json::json!({
                    "word": item.word, "start": item.start_time, "end": item.end_time,
                })).collect(),
                Err(e) => { eprintln!("[human-asr] Aligner failed: {e}"); vec![] }
            }
                };
            let qwen_alignment = align_asr(&qwen);
            let parakeet_alignment = align_asr(&parakeet);

            // Encode human audio as base64 for playback
            use base64::Engine;
            let human_wav_path = current_id.map(|id| format!("{}/human_{}.wav", audio_dir, id));
            let audio_b64 = if let Some(ref path) = human_wav_path {
                match std::fs::read(path) {
                    Ok(bytes) => Some(base64::engine::general_purpose::STANDARD.encode(&bytes)),
                    Err(_) => None,
                }
            } else {
                None
            };

            Ok(serde_json::json!({
                "qwen": qwen,
                "parakeet": parakeet,
                "qwen_alignment": qwen_alignment,
                "parakeet_alignment": parakeet_alignment,
                "alignment": alignment,
                "written_alignment": written_alignment,
                "audio_b64": audio_b64,
            }))
        })
        .await
        .map_err(|e| err(e))?
        .map_err(err)?;

    Ok(Json(result).into_response())
}

/// Load a WAV file from disk and resample to 16kHz mono.
// ==================== Vocab Review ====================

/// Convert a VocabRow into a fake SentenceRow for the review compute pipeline
fn vocab_to_sentence(vocab: &crate::db::VocabRow) -> crate::db::SentenceRow {
    crate::db::SentenceRow {
        id: vocab.id,
        text: vocab.term.clone(),
        spoken: vocab.spoken().to_string(),
        vocab_terms: "[]".to_string(),
        unknown_words: serde_json::json!([vocab.term]).to_string(),
        status: if vocab.reviewed {
            "approved"
        } else {
            "pending"
        }
        .to_string(),
        wav_path: None,
        alignment_json: None,
        tts_backend: None,
        parakeet_output: None,
        qwen_output: None,
        human_wav_path: None,
    }
}

pub async fn api_vocab_review_current(
    State(state): State<Arc<AppState>>,
) -> Result<Response, AppError> {
    // Get next from stable queue
    let Some(vocab_id) = next_vocab_id(&state) else {
        return Ok(Json(serde_json::json!({"sentence": null, "ready": true})).into_response());
    };

    // Track which term is currently being shown
    state.review.lock().unwrap().vocab_current_id = Some(vocab_id);

    serve_vocab_review(&state, vocab_id, true).await
}

/// Compute and return the review response for a specific vocab ID.
/// If `use_cache` is true, checks the precompute cache first (for /current).
/// If false, always recomputes (for pronunciation changes).
async fn serve_vocab_review(
    state: &Arc<AppState>,
    vocab_id: i64,
    use_cache: bool,
) -> Result<Response, AppError> {
    let vocab = {
        let db = state.db.lock().unwrap();
        db.get_vocab(vocab_id)
            .map_err(err)?
            .ok_or_else(|| err(anyhow::anyhow!("vocab gone")))?
    };

    let data = if use_cache {
        // Check precompute cache — if the precompute loop is actively computing
        // this item, wait for it instead of starting a competing computation
        'resolve: {
            // First check: already cached?
            if let Some(data) = state
                .review
                .lock()
                .unwrap()
                .vocab_precomputed
                .remove(&vocab_id)
            {
                break 'resolve data;
            }

            // If precompute loop is computing this exact item, wait for it
            let is_being_computed = state.review.lock().unwrap().vocab_computing == Some(vocab_id);
            if is_being_computed {
                for _ in 0..30 {
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                    let mut review = state.review.lock().unwrap();
                    if let Some(data) = review.vocab_precomputed.remove(&vocab_id) {
                        break 'resolve data;
                    }
                    if review.vocab_computing != Some(vocab_id) {
                        if let Some(data) = review.vocab_precomputed.remove(&vocab_id) {
                            break 'resolve data;
                        }
                        break;
                    }
                }
            }

            // Not cached and not being precomputed — compute inline
            let sentence = vocab_to_sentence(&vocab);
            let backend = state.review.lock().unwrap().backend.clone();
            let audio_dir = state.audio_dir.clone();
            let state2 = state.clone();
            let backend2 = backend.clone();
            tokio::task::spawn_blocking(move || {
                compute_for_sentence(&state2, &sentence, &backend2, &audio_dir)
            })
            .await
            .map_err(|e| err(e))?
            .map_err(err)?
        }
    } else {
        // Always recompute (pronunciation changed, cache is stale)
        let sentence = vocab_to_sentence(&vocab);
        let backend = state.review.lock().unwrap().backend.clone();
        let audio_dir = state.audio_dir.clone();
        let state2 = state.clone();
        let backend2 = backend.clone();
        tokio::task::spawn_blocking(move || {
            compute_for_sentence(&state2, &sentence, &backend2, &audio_dir)
        })
        .await
        .map_err(|e| err(e))?
        .map_err(err)?
    };

    // Trigger precompute of next terms
    state.vocab_precompute_notify.notify_one();

    let backend = state.review.lock().unwrap().backend.clone();
    let backends = state.tts.available_backends();
    let (reviewed, unreviewed, total) = {
        let db = state.db.lock().unwrap();
        db.vocab_review_counts().unwrap_or((0, 0, 0))
    };

    Ok(Json(serde_json::json!({
        "sentence": {
            "id": vocab.id,
            "text": vocab.term,
            "spoken": vocab.spoken(),
            "unknown_words": [vocab.term],
            "status": "pending",
        },
        "audio_b64": data.audio_b64,
        "alignment": data.alignment,
        "written_alignment": data.written_alignment,
        "asr": {
            "qwen": data.qwen_asr,
            "parakeet": data.parakeet_asr,
            "qwen_alignment": data.qwen_alignment,
            "parakeet_alignment": data.parakeet_alignment,
        },
        "backend": backend,
        "backends": backends,
        "progress": {
            "reviewed": reviewed,
            "total": total,
            "remaining": unreviewed,
        },
        "ready": true,
        "mode": "vocab",
    }))
    .into_response())
}

/// Ensure the vocab queue is populated with shuffled unreviewed IDs.
fn ensure_vocab_queue(state: &Arc<AppState>) {
    let mut review = state.review.lock().unwrap();
    if review.vocab_queue.is_empty() {
        drop(review); // release review lock before taking db lock
        let db = state.db.lock().unwrap();
        let mut ids = db.unreviewed_vocab_ids(200).unwrap_or_default();
        drop(db);
        use rand::seq::SliceRandom;
        ids.shuffle(&mut rand::rng());
        let mut review = state.review.lock().unwrap();
        review.vocab_queue = ids.into();
        eprintln!(
            "[vocab-review] queue refilled with {} terms",
            review.vocab_queue.len()
        );
    }
}

/// Invalidate the vocab queue (called after approve/reject)
fn invalidate_vocab_queue(state: &Arc<AppState>) {
    let mut review = state.review.lock().unwrap();
    review.vocab_queue.clear();
    review.vocab_precomputed.clear();
}

/// Pop the next vocab ID from the stable queue.
fn next_vocab_id(state: &Arc<AppState>) -> Option<i64> {
    ensure_vocab_queue(state);
    let mut review = state.review.lock().unwrap();
    review.vocab_queue.pop_front()
}

/// Eagerly precompute the next unreviewed vocab term. Called inline from approve/reject
/// so it races the frontend's fetch for /current.
async fn eager_vocab_precompute(state: &Arc<AppState>) {
    ensure_vocab_queue(state);
    let (id, vocab) = {
        let review = state.review.lock().unwrap();
        let db = state.db.lock().unwrap();
        // Find next queued ID that isn't precomputed yet
        let id = review
            .vocab_queue
            .iter()
            .find(|id| !review.vocab_precomputed.contains_key(id))
            .copied();
        match id {
            Some(id) => match db.get_vocab(id) {
                Ok(Some(v)) if !v.reviewed => (id, v),
                _ => return,
            },
            None => return,
        }
    };
    let sentence = vocab_to_sentence(&vocab);
    let backend = state.review.lock().unwrap().backend.clone();
    let audio_dir = state.audio_dir.clone();
    let state2 = state.clone();
    let result = tokio::task::spawn_blocking(move || {
        compute_for_sentence(&state2, &sentence, &backend, &audio_dir)
    })
    .await;
    if let Ok(Ok(data)) = result {
        let mut review = state.review.lock().unwrap();
        review.vocab_precomputed.insert(id, data);
        eprintln!("[eager-precompute] {} ready", vocab.term);
    }
}

pub async fn api_vocab_review_approve(
    State(state): State<Arc<AppState>>,
    Json(body): Json<serde_json::Value>,
) -> Result<Response, AppError> {
    let id = body["id"]
        .as_i64()
        .ok_or_else(|| err(anyhow::anyhow!("missing id")))?;
    {
        let db = state.db.lock().unwrap();
        db.mark_vocab_reviewed(id).map_err(err)?;
    }
    {
        let mut review = state.review.lock().unwrap();
        review.vocab_queue.retain(|&qid| qid != id);
        review.vocab_precomputed.remove(&id);
    }
    state.vocab_precompute_notify.notify_one();
    state.background_work_notify.notify_one();
    Ok(Json(serde_json::json!({"ok": true})).into_response())
}

pub async fn api_vocab_review_reject(
    State(state): State<Arc<AppState>>,
    Json(body): Json<serde_json::Value>,
) -> Result<Response, AppError> {
    let id = body["id"]
        .as_i64()
        .ok_or_else(|| err(anyhow::anyhow!("missing id")))?;
    {
        let db = state.db.lock().unwrap();
        db.reject_vocab(id).map_err(err)?;
    }
    {
        let mut review = state.review.lock().unwrap();
        review.vocab_queue.retain(|&qid| qid != id);
        review.vocab_precomputed.remove(&id);
    }
    state.vocab_precompute_notify.notify_one();
    state.background_work_notify.notify_one();
    Ok(Json(serde_json::json!({"ok": true})).into_response())
}

pub async fn api_vocab_review_pronunciation(
    State(state): State<Arc<AppState>>,
    Json(body): Json<PronunciationBody>,
) -> Result<Response, AppError> {
    // Update the vocab override and get the ID
    let vocab_id = {
        let db = state.db.lock().unwrap();
        let vocab = db
            .find_vocab_by_term(&body.word)
            .map_err(err)?
            .ok_or_else(|| err(anyhow::anyhow!("vocab term not found: {}", body.word)))?;
        db.update_vocab_override(vocab.id, Some(&body.spoken))
            .map_err(err)?;
        vocab.id
    };
    state.background_work_notify.notify_one();
    // Recompute for the SAME term (don't advance to next)
    serve_vocab_review(&state, vocab_id, false).await
}

pub async fn api_vocab_review_backend(
    State(state): State<Arc<AppState>>,
    Json(body): Json<BackendBody>,
) -> Result<Response, AppError> {
    let vocab_id = {
        let mut review = state.review.lock().unwrap();
        review.backend = body.backend;
        // Invalidate precomputed data (wrong backend now)
        review.vocab_precomputed.clear();
        review.vocab_current_id
    };
    match vocab_id {
        Some(id) => serve_vocab_review(&state, id, false).await,
        None => api_vocab_review_current(State(state)).await,
    }
}

fn load_wav_16k(wav_path: &str) -> anyhow::Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(wav_path)
        .map_err(|e| anyhow::anyhow!("WAV open {wav_path}: {e}"))?;
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
    let mono = if spec.channels > 1 {
        samples_f32
            .chunks(spec.channels as usize)
            .map(|ch| ch.iter().sum::<f32>() / ch.len() as f32)
            .collect()
    } else {
        samples_f32
    };
    tts::resample_to_16k(&mono, spec.sample_rate)
}
