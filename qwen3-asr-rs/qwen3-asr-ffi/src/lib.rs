use std::ffi::{c_char, c_float, CStr, CString};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use qwen3_asr::{AsrInference, StreamingOptions, StreamingState};
use serde_json::{json, Value};

const TARGET_SR: usize = 16000;

// ── Opaque handles ──────────────────────────────────────────────────────

pub struct AsrEngine {
    inner: Arc<AsrInference>,
}

pub struct AsrSession {
    engine: Arc<AsrInference>,
    state: StreamingState,
    transcript: String,
    pending_prefix: String,
    detected_language: String,
    session_samples: usize,
    session_limit: usize,
    chunk_size_sec: f32,
    language: Option<String>,
    debug_events: Vec<Value>,
    commit_candidate: Option<String>,
    commit_candidate_hits: usize,
}

// ── Options struct (repr C) ─────────────────────────────────────────────

#[repr(C)]
pub struct AsrSessionOptions {
    /// Audio chunk size in seconds (e.g. 0.5).
    pub chunk_size_sec: c_float,
    /// Maximum session duration in seconds before auto-rotation (e.g. 10.0).
    pub session_duration_sec: c_float,
    /// Force a specific language (e.g. "english", "french"). NULL for auto-detect.
    pub language: *const c_char,
    /// Vocabulary hint / prompt text. NULL for none.
    pub prompt: *const c_char,
}

// ── Helpers ─────────────────────────────────────────────────────────────

fn to_c_string(s: String) -> *mut c_char {
    match CString::new(s) {
        Ok(cs) => cs.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

fn set_error(out_err: *mut *mut c_char, msg: String) {
    if !out_err.is_null() {
        unsafe {
            *out_err = to_c_string(msg);
        }
    }
}

/// Return the last N chars of `s`.
fn tail_chars(s: &str, n: usize) -> &str {
    let byte_start = s
        .char_indices()
        .rev()
        .nth(n.saturating_sub(1))
        .map(|(i, _)| i)
        .unwrap_or(0);
    &s[byte_start..]
}

fn append_segment(dst: &mut String, segment: &str) {
    let s = segment.trim();
    if s.is_empty() {
        return;
    }
    if !dst.is_empty() {
        dst.push(' ');
    }
    dst.push_str(s);
}

fn join_segments(a: &str, b: &str) -> String {
    let mut out = String::new();
    append_segment(&mut out, a);
    append_segment(&mut out, b);
    out
}

fn normalize_language_name(language: &str) -> String {
    language.trim().to_lowercase()
}

fn now_unix_ms() -> u128 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => duration.as_millis(),
        Err(_) => 0,
    }
}

fn push_debug_event(s: &mut AsrSession, stage: &str, payload: Value) {
    s.debug_events.push(json!({
        "ts_unix_ms": now_unix_ms(),
        "stage": stage,
        "payload": payload,
    }));
}

fn is_sentence_terminal(ch: char) -> bool {
    matches!(ch, '.' | '!' | '?' | '。' | '！' | '？')
}

fn is_sentence_closer(ch: char) -> bool {
    matches!(
        ch,
        '"' | '\'' | ')' | ']' | '}' | '”' | '’' | '»' | '）' | '】' | '』'
    )
}

const MIN_COMMIT_WORDS: usize = 4;
const MIN_COMMIT_CHARS: usize = 16;
const MIN_COMPLETE_SENTENCES_BEFORE_COMMIT: usize = 2;
const MIN_COMMIT_STABLE_UPDATES: usize = 3;

fn content_word_count(text: &str) -> usize {
    text.split_whitespace()
        .filter(|token| token.chars().any(|ch| ch.is_alphanumeric()))
        .count()
}

/// Return byte indices right after complete sentence boundaries.
fn sentence_boundaries(text: &str) -> Vec<usize> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let len = text.len();
    let mut out = Vec::new();

    for (i, (_, ch)) in chars.iter().enumerate() {
        if !is_sentence_terminal(*ch) {
            continue;
        }

        let mut j = i + 1;
        while j < chars.len() && is_sentence_closer(chars[j].1) {
            j += 1;
        }

        let is_boundary = if j >= chars.len() {
            true
        } else {
            chars[j].1.is_whitespace()
        };
        if !is_boundary {
            continue;
        }

        while j < chars.len() && chars[j].1.is_whitespace() {
            j += 1;
        }
        let end = if j < chars.len() { chars[j].0 } else { len };
        out.push(end);
    }

    out
}

/// Commit policy:
/// when we have at least two complete sentences, commit only the first one.
/// This keeps a one-sentence lookahead while avoiding long growing prefixes.
fn split_for_commit(text: &str) -> Option<(&str, &str)> {
    let boundaries = sentence_boundaries(text);
    if boundaries.len() < MIN_COMPLETE_SENTENCES_BEFORE_COMMIT {
        return None;
    }
    let cut = boundaries[0];
    let committed = text[..cut].trim();
    if committed.is_empty() {
        return None;
    }
    if content_word_count(committed) < MIN_COMMIT_WORDS || committed.chars().count() < MIN_COMMIT_CHARS {
        return None;
    }
    let remainder = text[cut..].trim_start();
    Some((committed, remainder))
}

fn ends_at_sentence_terminal(text: &str) -> bool {
    text.trim_end()
        .chars()
        .last()
        .map(is_sentence_terminal)
        .unwrap_or(false)
}

fn reset_commit_candidate(s: &mut AsrSession) {
    s.commit_candidate = None;
    s.commit_candidate_hits = 0;
}

fn should_early_commit(s: &mut AsrSession, text: &str) -> bool {
    // If the current prediction itself ends at sentence punctuation, treat it
    // as unstable: we do not commit on this update because the model often
    // revises that boundary on subsequent chunks.
    if ends_at_sentence_terminal(text) {
        reset_commit_candidate(s);
        return false;
    }

    let Some((committed, _)) = split_for_commit(text) else {
        reset_commit_candidate(s);
        return false;
    };

    if s.commit_candidate.as_deref() == Some(committed) {
        s.commit_candidate_hits = s.commit_candidate_hits.saturating_add(1);
    } else {
        s.commit_candidate = Some(committed.to_string());
        s.commit_candidate_hits = 1;
    }

    s.commit_candidate_hits >= MIN_COMMIT_STABLE_UPDATES
}

fn build_context_text(s: &AsrSession) -> String {
    let combined = join_segments(&s.transcript, &s.pending_prefix);
    tail_chars(&combined, 200).to_string()
}

fn restart_subsession(s: &mut AsrSession) {
    let context = build_context_text(s);
    let mut opts = StreamingOptions::default().with_chunk_size_sec(s.chunk_size_sec);
    if !context.is_empty() {
        opts = opts.with_initial_text(context);
    }
    if let Some(lang) = &s.language {
        opts = opts.with_language(lang);
    }
    s.state = s.engine.init_streaming(opts);
    s.session_samples = 0;
    reset_commit_candidate(s);
}

// ── Engine API ──────────────────────────────────────────────────────────

/// Load a model from `model_dir`. Returns NULL on error (check `*out_err`).
///
/// The caller must free the engine with `asr_engine_free`.
#[no_mangle]
pub extern "C" fn asr_engine_load(
    model_dir: *const c_char,
    out_err: *mut *mut c_char,
) -> *mut AsrEngine {
    let result = std::panic::catch_unwind(|| {
        let dir = unsafe { CStr::from_ptr(model_dir) }
            .to_str()
            .map_err(|e| format!("invalid model_dir: {e}"))?;
        let device = qwen3_asr::best_device();
        let inference =
            AsrInference::load(std::path::Path::new(dir), device).map_err(|e| format!("{e:#}"))?;
        Ok::<_, String>(Box::into_raw(Box::new(AsrEngine {
            inner: Arc::new(inference),
        })))
    });

    match result {
        Ok(Ok(ptr)) => ptr,
        Ok(Err(msg)) => {
            set_error(out_err, msg);
            std::ptr::null_mut()
        }
        Err(_) => {
            set_error(out_err, "panic during engine load".into());
            std::ptr::null_mut()
        }
    }
}

/// Download a model from HuggingFace (if not cached) and load it.
///
/// `model_id`: e.g. "Qwen/Qwen3-ASR-0.6B"
/// `cache_dir`: local directory for caching model files.
///
/// Returns NULL on error (check `*out_err`). Free with `asr_engine_free`.
/// Requires the `hub` feature (enabled by default).
#[cfg(feature = "hub")]
#[no_mangle]
pub extern "C" fn asr_engine_from_pretrained(
    model_id: *const c_char,
    cache_dir: *const c_char,
    out_err: *mut *mut c_char,
) -> *mut AsrEngine {
    let result = std::panic::catch_unwind(|| {
        let id = unsafe { CStr::from_ptr(model_id) }
            .to_str()
            .map_err(|e| format!("invalid model_id: {e}"))?;
        let dir = unsafe { CStr::from_ptr(cache_dir) }
            .to_str()
            .map_err(|e| format!("invalid cache_dir: {e}"))?;
        eprintln!("[qwen3-asr-ffi] from_pretrained: model_id={id:?} cache_dir={dir:?}");
        let device = qwen3_asr::best_device();
        eprintln!("[qwen3-asr-ffi] device: {device:?}");
        let inference = AsrInference::from_pretrained(id, std::path::Path::new(dir), device)
            .map_err(|e| {
                let msg = format!("{e:#}");
                eprintln!("[qwen3-asr-ffi] from_pretrained FAILED: {msg}");
                msg
            })?;
        eprintln!("[qwen3-asr-ffi] model loaded successfully");
        Ok::<_, String>(Box::into_raw(Box::new(AsrEngine {
            inner: Arc::new(inference),
        })))
    });

    match result {
        Ok(Ok(ptr)) => ptr,
        Ok(Err(msg)) => {
            set_error(out_err, msg);
            std::ptr::null_mut()
        }
        Err(_) => {
            set_error(out_err, "panic during engine load".into());
            std::ptr::null_mut()
        }
    }
}

/// Download a GGUF-quantized model from HuggingFace and load it.
///
/// `base_repo_id`: full-precision repo for config + tokenizer (e.g. "Qwen/Qwen3-ASR-1.7B")
/// `gguf_repo_id`: repo hosting GGUF files (e.g. "Alkd/qwen3-asr-gguf")
/// `gguf_filename`: specific GGUF file (e.g. "qwen3_asr_1.7b_q4_k.gguf")
/// `cache_dir`: local directory for caching model files.
///
/// Returns NULL on error (check `*out_err`). Free with `asr_engine_free`.
#[cfg(feature = "hub")]
#[no_mangle]
pub extern "C" fn asr_engine_from_gguf(
    base_repo_id: *const c_char,
    gguf_repo_id: *const c_char,
    gguf_filename: *const c_char,
    cache_dir: *const c_char,
    out_err: *mut *mut c_char,
) -> *mut AsrEngine {
    let result = std::panic::catch_unwind(|| {
        let base_id = unsafe { CStr::from_ptr(base_repo_id) }
            .to_str()
            .map_err(|e| format!("invalid base_repo_id: {e}"))?;
        let gguf_id = unsafe { CStr::from_ptr(gguf_repo_id) }
            .to_str()
            .map_err(|e| format!("invalid gguf_repo_id: {e}"))?;
        let filename = unsafe { CStr::from_ptr(gguf_filename) }
            .to_str()
            .map_err(|e| format!("invalid gguf_filename: {e}"))?;
        let dir = unsafe { CStr::from_ptr(cache_dir) }
            .to_str()
            .map_err(|e| format!("invalid cache_dir: {e}"))?;
        eprintln!(
            "[qwen3-asr-ffi] from_gguf: base={base_id:?} gguf={gguf_id:?} file={filename:?} cache={dir:?}"
        );
        let device = qwen3_asr::best_device();
        eprintln!("[qwen3-asr-ffi] device: {device:?}");
        let inference = AsrInference::from_pretrained_gguf(
            base_id,
            gguf_id,
            filename,
            std::path::Path::new(dir),
            device,
        )
        .map_err(|e| {
            let msg = format!("{e:#}");
            eprintln!("[qwen3-asr-ffi] from_gguf FAILED: {msg}");
            msg
        })?;
        eprintln!("[qwen3-asr-ffi] GGUF model loaded successfully");
        Ok::<_, String>(Box::into_raw(Box::new(AsrEngine {
            inner: Arc::new(inference),
        })))
    });

    match result {
        Ok(Ok(ptr)) => ptr,
        Ok(Err(msg)) => {
            set_error(out_err, msg);
            std::ptr::null_mut()
        }
        Err(_) => {
            set_error(out_err, "panic during GGUF engine load".into());
            std::ptr::null_mut()
        }
    }
}

/// Free an engine. Safe to call with NULL.
#[no_mangle]
pub extern "C" fn asr_engine_free(engine: *mut AsrEngine) {
    if !engine.is_null() {
        unsafe {
            drop(Box::from_raw(engine));
        }
    }
}

/// Single-shot transcription from 16 kHz mono f32 samples.
///
/// Returns a newly-allocated C string with the transcript (free with `asr_string_free`).
/// Returns NULL on error (check `*out_err`).
#[no_mangle]
pub extern "C" fn asr_engine_transcribe_samples(
    engine: *const AsrEngine,
    samples: *const c_float,
    num_samples: usize,
    out_err: *mut *mut c_char,
) -> *mut c_char {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let engine_ref = unsafe { &*engine };
        let audio = unsafe { std::slice::from_raw_parts(samples, num_samples) };
        let opts = qwen3_asr::TranscribeOptions::default();
        let result = engine_ref
            .inner
            .transcribe_samples(audio, opts)
            .map_err(|e| format!("{e:#}"))?;
        Ok::<_, String>(result.text)
    }));

    match result {
        Ok(Ok(text)) => to_c_string(text),
        Ok(Err(msg)) => {
            set_error(out_err, msg);
            std::ptr::null_mut()
        }
        Err(_) => {
            set_error(out_err, "panic during transcribe_samples".into());
            std::ptr::null_mut()
        }
    }
}

// ── Session API ─────────────────────────────────────────────────────────

/// Create a streaming session. The caller must free it with `asr_session_free`.
#[no_mangle]
pub extern "C" fn asr_session_create(
    engine: *const AsrEngine,
    opts: AsrSessionOptions,
) -> *mut AsrSession {
    let engine_ref = unsafe { &*engine };
    let arc = engine_ref.inner.clone();
    let mut streaming_opts = StreamingOptions::default()
        .with_chunk_size_sec(opts.chunk_size_sec)
        // Keep a larger mutable suffix so punctuation/endings can be revised
        // instead of being frozen too aggressively between chunks.
        .with_unfixed_token_num(12);
    if !opts.language.is_null() {
        if let Ok(lang) = unsafe { CStr::from_ptr(opts.language) }.to_str() {
            if !lang.is_empty() {
                streaming_opts = streaming_opts.with_language(lang);
            }
        }
    }
    if !opts.prompt.is_null() {
        if let Ok(prompt) = unsafe { CStr::from_ptr(opts.prompt) }.to_str() {
            if !prompt.is_empty() {
                streaming_opts = streaming_opts.with_initial_text(prompt);
            }
        }
    }
    let state = arc.init_streaming(streaming_opts);
    let session_limit = (opts.session_duration_sec as usize) * TARGET_SR;

    let language = if !opts.language.is_null() {
        unsafe { CStr::from_ptr(opts.language) }
            .to_str()
            .ok()
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
    } else {
        None
    };

    let mut session = AsrSession {
        engine: arc,
        state,
        transcript: String::new(),
        pending_prefix: String::new(),
        detected_language: String::new(),
        session_samples: 0,
        session_limit,
        chunk_size_sec: opts.chunk_size_sec,
        language,
        debug_events: Vec::new(),
        commit_candidate: None,
        commit_candidate_hits: 0,
    };
    let language_for_event = session.language.clone();
    push_debug_event(
        &mut session,
        "session_create",
        json!({
            "chunk_size_sec": opts.chunk_size_sec,
            "session_duration_sec": opts.session_duration_sec,
            "session_limit_samples": session_limit,
            "language": language_for_event,
            "prompt_present": !opts.prompt.is_null(),
        }),
    );

    Box::into_raw(Box::new(session))
}

/// Feed 16 kHz mono f32 samples into the session.
///
/// Returns a newly-allocated C string with the current full transcript when a
/// chunk boundary is crossed (caller must free with `asr_string_free`).
/// Returns NULL if still buffering or on error (check `*out_err`).
#[no_mangle]
pub extern "C" fn asr_session_feed(
    session: *mut AsrSession,
    samples: *const c_float,
    num_samples: usize,
    out_err: *mut *mut c_char,
) -> *mut c_char {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let s = unsafe { &mut *session };
        let audio = unsafe { std::slice::from_raw_parts(samples, num_samples) };

        let session_samples_before = s.session_samples;
        s.session_samples += audio.len();
        push_debug_event(
            s,
            "feed_call",
            json!({
                "mode": "stream",
                "input_samples": audio.len(),
                "session_samples_before": session_samples_before,
                "session_samples_after": s.session_samples,
            }),
        );

        match s.engine.feed_audio_with_debug(&mut s.state, audio) {
            Ok((Some(result), debug_info)) => {
                push_debug_event(
                    s,
                    "feed_debug",
                    json!({
                        "mode": debug_info.mode,
                        "input_samples": debug_info.input_samples,
                        "speech_detected_before": debug_info.speech_detected_before,
                        "speech_detected_after": debug_info.speech_detected_after,
                        "speech_start_offset": debug_info.speech_start_offset,
                        "buffer_len_before": debug_info.buffer_len_before,
                        "buffer_len_after": debug_info.buffer_len_after,
                        "audio_accum_len_before": debug_info.audio_accum_len_before,
                        "audio_accum_len_after": debug_info.audio_accum_len_after,
                        "chunk_id_before": debug_info.chunk_id_before,
                        "chunk_id_after": debug_info.chunk_id_after,
                        "drop_reason": debug_info.drop_reason,
                        "drained_chunk": debug_info.drained_chunk,
                        "ran_inference": debug_info.ran_inference,
                    }),
                );

                s.detected_language = normalize_language_name(&result.language);
                let current_text = join_segments(&s.pending_prefix, &result.text);
                push_debug_event(
                    s,
                    "feed_result",
                    json!({
                        "mode": "stream",
                        "result_text": result.text,
                        "result_language": s.detected_language,
                        "current_text": current_text,
                    }),
                );

                // Commit the oldest sentence once we have >=3 complete sentences
                // buffered (two-sentence lookahead), then restart from that
                // text boundary.
                //
                // We first flush/finalize the current sub-session to avoid losing
                // any buffered samples that haven't crossed a chunk boundary yet.
                let commit_ready = should_early_commit(s, &current_text);
                push_debug_event(
                    s,
                    "feed_commit_gate",
                    json!({
                        "mode": "stream",
                        "commit_ready": commit_ready,
                        "candidate": s.commit_candidate.clone(),
                        "candidate_hits": s.commit_candidate_hits,
                        "required_hits": MIN_COMMIT_STABLE_UPDATES,
                        "ends_with_terminal": ends_at_sentence_terminal(&current_text),
                    }),
                );
                if commit_ready {
                    let (flushed, flush_debug) = s
                        .engine
                        .finish_streaming_with_debug(&mut s.state)
                        .map_err(|e| format!("{e:#}"))?;
                    push_debug_event(
                        s,
                        "feed_commit_flush_debug",
                        json!({
                            "buffer_len_before": flush_debug.buffer_len_before,
                            "buffer_len_after": flush_debug.buffer_len_after,
                            "audio_accum_len_before": flush_debug.audio_accum_len_before,
                            "audio_accum_len_after": flush_debug.audio_accum_len_after,
                            "chunk_id_before": flush_debug.chunk_id_before,
                            "chunk_id_after": flush_debug.chunk_id_after,
                            "flushed_remaining": flush_debug.flushed_remaining,
                            "returned_empty": flush_debug.returned_empty,
                        }),
                    );
                    s.detected_language = normalize_language_name(&flushed.language);
                    let full_tail = join_segments(&s.pending_prefix, &flushed.text);
                    if let Some((committed, remainder)) = split_for_commit(&full_tail) {
                        append_segment(&mut s.transcript, committed);
                        s.pending_prefix = remainder.to_string();
                        push_debug_event(
                            s,
                            "feed_commit_split",
                            json!({
                                "committed": committed,
                                "remainder": s.pending_prefix,
                                "transcript": s.transcript,
                            }),
                        );
                    } else {
                        s.pending_prefix = full_tail;
                    }
                    restart_subsession(s);
                    push_debug_event(
                        s,
                        "feed_commit_restart",
                        json!({
                            "session_samples": s.session_samples,
                            "transcript": s.transcript,
                            "pending_prefix": s.pending_prefix,
                        }),
                    );
                    let full_text = join_segments(&s.transcript, &s.pending_prefix);
                    push_debug_event(
                        s,
                        "feed_return",
                        json!({
                            "mode": "stream",
                            "full_text": full_text,
                        }),
                    );
                    return Ok(Some(full_text));
                }

                let full_text = join_segments(&s.transcript, &current_text);

                // Session rotation: if we've accumulated enough audio, finalize
                // and start a new sub-session.
                if s.session_samples >= s.session_limit {
                    push_debug_event(
                        s,
                        "feed_rotate",
                        json!({
                            "mode": "stream",
                            "session_samples": s.session_samples,
                            "session_limit": s.session_limit,
                        }),
                    );
                    rotate_session(s);
                }

                push_debug_event(
                    s,
                    "feed_return",
                    json!({
                        "mode": "stream",
                        "full_text": full_text,
                    }),
                );
                Ok(Some(full_text))
            }
            Ok((None, debug_info)) => {
                push_debug_event(
                    s,
                    "feed_debug",
                    json!({
                        "mode": debug_info.mode,
                        "input_samples": debug_info.input_samples,
                        "speech_detected_before": debug_info.speech_detected_before,
                        "speech_detected_after": debug_info.speech_detected_after,
                        "speech_start_offset": debug_info.speech_start_offset,
                        "buffer_len_before": debug_info.buffer_len_before,
                        "buffer_len_after": debug_info.buffer_len_after,
                        "audio_accum_len_before": debug_info.audio_accum_len_before,
                        "audio_accum_len_after": debug_info.audio_accum_len_after,
                        "chunk_id_before": debug_info.chunk_id_before,
                        "chunk_id_after": debug_info.chunk_id_after,
                        "drop_reason": debug_info.drop_reason,
                        "drained_chunk": debug_info.drained_chunk,
                        "ran_inference": debug_info.ran_inference,
                    }),
                );
                // Still buffering, but check if we need to rotate anyway.
                if s.session_samples >= s.session_limit {
                    push_debug_event(
                        s,
                        "feed_rotate",
                        json!({
                            "mode": "stream",
                            "session_samples": s.session_samples,
                            "session_limit": s.session_limit,
                        }),
                    );
                    rotate_session(s);
                }
                push_debug_event(
                    s,
                    "feed_return_none",
                    json!({
                        "mode": "stream",
                    }),
                );
                Ok(None)
            }
            Err(e) => Err(format!("{e:#}")),
        }
    }));

    match result {
        Ok(Ok(Some(text))) => to_c_string(text),
        Ok(Ok(None)) => std::ptr::null_mut(),
        Ok(Err(msg)) => {
            set_error(out_err, msg);
            std::ptr::null_mut()
        }
        Err(_) => {
            set_error(out_err, "panic during feed".into());
            std::ptr::null_mut()
        }
    }
}

/// Feed finalization-time 16 kHz mono f32 samples into the session.
///
/// Same contract as `asr_session_feed`, but this call is intended for the
/// stop/finalize path and will not drop low-energy post-speech chunks.
#[no_mangle]
pub extern "C" fn asr_session_feed_finalizing(
    session: *mut AsrSession,
    samples: *const c_float,
    num_samples: usize,
    out_err: *mut *mut c_char,
) -> *mut c_char {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let s = unsafe { &mut *session };
        let audio = unsafe { std::slice::from_raw_parts(samples, num_samples) };

        let session_samples_before = s.session_samples;
        s.session_samples += audio.len();
        push_debug_event(
            s,
            "feed_call",
            json!({
                "mode": "finalize",
                "input_samples": audio.len(),
                "session_samples_before": session_samples_before,
                "session_samples_after": s.session_samples,
            }),
        );

        match s.engine.feed_audio_finalizing_with_debug(&mut s.state, audio) {
            Ok((Some(result), debug_info)) => {
                push_debug_event(
                    s,
                    "feed_debug",
                    json!({
                        "mode": debug_info.mode,
                        "input_samples": debug_info.input_samples,
                        "speech_detected_before": debug_info.speech_detected_before,
                        "speech_detected_after": debug_info.speech_detected_after,
                        "speech_start_offset": debug_info.speech_start_offset,
                        "buffer_len_before": debug_info.buffer_len_before,
                        "buffer_len_after": debug_info.buffer_len_after,
                        "audio_accum_len_before": debug_info.audio_accum_len_before,
                        "audio_accum_len_after": debug_info.audio_accum_len_after,
                        "chunk_id_before": debug_info.chunk_id_before,
                        "chunk_id_after": debug_info.chunk_id_after,
                        "drop_reason": debug_info.drop_reason,
                        "drained_chunk": debug_info.drained_chunk,
                        "ran_inference": debug_info.ran_inference,
                    }),
                );
                s.detected_language = normalize_language_name(&result.language);
                let current_text = join_segments(&s.pending_prefix, &result.text);
                push_debug_event(
                    s,
                    "feed_result",
                    json!({
                        "mode": "finalize",
                        "result_text": result.text,
                        "result_language": s.detected_language,
                        "current_text": current_text,
                    }),
                );

                let commit_ready = should_early_commit(s, &current_text);
                push_debug_event(
                    s,
                    "feed_commit_gate",
                    json!({
                        "mode": "finalize",
                        "commit_ready": commit_ready,
                        "candidate": s.commit_candidate.clone(),
                        "candidate_hits": s.commit_candidate_hits,
                        "required_hits": MIN_COMMIT_STABLE_UPDATES,
                        "ends_with_terminal": ends_at_sentence_terminal(&current_text),
                    }),
                );
                if commit_ready {
                    let (flushed, flush_debug) = s
                        .engine
                        .finish_streaming_with_debug(&mut s.state)
                        .map_err(|e| format!("{e:#}"))?;
                    push_debug_event(
                        s,
                        "feed_commit_flush_debug",
                        json!({
                            "buffer_len_before": flush_debug.buffer_len_before,
                            "buffer_len_after": flush_debug.buffer_len_after,
                            "audio_accum_len_before": flush_debug.audio_accum_len_before,
                            "audio_accum_len_after": flush_debug.audio_accum_len_after,
                            "chunk_id_before": flush_debug.chunk_id_before,
                            "chunk_id_after": flush_debug.chunk_id_after,
                            "flushed_remaining": flush_debug.flushed_remaining,
                            "returned_empty": flush_debug.returned_empty,
                        }),
                    );
                    s.detected_language = normalize_language_name(&flushed.language);
                    let full_tail = join_segments(&s.pending_prefix, &flushed.text);
                    if let Some((committed, remainder)) = split_for_commit(&full_tail) {
                        append_segment(&mut s.transcript, committed);
                        s.pending_prefix = remainder.to_string();
                        push_debug_event(
                            s,
                            "feed_commit_split",
                            json!({
                                "committed": committed,
                                "remainder": s.pending_prefix,
                                "transcript": s.transcript,
                            }),
                        );
                    } else {
                        s.pending_prefix = full_tail;
                    }
                    restart_subsession(s);
                    let full_text = join_segments(&s.transcript, &s.pending_prefix);
                    push_debug_event(
                        s,
                        "feed_commit_restart",
                        json!({
                            "session_samples": s.session_samples,
                            "transcript": s.transcript,
                            "pending_prefix": s.pending_prefix,
                        }),
                    );
                    push_debug_event(
                        s,
                        "feed_return",
                        json!({
                            "mode": "finalize",
                            "full_text": full_text,
                        }),
                    );
                    return Ok(Some(full_text));
                }

                let full_text = join_segments(&s.transcript, &current_text);

                if s.session_samples >= s.session_limit {
                    push_debug_event(
                        s,
                        "feed_rotate",
                        json!({
                            "mode": "finalize",
                            "session_samples": s.session_samples,
                            "session_limit": s.session_limit,
                        }),
                    );
                    rotate_session(s);
                }

                push_debug_event(
                    s,
                    "feed_return",
                    json!({
                        "mode": "finalize",
                        "full_text": full_text,
                    }),
                );
                Ok(Some(full_text))
            }
            Ok((None, debug_info)) => {
                push_debug_event(
                    s,
                    "feed_debug",
                    json!({
                        "mode": debug_info.mode,
                        "input_samples": debug_info.input_samples,
                        "speech_detected_before": debug_info.speech_detected_before,
                        "speech_detected_after": debug_info.speech_detected_after,
                        "speech_start_offset": debug_info.speech_start_offset,
                        "buffer_len_before": debug_info.buffer_len_before,
                        "buffer_len_after": debug_info.buffer_len_after,
                        "audio_accum_len_before": debug_info.audio_accum_len_before,
                        "audio_accum_len_after": debug_info.audio_accum_len_after,
                        "chunk_id_before": debug_info.chunk_id_before,
                        "chunk_id_after": debug_info.chunk_id_after,
                        "drop_reason": debug_info.drop_reason,
                        "drained_chunk": debug_info.drained_chunk,
                        "ran_inference": debug_info.ran_inference,
                    }),
                );
                if s.session_samples >= s.session_limit {
                    push_debug_event(
                        s,
                        "feed_rotate",
                        json!({
                            "mode": "finalize",
                            "session_samples": s.session_samples,
                            "session_limit": s.session_limit,
                        }),
                    );
                    rotate_session(s);
                }
                push_debug_event(
                    s,
                    "feed_return_none",
                    json!({
                        "mode": "finalize",
                    }),
                );
                Ok(None)
            }
            Err(e) => Err(format!("{e:#}")),
        }
    }));

    match result {
        Ok(Ok(Some(text))) => to_c_string(text),
        Ok(Ok(None)) => std::ptr::null_mut(),
        Ok(Err(msg)) => {
            set_error(out_err, msg);
            std::ptr::null_mut()
        }
        Err(_) => {
            set_error(out_err, "panic during finalizing feed".into());
            std::ptr::null_mut()
        }
    }
}

/// Return the length (in UTF-16 code units) of the committed transcript prefix.
///
/// This excludes the current uncommitted/pending tail and is intended for UI
/// styling (e.g. committed vs pending fade levels).
#[no_mangle]
pub extern "C" fn asr_session_committed_utf16_len(session: *const AsrSession) -> usize {
    if session.is_null() {
        return 0;
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let s = unsafe { &*session };
        s.transcript.encode_utf16().count()
    }));

    result.unwrap_or(0)
}

/// Return the most recently detected language for the session.
///
/// Returns NULL if unavailable.
#[no_mangle]
pub extern "C" fn asr_session_last_language(session: *const AsrSession) -> *mut c_char {
    if session.is_null() {
        return std::ptr::null_mut();
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let s = unsafe { &*session };
        if s.detected_language.is_empty() {
            return None;
        }
        Some(s.detected_language.clone())
    }));

    match result {
        Ok(Some(language)) => to_c_string(language),
        _ => std::ptr::null_mut(),
    }
}

/// Force or clear language for an active streaming session.
///
/// `language=NULL` (or empty string) restores auto-detection.
/// Returns true on success, false on error (check `*out_err`).
#[no_mangle]
pub extern "C" fn asr_session_set_language(
    session: *mut AsrSession,
    language: *const c_char,
    out_err: *mut *mut c_char,
) -> bool {
    if session.is_null() {
        set_error(out_err, "session is null".into());
        return false;
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let s = unsafe { &mut *session };
        let requested = if language.is_null() {
            None
        } else {
            let raw = unsafe { CStr::from_ptr(language) }
                .to_str()
                .map_err(|e| format!("invalid language: {e}"))?;
            let normalized = normalize_language_name(raw);
            if normalized.is_empty() { None } else { Some(normalized) }
        };

        s.language = requested.clone();
        s.engine
            .set_streaming_language(&mut s.state, requested.as_deref());
        push_debug_event(
            s,
            "set_language",
            json!({
                "language": requested,
            }),
        );
        Ok::<(), String>(())
    }));

    match result {
        Ok(Ok(())) => true,
        Ok(Err(msg)) => {
            set_error(out_err, msg);
            false
        }
        Err(_) => {
            set_error(out_err, "panic during set_language".into());
            false
        }
    }
}

/// Finalize the session and return the complete transcript.
///
/// Returns a newly-allocated C string (caller must free with `asr_string_free`).
/// Returns NULL on error (check `*out_err`).
#[no_mangle]
pub extern "C" fn asr_session_finish(
    session: *mut AsrSession,
    out_err: *mut *mut c_char,
) -> *mut c_char {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let s = unsafe { &mut *session };
        push_debug_event(
            s,
            "finish_call",
            json!({
                "session_samples": s.session_samples,
                "transcript_chars_before": s.transcript.chars().count(),
                "pending_prefix_chars_before": s.pending_prefix.chars().count(),
            }),
        );
        let (final_result, finish_debug) = s
            .engine
            .finish_streaming_with_debug(&mut s.state)
            .map_err(|e| format!("{e:#}"))?;
        push_debug_event(
            s,
            "finish_debug",
            json!({
                "buffer_len_before": finish_debug.buffer_len_before,
                "buffer_len_after": finish_debug.buffer_len_after,
                "audio_accum_len_before": finish_debug.audio_accum_len_before,
                "audio_accum_len_after": finish_debug.audio_accum_len_after,
                "chunk_id_before": finish_debug.chunk_id_before,
                "chunk_id_after": finish_debug.chunk_id_after,
                "flushed_remaining": finish_debug.flushed_remaining,
                "returned_empty": finish_debug.returned_empty,
            }),
        );
        s.detected_language = normalize_language_name(&final_result.language);
        let full_tail = join_segments(&s.pending_prefix, &final_result.text);
        append_segment(&mut s.transcript, &full_tail);
        s.pending_prefix.clear();
        push_debug_event(
            s,
            "finish_result",
            json!({
                "language": s.detected_language,
                "final_result_text": final_result.text,
                "full_tail": full_tail,
                "transcript_chars_after": s.transcript.chars().count(),
            }),
        );
        Ok::<_, String>(s.transcript.clone())
    }));

    match result {
        Ok(Ok(text)) => to_c_string(text),
        Ok(Err(msg)) => {
            set_error(out_err, msg);
            std::ptr::null_mut()
        }
        Err(_) => {
            set_error(out_err, "panic during finish".into());
            std::ptr::null_mut()
        }
    }
}

/// Drain and return queued session debug events as a JSON array string.
///
/// Returns "[]" for null sessions, panics, or serialization failures.
#[no_mangle]
pub extern "C" fn asr_session_take_debug_events_json(session: *mut AsrSession) -> *mut c_char {
    if session.is_null() {
        return to_c_string("[]".into());
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let s = unsafe { &mut *session };
        let drained = std::mem::take(&mut s.debug_events);
        serde_json::to_string(&drained).unwrap_or_else(|_| "[]".to_string())
    }));

    match result {
        Ok(json_text) => to_c_string(json_text),
        Err(_) => to_c_string("[]".into()),
    }
}

/// Free a session. Safe to call with NULL.
#[no_mangle]
pub extern "C" fn asr_session_free(session: *mut AsrSession) {
    if !session.is_null() {
        unsafe {
            drop(Box::from_raw(session));
        }
    }
}

/// Free a string returned by any `asr_*` function. Safe to call with NULL.
#[no_mangle]
pub extern "C" fn asr_string_free(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            drop(CString::from_raw(s));
        }
    }
}

// ── Internal ────────────────────────────────────────────────────────────

fn rotate_session(s: &mut AsrSession) {
    push_debug_event(
        s,
        "rotate_begin",
        json!({
            "session_samples": s.session_samples,
            "transcript_chars_before": s.transcript.chars().count(),
            "pending_prefix_chars_before": s.pending_prefix.chars().count(),
        }),
    );
    // Finalize current sub-session.
    if let Ok((result, finish_debug)) = s.engine.finish_streaming_with_debug(&mut s.state) {
        push_debug_event(
            s,
            "rotate_finish_debug",
            json!({
                "buffer_len_before": finish_debug.buffer_len_before,
                "buffer_len_after": finish_debug.buffer_len_after,
                "audio_accum_len_before": finish_debug.audio_accum_len_before,
                "audio_accum_len_after": finish_debug.audio_accum_len_after,
                "chunk_id_before": finish_debug.chunk_id_before,
                "chunk_id_after": finish_debug.chunk_id_after,
                "flushed_remaining": finish_debug.flushed_remaining,
                "returned_empty": finish_debug.returned_empty,
            }),
        );
        s.detected_language = normalize_language_name(&result.language);
        let full_tail = join_segments(&s.pending_prefix, &result.text);
        append_segment(&mut s.transcript, &full_tail);
        s.pending_prefix.clear();
        push_debug_event(
            s,
            "rotate_finish_result",
            json!({
                "language": s.detected_language,
                "result_text": result.text,
                "full_tail": full_tail,
                "transcript_chars_after": s.transcript.chars().count(),
            }),
        );
    }
    restart_subsession(s);
    push_debug_event(
        s,
        "rotate_end",
        json!({
            "session_samples": s.session_samples,
            "transcript_chars": s.transcript.chars().count(),
            "pending_prefix_chars": s.pending_prefix.chars().count(),
        }),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_for_commit_needs_two_complete_sentences() {
        assert!(split_for_commit("First sentence has enough words now.").is_none());
        let (commit, rest) = split_for_commit(
            "First sentence has enough words now. Second sentence is complete too."
        )
        .expect("should split with two complete sentences");
        assert_eq!(commit, "First sentence has enough words now.");
        assert_eq!(rest, "Second sentence is complete too.");
    }

    #[test]
    fn split_for_commit_commits_oldest_sentence_only() {
        let (commit, rest) =
            split_for_commit(
                "First sentence has enough words now. Second sentence continues here. Third sentence is complete. Fourth"
            )
                .expect("should split");
        assert_eq!(commit, "First sentence has enough words now.");
        assert_eq!(rest, "Second sentence continues here. Third sentence is complete. Fourth");
    }

    #[test]
    fn split_for_commit_handles_unicode_punctuation() {
        let (commit, rest) =
            split_for_commit("Bonjour ceci est une phrase longue ! Encore une phrase ? Oui c'est complet. Suite")
                .expect("should split");
        assert_eq!(commit, "Bonjour ceci est une phrase longue !");
        assert_eq!(rest, "Encore une phrase ? Oui c'est complet. Suite");
    }

    #[test]
    fn split_for_commit_rejects_short_first_sentence() {
        assert!(
            split_for_commit("Yep. This is a full second sentence indeed. Third sentence is complete.")
                .is_none()
        );
    }

    #[test]
    fn join_segments_trims_and_separates() {
        assert_eq!(join_segments("", "hello"), "hello");
        assert_eq!(join_segments("hello", "world"), "hello world");
        assert_eq!(join_segments(" hello  ", "  world "), "hello world");
    }
}
