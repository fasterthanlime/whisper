use std::ffi::{c_char, c_float, CStr, CString};
use std::sync::Arc;

use qwen3_asr::{AsrInference, StreamingOptions, StreamingState};

const TARGET_SR: usize = 16000;

// ── Opaque handles ──────────────────────────────────────────────────────

pub struct AsrEngine {
    inner: Arc<AsrInference>,
}

pub struct AsrSession {
    engine: Arc<AsrInference>,
    state: StreamingState,
    transcript: String,
    session_samples: usize,
    session_limit: usize,
    chunk_size_sec: f32,
    language: Option<String>,
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
    let mut streaming_opts = StreamingOptions::default().with_chunk_size_sec(opts.chunk_size_sec);
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

    Box::into_raw(Box::new(AsrSession {
        engine: arc,
        state,
        transcript: String::new(),
        session_samples: 0,
        session_limit,
        chunk_size_sec: opts.chunk_size_sec,
        language,
    }))
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

        s.session_samples += audio.len();

        match s.engine.feed_audio(&mut s.state, audio) {
            Ok(Some(result)) => {
                let full_text = format!("{}{}", s.transcript, result.text);

                // Session rotation: if we've accumulated enough audio, finalize
                // and start a new sub-session.
                if s.session_samples >= s.session_limit {
                    rotate_session(s);
                }

                Ok(Some(full_text))
            }
            Ok(None) => {
                // Still buffering, but check if we need to rotate anyway.
                if s.session_samples >= s.session_limit {
                    rotate_session(s);
                }
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
        let final_result = s
            .engine
            .finish_streaming(&mut s.state)
            .map_err(|e| format!("{e:#}"))?;
        if !final_result.text.is_empty() {
            s.transcript.push_str(&final_result.text);
        }
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
    // Finalize current sub-session.
    if let Ok(result) = s.engine.finish_streaming(&mut s.state) {
        if !result.text.is_empty() {
            s.transcript.push_str(&result.text);
            s.transcript.push(' ');
        }
    }

    // Start a new sub-session with context from the tail of the transcript.
    let context = tail_chars(&s.transcript, 200);
    let mut opts = StreamingOptions::default()
        .with_chunk_size_sec(s.chunk_size_sec)
        .with_initial_text(context);
    if let Some(lang) = &s.language {
        opts = opts.with_language(lang);
    }
    s.state = s.engine.init_streaming(opts);
    s.session_samples = 0;
}
