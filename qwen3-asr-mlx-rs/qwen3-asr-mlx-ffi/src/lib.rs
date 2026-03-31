//! FFI layer for qwen3-asr-mlx — same C API as the candle FFI.

use std::ffi::{c_char, c_float, c_uint, CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, Once};
use std::io::Write;

use qwen3_asr_mlx::config::AsrConfig;
use qwen3_asr_mlx::forced_aligner::ForcedAligner;
use qwen3_asr_mlx::model::Qwen3ASRModel;
use qwen3_asr_mlx::streaming::{self, StreamingMode, StreamingOptions, StreamingState};

static INIT_LOGGER: Once = Once::new();

fn init_logger() {
    INIT_LOGGER.call_once(|| {
        env_logger::init();
    });
}

// ── Engine ──────────────────────────────────────────────────────────────

struct AsrEngineInner {
    model: Qwen3ASRModel,
    tokenizer: tokenizers::Tokenizer,
    model_dir: PathBuf,
}

// SAFETY: MLX arrays use heap-allocated Metal buffers accessed via Arc.
// Concurrent access is prevented by the Mutex in AsrSession.
unsafe impl Send for AsrEngineInner {}

pub struct AsrEngine {
    inner: Arc<Mutex<AsrEngineInner>>,
}

// ── Session ─────────────────────────────────────────────────────────────

pub struct AsrSession {
    engine: Arc<Mutex<AsrEngineInner>>,
    state: StreamingState,
    last_text: String,
}

// SAFETY: Same reasoning as AsrEngineInner — Metal buffers are heap-allocated,
// concurrent access prevented by external synchronization (Swift calls are sequential).
unsafe impl Send for AsrSession {}

// ── Options (must match C header layout) ────────────────────────────────

#[repr(C)]
pub struct AsrSessionOptions {
    pub chunk_size_sec: c_float,
    pub session_duration_sec: c_float,
    pub language: *const c_char,
    pub prompt: *const c_char,
    pub unfixed_chunk_num: c_uint,
    pub unfixed_token_num: c_uint,
    pub max_new_tokens_streaming: c_uint,
    pub max_new_tokens_final: c_uint,
}

// ── Helpers ─────────────────────────────────────────────────────────────

fn set_err(out_err: *mut *mut c_char, msg: &str) {
    if !out_err.is_null() {
        if let Ok(cs) = CString::new(msg) {
            unsafe { *out_err = cs.into_raw() };
        }
    }
}

fn to_c_string(s: &str) -> *mut c_char {
    CString::new(s).map(|cs| cs.into_raw()).unwrap_or(std::ptr::null_mut())
}

fn ffi_log(msg: &str) {
    let path = "/tmp/bee-ffi.log";
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(f, "[{:.3}] {}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs_f64(), msg);
    }
}

fn find_tokenizer(model_dir: &Path) -> Option<tokenizers::Tokenizer> {
    let paths = [
        model_dir.join("tokenizer.json"),
        dirs::home_dir()?.join("Library/Caches/qwen3-asr/Qwen--Qwen3-ASR-1.7B/tokenizer.json"),
        dirs::home_dir()?.join("Library/Caches/qwen3-asr/Qwen--Qwen3-ASR-0.6B/tokenizer.json"),
    ];
    for p in &paths {
        if p.exists() {
            if let Ok(t) = tokenizers::Tokenizer::from_file(p) {
                return Some(t);
            }
        }
    }
    None
}

fn find_vad_dir() -> Option<PathBuf> {
    let dir = dirs::home_dir()?.join("Library/Caches/qwen3-asr/aitytech--Silero-VAD-v5-MLX");
    if dir.exists() { Some(dir) } else { None }
}

fn find_aligner_dir() -> Option<PathBuf> {
    let base = dirs::home_dir()?.join("Library/Caches/qwen3-asr");
    // Prefer 4-bit quantized aligner
    let candidates = [
        "mlx-community--Qwen3-ForcedAligner-0.6B-4bit",
        "Qwen--Qwen3-ForcedAligner-0.6B",
    ];
    for name in &candidates {
        let dir = base.join(name);
        if dir.exists() { return Some(dir); }
    }
    None
}

// ── Engine API ──────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn asr_engine_load(
    model_dir: *const c_char,
    out_err: *mut *mut c_char,
) -> *mut AsrEngine {
    init_logger();

    let model_dir = match unsafe { CStr::from_ptr(model_dir) }.to_str() {
        Ok(s) => PathBuf::from(s),
        Err(e) => {
            set_err(out_err, &format!("invalid model_dir: {e}"));
            return std::ptr::null_mut();
        }
    };

    match load_engine(&model_dir) {
        Ok(engine) => Box::into_raw(Box::new(engine)),
        Err(e) => {
            set_err(out_err, &format!("{e}"));
            std::ptr::null_mut()
        }
    }
}

fn load_engine(model_dir: &Path) -> Result<AsrEngine, String> {
    let config_str = std::fs::read_to_string(model_dir.join("config.json"))
        .map_err(|e| format!("read config: {e}"))?;
    let config: AsrConfig = serde_json::from_str(&config_str)
        .map_err(|e| format!("parse config: {e}"))?;

    let mut model = Qwen3ASRModel::new(&config.thinker_config)
        .map_err(|e| format!("create model: {e}"))?;

    let stats = qwen3_asr_mlx::load::load_weights(&mut model, model_dir)
        .map_err(|e| format!("load weights: {e}"))?;

    use mlx_rs::module::ModuleParametersExt;
    model.eval().map_err(|e| format!("eval: {e}"))?;

    log::info!(
        "MLX engine loaded: {}/{} keys, {} quantized ({}bit)",
        stats.loaded, stats.total_keys, stats.quantized_layers, stats.bits,
    );

    let tokenizer = find_tokenizer(model_dir)
        .ok_or_else(|| "tokenizer.json not found".to_string())?;

    let inner = AsrEngineInner {
        model,
        tokenizer,
        model_dir: model_dir.to_path_buf(),
    };

    Ok(AsrEngine {
        inner: Arc::new(Mutex::new(inner)),
    })
}

#[no_mangle]
pub extern "C" fn asr_engine_from_pretrained(
    model_id: *const c_char,
    cache_dir: *const c_char,
    out_err: *mut *mut c_char,
) -> *mut AsrEngine {
    init_logger();

    let model_id = match unsafe { CStr::from_ptr(model_id) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_err(out_err, &format!("invalid model_id: {e}"));
            return std::ptr::null_mut();
        }
    };
    let cache_dir = match unsafe { CStr::from_ptr(cache_dir) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_err(out_err, &format!("invalid cache_dir: {e}"));
            return std::ptr::null_mut();
        }
    };

    // Resolve repo ID to local directory: "org/model" → "org--model" under cache_dir
    let dir_name = model_id.replace('/', "--");
    let model_dir = PathBuf::from(cache_dir).join(&dir_name);

    if !model_dir.exists() {
        set_err(out_err, &format!(
            "model not found at {}. Download it first.", model_dir.display()
        ));
        return std::ptr::null_mut();
    }

    match load_engine(&model_dir) {
        Ok(engine) => Box::into_raw(Box::new(engine)),
        Err(e) => {
            set_err(out_err, &format!("{e}"));
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn asr_engine_from_gguf(
    _base_repo_id: *const c_char,
    _gguf_repo_id: *const c_char,
    _gguf_filename: *const c_char,
    _cache_dir: *const c_char,
    out_err: *mut *mut c_char,
) -> *mut AsrEngine {
    set_err(out_err, "GGUF not supported by MLX backend — use safetensors models");
    std::ptr::null_mut()
}

#[no_mangle]
pub extern "C" fn asr_engine_transcribe_samples(
    _engine: *const AsrEngine,
    _samples: *const c_float,
    _num_samples: usize,
    out_err: *mut *mut c_char,
) -> *mut c_char {
    set_err(out_err, "single-shot transcription not yet implemented");
    std::ptr::null_mut()
}

#[no_mangle]
pub extern "C" fn asr_engine_free(engine: *mut AsrEngine) {
    if !engine.is_null() {
        unsafe { drop(Box::from_raw(engine)) };
    }
}

// ── Session API ─────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn asr_session_create(
    engine: *const AsrEngine,
    opts: AsrSessionOptions,
) -> *mut AsrSession {
    if engine.is_null() {
        return std::ptr::null_mut();
    }

    let engine_ref = unsafe { &*engine };
    let inner = engine_ref.inner.clone();

    let chunk_size = if opts.chunk_size_sec > 0.0 { opts.chunk_size_sec } else { 0.4 };
    let max_streaming = if opts.max_new_tokens_streaming > 0 { opts.max_new_tokens_streaming as usize } else { 32 };
    let max_final = if opts.max_new_tokens_final > 0 { opts.max_new_tokens_final as usize } else { 512 };

    let language = if !opts.language.is_null() {
        let s = unsafe { CStr::from_ptr(opts.language) }.to_str().unwrap_or("");
        if s.is_empty() { None } else { Some(s.to_string()) }
    } else {
        None
    };

    let mut streaming_opts = StreamingOptions::default()
        .with_mode(StreamingMode::Rotate)
        .with_chunk_size_sec(chunk_size)
        .with_max_new_tokens_streaming(max_streaming)
        .with_max_new_tokens_final(max_final);

    if let Some(lang) = language {
        streaming_opts = streaming_opts.with_language(lang);
    }

    // Load forced aligner if available
    let aligner = {
        let guard = inner.lock().unwrap();
        find_aligner_dir().and_then(|dir| {
            match ForcedAligner::load(&dir, guard.tokenizer.clone()) {
                Ok(a) => {
                    log::info!("Forced aligner loaded for session");
                    Some(a)
                }
                Err(e) => {
                    log::warn!("Failed to load forced aligner: {e}");
                    None
                }
            }
        })
    };

    let tokenizer = {
        let guard = inner.lock().unwrap();
        guard.tokenizer.clone()
    };

    let mut state = StreamingState::new(streaming_opts, tokenizer, aligner);

    // Load Silero VAD if available
    if let Some(vad_dir) = find_vad_dir() {
        match qwen3_asr_mlx::vad::SileroVad::load(&vad_dir) {
            Ok(vad) => {
                log::info!("Silero VAD loaded for session");
                ffi_log("Silero VAD loaded");
                state.vad = Some(vad);
            }
            Err(e) => {
                log::warn!("Failed to load Silero VAD: {e}");
                ffi_log(&format!("Failed to load Silero VAD: {e}"));
            }
        }
    }

    Box::into_raw(Box::new(AsrSession {
        engine: inner,
        state,
        last_text: String::new(),
    }))
}

#[repr(C)]
pub struct AsrFeedResult {
    pub text: *mut c_char,
    pub committed_utf16_len: usize,
    pub alignments_json: *mut c_char,
    pub debug_json: *mut c_char,
}

impl AsrFeedResult {
    fn empty() -> Self {
        Self {
            text: std::ptr::null_mut(),
            committed_utf16_len: 0,
            alignments_json: std::ptr::null_mut(),
            debug_json: std::ptr::null_mut(),
        }
    }
}

#[no_mangle]
pub extern "C" fn asr_session_feed(
    session: *mut AsrSession,
    samples: *const c_float,
    num_samples: usize,
    out_err: *mut *mut c_char,
) -> AsrFeedResult {
    feed_impl(session, samples, num_samples, out_err, false)
}

#[no_mangle]
pub extern "C" fn asr_session_feed_finalizing(
    session: *mut AsrSession,
    samples: *const c_float,
    num_samples: usize,
    out_err: *mut *mut c_char,
) -> AsrFeedResult {
    feed_impl(session, samples, num_samples, out_err, true)
}

fn feed_impl(
    session: *mut AsrSession,
    samples: *const c_float,
    num_samples: usize,
    out_err: *mut *mut c_char,
    finalizing: bool,
) -> AsrFeedResult {
    if session.is_null() || samples.is_null() {
        return AsrFeedResult::empty();
    }

    let session = unsafe { &mut *session };
    let audio = unsafe { std::slice::from_raw_parts(samples, num_samples) };

    let mut guard = session.engine.lock().unwrap();
    let result = if finalizing {
        streaming::feed_audio_finalizing(&mut guard.model, &mut session.state, audio)
    } else {
        streaming::feed_audio(&mut guard.model, &mut session.state, audio)
    };
    drop(guard);

    // Drain debug events regardless of result
    let debug_events = std::mem::take(&mut session.state.debug_events);
    let debug_json = if debug_events.is_empty() {
        std::ptr::null_mut()
    } else {
        to_c_string(&format!("[{}]", debug_events.join(",")))
    };

    match result {
        Ok(Some(text)) => {
            if text != session.last_text {
                ffi_log(&format!("FEED text changed:\n  was: {:?}\n  now: {:?}", session.last_text, text));
            }
            session.last_text = text.clone();

            // Compute committed UTF-16 length
            let committed_utf16_len = session.state.committed_text
                .encode_utf16().count();

            // Serialize alignments if any
            let alignments_json = if session.state.committed_alignments.is_empty() {
                std::ptr::null_mut()
            } else {
                let json = serde_json::to_string(&session.state.committed_alignments
                    .iter()
                    .map(|a| serde_json::json!({
                        "word": a.word,
                        "start": (a.start_time * 1000.0).round() / 1000.0,
                        "end": (a.end_time * 1000.0).round() / 1000.0,
                    }))
                    .collect::<Vec<_>>()
                ).unwrap_or_else(|_| "[]".to_string());
                to_c_string(&json)
            };

            AsrFeedResult {
                text: to_c_string(&text),
                committed_utf16_len,
                alignments_json,
                debug_json,
            }
        }
        Ok(None) => {
            AsrFeedResult {
                text: std::ptr::null_mut(),
                committed_utf16_len: 0,
                alignments_json: std::ptr::null_mut(),
                debug_json,
            }
        }
        Err(e) => {
            ffi_log(&format!("FEED => error: {e}"));
            set_err(out_err, &format!("{e}"));
            AsrFeedResult {
                text: std::ptr::null_mut(),
                committed_utf16_len: 0,
                alignments_json: std::ptr::null_mut(),
                debug_json,
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn asr_feed_result_free(result: AsrFeedResult) {
    if !result.text.is_null() { unsafe { drop(CString::from_raw(result.text)); } }
    if !result.alignments_json.is_null() { unsafe { drop(CString::from_raw(result.alignments_json)); } }
    if !result.debug_json.is_null() { unsafe { drop(CString::from_raw(result.debug_json)); } }
}

#[no_mangle]
pub extern "C" fn asr_session_finish(
    session: *mut AsrSession,
    out_err: *mut *mut c_char,
) -> *mut c_char {
    if session.is_null() {
        return std::ptr::null_mut();
    }

    let session = unsafe { &mut *session };
    let mut guard = session.engine.lock().unwrap();
    let result = streaming::finish_streaming(&mut guard.model, &mut session.state);
    drop(guard);

    match result {
        Ok(text) => {
            session.last_text = text.clone();
            to_c_string(&text)
        }
        Err(e) => {
            set_err(out_err, &format!("{e}"));
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn asr_session_last_language(session: *const AsrSession) -> *mut c_char {
    if session.is_null() {
        return std::ptr::null_mut();
    }
    let session = unsafe { &*session };
    to_c_string(&session.state.language)
}

#[no_mangle]
pub extern "C" fn asr_session_set_language(
    session: *mut AsrSession,
    language: *const c_char,
    _out_err: *mut *mut c_char,
) -> bool {
    if session.is_null() {
        return false;
    }
    // Language changes mid-stream not yet supported in MLX backend.
    // The language is set at session creation.
    true
}

#[no_mangle]
pub extern "C" fn asr_session_free(session: *mut AsrSession) {
    if !session.is_null() {
        unsafe { drop(Box::from_raw(session)) };
    }
}

#[no_mangle]
pub extern "C" fn asr_string_free(s: *mut c_char) {
    if !s.is_null() {
        unsafe { drop(CString::from_raw(s)) };
    }
}
