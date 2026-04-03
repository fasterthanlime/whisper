//! FFI layer for bee-transcribe — C API for streaming speech-to-text.

use std::ffi::{c_char, c_float, c_uint, CStr, CString};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, Once};
use std::time::{Duration, Instant};

use bee_transcribe::{Engine, EngineConfig, Language, Session, SessionOptions};

static INIT_LOGGER: Once = Once::new();

fn init_logger() {
    INIT_LOGGER.call_once(|| {
        env_logger::init();
    });
}

// ── Engine stats ────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy)]
pub struct AsrEngineStats {
    pub cpu_percent: c_float,
    pub gpu_percent: c_float,
    pub vram_used_mb: c_float,
    pub ram_used_mb: c_float,
}

impl Default for AsrEngineStats {
    fn default() -> Self {
        Self { cpu_percent: 0.0, gpu_percent: 0.0, vram_used_mb: 0.0, ram_used_mb: 0.0 }
    }
}

struct StatsSampler {
    latest: Arc<Mutex<AsrEngineStats>>,
}

impl StatsSampler {
    // EMA smoothing factor: α=0.25 at 400ms → time constant ~1.4s
    const ALPHA: f32 = 0.25;

    fn new() -> Self {
        let latest = Arc::new(Mutex::new(AsrEngineStats::default()));
        let shared = Arc::clone(&latest);
        std::thread::Builder::new()
            .name("bee-stats".into())
            .spawn(move || {
                let mut last_cpu_us: u64 = 0;
                let mut last_wall = Instant::now();
                let mut smooth = AsrEngineStats::default();
                // GPU is expensive (subprocess), sample it less often.
                let mut gpu_tick: u32 = 0;
                let mut last_gpu_percent = 0.0f32;
                let mut last_vram_mb = 0.0f32;

                loop {
                    std::thread::sleep(Duration::from_millis(400));
                    let now = Instant::now();
                    let wall_us = now.duration_since(last_wall).as_micros() as u64;
                    last_wall = now;

                    // CPU: cheap, every tick
                    let cpu_us = process_cpu_us();
                    let cpu_raw = if wall_us > 0 && last_cpu_us > 0 {
                        let delta = cpu_us.saturating_sub(last_cpu_us);
                        ((delta as f32 / wall_us as f32) * 100.0).min(100.0)
                    } else {
                        0.0
                    };
                    last_cpu_us = cpu_us;

                    // RAM: cheap, every tick
                    let ram_raw = process_ram_mb();

                    // GPU/VRAM: expensive subprocess, every ~2s (every 5th tick)
                    gpu_tick += 1;
                    if gpu_tick >= 5 {
                        gpu_tick = 0;
                        if let Some((g, v)) = sample_gpu_ioreg() {
                            last_gpu_percent = g;
                            last_vram_mb = v;
                        }
                    }

                    let a = Self::ALPHA;
                    smooth.cpu_percent  = a * cpu_raw          + (1.0 - a) * smooth.cpu_percent;
                    smooth.gpu_percent  = a * last_gpu_percent + (1.0 - a) * smooth.gpu_percent;
                    smooth.vram_used_mb = a * last_vram_mb     + (1.0 - a) * smooth.vram_used_mb;
                    smooth.ram_used_mb  = a * ram_raw          + (1.0 - a) * smooth.ram_used_mb;

                    if let Ok(mut s) = shared.lock() {
                        *s = smooth;
                    }
                }
            })
            .expect("failed to spawn stats thread");
        Self { latest }
    }

    fn get(&self) -> AsrEngineStats {
        self.latest.lock().map(|s| *s).unwrap_or_default()
    }
}

fn process_cpu_us() -> u64 {
    unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut usage);
        let user = usage.ru_utime.tv_sec as u64 * 1_000_000 + usage.ru_utime.tv_usec as u64;
        let sys  = usage.ru_stime.tv_sec as u64 * 1_000_000 + usage.ru_stime.tv_usec as u64;
        user + sys
    }
}

fn process_ram_mb() -> f32 {
    // mach task_basic_info gives current resident set size without shelling out.
    #[repr(C)]
    struct TaskBasicInfo {
        suspend_count: u32,
        virtual_size: usize,
        resident_size: usize,
        user_time: [i32; 2],
        system_time: [i32; 2],
        policy: i32,
    }
    extern "C" {
        static mach_task_self_: u32;
        fn task_info(
            target_task: u32,
            flavor: u32,
            task_info_out: *mut std::ffi::c_void,
            task_info_out_cnt: *mut u32,
        ) -> i32;
    }
    const TASK_BASIC_INFO: u32 = 5;
    unsafe {
        let mut info: TaskBasicInfo = std::mem::zeroed();
        let mut count = (std::mem::size_of::<TaskBasicInfo>() / 4) as u32;
        let kr = task_info(
            mach_task_self_,
            TASK_BASIC_INFO,
            &mut info as *mut _ as *mut std::ffi::c_void,
            &mut count,
        );
        if kr == 0 { info.resident_size as f32 / (1024.0 * 1024.0) } else { 0.0 }
    }
}

fn sample_gpu_ioreg() -> Option<(f32, f32)> {
    let out = std::process::Command::new("ioreg")
        .args(["-r", "-d", "1", "-w", "0", "-c", "AGXAccelerator"])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let line = text.lines().find(|l| l.contains("PerformanceStatistics"))?;
    let gpu = parse_ioreg_u64(line, "Device Utilization %").unwrap_or(0) as f32;
    let vram_bytes = parse_ioreg_u64(line, "In use system memory").unwrap_or(0);
    Some((gpu, vram_bytes as f32 / (1024.0 * 1024.0)))
}

fn parse_ioreg_u64(s: &str, key: &str) -> Option<u64> {
    let pat = format!("\"{}\"=", key);
    let pos = s.find(&pat)?;
    let rest = &s[pos + pat.len()..];
    let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    rest[..end].parse().ok()
}

// ── Engine ──────────────────────────────────────────────────────────────

pub struct AsrEngine {
    inner: Arc<Engine>,
    /// Pre-loaded VAD tensors (loaded once, cloned per session).
    vad_tensors: Option<std::collections::HashMap<String, mlx_rs::Array>>,
    stats: StatsSampler,
}

// SAFETY: Engine is immutable after construction. MLX arrays are heap-allocated
// Metal buffers; concurrent read access is safe.
unsafe impl Send for AsrEngine {}
unsafe impl Sync for AsrEngine {}

// ── Session ─────────────────────────────────────────────────────────────

pub struct AsrSession {
    // We store the Arc to keep the engine alive. The Session borrows it
    // via a raw pointer with 'static lifetime — safe because the Arc
    // guarantees the engine outlives the session.
    _engine: Arc<Engine>,
    session: Session<'static>,
    last_text: String,
    finished: bool,
}

// SAFETY: Session contains MLX arrays (Metal buffers) accessed sequentially.
// Swift calls are serialized by the caller.
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
    pub rollback_token_num: c_uint,
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
    CString::new(s)
        .map(|cs| cs.into_raw())
        .unwrap_or(std::ptr::null_mut())
}

fn ffi_log(msg: &str) {
    let path = "/tmp/bee.log";
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = writeln!(
            f,
            "[{:.3}] FFI: {}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64(),
            msg
        );
    }
}

fn find_vad_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("BEE_VAD_DIR") {
        let p = PathBuf::from(dir);
        if p.exists() {
            return Some(p);
        }
    }
    let dir = dirs::home_dir()?.join("Library/Caches/qwen3-asr/aitytech--Silero-VAD-v5-MLX");
    if dir.exists() {
        Some(dir)
    } else {
        None
    }
}

fn resolve_engine_config(model_dir: &Path) -> Result<EngineConfig<'static>, String> {
    // Tokenizer: env var, then model_dir/tokenizer.json
    let tokenizer_path: PathBuf = if let Ok(p) = std::env::var("BEE_TOKENIZER_PATH") {
        PathBuf::from(p)
    } else {
        model_dir.join("tokenizer.json")
    };
    if !tokenizer_path.exists() {
        return Err(format!(
            "tokenizer not found at {}",
            tokenizer_path.display()
        ));
    }

    // Aligner: env var, then well-known locations
    let aligner_dir: PathBuf = if let Ok(p) = std::env::var("BEE_ALIGNER_DIR") {
        PathBuf::from(p)
    } else {
        let home = dirs::home_dir().ok_or("no home dir")?;
        let base = home.join("Library/Caches/qwen3-asr");
        let candidates = [
            "mlx-community--Qwen3-ForcedAligner-0.6B-4bit",
            "Qwen--Qwen3-ForcedAligner-0.6B",
        ];
        candidates
            .iter()
            .map(|n| base.join(n))
            .find(|p| p.exists())
            .ok_or("forced aligner not found")?
    };

    // Leak the PathBufs to get 'static references (engine lives for process lifetime)
    let model_dir: &'static Path = Box::leak(model_dir.to_path_buf().into_boxed_path());
    let tokenizer_path: &'static Path = Box::leak(tokenizer_path.into_boxed_path());
    let aligner_dir: &'static Path = Box::leak(aligner_dir.into_boxed_path());

    Ok(EngineConfig {
        model_dir,
        tokenizer_path,
        aligner_dir,
    })
}

// ── Engine API ──────────────────────────────────────────────────────────

/// # Safety
/// `model_dir` must be a valid, nul-terminated UTF-8 pointer.
#[no_mangle]
pub unsafe extern "C" fn asr_engine_load(
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
            set_err(out_err, &e);
            std::ptr::null_mut()
        }
    }
}

fn load_engine(model_dir: &Path) -> Result<AsrEngine, String> {
    // Cap MLX's Metal buffer cache at 2GB to prevent unbounded memory growth
    bee_transcribe::set_mlx_cache_limit(2 * 1024 * 1024 * 1024);

    let config = resolve_engine_config(model_dir)?;
    let engine = Engine::load(&config).map_err(|e| format!("load engine: {e}"))?;

    let vad_tensors = find_vad_dir().and_then(|d| {
        let st_path = d.join("model.safetensors");
        match mlx_rs::Array::load_safetensors(&st_path) {
            Ok(tensors) => {
                ffi_log(&format!("Silero VAD loaded ({} tensors)", tensors.len()));
                Some(tensors)
            }
            Err(e) => {
                ffi_log(&format!("Failed to load VAD: {e}"));
                None
            }
        }
    });

    Ok(AsrEngine {
        inner: Arc::new(engine),
        vad_tensors,
        stats: StatsSampler::new(),
    })
}

/// # Safety
/// `model_id` and `cache_dir` must be valid, nul-terminated UTF-8 pointers.
#[no_mangle]
pub unsafe extern "C" fn asr_engine_from_pretrained(
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

    let dir_name = model_id.replace('/', "--");
    let model_dir = PathBuf::from(cache_dir).join(&dir_name);

    if !model_dir.exists() {
        set_err(
            out_err,
            &format!("model not found at {}", model_dir.display()),
        );
        return std::ptr::null_mut();
    }

    match load_engine(&model_dir) {
        Ok(engine) => Box::into_raw(Box::new(engine)),
        Err(e) => {
            set_err(out_err, &e);
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
    set_err(
        out_err,
        "GGUF not supported by MLX backend — use safetensors models",
    );
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

/// # Safety
/// `engine` must be a pointer returned by this module and not yet freed.
#[no_mangle]
pub unsafe extern "C" fn asr_engine_free(engine: *mut AsrEngine) {
    if !engine.is_null() {
        unsafe { drop(Box::from_raw(engine)) };
    }
}

// ── Session API ─────────────────────────────────────────────────────────

/// # Safety
/// `engine` must be a valid engine pointer.
#[no_mangle]
pub unsafe extern "C" fn asr_engine_get_stats(engine: *const AsrEngine) -> AsrEngineStats {
    if engine.is_null() {
        return AsrEngineStats::default();
    }
    unsafe { &*engine }.stats.get()
}

/// # Safety
/// `engine` must be a valid engine pointer.
#[no_mangle]
pub unsafe extern "C" fn asr_session_create(
    engine: *const AsrEngine,
    opts: AsrSessionOptions,
) -> *mut AsrSession {
    if engine.is_null() {
        return std::ptr::null_mut();
    }

    let engine_ref = unsafe { &*engine };
    let arc = engine_ref.inner.clone();

    let chunk_duration = if opts.chunk_size_sec > 0.0 {
        opts.chunk_size_sec
    } else {
        0.4
    };
    let max_streaming = if opts.max_new_tokens_streaming > 0 {
        opts.max_new_tokens_streaming as usize
    } else {
        32
    };
    let max_final = if opts.max_new_tokens_final > 0 {
        opts.max_new_tokens_final as usize
    } else {
        512
    };

    let language = if !opts.language.is_null() {
        let s = unsafe { CStr::from_ptr(opts.language) }
            .to_str()
            .unwrap_or("");
        if s.is_empty() {
            Language::default()
        } else {
            Language(s.to_string())
        }
    } else {
        Language::default()
    };

    let commit_token_count = if opts.unfixed_token_num > 0 {
        opts.unfixed_token_num as usize
    } else {
        12
    };
    let rollback_tokens = if opts.rollback_token_num > 0 {
        opts.rollback_token_num as usize
    } else {
        5
    };

    let session_opts = SessionOptions {
        chunk_duration,
        commit_token_count,
        rollback_tokens,
        max_tokens_streaming: max_streaming,
        max_tokens_final: max_final,
        language,
        ..SessionOptions::default()
    };

    // SAFETY: The Arc keeps the engine alive for the lifetime of the session.
    // We transmute the lifetime to 'static since FFI can't express borrows.
    let engine_ref: &Engine = &arc;
    let engine_static: &'static Engine = unsafe { std::mem::transmute(engine_ref) };
    let mut session = engine_static.session(session_opts);

    // Create VAD from pre-loaded tensors
    let asr_engine = unsafe { &*engine };
    if let Some(ref tensors) = asr_engine.vad_tensors {
        match bee_vad::SileroVad::from_tensors(tensors) {
            Ok(vad) => session.set_vad(vad),
            Err(e) => ffi_log(&format!("Failed to create VAD: {e}")),
        }
    }

    Box::into_raw(Box::new(AsrSession {
        _engine: arc,
        session,
        last_text: String::new(),
        finished: false,
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
    feed_impl(session, samples, num_samples, out_err)
}

#[no_mangle]
pub extern "C" fn asr_session_feed_finalizing(
    session: *mut AsrSession,
    samples: *const c_float,
    num_samples: usize,
    out_err: *mut *mut c_char,
) -> AsrFeedResult {
    // With the new API, there's no separate "finalizing" feed — just feed
    // normally and call finish() when done. For backward compat, treat this
    // the same as a regular feed.
    feed_impl(session, samples, num_samples, out_err)
}

fn feed_impl(
    session: *mut AsrSession,
    samples: *const c_float,
    num_samples: usize,
    out_err: *mut *mut c_char,
) -> AsrFeedResult {
    if session.is_null() || samples.is_null() {
        return AsrFeedResult::empty();
    }

    let session = unsafe { &mut *session };
    if session.finished {
        return AsrFeedResult::empty();
    }
    let audio = unsafe { std::slice::from_raw_parts(samples, num_samples) };

    match session.session.feed(audio) {
        Ok(Some(update)) => {
            if update.text != session.last_text {
                ffi_log(&format!(
                    "FEED text changed:\n  was: {:?}\n  now: {:?}",
                    session.last_text, update.text
                ));
            }
            session.last_text = update.text.clone();

            let committed_utf16_len = update.text[..update.committed_len].encode_utf16().count();

            let alignments_json = if update.alignments.is_empty() {
                std::ptr::null_mut()
            } else {
                let json = serde_json::to_string(
                    &update
                        .alignments
                        .iter()
                        .map(|a| {
                            serde_json::json!({
                                "word": a.word,
                                "start": (a.start * 1000.0).round() / 1000.0,
                                "end": (a.end * 1000.0).round() / 1000.0,
                            })
                        })
                        .collect::<Vec<_>>(),
                )
                .unwrap_or_else(|_| "[]".to_string());
                to_c_string(&json)
            };

            AsrFeedResult {
                text: to_c_string(&update.text),
                committed_utf16_len,
                alignments_json,
                debug_json: std::ptr::null_mut(),
            }
        }
        Ok(None) => AsrFeedResult::empty(),
        Err(e) => {
            ffi_log(&format!("FEED => error: {e}"));
            set_err(out_err, &e.to_string());
            AsrFeedResult::empty()
        }
    }
}

#[no_mangle]
pub extern "C" fn asr_feed_result_free(result: AsrFeedResult) {
    if !result.text.is_null() {
        unsafe { drop(CString::from_raw(result.text)) };
    }
    if !result.alignments_json.is_null() {
        unsafe { drop(CString::from_raw(result.alignments_json)) };
    }
    if !result.debug_json.is_null() {
        unsafe { drop(CString::from_raw(result.debug_json)) };
    }
}

/// # Safety
/// `session` must be a valid session pointer.
#[no_mangle]
pub unsafe extern "C" fn asr_session_finish(
    session: *mut AsrSession,
    out_err: *mut *mut c_char,
) -> *mut c_char {
    if session.is_null() {
        return std::ptr::null_mut();
    }

    // Take ownership of the session to call finish() which consumes it.
    // We reconstruct a new dummy session in its place.
    let session = unsafe { &mut *session };

    if session.finished {
        return std::ptr::null_mut();
    }
    session.finished = true;

    // We can't consume session.session through a pointer, so we use a
    // temporary: swap in a fresh session, finish the real one.
    let arc = session._engine.clone();
    let engine_ref: &Engine = &arc;
    let engine_static: &'static Engine = unsafe { std::mem::transmute(engine_ref) };
    let real_session = std::mem::replace(
        &mut session.session,
        engine_static.session(SessionOptions::default()),
    );

    match real_session.finish() {
        Ok(update) => {
            session.last_text = update.text.clone();
            to_c_string(&update.text)
        }
        Err(e) => {
            set_err(out_err, &e.to_string());
            std::ptr::null_mut()
        }
    }
}

/// # Safety
/// `session` must be a valid session pointer.
#[no_mangle]
pub unsafe extern "C" fn asr_session_last_language(_session: *const AsrSession) -> *mut c_char {
    // Language is set at session creation and doesn't change.
    to_c_string("English")
}

#[no_mangle]
pub extern "C" fn asr_session_set_language(
    _session: *mut AsrSession,
    _language: *const c_char,
    _out_err: *mut *mut c_char,
) -> bool {
    true
}

/// # Safety
/// `session` must be a valid session pointer.
#[no_mangle]
pub unsafe extern "C" fn asr_session_free(session: *mut AsrSession) {
    if !session.is_null() {
        unsafe { drop(Box::from_raw(session)) };
    }
}

/// # Safety
/// `s` must be either null or a pointer returned by this module.
#[no_mangle]
pub unsafe extern "C" fn asr_string_free(s: *mut c_char) {
    if !s.is_null() {
        unsafe { drop(CString::from_raw(s)) };
    }
}
