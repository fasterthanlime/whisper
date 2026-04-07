//! FFI layer for bee — vox-ffi service + legacy C API.

use std::ffi::{c_char, c_float, c_uint, CStr, CString};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Once};
use std::time::{Duration, Instant};

use bee_rpc::{BeeDispatcher, FeedResult, RepoDownload, RepoFile};
use bee_transcribe::{Engine, EngineConfig, Language, Session, SessionOptions};
use tracing::info;
use vox::acceptor_on;
use vox_ffi::declare_link_endpoint;

// ── Vox-FFI endpoint ───────────────────────────────────────────────────

declare_link_endpoint!(pub mod bee_ffi_endpoint { export = bee_ffi_v1_vtable; });

static VOX_BOOTSTRAPPED: AtomicBool = AtomicBool::new(false);

fn bootstrap_service_once() {
    if VOX_BOOTSTRAPPED.swap(true, Ordering::AcqRel) {
        return;
    }

    ffi_log("[bee-ffi] bootstrap_service_once: spawning runtime thread");
    std::thread::spawn(|| {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build();
        let Ok(runtime) = runtime else {
            ffi_log("[bee-ffi] failed to create tokio runtime");
            return;
        };

        runtime.block_on(async {
            ffi_log("[bee-ffi] runtime thread: waiting for accept");
            let Ok(link) = bee_ffi_endpoint::accept().await else {
                ffi_log("[bee-ffi] runtime thread: accept failed");
                return;
            };
            ffi_log("[bee-ffi] runtime thread: link accepted");

            let service = BeeService::new();
            let establish = acceptor_on(link)
                .on_connection(BeeDispatcher::new(service))
                .establish::<bee_rpc::BeeClient>()
                .await;

            match establish {
                Ok(client) => {
                    info!("bee-ffi: vox session established");
                    ffi_log("[bee-ffi] runtime thread: session established");
                    client.caller.closed().await;
                    ffi_log("[bee-ffi] runtime thread: caller closed");
                    if let Some(session) = client.session.as_ref() {
                        let _ = session.shutdown();
                    }
                }
                Err(error) => {
                    ffi_log(&format!("[bee-ffi] runtime thread: establish failed: {error}"));
                }
            }
        });
    });

    // Give the runtime thread a moment to start accepting
    std::thread::sleep(Duration::from_millis(10));
}

// ── BeeService impl ───────────────────────────────────────────────────

struct BeeServiceInner {
    engine: Mutex<Option<AsrEngine>>,
    sessions: Mutex<std::collections::HashMap<String, AsrSession>>,
    next_session_id: Mutex<u64>,
}

#[derive(Clone)]
struct BeeService {
    inner: Arc<BeeServiceInner>,
}

impl BeeService {
    fn new() -> Self {
        Self {
            inner: Arc::new(BeeServiceInner {
                engine: Mutex::new(None),
                sessions: Mutex::new(std::collections::HashMap::new()),
                next_session_id: Mutex::new(1),
            }),
        }
    }
}

const HF_BASE: &str = "https://huggingface.co";

fn hf_file_url(repo_id: &str, filename: &str) -> String {
    format!("{HF_BASE}/{repo_id}/resolve/main/{filename}")
}

impl bee_rpc::Bee for BeeService {
    async fn required_downloads(&self) -> Vec<RepoDownload> {
        vec![
            RepoDownload {
                repo_id: "mlx-community/Qwen3-ASR-1.7B-4bit".into(),
                local_dir: "mlx-community--Qwen3-ASR-1.7B-4bit".into(),
                files: vec![
                    RepoFile { name: "config.json".into(), url: hf_file_url("mlx-community/Qwen3-ASR-1.7B-4bit", "config.json"), size: 0 },
                    RepoFile { name: "tokenizer.json".into(), url: hf_file_url("mlx-community/Qwen3-ASR-1.7B-4bit", "tokenizer.json"), size: 0 },
                    RepoFile { name: "model.safetensors".into(), url: hf_file_url("mlx-community/Qwen3-ASR-1.7B-4bit", "model.safetensors"), size: 0 },
                    RepoFile { name: "generation_config.json".into(), url: hf_file_url("mlx-community/Qwen3-ASR-1.7B-4bit", "generation_config.json"), size: 0 },
                    RepoFile { name: "preprocessor_config.json".into(), url: hf_file_url("mlx-community/Qwen3-ASR-1.7B-4bit", "preprocessor_config.json"), size: 0 },
                ],
            },
            RepoDownload {
                repo_id: "mlx-community/Qwen3-ForcedAligner-0.6B-4bit".into(),
                local_dir: "mlx-community--Qwen3-ForcedAligner-0.6B-4bit".into(),
                files: vec![
                    RepoFile { name: "config.json".into(), url: hf_file_url("mlx-community/Qwen3-ForcedAligner-0.6B-4bit", "config.json"), size: 0 },
                    RepoFile { name: "model.safetensors".into(), url: hf_file_url("mlx-community/Qwen3-ForcedAligner-0.6B-4bit", "model.safetensors"), size: 0 },
                    RepoFile { name: "tokenizer.json".into(), url: hf_file_url("mlx-community/Qwen3-ForcedAligner-0.6B-4bit", "tokenizer.json"), size: 0 },
                ],
            },
            RepoDownload {
                repo_id: "aitytech/Silero-VAD-v5-MLX".into(),
                local_dir: "aitytech--Silero-VAD-v5-MLX".into(),
                files: vec![
                    RepoFile { name: "model.safetensors".into(), url: hf_file_url("aitytech/Silero-VAD-v5-MLX", "model.safetensors"), size: 0 },
                ],
            },
        ]
    }

    async fn load_engine(&self, cache_dir: String) -> String {
        let cache_base = PathBuf::from(&cache_dir);
        let model_dir = cache_base.join("mlx-community--Qwen3-ASR-1.7B-4bit");

        match load_engine(&model_dir, &cache_base) {
            Ok(engine) => {
                *self.inner.engine.lock().unwrap() = Some(engine);
                String::new()
            }
            Err(e) => e,
        }
    }

    async fn create_session(&self, language: String) -> String {
        let guard = self.inner.engine.lock().unwrap();
        let Some(engine) = guard.as_ref() else {
            return String::new();
        };

        let lang = if language.is_empty() {
            Language::default()
        } else {
            Language(language)
        };

        let opts = SessionOptions {
            language: lang,
            ..SessionOptions::default()
        };

        let engine_ref: &Engine = &engine.inner;
        let engine_static: &'static Engine = unsafe { std::mem::transmute(engine_ref) };
        let mut session = engine_static.session(opts);

        if let Some(ref tensors) = engine.vad_tensors {
            match bee_vad::SileroVad::from_tensors(tensors) {
                Ok(vad) => session.set_vad(vad),
                Err(e) => ffi_log(&format!("Failed to create VAD: {e}")),
            }
        }

        let mut id_counter = self.inner.next_session_id.lock().unwrap();
        let id = format!("session-{}", *id_counter);
        *id_counter += 1;

        self.inner.sessions.lock().unwrap().insert(
            id.clone(),
            AsrSession {
                _engine: engine.inner.clone(),
                session,
                last_text: String::new(),
                finished: false,
            },
        );

        id
    }

    async fn feed(&self, session_id: String, samples: Vec<f32>) -> FeedResult {
        let mut sessions = self.inner.sessions.lock().unwrap();
        let Some(session) = sessions.get_mut(&session_id) else {
            return FeedResult {
                text: String::new(),
                is_final: false,
            };
        };

        if session.finished {
            return FeedResult {
                text: String::new(),
                is_final: false,
            };
        }

        match session.session.feed(&samples) {
            Ok(Some(update)) => {
                session.last_text = update.text.clone();
                FeedResult {
                    text: update.text,
                    is_final: false,
                }
            }
            Ok(None) => FeedResult {
                text: session.last_text.clone(),
                is_final: false,
            },
            Err(e) => {
                ffi_log(&format!("FEED error: {e}"));
                FeedResult {
                    text: format!("error: {e}"),
                    is_final: false,
                }
            }
        }
    }

    async fn finish_session(&self, session_id: String) -> String {
        let mut sessions = self.inner.sessions.lock().unwrap();
        let Some(mut session) = sessions.remove(&session_id) else {
            return String::new();
        };

        if session.finished {
            return session.last_text;
        }
        session.finished = true;

        let arc = session._engine.clone();
        let engine_ref: &Engine = &arc;
        let engine_static: &'static Engine = unsafe { std::mem::transmute(engine_ref) };
        let real_session = std::mem::replace(
            &mut session.session,
            engine_static.session(SessionOptions::default()),
        );

        match real_session.finish() {
            Ok(update) => update.text,
            Err(e) => format!("error: {e}"),
        }
    }
}

// Trigger bootstrap when the vtable is first accessed
#[unsafe(no_mangle)]
pub extern "C" fn bee_ffi_bootstrap() {
    bootstrap_service_once();
}

static INIT_LOGGER: Once = Once::new();

fn init_logger() {
    INIT_LOGGER.call_once(|| {
        env_logger::init();
    });
}

static FFI_LOG_PATH: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Set the path for FFI diagnostic logs. Call this before loading any engine.
/// Pass NULL to disable file logging.
#[unsafe(no_mangle)]
pub extern "C" fn asr_set_log_path(path: *const c_char) {
    let mut guard = FFI_LOG_PATH.lock().unwrap();
    if path.is_null() {
        *guard = None;
    } else {
        let s = unsafe { CStr::from_ptr(path) };
        if let Ok(s) = s.to_str() {
            *guard = Some(PathBuf::from(s));
        }
    }
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
        Self {
            cpu_percent: 0.0,
            gpu_percent: 0.0,
            vram_used_mb: 0.0,
            ram_used_mb: 0.0,
        }
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

                loop {
                    std::thread::sleep(Duration::from_millis(400));
                    let now = Instant::now();
                    let wall_us = now.duration_since(last_wall).as_micros() as u64;
                    last_wall = now;

                    let cpu_us = process_cpu_us();
                    let cpu_raw = if wall_us > 0 && last_cpu_us > 0 {
                        let delta = cpu_us.saturating_sub(last_cpu_us);
                        ((delta as f32 / wall_us as f32) * 100.0).min(100.0)
                    } else {
                        0.0
                    };
                    last_cpu_us = cpu_us;

                    let ram_raw = process_ram_mb();
                    let (gpu_raw, vram_raw) = sample_gpu_iokit().unwrap_or((0.0, 0.0));

                    let a = Self::ALPHA;
                    smooth.cpu_percent = a * cpu_raw + (1.0 - a) * smooth.cpu_percent;
                    smooth.gpu_percent = a * gpu_raw + (1.0 - a) * smooth.gpu_percent;
                    smooth.vram_used_mb = a * vram_raw + (1.0 - a) * smooth.vram_used_mb;
                    smooth.ram_used_mb = a * ram_raw + (1.0 - a) * smooth.ram_used_mb;

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
        let sys = usage.ru_stime.tv_sec as u64 * 1_000_000 + usage.ru_stime.tv_usec as u64;
        user + sys
    }
}

fn process_ram_mb() -> f32 {
    // proc_pidinfo(PROC_PIDTASKINFO) gives current resident set size.
    #[repr(C)]
    struct ProcTaskinfo {
        pti_virtual_size: u64,
        pti_resident_size: u64,
        pti_total_user: u64,
        pti_total_system: u64,
        pti_threads_user: u64,
        pti_threads_system: u64,
        pti_policy: i32,
        pti_faults: i32,
        pti_pageins: i32,
        pti_cow_faults: i32,
        pti_messages_sent: i32,
        pti_messages_received: i32,
        pti_syscalls_mach: i32,
        pti_syscalls_unix: i32,
        pti_csw: i32,
        pti_threadnum: i32,
        pti_numrunning: i32,
        pti_priority: i32,
    }
    extern "C" {
        fn proc_pidinfo(
            pid: i32,
            flavor: i32,
            arg: u64,
            buffer: *mut std::ffi::c_void,
            buffersize: i32,
        ) -> i32;
    }
    const PROC_PIDTASKINFO: i32 = 4;
    unsafe {
        let mut info: ProcTaskinfo = std::mem::zeroed();
        let ret = proc_pidinfo(
            libc::getpid(),
            PROC_PIDTASKINFO,
            0,
            &mut info as *mut _ as *mut std::ffi::c_void,
            std::mem::size_of::<ProcTaskinfo>() as i32,
        );
        if ret > 0 {
            info.pti_resident_size as f32 / (1024.0 * 1024.0)
        } else {
            0.0
        }
    }
}

/// IOKit / CoreFoundation bindings — GPU stats without spawning a subprocess.
#[allow(non_upper_case_globals, non_snake_case)]
mod iokit {
    use std::ffi::{c_char, c_void};
    pub(crate) type IoServiceT = u32;
    pub(crate) const IO_OBJECT_NULL: IoServiceT = 0;
    pub(crate) const K_CF_STRING_ENCODING_UTF8: u32 = 0x0800_0100;
    pub(crate) const K_CF_NUMBER_SINT64_TYPE: i32 = 4;

    #[link(name = "IOKit", kind = "framework")]
    unsafe extern "C" {
        pub(crate) fn IOServiceMatching(name: *const c_char) -> *mut c_void;
        pub(crate) fn IOServiceGetMatchingService(
            masterPort: IoServiceT,
            matching: *mut c_void,
        ) -> IoServiceT;
        pub(crate) fn IORegistryEntryCreateCFProperties(
            entry: IoServiceT,
            properties: *mut *mut c_void,
            allocator: *const c_void,
            options: u32,
        ) -> i32;
        pub(crate) fn IOObjectRelease(object: IoServiceT) -> i32;
    }

    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        pub(crate) fn CFStringCreateWithCString(
            alloc: *const c_void,
            c_str: *const c_char,
            encoding: u32,
        ) -> *mut c_void;
        pub(crate) fn CFDictionaryGetValue(the_dict: *const c_void, key: *const c_void) -> *const c_void;
        pub(crate) fn CFNumberGetValue(
            number: *const c_void,
            the_type: i32,
            value_ptr: *mut c_void,
        ) -> bool;
        pub(crate) fn CFRelease(cf: *const c_void);
    }
}

fn cf_str(s: &str) -> *mut std::ffi::c_void {
    use iokit::*;
    let cs = std::ffi::CString::new(s).unwrap();
    unsafe { CFStringCreateWithCString(std::ptr::null(), cs.as_ptr(), K_CF_STRING_ENCODING_UTF8) }
}

fn cf_dict_i64(dict: *const std::ffi::c_void, key: &str) -> Option<i64> {
    use iokit::*;
    let k = cf_str(key);
    if k.is_null() {
        return None;
    }
    let val = unsafe { CFDictionaryGetValue(dict, k) };
    unsafe { CFRelease(k) };
    if val.is_null() {
        return None;
    }
    let mut out: i64 = 0;
    unsafe {
        CFNumberGetValue(
            val,
            K_CF_NUMBER_SINT64_TYPE,
            &mut out as *mut _ as *mut std::ffi::c_void,
        );
    }
    Some(out)
}

fn sample_gpu_iokit() -> Option<(f32, f32)> {
    use iokit::*;
    use std::ffi::CString;
    unsafe {
        let service_name = CString::new("AGXAccelerator").ok()?;
        let service = IOServiceGetMatchingService(0, IOServiceMatching(service_name.as_ptr()));
        if service == IO_OBJECT_NULL {
            return None;
        }
        let mut props: *mut std::ffi::c_void = std::ptr::null_mut();
        let kr = IORegistryEntryCreateCFProperties(service, &mut props, std::ptr::null(), 0);
        IOObjectRelease(service);
        if kr != 0 || props.is_null() {
            return None;
        }

        let perf_key = cf_str("PerformanceStatistics");
        let perf_dict = CFDictionaryGetValue(props, perf_key);
        CFRelease(perf_key);

        let result = if perf_dict.is_null() {
            None
        } else {
            let gpu = cf_dict_i64(perf_dict, "Device Utilization %").unwrap_or(0) as f32;
            let vram_bytes = cf_dict_i64(perf_dict, "In use system memory").unwrap_or(0);
            Some((gpu, vram_bytes as f32 / (1024.0 * 1024.0)))
        };

        CFRelease(props);
        result
    }
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

// ── Model paths (must match C header layout) ────────────────────────────

#[repr(C)]
pub struct AsrModelPaths {
    /// Base cache directory containing all model subdirectories.
    pub cache_dir: *const c_char,
}

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
    let path = FFI_LOG_PATH.lock().unwrap().clone();
    let Some(path) = path else { return };
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
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

fn find_vad_dir(cache_base: &Path) -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("BEE_VAD_DIR") {
        let p = PathBuf::from(dir);
        if p.exists() {
            return Some(p);
        }
    }
    let dir = cache_base.join("aitytech--Silero-VAD-v5-MLX");
    if dir.exists() {
        Some(dir)
    } else {
        None
    }
}

fn resolve_engine_config(
    model_dir: &Path,
    cache_base: &Path,
) -> Result<EngineConfig<'static>, String> {
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

    // Aligner: env var, then well-known locations under cache_base
    let aligner_dir: PathBuf = if let Ok(p) = std::env::var("BEE_ALIGNER_DIR") {
        PathBuf::from(p)
    } else {
        let candidates = [
            "mlx-community--Qwen3-ForcedAligner-0.6B-4bit",
            "Qwen--Qwen3-ForcedAligner-0.6B",
        ];
        candidates
            .iter()
            .map(|n| cache_base.join(n))
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
/// `model_dir` and `paths.cache_dir` must be valid, nul-terminated UTF-8 pointers.
#[no_mangle]
pub unsafe extern "C" fn asr_engine_load(
    model_dir: *const c_char,
    paths: AsrModelPaths,
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
    let cache_base = match unsafe { CStr::from_ptr(paths.cache_dir) }.to_str() {
        Ok(s) => PathBuf::from(s),
        Err(e) => {
            set_err(out_err, &format!("invalid paths.cache_dir: {e}"));
            return std::ptr::null_mut();
        }
    };

    match load_engine(&model_dir, &cache_base) {
        Ok(engine) => Box::into_raw(Box::new(engine)),
        Err(e) => {
            set_err(out_err, &e);
            std::ptr::null_mut()
        }
    }
}

fn load_engine(model_dir: &Path, cache_base: &Path) -> Result<AsrEngine, String> {
    // Cap MLX's Metal buffer cache at 2GB to prevent unbounded memory growth
    bee_transcribe::set_mlx_cache_limit(2 * 1024 * 1024 * 1024);

    let config = resolve_engine_config(model_dir, cache_base)?;
    let engine = Engine::load(&config).map_err(|e| format!("load engine: {e}"))?;

    let vad_tensors = find_vad_dir(cache_base).and_then(|d| {
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
/// `model_id` and `paths.cache_dir` must be valid, nul-terminated UTF-8 pointers.
#[no_mangle]
pub unsafe extern "C" fn asr_engine_from_pretrained(
    model_id: *const c_char,
    paths: AsrModelPaths,
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
    let cache_base = match unsafe { CStr::from_ptr(paths.cache_dir) }.to_str() {
        Ok(s) => PathBuf::from(s),
        Err(e) => {
            set_err(out_err, &format!("invalid paths.cache_dir: {e}"));
            return std::ptr::null_mut();
        }
    };

    let dir_name = model_id.replace('/', "--");
    let model_dir = cache_base.join(&dir_name);

    if !model_dir.exists() {
        set_err(
            out_err,
            &format!("model not found at {}", model_dir.display()),
        );
        return std::ptr::null_mut();
    }

    match load_engine(&model_dir, &cache_base) {
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
    _paths: AsrModelPaths,
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

// ── Correction Engine ──────────────────────────────────────────────────

use bee_correct::judge::{
    CorrectionEventSink, SpanDecision, TwoStageJudge,
};
use bee_correct::g2p::CachedEspeakG2p;
use bee_phonetic::{
    PhoneticIndex, SeedDataset, RetrievalQuery,
    query_index, score_shortlist,
    enumerate_transcript_spans_with, feature_tokens_for_ipa,
};
use bee_types::{CorrectionEvent, SpanContext, TranscriptSpan};

/// Opaque correction engine handle.
pub struct CorrectionEngine {
    judge: TwoStageJudge,
    index: PhoneticIndex,
    g2p: CachedEspeakG2p,
    events_path: Option<PathBuf>,
}

/// Opaque correction result handle.
pub struct CorrectionResult {
    session_id: String,
    best_text: String,
    edits_json: CString,
    spans: Vec<SpanWithDecision>,
}

struct SpanWithDecision {
    span: TranscriptSpan,
    decision: SpanDecision,
}

struct FileEventSink {
    file: Option<std::fs::File>,
}

impl CorrectionEventSink for FileEventSink {
    fn log_event(&mut self, event: &CorrectionEvent) {
        use std::io::Write;
        if let Some(ref mut f) = self.file {
            if let Ok(json) = facet_json::to_string(event) {
                let _ = writeln!(f, "{json}");
            }
        }
    }
}

/// Load a correction engine. Returns NULL on error.
///
/// # Safety
/// All string pointers must be valid nul-terminated UTF-8 or NULL.
#[no_mangle]
pub unsafe extern "C" fn bee_correct_engine_load(
    dataset_dir: *const c_char,
    events_path: *const c_char,
    gate_threshold: c_float,
    ranker_threshold: c_float,
    out_err: *mut *mut c_char,
) -> *mut CorrectionEngine {
    let dataset_dir = match ptr_to_str(dataset_dir) {
        Some(s) => PathBuf::from(s),
        None => {
            set_err(out_err, "dataset_dir is null or invalid");
            return std::ptr::null_mut();
        }
    };

    let events_path = ptr_to_str(events_path).map(PathBuf::from);

    match load_correction_engine(&dataset_dir, events_path, gate_threshold, ranker_threshold) {
        Ok(engine) => Box::into_raw(Box::new(engine)),
        Err(e) => {
            set_err(out_err, &e);
            std::ptr::null_mut()
        }
    }
}

fn ptr_to_str(p: *const c_char) -> Option<&'static str> {
    if p.is_null() {
        return None;
    }
    unsafe { CStr::from_ptr(p) }.to_str().ok()
}

fn load_correction_engine(
    dataset_dir: &Path,
    events_path: Option<PathBuf>,
    gate_threshold: c_float,
    ranker_threshold: c_float,
) -> Result<CorrectionEngine, String> {
    let dataset = SeedDataset::load(dataset_dir).map_err(|e| format!("load dataset: {e}"))?;
    let index = dataset.phonetic_index();

    let g2p = CachedEspeakG2p::english()
        .map_err(|e| format!("init g2p: {e}"))?;

    let gt = if gate_threshold > 0.0 { gate_threshold } else { 0.5 };
    let rt = if ranker_threshold > 0.0 { ranker_threshold } else { 0.2 };
    let judge = TwoStageJudge::new(gt, rt);

    Ok(CorrectionEngine {
        judge,
        index,
        g2p,
        events_path,
    })
}

/// Process a transcript and return correction results.
///
/// `transcript_json` is a JSON object: `{ "text": "...", "app_id": "..." }`
///
/// # Safety
/// `engine` must be a valid pointer. `transcript_json` must be valid UTF-8.
#[no_mangle]
pub unsafe extern "C" fn bee_correct_process(
    engine: *mut CorrectionEngine,
    transcript_json: *const c_char,
    out_err: *mut *mut c_char,
) -> *mut CorrectionResult {
    if engine.is_null() || transcript_json.is_null() {
        set_err(out_err, "null engine or transcript");
        return std::ptr::null_mut();
    }

    let engine = unsafe { &mut *engine };
    let json_str = match unsafe { CStr::from_ptr(transcript_json) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_err(out_err, &format!("invalid transcript JSON: {e}"));
            return std::ptr::null_mut();
        }
    };

    match process_transcript(engine, json_str) {
        Ok(result) => Box::into_raw(Box::new(result)),
        Err(e) => {
            set_err(out_err, &e);
            std::ptr::null_mut()
        }
    }
}

fn process_transcript(
    engine: &mut CorrectionEngine,
    json_str: &str,
) -> Result<CorrectionResult, String> {
    let input: serde_json::Value =
        serde_json::from_str(json_str).map_err(|e| format!("parse transcript JSON: {e}"))?;
    let text = input["text"]
        .as_str()
        .ok_or("missing 'text' field")?;
    let app_id = input["app_id"].as_str().map(String::from);

    let spans = enumerate_transcript_spans_with(
        text,
        3,
        None::<&[bee_types::TranscriptAlignmentToken]>,
        |span_text| engine.g2p.ipa_tokens(span_text).ok().flatten(),
    );

    let mut span_decisions = Vec::new();
    let mut edits = Vec::new();
    let session_id = format!("{:x}", std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos());

    for span in &spans {
        let query = RetrievalQuery {
            text: span.text.clone(),
            ipa_tokens: span.ipa_tokens.clone(),
            reduced_ipa_tokens: span.reduced_ipa_tokens.clone(),
            feature_tokens: feature_tokens_for_ipa(&span.ipa_tokens),
            token_count: (span.token_end - span.token_start) as u8,
        };
        let shortlist = query_index(&engine.index, &query, 50);
        if shortlist.is_empty() {
            continue;
        }
        let scored = score_shortlist(span, &shortlist, &engine.index);
        if scored.is_empty() {
            continue;
        }

        let candidates_with_flags: Vec<_> = scored
            .iter()
            .map(|c| {
                let flags = engine.index.aliases
                    .iter()
                    .find(|a| a.alias_id == c.alias_id)
                    .map(|a| a.identifier_flags.clone())
                    .unwrap_or_default();
                (c.clone(), flags)
            })
            .collect();

        let ctx = bee_correct::judge::extract_span_context(
            text,
            span.char_start,
            span.char_end,
        );
        let ctx = SpanContext {
            app_id: app_id.clone(),
            ..ctx
        };

        let decision = engine.judge.score_span(span, &candidates_with_flags, &ctx);

        if let Some(ref chosen) = decision.chosen {
            edits.push(serde_json::json!({
                "edit_id": format!("e{}", edits.len()),
                "span_start": span.char_start,
                "span_end": span.char_end,
                "original": span.text,
                "replacement": chosen.replacement_text,
                "term": chosen.term,
                "alias_id": chosen.alias_id,
                "ranker_prob": chosen.ranker_prob,
                "gate_prob": decision.gate_prob,
            }));
        }

        span_decisions.push(SpanWithDecision {
            span: span.clone(),
            decision,
        });
    }

    // Build best text by applying edits (non-overlapping, sorted by position)
    let mut best_text = text.to_string();
    let mut offset: i64 = 0;
    for edit in &edits {
        let start = edit["span_start"].as_u64().unwrap() as usize;
        let end = edit["span_end"].as_u64().unwrap() as usize;
        let replacement = edit["replacement"].as_str().unwrap();
        let adj_start = (start as i64 + offset) as usize;
        let adj_end = (end as i64 + offset) as usize;
        best_text.replace_range(adj_start..adj_end, replacement);
        offset += replacement.len() as i64 - (end - start) as i64;
    }

    let edits_value = serde_json::json!({
        "session_id": session_id,
        "edits": edits,
    });
    let edits_json = CString::new(edits_value.to_string()).unwrap_or_default();

    Ok(CorrectionResult {
        session_id,
        best_text,
        edits_json,
        spans: span_decisions,
    })
}

/// Get the session ID from a correction result.
#[no_mangle]
pub extern "C" fn bee_correction_result_session_id(
    result: *const CorrectionResult,
) -> *mut c_char {
    if result.is_null() {
        return std::ptr::null_mut();
    }
    to_c_string(&unsafe { &*result }.session_id)
}

/// Get the corrected text.
#[no_mangle]
pub extern "C" fn bee_correction_result_best_text(
    result: *const CorrectionResult,
) -> *mut c_char {
    if result.is_null() {
        return std::ptr::null_mut();
    }
    to_c_string(&unsafe { &*result }.best_text)
}

/// Get the edits as JSON.
#[no_mangle]
pub extern "C" fn bee_correction_result_json(
    result: *const CorrectionResult,
) -> *const c_char {
    if result.is_null() {
        return std::ptr::null();
    }
    unsafe { &*result }.edits_json.as_ptr()
}

/// Get the number of edits.
#[no_mangle]
pub extern "C" fn bee_correction_result_edit_count(
    result: *const CorrectionResult,
) -> u32 {
    if result.is_null() {
        return 0;
    }
    unsafe { &*result }
        .spans
        .iter()
        .filter(|s| s.decision.chosen.is_some())
        .count() as u32
}

/// Free a correction result.
#[no_mangle]
pub extern "C" fn bee_correction_result_free(result: *mut CorrectionResult) {
    if !result.is_null() {
        unsafe { drop(Box::from_raw(result)) };
    }
}

/// Teach the judge from user feedback.
///
/// `teach_json`: `{ "edits": [{"edit_id": "e0", "resolution": "accepted"}], ... }`
///
/// # Safety
/// `engine` must be valid. `session_id` and `teach_json` must be valid UTF-8.
#[no_mangle]
pub unsafe extern "C" fn bee_correct_teach(
    engine: *mut CorrectionEngine,
    _session_id: *const c_char,
    _teach_json: *const c_char,
    _out_err: *mut *mut c_char,
) {
    if engine.is_null() {
        return;
    }
    // TODO: implement teaching from stable session IDs
    // For now this is a stub — teaching requires storing the
    // process result keyed by session_id, which needs more design.
}

/// Save judge weights and flush events.
///
/// # Safety
/// `engine` must be a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn bee_correct_save(
    _engine: *mut CorrectionEngine,
    _out_err: *mut *mut c_char,
) {
    // TODO: implement persistence (weights + memory + events)
}

/// Free a correction engine.
#[no_mangle]
pub extern "C" fn bee_correct_engine_free(engine: *mut CorrectionEngine) {
    if !engine.is_null() {
        unsafe { drop(Box::from_raw(engine)) };
    }
}
