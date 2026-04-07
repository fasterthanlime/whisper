use bee_rpc::{RepoDownload, RepoFile};

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

pub(crate) fn required_downloads() -> Vec<RepoDownload> {
    vec![
        RepoDownload {
            repo_id: "mlx-community/Qwen3-ASR-1.7B-4bit".into(),
            local_dir: "mlx-community--Qwen3-ASR-1.7B-4bit".into(),
            files: vec![
                RepoFile {
                    name: "config.json".into(),
                    url: hf_file_url("mlx-community/Qwen3-ASR-1.7B-4bit", "config.json"),
                    size: 0,
                },
                RepoFile {
                    name: "tokenizer.json".into(),
                    url: hf_file_url("mlx-community/Qwen3-ASR-1.7B-4bit", "tokenizer.json"),
                    size: 0,
                },
                RepoFile {
                    name: "model.safetensors".into(),
                    url: hf_file_url("mlx-community/Qwen3-ASR-1.7B-4bit", "model.safetensors"),
                    size: 0,
                },
                RepoFile {
                    name: "generation_config.json".into(),
                    url: hf_file_url(
                        "mlx-community/Qwen3-ASR-1.7B-4bit",
                        "generation_config.json",
                    ),
                    size: 0,
                },
                RepoFile {
                    name: "preprocessor_config.json".into(),
                    url: hf_file_url(
                        "mlx-community/Qwen3-ASR-1.7B-4bit",
                        "preprocessor_config.json",
                    ),
                    size: 0,
                },
            ],
        },
        RepoDownload {
            repo_id: "mlx-community/Qwen3-ForcedAligner-0.6B-4bit".into(),
            local_dir: "mlx-community--Qwen3-ForcedAligner-0.6B-4bit".into(),
            files: vec![
                RepoFile {
                    name: "config.json".into(),
                    url: hf_file_url("mlx-community/Qwen3-ForcedAligner-0.6B-4bit", "config.json"),
                    size: 0,
                },
                RepoFile {
                    name: "model.safetensors".into(),
                    url: hf_file_url(
                        "mlx-community/Qwen3-ForcedAligner-0.6B-4bit",
                        "model.safetensors",
                    ),
                    size: 0,
                },
                RepoFile {
                    name: "tokenizer.json".into(),
                    url: hf_file_url(
                        "mlx-community/Qwen3-ForcedAligner-0.6B-4bit",
                        "tokenizer.json",
                    ),
                    size: 0,
                },
            ],
        },
        RepoDownload {
            repo_id: "aitytech/Silero-VAD-v5-MLX".into(),
            local_dir: "aitytech--Silero-VAD-v5-MLX".into(),
            files: vec![RepoFile {
                name: "model.safetensors".into(),
                url: hf_file_url("aitytech/Silero-VAD-v5-MLX", "model.safetensors"),
                size: 0,
            }],
        },
    ]
}
