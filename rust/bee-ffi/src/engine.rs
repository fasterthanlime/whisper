//! ASR engine loading and model download manifest.
//!
//! # Environment variables
//!
//! All optional — used for development/testing to override default paths:
//!
//! | Variable | Purpose | Default |
//! |----------|---------|---------|
//! | `BEE_VAD_DIR` | Silero VAD model directory | `{cache}/aitytech--Silero-VAD-v5-MLX` |
//! | `BEE_TOKENIZER_PATH` | Path to `tokenizer.json` | `{model_dir}/tokenizer.json` |
//! | `BEE_ALIGNER_DIR` | Forced-aligner model directory | First match in `{cache}/mlx-community--Qwen3-ForcedAligner-*` |
//! | `BEE_FFI_LOG_PATH` | Log file path (set by Swift before dlopen) | `/tmp/bee.log` |

use std::path::{Path, PathBuf};

use bee_rpc::{RepoDownload, RepoFile};
use bee_transcribe::{Engine, EngineConfig};

use crate::stats::StatsSampler;

const HF_BASE: &str = "https://huggingface.co";

fn hf_file_url(repo_id: &str, filename: &str) -> String {
    format!("{HF_BASE}/{repo_id}/resolve/main/{filename}")
}

pub(crate) struct AsrEngine {
    /// Leaked via `Box::leak` — lives for process lifetime. Gives us a genuine
    /// `&'static Engine` so sessions can borrow it without transmute.
    pub(crate) inner: &'static Engine,
    /// Pre-loaded VAD tensors (loaded once, cloned per session).
    pub(crate) vad_tensors: Option<std::collections::HashMap<String, mlx_rs::Array>>,
    pub(crate) stats: StatsSampler,
}

// SAFETY: Engine is immutable after construction. MLX arrays are heap-allocated
// Metal buffers; concurrent read access is safe.
unsafe impl Send for AsrEngine {}
unsafe impl Sync for AsrEngine {}

/// Locate the Silero VAD model directory.
///
/// Checks `BEE_VAD_DIR` env var first, then falls back to
/// `{cache_base}/aitytech--Silero-VAD-v5-MLX`. Returns `None` if neither exists
/// (VAD is optional — sessions work without it, just no silence detection).
pub(crate) fn find_vad_dir(cache_base: &Path) -> Option<PathBuf> {
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

/// Resolve paths for the ASR engine, checking env var overrides first.
///
/// Environment variables (all optional, for development/testing):
/// - `BEE_TOKENIZER_PATH`: path to `tokenizer.json`. Default: `{model_dir}/tokenizer.json`.
/// - `BEE_ALIGNER_DIR`: path to the forced-aligner model directory.
///   Default: first existing candidate in `{cache_base}/mlx-community--Qwen3-ForcedAligner-*`.
///
/// Paths are `Box::leak`ed to `'static` since the engine lives for the process lifetime.
fn resolve_engine_config(
    model_dir: &Path,
    cache_base: &Path,
) -> Result<EngineConfig<'static>, String> {
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

pub(crate) fn load_engine(model_dir: &Path, cache_base: &Path) -> Result<AsrEngine, String> {
    // Cap MLX's Metal buffer cache at 2GB to prevent unbounded memory growth
    bee_transcribe::set_mlx_cache_limit(2 * 1024 * 1024 * 1024)
        .map_err(|e| format!("Failed to set MLX cache limit: {e}"))?;

    let config = resolve_engine_config(model_dir, cache_base)?;
    let engine = Engine::load(&config).map_err(|e| format!("load engine: {e}"))?;

    let vad_tensors = find_vad_dir(cache_base).and_then(|d| {
        let st_path = d.join("model.safetensors");
        match mlx_rs::Array::load_safetensors(&st_path) {
            Ok(tensors) => {
                tracing::info!("Silero VAD loaded ({} tensors)", tensors.len());
                Some(tensors)
            }
            Err(e) => {
                tracing::warn!("Failed to load VAD: {e}");
                None
            }
        }
    });

    Ok(AsrEngine {
        inner: Box::leak(Box::new(engine)),
        vad_tensors,
        stats: StatsSampler::new(),
    })
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
                },
                RepoFile {
                    name: "tokenizer.json".into(),
                    url: hf_file_url("mlx-community/Qwen3-ASR-1.7B-4bit", "tokenizer.json"),

                },
                RepoFile {
                    name: "model.safetensors".into(),
                    url: hf_file_url("mlx-community/Qwen3-ASR-1.7B-4bit", "model.safetensors"),

                },
                RepoFile {
                    name: "generation_config.json".into(),
                    url: hf_file_url(
                        "mlx-community/Qwen3-ASR-1.7B-4bit",
                        "generation_config.json",
                    ),

                },
                RepoFile {
                    name: "preprocessor_config.json".into(),
                    url: hf_file_url(
                        "mlx-community/Qwen3-ASR-1.7B-4bit",
                        "preprocessor_config.json",
                    ),

                },
            ],
        },
        RepoDownload {
            repo_id: "mlx-community/Qwen3-ForcedAligner-0.6B-4bit".into(),
            local_dir: "mlx-community--Qwen3-ForcedAligner-0.6B-4bit".into(),
            files: vec![
                RepoFile {
                    name: "config.json".into(),
                    url: hf_file_url(
                        "mlx-community/Qwen3-ForcedAligner-0.6B-4bit",
                        "config.json",
                    ),

                },
                RepoFile {
                    name: "model.safetensors".into(),
                    url: hf_file_url(
                        "mlx-community/Qwen3-ForcedAligner-0.6B-4bit",
                        "model.safetensors",
                    ),

                },
                RepoFile {
                    name: "tokenizer.json".into(),
                    url: hf_file_url(
                        "mlx-community/Qwen3-ForcedAligner-0.6B-4bit",
                        "tokenizer.json",
                    ),

                },
            ],
        },
        RepoDownload {
            repo_id: "aitytech/Silero-VAD-v5-MLX".into(),
            local_dir: "aitytech--Silero-VAD-v5-MLX".into(),
            files: vec![RepoFile {
                name: "model.safetensors".into(),
                url: hf_file_url("aitytech/Silero-VAD-v5-MLX", "model.safetensors"),

            }],
        },
    ]
}
