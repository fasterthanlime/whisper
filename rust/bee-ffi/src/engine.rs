//! ASR engine loading and model download manifest.
//!
//! All paths are derived from the sandbox cache directory passed by Swift.
//! Env var overrides (`BEE_VAD_DIR`, `BEE_ALIGNER_DIR`, etc.) are for CLI
//! tools only — the dylib must not read them (sandbox blocks outside paths).

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
    pub(crate) stats: StatsSampler,
}

// SAFETY: Engine is immutable after construction. MLX arrays are heap-allocated
// Metal buffers; concurrent read access is safe.
unsafe impl Send for AsrEngine {}
unsafe impl Sync for AsrEngine {}

/// Locate the Silero VAD model directory within the cache.
///
/// Returns `None` if not found (VAD is optional — sessions work without it,
/// just no silence detection).
pub(crate) fn find_vad_dir(cache_base: &Path) -> Option<PathBuf> {
    let dir = cache_base.join("aitytech--Silero-VAD-v5-MLX");
    if dir.exists() { Some(dir) } else { None }
}

/// Resolve paths for the ASR engine from the sandbox cache directory.
///
/// Paths are `Box::leak`ed to `'static` since the engine lives for the process lifetime.
///
/// Note: env var overrides (`BEE_TOKENIZER_DIR`, `BEE_ALIGNER_DIR`, etc.) are handled
/// by CLI tools (beeml, transcribe) directly — the dylib uses only sandbox-safe paths.
fn resolve_engine_config(
    model_dir: &Path,
    cache_base: &Path,
) -> Result<EngineConfig<'static>, String> {
    let tokenizer_dir = model_dir.to_path_buf();

    let aligner_dir: PathBuf = {
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

    let silero_dir: PathBuf = find_vad_dir(cache_base).ok_or("Silero VAD not found")?;

    // Leak the PathBufs to get 'static references (engine lives for process lifetime)
    let model_dir: &'static Path = Box::leak(model_dir.to_path_buf().into_boxed_path());
    let tokenizer_dir: &'static Path = Box::leak(tokenizer_dir.into_boxed_path());
    let aligner_dir: &'static Path = Box::leak(aligner_dir.into_boxed_path());
    let silero_dir: &'static Path = Box::leak(silero_dir.into_boxed_path());

    // Look for the correction dataset directory
    let correction_dir_path = cache_base.join("bee-correct-dataset");
    let correction_dir: Option<&'static Path> = if correction_dir_path.exists() {
        Some(Box::leak(correction_dir_path.into_boxed_path()))
    } else {
        None
    };

    // Events path for online learning (next to correction dataset)
    let correction_events_path = correction_dir.map(|d| d.join("events.jsonl"));

    Ok(EngineConfig {
        model_dir,
        tokenizer_dir,
        aligner_dir,
        share_aligner_audio_tower: false,
        silero_dir,
        correction_dir,
        correction_events_path,
    })
}

pub(crate) fn load_engine(model_dir: &Path, cache_base: &Path) -> Result<AsrEngine, String> {
    // Cap MLX's Metal buffer cache at 2GB to prevent unbounded memory growth
    bee_transcribe::set_mlx_cache_limit(2 * 1024 * 1024 * 1024)
        .map_err(|e| format!("Failed to set MLX cache limit: {e}"))?;

    let config = resolve_engine_config(model_dir, cache_base)?;
    tracing::info!(
        "Engine config: model={}, tokenizer={}, aligner={}",
        config.model_dir.display(),
        config.tokenizer_dir.display(),
        config.aligner_dir.display(),
    );
    let engine = Engine::load(&config).map_err(|e| format!("load engine: {e}"))?;

    Ok(AsrEngine {
        inner: Box::leak(Box::new(engine)),
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
                    name: "vocab.json".into(),
                    url: hf_file_url("mlx-community/Qwen3-ASR-1.7B-4bit", "vocab.json"),
                },
                RepoFile {
                    name: "merges.txt".into(),
                    url: hf_file_url("mlx-community/Qwen3-ASR-1.7B-4bit", "merges.txt"),
                },
                RepoFile {
                    name: "tokenizer_config.json".into(),
                    url: hf_file_url("mlx-community/Qwen3-ASR-1.7B-4bit", "tokenizer_config.json"),
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
                    url: hf_file_url("mlx-community/Qwen3-ForcedAligner-0.6B-4bit", "config.json"),
                },
                RepoFile {
                    name: "model.safetensors".into(),
                    url: hf_file_url(
                        "mlx-community/Qwen3-ForcedAligner-0.6B-4bit",
                        "model.safetensors",
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
