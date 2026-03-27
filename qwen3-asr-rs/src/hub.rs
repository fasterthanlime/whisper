use anyhow::Context;
use log::info;
use std::path::Path;

pub(crate) fn hf_url(model_id: &str, filename: &str) -> String {
    format!(
        "https://huggingface.co/{}/resolve/main/{}",
        model_id, filename
    )
}

/// Make a GET request; returns `None` on 404, error on other failures.
pub(crate) fn hf_try_get(url: &str) -> anyhow::Result<Option<reqwest::blocking::Response>> {
    let client = reqwest::blocking::Client::builder().timeout(None).build()?;
    let mut b = client.get(url);
    if let Ok(tok) = std::env::var("HUGGING_FACE_HUB_TOKEN") {
        b = b.header("Authorization", format!("Bearer {}", tok));
    }
    let resp = b.send()?;
    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(None);
    }
    if !resp.status().is_success() {
        anyhow::bail!("HTTP {} for {}", resp.status(), url);
    }
    Ok(Some(resp))
}

/// GET a URL and return the full body as bytes.
pub(crate) fn hf_get_bytes(url: &str) -> anyhow::Result<Vec<u8>> {
    hf_try_get(url)?
        .ok_or_else(|| anyhow::anyhow!("404: {}", url))
        .and_then(|r| Ok(r.bytes()?.to_vec()))
}

/// Stream a URL to a file, printing progress to stderr.
pub(crate) fn hf_stream_to_file(url: &str, path: &std::path::Path) -> anyhow::Result<()> {
    use std::io::{Read, Write};
    info!("Downloading {}", url);
    let client = reqwest::blocking::Client::builder().timeout(None).build()?;
    let mut b = client.get(url);
    if let Ok(tok) = std::env::var("HUGGING_FACE_HUB_TOKEN") {
        b = b.header("Authorization", format!("Bearer {}", tok));
    }
    let mut resp = b.send()?;
    if !resp.status().is_success() {
        anyhow::bail!("HTTP {} for {}", resp.status(), url);
    }
    let mut file = std::fs::File::create(path)?;
    let mut downloaded = 0u64;
    let mut buf = [0u8; 65536];
    loop {
        let n = resp.read(&mut buf)?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        downloaded += n as u64;
    }
    info!("Downloaded {:.1} MB", downloaded as f64 / 1_048_576.0);
    Ok(())
}

/// Ensure model files for `model_id` exist under `cache_dir` and return the
/// model directory path.  Downloads from HuggingFace only when needed.
///
/// Cache layout: `{cache_dir}/{model_id.replace('/', '--')}/`
/// A `.complete` marker file signals that all files are present. If the
/// directory exists but `.complete` is missing (interrupted download), the
/// directory is removed and the download restarts.
pub(crate) fn ensure_model_cached(
    model_id: &str,
    cache_dir: &Path,
) -> anyhow::Result<std::path::PathBuf> {
    let sanitized = model_id.replace('/', "--");
    let model_dir = cache_dir.join(&sanitized);
    let marker = model_dir.join(".complete");

    // Fast path: already downloaded.
    if marker.exists() {
        info!("Using cached model at {}", model_dir.display());
        return Ok(model_dir);
    }

    // Partial / interrupted download — remove and restart.
    if model_dir.exists() {
        info!("Removing incomplete download at {}", model_dir.display());
        std::fs::remove_dir_all(&model_dir)?;
    }

    info!(
        "Downloading '{}' from HuggingFace to {}…",
        model_id,
        model_dir.display()
    );
    std::fs::create_dir_all(&model_dir)?;

    // config.json
    let config_bytes =
        hf_get_bytes(&hf_url(model_id, "config.json")).context("download config.json")?;
    std::fs::write(model_dir.join("config.json"), &config_bytes)?;

    // Weights: check for sharded index first.
    if let Some(resp) = hf_try_get(&hf_url(model_id, "model.safetensors.index.json"))? {
        let index_text = resp.text()?;
        std::fs::write(model_dir.join("model.safetensors.index.json"), &index_text)?;

        let index: serde_json::Value = serde_json::from_str(&index_text)?;
        let weight_map = index["weight_map"]
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("invalid model.safetensors.index.json"))?;
        let shards: std::collections::HashSet<String> = weight_map
            .values()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();
        for shard in &shards {
            hf_stream_to_file(&hf_url(model_id, shard), &model_dir.join(shard))
                .with_context(|| format!("download shard {}", shard))?;
        }
    } else {
        hf_stream_to_file(
            &hf_url(model_id, "model.safetensors"),
            &model_dir.join("model.safetensors"),
        )
        .context("download model.safetensors")?;
    }

    // Tokenizer: Qwen3-ASR ships tokenizer_config.json (with added_tokens_decoder)
    // but not tokenizer.json. Reconstruct from vocab.json + merges.txt + config.
    let tok_config = String::from_utf8(
        hf_get_bytes(&hf_url(model_id, "tokenizer_config.json"))
            .context("download tokenizer_config.json")?,
    )?;
    let vocab = String::from_utf8(
        hf_get_bytes(&hf_url(model_id, "vocab.json")).context("download vocab.json")?,
    )?;
    let merges = String::from_utf8(
        hf_get_bytes(&hf_url(model_id, "merges.txt")).context("download merges.txt")?,
    )?;
    let tok_json = build_qwen3_tokenizer_json(&vocab, &merges, &tok_config)?;
    std::fs::write(model_dir.join("tokenizer.json"), tok_json)?;

    // Mark download as complete.
    std::fs::write(&marker, b"")?;
    info!("Download complete, cached at {}", model_dir.display());

    Ok(model_dir)
}

/// Ensure a GGUF model is cached. Downloads config + tokenizer from
/// `base_repo_id` (e.g. "Qwen/Qwen3-ASR-1.7B") and the quantized weights
/// from `gguf_repo_id` (e.g. "Alkd/qwen3-asr-gguf").
///
/// Cache layout: `{cache_dir}/{gguf_repo_id}--{gguf_filename}/`
pub(crate) fn ensure_gguf_model_cached(
    base_repo_id: &str,
    gguf_repo_id: &str,
    gguf_filename: &str,
    cache_dir: &Path,
) -> anyhow::Result<std::path::PathBuf> {
    let sanitized = format!(
        "{}--{}",
        gguf_repo_id.replace('/', "--"),
        gguf_filename.replace('.', "_")
    );
    let model_dir = cache_dir.join(&sanitized);
    let marker = model_dir.join(".complete");

    if marker.exists() {
        info!("Using cached GGUF model at {}", model_dir.display());
        return Ok(model_dir);
    }

    if model_dir.exists() {
        info!(
            "Removing incomplete GGUF download at {}",
            model_dir.display()
        );
        std::fs::remove_dir_all(&model_dir)?;
    }

    info!(
        "Downloading GGUF model: {} from {}, config from {}",
        gguf_filename, gguf_repo_id, base_repo_id
    );
    std::fs::create_dir_all(&model_dir)?;

    // config.json from the base (non-quantized) repo
    let config_bytes = hf_get_bytes(&hf_url(base_repo_id, "config.json"))
        .context("download config.json from base repo")?;
    std::fs::write(model_dir.join("config.json"), &config_bytes)?;

    // GGUF weights from the quantized repo
    hf_stream_to_file(
        &hf_url(gguf_repo_id, gguf_filename),
        &model_dir.join(gguf_filename),
    )
    .with_context(|| format!("download {gguf_filename}"))?;

    // Tokenizer from the base repo
    let tok_config = String::from_utf8(
        hf_get_bytes(&hf_url(base_repo_id, "tokenizer_config.json"))
            .context("download tokenizer_config.json")?,
    )?;
    let vocab = String::from_utf8(
        hf_get_bytes(&hf_url(base_repo_id, "vocab.json")).context("download vocab.json")?,
    )?;
    let merges = String::from_utf8(
        hf_get_bytes(&hf_url(base_repo_id, "merges.txt")).context("download merges.txt")?,
    )?;
    let tok_json = build_qwen3_tokenizer_json(&vocab, &merges, &tok_config)?;
    std::fs::write(model_dir.join("tokenizer.json"), tok_json)?;

    std::fs::write(&marker, b"")?;
    info!("GGUF download complete, cached at {}", model_dir.display());

    Ok(model_dir)
}

pub(crate) use crate::tokenizer_build::build_qwen3_tokenizer_json;
