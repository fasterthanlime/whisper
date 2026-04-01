//! Weight loading for pre-quantized MLX community models.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use mlx_rs::error::Exception;
use mlx_rs::module::ModuleParameters;
use mlx_rs::nn::{QuantizedEmbedding, QuantizedLinear};
use mlx_rs::quantization::MaybeQuantized;
use mlx_rs::Array;

use crate::model::Qwen3ASRModel;

/// Load weights from safetensors files, handling both dense and pre-quantized formats.
///
/// For pre-quantized MLX models (e.g. mlx-community/Qwen3-ASR-1.7B-4bit),
/// layers with .scales/.biases keys are loaded as QuantizedLinear directly
/// from the pre-quantized data — no re-quantization needed.
pub fn load_weights(
    model: &mut Qwen3ASRModel,
    model_dir: &Path,
) -> Result<LoadStats, Exception> {
    // Find and load all safetensors files
    let mut st_files: Vec<_> = std::fs::read_dir(model_dir)
        .map_err(|e| Exception::custom(format!("read dir: {e}")))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "safetensors"))
        .map(|e| e.path())
        .collect();
    st_files.sort();

    let mut all_weights = HashMap::new();
    for f in &st_files {
        let tensors = Array::load_safetensors(f)
            .map_err(|e| Exception::custom(format!("load safetensors: {e}")))?;
        all_weights.extend(tensors);
    }

    // Remap keys: strip thinker. prefix, transpose conv2d weights
    let mut weights = HashMap::new();
    for (key, value) in all_weights {
        let mut new_key = key.clone();
        let had_thinker = new_key.starts_with("thinker.");
        if had_thinker {
            new_key = new_key["thinker.".len()..].to_string();
        }
        let value = if had_thinker
            && new_key.contains("conv2d")
            && new_key.ends_with(".weight")
            && value.ndim() == 4
        {
            value.transpose_axes(&[0, 2, 3, 1])?
        } else {
            value
        };
        weights.insert(new_key, value);
    }

    // Identify quantized layer prefixes (those with .scales keys)
    let quantized_prefixes: HashSet<String> = weights
        .keys()
        .filter(|k| k.ends_with(".scales"))
        .map(|k| k.strip_suffix(".scales").unwrap().to_string())
        .collect();

    // Detect quantization params from config.json
    let (group_size, bits) = if !quantized_prefixes.is_empty() {
        let config_path = model_dir.join("config.json");
        if let Ok(config_str) = std::fs::read_to_string(&config_path) {
            if let Ok(config) = serde_json::from_str::<serde_json::Value>(&config_str) {
                let gs = config.get("quantization")
                    .and_then(|q| q.get("group_size"))
                    .and_then(|v| v.as_i64())
                    .unwrap_or(64) as i32;
                let b = config.get("quantization")
                    .and_then(|q| q.get("bits"))
                    .and_then(|v| v.as_i64())
                    .unwrap_or(4) as i32;
                (gs, b)
            } else {
                (64, 4)
            }
        } else {
            (64, 4)
        }
    } else {
        (64, 4)
    };

    // Load dense parameters first
    let mut params = model.parameters_mut().flatten();
    let mut loaded = 0usize;
    let mut skipped = Vec::new();

    for (key, value) in &weights {
        if key.ends_with(".scales") || key.ends_with(".biases") {
            continue;
        }
        let prefix = if key.ends_with(".weight") {
            key.strip_suffix(".weight").unwrap()
        } else if key.ends_with(".bias") {
            key.strip_suffix(".bias").unwrap()
        } else {
            key.as_str()
        };

        if quantized_prefixes.contains(prefix) {
            continue;
        }

        if let Some(param) = params.get_mut(&**key) {
            **param = value.clone();
            loaded += 1;
        } else {
            skipped.push(key.clone());
        }
    }
    drop(params);

    // Now handle quantized layers by constructing QuantizedLinear/QuantizedEmbedding
    // directly from the pre-quantized data
    for prefix in &quantized_prefixes {
        let weight = weights.get(&format!("{prefix}.weight"));
        let scales = weights.get(&format!("{prefix}.scales"));
        let biases_q = weights.get(&format!("{prefix}.biases"));
        let bias = weights.get(&format!("{prefix}.bias"));

        let (Some(weight), Some(scales), Some(biases_q)) = (weight, scales, biases_q) else {
            skipped.push(format!("{prefix}.*"));
            continue;
        };

        let is_embedding = prefix.ends_with("embed_tokens");

        if is_embedding {
            let qe = QuantizedEmbedding::from_parts(
                weight.clone(),
                scales.clone(),
                biases_q.clone(),
                group_size,
                bits,
            );
            set_quantized_embedding(model, prefix, qe);
            loaded += 3;
        } else {
            let ql = QuantizedLinear::from_parts(
                weight.clone(),
                bias.cloned(),
                scales.clone(),
                biases_q.clone(),
                group_size,
                bits,
            );
            set_quantized_linear(model, prefix, ql);
            loaded += if bias.is_some() { 4 } else { 3 };
        }
    }

    // Handle weight-tied lm_head: if no lm_head weights in file, tie to embed_tokens
    if !weights.contains_key("lm_head.weight") {
        if let (Some(w), Some(s), Some(b)) = (
            weights.get("model.embed_tokens.weight"),
            weights.get("model.embed_tokens.scales"),
            weights.get("model.embed_tokens.biases"),
        ) {
            let ql = QuantizedLinear::from_parts(
                w.clone(), None, s.clone(), b.clone(), group_size, bits,
            );
            model.lm_head.set_quantized(ql);
            log::info!("lm_head weight-tied to embed_tokens (quantized)");
        } else if let Some(w) = weights.get("model.embed_tokens.weight") {
            // Dense embed_tokens — just copy weight into lm_head
            let mut params = model.parameters_mut().flatten();
            if let Some(param) = params.get_mut("lm_head.weight") {
                **param = w.clone();
                log::info!("lm_head weight-tied to embed_tokens (dense)");
            }
        }
    }

    Ok(LoadStats {
        total_keys: weights.len(),
        loaded,
        skipped: skipped.len(),
        quantized_layers: quantized_prefixes.len(),
        group_size,
        bits,
    })
}

pub struct LoadStats {
    pub total_keys: usize,
    pub loaded: usize,
    pub skipped: usize,
    pub quantized_layers: usize,
    pub group_size: i32,
    pub bits: i32,
}

/// Resolve a dotted prefix to the corresponding MaybeQuantized<nn::Linear> field.
fn resolve_linear_mut<'a>(
    model: &'a mut Qwen3ASRModel,
    prefix: &str,
) -> Option<&'a mut MaybeQuantized<mlx_rs::nn::Linear>> {
    let parts: Vec<&str> = prefix.split('.').collect();

    if parts.first() == Some(&"audio_tower") {
        if let Some(&"layers") = parts.get(1) {
            let idx = parts.get(2).and_then(|s| s.parse::<usize>().ok())?;
            let layer = model.audio_tower.layers.get_mut(idx)?;
            match parts.get(3).copied()? {
                "self_attn" => match parts.get(4).copied()? {
                    "q_proj" => return Some(&mut layer.self_attn.q_proj),
                    "k_proj" => return Some(&mut layer.self_attn.k_proj),
                    "v_proj" => return Some(&mut layer.self_attn.v_proj),
                    "out_proj" => return Some(&mut layer.self_attn.out_proj),
                    _ => return None,
                },
                "fc1" => return Some(&mut layer.fc1),
                "fc2" => return Some(&mut layer.fc2),
                _ => return None,
            }
        } else {
            return match parts.get(1).copied()? {
                "conv_out" => Some(&mut model.audio_tower.conv_out),
                "proj1" => Some(&mut model.audio_tower.proj1),
                "proj2" => Some(&mut model.audio_tower.proj2),
                _ => None,
            };
        }
    }

    if parts.first() == Some(&"model") {
        if let Some(&"layers") = parts.get(1) {
            let idx = parts.get(2).and_then(|s| s.parse::<usize>().ok())?;
            let layer = model.model.layers.get_mut(idx)?;
            match parts.get(3).copied()? {
                "self_attn" => match parts.get(4).copied()? {
                    "q_proj" => return Some(&mut layer.self_attn.q_proj),
                    "k_proj" => return Some(&mut layer.self_attn.k_proj),
                    "v_proj" => return Some(&mut layer.self_attn.v_proj),
                    "o_proj" => return Some(&mut layer.self_attn.o_proj),
                    _ => return None,
                },
                "mlp" => match parts.get(4).copied()? {
                    "gate_proj" => return Some(&mut layer.mlp.gate_proj),
                    "up_proj" => return Some(&mut layer.mlp.up_proj),
                    "down_proj" => return Some(&mut layer.mlp.down_proj),
                    _ => return None,
                },
                _ => return None,
            }
        }
    }

    if prefix == "lm_head" {
        return Some(&mut model.lm_head);
    }

    None
}

fn set_quantized_linear(model: &mut Qwen3ASRModel, prefix: &str, ql: QuantizedLinear) {
    if let Some(field) = resolve_linear_mut(model, prefix) {
        field.set_quantized(ql);
    } else {
        log::warn!("Unknown quantized linear path: {prefix}");
    }
}

fn set_quantized_embedding(model: &mut Qwen3ASRModel, prefix: &str, qe: QuantizedEmbedding) {
    if prefix == "model.embed_tokens" {
        model.model.embed_tokens.set_quantized(qe);
    } else {
        log::warn!("Unknown quantized embedding path: {prefix}");
    }
}
