use std::collections::HashMap;
use std::path::Path;

use mlx_rs::Array;
use mlx_rs::module::ModuleParameters;

use crate::model::T5ForConditionalGeneration;

pub struct LoadStats {
    pub loaded: usize,
    pub missing: Vec<String>,
    pub unexpected: Vec<String>,
}

/// Load safetensors weights into the T5 model.
///
/// HuggingFace T5 weight keys follow this pattern:
/// - `shared.weight`
/// - `encoder.block.{i}.layer.0.SelfAttention.{q,k,v,o}.weight`
/// - `encoder.block.{i}.layer.0.layer_norm.weight`
/// - `encoder.block.{i}.layer.1.DenseReluDense.wi_0.weight`
/// - `encoder.block.{i}.layer.1.DenseReluDense.wi_1.weight`
/// - `encoder.block.{i}.layer.1.DenseReluDense.wo.weight`
/// - `encoder.block.{i}.layer.1.layer_norm.weight`
/// - `encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight`
/// - `encoder.final_layer_norm.weight`
/// - `decoder.block.{i}.layer.0.SelfAttention.{q,k,v,o}.weight`
/// - `decoder.block.{i}.layer.1.EncDecAttention.{q,k,v,o}.weight`
/// - `decoder.block.{i}.layer.2.DenseReluDense.{wi_0,wi_1,wo}.weight`
/// - `decoder.block.{i}.layer.{0,1,2}.layer_norm.weight`
/// - `decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight`
/// - `decoder.final_layer_norm.weight`
/// - `lm_head.weight`
///
/// Our flattened parameter keys use dotted paths from the struct hierarchy.
pub fn load_weights(
    model: &mut T5ForConditionalGeneration,
    model_dir: &Path,
) -> Result<LoadStats, anyhow::Error> {
    let safetensors_path = model_dir.join("model.safetensors");
    let tensors = Array::load_safetensors(&safetensors_path)?;

    let key_map = build_key_map(model);
    let mut params = model.parameters_mut().flatten();

    let mut loaded = 0usize;
    let mut missing = Vec::new();
    let mut unexpected = Vec::new();

    for (hf_key, tensor) in &tensors {
        if let Some(our_key) = key_map.get(hf_key.as_str()) {
            if let Some(param) = params.get_mut(our_key.as_str()) {
                **param = tensor.clone();
                loaded += 1;
            } else {
                unexpected.push(hf_key.clone());
            }
        } else {
            unexpected.push(hf_key.clone());
        }
    }

    // Check for params we didn't load
    for (our_key, hf_key) in &key_map {
        if !tensors.contains_key(*our_key) && !tensors.contains_key(hf_key.as_str()) {
            // Check if tensor was actually loaded via the mapping
            if params.get(hf_key.as_str()).is_some() {
                // Already counted
            } else {
                missing.push(our_key.to_string());
            }
        }
    }

    // Eval all parameters after loading
    let all_params: Vec<Array> = params.values().map(|p| (**p).clone()).collect();
    if !all_params.is_empty() {
        mlx_rs::transforms::eval(all_params.iter())?;
    }

    Ok(LoadStats {
        loaded,
        missing,
        unexpected,
    })
}

fn build_key_map(model: &T5ForConditionalGeneration) -> HashMap<&'static str, String> {
    // We need to map HuggingFace keys -> our flattened parameter keys.
    // Our struct hierarchy produces keys like:
    //   shared.weight.weight  (Param<Embedding> -> weight field of Embedding)
    //   encoder.embed_tokens.weight.weight
    //   encoder.blocks.0.self_attn.q.weight
    //   etc.
    //
    // But it's easier to just build the map programmatically.
    let _ = model;

    // Actually, let's take a different approach: dump our parameter keys and the
    // HF keys, then build the mapping. For now, let's use a direct loading approach
    // where we iterate HF keys and manually assign.
    HashMap::new()
}

/// Load weights by directly iterating HuggingFace keys and assigning to model parameters.
pub fn load_weights_direct(
    model: &mut T5ForConditionalGeneration,
    model_dir: &Path,
) -> Result<LoadStats, anyhow::Error> {
    let safetensors_path = model_dir.join("model.safetensors");
    let tensors = Array::load_safetensors(&safetensors_path)?;

    // First, let's see what our parameter names look like
    let mut params = model.parameters_mut().flatten();

    // Print all our param keys for debugging
    let mut our_keys: Vec<_> = params.keys().map(|k| k.to_string()).collect();
    our_keys.sort();

    let mut hf_keys: Vec<_> = tensors.keys().map(|k| k.to_string()).collect();
    hf_keys.sort();

    // Build the mapping from HF key -> our key
    let mapping = build_hf_to_our_mapping(&our_keys, &hf_keys);

    let mut loaded = 0usize;
    let mut missing = Vec::new();
    let mut unexpected = Vec::new();

    for (hf_key, our_key) in &mapping {
        if let Some(tensor) = tensors.get(hf_key.as_str()) {
            if let Some(param) = params.get_mut(our_key.as_str()) {
                **param = tensor.clone();
                loaded += 1;
            } else {
                eprintln!("WARNING: mapped {hf_key} -> {our_key} but param not found");
                missing.push(hf_key.clone());
            }
        }
    }

    // Find unmapped HF keys
    for hf_key in &hf_keys {
        if !mapping.contains_key(hf_key) {
            unexpected.push(hf_key.clone());
        }
    }

    // Eval all loaded parameters
    let all_arrays: Vec<Array> = params.values().map(|p| (**p).clone()).collect();
    if !all_arrays.is_empty() {
        mlx_rs::transforms::eval(all_arrays.iter())?;
    }

    Ok(LoadStats {
        loaded,
        missing,
        unexpected,
    })
}

fn build_hf_to_our_mapping(our_keys: &[String], hf_keys: &[String]) -> HashMap<String, String> {
    let mut mapping = HashMap::new();

    for hf_key in hf_keys {
        if let Some(our_key) = hf_key_to_our_key(hf_key) {
            // Verify it exists in our params
            if our_keys.iter().any(|k| k == &our_key) {
                mapping.insert(hf_key.clone(), our_key);
            } else {
                eprintln!("WARNING: computed mapping {hf_key} -> {our_key} but key not in params");
            }
        }
    }

    mapping
}

fn hf_key_to_our_key(hf_key: &str) -> Option<String> {
    // shared.weight -> shared.weight.weight
    if hf_key == "shared.weight" {
        return Some("shared.weight".to_string());
    }

    // lm_head.weight -> lm_head.weight
    if hf_key == "lm_head.weight" {
        return Some("lm_head.weight".to_string());
    }

    // encoder.final_layer_norm.weight -> encoder.final_layer_norm.weight
    if hf_key == "encoder.final_layer_norm.weight" {
        return Some("encoder.final_layer_norm.weight".to_string());
    }
    if hf_key == "decoder.final_layer_norm.weight" {
        return Some("decoder.final_layer_norm.weight".to_string());
    }

    // encoder.embed_tokens.weight -> encoder.embed_tokens.weight
    if hf_key == "encoder.embed_tokens.weight" {
        return Some("encoder.embed_tokens.weight".to_string());
    }
    if hf_key == "decoder.embed_tokens.weight" {
        return Some("decoder.embed_tokens.weight".to_string());
    }

    // encoder.block.{i}.layer.0.SelfAttention.relative_attention_bias.weight
    // -> encoder.position_bias.relative_attention_bias.weight
    if hf_key == "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight" {
        return Some("encoder.position_bias.relative_attention_bias.weight".to_string());
    }
    if hf_key == "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight" {
        return Some("decoder.position_bias.relative_attention_bias.weight".to_string());
    }

    // encoder.block.{i}.layer.0.SelfAttention.{q,k,v,o}.weight
    // -> encoder.blocks.{i}.self_attn.{q,k,v,o}.weight
    if let Some(rest) = hf_key.strip_prefix("encoder.block.") {
        return map_encoder_block(rest);
    }

    // decoder.block.{i}.layer.{0,1,2}.*
    if let Some(rest) = hf_key.strip_prefix("decoder.block.") {
        return map_decoder_block(rest);
    }

    None
}

fn map_encoder_block(rest: &str) -> Option<String> {
    // rest = "{i}.layer.0.SelfAttention.{q,k,v,o}.weight"
    //   or  "{i}.layer.0.layer_norm.weight"
    //   or  "{i}.layer.1.DenseReluDense.{wi_0,wi_1,wo}.weight"
    //   or  "{i}.layer.1.layer_norm.weight"
    let (idx, rest) = rest.split_once('.')?;
    let rest = rest.strip_prefix("layer.")?;

    // layer.0 = self-attention sublayer
    if let Some(rest) = rest.strip_prefix("0.") {
        if let Some(rest) = rest.strip_prefix("SelfAttention.") {
            if rest == "relative_attention_bias.weight" {
                // Already handled above for block 0
                return None;
            }
            // q.weight, k.weight, v.weight, o.weight
            let key = rest;
            return Some(format!("encoder.blocks.{idx}.self_attn.{key}"));
        }
        if rest == "layer_norm.weight" {
            return Some(format!("encoder.blocks.{idx}.self_attn_norm.weight"));
        }
    }

    // layer.1 = FFN sublayer
    if let Some(rest) = rest.strip_prefix("1.") {
        if let Some(rest) = rest.strip_prefix("DenseReluDense.") {
            // wi_0.weight, wi_1.weight, wo.weight
            return Some(format!("encoder.blocks.{idx}.ff.{rest}"));
        }
        if rest == "layer_norm.weight" {
            return Some(format!("encoder.blocks.{idx}.ff_norm.weight"));
        }
    }

    None
}

fn map_decoder_block(rest: &str) -> Option<String> {
    let (idx, rest) = rest.split_once('.')?;
    let rest = rest.strip_prefix("layer.")?;

    // layer.0 = self-attention
    if let Some(rest) = rest.strip_prefix("0.") {
        if let Some(rest) = rest.strip_prefix("SelfAttention.") {
            if rest == "relative_attention_bias.weight" {
                return None; // handled separately for block 0
            }
            return Some(format!("decoder.blocks.{idx}.self_attn.{rest}"));
        }
        if rest == "layer_norm.weight" {
            return Some(format!("decoder.blocks.{idx}.self_attn_norm.weight"));
        }
    }

    // layer.1 = cross-attention (EncDecAttention)
    if let Some(rest) = rest.strip_prefix("1.") {
        if let Some(rest) = rest.strip_prefix("EncDecAttention.") {
            return Some(format!("decoder.blocks.{idx}.cross_attn.{rest}"));
        }
        if rest == "layer_norm.weight" {
            return Some(format!("decoder.blocks.{idx}.cross_attn_norm.weight"));
        }
    }

    // layer.2 = FFN
    if let Some(rest) = rest.strip_prefix("2.") {
        if let Some(rest) = rest.strip_prefix("DenseReluDense.") {
            return Some(format!("decoder.blocks.{idx}.ff.{rest}"));
        }
        if rest == "layer_norm.weight" {
            return Some(format!("decoder.blocks.{idx}.ff_norm.weight"));
        }
    }

    None
}
