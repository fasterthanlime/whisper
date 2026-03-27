use anyhow::{Context, Result, bail};
use candle_core::{Device, Tensor};
use peft_rs::{LoraConfig, LoraLayer, SaveLoad};
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Deserialize)]
pub struct MlxLoraParameters {
    pub rank: usize,
    pub dropout: f64,
    pub scale: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MlxAdapterConfig {
    pub fine_tune_type: String,
    pub model: String,
    pub num_layers: usize,
    pub lora_parameters: MlxLoraParameters,
}

pub struct ImportedLoraAdapters {
    pub config: MlxAdapterConfig,
    pub layers: BTreeMap<String, LoraLayer>,
    pub weights_path: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TensorKind {
    A,
    B,
}

#[derive(Default)]
struct ModuleWeights {
    lora_a: Option<Tensor>,
    lora_b: Option<Tensor>,
}

fn parse_module_tensor_name(name: &str) -> Option<(&str, TensorKind)> {
    if let Some(module_name) = name.strip_suffix(".lora_a") {
        Some((module_name, TensorKind::A))
    } else if let Some(module_name) = name.strip_suffix(".lora_b") {
        Some((module_name, TensorKind::B))
    } else {
        None
    }
}

fn alpha_from_mlx_scale(rank: usize, scale: f64) -> Result<usize> {
    if !scale.is_finite() || scale <= 0.0 {
        bail!("invalid MLX LoRA scale: {scale}");
    }
    let alpha = scale * rank as f64;
    let rounded = alpha.round();
    if (alpha - rounded).abs() > 1e-6 {
        bail!("non-integral alpha derived from MLX scale {scale} and rank {rank}");
    }
    Ok(rounded as usize)
}

pub fn load_mlx_lora_dir(dir: impl AsRef<Path>, device: &Device) -> Result<ImportedLoraAdapters> {
    let dir = dir.as_ref();
    let config_path = dir.join("adapter_config.json");
    let weights_path = dir.join("adapters.safetensors");

    let config_text = std::fs::read_to_string(&config_path)
        .with_context(|| format!("failed to read {}", config_path.display()))?;
    let config: MlxAdapterConfig = serde_json::from_str(&config_text)
        .with_context(|| format!("failed to parse {}", config_path.display()))?;

    if config.fine_tune_type != "lora" {
        bail!(
            "unsupported fine_tune_type {:?}, expected \"lora\"",
            config.fine_tune_type
        );
    }

    let tensors = candle_core::safetensors::load(&weights_path, device)
        .with_context(|| format!("failed to load {}", weights_path.display()))?;

    let mut grouped = BTreeMap::<String, ModuleWeights>::new();
    for (name, tensor) in tensors {
        let Some((module_name, tensor_kind)) = parse_module_tensor_name(&name) else {
            continue;
        };
        let weights = grouped.entry(module_name.to_string()).or_default();
        match tensor_kind {
            TensorKind::A => weights.lora_a = Some(tensor),
            TensorKind::B => weights.lora_b = Some(tensor),
        }
    }

    if grouped.is_empty() {
        bail!("no LoRA tensors found in {}", weights_path.display());
    }

    let alpha = alpha_from_mlx_scale(config.lora_parameters.rank, config.lora_parameters.scale)?;
    let base_config = LoraConfig {
        r: config.lora_parameters.rank,
        alpha,
        dropout: config.lora_parameters.dropout,
        target_modules: Vec::new(),
        ..Default::default()
    };

    let mut layers = BTreeMap::new();
    for (module_name, weights) in grouped {
        let lora_a = weights
            .lora_a
            .with_context(|| format!("missing lora_a tensor for {module_name}"))?;
        let lora_b = weights
            .lora_b
            .with_context(|| format!("missing lora_b tensor for {module_name}"))?;

        let lora_a_dims = lora_a.dims();
        let lora_b_dims = lora_b.dims();
        if lora_a_dims.len() != 2 || lora_b_dims.len() != 2 {
            bail!(
                "expected 2D LoRA tensors for {module_name}, got {:?} and {:?}",
                lora_a_dims,
                lora_b_dims
            );
        }
        if lora_a_dims[0] != config.lora_parameters.rank
            || lora_b_dims[1] != config.lora_parameters.rank
        {
            bail!(
                "rank mismatch for {module_name}: config rank {}, tensors {:?} and {:?}",
                config.lora_parameters.rank,
                lora_a_dims,
                lora_b_dims
            );
        }

        let in_features = lora_a_dims[1];
        let out_features = lora_b_dims[0];
        let mut layer_config = base_config.clone();
        layer_config.target_modules = vec![module_name.clone()];

        let mut layer = LoraLayer::new_with_zeros(in_features, out_features, layer_config, device)
            .with_context(|| format!("failed to create LoRA layer for {module_name}"))?;
        let state_dict = HashMap::from([
            ("lora_a.weight".to_string(), lora_a),
            ("lora_b.weight".to_string(), lora_b),
        ]);
        layer
            .load_state_dict(state_dict)
            .with_context(|| format!("failed to load LoRA weights for {module_name}"))?;
        layers.insert(module_name, layer);
    }

    Ok(ImportedLoraAdapters {
        config,
        layers,
        weights_path,
    })
}

#[cfg(test)]
mod tests {
    use super::{TensorKind, alpha_from_mlx_scale, parse_module_tensor_name};

    #[test]
    fn parses_mlx_tensor_names() {
        assert_eq!(
            parse_module_tensor_name("model.layers.20.self_attn.q_proj.lora_a"),
            Some(("model.layers.20.self_attn.q_proj", TensorKind::A))
        );
        assert_eq!(
            parse_module_tensor_name("model.layers.23.mlp.down_proj.lora_b"),
            Some(("model.layers.23.mlp.down_proj", TensorKind::B))
        );
        assert_eq!(
            parse_module_tensor_name("model.layers.23.mlp.down_proj.weight"),
            None
        );
    }

    #[test]
    fn mlx_scale_maps_to_alpha() {
        assert_eq!(alpha_from_mlx_scale(8, 20.0).unwrap(), 160);
    }
}
