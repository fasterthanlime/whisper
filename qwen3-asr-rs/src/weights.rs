use std::collections::HashMap;
use std::sync::Arc;

use candle_core::quantized::{gguf_file, QMatMul, QTensor};
use candle_core::{Device, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, LayerNorm, RmsNorm};
use log::info;

use crate::linear::LinearW;

/// Pre-loaded GGUF weights — all tensors read from disk into memory.
pub(crate) struct GgufWeights {
    tensors: HashMap<String, Arc<QTensor>>,
    device: Device,
}

impl GgufWeights {
    pub fn load(path: &std::path::Path, device: &Device) -> anyhow::Result<Self> {
        let mut file = std::io::BufReader::new(std::fs::File::open(path)?);
        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| anyhow::anyhow!("GGUF read error: {e}"))?;

        info!(
            "GGUF file: {} tensors, {:?}",
            content.tensor_infos.len(),
            content.magic
        );

        let mut tensors = HashMap::new();
        for name in content.tensor_infos.keys() {
            let qt = content
                .tensor(&mut file, name, device)
                .map_err(|e| anyhow::anyhow!("GGUF tensor '{name}': {e}"))?;
            tensors.insert(name.clone(), Arc::new(qt));
        }

        info!("Loaded {} GGUF tensors", tensors.len());
        Ok(Self {
            tensors,
            device: device.clone(),
        })
    }

    /// List all tensor names (for debugging / name mapping).
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(|s| s.as_str())
    }
}

/// Abstraction over weight sources — safetensors (`Dense`) or GGUF (`Quantized`).
///
/// Linear layers become `LinearW::Quant` when quantized, everything else
/// (norms, convolutions, embeddings) gets dequantized to regular `Tensor`.
pub(crate) enum Weights {
    Dense(HashMap<String, Tensor>),
    Quantized(GgufWeights),
}

impl Weights {
    // ── Tensor access ───────────────────────────────────────────────────────

    /// Get a regular tensor by name (dequantizes if GGUF).
    pub fn get_tensor(&self, name: &str) -> anyhow::Result<Tensor> {
        match self {
            Self::Dense(map) => map
                .get(name)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("weight not found: {name}")),
            Self::Quantized(g) => {
                let qt = g
                    .tensors
                    .get(name)
                    .ok_or_else(|| anyhow::anyhow!("GGUF tensor not found: {name}"))?;
                qt.dequantize(&g.device)
                    .map_err(|e| anyhow::anyhow!("dequantize '{name}': {e}"))
            }
        }
    }

    /// Try to get a tensor, returning `None` if not found.
    pub fn try_get_tensor(&self, name: &str) -> anyhow::Result<Option<Tensor>> {
        match self {
            Self::Dense(map) => Ok(map.get(name).cloned()),
            Self::Quantized(g) => match g.tensors.get(name) {
                Some(qt) => {
                    Ok(Some(qt.dequantize(&g.device).map_err(|e| {
                        anyhow::anyhow!("dequantize '{name}': {e}")
                    })?))
                }
                None => Ok(None),
            },
        }
    }

    // ── Layer constructors ──────────────────────────────────────────────────

    /// Load a linear layer. Returns `LinearW::Quant` for GGUF, `LinearW::Dense`
    /// for safetensors.
    pub fn load_linear(&self, prefix: &str) -> anyhow::Result<LinearW> {
        match self {
            Self::Dense(map) => {
                let w = map
                    .get(&format!("{prefix}.weight"))
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("weight not found: {prefix}.weight"))?;
                let b = map.get(&format!("{prefix}.bias")).cloned();
                Ok(LinearW::new(w, b))
            }
            Self::Quantized(g) => {
                let qt = g
                    .tensors
                    .get(&format!("{prefix}.weight"))
                    .ok_or_else(|| anyhow::anyhow!("GGUF tensor not found: {prefix}.weight"))?;
                let qmatmul = QMatMul::from_arc(qt.clone())
                    .map_err(|e| anyhow::anyhow!("QMatMul '{prefix}': {e}"))?;
                let bias = g
                    .tensors
                    .get(&format!("{prefix}.bias"))
                    .map(|qt| {
                        qt.dequantize(&g.device)
                            .map_err(|e| anyhow::anyhow!("dequantize '{prefix}.bias': {e}"))
                    })
                    .transpose()?;
                Ok(LinearW::from_qmatmul(qmatmul, bias))
            }
        }
    }

    /// Load a `LayerNorm` (always dequantized — norm weights are tiny).
    pub fn load_layer_norm(&self, prefix: &str, eps: f64) -> anyhow::Result<LayerNorm> {
        Ok(LayerNorm::new(
            self.get_tensor(&format!("{prefix}.weight"))?,
            self.get_tensor(&format!("{prefix}.bias"))?,
            eps,
        ))
    }

    /// Load an `RmsNorm` (always dequantized).
    pub fn load_rms_norm(&self, prefix: &str, eps: f64) -> anyhow::Result<RmsNorm> {
        Ok(RmsNorm::new(
            self.get_tensor(&format!("{prefix}.weight"))?,
            eps,
        ))
    }

    /// Load a `Conv2d` (always dequantized — conv weights are small).
    pub fn load_conv2d(
        &self,
        prefix: &str,
        stride: usize,
        padding: usize,
    ) -> anyhow::Result<Conv2d> {
        let w = self.get_tensor(&format!("{prefix}.weight"))?;
        let b = self.try_get_tensor(&format!("{prefix}.bias"))?;
        Ok(Conv2d::new(
            w,
            b,
            Conv2dConfig {
                stride,
                padding,
                ..Default::default()
            },
        ))
    }

    /// Convert BF16/F16 weight tensors to F32 when running on CPU (dense only).
    pub fn maybe_convert_for_cpu(&mut self, device: &Device) {
        if !device.is_cpu() {
            return;
        }
        if let Self::Dense(map) = self {
            let mut converted = 0usize;
            for (name, tensor) in map.iter_mut() {
                match tensor.dtype() {
                    candle_core::DType::BF16 | candle_core::DType::F16 => {
                        match tensor.to_dtype(candle_core::DType::F32) {
                            Ok(t) => {
                                *tensor = t;
                                converted += 1;
                            }
                            Err(e) => {
                                log::warn!("Failed to convert {name} to F32: {e}");
                            }
                        }
                    }
                    _ => {}
                }
            }
            if converted > 0 {
                info!(
                    "Converted {converted} weight tensors from BF16/F16 to F32 for CPU inference"
                );
            }
        }
        // Quantized weights don't need CPU dtype conversion — QMatMul handles it.
    }
}
