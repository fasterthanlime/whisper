use std::collections::HashMap;
use std::path::Path;

use log::info;
use mlx_rs::Array;

use crate::error::AsrError;

/// Loaded model weights from safetensors, with key remapping applied.
pub struct Weights {
    tensors: HashMap<String, Array>,
}

impl Weights {
    /// Load weights from one or more safetensors files in a directory.
    pub fn load_dir(dir: &Path) -> Result<Self, AsrError> {
        let mut tensors = HashMap::new();

        // Find all safetensors files
        let mut files: Vec<_> = std::fs::read_dir(dir)
            .map_err(|e| AsrError::ModelLoad(format!("cannot read dir {}: {e}", dir.display())))?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .path()
                    .extension()
                    .map_or(false, |ext| ext == "safetensors")
            })
            .collect();
        files.sort_by_key(|e| e.file_name());

        if files.is_empty() {
            return Err(AsrError::ModelLoad(format!(
                "no .safetensors files in {}",
                dir.display()
            )));
        }

        for entry in &files {
            let path = entry.path();
            let file_tensors = Array::load_safetensors(&path).map_err(|e| {
                AsrError::ModelLoad(format!("load {}: {e}", path.display()))
            })?;
            tensors.extend(file_tensors);
        }

        info!("Loaded {} tensors from {} files", tensors.len(), files.len());

        // Remap keys: strip thinker. prefix, transpose conv2d weights
        let remapped = remap_weights(tensors);

        Ok(Self { tensors: remapped })
    }

    /// Get a tensor by name.
    pub fn get(&self, name: &str) -> Result<&Array, AsrError> {
        self.tensors
            .get(name)
            .ok_or_else(|| AsrError::ModelLoad(format!("weight not found: {name}")))
    }

    /// Try to get a tensor, returning None if not found.
    pub fn try_get(&self, name: &str) -> Option<&Array> {
        self.tensors.get(name)
    }

    /// List all tensor names (for debugging).
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(|s| s.as_str())
    }

    /// Number of loaded tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }
}

/// Remap HuggingFace weight keys to model keys.
///
/// 1. Strip `thinker.` prefix from all keys
/// 2. Transpose Conv2d weights from PyTorch (out,in,kH,kW) to MLX (out,kH,kW,in)
fn remap_weights(weights: HashMap<String, Array>) -> HashMap<String, Array> {
    let mut remapped = HashMap::with_capacity(weights.len());

    for (key, value) in weights {
        let mut new_key = key.clone();
        let had_thinker = new_key.starts_with("thinker.");
        if had_thinker {
            new_key = new_key["thinker.".len()..].to_string();
        }

        // Transpose Conv2d weights: PyTorch (out,in,kH,kW) → MLX (out,kH,kW,in)
        let value = if had_thinker
            && new_key.contains("conv2d")
            && new_key.ends_with(".weight")
            && value.ndim() == 4
        {
            value.transpose_axes(&[0i32, 2, 3, 1]).unwrap_or(value)
        } else {
            value
        };

        remapped.insert(new_key, value);
    }

    info!("Remapped {} weight keys", remapped.len());
    remapped
}
