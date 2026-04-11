use std::collections::HashMap;
use std::path::PathBuf;

use mlx_rs::Array;

/// Load reference tensors from the dump_reference.py output.
pub fn load_reference() -> Option<(PathBuf, HashMap<String, Array>)> {
    let home = std::env::var("HOME").ok()?;
    let ref_dir = PathBuf::from(home).join(".bearcove/charsiu-g2p/reference");
    let ref_path = ref_dir.join("reference.safetensors");
    if !ref_path.exists() {
        eprintln!(
            "Reference artifacts not found at {}, skipping test",
            ref_path.display()
        );
        return None;
    }
    let tensors = Array::load_safetensors(&ref_path).ok()?;
    Some((ref_dir, tensors))
}

/// Load the model safetensors weights.
pub fn model_dir() -> Option<PathBuf> {
    let dir = PathBuf::from("/tmp/charsiu-g2p");
    if dir.join("model.safetensors").exists() {
        Some(dir)
    } else {
        eprintln!("Model safetensors not found at /tmp/charsiu-g2p, skipping test");
        None
    }
}

/// Assert two arrays are close within tolerance.
pub fn assert_close(name: &str, actual: &Array, expected: &Array, atol: f32, rtol: f32) {
    let close = actual
        .all_close(expected, atol as f64, rtol as f64, None)
        .unwrap();
    close.eval().unwrap();
    let is_close: bool = close.item();
    if !is_close {
        let diff = actual.subtract(expected).unwrap().abs().unwrap();
        let max_diff: f32 = diff.max(None).unwrap().item();
        let mean_diff: f32 = diff.mean(None).unwrap().item();
        panic!(
            "{name}: arrays not close! max_diff={max_diff:.6}, mean_diff={mean_diff:.6}, \
             actual shape={:?}, expected shape={:?}",
            actual.shape(),
            expected.shape()
        );
    }
}
