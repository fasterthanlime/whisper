//! Interleaved Multi-dimensional Rotary Position Embedding (MRoPE) for Qwen3-ASR.
//!
//! Critical correctness module. Uses stride-3 interleaving across 3 spatial
//! dimensions (temporal, height, width) with sections [24, 20, 20].

use mlx_rs::Array;
use mlx_rs::error::Exception;
use mlx_rs::ops;
use mlx_rs::ops::indexing::{IndexOp, NewAxis};

pub const MROPE_SECTION: [usize; 3] = [24, 20, 20];

#[derive(Debug, Clone)]
pub struct InterleavedMRoPE {
    inv_freq: Array,
    /// Overwrite masks for height and width dimensions
    overwrite_masks: Vec<Array>,
}

impl InterleavedMRoPE {
    pub fn new(head_dim: usize, base: f64, mrope_section: &[usize; 3]) -> Self {
        let half_dim = head_dim / 2;
        assert_eq!(
            mrope_section.iter().sum::<usize>(),
            half_dim,
            "mrope_section sum must equal head_dim/2"
        );

        // inv_freq: 1 / (base ^ (2i / head_dim))
        let inv_freq_data: Vec<f32> = (0..half_dim)
            .map(|i| (1.0 / base.powf((2 * i) as f64 / head_dim as f64)) as f32)
            .collect();
        let inv_freq = Array::from_slice(&inv_freq_data, &[half_dim as i32]);

        // Build overwrite masks for height (dim=1) and width (dim=2)
        let mut overwrite_masks = Vec::new();
        for (dim_idx, &offset) in [1usize, 2].iter().enumerate() {
            let length = mrope_section[dim_idx + 1] * 3;
            let stop = length.min(half_dim);

            let mut mask_data = vec![false; half_dim];
            let mut idx = offset;
            while idx < stop {
                mask_data[idx] = true;
                idx += 3;
            }

            // Convert to f32 mask (1.0 where true, 0.0 where false) for mx.where emulation
            let mask_f32: Vec<f32> = mask_data
                .iter()
                .map(|&b| if b { 1.0 } else { 0.0 })
                .collect();
            let mask = Array::from_slice(&mask_f32, &[1, 1, half_dim as i32]);
            overwrite_masks.push(mask);
        }

        Self {
            inv_freq,
            overwrite_masks,
        }
    }

    /// Compute interleaved MRoPE cos/sin embeddings.
    ///
    /// position_ids: (batch, 3, seq_len)
    /// Returns: (cos, sin) each (batch, seq_len, head_dim)
    pub fn forward(&self, position_ids: &Array) -> Result<(Array, Array), Exception> {
        // position_ids: (B, 3, L) → transpose → (3, B, L)
        let pos = position_ids
            .as_dtype(mlx_rs::Dtype::Float32)?
            .transpose_axes(&[1, 0, 2])?;

        // (3, B, L, 1) * (half_dim,) → (3, B, L, half_dim)
        let pos = ops::expand_dims(&pos, -1)?;
        let inv = self.inv_freq.index((NewAxis, NewAxis, NewAxis, ..));
        let freqs = pos.multiply(&inv)?; // (3, B, L, half_dim)

        // Start with temporal freqs (dim 0)
        let mut freqs_t = freqs.index((0, .., .., ..));

        // Overwrite height/width indices using masks
        for (dim_idx, mask) in self.overwrite_masks.iter().enumerate() {
            let dim = dim_idx + 1;
            let freqs_dim = freqs.index((dim as i32, .., .., ..));
            // freqs_t = where(mask > 0, freqs_dim, freqs_t)
            let mask_broad = mask.as_dtype(mlx_rs::Dtype::Bool)?;
            freqs_t = mlx_rs::ops::r#where(&mask_broad, &freqs_dim, &freqs_t)?;
        }

        // emb = [freqs_t, freqs_t] along last dim → (B, L, head_dim)
        let emb = ops::concatenate_axis(&[&freqs_t, &freqs_t], -1)?;
        let cos = emb.cos()?;
        let sin = emb.sin()?;

        Ok((cos, sin))
    }
}

/// Apply rotary position embeddings to Q and K tensors.
///
/// q, k: (batch, heads, seq_len, head_dim)
/// cos, sin: (batch, seq_len, head_dim)
///
/// Returns (q_rotated, k_rotated) same shapes as input.
pub fn apply_rotary_pos_emb(
    q: &Array,
    k: &Array,
    cos: &Array,
    sin: &Array,
) -> Result<(Array, Array), Exception> {
    // Expand for multi-head: (B, L, D) → (B, 1, L, D)
    let cos = if cos.ndim() == 3 {
        ops::expand_dims(cos, 1)?
    } else {
        cos.clone()
    };
    let sin = if sin.ndim() == 3 {
        ops::expand_dims(sin, 1)?
    } else {
        sin.clone()
    };

    let q_embed = q.multiply(&cos)?.add(&rotate_half(q)?.multiply(&sin)?)?;
    let k_embed = k.multiply(&cos)?.add(&rotate_half(k)?.multiply(&sin)?)?;

    Ok((q_embed, k_embed))
}

fn rotate_half(x: &Array) -> Result<Array, Exception> {
    let mid = x.shape()[x.ndim() - 1] / 2;
    let x1 = x.index((.., .., .., ..mid));
    let x2 = x.index((.., .., .., mid..));
    let neg_x2 = x2.negative()?;
    ops::concatenate_axis(&[&neg_x2, &x1], -1)
}
