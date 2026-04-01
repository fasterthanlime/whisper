//! Text decoder components for Qwen3-ASR in mlx-rs.
//!
//! Uses RMSNorm (NOT LayerNorm), GQA with QK-norm, interleaved MRoPE, SwiGLU MLP.

use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::macros::{ModuleParameters, Quantizable};
use mlx_rs::module::Module;
use mlx_rs::nn;
use mlx_rs::ops;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::quantization::MaybeQuantized;
use mlx_rs::Array;

use crate::config::TextDecoderConfig;
use crate::mrope::{self, InterleavedMRoPE};

// ── KV Cache ────────────────────────────────────────────────────────────

/// Per-layer key-value cache for autoregressive generation.
pub struct KVCache {
    keys: Vec<Option<Array>>,
    values: Vec<Option<Array>>,
    pub offset: usize,
}

impl KVCache {
    pub fn new(num_layers: usize) -> Self {
        Self {
            keys: vec![None; num_layers],
            values: vec![None; num_layers],
            offset: 0,
        }
    }

    /// Append new K/V and return full cached K/V.
    pub fn update(
        &mut self,
        key: Array,
        value: Array,
        layer_idx: usize,
    ) -> Result<(Array, Array), Exception> {
        if let Some(prev_k) = &self.keys[layer_idx] {
            let prev_v = self.values[layer_idx].as_ref().unwrap();
            self.keys[layer_idx] = Some(ops::concatenate_axis(&[prev_k, &key], 2)?);
            self.values[layer_idx] = Some(ops::concatenate_axis(&[prev_v, &value], 2)?);
        } else {
            self.keys[layer_idx] = Some(key);
            self.values[layer_idx] = Some(value);
        }

        let full_k = self.keys[layer_idx].as_ref().unwrap().clone();
        let full_v = self.values[layer_idx].as_ref().unwrap().clone();

        // Update offset after last layer
        if layer_idx == self.keys.len() - 1 {
            self.offset += full_k.shape()[2] as usize - (self.offset);
        }

        Ok((full_k, full_v))
    }

    /// Truncate the cache to keep only the first `seq_len` positions.
    /// Keys/values have shape (B, num_heads, seq_len, head_dim).
    pub fn truncate(&mut self, seq_len: usize) {
        for layer_k in &mut self.keys {
            if let Some(k) = layer_k {
                let current = k.shape()[2] as usize;
                if seq_len < current {
                    *k = k.index((.., .., ..seq_len as i32, ..));
                }
            }
        }
        for layer_v in &mut self.values {
            if let Some(v) = layer_v {
                let current = v.shape()[2] as usize;
                if seq_len < current {
                    *v = v.index((.., .., ..seq_len as i32, ..));
                }
            }
        }
        self.offset = seq_len;
    }
}

// ── TextAttention ───────────────────────────────────────────────────────

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct TextAttention {
    #[quantizable]
    #[param]
    pub q_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub k_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub v_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub o_proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub q_norm: nn::RmsNorm,
    #[param]
    pub k_norm: nn::RmsNorm,

    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
}

impl TextAttention {
    fn new(config: &TextDecoderConfig) -> Result<Self, Exception> {
        let h = config.hidden_size as i32;
        let nh = config.num_attention_heads as i32;
        let nkv = config.num_key_value_heads as i32;
        let hd = config.head_dim as i32;
        let bias = false; // Qwen3-ASR uses no bias on decoder attention

        Ok(Self {
            q_proj: MaybeQuantized::new(nn::LinearBuilder::new(h, nh * hd).bias(bias).build()?),
            k_proj: MaybeQuantized::new(nn::LinearBuilder::new(h, nkv * hd).bias(bias).build()?),
            v_proj: MaybeQuantized::new(nn::LinearBuilder::new(h, nkv * hd).bias(bias).build()?),
            o_proj: MaybeQuantized::new(nn::LinearBuilder::new(nh * hd, h).bias(bias).build()?),
            q_norm: nn::RmsNormBuilder::new(hd).eps(config.rms_norm_eps as f32).build()?,
            k_norm: nn::RmsNormBuilder::new(hd).eps(config.rms_norm_eps as f32).build()?,
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: hd,
        })
    }

    fn forward_attn(
        &mut self,
        x: &Array,
        cos: &Array,
        sin: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
        layer_idx: usize,
    ) -> Result<Array, Exception> {
        let b = x.shape()[0];
        let l = x.shape()[1];

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // (B, L, N*D) → (B, L, N, D)
        let mut q = q.reshape(&[b, l, self.num_heads, self.head_dim])?;
        let mut k = k.reshape(&[b, l, self.num_kv_heads, self.head_dim])?;
        let v = v.reshape(&[b, l, self.num_kv_heads, self.head_dim])?;

        // QK normalization
        q = self.q_norm.forward(&q)?;
        k = self.k_norm.forward(&k)?;

        // (B, L, N, D) → (B, N, L, D)
        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;

        // Apply MRoPE
        let (q, mut k) = mrope::apply_rotary_pos_emb(&q, &k, cos, sin)?;
        let mut v = v;

        // KV cache
        if let Some(cache) = cache {
            let (fk, fv) = cache.update(k, v, layer_idx)?;
            k = fk;
            v = fv;
        }

        // GQA: repeat KV heads
        let num_groups = self.num_heads / self.num_kv_heads;
        if num_groups > 1 {
            // (B, nkv, L, D) → repeat → (B, nkv*groups, L, D)
            k = repeat_kv(&k, num_groups)?;
            v = repeat_kv(&v, num_groups)?;
        }

        // Scaled dot-product attention
        let scale = Array::from_f32(1.0 / (self.head_dim as f32).sqrt());
        let attn = q.matmul(&k.transpose_axes(&[0, 1, 3, 2])?)?.multiply(&scale)?;
        let attn = match mask {
            Some(m) => attn.add(m)?,
            None => attn,
        };
        let attn = ops::softmax_axis(&attn, -1, None)?;
        let out = attn.matmul(&v)?;

        // (B, N, L, D) → (B, L, N*D)
        let out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, l, -1])?;
        self.o_proj.forward(&out)
    }
}

fn repeat_kv(x: &Array, n_rep: i32) -> Result<Array, Exception> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    // (B, nkv, L, D) → (B, nkv, 1, L, D) → expand → (B, nkv, n_rep, L, D) → reshape
    let sh = x.shape();
    let (b, nkv, l, d) = (sh[0], sh[1], sh[2], sh[3]);
    let x = ops::expand_dims(x, 2)?;
    // broadcast_to (B, nkv, n_rep, L, D)
    let expanded = ops::broadcast_to(&x, &[b, nkv, n_rep, l, d])?;
    expanded.reshape(&[b, nkv * n_rep, l, d])
}

// ── SwiGLU ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct SwiGLU {
    #[quantizable]
    #[param]
    pub gate_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub up_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub down_proj: MaybeQuantized<nn::Linear>,
}

impl SwiGLU {
    fn new(hidden_size: i32, intermediate_size: i32) -> Result<Self, Exception> {
        Ok(Self {
            gate_proj: MaybeQuantized::new(nn::LinearBuilder::new(hidden_size, intermediate_size).bias(false).build()?),
            up_proj: MaybeQuantized::new(nn::LinearBuilder::new(hidden_size, intermediate_size).bias(false).build()?),
            down_proj: MaybeQuantized::new(nn::LinearBuilder::new(intermediate_size, hidden_size).bias(false).build()?),
        })
    }
}

impl Module<&Array> for SwiGLU {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let gate = nn::silu(self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&gate.multiply(&up)?)
    }

    fn training_mode(&mut self, _mode: bool) {}
}

// ── TextDecoderLayer ────────────────────────────────────────────────────

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct TextDecoderLayer {
    #[param]
    pub input_layernorm: nn::RmsNorm,
    #[quantizable]
    #[param]
    pub self_attn: TextAttention,
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
    #[quantizable]
    #[param]
    pub mlp: SwiGLU,
}

impl TextDecoderLayer {
    fn new(config: &TextDecoderConfig) -> Result<Self, Exception> {
        let h = config.hidden_size as i32;
        let eps = config.rms_norm_eps as f32;
        Ok(Self {
            input_layernorm: nn::RmsNormBuilder::new(h).eps(eps).build()?,
            self_attn: TextAttention::new(config)?,
            post_attention_layernorm: nn::RmsNormBuilder::new(h).eps(eps).build()?,
            mlp: SwiGLU::new(h, config.intermediate_size as i32)?,
        })
    }

    fn forward_layer(
        &mut self,
        x: &Array,
        cos: &Array,
        sin: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
        layer_idx: usize,
    ) -> Result<Array, Exception> {
        // Self-attention
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward_attn(&normed, cos, sin, mask, cache, layer_idx)?;
        let x = x.add(&attn_out)?;

        // FFN
        let normed = self.post_attention_layernorm.forward(&x)?;
        let mlp_out = self.mlp.forward(&normed)?;
        x.add(&mlp_out)
    }
}

// ── TextDecoder ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct TextDecoder {
    #[quantizable]
    #[param]
    pub embed_tokens: MaybeQuantized<nn::Embedding>,
    #[quantizable]
    #[param]
    pub layers: Vec<TextDecoderLayer>,
    #[param]
    pub norm: nn::RmsNorm,

    pub rotary_emb: InterleavedMRoPE,
    pub config: TextDecoderConfig,
}

impl TextDecoder {
    pub fn new(config: &TextDecoderConfig) -> Result<Self, Exception> {
        let h = config.hidden_size as i32;
        let eps = config.rms_norm_eps as f32;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(TextDecoderLayer::new(config)?);
        }

        let mrope_section = config
            .rope_scaling
            .as_ref()
            .and_then(|s| {
                if s.mrope_section.len() == 3 {
                    Some([s.mrope_section[0], s.mrope_section[1], s.mrope_section[2]])
                } else {
                    None
                }
            })
            .unwrap_or(mrope::MROPE_SECTION);

        Ok(Self {
            embed_tokens: MaybeQuantized::new(nn::Embedding::new(config.vocab_size as i32, h)?),
            layers,
            norm: nn::RmsNormBuilder::new(h).eps(eps).build()?,
            rotary_emb: InterleavedMRoPE::new(
                config.head_dim,
                config.rope_theta,
                &mrope_section,
            ),
            config: config.clone(),
        })
    }

    /// Forward pass.
    ///
    /// Either `input_ids` (B, L) or `inputs_embeds` (B, L, D) must be provided.
    /// `position_ids` (B, 3, L) must always be provided for MRoPE.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn forward_decoder(
        &mut self,
        input_ids: Option<&Array>,
        inputs_embeds: Option<&Array>,
        position_ids: &Array,
        cache: &mut Option<KVCache>,
    ) -> Result<Array, Exception> {
        let h = match (input_ids, inputs_embeds) {
            (_, Some(embeds)) => embeds.clone(),
            (Some(ids), _) => self.embed_tokens.forward(ids)?,
            _ => return Err(Exception::custom("input_ids or inputs_embeds required")),
        };

        let seq_len = h.shape()[1] as usize;

        // Build causal mask
        let mask = if let Some(ref cache) = cache {
            if cache.offset > 0 {
                // Decoding with cache: single token needs no mask,
                // multi-token needs causal mask over prefix
                if seq_len == 1 {
                    None
                } else {
                    Some(create_causal_mask_with_prefix(seq_len, cache.offset))
                }
            } else {
                // Prefill: full causal mask
                if seq_len > 1 {
                    Some(create_causal_mask(seq_len))
                } else {
                    None
                }
            }
        } else {
            // No cache: full causal mask
            if seq_len > 1 {
                Some(create_causal_mask(seq_len))
            } else {
                None
            }
        };

        let (cos, sin) = self.rotary_emb.forward(position_ids)?;

        let mut h = h;
        let num_layers = self.layers.len();
        for i in 0..num_layers {
            h = self.layers[i].forward_layer(
                &h, &cos, &sin, mask.as_ref(),
                cache.as_mut(),
                i,
            )?;
        }

        self.norm.forward(&h)
    }
}

// ── Causal mask ─────────────────────────────────────────────────────────

pub fn create_causal_mask(seq_len: usize) -> Array {
    let mut mask_data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask_data[i * seq_len + j] = -1e9;
        }
    }
    let mask = Array::from_slice(&mask_data, &[seq_len as i32, seq_len as i32]);
    ops::expand_dims_axes(&mask, &[0, 1]).unwrap()
}

/// Causal mask for decoding with a cached prefix.
/// New tokens can attend to all prefix tokens and causally to each other.
pub fn create_causal_mask_with_prefix(seq_len: usize, prefix_len: usize) -> Array {
    let total_len = prefix_len + seq_len;
    let mut mask_data = vec![0.0f32; seq_len * total_len];
    for i in 0..seq_len {
        // Prefix region: all visible (0.0)
        // New tokens region: causal
        for j in 0..seq_len {
            if j > i {
                mask_data[i * total_len + prefix_len + j] = -1e9;
            }
        }
    }
    let mask = Array::from_slice(&mask_data, &[seq_len as i32, total_len as i32]);
    ops::expand_dims_axes(&mask, &[0, 1]).unwrap()
}
