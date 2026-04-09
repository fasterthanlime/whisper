use mlx_rs::Array;
use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::macros::{ModuleParameters, Quantizable};
use mlx_rs::module::Module;
use mlx_rs::nn;
use mlx_rs::ops;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::quantization::MaybeQuantized;

use crate::config::AudioEncoderConfig;

/// Windowed segment threshold — use per-window execution above this.
const WINDOWED_SEGMENT_MIN_WINDOWS: usize = 20;

// ── Sinusoidal PE (computed, not loaded) ────────────────────────────────

fn build_sinusoidal_pe(max_positions: usize, embedding_dim: usize) -> Array {
    let half_dim = embedding_dim / 2;
    let log_timescale_increment = (10000.0f32).ln() / (half_dim as f32 - 1.0);

    let inv_timescales: Vec<f32> = (0..half_dim)
        .map(|i| (-log_timescale_increment * i as f32).exp())
        .collect();
    let inv_t = Array::from_slice(&inv_timescales, &[1, half_dim as i32]);

    let positions: Vec<f32> = (0..max_positions).map(|i| i as f32).collect();
    let pos = Array::from_slice(&positions, &[max_positions as i32, 1]);

    let scaled_time = pos.matmul(&inv_t).unwrap();
    let sin_part = scaled_time.sin().unwrap();
    let cos_part = scaled_time.cos().unwrap();

    ops::concatenate_axis(&[&sin_part, &cos_part], -1).unwrap()
}

// ── AudioAttention ──────────────────────────────────────────────────────

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct AudioAttention {
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
    pub out_proj: MaybeQuantized<nn::Linear>,
    pub num_heads: i32,
    pub head_dim: i32,
}

impl AudioAttention {
    fn new(d_model: i32, num_heads: i32) -> Result<Self, Exception> {
        let head_dim = d_model / num_heads;
        Ok(Self {
            q_proj: MaybeQuantized::new(
                nn::LinearBuilder::new(d_model, d_model)
                    .bias(true)
                    .build()?,
            ),
            k_proj: MaybeQuantized::new(
                nn::LinearBuilder::new(d_model, d_model)
                    .bias(true)
                    .build()?,
            ),
            v_proj: MaybeQuantized::new(
                nn::LinearBuilder::new(d_model, d_model)
                    .bias(true)
                    .build()?,
            ),
            out_proj: MaybeQuantized::new(
                nn::LinearBuilder::new(d_model, d_model)
                    .bias(true)
                    .build()?,
            ),
            num_heads,
            head_dim,
        })
    }
}

struct AudioAttentionInput<'a> {
    x: &'a Array,
    mask: Option<&'a Array>,
}

impl Module<AudioAttentionInput<'_>> for AudioAttention {
    type Output = Array;
    type Error = Exception;

    fn forward(&self, input: AudioAttentionInput<'_>) -> Result<Array, Exception> {
        let AudioAttentionInput { x, mask } = input;
        let b = x.shape()[0];
        let l = x.shape()[1];

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // (B, L, D) → (B, L, H, Dh) → (B, H, L, Dh)
        let q = q
            .reshape(&[b, l, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = k
            .reshape(&[b, l, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = v
            .reshape(&[b, l, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // Scaled dot-product attention
        let scale = Array::from_f32(1.0 / (self.head_dim as f32).sqrt());
        let attn = q
            .matmul(&k.transpose_axes(&[0, 1, 3, 2])?)?
            .multiply(&scale)?;

        let attn = match mask {
            Some(m) => attn.add(m)?,
            None => attn,
        };

        let attn = ops::softmax_axis(&attn, -1, None)?;
        let out = attn.matmul(&v)?;

        // (B, H, L, Dh) → (B, L, H*Dh)
        let out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, l, -1])?;
        self.out_proj.forward(&out)
    }

    fn training_mode(&mut self, _mode: bool) {}
}

// ── AudioEncoderLayer ───────────────────────────────────────────────────

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct AudioEncoderLayer {
    #[param]
    pub self_attn_layer_norm: nn::LayerNorm,
    #[quantizable]
    #[param]
    pub self_attn: AudioAttention,
    #[param]
    pub final_layer_norm: nn::LayerNorm,
    #[quantizable]
    #[param]
    pub fc1: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub fc2: MaybeQuantized<nn::Linear>,
}

impl AudioEncoderLayer {
    fn new(d_model: i32, num_heads: i32, ffn_dim: i32) -> Result<Self, Exception> {
        Ok(Self {
            self_attn_layer_norm: nn::LayerNorm::new(d_model)?,
            self_attn: AudioAttention::new(d_model, num_heads)?,
            final_layer_norm: nn::LayerNorm::new(d_model)?,
            fc1: MaybeQuantized::new(
                nn::LinearBuilder::new(d_model, ffn_dim)
                    .bias(true)
                    .build()?,
            ),
            fc2: MaybeQuantized::new(
                nn::LinearBuilder::new(ffn_dim, d_model)
                    .bias(true)
                    .build()?,
            ),
        })
    }

    fn forward_with_mask(&self, x: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        // Self-attention (pre-norm)
        let normed = self.self_attn_layer_norm.forward(x)?;
        let attn_out = self
            .self_attn
            .forward(AudioAttentionInput { x: &normed, mask })?;
        let x = x.add(&attn_out)?;

        // FFN (pre-norm)
        let normed = self.final_layer_norm.forward(&x)?;
        let h = self.fc1.forward(&normed)?;
        let h = nn::gelu(&h)?;
        let h = self.fc2.forward(&h)?;
        let x = x.add(&h)?;

        Ok(x)
    }
}

// ── AudioEncoder ────────────────────────────────────────────────────────

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct AudioEncoder {
    #[param]
    pub conv2d1: nn::Conv2d,
    #[param]
    pub conv2d2: nn::Conv2d,
    #[param]
    pub conv2d3: nn::Conv2d,
    #[quantizable]
    #[param]
    pub conv_out: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub layers: Vec<AudioEncoderLayer>,
    #[param]
    pub ln_post: nn::LayerNorm,
    #[quantizable]
    #[param]
    pub proj1: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub proj2: MaybeQuantized<nn::Linear>,

    // Not a parameter — computed at init
    pub sinusoidal_pe: Array,
    pub config: AudioEncoderConfig,
}

impl AudioEncoder {
    pub fn new(config: &AudioEncoderConfig) -> Result<Self, Exception> {
        // Compute downsample_hidden_size from config:
        // After 3x stride-2 on 128 mel bins: 128→64→32→16
        let freq_after_conv = config.num_mel_bins / 8;
        // The conv output channels can be inferred from d_model and freq_after_conv
        // dhs = d_model * 8 / num_mel_bins... but actually we need to check the Python
        // For 1.7B: d_model=1024, dhs=480, freq_after_conv=16, so conv_out = 480*16=7680 → 1024
        // For 0.6B: d_model=896, dhs=?, freq_after_conv=16
        // The dhs is a config param that doesn't exist in our config. Let's compute from conv_chunksize.
        // Actually from the Python: dhs = config.downsample_hidden_size = 480 (1.7B) or similar
        // We need to add this to config or compute it. For now, compute from output_dim:
        // dhs * freq_after_conv feeds into conv_out(dhs*freq, d_model)
        // We can get dhs from the weight shapes at load time.
        // For scaffolding, use a reasonable default based on model size.
        let dhs = if config.d_model >= 1024 { 480 } else { 384 };

        let conv2d1 = nn::Conv2dBuilder::new(1, dhs as i32, 3)
            .stride(2)
            .padding(1)
            .build()?;
        let conv2d2 = nn::Conv2dBuilder::new(dhs as i32, dhs as i32, 3)
            .stride(2)
            .padding(1)
            .build()?;
        let conv2d3 = nn::Conv2dBuilder::new(dhs as i32, dhs as i32, 3)
            .stride(2)
            .padding(1)
            .build()?;

        let conv_out = MaybeQuantized::new(
            nn::LinearBuilder::new((dhs * freq_after_conv) as i32, config.d_model as i32)
                .bias(false)
                .build()?,
        );

        let sinusoidal_pe = build_sinusoidal_pe(config.max_source_positions, config.d_model);

        let mut layers = Vec::with_capacity(config.encoder_layers);
        for _ in 0..config.encoder_layers {
            layers.push(AudioEncoderLayer::new(
                config.d_model as i32,
                config.encoder_attention_heads as i32,
                config.encoder_ffn_dim as i32,
            )?);
        }

        let ln_post = nn::LayerNorm::new(config.d_model as i32)?;
        let proj1 = MaybeQuantized::new(
            nn::LinearBuilder::new(config.d_model as i32, config.d_model as i32)
                .bias(true)
                .build()?,
        );
        let proj2 = MaybeQuantized::new(
            nn::LinearBuilder::new(config.d_model as i32, config.output_dim as i32)
                .bias(true)
                .build()?,
        );

        Ok(Self {
            conv2d1,
            conv2d2,
            conv2d3,
            conv_out,
            sinusoidal_pe,
            layers,
            ln_post,
            proj1,
            proj2,
            config: config.clone(),
        })
    }

    /// Encode a mel spectrogram. Input: (n_mels, n_frames). Output: (n_tokens, output_dim).
    pub fn encode(&self, mel: &Array) -> Result<Array, Exception> {
        let total_frames = mel.shape()[1] as usize;
        let chunk_size = self.config.n_window * 2;
        let n_window_infer = self.config.n_window_infer;

        let n_full_chunks = total_frames / chunk_size;
        let mut chunk_token_lens: Vec<usize> = Vec::new();
        let mut chunk_conv_outputs: Vec<Array> = Vec::new();

        // Process full chunks
        if n_full_chunks > 0 {
            let full_frames = (n_full_chunks * chunk_size) as i32;
            let n_mels = mel.shape()[0];
            // mel[:, :full_frames] → reshape to (n_full, n_mels, chunk_size) → add channel
            let full_mel = mel.index((.., ..full_frames));
            let full_mel = full_mel
                .reshape(&[n_mels, n_full_chunks as i32, chunk_size as i32])?
                .transpose_axes(&[1, 0, 2])?; // (n_full, n_mels, chunk_size)

            // NHWC: (n_full, H=n_mels, W=chunk_size, C=1)
            let x = ops::expand_dims(&full_mel, -1)?;
            let x = self.apply_conv_stem(&x)?;

            // (n_full, F', T', C) → (n_full, T', C, F') → (n_full*T', C*F')
            let sh = x.shape();
            let (n, f_d, t_d, c_d) = (sh[0], sh[1], sh[2], sh[3]);
            let x = x
                .transpose_axes(&[0, 2, 3, 1])?
                .reshape(&[n * t_d, c_d * f_d])?;

            chunk_conv_outputs.push(x);
            for _ in 0..n_full_chunks {
                chunk_token_lens.push(t_d as usize);
            }
        }

        // Tail chunk
        let tail_start = (n_full_chunks * chunk_size) as i32;
        if (tail_start as usize) < total_frames {
            let tail_mel = mel.index((.., tail_start..));
            // NHWC: (1, n_mels, tail_len, 1)
            let x = ops::expand_dims(&tail_mel, 0)?;
            let x = ops::expand_dims(&x, -1)?;
            let x = self.apply_conv_stem(&x)?;

            let sh = x.shape();
            let (f_d, t_d, c_d) = (sh[1], sh[2], sh[3]);
            let x = x
                .transpose_axes(&[0, 2, 3, 1])?
                .reshape(&[t_d, c_d * f_d])?;
            chunk_token_lens.push(t_d as usize);
            chunk_conv_outputs.push(x);
        }

        if chunk_conv_outputs.is_empty() {
            return Err(Exception::custom("no audio frames"));
        }

        // Concatenate and project
        let refs: Vec<&Array> = chunk_conv_outputs.iter().collect();
        let x = ops::concatenate_axis(&refs, 0)?;
        let x = self.conv_out.forward(&x)?;

        // Per-chunk sinusoidal PE
        let max_chunk_tokens = *chunk_token_lens.iter().max().unwrap();
        let pe = self.sinusoidal_pe.index((..max_chunk_tokens as i32, ..));
        let mut pe_parts: Vec<Array> = Vec::new();
        for &ct in &chunk_token_lens {
            pe_parts.push(pe.index((..ct as i32, ..)));
        }
        let pe_refs: Vec<&Array> = pe_parts.iter().collect();
        let pe_full = ops::concatenate_axis(&pe_refs, 0)?;
        let x = x.add(&pe_full)?;

        // Windowed attention
        let total_tokens = x.shape()[0] as usize;
        let tokens_per_full_chunk = chunk_token_lens[0];
        let tokens_per_window = tokens_per_full_chunk * (n_window_infer / chunk_size);

        let mut cu_seqlens: Vec<usize> = vec![0];
        let mut pos = 0usize;
        while pos < total_tokens {
            let window_end = (pos + tokens_per_window).min(total_tokens);
            cu_seqlens.push(window_end);
            pos = window_end;
        }

        let num_windows = cu_seqlens.len() - 1;
        // Add batch dim: (1, total_tokens, d_model)
        let mut x = ops::expand_dims(&x, 0)?;

        if num_windows >= WINDOWED_SEGMENT_MIN_WINDOWS {
            for layer in &self.layers {
                let mut parts: Vec<Array> = Vec::new();
                for w in 0..num_windows {
                    let s = cu_seqlens[w] as i32;
                    let e = cu_seqlens[w + 1] as i32;
                    let window = x.index((.., s..e, ..));
                    parts.push(layer.forward_with_mask(&window, None)?);
                }
                let refs: Vec<&Array> = parts.iter().collect();
                x = ops::concatenate_axis(&refs, 1)?;
            }
        } else {
            let mask = create_windowed_mask(total_tokens, &cu_seqlens);
            for layer in &self.layers {
                x = layer.forward_with_mask(&x, mask.as_ref())?;
            }
        }

        // Remove batch dim
        let x = x.index((0, ..));

        // Post-processing
        let x = self.ln_post.forward(&x)?;
        let x = self.proj1.forward(&x)?;
        let x = nn::gelu(&x)?;
        let x = self.proj2.forward(&x)?;

        Ok(x) // (total_tokens, output_dim)
    }

    fn apply_conv_stem(&self, x: &Array) -> Result<Array, Exception> {
        let x = self.conv2d1.forward(x)?;
        let x = nn::gelu(&x)?;
        let x = self.conv2d2.forward(&x)?;
        let x = nn::gelu(&x)?;
        let x = self.conv2d3.forward(&x)?;
        nn::gelu(&x)
    }

    /// Replace the heavy shared backbone with another encoder's backbone while
    /// preserving this encoder's final output projection.
    pub fn share_backbone_from(&mut self, other: &AudioEncoder) {
        self.conv2d1 = other.conv2d1.clone();
        self.conv2d2 = other.conv2d2.clone();
        self.conv2d3 = other.conv2d3.clone();
        self.conv_out = other.conv_out.clone();
        self.layers = other.layers.clone();
        self.ln_post = other.ln_post.clone();
        self.proj1 = other.proj1.clone();
        self.sinusoidal_pe = other.sinusoidal_pe.clone();
    }
}

// ── EncoderCache ────────────────────────────────────────────────────

/// Cached encoder state for incremental (streaming) encoding.
///
/// Stores the post-projection output of fully-completed attention windows.
/// Since windowed attention makes each window independent, completed windows
/// never change and can be reused across streaming steps.
pub struct EncoderCache {
    /// Post-projection outputs for completed windows, each (window_tokens, output_dim).
    pub completed_windows: Vec<Array>,
    /// Number of full mel-frame chunks already committed to completed windows.
    pub committed_chunks: usize,
}

impl EncoderCache {
    pub fn new() -> Self {
        Self {
            completed_windows: Vec::new(),
            committed_chunks: 0,
        }
    }

    pub fn cached_tokens(&self) -> usize {
        self.completed_windows
            .iter()
            .map(|a| a.shape()[0] as usize)
            .sum()
    }
}

impl Default for EncoderCache {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioEncoder {
    /// Incremental encoding for streaming: reuses cached completed windows,
    /// only re-encodes the current (incomplete) window.
    /// Returns (total_tokens, output_dim).
    pub fn encode_incremental(
        &self,
        mel: &Array,
        cache: &mut EncoderCache,
    ) -> Result<Array, Exception> {
        // For streaming incremental encoding, we use the same full encode() path
        // but cache completed window outputs. Since windowed attention makes each
        // window independent, we can skip re-encoding cached windows.
        //
        // Strategy: run the full encode pipeline but intercept at the window level.
        // For simplicity and correctness, we reuse the existing encode() logic
        // and just cache/reuse window-level outputs.

        let num_frames = mel.shape()[1] as usize;
        let chunk_size = self.config.n_window * 2;
        let n_window_infer = self.config.n_window_infer;

        let n_full_chunks = num_frames / chunk_size;
        let chunks_per_window = n_window_infer / chunk_size;

        // How many complete windows from full chunks?
        let num_complete_windows = n_full_chunks / chunks_per_window;
        let committed_windows = cache.completed_windows.len();

        // Encode newly completed windows
        for win_idx in committed_windows..num_complete_windows {
            let window_output = self.encode_window(mel, chunk_size, win_idx, chunks_per_window)?;
            cache.completed_windows.push(window_output);
        }
        cache.committed_chunks = num_complete_windows * chunks_per_window;

        // Encode the partial (remaining) section — everything after complete windows
        let partial_start_frame = num_complete_windows * chunks_per_window * chunk_size;
        let remaining_frames = num_frames - partial_start_frame;

        let partial_output = if remaining_frames > 0 {
            let partial_mel = mel.index((.., partial_start_frame as i32..));
            // Use the same encode() logic for the partial section
            Some(self.encode_section(&partial_mel)?)
        } else {
            None
        };

        // Concatenate cached windows + partial
        let mut all_parts: Vec<&Array> = cache.completed_windows.iter().collect();
        if let Some(ref partial) = partial_output {
            all_parts.push(partial);
        }

        if all_parts.is_empty() {
            return Err(Exception::custom("no audio tokens produced"));
        }

        ops::concatenate_axis(&all_parts, 0)
    }

    /// Encode a single complete attention window (chunks_per_window full chunks).
    /// Returns (window_tokens, output_dim) with output projection applied.
    fn encode_window(
        &self,
        mel: &Array,
        chunk_size: usize,
        win_idx: usize,
        chunks_per_window: usize,
    ) -> Result<Array, Exception> {
        let start_frame = win_idx * chunks_per_window * chunk_size;
        let end_frame = start_frame + chunks_per_window * chunk_size;
        let window_mel = mel.index((.., start_frame as i32..end_frame as i32));
        self.encode_section(&window_mel)
    }

    /// Encode a section of mel spectrogram through the full pipeline:
    /// conv stem → conv_out → sinusoidal PE → transformer → output projection.
    /// Input: (n_mels, n_frames). Output: (n_tokens, output_dim).
    fn encode_section(&self, mel: &Array) -> Result<Array, Exception> {
        let total_frames = mel.shape()[1] as usize;
        let chunk_size = self.config.n_window * 2;

        let n_full_chunks = total_frames / chunk_size;
        let mut chunk_token_lens: Vec<usize> = Vec::new();
        let mut chunk_conv_outputs: Vec<Array> = Vec::new();

        let n_mels = mel.shape()[0];

        // Full chunks
        if n_full_chunks > 0 {
            let full_frames = (n_full_chunks * chunk_size) as i32;
            let full_mel = mel.index((.., ..full_frames));
            let full_mel = full_mel
                .reshape(&[n_mels, n_full_chunks as i32, chunk_size as i32])?
                .transpose_axes(&[1, 0, 2])?;
            let x = ops::expand_dims(&full_mel, -1)?;
            let x = self.apply_conv_stem(&x)?;
            let sh = x.shape();
            let (n, _f_d, t_d, c_d) = (sh[0], sh[1], sh[2], sh[3]);
            let x = x
                .transpose_axes(&[0, 2, 3, 1])?
                .reshape(&[n * t_d, c_d * _f_d])?;
            chunk_conv_outputs.push(x);
            for _ in 0..n_full_chunks {
                chunk_token_lens.push(t_d as usize);
            }
        }

        // Tail chunk
        let tail_start = (n_full_chunks * chunk_size) as i32;
        if (tail_start as usize) < total_frames {
            let tail_mel = mel.index((.., tail_start..));
            let x = ops::expand_dims(&tail_mel, 0)?;
            let x = ops::expand_dims(&x, -1)?;
            let x = self.apply_conv_stem(&x)?;
            let sh = x.shape();
            let (_f_d, t_d, c_d) = (sh[1], sh[2], sh[3]);
            let x = x
                .transpose_axes(&[0, 2, 3, 1])?
                .reshape(&[t_d, c_d * _f_d])?;
            chunk_token_lens.push(t_d as usize);
            chunk_conv_outputs.push(x);
        }

        if chunk_conv_outputs.is_empty() {
            return Err(Exception::custom("no audio in section"));
        }

        let refs: Vec<&Array> = chunk_conv_outputs.iter().collect();
        let x = ops::concatenate_axis(&refs, 0)?;
        let x = self.conv_out.forward(&x)?;

        // Per-chunk sinusoidal PE
        let max_chunk_tokens = *chunk_token_lens.iter().max().unwrap();
        let pe = self.sinusoidal_pe.index((..max_chunk_tokens as i32, ..));
        let mut pe_parts: Vec<Array> = Vec::new();
        for &ct in &chunk_token_lens {
            pe_parts.push(pe.index((..ct as i32, ..)));
        }
        let pe_refs: Vec<&Array> = pe_parts.iter().collect();
        let pe_full = ops::concatenate_axis(&pe_refs, 0)?;
        let x = x.add(&pe_full)?;

        // Transformer (no windowed mask — this is a single section)
        let mut x = ops::expand_dims(&x, 0)?;
        for layer in &self.layers {
            x = layer.forward_with_mask(&x, None)?;
        }
        let x = x.index((0, ..));

        // Output projection
        let x = self.ln_post.forward(&x)?;
        let x = self.proj1.forward(&x)?;
        let x = nn::gelu(&x)?;
        self.proj2.forward(&x)
    }
}

fn create_windowed_mask(seq_len: usize, cu_seqlens: &[usize]) -> Option<Array> {
    if cu_seqlens.len() <= 2 {
        return None;
    }

    let mut mask_data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        let win_i = cu_seqlens
            .windows(2)
            .position(|w| i >= w[0] && i < w[1])
            .unwrap();
        for j in 0..seq_len {
            let win_j = cu_seqlens
                .windows(2)
                .position(|w| j >= w[0] && j < w[1])
                .unwrap();
            if win_i != win_j {
                mask_data[i * seq_len + j] = -1e9;
            }
        }
    }

    let mask = Array::from_slice(&mask_data, &[seq_len as i32, seq_len as i32]);
    Some(ops::expand_dims_axes(&mask, &[0, 1]).unwrap())
}
