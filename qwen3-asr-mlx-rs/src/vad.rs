//! Silero VAD v5 (MLX) — lightweight neural voice activity detection.
//!
//! Architecture: STFT → 4×Conv1d+ReLU encoder → LSTM(128) → Conv1d decoder → sigmoid
//! Input: 512-sample chunks (32ms @ 16kHz) + 64-sample context
//! Output: speech probability 0.0–1.0

use std::path::Path;

use mlx_rs::error::Exception;
use mlx_rs::ops;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Array;

const CHUNK_SIZE: usize = 512;
const CONTEXT_SIZE: usize = 64;
const HOP_LENGTH: i32 = 128;

/// Silero VAD v5 model for streaming speech detection.
pub struct SileroVad {
    // STFT basis (precomputed DFT matrix)
    stft_weight: Array, // [258, 256, 1] — channels-last MLX format

    // Encoder: 4 Conv1d layers
    encoder_weights: Vec<Array>,
    encoder_biases: Vec<Array>,
    encoder_strides: Vec<i32>,

    // LSTM
    lstm_wx: Array,   // [512, 128]
    lstm_wh: Array,   // [512, 128]
    lstm_bias: Array,  // [512]

    // Decoder: Conv1d [128→1, k=1]
    decoder_weight: Array, // [1, 1, 128]
    decoder_bias: Array,   // [1]

    // Streaming state
    h: Option<Array>, // LSTM hidden state
    c: Option<Array>, // LSTM cell state
    context: Vec<f32>, // Last 64 samples for context
}

// SAFETY: MLX arrays are heap-allocated Metal buffers, only accessed sequentially.
unsafe impl Send for SileroVad {}

impl SileroVad {
    /// Load Silero VAD v5 from a model directory containing model.safetensors and config.json.
    pub fn load(model_dir: &Path) -> Result<Self, Exception> {
        let st_path = model_dir.join("model.safetensors");
        let tensors = Array::load_safetensors(&st_path)
            .map_err(|e| Exception::custom(format!("load VAD safetensors: {e}")))?;

        let get = |key: &str| -> Result<Array, Exception> {
            tensors.get(key)
                .cloned()
                .ok_or_else(|| Exception::custom(format!("VAD: missing key {key}")))
        };

        let stft_weight = get("stft.weight")?;

        let mut encoder_weights = Vec::new();
        let mut encoder_biases = Vec::new();
        for i in 0..4 {
            encoder_weights.push(get(&format!("encoder.{i}.weight"))?);
            encoder_biases.push(get(&format!("encoder.{i}.bias"))?);
        }

        // Strides from config: [1, 2, 2, 1]
        let encoder_strides = vec![1, 2, 2, 1];

        let lstm_wx = get("lstm.Wx")?;
        let lstm_wh = get("lstm.Wh")?;
        let lstm_bias = get("lstm.bias")?;

        let decoder_weight = get("decoder.weight")?;
        let decoder_bias = get("decoder.bias")?;

        log::info!(
            "Silero VAD loaded: stft={:?} encoder=4 layers, lstm=[{:?},{:?}], decoder={:?}",
            stft_weight.shape(), lstm_wx.shape(), lstm_wh.shape(), decoder_weight.shape(),
        );

        Ok(SileroVad {
            stft_weight,
            encoder_weights,
            encoder_biases,
            encoder_strides,
            lstm_wx,
            lstm_wh,
            lstm_bias,
            decoder_weight,
            decoder_bias,
            h: None,
            c: None,
            context: vec![0.0; CONTEXT_SIZE],
        })
    }

    /// Process a chunk of audio samples and return speech probability (0.0–1.0).
    ///
    /// For streaming use: call repeatedly with 512-sample chunks.
    /// LSTM state persists between calls.
    pub fn process_chunk(&mut self, samples: &[f32]) -> Result<f32, Exception> {
        assert!(samples.len() == CHUNK_SIZE, "VAD expects {CHUNK_SIZE} samples, got {}", samples.len());

        // Prepend context
        let mut input = Vec::with_capacity(CONTEXT_SIZE + CHUNK_SIZE);
        input.extend_from_slice(&self.context);
        input.extend_from_slice(samples);

        // Update context for next call
        self.context.copy_from_slice(&samples[CHUNK_SIZE - CONTEXT_SIZE..]);

        // Input shape: [1, N, 1] (batch, time, channels)
        let n = input.len() as i32;
        let x = Array::from_slice(&input, &[1, n, 1]);

        // STFT: conv1d with precomputed DFT basis
        let x = ops::conv1d(&x, &self.stft_weight, HOP_LENGTH, 0, 1, 1)?;
        // x shape: [1, T, 258] where 258 = 129 real + 129 imag

        // Magnitude: sqrt(real² + imag²)
        let real = x.index((.., .., ..129));
        let imag = x.index((.., .., 129..));
        let mag = ops::sqrt(&ops::add(&ops::square(&real)?, &ops::square(&imag)?)?)?;
        // mag shape: [1, T, 129]

        // Encoder: 4 Conv1d + ReLU layers
        let mut h = mag;
        for i in 0..4 {
            // Conv1d: padding = kernel_size/2 for 'same-ish' padding
            let padding = 1; // kernel_size=3, padding=1
            h = ops::conv1d(&h, &self.encoder_weights[i], self.encoder_strides[i], padding, 1, 1)?;
            h = ops::add(&h, &self.encoder_biases[i])?;
            h = ops::maximum(&h, &Array::from_f32(0.0))?; // ReLU
        }
        // h shape: [1, T', 128]

        // Flatten time dimension — take the last time step (or mean)
        // For small chunks, T' is typically 1-2 steps. Take the last.
        let t_dim = h.shape()[1] as i32;
        let h = if t_dim > 1 {
            // Take last time step
            h.index((.., (t_dim - 1)..t_dim, ..))
        } else {
            h
        };
        // h shape: [1, 1, 128]

        // Reshape for LSTM: [1, 128]
        let h = h.reshape(&[1, 128])?;

        // LSTM step (manual implementation to use our loaded weights)
        let (new_h, new_c) = self.lstm_step(&h)?;

        self.h = Some(new_h.clone());
        self.c = Some(new_c);

        // Decoder: conv1d [128→1, k=1] + sigmoid
        // Reshape for conv1d: [1, 1, 128]
        let dec_in = new_h.reshape(&[1, 1, 128])?;
        let out = ops::conv1d(&dec_in, &self.decoder_weight, 1, 0, 1, 1)?;
        let out = ops::add(&out, &self.decoder_bias)?;
        let prob = ops::sigmoid(&out)?;
        prob.eval()?;

        // Extract scalar — prob shape is [1, 1, 1]
        let prob_flat = prob.reshape(&[-1])?;
        let prob_val = prob_flat.index(0).item::<f32>();
        Ok(prob_val)
    }

    /// Manual LSTM step using loaded weights.
    fn lstm_step(&self, x: &Array) -> Result<(Array, Array), Exception> {
        // gates = x @ Wx.T + h @ Wh.T + bias
        let xw = ops::matmul(x, &self.lstm_wx.t())?;
        let gates = if let Some(ref h) = self.h {
            let hw = ops::matmul(h, &self.lstm_wh.t())?;
            ops::add(&ops::add(&xw, &hw)?, &self.lstm_bias)?
        } else {
            ops::add(&xw, &self.lstm_bias)?
        };

        // Split into 4 gates: [input, forget, cell_candidate, output] each [1, 128]
        let chunks = ops::split(&gates, 4, -1)?;
        let i_gate = ops::sigmoid(&chunks[0])?;
        let f_gate = ops::sigmoid(&chunks[1])?;
        let g_gate = ops::tanh(&chunks[2])?;
        let o_gate = ops::sigmoid(&chunks[3])?;

        // Cell update
        let new_c = if let Some(ref c) = self.c {
            ops::add(&ops::multiply(&f_gate, c)?, &ops::multiply(&i_gate, &g_gate)?)?
        } else {
            ops::multiply(&i_gate, &g_gate)?
        };

        // Hidden update
        let new_h = ops::multiply(&o_gate, &ops::tanh(&new_c)?)?;

        Ok((new_h, new_c))
    }

    /// Process a larger audio buffer, returning the max speech probability
    /// across all 512-sample sub-chunks.
    pub fn process_audio(&mut self, samples: &[f32]) -> Result<f32, Exception> {
        let mut max_prob: f32 = 0.0;
        let mut offset = 0;
        while offset + CHUNK_SIZE <= samples.len() {
            let prob = self.process_chunk(&samples[offset..offset + CHUNK_SIZE])?;
            max_prob = max_prob.max(prob);
            offset += CHUNK_SIZE;
        }
        // If no complete chunks were processed, return 0
        if offset == 0 {
            return Ok(0.0);
        }
        Ok(max_prob)
    }

    /// Reset streaming state (LSTM hidden/cell + context buffer).
    pub fn reset(&mut self) {
        self.h = None;
        self.c = None;
        self.context = vec![0.0; CONTEXT_SIZE];
    }
}
