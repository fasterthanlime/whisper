//! Forced aligner for Qwen3-ASR: produces word-level timestamps.
//!
//! Uses the Qwen3-ForcedAligner-0.6B model (same architecture as ASR but with
//! a timestamp classification head instead of a vocabulary LM head).
//!
//! The aligner takes audio + known text, runs a single forward pass (non-autoregressive),
//! and predicts timestamp buckets at designated [TS] positions in the prompt.

use std::path::Path;

use facet::Facet;
use mlx_rs::error::Exception;
use mlx_rs::module::{Module, ModuleParametersExt};
use mlx_rs::ops;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Array;

use crate::config::{AsrConfig, ThinkerConfig};
use crate::load;
use crate::model::Qwen3ASRModel;

/// One word with its time boundaries from forced alignment.
#[derive(Debug, Clone, Facet)]
pub struct ForcedAlignItem {
    pub word: String,
    pub start_time: f64,
    pub end_time: f64,
}

/// Forced aligner model.
pub struct ForcedAligner {
    model: Qwen3ASRModel,
    mel_extractor: crate::mel::MelExtractor,
    tokenizer: tokenizers::Tokenizer,
    timestamp_token_id: i64,
    timestamp_segment_time: f64,
    config: ThinkerConfig,
}

impl ForcedAligner {
    /// Load a forced aligner from a model directory.
    pub fn load(model_dir: &Path, tokenizer: tokenizers::Tokenizer) -> Result<Self, Exception> {
        let config_str = std::fs::read_to_string(model_dir.join("config.json"))
            .map_err(|e| Exception::custom(format!("read config: {e}")))?;
        let config: AsrConfig = serde_json::from_str(&config_str)
            .map_err(|e| Exception::custom(format!("parse config: {e}")))?;

        let timestamp_token_id = config
            .timestamp_token_id
            .ok_or_else(|| Exception::custom("config missing timestamp_token_id"))?;
        let timestamp_segment_time = config
            .timestamp_segment_time
            .ok_or_else(|| Exception::custom("config missing timestamp_segment_time"))?;

        let thinker = &config.thinker_config;
        let classify_num = thinker
            .classify_num
            .ok_or_else(|| Exception::custom("config missing thinker_config.classify_num"))?;

        // Create model (same architecture as ASR)
        let mut model = Qwen3ASRModel::new(thinker)?;

        // Load weights — the lm_head in the safetensors is [classify_num, hidden_size],
        // different from what Qwen3ASRModel creates (vocab_size). We'll load it separately.
        let stats = load::load_weights(&mut model, model_dir)?;
        model.eval()?;

        log::info!(
            "Aligner loaded: {}/{} keys, classify_num={}, ts_token={}, segment_time={}ms",
            stats.loaded,
            stats.total_keys,
            classify_num,
            timestamp_token_id,
            timestamp_segment_time,
        );

        // load_weights already loaded lm_head (quantized or dense) into model.lm_head.
        // For the aligner, lm_head maps to [classify_num] timestamp buckets instead of vocab.

        let mel_extractor =
            crate::mel::MelExtractor::new(400, 160, thinker.audio_config.num_mel_bins, 16000);

        Ok(ForcedAligner {
            model,
            mel_extractor,
            tokenizer,
            timestamp_token_id,
            timestamp_segment_time,
            config: thinker.clone(),
        })
    }

    /// Run forced alignment: given audio samples and known text, produce word-level timestamps.
    pub fn align(
        &mut self,
        samples: &[f32],
        text: &str,
    ) -> Result<Vec<ForcedAlignItem>, Exception> {
        let words = split_words(text);
        if words.is_empty() {
            return Ok(vec![]);
        }

        // Encode audio
        let (mel_data, n_mels, n_frames) = self
            .mel_extractor
            .extract(samples)
            .map_err(|e| Exception::custom(format!("mel: {e}")))?;
        let mel = Array::from_slice(&mel_data, &[n_mels as i32, n_frames as i32]);
        let audio_features = self.model.encode_audio(&mel)?;
        let n_audio_tokens = audio_features.shape()[0] as usize;

        // Build aligner prompt: [audio_start][pads][audio_end] word [TS][TS] word [TS][TS] ...
        let (input_ids, audio_start_pos) = self.build_prompt(n_audio_tokens, &words)?;
        let seq_len = input_ids.len();

        // Build embeddings with audio injection
        let before_ids: Vec<i32> = input_ids[..audio_start_pos]
            .iter()
            .map(|&x| x as i32)
            .collect();
        let after_ids: Vec<i32> = input_ids[audio_start_pos + n_audio_tokens..]
            .iter()
            .map(|&x| x as i32)
            .collect();

        let before_arr = Array::from_slice(&before_ids, &[1, before_ids.len() as i32]);
        let after_arr = Array::from_slice(&after_ids, &[1, after_ids.len() as i32]);

        let before_emb = self.model.model.embed_tokens.forward(&before_arr)?;
        let after_emb = self.model.model.embed_tokens.forward(&after_arr)?;
        let audio_emb = ops::expand_dims(&audio_features, 0)?; // (1, N, D)

        // Cast audio to match embedding dtype
        let audio_emb = audio_emb.as_dtype(before_emb.dtype())?;

        // Concatenate: [before_emb, audio_emb, after_emb] along seq dim
        let hidden_states = ops::concatenate_axis(&[&before_emb, &audio_emb, &after_emb], 1)?;

        // Position IDs
        let positions: Vec<i32> = (0..seq_len as i32).collect();
        let pos_arr = Array::from_slice(&positions, &[1, 1, seq_len as i32]);
        let position_ids = ops::broadcast_to(&pos_arr, &[1, 3, seq_len as i32])?;

        // Single forward pass through decoder (non-autoregressive)
        let hidden = self.model.model.forward_decoder(
            None,
            Some(&hidden_states),
            &position_ids,
            &mut None, // no KV cache needed for single pass
        )?;

        // Apply aligner classification head (lm_head maps hidden → classify_num buckets)
        let logits = self.model.lm_head.forward(&hidden)?; // (1, seq_len, classify_num)
        let output_ids = mlx_rs::ops::indexing::argmax_axis(logits.index((0, .., ..)), -1, false)?;
        output_ids.eval()?;

        // Extract timestamps at [TS] positions
        let mut raw_timestamps: Vec<f64> = Vec::new();
        let output_vec: Vec<i32> = (0..seq_len)
            .map(|i| output_ids.index(i as i32).item::<i32>())
            .collect();

        for (i, &input_id) in input_ids.iter().enumerate() {
            if input_id == self.timestamp_token_id {
                let bucket = output_vec[i] as f64;
                raw_timestamps.push(bucket * self.timestamp_segment_time);
            }
        }

        // LIS smoothing
        let smoothed = fix_timestamp(&raw_timestamps);

        // Two timestamps per word (start + end)
        let mut items = Vec::with_capacity(words.len());
        for (i, word) in words.iter().enumerate() {
            let start_ms = smoothed.get(i * 2).copied().unwrap_or(0.0);
            let end_ms = smoothed.get(i * 2 + 1).copied().unwrap_or(start_ms);
            items.push(ForcedAlignItem {
                word: word.clone(),
                start_time: (start_ms / 1000.0 * 1000.0).round() / 1000.0,
                end_time: (end_ms / 1000.0 * 1000.0).round() / 1000.0,
            });
        }

        Ok(items)
    }

    fn build_prompt(
        &self,
        n_audio_tokens: usize,
        words: &[String],
    ) -> Result<(Vec<i64>, usize), Exception> {
        let audio_start_token = self.config.audio_start_token_id;
        let audio_pad_token = self.config.audio_token_id;
        let audio_end_token = self.config.audio_end_token_id;

        let mut tokens: Vec<i64> = vec![audio_start_token];
        let audio_start_pos = tokens.len();
        tokens.extend(std::iter::repeat_n(audio_pad_token, n_audio_tokens));
        tokens.push(audio_end_token);

        for word in words {
            let enc = self
                .tokenizer
                .encode(word.as_str(), false)
                .map_err(|e| Exception::custom(format!("encode word '{}': {}", word, e)))?;
            tokens.extend(enc.get_ids().iter().map(|&id| id as i64));
            tokens.push(self.timestamp_token_id);
            tokens.push(self.timestamp_token_id);
        }

        Ok((tokens, audio_start_pos))
    }
}

// ── Word splitting ──────────────────────────────────────────────────────

fn is_cjk_char(ch: char) -> bool {
    let code = ch as u32;
    (0x4E00..=0x9FFF).contains(&code)
        || (0x3400..=0x4DBF).contains(&code)
        || (0x20000..=0x2A6DF).contains(&code)
}

fn split_words(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    for segment in text.split_whitespace() {
        let mut buf = String::new();
        for ch in segment.chars() {
            if is_cjk_char(ch) {
                if !buf.is_empty() {
                    tokens.push(std::mem::take(&mut buf));
                }
                tokens.push(ch.to_string());
            } else {
                buf.push(ch);
            }
        }
        if !buf.is_empty() {
            tokens.push(buf);
        }
    }
    tokens
}

// ── LIS-based timestamp smoothing ───────────────────────────────────────

fn fix_timestamp(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    if n == 0 {
        return vec![];
    }

    let mut dp = vec![1usize; n];
    let mut parent = vec![usize::MAX; n];

    for i in 1..n {
        for j in 0..i {
            if data[j] <= data[i] && dp[j] + 1 > dp[i] {
                dp[i] = dp[j] + 1;
                parent[i] = j;
            }
        }
    }

    let max_length = *dp.iter().max().unwrap();
    let max_idx = dp.iter().position(|&v| v == max_length).unwrap();

    let mut lis_indices = Vec::new();
    let mut idx = max_idx;
    loop {
        lis_indices.push(idx);
        if parent[idx] == usize::MAX {
            break;
        }
        idx = parent[idx];
    }
    lis_indices.reverse();

    let mut is_normal = vec![false; n];
    for &i in &lis_indices {
        is_normal[i] = true;
    }

    let mut result = data.to_vec();
    let mut i = 0;

    while i < n {
        if !is_normal[i] {
            let mut j = i;
            while j < n && !is_normal[j] {
                j += 1;
            }
            let anomaly_count = j - i;

            let left_val = (0..i).rev().find(|&k| is_normal[k]).map(|k| result[k]);
            let right_val = (j..n).find(|&k| is_normal[k]).map(|k| result[k]);

            if anomaly_count <= 2 {
                for (k, value) in result.iter_mut().enumerate().take(j).skip(i) {
                    *value = match (left_val, right_val) {
                        (None, Some(r)) => r,
                        (Some(l), None) => l,
                        (Some(l), Some(r)) => {
                            if (k as isize - (i as isize - 1)) <= (j as isize - k as isize) {
                                l
                            } else {
                                r
                            }
                        }
                        (None, None) => *value,
                    };
                }
            } else {
                match (left_val, right_val) {
                    (Some(l), Some(r)) => {
                        let step = (r - l) / (anomaly_count + 1) as f64;
                        for (k, value) in result.iter_mut().enumerate().take(j).skip(i) {
                            *value = l + step * (k - i + 1) as f64;
                        }
                    }
                    (Some(l), None) => {
                        for value in result.iter_mut().take(j).skip(i) {
                            *value = l;
                        }
                    }
                    (None, Some(r)) => {
                        for value in result.iter_mut().take(j).skip(i) {
                            *value = r;
                        }
                    }
                    (None, None) => {}
                }
            }
            i = j;
        } else {
            i += 1;
        }
    }

    result
}
