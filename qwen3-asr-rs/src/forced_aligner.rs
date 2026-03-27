use anyhow::Context;
use candle_core::{DType, Device, Tensor};
use log::info;
use std::path::Path;
use std::sync::Mutex;

use crate::config::AsrConfig;
use crate::decoder::{compute_mrope_cos_sin, create_causal_mask, KvCache, TextDecoder};
use crate::encoder::AudioEncoder;
use crate::error::AsrError;
use crate::inference::{load_safetensors_weights, MEL_SAMPLE_RATE};
use crate::linear::LinearW;
use crate::mel::MelExtractor;
use crate::weights::Weights;

const N_FFT: usize = 400;
const HOP_LENGTH: usize = 160;

/// One word with its time boundaries from forced alignment.
#[derive(Debug, Clone)]
pub struct ForcedAlignItem {
    pub word: String,
    /// Start time in seconds.
    pub start_time: f64,
    /// End time in seconds.
    pub end_time: f64,
}

struct ForcedAlignerInner {
    audio_encoder: AudioEncoder,
    text_decoder: TextDecoder,
    aligner_lm_head: LinearW,
    mel_extractor: MelExtractor,
    tokenizer: tokenizers::Tokenizer,
    config: AsrConfig,
    device: Device,
    timestamp_token_id: i64,
    timestamp_segment_time: f64,
}

// SAFETY: same reasoning as AsrInferenceInner — Metal tensor pointers are
// heap-allocated via Arc, not thread-local. Concurrent access prevented by Mutex.
unsafe impl Send for ForcedAlignerInner {}

pub struct ForcedAligner {
    inner: Mutex<ForcedAlignerInner>,
}

impl ForcedAligner {
    /// Load a ForcedAligner from a local model directory.
    pub fn load(model_dir: &Path, device: Device) -> crate::Result<Self> {
        info!("Loading forced aligner config...");
        let config = AsrConfig::from_file(&model_dir.join("config.json"))
            .context("load config")
            .map_err(AsrError::ModelLoad)?;

        let timestamp_token_id = config.timestamp_token_id.ok_or_else(|| {
            AsrError::ModelLoad(anyhow::anyhow!(
                "config.json missing `timestamp_token_id` — is this a ForcedAligner model?"
            ))
        })?;
        let timestamp_segment_time = config.timestamp_segment_time.ok_or_else(|| {
            AsrError::ModelLoad(anyhow::anyhow!(
                "config.json missing `timestamp_segment_time`"
            ))
        })?;

        info!("Loading weights...");
        let mut weights = load_safetensors_weights(model_dir, &device)
            .context("load safetensors weights")
            .map_err(AsrError::ModelLoad)?;
        info!(
            "Loaded {} weight tensors",
            match &weights {
                Weights::Dense(m) => m.len(),
                _ => 0,
            }
        );
        weights.maybe_convert_for_cpu(&device);

        // classify_num: prefer config, fall back to lm_head output dimension
        let classify_num = config.classify_num.unwrap_or_else(|| {
            let shape = weights
                .get_tensor("thinker.lm_head.weight")
                .map(|t| t.dims().to_vec())
                .unwrap_or_default();
            let n = shape.first().copied().unwrap_or(5000);
            info!(
                "classify_num not in config, inferred {} from lm_head shape {:?}",
                n, shape
            );
            n
        });

        info!(
            "ForcedAligner: classify_num={}, ts_token={}, segment_time={}ms",
            classify_num, timestamp_token_id, timestamp_segment_time
        );

        info!("Loading tokenizer...");
        let tokenizer_path = model_dir.join("tokenizer.json");
        if !tokenizer_path.exists() {
            info!("tokenizer.json not found, building from vocab.json + merges.txt...");
            let vocab = std::fs::read_to_string(model_dir.join("vocab.json"))
                .context("read vocab.json")
                .map_err(AsrError::ModelLoad)?;
            let merges = std::fs::read_to_string(model_dir.join("merges.txt"))
                .context("read merges.txt")
                .map_err(AsrError::ModelLoad)?;
            let tok_config = std::fs::read_to_string(model_dir.join("tokenizer_config.json"))
                .context("read tokenizer_config.json")
                .map_err(AsrError::ModelLoad)?;
            let tok_json =
                crate::tokenizer_build::build_qwen3_tokenizer_json(&vocab, &merges, &tok_config)
                    .context("build tokenizer.json")
                    .map_err(AsrError::ModelLoad)?;
            std::fs::write(&tokenizer_path, &tok_json)
                .context("write tokenizer.json")
                .map_err(AsrError::ModelLoad)?;
        }
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("tokenizer load failed: {}", e))
            .map_err(AsrError::ModelLoad)?;

        info!("Loading audio encoder...");
        let audio_encoder = AudioEncoder::load(
            &weights,
            "thinker.audio_tower",
            &config.thinker_config.audio_config,
            &device,
        )
        .context("load audio encoder")
        .map_err(AsrError::ModelLoad)?;

        info!("Loading text decoder...");
        let text_decoder = TextDecoder::load(
            &weights,
            "thinker.model",
            &config.thinker_config.text_config,
        )
        .context("load text decoder")
        .map_err(AsrError::ModelLoad)?;

        info!("Loading aligner lm_head...");
        let lm_head_weight = weights
            .get_tensor("thinker.lm_head.weight")
            .context("load thinker.lm_head.weight")
            .map_err(AsrError::ModelLoad)?;
        let lm_head_dims = lm_head_weight.dims().to_vec();
        info!("Aligner lm_head shape: {:?}", lm_head_dims);
        let aligner_lm_head = LinearW::new(lm_head_weight, None);

        let mel_extractor = MelExtractor::new(
            N_FFT,
            HOP_LENGTH,
            config.thinker_config.audio_config.num_mel_bins,
            MEL_SAMPLE_RATE,
        );

        info!("ForcedAligner loaded successfully.");

        let inner = ForcedAlignerInner {
            audio_encoder,
            text_decoder,
            aligner_lm_head,
            mel_extractor,
            tokenizer,
            config,
            device,
            timestamp_token_id,
            timestamp_segment_time,
        };

        Ok(ForcedAligner {
            inner: Mutex::new(inner),
        })
    }

    /// Download a ForcedAligner model from HuggingFace Hub and load it.
    #[cfg(feature = "hub")]
    pub fn from_pretrained(
        model_id: &str,
        cache_dir: &Path,
        device: Device,
    ) -> crate::Result<Self> {
        let model_dir =
            crate::hub::ensure_model_cached(model_id, cache_dir).map_err(AsrError::ModelLoad)?;
        Self::load(&model_dir, device)
    }

    /// Run forced alignment on audio samples with known text.
    ///
    /// `samples` must be 16 kHz mono f32.
    /// Returns one [`ForcedAlignItem`] per word with start/end times in seconds.
    pub fn align(&self, samples: &[f32], text: &str) -> crate::Result<Vec<ForcedAlignItem>> {
        let inner = self
            .inner
            .lock()
            .map_err(|_| AsrError::Inference(anyhow::anyhow!("mutex poisoned")))?;
        inner.align(samples, text).map_err(AsrError::Inference)
    }
}

impl ForcedAlignerInner {
    fn align(&self, samples: &[f32], text: &str) -> anyhow::Result<Vec<ForcedAlignItem>> {
        let words = split_words(text);
        if words.is_empty() {
            return Ok(vec![]);
        }

        // Encode audio
        let audio_embeds = self.encode_audio(samples)?;
        let num_audio_tokens = audio_embeds.dims()[0];
        info!("Audio tokens: {}", num_audio_tokens);

        // Build prompt (no chat template — raw audio + word + timestamp tokens)
        let (input_ids, audio_start_pos) = self.build_aligner_prompt(num_audio_tokens, &words)?;
        let seq_len = input_ids.len();
        info!("Prompt length: {} tokens ({} words)", seq_len, words.len());

        // Build embeddings with audio injection
        let before_ids: Vec<i64> = input_ids[..audio_start_pos].to_vec();
        let after_ids: Vec<i64> = input_ids[audio_start_pos + num_audio_tokens..].to_vec();

        let before_t =
            Tensor::from_vec(before_ids, (audio_start_pos,), &self.device)?.to_dtype(DType::U32)?;
        let after_t = Tensor::from_vec(
            after_ids,
            (seq_len - audio_start_pos - num_audio_tokens,),
            &self.device,
        )?
        .to_dtype(DType::U32)?;

        let before_emb = self.text_decoder.embed(&before_t)?;
        let after_emb = self.text_decoder.embed(&after_t)?;
        let audio_emb = audio_embeds.to_dtype(before_emb.dtype())?;

        let hidden_states = Tensor::cat(&[&before_emb, &audio_emb, &after_emb], 0)?.unsqueeze(0)?;

        // MRoPE cos/sin
        let text_cfg = &self.config.thinker_config.text_config;
        let all_pos: Vec<i64> = (0..seq_len as i64).collect();
        let full_ids: [Vec<i64>; 3] = [all_pos.clone(), all_pos.clone(), all_pos];
        let (cos, sin) = compute_mrope_cos_sin(
            &full_ids,
            text_cfg.head_dim,
            text_cfg.rope_theta,
            &text_cfg.mrope_section(),
            text_cfg.mrope_interleaved(),
            &self.device,
        )?;

        // Single forward pass (non-autoregressive)
        let mask = create_causal_mask(seq_len, 0, &self.device)?;
        let mut kv_cache = KvCache::new(text_cfg.num_hidden_layers);

        let hidden = self.text_decoder.forward_hidden(
            &hidden_states,
            &cos,
            &sin,
            &mut kv_cache,
            Some(&mask),
        )?;
        let logits = self.aligner_lm_head.forward(&hidden)?; // [1, seq_len, classify_num]
        let output_ids = logits.argmax(2)?.squeeze(0)?; // [seq_len]
        let output_ids_vec = output_ids.to_vec1::<u32>()?;

        // Extract predictions at timestamp positions
        let mut raw_timestamps: Vec<f64> = Vec::new();
        for (i, &input_id) in input_ids.iter().enumerate() {
            if input_id == self.timestamp_token_id {
                let bucket = output_ids_vec[i] as f64;
                raw_timestamps.push(bucket * self.timestamp_segment_time);
            }
        }

        info!(
            "Extracted {} timestamp predictions for {} words",
            raw_timestamps.len(),
            words.len()
        );

        // LIS smoothing
        let smoothed = fix_timestamp(&raw_timestamps);

        // Two timestamps per word (start + end)
        let mut items = Vec::with_capacity(words.len());
        for (i, word) in words.iter().enumerate() {
            let start_ms = smoothed[i * 2];
            let end_ms = smoothed[i * 2 + 1];
            items.push(ForcedAlignItem {
                word: word.clone(),
                start_time: (start_ms / 1000.0 * 1000.0).round() / 1000.0,
                end_time: (end_ms / 1000.0 * 1000.0).round() / 1000.0,
            });
        }

        Ok(items)
    }

    fn encode_audio(&self, samples: &[f32]) -> anyhow::Result<Tensor> {
        let (mel_data, n_mels, n_frames) = self.mel_extractor.extract(samples)?;
        let mel = Tensor::from_vec(mel_data, (n_mels, n_frames), &self.device)?;
        self.audio_encoder.forward(&mel)
    }

    /// Build the aligner prompt: `[AUDIO_START] [pad×N] [AUDIO_END] word_tokens [TS][TS] ...`
    ///
    /// No chat template wrapping — the ForcedAligner processor feeds text directly
    /// to the tokenizer without applying im_start/system/user/assistant markers.
    fn build_aligner_prompt(
        &self,
        num_audio_tokens: usize,
        words: &[String],
    ) -> anyhow::Result<(Vec<i64>, usize)> {
        let cfg = &self.config.thinker_config;

        let mut tokens: Vec<i64> = vec![cfg.audio_start_token_id];
        let audio_start_pos = tokens.len();
        tokens.extend(std::iter::repeat_n(cfg.audio_token_id, num_audio_tokens));
        tokens.push(cfg.audio_end_token_id);

        for word in words {
            let enc = self
                .tokenizer
                .encode(word.as_str(), false)
                .map_err(|e| anyhow::anyhow!("encode word '{}': {}", word, e))?;
            tokens.extend(enc.get_ids().iter().map(|&id| id as i64));
            tokens.push(self.timestamp_token_id);
            tokens.push(self.timestamp_token_id);
        }

        Ok((tokens, audio_start_pos))
    }
}

// ─── Word splitting ─────────────────────────────────────────────────────────

fn is_cjk_char(ch: char) -> bool {
    let code = ch as u32;
    (0x4E00..=0x9FFF).contains(&code)
        || (0x3400..=0x4DBF).contains(&code)
        || (0x20000..=0x2A6DF).contains(&code)
        || (0x2A700..=0x2B73F).contains(&code)
        || (0x2B740..=0x2B81F).contains(&code)
        || (0x2B820..=0x2CEAF).contains(&code)
        || (0xF900..=0xFAFF).contains(&code)
}

/// Split text into words, handling space-separated languages and CJK characters.
/// Keeps all characters (hyphens, dots, etc.) — the aligner model can handle them.
fn split_words(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    for segment in text.split_whitespace() {
        if segment.is_empty() {
            continue;
        }
        let cleaned = segment;
        // Split CJK characters into individual tokens
        let mut buf = String::new();
        for ch in cleaned.chars() {
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

// ─── LIS-based timestamp smoothing ──────────────────────────────────────────

/// Smooth non-monotonic timestamp predictions using Longest Increasing
/// Subsequence (LIS). Anomalous values are interpolated from their nearest
/// normal neighbours.
fn fix_timestamp(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    if n == 0 {
        return vec![];
    }

    // Compute LIS via O(n²) DP
    let mut dp = vec![1usize; n];
    let mut parent = vec![usize::MAX; n]; // MAX = no parent

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

    // Trace back to find LIS indices
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
            // Find the run of anomalous values
            let mut j = i;
            while j < n && !is_normal[j] {
                j += 1;
            }
            let anomaly_count = j - i;

            // Find nearest normal neighbours
            let left_val = (0..i).rev().find(|&k| is_normal[k]).map(|k| result[k]);
            let right_val = (j..n).find(|&k| is_normal[k]).map(|k| result[k]);

            if anomaly_count <= 2 {
                // Snap to nearest normal neighbour
                for k in i..j {
                    result[k] = match (left_val, right_val) {
                        (None, Some(r)) => r,
                        (Some(l), None) => l,
                        (Some(l), Some(r)) => {
                            if (k as isize - (i as isize - 1)) <= (j as isize - k as isize) {
                                l
                            } else {
                                r
                            }
                        }
                        (None, None) => result[k], // shouldn't happen with LIS
                    };
                }
            } else {
                // Linearly interpolate
                match (left_val, right_val) {
                    (Some(l), Some(r)) => {
                        let step = (r - l) / (anomaly_count + 1) as f64;
                        for k in i..j {
                            result[k] = l + step * (k - i + 1) as f64;
                        }
                    }
                    (Some(l), None) => {
                        for k in i..j {
                            result[k] = l;
                        }
                    }
                    (None, Some(r)) => {
                        for k in i..j {
                            result[k] = r;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_words_english() {
        let words = split_words("Hello, world! This is a test.");
        assert_eq!(words, vec!["Hello", "world", "This", "is", "a", "test"]);
    }

    #[test]
    fn test_split_words_apostrophe() {
        let words = split_words("don't won't");
        assert_eq!(words, vec!["don't", "won't"]);
    }

    #[test]
    fn test_split_words_cjk() {
        let words = split_words("你好世界");
        assert_eq!(words, vec!["你", "好", "世", "界"]);
    }

    #[test]
    fn test_split_words_mixed() {
        let words = split_words("hello你好world");
        assert_eq!(words, vec!["hello", "你", "好", "world"]);
    }

    #[test]
    fn test_split_words_empty() {
        let words = split_words("   ");
        assert!(words.is_empty());
    }

    #[test]
    fn test_fix_timestamp_monotonic() {
        let data = vec![0.0, 100.0, 200.0, 300.0];
        let result = fix_timestamp(&data);
        assert_eq!(result, data);
    }

    #[test]
    fn test_fix_timestamp_single_anomaly() {
        // Index 2 is anomalous (500 breaks monotonicity with 200 following)
        let data = vec![0.0, 100.0, 500.0, 200.0, 300.0, 400.0];
        let result = fix_timestamp(&data);
        // 500 should be corrected; the rest should remain
        assert!(result[0] <= result[1]);
        assert!(result[1] <= result[2]);
        assert!(result[2] <= result[3]);
        assert!(result[3] <= result[4]);
        assert!(result[4] <= result[5]);
    }

    #[test]
    fn test_fix_timestamp_empty() {
        let result = fix_timestamp(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_fix_timestamp_single() {
        let result = fix_timestamp(&[42.0]);
        assert_eq!(result, vec![42.0]);
    }
}
