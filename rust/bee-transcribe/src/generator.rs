//! Autoregressive token generation with mel extraction, encoder caching,
//! and prefix rollback.

use bee_qwen3_asr::encoder::EncoderCache;
use bee_qwen3_asr::generate::TokenLogprob;
use bee_qwen3_asr::mel::MelExtractor;
use bee_qwen3_asr::{generate, model::Qwen3ASRModel};
use mlx_rs::Array;
use mlx_rs::error::Exception;
use tokenizers::Tokenizer;

use crate::mlx_stuff::clear_mlx_cache;
use crate::types::TokenId;

/// Handles mel extraction, incremental encoding, autoregressive generation,
/// and prefix rollback.
pub struct Generator {
    encoder_cache: EncoderCache,
    mel_extractor: MelExtractor,
    raw_token_ids: Vec<TokenId>,
    raw_token_logprobs: Vec<TokenLogprob>,
    rollback_tokens: usize,
}

impl Generator {
    pub fn new(rollback_tokens: usize) -> Self {
        Self {
            encoder_cache: EncoderCache::new(),
            mel_extractor: MelExtractor::new(400, 160, 128, 16000),
            raw_token_ids: Vec::new(),
            raw_token_logprobs: Vec::new(),
            rollback_tokens,
        }
    }

    /// Run one decode step: extract mel, encode audio, generate tokens.
    pub fn decode_step(
        &mut self,
        model: &Qwen3ASRModel,
        tokenizer: &Tokenizer,
        audio: &[f32],
        language: &str,
        chunk_count: usize,
        max_tokens: usize,
    ) -> Result<(), Exception> {
        // Mel extraction
        let (mel_data, n_mels, n_frames) = self
            .mel_extractor
            .extract(audio)
            .map_err(|e| Exception::custom(format!("mel: {e}")))?;
        let mel = Array::from_slice(&mel_data, &[n_mels as i32, n_frames as i32]);

        // Encode audio (incremental)
        let audio_features = model.encode_incremental(&mel, &mut self.encoder_cache)?;
        let audio_features = mlx_rs::ops::expand_dims(&audio_features, 0)?;
        audio_features.eval()?;

        // Build prompt with prefix rollback
        let prefix_ids = self.compute_prefix(chunk_count);
        let mut prompt = generate::build_initial_prompt(
            audio_features.shape()[1] as usize,
            language,
            "",
            tokenizer,
        );
        if let Some(prefix) = &prefix_ids {
            prompt.extend(prefix.iter().map(|&t| t as i32));
        }

        // Generate
        let mut cache = None;
        let (generated, logprobs, _) = generate::prefill_and_decode(
            model,
            &prompt,
            &audio_features,
            &mut cache,
            0,
            max_tokens,
        )?;

        // Combine prefix + generated
        let prefix_len = prefix_ids.as_ref().map_or(0, |p| p.len());
        tracing::debug!(
            "generator: generated={} prefix={prefix_len} prompt_len={}",
            generated.len(),
            prompt.len(),
        );

        let (all_ids, all_logprobs): (Vec<TokenId>, Vec<TokenLogprob>) =
            if let Some(prefix) = prefix_ids {
                let prefix_len = prefix.len();
                let prefix_logprobs = if self.raw_token_logprobs.len() >= prefix_len {
                    self.raw_token_logprobs[..prefix_len].to_vec()
                } else {
                    let mut lps = self.raw_token_logprobs.clone();
                    lps.resize(
                        prefix_len,
                        TokenLogprob {
                            token_id: 0,
                            logprob: 0.0,
                            margin: 0.0,
                        },
                    );
                    lps
                };
                let mut ids = prefix;
                ids.extend(generated.iter().map(|&t| t as TokenId));
                let mut lps = prefix_logprobs;
                lps.extend_from_slice(&logprobs);
                (ids, lps)
            } else {
                (generated.iter().map(|&t| t as TokenId).collect(), logprobs)
            };

        tracing::debug!(
            "generator: total_ids={} (prefix={prefix_len} + generated={})",
            all_ids.len(),
            generated.len(),
        );

        if all_ids.is_empty() && !self.raw_token_ids.is_empty() {
            tracing::debug!(
                "generator: EOS with no output, preserving {} existing tokens",
                self.raw_token_ids.len()
            );
        } else {
            self.raw_token_ids = all_ids;
            self.raw_token_logprobs = all_logprobs;
        }

        drop(cache);
        clear_mlx_cache();

        Ok(())
    }

    /// Compute the fixed prefix to feed back to the model.
    /// Returns None during warm-up (chunk_count <= 2).
    fn compute_prefix(&self, chunk_count: usize) -> Option<Vec<TokenId>> {
        if chunk_count <= 2 || self.raw_token_ids.is_empty() {
            return None;
        }
        let keep = self
            .raw_token_ids
            .len()
            .saturating_sub(self.rollback_tokens);
        if keep == 0 {
            return None;
        }
        Some(self.raw_token_ids[..keep].to_vec())
    }

    /// Number of fixed (non-rollback) tokens.
    #[allow(dead_code)]
    pub fn fixed_token_count(&self) -> usize {
        self.raw_token_ids
            .len()
            .saturating_sub(self.rollback_tokens)
    }

    /// Drop the first `n` raw tokens and reset encoder cache (after commit).
    pub fn rotate(&mut self, n: usize) {
        self.raw_token_ids = self.raw_token_ids[n..].to_vec();
        self.raw_token_logprobs = if self.raw_token_logprobs.len() > n {
            self.raw_token_logprobs[n..].to_vec()
        } else {
            Vec::new()
        };
        self.encoder_cache = EncoderCache::new();
    }

    pub fn raw_token_ids(&self) -> &[TokenId] {
        &self.raw_token_ids
    }

    pub fn raw_token_logprobs(&self) -> &[TokenLogprob] {
        &self.raw_token_logprobs
    }
}
