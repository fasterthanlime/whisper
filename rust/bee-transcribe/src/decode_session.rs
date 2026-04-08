//! A decode sub-session: owns audio, encoder cache, and tokens together.
//!
//! Rotation = throw away the old DecodeSession and create a new one.
//! The start_time tracks where this sub-session begins in the timeline.

use bee_qwen3_asr::encoder::EncoderCache;
use bee_qwen3_asr::generate::{self, TokenLogprob};
use bee_qwen3_asr::mel::MelExtractor;
use bee_qwen3_asr::model::Qwen3ASRModel;
use mlx_rs::error::Exception;
use mlx_rs::Array;
use tokenizers::Tokenizer;

use crate::audio_buffer::{AudioBuffer, Seconds};
use crate::mlx_stuff::clear_mlx_cache;
use crate::text_buffer::{AsrToken, TokenCount, TokenEntry, TokenId, WordStart};

/// A decode sub-session. Replaced wholesale on rotation.
pub struct DecodeSession {
    /// Audio buffer for this sub-session.
    pub audio: AudioBuffer,
    /// When this sub-session starts in the session timeline.
    pub start_time: Seconds,
    /// Encoder cache for incremental encoding.
    encoder_cache: EncoderCache,
    /// Mel spectrogram extractor.
    mel_extractor: MelExtractor,
    /// Current tokens (metadata + text), merged from prefix + generated.
    tokens: Vec<AsrToken>,
    /// Index of the first text token (after `<asr_text>` tag).
    /// Everything before this is metadata.
    metadata_end: usize,
    /// How many recent tokens the model may revise each step.
    rollback_tokens: TokenCount,
    /// How many chunks have been decoded in this sub-session.
    chunk_count: usize,
}

impl DecodeSession {
    /// Create a new decode session.
    pub fn new(audio: AudioBuffer, start_time: Seconds, rollback_tokens: TokenCount) -> Self {
        Self {
            audio,
            start_time,
            encoder_cache: EncoderCache::new(),
            mel_extractor: MelExtractor::new(400, 160, 128, 16000),
            tokens: Vec::new(),
            metadata_end: 0,
            rollback_tokens,
            chunk_count: 0,
        }
    }

    /// The text tokens (after `<asr_text>`).
    pub fn text_tokens(&self) -> &[AsrToken] {
        &self.tokens[self.metadata_end..]
    }

    /// The metadata tokens (before `<asr_text>`).
    pub fn metadata_tokens(&self) -> &[AsrToken] {
        if self.metadata_end > 0 {
            // metadata_end includes the <asr_text> tag itself,
            // so metadata tokens are [0..metadata_end-1]
            &self.tokens[..self.metadata_end.saturating_sub(1)]
        } else {
            &[]
        }
    }

    /// Extract detected language from metadata tokens (e.g. "language English").
    /// Returns `None` if no language found.
    pub fn detected_language(&self, tokenizer: &tokenizers::Tokenizer) -> Option<String> {
        let metadata = self.metadata_tokens();
        if metadata.is_empty() {
            return None;
        }
        let ids: Vec<u32> = metadata.iter().map(|t| t.id).collect();
        let meta = tokenizer.decode(&ids, true).ok()?;
        let lang = meta.trim().strip_prefix("language ")?.trim();
        if lang.is_empty() || lang.eq_ignore_ascii_case("none") {
            None
        } else {
            Some(lang.to_string())
        }
    }

    /// Total token count (metadata + text).
    pub fn total_tokens(&self) -> TokenCount {
        TokenCount(self.tokens.len())
    }

    /// How many chunks have been processed.
    pub fn chunk_count(&self) -> usize {
        self.chunk_count
    }

    /// Run one decode step: mel extraction, encoding, generation.
    /// Updates internal tokens.
    pub fn decode_step(
        &mut self,
        model: &Qwen3ASRModel,
        tokenizer: &Tokenizer,
        language: &str,
        max_tokens: usize,
    ) -> Result<(), Exception> {
        self.chunk_count += 1;

        // Mel extraction
        let (mel_data, n_mels, n_frames) = self
            .mel_extractor
            .extract(self.audio.samples())
            .map_err(|e| Exception::custom(format!("mel: {e}")))?;
        let mel = Array::from_slice(&mel_data, &[n_mels as i32, n_frames as i32]);

        // Encode audio (incremental)
        let audio_features = model.encode_incremental(&mel, &mut self.encoder_cache)?;
        let audio_features = mlx_rs::ops::expand_dims(&audio_features, 0)?;
        audio_features.eval()?;

        // Build prompt with prefix rollback
        let prefix = self.compute_prefix();
        let mut prompt = generate::build_initial_prompt(
            audio_features.shape()[1] as usize,
            language,
            "",
            tokenizer,
        );
        if let Some(ref prefix_tokens) = prefix {
            prompt.extend(prefix_tokens.iter().map(|t| t.id as i32));
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

        // Merge prefix + generated into a single Vec<AsrToken>
        let prefix_len = prefix.as_ref().map_or(0, |p| p.len());
        let merged = self.merge_tokens(prefix, &generated, &logprobs);

        tracing::debug!(
            "decode_session: chunk={} generated={} prefix={prefix_len} total={}",
            self.chunk_count,
            generated.len(),
            merged.len(),
        );

        if merged.is_empty() && !self.tokens.is_empty() {
            tracing::debug!(
                "decode_session: EOS with no output, preserving {} tokens",
                self.tokens.len()
            );
        } else if !merged.is_empty() {
            self.tokens = merged;
            self.recompute_metadata_boundary();
        }

        drop(cache);
        clear_mlx_cache();
        Ok(())
    }

    /// Build pending TokenEntry values from text tokens (for TextBuffer).
    /// These don't have word boundaries yet — that comes from the tokenizer.
    pub fn pending_entries(&self, tokenizer: &Tokenizer) -> Vec<TokenEntry> {
        let text_tokens = self.text_tokens();
        if text_tokens.is_empty() {
            return Vec::new();
        }

        // Determine word boundaries by decoding incrementally and checking
        // for spaces at the start of each token's contribution.
        let ids: Vec<TokenId> = text_tokens.iter().map(|t| t.id).collect();
        let mut entries = Vec::with_capacity(text_tokens.len());

        for (i, token) in text_tokens.iter().enumerate() {
            let is_word_start = if i == 0 {
                true
            } else {
                // Decode with and without this token, check if it introduces a space
                let with = tokenizer.decode(&ids[..=i], true).unwrap_or_default();
                let without = tokenizer.decode(&ids[..i], true).unwrap_or_default();
                let contribution = if with.len() >= without.len() {
                    &with[without.len()..]
                } else {
                    ""
                };
                contribution.starts_with(' ') || contribution.starts_with('\n')
            };

            entries.push(TokenEntry {
                token: *token,
                word: if is_word_start {
                    Some(WordStart { alignment: None })
                } else {
                    None
                },
            });
        }

        entries
    }

    /// Clear all tokens (used after finish_commit when everything is committed).
    pub fn clear(&mut self) {
        self.tokens.clear();
        self.metadata_end = 0;
        self.encoder_cache = EncoderCache::new();
    }

    // ── Internal ────────────────────────────────────────────────────

    /// Compute the fixed prefix for rollback.
    fn compute_prefix(&self) -> Option<Vec<AsrToken>> {
        if self.chunk_count < 2 || self.tokens.is_empty() {
            return None;
        }
        let keep = self.tokens.len().saturating_sub(self.rollback_tokens.0);
        if keep == 0 {
            return None;
        }
        Some(self.tokens[..keep].to_vec())
    }

    /// Merge prefix tokens + newly generated tokens into a single Vec<AsrToken>.
    fn merge_tokens(
        &self,
        prefix: Option<Vec<AsrToken>>,
        generated: &[i32],
        logprobs: &[TokenLogprob],
    ) -> Vec<AsrToken> {
        if let Some(prefix) = prefix {
            let prefix_len = prefix.len();
            // Reuse logprobs from previous run for prefix tokens
            let prefix_logprobs: Vec<AsrToken> = if self.tokens.len() >= prefix_len {
                self.tokens[..prefix_len].to_vec()
            } else {
                let mut p = self.tokens.clone();
                p.resize(
                    prefix_len,
                    AsrToken {
                        id: 0,
                        logprob: 0.0,
                        margin: 0.0,
                    },
                );
                p
            };

            let mut merged = prefix_logprobs;
            for (i, &token_id) in generated.iter().enumerate() {
                let lp = logprobs.get(i).map_or(
                    AsrToken {
                        id: token_id as TokenId,
                        logprob: 0.0,
                        margin: 0.0,
                    },
                    |lp| AsrToken {
                        id: token_id as TokenId,
                        logprob: lp.logprob,
                        margin: lp.margin,
                    },
                );
                merged.push(lp);
            }
            merged
        } else {
            generated
                .iter()
                .enumerate()
                .map(|(i, &token_id)| {
                    let lp = logprobs.get(i);
                    AsrToken {
                        id: token_id as TokenId,
                        logprob: lp.map_or(0.0, |l| l.logprob),
                        margin: lp.map_or(0.0, |l| l.margin),
                    }
                })
                .collect()
        }
    }

    /// Find the <asr_text> boundary in tokens.
    fn recompute_metadata_boundary(&mut self) {
        let asr_text_id = generate::TOK_ASR_TEXT as TokenId;
        self.metadata_end = self
            .tokens
            .iter()
            .position(|t| t.id == asr_text_id)
            .map(|pos| pos + 1) // +1 to skip the tag itself
            .unwrap_or(0);
    }
}
