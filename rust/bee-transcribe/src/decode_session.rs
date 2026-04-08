//! A decode sub-session: owns audio, encoder cache, and tokens together.
//!
//! Rotation = throw away the old DecodeSession and create a new one.
//! The start_time tracks where this sub-session begins in the timeline.

use bee_qwen3_asr::encoder::EncoderCache;
use bee_qwen3_asr::generate::{self, TokenLogprob};
use bee_qwen3_asr::mel::MelExtractor;
use bee_qwen3_asr::model::Qwen3ASRModel;
use mlx_rs::Array;
use mlx_rs::error::Exception;
use tokenizers::Tokenizer;

use bee_qwen3_asr::forced_aligner::ForcedAligner;

use crate::audio_buffer::{AudioBuffer, Seconds};
use crate::mlx_stuff::clear_mlx_cache;
use crate::text_buffer::{
    self, AlignmentItem, AsrToken, TextBuffer, TokenCount, TokenEntry, TokenId, WordStart,
};

/// A decode sub-session. Replaced wholesale on rotation.
pub struct DecodeSession {
    /// Audio buffer for this sub-session.
    audio: AudioBuffer,
    /// When this sub-session starts in the session timeline.
    start_time: Seconds,
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

    /// Update the rollback window size (for adaptive rollback).
    pub fn set_rollback(&mut self, rollback: TokenCount) {
        self.rollback_tokens = rollback;
    }

    /// Append audio samples to this sub-session.
    pub fn append_audio(&mut self, chunk: &AudioBuffer) {
        self.audio.append(chunk);
    }

    /// Whether this sub-session has any audio.
    pub fn has_audio(&self) -> bool {
        !self.audio.is_empty()
    }

    /// Number of audio samples in this sub-session.
    pub fn audio_len(&self) -> usize {
        self.audio.len()
    }

    /// When this sub-session starts in the absolute timeline.
    pub fn start_time(&self) -> Seconds {
        self.start_time
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
        let merged = Self::merge_tokens(prefix, &generated, &logprobs);

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
    /// Word boundaries are detected via DecodeStream: if a token's
    /// contribution starts with a space or newline, it begins a new word.
    pub fn pending_entries(&self, tokenizer: &Tokenizer) -> Vec<TokenEntry> {
        let text_tokens = self.text_tokens();
        if text_tokens.is_empty() {
            return Vec::new();
        }

        let mut stream = tokenizer.decode_stream(true);
        let mut entries = Vec::with_capacity(text_tokens.len());

        for (i, token) in text_tokens.iter().enumerate() {
            let chunk = stream.step(token.id).ok().flatten();
            let is_word_start = if i == 0 {
                true
            } else {
                chunk
                    .as_ref()
                    .map_or(false, |c| c.starts_with(' ') || c.starts_with('\n'))
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

    /// Commit up to `n` text tokens. Snaps back to a word boundary, runs
    /// forced alignment, and splits self: returns an aligned TextBuffer and
    /// becomes the remainder (leftover audio, fresh cache).
    ///
    /// Returns `None` if nothing can be committed (no complete words, empty
    /// text, alignment failure, etc.).
    pub fn commit(
        &mut self,
        n: TokenCount,
        forced_aligner: &ForcedAligner,
        tokenizer: &Tokenizer,
    ) -> Result<Option<TextBuffer>, Exception> {
        // Build entries from text tokens, snap to word boundary
        let mut entries = TextBuffer::from_entries(self.pending_entries(tokenizer));
        let safe_n = entries.snap_to_word_boundary(n);
        tracing::debug!(
            requested = n.0,
            snapped = safe_n.0,
            total_text_tokens = self.text_tokens().len(),
            "commit: snap to word boundary"
        );
        if safe_n.0 == 0 {
            tracing::debug!("commit: nothing to commit (no complete words)");
            return Ok(None);
        }

        let to_commit = entries.split_off_front(safe_n);
        let commit_ids = to_commit.token_ids();
        let commit_text = tokenizer.decode(&commit_ids, true).unwrap_or_default();
        if commit_text.trim().is_empty() {
            tracing::debug!("commit: empty text after decode, skipping");
            return Ok(None);
        }

        tracing::debug!(
            tokens = safe_n.0,
            text = %commit_text.trim(),
            "commit: aligning text"
        );

        // Guard: can't align against empty audio
        if self.audio.is_empty() {
            tracing::warn!("commit: no audio to align against, skipping");
            return Ok(None);
        }

        // Run forced alignment against our audio
        let items = forced_aligner
            .align(self.audio.samples(), &commit_text)
            .map_err(|e| Exception::custom(format!("aligner: {e}")))?;
        if items.is_empty() {
            tracing::warn!("commit: forced aligner returned no items");
            return Ok(None);
        }

        // Build AlignmentItems and align the buffer (pure function)
        let alignment_items: Vec<AlignmentItem> = items
            .iter()
            .map(|item| AlignmentItem {
                word: item.word.clone(),
                start: Seconds(item.start_time as f64),
                end: Seconds(item.end_time as f64),
            })
            .collect();
        let aligned = text_buffer::align(to_commit, &alignment_items, &self.audio, self.start_time);

        // Rotate: trim audio and drop committed tokens, keep the rest
        // (like v1's Generator::rotate — remaining tokens provide context
        // so the model doesn't think it's starting a new utterance).
        let last_end = Seconds(items.last().unwrap().end_time as f64);
        let new_start = self.start_time + last_end;
        let (_, remaining) = self.audio.split_at(last_end);

        let remaining_text_tokens = self.text_tokens().len() - safe_n.0;
        tracing::info!(
            committed_tokens = safe_n.0,
            remaining_text_tokens,
            old_start = %format!("{:.3}s", self.start_time.0),
            new_start = %format!("{:.3}s", new_start.0),
            audio_before = self.audio.len(),
            audio_after = remaining.len(),
            aligned_words = items.len(),
            "commit: rotation"
        );

        self.audio = remaining;
        self.start_time = new_start;

        // Drop the committed text tokens, keep metadata + remaining text
        let drop_start = self.metadata_end;
        let drop_end = self.metadata_end + safe_n.0;
        self.tokens.drain(drop_start..drop_end);
        // metadata_end stays the same (metadata tokens unchanged)

        self.encoder_cache = EncoderCache::new();
        self.mel_extractor = MelExtractor::new(400, 160, 128, 16000);
        // Don't reset chunk_count — remaining tokens need prefix rollback to work

        Ok(Some(aligned))
    }

    /// Commit all text tokens. Same as `commit` but without a token limit.
    pub fn commit_all(
        &mut self,
        forced_aligner: &ForcedAligner,
        tokenizer: &Tokenizer,
    ) -> Result<Option<TextBuffer>, Exception> {
        let n = TokenCount(self.text_tokens().len());
        self.commit(n, forced_aligner, tokenizer)
    }

    /// Clear all tokens and reset state.
    pub fn clear(&mut self) {
        self.tokens.clear();
        self.metadata_end = 0;
        self.encoder_cache = EncoderCache::new();
        self.chunk_count = 0;
    }

    // ── Internal ────────────────────────────────────────────────────

    /// Compute the fixed prefix for rollback — text tokens only.
    ///
    /// Metadata is never included here: `build_initial_prompt` re-emits the
    /// language header on every step, so including metadata in the prefix
    /// would duplicate it in the model's context.
    fn compute_prefix(&self) -> Option<Vec<AsrToken>> {
        if self.chunk_count < 2 || self.tokens.is_empty() {
            return None;
        }
        let text_tokens = self.text_tokens();
        let text_keep = text_tokens.len().saturating_sub(self.rollback_tokens.0);
        if text_keep == 0 {
            return None;
        }
        Some(text_tokens[..text_keep].to_vec())
    }

    /// Merge prefix tokens + newly generated tokens into a single Vec<AsrToken>.
    ///
    /// The prefix already carries correct logprobs from the previous step.
    /// Generated tokens get fresh logprobs from this step.
    fn merge_tokens(
        prefix: Option<Vec<AsrToken>>,
        generated: &[i32],
        logprobs: &[TokenLogprob],
    ) -> Vec<AsrToken> {
        let mut merged: Vec<AsrToken> = prefix.unwrap_or_default();
        for (i, &token_id) in generated.iter().enumerate() {
            let lp = logprobs.get(i);
            merged.push(AsrToken {
                id: token_id as TokenId,
                logprob: lp.map_or(0.0, |l| l.logprob),
                margin: lp.map_or(0.0, |l| l.margin),
            });
        }
        merged
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
