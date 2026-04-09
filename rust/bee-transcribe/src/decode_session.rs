//! A decode sub-session: owns audio, encoder cache, and tokens together.
//!
//! Rotation = throw away the old DecodeSession and create a new one.
//! The start_time tracks where this sub-session begins in the timeline.

use bee_qwen3_asr::encoder::EncoderCache;
use bee_qwen3_asr::generate::{self, ConfidenceMode, TOP_K, TokenConfidence};
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
use crate::timing::{log_phase, log_phase_chunk, phase_start};

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
    /// First index in `tokens` that came from model generation rather than prompt prefix.
    generated_start: usize,
    /// Decode checkpoints used to estimate a rotation cut without re-discovering
    /// the full boundary from scratch at commit time.
    checkpoints: Vec<DecodeCheckpoint>,
}

#[derive(Debug, Clone, Copy)]
struct DecodeCheckpoint {
    audio_len_samples: usize,
    word_count: usize,
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
            generated_start: 0,
            checkpoints: Vec::new(),
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

    /// Number of leading text tokens that came from the fixed rollback prefix
    /// rather than the most recent generation pass.
    pub fn generated_text_start(&self) -> usize {
        self.generated_start.saturating_sub(self.metadata_end)
    }

    /// How many text tokens can be committed without splitting a word.
    pub fn committable_text_tokens(&self, tokenizer: &Tokenizer, n: TokenCount) -> TokenCount {
        let entries = TextBuffer::from_entries(self.pending_entries(tokenizer));
        entries.snap_to_word_boundary(n)
    }

    /// Run one decode step: mel extraction, encoding, generation.
    /// Updates internal tokens.
    pub fn decode_step(
        &mut self,
        model: &Qwen3ASRModel,
        tokenizer: &Tokenizer,
        language: &str,
        max_tokens: usize,
        confidence_mode: ConfidenceMode,
    ) -> Result<(), Exception> {
        self.chunk_count += 1;
        let decode_total_start = phase_start();

        // Mel extraction
        let mel_start = phase_start();
        let (mel_data, n_mels, n_frames) = self
            .mel_extractor
            .extract(self.audio.samples())
            .map_err(|e| Exception::custom(format!("mel: {e}")))?;
        log_phase_chunk("decode_step", "mel_extract", self.chunk_count, mel_start);
        let mel = Array::from_slice(&mel_data, &[n_mels as i32, n_frames as i32]);

        // Encode audio (incremental)
        let encode_start = phase_start();
        let audio_features = model.encode_incremental(&mel, &mut self.encoder_cache)?;
        let audio_features = mlx_rs::ops::expand_dims(&audio_features, 0)?;
        audio_features.eval()?;
        log_phase_chunk(
            "decode_step",
            "encode_incremental",
            self.chunk_count,
            encode_start,
        );

        // Build prompt with prefix rollback
        let prompt_start = phase_start();
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
        log_phase_chunk(
            "decode_step",
            "build_prompt",
            self.chunk_count,
            prompt_start,
        );

        // Generate
        let generate_start = phase_start();
        let mut cache = None;
        let (generated, logprobs, _) = generate::prefill_and_decode(
            model,
            &prompt,
            &audio_features,
            &mut cache,
            0,
            max_tokens,
            confidence_mode,
        )?;
        log_phase_chunk(
            "decode_step",
            "prefill_and_decode",
            self.chunk_count,
            generate_start,
        );

        // Merge prefix + generated into a single Vec<AsrToken>
        let merge_start = phase_start();
        let prefix_len = prefix.as_ref().map_or(0, |p| p.len());
        let merged = Self::merge_tokens(prefix, &generated, &logprobs);
        log_phase_chunk("decode_step", "merge_tokens", self.chunk_count, merge_start);

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
            self.generated_start = prefix_len;
            self.recompute_metadata_boundary();
            let word_count = self
                .pending_entries(tokenizer)
                .iter()
                .filter(|entry| entry.word.is_some())
                .count();
            self.checkpoints.push(DecodeCheckpoint {
                audio_len_samples: self.audio.len(),
                word_count,
            });
        }

        drop(cache);
        let clear_start = phase_start();
        clear_mlx_cache();
        log_phase_chunk(
            "decode_step",
            "clear_mlx_cache",
            self.chunk_count,
            clear_start,
        );
        log_phase_chunk("decode_step", "total", self.chunk_count, decode_total_start);
        Ok(())
    }

    /// Refresh confidence metadata for a text-token range.
    ///
    /// `text_start..text_end` are offsets into `self.text_tokens()`. This lets
    /// callers rescore only the text they are about to commit instead of the
    /// entire current sub-session.
    pub fn refresh_text_confidence(
        &mut self,
        model: &Qwen3ASRModel,
        tokenizer: &Tokenizer,
        language: &str,
        confidence_mode: ConfidenceMode,
        text_start: usize,
        text_end: usize,
    ) -> Result<(), Exception> {
        let text_len = self.text_tokens().len();
        let text_start = text_start.min(text_len);
        let text_end = text_end.min(text_len);
        if text_start >= text_end {
            return Ok(());
        }

        let refresh_total_start = phase_start();
        let mel_start = phase_start();
        let (mel_data, n_mels, n_frames) = self
            .mel_extractor
            .extract(self.audio.samples())
            .map_err(|e| Exception::custom(format!("mel: {e}")))?;
        log_phase("refresh_confidence", "mel_extract", mel_start);
        let mel = Array::from_slice(&mel_data, &[n_mels as i32, n_frames as i32]);
        let encode_start = phase_start();
        let audio_features = model.encode_incremental(&mel, &mut self.encoder_cache)?;
        let audio_features = mlx_rs::ops::expand_dims(&audio_features, 0)?;
        audio_features.eval()?;
        log_phase("refresh_confidence", "encode_incremental", encode_start);

        let prompt_start = phase_start();
        let mut prompt = generate::build_initial_prompt(
            audio_features.shape()[1] as usize,
            language,
            "",
            tokenizer,
        );
        prompt.extend(self.text_tokens()[..text_start].iter().map(|t| t.id as i32));
        log_phase("refresh_confidence", "build_prompt", prompt_start);

        let continuation: Vec<i32> = self.text_tokens()[text_start..text_end]
            .iter()
            .map(|t| t.id as i32)
            .collect();
        let score_start = phase_start();
        let confidences = generate::score_continuation(
            model,
            &prompt,
            &audio_features,
            &continuation,
            confidence_mode,
        )?;
        log_phase("refresh_confidence", "score_continuation", score_start);

        let apply_start = phase_start();
        let token_start = self.metadata_end + text_start;
        let token_end = token_start + continuation.len();
        for (token, confidence) in self.tokens[token_start..token_end]
            .iter_mut()
            .zip(confidences.into_iter())
        {
            token.concentration = confidence.concentration;
            token.margin = confidence.margin;
            token.alternative_count = confidence.alternative_count;
            token.top_ids = confidence.top_ids.map(|id| id as TokenId);
            token.top_logits = confidence.top_logits;
        }
        log_phase("refresh_confidence", "apply_confidence", apply_start);
        log_phase("refresh_confidence", "total", refresh_total_start);

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
        let commit_total_start = phase_start();
        let chunk_index = self.chunk_count;
        // Build entries from text tokens, snap to word boundary
        let boundary_start = phase_start();
        let mut entries = TextBuffer::from_entries(self.pending_entries(tokenizer));
        let safe_n = entries.snap_to_word_boundary(n);
        log_phase_chunk(
            "commit",
            "snap_to_word_boundary",
            chunk_index,
            boundary_start,
        );
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

        let split_start = phase_start();
        let to_commit = entries.split_off_front(safe_n);
        let commit_ids = to_commit.token_ids();
        let commit_text = tokenizer.decode(&commit_ids, true).unwrap_or_default();
        let committed_word_count = to_commit.words().count();
        log_phase_chunk("commit", "extract_commit_text", chunk_index, split_start);
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

        let align_prefix_end = self
            .estimated_commit_audio_cutoff(committed_word_count)
            .unwrap_or_else(|| self.audio.duration());
        let align_audio = self.audio.slice(crate::audio_buffer::TimeRange::new(
            Seconds::ZERO,
            align_prefix_end,
        ));

        // Run forced alignment against our audio
        let align_start = phase_start();
        let items = forced_aligner
            .align(align_audio.samples(), &commit_text)
            .map_err(|e| Exception::custom(format!("aligner: {e}")))?;
        log_phase_chunk("commit", "forced_align", chunk_index, align_start);
        if items.is_empty() {
            tracing::warn!("commit: forced aligner returned no items");
            return Ok(None);
        }

        // Build AlignmentItems and align the buffer (pure function)
        let align_buffer_start = phase_start();
        let alignment_items: Vec<AlignmentItem> = items
            .iter()
            .map(|item| AlignmentItem {
                word: item.word.clone(),
                start: Seconds(item.start_time as f64),
                end: Seconds(item.end_time as f64),
            })
            .collect();
        let aligned =
            text_buffer::align(to_commit, &alignment_items, &align_audio, self.start_time);
        log_phase_chunk(
            "commit",
            "build_aligned_buffer",
            chunk_index,
            align_buffer_start,
        );

        // Rotate: trim audio and drop committed tokens, keep the rest
        // (like v1's Generator::rotate — remaining tokens provide context
        // so the model doesn't think it's starting a new utterance).
        let rotate_start = phase_start();
        let last_end = Seconds(items.last().unwrap().end_time as f64);
        let trim_at = if align_prefix_end < last_end {
            align_prefix_end
        } else {
            last_end
        };
        let new_start = self.start_time + trim_at;
        let (_, remaining) = self.audio.split_at(trim_at);

        let remaining_text_tokens = self.text_tokens().len() - safe_n.0;
        tracing::info!(
            committed_word_count,
            checkpoints = self.checkpoints.len(),
            committed_tokens = safe_n.0,
            remaining_text_tokens,
            old_start = %format!("{:.3}s", self.start_time.0),
            new_start = %format!("{:.3}s", new_start.0),
            align_prefix_end = %format!("{:.3}s", align_prefix_end.0),
            aligned_end = %format!("{:.3}s", last_end.0),
            trim_at = %format!("{:.3}s", trim_at.0),
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
        self.generated_start = 0;
        self.checkpoints.clear();
        // Don't reset chunk_count — remaining tokens need prefix rollback to work
        log_phase_chunk("commit", "rotate_reset", chunk_index, rotate_start);
        log_phase_chunk("commit", "total", chunk_index, commit_total_start);

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
        self.generated_start = 0;
        self.checkpoints.clear();
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
    /// The prefix already carries correct confidence from the previous step.
    /// Generated tokens get fresh confidence from this step.
    fn merge_tokens(
        prefix: Option<Vec<AsrToken>>,
        generated: &[i32],
        logprobs: &[TokenConfidence],
    ) -> Vec<AsrToken> {
        let mut merged: Vec<AsrToken> = prefix.unwrap_or_default();
        for (i, &token_id) in generated.iter().enumerate() {
            let lp = logprobs.get(i);
            merged.push(AsrToken {
                id: token_id as TokenId,
                concentration: lp.map_or(0.0, |l| l.concentration),
                margin: lp.map_or(0.0, |l| l.margin),
                alternative_count: lp.map_or(0, |l| l.alternative_count),
                top_ids: lp.map_or([0; TOP_K], |l| l.top_ids.map(|id| id as TokenId)),
                top_logits: lp.map_or([0.0; TOP_K], |l| l.top_logits),
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

    fn estimated_commit_audio_cutoff(&self, committed_word_count: usize) -> Option<Seconds> {
        const ROTATION_CHECKPOINT_REWIND: usize = 1;

        let cutoff_index = self
            .checkpoints
            .iter()
            .position(|checkpoint| checkpoint.word_count >= committed_word_count)?;
        let chosen_index = cutoff_index.saturating_sub(ROTATION_CHECKPOINT_REWIND);
        let checkpoint = self.checkpoints.get(chosen_index)?;
        Some(Seconds::from_samples(
            checkpoint.audio_len_samples,
            self.audio.sample_rate(),
        ))
    }
}
