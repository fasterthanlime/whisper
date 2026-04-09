//! A decode sub-session: owns audio, encoder cache, and tokens together.
//!
//! Rotation = throw away the old DecodeSession and create a new one.
//! The start_time tracks where this sub-session begins in the timeline.

use bee_qwen3_asr::encoder::EncoderCache;
use bee_qwen3_asr::forced_aligner::{ForcedAlignItem, ForcedAligner};
use bee_qwen3_asr::generate::{self, ConfidenceMode, TOP_K, TokenConfidence};
use bee_qwen3_asr::mel::MelExtractor;
use bee_qwen3_asr::model::Qwen3ASRModel;
use bee_zipa_mlx::audio::AudioBuffer as ZipaAudioBuffer;
use bee_zipa_mlx::infer::ZipaInference;
use mlx_rs::Array;
use mlx_rs::error::Exception;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use tokenizers::Tokenizer;

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

#[derive(Debug, Clone)]
struct DecodeCheckpoint {
    audio_len_samples: usize,
    text_token_ids: Vec<TokenId>,
}

#[derive(Debug, Clone, Copy)]
struct SelectedCheckpoint {
    index: usize,
    audio_len_samples: usize,
    text_token_count: usize,
}

const MIN_ROTATION_TAIL_AUDIO: Seconds = Seconds(0.8);

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
            self.checkpoints.push(DecodeCheckpoint {
                audio_len_samples: self.audio.len(),
                text_token_ids: self.text_tokens().iter().map(|token| token.id).collect(),
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
        zipa: &ZipaInference,
        tokenizer: &Tokenizer,
    ) -> Result<Option<TextBuffer>, Exception> {
        let commit_total_start = phase_start();
        let chunk_index = self.chunk_count;
        // Build entries from text tokens, snap to word boundary
        let boundary_start = phase_start();
        let Some(selected_checkpoint) = self.select_rotation_checkpoint(tokenizer) else {
            tracing::warn!(
                requested_tokens = n.0,
                total_text_tokens = self.text_tokens().len(),
                checkpoints = self.checkpoints.len(),
                "commit: no compatible token checkpoint for rotation cutoff"
            );
            return Ok(None);
        };

        let checkpoint_tokens = TokenCount(selected_checkpoint.text_token_count);
        let mut entries = TextBuffer::from_entries(self.pending_entries(tokenizer));
        let safe_n = entries.snap_to_word_boundary(checkpoint_tokens);
        log_phase_chunk(
            "commit",
            "snap_to_word_boundary",
            chunk_index,
            boundary_start,
        );
        tracing::debug!(
            requested = n.0,
            checkpoint_tokens = checkpoint_tokens.0,
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

        let align_prefix_end = Seconds::from_samples(
            selected_checkpoint.audio_len_samples,
            self.audio.sample_rate(),
        );
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

        self.log_zipa_cut_report(zipa, &align_audio, &items, &commit_text)?;

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
        let trim_at = align_prefix_end;
        let new_start = self.start_time + trim_at;
        let (committed_audio, remaining) = self.audio.split_at(trim_at);
        let remaining_token_ids: Vec<TokenId> = self.text_tokens()[safe_n.0..]
            .iter()
            .map(|token| token.id)
            .collect();

        if let Err(error) = self.dump_rotation_debug_artifacts(
            tokenizer,
            selected_checkpoint,
            trim_at,
            &commit_ids,
            &remaining_token_ids,
            &commit_text,
            &committed_audio,
            &remaining,
        ) {
            tracing::warn!(%error, "commit: failed to dump rotation debug artifacts");
        }

        let remaining_text_tokens = self.text_tokens().len() - safe_n.0;
        tracing::info!(
            committed_word_count,
            checkpoints = self.checkpoints.len(),
            selected_checkpoint = selected_checkpoint.index,
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
        zipa: &ZipaInference,
        tokenizer: &Tokenizer,
    ) -> Result<Option<TextBuffer>, Exception> {
        let commit_total_start = phase_start();
        let chunk_index = self.chunk_count;

        let split_start = phase_start();
        let to_commit = TextBuffer::from_entries(self.pending_entries(tokenizer));
        let commit_ids = to_commit.token_ids();
        let commit_text = tokenizer.decode(&commit_ids, true).unwrap_or_default();
        log_phase_chunk(
            "commit_all",
            "extract_commit_text",
            chunk_index,
            split_start,
        );
        if commit_text.trim().is_empty() {
            tracing::debug!("commit_all: empty text after decode, skipping");
            return Ok(None);
        }

        if self.audio.is_empty() {
            tracing::warn!("commit_all: no audio to align against, skipping");
            return Ok(None);
        }

        let align_audio = self.audio.clone();
        let align_start = phase_start();
        let items = forced_aligner
            .align(align_audio.samples(), &commit_text)
            .map_err(|e| Exception::custom(format!("aligner: {e}")))?;
        log_phase_chunk("commit_all", "forced_align", chunk_index, align_start);
        if items.is_empty() {
            tracing::warn!("commit_all: forced aligner returned no items");
            return Ok(None);
        }

        self.log_zipa_cut_report(zipa, &align_audio, &items, &commit_text)?;

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
            "commit_all",
            "build_aligned_buffer",
            chunk_index,
            align_buffer_start,
        );

        tracing::info!(
            committed_tokens = self.text_tokens().len(),
            audio_samples = self.audio.len(),
            aligned_words = items.len(),
            "commit_all: final commit without checkpoint rotation"
        );

        self.audio = AudioBuffer::empty(self.audio.sample_rate());
        self.tokens.truncate(self.metadata_end);
        self.encoder_cache = EncoderCache::new();
        self.mel_extractor = MelExtractor::new(400, 160, 128, 16000);
        self.generated_start = 0;
        self.checkpoints.clear();
        log_phase_chunk("commit_all", "total", chunk_index, commit_total_start);

        Ok(Some(aligned))
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

    fn select_rotation_checkpoint(&self, tokenizer: &Tokenizer) -> Option<SelectedCheckpoint> {
        let current: Vec<TokenId> = self.text_tokens().iter().map(|token| token.id).collect();
        if current.is_empty() {
            return None;
        }

        let current_text = tokenizer
            .decode(
                &current.iter().map(|&id| id as u32).collect::<Vec<_>>(),
                true,
            )
            .unwrap_or_else(|_| "<decode failed>".to_string())
            .replace('\n', "\\n");
        let min_tail_samples = MIN_ROTATION_TAIL_AUDIO.to_samples(self.audio.sample_rate());

        let candidates = self
            .checkpoints
            .iter()
            .enumerate()
            .map(|(index, checkpoint)| {
                let lcp_len = common_prefix_len(&checkpoint.text_token_ids, &current);
                let checkpoint_len = checkpoint.text_token_ids.len();
                let stable_len = checkpoint_len.saturating_sub(self.rollback_tokens.0);
                let matches = stable_len > 0 && lcp_len >= stable_len;
                let current_prefix = &current[..checkpoint_len.min(current.len())];
                let checkpoint_text = tokenizer
                    .decode(
                        &checkpoint
                            .text_token_ids
                            .iter()
                            .map(|&id| id as u32)
                            .collect::<Vec<_>>(),
                        true,
                    )
                    .unwrap_or_else(|_| "<decode failed>".to_string())
                    .replace('\n', "\\n");
                let current_prefix_text = tokenizer
                    .decode(
                        &current_prefix
                            .iter()
                            .map(|&id| id as u32)
                            .collect::<Vec<_>>(),
                        true,
                    )
                    .unwrap_or_else(|_| "<decode failed>".to_string())
                    .replace('\n', "\\n");
                let divergence = if lcp_len < checkpoint_len && lcp_len < current.len() {
                    format!(
                        "cp[{}]={} current[{}]={}",
                        lcp_len, checkpoint.text_token_ids[lcp_len], lcp_len, current[lcp_len]
                    )
                } else {
                    "none".to_string()
                };
                (
                    index,
                    checkpoint.audio_len_samples,
                    checkpoint_len,
                    stable_len,
                    lcp_len,
                    self.audio
                        .len()
                        .saturating_sub(checkpoint.audio_len_samples),
                    matches,
                    checkpoint_text,
                    current_prefix_text,
                    divergence,
                )
            })
            .collect::<Vec<_>>();

        tracing::info!(
            rollback_tokens = self.rollback_tokens.0,
            current_tokens = current.len(),
            current_text = %current_text,
            min_tail_audio_secs = MIN_ROTATION_TAIL_AUDIO.0,
            checkpoint_candidates = %candidates
                .iter()
                .map(|(index, audio_len_samples, checkpoint_len, stable_len, lcp_len, tail_samples, matches, checkpoint_text, current_prefix_text, divergence)| {
                    format!(
                        "#{index}@{:.3}s match={} lcp={}/{} stable_len={} tail_samples={} cp_text={checkpoint_text} current_prefix={current_prefix_text} divergence={divergence}",
                        Seconds::from_samples(*audio_len_samples, self.audio.sample_rate()).0,
                        matches,
                        lcp_len,
                        checkpoint_len,
                        stable_len,
                        tail_samples
                    )
                })
                .collect::<Vec<_>>()
                .join(" | "),
            "checkpoint cutoff candidates"
        );

        let compatible_indices = candidates
            .iter()
            .filter(|(_, _, _, _, _, tail_samples, matches, _, _, _)| {
                *matches && *tail_samples >= min_tail_samples
            })
            .map(|(index, _, _, _, _, _, _, _, _, _)| *index)
            .collect::<Vec<_>>();

        let selected_index = match compatible_indices.as_slice() {
            [] => panic!(
                "no compatible checkpoint with enough tail audio: current_text={current_text}; checkpoints={}; min_tail_samples={min_tail_samples}",
                self.checkpoints.len(),
            ),
            [only] if *only == self.checkpoints.len().saturating_sub(1) => panic!(
                "only latest checkpoint is compatible: current_text={current_text}; latest_index={only}"
            ),
            [only] => *only,
            many => {
                let earlier = many[..many.len() - 1].last().copied();
                earlier.unwrap_or_else(|| {
                    panic!(
                        "compatible checkpoints only contained latest/current: current_text={current_text}; compatible={many:?}"
                    )
                })
            }
        };

        let checkpoint = &self.checkpoints[selected_index];
        tracing::info!(
            selected_checkpoint = selected_index,
            selected_audio_len_samples = checkpoint.audio_len_samples,
            selected_text_tokens = checkpoint.text_token_ids.len(),
            selected_audio_secs =
                Seconds::from_samples(checkpoint.audio_len_samples, self.audio.sample_rate()).0,
            "checkpoint selected"
        );

        Some(SelectedCheckpoint {
            index: selected_index,
            audio_len_samples: checkpoint.audio_len_samples,
            text_token_count: checkpoint.text_token_ids.len(),
        })
    }

    fn dump_rotation_debug_artifacts(
        &self,
        tokenizer: &Tokenizer,
        selected_checkpoint: SelectedCheckpoint,
        trim_at: Seconds,
        committed_token_ids: &[TokenId],
        remaining_token_ids: &[TokenId],
        commit_text: &str,
        committed_audio: &AudioBuffer,
        remaining_audio: &AudioBuffer,
    ) -> Result<(), Exception> {
        let round_dir = next_rotation_debug_dir()?;
        fs::create_dir_all(&round_dir)
            .map_err(|e| Exception::custom(format!("create {}: {e}", round_dir.display())))?;

        write_wav_file(&round_dir.join("before.wav"), &self.audio)?;
        write_wav_file(&round_dir.join("committed.wav"), committed_audio)?;
        write_wav_file(&round_dir.join("remaining.wav"), remaining_audio)?;

        let current_ids: Vec<TokenId> = self.text_tokens().iter().map(|token| token.id).collect();
        let current_text = decode_ids(tokenizer, &current_ids);
        let committed_text = decode_ids(tokenizer, committed_token_ids);
        let remaining_text = decode_ids(tokenizer, remaining_token_ids);

        fs::write(
            round_dir.join("summary.txt"),
            format!(
                "selected_checkpoint={}\nselected_audio_secs={:.3}\nselected_text_tokens={}\ntrim_at={:.3}\naudio_before_samples={}\naudio_committed_samples={}\naudio_remaining_samples={}\ncurrent_text={}\ncommitted_text={}\nremaining_text={}\ncommit_text_aligner={}\n",
                selected_checkpoint.index,
                Seconds::from_samples(selected_checkpoint.audio_len_samples, self.audio.sample_rate()).0,
                selected_checkpoint.text_token_count,
                trim_at.0,
                self.audio.len(),
                committed_audio.len(),
                remaining_audio.len(),
                current_text.replace('\n', "\\n"),
                committed_text.replace('\n', "\\n"),
                remaining_text.replace('\n', "\\n"),
                commit_text.replace('\n', "\\n"),
            ),
        )
        .map_err(|e| Exception::custom(format!("write summary: {e}")))?;

        fs::write(
            round_dir.join("current_tokens.txt"),
            render_token_dump(tokenizer, &current_ids),
        )
        .map_err(|e| Exception::custom(format!("write current tokens: {e}")))?;
        fs::write(
            round_dir.join("committed_tokens.txt"),
            render_token_dump(tokenizer, committed_token_ids),
        )
        .map_err(|e| Exception::custom(format!("write committed tokens: {e}")))?;
        fs::write(
            round_dir.join("remaining_tokens.txt"),
            render_token_dump(tokenizer, remaining_token_ids),
        )
        .map_err(|e| Exception::custom(format!("write remaining tokens: {e}")))?;
        fs::write(
            round_dir.join("checkpoints.txt"),
            self.render_checkpoint_ladder(tokenizer, &current_ids),
        )
        .map_err(|e| Exception::custom(format!("write checkpoints: {e}")))?;

        tracing::info!(dir = %round_dir.display(), "commit: wrote rotation debug artifacts");
        Ok(())
    }

    fn render_checkpoint_ladder(&self, tokenizer: &Tokenizer, current: &[TokenId]) -> String {
        self.checkpoints
            .iter()
            .enumerate()
            .map(|(index, checkpoint)| {
                let lcp_len = common_prefix_len(&checkpoint.text_token_ids, current);
                let text = decode_ids(tokenizer, &checkpoint.text_token_ids).replace('\n', "\\n");
                format!(
                    "#{index}\naudio_secs={:.3}\ntoken_count={}\nlcp_with_current={}\ntext={}\nids={:?}\n",
                    Seconds::from_samples(checkpoint.audio_len_samples, self.audio.sample_rate()).0,
                    checkpoint.text_token_ids.len(),
                    lcp_len,
                    text,
                    checkpoint.text_token_ids
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn log_zipa_cut_report(
        &self,
        zipa: &ZipaInference,
        align_audio: &AudioBuffer,
        items: &[ForcedAlignItem],
        commit_text: &str,
    ) -> Result<(), Exception> {
        let zipa_audio = ZipaAudioBuffer {
            samples: align_audio.samples().to_vec(),
            sample_rate_hz: align_audio.sample_rate().0,
        };
        let output = zipa
            .infer_audio(&zipa_audio)
            .map_err(|e| Exception::custom(format!("zipa inference: {e}")))?;

        let frame_count = output.log_probs_len;
        if frame_count == 0 || output.token_ids.is_empty() {
            tracing::info!(
                commit_text = %commit_text.trim(),
                "zipa: no frames available for cut analysis"
            );
            return Ok(());
        }

        let shape = output.log_probs.shape();
        let vocab_size = *shape.last().unwrap_or(&0) as usize;
        if vocab_size == 0 {
            tracing::info!(
                commit_text = %commit_text.trim(),
                "zipa: empty vocab in log_probs"
            );
            return Ok(());
        }
        let flat = output.log_probs.as_slice::<f32>();
        let blank_id = 0usize;
        let seconds_per_frame = align_audio.duration().0 / frame_count as f64;

        let mut runs = Vec::new();
        let mut frame = 0usize;
        while frame < output.token_ids.len() {
            if output.token_ids[frame] != blank_id {
                frame += 1;
                continue;
            }
            let start = frame;
            let mut max_blank = 0.0f32;
            let mut blank_sum = 0.0f32;
            while frame < output.token_ids.len() && output.token_ids[frame] == blank_id {
                let blank_log_prob = flat[frame * vocab_size + blank_id];
                let blank_prob = blank_log_prob.exp();
                max_blank = max_blank.max(blank_prob);
                blank_sum += blank_prob;
                frame += 1;
            }
            let end = frame;
            let len = end - start;
            if len >= 2 {
                let mean_blank = blank_sum / len as f32;
                let cut_time = Seconds((start + end) as f64 * 0.5 * seconds_per_frame);
                runs.push(ZipaCutRun {
                    start_frame: start,
                    end_frame: end,
                    cut_time,
                    mean_blank,
                    max_blank,
                });
            }
        }

        if runs.is_empty() {
            tracing::info!(
                commit_text = %commit_text.trim(),
                zipa_frames = frame_count,
                zipa_tokens = output.tokens.join(" "),
                "zipa: no blank-run cut candidates"
            );
            return Ok(());
        }

        let aligned_end = Seconds(items.last().unwrap().end_time as f64);
        let nearest = runs
            .iter()
            .min_by(|a, b| {
                let da = (a.cut_time.0 - aligned_end.0).abs();
                let db = (b.cut_time.0 - aligned_end.0).abs();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        let mut strongest = runs.clone();
        strongest.sort_by(|a, b| {
            b.mean_blank
                .partial_cmp(&a.mean_blank)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    b.max_blank
                        .partial_cmp(&a.max_blank)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });
        let strongest_summary = strongest
            .iter()
            .take(5)
            .map(|run| {
                format!(
                    "{:.3}s[f{}..{} mean={:.2} max={:.2}]",
                    run.cut_time.0, run.start_frame, run.end_frame, run.mean_blank, run.max_blank
                )
            })
            .collect::<Vec<_>>()
            .join(", ");

        tracing::info!(
            commit_text = %commit_text.trim(),
            zipa_frames = frame_count,
            zipa_frame_ms = seconds_per_frame * 1000.0,
            zipa_phone_count = output.tokens.len(),
            zipa_aligned_end = aligned_end.0,
            zipa_nearest_cut = nearest.cut_time.0,
            zipa_nearest_cut_delta_ms = (nearest.cut_time.0 - aligned_end.0) * 1000.0,
            zipa_strongest_blank_runs = %strongest_summary,
            "zipa cut report"
        );

        Ok(())
    }
}

fn common_prefix_len(left: &[TokenId], right: &[TokenId]) -> usize {
    left.iter()
        .zip(right.iter())
        .take_while(|(left, right)| left == right)
        .count()
}

fn decode_ids(tokenizer: &Tokenizer, ids: &[TokenId]) -> String {
    tokenizer
        .decode(&ids.iter().map(|&id| id).collect::<Vec<_>>(), true)
        .unwrap_or_else(|_| "<decode failed>".to_string())
}

fn render_token_dump(tokenizer: &Tokenizer, ids: &[TokenId]) -> String {
    let decoded = decode_ids(tokenizer, ids).replace('\n', "\\n");
    format!("decoded={decoded}\nids={ids:?}\n")
}

fn next_rotation_debug_dir() -> Result<PathBuf, Exception> {
    static RUN_ROOT: OnceLock<PathBuf> = OnceLock::new();
    static ROUND_COUNTER: AtomicUsize = AtomicUsize::new(0);

    let run_root = RUN_ROOT.get_or_init(|| {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Path::new(".artifacts")
            .join("gpu-hotline")
            .join("rotation-debug")
            .join(format!("run-{stamp}"))
    });

    fs::create_dir_all(run_root)
        .map_err(|e| Exception::custom(format!("create {}: {e}", run_root.display())))?;

    let round = ROUND_COUNTER.fetch_add(1, Ordering::SeqCst) + 1;
    Ok(run_root.join(format!("round-{round:02}")))
}

fn write_wav_file(path: &Path, audio: &AudioBuffer) -> Result<(), Exception> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: audio.sample_rate().0,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .map_err(|e| Exception::custom(format!("create wav {}: {e}", path.display())))?;
    for &sample in audio.samples() {
        let clamped = sample.clamp(-1.0, 1.0);
        let pcm = (clamped * i16::MAX as f32) as i16;
        writer
            .write_sample(pcm)
            .map_err(|e| Exception::custom(format!("write wav {}: {e}", path.display())))?;
    }
    writer
        .finalize()
        .map_err(|e| Exception::custom(format!("finalize wav {}: {e}", path.display())))?;
    Ok(())
}

#[derive(Clone)]
struct ZipaCutRun {
    start_frame: usize,
    end_frame: usize,
    cut_time: Seconds,
    mean_blank: f32,
    max_blank: f32,
}

#[cfg(test)]
mod tests {
    use super::common_prefix_len;

    #[test]
    fn common_prefix_len_stops_at_first_mismatch() {
        assert_eq!(common_prefix_len(&[1, 2, 3, 4], &[1, 2, 9, 4]), 2);
        assert_eq!(common_prefix_len(&[1, 2, 3], &[1, 2, 3, 4]), 3);
        assert_eq!(common_prefix_len(&[9, 2, 3], &[1, 2, 3]), 0);
    }
}
