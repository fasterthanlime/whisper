//! New transcription session built on foundation types.
//!
//! AudioFilterChain → DecodeSession → TextBuffer → forced alignment.
//! Rotation = new DecodeSession with leftover audio.
//! Two TextBuffers: committed (aligned, stable) and pending (volatile).

use bee_qwen3_asr::forced_aligner::ForcedAligner;
use bee_qwen3_asr::model::Qwen3ASRModel;
use bee_types::AlignedWord;
use bee_vad::SileroVad;
use mlx_rs::error::Exception;
use tokenizers::Tokenizer;

use crate::audio_buffer::{AudioBuffer, SampleRate, Seconds};
use crate::audio_filter::{self, AudioFilterChain};
use crate::corrector::Corrector;
use crate::decode_session::DecodeSession;
use crate::text_buffer::{self, TextBuffer, TokenCount, TokenEntry, TokenId};
use crate::types::{SessionOptions, Update};
use crate::{FinishResult, SharedCorrectionEngine};

/// A committed chunk waiting for right-context before correction.
struct BufferedCommit {
    raw_text: String,
    words: Vec<AlignedWord>,
}

pub struct SessionV2<'a> {
    model: &'a Qwen3ASRModel,
    tokenizer: &'a Tokenizer,
    forced_aligner: &'a ForcedAligner,

    filters: AudioFilterChain,
    decode: DecodeSession,
    committed: TextBuffer,
    pending: TextBuffer,
    detected_language: String,

    /// Correction state: shared engine + per-session corrector.
    correction: Option<(SharedCorrectionEngine, Corrector)>,
    /// One-commit-late buffer: waiting for right context before correction.
    buffered_commit: Option<BufferedCommit>,
    /// Raw words from the chunk before the buffered one (left context).
    prev_raw_words: Vec<AlignedWord>,

    options: SessionOptions,
    incoming: Vec<f32>,
    chunk_size_samples: usize,
}

impl<'a> SessionV2<'a> {
    pub fn new(
        model: &'a Qwen3ASRModel,
        tokenizer: &'a Tokenizer,
        forced_aligner: &'a ForcedAligner,
        vad: SileroVad,
        options: SessionOptions,
        correction: Option<(SharedCorrectionEngine, Corrector)>,
    ) -> Self {
        let chunk_size_samples = (options.chunk_duration * 16000.0) as usize;
        assert!(chunk_size_samples > 0, "chunk_duration too small");
        Self {
            model,
            tokenizer,
            forced_aligner,
            filters: audio_filter::default_filter_chain(vad, options.vad_threshold),
            decode: DecodeSession::new(
                AudioBuffer::empty(SampleRate::HZ_16000),
                Seconds::ZERO,
                TokenCount(options.rollback_tokens),
            ),
            committed: TextBuffer::new(),
            pending: TextBuffer::new(),
            detected_language: String::new(),
            correction,
            buffered_commit: None,
            prev_raw_words: Vec::new(),
            options,
            incoming: Vec::new(),
            chunk_size_samples,
        }
    }

    pub fn feed(&mut self, samples: &[f32]) -> Result<Option<Update>, Exception> {
        self.incoming.extend_from_slice(samples);
        tracing::trace!(
            incoming_len = self.incoming.len(),
            chunk_size = self.chunk_size_samples,
            "feed: buffering audio"
        );

        let mut did_decode = false;
        while self.incoming.len() >= self.chunk_size_samples {
            let chunk_samples: Vec<f32> =
                self.incoming.drain(..self.chunk_size_samples).collect();
            let chunk = AudioBuffer::new(chunk_samples, SampleRate::HZ_16000);

            if let Some(chunk) = self.filters.process(chunk) {
                tracing::debug!(
                    audio_samples = chunk.len(),
                    "feed: speech chunk passed filters"
                );
                self.decode.append_audio(&chunk);
                self.decode_and_maybe_commit(self.options.max_tokens_streaming)?;
                did_decode = true;
            } else {
                tracing::trace!("feed: chunk filtered out (silence/VAD)");
            }
        }

        if did_decode {
            Ok(Some(self.make_update()))
        } else {
            Ok(None)
        }
    }

    pub fn finish(mut self) -> Result<FinishResult, Exception> {
        tracing::info!(
            incoming = self.incoming.len(),
            committed_tokens = self.committed.len().0,
            pending_tokens = self.pending.len().0,
            "finish: finalizing session"
        );

        if !self.incoming.is_empty() {
            let remaining = std::mem::take(&mut self.incoming);
            let chunk = AudioBuffer::new(remaining, SampleRate::HZ_16000);
            if let Some(chunk) = self.filters.process(chunk) {
                self.decode.append_audio(&chunk);
            }
        }

        if self.decode.has_audio() {
            tracing::debug!(max_tokens = self.options.max_tokens_final, "finish: final decode");
            self.decode_and_maybe_commit(self.options.max_tokens_final)?;
        }

        // Commit everything remaining
        if !self.pending.is_empty() {
            tracing::debug!(
                pending_tokens = self.pending.len().0,
                "finish: committing remaining pending tokens"
            );
            debug_assert!(
                self.decode.has_audio(),
                "pending text exists without decode audio"
            );
            if let Some(aligned) = self.decode.commit_all(
                self.forced_aligner,
                self.tokenizer,
            )? {
                let final_words: Vec<AlignedWord> = aligned.words()
                    .filter_map(|entries| Self::word_to_aligned(self.tokenizer, entries))
                    .collect();

                // Flush previous buffered commit with these final words as right context
                self.flush_buffered_commit(&final_words);

                // Correct this final chunk too (no right context)
                let raw_text = final_words.iter().map(|w| w.word.as_str()).collect::<Vec<_>>().join(" ");
                if let Some((ref engine_arc, ref mut corrector)) = self.correction {
                    let mut engine = engine_arc.lock().expect("correction engine lock poisoned");
                    corrector.process_chunk_with_context(
                        &mut engine,
                        &raw_text,
                        &final_words,
                        &self.prev_raw_words,
                        &[],
                        self.options.app_id.as_deref(),
                    );
                }

                self.committed.append(aligned);
                self.pending.replace(self.decode.pending_entries(self.tokenizer));
            }
        } else {
            // No pending tokens, but may still have a buffered commit to flush
            self.flush_buffered_commit(&[]);
        }

        let update = self.make_update();
        tracing::info!(
            committed_tokens = self.committed.len().0,
            text_len = update.text.len(),
            alignments = update.alignments.len(),
            "finish: done"
        );
        let corrector = self.correction.take().map(|(_, c)| c);
        Ok(FinishResult { update, corrector })
    }

    // ── Internal ────────────────────────────────────────────────────

    fn decode_and_maybe_commit(&mut self, max_tokens: usize) -> Result<(), Exception> {
        self.decode.decode_step(
            self.model,
            self.tokenizer,
            self.options.language.as_str(),
            max_tokens,
        )?;

        if let Some(lang) = self.decode.detected_language(self.tokenizer) {
            tracing::debug!(language = %lang, "detected language");
            self.detected_language = lang;
        }

        // Refresh pending from decode session's current text tokens
        self.pending
            .replace(self.decode.pending_entries(self.tokenizer));

        // Check if enough fixed tokens to try committing
        let text_count = self.decode.text_tokens().len();
        let fixed = text_count.saturating_sub(self.options.rollback_tokens);
        tracing::debug!(
            text_tokens = text_count,
            fixed,
            rollback = self.options.rollback_tokens,
            commit_threshold = self.options.commit_token_count * 2,
            "decode_and_maybe_commit: token counts"
        );

        if fixed >= self.options.commit_token_count * 2 {
            tracing::info!(
                commit_n = self.options.commit_token_count,
                text_tokens = text_count,
                "rotating: committing tokens and starting fresh decode"
            );
            if let Some(aligned) = self.decode.commit(
                TokenCount(self.options.commit_token_count),
                self.forced_aligner,
                self.tokenizer,
            )? {
                let new_words: Vec<AlignedWord> = aligned.words()
                    .filter_map(|entries| Self::word_to_aligned(self.tokenizer, entries))
                    .collect();
                let committed_word_texts: Vec<_> = new_words.iter().map(|w| w.word.clone()).collect();
                tracing::info!(
                    words = ?committed_word_texts,
                    remaining_text_tokens = self.decode.text_tokens().len(),
                    remaining_audio_samples = self.decode.audio_len(),
                    "rotation complete: committed words"
                );

                // One-commit-late correction: correct the *previous* buffered
                // chunk using the new chunk's words as right context.
                self.flush_buffered_commit(&new_words);

                // Buffer this chunk for correction when the next commit arrives.
                let raw_text = new_words.iter().map(|w| w.word.as_str()).collect::<Vec<_>>().join(" ");
                self.buffered_commit = Some(BufferedCommit {
                    raw_text,
                    words: new_words,
                });

                self.committed.append(aligned);
                // Refresh pending after rotation
                self.pending
                    .replace(self.decode.pending_entries(self.tokenizer));
            }
        }

        Ok(())
    }

    /// Correct the previously buffered commit using right-context words,
    /// then shift the buffer state.
    fn flush_buffered_commit(&mut self, right_context: &[AlignedWord]) {
        if let Some(buffered) = self.buffered_commit.take() {
            if let Some((ref engine_arc, ref mut corrector)) = self.correction {
                let mut engine = engine_arc.lock().expect("correction engine lock poisoned");
                corrector.process_chunk_with_context(
                    &mut engine,
                    &buffered.raw_text,
                    &buffered.words,
                    &self.prev_raw_words,
                    right_context,
                    self.options.app_id.as_deref(),
                );
                tracing::debug!(
                    corrected_text_len = corrector.committed_text().len(),
                    edits = corrector.committed_edits().len(),
                    "correction: processed buffered chunk"
                );
            }
            self.prev_raw_words = buffered.words;
        }
    }

    fn word_to_aligned(tokenizer: &Tokenizer, entries: &[TokenEntry]) -> Option<AlignedWord> {
        let a = entries[0].word.as_ref()?.alignment.as_ref()?;
        let word_ids: Vec<TokenId> = entries.iter().map(|e| e.token.id).collect();
        let word = tokenizer.decode(&word_ids, true).unwrap_or_default();
        Some(AlignedWord {
            word,
            start: a.time.start.0,
            end: a.time.end.0,
            confidence: text_buffer::confidence(entries),
        })
    }

    fn make_update(&self) -> Update {
        let mut text = String::new();
        let mut alignments = Vec::new();

        // If correction is active, use corrector's committed text for the
        // corrected portion, then append raw committed words that haven't
        // been corrected yet (the buffered commit).
        let correction_text = self
            .correction
            .as_ref()
            .map(|(_, c)| c.committed_text())
            .unwrap_or("");

        if !correction_text.is_empty() {
            text.push_str(correction_text);
        }

        // Decode committed words for alignments (always needed) and for
        // text when no corrector is active.
        for entries in self.committed.words() {
            if let Some(aligned) = Self::word_to_aligned(self.tokenizer, entries) {
                if correction_text.is_empty() {
                    // No corrector — build text from raw words
                    if !text.is_empty() && !aligned.word.starts_with(' ') {
                        text.push(' ');
                    }
                    text.push_str(&aligned.word);
                }
                alignments.push(aligned);
            }
        }

        // Add buffered commit text (not yet corrected, waiting for right context)
        if let Some(ref buffered) = self.buffered_commit {
            if !text.is_empty() && !buffered.raw_text.starts_with(' ') {
                text.push(' ');
            }
            text.push_str(&buffered.raw_text);
        }

        // Decode pending tokens as one block
        let pending_ids = self.pending.token_ids();
        if !pending_ids.is_empty() {
            let pending_text = self.tokenizer.decode(&pending_ids, true).unwrap_or_default();
            if !text.is_empty() && !pending_text.starts_with(' ') {
                text.push(' ');
            }
            text.push_str(&pending_text);
        }

        Update {
            text,
            alignments,
            detected_language: self.detected_language.clone(),
        }
    }
}
