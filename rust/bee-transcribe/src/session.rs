//! New transcription session built on foundation types.
//!
//! AudioFilterChain → DecodeSession → TextBuffer → forced alignment.
//! Rotation = new DecodeSession with leftover audio.
//! Two TextBuffers: committed (aligned, stable) and pending (volatile).

use bee_qwen3_asr::forced_aligner::ForcedAligner;
use bee_qwen3_asr::generate::ConfidenceMode;
use bee_qwen3_asr::model::Qwen3ASRModel;
use bee_types::AlignedWord;
use bee_vad::SileroVad;
use bee_zipa_mlx::infer::ZipaInference;
use mlx_rs::error::Exception;
use tokenizers::Tokenizer;

use crate::audio_buffer::{AudioBuffer, SampleRate, Seconds};
use crate::audio_filter::{self, AudioFilterChain};
use crate::corrector::Corrector;
use crate::decode_session::DecodeSession;
use crate::text_buffer::{self, TextBuffer, TokenCount, TokenEntry, TokenId};
use crate::timing::{log_phase, log_phase_chunk, phase_start};
use crate::types::{
    CutEvent, CutSink, PendingToken, RotationCutStrategy, SessionAmbiguitySummary, SessionOptions,
    SessionSnapshot, TokenAlternative,
};
use crate::{FinishResult, SharedCorrectionEngine};

/// A committed chunk waiting for right-context before correction.
struct BufferedCommit {
    raw_text: String,
    words: Vec<AlignedWord>,
    /// The aligned token entries, deferred until flush.
    aligned: TextBuffer,
}

pub struct Session<'a> {
    model: &'a Qwen3ASRModel,
    tokenizer: &'a Tokenizer,
    forced_aligner: &'a ForcedAligner,
    zipa: &'a ZipaInference,

    filters: AudioFilterChain,
    decode: DecodeSession,
    committed: TextBuffer,
    pending: TextBuffer,
    detected_language: String,
    /// Language locked after the first commit (rotation). In auto-detect mode,
    /// we wait until we have a full committed segment before trusting the
    /// detection — then freeze it so subsequent sub-sessions don't drift.
    locked_language: String,

    /// Correction state: shared engine + per-session corrector.
    correction: Option<(SharedCorrectionEngine, Corrector)>,
    /// One-commit-late buffer: waiting for right context before correction.
    buffered_commit: Option<BufferedCommit>,
    /// Raw words from the chunk before the buffered one (left context).
    prev_raw_words: Vec<AlignedWord>,

    options: SessionOptions,
    incoming: Vec<f32>,
    chunk_size_samples: usize,
    revision: u64,
    session_audio: AudioBuffer,
    cut_sink: Option<CutSink>,
}

impl<'a> Session<'a> {
    pub fn new(
        model: &'a Qwen3ASRModel,
        tokenizer: &'a Tokenizer,
        forced_aligner: &'a ForcedAligner,
        zipa: &'a ZipaInference,
        vad: SileroVad,
        options: SessionOptions,
        correction: Option<(SharedCorrectionEngine, Corrector)>,
        cut_sink: Option<CutSink>,
    ) -> Self {
        let chunk_size_samples = (options.chunk_duration * 16000.0) as usize;
        assert!(chunk_size_samples > 0, "chunk_duration too small");
        Self {
            model,
            tokenizer,
            forced_aligner,
            zipa,
            filters: audio_filter::default_filter_chain(vad, options.vad_threshold),
            decode: DecodeSession::new(
                AudioBuffer::empty(SampleRate::HZ_16000),
                Seconds::ZERO,
                TokenCount(options.rollback_tokens),
            ),
            committed: TextBuffer::new(),
            pending: TextBuffer::new(),
            detected_language: String::new(),
            locked_language: String::new(),
            correction,
            buffered_commit: None,
            prev_raw_words: Vec::new(),
            options,
            incoming: Vec::new(),
            chunk_size_samples,
            revision: 0,
            session_audio: AudioBuffer::empty(SampleRate::HZ_16000),
            cut_sink,
        }
    }

    /// Access the current pending token entries (for inspecting alternatives).
    pub fn pending_entries(&self) -> &[TokenEntry] {
        self.pending.entries()
    }

    /// Access the tokenizer (for decoding token IDs to text).
    pub fn tokenizer(&self) -> &Tokenizer {
        self.tokenizer
    }

    pub fn feed(&mut self, samples: &[f32]) -> Result<Option<SessionSnapshot>, Exception> {
        let feed_total_start = phase_start();
        self.incoming.extend_from_slice(samples);
        tracing::trace!(
            incoming_len = self.incoming.len(),
            chunk_size = self.chunk_size_samples,
            "feed: buffering audio"
        );

        let mut did_decode = false;
        while self.incoming.len() >= self.chunk_size_samples {
            let chunk_samples: Vec<f32> = self.incoming.drain(..self.chunk_size_samples).collect();
            let chunk = AudioBuffer::new(chunk_samples, SampleRate::HZ_16000);

            let filter_start = phase_start();
            let chunk_index = self.decode.chunk_count() + 1;
            if let Some(chunk) = self.filters.process(chunk) {
                log_phase_chunk("feed", "filter_vad", chunk_index, filter_start);
                tracing::debug!(
                    audio_samples = chunk.len(),
                    "feed: speech chunk passed filters"
                );
                self.session_audio.append(&chunk);
                self.decode.append_audio(&chunk);
                let decode_start = phase_start();
                self.decode_and_maybe_commit(
                    self.options.max_tokens_streaming,
                    ConfidenceMode::Streaming,
                )?;
                log_phase_chunk("feed", "decode_and_maybe_commit", chunk_index, decode_start);
                did_decode = true;
            } else {
                log_phase_chunk("feed", "filter_vad", chunk_index, filter_start);
                tracing::trace!("feed: chunk filtered out (silence/VAD)");
            }
        }

        if did_decode {
            let snapshot_start = phase_start();
            let snapshot = self.make_snapshot();
            log_phase("feed", "make_snapshot", snapshot_start);
            log_phase("feed", "total", feed_total_start);
            Ok(Some(snapshot))
        } else {
            log_phase("feed", "total", feed_total_start);
            Ok(None)
        }
    }

    pub fn finish(mut self) -> Result<FinishResult, Exception> {
        let finish_total_start = phase_start();
        tracing::info!(
            incoming = self.incoming.len(),
            committed_tokens = self.committed.len().0,
            pending_tokens = self.pending.len().0,
            "finish: finalizing session"
        );

        if !self.incoming.is_empty() {
            let remaining = std::mem::take(&mut self.incoming);
            let chunk = AudioBuffer::new(remaining, SampleRate::HZ_16000);
            let filter_start = phase_start();
            if let Some(chunk) = self.filters.process(chunk) {
                log_phase("finish", "filter_vad", filter_start);
                self.decode.append_audio(&chunk);
            } else {
                log_phase("finish", "filter_vad", filter_start);
            }
        }

        if self.decode.has_audio() {
            tracing::debug!(
                max_tokens = self.options.max_tokens_final,
                "finish: final decode"
            );
            let final_decode_start = phase_start();
            self.decode_and_maybe_commit(self.options.max_tokens_final, ConfidenceMode::Full)?;
            log_phase("finish", "final_decode", final_decode_start);
        }

        // Commit everything remaining
        if !self.pending.is_empty() {
            tracing::debug!(
                pending_tokens = self.pending.len().0,
                "finish: committing remaining pending tokens"
            );
            if self.decode.has_audio() {
                let commit_all_start = phase_start();
                if let Some(aligned) =
                    self.decode
                        .commit_all(self.forced_aligner, self.zipa, self.tokenizer)?
                {
                    log_phase("finish", "commit_all", commit_all_start);
                    let final_words: Vec<AlignedWord> = aligned
                        .words()
                        .filter_map(|entries| Self::word_to_aligned(self.tokenizer, entries))
                        .collect();

                    // Flush previous buffered commit with these final words as right context
                    self.flush_buffered_commit(&final_words);

                    // Correct this final chunk too (no right context)
                    let raw_text = final_words
                        .iter()
                        .map(|w| w.word.as_str())
                        .collect::<Vec<_>>()
                        .join(" ");
                    if let Some((ref engine_arc, ref mut corrector)) = self.correction {
                        let mut engine =
                            engine_arc.lock().expect("correction engine lock poisoned");
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
                    self.pending
                        .replace(self.decode.pending_entries(self.tokenizer));
                } else {
                    log_phase("finish", "commit_all", commit_all_start);
                }
            } else {
                tracing::warn!(
                    pending_tokens = self.pending.len().0,
                    "finish: pending text without decode audio; preserving raw tail"
                );
                self.flush_buffered_commit(&[]);
            }
        } else {
            // No pending tokens, but may still have a buffered commit to flush
            self.flush_buffered_commit(&[]);
        }

        let snapshot_start = phase_start();
        let snapshot = self.make_snapshot();
        log_phase("finish", "make_snapshot", snapshot_start);
        tracing::info!(
            committed_tokens = self.committed.len().0,
            text_len = snapshot.full_text.len(),
            alignments = snapshot.committed_words.len(),
            "finish: done"
        );
        log_phase("finish", "total", finish_total_start);
        let corrector = self.correction.take().map(|(_, c)| c);
        Ok(FinishResult {
            snapshot,
            session_audio: self.session_audio,
            corrector,
        })
    }

    // ── Internal ────────────────────────────────────────────────────

    /// Language to pass to the model: explicit > locked > auto.
    fn effective_language(&self) -> &str {
        if !self.options.language.as_str().is_empty() {
            self.options.language.as_str()
        } else {
            &self.locked_language
        }
    }

    /// Conservative early rollback: before the first committed chunk, keep the
    /// full configured rewrite window so the model can revise shaky cold-start
    /// tokens. After the session has rotated once, fall back to the narrower
    /// adaptive window so later chunks stabilize faster.
    fn effective_rollback(&self) -> usize {
        compute_effective_rollback(
            self.has_committed_context(),
            self.options.rollback_tokens,
            self.decode.text_tokens().len(),
        )
    }

    fn effective_commit_threshold(&self) -> usize {
        match self.options.rotation_cut_strategy {
            RotationCutStrategy::Uncut => usize::MAX,
            RotationCutStrategy::ManualTargetCommittedTextTokens(target) => target.max(1) as usize,
            RotationCutStrategy::Qwen3 | RotationCutStrategy::Zipa => {
                compute_effective_commit_threshold(
                    self.has_committed_context(),
                    self.options.commit_token_count,
                )
            }
        }
    }

    fn requested_commit_tokens(&self) -> TokenCount {
        match self.options.rotation_cut_strategy {
            RotationCutStrategy::Uncut => TokenCount(usize::MAX),
            RotationCutStrategy::Qwen3 | RotationCutStrategy::Zipa => {
                TokenCount(self.options.commit_token_count)
            }
            RotationCutStrategy::ManualTargetCommittedTextTokens(target) => {
                TokenCount(target.max(1) as usize)
            }
        }
    }

    fn has_committed_context(&self) -> bool {
        !self.prev_raw_words.is_empty()
            || self.buffered_commit.is_some()
            || !self.committed.is_empty()
    }

    fn decode_and_maybe_commit(
        &mut self,
        max_tokens: usize,
        confidence_mode: ConfidenceMode,
    ) -> Result<(), Exception> {
        let total_start = phase_start();
        let chunk_index = self.decode.chunk_count() + 1;
        // Update rollback window before decode so compute_prefix uses the adaptive value
        let rollback = self.effective_rollback();
        self.decode.set_rollback(text_buffer::TokenCount(rollback));

        let language = self.effective_language().to_owned();
        let decode_step_start = phase_start();
        self.decode.decode_step(
            self.model,
            self.tokenizer,
            &language,
            max_tokens,
            confidence_mode,
        )?;
        log_phase_chunk(
            "decode_and_maybe_commit",
            "decode_step",
            chunk_index,
            decode_step_start,
        );

        if let Some(lang) = self.decode.detected_language(self.tokenizer) {
            tracing::debug!(language = %lang, "detected language");
            self.detected_language = lang;
        }

        // Refresh pending from decode session's current text tokens
        let pending_start = phase_start();
        self.pending
            .replace(self.decode.pending_entries(self.tokenizer));
        log_phase_chunk(
            "decode_and_maybe_commit",
            "refresh_pending",
            chunk_index,
            pending_start,
        );

        // Check if enough fixed tokens to try committing (use adaptive rollback)
        let text_count = self.decode.text_tokens().len();
        let rollback = self.effective_rollback();
        let fixed = text_count.saturating_sub(rollback);
        let commit_threshold = self.effective_commit_threshold();
        tracing::debug!(
            text_tokens = text_count,
            fixed,
            rollback,
            commit_threshold,
            "decode_and_maybe_commit: token counts"
        );

        if fixed >= commit_threshold {
            let requested_commit_tokens = self.requested_commit_tokens();
            tracing::info!(
                commit_n = requested_commit_tokens.0,
                text_tokens = text_count,
                threshold = commit_threshold,
                rotation_cut_strategy = ?self.options.rotation_cut_strategy,
                "rotating: committing tokens and starting fresh decode"
            );
            let commit_n = self
                .decode
                .committable_text_tokens(self.tokenizer, requested_commit_tokens);
            if commit_n.0 > 0 {
                let refresh_start = phase_start();
                self.decode.refresh_text_confidence(
                    self.model,
                    self.tokenizer,
                    &language,
                    ConfidenceMode::Full,
                    0,
                    commit_n.0,
                )?;
                log_phase_chunk(
                    "decode_and_maybe_commit",
                    "refresh_text_confidence",
                    chunk_index,
                    refresh_start,
                );
            }
            let commit_start = phase_start();
            if let Some(aligned) = self.decode.commit(
                requested_commit_tokens,
                &self.options.rotation_cut_strategy,
                self.forced_aligner,
                self.zipa,
                self.tokenizer,
            )? {
                log_phase_chunk(
                    "decode_and_maybe_commit",
                    "commit",
                    chunk_index,
                    commit_start,
                );
                let new_words: Vec<AlignedWord> = aligned
                    .words()
                    .filter_map(|entries| Self::word_to_aligned(self.tokenizer, entries))
                    .collect();
                let committed_word_texts: Vec<_> =
                    new_words.iter().map(|w| w.word.clone()).collect();
                tracing::info!(
                    words = ?committed_word_texts,
                    remaining_text_tokens = self.decode.text_tokens().len(),
                    remaining_audio_samples = self.decode.audio_len(),
                    "rotation complete: committed words"
                );
                if let Some(sink) = self.cut_sink.as_mut() {
                    sink(CutEvent {
                        committed_words: new_words.clone(),
                    });
                }

                // Lock language on first rotation (auto-detect mode only).
                // We wait for a full committed segment before trusting the
                // detected language, so subsequent sub-sessions don't drift.
                if self.locked_language.is_empty()
                    && !self.detected_language.is_empty()
                    && self.options.language.as_str().is_empty()
                {
                    tracing::info!(language = %self.detected_language, "locking detected language after first rotation");
                    self.locked_language = self.detected_language.clone();
                }

                // One-commit-late correction: correct the *previous* buffered
                // chunk using the new chunk's words as right context.
                self.flush_buffered_commit(&new_words);

                // Buffer this chunk for correction when the next commit arrives.
                let raw_text = {
                    let mut s = String::new();
                    for w in &new_words {
                        if !s.is_empty() && !w.word.starts_with(' ') {
                            s.push(' ');
                        }
                        s.push_str(&w.word);
                    }
                    s
                };
                self.buffered_commit = Some(BufferedCommit {
                    raw_text,
                    words: new_words,
                    aligned,
                });
                // Refresh pending after rotation
                self.pending
                    .replace(self.decode.pending_entries(self.tokenizer));
            } else {
                log_phase_chunk(
                    "decode_and_maybe_commit",
                    "commit",
                    chunk_index,
                    commit_start,
                );
            }
        }

        log_phase_chunk("decode_and_maybe_commit", "total", chunk_index, total_start);
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
            // Now that correction has processed this chunk, move the aligned
            // tokens into the committed buffer (for make_update's word iteration).
            self.committed.append(buffered.aligned);
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

    fn make_snapshot(&mut self) -> SessionSnapshot {
        let snapshot_total_start = phase_start();
        const LOW_CONCENTRATION: f32 = 3.0;
        const LOW_MARGIN: f32 = 2.0;

        self.revision += 1;

        let mut committed_text = String::new();
        let mut pending_text = String::new();
        let mut committed_words = Vec::new();

        // If correction is active, use corrector's committed text for the
        // corrected portion, then append raw committed words that haven't
        // been corrected yet (the buffered commit).
        let correction_text = self
            .correction
            .as_ref()
            .map(|(_, c)| c.committed_text())
            .unwrap_or("");

        if !correction_text.is_empty() {
            committed_text.push_str(correction_text);
        }

        // Decode committed words for alignments (always needed) and for
        // text when no corrector is active.
        let committed_start = phase_start();
        for entries in self.committed.words() {
            if let Some(aligned) = Self::word_to_aligned(self.tokenizer, entries) {
                if correction_text.is_empty() {
                    // No corrector — build text from raw words
                    if !committed_text.is_empty() && !aligned.word.starts_with(' ') {
                        committed_text.push(' ');
                    }
                    committed_text.push_str(&aligned.word);
                }
                committed_words.push(aligned);
            }
        }
        log_phase("make_snapshot", "committed_words", committed_start);

        // Add buffered commit text (not yet corrected, waiting for right context)
        if let Some(ref buffered) = self.buffered_commit {
            if !pending_text.is_empty() && !buffered.raw_text.starts_with(' ') {
                pending_text.push(' ');
            }
            pending_text.push_str(&buffered.raw_text);
        }

        // Decode pending tokens, trimming trailing low-confidence ones.
        // This prevents uncertain tokens from flickering in the UI.
        let pending_entries_start = phase_start();
        let pending_entries = self.pending.entries();
        if !pending_entries.is_empty() {
            const CONFIDENCE_GATE: f32 = 3.0; // concentration threshold (top1 - mean(rest))
            let confident_count = pending_entries
                .iter()
                .rposition(|e| e.token.concentration >= CONFIDENCE_GATE)
                .map(|i| i + 1)
                .unwrap_or(0);
            if confident_count > 0 {
                let pending_ids: Vec<_> = pending_entries[..confident_count]
                    .iter()
                    .map(|e| e.token.id)
                    .collect();
                let decoded_pending = self
                    .tokenizer
                    .decode(&pending_ids, true)
                    .unwrap_or_default();
                if !decoded_pending.is_empty() {
                    if !pending_text.is_empty() && !decoded_pending.starts_with(' ') {
                        pending_text.push(' ');
                    }
                    pending_text.push_str(&decoded_pending);
                }
            }
        }
        log_phase("make_snapshot", "pending_text", pending_entries_start);

        let pending_tokens_start = phase_start();
        let full_text = format!("{committed_text}{pending_text}");
        let pending_tokens: Vec<_> = self
            .pending_entries()
            .iter()
            .map(|entry| PendingToken {
                token_id: entry.token.id,
                text: self
                    .tokenizer
                    .decode(&[entry.token.id], true)
                    .unwrap_or_default(),
                concentration: entry.token.concentration,
                margin: entry.token.margin,
                alternatives: entry
                    .token
                    .top_ids
                    .iter()
                    .zip(entry.token.top_logits.iter())
                    .take(entry.token.alternative_count as usize)
                    .map(|(&token_id, &logit)| TokenAlternative {
                        token_id,
                        text: self.tokenizer.decode(&[token_id], true).unwrap_or_default(),
                        logit,
                    })
                    .collect(),
            })
            .collect();
        log_phase("make_snapshot", "pending_tokens", pending_tokens_start);

        let pending_token_count = pending_tokens.len() as u32;
        let ambiguity_start = phase_start();
        let ambiguity = if pending_tokens.is_empty() {
            SessionAmbiguitySummary {
                pending_token_count: 0,
                low_concentration_count: 0,
                low_margin_count: 0,
                volatile_token_count: 0,
                mean_concentration: 0.0,
                mean_margin: 0.0,
                min_concentration: 0.0,
                min_margin: 0.0,
            }
        } else {
            let n = pending_tokens.len() as f32;
            let low_concentration_count = pending_tokens
                .iter()
                .filter(|token| token.concentration < LOW_CONCENTRATION)
                .count() as u32;
            let low_margin_count = pending_tokens
                .iter()
                .filter(|token| token.margin < LOW_MARGIN)
                .count() as u32;
            let volatile_token_count = pending_tokens
                .iter()
                .filter(|token| {
                    token.concentration < LOW_CONCENTRATION || token.margin < LOW_MARGIN
                })
                .count() as u32;

            SessionAmbiguitySummary {
                pending_token_count,
                low_concentration_count,
                low_margin_count,
                volatile_token_count,
                mean_concentration: pending_tokens
                    .iter()
                    .map(|token| token.concentration)
                    .sum::<f32>()
                    / n,
                mean_margin: pending_tokens.iter().map(|token| token.margin).sum::<f32>() / n,
                min_concentration: pending_tokens
                    .iter()
                    .map(|token| token.concentration)
                    .fold(f32::INFINITY, f32::min),
                min_margin: pending_tokens
                    .iter()
                    .map(|token| token.margin)
                    .fold(f32::INFINITY, f32::min),
            }
        };
        log_phase("make_snapshot", "ambiguity", ambiguity_start);
        log_phase("make_snapshot", "total", snapshot_total_start);

        SessionSnapshot {
            revision: self.revision,
            committed_text,
            pending_text,
            full_text,
            committed_token_count: self.committed.len().0 as u32,
            pending_token_count,
            committed_words,
            pending_tokens,
            ambiguity,
            detected_language: self.detected_language.clone(),
        }
    }
}

fn compute_effective_rollback(
    has_committed_context: bool,
    configured_rollback: usize,
    text_token_count: usize,
) -> usize {
    if !has_committed_context {
        configured_rollback
    } else if text_token_count <= 1 {
        0
    } else {
        (text_token_count - 1).min(configured_rollback)
    }
}

fn compute_effective_commit_threshold(
    has_committed_context: bool,
    configured_commit_threshold: usize,
) -> usize {
    if has_committed_context {
        configured_commit_threshold
    } else {
        configured_commit_threshold * 2
    }
}

#[cfg(test)]
mod tests {
    use super::{compute_effective_commit_threshold, compute_effective_rollback};

    #[test]
    fn cold_start_keeps_full_rollback_window() {
        assert_eq!(compute_effective_rollback(false, 5, 0), 5);
        assert_eq!(compute_effective_rollback(false, 5, 2), 5);
        assert_eq!(compute_effective_rollback(false, 5, 10), 5);
    }

    #[test]
    fn post_commit_rollback_shrinks_with_context() {
        assert_eq!(compute_effective_rollback(true, 5, 0), 0);
        assert_eq!(compute_effective_rollback(true, 5, 1), 0);
        assert_eq!(compute_effective_rollback(true, 5, 2), 1);
        assert_eq!(compute_effective_rollback(true, 5, 4), 3);
        assert_eq!(compute_effective_rollback(true, 5, 10), 5);
    }

    #[test]
    fn cold_start_uses_larger_commit_threshold() {
        assert_eq!(compute_effective_commit_threshold(false, 12), 24);
        assert_eq!(compute_effective_commit_threshold(true, 12), 12);
    }
}
