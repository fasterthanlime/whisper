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
use crate::decode_session::DecodeSession;
use crate::text_buffer::{self, TextBuffer, TokenCount, TokenEntry, TokenId};
use crate::types::{SessionOptions, Update};

pub struct SessionV2<'a> {
    model: &'a Qwen3ASRModel,
    tokenizer: &'a Tokenizer,
    forced_aligner: &'a ForcedAligner,

    filters: AudioFilterChain,
    decode: DecodeSession,
    committed: TextBuffer,
    pending: TextBuffer,
    detected_language: String,

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
            options,
            incoming: Vec::new(),
            chunk_size_samples,
        }
    }

    pub fn feed(&mut self, samples: &[f32]) -> Result<Option<Update>, Exception> {
        self.incoming.extend_from_slice(samples);

        let mut did_decode = false;
        while self.incoming.len() >= self.chunk_size_samples {
            let chunk_samples: Vec<f32> =
                self.incoming.drain(..self.chunk_size_samples).collect();
            let chunk = AudioBuffer::new(chunk_samples, SampleRate::HZ_16000);

            if let Some(chunk) = self.filters.process(chunk) {
                self.decode.append_audio(&chunk);
                self.decode_and_maybe_commit(self.options.max_tokens_streaming)?;
                did_decode = true;
            }
        }

        if did_decode {
            Ok(Some(self.make_update()))
        } else {
            Ok(None)
        }
    }

    pub fn finish(&mut self) -> Result<Update, Exception> {
        if !self.incoming.is_empty() {
            let remaining = std::mem::take(&mut self.incoming);
            let chunk = AudioBuffer::new(remaining, SampleRate::HZ_16000);
            if let Some(chunk) = self.filters.process(chunk) {
                self.decode.append_audio(&chunk);
            }
        }

        if self.decode.has_audio() {
            self.decode_and_maybe_commit(self.options.max_tokens_final)?;
        }

        // Commit everything remaining
        if !self.pending.is_empty() {
            debug_assert!(
                self.decode.has_audio(),
                "pending text exists without decode audio"
            );
            if let Some(aligned) = self.decode.commit_all(
                self.forced_aligner,
                self.tokenizer,
            )? {
                self.committed.append(aligned);
                self.pending.replace(self.decode.pending_entries(self.tokenizer));
            }
        }

        Ok(self.make_update())
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
            self.detected_language = lang;
        }

        // Refresh pending from decode session's current text tokens
        self.pending
            .replace(self.decode.pending_entries(self.tokenizer));

        // Check if enough fixed tokens to try committing
        let text_count = self.decode.text_tokens().len();
        let fixed = text_count.saturating_sub(self.options.rollback_tokens);
        if fixed >= self.options.commit_token_count * 2 {
            if let Some(aligned) = self.decode.commit(
                TokenCount(self.options.commit_token_count),
                self.forced_aligner,
                self.tokenizer,
            )? {
                self.committed.append(aligned);
                // Refresh pending after rotation
                self.pending
                    .replace(self.decode.pending_entries(self.tokenizer));
            }
        }

        Ok(())
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
        // Decode committed words individually to avoid cross-boundary
        // tokenizer artifacts (leading-space tokens get misinterpreted
        // when concatenated token IDs span rotation boundaries).
        let mut text = String::new();
        let mut alignments = Vec::new();
        for entries in self.committed.words() {
            if let Some(aligned) = Self::word_to_aligned(self.tokenizer, entries) {
                if !text.is_empty() && !aligned.word.starts_with(' ') {
                    text.push(' ');
                }
                text.push_str(&aligned.word);
                alignments.push(aligned);
            }
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
