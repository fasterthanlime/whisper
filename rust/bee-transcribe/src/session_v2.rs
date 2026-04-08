//! New transcription session built on foundation types.
//!
//! AudioFilterChain → DecodeSession → TextBuffer → forced alignment.
//! Rotation = new DecodeSession with leftover audio.
//! Committed state lives in TextBuffer.

use bee_qwen3_asr::forced_aligner::ForcedAligner;
use bee_qwen3_asr::model::Qwen3ASRModel;
use bee_types::AlignedWord;
use bee_vad::SileroVad;
use mlx_rs::error::Exception;
use tokenizers::Tokenizer;

use crate::audio_buffer::{AudioBuffer, SampleRate, Seconds, TimeRange};
use crate::audio_filter::{self, AudioFilterChain};
use crate::decode_session::DecodeSession;
use crate::text_buffer::{TextBuffer, TokenCount, TokenId, TokenIndex, WordAlignment};
use crate::types::{SessionOptions, Update};

/// A live transcription session.
pub struct SessionV2<'a> {
    model: &'a Qwen3ASRModel,
    tokenizer: &'a Tokenizer,
    forced_aligner: &'a ForcedAligner,

    filters: AudioFilterChain,
    decode: DecodeSession,
    text: TextBuffer,
    /// Audio offset from previous rotations, for absolute timestamps.
    audio_offset: Seconds,
    detected_language: String,

    options: SessionOptions,
    /// Sub-chunk accumulation buffer.
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
            text: TextBuffer::new(),
            audio_offset: Seconds::ZERO,
            detected_language: String::new(),
            options,
            incoming: Vec::new(),
            chunk_size_samples,
        }
    }

    pub fn feed(&mut self, samples: &[f32]) -> Result<Option<Update>, Exception> {
        self.incoming.extend_from_slice(samples);
        if self.incoming.len() < self.chunk_size_samples {
            return Ok(None);
        }

        let chunk_samples: Vec<f32> = self.incoming.drain(..self.chunk_size_samples).collect();
        let chunk = AudioBuffer::new(chunk_samples, SampleRate::HZ_16000);

        let chunk = match self.filters.process(chunk) {
            Some(c) => c,
            None => return Ok(None),
        };

        self.decode.audio.append(&chunk);
        self.decode_and_commit(self.options.max_tokens_streaming)?;
        Ok(Some(self.make_update()))
    }

    pub fn finish(&mut self) -> Result<Update, Exception> {
        if !self.incoming.is_empty() {
            let remaining = std::mem::take(&mut self.incoming);
            let chunk = AudioBuffer::new(remaining, SampleRate::HZ_16000);
            if let Some(chunk) = self.filters.process(chunk) {
                self.decode.audio.append(&chunk);
            }
        }

        if !self.decode.audio.is_empty() {
            self.decode_and_commit(self.options.max_tokens_final)?;
        }

        // Commit everything remaining
        let pending_count = self.text.pending_len();
        if pending_count.0 > 0 && !self.decode.audio.is_empty() {
            self.commit(pending_count)?;
        }

        Ok(self.make_update())
    }

    // ── Internal ────────────────────────────────────────────────────

    /// Decode, refresh pending tail, maybe commit.
    fn decode_and_commit(&mut self, max_tokens: usize) -> Result<(), Exception> {
        self.decode.decode_step(
            self.model,
            self.tokenizer,
            self.options.language.as_str(),
            max_tokens,
        )?;

        self.refresh_language();

        let pending = self.decode.pending_entries(self.tokenizer);
        self.text.replace_pending(pending);

        // Check if enough fixed tokens to commit
        let text_token_count = self.decode.text_tokens().len();
        let fixed = text_token_count.saturating_sub(self.options.rollback_tokens);
        if fixed >= self.options.commit_token_count * 2 {
            self.commit(TokenCount(self.options.commit_token_count))?;
        }

        Ok(())
    }

    /// Commit `n` pending tokens: advance boundary, run forced alignment,
    /// set WordAlignment on newly committed entries, then rotate.
    fn commit(&mut self, n: TokenCount) -> Result<(), Exception> {
        // Decode the text we're about to commit
        let pending = self.text.pending_entries();
        let commit_ids: Vec<TokenId> = pending[..n.0].iter().map(|e| e.token.id).collect();
        let commit_text = self.tokenizer.decode(&commit_ids, true).unwrap_or_default();

        if commit_text.trim().is_empty() {
            return Ok(());
        }

        // Forced alignment against decode session's audio
        let items = self
            .forced_aligner
            .align(self.decode.audio.samples(), &commit_text)
            .map_err(|e| Exception::custom(format!("aligner: {e}")))?;

        // Advance the committed boundary
        self.text.advance_committed(n);

        // Set alignment on newly committed word-start entries
        if !items.is_empty() {
            let commit_start = self.text.committed_index().0 - n.0;
            let mut word_idx = 0;
            let mut align_pairs = Vec::new();

            for (offset, entry) in self.text.committed_entries()[commit_start..].iter().enumerate() {
                if entry.word.is_some() && word_idx < items.len() {
                    let item = &items[word_idx];
                    let start = Seconds(item.start_time as f64);
                    let end = Seconds(item.end_time as f64);
                    let time = TimeRange::new(
                        start + self.audio_offset,
                        end + self.audio_offset,
                    );
                    let audio = self.decode.audio.slice(TimeRange::new(start, end));
                    align_pairs.push((offset, WordAlignment { time, audio }));
                    word_idx += 1;
                }
            }

            self.text.set_alignments(
                TokenIndex(commit_start),
                align_pairs,
            );

            // Rotate: trim audio, create fresh decode session with remainder
            let last_end = Seconds(items.last().unwrap().end_time as f64);
            self.audio_offset = self.audio_offset + last_end;

            let (_, remaining_audio) = self.decode.audio.split_at(last_end);
            self.decode = DecodeSession::new(
                remaining_audio,
                self.audio_offset,
                TokenCount(self.options.rollback_tokens),
            );
        }

        Ok(())
    }

    fn refresh_language(&mut self) {
        if let Some(lang) = self.decode.detected_language(self.tokenizer) {
            self.detected_language = lang;
        }
    }

    fn make_update(&self) -> Update {
        let committed_ids = self.text.committed_token_ids();
        let committed_text = if committed_ids.is_empty() {
            String::new()
        } else {
            self.tokenizer.decode(&committed_ids, true).unwrap_or_default()
        };

        let pending_ids = self.text.pending_token_ids();
        let pending_text = if pending_ids.is_empty() {
            String::new()
        } else {
            self.tokenizer.decode(&pending_ids, true).unwrap_or_default()
        };

        let text = format!("{committed_text}{pending_text}");
        let asr_committed_len = committed_text.len();

        // Build alignments from TextBuffer's committed words
        let mut alignments = Vec::new();
        for (_offset, entries) in self.text.committed_words() {
            if let Some(ref ws) = entries[0].word {
                if let Some(ref alignment) = ws.alignment {
                    let word_ids: Vec<TokenId> = entries.iter().map(|e| e.token.id).collect();
                    let word_text = self.tokenizer.decode(&word_ids, true).unwrap_or_default();

                    let n = entries.len() as f32;
                    alignments.push(AlignedWord {
                        word: word_text,
                        start: alignment.time.start.0,
                        end: alignment.time.end.0,
                        confidence: bee_types::Confidence {
                            mean_lp: entries.iter().map(|e| e.token.logprob).sum::<f32>() / n,
                            min_lp: entries.iter().map(|e| e.token.logprob).fold(f32::INFINITY, f32::min),
                            mean_m: entries.iter().map(|e| e.token.margin).sum::<f32>() / n,
                            min_m: entries.iter().map(|e| e.token.margin).fold(f32::INFINITY, f32::min),
                        },
                    });
                }
            }
        }

        Update {
            text,
            asr_committed_len,
            correction_committed_len: 0,
            alignments,
            detected_language: self.detected_language.clone(),
        }
    }
}
