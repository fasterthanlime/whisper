//! ASR commit logic: track committed state, decide when to commit,
//! run forced alignment on text-only tokens.

use bee_qwen3_asr::forced_aligner::ForcedAligner;
use bee_qwen3_asr::generate::TokenLogprob;
use bee_types::AlignedWord;
use mlx_rs::error::Exception;
use tokenizers::Tokenizer;

use crate::types::TokenId;
use crate::word_logprob_stats;

/// Instructions for rotating the generator and speech gate after a commit.
pub struct RotateInstruction {
    /// How many raw tokens to drop from the generator
    /// (metadata + committed text tokens).
    pub raw_tokens_to_drop: usize,
    /// How many audio samples to trim from the speech gate.
    pub audio_cut_samples: usize,
}

/// A chunk of aligned, ASR-committed words.
#[allow(dead_code)]
pub struct AlignedChunk {
    /// Aligned words with confidence and absolute timing.
    pub words: Vec<AlignedWord>,
    /// The text that was committed.
    pub text: String,
    /// Rotation instructions for upstream layers.
    pub rotate: RotateInstruction,
}

/// Tracks ASR-committed state, decides when to commit, runs forced alignment.
pub struct Aligner {
    committed_text_tokens: Vec<TokenId>,
    committed_logprobs: Vec<TokenLogprob>,
    committed_alignments: Vec<AlignedWord>,
    committed_audio_offset: f64,
    detected_language: String,
    commit_token_count: usize,
}

impl Aligner {
    pub fn new(commit_token_count: usize) -> Self {
        Self {
            committed_text_tokens: Vec::new(),
            committed_logprobs: Vec::new(),
            committed_alignments: Vec::new(),
            committed_audio_offset: 0.0,
            detected_language: String::new(),
            commit_token_count,
        }
    }

    /// Extract language from metadata tokens (e.g. "language English").
    pub fn detect_language(&mut self, tokenizer: &Tokenizer, metadata_ids: &[TokenId]) {
        if metadata_ids.is_empty() {
            return;
        }
        let meta = tokenizer
            .decode(
                metadata_ids,
                true,
            )
            .unwrap_or_default();
        let lang = meta
            .trim()
            .strip_prefix("language ")
            .map(|l| l.trim().to_string())
            .filter(|l| !l.eq_ignore_ascii_case("none"))
            .unwrap_or_default();
        if !lang.is_empty() {
            self.detected_language = lang;
        }
    }

    /// Check if we should commit and rotate. Returns an `AlignedChunk` if so.
    ///
    /// `text_ids` and `text_logprobs` are the **text-only** portion (metadata stripped).
    /// `metadata_token_count` is the number of raw tokens consumed by metadata
    /// (needed to compute rotation offset for the generator).
    /// `rollback_tokens` is how many tail tokens may still be revised.
    #[allow(clippy::too_many_arguments)]
    pub fn maybe_commit(
        &mut self,
        forced_aligner: &ForcedAligner,
        tokenizer: &Tokenizer,
        audio: &[f32],
        text_ids: &[TokenId],
        text_logprobs: &[TokenLogprob],
        metadata_token_count: usize,
        rollback_tokens: usize,
    ) -> Result<Option<AlignedChunk>, Exception> {
        let fixed_count = text_ids.len().saturating_sub(rollback_tokens);

        if fixed_count < self.commit_token_count * 2 {
            return Ok(None);
        }

        let commit_count = self.commit_token_count;
        self.do_commit(
            forced_aligner,
            tokenizer,
            audio,
            &text_ids[..commit_count],
            &text_logprobs[..commit_count.min(text_logprobs.len())],
            metadata_token_count,
            commit_count,
        )
    }

    /// Commit remaining text tokens at finish time.
    pub fn finish_commit(
        &mut self,
        forced_aligner: &ForcedAligner,
        tokenizer: &Tokenizer,
        audio: &[f32],
        text_ids: &[TokenId],
        text_logprobs: &[TokenLogprob],
    ) -> Result<Option<AlignedChunk>, Exception> {
        if text_ids.is_empty() || audio.is_empty() {
            return Ok(None);
        }
        // No rotation needed at finish — commit everything, no metadata offset
        self.do_commit(forced_aligner, tokenizer, audio, text_ids, text_logprobs, 0, text_ids.len())
    }

    #[allow(clippy::too_many_arguments)]
    fn do_commit(
        &mut self,
        forced_aligner: &ForcedAligner,
        tokenizer: &Tokenizer,
        audio: &[f32],
        commit_text_ids: &[TokenId],
        commit_logprobs: &[TokenLogprob],
        metadata_token_count: usize,
        commit_count: usize,
    ) -> Result<Option<AlignedChunk>, Exception> {
        let commit_text = tokenizer
            .decode(
                commit_text_ids,
                true,
            )
            .unwrap_or_default();

        if commit_text.trim().is_empty() {
            return Ok(None);
        }

        let items = forced_aligner
            .align(audio, &commit_text)
            .map_err(|e| Exception::custom(format!("aligner: {e}")))?;

        if items.is_empty() {
            return Ok(None);
        }

        let last = &items[items.len() - 1];
        let audio_cut_samples = (last.end_time * 16000.0) as usize;

        let wstats = word_logprob_stats(tokenizer, commit_text_ids, commit_logprobs, items.len())
            .map_err(|e| Exception::custom(format!("{e}")))?;

        let offset = self.committed_audio_offset;
        let mut words = Vec::with_capacity(items.len());
        for (i, item) in items.iter().enumerate() {
            let word = AlignedWord {
                word: item.word.clone(),
                start: item.start_time + offset,
                end: item.end_time + offset,
                confidence: wstats[i].clone(),
            };
            self.committed_alignments.push(word.clone());
            words.push(word);
        }

        self.committed_text_tokens
            .extend_from_slice(commit_text_ids);
        self.committed_logprobs
            .extend_from_slice(commit_logprobs);

        let cut = audio_cut_samples.min(audio.len());
        self.committed_audio_offset += cut as f64 / 16000.0;

        log::info!(
            "Committed {} text tokens | audio cut at {:.1}s",
            commit_text_ids.len(),
            cut as f64 / 16000.0,
        );

        Ok(Some(AlignedChunk {
            words,
            text: commit_text,
            rotate: RotateInstruction {
                raw_tokens_to_drop: metadata_token_count + commit_count,
                audio_cut_samples: cut,
            },
        }))
    }

    pub fn committed_alignments(&self) -> &[AlignedWord] {
        &self.committed_alignments
    }

    pub fn committed_text_tokens(&self) -> &[TokenId] {
        &self.committed_text_tokens
    }

    pub fn detected_language(&self) -> &str {
        &self.detected_language
    }

    #[allow(dead_code)]
    pub fn committed_audio_offset(&self) -> f64 {
        self.committed_audio_offset
    }
}
