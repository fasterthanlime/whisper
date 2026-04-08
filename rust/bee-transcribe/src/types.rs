// ── Data types ──────────────────────────────────────────────────────────

use std::path::Path;

use bee_types::AlignedWord;
use facet::Facet;

/// Numeric identifier for a token (for the tokenizer used by Qwen3-ASR, see the tokenizer crate)
pub type TokenId = u32;

/// Any transcription error
#[derive(thiserror::Error, Debug)]
pub enum TranscribeError {
    #[error("missing token logprobs")]
    MissingTokenLogProbs,
}

/// Language of the audio being transcribed.
///
/// Passed to the model as `"language {name}"`. Common values:
/// `"English"`, `"Chinese"`, `"Japanese"`, `"Korean"`, `"French"`,
/// `"German"`, `"Spanish"`.
#[derive(Default, Debug, Clone, Facet)]
pub struct Language(pub String);

impl Language {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Configuration for a transcription session.
#[derive(Debug, Clone, Facet)]
pub struct SessionOptions {
    /// Seconds of audio per processing chunk. Default: 0.4
    pub chunk_duration: f32,

    /// VAD speech probability threshold (0.0-1.0). Default: 0.5
    pub vad_threshold: f32,

    /// How many recent tokens the model is allowed to revise each step.
    /// Everything before this tail is fed back as fixed context that
    /// the model must continue from. Higher values give the model more
    /// freedom to correct itself but make the output less stable between
    /// steps. Default: 5
    pub rollback_tokens: usize,

    /// Minimum number of fixed tokens (total minus `rollback_tokens`)
    /// before we commit and rotate the session. When the fixed portion
    /// reaches this threshold, we find a word boundary, run the aligner,
    /// and start a fresh session. Default: 12
    pub commit_token_count: usize,

    /// Max tokens to generate per streaming step. Default: 32
    pub max_tokens_streaming: usize,

    /// Max tokens for the final decode when `finish()` is called. Default: 512
    pub max_tokens_final: usize,

    /// Language of the audio.
    pub language: Language,

    /// App bundle ID for correction context features.
    pub app_id: Option<String>,
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            chunk_duration: 0.4,
            vad_threshold: 0.5,
            rollback_tokens: 5,
            commit_token_count: 12,
            max_tokens_streaming: 32,
            max_tokens_final: 512,
            language: Language::default(),
            app_id: None,
        }
    }
}

/// Result of a `feed()` or `finish()` call.
#[derive(Debug, Clone, Facet)]
pub struct Update {
    /// Full transcription so far (corrected where available + ASR for the rest + in-progress tail).
    pub text: String,

    /// Byte offset: `text[..asr_committed_len]` is ASR-committed
    /// (won't change via rollback, but may still be corrected).
    pub asr_committed_len: usize,

    /// Byte offset: `text[..correction_committed_len]` has been through the
    /// correction pipeline and is truly final.
    pub correction_committed_len: usize,

    /// Word-level timestamps for committed words.
    pub alignments: Vec<AlignedWord>,

    /// Language detected by the model (empty if language was forced).
    pub detected_language: String,
}

/// Paths required to load an engine.
pub struct EngineConfig<'a> {
    /// Directory containing `config.json` and `*.safetensors` for the
    /// ASR model weights.
    pub model_dir: &'a Path,

    /// Directory containing tokenizer files — either a single `tokenizer.json`,
    /// or `vocab.json` + `merges.txt` (GPT-2 style BPE).
    pub tokenizer_dir: &'a Path,

    /// Directory containing the forced aligner model weights.
    pub aligner_dir: &'a Path,

    /// Directory containing silero VAD weights
    pub silero_dir: &'a Path,
}
