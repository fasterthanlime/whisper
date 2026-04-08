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

/// One alternative for a pending ASR token.
#[derive(Debug, Clone, Facet)]
pub struct TokenAlternative {
    pub token_id: TokenId,
    pub text: String,
    pub logit: f32,
}

/// A pending ASR token with top-k alternatives.
#[derive(Debug, Clone, Facet)]
pub struct PendingToken {
    pub token_id: TokenId,
    pub text: String,
    pub concentration: f32,
    pub margin: f32,
    pub alternatives: Vec<TokenAlternative>,
}

/// Native Session snapshot returned by `feed()` and `finish()`.
#[derive(Debug, Clone, Facet)]
pub struct SessionSnapshot {
    /// Fully committed text only.
    pub committed_text: String,

    /// Uncommitted text (buffered commit + pending decode tail).
    pub pending_text: String,

    /// Full visible text (`committed_text + pending_text`).
    pub full_text: String,

    /// Word-level timestamps for committed words only.
    pub committed_words: Vec<AlignedWord>,

    /// Current pending token alternatives from the decode tail.
    pub pending_tokens: Vec<PendingToken>,

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

    /// Optional directory containing the correction dataset (phonetic index,
    /// espeak data, seed weights). If set, inline corrections are enabled.
    pub correction_dir: Option<&'a Path>,

    /// Optional path to the correction events JSONL file (for online learning).
    pub correction_events_path: Option<std::path::PathBuf>,
}
