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

/// Strategy for choosing rotation cut points inside a decode session.
#[repr(u8)]
#[derive(Debug, Clone, Facet)]
pub enum RotationCutStrategy {
    /// Never rotate during streaming; commit only once at finish().
    Uncut,
    /// Automatic checkpoint-based rotation using Qwen/aligner timing.
    Qwen3,
    /// Automatic checkpoint-based rotation using ZIPA-derived cut timing.
    Zipa,
    /// Manual behavior: commit at the latest compatible checkpoint whose
    /// text-token length is at most this target.
    ManualTargetCommittedTextTokens(u32),
}

impl Default for RotationCutStrategy {
    fn default() -> Self {
        Self::Qwen3
    }
}

/// Event fired each time the session commits a rotation cut.
pub struct CutEvent {
    /// The words that were just committed at this cut point.
    pub committed_words: Vec<AlignedWord>,
    /// Audio that was committed (the slice that got locked in).
    pub committed_audio: crate::audio_buffer::AudioBuffer,
    /// Audio that was retained and fed back into the next decode session.
    pub remaining_audio: crate::audio_buffer::AudioBuffer,
}

/// A sink that receives cut events during a transcription session.
pub type CutSink = Box<dyn FnMut(CutEvent)>;

/// Event fired for every audio chunk processed by `feed()`.
pub struct ChunkEvent {
    /// Raw audio before any filtering.
    pub raw_audio: crate::audio_buffer::AudioBuffer,
    /// Post-filter audio. `None` if VAD gated the chunk.
    pub filtered_audio: Option<crate::audio_buffer::AudioBuffer>,
}

/// A sink that receives every audio chunk, including VAD-dropped ones.
pub type ChunkSink = Box<dyn FnMut(ChunkEvent)>;

/// Configuration for a transcription session.
#[derive(Debug, Clone, Facet)]
pub struct SessionOptions {
    /// Seconds of audio per processing chunk. Default: 0.4
    pub chunk_duration: f32,

    /// VAD speech probability threshold (0.0-1.0). Default: 0.5
    pub vad_threshold: f32,

    /// Streaming revision window: how many tail tokens the model may rewrite
    /// on each streaming step. Everything before this tail is fed back as a
    /// fixed prefix the model must continue from. Higher = more self-correction,
    /// less stability. Default: 5
    pub rollback_tokens: usize,

    /// How many tokens from the just-committed text are carried forward as
    /// fixed context into the next decode session after a rotation cut.
    /// Independent of `rollback_tokens`. Default: 0
    pub context_tokens: usize,

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

    /// How rotation cut points are selected.
    pub rotation_cut_strategy: RotationCutStrategy,

    /// Skip all audio filters (VAD, DC removal, RMS normalization).
    /// Every chunk is fed directly to the decoder as-is.
    pub bypass_audio_filters: bool,
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            chunk_duration: 0.4,
            vad_threshold: 0.5,
            rollback_tokens: 5,
            context_tokens: 0,
            commit_token_count: 12,
            max_tokens_streaming: 32,
            max_tokens_final: 512,
            language: Language::default(),
            app_id: None,
            rotation_cut_strategy: RotationCutStrategy::Qwen3,
            bypass_audio_filters: false,
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

/// Summary of how volatile the current pending decode tail is.
#[derive(Debug, Clone, Facet)]
pub struct SessionAmbiguitySummary {
    /// Number of pending tail tokens included in this summary.
    pub pending_token_count: u32,

    /// Tokens whose concentration falls below the stability gate.
    pub low_concentration_count: u32,

    /// Tokens whose top-1 vs top-2 margin falls below the tie gate.
    pub low_margin_count: u32,

    /// Tokens that are unstable by either concentration or margin.
    pub volatile_token_count: u32,

    /// Mean concentration across pending tokens, or 0 when none exist.
    pub mean_concentration: f32,

    /// Mean margin across pending tokens, or 0 when none exist.
    pub mean_margin: f32,

    /// Minimum concentration across pending tokens, or 0 when none exist.
    pub min_concentration: f32,

    /// Minimum margin across pending tokens, or 0 when none exist.
    pub min_margin: f32,
}

/// Native Session snapshot returned by `feed()` and `finish()`.
#[derive(Debug, Clone, Facet)]
pub struct SessionSnapshot {
    /// Monotonic per-session revision. Increments whenever a new snapshot is produced.
    pub revision: u64,

    /// Fully committed text only.
    pub committed_text: String,

    /// Uncommitted text (buffered commit + pending decode tail).
    pub pending_text: String,

    /// Full visible text (`committed_text + pending_text`).
    pub full_text: String,

    /// Number of committed tokens currently frozen into `committed_text`.
    pub committed_token_count: u32,

    /// Number of volatile tail tokens currently represented by `pending_tokens`.
    pub pending_token_count: u32,

    /// Word-level timestamps for committed words only.
    pub committed_words: Vec<AlignedWord>,

    /// Current pending token alternatives from the decode tail.
    ///
    /// These are decoder-local token alternatives, not final word alternatives.
    /// They are volatile and may be rewritten on the next `feed()`.
    pub pending_tokens: Vec<PendingToken>,

    /// Summary of pending-tail instability derived from `pending_tokens`.
    pub ambiguity: SessionAmbiguitySummary,

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

    /// Experiment: make the forced aligner reuse the ASR audio encoder
    /// backbone instead of loading its own copy. The aligner keeps its
    /// own final projection head so feature dimensionality still matches
    /// the aligner text model.
    pub share_aligner_audio_tower: bool,

    /// Directory containing silero VAD weights
    pub silero_dir: &'a Path,

    /// Optional directory containing the correction dataset (phonetic index,
    /// espeak data, seed weights). If set, inline corrections are enabled.
    pub correction_dir: Option<&'a Path>,

    /// Optional path to the correction events JSONL file (for online learning).
    pub correction_events_path: Option<std::path::PathBuf>,

    /// Directory containing the ZIPA-CR bundle (`config.json`,
    /// `model.safetensors`, `tokens.txt`).
    pub zipa_bundle_dir: &'a Path,
}
