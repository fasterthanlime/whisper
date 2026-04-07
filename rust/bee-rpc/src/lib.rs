use bee_types::AlignedWord;
use facet::Facet;
use vox::service;

/// Methods the IME exposes — app calls into IME.
#[service]
pub trait Ime {
    /// App sends marked text to the IME for display.
    async fn set_marked_text(&self, session_id: String, text: String) -> bool;

    /// App tells IME to commit text and end the session.
    async fn commit_text(&self, session_id: String, text: String) -> bool;

    /// App tells IME to stop dictating (cancel/abort).
    async fn stop_dictating(&self, session_id: String) -> bool;

    /// App pushes a prepared session to the IME. IME should claim it
    /// when activateServer fires for the matching PID.
    async fn prepare_session(&self, session_id: String, target_pid: i32) -> bool;

    /// App asks IME to replace previously committed text.
    async fn replace_text(&self, session_id: String, old_text: String, new_text: String) -> bool;
}

/// Methods the app exposes — IME calls into app.
#[service]
pub trait App {
    /// IME says hello, returns app instance ID.
    async fn ime_hello(&self) -> String;

    /// IME claims the prepared session. Returns session ID if one
    /// was waiting, or empty string if none.
    async fn claim_session(&self) -> String;

    /// IME attached to the session (activateServer confirmed).
    async fn ime_attach(&self, session_id: String) -> bool;

    /// IME notifies app that activation was revoked (spurious deactivate).
    async fn ime_activation_revoked(&self) -> bool;

    /// IME notifies app that it lost context (deactivateServer with no session).
    async fn ime_context_lost(&self, had_marked_text: bool) -> bool;

    /// IME notifies app of a key event (submit, cancel, user typed).
    async fn ime_key_event(
        &self,
        session_id: String,
        event_type: String,
        key_code: u32,
        characters: String,
    ) -> bool;
}

/// The main bee service — replaces all C FFI.
/// Swift (initiator/client) talks to Rust (acceptor/server) over vox-ffi.
#[service]
pub trait Bee {
    // ── Model management ───────────────────────────────────────────────

    /// Full manifest of all required model repos.
    async fn required_downloads(&self) -> Vec<RepoDownload>;

    /// Load ASR engine from the given cache dir.
    async fn load_engine(&self, cache_dir: String) -> Result<bool, BeeError>;

    // ── Transcription ──────────────────────────────────────────────────

    /// Create a transcription session, returns session ID.
    async fn create_session(&self, opts: SessionConfig) -> Result<String, BeeError>;

    /// Feed audio samples to a session.
    async fn feed(&self, session_id: String, samples: Vec<f32>) -> Result<Option<FeedResult>, BeeError>;

    /// Finalize a session, returns final transcription.
    async fn finish_session(&self, session_id: String) -> Result<String, BeeError>;

    /// Set the language for a session.
    async fn set_language(
        &self,
        session_id: String,
        language: String,
    ) -> Result<bool, BeeError>;

    /// Single-shot transcription of raw 16kHz f32 samples.
    async fn transcribe_samples(&self, samples: Vec<f32>) -> Result<String, BeeError>;

    /// Get engine resource usage stats.
    async fn get_stats(&self) -> EngineStats;

    // ── Correction ─────────────────────────────────────────────────────

    /// Load the correction engine.
    async fn correct_load(
        &self,
        dataset_dir: String,
        events_path: String,
        gate_threshold: f32,
        ranker_threshold: f32,
    ) -> Result<bool, BeeError>;

    /// Run correction on text.
    async fn correct_process(&self, text: String, app_id: String) -> CorrectionOutput;

    /// Teach the correction engine from user resolutions.
    async fn correct_teach(
        &self,
        session_id: String,
        resolutions: Vec<EditResolution>,
    ) -> Result<bool, BeeError>;

    /// Save correction engine state to disk.
    async fn correct_save(&self) -> Result<bool, BeeError>;
}

// ── Error type ────────────────────────────────────────────────────────

/// Errors returned by the Bee service.
#[derive(Debug, Clone, Facet)]
#[repr(u8)]
pub enum BeeError {
    /// Engine not loaded yet.
    EngineNotLoaded,
    /// Session not found (already finished or never created).
    SessionNotFound { session_id: String },
    /// Engine failed to load.
    LoadFailed { message: String },
    /// Transcription error (feed or finish).
    TranscriptionError { message: String },
    /// Correction engine error.
    CorrectionError { message: String },
    /// Not yet implemented.
    NotImplemented,
}

// ── Session config ─────────────────────────────────────────────────────

/// Configuration for creating a transcription session.
#[derive(Debug, Clone, Facet)]
pub struct SessionConfig {
    /// Language code (e.g. "en", "auto"). Empty = auto-detect.
    pub language: String,
    /// Seconds of audio per processing chunk. 0 = use default (0.4).
    pub chunk_duration: f32,
    /// VAD speech probability threshold (0.0-1.0). 0 = use default (0.5).
    pub vad_threshold: f32,
    /// How many recent tokens the model may revise each step. 0 = use default (5).
    pub rollback_tokens: u32,
    /// Fixed token count before commit+rotate. 0 = use default (12).
    pub commit_token_count: u32,
}

// ── Model types ────────────────────────────────────────────────────────

/// A HuggingFace repo that needs to be downloaded.
#[derive(Debug, Clone, Facet)]
pub struct RepoDownload {
    pub repo_id: String,
    pub local_dir: String,
    pub files: Vec<RepoFile>,
}

/// A single file within a HuggingFace repo.
#[derive(Debug, Clone, Facet)]
pub struct RepoFile {
    pub name: String,
    pub url: String,
}

// ── Transcription types ────────────────────────────────────────────────

/// Result of feeding audio samples to a transcription session.
#[derive(Debug, Clone, Facet)]
pub struct FeedResult {
    pub text: String,
    pub committed_utf16_len: u32,
    pub alignments: Vec<AlignedWord>,
    pub is_final: bool,
    /// Language detected by the model (empty if language was forced).
    pub detected_language: String,
}

/// Engine resource usage stats.
#[derive(Debug, Clone, Facet)]
pub struct EngineStats {
    pub cpu_percent: f32,
    pub gpu_percent: f32,
    pub vram_used_mb: f32,
    pub ram_used_mb: f32,
}

// ── Correction types ───────────────────────────────────────────────────

/// Output from the correction engine.
#[derive(Debug, Clone, Facet)]
pub struct CorrectionOutput {
    pub session_id: String,
    pub best_text: String,
    pub edits: Vec<CorrectionEdit>,
}

/// A single correction edit.
#[derive(Debug, Clone, Facet)]
pub struct CorrectionEdit {
    pub edit_id: String,
    pub span_start: u32,
    pub span_end: u32,
    pub original: String,
    pub replacement: String,
    pub term: String,
    pub alias_id: i32,
    pub ranker_prob: f64,
    pub gate_prob: f64,
}

/// User resolution for a correction edit.
#[derive(Debug, Clone, Facet)]
pub struct EditResolution {
    pub edit_id: String,
    pub accepted: bool,
}
