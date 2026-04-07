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
    /// Full manifest of all required model repos.
    /// Swift checks locally which ones are already present and skips them.
    async fn required_downloads(&self) -> Vec<RepoDownload>;

    /// Load ASR engine from the given cache dir.
    /// Returns empty string on success, error message on failure.
    async fn load_engine(&self, cache_dir: String) -> String;

    /// Create a transcription session, returns session ID.
    async fn create_session(&self, language: String) -> String;

    /// Feed audio samples to a session.
    async fn feed(&self, session_id: String, samples: Vec<f32>) -> FeedResult;

    /// Finalize a session, returns final transcription.
    async fn finish_session(&self, session_id: String) -> String;
}

/// A HuggingFace repo that needs to be downloaded.
#[derive(Debug, Clone, Facet)]
pub struct RepoDownload {
    /// e.g. "mlx-community/Qwen3-ASR-1.7B-4bit"
    pub repo_id: String,
    /// Local directory name under cache_dir, e.g. "mlx-community--Qwen3-ASR-1.7B-4bit"
    pub local_dir: String,
    /// Files to download
    pub files: Vec<RepoFile>,
}

/// A single file within a HuggingFace repo.
#[derive(Debug, Clone, Facet)]
pub struct RepoFile {
    /// Filename, e.g. "model-00001-of-00002.safetensors"
    pub name: String,
    /// Full download URL
    pub url: String,
    /// Expected size in bytes (0 if unknown)
    pub size: u64,
}

/// Result of feeding audio samples to a transcription session.
#[derive(Debug, Clone, Facet)]
pub struct FeedResult {
    /// Current transcription text
    pub text: String,
    /// Whether this is a finalized segment
    pub is_final: bool,
}
