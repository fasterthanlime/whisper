use vox::service;

/// RPC service between the bee app and the beeInput IME.
///
/// The app is the server (acceptor), the IME is the client (initiator).
#[service]
pub trait BeeIPC {
    /// IME says hello, returns its instance ID.
    async fn ime_hello(&self) -> String;

    /// App prepares a dictation session. The IME should claim it
    /// when activateServer fires for the matching PID.
    async fn prepare_session(
        &self,
        session_id: String,
        target_pid: i32,
    ) -> bool;

    /// IME claims the prepared session. Returns the session ID if one
    /// was waiting, or empty string if none.
    async fn claim_session(&self) -> String;

    /// IME attached to the session (activateServer confirmed).
    async fn ime_attach(&self, session_id: String) -> bool;

    /// App sends marked text to the IME for display.
    async fn set_marked_text(&self, session_id: String, text: String) -> bool;

    /// App tells IME to commit text and end the session.
    async fn commit_text(&self, session_id: String, text: String) -> bool;

    /// App tells IME to stop dictating (cancel/abort).
    async fn stop_dictating(&self, session_id: String) -> bool;

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
