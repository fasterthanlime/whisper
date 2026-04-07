use std::sync::Arc;

use bee_transcribe::{Engine, Session};

pub(crate) struct AsrSession {
    // We store the Arc to keep the engine alive. The Session borrows it
    // via a raw pointer with 'static lifetime — safe because the Arc
    // guarantees the engine outlives the session.
    pub(crate) _engine: Arc<Engine>,
    pub(crate) session: Session<'static>,
    pub(crate) last_text: String,
    pub(crate) finished: bool,
}

// SAFETY: Session contains MLX arrays (Metal buffers) accessed sequentially.
// Swift calls are serialized by the caller.
unsafe impl Send for AsrSession {}
