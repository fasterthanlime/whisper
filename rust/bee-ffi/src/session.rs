use bee_transcribe::Session;

pub(crate) struct AsrSession {
    /// Borrows the `&'static Engine` (Box::leaked in `load_engine`).
    pub(crate) session: Session<'static>,
    pub(crate) last_text: String,
    pub(crate) finished: bool,
}

// SAFETY: Session contains MLX arrays (Metal buffers) accessed sequentially.
// Swift calls are serialized by the caller.
unsafe impl Send for AsrSession {}
