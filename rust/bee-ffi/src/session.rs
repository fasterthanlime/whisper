use bee_transcribe::Session;

pub(crate) struct SessionInner {
    pub(crate) session: Option<Session<'static>>,
}

// SAFETY: Session contains MLX arrays (Metal buffers) accessed sequentially.
// The tokio::Mutex ensures exclusive access.
unsafe impl Send for SessionInner {}
