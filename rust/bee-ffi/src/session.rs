use bee_rpc::SessionConfig;
use bee_transcribe::Session;

pub(crate) struct SessionInner {
    pub(crate) session: Option<Session<'static>>,
    pub(crate) config: SessionConfig,
}

// SAFETY: Session contains MLX arrays (Metal buffers) accessed sequentially.
// The tokio::Mutex ensures exclusive access.
unsafe impl Send for SessionInner {}
