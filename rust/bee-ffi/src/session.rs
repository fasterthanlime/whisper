use bee_rpc::SessionConfig;
use bee_transcribe::session_v2::SessionV2;

pub(crate) struct SessionInner {
    pub(crate) session: Option<SessionV2<'static>>,
    pub(crate) config: SessionConfig,
}

// SAFETY: Session contains MLX arrays (Metal buffers) accessed sequentially.
// The tokio::Mutex ensures exclusive access.
unsafe impl Send for SessionInner {}
