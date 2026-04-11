/// Policy hook that decides where to cut a ready chunk.
///
/// Intent:
/// - the cutter owns the small amount of business logic that chooses a cut
/// - the cutter never mutates utterance state directly
pub(crate) trait Cutter {
    /// Chooses a cut for `chunk`.
    ///
    /// Invariant:
    /// - returned cuts must refer to utterance-global token coordinates
    fn cut(&mut self, chunk: &super::ChunkInfo) -> super::Cut;
}

/// Observer hook for utterance lifecycle events.
///
/// Intent:
/// - side effects, logging, debug capture, and inspection live here
/// - listeners observe state transitions; they do not decide cuts
pub(crate) trait Listener {
    /// Called when the utterance has produced a new chunk ready for a cut decision.
    fn on_chunk(&mut self, _chunk: &super::ChunkInfo) {}

    /// Called immediately after the utterance applies a cut.
    fn on_cut(&mut self, _chunk: &super::ChunkInfo, _cut: super::Cut) {}
}

/// Streaming utterance state for the next rollback model.
///
/// Intent:
/// - audio is the only public ingress
/// - inference and chunk construction happen inside this type
/// - cut application happens inside this type
/// - external policy is delegated to [`Cutter`]
/// - external observation is delegated to [`Listener`]
///
/// Non-goals:
/// - this scaffold does not yet implement decode scheduling or cut application
pub(crate) struct Utterance {
    /// Buffered audio owned in utterance-global sample space.
    pub(crate) audio: super::AudioBuffer,
    /// Most recent chunk information made available for cutting.
    pub(crate) chunk: Option<super::ChunkInfo>,
    /// Boxed cut policy used by this utterance.
    pub(crate) cutter: Box<dyn Cutter>,
    /// Boxed event sink used by this utterance.
    pub(crate) listener: Box<dyn Listener>,
}

impl Utterance {
    /// Creates a new utterance with empty audio and no chunk yet.
    pub(crate) fn new(cutter: Box<dyn Cutter>, listener: Box<dyn Listener>) -> Self {
        Self {
            audio: super::AudioBuffer::new(super::SampleIndex::new(0), Vec::new()),
            chunk: None,
            cutter,
            listener,
        }
    }

    /// Appends audio to the utterance.
    ///
    /// Intent:
    /// - audio is the only public input into utterance state
    /// - future implementations will decide internally when enough audio exists
    ///   to run inference and refresh [`super::ChunkInfo`]
    pub(crate) fn push_audio(&mut self, buffer: super::AudioBuffer) {
        if self.audio.is_empty() {
            self.audio = buffer;
        } else {
            self.audio.push_end(buffer);
        }
    }
}
