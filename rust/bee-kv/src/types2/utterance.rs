use crate::types2::{AudioBuffer, ChunkInfo, Cut, SampleIndex, TokenIndex};

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
    fn cut(&mut self, chunk: &ChunkInfo) -> Cut;
}

/// Observer hook for utterance lifecycle events.
///
/// Intent:
/// - side effects, logging, debug capture, and inspection live here
/// - listeners observe state transitions; they do not decide cuts
pub(crate) trait Listener {
    /// Called when the utterance has produced a new chunk ready for a cut decision.
    fn on_chunk(&mut self, _chunk: &ChunkInfo) {}

    /// Called immediately after the utterance applies a cut.
    fn on_cut(&mut self, _chunk: &ChunkInfo, _cut: Cut) {}
}

/// Streaming utterance state for the next rollback model.
///
/// Intent:
/// - audio is the only public ingress
/// - utterance audio is append-only and remains anchored at utterance sample 0
/// - inference and chunk construction happen inside this type
/// - cut application happens inside this type
/// - the moving committed boundary is tracked in token space
/// - external policy is delegated to [`Cutter`]
/// - external observation is delegated to [`Listener`]
///
/// Non-goals:
/// - this scaffold does not yet implement decode scheduling or cut application
pub(crate) struct Utterance {
    /// Append-only utterance audio owned in utterance-global sample space.
    ///
    /// Invariant:
    /// - this buffer remains anchored at utterance sample 0
    pub(crate) audio: AudioBuffer,
    /// Token boundary through which the utterance has been committed.
    ///
    /// Invariant:
    /// - this is the primary moving boundary in utterance state
    /// - any sample/time boundaries are derived from this token boundary, not stored separately
    pub(crate) committed_through: Option<TokenIndex>,
    /// Boxed cut policy used by this utterance.
    pub(crate) cutter: Box<dyn Cutter>,
    /// Boxed event sink used by this utterance.
    pub(crate) listener: Box<dyn Listener>,
}

impl Utterance {
    /// Creates a new utterance with empty audio and no committed tokens yet.
    pub(crate) fn new(cutter: Box<dyn Cutter>, listener: Box<dyn Listener>) -> Self {
        Self {
            audio: AudioBuffer::new(SampleIndex::new(0), Vec::new()),
            committed_through: None,
            cutter,
            listener,
        }
    }

    /// Appends raw samples to the utterance recording buffer.
    ///
    /// Intent:
    /// - audio is the only public input into utterance state
    /// - callers provide only raw samples, never timed audio buffers
    /// - utterance timing stays internal and is derived from sample position in this append-only buffer
    /// - future implementations will decide internally when enough audio exists
    ///   to run inference and construct transient [`ChunkInfo`] values
    pub(crate) fn push_audio(&mut self, samples: Vec<f32>) {
        assert!(
            self.audio.utterance_start == SampleIndex::new(0),
            "utterance audio buffer must remain anchored at sample 0"
        );
        self.audio.samples.extend(samples);
    }
}
