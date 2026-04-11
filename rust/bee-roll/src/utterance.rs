use crate::{AudioBuffer, Cut, SampleOffset, TimedToken, TokenIndex};

/// Policy hook that decides where to cut a ready chunk.
///
/// Intent:
/// - the cutter owns the small amount of business logic that chooses a cut
/// - the cutter never mutates utterance state directly
pub trait Cutter {
    /// Chooses a cut for `tokens`.
    ///
    /// Invariant:
    /// - returned cuts must refer to utterance-global token coordinates
    fn cut(&mut self, tokens: &[TimedToken]) -> Cut;
}

/// Observer hook for utterance lifecycle events.
///
/// Intent:
/// - side effects, logging, debug capture, and inspection live here
/// - event methods are intentionally deferred until the debug/HTML contract exists
pub trait Listener {}

/// Streaming utterance state for the next rollback model.
///
/// Intent:
/// - audio is the only public ingress
/// - utterance audio is append-only and remains anchored at utterance sample 0
/// - utterance-owned token, commit, and ASR state stay synchronized under one owner
/// - inference and chunk construction happen inside this type
/// - cut application happens inside this type
/// - the moving committed boundary is tracked in token space
/// - external policy is delegated to [`Cutter`]
/// - external observation is delegated to [`Listener`]
///
/// Non-goals:
/// - this scaffold does not yet implement decode scheduling or cut application
pub struct Utterance {
    /// Append-only utterance audio owned in utterance-global sample space.
    ///
    /// Invariant:
    /// - this buffer remains anchored at utterance sample 0
    audio: AudioBuffer,

    /// Token boundary through which the utterance has been committed.
    ///
    /// Invariant:
    /// - zero means no tokens are committed yet
    /// - this is the primary moving boundary in utterance state
    /// - any sample/time boundaries are derived from this token boundary, not stored separately
    committed_through: TokenIndex,

    /// Boxed cut policy used by this utterance.
    cutter: Box<dyn Cutter>,

    /// Boxed event sink used by this utterance.
    listener: Box<dyn Listener>,
}

impl Utterance {
    // Boxed trait objects are intentional here. We accept the cost and do not want
    // further reminders to genericize or optimize this construction path.
    /// Creates a new utterance with empty audio and committed boundary 0.
    pub fn new(cutter: Box<dyn Cutter>, listener: Box<dyn Listener>) -> Self {
        Self {
            audio: AudioBuffer::new(SampleOffset::new(0), Vec::new()),
            committed_through: TokenIndex::new(0),
            cutter,
            listener,
        }
    }

    /// Feeds raw samples into the utterance recording buffer.
    ///
    /// Intent:
    /// - audio is the only public input into utterance state
    /// - callers provide only raw samples, never timed audio buffers
    /// - utterance timing stays internal and is derived from sample position in this append-only buffer
    /// - future implementations will decide internally when enough audio exists
    ///   to run inference and construct transient token slices for cutting
    pub fn feed(&mut self, samples: Vec<f32>) {
        self.audio.extend_samples(samples);
    }
}
