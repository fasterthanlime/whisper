use crate::tokens::UtteranceTokenRange;
use crate::{
    AudioBuffer, ComparisonPhone, Cut, FeedOutput, OutputToken, SampleOffset, SampleRange, Tape,
    TimedToken, TokenId, TokenIndex, ZipaTiming,
};

const DEFAULT_PREVIEW_REWRITE_TOKENS: usize = 5;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DecodeMode {
    RebuildPromptEachFeed,
    PersistentKv,
}

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
    fn cut(&mut self, tokens: &[OutputToken]) -> Cut;
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
/// - the moving stable/carry/preview boundaries are tracked in token space
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

    /// Token boundary through which the utterance has been promoted into `stable`.
    ///
    /// Invariant:
    /// - zero means no tokens are stable yet
    /// - tokens before this boundary are `stable`
    /// - any sample/time boundaries are derived from this token boundary, not stored separately
    stable_through: TokenIndex,

    /// Token boundary through which the utterance is retained as `carry`.
    ///
    /// Invariant:
    /// - `stable_through <= carry_through <= tape.end()`
    /// - tokens in `[stable_through, carry_through)` are the replay bridge
    /// - tokens in `[carry_through, tape.end())` are the current `preview`
    carry_through: TokenIndex,

    /// Canonical token-aligned output tape plus synchronized ASR rollback state.
    tape: Tape,

    /// How many tail tokens the next feed may rewrite before a new cut is chosen.
    preview_rewrite_tokens: usize,

    /// Streaming decode behavior for this utterance.
    decode_mode: DecodeMode,

    /// Number of feed steps processed so far.
    feed_count: usize,

    /// Next decoder-visible token position for persistent-KV mode.
    decoder_position: usize,

    /// Boxed cut policy used by this utterance.
    cutter: Box<dyn Cutter>,

    /// Boxed event sink used by this utterance.
    listener: Box<dyn Listener>,
}

impl Utterance {
    // Boxed trait objects are intentional here. We accept the cost and do not want
    // further reminders to genericize or optimize this construction path.
    /// Creates a new utterance with empty audio and all token boundaries at 0.
    pub fn new(num_layers: usize, cutter: Box<dyn Cutter>, listener: Box<dyn Listener>) -> Self {
        Self {
            audio: AudioBuffer::new(SampleOffset::new(0), Vec::new()),
            stable_through: TokenIndex::new(0),
            carry_through: TokenIndex::new(0),
            tape: Tape::new(num_layers),
            preview_rewrite_tokens: DEFAULT_PREVIEW_REWRITE_TOKENS,
            decode_mode: DecodeMode::PersistentKv,
            feed_count: 0,
            decoder_position: 0,
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
    pub fn feed(&mut self, samples: Vec<f32>) -> FeedOutput<'_> {
        self.audio.extend_samples(samples);
        self.feed_count += 1;

        self.rewrite_preview_from_carry();
        // TODO: decide when enough audio exists to run ASR
        // TODO: decode preview from carry boundary using `decode_mode`
        // TODO: optionally run cutter and promote stable/carry boundaries

        FeedOutput::new(self.tape.tokens(), self.tape.detected_language())
    }

    /// Current stable token slice.
    ///
    /// Invariant:
    /// - this is the prefix `[0, stable_through)`
    fn stable_tokens(&self) -> &[OutputToken] {
        self.tape.slice(UtteranceTokenRange::new(
            TokenIndex::new(0),
            self.stable_through,
        ))
    }

    /// Current carry token slice.
    ///
    /// Invariant:
    /// - this is the bridge `[stable_through, carry_through)`
    fn carry_tokens(&self) -> &[OutputToken] {
        self.tape.slice(UtteranceTokenRange::new(
            self.stable_through,
            self.carry_through,
        ))
    }

    /// Current preview token slice.
    ///
    /// Invariant:
    /// - this is the live tail `[carry_through, tape.end())`
    fn preview_tokens(&self) -> &[OutputToken] {
        self.tape.slice(UtteranceTokenRange::new(
            self.carry_through,
            self.tape.end(),
        ))
    }

    /// Replace the current carry/preview partition after a cut.
    ///
    /// Invariant:
    /// - `stable_through <= carry_through <= tape.end()`
    fn set_stable_and_carry(&mut self, stable_through: TokenIndex, carry_through: TokenIndex) {
        assert!(
            stable_through <= carry_through,
            "stable must not exceed carry"
        );
        assert!(
            carry_through <= self.tape.end(),
            "carry must lie within the tape"
        );
        self.stable_through = stable_through;
        self.carry_through = carry_through;
    }

    /// Rewind the live tail so the next decode pass can regenerate preview from
    /// the current carry boundary.
    ///
    /// Intent:
    /// - the preview suffix is provisional and can be discarded wholesale
    /// - stable/carry boundaries remain intact
    /// - persistent-KV mode tracks the visible decoder position at the carry cut
    fn rewrite_preview_from_carry(&mut self) {
        self.tape.truncate_to(self.carry_through);
        if matches!(self.decode_mode, DecodeMode::PersistentKv) {
            self.decoder_position = self.carry_through.as_usize();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Cutter, Listener, Utterance};
    use crate::{
        ComparisonPhone, Cut, OutputToken, SampleOffset, SampleRange, TimedToken, TokenId,
        TokenIndex, ZipaTiming,
    };

    struct NoCut;
    impl Cutter for NoCut {
        fn cut(&mut self, _tokens: &[crate::OutputToken]) -> Cut {
            Cut::NoCut
        }
    }

    struct NullListener;
    impl Listener for NullListener {}

    #[test]
    fn new_utterance_starts_with_empty_stable_carry_preview() {
        let utterance = Utterance::new(2, Box::new(NoCut), Box::new(NullListener));
        assert!(utterance.stable_tokens().is_empty());
        assert!(utterance.carry_tokens().is_empty());
        assert!(utterance.preview_tokens().is_empty());
    }

    #[test]
    fn feed_rewrites_preview_from_carry_boundary() {
        let mut utterance = Utterance::new(2, Box::new(NoCut), Box::new(NullListener));
        utterance.tape.append(vec![
            dummy_output_token(0, 10),
            dummy_output_token(1, 11),
            dummy_output_token(2, 12),
            dummy_output_token(3, 13),
        ]);
        utterance.set_stable_and_carry(TokenIndex::new(1), TokenIndex::new(3));

        let output_len = {
            let output = utterance.feed(vec![0.0; 320]);
            output.tokens().len()
        };

        assert_eq!(utterance.stable_tokens().len(), 1);
        assert_eq!(utterance.carry_tokens().len(), 2);
        assert_eq!(utterance.preview_tokens().len(), 0);
        assert_eq!(output_len, 3);
        assert_eq!(utterance.decoder_position, 3);
    }

    fn dummy_output_token(index: usize, token_id: u32) -> OutputToken {
        OutputToken::new(
            TimedToken::new(
                TokenIndex::new(index),
                TokenId::new(token_id),
                SampleRange::new(
                    SampleOffset::new(index * 160),
                    SampleOffset::new((index + 1) * 160),
                ),
            ),
            None,
            None,
            Vec::<ComparisonPhone>::new(),
            Vec::new(),
            ZipaTiming::Invalid,
        )
    }
}
