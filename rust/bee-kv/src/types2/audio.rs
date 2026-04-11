use crate::types2::{TimeRange, UtteranceTime};

/// An utterance-global sample offset.
///
/// Invariant:
/// - zero is the first sample in the utterance
/// - values are absolute within the utterance, never rebased per buffer/window
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct SampleOffset(usize);

impl SampleOffset {
    pub(crate) fn new(samples: usize) -> Self {
        Self(samples)
    }

    pub(crate) fn as_usize(self) -> usize {
        self.0
    }

    /// Converts this utterance-global sample offset into utterance-global time.
    pub(crate) fn to_time(self) -> UtteranceTime {
        UtteranceTime::from_secs(self.0 as f64 / crate::SAMPLE_RATE as f64)
    }

    pub(crate) fn saturating_add(self, count: SampleCount) -> Self {
        Self(self.0.saturating_add(count.as_usize()))
    }
}

/// A count of samples.
///
/// Invariant:
/// - this is a length, not a position
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct SampleCount(usize);

impl SampleCount {
    pub(crate) fn new(count: usize) -> Self {
        Self(count)
    }

    pub(crate) fn as_usize(self) -> usize {
        self.0
    }

    /// Converts this sample count into seconds at the fixed ASR sample rate.
    pub(crate) fn to_secs(self) -> f64 {
        self.0 as f64 / crate::SAMPLE_RATE as f64
    }
}

/// A half-open utterance-global sample range.
///
/// Invariant:
/// - `start <= end`
/// - both coordinates are utterance-global
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct SampleRange {
    /// Inclusive start of the half-open sample range in utterance-global coordinates.
    pub(crate) start: SampleOffset,
    /// Exclusive end of the half-open sample range in utterance-global coordinates.
    pub(crate) end: SampleOffset,
}

impl SampleRange {
    pub(crate) fn new(start: SampleOffset, end: SampleOffset) -> Self {
        assert!(start <= end, "sample ranges must be ordered");
        Self { start, end }
    }

    pub(crate) fn len(self) -> SampleCount {
        SampleCount::new(self.end.as_usize().saturating_sub(self.start.as_usize()))
    }

    /// Converts this utterance-global sample range into utterance-global time.
    pub(crate) fn to_time_range(self) -> TimeRange {
        TimeRange::new(self.start.to_time(), self.end.to_time())
    }
}

/// Owned audio samples anchored in utterance-global sample space.
///
/// Intent:
/// - this is the mutable real-time audio primitive for the next rollback model
/// - copies are acceptable; clarity is preferred over clever borrowing
///
/// Invariants:
/// - `utterance_start` is utterance-global
/// - the buffer covers the half-open range `[utterance_start, utterance_start + samples.len())`
/// - no end index is stored redundantly
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct AudioBuffer {
    /// Inclusive utterance-global sample index of the first stored sample.
    pub(crate) utterance_start: SampleOffset,
    /// Owned PCM samples covering a contiguous utterance-global sample interval.
    pub(crate) samples: Vec<f32>,
}

impl AudioBuffer {
    pub(crate) fn new(utterance_start: SampleOffset, samples: Vec<f32>) -> Self {
        Self {
            utterance_start,
            samples,
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub(crate) fn sample_count(&self) -> SampleCount {
        SampleCount::new(self.samples.len())
    }

    /// Returns the utterance-global sample range covered by this buffer.
    pub(crate) fn utterance_range(&self) -> SampleRange {
        SampleRange::new(
            self.utterance_start,
            self.utterance_start.saturating_add(self.sample_count()),
        )
    }

    /// Returns a copied utterance-global slice of this buffer.
    ///
    /// Intent:
    /// - callers can retain or inspect a stable timed audio slice without
    ///   mutating the source buffer
    ///
    /// Invariants:
    /// - `range` must lie entirely within this buffer's utterance-global span
    pub(crate) fn slice(&self, range: SampleRange) -> Self {
        let self_range = self.utterance_range();
        assert!(
            range.start >= self_range.start && range.end <= self_range.end,
            "audio slice must lie within the source buffer"
        );
        let local_start = range.start.as_usize() - self_range.start.as_usize();
        let local_end = range.end.as_usize() - self_range.start.as_usize();
        Self::new(range.start, self.samples[local_start..local_end].to_vec())
    }

    /// Appends `other` to the end of this buffer.
    ///
    /// Invariant:
    /// - `other` must begin exactly where `self` ends
    pub(crate) fn push_end(&mut self, other: Self) {
        assert!(
            other.utterance_start == self.utterance_range().end,
            "audio buffers must be contiguous when appended"
        );
        self.samples.extend(other.samples);
    }

    /// Drops `count` samples from the front of this buffer.
    ///
    /// Invariant:
    /// - `count` must not exceed the current sample count
    pub(crate) fn drop_front(&mut self, count: SampleCount) {
        assert!(
            count <= self.sample_count(),
            "cannot drop more samples than the buffer contains"
        );
        self.samples.drain(..count.as_usize());
        self.utterance_start = self.utterance_start.saturating_add(count);
    }
}
