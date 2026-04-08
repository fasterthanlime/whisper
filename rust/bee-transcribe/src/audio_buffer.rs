//! Typed audio buffer with sample rate metadata and time-based operations.

use std::ops::{Add, Sub};

// ── Sample Rate ────────────────────────────────────────────────────────

/// Audio sample rate in Hz.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SampleRate(pub u32);

impl SampleRate {
    pub const HZ_16000: SampleRate = SampleRate(16000);
}

// ── Seconds ────────────────────────────────────────────────────────────

/// A duration or timestamp in seconds.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct Seconds(pub f64);

impl Seconds {
    pub const ZERO: Seconds = Seconds(0.0);

    /// Convert to a sample index at the given rate, rounding down.
    pub fn to_samples(self, rate: SampleRate) -> usize {
        (self.0 * rate.0 as f64) as usize
    }

    /// Convert from a sample count at the given rate.
    pub fn from_samples(n: usize, rate: SampleRate) -> Self {
        Seconds(n as f64 / rate.0 as f64)
    }
}

impl Add for Seconds {
    type Output = Seconds;
    fn add(self, rhs: Seconds) -> Seconds {
        Seconds(self.0 + rhs.0)
    }
}

impl Sub for Seconds {
    type Output = Seconds;
    fn sub(self, rhs: Seconds) -> Seconds {
        Seconds(self.0 - rhs.0)
    }
}

impl std::ops::AddAssign for Seconds {
    fn add_assign(&mut self, rhs: Seconds) {
        self.0 += rhs.0;
    }
}

impl std::fmt::Display for Seconds {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.3}s", self.0)
    }
}

// ── TimeRange ──────────────────────────────────────────────────────────

/// A half-open time range [start, end).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeRange {
    pub start: Seconds,
    pub end: Seconds,
}

impl TimeRange {
    pub fn new(start: Seconds, end: Seconds) -> Self {
        debug_assert!(
            end.0 >= start.0,
            "TimeRange end ({end}) must be >= start ({start})"
        );
        Self { start, end }
    }

    /// Duration of this range.
    pub fn duration(&self) -> Seconds {
        self.end - self.start
    }

    /// Whether this range overlaps with another.
    pub fn overlaps(&self, other: &TimeRange) -> bool {
        self.start.0 < other.end.0 && other.start.0 < self.end.0
    }
}

// ── AudioBuffer ────────────────────────────────────────────────────────

/// A buffer of audio samples at a known sample rate.
#[derive(Clone)]
pub struct AudioBuffer {
    samples: Vec<f32>,
    sample_rate: SampleRate,
}

impl AudioBuffer {
    /// Create a new audio buffer.
    pub fn new(samples: Vec<f32>, sample_rate: SampleRate) -> Self {
        Self {
            samples,
            sample_rate,
        }
    }

    /// Create an empty buffer at the given sample rate.
    pub fn empty(sample_rate: SampleRate) -> Self {
        Self {
            samples: Vec::new(),
            sample_rate,
        }
    }

    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    pub fn sample_rate(&self) -> SampleRate {
        self.sample_rate
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Duration of this buffer.
    pub fn duration(&self) -> Seconds {
        Seconds::from_samples(self.samples.len(), self.sample_rate)
    }

    /// Append another buffer. Panics if sample rates differ.
    pub fn append(&mut self, other: &AudioBuffer) {
        assert_eq!(
            self.sample_rate, other.sample_rate,
            "cannot append buffers with different sample rates ({:?} vs {:?})",
            self.sample_rate, other.sample_rate,
        );
        self.samples.extend_from_slice(&other.samples);
    }

    /// Slice a time range, returning a new buffer.
    /// Clamps to buffer bounds.
    pub fn slice(&self, range: TimeRange) -> AudioBuffer {
        let start = range
            .start
            .to_samples(self.sample_rate)
            .min(self.samples.len());
        let end = range
            .end
            .to_samples(self.sample_rate)
            .min(self.samples.len());
        AudioBuffer {
            samples: self.samples[start..end].to_vec(),
            sample_rate: self.sample_rate,
        }
    }

    /// Drop all samples before the given time point. Keeps the rest.
    pub fn trim_before(&mut self, t: Seconds) {
        let cut = t.to_samples(self.sample_rate).min(self.samples.len());
        self.samples.drain(..cut);
    }

    /// Split at a time point. Returns (before, after).
    pub fn split_at(&self, t: Seconds) -> (AudioBuffer, AudioBuffer) {
        let idx = t.to_samples(self.sample_rate).min(self.samples.len());
        (
            AudioBuffer {
                samples: self.samples[..idx].to_vec(),
                sample_rate: self.sample_rate,
            },
            AudioBuffer {
                samples: self.samples[idx..].to_vec(),
                sample_rate: self.sample_rate,
            },
        )
    }
}

impl std::fmt::Debug for AudioBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "AudioBuffer({} samples, {:?}, {})",
            self.samples.len(),
            self.sample_rate,
            self.duration(),
        )
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seconds_conversion() {
        let rate = SampleRate::HZ_16000;
        assert_eq!(Seconds(1.0).to_samples(rate), 16000);
        assert_eq!(Seconds(0.5).to_samples(rate), 8000);
        assert_eq!(Seconds::from_samples(16000, rate), Seconds(1.0));
        assert_eq!(Seconds::from_samples(8000, rate), Seconds(0.5));
    }

    #[test]
    fn seconds_arithmetic() {
        assert_eq!(Seconds(1.5) + Seconds(0.5), Seconds(2.0));
        assert_eq!(Seconds(2.0) - Seconds(0.5), Seconds(1.5));
        let mut s = Seconds(1.0);
        s += Seconds(0.5);
        assert_eq!(s, Seconds(1.5));
    }

    #[test]
    fn time_range_basics() {
        let r = TimeRange::new(Seconds(1.0), Seconds(2.0));
        assert_eq!(r.duration(), Seconds(1.0));
    }

    #[test]
    fn time_range_overlap() {
        let a = TimeRange::new(Seconds(1.0), Seconds(3.0));
        let b = TimeRange::new(Seconds(2.0), Seconds(4.0));
        let c = TimeRange::new(Seconds(3.0), Seconds(5.0));
        assert!(a.overlaps(&b));
        assert!(!a.overlaps(&c)); // half-open: [1,3) and [3,5) don't overlap
    }

    #[test]
    fn audio_buffer_duration() {
        let rate = SampleRate::HZ_16000;
        let buf = AudioBuffer::new(vec![0.0; 16000], rate);
        assert_eq!(buf.duration(), Seconds(1.0));
        assert_eq!(buf.len(), 16000);
    }

    #[test]
    fn audio_buffer_append() {
        let rate = SampleRate::HZ_16000;
        let mut buf = AudioBuffer::new(vec![1.0; 8000], rate);
        let other = AudioBuffer::new(vec![2.0; 8000], rate);
        buf.append(&other);
        assert_eq!(buf.len(), 16000);
        assert_eq!(buf.duration(), Seconds(1.0));
        assert_eq!(buf.samples()[0], 1.0);
        assert_eq!(buf.samples()[8000], 2.0);
    }

    #[test]
    #[should_panic(expected = "different sample rates")]
    fn audio_buffer_append_mismatched_rate() {
        let mut buf = AudioBuffer::new(vec![1.0; 100], SampleRate(16000));
        let other = AudioBuffer::new(vec![2.0; 100], SampleRate(44100));
        buf.append(&other);
    }

    #[test]
    fn audio_buffer_trim_before() {
        let rate = SampleRate::HZ_16000;
        let mut buf = AudioBuffer::new(vec![0.0; 32000], rate); // 2 seconds
        buf.trim_before(Seconds(0.5)); // cut first 0.5s = 8000 samples
        assert_eq!(buf.len(), 24000);
        assert_eq!(buf.duration(), Seconds(1.5));
    }

    #[test]
    fn audio_buffer_split_at() {
        let rate = SampleRate::HZ_16000;
        let buf = AudioBuffer::new(vec![0.0; 32000], rate); // 2 seconds
        let (before, after) = buf.split_at(Seconds(0.5));
        assert_eq!(before.len(), 8000);
        assert_eq!(after.len(), 24000);
    }

    #[test]
    fn audio_buffer_slice() {
        let rate = SampleRate::HZ_16000;
        // Fill with index values so we can verify the right slice
        let samples: Vec<f32> = (0..32000).map(|i| i as f32).collect();
        let buf = AudioBuffer::new(samples, rate);
        let sliced = buf.slice(TimeRange::new(Seconds(0.5), Seconds(1.0)));
        assert_eq!(sliced.len(), 8000);
        assert_eq!(sliced.samples()[0], 8000.0);
    }

    #[test]
    fn audio_buffer_slice_clamped() {
        let rate = SampleRate::HZ_16000;
        let buf = AudioBuffer::new(vec![0.0; 16000], rate); // 1 second
        let sliced = buf.slice(TimeRange::new(Seconds(0.5), Seconds(5.0))); // end beyond buffer
        assert_eq!(sliced.len(), 8000); // clamped to buffer end
    }

    #[test]
    fn audio_buffer_empty() {
        let buf = AudioBuffer::empty(SampleRate::HZ_16000);
        assert!(buf.is_empty());
        assert_eq!(buf.duration(), Seconds(0.0));
    }

    #[test]
    fn audio_buffer_trim_beyond_end() {
        let rate = SampleRate::HZ_16000;
        let mut buf = AudioBuffer::new(vec![0.0; 16000], rate);
        buf.trim_before(Seconds(5.0)); // way past the end
        assert!(buf.is_empty());
    }
}
