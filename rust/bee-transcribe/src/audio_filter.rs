//! Composable audio processing filters.
//!
//! Audio samples pass through a chain of filters before reaching the decoder.
//! Filters can transform audio (normalization, DC removal) or gate it (VAD).
//! Order matters: signal hygiene (DC removal, clipping guard) runs first,
//! then VAD (trained on raw-ish audio), then heavier normalization.

use crate::audio_buffer::AudioBuffer;

/// A single audio processing filter.
///
/// Returns `Some(processed)` to pass audio downstream,
/// or `None` to gate it (e.g. silence detected by VAD).
pub trait AudioFilter: Send {
    fn process(&mut self, chunk: AudioBuffer) -> Option<AudioBuffer>;

    /// Reset filter state (e.g. between sessions).
    fn reset(&mut self) {}
}

/// A chain of audio filters. Processes audio through each filter in order.
/// If any filter returns `None`, the chain short-circuits.
pub struct AudioFilterChain {
    filters: Vec<Box<dyn AudioFilter>>,
}

impl AudioFilterChain {
    pub fn new() -> Self {
        Self {
            filters: Vec::new(),
        }
    }

    /// Add a filter to the end of the chain.
    pub fn push(&mut self, filter: impl AudioFilter + 'static) {
        self.filters.push(Box::new(filter));
    }

    /// Process audio through the chain. Returns `None` if any filter gates.
    pub fn process(&mut self, chunk: AudioBuffer) -> Option<AudioBuffer> {
        let mut current = chunk;
        for filter in &mut self.filters {
            current = filter.process(current)?;
        }
        Some(current)
    }

    /// Reset all filters.
    pub fn reset(&mut self) {
        for filter in &mut self.filters {
            filter.reset();
        }
    }
}

impl Default for AudioFilterChain {
    fn default() -> Self {
        Self::new()
    }
}

// ── Signal hygiene filters ─────────────────────────────────────────────

/// Removes DC offset from audio by subtracting a running mean.
/// This is lightweight signal hygiene — runs before VAD.
pub struct DcOffsetFilter {
    running_mean: f64,
    /// Smoothing factor (0..1). Smaller = slower adaptation.
    alpha: f64,
}

impl DcOffsetFilter {
    pub fn new() -> Self {
        Self {
            running_mean: 0.0,
            alpha: 0.001,
        }
    }
}

impl AudioFilter for DcOffsetFilter {
    fn process(&mut self, chunk: AudioBuffer) -> Option<AudioBuffer> {
        let rate = chunk.sample_rate();
        let mut samples: Vec<f32> = chunk.samples().to_vec();
        for sample in &mut samples {
            self.running_mean =
                self.running_mean * (1.0 - self.alpha) + *sample as f64 * self.alpha;
            *sample -= self.running_mean as f32;
        }
        Some(AudioBuffer::new(samples, rate))
    }

    fn reset(&mut self) {
        self.running_mean = 0.0;
    }
}

impl Default for DcOffsetFilter {
    fn default() -> Self {
        Self::new()
    }
}

/// Clamps audio samples to [-1.0, 1.0] to prevent clipping artifacts
/// from reaching the model.
pub struct ClippingGuard;

impl ClippingGuard {
    pub fn new() -> Self {
        Self
    }
}

impl AudioFilter for ClippingGuard {
    fn process(&mut self, chunk: AudioBuffer) -> Option<AudioBuffer> {
        let rate = chunk.sample_rate();
        let samples: Vec<f32> = chunk
            .samples()
            .iter()
            .map(|&s| s.clamp(-1.0, 1.0))
            .collect();
        Some(AudioBuffer::new(samples, rate))
    }
}

impl Default for ClippingGuard {
    fn default() -> Self {
        Self::new()
    }
}

// ── VAD filter ─────────────────────────────────────────────────────────

/// Voice Activity Detection filter using Silero VAD.
/// Gates audio: returns `None` for silence chunks.
/// Tracks speech → silence transitions: after `SILENCE_CHUNKS_TO_GATE`
/// consecutive silence chunks, gates again (end-of-speech).
pub struct VadFilter {
    vad: bee_vad::SileroVad,
    speech_detected: bool,
    /// Consecutive silence chunks since last speech.
    silence_streak: usize,
    threshold: f32,
}

/// How many consecutive silence chunks after speech before we gate again.
const SILENCE_CHUNKS_TO_GATE: usize = 3;

impl VadFilter {
    pub fn new(vad: bee_vad::SileroVad, threshold: f32) -> Self {
        Self {
            vad,
            speech_detected: false,
            silence_streak: 0,
            threshold,
        }
    }
}

impl AudioFilter for VadFilter {
    fn process(&mut self, chunk: AudioBuffer) -> Option<AudioBuffer> {
        let prob = self.vad.process_audio(chunk.samples()).unwrap_or(0.0);

        if prob >= self.threshold {
            // Speech detected
            if !self.speech_detected {
                tracing::info!("vad_filter: speech detected (prob={prob:.3})");
            }
            self.speech_detected = true;
            self.silence_streak = 0;
            return Some(chunk);
        }

        // Below threshold — silence
        if !self.speech_detected {
            tracing::debug!("vad_filter: pre-speech silence (prob={prob:.3})");
            return None;
        }

        // Mid/post-speech silence
        self.silence_streak += 1;
        if self.silence_streak >= SILENCE_CHUNKS_TO_GATE {
            tracing::info!(
                "vad_filter: end-of-speech after {} silence chunks (prob={prob:.3})",
                self.silence_streak
            );
            self.speech_detected = false;
            self.silence_streak = 0;
            return None;
        }

        tracing::debug!(
            "vad_filter: mid-speech silence {}/{} (prob={prob:.3})",
            self.silence_streak,
            SILENCE_CHUNKS_TO_GATE,
        );
        Some(chunk)
    }

    fn reset(&mut self) {
        self.speech_detected = false;
        self.silence_streak = 0;
    }
}

// ── Post-VAD normalization ─────────────────────────────────────────────

/// RMS normalization: scales audio so its RMS energy matches a target level.
/// Runs after VAD so silence chunks (already gated) don't dilute the estimate.
/// Uses a per-chunk approach — each chunk is independently normalized.
pub struct RmsNormalizer {
    /// Target RMS level. Speech in [-1, 1] typically sits around 0.05–0.1.
    target_rms: f32,
}

impl RmsNormalizer {
    pub fn new(target_rms: f32) -> Self {
        Self { target_rms }
    }
}

impl AudioFilter for RmsNormalizer {
    fn process(&mut self, chunk: AudioBuffer) -> Option<AudioBuffer> {
        let samples = chunk.samples();
        let rms = (samples.iter().map(|&s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
        if rms < 1e-8 {
            // Near-silence, don't amplify noise
            return Some(chunk);
        }
        let gain = self.target_rms / rms;
        // Cap gain to avoid blowing up quiet chunks
        let gain = gain.min(10.0);
        let rate = chunk.sample_rate();
        let normalized: Vec<f32> = samples
            .iter()
            .map(|&s| (s * gain).clamp(-1.0, 1.0))
            .collect();
        tracing::trace!(rms, gain, target = self.target_rms, "rms_normalizer");
        Some(AudioBuffer::new(normalized, rate))
    }
}

/// Build the default audio filter chain:
/// DC removal → clipping guard → VAD → RMS normalization.
pub fn default_filter_chain(vad: bee_vad::SileroVad, vad_threshold: f32) -> AudioFilterChain {
    let mut chain = AudioFilterChain::new();
    chain.push(DcOffsetFilter::new());
    chain.push(ClippingGuard::new());
    chain.push(VadFilter::new(vad, vad_threshold));
    chain.push(RmsNormalizer::new(0.08));
    chain
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_buffer::SampleRate;

    #[test]
    fn clipping_guard_clamps() {
        let buf = AudioBuffer::new(vec![-2.0, -0.5, 0.0, 0.5, 2.0], SampleRate::HZ_16000);
        let guard = &mut ClippingGuard::new();
        let result = guard.process(buf).unwrap();
        assert_eq!(result.samples(), &[-1.0, -0.5, 0.0, 0.5, 1.0]);
    }

    #[test]
    fn dc_offset_removal() {
        // Feed several chunks of DC-offset audio through the filter
        let filter = &mut DcOffsetFilter::new();
        let mut last = 0.5f32;
        for _ in 0..20 {
            let samples: Vec<f32> = (0..1600).map(|_| 0.5).collect();
            let buf = AudioBuffer::new(samples, SampleRate::HZ_16000);
            let result = filter.process(buf).unwrap();
            last = *result.samples().last().unwrap();
        }
        // After many chunks, DC offset should be mostly removed
        assert!(
            last.abs() < 0.05,
            "DC offset should be mostly removed, got {last}"
        );
    }

    #[test]
    fn chain_short_circuits() {
        struct AlwaysGate;
        impl AudioFilter for AlwaysGate {
            fn process(&mut self, _chunk: AudioBuffer) -> Option<AudioBuffer> {
                None
            }
        }
        struct Unreachable;
        impl AudioFilter for Unreachable {
            fn process(&mut self, _chunk: AudioBuffer) -> Option<AudioBuffer> {
                panic!("should not be called");
            }
        }

        let mut chain = AudioFilterChain::new();
        chain.push(AlwaysGate);
        chain.push(Unreachable);

        let buf = AudioBuffer::new(vec![0.0; 100], SampleRate::HZ_16000);
        assert!(chain.process(buf).is_none());
    }

    #[test]
    fn chain_passes_through() {
        let mut chain = AudioFilterChain::new();
        chain.push(ClippingGuard::new());
        chain.push(DcOffsetFilter::new());

        let buf = AudioBuffer::new(vec![0.1, 0.2, 0.3], SampleRate::HZ_16000);
        let result = chain.process(buf);
        assert!(result.is_some());
        assert_eq!(result.unwrap().len(), 3);
    }
}
