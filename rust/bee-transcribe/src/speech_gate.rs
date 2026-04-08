//! Audio buffering, chunking, and VAD gating.

use bee_vad::SileroVad;

/// Whether a chunk should be decoded or was silence.
pub enum FeedResult {
    /// Chunk was silence — skip decoding.
    Silence,
    /// Chunk contains speech — proceed to decode.
    Decode,
}

/// Buffers incoming audio, chunks it, and gates on speech detection via VAD.
pub struct SpeechGate {
    vad: SileroVad,
    buffer: Vec<f32>,
    audio: Vec<f32>,
    chunk_size_samples: usize,
    pub(crate) chunk_count: usize,
    speech_detected: bool,
    vad_threshold: f32,
}

impl SpeechGate {
    pub fn new(vad: SileroVad, chunk_size_samples: usize, vad_threshold: f32) -> Self {
        Self {
            vad,
            buffer: Vec::new(),
            audio: Vec::new(),
            chunk_size_samples,
            chunk_count: 0,
            speech_detected: false,
            vad_threshold,
        }
    }

    /// Feed raw 16kHz mono samples. Returns `Some(Decode)` when a speech chunk
    /// was appended to the audio buffer, `Some(Silence)` when a chunk was
    /// silence, or `None` if still buffering.
    pub fn feed(&mut self, samples: &[f32]) -> Option<FeedResult> {
        self.buffer.extend_from_slice(samples);

        if self.buffer.len() < self.chunk_size_samples {
            tracing::trace!(
                "speech_gate: buffering {}/{}",
                self.buffer.len(),
                self.chunk_size_samples
            );
            return None;
        }

        let chunk: Vec<f32> = self.buffer.drain(..self.chunk_size_samples).collect();

        // Pre-speech gate: wait for first speech detection
        if !self.speech_detected {
            let prob = self.vad.process_audio(&chunk).unwrap_or(0.0);
            if prob < self.vad_threshold {
                tracing::debug!("speech_gate: pre-speech silence (vad={prob:.3})");
                return Some(FeedResult::Silence);
            }
            tracing::info!("speech_gate: speech detected (vad={prob:.3})");
            self.speech_detected = true;
        }

        self.audio.extend_from_slice(&chunk);
        self.chunk_count += 1;

        // Mid-speech silence: skip decode but keep audio
        if self.chunk_count > 1 {
            let prob = self.vad.process_audio(&chunk).unwrap_or(0.0);
            if prob < self.vad_threshold {
                tracing::debug!("speech_gate: mid-speech silence (vad={prob:.3}), skipping decode");
                return Some(FeedResult::Silence);
            }
        }

        Some(FeedResult::Decode)
    }

    /// Flush remaining buffer into the audio (for finish).
    pub fn flush(&mut self) {
        if !self.buffer.is_empty() {
            self.audio.append(&mut self.buffer);
            self.chunk_count += 1;
        }
    }

    /// The accumulated audio samples for the current segment.
    pub fn audio(&self) -> &[f32] {
        &self.audio
    }

    /// Trim audio after a commit. Keeps samples after `cut_samples`.
    pub fn rotate(&mut self, cut_samples: usize) {
        let cut = cut_samples.min(self.audio.len());
        self.audio = self.audio[cut..].to_vec();
    }

    /// Skip warm-up phase (used after rotation).
    pub fn skip_warmup(&mut self) {
        self.chunk_count = 3;
    }
}
