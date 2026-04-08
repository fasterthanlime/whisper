use rustfft::{num_complex::Complex, FftPlanner};

use crate::audio::AudioBuffer;

#[derive(Debug, Clone, PartialEq)]
pub struct FbankParams {
    pub sample_rate_hz: u32,
    pub frame_length_ms: f32,
    pub frame_shift_ms: f32,
    pub round_to_power_of_two: bool,
    pub remove_dc_offset: bool,
    pub preemph_coeff: f32,
    pub window_type: WindowType,
    pub dither: f32,
    pub snip_edges: bool,
    pub energy_floor: f32,
    pub raw_energy: bool,
    pub use_energy: bool,
    pub use_fft_mag: bool,
    pub low_freq_hz: f32,
    pub high_freq_hz: f32,
    pub num_filters: usize,
    pub torchaudio_compatible_mel_scale: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowType {
    Povey,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FbankFeatures {
    pub data: Vec<f32>,
    pub num_frames: usize,
    pub num_filters: usize,
}

pub struct FbankExtractor {
    params: FbankParams,
    window_length: usize,
    window_shift: usize,
    fft_length: usize,
    window: Vec<f32>,
    mel_filters: Vec<f32>, // [fft_bins, num_filters]
    fft_bins: usize,
}

impl Default for FbankParams {
    fn default() -> Self {
        Self {
            sample_rate_hz: 16_000,
            frame_length_ms: 25.0,
            frame_shift_ms: 10.0,
            round_to_power_of_two: true,
            remove_dc_offset: true,
            preemph_coeff: 0.97,
            window_type: WindowType::Povey,
            dither: 0.0,
            snip_edges: false,
            energy_floor: 1e-10,
            raw_energy: true,
            use_energy: false,
            use_fft_mag: false,
            low_freq_hz: 20.0,
            high_freq_hz: -400.0,
            num_filters: 80,
            torchaudio_compatible_mel_scale: true,
        }
    }
}

impl FbankExtractor {
    pub fn new(params: FbankParams) -> Self {
        let window_length =
            ((params.frame_length_ms / 1000.0) * params.sample_rate_hz as f32).floor() as usize;
        let window_shift =
            ((params.frame_shift_ms / 1000.0) * params.sample_rate_hz as f32).floor() as usize;
        let fft_length = if params.round_to_power_of_two {
            next_power_of_two(window_length)
        } else {
            window_length
        };
        let fft_bins = fft_length / 2 + 1;
        let window = create_frame_window(window_length, params.window_type);
        let mel_filters = if params.torchaudio_compatible_mel_scale {
            create_torchaudio_mel_bank(
                params.num_filters,
                fft_length,
                params.sample_rate_hz as f32,
                params.low_freq_hz,
                params.high_freq_hz,
            )
        } else {
            create_torchaudio_mel_bank(
                params.num_filters,
                fft_length,
                params.sample_rate_hz as f32,
                params.low_freq_hz,
                params.high_freq_hz,
            )
        };
        Self {
            params,
            window_length,
            window_shift,
            fft_length,
            window,
            mel_filters,
            fft_bins,
        }
    }

    pub fn extract(&self, audio: &AudioBuffer) -> anyhow::Result<FbankFeatures> {
        anyhow::ensure!(
            audio.sample_rate_hz == self.params.sample_rate_hz,
            "expected {} Hz audio, got {} Hz",
            self.params.sample_rate_hz,
            audio.sample_rate_hz
        );

        let frames = get_strided_frames(&audio.samples, self.window_length, self.window_shift);
        let num_frames = frames.len();
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(self.fft_length);
        let mut fft_input = vec![Complex::<f32>::new(0.0, 0.0); self.fft_length];
        let mut data = vec![0.0f32; num_frames * self.params.num_filters];

        for (frame_idx, frame) in frames.iter().enumerate() {
            let mut processed = frame.clone();

            if self.params.remove_dc_offset {
                let mean = processed.iter().sum::<f32>() / processed.len() as f32;
                for sample in &mut processed {
                    *sample -= mean;
                }
            }

            if self.params.preemph_coeff != 0.0 {
                let mut prev = processed[0];
                for sample in processed.iter_mut().skip(1) {
                    let current = *sample;
                    *sample = current - self.params.preemph_coeff * prev;
                    prev = current;
                }
                processed[0] -= self.params.preemph_coeff * processed[0];
            }

            for (dst, (&sample, &window)) in fft_input
                .iter_mut()
                .zip(processed.iter().zip(self.window.iter()))
            {
                *dst = Complex::new(sample * window, 0.0);
            }
            for dst in fft_input.iter_mut().skip(self.window_length) {
                *dst = Complex::new(0.0, 0.0);
            }

            fft.process(&mut fft_input);

            for filter_idx in 0..self.params.num_filters {
                let mut energy = 0.0f32;
                for bin_idx in 0..self.fft_bins {
                    let c = fft_input[bin_idx];
                    let spec = if self.params.use_fft_mag {
                        (c.re * c.re + c.im * c.im).sqrt()
                    } else {
                        c.re * c.re + c.im * c.im
                    };
                    energy +=
                        spec * self.mel_filters[bin_idx * self.params.num_filters + filter_idx];
                }
                data[frame_idx * self.params.num_filters + filter_idx] =
                    energy.max(f32::EPSILON).ln();
            }
        }

        Ok(FbankFeatures {
            data,
            num_frames,
            num_filters: self.params.num_filters,
        })
    }
}

fn get_strided_frames(samples: &[f32], window_length: usize, window_shift: usize) -> Vec<Vec<f32>> {
    let num_samples = samples.len();
    let num_frames = (num_samples + (window_shift / 2)) / window_shift;
    let new_num_samples = (num_frames.saturating_sub(1)) * window_shift + window_length;
    let npad = new_num_samples as isize - num_samples as isize;
    let npad_left = (window_length - window_shift) / 2;
    let npad_right = npad - npad_left as isize;

    let mut padded = Vec::with_capacity(npad_left + num_samples + npad_right.max(0) as usize);
    padded.extend(samples[..npad_left.min(num_samples)].iter().rev().copied());
    padded.extend_from_slice(samples);
    if npad_right > 0 {
        padded.extend(
            samples[num_samples - npad_right as usize..]
                .iter()
                .rev()
                .copied(),
        );
    }

    (0..num_frames)
        .map(|frame_idx| {
            let start = frame_idx * window_shift;
            padded[start..start + window_length].to_vec()
        })
        .collect()
}

fn next_power_of_two(x: usize) -> usize {
    if x == 0 {
        1
    } else {
        x.next_power_of_two()
    }
}

fn create_frame_window(window_size: usize, window_type: WindowType) -> Vec<f32> {
    match window_type {
        WindowType::Povey => (0..window_size)
            .map(|i| {
                let angle = 2.0 * std::f32::consts::PI * i as f32 / (window_size - 1) as f32;
                (0.5 - 0.5 * angle.cos()).powf(0.85)
            })
            .collect(),
    }
}

fn lin2mel(x: f32) -> f32 {
    1127.0 * (1.0 + x / 700.0).ln()
}

fn create_torchaudio_mel_bank(
    num_filters: usize,
    fft_length: usize,
    sample_freq: f32,
    low_freq: f32,
    mut high_freq: f32,
) -> Vec<f32> {
    let num_fft_bins = fft_length / 2;
    let nyquist = 0.5 * sample_freq;
    if high_freq <= 0.0 {
        high_freq += nyquist;
    }
    let fft_bin_width = sample_freq / fft_length as f32;
    let mel_low_freq = lin2mel(low_freq);
    let mel_high_freq = lin2mel(high_freq);
    let mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_filters + 1) as f32;

    let mut bins = vec![0.0f32; (num_fft_bins + 1) * num_filters];
    for bin in 0..num_filters {
        let left_mel = mel_low_freq + bin as f32 * mel_freq_delta;
        let center_mel = mel_low_freq + (bin as f32 + 1.0) * mel_freq_delta;
        let right_mel = mel_low_freq + (bin as f32 + 2.0) * mel_freq_delta;

        for fft_bin in 0..num_fft_bins {
            let mel = lin2mel(fft_bin_width * fft_bin as f32);
            let up_slope = (mel - left_mel) / (center_mel - left_mel);
            let down_slope = (right_mel - mel) / (right_mel - center_mel);
            bins[fft_bin * num_filters + bin] = up_slope.min(down_slope).max(0.0);
        }
    }
    bins
}

#[cfg(test)]
mod tests {
    use super::{FbankExtractor, FbankParams};
    use crate::audio::load_wav_mono_f32;

    #[test]
    fn matches_reference_summary_for_known_wav() {
        let home = std::env::var_os("HOME").unwrap();
        let path = std::path::PathBuf::from(home)
            .join("bearcove/bee/data/phonetic-seed/audio-wav/authored_282_take_1.wav");
        let audio = load_wav_mono_f32(path).unwrap();
        let extractor = FbankExtractor::new(FbankParams::default());
        let feats = extractor.extract(&audio).unwrap();

        assert_eq!(feats.num_frames, 120);
        assert_eq!(feats.num_filters, 80);

        let row0 = &feats.data[..80];
        let last = &feats.data[(feats.num_frames - 1) * 80..feats.num_frames * 80];

        let expected_first = [
            -7.26802, -6.653513, -7.136528, -7.233698, -7.209892, -7.214974, -7.196978, -7.560191,
        ];
        let expected_last = [
            -6.151392, -6.141901, -7.884505, -8.159524, -7.953354, -7.431382, -6.63212, -5.984893,
        ];

        for (actual, expected) in row0.iter().take(8).zip(expected_first) {
            assert!((actual - expected).abs() < 1e-3, "{actual} vs {expected}");
        }
        for (actual, expected) in last.iter().take(8).zip(expected_last) {
            assert!((actual - expected).abs() < 1e-3, "{actual} vs {expected}");
        }

        let sum: f32 = feats.data.iter().sum();
        let mean = sum / feats.data.len() as f32;
        assert!((sum - (-47094.851562)).abs() < 0.1, "{sum}");
        assert!((mean - (-4.905714)).abs() < 1e-4, "{mean}");
    }
}
