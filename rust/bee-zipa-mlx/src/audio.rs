use std::path::Path;

use crate::Result;
use crate::error::ZipaError;

#[derive(Debug, Clone, PartialEq)]
pub struct AudioBuffer {
    pub samples: Vec<f32>,
    pub sample_rate_hz: u32,
}

pub fn load_wav_mono_f32(path: impl AsRef<Path>) -> Result<AudioBuffer> {
    let path = path.as_ref();
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();

    if spec.channels != 1 {
        return Err(ZipaError::UnsupportedWav {
            path: path.to_path_buf(),
            reason: format!("expected mono, got {} channels", spec.channels),
        });
    }

    let samples = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            if bits == 0 || bits > 32 {
                return Err(ZipaError::UnsupportedWav {
                    path: path.to_path_buf(),
                    reason: format!("unsupported integer bit depth {bits}"),
                });
            }
            let scale = (1_i64 << (bits - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|sample| sample.map(|s| s as f32 / scale))
                .collect::<std::result::Result<Vec<_>, _>>()?
        }
    };

    Ok(AudioBuffer {
        samples,
        sample_rate_hz: spec.sample_rate,
    })
}

#[cfg(test)]
mod tests {
    use super::load_wav_mono_f32;

    #[test]
    fn loads_reference_wav() {
        let home = std::env::var_os("HOME").unwrap();
        let path = std::path::PathBuf::from(home)
            .join("bearcove/bee/data/phonetic-seed/audio-wav/authored_282_take_1.wav");
        let audio = load_wav_mono_f32(path).unwrap();
        assert_eq!(audio.sample_rate_hz, 16_000);
        assert!(!audio.samples.is_empty());
    }
}
