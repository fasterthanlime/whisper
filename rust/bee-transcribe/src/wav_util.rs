use std::io::Cursor;

/// Decode WAV bytes to 16kHz mono f32 samples.
pub fn decode_wav(bytes: &[u8]) -> Result<Vec<f32>, mlx_rs::error::Exception> {
    let cursor = Cursor::new(bytes);
    let mut reader = hound::WavReader::new(cursor)
        .map_err(|e| mlx_rs::error::Exception::custom(format!("invalid WAV: {e}")))?;
    let spec = reader.spec();

    if spec.sample_rate != 16_000 {
        return Err(mlx_rs::error::Exception::custom(format!(
            "expected 16kHz WAV, got {}Hz",
            spec.sample_rate
        )));
    }

    let channels = spec.channels.max(1) as usize;
    let mut mono = Vec::new();

    match spec.sample_format {
        hound::SampleFormat::Float => {
            let mut acc = 0.0f32;
            let mut idx = 0usize;
            for sample in reader.samples::<f32>() {
                acc += sample.map_err(|e| mlx_rs::error::Exception::custom(format!("{e}")))?;
                idx += 1;
                if idx == channels {
                    mono.push(acc / channels as f32);
                    acc = 0.0;
                    idx = 0;
                }
            }
        }
        hound::SampleFormat::Int => {
            let scale = if spec.bits_per_sample <= 16 {
                i16::MAX as f32
            } else {
                ((1_i64 << (spec.bits_per_sample - 1)) - 1) as f32
            };
            let mut acc = 0.0f32;
            let mut idx = 0usize;
            for sample in reader.samples::<i32>() {
                acc += sample.map_err(|e| mlx_rs::error::Exception::custom(format!("{e}")))? as f32
                    / scale;
                idx += 1;
                if idx == channels {
                    mono.push(acc / channels as f32);
                    acc = 0.0;
                    idx = 0;
                }
            }
        }
    }

    if mono.is_empty() {
        return Err(mlx_rs::error::Exception::custom("WAV is empty"));
    }

    Ok(mono)
}
