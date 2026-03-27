use anyhow::Result;
use clap::Parser;

#[derive(Parser)]
struct Args {
    /// Text to synthesize
    #[arg(
        short,
        long,
        default_value = "The serde crate handles serialization and deserialization in Rust."
    )]
    text: String,

    /// Path to a voice reference WAV file
    #[arg(short, long, default_value = "voices/amos.wav")]
    voice: String,

    /// Output WAV path
    #[arg(short, long, default_value = "output.wav")]
    output: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("Loading pocket-tts model (quantized)...");
    let model = pocket_tts::TTSModel::load_quantized("b6369a24")?;

    println!("Loading voice from '{}'...", args.voice);
    let voice_state = model.get_voice_state(&args.voice)?;

    println!("Synthesizing: {:?}", args.text);
    let audio = model.generate(&args.text, &voice_state)?;

    let sample_rate = model.sample_rate as u32;
    let samples: Vec<f32> = audio.flatten_all()?.to_vec1()?;

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&args.output, spec)?;
    for &s in &samples {
        let clamped: f32 = (s * 32767.0f32).clamp(-32768.0f32, 32767.0f32);
        writer.write_sample(clamped as i16)?;
    }
    writer.finalize()?;

    let duration = samples.len() as f32 / sample_rate as f32;
    println!(
        "Wrote {} ({:.1}s, {} Hz, {} samples)",
        args.output,
        duration,
        sample_rate,
        samples.len()
    );
    Ok(())
}
