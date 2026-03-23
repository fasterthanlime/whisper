use anyhow::Result;
use clap::Parser;
use parakeet_rs::Transcriber;

#[derive(Parser)]
struct Args {
    /// Path to model directory
    #[arg(short, long, default_value = "models/parakeet-tdt")]
    model: String,

    /// Path to WAV file to transcribe
    #[arg(short, long)]
    audio: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("Loading Parakeet TDT model from '{}'...", args.model);
    let mut parakeet = parakeet_rs::ParakeetTDT::from_pretrained(&args.model, None)?;

    println!("Transcribing '{}'...", args.audio);

    // Load WAV
    let reader = hound::WavReader::open(&args.audio)?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.into_samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader.into_samples::<i32>().map(|s| s.unwrap() as f32 / max).collect()
        }
    };

    let result = parakeet.transcribe_samples(
        samples,
        spec.sample_rate,
        spec.channels,
        None,
    )?;

    println!("Text: {}", result.text);
    Ok(())
}
