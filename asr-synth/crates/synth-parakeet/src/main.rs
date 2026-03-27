use anyhow::Result;
use clap::Parser;
use parakeet_rs::Transcriber;
use std::io::{self, BufRead, Write};

#[derive(Parser)]
struct Args {
    /// Path to model directory
    #[arg(short, long, default_value = "models/parakeet-tdt")]
    model: String,

    /// Path to WAV file to transcribe (if omitted, reads paths from stdin)
    #[arg(short, long)]
    audio: Option<String>,
}

fn transcribe_wav(parakeet: &mut parakeet_rs::ParakeetTDT, path: &str) -> Result<String> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.into_samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.unwrap() as f32 / max)
                .collect()
        }
    };

    let result = parakeet.transcribe_samples(samples, spec.sample_rate, spec.channels, None)?;
    Ok(result.text)
}

fn main() -> Result<()> {
    let args = Args::parse();

    eprintln!("Loading Parakeet TDT from '{}'...", args.model);
    let mut parakeet = parakeet_rs::ParakeetTDT::from_pretrained(&args.model, None)?;
    eprintln!("Parakeet ready");

    if let Some(audio) = &args.audio {
        // Single file mode
        let text = transcribe_wav(&mut parakeet, audio)?;
        println!("Text: {text}");
    } else {
        // Server mode: read WAV paths from stdin, write JSON to stdout
        eprintln!("Server mode: reading WAV paths from stdin");
        let stdin = io::stdin();
        let mut stdout = io::stdout();
        for line in stdin.lock().lines() {
            let path = line?;
            let path = path.trim();
            if path.is_empty() {
                continue;
            }
            match transcribe_wav(&mut parakeet, path) {
                Ok(text) => {
                    serde_json::to_writer(&mut stdout, &serde_json::json!({"text": text}))?;
                    stdout.write_all(b"\n")?;
                    stdout.flush()?;
                }
                Err(e) => {
                    serde_json::to_writer(
                        &mut stdout,
                        &serde_json::json!({"error": e.to_string()}),
                    )?;
                    stdout.write_all(b"\n")?;
                    stdout.flush()?;
                }
            }
        }
    }

    Ok(())
}
