use anyhow::Result;
use clap::Parser;
use qwen3_asr::{AsrInference, TranscribeOptions};
use std::path::Path;

#[derive(Parser)]
struct Args {
    /// Path to model directory
    #[arg(short, long, default_value = "~/Library/Caches/qwen3-asr/Qwen--Qwen3-ASR-0.6B")]
    model: String,

    /// Path to WAV file to transcribe
    #[arg(short, long)]
    audio: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let model_dir = shellexpand::tilde(&args.model).to_string();
    println!("Loading Qwen3 ASR from '{}'...", model_dir);

    let device = qwen3_asr::best_device();
    println!("Device: {:?}", device);

    let engine = AsrInference::load(Path::new(&model_dir), device)?;

    println!("Transcribing '{}'...", args.audio);
    let result = engine.transcribe(&args.audio, TranscribeOptions::default())?;

    println!("Language: {}", result.language);
    println!("Text: {}", result.text);
    Ok(())
}
