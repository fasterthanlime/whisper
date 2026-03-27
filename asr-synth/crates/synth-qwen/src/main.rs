use anyhow::Result;
use clap::Parser;
use qwen3_asr::{AsrInference, TranscribeOptions};
use std::io::{self, BufRead, Write};
use std::path::Path;

#[derive(Parser)]
struct Args {
    /// Path to model directory
    #[arg(
        short,
        long,
        default_value = "~/Library/Caches/qwen3-asr/Alkd--qwen3-asr-gguf--qwen3_asr_1_7b_q8_0_gguf"
    )]
    model: String,

    /// Path to WAV file to transcribe (if omitted, reads paths from stdin)
    #[arg(short, long)]
    audio: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let model_dir = shellexpand::tilde(&args.model).to_string();

    eprintln!("Loading Qwen3 ASR from '{}'...", model_dir);
    let device = qwen3_asr::best_device();
    eprintln!("Device: {:?}", device);
    let engine = AsrInference::load(Path::new(&model_dir), device)?;
    eprintln!("Qwen3 ready");

    if let Some(audio) = &args.audio {
        // Single file mode
        let result = engine.transcribe(audio, TranscribeOptions::default())?;
        println!("Language: {}", result.language);
        println!("Text: {}", result.text);
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
            match engine.transcribe(path, TranscribeOptions::default()) {
                Ok(result) => {
                    serde_json::to_writer(
                        &mut stdout,
                        &serde_json::json!({
                            "text": result.text,
                            "language": result.language,
                        }),
                    )?;
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
