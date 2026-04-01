use anyhow::Result;
use clap::Parser;
use beeml_pipeline::{PipelineConfig, PipelineEvent};

#[derive(Parser)]
struct Args {
    /// Voice reference WAV for TTS
    #[arg(long, default_value = "voices/amos.wav")]
    voice: String,

    /// Parakeet TDT model directory
    #[arg(long, default_value = "models/parakeet-tdt")]
    parakeet_model: String,

    /// Qwen3 ASR model directory
    #[arg(
        long,
        default_value = "~/Library/Caches/qwen3-asr/Alkd--qwen3-asr-gguf--qwen3_asr_1_7b_q8_0_gguf"
    )]
    qwen_model: String,

    /// Pre-generated sentences JSONL (if provided, skips textgen)
    #[arg(long)]
    sentences: Option<String>,

    /// Root directory for docs scanning (used if --sentences not provided)
    #[arg(long, default_value = "~/bearcove")]
    docs_root: String,

    /// Number of sentences to generate (ignored if --sentences provided)
    #[arg(short, long, default_value = "10")]
    count: usize,

    /// Output JSONL file
    #[arg(short, long, default_value = "data/output.jsonl")]
    output: String,

    /// Save TTS audio as WAVs in this directory (for review)
    #[arg(long)]
    save_audio: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let sentences = if let Some(ref path) = args.sentences {
        eprintln!("Loading sentences from {path}...");
        beeml_pipeline::load_sentences(path)?
    } else {
        let docs_root = shellexpand::tilde(&args.docs_root).to_string();
        eprintln!("Scanning {docs_root} for vocabulary...");
        let vocab = beeml_textgen::corpus::extract_vocab(&docs_root)?;
        eprintln!("Extracted {} terms", vocab.len());
        beeml_textgen::templates::generate(&vocab, args.count, None, None)
    };
    eprintln!("{} sentences ready", sentences.len());

    let config = PipelineConfig {
        voice: args.voice,
        parakeet_model: args.parakeet_model,
        qwen_model: shellexpand::tilde(&args.qwen_model).to_string(),
        save_audio: args.save_audio,
        voice_id: "amos".into(),
    };

    let pairs = beeml_pipeline::run_pipeline(&config, &sentences, |event| match event {
        PipelineEvent::Status(msg) => eprintln!("{msg}"),
        PipelineEvent::SentenceStart { index, total, text } => {
            eprint!("[{}/{}] \"{}\" ... ", index + 1, total, text);
        }
        PipelineEvent::SentenceDone { parakeet, qwen, .. } => {
            eprintln!("OK");
            eprintln!("  Parakeet: {parakeet}");
            eprintln!("  Qwen3:    {qwen}");
        }
        PipelineEvent::SentenceError { error, .. } => {
            eprintln!("ERROR: {error}");
        }
        PipelineEvent::Done { count } => {
            eprintln!("Pipeline complete: {count} pairs generated");
        }
    })?;

    beeml_pipeline::write_pairs(&args.output, &pairs)?;
    eprintln!("Wrote {} pairs to {}", pairs.len(), args.output);

    Ok(())
}
