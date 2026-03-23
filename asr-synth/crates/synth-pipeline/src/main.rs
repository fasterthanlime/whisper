use anyhow::{Context, Result};
use clap::Parser;
use parakeet_rs::Transcriber;
use std::io::Write;
use std::path::Path;

#[derive(Parser)]
struct Args {
    /// Voice reference WAV for TTS
    #[arg(long, default_value = "voices/amos.wav")]
    voice: String,

    /// Parakeet TDT model directory
    #[arg(long, default_value = "models/parakeet-tdt")]
    parakeet_model: String,

    /// Qwen3 ASR model directory
    #[arg(long, default_value = "~/Library/Caches/qwen3-asr/Alkd--qwen3-asr-gguf--qwen3_asr_1_7b_q8_0_gguf")]
    qwen_model: String,

    /// Root directory for docs scanning
    #[arg(long, default_value = "~/bearcove")]
    docs_root: String,

    /// Number of sentences to generate
    #[arg(short, long, default_value = "10")]
    count: usize,

    /// Output JSONL file
    #[arg(short, long, default_value = "data/output.jsonl")]
    output: String,

    /// Save TTS audio as WAVs in this directory (for review)
    #[arg(long)]
    save_audio: Option<String>,
}

#[derive(serde::Serialize)]
struct TrainingPair {
    original_text: String,
    spoken_text: String,
    parakeet_output: String,
    qwen_output: String,
    vocab: Vec<String>,
    voice_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    audio_file: Option<String>,
}

fn resample_24k_to_16k(samples: &[f32]) -> Result<Vec<f32>> {
    use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction};

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut resampler = SincFixedIn::<f32>::new(
        16000.0 / 24000.0,
        2.0,
        params,
        samples.len(),
        1,
    )?;
    let output = resampler.process(&[samples], None)?;
    Ok(output.into_iter().next().unwrap_or_default())
}

fn main() -> Result<()> {
    let args = Args::parse();

    // --- Step 1: Generate text ---
    let docs_root = shellexpand::tilde(&args.docs_root).to_string();
    eprintln!("Scanning {docs_root} for vocabulary...");
    let vocab = synth_textgen::corpus::extract_vocab(&docs_root)?;
    eprintln!("Extracted {} terms", vocab.len());

    let sentences = synth_textgen::templates::generate(&vocab, args.count);
    eprintln!("Generated {} sentences", sentences.len());

    // --- Step 2: Load TTS ---
    eprintln!("Loading pocket-tts (quantized)...");
    let tts = pocket_tts::TTSModel::load_quantized("b6369a24")?;
    let voice_state = tts.get_voice_state(&args.voice)
        .context("loading voice reference WAV")?;
    let tts_sample_rate = tts.sample_rate as u32;
    eprintln!("TTS ready ({tts_sample_rate} Hz)");

    // --- Step 3: Load ASR engines ---
    eprintln!("Loading Parakeet TDT...");
    let mut parakeet = parakeet_rs::ParakeetTDT::from_pretrained(&args.parakeet_model, None)?;
    eprintln!("Parakeet ready");

    eprintln!("Loading Qwen3 ASR...");
    let qwen_model_dir = shellexpand::tilde(&args.qwen_model).to_string();
    let qwen = qwen3_asr::AsrInference::load(
        Path::new(&qwen_model_dir),
        qwen3_asr::best_device(),
    )?;
    eprintln!("Qwen3 ready");

    // --- Step 4: Pipeline ---
    let output_path = Path::new(&args.output);
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut out = std::io::BufWriter::new(std::fs::File::create(output_path)?);

    // Set up audio save directory if requested
    if let Some(ref audio_dir) = args.save_audio {
        std::fs::create_dir_all(audio_dir)?;
    }

    for (i, sentence) in sentences.iter().enumerate() {
        eprint!("[{}/{}] \"{}\" ... ", i + 1, sentences.len(), &sentence.text);
        eprintln!("  (spoken: \"{}\")", &sentence.spoken);

        // TTS — use the spoken form so pronunciation is correct
        let audio = match tts.generate(&sentence.spoken, &voice_state) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("TTS error: {e}");
                continue;
            }
        };
        let samples_24k: Vec<f32> = audio.flatten_all()?.to_vec1()?;

        // Resample 24kHz → 16kHz
        let samples_16k = resample_24k_to_16k(&samples_24k)?;

        // Parakeet ASR
        let parakeet_text = match parakeet.transcribe_samples(
            samples_16k.clone(), 16000, 1, None,
        ) {
            Ok(r) => r.text,
            Err(e) => {
                eprintln!("Parakeet error: {e}");
                continue;
            }
        };

        // Qwen3 ASR
        let qwen_text = match qwen.transcribe_samples(
            &samples_16k,
            qwen3_asr::TranscribeOptions::default(),
        ) {
            Ok(r) => r.text,
            Err(e) => {
                eprintln!("Qwen3 error: {e}");
                continue;
            }
        };

        eprintln!("OK");
        eprintln!("  Parakeet: {parakeet_text}");
        eprintln!("  Qwen3:    {qwen_text}");

        let pair = TrainingPair {
            original_text: sentence.text.clone(),
            parakeet_output: parakeet_text,
            qwen_output: qwen_text,
            vocab: sentence.vocab_terms.clone(),
            voice_id: "amos".to_string(),
        };

        serde_json::to_writer(&mut out, &pair)?;
        out.write_all(b"\n")?;
        out.flush()?;
    }

    eprintln!("Wrote {} to {}", sentences.len(), args.output);
    Ok(())
}
