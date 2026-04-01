use anyhow::{Context, Result};
use parakeet_rs::Transcriber;
use std::io::Write;
use std::path::Path;

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct TrainingPair {
    pub original_text: String,
    pub spoken_text: String,
    pub parakeet_output: String,
    pub qwen_output: String,
    pub vocab: Vec<String>,
    pub voice_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_file: Option<String>,
}

pub fn resample_24k_to_16k(samples: &[f32]) -> Result<Vec<f32>> {
    use rubato::{
        Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
    };

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut resampler = SincFixedIn::<f32>::new(16000.0 / 24000.0, 2.0, params, samples.len(), 1)?;
    let output = resampler.process(&[samples], None)?;
    Ok(output.into_iter().next().unwrap_or_default())
}

/// Progress callback for pipeline execution
pub enum PipelineEvent {
    Status(String),
    SentenceStart {
        index: usize,
        total: usize,
        text: String,
    },
    SentenceDone {
        index: usize,
        parakeet: String,
        qwen: String,
    },
    SentenceError {
        index: usize,
        error: String,
    },
    Done {
        count: usize,
    },
}

pub struct PipelineConfig {
    pub voice: String,
    pub parakeet_model: String,
    pub qwen_model: String,
    pub save_audio: Option<String>,
    pub voice_id: String,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            voice: "voices/amos.wav".into(),
            parakeet_model: "models/parakeet-tdt".into(),
            qwen_model: shellexpand::tilde(
                "~/Library/Caches/qwen3-asr/Alkd--qwen3-asr-gguf--qwen3_asr_1_7b_q8_0_gguf",
            )
            .to_string(),
            save_audio: None,
            voice_id: "amos".into(),
        }
    }
}

/// Run the full TTS→ASR pipeline on a set of sentences, calling `on_event` for progress.
/// Returns the generated training pairs.
pub fn run_pipeline(
    config: &PipelineConfig,
    sentences: &[beeml_textgen::templates::GeneratedSentence],
    mut on_event: impl FnMut(PipelineEvent),
) -> Result<Vec<TrainingPair>> {
    on_event(PipelineEvent::Status(
        "Loading pocket-tts (quantized)...".into(),
    ));
    let tts = pocket_tts::TTSModel::load_quantized("b6369a24")?;
    let voice_state = tts
        .get_voice_state(&config.voice)
        .context("loading voice reference WAV")?;
    let tts_sample_rate = tts.sample_rate as u32;
    on_event(PipelineEvent::Status(format!(
        "TTS ready ({tts_sample_rate} Hz)"
    )));

    on_event(PipelineEvent::Status("Loading Parakeet TDT...".into()));
    let mut parakeet = parakeet_rs::ParakeetTDT::from_pretrained(&config.parakeet_model, None)?;
    on_event(PipelineEvent::Status("Parakeet ready".into()));

    on_event(PipelineEvent::Status("Loading Qwen3 ASR...".into()));
    let qwen =
        qwen3_asr::AsrInference::load(Path::new(&config.qwen_model), qwen3_asr::best_device())?;
    on_event(PipelineEvent::Status("Qwen3 ready".into()));

    if let Some(ref audio_dir) = config.save_audio {
        std::fs::create_dir_all(audio_dir)?;
    }

    let mut pairs = Vec::new();

    for (i, sentence) in sentences.iter().enumerate() {
        on_event(PipelineEvent::SentenceStart {
            index: i,
            total: sentences.len(),
            text: sentence.text.clone(),
        });

        // TTS — use the spoken form so pronunciation is correct
        let audio = match tts.generate(&sentence.spoken, &voice_state) {
            Ok(a) => a,
            Err(e) => {
                on_event(PipelineEvent::SentenceError {
                    index: i,
                    error: format!("TTS: {e}"),
                });
                continue;
            }
        };
        let samples_24k: Vec<f32> = audio.flatten_all()?.to_vec1()?;
        let samples_16k = resample_24k_to_16k(&samples_24k)?;

        // Parakeet ASR
        let parakeet_text = match parakeet.transcribe_samples(samples_16k.clone(), 16000, 1, None) {
            Ok(r) => r.text,
            Err(e) => {
                on_event(PipelineEvent::SentenceError {
                    index: i,
                    error: format!("Parakeet: {e}"),
                });
                continue;
            }
        };

        // Qwen3 ASR
        let qwen_text =
            match qwen.transcribe_samples(&samples_16k, qwen3_asr::TranscribeOptions::default()) {
                Ok(r) => r.text,
                Err(e) => {
                    on_event(PipelineEvent::SentenceError {
                        index: i,
                        error: format!("Qwen3: {e}"),
                    });
                    continue;
                }
            };

        on_event(PipelineEvent::SentenceDone {
            index: i,
            parakeet: parakeet_text.clone(),
            qwen: qwen_text.clone(),
        });

        pairs.push(TrainingPair {
            original_text: sentence.text.clone(),
            spoken_text: sentence.spoken.clone(),
            parakeet_output: parakeet_text,
            qwen_output: qwen_text,
            vocab: sentence.vocab_terms.clone(),
            voice_id: config.voice_id.clone(),
            audio_file: None,
        });
    }

    on_event(PipelineEvent::Done { count: pairs.len() });
    Ok(pairs)
}

/// Load sentences from a JSONL file
pub fn load_sentences(path: &str) -> Result<Vec<beeml_textgen::templates::GeneratedSentence>> {
    let content = std::fs::read_to_string(path)?;
    let mut sentences = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line)?;
        sentences.push(beeml_textgen::templates::GeneratedSentence {
            text: v["text"].as_str().unwrap_or("").to_string(),
            spoken: v["spoken"].as_str().unwrap_or("").to_string(),
            vocab_terms: v["vocab_terms"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default(),
        });
    }
    Ok(sentences)
}

/// Write training pairs to a JSONL file
pub fn write_pairs(path: &str, pairs: &[TrainingPair]) -> Result<()> {
    if let Some(parent) = Path::new(path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
    for pair in pairs {
        serde_json::to_writer(&mut out, pair)?;
        out.write_all(b"\n")?;
    }
    out.flush()?;
    Ok(())
}
