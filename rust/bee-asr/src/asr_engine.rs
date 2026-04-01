use std::io::Cursor;
use std::path::Path;

use anyhow::{Context, Result};
use mlx_rs::module::ModuleParametersExt;
use mlx_rs::{ops, Array};

use crate::config::AsrConfig;
use crate::forced_aligner::{ForcedAlignItem, ForcedAligner};
use crate::generate;
use crate::load;
use crate::mel::MelExtractor;
use crate::model::{Qwen3ASRModel, AUDIO_END_TOKEN_ID, AUDIO_PAD_TOKEN_ID, AUDIO_START_TOKEN_ID};
use crate::tokenizers;

const TOK_IM_START: i32 = 151644;
const TOK_IM_END: i32 = 151645;
const TOK_SYSTEM: i32 = 8948;
const TOK_USER: i32 = 872;
const TOK_ASSISTANT: i32 = 77091;
const TOK_NEWLINE: i32 = 198;

pub struct AsrEngine {
    model: Qwen3ASRModel,
    tokenizer: tokenizers::Tokenizer,
    mel_extractor: MelExtractor,
    aligner: Option<ForcedAligner>,
}

impl AsrEngine {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .with_context(|| format!("reading {}", config_path.display()))?;
        let config: AsrConfig = serde_json::from_str(&config_str)
            .with_context(|| format!("parsing {}", config_path.display()))?;

        let mut model = Qwen3ASRModel::new(&config.thinker_config).context("creating ASR model")?;
        let _stats = load::load_weights(&mut model, model_dir).context("loading ASR weights")?;
        model.eval().context("setting model eval mode")?;

        let tokenizer = find_tokenizer(model_dir).ok_or_else(|| {
            anyhow::anyhow!(
                "no tokenizer.json found in {} or common cache locations",
                model_dir.display()
            )
        })?;
        let aligner = find_aligner_dir().and_then(|aligner_dir| {
            match ForcedAligner::load(&aligner_dir, tokenizer.clone()) {
                Ok(aligner) => Some(aligner),
                Err(error) => {
                    log::warn!(
                        "failed to load forced aligner from {}: {error}",
                        aligner_dir.display()
                    );
                    None
                }
            }
        });

        Ok(Self {
            model,
            tokenizer,
            mel_extractor: MelExtractor::new(400, 160, 128, 16000),
            aligner,
        })
    }

    pub fn transcribe_wav(&mut self, wav_bytes: &[u8]) -> Result<String> {
        let samples = decode_wav_to_f32_16k_mono(wav_bytes)?;
        self.transcribe_samples(&samples)
    }

    pub fn transcribe_wav_with_alignments(
        &mut self,
        wav_bytes: &[u8],
    ) -> Result<(String, Vec<ForcedAlignItem>)> {
        let samples = decode_wav_to_f32_16k_mono(wav_bytes)?;
        self.transcribe_samples_with_alignments(&samples)
    }

    pub fn transcribe_samples(&mut self, samples: &[f32]) -> Result<String> {
        let (mel_data, n_mels, n_frames) = self
            .mel_extractor
            .extract(samples)
            .context("extracting mel features")?;
        let mel = Array::from_slice(&mel_data, &[n_mels as i32, n_frames as i32]);

        let audio_features = self
            .model
            .encode_audio(&mel)
            .context("running audio encoder")?;
        audio_features
            .eval()
            .context("evaluating audio encoder output")?;
        let n_audio_tokens = audio_features.shape()[0] as usize;
        let audio_features =
            ops::expand_dims(&audio_features, 0).context("expanding audio batch dim")?;

        let mut prompt_tokens: Vec<i32> = vec![
            TOK_IM_START,
            TOK_SYSTEM,
            TOK_NEWLINE,
            TOK_IM_END,
            TOK_NEWLINE,
            TOK_IM_START,
            TOK_USER,
            TOK_NEWLINE,
            AUDIO_START_TOKEN_ID,
        ];
        prompt_tokens.extend(std::iter::repeat_n(AUDIO_PAD_TOKEN_ID, n_audio_tokens));
        prompt_tokens.extend_from_slice(&[
            AUDIO_END_TOKEN_ID,
            TOK_IM_END,
            TOK_NEWLINE,
            TOK_IM_START,
            TOK_ASSISTANT,
            TOK_NEWLINE,
        ]);

        let seq_len = prompt_tokens.len();
        let input_ids_arr = Array::from_slice(&prompt_tokens, &[1, seq_len as i32]);
        let input_ids = &input_ids_arr;

        let positions: Vec<i32> = (0..seq_len as i32).collect();
        let pos_arr = Array::from_slice(&positions, &[1, 1, seq_len as i32]);
        let position_ids = ops::broadcast_to(&pos_arr, &[1, 3, seq_len as i32])
            .context("building position ids")?;

        let output_tokens = generate::generate(
            &mut self.model,
            input_ids,
            &audio_features,
            &position_ids,
            512,
        )
        .context("running decoder")?;

        let ids: Vec<u32> = output_tokens.iter().map(|&t| t as u32).collect();
        let text = self
            .tokenizer
            .decode(&ids, true)
            .map_err(|e| anyhow::anyhow!("tokenizer decode: {e}"))?;

        Ok(text)
    }

    pub fn transcribe_samples_with_alignments(
        &mut self,
        samples: &[f32],
    ) -> Result<(String, Vec<ForcedAlignItem>)> {
        let text = self.transcribe_samples(samples)?;
        let alignments = if let Some(aligner) = &mut self.aligner {
            aligner.align(samples, &text)?
        } else {
            Vec::new()
        };
        Ok((text, alignments))
    }
}

fn find_tokenizer(model_dir: &Path) -> Option<tokenizers::Tokenizer> {
    let mut paths = Vec::new();
    paths.push(model_dir.join("tokenizer.json"));

    let home = dirs::home_dir()?;
    paths.push(home.join("Library/Caches/qwen3-asr/Qwen--Qwen3-ASR-1.7B/tokenizer.json"));
    paths.push(home.join("Library/Caches/qwen3-asr/Qwen--Qwen3-ASR-0.6B/tokenizer.json"));

    for path in &paths {
        if path.exists() {
            if let Ok(tokenizer) = tokenizers::Tokenizer::from_file(path) {
                return Some(tokenizer);
            }
        }
    }

    None
}

fn find_aligner_dir() -> Option<std::path::PathBuf> {
    let home = dirs::home_dir()?;
    let base = home.join("Library/Caches/qwen3-asr");
    let candidates = [
        "mlx-community--Qwen3-ForcedAligner-0.6B-4bit",
        "Qwen--Qwen3-ForcedAligner-0.6B",
    ];
    for name in candidates {
        let dir = base.join(name);
        if dir.exists() {
            return Some(dir);
        }
    }
    None
}

fn decode_wav_to_f32_16k_mono(bytes: &[u8]) -> Result<Vec<f32>> {
    let cursor = Cursor::new(bytes);
    let mut reader = hound::WavReader::new(cursor).context("invalid WAV bytes")?;
    let spec = reader.spec();

    if spec.sample_rate != 16_000 {
        anyhow::bail!("expected 16kHz WAV input, got {} Hz", spec.sample_rate);
    }

    let channels = spec.channels.max(1) as usize;
    let mut mono = Vec::new();

    match spec.sample_format {
        hound::SampleFormat::Float => {
            let mut acc = 0.0f32;
            let mut idx = 0usize;
            for sample in reader.samples::<f32>() {
                acc += sample.context("reading float WAV sample")?;
                idx += 1;
                if idx == channels {
                    mono.push(acc / channels as f32);
                    acc = 0.0;
                    idx = 0;
                }
            }
        }
        hound::SampleFormat::Int => {
            if spec.bits_per_sample <= 16 {
                let scale = i16::MAX as f32;
                let mut acc = 0.0f32;
                let mut idx = 0usize;
                for sample in reader.samples::<i16>() {
                    acc += sample.context("reading i16 WAV sample")? as f32 / scale;
                    idx += 1;
                    if idx == channels {
                        mono.push(acc / channels as f32);
                        acc = 0.0;
                        idx = 0;
                    }
                }
            } else {
                let max = ((1_i64 << (spec.bits_per_sample - 1)) - 1) as f32;
                let mut acc = 0.0f32;
                let mut idx = 0usize;
                for sample in reader.samples::<i32>() {
                    acc += sample.context("reading i32 WAV sample")? as f32 / max;
                    idx += 1;
                    if idx == channels {
                        mono.push(acc / channels as f32);
                        acc = 0.0;
                        idx = 0;
                    }
                }
            }
        }
    }

    if mono.is_empty() {
        anyhow::bail!("decoded WAV is empty");
    }

    Ok(mono)
}
