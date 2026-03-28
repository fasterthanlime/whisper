//! In-process LLM inference using Candle and quantized GGUF Qwen models.
//!
//! Replaces the flaky mlx_lm.server Python subprocess with direct GGUF model loading.

use crate::adapter_import;
use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use peft_rs::LoraLayer;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::time::Instant;

pub(crate) trait AdapterModel {
    fn forward(&mut self, input: &Tensor, offset: usize) -> candle_core::Result<Tensor>;
    fn clear_kv_cache(&mut self);
    fn set_lora_layers(&mut self, layers: BTreeMap<String, LoraLayer>);
}

impl AdapterModel for crate::qwen2::ModelWeights {
    fn forward(&mut self, input: &Tensor, offset: usize) -> candle_core::Result<Tensor> {
        self.forward(input, offset)
    }

    fn clear_kv_cache(&mut self) {
        self.clear_kv_cache();
    }

    fn set_lora_layers(&mut self, layers: BTreeMap<String, LoraLayer>) {
        self.set_lora_layers(layers);
    }
}

impl AdapterModel for crate::qwen3_5::ModelWeights {
    fn forward(&mut self, input: &Tensor, offset: usize) -> candle_core::Result<Tensor> {
        self.forward(input, offset)
    }

    fn clear_kv_cache(&mut self) {
        self.clear_kv_cache();
    }

    fn set_lora_layers(&mut self, layers: BTreeMap<String, LoraLayer>) {
        self.set_lora_layers(layers);
    }
}

fn ensure_model_files(
    gguf_repo: &str,
    gguf_file: &str,
    tokenizer_repo: &str,
) -> Result<(std::path::PathBuf, std::path::PathBuf)> {
    use hf_hub::api::sync::Api;

    let api = Api::new().context("failed to create HF Hub API")?;

    eprintln!("[llm] Ensuring GGUF model: {gguf_repo}/{gguf_file}");
    let gguf_repo = api.model(gguf_repo.to_string());
    let model_path = gguf_repo
        .get(gguf_file)
        .context("failed to download GGUF model")?;

    eprintln!("[llm] Ensuring tokenizer: {tokenizer_repo}/tokenizer.json");
    let tok_repo = api.model(tokenizer_repo.to_string());
    let tokenizer_path = tok_repo
        .get("tokenizer.json")
        .context("failed to download tokenizer.json")?;

    Ok((model_path, tokenizer_path))
}

fn best_device() -> Device {
    #[cfg(feature = "metal")]
    {
        if let Ok(device) = Device::new_metal(0) {
            return device;
        }
    }
    Device::Cpu
}

pub(crate) struct LocalModel<M> {
    model: M,
    tokenizer: tokenizers::Tokenizer,
    device: Device,
    stop_tokens: BTreeSet<u32>,
}

#[derive(Debug, Clone)]
pub(crate) struct GenerateStats {
    pub text: String,
    pub prompt_tokens: usize,
    pub output_tokens: usize,
    pub prefill_ms: u64,
    pub decode_ms: u64,
}

impl<M: AdapterModel> LocalModel<M> {
    fn discover_stop_tokens(tokenizer: &tokenizers::Tokenizer) -> BTreeSet<u32> {
        const STOP_TEXTS: &[&str] = &[
            "<|eot_id|>",
            "<|im_end|>",
            "<|end|>",
            "<end_of_turn>",
            "<|endoftext|>",
            "<|end_of_text|>",
            "<EOT>",
            "_<EOT>",
            "[EOT]",
            "<｜end▁of▁sentence｜>",
            "<end_of_utterance>",
        ];

        let mut stop_tokens = BTreeSet::new();
        for text in STOP_TEXTS {
            if let Some(id) = tokenizer.token_to_id(text) {
                stop_tokens.insert(id);
            }
        }
        stop_tokens
    }

    fn sample_next_token(logits_processor: &mut LogitsProcessor, logits: Tensor) -> Result<u32> {
        let logits = logits.squeeze(0)?;
        logits_processor
            .sample(&logits)
            .map_err(|e| anyhow::anyhow!("sample failed: {e}"))
    }

    fn from_loaded_parts(model: M, tokenizer: tokenizers::Tokenizer, device: Device) -> Self {
        let stop_tokens = Self::discover_stop_tokens(&tokenizer);
        eprintln!("[llm] stop tokens: {stop_tokens:?}");
        Self {
            model,
            tokenizer,
            device,
            stop_tokens,
        }
    }

    fn is_stop_token(&self, token: u32) -> bool {
        self.stop_tokens.contains(&token)
    }

    fn normalize_piece(piece: &str) -> String {
        piece.trim_matches(|ch: char| {
            ch.is_ascii_whitespace()
                || matches!(
                    ch,
                    '\u{2581}' | '<' | '>' | '|' | '_' | '\n' | '\r' | '\t'
                )
        })
        .to_ascii_lowercase()
    }

    fn has_repeated_suffix<T: Eq>(items: &[T], unit: usize, reps: usize) -> bool {
        if unit == 0 || reps < 2 || items.len() < unit * reps {
            return false;
        }
        let start = items.len() - unit * reps;
        let base = &items[start..start + unit];
        (1..reps).all(|rep| {
            let chunk_start = start + rep * unit;
            &items[chunk_start..chunk_start + unit] == base
        })
    }

    fn should_abort_repetition(&self, generated_tokens: &[u32]) -> bool {
        if Self::has_repeated_suffix(generated_tokens, 1, 6)
            || Self::has_repeated_suffix(generated_tokens, 2, 4)
            || Self::has_repeated_suffix(generated_tokens, 3, 3)
        {
            return true;
        }

        let pieces: Vec<_> = generated_tokens
            .iter()
            .filter_map(|&token| self.tokenizer.id_to_token(token))
            .map(|piece| Self::normalize_piece(&piece))
            .filter(|piece| !piece.is_empty())
            .collect();

        if Self::has_repeated_suffix(&pieces, 1, 4)
            || Self::has_repeated_suffix(&pieces, 2, 3)
            || Self::has_repeated_suffix(&pieces, 3, 3)
        {
            return true;
        }

        let Ok(text) = self.tokenizer.decode(generated_tokens, true) else {
            return false;
        };
        let lower = text.trim().to_ascii_lowercase();
        lower.contains("<task>")
            || lower.contains("<rules>")
            || lower.contains("<qwen>")
            || lower.contains("<fixd>")
    }

    fn should_stop_early(&self, generated_tokens: &[u32]) -> bool {
        if self.should_abort_repetition(generated_tokens) {
            return true;
        }
        if generated_tokens.len() < 6 {
            return false;
        }

        let Ok(text) = self.tokenizer.decode(generated_tokens, true) else {
            return false;
        };
        let text = text.trim_end();
        if text.is_empty() {
            return false;
        }

        text.ends_with('.') || text.ends_with('!') || text.ends_with('?')
    }

    pub fn attach_mlx_adapters(&mut self, adapter_dir: impl AsRef<Path>) -> Result<()> {
        let imported = adapter_import::load_mlx_lora_dir(adapter_dir.as_ref(), &self.device)?;
        let layer_count = imported.layers.len();
        self.model.set_lora_layers(imported.layers);
        eprintln!(
            "[llm] Attached {} LoRA modules from {}",
            layer_count,
            imported.weights_path.display()
        );
        Ok(())
    }

    pub fn encode_chat(&self, system: &str, user: &str) -> Result<Vec<u32>> {
        let prompt = format!(
            "<|im_start|>system\n{system}<|im_end|>\n\
             <|im_start|>user\n{user}<|im_end|>\n\
             <|im_start|>assistant\n"
        );
        let encoding = self
            .tokenizer
            .encode(prompt.as_str(), true)
            .map_err(|e| anyhow::anyhow!("tokenizer encode failed: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn encode_text(&self, prompt: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("tokenizer encode failed: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn generate_with_stats(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        temperature: f64,
        seed: u64,
    ) -> Result<GenerateStats> {
        let started = Instant::now();
        self.model.clear_kv_cache();

        let sampling = if temperature < 1e-7 {
            Sampling::ArgMax
        } else {
            Sampling::TopP {
                p: 0.9,
                temperature,
            }
        };
        let mut logits_processor = LogitsProcessor::from_sampling(seed, sampling);

        let prefill_started = Instant::now();
        let input = Tensor::new(prompt_tokens, &self.device)?.unsqueeze(0)?;
        let logits = self
            .model
            .forward(&input, 0)
            .map_err(|e| anyhow::anyhow!("forward (prefill) failed: {e}"))?;
        let prefill_elapsed = prefill_started.elapsed();
        eprintln!(
            "[llm] prefill done: prompt_tokens={} prefill_ms={}",
            prompt_tokens.len(),
            prefill_elapsed.as_millis(),
        );
        let mut next_token = Self::sample_next_token(&mut logits_processor, logits)?;

        let mut generated_tokens = Vec::new();
        if self.is_stop_token(next_token) {
            eprintln!(
                "[llm] generate done: prompt_tokens={} output_tokens=0 prefill_ms={} decode_ms=0 total_ms={}",
                prompt_tokens.len(),
                prefill_elapsed.as_millis(),
                started.elapsed().as_millis(),
            );
            return Ok(GenerateStats {
                text: String::new(),
                prompt_tokens: prompt_tokens.len(),
                output_tokens: 0,
                prefill_ms: prefill_elapsed.as_millis() as u64,
                decode_ms: 0,
            });
        }
        generated_tokens.push(next_token);
        if self.should_stop_early(&generated_tokens) {
            eprintln!(
                "[llm] early stop after {} token(s) of output",
                generated_tokens.len()
            );
            let text = self
                .tokenizer
                .decode(&generated_tokens, true)
                .map_err(|e| anyhow::anyhow!("tokenizer decode failed: {e}"))?;
            return Ok(GenerateStats {
                text,
                prompt_tokens: prompt_tokens.len(),
                output_tokens: generated_tokens.len(),
                prefill_ms: prefill_elapsed.as_millis() as u64,
                decode_ms: 0,
            });
        }

        let decode_started = Instant::now();
        for i in 0..max_tokens.saturating_sub(1) {
            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let logits = self
                .model
                .forward(&input, prompt_tokens.len() + i + 1)
                .map_err(|e| anyhow::anyhow!("forward (decode step {i}) failed: {e}"))?;
            next_token = Self::sample_next_token(&mut logits_processor, logits)?;

            if self.is_stop_token(next_token) {
                break;
            }
            generated_tokens.push(next_token);
            if self.should_stop_early(&generated_tokens) {
                eprintln!(
                    "[llm] early stop after {} token(s) of output",
                    generated_tokens.len()
                );
                break;
            }
        }
        let decode_elapsed = decode_started.elapsed();

        eprintln!(
            "[llm] generate done: prompt_tokens={} output_tokens={} prefill_ms={} decode_ms={} total_ms={}",
            prompt_tokens.len(),
            generated_tokens.len(),
            prefill_elapsed.as_millis(),
            decode_elapsed.as_millis(),
            started.elapsed().as_millis(),
        );

        let text = self
            .tokenizer
            .decode(&generated_tokens, true)
            .map_err(|e| anyhow::anyhow!("tokenizer decode failed: {e}"))?;
        Ok(GenerateStats {
            text,
            prompt_tokens: prompt_tokens.len(),
            output_tokens: generated_tokens.len(),
            prefill_ms: prefill_elapsed.as_millis() as u64,
            decode_ms: decode_elapsed.as_millis() as u64,
        })
    }

    pub fn generate(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        temperature: f64,
        seed: u64,
    ) -> Result<String> {
        Ok(self
            .generate_with_stats(prompt_tokens, max_tokens, temperature, seed)?
            .text)
    }
}

pub(crate) type Qwen2Model = LocalModel<crate::qwen2::ModelWeights>;
pub(crate) type Qwen3_5Model = LocalModel<crate::qwen3_5::ModelWeights>;

impl LocalModel<crate::qwen2::ModelWeights> {
    pub fn load(gguf_repo: &str, gguf_file: &str, tokenizer_repo: &str) -> Result<Self> {
        let device = best_device();
        eprintln!("[llm] Using device: {device:?}");

        let (model_path, tokenizer_path) =
            ensure_model_files(gguf_repo, gguf_file, tokenizer_repo)?;

        eprintln!("[llm] Loading GGUF weights from {}", model_path.display());
        let mut file = std::fs::File::open(&model_path)?;
        let content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .context("failed to read GGUF file")?;
        let model = crate::qwen2::ModelWeights::from_gguf(content, &mut file, &device)
            .map_err(|e| anyhow::anyhow!("failed to load Qwen2 weights: {e}"))?;

        eprintln!("[llm] Loading tokenizer from {}", tokenizer_path.display());
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

        eprintln!("[llm] Model loaded successfully");
        Ok(Self::from_loaded_parts(model, tokenizer, device))
    }

    pub fn load_with_mlx_adapters(
        gguf_repo: &str,
        gguf_file: &str,
        tokenizer_repo: &str,
        adapter_dir: impl AsRef<Path>,
    ) -> Result<Self> {
        let mut model = Self::load(gguf_repo, gguf_file, tokenizer_repo)?;
        model.attach_mlx_adapters(adapter_dir)?;
        Ok(model)
    }
}

impl LocalModel<crate::qwen3_5::ModelWeights> {
    pub fn load(gguf_repo: &str, gguf_file: &str, tokenizer_repo: &str) -> Result<Self> {
        let device = best_device();
        eprintln!("[llm] Using device: {device:?}");

        let (model_path, tokenizer_path) =
            ensure_model_files(gguf_repo, gguf_file, tokenizer_repo)?;

        eprintln!("[llm] Loading GGUF weights from {}", model_path.display());
        let mut file = std::fs::File::open(&model_path)?;
        let content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .context("failed to read GGUF file")?;
        let model = crate::qwen3_5::ModelWeights::from_gguf(content, &mut file, &device)
            .map_err(|e| anyhow::anyhow!("failed to load Qwen3.5 weights: {e}"))?;

        eprintln!("[llm] Loading tokenizer from {}", tokenizer_path.display());
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

        eprintln!("[llm] Model loaded successfully");
        Ok(Self::from_loaded_parts(model, tokenizer, device))
    }

    pub fn load_with_mlx_adapters(
        gguf_repo: &str,
        gguf_file: &str,
        tokenizer_repo: &str,
        adapter_dir: impl AsRef<Path>,
    ) -> Result<Self> {
        let mut model = Self::load(gguf_repo, gguf_file, tokenizer_repo)?;
        model.attach_mlx_adapters(adapter_dir)?;
        Ok(model)
    }
}

pub(crate) enum CorrectionModel {
    Qwen2(Qwen2Model),
    Qwen3_5(Qwen3_5Model),
}

impl CorrectionModel {
    pub fn encode_text(&self, prompt: &str) -> Result<Vec<u32>> {
        match self {
            Self::Qwen2(model) => model.encode_text(prompt),
            Self::Qwen3_5(model) => model.encode_text(prompt),
        }
    }

    pub fn generate_with_stats(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        temperature: f64,
        seed: u64,
    ) -> Result<GenerateStats> {
        match self {
            Self::Qwen2(model) => {
                model.generate_with_stats(prompt_tokens, max_tokens, temperature, seed)
            }
            Self::Qwen3_5(model) => {
                model.generate_with_stats(prompt_tokens, max_tokens, temperature, seed)
            }
        }
    }
}
