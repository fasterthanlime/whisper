//! In-process LLM inference using Candle and quantized GGUF Qwen models.
//!
//! Replaces the flaky mlx_lm.server Python subprocess with direct GGUF model loading.

use crate::adapter_import;
use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use peft_rs::LoraLayer;
use std::collections::BTreeMap;
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
}

impl<M: AdapterModel> LocalModel<M> {
    fn sample_next_token(logits_processor: &mut LogitsProcessor, logits: Tensor) -> Result<u32> {
        let logits = logits.squeeze(0)?;
        logits_processor
            .sample(&logits)
            .map_err(|e| anyhow::anyhow!("sample failed: {e}"))
    }

    fn from_loaded_parts(model: M, tokenizer: tokenizers::Tokenizer, device: Device) -> Self {
        Self {
            model,
            tokenizer,
            device,
        }
    }

    fn should_stop_early(&self, generated_tokens: &[u32]) -> bool {
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

    pub fn generate(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        temperature: f64,
        seed: u64,
    ) -> Result<String> {
        const IM_END_TOKEN: u32 = 151645;
        const ENDOFTEXT_TOKEN: u32 = 151643;

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
        if next_token == IM_END_TOKEN || next_token == ENDOFTEXT_TOKEN {
            eprintln!(
                "[llm] generate done: prompt_tokens={} output_tokens=0 prefill_ms={} decode_ms=0 total_ms={}",
                prompt_tokens.len(),
                prefill_elapsed.as_millis(),
                started.elapsed().as_millis(),
            );
            return Ok(String::new());
        }
        generated_tokens.push(next_token);
        if self.should_stop_early(&generated_tokens) {
            eprintln!(
                "[llm] early stop after {} token(s) of output",
                generated_tokens.len()
            );
            return self
                .tokenizer
                .decode(&generated_tokens, true)
                .map_err(|e| anyhow::anyhow!("tokenizer decode failed: {e}"));
        }

        let decode_started = Instant::now();
        for i in 0..max_tokens.saturating_sub(1) {
            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let logits = self
                .model
                .forward(&input, prompt_tokens.len() + i + 1)
                .map_err(|e| anyhow::anyhow!("forward (decode step {i}) failed: {e}"))?;
            next_token = Self::sample_next_token(&mut logits_processor, logits)?;

            if next_token == IM_END_TOKEN || next_token == ENDOFTEXT_TOKEN {
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

        self.tokenizer
            .decode(&generated_tokens, true)
            .map_err(|e| anyhow::anyhow!("tokenizer decode failed: {e}"))
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

    pub fn generate(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        temperature: f64,
        seed: u64,
    ) -> Result<String> {
        match self {
            Self::Qwen2(model) => model.generate(prompt_tokens, max_tokens, temperature, seed),
            Self::Qwen3_5(model) => model.generate(prompt_tokens, max_tokens, temperature, seed),
        }
    }
}
