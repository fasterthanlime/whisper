//! In-process LLM inference using candle + quantized Qwen2.
//!
//! Replaces the flaky mlx_lm.server Python subprocess with direct GGUF model loading.

use crate::qwen2::ModelWeights;
use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;

/// Download model files from HuggingFace Hub if not cached.
/// Returns `(model_path, tokenizer_path)`.
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

/// A loaded GGUF Qwen2 model ready for chat-style text generation.
pub struct Qwen2Model {
    model: ModelWeights,
    tokenizer: tokenizers::Tokenizer,
    device: Device,
}

impl Qwen2Model {
    /// Load a quantized Qwen2 model from HuggingFace Hub.
    pub fn load(gguf_repo: &str, gguf_file: &str, tokenizer_repo: &str) -> Result<Self> {
        let device = best_device();
        eprintln!("[llm] Using device: {device:?}");

        let (model_path, tokenizer_path) =
            ensure_model_files(gguf_repo, gguf_file, tokenizer_repo)?;

        eprintln!("[llm] Loading GGUF weights from {}", model_path.display());
        let mut file = std::fs::File::open(&model_path)?;
        let content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .context("failed to read GGUF file")?;
        let model = ModelWeights::from_gguf(content, &mut file, &device)
            .map_err(|e| anyhow::anyhow!("failed to load Qwen2 weights: {e}"))?;

        eprintln!("[llm] Loading tokenizer from {}", tokenizer_path.display());
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

        eprintln!("[llm] Model loaded successfully");
        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    /// Build a Qwen2.5 ChatML prompt from system + user messages, encode to token IDs.
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

    /// Run autoregressive generation. Returns generated text (not including the prompt).
    pub fn generate(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        temperature: f64,
        seed: u64,
    ) -> Result<String> {
        const IM_END_TOKEN: u32 = 151645;
        const ENDOFTEXT_TOKEN: u32 = 151643;

        self.model.clear_kv_cache();

        let sampling = if temperature < 1e-7 {
            candle_transformers::generation::Sampling::ArgMax
        } else {
            candle_transformers::generation::Sampling::TopP {
                p: 0.9,
                temperature,
            }
        };
        let mut logits_processor = LogitsProcessor::from_sampling(seed, sampling);

        // Prefill: feed the entire prompt
        let input = Tensor::new(prompt_tokens, &self.device)?.unsqueeze(0)?;
        let logits = self
            .model
            .forward(&input, 0)
            .map_err(|e| anyhow::anyhow!("forward (prefill) failed: {e}"))?;
        let mut next_token = logits_processor
            .sample(&logits)
            .map_err(|e| anyhow::anyhow!("sample failed: {e}"))?;

        let mut generated_tokens = Vec::new();

        if next_token == IM_END_TOKEN || next_token == ENDOFTEXT_TOKEN {
            return Ok(String::new());
        }
        generated_tokens.push(next_token);

        // Decode: one token at a time
        for i in 0..max_tokens.saturating_sub(1) {
            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let logits = self
                .model
                .forward(&input, prompt_tokens.len() + i + 1)
                .map_err(|e| anyhow::anyhow!("forward (decode step {i}) failed: {e}"))?;
            next_token = logits_processor
                .sample(&logits)
                .map_err(|e| anyhow::anyhow!("sample failed: {e}"))?;

            if next_token == IM_END_TOKEN || next_token == ENDOFTEXT_TOKEN {
                break;
            }
            generated_tokens.push(next_token);
        }

        let text = self
            .tokenizer
            .decode(&generated_tokens, true)
            .map_err(|e| anyhow::anyhow!("tokenizer decode failed: {e}"))?;

        Ok(text)
    }
}
