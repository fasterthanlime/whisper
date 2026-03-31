//! Top-level Qwen3-ASR model: audio encoder + text decoder + LM head.

use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::macros::ModuleParameters;
use mlx_rs::module::Module;
use mlx_rs::nn;
use mlx_rs::ops;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Array;

use crate::config::ThinkerConfig;
use crate::decoder::{self, KVCache, TextDecoder};
use crate::encoder::AudioEncoder;

/// Audio pad token ID used in Qwen3-ASR prompts.
pub const AUDIO_PAD_TOKEN_ID: i32 = 151676;
pub const AUDIO_START_TOKEN_ID: i32 = 151669;
pub const AUDIO_END_TOKEN_ID: i32 = 151670;
pub const EOS_TOKEN_IDS: &[i32] = &[151643, 151645];

#[derive(Debug, Clone, ModuleParameters)]
pub struct Qwen3ASRModel {
    #[param]
    pub audio_tower: AudioEncoder,
    #[param]
    pub model: TextDecoder,
    #[param]
    pub lm_head: nn::Linear,

    audio_token_id: i32,
}

impl Qwen3ASRModel {
    pub fn new(config: &ThinkerConfig) -> Result<Self, Exception> {
        let audio_tower = AudioEncoder::new(&config.audio_config)?;
        let model = TextDecoder::new(&config.text_config)?;
        let lm_head = nn::LinearBuilder::new(
            config.text_config.hidden_size as i32,
            config.text_config.vocab_size as i32,
        )
        .bias(false)
        .build()?;

        Ok(Self {
            audio_tower,
            model,
            lm_head,
            audio_token_id: config.audio_token_id as i32,
        })
    }

    /// Encode audio mel spectrogram into features.
    /// mel: (n_mels, n_frames). Returns (n_tokens, output_dim).
    pub fn encode_audio(&mut self, mel: &Array) -> Result<Array, Exception> {
        self.audio_tower.encode(mel)
    }

    /// Prefill: process prompt with injected audio features, populate KV cache.
    /// Returns logits for the last position.
    pub fn prefill(
        &mut self,
        input_ids: &Array,
        audio_features: &Array,
        position_ids: &Array,
        cache: &mut Option<KVCache>,
    ) -> Result<Array, Exception> {
        // Get embeddings
        let embeds = self.model.embed_tokens.forward(input_ids)?;

        // Inject audio features at placeholder positions
        let audio_mask = input_ids.eq(&Array::from_int(self.audio_token_id))?;
        let embeds = inject_audio_features(&embeds, audio_features, &audio_mask)?;

        // Run decoder
        let hidden = self.model.forward_decoder(
            None,
            Some(&embeds),
            position_ids,
            cache,
        )?;

        // Logits for last position
        let last = hidden.index((.., -1.., ..));
        self.lm_head.forward(&last)
    }

    /// Single autoregressive step with KV cache.
    pub fn step(
        &mut self,
        input_ids: &Array,
        position_ids: &Array,
        cache: &mut Option<KVCache>,
    ) -> Result<Array, Exception> {
        let embeds = self.model.embed_tokens.forward(input_ids)?;
        let hidden = self.model.forward_decoder(
            None,
            Some(&embeds),
            position_ids,
            cache,
        )?;
        self.lm_head.forward(&hidden)
    }

    pub fn create_cache(&self) -> KVCache {
        KVCache::new(self.model.num_layers())
    }
}

/// Replace audio placeholder positions with encoded audio features.
fn inject_audio_features(
    embeds: &Array,
    audio_features: &Array,
    audio_mask: &Array,
) -> Result<Array, Exception> {
    let b = embeds.shape()[0];

    // cumulative index: map each placeholder to its audio feature index
    let cum_idx = audio_mask.as_dtype(mlx_rs::Dtype::Int32)?.cumsum(1, None, None)?
        .subtract(&Array::from_int(1))?;
    let cum_idx = ops::maximum(&cum_idx, &Array::from_int(0))?;

    // Gather audio features per batch element
    let mut parts: Vec<Array> = Vec::with_capacity(b as usize);
    for bi in 0..b {
        let idx = cum_idx.index((bi, ..)); // (L,)
        let audio_b = audio_features.index((bi, ..)); // (N_audio, D)
        // audio_b[idx] → (L, D)
        let expanded = audio_b.index(idx);
        parts.push(expanded);
    }
    // Stack by adding batch dim to each and concatenating
    let expanded: Vec<Array> = parts.iter()
        .map(|p| ops::expand_dims(p, 0).unwrap())
        .collect();
    let refs: Vec<&Array> = expanded.iter().collect();
    let audio_expanded = ops::concatenate_axis(&refs, 0)?; // (B, L, D)

    // Select: audio where mask, text embeds elsewhere
    let mask_3d = ops::expand_dims(audio_mask, -1)?; // (B, L, 1)
    mlx_rs::ops::r#where(&mask_3d, &audio_expanded, embeds)
}
