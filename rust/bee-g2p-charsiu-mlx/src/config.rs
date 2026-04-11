/// Configuration for the Charsiu G2P ByT5 model.
///
/// Matches the HuggingFace config.json for `charsiu/g2p_multilingual_byT5_tiny_16_layers`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct T5Config {
    pub d_model: i32,
    pub d_ff: i32,
    pub d_kv: i32,
    pub num_heads: i32,
    pub num_layers: i32,
    pub num_decoder_layers: i32,
    pub vocab_size: i32,
    pub relative_attention_num_buckets: i32,
    pub decoder_start_token_id: i32,
    pub eos_token_id: i32,
    pub pad_token_id: i32,
    #[serde(default = "default_layer_norm_epsilon")]
    pub layer_norm_epsilon: f32,
}

fn default_layer_norm_epsilon() -> f32 {
    1e-6
}

impl T5Config {
    /// The config for charsiu/g2p_multilingual_byT5_tiny_16_layers
    pub fn charsiu_g2p() -> Self {
        Self {
            d_model: 256,
            d_ff: 1024,
            d_kv: 64,
            num_heads: 6,
            num_layers: 12,
            num_decoder_layers: 4,
            vocab_size: 384,
            relative_attention_num_buckets: 32,
            decoder_start_token_id: 0,
            eos_token_id: 1,
            pad_token_id: 0,
            layer_norm_epsilon: 1e-6,
        }
    }
}
