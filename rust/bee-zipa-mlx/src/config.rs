#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZipaModelConfig {
    pub variant: ZipaVariant,
    pub feature_dim: usize,
    pub subsampling_factor: usize,
    pub vocab_size: usize,
    pub num_encoder_layers: Vec<usize>,
    pub downsampling_factor: Vec<usize>,
    pub feedforward_dim: Vec<usize>,
    pub num_heads: Vec<usize>,
    pub encoder_dim: Vec<usize>,
    pub encoder_unmasked_dim: Vec<usize>,
    pub cnn_module_kernel: Vec<usize>,
    pub query_head_dim: usize,
    pub value_head_dim: usize,
    pub pos_head_dim: usize,
    pub pos_dim: usize,
    pub causal: bool,
    pub use_ctc: bool,
    pub use_cr_ctc: bool,
    pub use_transducer: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZipaVariant {
    SmallCrCtcNsNoDiacritics700k,
}

impl ZipaModelConfig {
    pub fn for_variant(variant: ZipaVariant) -> Self {
        match variant {
            ZipaVariant::SmallCrCtcNsNoDiacritics700k => Self {
                variant,
                feature_dim: 80,
                subsampling_factor: 4,
                vocab_size: 127,
                num_encoder_layers: vec![2, 2, 3, 4, 3, 2],
                downsampling_factor: vec![1, 2, 4, 8, 4, 2],
                feedforward_dim: vec![512, 768, 1024, 1536, 1024, 768],
                num_heads: vec![4, 4, 4, 8, 4, 4],
                encoder_dim: vec![192, 256, 384, 512, 384, 256],
                encoder_unmasked_dim: vec![192, 192, 256, 256, 256, 192],
                cnn_module_kernel: vec![31, 31, 15, 15, 15, 31],
                query_head_dim: 32,
                value_head_dim: 12,
                pos_head_dim: 4,
                pos_dim: 48,
                causal: false,
                use_ctc: true,
                use_cr_ctc: true,
                use_transducer: false,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ZipaModelConfig, ZipaVariant};

    #[test]
    fn small_variant_matches_reference_shape() {
        let config = ZipaModelConfig::for_variant(ZipaVariant::SmallCrCtcNsNoDiacritics700k);
        assert_eq!(config.feature_dim, 80);
        assert_eq!(config.vocab_size, 127);
        assert_eq!(config.num_encoder_layers, vec![2, 2, 3, 4, 3, 2]);
        assert_eq!(config.encoder_dim, vec![192, 256, 384, 512, 384, 256]);
        assert!(config.use_ctc);
        assert!(config.use_cr_ctc);
        assert!(!config.use_transducer);
    }
}
