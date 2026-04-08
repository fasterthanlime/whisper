use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::macros::ModuleParameters;
use mlx_rs::module::Param;
use mlx_rs::nn;
use mlx_rs::ops::zeros;
use mlx_rs::Array;

use crate::config::ZipaModelConfig;
use crate::model::BiasNorm;

#[derive(Debug, Clone, ModuleParameters)]
pub struct BypassModule {
    #[param]
    pub bypass_scale: Param<Array>,
}

impl BypassModule {
    pub fn new(embed_dim: i32) -> Result<Self, Exception> {
        Ok(Self {
            bypass_scale: Param::new(zeros::<f32>(&[embed_dim])?),
        })
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct RelPositionAttentionWeights {
    #[param]
    pub in_proj: nn::Linear,
    #[param]
    pub linear_pos: nn::Linear,
}

impl RelPositionAttentionWeights {
    pub fn new(
        embed_dim: i32,
        pos_dim: i32,
        num_heads: i32,
        query_head_dim: i32,
        pos_head_dim: i32,
    ) -> Result<Self, Exception> {
        let in_proj_dim = (query_head_dim * 2 + pos_head_dim) * num_heads;
        Ok(Self {
            in_proj: nn::LinearBuilder::new(embed_dim, in_proj_dim).build()?,
            linear_pos: nn::LinearBuilder::new(pos_dim, num_heads * pos_head_dim)
                .bias(false)
                .build()?,
        })
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct SelfAttention {
    #[param]
    pub in_proj: nn::Linear,
    #[param]
    pub out_proj: nn::Linear,
}

impl SelfAttention {
    pub fn new(embed_dim: i32, num_heads: i32, value_head_dim: i32) -> Result<Self, Exception> {
        let value_dim = num_heads * value_head_dim;
        Ok(Self {
            in_proj: nn::LinearBuilder::new(embed_dim, value_dim).build()?,
            out_proj: nn::LinearBuilder::new(value_dim, embed_dim).build()?,
        })
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct FeedForwardModule {
    #[param]
    pub in_proj: nn::Linear,
    #[param]
    pub out_proj: nn::Linear,
}

impl FeedForwardModule {
    pub fn new(embed_dim: i32, feedforward_dim: i32) -> Result<Self, Exception> {
        Ok(Self {
            in_proj: nn::LinearBuilder::new(embed_dim, feedforward_dim).build()?,
            out_proj: nn::LinearBuilder::new(feedforward_dim, embed_dim).build()?,
        })
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct NonlinAttention {
    #[param]
    pub in_proj: nn::Linear,
    #[param]
    pub out_proj: nn::Linear,
}

impl NonlinAttention {
    pub fn new(embed_dim: i32, hidden_channels: i32) -> Result<Self, Exception> {
        Ok(Self {
            in_proj: nn::LinearBuilder::new(embed_dim, hidden_channels * 3).build()?,
            out_proj: nn::LinearBuilder::new(hidden_channels, embed_dim).build()?,
        })
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct ConvolutionModule {
    #[param]
    pub in_proj: nn::Linear,
    #[param]
    pub depthwise_conv: nn::Conv1d,
    #[param]
    pub out_proj: nn::Linear,
}

impl ConvolutionModule {
    pub fn new(embed_dim: i32, kernel_size: i32) -> Result<Self, Exception> {
        Ok(Self {
            in_proj: nn::LinearBuilder::new(embed_dim, embed_dim * 2).build()?,
            depthwise_conv: nn::Conv1dBuilder::new(1, embed_dim, kernel_size)
                .groups(embed_dim)
                .padding(kernel_size / 2)
                .build()?,
            out_proj: nn::LinearBuilder::new(embed_dim, embed_dim).build()?,
        })
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct ZipformerEncoderLayer {
    #[param]
    pub self_attn_weights: RelPositionAttentionWeights,
    #[param]
    pub self_attn1: SelfAttention,
    #[param]
    pub self_attn2: SelfAttention,
    #[param]
    pub feed_forward1: FeedForwardModule,
    #[param]
    pub feed_forward2: FeedForwardModule,
    #[param]
    pub feed_forward3: FeedForwardModule,
    #[param]
    pub nonlin_attention: NonlinAttention,
    #[param]
    pub conv_module1: ConvolutionModule,
    #[param]
    pub conv_module2: ConvolutionModule,
    #[param]
    pub bypass_mid: BypassModule,
    #[param]
    pub norm: BiasNorm,
    #[param]
    pub bypass: BypassModule,
}

impl ZipformerEncoderLayer {
    pub fn new_for_stage(config: &ZipaModelConfig, stage: usize) -> Result<Self, Exception> {
        let embed_dim = config.encoder_dim[stage] as i32;
        let pos_dim = config.pos_dim as i32;
        let num_heads = config.num_heads[stage] as i32;
        let query_head_dim = config.query_head_dim as i32;
        let pos_head_dim = config.pos_head_dim as i32;
        let value_head_dim = config.value_head_dim as i32;
        let ff2_dim = config.feedforward_dim[stage] as i32;
        let ff1_dim = (ff2_dim * 3) / 4;
        let ff3_dim = (ff2_dim * 5) / 4;
        let nonlin_hidden = (embed_dim * 3) / 4;
        let kernel_size = config.cnn_module_kernel[stage] as i32;

        Ok(Self {
            self_attn_weights: RelPositionAttentionWeights::new(
                embed_dim,
                pos_dim,
                num_heads,
                query_head_dim,
                pos_head_dim,
            )?,
            self_attn1: SelfAttention::new(embed_dim, num_heads, value_head_dim)?,
            self_attn2: SelfAttention::new(embed_dim, num_heads, value_head_dim)?,
            feed_forward1: FeedForwardModule::new(embed_dim, ff1_dim)?,
            feed_forward2: FeedForwardModule::new(embed_dim, ff2_dim)?,
            feed_forward3: FeedForwardModule::new(embed_dim, ff3_dim)?,
            nonlin_attention: NonlinAttention::new(embed_dim, nonlin_hidden)?,
            conv_module1: ConvolutionModule::new(embed_dim, kernel_size)?,
            conv_module2: ConvolutionModule::new(embed_dim, kernel_size)?,
            bypass_mid: BypassModule::new(embed_dim)?,
            norm: BiasNorm::new(embed_dim)?,
            bypass: BypassModule::new(embed_dim)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::config::{ZipaModelConfig, ZipaVariant};

    use super::ZipformerEncoderLayer;

    #[test]
    fn stage0_layer_matches_reference_shapes() {
        let config = ZipaModelConfig::for_variant(ZipaVariant::SmallCrCtcNsNoDiacritics700k);
        let layer = ZipformerEncoderLayer::new_for_stage(&config, 0).unwrap();

        assert_eq!(layer.self_attn_weights.in_proj.weight.shape(), vec![272, 192]);
        assert_eq!(layer.self_attn_weights.linear_pos.weight.shape(), vec![16, 48]);
        assert_eq!(layer.self_attn1.in_proj.weight.shape(), vec![48, 192]);
        assert_eq!(layer.self_attn1.out_proj.weight.shape(), vec![192, 48]);
        assert_eq!(layer.feed_forward1.in_proj.weight.shape(), vec![384, 192]);
        assert_eq!(layer.feed_forward1.out_proj.weight.shape(), vec![192, 384]);
        assert_eq!(layer.nonlin_attention.in_proj.weight.shape(), vec![432, 192]);
        assert_eq!(layer.nonlin_attention.out_proj.weight.shape(), vec![192, 144]);
        assert_eq!(layer.conv_module1.in_proj.weight.shape(), vec![384, 192]);
        assert_eq!(layer.conv_module1.depthwise_conv.weight.shape(), vec![192, 31, 1]);
        assert_eq!(layer.conv_module1.out_proj.weight.shape(), vec![192, 192]);
        assert_eq!(layer.feed_forward2.in_proj.weight.shape(), vec![512, 192]);
        assert_eq!(layer.feed_forward2.out_proj.weight.shape(), vec![192, 512]);
        assert_eq!(layer.self_attn2.in_proj.weight.shape(), vec![48, 192]);
        assert_eq!(layer.self_attn2.out_proj.weight.shape(), vec![192, 48]);
        assert_eq!(layer.conv_module2.in_proj.weight.shape(), vec![384, 192]);
        assert_eq!(layer.conv_module2.depthwise_conv.weight.shape(), vec![192, 31, 1]);
        assert_eq!(layer.conv_module2.out_proj.weight.shape(), vec![192, 192]);
        assert_eq!(layer.feed_forward3.in_proj.weight.shape(), vec![640, 192]);
        assert_eq!(layer.feed_forward3.out_proj.weight.shape(), vec![192, 640]);
        assert_eq!(layer.bypass_mid.bypass_scale.shape(), vec![192]);
        assert_eq!(layer.norm.bias.shape(), vec![192]);
        assert_eq!(layer.bypass.bypass_scale.shape(), vec![192]);
    }
}
