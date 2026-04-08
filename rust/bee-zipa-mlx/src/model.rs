use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::macros::ModuleParameters;
use mlx_rs::module::Param;
use mlx_rs::nn;
use mlx_rs::ops::zeros;
use mlx_rs::Array;

use crate::config::{ZipaModelConfig, ZipaVariant};

#[derive(Debug, Clone, ModuleParameters)]
pub struct BiasNorm {
    #[param]
    pub log_scale: Param<Array>,
    #[param]
    pub bias: Param<Array>,
}

impl BiasNorm {
    pub fn new(dim: i32) -> Result<Self, Exception> {
        Ok(Self {
            log_scale: Param::new(zeros::<f32>(&[])?),
            bias: Param::new(zeros::<f32>(&[dim])?),
        })
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct ConvNeXtFrontend {
    #[param]
    pub depthwise_conv: nn::Conv2d,
    #[param]
    pub pointwise_conv1: nn::Conv2d,
    #[param]
    pub pointwise_conv2: nn::Conv2d,
}

impl ConvNeXtFrontend {
    pub fn new(channels: i32) -> Result<Self, Exception> {
        Ok(Self {
            depthwise_conv: nn::Conv2dBuilder::new(channels, channels, (7, 7))
                .groups(channels)
                .padding((3, 3))
                .build()?,
            pointwise_conv1: nn::Conv2dBuilder::new(channels, channels * 3, (1, 1)).build()?,
            pointwise_conv2: nn::Conv2dBuilder::new(channels * 3, channels, (1, 1)).build()?,
        })
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct EncoderEmbed {
    #[param]
    pub conv0: nn::Conv2d,
    #[param]
    pub conv1: nn::Conv2d,
    #[param]
    pub conv2: nn::Conv2d,
    #[param]
    pub convnext: ConvNeXtFrontend,
    #[param]
    pub out: nn::Linear,
    #[param]
    pub out_norm: BiasNorm,
}

impl EncoderEmbed {
    pub const LAYER1_CHANNELS: i32 = 8;
    pub const LAYER2_CHANNELS: i32 = 32;
    pub const LAYER3_CHANNELS: i32 = 128;

    pub fn new(config: &ZipaModelConfig) -> Result<Self, Exception> {
        let input_features = config.feature_dim as i32;
        let out_width = ((((input_features - 1) / 2) - 1) / 2).max(1);
        Ok(Self {
            conv0: nn::Conv2dBuilder::new(1, Self::LAYER1_CHANNELS, 3)
                .padding((0, 1))
                .build()?,
            conv1: nn::Conv2dBuilder::new(Self::LAYER1_CHANNELS, Self::LAYER2_CHANNELS, 3)
                .stride(2)
                .build()?,
            conv2: nn::Conv2dBuilder::new(Self::LAYER2_CHANNELS, Self::LAYER3_CHANNELS, (3, 3))
                .stride((1, 2))
                .build()?,
            convnext: ConvNeXtFrontend::new(Self::LAYER3_CHANNELS)?,
            out: nn::LinearBuilder::new(out_width * Self::LAYER3_CHANNELS, config.encoder_dim[0] as i32)
                .build()?,
            out_norm: BiasNorm::new(config.encoder_dim[0] as i32)?,
        })
    }

    pub fn output_width(input_features: i32) -> i32 {
        ((((input_features - 1) / 2) - 1) / 2).max(1)
    }

    pub fn output_frames(input_frames: i32) -> i32 {
        ((input_frames - 7) / 2).max(0)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct CtcHead {
    #[param]
    pub linear: nn::Linear,
}

impl CtcHead {
    pub fn new(config: &ZipaModelConfig) -> Result<Self, Exception> {
        Ok(Self {
            linear: nn::LinearBuilder::new(
                *config.encoder_dim.last().unwrap_or(&config.encoder_dim[0]) as i32,
                config.vocab_size as i32,
            )
            .build()?,
        })
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct ZipaModel {
    #[param]
    pub encoder_embed: EncoderEmbed,
    #[param]
    pub ctc_head: CtcHead,
}

impl ZipaModel {
    pub fn new(config: &ZipaModelConfig) -> Result<Self, Exception> {
        Ok(Self {
            encoder_embed: EncoderEmbed::new(config)?,
            ctc_head: CtcHead::new(config)?,
        })
    }

    pub fn small_no_diacritics() -> Result<Self, Exception> {
        let config = ZipaModelConfig::for_variant(ZipaVariant::SmallCrCtcNsNoDiacritics700k);
        Self::new(&config)
    }
}

#[cfg(test)]
mod tests {
    use super::{EncoderEmbed, ZipaModel};
    use crate::config::{ZipaModelConfig, ZipaVariant};
    use mlx_rs::module::ModuleParameters;

    #[test]
    fn encoder_embed_matches_reference_frontend_shapes() {
        let config = ZipaModelConfig::for_variant(ZipaVariant::SmallCrCtcNsNoDiacritics700k);
        let model = ZipaModel::new(&config).unwrap();

        assert_eq!(model.encoder_embed.conv0.weight.shape(), vec![8, 3, 3, 1]);
        assert_eq!(model.encoder_embed.conv1.weight.shape(), vec![32, 3, 3, 8]);
        assert_eq!(model.encoder_embed.conv2.weight.shape(), vec![128, 3, 3, 32]);
        assert_eq!(
            model.encoder_embed.convnext.depthwise_conv.weight.shape(),
            vec![128, 7, 7, 128]
        );
        assert_eq!(model.encoder_embed.convnext.depthwise_conv.groups, 128);
        assert_eq!(
            model.encoder_embed.convnext.pointwise_conv1.weight.shape(),
            vec![384, 1, 1, 128]
        );
        assert_eq!(
            model.encoder_embed.convnext.pointwise_conv2.weight.shape(),
            vec![128, 1, 1, 384]
        );
        assert_eq!(model.encoder_embed.out.weight.shape(), vec![192, 2432]);
        assert_eq!(model.ctc_head.linear.weight.shape(), vec![127, 256]);
    }

    #[test]
    fn frontend_length_formula_matches_zipa_comments() {
        assert_eq!(EncoderEmbed::output_width(80), 19);
        assert_eq!(EncoderEmbed::output_frames(245), 119);
        assert_eq!(EncoderEmbed::output_frames(247), 120);
    }

    #[test]
    fn model_exposes_parameters() {
        let model = ZipaModel::small_no_diacritics().unwrap();
        let flat = model.parameters().flatten();
        assert!(flat.contains_key("encoder_embed.conv0.weight"));
        assert!(flat.contains_key("encoder_embed.convnext.depthwise_conv.weight"));
        assert!(flat.contains_key("encoder_embed.out_norm.log_scale"));
        assert!(flat.contains_key("ctc_head.linear.weight"));
    }
}
