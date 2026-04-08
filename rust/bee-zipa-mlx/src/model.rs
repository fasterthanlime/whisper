use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::macros::{ModuleParameters, Quantizable};
use mlx_rs::module::Module;
use mlx_rs::module::Param;
use mlx_rs::nn;
use mlx_rs::nn::QuantizedLinear;
use mlx_rs::ops;
use mlx_rs::ops::zeros;
use mlx_rs::quantization::MaybeQuantized;
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
            log_scale: Param::new(Array::from_f32(1.0)),
            bias: Param::new(zeros::<f32>(&[dim])?),
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array, Exception> {
        let bias = self.bias.expand_dims_axes(&[0, 1])?;
        let centered = x.subtract(&bias)?;
        let scales = centered
            .square()?
            .mean_axis(-1, true)?
            .rsqrt()?
            .multiply(self.log_scale.value.exp()?)?;
        x.multiply(scales)
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
            depthwise_conv: nn::Conv2dBuilder::new(1, channels, (7, 7))
                .groups(channels)
                .padding((3, 3))
                .build()?,
            pointwise_conv1: nn::Conv2dBuilder::new(channels, channels * 3, (1, 1)).build()?,
            pointwise_conv2: nn::Conv2dBuilder::new(channels * 3, channels, (1, 1)).build()?,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array, Exception> {
        let bypass = x.clone();
        let x = self.depthwise_conv.forward(x)?;
        let x = self.pointwise_conv1.forward(&x)?;
        let x = swoosh_l(&x)?;
        let x = self.pointwise_conv2.forward(&x)?;
        x.add(&bypass)
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
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
    #[quantizable]
    pub out: MaybeQuantized<nn::Linear>,
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
            out: MaybeQuantized::new(
                nn::LinearBuilder::new(
                    out_width * Self::LAYER3_CHANNELS,
                    config.encoder_dim[0] as i32,
                )
                .build()?,
            ),
            out_norm: BiasNorm::new(config.encoder_dim[0] as i32)?,
        })
    }

    pub fn output_width(input_features: i32) -> i32 {
        ((((input_features - 1) / 2) - 1) / 2).max(1)
    }

    pub fn output_frames(input_frames: i32) -> i32 {
        ((input_frames - 7) / 2).max(0)
    }

    pub fn forward(&self, features: &Array) -> Result<Array, Exception> {
        let x = ops::expand_dims(features, -1)?;
        let x = self.conv0.forward(&x)?;
        let x = swoosh_r(&x)?;
        let x = self.conv1.forward(&x)?;
        let x = swoosh_r(&x)?;
        let x = self.conv2.forward(&x)?;
        let x = swoosh_r(&x)?;
        let x = self.convnext.forward(&x)?;

        let shape = x.shape();
        let (b, t, f, c) = (shape[0], shape[1], shape[2], shape[3]);
        let x = x.transpose_axes(&[0, 1, 3, 2])?.reshape(&[b, t, c * f])?;
        let x = self.out.forward(&x)?;
        self.out_norm.forward(&x)
    }

    pub fn quantize_linears(&mut self, max_group_size: i32, bits: i32) -> Result<(), Exception> {
        quantize_linear_adaptive(&mut self.out, max_group_size, bits)?;
        Ok(())
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct CtcHead {
    #[param]
    #[quantizable]
    pub linear: MaybeQuantized<nn::Linear>,
}

impl CtcHead {
    pub fn new(config: &ZipaModelConfig) -> Result<Self, Exception> {
        let encoder_dim = config
            .encoder_dim
            .iter()
            .copied()
            .max()
            .unwrap_or(config.encoder_dim[0]) as i32;
        Ok(Self {
            linear: MaybeQuantized::new(
                nn::LinearBuilder::new(encoder_dim, config.vocab_size as i32).build()?,
            ),
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array, Exception> {
        nn::log_softmax(self.linear.forward(x)?, -1)
    }

    pub fn quantize_linears(&mut self, max_group_size: i32, bits: i32) -> Result<(), Exception> {
        quantize_linear_adaptive(&mut self.linear, max_group_size, bits)?;
        Ok(())
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct ZipaModel {
    #[param]
    #[quantizable]
    pub encoder_embed: EncoderEmbed,
    #[param]
    #[quantizable]
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

    pub fn forward_frontend(&self, features: &Array) -> Result<Array, Exception> {
        self.encoder_embed.forward(features)
    }

    pub fn quantize_linears(&mut self, max_group_size: i32, bits: i32) -> Result<(), Exception> {
        self.encoder_embed.quantize_linears(max_group_size, bits)?;
        self.ctc_head.quantize_linears(max_group_size, bits)?;
        Ok(())
    }
}

pub(crate) fn adaptive_group_size(input_dims: i32, max_group_size: i32) -> Option<i32> {
    if input_dims <= 0 || max_group_size <= 0 {
        return None;
    }
    for group_size in [128, 64, 32] {
        if group_size <= max_group_size && group_size <= input_dims && input_dims % group_size == 0
        {
            return Some(group_size);
        }
    }
    None
}

pub(crate) fn quantize_linear_adaptive(
    linear: &mut MaybeQuantized<nn::Linear>,
    max_group_size: i32,
    bits: i32,
) -> Result<Option<i32>, Exception> {
    let input_dims = match linear {
        MaybeQuantized::Original(linear) => linear.shape().1,
        MaybeQuantized::Quantized(linear) => return Ok(Some(linear.group_size)),
    };
    let Some(group_size) = adaptive_group_size(input_dims, max_group_size) else {
        return Ok(None);
    };
    let quantized = std::mem::replace(
        linear,
        MaybeQuantized::new(nn::LinearBuilder::new(1, 1).build()?),
    )
    .quantize_with(|linear| QuantizedLinear::try_from_linear(linear, group_size, bits))?;
    *linear = quantized;
    Ok(Some(group_size))
}

#[cfg(test)]
fn linear_weight_shape(linear: &MaybeQuantized<nn::Linear>) -> Vec<i32> {
    match linear {
        MaybeQuantized::Original(linear) => linear.weight.shape().to_vec(),
        MaybeQuantized::Quantized(linear) => {
            let shape = linear.inner.weight.shape();
            vec![shape[0], shape[1] * (32 / linear.bits)]
        }
    }
}

pub(crate) fn swoosh_l(x: &Array) -> Result<Array, Exception> {
    let zero = Array::from_f32(0.0);
    ops::logaddexp(&zero, &x.subtract(Array::from_f32(4.0))?)?
        .subtract(x.multiply(Array::from_f32(0.08))?)?
        .subtract(Array::from_f32(0.035))
}

pub(crate) fn swoosh_r(x: &Array) -> Result<Array, Exception> {
    let zero = Array::from_f32(0.0);
    ops::logaddexp(&zero, &x.subtract(Array::from_f32(1.0))?)?
        .subtract(x.multiply(Array::from_f32(0.08))?)?
        .subtract(Array::from_f32(0.313261687))
}

#[cfg(test)]
mod tests {
    use super::{linear_weight_shape, EncoderEmbed, ZipaModel};
    use crate::config::{ZipaModelConfig, ZipaVariant};
    use crate::load::load_frontend_and_ctc_weights;
    use mlx_rs::array;
    use mlx_rs::module::ModuleParameters;
    use mlx_rs::Array;
    use std::path::PathBuf;

    #[test]
    fn encoder_embed_matches_reference_frontend_shapes() {
        let config = ZipaModelConfig::for_variant(ZipaVariant::SmallCrCtcNsNoDiacritics700k);
        let model = ZipaModel::new(&config).unwrap();

        assert_eq!(model.encoder_embed.conv0.weight.shape(), vec![8, 3, 3, 1]);
        assert_eq!(model.encoder_embed.conv1.weight.shape(), vec![32, 3, 3, 8]);
        assert_eq!(
            model.encoder_embed.conv2.weight.shape(),
            vec![128, 3, 3, 32]
        );
        assert_eq!(
            model.encoder_embed.convnext.depthwise_conv.weight.shape(),
            vec![128, 7, 7, 1]
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
        assert_eq!(
            linear_weight_shape(&model.encoder_embed.out),
            vec![192, 2432]
        );
        assert_eq!(linear_weight_shape(&model.ctc_head.linear), vec![127, 512]);
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

    #[test]
    fn frontend_forward_has_expected_output_shape() {
        let model = ZipaModel::small_no_diacritics().unwrap();
        let features = Array::zeros::<f32>(&[1, 120, 80]).unwrap();
        let out = model.forward_frontend(&features).unwrap();
        assert_eq!(out.shape(), vec![1, 56, 192]);
    }

    #[test]
    fn bias_norm_preserves_shape() {
        let norm = super::BiasNorm::new(3).unwrap();
        let x = array!([[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]]);
        let y = norm.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 2, 3]);
    }

    #[test]
    fn frontend_matches_onnx_reference_when_local_artifacts_exist() {
        let home = match std::env::var_os("HOME") {
            Some(home) => PathBuf::from(home),
            None => return,
        };
        let weights = home.join(
            "bearcove/zipa/checkpoints/zipa-cr-ns-small-nodiacritics-700k/exp/frontend_ctc.safetensors",
        );
        let reference = home.join(
            "bearcove/zipa/checkpoints/zipa-cr-ns-small-nodiacritics-700k/exp/authored_282_take_1_frontend_ref.safetensors",
        );
        if !(weights.exists() && reference.exists()) {
            return;
        }

        let mut model = ZipaModel::small_no_diacritics().unwrap();
        let stats = load_frontend_and_ctc_weights(&mut model, &weights).unwrap();
        assert!(stats.missing.is_empty(), "missing: {:?}", stats.missing);

        let tensors = Array::load_safetensors(&reference).unwrap();
        let features = tensors.get("features").unwrap();
        let expected = tensors.get("frontend_out").unwrap();

        let actual = model.forward_frontend(features).unwrap();
        assert_eq!(actual.shape(), expected.shape());
        assert!(
            actual
                .all_close(expected, 1e-4, 1e-4, None)
                .unwrap()
                .item::<bool>(),
            "frontend output diverged from ONNX reference"
        );
    }
}
