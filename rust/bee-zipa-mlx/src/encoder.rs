use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::macros::ModuleParameters;
use mlx_rs::module::Module;
use mlx_rs::module::Param;
use mlx_rs::nn;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::softmax_axis;
use mlx_rs::ops::sum_axes;
use mlx_rs::ops::tanh;
use mlx_rs::ops::zeros;
use mlx_rs::Array;

use crate::config::ZipaModelConfig;
use crate::model::BiasNorm;

#[derive(Debug, Clone)]
pub struct CompactRelPositionalEncoding {
    pub embed_dim: i32,
    pub length_factor: f32,
}

impl CompactRelPositionalEncoding {
    pub fn new(embed_dim: i32, length_factor: f32) -> Self {
        Self {
            embed_dim,
            length_factor,
        }
    }

    pub fn forward(&self, seq_len: i32, left_context_len: i32) -> Result<Array, Exception> {
        let t = seq_len + left_context_len;
        let total = 2 * t - 1;
        let half_dim = self.embed_dim / 2;
        let compression_length = (self.embed_dim as f32).sqrt();
        let length_scale =
            self.length_factor * self.embed_dim as f32 / (2.0 * std::f32::consts::PI);

        let mut pe = vec![0.0f32; (total * self.embed_dim) as usize];
        for row in 0..total {
            let rel = row - (t - 1);
            let rel_f = rel as f32;
            let x_compressed = compression_length
                * rel_f.signum()
                * ((rel_f.abs() + compression_length).ln() - compression_length.ln());
            let x_atan = (x_compressed / length_scale).atan();

            for i in 0..half_dim {
                let angle = x_atan * (i + 1) as f32;
                let base = (row * self.embed_dim + 2 * i) as usize;
                pe[base] = angle.cos();
                pe[base + 1] = angle.sin();
            }
            pe[(row * self.embed_dim + (self.embed_dim - 1)) as usize] = 1.0;
        }

        let start = t - seq_len;
        let end = t + seq_len - 1;
        let mut out = vec![0.0f32; ((2 * seq_len - 1) * self.embed_dim) as usize];
        for row in start..end {
            let src = (row * self.embed_dim) as usize;
            let dst = ((row - start) * self.embed_dim) as usize;
            out[dst..dst + self.embed_dim as usize]
                .copy_from_slice(&pe[src..src + self.embed_dim as usize]);
        }

        Ok(Array::from_slice(
            &out,
            &[1, 2 * seq_len - 1, self.embed_dim],
        ))
    }
}

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

    pub fn forward(&self, src_orig: &Array, src: &Array) -> Result<Array, Exception> {
        src_orig.add(
            src.subtract(src_orig)?
                .multiply(self.bypass_scale.as_ref())?,
        )
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct Downsample2 {
    #[param]
    pub weights: Param<Array>,
    pub output_dim: i32,
}

impl Downsample2 {
    pub fn new(output_dim: i32) -> Result<Self, Exception> {
        Ok(Self {
            weights: Param::new(Array::from_slice(&[0.5f32, 0.5f32], &[2, 1, 1])),
            output_dim,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array, Exception> {
        let shape = x.shape();
        let seq_len = shape[0];
        let batch = shape[1];
        let channels = shape[2];
        let factor = self.weights.shape()[0];

        let x = if channels == self.output_dim {
            x.clone()
        } else {
            let pad_channels = self.output_dim - channels;
            let pad = zeros::<f32>(&[seq_len, batch, pad_channels])?;
            mlx_rs::ops::concatenate_axis(&[x.clone(), pad], 2)?
        };
        let padded_seq_len = x.shape()[0];
        let batch = x.shape()[1];
        let channels = x.shape()[2];

        let remainder = padded_seq_len % factor;
        let x = if remainder == 0 {
            x
        } else {
            let pad = zeros::<f32>(&[factor - remainder, batch, channels])?;
            mlx_rs::ops::concatenate_axis(&[x, pad], 0)?
        };
        let downsampled_len = x.shape()[0] / factor;
        let x = x.reshape(&[downsampled_len, factor, batch, channels])?;
        let weighted = x.multiply(self.weights.as_ref())?;
        sum_axes(&weighted, &[1], false)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct RelPositionAttentionWeights {
    #[param]
    pub in_proj: nn::Linear,
    #[param]
    pub linear_pos: nn::Linear,
    pub num_heads: i32,
    pub query_head_dim: i32,
    pub pos_head_dim: i32,
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
            num_heads,
            query_head_dim,
            pos_head_dim,
        })
    }

    pub fn forward_with_position(&self, x: &Array, pos_emb: &Array) -> Result<Array, Exception> {
        let shape = x.shape();
        let seq_len = shape[0];
        let batch_size = shape[1];
        let query_dim = self.num_heads * self.query_head_dim;
        let pos_query_dim = self.num_heads * self.pos_head_dim;

        let proj = self.in_proj.forward(x)?;
        let q = proj
            .index((.., .., 0..query_dim))
            .reshape(&[seq_len, batch_size, self.num_heads, self.query_head_dim])?
            .transpose_axes(&[2, 1, 0, 3])?;
        let k = proj
            .index((.., .., query_dim..(2 * query_dim)))
            .reshape(&[seq_len, batch_size, self.num_heads, self.query_head_dim])?
            .transpose_axes(&[2, 1, 3, 0])?;
        let p = proj
            .index((.., .., (2 * query_dim)..(2 * query_dim + pos_query_dim)))
            .reshape(&[seq_len, batch_size, self.num_heads, self.pos_head_dim])?
            .transpose_axes(&[2, 1, 0, 3])?;

        let attn_scores = q.matmul(&k)?;
        let pos = self
            .linear_pos
            .forward(pos_emb)?
            .reshape(&[1, 2 * seq_len - 1, self.num_heads, self.pos_head_dim])?
            .transpose_axes(&[2, 0, 3, 1])?;
        let rel_scores = p.matmul(&pos)?;
        let abs_scores = relative_to_absolute(&rel_scores, seq_len)?;
        softmax_axis(attn_scores.add(&abs_scores)?, -1, false)
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

    pub fn forward(&self, x: &Array, attn_weights: &Array) -> Result<Array, Exception> {
        let shape = x.shape();
        let seq_len = shape[0];
        let batch_size = shape[1];
        let num_heads = attn_weights.shape()[0];
        let value_dim = self.in_proj.weight.shape()[0];
        let value_head_dim = value_dim / num_heads;

        let values = self
            .in_proj
            .forward(x)?
            .reshape(&[seq_len, batch_size, num_heads, value_head_dim])?
            .transpose_axes(&[2, 1, 0, 3])?;
        let mixed = attn_weights.matmul(&values)?;
        let merged = mixed
            .transpose_axes(&[2, 1, 0, 3])?
            .reshape(&[seq_len, batch_size, value_dim])?;
        self.out_proj.forward(&merged)
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

    pub fn forward(&self, x: &Array) -> Result<Array, Exception> {
        let x = self.in_proj.forward(x)?;
        let x = crate::model::swoosh_l(&x)?;
        self.out_proj.forward(&x)
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

    pub fn forward(&self, x: &Array, attn_weights: &Array) -> Result<Array, Exception> {
        let shape = x.shape();
        let seq_len = shape[0];
        let batch_size = shape[1];
        let hidden = self.out_proj.weight.shape()[1];
        let third = hidden;

        let proj = self.in_proj.forward(x)?;
        let s = tanh(proj.index((.., .., 0..third)))?;
        let x_gate = proj.index((.., .., third..(2 * third))).multiply(&s)?;
        let y = proj.index((.., .., (2 * third)..(3 * third)));

        let num_heads = attn_weights.shape()[0];
        let head_dim = hidden / num_heads;
        let x_heads = x_gate
            .reshape(&[seq_len, batch_size, num_heads, head_dim])?
            .transpose_axes(&[2, 1, 0, 3])?;
        let mixed = attn_weights
            .matmul(&x_heads)?
            .transpose_axes(&[2, 1, 0, 3])?
            .reshape(&[seq_len, batch_size, hidden])?;
        self.out_proj.forward(&mixed.multiply(&y)?)
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

    pub fn forward(&self, x: &Array) -> Result<Array, Exception> {
        let channels = self.out_proj.weight.shape()[0];
        let proj = self.in_proj.forward(x)?;
        let main = proj.index((.., .., 0..channels));
        let gate = nn::sigmoid(proj.index((.., .., channels..(2 * channels))))?;
        let mixed = main.multiply(&gate)?;
        let conv_in = mixed.transpose_axes(&[1, 0, 2])?;
        let conv_out = self
            .depthwise_conv
            .forward(&conv_in)?
            .transpose_axes(&[1, 0, 2])?;
        let x = crate::model::swoosh_r(&conv_out)?;
        self.out_proj.forward(&x)
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

#[derive(Debug, Clone)]
pub struct Stage0Encoder {
    pub encoder_pos: CompactRelPositionalEncoding,
    pub layer0: ZipformerEncoderLayer,
    pub layer1: ZipformerEncoderLayer,
}

#[derive(Debug, Clone)]
pub struct Stage1EncoderPrefix {
    pub downsample: Downsample2,
    pub encoder_pos: CompactRelPositionalEncoding,
    pub layer0: ZipformerEncoderLayer,
    pub layer1: ZipformerEncoderLayer,
    pub out_combiner: BypassModule,
}

#[derive(Debug, Clone)]
pub struct StageEncoder {
    pub stage: usize,
    pub downsample: Downsample2,
    pub encoder_pos: CompactRelPositionalEncoding,
    pub layers: Vec<ZipformerEncoderLayer>,
    pub out_combiner: BypassModule,
}

impl Stage0Encoder {
    pub fn new(config: &ZipaModelConfig) -> Result<Self, Exception> {
        Ok(Self {
            encoder_pos: CompactRelPositionalEncoding::new(config.pos_dim as i32, 1.0),
            layer0: ZipformerEncoderLayer::new_for_stage(config, 0)?,
            layer1: ZipformerEncoderLayer::new_for_stage(config, 0)?,
        })
    }

    pub fn forward(&self, src: &Array) -> Result<Array, Exception> {
        let seq_len = src.shape()[0];
        let pos_emb = self.encoder_pos.forward(seq_len, 0)?;
        let src = self.layer0.forward(src, &pos_emb)?;
        self.layer1.forward(&src, &pos_emb)
    }
}

impl Stage1EncoderPrefix {
    pub fn new(config: &ZipaModelConfig) -> Result<Self, Exception> {
        Ok(Self {
            downsample: Downsample2::new(config.encoder_dim[1] as i32)?,
            encoder_pos: CompactRelPositionalEncoding::new(config.pos_dim as i32, 1.0),
            layer0: ZipformerEncoderLayer::new_for_stage(config, 1)?,
            layer1: ZipformerEncoderLayer::new_for_stage(config, 1)?,
            out_combiner: BypassModule::new(config.encoder_dim[1] as i32)?,
        })
    }

    pub fn forward(&self, src: &Array) -> Result<Array, Exception> {
        let src_padded = pad_channels(src, self.out_combiner.bypass_scale.shape()[0])?;
        let src = self.downsample.forward(src)?;
        let seq_len = src.shape()[0];
        let pos_emb = self.encoder_pos.forward(seq_len, 0)?;
        let src = self.layer0.forward(&src, &pos_emb)?;
        let src = self.layer1.forward(&src, &pos_emb)?;
        let src = upsample_by_repeat(&src, src_padded.shape()[0])?;
        self.out_combiner.forward(&src_padded, &src)
    }
}

impl StageEncoder {
    pub fn new(config: &ZipaModelConfig, stage: usize) -> Result<Self, Exception> {
        assert!(
            stage > 0,
            "StageEncoder is only for stages with downsample/out_combiner"
        );
        let mut layers = Vec::with_capacity(config.num_encoder_layers[stage]);
        for _ in 0..config.num_encoder_layers[stage] {
            layers.push(ZipformerEncoderLayer::new_for_stage(config, stage)?);
        }
        Ok(Self {
            stage,
            downsample: Downsample2::new(config.encoder_dim[stage] as i32)?,
            encoder_pos: CompactRelPositionalEncoding::new(config.pos_dim as i32, 1.0),
            layers,
            out_combiner: BypassModule::new(config.encoder_dim[stage] as i32)?,
        })
    }

    pub fn forward(&self, src: &Array) -> Result<Array, Exception> {
        let src_padded = pad_channels(src, self.out_combiner.bypass_scale.shape()[0])?;
        let mut src = self.downsample.forward(&src_padded)?;
        let seq_len = src.shape()[0];
        let pos_emb = self.encoder_pos.forward(seq_len, 0)?;
        for layer in &self.layers {
            src = layer.forward(&src, &pos_emb)?;
        }
        let src = upsample_by_repeat(&src, src_padded.shape()[0])?;
        self.out_combiner.forward(&src_padded, &src)
    }
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

    pub fn forward_with_attn_weights(
        &self,
        src: &Array,
        attn_weights: &Array,
    ) -> Result<Array, Exception> {
        let src_orig = src.clone();
        let selected_attn_weights = attn_weights.index((0..1, .., .., ..));

        let src = src.add(&self.feed_forward1.forward(src)?)?;
        let src = src.add(
            &self
                .nonlin_attention
                .forward(&src, &selected_attn_weights)?,
        )?;
        let src = src.add(&self.self_attn1.forward(&src, attn_weights)?)?;
        let src = src.add(&self.conv_module1.forward(&src)?)?;
        let src = src.add(&self.feed_forward2.forward(&src)?)?;
        let src = self.bypass_mid.forward(&src_orig, &src)?;
        let src = src.add(&self.self_attn2.forward(&src, attn_weights)?)?;
        let src = src.add(&self.conv_module2.forward(&src)?)?;
        let src = src.add(&self.feed_forward3.forward(&src)?)?;
        let src = self.norm.forward(&src)?;
        self.bypass.forward(&src_orig, &src)
    }

    pub fn forward(&self, src: &Array, pos_emb: &Array) -> Result<Array, Exception> {
        let attn_weights = self.self_attn_weights.forward_with_position(src, pos_emb)?;
        self.forward_with_attn_weights(src, &attn_weights)
    }
}

fn relative_to_absolute(pos_scores: &Array, seq_len: i32) -> Result<Array, Exception> {
    let mut rows = Vec::with_capacity(seq_len as usize);
    for t in 0..seq_len {
        let start = seq_len - 1 - t;
        let row = pos_scores.index((.., .., t, start..(start + seq_len)));
        rows.push(row.expand_dims_axes(&[2])?);
    }
    mlx_rs::ops::concatenate_axis(&rows, 2)
}

fn pad_channels(x: &Array, output_dim: i32) -> Result<Array, Exception> {
    let shape = x.shape();
    let seq_len = shape[0];
    let batch = shape[1];
    let channels = shape[2];
    if channels == output_dim {
        return Ok(x.clone());
    }
    if channels > output_dim {
        return Ok(x.index((.., .., 0..output_dim)));
    }
    let pad_channels = output_dim - channels;
    let pad = zeros::<f32>(&[seq_len, batch, pad_channels])?;
    mlx_rs::ops::concatenate_axis(&[x.clone(), pad], 2)
}

fn upsample_by_repeat(x: &Array, target_seq_len: i32) -> Result<Array, Exception> {
    let shape = x.shape();
    let seq_len = shape[0];
    let batch = shape[1];
    let channels = shape[2];
    let repeat_factor = (target_seq_len + seq_len - 1) / seq_len;
    let expanded = x.expand_dims_axes(&[1])?;
    let repeated =
        mlx_rs::ops::broadcast_to(&expanded, &[seq_len, repeat_factor, batch, channels])?;
    let reshaped = repeated.reshape(&[seq_len * repeat_factor, batch, channels])?;
    Ok(reshaped.index((0..target_seq_len, .., ..)))
}

#[cfg(test)]
mod tests {
    use crate::config::{ZipaModelConfig, ZipaVariant};
    use crate::load::{
        load_bypass_scale_from_map, load_downsample_weights_from_map,
        load_stage0_layer_weights_from_map, load_stage_layer_weights_from_map,
    };

    use super::{
        CompactRelPositionalEncoding, Downsample2, Stage0Encoder, Stage1EncoderPrefix,
        StageEncoder, ZipformerEncoderLayer,
    };
    use mlx_rs::ops::indexing::IndexOp;
    use mlx_rs::Array;
    use std::path::PathBuf;

    #[test]
    fn stage0_layer_matches_reference_shapes() {
        let config = ZipaModelConfig::for_variant(ZipaVariant::SmallCrCtcNsNoDiacritics700k);
        let layer = ZipformerEncoderLayer::new_for_stage(&config, 0).unwrap();

        assert_eq!(
            layer.self_attn_weights.in_proj.weight.shape(),
            vec![272, 192]
        );
        assert_eq!(
            layer.self_attn_weights.linear_pos.weight.shape(),
            vec![16, 48]
        );
        assert_eq!(layer.self_attn1.in_proj.weight.shape(), vec![48, 192]);
        assert_eq!(layer.self_attn1.out_proj.weight.shape(), vec![192, 48]);
        assert_eq!(layer.feed_forward1.in_proj.weight.shape(), vec![384, 192]);
        assert_eq!(layer.feed_forward1.out_proj.weight.shape(), vec![192, 384]);
        assert_eq!(
            layer.nonlin_attention.in_proj.weight.shape(),
            vec![432, 192]
        );
        assert_eq!(
            layer.nonlin_attention.out_proj.weight.shape(),
            vec![192, 144]
        );
        assert_eq!(layer.conv_module1.in_proj.weight.shape(), vec![384, 192]);
        assert_eq!(
            layer.conv_module1.depthwise_conv.weight.shape(),
            vec![192, 31, 1]
        );
        assert_eq!(layer.conv_module1.out_proj.weight.shape(), vec![192, 192]);
        assert_eq!(layer.feed_forward2.in_proj.weight.shape(), vec![512, 192]);
        assert_eq!(layer.feed_forward2.out_proj.weight.shape(), vec![192, 512]);
        assert_eq!(layer.self_attn2.in_proj.weight.shape(), vec![48, 192]);
        assert_eq!(layer.self_attn2.out_proj.weight.shape(), vec![192, 48]);
        assert_eq!(layer.conv_module2.in_proj.weight.shape(), vec![384, 192]);
        assert_eq!(
            layer.conv_module2.depthwise_conv.weight.shape(),
            vec![192, 31, 1]
        );
        assert_eq!(layer.conv_module2.out_proj.weight.shape(), vec![192, 192]);
        assert_eq!(layer.feed_forward3.in_proj.weight.shape(), vec![640, 192]);
        assert_eq!(layer.feed_forward3.out_proj.weight.shape(), vec![192, 640]);
        assert_eq!(layer.bypass_mid.bypass_scale.shape(), vec![192]);
        assert_eq!(layer.norm.bias.shape(), vec![192]);
        assert_eq!(layer.bypass.bypass_scale.shape(), vec![192]);
    }

    #[test]
    fn compact_positional_encoding_matches_onnx_reference_when_local_artifacts_exist() {
        let home = match std::env::var_os("HOME") {
            Some(home) => PathBuf::from(home),
            None => return,
        };
        let reference = home.join(
            "bearcove/zipa/checkpoints/zipa-cr-ns-small-nodiacritics-700k/exp/authored_282_take_1_layer0_ref.safetensors",
        );
        if !reference.exists() {
            return;
        }

        let tensors = Array::load_safetensors(&reference).unwrap();
        let expected = tensors.get("pos_emb").unwrap();
        let seq_len = tensors.get("layer0_in").unwrap().shape()[0];
        let pos = CompactRelPositionalEncoding::new(48, 1.0)
            .forward(seq_len, 0)
            .unwrap();
        assert_close("pos_emb", &pos, expected);
    }

    #[test]
    fn stage0_layer_matches_onnx_reference_when_local_artifacts_exist() {
        let home = match std::env::var_os("HOME") {
            Some(home) => PathBuf::from(home),
            None => return,
        };
        let weights = home.join(
            "bearcove/zipa/checkpoints/zipa-cr-ns-small-nodiacritics-700k/exp/frontend_ctc.safetensors",
        );
        let reference = home.join(
            "bearcove/zipa/checkpoints/zipa-cr-ns-small-nodiacritics-700k/exp/authored_282_take_1_layer0_ref.safetensors",
        );
        if !(weights.exists() && reference.exists()) {
            return;
        }

        let config = ZipaModelConfig::for_variant(ZipaVariant::SmallCrCtcNsNoDiacritics700k);
        let mut layer = ZipformerEncoderLayer::new_for_stage(&config, 0).unwrap();
        let params = Array::load_safetensors(&weights).unwrap();
        let stats = load_stage0_layer_weights_from_map(&mut layer, &params).unwrap();
        assert!(stats.missing.is_empty(), "missing: {:?}", stats.missing);

        let tensors = Array::load_safetensors(&reference).unwrap();
        let layer0_in = tensors.get("layer0_in").unwrap();
        let pos_emb = tensors.get("pos_emb").unwrap();
        let attn_weights = tensors.get("attn_weights").unwrap();
        let src_orig = layer0_in.clone();

        let actual_attn = layer
            .self_attn_weights
            .forward_with_position(layer0_in, pos_emb)
            .unwrap();
        assert_close("attn_weights", &actual_attn, attn_weights);

        let add0 = layer0_in
            .add(&layer.feed_forward1.forward(layer0_in).unwrap())
            .unwrap();
        assert_close("add0", &add0, tensors.get("add0").unwrap());

        let selected_attn_weights = attn_weights.index((0..1, .., .., ..));
        let add1 = add0
            .add(
                &layer
                    .nonlin_attention
                    .forward(&add0, &selected_attn_weights)
                    .unwrap(),
            )
            .unwrap();
        assert_close("add1", &add1, tensors.get("add1").unwrap());

        let add2 = add1
            .add(&layer.self_attn1.forward(&add1, attn_weights).unwrap())
            .unwrap();
        assert_close("add2", &add2, tensors.get("add2").unwrap());

        let add3 = add2
            .add(&layer.conv_module1.forward(&add2).unwrap())
            .unwrap();
        assert_close("add3", &add3, tensors.get("add3").unwrap());

        let add4 = add3
            .add(&layer.feed_forward2.forward(&add3).unwrap())
            .unwrap();
        assert_close("add4", &add4, tensors.get("add4").unwrap());

        let mid = layer.bypass_mid.forward(&src_orig, &add4).unwrap();
        assert_close("mid", &mid, tensors.get("mid").unwrap());

        let add5 = mid
            .add(&layer.self_attn2.forward(&mid, attn_weights).unwrap())
            .unwrap();
        assert_close("add5", &add5, tensors.get("add5").unwrap());

        let add6 = add5
            .add(&layer.conv_module2.forward(&add5).unwrap())
            .unwrap();
        assert_close("add6", &add6, tensors.get("add6").unwrap());

        let add7 = add6
            .add(&layer.feed_forward3.forward(&add6).unwrap())
            .unwrap();
        assert_close("add7", &add7, tensors.get("add7").unwrap());

        let norm = layer.norm.forward(&add7).unwrap();
        assert_close("norm", &norm, tensors.get("norm").unwrap());

        let actual = layer.bypass.forward(&src_orig, &norm).unwrap();
        assert_close("layer0_out", &actual, tensors.get("layer0_out").unwrap());

        let full_actual = layer.forward(layer0_in, pos_emb).unwrap();
        assert_close(
            "layer0_out_full",
            &full_actual,
            tensors.get("layer0_out").unwrap(),
        );
    }

    #[test]
    fn stage0_encoder_wrapper_matches_onnx_reference_when_local_artifacts_exist() {
        let home = match std::env::var_os("HOME") {
            Some(home) => PathBuf::from(home),
            None => return,
        };
        let weights = home.join(
            "bearcove/zipa/checkpoints/zipa-cr-ns-small-nodiacritics-700k/exp/frontend_ctc.safetensors",
        );
        let reference = home.join(
            "bearcove/zipa/checkpoints/zipa-cr-ns-small-nodiacritics-700k/exp/authored_282_take_1_layer0_ref.safetensors",
        );
        if !(weights.exists() && reference.exists()) {
            return;
        }

        let config = ZipaModelConfig::for_variant(ZipaVariant::SmallCrCtcNsNoDiacritics700k);
        let mut stage0 = Stage0Encoder::new(&config).unwrap();
        let params = Array::load_safetensors(&weights).unwrap();
        let layer0_stats = load_stage0_layer_weights_from_map(&mut stage0.layer0, &params).unwrap();
        assert!(
            layer0_stats.missing.is_empty(),
            "missing: {:?}",
            layer0_stats.missing
        );
        let layer1_stats =
            load_stage_layer_weights_from_map(&mut stage0.layer1, "encoder.stage0.layer1", &params)
                .unwrap();
        assert!(
            layer1_stats.missing.is_empty(),
            "missing: {:?}",
            layer1_stats.missing
        );

        let tensors = Array::load_safetensors(&reference).unwrap();
        let layer0_in = tensors.get("layer0_in").unwrap();
        let expected = tensors.get("stage0_out").unwrap();

        let actual = stage0.forward(layer0_in).unwrap();
        assert_close("stage0_wrapper", &actual, expected);
    }

    #[test]
    fn stage0_layer1_matches_onnx_reference_when_local_artifacts_exist() {
        let home = match std::env::var_os("HOME") {
            Some(home) => PathBuf::from(home),
            None => return,
        };
        let weights = home.join(
            "bearcove/zipa/checkpoints/zipa-cr-ns-small-nodiacritics-700k/exp/frontend_ctc.safetensors",
        );
        let reference = home.join(
            "bearcove/zipa/checkpoints/zipa-cr-ns-small-nodiacritics-700k/exp/authored_282_take_1_layer0_ref.safetensors",
        );
        if !(weights.exists() && reference.exists()) {
            return;
        }

        let config = ZipaModelConfig::for_variant(ZipaVariant::SmallCrCtcNsNoDiacritics700k);
        let mut layer = ZipformerEncoderLayer::new_for_stage(&config, 0).unwrap();
        let params = Array::load_safetensors(&weights).unwrap();
        let stats = load_stage_layer_weights_from_map(&mut layer, "encoder.stage0.layer1", &params)
            .unwrap();
        assert!(stats.missing.is_empty(), "missing: {:?}", stats.missing);

        let tensors = Array::load_safetensors(&reference).unwrap();
        let layer_input = tensors.get("layer0_out").unwrap();
        let pos_emb = tensors.get("pos_emb").unwrap();
        let expected_attn = tensors.get("layer1_attn_weights").unwrap();
        let expected = tensors.get("layer1_out").unwrap();

        let actual_attn = layer
            .self_attn_weights
            .forward_with_position(layer_input, pos_emb)
            .unwrap();
        assert_close("layer1_attn_weights", &actual_attn, expected_attn);

        let actual = layer.forward(layer_input, pos_emb).unwrap();
        assert_close("layer1_out", &actual, expected);
    }

    #[test]
    fn stage1_layer_matches_reference_shapes() {
        let config = ZipaModelConfig::for_variant(ZipaVariant::SmallCrCtcNsNoDiacritics700k);
        let layer = ZipformerEncoderLayer::new_for_stage(&config, 1).unwrap();

        assert_eq!(
            layer.self_attn_weights.in_proj.weight.shape(),
            vec![272, 256]
        );
        assert_eq!(
            layer.self_attn_weights.linear_pos.weight.shape(),
            vec![16, 48]
        );
        assert_eq!(layer.self_attn1.in_proj.weight.shape(), vec![48, 256]);
        assert_eq!(layer.self_attn1.out_proj.weight.shape(), vec![256, 48]);
        assert_eq!(layer.feed_forward1.in_proj.weight.shape(), vec![576, 256]);
        assert_eq!(layer.feed_forward1.out_proj.weight.shape(), vec![256, 576]);
        assert_eq!(
            layer.nonlin_attention.in_proj.weight.shape(),
            vec![576, 256]
        );
        assert_eq!(
            layer.nonlin_attention.out_proj.weight.shape(),
            vec![256, 192]
        );
        assert_eq!(layer.conv_module1.in_proj.weight.shape(), vec![512, 256]);
        assert_eq!(
            layer.conv_module1.depthwise_conv.weight.shape(),
            vec![256, 31, 1]
        );
        assert_eq!(layer.conv_module1.out_proj.weight.shape(), vec![256, 256]);
        assert_eq!(layer.feed_forward2.in_proj.weight.shape(), vec![768, 256]);
        assert_eq!(layer.feed_forward2.out_proj.weight.shape(), vec![256, 768]);
        assert_eq!(layer.self_attn2.in_proj.weight.shape(), vec![48, 256]);
        assert_eq!(layer.self_attn2.out_proj.weight.shape(), vec![256, 48]);
        assert_eq!(layer.conv_module2.in_proj.weight.shape(), vec![512, 256]);
        assert_eq!(
            layer.conv_module2.depthwise_conv.weight.shape(),
            vec![256, 31, 1]
        );
        assert_eq!(layer.conv_module2.out_proj.weight.shape(), vec![256, 256]);
        assert_eq!(layer.feed_forward3.in_proj.weight.shape(), vec![960, 256]);
        assert_eq!(layer.feed_forward3.out_proj.weight.shape(), vec![256, 960]);
        assert_eq!(layer.bypass_mid.bypass_scale.shape(), vec![256]);
        assert_eq!(layer.norm.bias.shape(), vec![256]);
        assert_eq!(layer.bypass.bypass_scale.shape(), vec![256]);
    }

    #[test]
    fn stage1_downsample_matches_onnx_reference_when_local_artifacts_exist() {
        let home = match std::env::var_os("HOME") {
            Some(home) => PathBuf::from(home),
            None => return,
        };
        let weights = home.join(
            "bearcove/zipa/checkpoints/zipa-cr-ns-small-nodiacritics-700k/exp/frontend_ctc.safetensors",
        );
        let reference = home.join(
            "bearcove/zipa/checkpoints/zipa-cr-ns-small-nodiacritics-700k/exp/authored_282_take_1_layer0_ref.safetensors",
        );
        if !(weights.exists() && reference.exists()) {
            return;
        }

        let mut downsample = Downsample2::new(256).unwrap();
        let params = Array::load_safetensors(&weights).unwrap();
        let stats = load_downsample_weights_from_map(
            &mut downsample,
            "encoder.stage1.downsample.weights",
            &params,
        )
        .unwrap();
        assert!(stats.missing.is_empty(), "missing: {:?}", stats.missing);

        let tensors = Array::load_safetensors(&reference).unwrap();
        let stage0_out = tensors.get("stage0_out").unwrap();
        let expected = tensors.get("stage1_downsample_out").unwrap();

        let actual = downsample.forward(stage0_out).unwrap();
        assert_close("stage1_downsample_out", &actual, expected);
    }

    #[test]
    fn stage1_layer0_matches_onnx_reference_when_local_artifacts_exist() {
        let home = match std::env::var_os("HOME") {
            Some(home) => PathBuf::from(home),
            None => return,
        };
        let weights = home.join(
            "bearcove/zipa/checkpoints/zipa-cr-ns-small-nodiacritics-700k/exp/frontend_ctc.safetensors",
        );
        let reference = home.join(
            "bearcove/zipa/checkpoints/zipa-cr-ns-small-nodiacritics-700k/exp/authored_282_take_1_layer0_ref.safetensors",
        );
        if !(weights.exists() && reference.exists()) {
            return;
        }

        let config = ZipaModelConfig::for_variant(ZipaVariant::SmallCrCtcNsNoDiacritics700k);
        let mut layer = ZipformerEncoderLayer::new_for_stage(&config, 1).unwrap();
        let params = Array::load_safetensors(&weights).unwrap();
        let stats = load_stage_layer_weights_from_map(&mut layer, "encoder.stage1.layer0", &params)
            .unwrap();
        assert!(stats.missing.is_empty(), "missing: {:?}", stats.missing);

        let tensors = Array::load_safetensors(&reference).unwrap();
        let layer_input = tensors.get("stage1_downsample_out").unwrap();
        let pos_emb = tensors.get("stage1_pos_emb").unwrap();
        let expected_attn = tensors.get("stage1_layer0_attn_weights").unwrap();
        let expected = tensors.get("stage1_layer0_out").unwrap();

        let actual_attn = layer
            .self_attn_weights
            .forward_with_position(layer_input, pos_emb)
            .unwrap();
        assert_close("stage1_layer0_attn_weights", &actual_attn, expected_attn);

        let actual = layer.forward(layer_input, pos_emb).unwrap();
        assert_close("stage1_layer0_out", &actual, expected);
    }

    #[test]
    fn stage1_layer1_matches_onnx_reference_when_local_artifacts_exist() {
        let home = match std::env::var_os("HOME") {
            Some(home) => PathBuf::from(home),
            None => return,
        };
        let weights = home.join(
            "bearcove/zipa/checkpoints/zipa-cr-ns-small-nodiacritics-700k/exp/frontend_ctc.safetensors",
        );
        let reference = home.join(
            "bearcove/zipa/checkpoints/zipa-cr-ns-small-nodiacritics-700k/exp/authored_282_take_1_layer0_ref.safetensors",
        );
        if !(weights.exists() && reference.exists()) {
            return;
        }

        let config = ZipaModelConfig::for_variant(ZipaVariant::SmallCrCtcNsNoDiacritics700k);
        let mut layer = ZipformerEncoderLayer::new_for_stage(&config, 1).unwrap();
        let params = Array::load_safetensors(&weights).unwrap();
        let stats = load_stage_layer_weights_from_map(&mut layer, "encoder.stage1.layer1", &params)
            .unwrap();
        assert!(stats.missing.is_empty(), "missing: {:?}", stats.missing);

        let tensors = Array::load_safetensors(&reference).unwrap();
        let layer_input = tensors.get("stage1_layer0_out").unwrap();
        let pos_emb = tensors.get("stage1_pos_emb").unwrap();
        let expected_attn = tensors.get("stage1_layer1_attn_weights").unwrap();
        let expected = tensors.get("stage1_layer1_out").unwrap();

        let actual_attn = layer
            .self_attn_weights
            .forward_with_position(layer_input, pos_emb)
            .unwrap();
        assert_close("stage1_layer1_attn_weights", &actual_attn, expected_attn);

        let actual = layer.forward(layer_input, pos_emb).unwrap();
        assert_close("stage1_layer1_out", &actual, expected);
    }

    #[test]
    fn stage1_prefix_matches_onnx_reference_when_local_artifacts_exist() {
        let home = match std::env::var_os("HOME") {
            Some(home) => PathBuf::from(home),
            None => return,
        };
        let weights = home.join(
            "bearcove/zipa/checkpoints/zipa-cr-ns-small-nodiacritics-700k/exp/frontend_ctc.safetensors",
        );
        let reference = home.join(
            "bearcove/zipa/checkpoints/zipa-cr-ns-small-nodiacritics-700k/exp/authored_282_take_1_layer0_ref.safetensors",
        );
        if !(weights.exists() && reference.exists()) {
            return;
        }

        let config = ZipaModelConfig::for_variant(ZipaVariant::SmallCrCtcNsNoDiacritics700k);
        let mut stage1 = Stage1EncoderPrefix::new(&config).unwrap();
        let params = Array::load_safetensors(&weights).unwrap();
        let downsample_stats = load_downsample_weights_from_map(
            &mut stage1.downsample,
            "encoder.stage1.downsample.weights",
            &params,
        )
        .unwrap();
        assert!(
            downsample_stats.missing.is_empty(),
            "missing: {:?}",
            downsample_stats.missing
        );
        let layer_stats =
            load_stage_layer_weights_from_map(&mut stage1.layer0, "encoder.stage1.layer0", &params)
                .unwrap();
        assert!(
            layer_stats.missing.is_empty(),
            "missing: {:?}",
            layer_stats.missing
        );
        let layer1_stats =
            load_stage_layer_weights_from_map(&mut stage1.layer1, "encoder.stage1.layer1", &params)
                .unwrap();
        assert!(
            layer1_stats.missing.is_empty(),
            "missing: {:?}",
            layer1_stats.missing
        );
        let out_combiner_stats = load_bypass_scale_from_map(
            &mut stage1.out_combiner,
            "encoder.stage1.out_combiner.bypass_scale",
            &params,
        )
        .unwrap();
        assert!(
            out_combiner_stats.missing.is_empty(),
            "missing: {:?}",
            out_combiner_stats.missing
        );

        let tensors = Array::load_safetensors(&reference).unwrap();
        let stage0_out = tensors.get("stage0_out").unwrap();
        let expected = tensors.get("stage1_out").unwrap();

        let actual = stage1.forward(stage0_out).unwrap();
        assert_close("stage1_prefix", &actual, expected);
    }

    #[test]
    fn stage2_layer_matches_reference_shapes() {
        let config = ZipaModelConfig::for_variant(ZipaVariant::SmallCrCtcNsNoDiacritics700k);
        let layer = ZipformerEncoderLayer::new_for_stage(&config, 2).unwrap();

        assert_eq!(
            layer.self_attn_weights.in_proj.weight.shape(),
            vec![272, 384]
        );
        assert_eq!(
            layer.self_attn_weights.linear_pos.weight.shape(),
            vec![16, 48]
        );
        assert_eq!(layer.self_attn1.in_proj.weight.shape(), vec![48, 384]);
        assert_eq!(layer.self_attn1.out_proj.weight.shape(), vec![384, 48]);
        assert_eq!(layer.feed_forward1.in_proj.weight.shape(), vec![768, 384]);
        assert_eq!(layer.feed_forward1.out_proj.weight.shape(), vec![384, 768]);
        assert_eq!(
            layer.nonlin_attention.in_proj.weight.shape(),
            vec![864, 384]
        );
        assert_eq!(
            layer.nonlin_attention.out_proj.weight.shape(),
            vec![384, 288]
        );
        assert_eq!(layer.conv_module1.in_proj.weight.shape(), vec![768, 384]);
        assert_eq!(
            layer.conv_module1.depthwise_conv.weight.shape(),
            vec![384, 15, 1]
        );
        assert_eq!(layer.conv_module1.out_proj.weight.shape(), vec![384, 384]);
        assert_eq!(layer.feed_forward2.in_proj.weight.shape(), vec![1024, 384]);
        assert_eq!(layer.feed_forward2.out_proj.weight.shape(), vec![384, 1024]);
        assert_eq!(layer.self_attn2.in_proj.weight.shape(), vec![48, 384]);
        assert_eq!(layer.self_attn2.out_proj.weight.shape(), vec![384, 48]);
        assert_eq!(layer.conv_module2.in_proj.weight.shape(), vec![768, 384]);
        assert_eq!(
            layer.conv_module2.depthwise_conv.weight.shape(),
            vec![384, 15, 1]
        );
        assert_eq!(layer.conv_module2.out_proj.weight.shape(), vec![384, 384]);
        assert_eq!(layer.feed_forward3.in_proj.weight.shape(), vec![1280, 384]);
        assert_eq!(layer.feed_forward3.out_proj.weight.shape(), vec![384, 1280]);
        assert_eq!(layer.bypass_mid.bypass_scale.shape(), vec![384]);
        assert_eq!(layer.norm.bias.shape(), vec![384]);
        assert_eq!(layer.bypass.bypass_scale.shape(), vec![384]);
    }

    #[test]
    fn stage2_layer0_matches_onnx_reference_when_local_artifacts_exist() {
        assert_stage_layer_matches_reference(
            "encoder.stage2.layer0",
            2,
            "stage2_downsample_out",
            "stage2_pos_emb",
            "stage2_layer0_attn_weights",
            "stage2_layer0_out",
        );
    }

    #[test]
    fn stage2_layer1_matches_onnx_reference_when_local_artifacts_exist() {
        assert_stage_layer_matches_reference(
            "encoder.stage2.layer1",
            2,
            "stage2_layer0_out",
            "stage2_pos_emb",
            "stage2_layer1_attn_weights",
            "stage2_layer1_out",
        );
    }

    #[test]
    fn stage2_layer2_matches_onnx_reference_when_local_artifacts_exist() {
        assert_stage_layer_matches_reference(
            "encoder.stage2.layer2",
            2,
            "stage2_layer1_out",
            "stage2_pos_emb",
            "stage2_layer2_attn_weights",
            "stage2_layer2_out",
        );
    }

    #[test]
    fn stage2_wrapper_matches_onnx_reference_when_local_artifacts_exist() {
        assert_stage_wrapper_matches_reference(2, "stage1_out", "stage2_out");
    }

    #[test]
    fn stage3_layer0_matches_onnx_reference_when_local_artifacts_exist() {
        assert_stage_layer_matches_reference(
            "encoder.stage3.layer0",
            3,
            "stage3_downsample_out",
            "stage3_pos_emb",
            "stage3_layer0_attn_weights",
            "stage3_layer0_out",
        );
    }

    #[test]
    fn stage3_layer1_matches_onnx_reference_when_local_artifacts_exist() {
        assert_stage_layer_matches_reference(
            "encoder.stage3.layer1",
            3,
            "stage3_layer0_out",
            "stage3_pos_emb",
            "stage3_layer1_attn_weights",
            "stage3_layer1_out",
        );
    }

    #[test]
    fn stage3_layer2_matches_onnx_reference_when_local_artifacts_exist() {
        assert_stage_layer_matches_reference(
            "encoder.stage3.layer2",
            3,
            "stage3_layer1_out",
            "stage3_pos_emb",
            "stage3_layer2_attn_weights",
            "stage3_layer2_out",
        );
    }

    #[test]
    fn stage3_layer3_matches_onnx_reference_when_local_artifacts_exist() {
        assert_stage_layer_matches_reference(
            "encoder.stage3.layer3",
            3,
            "stage3_layer2_out",
            "stage3_pos_emb",
            "stage3_layer3_attn_weights",
            "stage3_layer3_out",
        );
    }

    #[test]
    fn stage3_wrapper_matches_onnx_reference_when_local_artifacts_exist() {
        assert_stage_wrapper_matches_reference(3, "stage2_out", "stage3_out");
    }

    #[test]
    fn stage4_layer0_matches_onnx_reference_when_local_artifacts_exist() {
        assert_stage_layer_matches_reference(
            "encoder.stage4.layer0",
            4,
            "stage4_downsample_out",
            "stage4_pos_emb",
            "stage4_layer0_attn_weights",
            "stage4_layer0_out",
        );
    }

    #[test]
    fn stage4_layer1_matches_onnx_reference_when_local_artifacts_exist() {
        assert_stage_layer_matches_reference(
            "encoder.stage4.layer1",
            4,
            "stage4_layer0_out",
            "stage4_pos_emb",
            "stage4_layer1_attn_weights",
            "stage4_layer1_out",
        );
    }

    #[test]
    fn stage4_layer2_matches_onnx_reference_when_local_artifacts_exist() {
        assert_stage_layer_matches_reference(
            "encoder.stage4.layer2",
            4,
            "stage4_layer1_out",
            "stage4_pos_emb",
            "stage4_layer2_attn_weights",
            "stage4_layer2_out",
        );
    }

    #[test]
    fn stage4_wrapper_matches_onnx_reference_when_local_artifacts_exist() {
        assert_stage_wrapper_matches_reference(4, "stage3_out", "stage4_out");
    }

    #[test]
    fn stage5_layer0_matches_onnx_reference_when_local_artifacts_exist() {
        assert_stage_layer_matches_reference(
            "encoder.stage5.layer0",
            5,
            "stage5_downsample_out",
            "stage5_pos_emb",
            "stage5_layer0_attn_weights",
            "stage5_layer0_out",
        );
    }

    #[test]
    fn stage5_layer1_matches_onnx_reference_when_local_artifacts_exist() {
        assert_stage_layer_matches_reference(
            "encoder.stage5.layer1",
            5,
            "stage5_layer0_out",
            "stage5_pos_emb",
            "stage5_layer1_attn_weights",
            "stage5_layer1_out",
        );
    }

    #[test]
    fn stage5_wrapper_matches_onnx_reference_when_local_artifacts_exist() {
        assert_stage_wrapper_matches_reference(5, "stage4_out", "stage5_out");
    }

    fn assert_stage_layer_matches_reference(
        prefix: &str,
        stage_index: usize,
        input_key: &str,
        pos_key: &str,
        attn_key: &str,
        output_key: &str,
    ) {
        let home = match std::env::var_os("HOME") {
            Some(home) => PathBuf::from(home),
            None => return,
        };
        let weights = home.join(
            "bearcove/zipa/checkpoints/zipa-cr-ns-small-nodiacritics-700k/exp/frontend_ctc.safetensors",
        );
        let reference = home.join(
            "bearcove/zipa/checkpoints/zipa-cr-ns-small-nodiacritics-700k/exp/authored_282_take_1_layer0_ref.safetensors",
        );
        if !(weights.exists() && reference.exists()) {
            return;
        }

        let config = ZipaModelConfig::for_variant(ZipaVariant::SmallCrCtcNsNoDiacritics700k);
        let mut layer = ZipformerEncoderLayer::new_for_stage(&config, stage_index).unwrap();
        let params = Array::load_safetensors(&weights).unwrap();
        let stats = load_stage_layer_weights_from_map(&mut layer, prefix, &params).unwrap();
        assert!(stats.missing.is_empty(), "missing: {:?}", stats.missing);

        let tensors = Array::load_safetensors(&reference).unwrap();
        let layer_input = tensors.get(input_key).unwrap();
        let pos_emb = tensors.get(pos_key).unwrap();
        let expected_attn = tensors.get(attn_key).unwrap();
        let expected = tensors.get(output_key).unwrap();

        let actual_attn = layer
            .self_attn_weights
            .forward_with_position(layer_input, pos_emb)
            .unwrap();
        assert_close(attn_key, &actual_attn, expected_attn);

        let actual = layer.forward(layer_input, pos_emb).unwrap();
        assert_close(output_key, &actual, expected);
    }

    fn assert_stage_wrapper_matches_reference(
        stage_index: usize,
        input_key: &str,
        output_key: &str,
    ) {
        let home = match std::env::var_os("HOME") {
            Some(home) => PathBuf::from(home),
            None => return,
        };
        let weights = home.join(
            "bearcove/zipa/checkpoints/zipa-cr-ns-small-nodiacritics-700k/exp/frontend_ctc.safetensors",
        );
        let reference = home.join(
            "bearcove/zipa/checkpoints/zipa-cr-ns-small-nodiacritics-700k/exp/authored_282_take_1_layer0_ref.safetensors",
        );
        if !(weights.exists() && reference.exists()) {
            return;
        }

        let config = ZipaModelConfig::for_variant(ZipaVariant::SmallCrCtcNsNoDiacritics700k);
        let mut stage = StageEncoder::new(&config, stage_index).unwrap();
        let params = Array::load_safetensors(&weights).unwrap();
        let downsample_stats = load_downsample_weights_from_map(
            &mut stage.downsample,
            &format!("encoder.stage{stage_index}.downsample.weights"),
            &params,
        )
        .unwrap();
        assert!(
            downsample_stats.missing.is_empty(),
            "missing: {:?}",
            downsample_stats.missing
        );
        for (layer_index, layer) in stage.layers.iter_mut().enumerate() {
            let prefix = format!("encoder.stage{stage_index}.layer{layer_index}");
            let layer_stats = load_stage_layer_weights_from_map(layer, &prefix, &params).unwrap();
            assert!(
                layer_stats.missing.is_empty(),
                "missing: {:?}",
                layer_stats.missing
            );
        }
        let out_combiner_stats = load_bypass_scale_from_map(
            &mut stage.out_combiner,
            &format!("encoder.stage{stage_index}.out_combiner.bypass_scale"),
            &params,
        )
        .unwrap();
        assert!(
            out_combiner_stats.missing.is_empty(),
            "missing: {:?}",
            out_combiner_stats.missing
        );

        let tensors = Array::load_safetensors(&reference).unwrap();
        let input = tensors.get(input_key).unwrap();
        let expected = tensors.get(output_key).unwrap();

        let actual = stage.forward(input).unwrap();
        assert_close(output_key, &actual, expected);
    }

    fn assert_close(name: &str, actual: &Array, expected: &Array) {
        assert_eq!(actual.shape(), expected.shape(), "{name} shape mismatch");
        assert!(
            actual
                .all_close(expected, 1e-4, 1e-4, None)
                .unwrap()
                .item::<bool>(),
            "{name} diverged from ONNX reference"
        );
    }
}
