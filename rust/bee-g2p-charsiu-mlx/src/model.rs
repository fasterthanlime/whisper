use mlx_rs::Array;
use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::macros::ModuleParameters;
use mlx_rs::module::Module;
use mlx_rs::nn;
use mlx_rs::ops;
use mlx_rs::ops::indexing::{IndexOp, argmax_axis};

use crate::config::T5Config;

// ---------------------------------------------------------------------------
// T5 Relative Position Bias
// ---------------------------------------------------------------------------

/// Computes relative position bias for T5 attention.
///
/// T5 uses learned relative position buckets rather than absolute position
/// embeddings or RoPE. The bias is computed once and shared across all heads.
#[derive(Debug, Clone, ModuleParameters)]
pub struct T5RelativePositionBias {
    #[param]
    pub relative_attention_bias: nn::Embedding,
    pub num_buckets: i32,
    pub num_heads: i32,
    pub is_decoder: bool,
}

impl T5RelativePositionBias {
    pub fn new(config: &T5Config, is_decoder: bool) -> Result<Self, Exception> {
        Ok(Self {
            relative_attention_bias: nn::Embedding::new(
                config.relative_attention_num_buckets,
                config.num_heads,
            )?,
            num_buckets: config.relative_attention_num_buckets,
            num_heads: config.num_heads,
            is_decoder,
        })
    }

    /// Compute relative position bias: returns shape [1, num_heads, qlen, klen]
    pub fn forward(&self, query_length: i32, key_length: i32) -> Result<Array, Exception> {
        let context_position =
            ops::arange::<_, i32>(0, query_length, None)?.reshape(&[query_length, 1])?;
        let memory_position =
            ops::arange::<_, i32>(0, key_length, None)?.reshape(&[1, key_length])?;
        let relative_position = memory_position.subtract(&context_position)?;

        let buckets = self.relative_position_bucket(&relative_position)?;

        // Look up bias: [qlen, klen, num_heads]
        let values = self.relative_attention_bias.forward(&buckets)?;
        // Permute to [1, num_heads, qlen, klen]
        values
            .transpose_axes(&[2, 0, 1])?
            .reshape(&[1, self.num_heads, query_length, key_length])
    }

    fn relative_position_bucket(&self, relative_position: &Array) -> Result<Array, Exception> {
        // Exact port of HuggingFace T5Attention._relative_position_bucket
        let mut num_buckets = self.num_buckets;
        let max_distance = 128i32;
        let bidirectional = !self.is_decoder;

        let mut relative_buckets = ops::zeros_like(relative_position)?.as_type::<i32>()?;

        let n = if bidirectional {
            num_buckets /= 2;
            // Positive relative positions get offset by num_buckets
            let is_positive = relative_position.gt(&Array::from_int(0))?;
            relative_buckets = ops::r#where(
                &is_positive,
                &Array::from_int(num_buckets),
                &relative_buckets,
            )?;
            relative_position.abs()?
        } else {
            // For causal: relative_position = -min(relative_position, 0)
            let zero = ops::zeros_like(relative_position)?;
            let clamped = ops::minimum(relative_position, &zero)?;
            clamped.negative()?
        };

        // n is now non-negative
        let max_exact = num_buckets / 2;
        let is_small = n.lt(&Array::from_int(max_exact))?;

        let val_if_large = {
            let n_f = n.as_type::<f32>()?;
            let log_ratio = (n_f / max_exact as f32).log()?;
            let log_max = (max_distance as f32 / max_exact as f32).ln();
            let scaled = log_ratio / log_max * (num_buckets - max_exact) as f32;
            let bucket = scaled.add(Array::from_f32(max_exact as f32))?;
            ops::minimum(&bucket.as_type::<i32>()?, &Array::from_int(num_buckets - 1))?
        };

        let n_i32 = n.as_type::<i32>()?;
        let position_buckets = ops::r#where(&is_small, &n_i32, &val_if_large)?;
        relative_buckets.add(&position_buckets)
    }
}

// ---------------------------------------------------------------------------
// T5 Attention
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
pub struct T5Attention {
    #[param]
    pub q: nn::Linear,
    #[param]
    pub k: nn::Linear,
    #[param]
    pub v: nn::Linear,
    #[param]
    pub o: nn::Linear,
    pub num_heads: i32,
    pub d_kv: i32,
}

impl T5Attention {
    pub fn new(config: &T5Config) -> Result<Self, Exception> {
        let inner_dim = config.num_heads * config.d_kv;
        Ok(Self {
            q: nn::LinearBuilder::new(config.d_model, inner_dim)
                .bias(false)
                .build()?,
            k: nn::LinearBuilder::new(config.d_model, inner_dim)
                .bias(false)
                .build()?,
            v: nn::LinearBuilder::new(config.d_model, inner_dim)
                .bias(false)
                .build()?,
            o: nn::LinearBuilder::new(inner_dim, config.d_model)
                .bias(false)
                .build()?,
            num_heads: config.num_heads,
            d_kv: config.d_kv,
        })
    }

    pub fn forward(
        &self,
        x: &Array,
        kv_input: &Array,
        position_bias: Option<&Array>,
        mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        let batch_size = x.shape()[0];
        let q_len = x.shape()[1];
        let k_len = kv_input.shape()[1];

        let q = self
            .q
            .forward(x)?
            .reshape(&[batch_size, q_len, self.num_heads, self.d_kv])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let k = self
            .k
            .forward(kv_input)?
            .reshape(&[batch_size, k_len, self.num_heads, self.d_kv])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let v = self
            .v
            .forward(kv_input)?
            .reshape(&[batch_size, k_len, self.num_heads, self.d_kv])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // scores: [batch, heads, qlen, klen]
        let mut scores = ops::matmul(&q, &k.transpose_axes(&[0, 1, 3, 2])?)?;

        if let Some(bias) = position_bias {
            scores = scores.add(bias)?;
        }
        if let Some(m) = mask {
            scores = scores.add(m)?;
        }

        let weights = ops::softmax_axis(&scores, -1, false)?;
        let context = ops::matmul(&weights, &v)?;

        let context = context.transpose_axes(&[0, 2, 1, 3])?.reshape(&[
            batch_size,
            q_len,
            self.num_heads * self.d_kv,
        ])?;

        self.o.forward(&context)
    }
}

// ---------------------------------------------------------------------------
// T5 Gated-GELU Feed-Forward
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
pub struct T5GatedGeluDense {
    #[param]
    pub wi_0: nn::Linear,
    #[param]
    pub wi_1: nn::Linear,
    #[param]
    pub wo: nn::Linear,
}

impl T5GatedGeluDense {
    pub fn new(config: &T5Config) -> Result<Self, Exception> {
        Ok(Self {
            wi_0: nn::LinearBuilder::new(config.d_model, config.d_ff)
                .bias(false)
                .build()?,
            wi_1: nn::LinearBuilder::new(config.d_model, config.d_ff)
                .bias(false)
                .build()?,
            wo: nn::LinearBuilder::new(config.d_ff, config.d_model)
                .bias(false)
                .build()?,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array, Exception> {
        let gated = nn::gelu_approximate(self.wi_0.forward(x)?)?;
        let linear = self.wi_1.forward(x)?;
        let hidden = gated.multiply(&linear)?;
        self.wo.forward(&hidden)
    }
}

// ---------------------------------------------------------------------------
// T5 Encoder Block
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
pub struct T5EncoderBlock {
    #[param]
    pub self_attn: T5Attention,
    #[param]
    pub self_attn_norm: nn::RmsNorm,
    #[param]
    pub ff: T5GatedGeluDense,
    #[param]
    pub ff_norm: nn::RmsNorm,
}

impl T5EncoderBlock {
    pub fn new(config: &T5Config) -> Result<Self, Exception> {
        Ok(Self {
            self_attn: T5Attention::new(config)?,
            self_attn_norm: nn::RmsNormBuilder::new(config.d_model)
                .eps(config.layer_norm_epsilon)
                .build()?,
            ff: T5GatedGeluDense::new(config)?,
            ff_norm: nn::RmsNormBuilder::new(config.d_model)
                .eps(config.layer_norm_epsilon)
                .build()?,
        })
    }

    pub fn forward(&self, x: &Array, position_bias: Option<&Array>) -> Result<Array, Exception> {
        let normed = self.self_attn_norm.forward(x)?;
        let attn_out = self
            .self_attn
            .forward(&normed, &normed, position_bias, None)?;
        let x = x.add(&attn_out)?;

        let normed = self.ff_norm.forward(&x)?;
        let ff_out = self.ff.forward(&normed)?;
        x.add(&ff_out)
    }
}

// ---------------------------------------------------------------------------
// T5 Decoder Block
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
pub struct T5DecoderBlock {
    #[param]
    pub self_attn: T5Attention,
    #[param]
    pub self_attn_norm: nn::RmsNorm,
    #[param]
    pub cross_attn: T5Attention,
    #[param]
    pub cross_attn_norm: nn::RmsNorm,
    #[param]
    pub ff: T5GatedGeluDense,
    #[param]
    pub ff_norm: nn::RmsNorm,
}

impl T5DecoderBlock {
    pub fn new(config: &T5Config) -> Result<Self, Exception> {
        Ok(Self {
            self_attn: T5Attention::new(config)?,
            self_attn_norm: nn::RmsNormBuilder::new(config.d_model)
                .eps(config.layer_norm_epsilon)
                .build()?,
            cross_attn: T5Attention::new(config)?,
            cross_attn_norm: nn::RmsNormBuilder::new(config.d_model)
                .eps(config.layer_norm_epsilon)
                .build()?,
            ff: T5GatedGeluDense::new(config)?,
            ff_norm: nn::RmsNormBuilder::new(config.d_model)
                .eps(config.layer_norm_epsilon)
                .build()?,
        })
    }

    pub fn forward(
        &self,
        x: &Array,
        encoder_output: &Array,
        self_attn_bias: Option<&Array>,
        cross_attn_bias: Option<&Array>,
        causal_mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        let normed = self.self_attn_norm.forward(x)?;
        let attn_out = self
            .self_attn
            .forward(&normed, &normed, self_attn_bias, causal_mask)?;
        let x = x.add(&attn_out)?;

        let normed = self.cross_attn_norm.forward(&x)?;
        let cross_out = self
            .cross_attn
            .forward(&normed, encoder_output, cross_attn_bias, None)?;
        let x = x.add(&cross_out)?;

        let normed = self.ff_norm.forward(&x)?;
        let ff_out = self.ff.forward(&normed)?;
        x.add(&ff_out)
    }
}

// ---------------------------------------------------------------------------
// T5 Encoder
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
pub struct T5Encoder {
    #[param]
    pub embed_tokens: nn::Embedding,
    #[param]
    pub position_bias: T5RelativePositionBias,
    #[param]
    pub blocks: Vec<T5EncoderBlock>,
    #[param]
    pub final_layer_norm: nn::RmsNorm,
}

impl T5Encoder {
    pub fn new(config: &T5Config) -> Result<Self, Exception> {
        let mut blocks = Vec::with_capacity(config.num_layers as usize);
        for _ in 0..config.num_layers {
            blocks.push(T5EncoderBlock::new(config)?);
        }
        Ok(Self {
            embed_tokens: nn::Embedding::new(config.vocab_size, config.d_model)?,
            position_bias: T5RelativePositionBias::new(config, false)?,
            blocks,
            final_layer_norm: nn::RmsNormBuilder::new(config.d_model)
                .eps(config.layer_norm_epsilon)
                .build()?,
        })
    }

    pub fn forward(&self, input_ids: &Array) -> Result<Array, Exception> {
        let seq_len = input_ids.shape()[1] as i32;
        let mut x = self.embed_tokens.forward(input_ids)?;
        let bias = self.position_bias.forward(seq_len, seq_len)?;

        for block in &self.blocks {
            x = block.forward(&x, Some(&bias))?;
        }

        self.final_layer_norm.forward(&x)
    }
}

// ---------------------------------------------------------------------------
// T5 Decoder
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
pub struct T5Decoder {
    #[param]
    pub embed_tokens: nn::Embedding,
    #[param]
    pub position_bias: T5RelativePositionBias,
    #[param]
    pub blocks: Vec<T5DecoderBlock>,
    #[param]
    pub final_layer_norm: nn::RmsNorm,
}

impl T5Decoder {
    pub fn new(config: &T5Config) -> Result<Self, Exception> {
        let mut blocks = Vec::with_capacity(config.num_decoder_layers as usize);
        for _ in 0..config.num_decoder_layers {
            blocks.push(T5DecoderBlock::new(config)?);
        }
        Ok(Self {
            embed_tokens: nn::Embedding::new(config.vocab_size, config.d_model)?,
            position_bias: T5RelativePositionBias::new(config, true)?,
            blocks,
            final_layer_norm: nn::RmsNormBuilder::new(config.d_model)
                .eps(config.layer_norm_epsilon)
                .build()?,
        })
    }

    pub fn forward(
        &self,
        decoder_input_ids: &Array,
        encoder_output: &Array,
    ) -> Result<Array, Exception> {
        let dec_len = decoder_input_ids.shape()[1] as i32;

        let mut x = self.embed_tokens.forward(decoder_input_ids)?;
        // The position bias from T5RelativePositionBias(is_decoder=true) already
        // encodes causality — future positions get large negative bias values.
        // No separate causal mask is needed.
        let self_attn_bias = self.position_bias.forward(dec_len, dec_len)?;

        for block in &self.blocks {
            x = block.forward(&x, encoder_output, Some(&self_attn_bias), None, None)?;
        }

        self.final_layer_norm.forward(&x)
    }
}

// ---------------------------------------------------------------------------
// T5 For Conditional Generation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
pub struct T5ForConditionalGeneration {
    #[param]
    pub shared: nn::Embedding,
    #[param]
    pub encoder: T5Encoder,
    #[param]
    pub decoder: T5Decoder,
    #[param]
    pub lm_head: nn::Linear,
    pub config: T5Config,
}

impl T5ForConditionalGeneration {
    pub fn new(config: T5Config) -> Result<Self, Exception> {
        let shared = nn::Embedding::new(config.vocab_size, config.d_model)?;
        let encoder = T5Encoder::new(&config)?;
        let decoder = T5Decoder::new(&config)?;
        let lm_head = nn::LinearBuilder::new(config.d_model, config.vocab_size)
            .bias(false)
            .build()?;

        Ok(Self {
            shared,
            encoder,
            decoder,
            lm_head,
            config,
        })
    }

    pub fn encode(&self, input_ids: &Array) -> Result<Array, Exception> {
        self.encoder.forward(input_ids)
    }

    pub fn decode(
        &self,
        decoder_input_ids: &Array,
        encoder_output: &Array,
    ) -> Result<Array, Exception> {
        let hidden = self.decoder.forward(decoder_input_ids, encoder_output)?;
        self.lm_head.forward(&hidden)
    }

    pub fn generate(&self, input_ids: &Array, max_length: i32) -> Result<Vec<i32>, Exception> {
        let encoder_output = self.encode(input_ids)?;
        let mut generated = vec![self.config.decoder_start_token_id];

        for _ in 0..max_length {
            let decoder_ids =
                Array::from_slice(&generated, &[1, generated.len() as i32]).as_type::<i32>()?;
            let logits = self.decode(&decoder_ids, &encoder_output)?;

            let last_logits = logits.index((.., -1, ..));
            let next_token = argmax_axis(&last_logits, -1, None)?;
            next_token.eval()?;
            let token_id: i32 = next_token.item();

            if token_id == self.config.eos_token_id {
                break;
            }
            generated.push(token_id);
        }

        Ok(generated[1..].to_vec())
    }
}
