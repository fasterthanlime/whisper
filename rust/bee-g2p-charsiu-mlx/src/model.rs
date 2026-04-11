use mlx_rs::Array;
use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::macros::ModuleParameters;
use mlx_rs::module::Module;
use mlx_rs::nn;
use mlx_rs::ops;
use mlx_rs::ops::concatenate_axis;
use mlx_rs::ops::indexing::{IndexOp, argmax_axis};

use crate::config::T5Config;

// ---------------------------------------------------------------------------
// KV Cache for decoder attention
// ---------------------------------------------------------------------------

/// Per-layer cache for one attention sublayer (self-attn or cross-attn).
#[derive(Default)]
pub struct KvCache {
    pub keys: Option<Array>,
    pub values: Option<Array>,
}

impl KvCache {
    pub fn update(&mut self, k: Array, v: Array) -> Result<(Array, Array), Exception> {
        match (self.keys.take(), self.values.take()) {
            (Some(prev_k), Some(prev_v)) => {
                let new_k = concatenate_axis(&[prev_k, k], 2)?;
                let new_v = concatenate_axis(&[prev_v, v], 2)?;
                self.keys = Some(new_k.clone());
                self.values = Some(new_v.clone());
                Ok((new_k, new_v))
            }
            _ => {
                self.keys = Some(k.clone());
                self.values = Some(v.clone());
                Ok((k, v))
            }
        }
    }

    pub fn get(&self) -> Option<(&Array, &Array)> {
        match (&self.keys, &self.values) {
            (Some(k), Some(v)) => Some((k, v)),
            _ => None,
        }
    }
}

/// Per-layer cache: self-attention + cross-attention.
#[derive(Default)]
pub struct DecoderLayerCache {
    pub self_attn: KvCache,
    pub cross_attn: KvCache,
}

/// Full decoder cache: one entry per decoder layer.
pub struct DecoderCache {
    pub layers: Vec<DecoderLayerCache>,
}

impl DecoderCache {
    pub fn new(num_layers: usize) -> Self {
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(DecoderLayerCache::default());
        }
        Self { layers }
    }
}

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

        self.attend(&q, &k, &v, position_bias, mask)
    }

    /// Cached forward: projects Q from x, K/V from kv_input, appends to cache.
    pub fn forward_cached(
        &self,
        x: &Array,
        kv_input: &Array,
        cache: &mut KvCache,
        position_bias: Option<&Array>,
        mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        let batch_size = x.shape()[0];
        let q_len = x.shape()[1];
        let kv_len = kv_input.shape()[1];

        let q = self
            .q
            .forward(x)?
            .reshape(&[batch_size, q_len, self.num_heads, self.d_kv])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let k = self
            .k
            .forward(kv_input)?
            .reshape(&[batch_size, kv_len, self.num_heads, self.d_kv])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let v = self
            .v
            .forward(kv_input)?
            .reshape(&[batch_size, kv_len, self.num_heads, self.d_kv])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let (k, v) = cache.update(k, v)?;

        self.attend(&q, &k, &v, position_bias, mask)
    }

    /// Use pre-computed K/V from cache (for cross-attention after first step).
    pub fn forward_cross_cached(
        &self,
        x: &Array,
        cache: &KvCache,
        mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        let batch_size = x.shape()[0];
        let q_len = x.shape()[1];

        let q = self
            .q
            .forward(x)?
            .reshape(&[batch_size, q_len, self.num_heads, self.d_kv])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let (k, v) = cache
            .get()
            .expect("cross-attention cache must be populated");

        self.attend(&q, k, v, None, mask)
    }

    fn attend(
        &self,
        q: &Array,
        k: &Array,
        v: &Array,
        position_bias: Option<&Array>,
        mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        let batch_size = q.shape()[0];
        let q_len = q.shape()[2];

        // scores: [batch, heads, qlen, klen]
        let mut scores = ops::matmul(q, &k.transpose_axes(&[0, 1, 3, 2])?)?;

        if let Some(bias) = position_bias {
            scores = scores.add(bias)?;
        }
        if let Some(m) = mask {
            scores = scores.add(m)?;
        }

        let weights = ops::softmax_axis(&scores, -1, false)?;
        let context = ops::matmul(&weights, v)?;

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

    /// Cached forward: only processes the new token, using KV cache.
    pub fn forward_cached(
        &self,
        x: &Array,
        encoder_output: &Array,
        self_attn_bias: Option<&Array>,
        cache: &mut DecoderLayerCache,
        cross_attn_mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        let normed = self.self_attn_norm.forward(x)?;
        let attn_out = self.self_attn.forward_cached(
            &normed,
            &normed,
            &mut cache.self_attn,
            self_attn_bias,
            None,
        )?;
        let x = x.add(&attn_out)?;

        let normed = self.cross_attn_norm.forward(&x)?;
        let cross_out = if cache.cross_attn.get().is_some() {
            // Reuse cached encoder K/V — still need cross_attn_mask
            self.cross_attn
                .forward_cross_cached(&normed, &cache.cross_attn, cross_attn_mask)?
        } else {
            // First step: compute and cache encoder K/V
            self.cross_attn.forward_cached(
                &normed,
                encoder_output,
                &mut cache.cross_attn,
                None,
                cross_attn_mask,
            )?
        };
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
        self.forward_with_mask(input_ids, None)
    }

    /// Forward with optional attention mask for padding.
    /// `attention_mask`: [batch, seq_len] with 1 for real, 0 for padding.
    pub fn forward_with_mask(
        &self,
        input_ids: &Array,
        attention_mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        let seq_len = input_ids.shape()[1] as i32;
        let mut x = self.embed_tokens.forward(input_ids)?;
        let mut bias = self.position_bias.forward(seq_len, seq_len)?;

        // Apply padding mask: where mask==0, set bias to -inf so those positions are ignored
        if let Some(mask) = attention_mask {
            // mask: [batch, seq_len] -> [batch, 1, 1, seq_len]
            let batch_size = mask.shape()[0];
            let mask_4d = mask.reshape(&[batch_size, 1, 1, seq_len])?;
            let neg_inf = Array::from_f32(f32::NEG_INFINITY);
            let zero = Array::from_f32(0.0);
            let mask_bias = ops::r#where(&mask_4d.as_type::<bool>()?, &zero, &neg_inf)?;
            bias = bias.add(&mask_bias)?;
        }

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

    /// Cached step: processes one new token per batch element using KV cache.
    /// `token_ids` shape: [batch, 1]. `step` is the current decode position (0-indexed).
    /// `cross_attn_mask`: optional [batch, 1, 1, enc_len] mask for encoder padding.
    pub fn step_cached(
        &self,
        token_ids: &Array,
        encoder_output: &Array,
        cache: &mut DecoderCache,
        step: i32,
        cross_attn_mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        let mut x = self.embed_tokens.forward(token_ids)?;

        // Position bias for this step: need bias for position `step` attending to all positions 0..=step
        // Shape: [1, num_heads, 1, step+1]
        let full_bias = self.position_bias.forward(step + 1, step + 1)?;
        // Take only the last row (the new token attending to all cached + itself)
        let step_bias = full_bias.index((.., .., -1.., ..));

        for (block, layer_cache) in self.blocks.iter().zip(cache.layers.iter_mut()) {
            x = block.forward_cached(
                &x,
                encoder_output,
                Some(&step_bias),
                layer_cache,
                cross_attn_mask,
            )?;
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

    pub fn encode_with_mask(
        &self,
        input_ids: &Array,
        attention_mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        self.encoder.forward_with_mask(input_ids, attention_mask)
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
        encoder_output.eval()?;

        let mut cache = DecoderCache::new(self.config.num_decoder_layers as usize);
        let mut generated = Vec::new();
        let start_token = self.config.decoder_start_token_id;
        let mut current_ids = Array::from_int(start_token).reshape(&[1, 1])?;

        for step in 0..max_length {
            let hidden =
                self.decoder
                    .step_cached(&current_ids, &encoder_output, &mut cache, step, None)?;
            let logits = self.lm_head.forward(&hidden)?;
            let next_ids = argmax_axis(&logits.index((.., -1, ..)), -1, None)?;
            next_ids.eval()?;
            let token_id: i32 = next_ids.item();

            if token_id == self.config.eos_token_id {
                break;
            }
            generated.push(token_id);
            current_ids = Array::from_int(token_id).reshape(&[1, 1])?;
        }

        Ok(generated)
    }

    /// Batched generation: encode and decode multiple inputs in parallel.
    /// `input_ids` shape: [batch, seq_len] (padded with 0).
    /// Returns one Vec<i32> per batch element.
    pub fn generate_batch(
        &self,
        input_ids: &Array,
        max_length: i32,
    ) -> Result<Vec<Vec<i32>>, Exception> {
        let batch_size = input_ids.shape()[0] as usize;

        // Create attention mask: 1 where input_ids != 0 (PAD), 0 where padding
        let attention_mask = input_ids.ne(&Array::from_int(0))?.as_type::<i32>()?;
        let encoder_output = self.encode_with_mask(input_ids, Some(&attention_mask))?;
        encoder_output.eval()?;

        // Cross-attention mask: [batch, 1, 1, enc_len] — prevents decoder attending to encoder padding
        let enc_len = input_ids.shape()[1];
        let cross_attn_mask = {
            let mask_2d = attention_mask.reshape(&[batch_size as i32, 1, 1, enc_len as i32])?;
            let neg_inf = Array::from_f32(f32::NEG_INFINITY);
            let zero = Array::from_f32(0.0);
            ops::r#where(&mask_2d.as_type::<bool>()?, &zero, &neg_inf)?
        };

        let mut cache = DecoderCache::new(self.config.num_decoder_layers as usize);
        let mut generated: Vec<Vec<i32>> = vec![Vec::new(); batch_size];
        let mut finished = vec![false; batch_size];

        let start_token = self.config.decoder_start_token_id;
        let mut current_ids =
            Array::from_slice(&vec![start_token; batch_size], &[batch_size as i32, 1])
                .as_type::<i32>()?;

        for step in 0..max_length {
            let hidden = self.decoder.step_cached(
                &current_ids,
                &encoder_output,
                &mut cache,
                step,
                Some(&cross_attn_mask),
            )?;
            let logits = self.lm_head.forward(&hidden)?;
            // next_ids: [batch, 1]
            let next_ids = argmax_axis(&logits.index((.., -1, ..)), -1, None)?;
            next_ids.eval()?;

            let mut all_done = true;
            let mut next_tokens = Vec::with_capacity(batch_size);
            for i in 0..batch_size {
                let token_id: i32 = next_ids.index((i as i32,)).item();
                if !finished[i] {
                    if token_id == self.config.eos_token_id {
                        finished[i] = true;
                    } else {
                        generated[i].push(token_id);
                    }
                }
                // Keep feeding the token even if finished (will be ignored in output)
                next_tokens.push(if finished[i] {
                    self.config.eos_token_id
                } else {
                    token_id
                });
                if !finished[i] {
                    all_done = false;
                }
            }

            if all_done {
                break;
            }

            current_ids =
                Array::from_slice(&next_tokens, &[batch_size as i32, 1]).as_type::<i32>()?;
        }

        Ok(generated)
    }
}
