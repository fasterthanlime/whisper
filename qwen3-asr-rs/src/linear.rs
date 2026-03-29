use candle_core::quantized::QMatMul;
use candle_core::{DType, Result, Tensor};
use candle_nn::{Linear, Module};

/// Linear layer that supports both dense (safetensors) and quantized (GGUF)
/// weights behind a single `forward()` interface.
pub(crate) enum LinearW {
    /// Standard dense linear layer (weight + optional bias).
    Dense(Linear),
    /// Quantized matmul (GGUF Q4_K, Q8_0, etc.) + optional dequantized bias.
    Quant {
        qmatmul: QMatMul,
        bias: Option<Tensor>,
    },
}

impl LinearW {
    /// Create a dense linear layer (safetensors path).
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self::Dense(Linear::new(weight, bias))
    }

    /// Create a quantized linear layer from a QMatMul and optional bias tensor.
    pub fn from_qmatmul(qmatmul: QMatMul, bias: Option<Tensor>) -> Self {
        Self::Quant { qmatmul, bias }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(linear) => linear.forward(x),
            Self::Quant { qmatmul, bias } => {
                // Candle's Metal quantized matmul path currently asserts F32
                // inputs. Cast reduced-precision activations before dispatch.
                let target_dtype = x.dtype();
                let q_input = if target_dtype == DType::F32 {
                    x.clone()
                } else {
                    x.to_dtype(DType::F32)?
                };
                let mut out = qmatmul.forward(&q_input)?;
                if out.dtype() != target_dtype {
                    out = out.to_dtype(target_dtype)?;
                }
                match bias {
                    Some(b) => {
                        if b.dtype() == out.dtype() {
                            out.broadcast_add(b)
                        } else {
                            let b_cast = b.to_dtype(out.dtype())?;
                            out.broadcast_add(&b_cast)
                        }
                    }
                    None => Ok(out),
                }
            }
        }
    }
}
