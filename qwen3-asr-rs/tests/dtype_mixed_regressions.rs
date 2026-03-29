use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{LayerNorm, Module, RmsNorm};

#[test]
fn layer_norm_mixed_dtype_contiguous_regression() -> Result<()> {
    let device = &Device::Cpu;
    let x = Tensor::new(&[[[1f32, 2f32, 3f32]]], device)?.to_dtype(DType::BF16)?;
    let w = Tensor::new(&[1f32, 1f32, 1f32], device)?;
    let b = Tensor::new(&[0f32, 0f32, 0f32], device)?;
    let ln = LayerNorm::new(w, b, 1e-5);
    let y = ln.forward(&x)?;
    assert_eq!(y.dims(), x.dims());
    Ok(())
}

#[test]
fn layer_norm_mixed_dtype_noncontiguous_regression() -> Result<()> {
    let device = &Device::Cpu;
    let x = Tensor::new(&[[[1f32, 2f32, 3f32], [4f32, 5f32, 6f32]]], device)?
        .to_dtype(DType::BF16)?
        .transpose(1, 2)?;
    let w = Tensor::new(&[1f32, 1f32], device)?;
    let b = Tensor::new(&[0f32, 0f32], device)?;
    let ln = LayerNorm::new(w, b, 1e-5);
    let y = ln.forward(&x)?;
    assert_eq!(y.dims(), x.dims());
    Ok(())
}

#[test]
fn rms_norm_mixed_dtype_noncontiguous_regression() -> Result<()> {
    let device = &Device::Cpu;
    let x = Tensor::new(&[[[1f32, 2f32, 3f32], [4f32, 5f32, 6f32]]], device)?
        .to_dtype(DType::BF16)?
        .transpose(1, 2)?;
    let w = Tensor::new(&[1f32, 1f32], device)?;
    let rms = RmsNorm::new(w, 1e-5);
    let y = rms.forward(&x)?;
    assert_eq!(y.dims(), x.dims());
    Ok(())
}
