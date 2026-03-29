#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use candle::{test_utils, Device, Tensor};
use candle_nn::{LayerNorm, Module, RmsNorm};

#[test]
fn layer_norm() -> Result<()> {
    let device = &Device::Cpu;
    let w = Tensor::new(&[3f32], device)?;
    let b = Tensor::new(&[0.5f32], device)?;
    let ln2 = LayerNorm::new(Tensor::cat(&[&w, &w], 0)?, Tensor::cat(&[&b, &b], 0)?, 1e-8);
    let ln3 = LayerNorm::new(
        Tensor::cat(&[&w, &w, &w], 0)?,
        Tensor::cat(&[&b, &b, &b], 0)?,
        1e-8,
    );
    let ln = LayerNorm::new(w, b, 1e-8);

    let two = Tensor::new(&[[[2f32]]], device)?;
    let res = ln.forward(&two)?.flatten_all()?;
    assert_eq!(res.to_vec1::<f32>()?, [0.5f32]);

    let inp = Tensor::new(&[[[4f32, 0f32]]], device)?;
    let res = ln2.forward(&inp)?;
    assert_eq!(res.to_vec3::<f32>()?, [[[3.5f32, -2.5]]]);

    let inp = Tensor::new(&[[[1f32, 2., 3.], [4., 5., 6.], [9., 8., 7.]]], device)?;
    let res = ln3.forward(&inp)?;
    assert_eq!(
        test_utils::to_vec3_round(&res, 4)?,
        [[
            [-3.1742, 0.5, 4.1742],
            [-3.1742, 0.5, 4.1742],
            [4.1742, 0.5, -3.1742]
        ]]
    );
    let mean = (res.sum_keepdim(2)? / 3.0)?;
    // The average value should be `b`.
    assert_eq!(
        test_utils::to_vec3_round(&mean, 4)?,
        [[[0.5], [0.5], [0.5]]]
    );
    let std = (res.broadcast_sub(&mean)?.sqr()?.sum_keepdim(2)?.sqrt()? / 3.0)?;
    // The standard deviation should be sqrt(`w`).
    assert_eq!(
        test_utils::to_vec3_round(&std, 4)?,
        [[[1.7321], [1.7321], [1.7321]]]
    );
    Ok(())
}

#[test]
fn layer_norm_mixed_dtype_contiguous() -> Result<()> {
    let device = &Device::Cpu;
    let x = Tensor::new(&[[[1f32, 2f32, 3f32]]], device)?.to_dtype(candle::DType::BF16)?;
    let w = Tensor::new(&[1f32, 1f32, 1f32], device)?;
    let b = Tensor::new(&[0f32, 0f32, 0f32], device)?;
    let ln = LayerNorm::new(w, b, 1e-5);
    let y = ln.forward(&x)?;
    assert_eq!(y.dims(), x.dims());
    Ok(())
}

#[test]
fn rms_norm_mixed_dtype_noncontiguous() -> Result<()> {
    let device = &Device::Cpu;
    let x = Tensor::new(&[[[1f32, 2f32, 3f32], [4f32, 5f32, 6f32]]], device)?
        .to_dtype(candle::DType::BF16)?
        .transpose(1, 2)?;
    let w = Tensor::new(&[1f32, 1f32], device)?;
    let rms = RmsNorm::new(w, 1e-5);
    let y = rms.forward(&x)?;
    assert_eq!(y.dims(), x.dims());
    Ok(())
}
