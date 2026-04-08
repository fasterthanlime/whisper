use std::env;
use std::path::PathBuf;

use bee_zipa_mlx::infer::ZipaInference;
use mlx_rs::ops::indexing::{argmax_axis, IndexOp};
use mlx_rs::Array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args_os();
    let _program = args.next();
    let wav = match args.next() {
        Some(path) => PathBuf::from(path),
        None => {
            eprintln!("usage: zipa-compare <wav-path> <reference-safetensors>");
            std::process::exit(2);
        }
    };
    let reference = match args.next() {
        Some(path) => PathBuf::from(path),
        None => {
            eprintln!("usage: zipa-compare <wav-path> <reference-safetensors>");
            std::process::exit(2);
        }
    };

    let inference = ZipaInference::load_reference_small_no_diacritics()?;
    let output = inference.infer_wav(&wav)?;

    let tensors = Array::load_safetensors(&reference)?;
    let expected = tensors
        .get("log_probs")
        .ok_or("reference missing log_probs")?
        .clone();

    let diff = output.log_probs.subtract(&expected)?;
    let max_abs = diff.abs()?.max(None)?.item::<f32>();
    let mean_abs = diff.abs()?.mean(None)?.item::<f32>();

    let actual_ids = argmax_axis(output.log_probs.index((0, .., ..)), -1, false)?;
    let expected_ids = argmax_axis(expected.index((0, .., ..)), -1, false)?;
    let actual_ids = actual_ids.as_slice::<u32>();
    let expected_ids = expected_ids.as_slice::<u32>();

    let mismatch_frames = actual_ids
        .iter()
        .zip(expected_ids.iter())
        .filter(|(a, b)| a != b)
        .count();

    let tail = 8usize.min(actual_ids.len());
    println!("wav: {}", wav.display());
    println!("reference: {}", reference.display());
    println!("frames: {}", actual_ids.len());
    println!("max_abs_diff: {max_abs:.8}");
    println!("mean_abs_diff: {mean_abs:.8}");
    println!("argmax_mismatch_frames: {mismatch_frames}");
    println!(
        "actual_tail_ids: {:?}",
        &actual_ids[actual_ids.len().saturating_sub(tail)..]
    );
    println!(
        "expected_tail_ids: {:?}",
        &expected_ids[expected_ids.len().saturating_sub(tail)..]
    );
    Ok(())
}
