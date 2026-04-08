use std::env;
use std::path::PathBuf;
use std::time::Instant;

use bee_zipa_mlx::infer::ZipaInference;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args_os();
    let _program = args.next();
    let mut bits = None;
    let mut group_size = None;
    let mut wav = None;

    while let Some(arg) = args.next() {
        match arg.to_string_lossy().as_ref() {
            "--bits" => {
                let value = args.next().ok_or("--bits requires a value")?;
                bits = Some(value.to_string_lossy().parse::<i32>()?);
            }
            "--group-size" => {
                let value = args.next().ok_or("--group-size requires a value")?;
                group_size = Some(value.to_string_lossy().parse::<i32>()?);
            }
            _ if wav.is_none() => wav = Some(PathBuf::from(arg)),
            _ => {
                eprintln!("usage: zipa-infer [--bits N] [--group-size N] <wav-path>");
                std::process::exit(2);
            }
        }
    }

    let wav = match wav {
        Some(path) => path,
        None => {
            eprintln!("usage: zipa-infer [--bits N] [--group-size N] <wav-path>");
            std::process::exit(2);
        }
    };

    let load_start = Instant::now();
    let mut inference = ZipaInference::load_reference_small_no_diacritics()?;
    let load_elapsed = load_start.elapsed();

    if let Some(bits) = bits {
        inference.quantize_linears(group_size.unwrap_or(64), bits)?;
    }

    let infer_start = Instant::now();
    let output = inference.infer_wav(&wav)?;
    let infer_elapsed = infer_start.elapsed();

    println!("wav: {}", wav.display());
    println!("load_ms: {:.3}", load_elapsed.as_secs_f64() * 1_000.0);
    if let Some(bits) = bits {
        println!("quantized_bits: {bits}");
        println!("quantized_group_size: {}", group_size.unwrap_or(64));
    }
    println!("infer_ms: {:.3}", infer_elapsed.as_secs_f64() * 1_000.0);
    println!("frames: {}", output.log_probs_len);
    println!("tokens: {}", output.tokens.join(" "));
    Ok(())
}
