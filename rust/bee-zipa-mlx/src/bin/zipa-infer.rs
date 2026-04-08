use std::env;
use std::path::PathBuf;
use std::time::Instant;

use bee_zipa_mlx::infer::ZipaInference;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args_os();
    let _program = args.next();
    let wav = match args.next() {
        Some(path) => PathBuf::from(path),
        None => {
            eprintln!("usage: zipa-infer <wav-path>");
            std::process::exit(2);
        }
    };

    let load_start = Instant::now();
    let inference = ZipaInference::load_reference_small_no_diacritics()?;
    let load_elapsed = load_start.elapsed();

    let infer_start = Instant::now();
    let output = inference.infer_wav(&wav)?;
    let infer_elapsed = infer_start.elapsed();

    println!("wav: {}", wav.display());
    println!("load_ms: {:.3}", load_elapsed.as_secs_f64() * 1_000.0);
    println!("infer_ms: {:.3}", infer_elapsed.as_secs_f64() * 1_000.0);
    println!("frames: {}", output.log_probs_len);
    println!("tokens: {}", output.tokens.join(" "));
    Ok(())
}
