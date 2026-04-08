use std::env;
use std::path::PathBuf;

use bee_zipa_mlx::infer::ZipaInference;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args_os();
    let _program = args.next();
    let mut bits = 8;
    let mut group_size = 64;
    let mut output = None;

    while let Some(arg) = args.next() {
        match arg.to_string_lossy().as_ref() {
            "--bits" => {
                let value = args.next().ok_or("--bits requires a value")?;
                bits = value.to_string_lossy().parse::<i32>()?;
            }
            "--group-size" => {
                let value = args.next().ok_or("--group-size requires a value")?;
                group_size = value.to_string_lossy().parse::<i32>()?;
            }
            _ if output.is_none() => output = Some(PathBuf::from(arg)),
            _ => {
                eprintln!("usage: zipa-export-quantized [--bits N] [--group-size N] <output.safetensors>");
                std::process::exit(2);
            }
        }
    }

    let output = match output {
        Some(path) => path,
        None => {
            eprintln!("usage: zipa-export-quantized [--bits N] [--group-size N] <output.safetensors>");
            std::process::exit(2);
        }
    };

    let mut inference = ZipaInference::load_reference_small_no_diacritics()?;
    inference.quantize_linears(group_size, bits)?;
    inference.save_quantized_safetensors(&output, group_size, bits)?;

    println!("output: {}", output.display());
    println!("quantized_bits: {bits}");
    println!("quantized_group_size: {group_size}");
    Ok(())
}
