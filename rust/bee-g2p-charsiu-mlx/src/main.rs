use std::path::PathBuf;

use anyhow::Result;
use bee_g2p_charsiu_mlx::config::T5Config;
use bee_g2p_charsiu_mlx::load::load_weights_direct;
use bee_g2p_charsiu_mlx::model::T5ForConditionalGeneration;
use bee_g2p_charsiu_mlx::tokenize;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <model_dir> <word> [word...]", args[0]);
        eprintln!(
            "Example: {} /path/to/charsiu/g2p_model Facet Wednesday",
            args[0]
        );
        std::process::exit(1);
    }

    let model_dir = PathBuf::from(&args[1]);
    let words: Vec<&str> = args[2..].iter().map(|s| s.as_str()).collect();
    let lang_code = "eng-us";

    eprintln!("Loading model from {} ...", model_dir.display());
    let config = T5Config::charsiu_g2p();
    let mut model = T5ForConditionalGeneration::new(config)?;

    let stats = load_weights_direct(&mut model, &model_dir)?;
    eprintln!(
        "Loaded {} tensors, {} missing, {} unexpected",
        stats.loaded,
        stats.missing.len(),
        stats.unexpected.len()
    );
    if !stats.missing.is_empty() {
        eprintln!("Missing: {:?}", stats.missing);
    }
    if !stats.unexpected.is_empty() {
        eprintln!("Unexpected: {:?}", stats.unexpected);
    }

    let mut total_us = 0u128;
    for word in &words {
        let start = std::time::Instant::now();
        let input = tokenize::format_g2p_input(word, lang_code);
        let input_ids = tokenize::encode_to_array(&input)?;
        let output_ids = model.generate(&input_ids, 64)?;
        let elapsed = start.elapsed();
        total_us += elapsed.as_micros();
        let ipa = tokenize::decode_byt5(&output_ids);
        println!("{word} -> {ipa}  ({:.1}ms)", elapsed.as_secs_f64() * 1000.0);
    }
    let n = words.len();
    eprintln!(
        "\n{n} words in {:.1}ms total, {:.1}ms/word average",
        total_us as f64 / 1000.0,
        total_us as f64 / 1000.0 / n as f64,
    );

    Ok(())
}
