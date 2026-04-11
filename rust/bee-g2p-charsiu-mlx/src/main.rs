use std::path::PathBuf;

use anyhow::Result;
use bee_g2p_charsiu_mlx::config::T5Config;
use bee_g2p_charsiu_mlx::load::load_weights_direct;
use bee_g2p_charsiu_mlx::model::T5ForConditionalGeneration;
use bee_g2p_charsiu_mlx::tokenize;
use mlx_rs::ops::indexing::IndexOp;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage: {} <model_dir> [--batch N] <word> [word...]",
            args[0]
        );
        std::process::exit(1);
    }

    let model_dir = PathBuf::from(&args[1]);
    let lang_code = "eng-us";

    // Parse flags
    let mut batch_size: Option<usize> = None;
    let mut cross_attention = false;
    let mut words: Vec<&str> = Vec::new();
    let mut i = 2;
    while i < args.len() {
        if args[i] == "--batch" {
            i += 1;
            batch_size = Some(args[i].parse()?);
        } else if args[i] == "--cross-attention" {
            cross_attention = true;
        } else {
            words.push(&args[i]);
        }
        i += 1;
    }

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

    if cross_attention {
        run_cross_attention(&model, &words, lang_code)?;
    } else if let Some(bs) = batch_size {
        run_batched(&model, &words, lang_code, bs)?;
    } else {
        run_sequential(&model, &words, lang_code)?;
    }

    Ok(())
}

fn run_sequential(
    model: &T5ForConditionalGeneration,
    words: &[&str],
    lang_code: &str,
) -> Result<()> {
    let mut total_us = 0u128;
    for word in words {
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

fn run_cross_attention(
    model: &T5ForConditionalGeneration,
    words: &[&str],
    lang_code: &str,
) -> Result<()> {
    for word in words {
        let input = tokenize::format_g2p_input(word, lang_code);
        let input_ids = tokenize::encode_to_array(&input)?;

        let start = std::time::Instant::now();
        let (generated, cross_attn) = model.generate_with_cross_attention(&input_ids, 64)?;
        let elapsed = start.elapsed();

        let ipa = tokenize::decode_byt5(&generated);
        println!("{word} -> {ipa}  ({:.1}ms)", elapsed.as_secs_f64() * 1000.0);

        // cross_attn shape: [dec_len, enc_len]
        let dec_len = cross_attn.shape()[0];
        let enc_len = cross_attn.shape()[1];
        println!("  cross-attention: [{dec_len}, {enc_len}]");

        // Show which input byte each output byte attends to most
        // Map byte positions back to the UTF-8 characters they belong to
        let input_bytes = input.as_bytes();
        let output_bytes: Vec<u8> = generated
            .iter()
            .filter(|&&id| id >= 3)
            .map(|&id| (id - 3) as u8)
            .collect();

        // Build byte→char labels: each byte gets the full character it belongs to
        let ipa_chars: Vec<(usize, char)> = {
            let ipa_str = String::from_utf8_lossy(&output_bytes);
            let mut result = Vec::new();
            for (byte_offset, ch) in ipa_str.char_indices() {
                let byte_len = ch.len_utf8();
                for b in 0..byte_len {
                    result.push((byte_offset + b, ch));
                }
            }
            result
        };
        let input_chars: Vec<char> = {
            let s = String::from_utf8_lossy(input_bytes);
            let mut result = vec![' '; input_bytes.len()];
            for (byte_offset, ch) in s.char_indices() {
                let byte_len = ch.len_utf8();
                for b in 0..byte_len {
                    if byte_offset + b < result.len() {
                        result[byte_offset + b] = ch;
                    }
                }
            }
            result
        };

        for out_idx in 0..dec_len as usize {
            let row = cross_attn.index((out_idx as i32, ..));
            let top_idx: i32 = mlx_rs::ops::indexing::argmax_axis(&row, 0, None)?.item();
            let top_score: f32 = row.index((top_idx,)).item();

            let out_label = if out_idx < ipa_chars.len() {
                format!("{}", ipa_chars[out_idx].1)
            } else {
                "?".to_string()
            };
            let in_label = if (top_idx as usize) < input_chars.len() {
                format!("{}", input_chars[top_idx as usize])
            } else {
                "?".to_string()
            };

            println!(
                "  out[{out_idx:>2}] {out_label:<4} -> in[{top_idx:>2}] {in_label:<4} score={top_score:.4}"
            );
        }
        println!();
    }
    Ok(())
}

fn run_batched(
    model: &T5ForConditionalGeneration,
    words: &[&str],
    lang_code: &str,
    batch_size: usize,
) -> Result<()> {
    let mut total_us = 0u128;
    let mut total_words = 0usize;

    for chunk in words.chunks(batch_size) {
        let prompts: Vec<String> = chunk
            .iter()
            .map(|w| tokenize::format_g2p_input(w, lang_code))
            .collect();
        let input_ids = tokenize::encode_batch_to_array(&prompts)?;

        let start = std::time::Instant::now();
        let results = model.generate_batch(&input_ids, 64)?;
        let elapsed = start.elapsed();
        total_us += elapsed.as_micros();
        total_words += chunk.len();

        let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
        for (word, output_ids) in chunk.iter().zip(results.iter()) {
            let ipa = tokenize::decode_byt5(output_ids);
            println!("{word} -> {ipa}");
        }
        eprintln!(
            "  batch of {} in {:.1}ms ({:.1}ms/word)",
            chunk.len(),
            elapsed_ms,
            elapsed_ms / chunk.len() as f64,
        );
    }

    eprintln!(
        "\n{total_words} words in {:.1}ms total, {:.1}ms/word average (batched)",
        total_us as f64 / 1000.0,
        total_us as f64 / 1000.0 / total_words as f64,
    );
    Ok(())
}
