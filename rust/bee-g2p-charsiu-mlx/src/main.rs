use std::path::PathBuf;

use anyhow::Result;
use bee_g2p_charsiu_mlx::engine::G2pEngine;
use bee_g2p_charsiu_mlx::ownership::ByteSpan;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage: {} <model_dir> [--batch N] [--cross-attention] <word> [word...]",
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
    let mut engine = G2pEngine::load(&model_dir)?;
    eprintln!("Model loaded.");

    if cross_attention {
        run_cross_attention(&mut engine, &words, lang_code)?;
    } else if let Some(bs) = batch_size {
        run_batched(&mut engine, &words, lang_code, bs)?;
    } else {
        run_sequential(&mut engine, &words, lang_code)?;
    }

    Ok(())
}

fn run_sequential(engine: &mut G2pEngine, words: &[&str], lang_code: &str) -> Result<()> {
    let mut total_us = 0u128;
    for word in words {
        let start = std::time::Instant::now();
        let ipa = engine.g2p(word, lang_code)?;
        let elapsed = start.elapsed();
        total_us += elapsed.as_micros();
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

fn run_cross_attention(engine: &mut G2pEngine, words: &[&str], lang_code: &str) -> Result<()> {
    for word in words {
        // Create per-character byte spans for the word
        let spans: Vec<ByteSpan> = word
            .char_indices()
            .map(|(byte_offset, ch)| ByteSpan {
                label: ch.to_string(),
                byte_start: byte_offset,
                byte_end: byte_offset + ch.len_utf8(),
            })
            .collect();

        let start = std::time::Instant::now();
        let output = engine.probe(word, lang_code, &spans)?;
        let elapsed = start.elapsed();

        println!(
            "{word} -> {}  ({:.1}ms)",
            output.ipa,
            elapsed.as_secs_f64() * 1000.0
        );
        println!(
            "  cross-attention: [{}, {}]",
            output.dec_len, output.enc_len
        );

        // Show ownership spans
        println!("  ownership:");
        for span in &output.ownership {
            println!(
                "    {:>6} -> {:<4} (score={:.4})",
                format!("{:?}", span.label),
                span.ipa_text,
                span.avg_score,
            );
        }
        println!();
    }
    Ok(())
}

fn run_batched(
    engine: &mut G2pEngine,
    words: &[&str],
    lang_code: &str,
    batch_size: usize,
) -> Result<()> {
    let mut total_us = 0u128;
    let mut total_words = 0usize;

    for chunk in words.chunks(batch_size) {
        let start = std::time::Instant::now();
        let ipas = engine.g2p_batch(chunk, lang_code)?;
        let elapsed = start.elapsed();
        total_us += elapsed.as_micros();
        total_words += chunk.len();

        let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
        for (word, ipa) in chunk.iter().zip(ipas.iter()) {
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
