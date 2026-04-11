use std::env;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};
use bee_qwen3_asr::tokenizers::Tokenizer;

fn main() -> Result<()> {
    let (tokenizer_path, text) = parse_args()?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow!("loading {}: {e}", tokenizer_path.display()))?;
    let encoding = tokenizer
        .encode(text.as_str(), false)
        .map_err(|e| anyhow!("encoding text: {e}"))?;

    println!("text: {text}");
    for (index, ((id, token), (start, end))) in encoding
        .get_ids()
        .iter()
        .zip(encoding.get_tokens())
        .zip(encoding.get_offsets())
        .enumerate()
    {
        println!("{index}\t{id}\t{start}..{end}\t{token}");
    }

    Ok(())
}

fn parse_args() -> Result<(PathBuf, String)> {
    let mut args = env::args().skip(1);
    let mut tokenizer_path = env::var_os("BEE_TOKENIZER_PATH").map(PathBuf::from);
    let mut text = None::<String>;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--tokenizer-path" => {
                let value = args.next().context("--tokenizer-path requires a value")?;
                tokenizer_path = Some(PathBuf::from(value));
            }
            "--text" => {
                text = Some(args.next().context("--text requires a value")?);
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(anyhow!("unexpected argument: {other}")),
        }
    }

    let tokenizer_path = tokenizer_path.ok_or_else(|| {
        anyhow!("missing tokenizer path; set BEE_TOKENIZER_PATH or pass --tokenizer-path")
    })?;
    let text = text.ok_or_else(|| anyhow!("missing --text"))?;
    Ok((tokenizer_path, text))
}

fn print_usage() {
    eprintln!("usage: qwen3-tokenize [--tokenizer-path PATH] --text TEXT");
}
