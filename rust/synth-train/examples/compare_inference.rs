use anyhow::Result;
use clap::Parser;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value = "training/adapters")]
    adapters: String,
    #[arg(long, default_value = "Qwen/Qwen2.5-0.5B")]
    model: String,
    #[arg(long, default_value_t = 64)]
    max_tokens: usize,
    #[arg(long, default_value = "")]
    parakeet: String,
    #[arg(long)]
    qwen: String,
}

fn print_result(label: &str, result: &beeml_train::InferenceOutput) {
    println!("== {label} ==");
    println!("sanitized: {}", result.text);
    println!("raw: {}", result.raw_text);
    println!(
        "timing: total={}ms encode={}ms prefill={}ms decode={}ms generate={}ms prompt_toks={} output_toks={}",
        result.stats.total_ms,
        result.stats.encode_ms,
        result.stats.prefill_ms,
        result.stats.decode_ms,
        result.stats.generate_ms,
        result.stats.prompt_tokens,
        result.stats.output_tokens,
    );
    println!();
}

fn main() -> Result<()> {
    let args = Args::parse();
    let prompt = beeml_train::build_correction_prompt(&args.parakeet, &args.qwen);

    let base_config = beeml_train::InferenceConfig {
        model: args.model.clone(),
        adapters: args.adapters.clone(),
        attach_adapters: false,
        max_tokens: args.max_tokens,
        ..Default::default()
    };
    let adapter_config = beeml_train::InferenceConfig {
        model: args.model,
        adapters: args.adapters,
        attach_adapters: true,
        max_tokens: args.max_tokens,
        ..Default::default()
    };

    match beeml_train::InferenceServer::start(&base_config)
        .and_then(|mut server| server.infer_with_stats(&prompt))
    {
        Ok(result) => print_result("base", &result),
        Err(err) => {
            println!("== base ==");
            println!("error: {err}");
            println!();
        }
    }

    match beeml_train::InferenceServer::start(&adapter_config)
        .and_then(|mut server| server.infer_with_stats(&prompt))
    {
        Ok(result) => print_result("adapter", &result),
        Err(err) => {
            println!("== adapter ==");
            println!("error: {err}");
            println!();
        }
    }

    Ok(())
}
