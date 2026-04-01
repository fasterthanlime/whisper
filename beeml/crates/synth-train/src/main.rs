use anyhow::Result;
use clap::Parser;

#[derive(Parser)]
#[command(about = "Prepare training data from corrupted corpus + run MLX-LM LoRA training")]
struct Args {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(clap::Subcommand)]
enum Cmd {
    /// Convert corpus JSONL → MLX-LM completions format (train/valid splits)
    Prepare {
        #[arg(short, long, default_value = "data/corpus_dashboard.jsonl")]
        input: String,
        #[arg(short, long, default_value = "training/data")]
        output: String,
        #[arg(long, default_value = "12000")]
        total_examples: usize,
        #[arg(long, default_value = "0.5")]
        error_rate: f64,
    },
    /// Run MLX-LM LoRA training (wraps uvx)
    Train {
        #[arg(long, default_value = "training/data")]
        data: String,
        #[arg(long, default_value = "training/adapters")]
        adapters: String,
        #[arg(long, default_value = "Qwen/Qwen2.5-0.5B")]
        model: String,
        #[arg(long, default_value = "1000")]
        iters: usize,
        #[arg(long, default_value = "4")]
        batch_size: usize,
        #[arg(long, default_value = "8")]
        num_layers: usize,
    },
}

fn main() -> Result<()> {
    match Args::parse().cmd {
        Cmd::Prepare {
            input,
            output,
            total_examples,
            error_rate,
        } => {
            let config = synth_train::PrepareConfig {
                input,
                output,
                total_examples,
                error_rate,
            };
            let stats = synth_train::prepare(&config, |msg| eprintln!("{msg}"))?;
            eprintln!(
                "({} error + {} identity = {} total)",
                stats.correction_examples, stats.identity_examples, stats.total
            );
            Ok(())
        }
        Cmd::Train {
            data,
            adapters,
            model,
            iters,
            batch_size,
            num_layers,
        } => {
            eprintln!("=== ASR Correction Model Training ===");
            eprintln!("Model:    {model}");
            eprintln!("Data:     {data}");
            eprintln!("Adapters: {adapters}");
            eprintln!("Iters:    {iters}");

            let config = synth_train::TrainConfig {
                data,
                adapters: adapters.clone(),
                model,
                iters,
                batch_size,
                num_layers,
                ..Default::default()
            };
            let status = synth_train::train(&config)?;
            if !status.success() {
                anyhow::bail!("Training failed with exit code: {:?}", status.code());
            }
            eprintln!("Training complete. Adapters saved to {adapters}");
            Ok(())
        }
    }
}
