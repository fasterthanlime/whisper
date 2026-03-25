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
    /// Convert corpus JSONL → MLX-LM completions format (train/valid/test splits)
    Prepare {
        #[arg(short, long, default_value = "data/corpus_5k.jsonl")]
        input: String,
        #[arg(short, long, default_value = "training/data")]
        output: String,
        #[arg(long, default_value = "95000")]
        identity_count: usize,
        #[arg(long, default_value = "~/.claude/history.jsonl")]
        claude_history: String,
        #[arg(long, default_value = "~/.codex/history.jsonl")]
        codex_history: String,
        #[arg(long, default_value = "0.8")]
        train_ratio: f64,
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
        #[arg(long, default_value = "1")]
        batch_size: usize,
        #[arg(long, default_value = "4")]
        num_layers: usize,
    },
}

fn main() -> Result<()> {
    match Args::parse().cmd {
        Cmd::Prepare {
            input,
            output,
            identity_count,
            claude_history,
            codex_history,
            train_ratio,
        } => {
            let config = synth_train::PrepareConfig {
                input,
                output,
                identity_count,
                claude_history,
                codex_history,
                train_ratio,
            };
            let stats = synth_train::prepare(&config, |msg| eprintln!("{msg}"))?;
            eprintln!(
                "({} correction + {} identity examples)",
                stats.correction_examples, stats.identity_examples
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
