use anyhow::Result;
use rand::prelude::*;
use std::io::Write;

/// Stats returned from the prepare step
#[derive(Debug, serde::Serialize)]
pub struct PrepareStats {
    pub correction_examples: usize,
    pub identity_examples: usize,
    pub total: usize,
    pub train_count: usize,
    pub valid_count: usize,
}

pub struct PrepareConfig {
    /// Path to corpus JSONL (each line has original/qwen/parakeet fields)
    pub input: String,
    /// Output directory for train.jsonl + valid.jsonl
    pub output: String,
    /// Total number of training examples to generate
    pub total_examples: usize,
    /// Fraction of examples that contain a real ASR error (0.0–1.0)
    pub error_rate: f64,
}

impl Default for PrepareConfig {
    fn default() -> Self {
        Self {
            input: "data/corpus_dashboard.jsonl".into(),
            output: "training/data".into(),
            total_examples: 12000,
            error_rate: 0.5,
        }
    }
}

pub struct TrainConfig {
    pub data: String,
    pub adapters: String,
    pub model: String,
    pub iters: usize,
    pub batch_size: usize,
    pub num_layers: usize,
    /// Stop training if val loss doesn't improve for this many evals. 0 = disabled.
    pub early_stop_patience: usize,
    /// How often to run validation (in training steps).
    pub steps_per_eval: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            data: "training/data".into(),
            adapters: "training/adapters".into(),
            model: "Qwen/Qwen2.5-0.5B".into(),
            iters: 2000,
            batch_size: 4,
            num_layers: 8,
            early_stop_patience: 10,
            steps_per_eval: 500,
        }
    }
}

/// Prepare training data from corpus JSONL → MLX-LM completions format.
///
/// Generates `total_examples` training prompts:
/// - `error_rate` fraction are correction examples (randomly sampled from corpus pairs)
/// - The rest are identity examples (randomly sampled originals where both ASR lanes are correct)
///
/// All examples have the same shape. Fixed 90/10 train/valid split.
pub fn prepare(config: &PrepareConfig, mut on_status: impl FnMut(&str)) -> Result<PrepareStats> {
    let content = std::fs::read_to_string(&config.input)?;
    let lines: Vec<&str> = content.lines().filter(|l| !l.trim().is_empty()).collect();

    // Parse corpus pairs
    let mut corpus_pairs: Vec<(String, String, String)> = Vec::new(); // (original, parakeet, qwen)
    let mut originals: Vec<String> = Vec::new(); // unique originals for identity examples

    for line in &lines {
        let v: serde_json::Value = serde_json::from_str(line)?;
        let original = v["original"].as_str().unwrap_or("").to_string();
        let parakeet = v["parakeet"].as_str().unwrap_or("").to_string();
        let qwen = v["qwen"].as_str().unwrap_or("").to_string();
        if original.is_empty() { continue; }
        originals.push(original.clone());
        corpus_pairs.push((original, parakeet, qwen));
    }

    on_status(&format!("Loaded {} corpus pairs, {} unique originals", corpus_pairs.len(), originals.len()));

    if corpus_pairs.is_empty() {
        anyhow::bail!("No corpus pairs found in {}", config.input);
    }

    let mut rng = rand::rng();
    let n_error = (config.total_examples as f64 * config.error_rate).round() as usize;
    let n_identity = config.total_examples - n_error;

    on_status(&format!("Generating {} error + {} identity = {} total examples",
        n_error, n_identity, config.total_examples));

    let mut examples = Vec::with_capacity(config.total_examples);
    let mut correction_count = 0usize;
    let mut identity_count = 0usize;

    // Error examples: randomly sample from corpus pairs (with replacement)
    for _ in 0..n_error {
        let (original, parakeet, qwen) = &corpus_pairs[rng.random_range(0..corpus_pairs.len())];
        examples.push(serde_json::json!({
            "prompt": format!("<keet> {}\n<qwen> {}\n<fixd>", parakeet, qwen),
            "completion": format!(" {}<|endoftext|>", original),
        }));
        correction_count += 1;
    }

    // Identity examples: same shape, but both ASR lanes have the correct text
    for _ in 0..n_identity {
        let text = &originals[rng.random_range(0..originals.len())];
        examples.push(serde_json::json!({
            "prompt": format!("<keet> {}\n<qwen> {}\n<fixd>", text, text),
            "completion": format!(" {}<|endoftext|>", text),
        }));
        identity_count += 1;
    }

    examples.shuffle(&mut rng);

    // Fixed 90/10 train/valid split
    let n = examples.len();
    let n_train = (n as f64 * 0.9) as usize;
    let train = &examples[..n_train];
    let valid = &examples[n_train..];

    std::fs::create_dir_all(&config.output)?;
    write_jsonl(&format!("{}/train.jsonl", config.output), train)?;
    write_jsonl(&format!("{}/valid.jsonl", config.output), valid)?;

    let stats = PrepareStats {
        correction_examples: correction_count,
        identity_examples: identity_count,
        total: n,
        train_count: train.len(),
        valid_count: valid.len(),
    };

    on_status(&format!(
        "Wrote {} train + {} valid to {}",
        stats.train_count, stats.valid_count, config.output
    ));

    Ok(stats)
}

/// Run MLX-LM LoRA training (wraps uvx)
pub fn train(config: &TrainConfig) -> Result<std::process::ExitStatus> {
    let status = std::process::Command::new("uvx")
        .args([
            "--from",
            "mlx-lm",
            "mlx_lm.lora",
            "--model",
            &config.model,
            "--data",
            &config.data,
            "--train",
            "--iters",
            &config.iters.to_string(),
            "--batch-size",
            &config.batch_size.to_string(),
            "--num-layers",
            &config.num_layers.to_string(),
            "--adapter-path",
            &config.adapters,
            "--mask-prompt",
        ])
        .status()?;
    Ok(status)
}

/// Run MLX-LM LoRA training with streaming output. Each line from stderr
/// is passed to `on_line` for progress tracking.
pub fn train_streaming(
    config: &TrainConfig,
    mut on_line: impl FnMut(&str),
) -> Result<std::process::ExitStatus> {
    use std::io::BufRead;
    use std::process::{Command, Stdio};

    let mut child = Command::new("uvx")
        .args([
            "--from", "mlx-lm",
            "mlx_lm.lora",
            "--model", &config.model,
            "--data", &config.data,
            "--train",
            "--iters", &config.iters.to_string(),
            "--batch-size", &config.batch_size.to_string(),
            "--num-layers", &config.num_layers.to_string(),
            "--adapter-path", &config.adapters,
            "--steps-per-eval", &config.steps_per_eval.to_string(),
            "--mask-prompt",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    // MLX-LM writes progress bars to stderr (with \r) and training results to stdout.
    // Merge both streams and parse line-by-line, monitoring val loss for early stopping.
    let mut best_val_loss = f64::INFINITY;
    let mut patience_remaining = config.early_stop_patience;
    let mut should_kill = false;

    // Read both stdout and stderr on separate threads, funnel into one channel
    let (line_tx, line_rx) = std::sync::mpsc::channel::<String>();

    // Stderr reader (handles \r progress bars)
    if let Some(stderr) = child.stderr.take() {
        let tx = line_tx.clone();
        std::thread::spawn(move || {
            use std::io::Read;
            let mut reader = std::io::BufReader::new(stderr);
            let mut buf = Vec::new();
            let mut byte = [0u8; 1];
            while reader.read(&mut byte).unwrap_or(0) == 1 {
                match byte[0] {
                    b'\n' => {
                        let line = String::from_utf8_lossy(&buf).trim().to_string();
                        if !line.is_empty() { let _ = tx.send(line); }
                        buf.clear();
                    }
                    b'\r' => { buf.clear(); }
                    _ => buf.push(byte[0]),
                }
            }
            if !buf.is_empty() {
                let line = String::from_utf8_lossy(&buf).trim().to_string();
                if !line.is_empty() { let _ = tx.send(line); }
            }
        });
    }

    // Stdout reader (line-based, training output)
    if let Some(stdout) = child.stdout.take() {
        let tx = line_tx.clone();
        std::thread::spawn(move || {
            use std::io::BufRead;
            let reader = std::io::BufReader::new(stdout);
            for line in reader.lines() {
                let Ok(line) = line else { break };
                let line = line.trim().to_string();
                if !line.is_empty() { let _ = tx.send(line); }
            }
        });
    }

    drop(line_tx); // Close sender so receiver knows when both threads are done

    for line in line_rx {
        on_line(&line);

        // Check for val loss for early stopping
        if config.early_stop_patience > 0 {
            let lower = line.to_lowercase();
            if lower.contains("val") && lower.contains("loss") {
                if let Some(loss) = lower.split_whitespace()
                    .filter_map(|w| w.trim_matches(',').parse::<f64>().ok())
                    .next()
                {
                    if loss < best_val_loss {
                        best_val_loss = loss;
                        patience_remaining = config.early_stop_patience;
                    } else {
                        patience_remaining = patience_remaining.saturating_sub(1);
                        if patience_remaining == 0 {
                            on_line(&format!(
                                "Early stopping: val loss {loss:.4} did not improve from {best_val_loss:.4} for {} evals",
                                config.early_stop_patience
                            ));
                            let _ = child.kill();
                            should_kill = true;
                            break;
                        }
                    }
                }
            }
        }
    }

    let status = child.wait()?;
    Ok(status)
}

/// Configuration for inference with the trained correction model
pub struct InferenceConfig {
    pub model: String,
    pub adapters: String,
    pub max_tokens: usize,
    pub port: u16,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            model: "Qwen/Qwen2.5-0.5B".into(),
            adapters: "training/adapters".into(),
            max_tokens: 256,
            port: 8899,
        }
    }
}

/// Build the prompt for the correction model from ASR outputs
pub fn build_correction_prompt(parakeet: &str, qwen: &str) -> String {
    format!("<keet> {}\n<qwen> {}\n<fixd>", parakeet, qwen)
}

/// An inference server that keeps the model loaded. Wraps mlx_lm.server.
pub struct InferenceServer {
    child: std::process::Child,
    pub port: u16,
    pub max_tokens: usize,
}

impl InferenceServer {
    /// Start the mlx_lm.server subprocess. Blocks until the server is ready.
    pub fn start(config: &InferenceConfig) -> Result<Self> {
        use std::process::{Command, Stdio};

        eprintln!("[inference] Starting mlx_lm.server on port {}...", config.port);
        let child = Command::new("uvx")
            .args([
                "--from", "mlx-lm",
                "mlx_lm.server",
                "--model", &config.model,
                "--adapter-path", &config.adapters,
                "--port", &config.port.to_string(),
                "--max-tokens", &config.max_tokens.to_string(),
                "--temp", "0",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        // Wait for the server to be ready (poll /v1/models)
        let url = format!("http://127.0.0.1:{}/v1/models", config.port);
        let start = std::time::Instant::now();
        loop {
            if start.elapsed() > std::time::Duration::from_secs(120) {
                anyhow::bail!("mlx_lm.server failed to start within 120s");
            }
            if let Ok(resp) = ureq::get(&url).call() {
                if resp.status() == 200 { break; }
            }
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
        eprintln!("[inference] Server ready on port {}", config.port);

        Ok(Self { child, port: config.port, max_tokens: config.max_tokens })
    }

    /// Run inference on a single prompt. Returns the corrected text.
    /// Times out after 30 seconds to prevent hangs on degenerate inputs.
    pub fn infer(&self, prompt: &str) -> Result<String> {
        let url = format!("http://127.0.0.1:{}/v1/completions", self.port);
        let body = serde_json::json!({
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
        });

        let agent = ureq::Agent::config_builder()
            .timeout_global(Some(std::time::Duration::from_secs(30)))
            .build()
            .new_agent();

        let mut resp = agent.post(&url)
            .header("Content-Type", "application/json")
            .send_json(&body)
            .map_err(|e| anyhow::anyhow!("inference request failed: {e}"))?;

        let json: serde_json::Value = resp.body_mut().read_json()
            .map_err(|e| anyhow::anyhow!("inference response parse failed: {e}"))?;

        let text = json["choices"][0]["text"]
            .as_str()
            .unwrap_or("")
            .replace("<|endoftext|>", "")
            .replace("<|end_of_text|>", "");

        Ok(text.trim().to_string())
    }
}

impl Drop for InferenceServer {
    fn drop(&mut self) {
        eprintln!("[inference] Shutting down mlx_lm.server");
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub fn write_jsonl(path: &str, entries: &[serde_json::Value]) -> Result<()> {
    let mut f = std::fs::File::create(path)?;
    for entry in entries {
        serde_json::to_writer(&mut f, entry)?;
        f.write_all(b"\n")?;
    }
    f.flush()?;
    Ok(())
}
