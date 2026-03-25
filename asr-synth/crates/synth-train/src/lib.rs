use anyhow::Result;
use rand::prelude::*;
use std::io::Write;
use std::path::Path;

/// Stats returned from the prepare step
#[derive(Debug, serde::Serialize)]
pub struct PrepareStats {
    pub correction_examples: usize,
    pub identity_examples: usize,
    pub train_count: usize,
    pub valid_count: usize,
    pub test_count: usize,
}

pub struct PrepareConfig {
    pub input: String,
    pub output: String,
    pub identity_count: usize,
    pub claude_history: String,
    pub codex_history: String,
    pub train_ratio: f64,
}

impl Default for PrepareConfig {
    fn default() -> Self {
        Self {
            input: "data/corpus_5k.jsonl".into(),
            output: "training/data".into(),
            identity_count: 95000,
            claude_history: "~/.claude/history.jsonl".into(),
            codex_history: "~/.codex/history.jsonl".into(),
            train_ratio: 0.8,
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
            batch_size: 1,
            num_layers: 4,
            early_stop_patience: 3,
            steps_per_eval: 100,
        }
    }
}

/// Prepare training data from corpus JSONL → MLX-LM completions format
pub fn prepare(config: &PrepareConfig, mut on_status: impl FnMut(&str)) -> Result<PrepareStats> {
    let content = std::fs::read_to_string(&config.input)?;
    let lines: Vec<&str> = content.lines().filter(|l| !l.trim().is_empty()).collect();
    on_status(&format!("Read {} corpus entries from {}", lines.len(), config.input));

    let mut examples = Vec::new();
    let mut rng = rand::rng();

    for line in &lines {
        let v: serde_json::Value = serde_json::from_str(line)?;
        let original = v["original"].as_str().unwrap_or("");
        let parakeet = v["parakeet"].as_str().unwrap_or("");
        let qwen = v["qwen"].as_str().unwrap_or("");

        if original.is_empty() {
            continue;
        }

        let prompt = format!("<parakeet> {} <qwen> {} <correct>", parakeet, qwen);
        examples.push(serde_json::json!({
            "prompt": prompt,
            "completion": format!(" {}<|endoftext|>", original),
        }));
    }

    let n_corrections = examples.len();

    // Build markov chain from history + blog posts for identity examples
    let mut chain = synth_corrupt::markov::MarkovChain::new();
    let mut raw_texts: Vec<String> = Vec::new();

    for path in [&config.claude_history, &config.codex_history] {
        let expanded = shellexpand::tilde(path).to_string();
        if let Ok(content) = std::fs::read_to_string(&expanded) {
            for line in content.lines() {
                if let Ok(d) = serde_json::from_str::<serde_json::Value>(line) {
                    let text = d["display"]
                        .as_str()
                        .or_else(|| d["text"].as_str())
                        .unwrap_or("");
                    if text.len() >= 20
                        && text.len() <= 200
                        && !text.contains("[Pasted")
                        && !text.contains("[Image")
                        && !text.starts_with('/')
                    {
                        chain.feed(text);
                        raw_texts.push(text.to_string());
                    }
                }
            }
        }
    }

    // Blog posts from ~/bearcove/fasterthanli.me
    let blog_dir = shellexpand::tilde("~/bearcove/fasterthanli.me").to_string();
    let mut blog_paragraphs = 0usize;
    if let Ok(entries) = glob_md(&blog_dir) {
        for path in entries {
            if let Ok(content) = std::fs::read_to_string(&path) {
                let mut in_code_block = false;
                let mut current_para = String::new();

                let mut opts = pulldown_cmark::Options::empty();
                opts.insert(pulldown_cmark::Options::ENABLE_TABLES);
                for event in pulldown_cmark::Parser::new_ext(&content, opts) {
                    match event {
                        pulldown_cmark::Event::Start(pulldown_cmark::Tag::CodeBlock(_)) => {
                            in_code_block = true;
                        }
                        pulldown_cmark::Event::End(pulldown_cmark::TagEnd::CodeBlock) => {
                            in_code_block = false;
                        }
                        pulldown_cmark::Event::Text(text) if !in_code_block => {
                            current_para.push_str(&text);
                        }
                        pulldown_cmark::Event::SoftBreak
                        | pulldown_cmark::Event::HardBreak
                            if !in_code_block =>
                        {
                            current_para.push(' ');
                        }
                        // Table cells: add space between cells
                        pulldown_cmark::Event::End(pulldown_cmark::TagEnd::TableCell) => {
                            current_para.push(' ');
                        }
                        // Table rows: flush like paragraphs
                        pulldown_cmark::Event::End(
                            pulldown_cmark::TagEnd::TableHead | pulldown_cmark::TagEnd::TableRow,
                        ) => {
                            let clean = current_para.trim().to_string();
                            if clean.len() >= 30 && clean.len() <= 300 {
                                chain.feed(&clean);
                                blog_paragraphs += 1;
                            }
                            current_para.clear();
                        }
                        pulldown_cmark::Event::End(pulldown_cmark::TagEnd::Paragraph) => {
                            let clean = current_para.trim().to_string();
                            if clean.len() >= 30 && clean.len() <= 300 {
                                chain.feed(&clean);
                                blog_paragraphs += 1;
                            }
                            current_para.clear();
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    on_status(&format!("Blog: {} paragraphs from fasterthanli.me", blog_paragraphs));
    on_status(&format!(
        "Markov chain: {} transitions, {} raw texts",
        chain.transition_count(),
        raw_texts.len()
    ));

    let mut identity_generated = 0;
    for _ in 0..config.identity_count {
        let text = if !raw_texts.is_empty() && rng.random_bool(0.5) {
            raw_texts[rng.random_range(0..raw_texts.len())].clone()
        } else {
            let target_len: usize = 10 + rng.random_range(0..10);
            match chain.generate(&mut rng, target_len) {
                Some(t) => t,
                None => continue,
            }
        };

        let prompt = format!("<parakeet> {} <qwen> {} <correct>", text, text);
        examples.push(serde_json::json!({
            "prompt": prompt,
            "completion": format!(" {}<|endoftext|>", text),
        }));
        identity_generated += 1;
    }
    on_status(&format!("Generated {} identity examples", identity_generated));

    examples.shuffle(&mut rng);

    let n = examples.len();
    let n_train = (n as f64 * config.train_ratio) as usize;
    let n_remaining = n - n_train;
    let n_valid = n_remaining / 2;

    let train = &examples[..n_train];
    let valid = &examples[n_train..n_train + n_valid];
    let test = &examples[n_train + n_valid..];

    std::fs::create_dir_all(&config.output)?;
    write_jsonl(&format!("{}/train.jsonl", config.output), train)?;
    write_jsonl(&format!("{}/valid.jsonl", config.output), valid)?;
    write_jsonl(&format!("{}/test.jsonl", config.output), test)?;

    let stats = PrepareStats {
        correction_examples: n_corrections,
        identity_examples: identity_generated,
        train_count: train.len(),
        valid_count: valid.len(),
        test_count: test.len(),
    };

    on_status(&format!(
        "Wrote {} train, {} valid, {} test to {}",
        stats.train_count, stats.valid_count, stats.test_count, config.output
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

    // MLX-LM writes progress to stderr using \r for progress bars.
    // Read byte-by-byte and split on both \r and \n.
    // Monitor val loss for early stopping.
    let mut best_val_loss = f64::INFINITY;
    let mut patience_remaining = config.early_stop_patience;

    if let Some(stderr) = child.stderr.take() {
        use std::io::Read;
        let mut reader = std::io::BufReader::new(stderr);
        let mut buf = Vec::new();
        let mut byte = [0u8; 1];
        while reader.read(&mut byte).unwrap_or(0) == 1 {
            match byte[0] {
                b'\n' => {
                    let line = String::from_utf8_lossy(&buf).to_string();
                    let line = line.trim();
                    if !line.is_empty() {
                        on_line(line);

                        // Check for val loss: "Val loss 2.345, Val took 1.2s"
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
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    buf.clear();
                }
                b'\r' => {
                    buf.clear();
                }
                _ => buf.push(byte[0]),
            }
        }
        if !buf.is_empty() {
            let line = String::from_utf8_lossy(&buf).to_string();
            let line = line.trim();
            if !line.is_empty() { on_line(line); }
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
    format!("<parakeet> {} <qwen> {} <correct>", parakeet, qwen)
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
    pub fn infer(&self, prompt: &str) -> Result<String> {
        let url = format!("http://127.0.0.1:{}/v1/completions", self.port);
        let body = serde_json::json!({
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
        });

        let mut resp = ureq::post(&url)
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

pub fn glob_md(root: &str) -> Result<Vec<std::path::PathBuf>> {
    let mut results = Vec::new();
    fn walk(dir: &Path, results: &mut Vec<std::path::PathBuf>) {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    walk(&path, results);
                } else if path.extension().is_some_and(|e| e == "md") {
                    results.push(path);
                }
            }
        }
    }
    walk(Path::new(root), &mut results);
    Ok(results)
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
