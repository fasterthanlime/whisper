use anyhow::Result;
use rand::prelude::*;
use std::io::Write;
use std::time::Instant;

pub mod adapter_import;
pub mod llm;
mod qwen2;
mod qwen3_5;

const MLX_LM_PACKAGE: &str = "mlx-lm==0.31.1";
const AGX_RELAX_CDM_CTXSTORE_TIMEOUT_ENV: &str = "AGX_RELAX_CDM_CTXSTORE_TIMEOUT";
const AGX_RELAX_CDM_CTXSTORE_TIMEOUT_VALUE: &str = "1";

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
            model: "mlx-community/Qwen3.5-2B-4bit".into(),
            iters: 2000,
            batch_size: 4,
            num_layers: 8,
            early_stop_patience: 10,
            steps_per_eval: 500,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
struct TrainingResourceSample {
    timestamp_ms: u64,
    launcher_pid: u32,
    process_tree_pids: Vec<u32>,
    launcher_rss_mb: f64,
    process_tree_rss_mb: f64,
    system_total_memory_mb: f64,
    system_used_memory_mb: f64,
    system_free_memory_mb: f64,
    system_available_memory_mb: f64,
    gpu_device_util_percent: Option<f64>,
    gpu_in_use_system_memory_mb: Option<f64>,
    gpu_alloc_system_memory_mb: Option<f64>,
    gpu_driver_system_memory_mb: Option<f64>,
}

fn mlx_uvx_command() -> std::process::Command {
    let mut command = std::process::Command::new("uvx");
    command.env(
        AGX_RELAX_CDM_CTXSTORE_TIMEOUT_ENV,
        AGX_RELAX_CDM_CTXSTORE_TIMEOUT_VALUE,
    );
    command
}

fn kb_to_mb(value_kb: u64) -> f64 {
    value_kb as f64 / 1024.0
}

fn bytes_to_mb(value_bytes: u64) -> f64 {
    value_bytes as f64 / (1024.0 * 1024.0)
}

fn process_tree_rss_kb(root_pid: u32) -> Result<(Vec<u32>, u64, u64)> {
    let output = std::process::Command::new("ps")
        .args(["-axo", "pid=,ppid=,rss="])
        .output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut snapshots = std::collections::HashMap::<u32, (u32, u64)>::new();
    for line in stdout.lines() {
        let mut parts = line.split_whitespace();
        let Some(pid) = parts.next().and_then(|v| v.parse::<u32>().ok()) else {
            continue;
        };
        let Some(ppid) = parts.next().and_then(|v| v.parse::<u32>().ok()) else {
            continue;
        };
        let Some(rss_kb) = parts.next().and_then(|v| v.parse::<u64>().ok()) else {
            continue;
        };
        snapshots.insert(pid, (ppid, rss_kb));
    }

    let mut queue = std::collections::VecDeque::from([root_pid]);
    let mut tree_pids = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut tree_rss_kb = 0u64;
    let mut launcher_rss_kb = 0u64;

    while let Some(pid) = queue.pop_front() {
        if !seen.insert(pid) {
            continue;
        }
        tree_pids.push(pid);
        if let Some((_, rss_kb)) = snapshots.get(&pid).copied() {
            tree_rss_kb += rss_kb;
            if pid == root_pid {
                launcher_rss_kb = rss_kb;
            }
        }
        for (&child_pid, &(ppid, _)) in &snapshots {
            if ppid == pid && !seen.contains(&child_pid) {
                queue.push_back(child_pid);
            }
        }
    }

    tree_pids.sort_unstable();
    Ok((tree_pids, launcher_rss_kb, tree_rss_kb))
}

fn performance_statistics_block(ioreg_output: &str) -> Option<&str> {
    let start = ioreg_output.find("\"PerformanceStatistics\"")?;
    let rest = &ioreg_output[start..];
    let open = rest.find('{')?;
    let stats = &rest[open + 1..];
    let close = stats.find('}')?;
    Some(&stats[..close])
}

fn extract_stat_number(stats: &str, key: &str) -> Option<f64> {
    let key_pos = stats.find(&format!("\"{key}\""))?;
    let rest = &stats[key_pos + key.len() + 2..];
    let eq = rest.find('=')?;
    let value = rest[eq + 1..].trim_start();
    let digits: String = value
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
        .collect();
    digits.parse::<f64>().ok()
}

fn sample_gpu_metrics() -> (Option<f64>, Option<f64>, Option<f64>, Option<f64>) {
    let output = std::process::Command::new("ioreg")
        .args(["-r", "-d", "1", "-c", "IOAccelerator"])
        .output();
    let Ok(output) = output else {
        return (None, None, None, None);
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_gpu_metrics(&stdout)
}

fn parse_gpu_metrics(ioreg_output: &str) -> (Option<f64>, Option<f64>, Option<f64>, Option<f64>) {
    let Some(stats) = performance_statistics_block(ioreg_output) else {
        return (None, None, None, None);
    };

    let device_util_percent = extract_stat_number(stats, "Device Utilization %");
    let in_use_mb =
        extract_stat_number(stats, "In use system memory").map(|bytes| bytes / (1024.0 * 1024.0));
    let alloc_mb =
        extract_stat_number(stats, "Alloc system memory").map(|bytes| bytes / (1024.0 * 1024.0));
    let driver_mb = extract_stat_number(stats, "In use system memory (driver)")
        .map(|bytes| bytes / (1024.0 * 1024.0));
    (device_util_percent, in_use_mb, alloc_mb, driver_mb)
}

fn sample_training_resources(root_pid: u32, system: &mut sysinfo::System) -> Result<String> {
    system.refresh_memory();
    let (tree_pids, launcher_rss_kb, tree_rss_kb) = process_tree_rss_kb(root_pid)?;
    let (
        gpu_device_util_percent,
        gpu_in_use_system_memory_mb,
        gpu_alloc_system_memory_mb,
        gpu_driver_system_memory_mb,
    ) = sample_gpu_metrics();

    let timestamp_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);

    let sample = TrainingResourceSample {
        timestamp_ms,
        launcher_pid: root_pid,
        process_tree_pids: tree_pids,
        launcher_rss_mb: kb_to_mb(launcher_rss_kb),
        process_tree_rss_mb: kb_to_mb(tree_rss_kb),
        system_total_memory_mb: bytes_to_mb(system.total_memory()),
        system_used_memory_mb: bytes_to_mb(system.used_memory()),
        system_free_memory_mb: bytes_to_mb(system.free_memory()),
        system_available_memory_mb: bytes_to_mb(system.available_memory()),
        gpu_device_util_percent,
        gpu_in_use_system_memory_mb,
        gpu_alloc_system_memory_mb,
        gpu_driver_system_memory_mb,
    };

    Ok(format!("RESOURCE {}", serde_json::to_string(&sample)?))
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
        if original.is_empty() {
            continue;
        }
        originals.push(original.clone());
        corpus_pairs.push((original, parakeet, qwen));
    }

    on_status(&format!(
        "Loaded {} corpus pairs, {} unique originals",
        corpus_pairs.len(),
        originals.len()
    ));

    if corpus_pairs.is_empty() {
        anyhow::bail!("No corpus pairs found in {}", config.input);
    }

    let mut rng = rand::rng();
    let n_error = (config.total_examples as f64 * config.error_rate).round() as usize;
    let n_identity = config.total_examples - n_error;

    on_status(&format!(
        "Generating {} error + {} identity = {} total examples",
        n_error, n_identity, config.total_examples
    ));

    let mut examples = Vec::with_capacity(config.total_examples);
    let mut correction_count = 0usize;
    let mut identity_count = 0usize;

    // Error examples: randomly sample from corpus pairs (with replacement)
    for _ in 0..n_error {
        let (original, _parakeet, qwen) = &corpus_pairs[rng.random_range(0..corpus_pairs.len())];
        let prompt = format!("<qwen> {}\n<fixd>", qwen);
        examples.push(serde_json::json!({
            "prompt": prompt,
            "completion": format!(" {}<|endoftext|>", original),
        }));
        correction_count += 1;
    }

    // Identity examples: same shape, but ASR output matches the correct text
    for _ in 0..n_identity {
        let text = &originals[rng.random_range(0..originals.len())];
        examples.push(serde_json::json!({
            "prompt": format!("<qwen> {}\n<fixd>", text),
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
    let mut command = mlx_uvx_command();
    let status = command
        .args([
            "--refresh",
            "--from",
            MLX_LM_PACKAGE,
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
    should_cancel: impl Fn() -> bool,
    mut on_line: impl FnMut(&str),
) -> Result<std::process::ExitStatus> {
    use std::process::Stdio;
    use std::sync::mpsc::RecvTimeoutError;
    use std::time::Duration;

    let mut command = mlx_uvx_command();
    let mut child = command
        .args([
            "--refresh",
            "--from",
            MLX_LM_PACKAGE,
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
            "--steps-per-eval",
            &config.steps_per_eval.to_string(),
            "--mask-prompt",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let mut system = sysinfo::System::new();
    let mut last_resource_sample = std::time::Instant::now() - Duration::from_secs(1);

    // MLX-LM writes progress bars to stderr (with \r) and training results to stdout.
    // Merge both streams and parse line-by-line, monitoring val loss for early stopping.
    let mut best_val_loss = f64::INFINITY;
    let mut patience_remaining = config.early_stop_patience;

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
                        if !line.is_empty() {
                            let _ = tx.send(line);
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
                let line = String::from_utf8_lossy(&buf).trim().to_string();
                if !line.is_empty() {
                    let _ = tx.send(line);
                }
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
                if !line.is_empty() {
                    let _ = tx.send(line);
                }
            }
        });
    }

    drop(line_tx); // Close sender so receiver knows when both threads are done

    loop {
        match line_rx.recv_timeout(Duration::from_millis(250)) {
            Ok(line) => {
                on_line(&line);

                // Check for val loss for early stopping
                if config.early_stop_patience > 0 {
                    let lower = line.to_lowercase();
                    if lower.contains("val") && lower.contains("loss") {
                        if let Some(loss) = lower
                            .split_whitespace()
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
            Err(RecvTimeoutError::Timeout) => {
                if last_resource_sample.elapsed() >= Duration::from_secs(1) {
                    last_resource_sample = std::time::Instant::now();
                    if let Ok(line) = sample_training_resources(child.id(), &mut system) {
                        on_line(&line);
                    }
                }
                if should_cancel() {
                    on_line("Stop requested. Terminating training process...");
                    let _ = child.kill();
                    break;
                }
                if let Some(_status) = child.try_wait()? {
                    break;
                }
            }
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }

    let status = child.wait()?;
    Ok(status)
}

/// Configuration for inference with the trained correction model
#[derive(Clone)]
pub struct InferenceConfig {
    pub model: String,
    pub adapters: String,
    pub attach_adapters: bool,
    pub max_tokens: usize,
    pub port: u16,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct InferenceScoreOutput {
    pub tokens: Vec<String>,
    pub token_ids: Vec<u32>,
    pub logits: Vec<f32>,
    pub probs: Vec<f32>,
    pub stats: InferenceStats,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            model: "mlx-community/Qwen3.5-2B-4bit".into(),
            adapters: "training/adapters".into(),
            attach_adapters: true,
            max_tokens: 64,
            port: 8899,
        }
    }
}

/// Build the prompt for the correction model from ASR outputs.
/// In dual mode: both Parakeet and Qwen lanes.
/// In single mode: only Qwen lane (pass parakeet as empty or None).
pub fn build_correction_prompt(_parakeet: &str, qwen: &str) -> String {
    format!(
        "<task> Rewrite the ASR sentence into corrected technical text.\n\
<rules> Keep the same sentence. Fix recognition mistakes only. Do not add or remove meaning. Output one corrected sentence only.\n\
<qwen> {}\n\
<fixd>",
        qwen.trim()
    )
}

fn normalized_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|token| {
            token
                .chars()
                .filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-' || *c == '\'')
                .collect::<String>()
                .to_ascii_lowercase()
        })
        .filter(|token| !token.is_empty())
        .collect()
}

fn shared_word_ratio(candidate: &str, source: &str) -> f32 {
    let candidate_words = normalized_words(candidate);
    if candidate_words.is_empty() {
        return 0.0;
    }
    let source_words: std::collections::HashSet<_> = normalized_words(source).into_iter().collect();
    let shared = candidate_words
        .iter()
        .filter(|word| source_words.contains(*word))
        .count();
    shared as f32 / candidate_words.len() as f32
}

fn has_excessive_char_run(text: &str) -> bool {
    let mut prev = '\0';
    let mut run = 0usize;
    for ch in text.chars() {
        if ch == prev {
            run += 1;
            if run >= 7 {
                return true;
            }
        } else {
            prev = ch;
            run = 1;
        }
    }
    false
}

fn has_repeated_token_run(text: &str) -> bool {
    let words = normalized_words(text);
    let mut prev = "";
    let mut run = 0usize;
    for word in &words {
        if word == prev {
            run += 1;
            if run >= 4 {
                return true;
            }
        } else {
            prev = word;
            run = 1;
        }
    }
    false
}

fn extract_qwen_from_prompt(prompt: &str) -> &str {
    prompt
        .lines()
        .find_map(|line| line.strip_prefix("<qwen> "))
        .unwrap_or("")
}

fn strip_explanatory_prefixes(mut text: String) -> String {
    loop {
        let trimmed = text.trim().to_string();
        let lower = trimmed.to_ascii_lowercase();
        let prefixes = [
            "the corrected sentence is:",
            "corrected sentence:",
            "corrected:",
            "correction:",
            "assistant:",
            "answer:",
        ];
        if let Some(prefix) = prefixes.iter().find(|prefix| lower.starts_with(**prefix)) {
            text = trimmed[prefix.len()..].trim().trim_matches('"').to_string();
            continue;
        }
        if lower.starts_with("the sentence is correct.")
            || lower.starts_with("the sentence is correct,")
        {
            if let Some(idx) = trimmed.find(':') {
                text = trimmed[idx + 1..].trim().trim_matches('"').to_string();
                continue;
            }
        }
        return trimmed.trim_matches('"').to_string();
    }
}

fn sanitize_correction_output(prompt: &str, raw: &str) -> String {
    let mut text = raw
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("")
        .to_string();
    text = strip_explanatory_prefixes(text);
    if let Some(idx) = text.find('\n') {
        text.truncate(idx);
    }
    text = text.trim().trim_matches('"').to_string();
    if text.is_empty() {
        return String::new();
    }
    if has_excessive_char_run(&text) || has_repeated_token_run(&text) {
        return String::new();
    }

    let qwen = extract_qwen_from_prompt(prompt);
    let output_words = normalized_words(&text);
    let qwen_words = normalized_words(qwen);
    if !output_words.is_empty() && !qwen_words.is_empty() {
        let ratio = shared_word_ratio(&text, qwen);
        if output_words.len() >= 3 && ratio < 0.34 {
            return String::new();
        }
        if output_words.len() > qwen_words.len().saturating_add(4) {
            return String::new();
        }
    }

    text
}

#[derive(Debug, Clone)]
struct CorrectionModelSpec {
    base_model: String,
    family: CorrectionModelFamily,
    gguf_repo: String,
    gguf_file: String,
    tokenizer_repo: String,
}

#[derive(Debug, Clone, Copy)]
enum CorrectionModelFamily {
    Qwen2,
    Qwen3_5,
}

fn load_adapter_base_model(adapters_dir: &str) -> Option<String> {
    let path = std::path::Path::new(adapters_dir).join("adapter_config.json");
    let text = std::fs::read_to_string(path).ok()?;
    let cfg: adapter_import::MlxAdapterConfig = serde_json::from_str(&text).ok()?;
    Some(cfg.model)
}

fn resolve_correction_model(config: &InferenceConfig) -> Result<CorrectionModelSpec> {
    let base_model =
        load_adapter_base_model(&config.adapters).unwrap_or_else(|| config.model.clone());
    let base_model = match base_model.as_str() {
        "mlx-community/Qwen3.5-0.8B-4bit" => "Qwen/Qwen3.5-0.8B".to_string(),
        "mlx-community/Qwen3.5-2B-4bit" => "Qwen/Qwen3.5-2B".to_string(),
        other => other.to_string(),
    };
    match base_model.as_str() {
        "Qwen/Qwen2.5-0.5B" => Ok(CorrectionModelSpec {
            base_model,
            family: CorrectionModelFamily::Qwen2,
            gguf_repo: "QuantFactory/Qwen2.5-0.5B-GGUF".into(),
            gguf_file: "Qwen2.5-0.5B.Q4_K_M.gguf".into(),
            tokenizer_repo: "Qwen/Qwen2.5-0.5B".into(),
        }),
        "Qwen/Qwen2.5-0.5B-Instruct" => Ok(CorrectionModelSpec {
            base_model,
            family: CorrectionModelFamily::Qwen2,
            gguf_repo: "Qwen/Qwen2.5-0.5B-Instruct-GGUF".into(),
            gguf_file: "qwen2.5-0.5b-instruct-q4_k_m.gguf".into(),
            tokenizer_repo: "Qwen/Qwen2.5-0.5B-Instruct".into(),
        }),
        "Qwen/Qwen3.5-0.8B" => Ok(CorrectionModelSpec {
            base_model,
            family: CorrectionModelFamily::Qwen3_5,
            gguf_repo: "unsloth/Qwen3.5-0.8B-GGUF".into(),
            gguf_file: "Qwen3.5-0.8B-Q4_K_M.gguf".into(),
            tokenizer_repo: "Qwen/Qwen3.5-0.8B".into(),
        }),
        "Qwen/Qwen3.5-2B" => Ok(CorrectionModelSpec {
            base_model,
            family: CorrectionModelFamily::Qwen3_5,
            gguf_repo: "unsloth/Qwen3.5-2B-GGUF".into(),
            gguf_file: "Qwen3.5-2B-Q4_K_M.gguf".into(),
            tokenizer_repo: "Qwen/Qwen3.5-2B".into(),
        }),
        "Qwen/Qwen3-Reranker-0.6B" => Ok(CorrectionModelSpec {
            base_model,
            family: CorrectionModelFamily::Qwen3_5,
            gguf_repo: "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF".into(),
            gguf_file: "qwen3-reranker-0.6b-q8_0.gguf".into(),
            tokenizer_repo: "Qwen/Qwen3-Reranker-0.6B".into(),
        }),
        other => anyhow::bail!("unsupported correction base model: {other}"),
    }
}

pub fn resolved_correction_base_model(config: &InferenceConfig) -> Result<String> {
    Ok(resolve_correction_model(config)?.base_model)
}

/// An in-process correction model that keeps the GGUF weights loaded in Rust.
pub struct InferenceServer {
    model_runtime: llm::CorrectionModel,
    pub port: u16,
    pub max_tokens: usize,
    pub model: String,
    pub adapters: String,
    pub attach_adapters: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct InferenceStats {
    pub prompt_tokens: usize,
    pub output_tokens: usize,
    pub encode_ms: u64,
    pub prefill_ms: u64,
    pub decode_ms: u64,
    pub generate_ms: u64,
    pub total_ms: u64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct InferenceOutput {
    pub text: String,
    pub raw_text: String,
    pub stats: InferenceStats,
}

impl InferenceServer {
    /// Load the correction model and attach the current MLX LoRA adapters.
    pub fn start(config: &InferenceConfig) -> Result<Self> {
        let spec = resolve_correction_model(config)?;
        eprintln!(
            "[inference] Loading in-process correction model {} from {}/{}...",
            spec.base_model, spec.gguf_repo, spec.gguf_file
        );
        let model_runtime = match spec.family {
            CorrectionModelFamily::Qwen2 => {
                let model = if config.attach_adapters {
                    llm::Qwen2Model::load_with_mlx_adapters(
                        &spec.gguf_repo,
                        &spec.gguf_file,
                        &spec.tokenizer_repo,
                        &config.adapters,
                    )?
                } else {
                    llm::Qwen2Model::load(&spec.gguf_repo, &spec.gguf_file, &spec.tokenizer_repo)?
                };
                llm::CorrectionModel::Qwen2(model)
            }
            CorrectionModelFamily::Qwen3_5 => {
                let model = if config.attach_adapters {
                    llm::Qwen3_5Model::load_with_mlx_adapters(
                        &spec.gguf_repo,
                        &spec.gguf_file,
                        &spec.tokenizer_repo,
                        &config.adapters,
                    )?
                } else {
                    llm::Qwen3_5Model::load(&spec.gguf_repo, &spec.gguf_file, &spec.tokenizer_repo)?
                };
                llm::CorrectionModel::Qwen3_5(model)
            }
        };
        eprintln!("[inference] Correction model ready");

        Ok(Self {
            model_runtime,
            port: config.port,
            max_tokens: config.max_tokens,
            model: spec.base_model,
            adapters: config.adapters.clone(),
            attach_adapters: config.attach_adapters,
        })
    }

    fn infer_tokens(&mut self, prompt_tokens: &[u32]) -> Result<InferenceOutput> {
        let started = Instant::now();
        let generate_started = Instant::now();
        let generation = self
            .model_runtime
            .generate_with_stats(prompt_tokens, self.max_tokens, 0.0, 0)?;
        let generate_elapsed = generate_started.elapsed();
        let raw_text = generation.text.trim().to_string();
        Ok(InferenceOutput {
            text: raw_text.clone(),
            raw_text,
            stats: InferenceStats {
                prompt_tokens: generation.prompt_tokens,
                output_tokens: generation.output_tokens,
                encode_ms: 0,
                prefill_ms: generation.prefill_ms,
                decode_ms: generation.decode_ms,
                generate_ms: generate_elapsed.as_millis() as u64,
                total_ms: started.elapsed().as_millis() as u64,
            },
        })
    }

    /// Run inference on a single prompt. Returns the corrected text.
    pub fn infer(&mut self, prompt: &str) -> Result<String> {
        Ok(self.infer_with_stats(prompt)?.text)
    }

    pub fn infer_chat_with_stats(&mut self, system: &str, user: &str) -> Result<InferenceOutput> {
        let started = Instant::now();
        let encode_started = Instant::now();
        let prompt_tokens = self.model_runtime.encode_chat(system, user)?;
        let encode_elapsed = encode_started.elapsed();
        let mut output = self.infer_tokens(&prompt_tokens)?;
        output.stats.encode_ms = encode_elapsed.as_millis() as u64;
        output.stats.total_ms = started.elapsed().as_millis() as u64;
        Ok(output)
    }

    pub fn infer_with_stats(&mut self, prompt: &str) -> Result<InferenceOutput> {
        let started = Instant::now();
        let encode_started = Instant::now();
        let prompt_tokens = self.model_runtime.encode_text(prompt)?;
        let encode_elapsed = encode_started.elapsed();
        let mut output = self.infer_tokens(&prompt_tokens)?;
        let mut text = output.text;

        for marker in [
            "<|endoftext|>",
            "<|end_of_text|>",
            "<fixd>",
            "<qwen>",
            "<keet>",
        ] {
            if let Some(idx) = text.find(marker) {
                text.truncate(idx);
            }
        }

        let raw_text = text.trim().to_string();
        let sanitized = sanitize_correction_output(prompt, &raw_text);
        output.stats.encode_ms = encode_elapsed.as_millis() as u64;
        output.stats.total_ms = started.elapsed().as_millis() as u64;
        eprintln!(
            "[inference] infer done: prompt_tokens={} encode_ms={} prefill_ms={} decode_ms={} generate_ms={} total_ms={}",
            output.stats.prompt_tokens,
            output.stats.encode_ms,
            output.stats.prefill_ms,
            output.stats.decode_ms,
            output.stats.generate_ms,
            output.stats.total_ms,
        );
        Ok(InferenceOutput {
            text: sanitized,
            raw_text,
            stats: output.stats,
        })
    }

    pub fn score_next_tokens_with_stats(
        &mut self,
        prompt: &str,
        add_special_tokens: bool,
        candidate_tokens: &[&str],
    ) -> Result<InferenceScoreOutput> {
        let started = Instant::now();
        let encode_started = Instant::now();
        let prompt_tokens = self
            .model_runtime
            .encode_text_with_special_tokens(prompt, add_special_tokens)?;
        let encode_elapsed = encode_started.elapsed();
        let token_ids = candidate_tokens
            .iter()
            .map(|token| {
                self.model_runtime
                    .token_id(token)
                    .ok_or_else(|| anyhow::anyhow!("tokenizer has no single token for {token:?}"))
            })
            .collect::<Result<Vec<_>>>()?;
        let scored = self
            .model_runtime
            .score_next_tokens(&prompt_tokens, &token_ids)?;
        Ok(InferenceScoreOutput {
            tokens: candidate_tokens.iter().map(|t| (*t).to_string()).collect(),
            token_ids,
            logits: scored.logits,
            probs: scored.probs,
            stats: InferenceStats {
                prompt_tokens: scored.prompt_tokens,
                output_tokens: 0,
                encode_ms: encode_elapsed.as_millis() as u64,
                prefill_ms: scored.prefill_ms,
                decode_ms: 0,
                generate_ms: scored.prefill_ms,
                total_ms: started.elapsed().as_millis() as u64,
            },
        })
    }
}

impl InferenceServer {
    pub fn kill(&mut self) {
        eprintln!("[inference] Releasing in-process correction model");
    }

    pub fn matches(&self, config: &InferenceConfig) -> bool {
        let resolved_model = resolve_correction_model(config)
            .map(|spec| spec.base_model)
            .unwrap_or_else(|_| config.model.clone());
        self.port == config.port
            && self.max_tokens == config.max_tokens
            && self.model == resolved_model
            && self.adapters == config.adapters
            && self.attach_adapters == config.attach_adapters
    }
}

impl Drop for InferenceServer {
    fn drop(&mut self) {
        eprintln!("[inference] Releasing in-process correction model");
    }
}

// ==================== Sentence Generator ====================

/// Configuration for the local LLM sentence generator.
pub struct SentenceGeneratorConfig {
    /// HuggingFace repo containing the GGUF model file.
    pub gguf_repo: String,
    /// GGUF filename within the repo.
    pub gguf_file: String,
    /// HuggingFace repo containing tokenizer.json (usually the base model repo).
    pub tokenizer_repo: String,
    pub max_tokens: usize,
    pub temperature: f32,
}

impl Default for SentenceGeneratorConfig {
    fn default() -> Self {
        Self {
            gguf_repo: "Qwen/Qwen2.5-1.5B-Instruct-GGUF".into(),
            gguf_file: "qwen2.5-1.5b-instruct-q4_k_m.gguf".into(),
            tokenizer_repo: "Qwen/Qwen2.5-1.5B-Instruct".into(),
            max_tokens: 128,
            temperature: 0.9,
        }
    }
}

/// In-process LLM for generating natural sentences containing vocab terms.
/// Uses a quantized Qwen2 model via candle (no Python subprocess).
pub struct SentenceGenerator {
    model: llm::Qwen2Model,
    max_tokens: usize,
    temperature: f32,
    rng_seed: u64,
}

impl SentenceGenerator {
    /// Load the model. Downloads from HuggingFace Hub on first use.
    pub fn start(config: &SentenceGeneratorConfig) -> Result<Self> {
        eprintln!(
            "[sentgen] Loading {} (in-process candle inference)...",
            config.gguf_file
        );
        let model =
            llm::Qwen2Model::load(&config.gguf_repo, &config.gguf_file, &config.tokenizer_repo)?;
        eprintln!("[sentgen] Ready.");

        Ok(Self {
            model,
            max_tokens: config.max_tokens,
            temperature: config.temperature,
            rng_seed: 42,
        })
    }

    /// Generate a single sentence containing the given term.
    /// Returns None if the model's output doesn't pass validation.
    pub fn generate_sentence(
        &mut self,
        term: &str,
        description: Option<&str>,
    ) -> Result<Option<String>> {
        let context = description.unwrap_or("a programming/tech term");
        let system = "You output a single English sentence. No explanations, no quotes, no prefixes \u{2014} just the sentence itself.";
        let user = format!(
            "Generate a natural sentence (8-20 words) that a developer might say in a podcast or code review. \
             It must contain the exact token: {term}\n\
             Context: {term} is {context}.\n\
             Just the sentence:"
        );

        let prompt_tokens = self.model.encode_chat(system, &user)?;

        // Vary seed per call for diversity
        self.rng_seed = self.rng_seed.wrapping_add(1);

        let text = self.model.generate(
            &prompt_tokens,
            self.max_tokens,
            self.temperature as f64,
            self.rng_seed,
        )?;

        let text = text.trim().trim_matches('"').to_string();

        // Validate output
        let lower = text.to_lowercase();
        if !lower.contains(&term.to_lowercase()) {
            return Ok(None);
        }
        if lower.contains("write one sentence")
            || lower.contains("generate a")
            || lower.starts_with("sure")
            || lower.starts_with("here")
        {
            return Ok(None);
        }
        let word_count = text.split_whitespace().count();
        if word_count < 5 || word_count > 30 {
            return Ok(None);
        }
        Ok(Some(text))
    }

    /// Generate a sentence, retrying up to `max_attempts` if validation fails.
    pub fn generate_sentence_retry(
        &mut self,
        term: &str,
        description: Option<&str>,
        max_attempts: usize,
    ) -> Result<Option<String>> {
        for _ in 0..max_attempts {
            match self.generate_sentence(term, description)? {
                Some(s) => return Ok(Some(s)),
                None => continue,
            }
        }
        Ok(None)
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

#[cfg(test)]
mod tests {
    use super::{InferenceConfig, parse_gpu_metrics, resolve_correction_model};

    #[test]
    fn parses_ioreg_performance_statistics() {
        let sample = r#"
            "PerformanceStatistics" = {"In use system memory (driver)"=1048576,"Alloc system memory"=8589934592,"In use system memory"=2147483648,"Device Utilization %"=67}
        "#;
        let (util, in_use, alloc, driver) = parse_gpu_metrics(sample);
        assert_eq!(util, Some(67.0));
        assert_eq!(in_use, Some(2048.0));
        assert_eq!(alloc, Some(8192.0));
        assert_eq!(driver, Some(1.0));
    }

    #[test]
    fn resolves_qwen3_5_correction_models() {
        let config = InferenceConfig {
            model: "Qwen/Qwen3.5-2B".into(),
            adapters: "/definitely/missing".into(),
            attach_adapters: true,
            max_tokens: 64,
            port: 8899,
        };
        let spec = resolve_correction_model(&config).expect("Qwen3.5-2B should resolve");
        assert_eq!(spec.base_model, "Qwen/Qwen3.5-2B");
        assert_eq!(spec.gguf_repo, "unsloth/Qwen3.5-2B-GGUF");
        assert_eq!(spec.gguf_file, "Qwen3.5-2B-Q4_K_M.gguf");
        assert_eq!(spec.tokenizer_repo, "Qwen/Qwen3.5-2B");
    }
}
