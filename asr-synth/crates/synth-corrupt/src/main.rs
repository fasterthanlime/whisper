use anyhow::Result;
use clap::Parser;
use rand::prelude::*;
use rayon::prelude::*;
use std::sync::Arc;

mod cmudict;
mod corrupt;
mod features;
mod g2p;
mod markov;

#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    cmd: Cmd,

    /// Path to CMUdict file
    #[arg(long, default_value = "data/cmudict.txt", global = true)]
    dict: String,

    /// Path to Phonetisaurus FST model
    #[arg(long, default_value = "models/g2p.fst", global = true)]
    fst: String,

    /// Path to pronunciation overrides JSONL
    #[arg(long, default_value = "data/pronunciations.jsonl", global = true)]
    pronunciations: String,
}

#[derive(clap::Subcommand)]
enum Cmd {
    /// Find phoneme confusions for a single term
    Term {
        word: String,
        #[arg(long, default_value = "1.5")]
        max_cost: f32,
        #[arg(long, default_value = "10")]
        max_results: usize,
    },
    /// Batch-corrupt terms from a JSONL file
    Batch {
        input: String,
        #[arg(short, long)]
        output: Option<String>,
        #[arg(long, default_value = "1.0")]
        max_cost: f32,
        #[arg(long, default_value = "5")]
        max_results: usize,
    },
    /// Generate corrupted training corpus from conversation history
    Generate {
        /// Claude Code history.jsonl
        #[arg(long, default_value = "~/.claude/history.jsonl")]
        claude_history: String,
        /// Codex history.jsonl
        #[arg(long, default_value = "~/.codex/history.jsonl")]
        codex_history: String,
        /// Corruptions JSONL (from batch mode)
        #[arg(long, default_value = "data/corruptions_2500.jsonl")]
        corruptions: String,
        /// Number of sentences to generate
        #[arg(short, long, default_value = "1000")]
        count: usize,
        /// Output JSONL
        #[arg(short, long, default_value = "data/corpus.jsonl")]
        output: String,
        /// Probability of corrupting each eligible term (0.0-1.0)
        #[arg(long, default_value = "0.5")]
        corrupt_prob: f64,
    },
}

fn main() -> Result<()> {
    let args = Args::parse();

    eprintln!("Loading CMUdict...");
    let dict = Arc::new(cmudict::load(&args.dict)?);
    eprintln!("Loaded {} entries", dict.len());

    eprintln!("Loading G2P...");
    let g2p = Arc::new(g2p::G2p::load(&args.fst, Some(&args.pronunciations))?);
    eprintln!("G2P ready");

    eprintln!("Building phoneme index...");
    let index = Arc::new(corrupt::PhonemeIndex::new(&dict));
    eprintln!("Ready");

    match args.cmd {
        Cmd::Term { word, max_cost, max_results } => {
            let max_dist = (max_cost * 100.0) as usize;
            let phonemes = g2p.phonemize(&word, &dict);
            let ipa = g2p.ipa(&word);
            println!("\nTerm: {}  →  IPA: {}  ARPAbet: {}", word, ipa, phonemes.join(" "));
            let singles = index.find_single_word(&phonemes, max_dist, max_results);
            let doubles = index.find_two_word(&phonemes, max_dist, max_results);
            if !singles.is_empty() {
                println!("  Single-word:");
                for (w, d) in &singles { println!("    {:<25} (cost={:.2})", w, *d as f32 / 100.0); }
            }
            if !doubles.is_empty() {
                println!("  Two-word:");
                for (p, d) in &doubles { println!("    {:<30} (cost={:.2})", p, *d as f32 / 100.0); }
            }
        }
        Cmd::Batch { input, output, max_cost, max_results } => {
            let max_dist = (max_cost * 100.0) as usize;
            let content = std::fs::read_to_string(&input)?;
            let terms: Vec<String> = content.lines()
                .filter(|l| !l.trim().is_empty())
                .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok()
                    .and_then(|v| v["term"].as_str().map(String::from)))
                .collect();
            eprintln!("Processing {} terms...", terms.len());
            let results: Vec<_> = terms.par_iter().map(|term| {
                let phonemes = g2p.phonemize(term, &dict);
                let singles = index.find_single_word(&phonemes, max_dist, max_results);
                let doubles = index.find_two_word(&phonemes, max_dist, max_results);
                let mut confusions: Vec<(String, f32)> = Vec::new();
                for (w, d) in singles { confusions.push((w, d as f32 / 100.0)); }
                for (p, d) in doubles { confusions.push((p, d as f32 / 100.0)); }
                confusions.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                confusions.truncate(max_results);
                serde_json::json!({
                    "term": term,
                    "phonemes": phonemes.join(" "),
                    "confusions": confusions.iter().map(|(w, c)| serde_json::json!({"text": w, "cost": (*c * 100.0).round() / 100.0})).collect::<Vec<_>>(),
                })
            }).collect();
            let mut out: Box<dyn std::io::Write> = match &output {
                Some(p) => { std::fs::create_dir_all(std::path::Path::new(p).parent().unwrap_or(std::path::Path::new(".")))?; Box::new(std::fs::File::create(p)?) }
                None => Box::new(std::io::stdout()),
            };
            let mut with = 0;
            for r in &results {
                if !r["confusions"].as_array().unwrap().is_empty() { with += 1; }
                serde_json::to_writer(&mut out, r)?;
                std::io::Write::write_all(&mut out, b"\n")?;
            }
            eprintln!("Done: {with}/{} terms have confusions", results.len());
        }
        Cmd::Generate { claude_history, codex_history, corruptions, count, output, corrupt_prob } => {
            // Load corruption table
            let corruption_map = load_corruption_map(&corruptions)?;
            eprintln!("Loaded corruptions for {} terms", corruption_map.len());

            // Build markov chain from history
            let mut chain = markov::MarkovChain::new();
            for path in [&claude_history, &codex_history] {
                let expanded = shellexpand::tilde(path).to_string();
                if let Ok(content) = std::fs::read_to_string(&expanded) {
                    for line in content.lines() {
                        if let Ok(d) = serde_json::from_str::<serde_json::Value>(line) {
                            let text = d["display"].as_str()
                                .or_else(|| d["text"].as_str())
                                .unwrap_or("");
                            if text.len() > 20 && text.len() < 300
                                && !text.contains("[Pasted")
                                && !text.contains("[Image")
                                && !text.starts_with('/')
                            {
                                chain.feed(text);
                            }
                        }
                    }
                }
            }
            eprintln!("Markov chain: {} transitions", chain.transition_count());

            // Generate sentences with corruptions
            let mut rng = rand::rng();
            let mut out = std::fs::File::create(&output)?;
            let mut generated = 0;

            for _ in 0..count * 3 {
                if generated >= count { break; }

                let target_len = 12 + rng.random_range(0..10);
                let Some(sentence) = chain.generate(&mut rng, target_len) else {
                    continue;
                };

                // Generate two independent corruptions (simulating Parakeet + Qwen)
                let corrupt_once = |rng: &mut rand::rngs::ThreadRng| -> (String, Vec<serde_json::Value>) {
                    let mut corrupted = sentence.clone();
                    let mut applied = Vec::new();
                    for (term, confusions) in &corruption_map {
                        let lower_sent = corrupted.to_lowercase();
                        let lower_term = term.to_lowercase();
                        if lower_sent.contains(&lower_term) && rng.random_bool(corrupt_prob) {
                            if let Some(replacement) = confusions.choose(rng) {
                                if let Some(pos) = lower_sent.find(&lower_term) {
                                    corrupted = format!(
                                        "{}{}{}",
                                        &corrupted[..pos],
                                        replacement,
                                        &corrupted[pos + term.len()..]
                                    );
                                    applied.push(serde_json::json!({
                                        "term": term,
                                        "replacement": replacement,
                                    }));
                                }
                            }
                        }
                    }
                    (corrupted, applied)
                };

                let (parakeet, parakeet_applied) = corrupt_once(&mut rng);
                let (qwen, qwen_applied) = corrupt_once(&mut rng);

                let entry = serde_json::json!({
                    "original": sentence,
                    "parakeet": parakeet,
                    "qwen": qwen,
                    "parakeet_corruptions": parakeet_applied,
                    "qwen_corruptions": qwen_applied,
                });
                serde_json::to_writer(&mut out, &entry)?;
                std::io::Write::write_all(&mut out, b"\n")?;
                generated += 1;
            }

            eprintln!("Generated {generated} sentences to {output}");
        }
    }

    Ok(())
}

/// Load corruption map: term → vec of possible confusions
fn load_corruption_map(path: &str) -> Result<std::collections::HashMap<String, Vec<String>>> {
    let mut map = std::collections::HashMap::new();
    let content = std::fs::read_to_string(path)?;
    for line in content.lines() {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
            let term = v["term"].as_str().unwrap_or("").to_string();
            let confusions: Vec<String> = v["confusions"]
                .as_array()
                .map(|arr| arr.iter().filter_map(|c| c["text"].as_str().map(String::from)).collect())
                .unwrap_or_default();
            if !term.is_empty() && !confusions.is_empty() {
                map.insert(term, confusions);
            }
        }
    }
    Ok(map)
}
