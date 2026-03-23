use anyhow::Result;
use clap::Parser;
use rayon::prelude::*;
use std::sync::Arc;

mod cmudict;
mod corrupt;
mod features;
mod g2p;

#[derive(Parser)]
struct Args {
    /// Single term to find confusions for
    term: Option<String>,

    /// Batch mode: read terms from JSONL (expects {"term": "...", ...} per line)
    #[arg(long)]
    batch: Option<String>,

    /// Path to CMUdict file
    #[arg(long, default_value = "data/cmudict.txt")]
    dict: String,

    /// Path to Phonetisaurus FST model
    #[arg(long, default_value = "models/g2p.fst")]
    fst: String,

    /// Path to pronunciation overrides JSONL
    #[arg(long, default_value = "data/pronunciations.jsonl")]
    pronunciations: String,

    /// Max phoneme cost for matches (e.g. 1.5)
    #[arg(long, default_value = "1.5")]
    max_cost: f32,

    /// Max results per term (single + two-word combined)
    #[arg(long, default_value = "5")]
    max_results: usize,

    /// Output JSONL (batch mode only)
    #[arg(short, long)]
    output: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let max_dist = (args.max_cost * 100.0) as usize;

    eprintln!("Loading CMUdict from {}...", args.dict);
    let dict = Arc::new(cmudict::load(&args.dict)?);
    eprintln!("Loaded {} entries", dict.len());

    eprintln!("Loading G2P model from {}...", args.fst);
    let g2p = Arc::new(g2p::G2p::load(&args.fst, Some(&args.pronunciations))?);
    eprintln!("G2P ready");

    eprintln!("Building phoneme index...");
    let index = Arc::new(corrupt::PhonemeIndex::new(&dict));
    eprintln!("Index ready");

    if let Some(batch_path) = &args.batch {
        // Batch mode: read JSONL, process in parallel, output JSONL
        let content = std::fs::read_to_string(batch_path)?;
        let terms: Vec<String> = content
            .lines()
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| {
                serde_json::from_str::<serde_json::Value>(l)
                    .ok()
                    .and_then(|v| v["term"].as_str().map(String::from))
            })
            .collect();

        eprintln!("Processing {} terms with rayon...", terms.len());

        let results: Vec<_> = terms
            .par_iter()
            .map(|term| {
                let phonemes = g2p.phonemize(term, &dict);
                let singles = index.find_single_word(&phonemes, max_dist, args.max_results);
                let doubles = index.find_two_word(&phonemes, max_dist, args.max_results);

                let mut confusions: Vec<(String, f32)> = Vec::new();
                for (word, dist) in singles {
                    confusions.push((word, dist as f32 / 100.0));
                }
                for (phrase, dist) in doubles {
                    confusions.push((phrase, dist as f32 / 100.0));
                }
                confusions.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                confusions.truncate(args.max_results);

                serde_json::json!({
                    "term": term,
                    "phonemes": phonemes.join(" "),
                    "confusions": confusions.iter().map(|(w, c)| {
                        serde_json::json!({"text": w, "cost": (*c * 100.0).round() / 100.0})
                    }).collect::<Vec<_>>(),
                })
            })
            .collect();

        // Write output
        let mut out: Box<dyn std::io::Write> = if let Some(path) = &args.output {
            if let Some(parent) = std::path::Path::new(path).parent() {
                std::fs::create_dir_all(parent)?;
            }
            Box::new(std::fs::File::create(path)?)
        } else {
            Box::new(std::io::stdout())
        };

        let mut with_confusions = 0;
        for r in &results {
            let arr = r["confusions"].as_array().unwrap();
            if !arr.is_empty() {
                with_confusions += 1;
            }
            serde_json::to_writer(&mut out, r)?;
            out.write_all(b"\n")?;
        }
        eprintln!("Done: {}/{} terms have confusions", with_confusions, results.len());
    } else if let Some(term) = &args.term {
        show_confusions(term, &dict, &g2p, &index, max_dist, args.max_results);
    } else {
        // Demo mode
        let terms = [
            "serde", "tokio", "axum", "ratatui", "kajit", "reqwest",
            "facet", "clippy", "nextest", "backtraces", "minijinja",
            "bearcove", "fasterthanlime",
        ];
        for term in terms {
            show_confusions(term, &dict, &g2p, &index, max_dist, args.max_results);
        }
    }

    Ok(())
}

fn show_confusions(
    term: &str,
    dict: &cmudict::CmuDict,
    g2p: &g2p::G2p,
    index: &corrupt::PhonemeIndex,
    max_dist: usize,
    max_results: usize,
) {
    let phonemes = g2p.phonemize(term, dict);
    let ipa = g2p.ipa(term);
    println!("\n{}", "=".repeat(60));
    println!("Term: {}  →  IPA: {}  ARPAbet: {}", term, ipa, phonemes.join(" "));

    let singles = index.find_single_word(&phonemes, max_dist, max_results);
    let doubles = index.find_two_word(&phonemes, max_dist, max_results);

    if !singles.is_empty() {
        println!("  Single-word:");
        for (word, dist) in &singles {
            println!("    {:<25} (cost={:.2})", word, *dist as f32 / 100.0);
        }
    }
    if !doubles.is_empty() {
        println!("  Two-word:");
        for (phrase, dist) in &doubles {
            println!("    {:<30} (cost={:.2})", phrase, *dist as f32 / 100.0);
        }
    }
    if singles.is_empty() && doubles.is_empty() {
        println!("  (no confusions found)");
    }
}
