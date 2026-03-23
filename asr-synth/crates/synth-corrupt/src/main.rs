use anyhow::Result;
use clap::Parser;

mod cmudict;
mod corrupt;
mod features;
mod g2p;

#[derive(Parser)]
struct Args {
    /// Term to find confusions for (demo mode if omitted)
    term: Option<String>,

    /// Path to CMUdict file
    #[arg(long, default_value = "data/cmudict.txt")]
    dict: String,

    /// Path to Phonetisaurus FST model
    #[arg(long, default_value = "models/g2p.fst")]
    fst: String,

    /// Max phoneme edit distance (×100, so 200 = 2 old-style edits, 120 = ~1.2 weighted edits)
    #[arg(long, default_value = "200")]
    max_dist: usize,

    /// Max results per category
    #[arg(long, default_value = "10")]
    max_results: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    eprintln!("Loading CMUdict from {}...", args.dict);
    let dict = cmudict::load(&args.dict)?;
    eprintln!("Loaded {} entries", dict.len());

    eprintln!("Loading G2P model from {}...", args.fst);
    let g2p = g2p::G2p::load(&args.fst)?;
    eprintln!("G2P ready");

    eprintln!("Building phoneme index...");
    let index = corrupt::PhonemeIndex::new(&dict);
    eprintln!("Index built: {} length buckets", index.bucket_count());

    if let Some(term) = &args.term {
        show_confusions(term, &dict, &g2p, &index, &args);
    } else {
        let terms = [
            "serde", "tokio", "axum", "ratatui", "kajit", "reqwest",
            "facet", "clippy", "nextest", "backtraces", "minijinja",
            "bearcove", "fasterthanlime",
        ];
        for term in terms {
            show_confusions(term, &dict, &g2p, &index, &args);
        }
    }

    Ok(())
}

fn show_confusions(
    term: &str,
    dict: &cmudict::CmuDict,
    g2p: &g2p::G2p,
    index: &corrupt::PhonemeIndex,
    args: &Args,
) {
    let phonemes = g2p.phonemize(term, dict);
    let ipa = g2p.ipa(term);
    println!("\n{}", "=".repeat(60));
    println!("Term: {}  →  IPA: {}  ARPAbet: {}", term, ipa, phonemes.join(" "));

    let singles = index.find_single_word(&phonemes, args.max_dist, args.max_results);
    let doubles = index.find_two_word(&phonemes, 200, args.max_results);

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
