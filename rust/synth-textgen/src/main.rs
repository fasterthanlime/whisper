use anyhow::Result;
use clap::Parser;

mod corpus;
mod templates;

#[derive(Parser)]
struct Args {
    /// Directory to scan for markdown docs (glob: <dir>/*/docs/**/*.md)
    #[arg(short, long, default_value = "~/bearcove")]
    docs_root: String,

    /// Number of sentences to generate
    #[arg(short, long, default_value = "50")]
    count: usize,

    /// Print extracted vocabulary before generating
    #[arg(long)]
    show_vocab: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let root = shellexpand::tilde(&args.docs_root).to_string();

    eprintln!("Scanning {root}/*/docs/**/*.md for vocabulary...");
    let vocab = corpus::extract_vocab(&root)?;
    eprintln!("Extracted {} terms", vocab.len());

    if args.show_vocab {
        for entry in &vocab {
            println!("{}", entry.term);
        }
        return Ok(());
    }

    let sentences = templates::generate(&vocab, args.count, None, None);
    for s in &sentences {
        let json = serde_json::to_string(s)?;
        println!("{json}");
    }
    eprintln!("Generated {} sentences", sentences.len());
    Ok(())
}
