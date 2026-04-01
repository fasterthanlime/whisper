use anyhow::Result;
use std::collections::HashSet;
use std::path::Path;

/// Returns true if a term is plausibly a dictatable word (not junk).
pub fn is_valid_vocab_term(term: &str) -> bool {
    if term.len() < 2 {
        return false;
    }
    if term.contains('/') || term.contains('=') || term.contains('_') {
        return false;
    }
    if term.chars().any(|c| !c.is_ascii()) {
        return false;
    }
    if term.starts_with(|c: char| c.is_ascii_digit()) {
        return false;
    }
    true
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct VocabEntry {
    /// Written form (target for correction model): `SHA-256`, `--doc`, `.asm`
    pub term: String,
    /// How a human would say it aloud: `shaw two fifty six`, `dash dash doc`, `dot asm`
    pub spoken: String,
}

/// Extract interesting technical terms from markdown files in <root>/*/docs/**/*.md
pub fn extract_vocab(root: &str) -> Result<Vec<VocabEntry>> {
    let mut terms = HashSet::new();

    // Walk <root>/*/docs/ looking for markdown files
    let root = Path::new(root);
    let entries = std::fs::read_dir(root)?;
    for entry in entries.flatten() {
        let docs_dir = entry.path().join("docs");
        if docs_dir.is_dir() {
            walk_md(&docs_dir, &mut terms)?;
        }
    }

    // Also add some well-known terms that are commonly misrecognized
    for term in SEED_VOCAB {
        terms.insert(term.to_string());
    }

    let mut vocab: Vec<VocabEntry> = terms
        .into_iter()
        .filter(|term| is_valid_vocab_term(term))
        .map(|term| {
            let spoken = to_spoken(&term);
            VocabEntry { term, spoken }
        })
        .collect();
    vocab.sort_by(|a, b| a.term.to_lowercase().cmp(&b.term.to_lowercase()));
    Ok(vocab)
}

fn walk_md(dir: &Path, terms: &mut HashSet<String>) -> Result<()> {
    for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk_md(&path, terms)?;
        } else if path.extension().is_some_and(|e| e == "md") {
            extract_from_file(&path, terms);
        }
    }
    Ok(())
}

fn extract_from_file(path: &Path, terms: &mut HashSet<String>) {
    let Ok(content) = std::fs::read_to_string(path) else {
        return;
    };

    use pulldown_cmark::{Event, Parser, Tag};

    let parser = Parser::new(&content);
    let mut in_code_block = false;

    for event in parser {
        match event {
            Event::Start(Tag::CodeBlock(_)) => in_code_block = true,
            Event::End(pulldown_cmark::TagEnd::CodeBlock) => in_code_block = false,

            // Inline code spans — the most valuable source of terms
            Event::Code(code) if !in_code_block => {
                let clean = code.trim();
                // Single token inline code like `serde`, `SHA-256`, `f32`
                // Skip multi-word code spans and paths
                if !clean.contains(' ') && !clean.contains('/') && !clean.contains('\\') {
                    let trimmed = clean.trim_matches(|c: char| !c.is_alphanumeric() && c != '-');
                    if is_interesting_term(trimmed) {
                        terms.insert(trimmed.to_string());
                    }
                }
            }

            // Prose text — look for CamelCase identifiers
            Event::Text(text) if !in_code_block => {
                for word in text.split_whitespace() {
                    let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
                    if is_interesting_term(clean) {
                        terms.insert(clean.to_string());
                    }
                }
            }

            _ => {}
        }
    }
}

/// A term is interesting if ASR is likely to mangle it.
/// We want: unusual words, acronyms, names with weird spellings.
/// We do NOT want: regular English words joined by underscores/hyphens.
fn is_interesting_term(s: &str) -> bool {
    if s.len() < 2 || s.len() > 30 {
        return false;
    }
    // Skip pure numbers
    if s.chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    // Must contain at least one letter
    if !s.chars().any(|c| c.is_alphabetic()) {
        return false;
    }
    // Skip common English words
    if STOP_WORDS.contains(&s.to_lowercase().as_str()) {
        return false;
    }
    // Skip paths, URLs, punctuation-heavy things
    if s.contains('/')
        || s.contains('\\')
        || s.contains('[')
        || s.contains(']')
        || s.contains('(')
        || s.contains(')')
        || s.contains('{')
        || s.contains('}')
        || s.contains('=')
        || s.contains('"')
        || s.contains('\'')
        || s.contains(',')
        || s.contains(';')
        || s.contains('<')
        || s.contains('>')
        || s.contains('|')
        || s.contains('#')
        || s.contains('@')
        || s.contains('$')
        || s.contains('%')
        || s.contains('+')
        || s.contains('_')
        || s.contains('.')
    {
        return false;
    }
    // Must be pronounceable-ish: at least 50% letters
    let letter_count = s.chars().filter(|c| c.is_alphabetic()).count();
    if letter_count * 2 < s.len() {
        return false;
    }

    // Skip CLI flags (--foo, --foo-bar)
    if s.starts_with("--")
        || s.starts_with('-')
            && s.chars().nth(1).is_some_and(|c| c.is_alphabetic())
            && s.chars().all(|c| c.is_alphanumeric() || c == '-')
            && s.chars().filter(|c| c.is_uppercase()).count() == 0
    {
        return false;
    }

    let has_upper = s.chars().any(|c| c.is_uppercase());
    let has_lower = s.chars().any(|c| c.is_lowercase());
    let has_digit = s.chars().any(|c| c.is_ascii_digit());
    let is_mixed_case = has_upper && has_lower;

    // Interesting if: mixed case (CamelCase), or digits mixed with letters
    // Hyphenated regular words like "channel-health" are not interesting —
    // ASR won't mangle them. Only keep hyphenated terms with digits (SHA-256).
    let has_interesting_hyphen = s.contains('-') && has_digit;

    is_mixed_case || has_digit || has_interesting_hyphen
}

/// Convert a written technical term to how a human would say it aloud.
///
/// Conservative: only transforms things that genuinely sound different from how
/// they're spelled. Does NOT lowercase — case changes aren't pronunciation changes.
///
/// Examples:
///   `SHA-256`    → `shaw two fifty six`  (override)
///   `serde`      → `ser dee`             (override)
///   `gRPC`       → `g r p c`            (override)
///   `CamelCase`  → `CamelCase`          (no change — pronounceable as-is)
///   `f32`        → `f thirty two`       (override)
///   `ESNext`     → `ESNext`             (no change)
pub fn to_spoken(term: &str) -> String {
    // Check overrides first — this is the primary mechanism
    if let Some(&spoken) = PRONUNCIATION_OVERRIDES.iter().find_map(|(k, v)| {
        if k.eq_ignore_ascii_case(term) {
            Some(v)
        } else {
            None
        }
    }) {
        return spoken.to_string();
    }

    // For everything else, return the term as-is.
    // The spoken form is the same as the written form unless there's an override.
    // Pronunciation corrections come from human review in the dashboard.
    term.to_string()
}

fn number_to_words(num: &str) -> String {
    // Simple number pronunciation for common cases
    match num {
        "0" => "zero".to_string(),
        "1" => "one".to_string(),
        "2" => "two".to_string(),
        "3" => "three".to_string(),
        "4" => "four".to_string(),
        "5" => "five".to_string(),
        "6" => "six".to_string(),
        "7" => "seven".to_string(),
        "8" => "eight".to_string(),
        "9" => "nine".to_string(),
        "10" => "ten".to_string(),
        "16" => "sixteen".to_string(),
        "32" => "thirty two".to_string(),
        "64" => "sixty four".to_string(),
        "128" => "one twenty eight".to_string(),
        "256" => "two fifty six".to_string(),
        "512" => "five twelve".to_string(),
        "1024" => "ten twenty four".to_string(),
        _ => {
            // Fall back to digit-by-digit for unknown numbers
            num.chars()
                .map(|c| match c {
                    '0' => "zero",
                    '1' => "one",
                    '2' => "two",
                    '3' => "three",
                    '4' => "four",
                    '5' => "five",
                    '6' => "six",
                    '7' => "seven",
                    '8' => "eight",
                    '9' => "nine",
                    _ => "",
                })
                .collect::<Vec<_>>()
                .join(" ")
        }
    }
}

/// Manual pronunciation overrides for terms where the spelling genuinely
/// doesn't match the pronunciation. Everything else gets added via the dashboard.
pub const PRONUNCIATION_OVERRIDES: &[(&str, &str)] =
    &[("serde", "sir day"), ("SQLite", "sequel light")];

const STOP_WORDS: &[&str] = &[
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were", "been", "have",
    "has", "had", "not", "but", "can", "will", "all", "each", "which", "their", "there", "when",
    "would", "make", "like", "just", "over", "such", "take", "also", "into", "than", "them",
    "very", "some", "could", "they", "other", "then", "its", "about", "use", "how", "any", "these",
    "may", "should", "does", "more", "most", "only", "what", "where", "why", "here", "still",
    "both", "between", "own", "under", "never", "being",
];

/// Well-known technical terms that are commonly misrecognized by ASR
const SEED_VOCAB: &[&str] = &[
    // Rust ecosystem
    "serde",
    "tokio",
    "axum",
    "hyper",
    "candle",
    "rubato",
    "rustfft",
    "wgpu",
    "naga",
    "ratatui",
    "clap",
    "anyhow",
    "thiserror",
    "tracing",
    "rayon",
    "crossbeam",
    "mio",
    "reqwest",
    "ureq",
    // ML/AI terms
    "GGUF",
    "GGML",
    "safetensors",
    "ONNX",
    "MLX",
    "LoRA",
    "QLoRA",
    // Tools & services
    "HuggingFace",
    "GitHub",
    "Xcode",
    "Homebrew",
    "ffmpeg",
    // General tech
    "WebSocket",
    "gRPC",
    "protobuf",
    "SQLite",
    "PostgreSQL",
    "OAuth",
    "JWT",
    "SHA-256",
    "Blake3",
];
