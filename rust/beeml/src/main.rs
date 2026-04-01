fn main() {
    let tokens = beeml_phonetic::parse_reviewed_ipa("s t a r t");
    eprintln!("Starting fresh aw yiss. Parsed {} phonemes.", tokens.len());
}
