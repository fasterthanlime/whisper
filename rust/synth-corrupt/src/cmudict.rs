use anyhow::Result;
use std::collections::HashMap;

/// Word → phoneme sequence (ARPAbet, stress stripped)
pub type CmuDict = HashMap<String, Vec<String>>;

pub fn load(path: &str) -> Result<CmuDict> {
    let bytes = std::fs::read(path)?;
    let content = String::from_utf8_lossy(&bytes);
    let mut dict = HashMap::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with(";;;") {
            continue;
        }
        let mut parts = line.split_whitespace();
        let word = match parts.next() {
            Some(w) => w.to_uppercase(),
            None => continue,
        };
        // Skip alternate pronunciations: WORD(2), WORD(3)
        if word.contains('(') {
            continue;
        }
        // Strip stress markers (0,1,2) from vowel phonemes
        let phonemes: Vec<String> = parts
            .map(|p| p.trim_end_matches(|c: char| c.is_ascii_digit()).to_string())
            .collect();
        if !phonemes.is_empty() {
            dict.insert(word, phonemes);
        }
    }
    Ok(dict)
}
