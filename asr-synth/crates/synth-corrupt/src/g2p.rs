use crate::cmudict::CmuDict;
use anyhow::Result;
use phonetisaurus_g2p::PhonetisaurusModel;
use std::collections::HashMap;

/// G2P engine: overrides → CMUdict → acronym spelling → Phonetisaurus FST.
pub struct G2p {
    model: PhonetisaurusModel,
    overrides: HashMap<String, Vec<String>>,
}

impl G2p {
    pub fn load(fst_path: &str, overrides_path: Option<&str>) -> Result<Self> {
        let path = std::path::Path::new(fst_path);
        let model = PhonetisaurusModel::try_from(path)?;

        let mut overrides = HashMap::new();
        if let Some(path) = overrides_path {
            if let Ok(content) = std::fs::read_to_string(path) {
                for line in content.lines() {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                        if let (Some(term), Some(phonemes)) =
                            (v["term"].as_str(), v["phonemes"].as_str())
                        {
                            let ph: Vec<String> =
                                phonemes.split_whitespace().map(String::from).collect();
                            if !ph.is_empty() {
                                // Store both original case and uppercase for lookup
                                overrides.insert(term.to_string(), ph.clone());
                                overrides.insert(term.to_uppercase(), ph.clone());
                                overrides.insert(term.to_lowercase(), ph);
                            }
                        }
                    }
                }
                eprintln!("Loaded {} pronunciation overrides", overrides.len() / 3);
            }
        }

        Ok(Self { model, overrides })
    }

    /// Get ARPAbet phonemes for a word.
    /// Priority: overrides → CMUdict → acronym spelling → Phonetisaurus FST → crude fallback.
    pub fn phonemize(&self, word: &str, dict: &CmuDict) -> Vec<String> {
        let upper = word
            .to_uppercase()
            .trim_matches(|c: char| !c.is_alphanumeric())
            .to_string();

        // 1. Check pronunciation overrides first (highest priority)
        if let Some(phonemes) = self
            .overrides
            .get(word)
            .or_else(|| self.overrides.get(&upper))
            .or_else(|| self.overrides.get(&word.to_lowercase()))
        {
            return phonemes.clone();
        }

        // 2. Try CMUdict (authoritative for standard English)
        if let Some(phonemes) = dict.get(&upper) {
            return phonemes.clone();
        }

        // 3. Acronyms: all-caps, 2-5 letters → spell out letter by letter
        if upper.len() >= 2
            && upper.len() <= 5
            && upper.chars().all(|c| c.is_ascii_alphabetic())
            && word.chars().all(|c| c.is_uppercase() || !c.is_alphabetic())
        {
            let mut phonemes = Vec::new();
            for ch in upper.chars() {
                let letter = ch.to_string();
                if let Some(p) = dict.get(&letter) {
                    phonemes.extend(p.iter().cloned());
                }
            }
            if !phonemes.is_empty() {
                return phonemes;
            }
        }

        // Phonetisaurus FST for OOV words
        match self.model.phonemize_word(&upper.to_lowercase()) {
            Ok(result) => {
                // result.phonemes is IPA — convert to ARPAbet
                ipa_to_arpabet(&result.phonemes)
            }
            Err(_) => {
                // Last resort: crude letter mapping
                crude_g2p(&upper)
            }
        }
    }

    /// Get raw IPA for display.
    pub fn ipa(&self, word: &str) -> String {
        match self.model.phonemize_word(&word.to_lowercase()) {
            Ok(result) => result.phonemes,
            Err(_) => word.to_lowercase(),
        }
    }
}

/// Convert IPA string to ARPAbet phoneme sequence (approximate).
fn ipa_to_arpabet(ipa: &str) -> Vec<String> {
    let mut result = Vec::new();
    let chars: Vec<char> = ipa.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        // Try two-char IPA digraphs first
        if i + 1 < chars.len() {
            let digraph = format!("{}{}", chars[i], chars[i + 1]);
            let matched = match digraph.as_str() {
                "aɪ" => Some("AY"),
                "aʊ" => Some("AW"),
                "eɪ" => Some("EY"),
                "oʊ" => Some("OW"),
                "ɔɪ" => Some("OY"),
                "tʃ" => Some("CH"),
                "dʒ" => Some("JH"),
                _ => None,
            };
            if let Some(p) = matched {
                result.push(p.to_string());
                i += 2;
                continue;
            }
        }

        // Single-char IPA → ARPAbet
        let p = match chars[i] {
            'ɑ' | 'ɒ' => Some("AA"),
            'æ' => Some("AE"),
            'ʌ' | 'ə' => Some("AH"),
            'ɔ' => Some("AO"),
            'ɛ' | 'e' => Some("EH"),
            'ɝ' | 'ɜ' => Some("ER"),
            'ɪ' => Some("IH"),
            'i' => Some("IY"),
            'ʊ' => Some("UH"),
            'u' => Some("UW"),
            'a' => Some("AA"),
            'o' => Some("OW"),
            'b' => Some("B"),
            'd' => Some("D"),
            'f' => Some("F"),
            'ɡ' | 'g' => Some("G"),
            'h' => Some("HH"),
            'k' => Some("K"),
            'l' => Some("L"),
            'm' => Some("M"),
            'n' => Some("N"),
            'ŋ' => Some("NG"),
            'p' => Some("P"),
            'ɹ' | 'r' => Some("R"),
            's' => Some("S"),
            'ʃ' => Some("SH"),
            't' => Some("T"),
            'θ' => Some("TH"),
            'ð' => Some("DH"),
            'v' => Some("V"),
            'w' => Some("W"),
            'j' => Some("Y"),
            'z' => Some("Z"),
            'ʒ' => Some("ZH"),
            // Skip stress/prosody markers
            'ˈ' | 'ˌ' | '.' | ':' | 'ː' | ' ' | '|' => None,
            _ => None,
        };

        if let Some(p) = p {
            result.push(p.to_string());
        }
        i += 1;
    }

    result
}

/// Last-resort crude letter→phoneme mapping.
fn crude_g2p(word: &str) -> Vec<String> {
    let mut phonemes = Vec::new();
    for ch in word.chars() {
        match ch.to_ascii_uppercase() {
            'A' => phonemes.push("AE".into()),
            'B' => phonemes.push("B".into()),
            'C' => phonemes.push("K".into()),
            'D' => phonemes.push("D".into()),
            'E' => phonemes.push("EH".into()),
            'F' => phonemes.push("F".into()),
            'G' => phonemes.push("G".into()),
            'H' => phonemes.push("HH".into()),
            'I' => phonemes.push("IH".into()),
            'J' => phonemes.push("JH".into()),
            'K' => phonemes.push("K".into()),
            'L' => phonemes.push("L".into()),
            'M' => phonemes.push("M".into()),
            'N' => phonemes.push("N".into()),
            'O' => phonemes.push("AO".into()),
            'P' => phonemes.push("P".into()),
            'R' => phonemes.push("R".into()),
            'S' => phonemes.push("S".into()),
            'T' => phonemes.push("T".into()),
            'U' => phonemes.push("AH".into()),
            'V' => phonemes.push("V".into()),
            'W' => phonemes.push("W".into()),
            'X' => {
                phonemes.push("K".into());
                phonemes.push("S".into());
            }
            'Y' => phonemes.push("Y".into()),
            'Z' => phonemes.push("Z".into()),
            _ => {}
        }
    }
    phonemes
}
