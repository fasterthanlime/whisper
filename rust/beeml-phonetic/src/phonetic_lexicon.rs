use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::types::{ReviewedConfusionSurfaceRow, VocabRow};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AliasSource {
    Canonical,
    Spoken,
    Confusion,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct IdentifierFlags {
    pub acronym_like: bool,
    pub has_digits: bool,
    pub snake_like: bool,
    pub camel_like: bool,
    pub symbol_like: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LexiconAlias {
    pub alias_id: u32,
    pub term: String,
    pub alias_text: String,
    pub alias_source: AliasSource,
    pub ipa_tokens: Vec<String>,
    pub reduced_ipa_tokens: Vec<String>,
    pub token_count: u8,
    pub phone_count: u8,
    pub identifier_flags: IdentifierFlags,
}

pub fn build_phonetic_lexicon(
    vocab: &[VocabRow],
    confusion_forms: &HashMap<String, Vec<ReviewedConfusionSurfaceRow>>,
) -> Vec<LexiconAlias> {
    let mut out = Vec::new();
    let mut next_alias_id = 0u32;
    let mut seen = std::collections::HashSet::new();

    let mut add_alias = |term: &str, alias_text: &str, alias_source: AliasSource, ipa_text: &str| {
        let alias_text = alias_text.trim();
        if alias_text.is_empty() {
            return;
        }
        let ipa_tokens = crate::prototype::parse_reviewed_ipa(ipa_text);
        if ipa_tokens.is_empty() {
            return;
        }
        let key = (
            term.trim().to_ascii_lowercase(),
            alias_text.to_ascii_lowercase(),
            alias_source,
            ipa_tokens.clone(),
        );
        if !seen.insert(key) {
            return;
        }
        let token_count = alias_text.split_whitespace().count().min(u8::MAX as usize) as u8;
        let phone_count = ipa_tokens.len().min(u8::MAX as usize) as u8;
        out.push(LexiconAlias {
            alias_id: next_alias_id,
            term: term.trim().to_string(),
            alias_text: alias_text.to_string(),
            alias_source,
            reduced_ipa_tokens: reduce_ipa_tokens(&ipa_tokens),
            ipa_tokens,
            token_count,
            phone_count,
            identifier_flags: derive_identifier_flags(alias_text),
        });
        next_alias_id += 1;
    };

    for row in vocab {
        let Some(reviewed_ipa) = row.reviewed_ipa.as_deref() else {
            continue;
        };
        add_alias(&row.term, &row.term, AliasSource::Canonical, reviewed_ipa);
        add_alias(&row.term, row.spoken(), AliasSource::Spoken, reviewed_ipa);
        if let Some(forms) = confusion_forms.get(&row.term) {
            for form in forms {
                let Some(reviewed_ipa) = form.reviewed_ipa.as_deref() else {
                    continue;
                };
                add_alias(&row.term, &form.surface_form, AliasSource::Confusion, reviewed_ipa);
            }
        }
    }

    out
}

pub fn reduce_ipa_tokens(tokens: &[String]) -> Vec<String> {
    tokens.iter().map(|token| reduce_ipa_token(token)).collect()
}

pub fn derive_identifier_flags(text: &str) -> IdentifierFlags {
    let mut acronym_like = false;
    let mut has_digits = false;
    let mut snake_like = false;
    let mut camel_like = false;
    let mut symbol_like = false;

    let has_lower = text.chars().any(|c| c.is_ascii_lowercase());
    let has_upper = text.chars().any(|c| c.is_ascii_uppercase());

    if text.contains('_') {
        snake_like = true;
        symbol_like = true;
    }
    if text.contains('-') || text.contains('/') || text.contains('.') {
        symbol_like = true;
    }
    if has_lower && has_upper {
        camel_like = true;
    }
    if text.chars().any(|c| c.is_ascii_digit()) {
        has_digits = true;
    }
    let letters_only: String = text.chars().filter(|c| c.is_ascii_alphabetic()).collect();
    if !letters_only.is_empty()
        && letters_only.len() <= 6
        && letters_only.chars().all(|c| c.is_ascii_uppercase())
    {
        acronym_like = true;
    }

    IdentifierFlags {
        acronym_like,
        has_digits,
        snake_like,
        camel_like,
        symbol_like,
    }
}

fn reduce_ipa_token(token: &str) -> String {
    let mut out = String::new();
    for ch in token.chars() {
        match ch {
            'ˈ' | 'ˌ' | 'ː' | '˞' | 'ʰ' | 'ʲ' | 'ʷ' | '̃' | '̩' | '̯' | '.' => {}
            _ => out.push(ch),
        }
    }
    match out.as_str() {
        "ɚ" | "ɝ" => "ə".to_string(),
        "oʊ" => "əʊ".to_string(),
        _ => out,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row_with_reviewed_ipa(term: &str, spoken: &str, reviewed_ipa: &str) -> VocabRow {
        VocabRow {
            id: 1,
            term: term.to_string(),
            spoken_auto: spoken.to_string(),
            spoken_override: Some(spoken.to_string()),
            reviewed_ipa: Some(reviewed_ipa.to_string()),
            reviewed: true,
            description: None,
        }
    }

    fn confusion(term: &str, surface_form: &str, reviewed_ipa: &str) -> ReviewedConfusionSurfaceRow {
        ReviewedConfusionSurfaceRow {
            id: 1,
            term: term.to_string(),
            surface_form: surface_form.to_string(),
            reviewed_ipa: Some(reviewed_ipa.to_string()),
            status: "reviewed".to_string(),
            source: Some("test".to_string()),
            created_at: String::new(),
            updated_at: String::new(),
        }
    }

    #[test]
    fn builds_aliases_from_vocab_and_confusions() {
        let vocab = vec![row_with_reviewed_ipa(
            "AArch64",
            "A arch sixty-four",
            "eɪ ɑː tʃ s ɪ k s t ɪ f ə",
        )];
        let confusion_forms = HashMap::from([(
            "AArch64".to_string(),
            vec![confusion("AArch64", "ARC sixty four", "ɑːɹ s ɪ k s t i f ɔ ɹ")],
        )]);

        let aliases = build_phonetic_lexicon(&vocab, &confusion_forms);
        assert_eq!(aliases.len(), 3, "{aliases:#?}");
        assert!(aliases.iter().any(|a| a.alias_source == AliasSource::Canonical));
        assert!(aliases.iter().any(|a| a.alias_source == AliasSource::Spoken));
        assert!(aliases.iter().any(|a| a.alias_source == AliasSource::Confusion));
    }

    #[test]
    fn identifier_flags_detect_code_like_terms() {
        let flags = derive_identifier_flags("serde_json");
        assert!(flags.snake_like);
        assert!(flags.symbol_like);
        assert!(!flags.acronym_like);

        let flags = derive_identifier_flags("QEMU");
        assert!(flags.acronym_like);
        assert!(!flags.snake_like);

        let flags = derive_identifier_flags("AArch64");
        assert!(flags.has_digits);
        assert!(flags.camel_like);
    }
}
