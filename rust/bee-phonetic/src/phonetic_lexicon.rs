use std::collections::HashMap;

use facet::Facet;

use crate::types::{ReviewedConfusionSurfaceRow, VocabRow};
use crate::word_split::count_sentence_words;

pub use bee_types::{AliasSource, IdentifierFlags};

#[derive(Debug, Clone, Facet, PartialEq, Eq)]
pub struct LexiconAlias {
    pub alias_id: u32,
    pub term: String,
    pub alias_text: String,
    pub alias_source: AliasSource,
    pub ipa_tokens: Vec<String>,
    pub reduced_ipa_tokens: Vec<String>,
    pub feature_tokens: Vec<String>,
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

    let mut add_alias =
        |term: &str, alias_text: &str, alias_source: AliasSource, ipa_text: &str| {
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
            let token_count = count_sentence_words(alias_text).min(u8::MAX as usize) as u8;
            let phone_count = ipa_tokens.len().min(u8::MAX as usize) as u8;
            out.push(LexiconAlias {
                alias_id: next_alias_id,
                term: term.trim().to_string(),
                alias_text: alias_text.to_string(),
                alias_source,
                reduced_ipa_tokens: reduce_ipa_tokens(&ipa_tokens),
                feature_tokens: crate::feature_view::feature_tokens_for_ipa(&ipa_tokens),
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
        for alias_text in generate_identifier_aliases(&row.term, row.spoken()) {
            add_alias(
                &row.term,
                &alias_text,
                AliasSource::Identifier,
                reviewed_ipa,
            );
        }
        if let Some(forms) = confusion_forms.get(&row.term) {
            for form in forms {
                let Some(reviewed_ipa) = form.reviewed_ipa.as_deref() else {
                    continue;
                };
                add_alias(
                    &row.term,
                    &form.surface_form,
                    AliasSource::Confusion,
                    reviewed_ipa,
                );
            }
        }
    }

    out
}

fn generate_identifier_aliases(term: &str, spoken: &str) -> Vec<String> {
    let mut aliases = Vec::new();
    let mut seen = std::collections::HashSet::new();

    let mut push = |value: String| {
        let normalized = normalize_alias_text(&value);
        if normalized.is_empty() {
            return;
        }
        let lowered = normalized.to_ascii_lowercase();
        let term_lowered = term.trim().to_ascii_lowercase();
        let spoken_lowered = normalize_alias_text(spoken).to_ascii_lowercase();
        if lowered == term_lowered || lowered == spoken_lowered || !seen.insert(lowered) {
            return;
        }
        aliases.push(normalized);
    };

    push(spoken.replace(['-', '_', '/', '.'], " "));

    let parts = split_identifier_parts(term);
    if parts.len() >= 2 {
        let space_joined = parts
            .iter()
            .map(|part| normalize_identifier_part(part))
            .collect::<Vec<_>>()
            .join(" ");
        push(space_joined);

        let compact = parts
            .iter()
            .map(|part| normalize_identifier_part(part))
            .collect::<String>();
        push(compact);

        let cardinal = parts
            .iter()
            .map(|part| verbalize_identifier_part(part, NumberMode::Cardinal))
            .collect::<Vec<_>>()
            .join(" ");
        push(cardinal);

        let digits = parts
            .iter()
            .map(|part| verbalize_identifier_part(part, NumberMode::DigitByDigit))
            .collect::<Vec<_>>()
            .join(" ");
        push(digits);
    }

    aliases
}

#[derive(Clone, Copy)]
enum NumberMode {
    Cardinal,
    DigitByDigit,
}

fn normalize_alias_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn split_identifier_parts(text: &str) -> Vec<String> {
    let chars = text.chars().collect::<Vec<_>>();
    let mut parts = Vec::new();
    let mut current = String::new();

    for (idx, ch) in chars.iter().copied().enumerate() {
        if matches!(ch, '_' | '-' | '/' | '.' | ' ') {
            if !current.is_empty() {
                parts.push(std::mem::take(&mut current));
            }
            continue;
        }

        let should_split = if current.is_empty() {
            false
        } else {
            let prev = current.chars().last().unwrap();
            let next = chars.get(idx + 1).copied();
            identifier_boundary(prev, ch, next)
        };

        if should_split && !current.is_empty() {
            parts.push(std::mem::take(&mut current));
        }
        current.push(ch);
    }

    if !current.is_empty() {
        parts.push(current);
    }

    parts
}

fn identifier_boundary(prev: char, current: char, next: Option<char>) -> bool {
    if prev.is_ascii_digit() != current.is_ascii_digit() {
        return true;
    }
    if prev.is_ascii_lowercase() && current.is_ascii_uppercase() {
        return true;
    }
    if prev.is_ascii_uppercase()
        && current.is_ascii_uppercase()
        && next.is_some_and(|next| next.is_ascii_lowercase())
    {
        return true;
    }
    false
}

fn normalize_identifier_part(part: &str) -> String {
    if part.chars().all(|ch| ch.is_ascii_digit()) {
        part.to_string()
    } else {
        part.to_ascii_lowercase()
    }
}

fn verbalize_identifier_part(part: &str, number_mode: NumberMode) -> String {
    if part.chars().all(|ch| ch.is_ascii_digit()) {
        match number_mode {
            NumberMode::Cardinal => number_to_words(part).unwrap_or_else(|| {
                part.chars()
                    .filter_map(digit_word)
                    .collect::<Vec<_>>()
                    .join(" ")
            }),
            NumberMode::DigitByDigit => part
                .chars()
                .filter_map(digit_word)
                .collect::<Vec<_>>()
                .join(" "),
        }
    } else {
        part.to_ascii_lowercase()
    }
}

fn digit_word(ch: char) -> Option<&'static str> {
    Some(match ch {
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
        _ => return None,
    })
}

fn number_to_words(text: &str) -> Option<String> {
    let number = text.parse::<u32>().ok()?;
    if number > 9999 {
        return None;
    }

    fn under_100(n: u32) -> String {
        match n {
            0 => String::new(),
            1 => "one".to_string(),
            2 => "two".to_string(),
            3 => "three".to_string(),
            4 => "four".to_string(),
            5 => "five".to_string(),
            6 => "six".to_string(),
            7 => "seven".to_string(),
            8 => "eight".to_string(),
            9 => "nine".to_string(),
            10 => "ten".to_string(),
            11 => "eleven".to_string(),
            12 => "twelve".to_string(),
            13 => "thirteen".to_string(),
            14 => "fourteen".to_string(),
            15 => "fifteen".to_string(),
            16 => "sixteen".to_string(),
            17 => "seventeen".to_string(),
            18 => "eighteen".to_string(),
            19 => "nineteen".to_string(),
            20 => "twenty".to_string(),
            30 => "thirty".to_string(),
            40 => "forty".to_string(),
            50 => "fifty".to_string(),
            60 => "sixty".to_string(),
            70 => "seventy".to_string(),
            80 => "eighty".to_string(),
            90 => "ninety".to_string(),
            _ => {
                let tens = (n / 10) * 10;
                let ones = n % 10;
                format!("{} {}", under_100(tens), under_100(ones))
            }
        }
    }

    fn under_1000(n: u32) -> String {
        if n < 100 {
            return under_100(n);
        }
        let hundreds = n / 100;
        let remainder = n % 100;
        if remainder == 0 {
            format!("{} hundred", under_100(hundreds))
        } else {
            format!("{} hundred {}", under_100(hundreds), under_100(remainder))
        }
    }

    let words = if number < 1000 {
        under_1000(number)
    } else {
        let thousands = number / 1000;
        let remainder = number % 1000;
        if remainder == 0 {
            format!("{} thousand", under_100(thousands))
        } else {
            format!(
                "{} thousand {}",
                under_100(thousands),
                under_1000(remainder)
            )
        }
    };
    Some(words.trim().to_string())
}

pub fn reduce_ipa_tokens(tokens: &[String]) -> Vec<String> {
    tokens.iter().map(|token| reduce_ipa_token(token)).collect()
}

pub fn normalize_ipa_for_comparison(tokens: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    let reduced = reduce_ipa_tokens(tokens);
    let mut index = 0usize;
    while index < reduced.len() {
        let token = &reduced[index];
        let next = reduced.get(index + 1).map(String::as_str);
        if next == Some("ɹ") && is_vowel_like(token) {
            out.push(normalize_vowel_family(token));
            index += 2;
            continue;
        }

        match token.as_str() {
            "t͡ʃ" | "tʃ" => {
                out.push("t".to_string());
                out.push("ʃ".to_string());
            }
            "d͡ʒ" | "dʒ" => {
                out.push("d".to_string());
                out.push("ʒ".to_string());
            }
            "eɪ" => {
                out.push("ɛ".to_string());
                out.push("ɪ".to_string());
            }
            "aɪ" => {
                out.push("a".to_string());
                out.push("ɪ".to_string());
            }
            "ɔɪ" => {
                out.push("ɔ".to_string());
                out.push("ɪ".to_string());
            }
            "aʊ" => {
                out.push("a".to_string());
                out.push("ʊ".to_string());
            }
            "oʊ" | "əʊ" => {
                out.push("ə".to_string());
                out.push("ʊ".to_string());
            }
            "ɪə" => {
                out.push("ɪ".to_string());
                out.push("ə".to_string());
            }
            "eə" => {
                out.push("ɛ".to_string());
                out.push("ə".to_string());
            }
            "ʊə" => {
                out.push("ʊ".to_string());
                out.push("ə".to_string());
            }
            _ => out.push(normalize_vowel_family(token)),
        }
        index += 1;
    }
    out
}

fn is_vowel_like(token: &str) -> bool {
    matches!(
        token,
        "a"
            | "æ"
            | "ɑ"
            | "ɒ"
            | "ɔ"
            | "e"
            | "ɛ"
            | "ɜ"
            | "ɐ"
            | "ə"
            | "ʌ"
            | "ɘ"
            | "ɞ"
            | "i"
            | "ɪ"
            | "o"
            | "ɵ"
            | "ɤ"
            | "u"
            | "ʊ"
            | "ɯ"
            | "ʉ"
    )
}

fn normalize_vowel_family(token: &str) -> String {
    match token {
        "ɐ" | "ə" | "ʌ" | "ɜ" | "ɘ" | "ɞ" | "ᵻ" => "ə".to_string(),
        "ɑ" | "ɒ" => "ɑ".to_string(),
        "æ" | "ɛ" | "e" => "ɛ".to_string(),
        "ɔ" | "o" | "ɵ" | "ɤ" => "ɔ".to_string(),
        "i" | "ɪ" | "y" | "ʏ" => "ɪ".to_string(),
        "u" | "ʊ" | "ɯ" | "ʉ" => "ʊ".to_string(),
        _ => token.to_string(),
    }
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

pub fn is_identifier_like(flags: &IdentifierFlags) -> bool {
    flags.acronym_like
        || flags.has_digits
        || flags.snake_like
        || flags.camel_like
        || flags.symbol_like
}

pub fn looks_like_name(text: &str) -> bool {
    let mut parts = text.split_whitespace();
    let Some(token) = parts.next() else {
        return false;
    };
    if parts.next().is_some() {
        return false;
    }

    let mut chars = token.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !first.is_ascii_uppercase() {
        return false;
    }
    let mut saw_lower = false;
    for ch in chars {
        if !ch.is_ascii_lowercase() {
            return false;
        }
        saw_lower = true;
    }
    saw_lower
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

    fn confusion(
        term: &str,
        surface_form: &str,
        reviewed_ipa: &str,
    ) -> ReviewedConfusionSurfaceRow {
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
            vec![confusion(
                "AArch64",
                "ARC sixty four",
                "ɑːɹ s ɪ k s t i f ɔ ɹ",
            )],
        )]);

        let aliases = build_phonetic_lexicon(&vocab, &confusion_forms);
        assert_eq!(aliases.len(), 6, "{aliases:#?}");
        assert!(aliases
            .iter()
            .any(|a| a.alias_source == AliasSource::Canonical));
        assert!(aliases
            .iter()
            .any(|a| a.alias_source == AliasSource::Spoken));
        assert!(aliases
            .iter()
            .any(|a| a.alias_source == AliasSource::Identifier));
        assert!(aliases
            .iter()
            .any(|a| a.alias_source == AliasSource::Confusion));
    }

    #[test]
    fn generates_identifier_aliases_for_code_like_terms() {
        let aliases = generate_identifier_aliases("serde_json", "sirday jason");
        assert!(aliases.iter().any(|alias| alias == "serde json"));
        assert!(aliases.iter().any(|alias| alias == "serdejson"));

        let aliases = generate_identifier_aliases("AArch64", "A arch sixty-four");
        assert!(aliases.iter().any(|alias| alias == "a arch 64"));
        assert!(aliases.iter().any(|alias| alias == "a arch six four"));

        let aliases = generate_identifier_aliases("MachO", "mach oh");
        assert!(aliases.iter().any(|alias| alias == "mach o"));
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

    #[test]
    fn looks_like_name_detects_simple_proper_names() {
        assert!(looks_like_name("Quinn"));
        assert!(looks_like_name("Marco"));
        assert!(!looks_like_name("qwen"));
        assert!(!looks_like_name("MachO"));
        assert!(!looks_like_name("third day"));
    }

    #[test]
    fn reduce_ipa_tokens_strips_prosody_and_normalizes_rhotic_vowels() {
        assert_eq!(
            reduce_ipa_tokens(&[
                "ɚ".to_string(),
                "ɝ".to_string(),
                "oʊ".to_string(),
                "iː".to_string()
            ]),
            vec!["ə", "ə", "əʊ", "i"]
        );
    }

    #[test]
    fn normalize_ipa_for_comparison_splits_affricates_and_diphthongs() {
        assert_eq!(
            normalize_ipa_for_comparison(&[
                "tʃ".to_string(),
                "d͡ʒ".to_string(),
                "aɪ".to_string(),
                "eɪ".to_string(),
                "oʊ".to_string(),
                "əʊ".to_string(),
            ]),
            vec!["t", "ʃ", "d", "ʒ", "a", "ɪ", "ɛ", "ɪ", "ə", "ʊ", "ə", "ʊ"]
        );
    }

    #[test]
    fn normalize_ipa_for_comparison_matches_split_spellings() {
        assert_eq!(
            normalize_ipa_for_comparison(&["aɪ".to_string(), "tʃ".to_string(), "ɚ".to_string()]),
            normalize_ipa_for_comparison(&[
                "a".to_string(),
                "ɪ".to_string(),
                "t".to_string(),
                "ʃ".to_string(),
                "ə".to_string(),
            ])
        );
    }

    #[test]
    fn normalize_ipa_for_comparison_collapses_vowel_families() {
        assert_eq!(
            normalize_ipa_for_comparison(&[
                "ɐ".to_string(),
                "ʌ".to_string(),
                "ɒ".to_string(),
                "e".to_string(),
                "i".to_string(),
                "u".to_string(),
            ]),
            vec!["ə", "ə", "ɑ", "ɛ", "ɪ", "ʊ"]
        );
    }

    #[test]
    fn normalize_ipa_for_comparison_treats_postvocalic_r_as_rhoticity() {
        assert_eq!(
            normalize_ipa_for_comparison(&[
                "ə".to_string(),
                "ɹ".to_string(),
                "ɑ".to_string(),
                "ɹ".to_string(),
                "k".to_string(),
            ]),
            vec!["ə", "ɑ", "k"]
        );
        assert_eq!(
            normalize_ipa_for_comparison(&["ɚ".to_string(), "ɑ".to_string()]),
            vec!["ə", "ɑ"]
        );
    }
}
