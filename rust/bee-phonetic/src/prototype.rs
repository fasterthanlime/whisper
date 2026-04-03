pub fn parse_reviewed_ipa(ipa_text: &str) -> Vec<String> {
    const MULTI_SYMBOL_PHONES: &[&str] = &[
        "t͡ʃ", "d͡ʒ", "tʃ", "dʒ", "eɪ", "aɪ", "ɔɪ", "aʊ", "oʊ", "əʊ", "ɪə", "eə", "ʊə",
    ];

    let mut out = Vec::new();
    let mut i = 0;

    while i < ipa_text.len() {
        let rest = &ipa_text[i..];
        let ch = rest
            .chars()
            .next()
            .expect("slice at char boundary should have a char");

        if is_ipa_separator(ch) {
            i += ch.len_utf8();
            continue;
        }

        if let Some(seq) = MULTI_SYMBOL_PHONES
            .iter()
            .find(|seq| rest.starts_with(**seq))
        {
            out.push((*seq).to_string());
            i += seq.len();
            continue;
        }

        let mut token = ch.to_string();
        i += ch.len_utf8();

        while i < ipa_text.len() {
            let next = ipa_text[i..]
                .chars()
                .next()
                .expect("slice at char boundary should have a char");
            if !is_ipa_modifier(next) {
                break;
            }
            token.push(next);
            i += next.len_utf8();
        }

        out.push(token);
    }

    out
}

pub fn phoneme_similarity(a: &[String], b: &[String]) -> Option<f32> {
    if a.is_empty() || b.is_empty() {
        return None;
    }

    let distance = levenshtein(a, b) as f32;
    let max_len = a.len().max(b.len()) as f32;
    let normalized = 1.0 - (distance / max_len);
    Some(normalized.clamp(0.0, 1.0))
}

fn levenshtein(a: &[String], b: &[String]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr = vec![0usize; b.len() + 1];

    for (i, ax) in a.iter().enumerate() {
        curr[0] = i + 1;
        let mut prev_left = i;
        for (j, by) in b.iter().enumerate() {
            let cost = usize::from(ax != by);
            curr[j + 1] = (curr[j] + 1).min(prev[j + 1] + 1).min(prev_left + cost);
            prev_left = prev[j + 1];
            prev[j + 1] = curr[j + 1];
        }
        prev.copy_from_slice(&curr);
    }

    prev[b.len()]
}

fn is_ipa_separator(ch: char) -> bool {
    ch.is_whitespace() || matches!(ch, 'ˈ' | 'ˌ' | '.')
}

fn is_ipa_modifier(ch: char) -> bool {
    matches!(ch, 'ː' | '˞' | 'ʰ' | 'ʲ' | 'ʷ' | '̃' | '̩' | '̯')
}

#[cfg(test)]
mod tests {
    use super::parse_reviewed_ipa;

    #[test]
    fn tokenizes_compact_reviewed_ipa_into_phones() {
        assert_eq!(parse_reviewed_ipa("sˈɜːdeɪ"), vec!["s", "ɜː", "d", "eɪ"]);
    }

    #[test]
    fn tokenizes_affricates_and_long_vowels() {
        assert_eq!(
            parse_reviewed_ipa("ˈeɪ ˈɑːtʃ sˈɪkstɪfə"),
            vec!["eɪ", "ɑː", "tʃ", "s", "ɪ", "k", "s", "t", "ɪ", "f", "ə"]
        );
    }

    #[test]
    fn tokenizes_espeak_probe_output_consistently() {
        assert_eq!(
            parse_reviewed_ipa("sˌɜː dˈe ɪ"),
            vec!["s", "ɜː", "d", "e", "ɪ"]
        );
    }
}
