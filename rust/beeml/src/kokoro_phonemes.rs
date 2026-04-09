const KOKORO_ALLOWED_PHONEME_CHARS: &str = " !\"$',.:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz¡«»¿æçðøħŋœǀǁǂǃɐɑɒɓɔɕɖɗɘəɚɛɜɝɞɟɠɡɢɣɤɥɦɧɨɪɫɬɭɮɯɰɱɲɳɴɵɶɸɹɺɻɽɾʀʁʂʃʄʈʉʊʋʌʍʎʏʐʑʒʔʕʘʙʛʜʝʟʡʢʤʧʰʱʲʴʷʼˈˌːˑ˞ˠˤ̩βθχᵻ—“”…↑→↓↗↘ⱱ";

pub fn sanitize_for_kokoro(phonemes: &str) -> String {
    let normalized = phonemes
        .replace("ʲ", "j")
        .replace('r', "ɹ")
        .replace('x', "k")
        .replace("ɬ", "l")
        .replace('∅', " ");
    let filtered = normalized
        .chars()
        .filter(|ch| KOKORO_ALLOWED_PHONEME_CHARS.contains(*ch))
        .collect::<String>();
    filtered.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn join_tokens_for_kokoro(tokens: &[String]) -> String {
    sanitize_for_kokoro(&tokens.join(" "))
}

#[cfg(test)]
mod tests {
    use super::{join_tokens_for_kokoro, sanitize_for_kokoro};

    #[test]
    fn keeps_supported_ipa_and_drops_display_gaps() {
        assert_eq!(sanitize_for_kokoro("m ɛ ∅ ɪ"), "m ɛ ɪ");
        assert_eq!(sanitize_for_kokoro("m ɛ ɹ ɪ"), "m ɛ ɹ ɪ");
    }

    #[test]
    fn mirrors_kokoro_tokenizer_normalizations() {
        assert_eq!(sanitize_for_kokoro("r ʲ x ɬ"), "ɹ j k l");
    }

    #[test]
    fn removes_unknown_symbols_but_keeps_spacing_readable() {
        assert_eq!(sanitize_for_kokoro("m § ɛ ¤ ɪ"), "m ɛ ɪ");
    }

    #[test]
    fn joins_tokens_before_sanitizing() {
        let tokens = vec![
            "m".to_string(),
            "ɛ".to_string(),
            "∅".to_string(),
            "ɪ".to_string(),
        ];
        assert_eq!(join_tokens_for_kokoro(&tokens), "m ɛ ɪ");
    }
}
