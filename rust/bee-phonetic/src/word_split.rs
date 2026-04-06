pub use bee_types::SentenceWordToken;

pub fn split_sentence_words(text: &str) -> Vec<String> {
    sentence_word_tokens(text)
        .into_iter()
        .map(|token| token.text)
        .collect()
}

pub fn count_sentence_words(text: &str) -> usize {
    sentence_word_tokens(text).len()
}

pub fn sentence_word_tokens(text: &str) -> Vec<SentenceWordToken> {
    let mut tokens = Vec::new();
    let mut current_start = None;

    for (idx, ch) in text.char_indices() {
        if is_cjk_char(ch) {
            if let Some(start) = current_start.take() {
                tokens.push(SentenceWordToken {
                    char_start: start,
                    char_end: idx,
                    text: text[start..idx].to_string(),
                });
            }
            tokens.push(SentenceWordToken {
                char_start: idx,
                char_end: idx + ch.len_utf8(),
                text: ch.to_string(),
            });
            continue;
        }

        if is_sentence_word_char(ch) {
            current_start.get_or_insert(idx);
            continue;
        }

        if let Some(start) = current_start.take() {
            tokens.push(SentenceWordToken {
                char_start: start,
                char_end: idx,
                text: text[start..idx].to_string(),
            });
        }
    }

    if let Some(start) = current_start {
        tokens.push(SentenceWordToken {
            char_start: start,
            char_end: text.len(),
            text: text[start..].to_string(),
        });
    }

    tokens
}

fn is_sentence_word_char(ch: char) -> bool {
    ch.is_alphanumeric() || matches!(ch, '_' | '\'' | '-')
}

fn is_cjk_char(ch: char) -> bool {
    let code = ch as u32;
    (0x4E00..=0x9FFF).contains(&code)
        || (0x3400..=0x4DBF).contains(&code)
        || (0x20000..=0x2A6DF).contains(&code)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_words_on_punctuation_and_tracks_ranges() {
        let tokens = sentence_word_tokens("hello, world-test");
        assert_eq!(
            tokens,
            vec![
                SentenceWordToken {
                    char_start: 0,
                    char_end: 5,
                    text: "hello".to_string(),
                },
                SentenceWordToken {
                    char_start: 7,
                    char_end: 17,
                    text: "world-test".to_string(),
                },
            ]
        );
    }

    #[test]
    fn splits_cjk_chars_as_individual_tokens() {
        assert_eq!(
            split_sentence_words("你好 world"),
            vec!["你".to_string(), "好".to_string(), "world".to_string()]
        );
    }
}
