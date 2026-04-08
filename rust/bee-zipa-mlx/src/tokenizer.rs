use std::collections::HashMap;
use std::path::Path;

use crate::error::ZipaError;
use crate::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenTable {
    tokens_by_id: Vec<String>,
}

impl TokenTable {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Self::from_str(&text)
    }

    pub fn from_str(text: &str) -> Result<Self> {
        let mut tokens = HashMap::<usize, String>::new();
        let mut max_id = 0usize;

        for raw_line in text.lines() {
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }
            let mut parts = line.split_whitespace();
            let token = parts
                .next()
                .ok_or_else(|| ZipaError::InvalidTokenLine(raw_line.to_owned()))?;
            let id = parts
                .next()
                .ok_or_else(|| ZipaError::InvalidTokenLine(raw_line.to_owned()))?
                .parse::<usize>()
                .map_err(|_| ZipaError::InvalidTokenLine(raw_line.to_owned()))?;
            max_id = max_id.max(id);
            tokens.insert(id, token.to_owned());
        }

        let mut tokens_by_id = vec![String::new(); max_id + 1];
        for (id, token) in tokens {
            tokens_by_id[id] = token;
        }

        Ok(Self { tokens_by_id })
    }

    pub fn len(&self) -> usize {
        self.tokens_by_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens_by_id.is_empty()
    }

    pub fn get(&self, id: usize) -> Option<&str> {
        self.tokens_by_id.get(id).map(String::as_str)
    }

    pub fn decode_ctc_greedy(&self, frame_token_ids: &[usize], blank_id: usize) -> Vec<String> {
        let mut decoded = Vec::new();
        let mut prev = None;

        for &token_id in frame_token_ids {
            if token_id != blank_id && Some(token_id) != prev {
                if let Some(token) = self.get(token_id) {
                    if !token.is_empty() {
                        decoded.push(token.to_owned());
                    }
                }
            }
            prev = Some(token_id);
        }

        decoded
    }
}

#[cfg(test)]
mod tests {
    use super::TokenTable;

    #[test]
    fn parses_tokens_txt() {
        let table = TokenTable::from_str("<blk> 0\n▁ 1\nə 2\n").unwrap();
        assert_eq!(table.len(), 3);
        assert_eq!(table.get(2), Some("ə"));
    }

    #[test]
    fn greedy_ctc_decode_drops_blank_and_repeats() {
        let table = TokenTable::from_str("<blk> 0\n▁ 1\nə 2\nn 3\n").unwrap();
        let ids = [0, 1, 1, 0, 2, 2, 3, 0];
        assert_eq!(table.decode_ctc_greedy(&ids, 0), vec!["▁", "ə", "n"]);
    }
}
