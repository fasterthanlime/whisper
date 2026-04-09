use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use bee_phonetic::{parse_reviewed_ipa, sentence_word_tokens};
use espeak_ng::{EspeakNg, install_bundled_language};

pub struct CachedEspeakG2p {
    engine: EspeakNg,
    cache: HashMap<String, Vec<String>>,
    persist_path: Option<PathBuf>,
    last_lookup_hit_cache: bool,
}

impl CachedEspeakG2p {
    pub fn english(base_dir: &Path) -> Result<Self> {
        Self::english_with_persist_path(base_dir, None)
    }

    pub fn english_with_persist_path(
        base_dir: &Path,
        persist_path: Option<PathBuf>,
    ) -> Result<Self> {
        let data_dir = bundled_espeak_data_dir(base_dir, "en")?;
        let cache = match persist_path.as_deref() {
            Some(path) => load_cache_file(path)?,
            None => HashMap::new(),
        };
        Ok(Self {
            engine: EspeakNg::with_data_dir("en", &data_dir)
                .context("initializing embedded espeak-ng engine")?,
            cache,
            persist_path,
            last_lookup_hit_cache: false,
        })
    }

    pub fn ipa_tokens(&mut self, text: &str) -> Result<Option<Vec<String>>> {
        let key = text.trim();
        if key.is_empty() {
            self.last_lookup_hit_cache = false;
            return Ok(None);
        }

        if let Some(tokens) = self.cache.get(key) {
            self.last_lookup_hit_cache = true;
            return Ok(Some(tokens.clone()));
        }

        self.last_lookup_hit_cache = false;
        let ipa = self
            .engine
            .text_to_phonemes(key)
            .with_context(|| format!("espeak-ng text_to_phonemes failed for '{key}'"))?;
        let tokens = parse_reviewed_ipa(ipa.trim());
        if tokens.is_empty() {
            return Ok(None);
        }

        if let Some(path) = &self.persist_path {
            append_cache_entry(path, key, &tokens)?;
        }

        self.cache.insert(key.to_string(), tokens.clone());
        Ok(Some(tokens))
    }

    pub fn ipa_word_tokens_in_utterance(&mut self, text: &str) -> Result<Option<Vec<Vec<String>>>> {
        let key = text.trim();
        if key.is_empty() {
            self.last_lookup_hit_cache = false;
            return Ok(None);
        }

        let words = sentence_word_tokens(key);
        if words.is_empty() {
            return Ok(None);
        }

        let mut out = Vec::with_capacity(words.len());
        let mut previous_prefix = Vec::<String>::new();
        for word in words {
            let prefix_text = &key[..word.char_end];
            let prefix_tokens = self
                .ipa_tokens(prefix_text)?
                .ok_or_else(|| anyhow!("espeak produced no tokens for '{prefix_text}'"))?;
            let common_prefix = previous_prefix
                .iter()
                .zip(prefix_tokens.iter())
                .take_while(|(left, right)| left == right)
                .count();
            out.push(prefix_tokens[common_prefix..].to_vec());
            previous_prefix = prefix_tokens;
        }

        if out.is_empty() {
            return Ok(None);
        }

        Ok(Some(out))
    }

    pub fn last_lookup_hit_cache(&self) -> bool {
        self.last_lookup_hit_cache
    }
}

fn bundled_espeak_data_dir(base_dir: &Path, lang: &str) -> Result<PathBuf> {
    let data_dir = base_dir.join("espeak-ng-data").join(lang);
    fs::create_dir_all(&data_dir)
        .with_context(|| format!("creating bundled espeak-ng data dir {}", data_dir.display()))?;
    install_bundled_language(&data_dir, lang)
        .with_context(|| format!("installing bundled espeak-ng data for {lang}"))?;
    Ok(data_dir)
}

fn load_cache_file(path: &Path) -> Result<HashMap<String, Vec<String>>> {
    let mut cache = HashMap::new();
    if !path.exists() {
        return Ok(cache);
    }

    let file = fs::File::open(path)
        .with_context(|| format!("opening g2p cache file {}", path.display()))?;
    for (line_no, line) in BufReader::new(file).lines().enumerate() {
        let line = line.with_context(|| {
            format!(
                "reading g2p cache file {} line {}",
                path.display(),
                line_no + 1
            )
        })?;
        if line.is_empty() {
            continue;
        }
        let Some((key, ipa)) = line.split_once('\t') else {
            continue;
        };
        let key = unescape_field(key)
            .with_context(|| format!("decoding g2p cache key on line {}", line_no + 1))?;
        let ipa = unescape_field(ipa)
            .with_context(|| format!("decoding g2p cache value on line {}", line_no + 1))?;
        let tokens = parse_reviewed_ipa(&ipa);
        if !tokens.is_empty() {
            cache.insert(key, tokens);
        }
    }
    Ok(cache)
}

fn append_cache_entry(path: &Path, key: &str, tokens: &[String]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating g2p cache parent dir {}", parent.display()))?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("opening g2p cache file for append {}", path.display()))?;
    let ipa = tokens.join(" ");
    writeln!(file, "{}\t{}", escape_field(key), escape_field(&ipa))
        .with_context(|| format!("writing g2p cache entry to {}", path.display()))?;
    Ok(())
}

fn escape_field(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            _ => out.push(ch),
        }
    }
    out
}

fn unescape_field(text: &str) -> Result<String> {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            out.push(ch);
            continue;
        }

        let escaped = chars
            .next()
            .context("unterminated escape in g2p cache entry")?;
        match escaped {
            '\\' => out.push('\\'),
            'n' => out.push('\n'),
            't' => out.push('\t'),
            other => {
                out.push('\\');
                out.push(other);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utterance_word_groups_preserve_contextual_onset() {
        let base_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("crate should live under rust/");
        let mut g2p = CachedEspeakG2p::english(base_dir).expect("init g2p");
        let groups = g2p
            .ipa_word_tokens_in_utterance("I used Thursday in the parser.")
            .expect("phonemize utterance")
            .expect("non-empty utterance");

        assert_eq!(groups.len(), 6);
        assert_eq!(groups[0], vec!["aɪ".to_string()]);
        assert!(!groups[1].is_empty());
    }
}
