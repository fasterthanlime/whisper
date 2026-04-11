//! High-level G2P engine with word-level IPA and probe caches.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use mlx_rs::ops::indexing::IndexOp;

use crate::config::T5Config;
use crate::load::load_weights_direct;
use crate::model::T5ForConditionalGeneration;
use crate::ownership::{ByteSpan, OwnershipSpan, compute_ownership};
use crate::tokenize;

/// Output of a cross-attention probe.
#[derive(Debug, Clone)]
pub struct ProbeOutput {
    /// The decoded IPA string.
    pub ipa: String,
    /// Generated token IDs (ByT5 byte IDs).
    pub generated_ids: Vec<i32>,
    /// Per-span ownership: which IPA substring belongs to which input span.
    pub ownership: Vec<OwnershipSpan>,
    /// Raw attention matrix, row-major `[dec_len, enc_len]`.
    pub attention_matrix: Vec<f32>,
    /// Number of decoder output steps.
    pub dec_len: usize,
    /// Number of encoder input positions.
    pub enc_len: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ProbeCacheKey {
    lang_code: String,
    word: String,
    spans: Vec<ByteSpan>,
}

/// High-level G2P engine. Bundles model + word-level IPA cache.
pub struct G2pEngine {
    model: T5ForConditionalGeneration,
    ipa_cache: HashMap<(String, String), String>,
    probe_cache: HashMap<ProbeCacheKey, ProbeOutput>,
}

impl G2pEngine {
    /// Load the Charsiu G2P model from a directory containing `model.safetensors`.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let config = T5Config::charsiu_g2p();
        let mut model = T5ForConditionalGeneration::new(config)?;
        let stats = load_weights_direct(&mut model, model_dir)?;
        if !stats.missing.is_empty() {
            anyhow::bail!(
                "Missing {} tensors: {:?}",
                stats.missing.len(),
                &stats.missing[..stats.missing.len().min(5)]
            );
        }
        Ok(Self {
            model,
            ipa_cache: HashMap::new(),
            probe_cache: HashMap::new(),
        })
    }

    fn probe_cache_key(word: &str, lang_code: &str, spans: &[ByteSpan]) -> ProbeCacheKey {
        ProbeCacheKey {
            lang_code: lang_code.to_owned(),
            word: word.to_owned(),
            spans: spans.to_vec(),
        }
    }

    /// Look up or compute IPA for a single word.
    pub fn g2p(&mut self, word: &str, lang_code: &str) -> Result<String> {
        let key = (lang_code.to_string(), word.to_string());
        if let Some(cached) = self.ipa_cache.get(&key) {
            return Ok(cached.clone());
        }

        let input = tokenize::format_g2p_input(word, lang_code);
        let input_ids = tokenize::encode_to_array(&input)?;
        let output_ids = self.model.generate(&input_ids, 64)?;
        let ipa = tokenize::decode_byt5(&output_ids);

        self.ipa_cache.insert(key, ipa.clone());
        Ok(ipa)
    }

    /// Batch G2P: compute IPA for multiple words, using cache where possible.
    /// Returns one IPA string per word, in the same order.
    pub fn g2p_batch(&mut self, words: &[&str], lang_code: &str) -> Result<Vec<String>> {
        // Split into cached and uncached
        let mut results = vec![String::new(); words.len()];
        let mut uncached_indices = Vec::new();
        let mut uncached_words = Vec::new();

        for (i, word) in words.iter().enumerate() {
            let key = (lang_code.to_string(), word.to_string());
            if let Some(cached) = self.ipa_cache.get(&key) {
                results[i] = cached.clone();
            } else {
                uncached_indices.push(i);
                uncached_words.push(*word);
            }
        }

        if !uncached_words.is_empty() {
            let prompts: Vec<String> = uncached_words
                .iter()
                .map(|w| tokenize::format_g2p_input(w, lang_code))
                .collect();
            let input_ids = tokenize::encode_batch_to_array(&prompts)?;
            let batch_results = self.model.generate_batch(&input_ids, 64)?;

            for (j, idx) in uncached_indices.into_iter().enumerate() {
                let ipa = tokenize::decode_byt5(&batch_results[j]);
                let key = (lang_code.to_string(), uncached_words[j].to_string());
                self.ipa_cache.insert(key, ipa.clone());
                results[idx] = ipa;
            }
        }

        Ok(results)
    }

    /// Generate IPA and extract cross-attention ownership for token-piece alignment.
    ///
    /// `spans` defines the byte ranges in the input word to score (e.g., Qwen token pieces).
    /// The spans' `byte_start`/`byte_end` are relative to the word text, not the full prompt.
    pub fn probe(
        &mut self,
        word: &str,
        lang_code: &str,
        spans: &[ByteSpan],
    ) -> Result<ProbeOutput> {
        let probe_key = Self::probe_cache_key(word, lang_code, spans);
        if let Some(cached) = self.probe_cache.get(&probe_key) {
            return Ok(cached.clone());
        }

        let input = tokenize::format_g2p_input(word, lang_code);
        let input_ids = tokenize::encode_to_array(&input)?;

        let (generated_ids, cross_attn) =
            self.model.generate_with_cross_attention(&input_ids, 64)?;
        let ipa = tokenize::decode_byt5(&generated_ids);

        let ipa_key = (lang_code.to_string(), word.to_string());
        self.ipa_cache.insert(ipa_key, ipa.clone());

        // Extract attention matrix to CPU
        let dec_len = cross_attn.shape()[0] as usize;
        let enc_len = cross_attn.shape()[1] as usize;
        let mut flat = Vec::with_capacity(dec_len * enc_len);
        for d in 0..dec_len {
            for e in 0..enc_len {
                let val: f32 = cross_attn.index((d as i32, e as i32)).item();
                flat.push(val);
            }
        }

        // Compute text byte offset: where the word starts in the prompt
        let prompt_bytes = input.as_bytes();
        let word_bytes = word.as_bytes();
        let text_byte_offset = prompt_bytes.len() - word_bytes.len();

        let ownership = compute_ownership(
            &flat,
            enc_len,
            dec_len,
            text_byte_offset,
            spans,
            &ipa,
            &generated_ids,
        );

        let output = ProbeOutput {
            ipa,
            generated_ids,
            ownership,
            attention_matrix: flat,
            dec_len,
            enc_len,
        };

        self.probe_cache.insert(probe_key, output.clone());

        Ok(output)
    }

    /// Number of cached IPA entries.
    pub fn cache_len(&self) -> usize {
        self.ipa_cache.len()
    }

    /// Number of cached probe entries.
    pub fn probe_cache_len(&self) -> usize {
        self.probe_cache.len()
    }

    /// Clear the IPA and probe caches.
    pub fn clear_cache(&mut self) {
        self.ipa_cache.clear();
        self.probe_cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_cache_key_includes_span_layout_and_order() {
        let spans_a = vec![
            ByteSpan {
                label: "a".into(),
                byte_start: 0,
                byte_end: 1,
            },
            ByteSpan {
                label: "b".into(),
                byte_start: 1,
                byte_end: 3,
            },
        ];
        let spans_b = vec![
            ByteSpan {
                label: "a".into(),
                byte_start: 0,
                byte_end: 1,
            },
            ByteSpan {
                label: "b".into(),
                byte_start: 1,
                byte_end: 3,
            },
        ];
        let spans_c = vec![
            ByteSpan {
                label: "b".into(),
                byte_start: 1,
                byte_end: 3,
            },
            ByteSpan {
                label: "a".into(),
                byte_start: 0,
                byte_end: 1,
            },
        ];

        let key_a = G2pEngine::probe_cache_key("word", "eng-us", &spans_a);
        let key_b = G2pEngine::probe_cache_key("word", "eng-us", &spans_b);
        let key_c = G2pEngine::probe_cache_key("word", "eng-us", &spans_c);
        let key_d = G2pEngine::probe_cache_key("word", "eng-gb", &spans_a);

        assert_eq!(key_a, key_b);
        assert_ne!(key_a, key_c);
        assert_ne!(key_a, key_d);
    }
}
