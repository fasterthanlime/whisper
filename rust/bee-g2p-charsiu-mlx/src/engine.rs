//! High-level G2P engine with word-level IPA and probe caches.

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use anyhow::{Result, bail};
use mlx_rs::ops::flatten;
use mlx_rs::{Array, Dtype};

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
        let start = Instant::now();
        // Split into cached and uncached
        let mut results = vec![String::new(); words.len()];
        let mut cache_hits = 0usize;
        let mut uncached_unique_indices = HashMap::<&str, usize>::new();
        let mut uncached_unique_words = Vec::new();
        let mut uncached_unique_positions = Vec::<Vec<usize>>::new();

        for (i, word) in words.iter().enumerate() {
            let key = (lang_code.to_string(), word.to_string());
            if let Some(cached) = self.ipa_cache.get(&key) {
                results[i] = cached.clone();
                cache_hits += 1;
            } else {
                let unique_index = *uncached_unique_indices.entry(*word).or_insert_with(|| {
                    let unique_index = uncached_unique_words.len();
                    uncached_unique_words.push(*word);
                    uncached_unique_positions.push(Vec::new());
                    unique_index
                });
                uncached_unique_positions[unique_index].push(i);
            }
        }

        if !uncached_unique_words.is_empty() {
            let prompts: Vec<String> = uncached_unique_words
                .iter()
                .map(|w| tokenize::format_g2p_input(w, lang_code))
                .collect();
            let input_ids = tokenize::encode_batch_to_array(&prompts)?;
            let batch_results = self.model.generate_batch(&input_ids, 64)?;

            for (j, word) in uncached_unique_words.iter().enumerate() {
                let ipa = tokenize::decode_byt5(&batch_results[j]);
                let key = (lang_code.to_string(), (*word).to_string());
                self.ipa_cache.insert(key, ipa.clone());
                for &idx in &uncached_unique_positions[j] {
                    results[idx] = ipa.clone();
                }
            }
        }

        tracing::trace!(
            target: "bee_phase",
            component = "g2p",
            phase = "g2p_batch",
            words_seen = words.len(),
            cache_hits,
            unique_uncached_words = uncached_unique_words.len(),
            duplicate_uncached_words = words.len().saturating_sub(cache_hits + uncached_unique_words.len()),
            ms = start.elapsed().as_secs_f64() * 1000.0,
            "batch timing"
        );

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

        let (flat, dec_len, enc_len) = extract_attention_matrix(&cross_attn)?;

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

fn extract_attention_matrix(cross_attn: &Array) -> Result<(Vec<f32>, usize, usize)> {
    let cross_attn = if cross_attn.dtype() == Dtype::Float32 {
        cross_attn.clone()
    } else {
        cross_attn.as_type::<f32>()?
    };

    let shape = cross_attn.shape();
    if shape.len() != 2 {
        bail!(
            "cross-attention matrix must be rank 2, got shape {:?}",
            shape
        );
    }

    let dec_len = shape[0] as usize;
    let enc_len = shape[1] as usize;

    // Normalize once, then bulk-read the full matrix in row-major order.
    let flat = flatten(&cross_attn, None, None)?;
    let matrix = flat.as_slice::<f32>().to_vec();

    Ok((matrix, dec_len, enc_len))
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

    #[test]
    fn extract_attention_matrix_bulk_reads_row_major_f32() {
        let cross_attn = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

        let (matrix, dec_len, enc_len) = extract_attention_matrix(&cross_attn).unwrap();

        assert_eq!(dec_len, 2);
        assert_eq!(enc_len, 3);
        assert_eq!(matrix, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }
}
