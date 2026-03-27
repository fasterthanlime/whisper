/// Build the Qwen3 tokenizer JSON from vocab.json, merges.txt, and tokenizer_config.json.
/// The added_tokens list is derived from tokenizer_config.json's added_tokens_decoder field,
/// so no special tokens need to be hardcoded here.
pub(crate) fn build_qwen3_tokenizer_json(
    vocab: &str,
    merges: &str,
    tok_config: &str,
) -> anyhow::Result<Vec<u8>> {
    let vocab_val: serde_json::Value = serde_json::from_str(vocab)?;
    let merges_vec: Vec<&str> = merges
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
        .collect();

    // Build added_tokens from tokenizer_config.json's added_tokens_decoder.
    let tok_cfg: serde_json::Value = serde_json::from_str(tok_config)?;
    let mut added_tokens: Vec<serde_json::Value> = Vec::new();
    if let Some(decoder_map) = tok_cfg["added_tokens_decoder"].as_object() {
        let mut entries: Vec<(u64, &serde_json::Value)> = decoder_map
            .iter()
            .filter_map(|(k, v)| k.parse::<u64>().ok().map(|id| (id, v)))
            .collect();
        entries.sort_by_key(|(id, _)| *id);
        for (id, v) in &entries {
            added_tokens.push(serde_json::json!({
                "id": id,
                "content": v["content"],
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": v["special"]
            }));
        }
    }
    let added_tokens = serde_json::Value::Array(added_tokens);

    let tokenizer_json = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added_tokens,
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},
                    "behavior": "Isolated",
                    "invert": false
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": false,
                    "trim_offsets": false,
                    "use_regex": false
                }
            ]
        },
        "post_processor": {
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": false,
            "use_regex": false
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": false,
            "use_regex": false
        },
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": false,
            "byte_fallback": false,
            "ignore_merges": false,
            "vocab": vocab_val,
            "merges": merges_vec
        }
    });

    serde_json::to_vec(&tokenizer_json).map_err(Into::into)
}
