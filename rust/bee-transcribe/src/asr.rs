use std::path::Path;

/// Load a tokenizer from a directory. Tries `tokenizer.json` first (consolidated
/// HuggingFace format), then falls back to building from `vocab.json` + `merges.txt`.
pub(crate) fn load_tokenizer(
    dir: &Path,
) -> Result<tokenizers::Tokenizer, mlx_rs::error::Exception> {
    let consolidated = dir.join("tokenizer.json");
    if consolidated.exists() {
        return tokenizers::Tokenizer::from_file(&consolidated)
            .map_err(|e| mlx_rs::error::Exception::custom(format!("load tokenizer.json: {e}")));
    }

    let vocab_path = dir.join("vocab.json");
    let merges_path = dir.join("merges.txt");
    if vocab_path.exists() && merges_path.exists() {
        let bpe = tokenizers::models::bpe::BPE::from_file(
            vocab_path.to_str().unwrap(),
            merges_path.to_str().unwrap(),
        )
        .byte_fallback(true)
        .build()
        .map_err(|e| mlx_rs::error::Exception::custom(format!("build BPE tokenizer: {e}")))?;
        let mut tokenizer = tokenizers::Tokenizer::new(bpe);

        // GPT-2 style BPE uses byte-level pre-tokenizer and decoder to handle
        // the Ġ (U+0120) → space mapping and other byte-level encodings.
        tokenizer.with_pre_tokenizer(Some(
            tokenizers::pre_tokenizers::byte_level::ByteLevel::new(false, true, false),
        ));
        tokenizer.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
        return Ok(tokenizer);
    }

    Err(mlx_rs::error::Exception::custom(format!(
        "no tokenizer found in {} (need tokenizer.json or vocab.json + merges.txt)",
        dir.display()
    )))
}
