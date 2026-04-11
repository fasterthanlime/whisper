use std::path::Path;
use std::sync::OnceLock;

use anyhow::Result;
use bee_qwen3_asr::tokenizers::Tokenizer;

static TOKENIZER: OnceLock<Tokenizer> = OnceLock::new();

/// Loads the tokenizer from `path` and installs it into the process-global slot.
///
/// Invariants:
/// - initialization happens exactly once
/// - every decode helper in `types2` reads from the same tokenizer instance
pub(crate) fn init_tokenizer(path: &Path) -> &'static Tokenizer {
    let loaded =
        Tokenizer::from_file(path).unwrap_or_else(|e| panic!("loading {}: {e}", path.display()));
    TOKENIZER
        .set(loaded)
        .unwrap_or_else(|_| panic!("types2 tokenizer already initialized"));
    tokenizer()
}

/// Returns the process-global tokenizer slot.
///
/// Invariant:
/// - callers must initialize the slot through [`init_tokenizer`] before use
pub(crate) fn tokenizer() -> &'static Tokenizer {
    TOKENIZER
        .get()
        .unwrap_or_else(|| panic!("types2 tokenizer not initialized"))
}

pub(crate) fn decode_token_ids(token_ids: &[super::TokenId]) -> Result<String> {
    let ids: Vec<u32> = token_ids.iter().map(|id| id.as_u32()).collect();
    tokenizer()
        .decode(&ids, true)
        .map_err(|e| anyhow::anyhow!("decoding token ids: {e}"))
}
