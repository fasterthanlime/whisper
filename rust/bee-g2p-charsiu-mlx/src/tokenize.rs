use mlx_rs::Array;
use mlx_rs::error::Exception;

/// ByT5 tokenization: each UTF-8 byte maps to token_id = byte + 3.
///
/// Special tokens: 0 = pad, 1 = eos, 2 = unk.
/// Byte values 0-255 map to token ids 3-258.
pub fn encode_byt5(text: &str) -> Vec<i32> {
    text.as_bytes().iter().map(|&b| b as i32 + 3).collect()
}

/// Decode ByT5 token ids back to a string.
///
/// Tokens 0-2 are special (pad/eos/unk) and are skipped.
/// Tokens 3-258 map to bytes 0-255.
pub fn decode_byt5(token_ids: &[i32]) -> String {
    let bytes: Vec<u8> = token_ids
        .iter()
        .filter(|&&id| id >= 3)
        .map(|&id| (id - 3) as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Format a word for Charsiu G2P input.
pub fn format_g2p_input(word: &str, lang_code: &str) -> String {
    format!("<{lang_code}>: {word}")
}

/// Encode text to a batch of token ids as an MLX array.
///
/// Returns shape [1, seq_len] suitable for model input.
pub fn encode_to_array(text: &str) -> Result<Array, Exception> {
    let ids = encode_byt5(text);
    let len = ids.len() as i32;
    Array::from_slice(&ids, &[1, len]).as_type::<i32>()
}
