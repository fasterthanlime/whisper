//! Parse raw generator output into structured form, splitting metadata
//! from text tokens at the `<asr_text>` boundary.

use bee_qwen3_asr::generate::{self, TokenLogprob};

use crate::types::TokenId;

/// Raw ASR output parsed into metadata and text portions.
///
/// The `<asr_text>` token marks the boundary: everything before is metadata
/// (language tag, etc.), everything after is actual transcription.
/// Downstream layers (aligner, corrector) only see `text_ids` / `text_logprobs`.
pub struct StructuredAsrOutput<'a> {
    /// Metadata tokens before `<asr_text>` (language tag, etc.)
    pub metadata_ids: &'a [TokenId],
    /// Text tokens after `<asr_text>` — the actual transcription
    pub text_ids: &'a [TokenId],
    /// Logprobs corresponding to `text_ids`
    pub text_logprobs: &'a [TokenLogprob],
}

impl<'a> StructuredAsrOutput<'a> {
    /// Split raw token arrays at the `<asr_text>` boundary.
    ///
    /// If no `<asr_text>` token is found, all tokens are treated as text
    /// (metadata is empty).
    pub fn from_raw(raw_ids: &'a [TokenId], raw_logprobs: &'a [TokenLogprob]) -> Self {
        let asr_text_id = generate::TOK_ASR_TEXT as TokenId;

        if let Some(tag_pos) = raw_ids.iter().position(|&id| id == asr_text_id) {
            let metadata_ids = &raw_ids[..tag_pos];
            let text_ids = &raw_ids[tag_pos + 1..];
            // Logprobs array may be shorter than ids (prefix tokens reuse old logprobs).
            // The metadata portion consumes `tag_pos` logprobs, plus 1 for the tag itself.
            let logprob_offset = (tag_pos + 1).min(raw_logprobs.len());
            let text_logprobs = &raw_logprobs[logprob_offset..];

            Self {
                metadata_ids,
                text_ids,
                text_logprobs,
            }
        } else {
            // No marker — treat everything as text
            Self {
                metadata_ids: &[],
                text_ids: raw_ids,
                text_logprobs: raw_logprobs,
            }
        }
    }

    /// Number of raw tokens consumed by metadata (including the `<asr_text>` tag itself).
    /// Useful for computing rotation offsets.
    pub fn metadata_token_count(&self) -> usize {
        if self.metadata_ids.is_empty() {
            0
        } else {
            self.metadata_ids.len() + 1 // +1 for the <asr_text> tag
        }
    }
}
