//! Attention matrix → ownership spans.
//!
//! Pure computation, no MLX dependency. Operates on a flat `&[f32]` attention
//! matrix extracted from the model.

/// A byte range in the input text (e.g., a word span or Qwen token-piece span).
#[derive(Debug, Clone)]
pub struct ByteSpan {
    pub label: String,
    pub byte_start: usize,
    pub byte_end: usize,
}

/// Attention mass on a span for one output step.
#[derive(Debug, Clone)]
pub struct SpanScore {
    pub span_index: usize,
    pub score: f32,
}

/// Contiguous IPA bytes owned by one input span.
#[derive(Debug, Clone)]
pub struct OwnershipSpan {
    /// Index into the original `spans` slice.
    pub span_index: usize,
    /// Label from the span (e.g., token surface text).
    pub label: String,
    /// Byte offset into the IPA string where this span's ownership starts.
    pub ipa_byte_start: usize,
    /// Byte offset into the IPA string where this span's ownership ends.
    pub ipa_byte_end: usize,
    /// The IPA substring owned by this span.
    pub ipa_text: String,
    /// Average attention score across the output steps in this run.
    pub avg_score: f32,
}

/// Sum attention weights over each span's byte range for one output step.
///
/// `attention_row` has length `enc_len` — one weight per encoder position (input byte).
/// `text_byte_offset` is the byte offset where the actual text starts in the prompt
/// (to skip the `<lang>: ` prefix bytes).
pub fn score_spans(
    attention_row: &[f32],
    text_byte_offset: usize,
    spans: &[ByteSpan],
) -> Vec<SpanScore> {
    spans
        .iter()
        .enumerate()
        .map(|(i, span)| {
            let start = text_byte_offset + span.byte_start;
            let end = text_byte_offset + span.byte_end;
            let score: f32 = attention_row
                [start.min(attention_row.len())..end.min(attention_row.len())]
                .iter()
                .sum();
            SpanScore {
                span_index: i,
                score,
            }
        })
        .collect()
}

/// Compute ownership: for each output step, find the top-scoring span,
/// then collapse contiguous runs of the same span into `OwnershipSpan`s.
///
/// `matrix` is row-major `[dec_len, enc_len]`.
/// `ipa` is the decoded IPA string (used to extract substrings for each run).
/// `generated_ids` are the raw ByT5 token IDs (3+ = byte, used to map output steps to IPA bytes).
pub fn compute_ownership(
    matrix: &[f32],
    enc_len: usize,
    dec_len: usize,
    text_byte_offset: usize,
    spans: &[ByteSpan],
    ipa: &str,
    generated_ids: &[i32],
) -> Vec<OwnershipSpan> {
    if spans.is_empty() || dec_len == 0 {
        return Vec::new();
    }

    // Map each output step to its byte offset in the IPA string.
    // The attention matrix has dec_len rows. The decoder_input_ids are:
    //   [decoder_start_token_id] + generated_ids
    // So dec_len = 1 + len(generated_ids). The first step (start token) emits nothing.
    let ipa_bytes = ipa.as_bytes();
    // step_byte_offsets[i] = byte offset in IPA string BEFORE step i produces output.
    // Step 0 = start token (no output). Steps 1..dec_len correspond to generated_ids.
    let mut step_byte_offsets: Vec<usize> = Vec::with_capacity(dec_len);
    let mut byte_pos = 0usize;
    step_byte_offsets.push(0); // step 0: start token, at byte 0
    for &id in generated_ids {
        step_byte_offsets.push(byte_pos);
        if id >= 3 {
            byte_pos += 1;
        }
    }
    let total_bytes = byte_pos;
    step_byte_offsets.truncate(dec_len);

    // Group output steps into UTF-8 characters. Only the last byte of a
    // multi-byte character "emits" text — matching bee-g2p-charsiu's approach
    // where pending_rows accumulate until emitted_text is non-empty.
    //
    // For each character group, score spans using all bytes in the group,
    // then pick the top span as the anchor (matching the existing logic).
    struct CharGroup {
        /// Steps (output indices) that make up this character.
        steps: Vec<usize>,
        /// Byte offset where this character starts in the IPA string.
        ipa_byte_start: usize,
        /// Byte offset where this character ends.
        ipa_byte_end: usize,
    }

    let mut char_groups: Vec<CharGroup> = Vec::new();
    let mut current_steps: Vec<usize> = Vec::new();
    let mut current_byte_start: Option<usize> = None;

    // Skip step 0 (decoder start token) — it produces no IPA output
    for step in 1..dec_len {
        let byte_off = step_byte_offsets[step];
        if current_byte_start.is_none() {
            current_byte_start = Some(byte_off);
        }
        current_steps.push(step);

        // Check if this step completes a UTF-8 character
        let next_byte_off = if step + 1 < dec_len {
            step_byte_offsets[step + 1]
        } else {
            total_bytes
        };

        // A character is complete when the next step starts a new byte position
        // AND that position is on a UTF-8 character boundary in the IPA string
        let is_char_boundary = next_byte_off <= ipa_bytes.len()
            && (next_byte_off == total_bytes || ipa.is_char_boundary(next_byte_off));
        let advanced = next_byte_off > byte_off;

        if advanced && is_char_boundary {
            char_groups.push(CharGroup {
                steps: std::mem::take(&mut current_steps),
                ipa_byte_start: current_byte_start.take().unwrap(),
                ipa_byte_end: next_byte_off,
            });
            current_byte_start = None;
        }
    }
    // Flush remaining steps (shouldn't happen with valid UTF-8, but be safe)
    if !current_steps.is_empty() {
        let start = current_byte_start.unwrap_or(total_bytes);
        char_groups.push(CharGroup {
            steps: current_steps,
            ipa_byte_start: start,
            ipa_byte_end: total_bytes,
        });
    }

    // For each character group, accumulate span scores across all decoder steps
    // that contributed bytes to that emitted character. Using the whole group is
    // more stable than anchoring on the final byte step only, which can shove a
    // trailing consonant onto the next token piece.
    let mut group_assignments: Vec<(usize, f32)> = Vec::with_capacity(char_groups.len());
    for group in &char_groups {
        let mut accum = vec![0.0f32; spans.len()];
        for &step in &group.steps {
            let row = &matrix[step * enc_len..(step + 1) * enc_len];
            for score in score_spans(row, text_byte_offset, spans) {
                accum[score.span_index] += score.score;
            }
        }
        let (span_index, score) = accum
            .iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        let avg_score = score / group.steps.len() as f32;
        group_assignments.push((span_index, avg_score));
    }

    // Collapse contiguous character groups assigned to the same span
    let mut result: Vec<OwnershipSpan> = Vec::new();
    let mut run_start = 0usize;

    while run_start < char_groups.len() {
        let (span_idx, _) = group_assignments[run_start];
        let mut run_end = run_start + 1;
        while run_end < char_groups.len() && group_assignments[run_end].0 == span_idx {
            run_end += 1;
        }

        let avg_score: f32 = group_assignments[run_start..run_end]
            .iter()
            .map(|(_, s)| s)
            .sum::<f32>()
            / (run_end - run_start) as f32;

        let ipa_start = char_groups[run_start].ipa_byte_start.min(ipa_bytes.len());
        let ipa_end = char_groups[run_end - 1].ipa_byte_end.min(ipa_bytes.len());
        let ipa_text = String::from_utf8_lossy(&ipa_bytes[ipa_start..ipa_end]).into_owned();

        result.push(OwnershipSpan {
            span_index: span_idx,
            label: spans[span_idx].label.clone(),
            ipa_byte_start: ipa_start,
            ipa_byte_end: ipa_end,
            ipa_text,
            avg_score,
        });

        run_start = run_end;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_spans_basic() {
        // 5 encoder positions, weights sum to 1
        let row = [0.1, 0.2, 0.3, 0.25, 0.15];
        let spans = vec![
            ByteSpan {
                label: "ab".into(),
                byte_start: 0,
                byte_end: 2,
            },
            ByteSpan {
                label: "cde".into(),
                byte_start: 2,
                byte_end: 5,
            },
        ];
        let scores = score_spans(&row, 0, &spans);
        assert_eq!(scores.len(), 2);
        assert!((scores[0].score - 0.3).abs() < 1e-6); // 0.1 + 0.2
        assert!((scores[1].score - 0.7).abs() < 1e-6); // 0.3 + 0.25 + 0.15
    }

    #[test]
    fn test_compute_ownership_simple() {
        // dec_len=3 (start token + 2 generated), enc_len=6
        // Prompt: "<>: ab" — text_byte_offset=4, text bytes at positions 4,5
        // Span A covers byte 0 (position 4), Span B covers byte 1 (position 5)
        let matrix = [
            // step 0 (start token): don't care
            0.0, 0.0, 0.0, 0.0, 0.5, 0.5, // step 1: attends to position 4 (span A)
            0.0, 0.0, 0.0, 0.0, 0.8, 0.2, // step 2: attends to position 5 (span B)
            0.0, 0.0, 0.0, 0.0, 0.1, 0.9,
        ];
        let spans = vec![
            ByteSpan {
                label: "a".into(),
                byte_start: 0,
                byte_end: 1,
            },
            ByteSpan {
                label: "b".into(),
                byte_start: 1,
                byte_end: 2,
            },
        ];
        // Generated IDs: both are valid bytes (id >= 3)
        let generated_ids = [100, 101];
        let ipa = "xy";

        let ownership = compute_ownership(&matrix, 6, 3, 4, &spans, ipa, &generated_ids);
        assert_eq!(ownership.len(), 2);
        assert_eq!(ownership[0].label, "a");
        assert_eq!(ownership[0].ipa_text, "x");
        assert_eq!(ownership[1].label, "b");
        assert_eq!(ownership[1].ipa_text, "y");
    }
}
