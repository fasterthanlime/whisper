//! Autoregressive text generation for Qwen3-ASR.

use mlx_rs::Array;
use mlx_rs::error::Exception;
use mlx_rs::ops;
use mlx_rs::ops::indexing::{self, IndexOp};

use crate::decoder::KVCache;
use crate::model::{AUDIO_END_TOKEN_ID, AUDIO_PAD_TOKEN_ID, AUDIO_START_TOKEN_ID};
use crate::model::{EOS_TOKEN_IDS, Qwen3ASRModel};

/// TokenID type used by qwen3-asr generation
pub type TokenId = i32;

/// Number of top-k alternatives stored per token.
pub const TOP_K: usize = 4;
pub const STREAMING_TOP_K: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfidenceMode {
    Streaming,
    Full,
}

/// Per-token confidence information extracted during decoding.
///
/// Stores the top-k token alternatives and their raw logits.
/// Derived metrics (margin, concentration) are computed from these.
#[derive(Debug, Clone, Copy)]
pub struct TokenConfidence {
    /// The chosen (top-1) token ID.
    pub token_id: TokenId,

    /// Number of populated entries in `top_ids` / `top_logits`.
    pub alternative_count: u8,

    /// Top-k token IDs, sorted by descending logit.
    pub top_ids: [TokenId; TOP_K],

    /// Top-k raw logits (unnormalized), sorted descending.
    pub top_logits: [f32; TOP_K],

    /// Concentration: how far the winner is above the pack.
    /// `top1_logit - mean(top2..topk_logits)`. Higher = more confident.
    pub concentration: f32,

    /// Margin: difference between top-1 and top-2 logits.
    /// Higher = more decisive choice. Useful for gating uncertain tokens.
    pub margin: f32,
}

const REPETITION_THRESHOLD: usize = 20;

// Chat template token IDs
pub const TOK_IM_START: TokenId = 151644;
pub const TOK_IM_END: TokenId = 151645;
pub const TOK_SYSTEM: TokenId = 8948;
pub const TOK_USER: TokenId = 872;
pub const TOK_ASSISTANT: TokenId = 77091;
pub const TOK_NEWLINE: TokenId = 198;
pub const TOK_ASR_TEXT: TokenId = 151704;

/// Greedy autoregressive generation (batch mode, fresh cache).
pub fn generate(
    model: &Qwen3ASRModel,
    input_ids: &Array,
    audio_features: &Array,
    position_ids: &Array,
    max_new_tokens: usize,
) -> Result<Vec<TokenId>, Exception> {
    let mut cache = Some(model.create_cache());

    let logits = model.prefill(input_ids, audio_features, position_ids, &mut cache)?;
    let mut token = argmax(&logits)?;
    let mut generated = vec![token];

    if is_eos(token) || max_new_tokens <= 1 {
        return Ok(strip_eos(generated));
    }

    let seq_len = input_ids.shape()[1];

    for step in 1..max_new_tokens {
        let next_ids = Array::from_slice(&[token], &[1, 1]);
        let pos = seq_len + step as i32;
        let pos_arr = Array::from_slice(&[pos, pos, pos], &[1, 3, 1]);

        let logits = model.step(&next_ids, &pos_arr, &mut cache)?;
        token = argmax(&logits)?;
        generated.push(token);

        if is_eos(token) {
            break;
        }
        if detect_repetition(&generated) {
            break;
        }
        if step % 8 == 0 {
            logits.eval()?;
        }
    }

    Ok(strip_eos(generated))
}

/// Build the initial prompt tokens for the first streaming chunk.
///
/// Template:
///   <|im_start|>system\n{context}<|im_end|>\n<|im_start|>user\n
///   <|audio_start|><|audio_pad|>*N<|audio_end|><|im_end|>\n
///   <|im_start|>assistant\n
///   [language {lang}<asr_text>]   ← only when language is non-empty
pub fn build_initial_prompt(
    n_audio_tokens: usize,
    language: &str,
    context: &str,
    tokenizer: &tokenizers::Tokenizer,
) -> Vec<TokenId> {
    let mut prompt: Vec<TokenId> = vec![TOK_IM_START, TOK_SYSTEM, TOK_NEWLINE];
    if !context.is_empty() {
        prompt.extend(tokenize(tokenizer, context));
    }
    prompt.extend_from_slice(&[
        TOK_IM_END,
        TOK_NEWLINE,
        TOK_IM_START,
        TOK_USER,
        TOK_NEWLINE,
        AUDIO_START_TOKEN_ID,
    ]);
    prompt.extend(std::iter::repeat_n(AUDIO_PAD_TOKEN_ID, n_audio_tokens));
    prompt.extend_from_slice(&[
        AUDIO_END_TOKEN_ID,
        TOK_IM_END,
        TOK_NEWLINE,
        TOK_IM_START,
        TOK_ASSISTANT,
        TOK_NEWLINE,
    ]);
    if !language.is_empty() {
        let lang_header = format!("language {language}");
        prompt.extend(tokenize(tokenizer, &lang_header));
        prompt.push(TOK_ASR_TEXT);
    }
    prompt
}

/// Build the follow-up prompt for subsequent streaming chunks.
///
/// Template:
///   <|im_end|>\n<|im_start|>user\n
///   <|audio_start|><|audio_pad|>*N<|audio_end|><|im_end|>\n
///   <|im_start|>assistant\n
///   [language {lang}<asr_text>]   ← only when language is non-empty
pub fn build_followup_prompt(
    n_audio_tokens: usize,
    language: &str,
    tokenizer: &tokenizers::Tokenizer,
) -> Vec<TokenId> {
    let mut prompt: Vec<TokenId> = vec![
        TOK_IM_END,
        TOK_NEWLINE,
        TOK_IM_START,
        TOK_USER,
        TOK_NEWLINE,
        AUDIO_START_TOKEN_ID,
    ];
    prompt.extend(std::iter::repeat_n(AUDIO_PAD_TOKEN_ID, n_audio_tokens));
    prompt.extend_from_slice(&[
        AUDIO_END_TOKEN_ID,
        TOK_IM_END,
        TOK_NEWLINE,
        TOK_IM_START,
        TOK_ASSISTANT,
        TOK_NEWLINE,
    ]);
    if !language.is_empty() {
        let lang_header = format!("language {language}");
        prompt.extend(tokenize(tokenizer, &lang_header));
        prompt.push(TOK_ASR_TEXT);
    }
    prompt
}

/// Prefill a prompt (initial or followup) into an existing cache, then
/// autoregressively decode up to max_new_tokens.
///
/// Returns generated token IDs, per-token confidence (top-k info), and the
/// updated next_position. Position tracking: the cache contains exactly the prompt
/// tokens plus each non-EOS generated token that was stepped through. EOS
/// tokens are NOT added to the cache (matching the Python reference).
pub fn prefill_and_decode(
    model: &Qwen3ASRModel,
    prompt_tokens: &[TokenId],
    audio_features: &Array,
    cache: &mut Option<KVCache>,
    start_position: usize,
    max_new_tokens: usize,
    confidence_mode: ConfidenceMode,
) -> Result<(Vec<TokenId>, Vec<TokenConfidence>, usize), Exception> {
    let seq_len = prompt_tokens.len();
    let input_ids = Array::from_slice(prompt_tokens, &[1, seq_len as i32]);

    let positions: Vec<i32> = (start_position as i32..(start_position + seq_len) as i32).collect();
    let pos_arr = Array::from_slice(&positions, &[1, 1, seq_len as i32]);
    let position_ids = ops::broadcast_to(&pos_arr, &[1, 3, seq_len as i32])?;

    // Ensure cache exists
    if cache.is_none() {
        *cache = Some(model.create_cache());
    }

    // Prefill — appends to existing cache
    let logits = model.prefill(&input_ids, audio_features, &position_ids, cache)?;
    // Position after prefill: prompt tokens are in the cache
    let mut position = start_position + seq_len;

    let tlp = confidence_for_logits(&logits, confidence_mode)?;
    let mut token = tlp.token_id;
    let mut generated = Vec::new();
    let mut confidences = Vec::new();

    tracing::debug!(
        "prefill_and_decode: first_token={token} concentration={:.3} margin={:.3} is_eos={} prompt_len={seq_len} max_new={max_new_tokens}",
        tlp.concentration,
        tlp.margin,
        is_eos(token),
    );

    if is_eos(token) || max_new_tokens <= 1 {
        // EOS on first token — nothing added to cache beyond prompt
        if !is_eos(token) {
            generated.push(token);
            confidences.push(tlp);
        }
        return Ok((generated, confidences, position));
    }

    generated.push(token);
    confidences.push(tlp);

    // Autoregressive decode — each step adds the token to the cache
    for _ in 1..max_new_tokens {
        let next_ids = Array::from_slice(&[token], &[1, 1]);
        let pos_arr = Array::from_slice(
            &[position as i32, position as i32, position as i32],
            &[1, 3, 1],
        );

        let logits = model.step(&next_ids, &pos_arr, cache)?;
        position += 1; // token was fed into cache

        let tlp = confidence_for_logits(&logits, confidence_mode)?;
        token = tlp.token_id;

        if is_eos(token) {
            break; // EOS not fed into cache
        }

        generated.push(token);
        confidences.push(tlp);

        if detect_repetition(&generated) {
            break;
        }

        if generated.len() % 8 == 0 {
            logits.eval()?;
        }
    }

    Ok((generated, confidences, position))
}

/// Score an existing generated continuation against a prompt.
///
/// This is used to cheaply stream with reduced confidence metadata, then
/// refresh the same token sequence with full top-k confidence right before
/// commit/finalization when the correction pipeline needs richer metadata.
pub fn score_continuation(
    model: &Qwen3ASRModel,
    prompt_tokens: &[TokenId],
    audio_features: &Array,
    continuation_tokens: &[TokenId],
    confidence_mode: ConfidenceMode,
) -> Result<Vec<TokenConfidence>, Exception> {
    if continuation_tokens.is_empty() {
        return Ok(Vec::new());
    }

    let seq_len = prompt_tokens.len();
    let input_ids = Array::from_slice(prompt_tokens, &[1, seq_len as i32]);
    let positions: Vec<i32> = (0..seq_len as i32).collect();
    let pos_arr = Array::from_slice(&positions, &[1, 1, seq_len as i32]);
    let position_ids = ops::broadcast_to(&pos_arr, &[1, 3, seq_len as i32])?;

    let mut cache = Some(model.create_cache());
    let logits = model.prefill(&input_ids, audio_features, &position_ids, &mut cache)?;

    let mut scored = Vec::with_capacity(continuation_tokens.len());
    scored.push(confidence_for_logits_forced(
        &logits,
        continuation_tokens[0],
        confidence_mode,
    )?);

    let mut position = seq_len;
    for window in continuation_tokens.windows(2) {
        let prev = window[0];
        let expected = window[1];
        let next_ids = Array::from_slice(&[prev], &[1, 1]);
        let pos_arr = Array::from_slice(
            &[position as i32, position as i32, position as i32],
            &[1, 3, 1],
        );
        let logits = model.step(&next_ids, &pos_arr, &mut cache)?;
        position += 1;
        scored.push(confidence_for_logits_forced(
            &logits,
            expected,
            confidence_mode,
        )?);
    }

    Ok(scored)
}

fn argmax(logits: &Array) -> Result<i32, Exception> {
    let flat = logits.reshape(&[-1])?;
    let idx = indexing::argmax(&flat, None)?;
    Ok(idx.item::<i32>())
}

fn confidence_for_logits(
    logits: &Array,
    confidence_mode: ConfidenceMode,
) -> Result<TokenConfidence, Exception> {
    match confidence_mode {
        ConfidenceMode::Streaming => top2_confidence(logits),
        ConfidenceMode::Full => topk_confidence(logits),
    }
}

fn confidence_for_logits_forced(
    logits: &Array,
    token_id: TokenId,
    confidence_mode: ConfidenceMode,
) -> Result<TokenConfidence, Exception> {
    let mut confidence = confidence_for_logits(logits, confidence_mode)?;
    confidence.token_id = token_id;
    Ok(confidence)
}

fn top2_confidence(logits: &Array) -> Result<TokenConfidence, Exception> {
    let flat = logits.reshape(&[-1])?;

    let neg = ops::negative(&flat)?;
    let partitioned_indices = ops::argpartition(&neg, 1)?;
    let top2_indices = partitioned_indices.index(..STREAMING_TOP_K as i32);
    let top2_values = flat.index(&top2_indices);

    top2_indices.eval()?;
    top2_values.eval()?;

    let idx0 = top2_indices.index(0).item::<i32>();
    let idx1 = top2_indices.index(1).item::<i32>();
    let val0 = top2_values.index(0).item::<f32>();
    let val1 = top2_values.index(1).item::<f32>();

    let (winner_id, winner_logit, runner_id, runner_logit) = if val0 >= val1 {
        (idx0, val0, idx1, val1)
    } else {
        (idx1, val1, idx0, val0)
    };

    let mut top_ids = [0i32; TOP_K];
    let mut top_logits = [0.0f32; TOP_K];
    top_ids[0] = winner_id;
    top_ids[1] = runner_id;
    top_logits[0] = winner_logit;
    top_logits[1] = runner_logit;

    let margin = winner_logit - runner_logit;

    Ok(TokenConfidence {
        token_id: winner_id,
        alternative_count: STREAMING_TOP_K as u8,
        top_ids,
        top_logits,
        concentration: margin,
        margin,
    })
}

/// Extract top-k tokens and their raw logits efficiently.
///
/// Uses argpartition (O(n) partial sort) on negated logits to find
/// top-k indices, then gathers their values. Single pass over the vocab.
///
/// Requires TOP_K >= 2 (margin needs at least two candidates).
/// Derives concentration (top1 - mean of rest) as a confidence signal.
fn topk_confidence(logits: &Array) -> Result<TokenConfidence, Exception> {
    const { assert!(TOP_K >= 2, "TOP_K must be >= 2 for margin computation") };
    let flat = logits.reshape(&[-1])?;

    // argpartition on negated logits: first k elements are the k largest
    let neg = ops::negative(&flat)?;
    let partitioned_indices = ops::argpartition(&neg, TOP_K as i32 - 1)?;

    // Take first TOP_K indices, gather their logit values, then sort descending
    let topk_indices = partitioned_indices.index(..TOP_K as i32);
    let topk_values = flat.index(&topk_indices);

    // Sort these k values descending (sort ascending on negated)
    let neg_topk = ops::negative(&topk_values)?;
    let sort_order = ops::argsort(&neg_topk)?;
    let sorted_indices = topk_indices.index(&sort_order);
    let sorted_values = topk_values.index(&sort_order);

    sorted_indices.eval()?;
    sorted_values.eval()?;

    let mut top_ids = [0i32; TOP_K];
    let mut top_logits = [0.0f32; TOP_K];
    for i in 0..TOP_K {
        top_ids[i] = sorted_indices.index(i as i32).item::<i32>();
        top_logits[i] = sorted_values.index(i as i32).item::<f32>();
    }

    // Concentration: top1 - mean(top2..topk)
    let rest_sum: f32 = top_logits[1..].iter().sum();
    let rest_mean = rest_sum / (TOP_K - 1) as f32;
    let concentration = top_logits[0] - rest_mean;

    // Margin: top1 - top2
    let margin = top_logits[0] - top_logits[1];

    Ok(TokenConfidence {
        token_id: top_ids[0],
        alternative_count: TOP_K as u8,
        top_ids,
        top_logits,
        concentration,
        margin,
    })
}

fn is_eos(token: i32) -> bool {
    EOS_TOKEN_IDS.contains(&token)
}

fn strip_eos(mut tokens: Vec<i32>) -> Vec<i32> {
    if let Some(&last) = tokens.last() {
        if is_eos(last) {
            tokens.pop();
        }
    }
    tokens
}

fn tokenize(tokenizer: &tokenizers::Tokenizer, text: &str) -> Vec<i32> {
    tokenizer
        .encode(text, false)
        .map(|enc| enc.get_ids().iter().map(|&id| id as i32).collect())
        .unwrap_or_default()
}

fn detect_repetition(tokens: &[i32]) -> bool {
    if tokens.len() < REPETITION_THRESHOLD {
        return false;
    }
    let last = tokens[tokens.len() - 1];
    tokens[tokens.len() - REPETITION_THRESHOLD..]
        .iter()
        .all(|&t| t == last)
}
