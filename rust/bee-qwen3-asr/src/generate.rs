//! Autoregressive text generation for Qwen3-ASR.

use mlx_rs::error::Exception;
use mlx_rs::nn;
use mlx_rs::ops;
use mlx_rs::ops::indexing::{self, IndexOp};
use mlx_rs::Array;

use crate::decoder::KVCache;
use crate::model::{Qwen3ASRModel, EOS_TOKEN_IDS};
use crate::model::{AUDIO_END_TOKEN_ID, AUDIO_PAD_TOKEN_ID, AUDIO_START_TOKEN_ID};

/// Per-token confidence information extracted during decoding.
#[derive(Debug, Clone, Copy)]
pub struct TokenLogprob {
    /// The chosen token ID.
    pub token_id: i32,
    /// Log-probability of the chosen token.
    pub logprob: f32,
    /// Gap between the top-1 and top-2 log-probabilities (always >= 0).
    pub margin: f32,
}

const REPETITION_THRESHOLD: usize = 20;

// Chat template token IDs
pub const TOK_IM_START: i32 = 151644;
pub const TOK_IM_END: i32 = 151645;
pub const TOK_SYSTEM: i32 = 8948;
pub const TOK_USER: i32 = 872;
pub const TOK_ASSISTANT: i32 = 77091;
pub const TOK_NEWLINE: i32 = 198;

/// Greedy autoregressive generation (batch mode, fresh cache).
pub fn generate(
    model: &Qwen3ASRModel,
    input_ids: &Array,
    audio_features: &Array,
    position_ids: &Array,
    max_new_tokens: usize,
) -> Result<Vec<i32>, Exception> {
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
///   <|im_start|>system\n<|im_end|>\n<|im_start|>user\n
///   <|audio_start|><|audio_pad|>*N<|audio_end|><|im_end|>\n
///   <|im_start|>assistant\n
///   language {lang}<asr_text>
pub fn build_initial_prompt(
    n_audio_tokens: usize,
    language_tokens: &[i32],
    asr_text_tokens: &[i32],
) -> Vec<i32> {
    let mut prompt: Vec<i32> = vec![
        TOK_IM_START,
        TOK_SYSTEM,
        TOK_NEWLINE,
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
    prompt.extend_from_slice(language_tokens);
    prompt.extend_from_slice(asr_text_tokens);
    prompt
}

/// Build the follow-up prompt for subsequent streaming chunks.
///
/// Template:
///   <|im_end|>\n<|im_start|>user\n
///   <|audio_start|><|audio_pad|>*N<|audio_end|><|im_end|>\n
///   <|im_start|>assistant\n
///   language {lang}<asr_text>
pub fn build_followup_prompt(
    n_audio_tokens: usize,
    language_tokens: &[i32],
    asr_text_tokens: &[i32],
) -> Vec<i32> {
    let mut prompt: Vec<i32> = vec![
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
    prompt.extend_from_slice(language_tokens);
    prompt.extend_from_slice(asr_text_tokens);
    prompt
}

/// Prefill a prompt (initial or followup) into an existing cache, then
/// autoregressively decode up to max_new_tokens.
///
/// Returns generated token IDs, per-token logprob info, and the updated
/// next_position. Position tracking: the cache contains exactly the prompt
/// tokens plus each non-EOS generated token that was stepped through. EOS
/// tokens are NOT added to the cache (matching the Python reference).
pub fn prefill_and_decode(
    model: &Qwen3ASRModel,
    prompt_tokens: &[i32],
    audio_features: &Array,
    cache: &mut Option<KVCache>,
    start_position: usize,
    max_new_tokens: usize,
) -> Result<(Vec<i32>, Vec<TokenLogprob>, usize), Exception> {
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

    let tlp = argmax_with_logprob(&logits)?;
    let mut token = tlp.token_id;
    let mut generated = Vec::new();
    let mut logprobs = Vec::new();

    tracing::debug!(
        "prefill_and_decode: first_token={token} logprob={:.3} is_eos={} prompt_len={seq_len} max_new={max_new_tokens}",
        tlp.logprob,
        is_eos(token),
    );

    if is_eos(token) || max_new_tokens <= 1 {
        // EOS on first token — nothing added to cache beyond prompt
        if !is_eos(token) {
            generated.push(token);
            logprobs.push(tlp);
        }
        return Ok((generated, logprobs, position));
    }

    generated.push(token);
    logprobs.push(tlp);

    // Autoregressive decode — each step adds the token to the cache
    for _ in 1..max_new_tokens {
        let next_ids = Array::from_slice(&[token], &[1, 1]);
        let pos_arr = Array::from_slice(
            &[position as i32, position as i32, position as i32],
            &[1, 3, 1],
        );

        let logits = model.step(&next_ids, &pos_arr, cache)?;
        position += 1; // token was fed into cache

        let tlp = argmax_with_logprob(&logits)?;
        token = tlp.token_id;

        if is_eos(token) {
            break; // EOS not fed into cache
        }

        generated.push(token);
        logprobs.push(tlp);

        if detect_repetition(&generated) {
            break;
        }

        if generated.len() % 8 == 0 {
            logits.eval()?;
        }
    }

    Ok((generated, logprobs, position))
}

fn argmax(logits: &Array) -> Result<i32, Exception> {
    let flat = logits.reshape(&[-1])?;
    let idx = indexing::argmax(&flat, None)?;
    Ok(idx.item::<i32>())
}

/// Extract the argmax token along with its logprob and top1-top2 margin.
fn argmax_with_logprob(logits: &Array) -> Result<TokenLogprob, Exception> {
    let flat = logits.reshape(&[-1])?;
    let idx = indexing::argmax(&flat, None)?;
    let token_id = idx.item::<i32>();

    // Compute log-softmax, then index the chosen token's logprob directly
    let log_probs = nn::log_softmax(&flat, -1)?;
    log_probs.eval()?;
    let top1_lp = log_probs.index(token_id).item::<f32>();

    // For margin: get top-2 values (unsorted!), then sort descending
    let top2 = indexing::topk(&log_probs, 2)?;
    top2.eval()?;
    let a = top2.index(0).item::<f32>();
    let b = top2.index(1).item::<f32>();
    let margin = if a >= b { a - b } else { b - a };

    Ok(TokenLogprob {
        token_id,
        logprob: top1_lp,
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

fn detect_repetition(tokens: &[i32]) -> bool {
    if tokens.len() < REPETITION_THRESHOLD {
        return false;
    }
    let last = tokens[tokens.len() - 1];
    tokens[tokens.len() - REPETITION_THRESHOLD..]
        .iter()
        .all(|&t| t == last)
}
