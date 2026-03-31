//! Streaming transcription with three modes:
//!
//! - **Accumulate**: Re-encode all audio each step, prefix rollback. Best quality, O(total) cost.
//! - **Overlap**: Per-chunk encoding with audio overlap + text merge. O(chunk) cost, simplest.
//! - **Rotate**: Like accumulate, but sentence commit triggers session rotation. Bounded cost.

use mlx_rs::error::Exception;
use mlx_rs::Array;

use crate::encoder::EncoderCache;
use crate::generate;
use crate::mel::MelExtractor;
use crate::model::Qwen3ASRModel;

// MLX memory management (not yet wrapped in mlx-rs)
extern "C" {
    fn mlx_clear_cache() -> std::ffi::c_int;
    fn mlx_get_active_memory(res: *mut usize) -> std::ffi::c_int;
    fn mlx_get_peak_memory(res: *mut usize) -> std::ffi::c_int;
    fn mlx_get_cache_memory(res: *mut usize) -> std::ffi::c_int;
    fn mlx_reset_peak_memory() -> std::ffi::c_int;
}

pub fn mlx_memory_stats() -> (usize, usize, usize) {
    let (mut active, mut peak, mut cache) = (0usize, 0usize, 0usize);
    unsafe {
        mlx_get_active_memory(&mut active);
        mlx_get_peak_memory(&mut peak);
        mlx_get_cache_memory(&mut cache);
    }
    (active, peak, cache)
}

fn log_memory(label: &str) {
    let (active, peak, cache) = mlx_memory_stats();
    log::info!(
        "[mem] {}: active={:.1}MB peak={:.1}MB cache={:.1}MB",
        label,
        active as f64 / 1e6,
        peak as f64 / 1e6,
        cache as f64 / 1e6,
    );
}

// ── VAD constants ───────────────────────────────────────────────────────

const VAD_WINDOW_SIZE: usize = 160; // 10ms at 16kHz
const VAD_SPEECH_RMS_THRESHOLD: f32 = 0.01;
const POST_SPEECH_SILENCE_RMS_THRESHOLD: f32 = 0.006;

// ── Streaming mode ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamingMode {
    /// Re-encode all accumulated audio each step, prefix rollback.
    Accumulate,
    /// Per-chunk encoding with audio overlap + text merge.
    Overlap,
    /// Like Accumulate, but sentence commit triggers session rotation.
    Rotate,
}

// ── StreamingOptions ────────────────────────────────────────────────────

#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct StreamingOptions {
    pub mode: StreamingMode,
    pub chunk_size_sec: f32,
    pub unfixed_chunk_num: usize,
    pub unfixed_token_num: usize,
    pub max_new_tokens_streaming: usize,
    pub max_new_tokens_final: usize,
    pub language: Option<String>,
    pub initial_text: Option<String>,
    /// Overlap duration in seconds for Overlap mode. Default: 0.5
    pub overlap_sec: f32,
    /// Number of stable updates before committing a sentence (Rotate mode). Default: 3
    pub commit_after_stable: usize,
}

impl Default for StreamingOptions {
    fn default() -> Self {
        Self {
            mode: StreamingMode::Accumulate,
            chunk_size_sec: 2.0,
            unfixed_chunk_num: 2,
            unfixed_token_num: 5,
            max_new_tokens_streaming: 32,
            max_new_tokens_final: 512,
            language: None,
            initial_text: None,
            overlap_sec: 0.5,
            commit_after_stable: 3,
        }
    }
}

impl StreamingOptions {
    pub fn with_mode(mut self, v: StreamingMode) -> Self { self.mode = v; self }
    pub fn with_chunk_size_sec(mut self, v: f32) -> Self { self.chunk_size_sec = v; self }
    pub fn with_unfixed_chunk_num(mut self, v: usize) -> Self { self.unfixed_chunk_num = v; self }
    pub fn with_unfixed_token_num(mut self, v: usize) -> Self { self.unfixed_token_num = v; self }
    pub fn with_max_new_tokens_streaming(mut self, v: usize) -> Self { self.max_new_tokens_streaming = v; self }
    pub fn with_max_new_tokens_final(mut self, v: usize) -> Self { self.max_new_tokens_final = v; self }
    pub fn with_language(mut self, v: impl Into<String>) -> Self { self.language = Some(v.into()); self }
    pub fn with_initial_text(mut self, v: impl Into<String>) -> Self { self.initial_text = Some(v.into()); self }
    pub fn with_overlap_sec(mut self, v: f32) -> Self { self.overlap_sec = v; self }
    pub fn with_commit_after_stable(mut self, v: usize) -> Self { self.commit_after_stable = v; self }
}

// ── StreamingState ──────────────────────────────────────────────────────

pub struct StreamingState {
    // Audio buffering
    buffer: Vec<f32>,
    audio_accum: Vec<f32>,
    chunk_size_samples: usize,
    pub chunk_id: usize,

    // Token state (Accumulate/Rotate modes)
    raw_token_ids: Vec<u32>,

    // Config
    pub options: StreamingOptions,
    pub language: String,
    pub text: String,

    // Encoder cache (Accumulate/Rotate modes)
    encoder_cache: EncoderCache,

    // VAD
    speech_detected: bool,

    // Tokenizer + mel
    tokenizer: tokenizers::Tokenizer,
    mel_extractor: MelExtractor,
    language_tokens: Vec<i32>,
    asr_text_tokens: Vec<i32>,

    // Overlap mode state
    prev_chunk_tail: Vec<f32>,
    overlap_samples: usize,

    // Rotate mode state
    committed_text: String,
    last_first_sentence: String,
    stable_count: usize,
}

impl StreamingState {
    pub fn new(options: StreamingOptions, tokenizer: tokenizers::Tokenizer) -> Self {
        let chunk_size_samples = (options.chunk_size_sec * 16000.0) as usize;
        let overlap_samples = (options.overlap_sec * 16000.0) as usize;
        let language = options.language.clone().unwrap_or_else(|| "English".to_string());

        let lang_header = format!("language {language}");
        let language_tokens = tokenize_to_i32(&tokenizer, &lang_header);
        let asr_text_tokens = tokenize_to_i32(&tokenizer, "<asr_text>");

        Self {
            buffer: Vec::new(),
            audio_accum: Vec::new(),
            chunk_size_samples,
            chunk_id: 0,
            raw_token_ids: Vec::new(),
            options,
            language,
            text: String::new(),
            encoder_cache: EncoderCache::new(),
            speech_detected: false,
            tokenizer,
            mel_extractor: MelExtractor::new(400, 160, 128, 16000),
            language_tokens,
            asr_text_tokens,
            prev_chunk_tail: Vec::new(),
            overlap_samples,
            committed_text: String::new(),
            last_first_sentence: String::new(),
            stable_count: 0,
        }
    }
}

fn tokenize_to_i32(tokenizer: &tokenizers::Tokenizer, text: &str) -> Vec<i32> {
    tokenizer
        .encode(text, false)
        .map(|enc| enc.get_ids().iter().map(|&id| id as i32).collect())
        .unwrap_or_default()
}

// ── Public API ──────────────────────────────────────────────────────────

pub fn feed_audio(
    model: &mut Qwen3ASRModel,
    state: &mut StreamingState,
    samples: &[f32],
) -> Result<Option<String>, Exception> {
    feed_audio_inner(model, state, samples, false)
}

pub fn feed_audio_finalizing(
    model: &mut Qwen3ASRModel,
    state: &mut StreamingState,
    samples: &[f32],
) -> Result<Option<String>, Exception> {
    feed_audio_inner(model, state, samples, true)
}

pub fn finish_streaming(
    model: &mut Qwen3ASRModel,
    state: &mut StreamingState,
) -> Result<String, Exception> {
    match state.options.mode {
        StreamingMode::Accumulate | StreamingMode::Rotate => finish_accumulate(model, state),
        StreamingMode::Overlap => finish_overlap(model, state),
    }
}

// ── Feed dispatch ───────────────────────────────────────────────────────

fn feed_audio_inner(
    model: &mut Qwen3ASRModel,
    state: &mut StreamingState,
    samples: &[f32],
    finalizing: bool,
) -> Result<Option<String>, Exception> {
    // VAD gate
    if !state.speech_detected {
        if let Some(onset) = detect_speech_onset(samples) {
            state.speech_detected = true;
            state.buffer.extend_from_slice(&samples[onset..]);
        } else {
            return Ok(None);
        }
    } else {
        state.buffer.extend_from_slice(samples);
    }

    if state.buffer.len() < state.chunk_size_samples {
        return Ok(None);
    }

    match state.options.mode {
        StreamingMode::Accumulate => feed_accumulate(model, state, finalizing),
        StreamingMode::Overlap => feed_overlap(model, state, finalizing),
        StreamingMode::Rotate => feed_rotate(model, state, finalizing),
    }
}

// ── Mode: Accumulate ────────────────────────────────────────────────────

fn feed_accumulate(
    model: &mut Qwen3ASRModel,
    state: &mut StreamingState,
    finalizing: bool,
) -> Result<Option<String>, Exception> {
    let chunk: Vec<f32> = state.buffer.drain(..state.chunk_size_samples).collect();
    state.audio_accum.extend_from_slice(&chunk);
    state.chunk_id += 1;

    if !finalizing && state.chunk_id > 1 {
        let rms = compute_rms(&chunk);
        if rms < POST_SPEECH_SILENCE_RMS_THRESHOLD {
            return Ok(None);
        }
    }

    run_accumulate_step(model, state, state.options.max_new_tokens_streaming)?;
    Ok(Some(state.text.clone()))
}

fn run_accumulate_step(
    model: &mut Qwen3ASRModel,
    state: &mut StreamingState,
    max_new_tokens: usize,
) -> Result<(), Exception> {
    let (mel_data, n_mels, n_frames) = state.mel_extractor.extract(&state.audio_accum)
        .map_err(|e| Exception::custom(format!("mel: {e}")))?;
    let mel = Array::from_slice(&mel_data, &[n_mels as i32, n_frames as i32]);
    let audio_features = model.encode_incremental(&mel, &mut state.encoder_cache)?;
    let audio_features = mlx_rs::ops::expand_dims(&audio_features, 0)?;

    let prefix_ids = compute_prefix_ids(state);

    let prompt = generate::build_initial_prompt(
        audio_features.shape()[1] as usize,
        &state.language_tokens,
        &state.asr_text_tokens,
    );

    // Append prefix tokens to prompt
    let mut full_prompt = prompt;
    let prompt_base_len = full_prompt.len();
    if let Some(prefix) = prefix_ids {
        full_prompt.extend(prefix.iter().map(|&t| t as i32));
        log::debug!(
            "chunk {}: prefix {} tokens (kept {}/{}), prompt {} + {} = {} tokens",
            state.chunk_id,
            prefix.len(),
            prefix.len(),
            state.raw_token_ids.len(),
            prompt_base_len,
            prefix.len(),
            full_prompt.len(),
        );
    } else {
        log::debug!("chunk {}: no prefix (cold start), prompt {} tokens", state.chunk_id, full_prompt.len());
    }

    let mut cache = None;
    let (generated, _) = generate::prefill_and_decode(
        model,
        &full_prompt,
        &audio_features,
        &mut cache,
        0,
        max_new_tokens,
    )?;
    // Explicitly drop the KV cache and clear MLX memory cache
    drop(cache);
    unsafe { mlx_clear_cache(); }
    log_memory(&format!("after chunk {}", state.chunk_id));

    log::debug!("chunk {}: generated {} tokens: {:?}", state.chunk_id, generated.len(), &generated[..generated.len().min(10)]);

    let all_ids = combine_prefix_and_generated(state, &generated);

    // Debug: decode prefix, generated, and combined separately
    if log::log_enabled!(log::Level::Debug) {
        let gen_u32: Vec<u32> = generated.iter().map(|&t| t as u32).collect();
        let gen_text = state.tokenizer.decode(&gen_u32, true).unwrap_or_default();
        let all_u32: Vec<u32> = all_ids.iter().map(|&t| t as u32).collect();
        let all_text = state.tokenizer.decode(&all_u32, true).unwrap_or_default();
        log::debug!("chunk {}: gen_text={:?}", state.chunk_id, gen_text);
        log::debug!("chunk {}: all_text={:?}", state.chunk_id, all_text);
    }

    let ids_u32: Vec<u32> = all_ids.iter().map(|&t| t as u32).collect();
    let text = state.tokenizer.decode(&ids_u32, true).unwrap_or_default();

    state.raw_token_ids = ids_u32;
    state.text = text;
    Ok(())
}

fn finish_accumulate(
    model: &mut Qwen3ASRModel,
    state: &mut StreamingState,
) -> Result<String, Exception> {
    if !state.buffer.is_empty() {
        state.audio_accum.extend(state.buffer.drain(..));
        state.chunk_id += 1;
    }
    if state.audio_accum.is_empty() {
        return Ok(state.text.clone());
    }
    run_accumulate_step(model, state, state.options.max_new_tokens_final)?;

    // For Rotate mode, merge committed text with final session text
    if state.options.mode == StreamingMode::Rotate && !state.committed_text.is_empty() {
        state.text = append_chunk_text(&state.committed_text, &state.text);
    }

    Ok(state.text.clone())
}

fn compute_prefix_ids(state: &StreamingState) -> Option<&[u32]> {
    if state.chunk_id <= state.options.unfixed_chunk_num {
        return None;
    }
    if state.raw_token_ids.is_empty() {
        return None;
    }
    let keep = state.raw_token_ids.len().saturating_sub(state.options.unfixed_token_num);
    if keep == 0 {
        return None;
    }
    Some(&state.raw_token_ids[..keep])
}

fn combine_prefix_and_generated(state: &StreamingState, generated: &[i32]) -> Vec<i32> {
    if state.raw_token_ids.is_empty() || state.chunk_id <= state.options.unfixed_chunk_num {
        return generated.to_vec();
    }
    let keep = state.raw_token_ids.len().saturating_sub(state.options.unfixed_token_num);
    if keep == 0 {
        return generated.to_vec();
    }
    let mut combined: Vec<i32> = state.raw_token_ids[..keep].iter().map(|&t| t as i32).collect();
    combined.extend_from_slice(generated);
    combined
}

// ── Mode: Overlap ───────────────────────────────────────────────────────

fn feed_overlap(
    model: &mut Qwen3ASRModel,
    state: &mut StreamingState,
    finalizing: bool,
) -> Result<Option<String>, Exception> {
    let chunk: Vec<f32> = state.buffer.drain(..state.chunk_size_samples).collect();
    state.chunk_id += 1;

    if !finalizing && state.chunk_id > 1 {
        let rms = compute_rms(&chunk);
        if rms < POST_SPEECH_SILENCE_RMS_THRESHOLD {
            return Ok(None);
        }
    }

    // Build chunk with overlap prepended
    let mut encode_samples = Vec::new();
    if !state.prev_chunk_tail.is_empty() {
        encode_samples.extend_from_slice(&state.prev_chunk_tail);
    }
    encode_samples.extend_from_slice(&chunk);

    // Save tail for next chunk's overlap
    let overlap = state.overlap_samples.min(chunk.len());
    state.prev_chunk_tail = chunk[chunk.len() - overlap..].to_vec();

    // Encode just this chunk (with overlap)
    let (mel_data, n_mels, n_frames) = state.mel_extractor.extract(&encode_samples)
        .map_err(|e| Exception::custom(format!("mel: {e}")))?;
    let mel = Array::from_slice(&mel_data, &[n_mels as i32, n_frames as i32]);
    let audio_features = model.encode_audio(&mel)?;
    let audio_features = mlx_rs::ops::expand_dims(&audio_features, 0)?;

    let prompt = generate::build_initial_prompt(
        audio_features.shape()[1] as usize,
        &state.language_tokens,
        &state.asr_text_tokens,
    );

    let (generated, _) = generate::prefill_and_decode(
        model,
        &prompt,
        &audio_features,
        &mut None,
        0,
        state.options.max_new_tokens_streaming,
    )?;

    let ids_u32: Vec<u32> = generated.iter().map(|&t| t as u32).collect();
    let chunk_text = state.tokenizer.decode(&ids_u32, true).unwrap_or_default();

    // Merge with existing text
    log::debug!("chunk {}: chunk_text={:?}", state.chunk_id, chunk_text);
    let prev = state.text.clone();
    state.text = append_chunk_text(&state.text, &chunk_text);
    log::debug!("chunk {}: merged={:?} (prev={:?})", state.chunk_id, state.text, prev);
    Ok(Some(state.text.clone()))
}

fn finish_overlap(
    model: &mut Qwen3ASRModel,
    state: &mut StreamingState,
) -> Result<String, Exception> {
    if state.buffer.is_empty() {
        return Ok(state.text.clone());
    }

    let tail: Vec<f32> = state.buffer.drain(..).collect();
    state.chunk_id += 1;

    let mut encode_samples = Vec::new();
    if !state.prev_chunk_tail.is_empty() {
        encode_samples.extend_from_slice(&state.prev_chunk_tail);
    }
    encode_samples.extend_from_slice(&tail);

    let (mel_data, n_mels, n_frames) = state.mel_extractor.extract(&encode_samples)
        .map_err(|e| Exception::custom(format!("mel: {e}")))?;
    let mel = Array::from_slice(&mel_data, &[n_mels as i32, n_frames as i32]);
    let audio_features = model.encode_audio(&mel)?;
    let audio_features = mlx_rs::ops::expand_dims(&audio_features, 0)?;

    let prompt = generate::build_initial_prompt(
        audio_features.shape()[1] as usize,
        &state.language_tokens,
        &state.asr_text_tokens,
    );

    let (generated, _) = generate::prefill_and_decode(
        model,
        &prompt,
        &audio_features,
        &mut None,
        0,
        state.options.max_new_tokens_final,
    )?;

    let ids_u32: Vec<u32> = generated.iter().map(|&t| t as u32).collect();
    let chunk_text = state.tokenizer.decode(&ids_u32, true).unwrap_or_default();
    state.text = append_chunk_text(&state.text, &chunk_text);
    Ok(state.text.clone())
}

/// Append new chunk text with overlap dedup (port from Python _append_chunk_text).
fn append_chunk_text(current: &str, addition: &str) -> String {
    let curr = current.trim();
    let add = addition.trim();
    if add.is_empty() { return curr.to_string(); }
    if curr.is_empty() { return add.to_string(); }
    if curr == add || curr.ends_with(add) { return curr.to_string(); }
    if add.starts_with(curr) { return add.to_string(); }

    let curr_words: Vec<&str> = curr.split_whitespace().collect();
    let add_words: Vec<&str> = add.split_whitespace().collect();

    // Superset detection: if first 3 words match and addition is longer, replace
    let prefix_check = 3.min(curr_words.len()).min(add_words.len());
    if prefix_check > 0 && curr_words[..prefix_check] == add_words[..prefix_check] {
        if add_words.len() >= curr_words.len() {
            return add.to_string();
        }
    }

    // Overlap detection: find longest suffix of current that matches prefix of addition
    let max_overlap = curr_words.len().min(add_words.len());
    for k in (1..=max_overlap).rev() {
        if curr_words[curr_words.len() - k..] == add_words[..k] {
            let mut merged = curr_words.clone();
            merged.extend_from_slice(&add_words[k..]);
            return merged.join(" ");
        }
    }

    // Fallback: concatenate with space
    format!("{curr} {add}")
}

// ── Mode: Rotate ────────────────────────────────────────────────────────

fn feed_rotate(
    model: &mut Qwen3ASRModel,
    state: &mut StreamingState,
    finalizing: bool,
) -> Result<Option<String>, Exception> {
    // Run normal accumulate step
    let result = feed_accumulate(model, state, finalizing)?;

    if result.is_some() {
        let current_text = state.text.clone();
        if let Some(first_sentence) = find_first_complete_sentence(&current_text) {
            if first_sentence == state.last_first_sentence {
                state.stable_count += 1;
            } else {
                state.last_first_sentence = first_sentence.to_string();
                state.stable_count = 1;
            }

            // Need at least 2 sentences and stable updates before committing
            let has_more = current_text.len() > first_sentence.len() + 1;

            if state.stable_count >= state.options.commit_after_stable && has_more {
                let committed_sentence = state.last_first_sentence.clone();

                // Estimate audio boundary using proportional token count.
                // TODO: replace with forced aligner for exact boundaries.
                let committed_audio_samples = estimate_audio_boundary(
                    &committed_sentence,
                    &current_text,
                    state.audio_accum.len(),
                );

                // Append to committed text
                if !state.committed_text.is_empty() {
                    state.committed_text = format!("{} {}", state.committed_text, committed_sentence);
                } else {
                    state.committed_text = committed_sentence.clone();
                }

                // Keep audio after the committed boundary
                let keep_from = committed_audio_samples.min(state.audio_accum.len());
                let remaining_audio: Vec<f32> = state.audio_accum[keep_from..].to_vec();

                log::info!(
                    "Session rotated. Committed: {:?} | audio: kept {:.1}s of {:.1}s (boundary at {:.1}s)",
                    state.committed_text,
                    remaining_audio.len() as f64 / 16000.0,
                    state.audio_accum.len() as f64 / 16000.0,
                    keep_from as f64 / 16000.0,
                );

                // Tokenize last ~200 chars of committed text for prefix context
                let context_text = if state.committed_text.len() > 200 {
                    &state.committed_text[state.committed_text.len() - 200..]
                } else {
                    &state.committed_text
                };
                let context_ids: Vec<u32> = state.tokenizer
                    .encode(context_text, false)
                    .map(|enc| enc.get_ids().to_vec())
                    .unwrap_or_default();

                // Reset streaming state for new session with remaining audio
                state.audio_accum = remaining_audio;
                state.encoder_cache = EncoderCache::new();
                state.raw_token_ids = context_ids;
                state.chunk_id = state.options.unfixed_chunk_num + 1;
                state.stable_count = 0;
                state.last_first_sentence.clear();

                unsafe { mlx_clear_cache(); }
                log_memory("after rotation");
            }
        }

        // Display: committed text + current session text
        if !state.committed_text.is_empty() {
            let display = append_chunk_text(&state.committed_text, &state.text);
            state.text = display;
        }
    }

    Ok(Some(state.text.clone()))
}

/// Estimate how many audio samples correspond to the committed text.
/// Uses proportional character count as approximation.
/// TODO: replace with forced aligner for precise word-level timestamps.
fn estimate_audio_boundary(
    committed_text: &str,
    full_text: &str,
    total_audio_samples: usize,
) -> usize {
    let committed_chars = committed_text.trim().len();
    let total_chars = full_text.trim().len();
    if total_chars == 0 {
        return 0;
    }
    let fraction = committed_chars as f64 / total_chars as f64;
    // Add a small margin (0.5s) to ensure we keep enough audio context at the boundary
    let boundary = (fraction * total_audio_samples as f64) as usize;
    // Clamp: don't go past the audio, and leave at least 1s
    boundary.min(total_audio_samples.saturating_sub(16000))
}

/// Find the first complete sentence (ending with . ! ? or similar).
fn find_first_complete_sentence(text: &str) -> Option<&str> {
    let text = text.trim();
    for (i, c) in text.char_indices() {
        if (c == '.' || c == '!' || c == '?') && i > 0 {
            // Make sure it's not an abbreviation (check next char is space/end)
            let rest = &text[i + c.len_utf8()..];
            if rest.is_empty() || rest.starts_with(' ') || rest.starts_with('\n') {
                return Some(&text[..=i]);
            }
        }
    }
    None
}

// ── VAD ─────────────────────────────────────────────────────────────────

fn detect_speech_onset(samples: &[f32]) -> Option<usize> {
    let mut consecutive_speech = 0;
    let mut first_speech_idx = 0;

    for (i, window) in samples.chunks(VAD_WINDOW_SIZE).enumerate() {
        let rms = compute_rms(window);
        if rms >= VAD_SPEECH_RMS_THRESHOLD {
            if consecutive_speech == 0 {
                first_speech_idx = i * VAD_WINDOW_SIZE;
            }
            consecutive_speech += 1;
            if consecutive_speech >= 2 {
                return Some(first_speech_idx);
            }
        } else {
            consecutive_speech = 0;
        }
    }
    None
}

fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}
