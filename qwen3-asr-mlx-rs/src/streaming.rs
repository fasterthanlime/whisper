//! Streaming transcription with three modes:
//!
//! - **Accumulate**: Re-encode all audio each step, prefix rollback. Best quality, O(total) cost.
//! - **Overlap**: Per-chunk encoding with audio overlap + text merge. O(chunk) cost, simplest.
//! - **Rotate**: Like accumulate, but sentence commit triggers session rotation. Bounded cost.

use mlx_rs::error::Exception;
use mlx_rs::Array;

use crate::encoder::EncoderCache;
use crate::forced_aligner::ForcedAligner;
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

fn stream_log(msg: &str) {
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/bee.log") {
        let _ = writeln!(f, "[{:.3}] {}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs_f64(), msg);
    }
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
    /// Like Rotate, but reuses decoder KV cache across steps via multi-turn prompts.
    /// Only prefills new audio each step instead of the full prompt.
    RotateCached,
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
    /// Number of stable updates before committing the prefix (Rotate mode). Default: 3
    pub commit_after_stable: usize,
    /// Number of leading tokens to commit (Rotate mode). Default: 12
    pub commit_token_count: usize,
    /// Minimum trailing tokens beyond commit_token_count before committing. Default: 6
    pub min_trailing_tokens: usize,
    /// VAD speech probability threshold (0.0–1.0). Default: 0.5
    pub vad_threshold: f32,
}

impl Default for StreamingOptions {
    fn default() -> Self {
        Self {
            mode: StreamingMode::Accumulate,
            chunk_size_sec: 0.4,
            unfixed_chunk_num: 2,
            unfixed_token_num: 5,
            max_new_tokens_streaming: 32,
            max_new_tokens_final: 512,
            language: None,
            initial_text: None,
            overlap_sec: 0.5,
            commit_after_stable: 3,
            commit_token_count: 12,
            min_trailing_tokens: 6,
            vad_threshold: 0.5,
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
    pub fn with_commit_token_count(mut self, v: usize) -> Self { self.commit_token_count = v; self }
    pub fn with_min_trailing_tokens(mut self, v: usize) -> Self { self.min_trailing_tokens = v; self }
    pub fn with_vad_threshold(mut self, v: f32) -> Self { self.vad_threshold = v; self }
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

    // TODO: mel caching (skip recompute when audio hasn't grown)
    // Not implemented yet — mel is ~1% of runtime per profiling,
    // and rotate mode keeps sessions short.

    // VAD
    speech_detected: bool,
    pub vad: Option<crate::vad::SileroVad>,

    // Tokenizer + mel
    tokenizer: tokenizers::Tokenizer,
    mel_extractor: MelExtractor,
    language_tokens: Vec<i32>,
    asr_text_tokens: Vec<i32>,

    // Overlap mode state
    prev_chunk_tail: Vec<f32>,
    overlap_samples: usize,

    // Rotate mode state
    pub committed_text: String,
    committed_token_ids: Vec<u32>,
    /// The first `commit_token_count` tokens from the last update, for stability tracking.
    last_prefix_tokens: Vec<u32>,
    /// How many consecutive updates the prefix tokens have been identical.
    stable_count: usize,

    /// Forced aligner for precise audio boundaries (Rotate mode).
    aligner: Option<ForcedAligner>,

    /// Accumulated word-level alignments from all commits.
    pub committed_alignments: Vec<crate::forced_aligner::ForcedAlignItem>,
    /// Audio time (seconds) trimmed before current audio_accum. Used to offset aligner timestamps.
    committed_audio_offset: f64,
    /// Debug events accumulated since last drain.
    pub debug_events: Vec<String>,

    /// Persistent decoder KV cache for RotateCached mode.
    decoder_cache: Option<crate::decoder::KVCache>,
    /// Next position in the decoder KV cache.
    decoder_next_pos: usize,
    /// Whether first step in current session (needs full prefill vs followup).
    is_first_step: bool,
}

impl StreamingState {
    pub fn new(options: StreamingOptions, tokenizer: tokenizers::Tokenizer, aligner: Option<ForcedAligner>) -> Self {
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
            vad: None,
            tokenizer,
            mel_extractor: MelExtractor::new(400, 160, 128, 16000),
            language_tokens,
            asr_text_tokens,
            prev_chunk_tail: Vec::new(),
            overlap_samples,
            committed_text: String::new(),
            committed_token_ids: Vec::new(),
            last_prefix_tokens: Vec::new(),
            stable_count: 0,
            aligner,
            committed_alignments: Vec::new(),
            committed_audio_offset: 0.0,
            debug_events: Vec::new(),
            decoder_cache: None,
            decoder_next_pos: 0,
            is_first_step: true,
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
        StreamingMode::Accumulate | StreamingMode::Rotate | StreamingMode::RotateCached => finish_accumulate(model, state),
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
    // VAD gate: use Silero VAD if available, else fall back to RMS
    if !state.speech_detected {
        if let Some(ref mut vad) = state.vad {
            let prob = vad.process_audio(samples)
                .unwrap_or(0.0);
            if prob >= state.options.vad_threshold {
                state.speech_detected = true;
                state.buffer.extend_from_slice(samples);
            } else {
                return Ok(None);
            }
        } else if let Some(onset) = detect_speech_onset(samples) {
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
        StreamingMode::RotateCached => feed_rotate_cached(model, state, finalizing),
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
        let (is_silence, vad_prob) = if let Some(ref mut vad) = state.vad {
            let prob = vad.process_audio(&chunk).unwrap_or(0.0);
            (prob < state.options.vad_threshold, Some(prob))
        } else {
            (compute_rms(&chunk) < POST_SPEECH_SILENCE_RMS_THRESHOLD, None)
        };
        if let Some(prob) = vad_prob {
            state.debug_events.push(format!(
                r#"{{"type":"vad","chunk":{},"prob":{:.3},"speech":{}}}"#,
                state.chunk_id, prob, !is_silence,
            ));
        }
        if is_silence {
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
    let t_step = std::time::Instant::now();

    let t0 = std::time::Instant::now();
    let (mel_data, n_mels, n_frames) = state.mel_extractor.extract(&state.audio_accum)
        .map_err(|e| Exception::custom(format!("mel: {e}")))?;
    let mel_ms = t0.elapsed().as_millis();

    let t0 = std::time::Instant::now();
    let mel = Array::from_slice(&mel_data, &[n_mels as i32, n_frames as i32]);
    let audio_features = model.encode_incremental(&mel, &mut state.encoder_cache)?;
    let audio_features = mlx_rs::ops::expand_dims(&audio_features, 0)?;
    let enc_ms = t0.elapsed().as_millis();

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

    let t0 = std::time::Instant::now();
    let mut cache = None;
    let (generated, _) = generate::prefill_and_decode(
        model,
        &full_prompt,
        &audio_features,
        &mut cache,
        0,
        max_new_tokens,
    )?;
    let decode_ms = t0.elapsed().as_millis();
    // Explicitly drop the KV cache and clear MLX memory cache
    drop(cache);
    unsafe { mlx_clear_cache(); }

    log::info!(
        "step {}: mel={:.0}ms enc={:.0}ms decode={:.0}ms total={:.0}ms (audio={:.1}s, prompt={} tokens, gen={} tokens)",
        state.chunk_id, mel_ms, enc_ms, decode_ms, t_step.elapsed().as_millis(),
        state.audio_accum.len() as f64 / 16000.0, full_prompt.len(), generated.len(),
    );

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

    // For Rotate mode, merge committed text with final session text (token-deduped)
    if state.options.mode == StreamingMode::Rotate && !state.committed_token_ids.is_empty() {
        let skip = find_token_overlap(&state.committed_token_ids, &state.raw_token_ids);
        let remaining_ids: Vec<u32> = state.raw_token_ids[skip..].to_vec();
        let new_text = state.tokenizer.decode(&remaining_ids, true).unwrap_or_default();
        state.text = join_committed(&state.committed_text, &new_text);
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
        let is_silence = if let Some(ref mut vad) = state.vad {
            let prob = vad.process_audio(&chunk).unwrap_or(0.0);
            prob < state.options.vad_threshold
        } else {
            compute_rms(&chunk) < POST_SPEECH_SILENCE_RMS_THRESHOLD
        };
        if is_silence {
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
    let result = feed_accumulate(model, state, finalizing)?;

    if result.is_some() && !state.raw_token_ids.is_empty() {
        // Track stability of the first N tokens
        let n = state.options.commit_token_count.min(state.raw_token_ids.len());
        let current_prefix: Vec<u32> = state.raw_token_ids[..n].to_vec();

        if current_prefix == state.last_prefix_tokens {
            state.stable_count += 1;
        } else {
            state.last_prefix_tokens = current_prefix.clone();
            state.stable_count = 1;
        }

        log::debug!(
            "chunk {}: {} tokens, prefix stable for {}/{} rounds (need {} tokens)",
            state.chunk_id, state.raw_token_ids.len(), state.stable_count,
            state.options.commit_after_stable, state.options.commit_token_count,
        );
        let total = state.raw_token_ids.len();
        let need = state.options.commit_token_count + state.options.min_trailing_tokens;
        stream_log(&format!(
            "chunk {}: {} tokens (need {}), prefix[..{}] stable {}/{}, mode={:?}",
            state.chunk_id, total, need, n, state.stable_count,
            state.options.commit_after_stable, state.options.mode,
        ));

        // Commit when prefix is stable AND we have enough tokens
        if state.stable_count >= state.options.commit_after_stable
            && n >= state.options.commit_token_count
            && state.raw_token_ids.len() >= state.options.commit_token_count + state.options.min_trailing_tokens
        {
            // Back up to a word boundary so we never commit mid-word.
            let Some((n, current_prefix, committed_text_new)) =
                find_word_boundary_commit(&state.raw_token_ids, n, &state.tokenizer)
            else {
                // No word boundary found in these tokens — reset stability
                // and wait for more tokens.
                state.stable_count = 0;
                state.last_prefix_tokens.clear();
                return Ok(Some(state.text.clone()));
            };

            // Find audio boundary and store alignments
            let committed_audio_samples = if let Some(ref mut aligner) = state.aligner {
                let t_align = std::time::Instant::now();
                let result = aligner.align(&state.audio_accum, &committed_text_new);
                let align_ms = t_align.elapsed().as_millis();
                log::info!("Aligner took {align_ms}ms on {:.1}s audio",
                    state.audio_accum.len() as f64 / 16000.0);
                match result {
                    Ok(items) if !items.is_empty() => {
                        let last_word = &items[items.len() - 1];
                        let samples = (last_word.end_time * 16000.0) as usize;
                        log::info!("Aligner: boundary at {:.3}s, last word: {:?}",
                            last_word.end_time, last_word.word);
                        // Store alignments with absolute timestamps
                        let offset = state.committed_audio_offset;
                        for item in &items {
                            state.committed_alignments.push(crate::forced_aligner::ForcedAlignItem {
                                word: item.word.clone(),
                                start_time: item.start_time + offset,
                                end_time: item.end_time + offset,
                            });
                        }
                        state.debug_events.push(format!(
                            r#"{{"type":"align","ms":{},"words":{},"boundary":{:.3}}}"#,
                            align_ms, items.len(), last_word.end_time + offset,
                        ));
                        samples
                    }
                    _ => estimate_audio_boundary(&committed_text_new, &state.text, state.audio_accum.len()),
                }
            } else {
                estimate_audio_boundary(&committed_text_new, &state.text, state.audio_accum.len())
            };

            // Append to committed token IDs and text
            state.committed_token_ids.extend_from_slice(&current_prefix);
            if !state.committed_text.is_empty() {
                state.committed_text = join_committed(&state.committed_text, &committed_text_new);
            } else {
                state.committed_text = committed_text_new;
            }

            let keep_from = committed_audio_samples.min(state.audio_accum.len());
            let trimmed_secs = keep_from as f64 / 16000.0;
            state.committed_audio_offset += trimmed_secs;
            let remaining_audio: Vec<f32> = state.audio_accum[keep_from..].to_vec();

            state.debug_events.push(format!(
                r#"{{"type":"commit","tokens":{},"text":{},"audio_offset":{:.3}}}"#,
                n, serde_json::to_string(&state.committed_text).unwrap_or_default(),
                state.committed_audio_offset,
            ));

            log::info!(
                "Committed {} tokens: {:?} | audio: kept {:.1}s of {:.1}s (boundary at {:.1}s)",
                n, state.committed_text,
                remaining_audio.len() as f64 / 16000.0,
                state.audio_accum.len() as f64 / 16000.0,
                keep_from as f64 / 16000.0,
            );
            stream_log(&format!(
                "COMMIT {} tokens: {:?} | kept {:.1}s of {:.1}s",
                n, state.committed_text,
                remaining_audio.len() as f64 / 16000.0,
                state.audio_accum.len() as f64 / 16000.0,
            ));

            // Keep tokens for the uncommitted tail
            let remaining_tokens: Vec<u32> = state.raw_token_ids[n..].to_vec();
            log::info!(
                "Seeding new session with {} remaining tokens",
                remaining_tokens.len(),
            );

            // Reset for new session — seed with remaining tokens + audio
            state.audio_accum = remaining_audio;
            state.encoder_cache = EncoderCache::new();
            state.raw_token_ids = remaining_tokens;
            // Skip cold start so prefix rollback kicks in immediately
            state.chunk_id = state.options.unfixed_chunk_num + 1;
            state.stable_count = 0;
            state.last_prefix_tokens.clear();

            unsafe { mlx_clear_cache(); }
            log_memory("after rotation");
        }

        // Display: committed text + current session text, with token-based dedup
        if !state.committed_token_ids.is_empty() {
            // Strip any leading tokens in the new session that overlap with
            // committed tokens (the model may re-produce them from overlapping audio).
            let new_ids = &state.raw_token_ids;
            let skip = find_token_overlap(&state.committed_token_ids, new_ids);
            if skip > 0 {
                stream_log(&format!("TOKEN DEDUP: stripping {} overlapping tokens from session start", skip));
            }
            // Decode the new (non-overlapping) tokens separately, then join at text level
            // to preserve proper spacing (concatenating token IDs loses inter-segment spaces).
            let remaining_ids: Vec<u32> = new_ids[skip..].to_vec();
            let new_text = state.tokenizer.decode(&remaining_ids, true).unwrap_or_default();
            state.text = join_committed(&state.committed_text, &new_text);
        }
    }

    Ok(Some(state.text.clone()))
}

/// Estimate how many audio samples correspond to the committed text.
/// Uses proportional character count as approximation.
/// TODO: replace with forced aligner for precise word-level timestamps.
// ── Mode: RotateCached ──────────────────────────────────────────────────

fn feed_rotate_cached(
    model: &mut Qwen3ASRModel,
    state: &mut StreamingState,
    finalizing: bool,
) -> Result<Option<String>, Exception> {
    // VAD + buffering already handled by feed_audio_inner, but we need the chunk
    let chunk: Vec<f32> = state.buffer.drain(..state.chunk_size_samples).collect();
    state.chunk_id += 1;

    if !finalizing && state.chunk_id > 1 {
        let is_silence = if let Some(ref mut vad) = state.vad {
            let prob = vad.process_audio(&chunk).unwrap_or(0.0);
            prob < state.options.vad_threshold
        } else {
            compute_rms(&chunk) < POST_SPEECH_SILENCE_RMS_THRESHOLD
        };
        if is_silence {
            return Ok(None);
        }
    }

    // Encode just this chunk's audio
    let t_step = std::time::Instant::now();
    let t0 = std::time::Instant::now();

    let (mel_data, n_mels, n_frames) = state.mel_extractor.extract(&chunk)
        .map_err(|e| Exception::custom(format!("mel: {e}")))?;
    let mel = Array::from_slice(&mel_data, &[n_mels as i32, n_frames as i32]);
    let audio_features = model.encode_audio(&mel)?;
    let audio_features = mlx_rs::ops::expand_dims(&audio_features, 0)?;
    let n_audio_tokens = audio_features.shape()[1] as usize;
    let enc_ms = t0.elapsed().as_millis();

    // Also keep accumulated audio for the aligner (it needs the full session audio)
    state.audio_accum.extend_from_slice(&chunk);

    // Build prompt: full on first step, followup on subsequent
    let prompt = if state.is_first_step {
        state.is_first_step = false;
        generate::build_initial_prompt(
            n_audio_tokens,
            &state.language_tokens,
            &state.asr_text_tokens,
        )
    } else {
        // Truncate KV cache: roll back unfixed tokens from previous generation
        if let Some(ref mut cache) = state.decoder_cache {
            let rollback = state.options.unfixed_token_num.min(
                state.decoder_next_pos.saturating_sub(1)
            );
            if rollback > 0 {
                let new_len = state.decoder_next_pos - rollback;
                cache.truncate(new_len);
                state.decoder_next_pos = new_len;
                // Also trim the raw token IDs
                let keep = state.raw_token_ids.len().saturating_sub(rollback);
                state.raw_token_ids.truncate(keep);
            }
        }
        generate::build_followup_prompt(
            n_audio_tokens,
            &state.language_tokens,
            &state.asr_text_tokens,
        )
    };

    // Prefill + decode with persistent cache
    let t0 = std::time::Instant::now();
    let (generated, new_pos) = generate::prefill_and_decode(
        model,
        &prompt,
        &audio_features,
        &mut state.decoder_cache,
        state.decoder_next_pos,
        state.options.max_new_tokens_streaming,
    )?;
    let decode_ms = t0.elapsed().as_millis();
    state.decoder_next_pos = new_pos;

    // Accumulate tokens
    state.raw_token_ids.extend(generated.iter().map(|&t| t as u32));

    // Decode all tokens to text
    let text = state.tokenizer
        .decode(&state.raw_token_ids, true)
        .unwrap_or_default();
    state.text = text;

    log::info!(
        "step {}: enc={:.0}ms decode={:.0}ms total={:.0}ms (prompt={} tokens, gen={} tokens, cache_pos={})",
        state.chunk_id, enc_ms, decode_ms, t_step.elapsed().as_millis(),
        prompt.len(), generated.len(), state.decoder_next_pos,
    );

    // --- Commit logic (same as rotate mode) ---
    if !state.raw_token_ids.is_empty() {
        let n = state.options.commit_token_count.min(state.raw_token_ids.len());
        let current_prefix: Vec<u32> = state.raw_token_ids[..n].to_vec();

        if current_prefix == state.last_prefix_tokens {
            state.stable_count += 1;
        } else {
            state.last_prefix_tokens = current_prefix.clone();
            state.stable_count = 1;
        }

        if state.stable_count >= state.options.commit_after_stable
            && n >= state.options.commit_token_count
            && state.raw_token_ids.len() >= state.options.commit_token_count + state.options.min_trailing_tokens
        {
            // Back up to a word boundary so we never commit mid-word.
            let Some((n, current_prefix, committed_text_new)) =
                find_word_boundary_commit(&state.raw_token_ids, n, &state.tokenizer)
            else {
                state.stable_count = 0;
                state.last_prefix_tokens.clear();
                return Ok(Some(state.text.clone()));
            };

            let committed_audio_samples = if let Some(ref mut aligner) = state.aligner {
                let t_align = std::time::Instant::now();
                let result = aligner.align(&state.audio_accum, &committed_text_new);
                log::info!("Aligner took {:.0}ms on {:.1}s audio",
                    t_align.elapsed().as_millis(), state.audio_accum.len() as f64 / 16000.0);
                match result {
                    Ok(items) if !items.is_empty() => {
                        let last_word = &items[items.len() - 1];
                        let samples = (last_word.end_time * 16000.0) as usize;
                        log::info!("Aligner: boundary at {:.3}s, last word: {:?}",
                            last_word.end_time, last_word.word);
                        samples
                    }
                    _ => estimate_audio_boundary(&committed_text_new, &state.text, state.audio_accum.len()),
                }
            } else {
                estimate_audio_boundary(&committed_text_new, &state.text, state.audio_accum.len())
            };

            if !state.committed_text.is_empty() {
                state.committed_text = join_committed(&state.committed_text, &committed_text_new);
            } else {
                state.committed_text = committed_text_new;
            }

            let keep_from = committed_audio_samples.min(state.audio_accum.len());
            let remaining_audio: Vec<f32> = state.audio_accum[keep_from..].to_vec();
            let remaining_tokens: Vec<u32> = state.raw_token_ids[n..].to_vec();

            log::info!(
                "Session rotated. Committed: {:?} | audio: kept {:.1}s (boundary at {:.1}s) | seeding {} tokens",
                state.committed_text,
                remaining_audio.len() as f64 / 16000.0,
                keep_from as f64 / 16000.0,
                remaining_tokens.len(),
            );

            // Reset session — clear KV cache, start fresh
            state.audio_accum = remaining_audio;
            state.encoder_cache = EncoderCache::new();
            state.raw_token_ids = remaining_tokens;
            state.chunk_id = state.options.unfixed_chunk_num + 1;
            state.stable_count = 0;
            state.last_prefix_tokens.clear();
            state.decoder_cache = None;
            state.decoder_next_pos = 0;
            state.is_first_step = true;

            unsafe { mlx_clear_cache(); }
            log_memory("after rotation");
        }

        if !state.committed_text.is_empty() {
            state.text = join_committed(&state.committed_text, &state.text);
        }
    }

    Ok(Some(state.text.clone()))
}

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

/// Join committed text with new session text, stripping any overlap where the
/// session re-produced words already in committed text.
fn join_committed(committed: &str, new: &str) -> String {
    let committed = committed.trim();
    let new = new.trim();
    if new.is_empty() { return committed.to_string(); }
    if committed.is_empty() { return new.to_string(); }

    // Fix capitalization at boundary
    let needs_lowercase = !matches!(
        committed.chars().last(),
        Some('.' | '!' | '?' | '。' | '！' | '？') | None
    );
    let fixed = if needs_lowercase {
        let mut chars = new.chars();
        match chars.next() {
            Some(c) if c.is_uppercase() => c.to_lowercase().to_string() + chars.as_str(),
            _ => new.to_string(),
        }
    } else {
        new.to_string()
    };

    // Strip overlap: after rotation the model may re-produce words from committed
    // text because the remaining audio still contains some overlap. The repeated
    // words might appear at the START of the new text (prefix overlap) or somewhere
    // inside it (the model transcribes new words then replays old ones).
    //
    // Strategy: find the longest suffix of committed (by words) that appears
    // anywhere in the new text, and strip everything up to and including that match.
    let committed_words: Vec<&str> = committed.split_whitespace().collect();
    let new_words: Vec<&str> = fixed.split_whitespace().collect();

    // Try progressively shorter suffixes of committed (skip the last word
    // which may be a fragment like "It" from "It's" split at token boundary).
    // We search for each suffix anywhere in the new text.
    let max_suffix = committed_words.len().min(new_words.len());
    for suffix_len in (2..=max_suffix).rev() {
        // Try both with and without the last committed word (it may be a fragment)
        for skip_last in [0usize, 1] {
            if suffix_len <= skip_last { continue; }
            let end = committed_words.len() - skip_last;
            let start_idx = end.saturating_sub(suffix_len);
            let comm_suffix = &committed_words[start_idx..end];
            let match_len = comm_suffix.len();
            if match_len < 2 { continue; }

            for start in 0..=new_words.len().saturating_sub(match_len) {
                let candidate = &new_words[start..start + match_len];
                if comm_suffix.iter().zip(candidate.iter())
                    .all(|(a, b)| strip_punct(&a.to_lowercase()) == strip_punct(&b.to_lowercase()))
                {
                    // Found match — keep everything after the match
                    let after = start + match_len;
                    let remaining = &new_words[after..];
                    return if remaining.is_empty() {
                        committed.to_string()
                    } else {
                        format!("{committed} {}", remaining.join(" "))
                    };
                }
            }
        }
    }

    // Committed text always ends at a word boundary (enforced at commit time).
    // Don't add space if new text starts with punctuation that attaches to
    // the preceding word.
    let sep = match fixed.chars().next() {
        Some('\'' | '\u{2019}' | ',' | '.' | '!' | '?' | ';' | ':') => "",
        _ => " ",
    };
    format!("{committed}{sep}{fixed}")
}

/// Find the largest token count <= `max_tokens` such that the decoded text
/// ends at a word boundary (not mid-word). Returns (token_count, decoded_text).
/// Returns None if no word boundary can be found (e.g., single token that is mid-word).
fn find_word_boundary_commit(
    token_ids: &[u32],
    max_tokens: usize,
    tokenizer: &tokenizers::Tokenizer,
) -> Option<(usize, Vec<u32>, String)> {
    let mut n = max_tokens.min(token_ids.len());
    while n > 0 {
        let prefix = &token_ids[..n];
        let text = tokenizer.decode(prefix, true).unwrap_or_default();
        let trimmed = text.trim_end();
        // A word boundary: text ends with non-alphanumeric (space, punctuation)
        // or is empty
        if trimmed.is_empty()
            || matches!(trimmed.chars().last(), Some(c) if !c.is_alphanumeric())
        {
            return Some((n, prefix.to_vec(), text));
        }
        n -= 1;
    }
    None
}

fn strip_punct(s: &str) -> String {
    s.chars().filter(|c| c.is_alphanumeric() || *c == '\'').collect()
}

/// Find how many tokens at the start of `new_ids` overlap with a suffix of `committed_ids`.
/// Returns the number of tokens to skip from `new_ids`.
fn find_token_overlap(committed_ids: &[u32], new_ids: &[u32]) -> usize {
    let max_overlap = committed_ids.len().min(new_ids.len());
    for k in (1..=max_overlap).rev() {
        let comm_suffix = &committed_ids[committed_ids.len() - k..];
        let new_prefix = &new_ids[..k];
        if comm_suffix == new_prefix {
            return k;
        }
    }
    // Also check if new_ids contains committed suffix anywhere (not just at prefix)
    // This handles cases where the model produces some new tokens THEN replays committed tokens
    for k in (3..=max_overlap).rev() {
        let comm_suffix = &committed_ids[committed_ids.len() - k..];
        for start in 1..=new_ids.len().saturating_sub(k) {
            if &new_ids[start..start + k] == comm_suffix {
                // Strip everything up to and including the match
                return start + k;
            }
        }
    }
    0
}

fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}
