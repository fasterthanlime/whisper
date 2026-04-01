//! High-level streaming transcription built on `bee-qwen3-asr`.
//!
//! `Engine` holds the loaded model weights and is immutable after construction —
//! multiple `Session`s can borrow it concurrently.
//!
//! Each `Session` processes a single audio stream, producing incremental text
//! updates with word-level timestamps.

use std::io::Cursor;
use std::path::Path;

extern "C" {
    fn mlx_set_cache_limit(res: *mut usize, limit: usize) -> std::ffi::c_int;
    fn mlx_clear_cache() -> std::ffi::c_int;
}

/// Release unused MLX Metal buffers from the pool back to the system.
/// Safe to call concurrently — only frees buffers with no live references.
pub fn clear_mlx_cache() {
    unsafe { mlx_clear_cache(); }
}

/// Set the MLX Metal buffer cache limit. Buffers beyond this are returned
/// to the system instead of being pooled for reuse.
pub fn set_mlx_cache_limit(limit: usize) -> usize {
    let mut prev = 0usize;
    unsafe { mlx_set_cache_limit(&mut prev, limit); }
    prev
}

use bee_qwen3_asr::config::AsrConfig;
use bee_qwen3_asr::encoder::EncoderCache;
use bee_qwen3_asr::forced_aligner::ForcedAligner;
use bee_qwen3_asr::mel::MelExtractor;
use bee_qwen3_asr::model::Qwen3ASRModel;
use bee_qwen3_asr::{generate, load};
use facet::Facet;
use mlx_rs::error::Exception;
use mlx_rs::module::ModuleParametersExt;
use mlx_rs::Array;

// ── Data types ──────────────────────────────────────────────────────────

/// Language of the audio being transcribed.
///
/// Passed to the model as `"language {name}"`. Common values:
/// `"English"`, `"Chinese"`, `"Japanese"`, `"Korean"`, `"French"`,
/// `"German"`, `"Spanish"`.
#[derive(Debug, Clone, Facet)]
pub struct Language(pub String);

impl Language {
    fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for Language {
    fn default() -> Self {
        Language("English".into())
    }
}

/// Configuration for a transcription session.
#[derive(Debug, Clone, Facet)]
pub struct SessionOptions {
    /// Seconds of audio per processing chunk. Default: 0.4
    pub chunk_duration: f32,

    /// VAD speech probability threshold (0.0-1.0). Default: 0.5
    pub vad_threshold: f32,

    /// How many recent tokens the model is allowed to revise each step.
    /// Everything before this tail is fed back as fixed context that
    /// the model must continue from. Higher values give the model more
    /// freedom to correct itself but make the output less stable between
    /// steps. Default: 5
    pub rollback_tokens: usize,

    /// Minimum number of fixed tokens (total minus `rollback_tokens`)
    /// before we commit and rotate the session. When the fixed portion
    /// reaches this threshold, we find a word boundary, run the aligner,
    /// and start a fresh session. Default: 12
    pub commit_token_count: usize,

    /// Max tokens to generate per streaming step. Default: 32
    pub max_tokens_streaming: usize,

    /// Max tokens for the final decode when `finish()` is called. Default: 512
    pub max_tokens_final: usize,

    /// Language of the audio.
    pub language: Language,
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            chunk_duration: 0.4,
            vad_threshold: 0.5,
            rollback_tokens: 5,
            commit_token_count: 12,
            max_tokens_streaming: 32,
            max_tokens_final: 512,
            language: Language::default(),
        }
    }
}

/// Result of a `feed()` or `finish()` call.
#[derive(Debug, Clone, Facet)]
pub struct Update {
    /// Full transcription so far (committed + in-progress).
    pub text: String,

    /// Byte offset into `text` where the committed (stable) portion ends.
    /// `text[..committed_len]` is final and won't change.
    /// `text[committed_len..]` is the in-progress tail that may be revised.
    pub committed_len: usize,

    /// Word-level timestamps for committed words.
    pub alignments: Vec<AlignedWord>,
}

/// A single word with its time boundaries from forced alignment.
#[derive(Debug, Clone, Facet)]
pub struct AlignedWord {
    /// The word text.
    pub word: String,
    /// Start time in seconds from the beginning of the audio stream.
    pub start: f64,
    /// End time in seconds from the beginning of the audio stream.
    pub end: f64,
}

// ── Engine ──────────────────────────────────────────────────────────────

/// Paths required to load an engine.
pub struct EngineConfig<'a> {
    /// Directory containing `config.json` and `*.safetensors` for the
    /// ASR model weights.
    pub model_dir: &'a Path,
    /// Path to `tokenizer.json`.
    pub tokenizer_path: &'a Path,
    /// Directory containing the forced aligner model weights.
    pub aligner_dir: &'a Path,
}

/// Holds loaded model weights, tokenizer, and forced aligner.
///
/// Immutable after construction — multiple sessions can borrow it
/// concurrently via `&Engine`.
pub struct Engine {
    model: Qwen3ASRModel,
    tokenizer: tokenizers::Tokenizer,
    aligner: ForcedAligner,
}

// SAFETY: Engine is immutable after construction. The MLX arrays inside are
// heap-allocated Metal buffers that are safe to read concurrently.
unsafe impl Send for Engine {}
unsafe impl Sync for Engine {}

impl Engine {
    /// Load an engine from explicit paths.
    pub fn load(config: &EngineConfig<'_>) -> Result<Self, Exception> {
        let config_str = std::fs::read_to_string(config.model_dir.join("config.json"))
            .map_err(|e| Exception::custom(format!("read config: {e}")))?;
        let asr_config: AsrConfig = serde_json::from_str(&config_str)
            .map_err(|e| Exception::custom(format!("parse config: {e}")))?;

        let mut model = Qwen3ASRModel::new(&asr_config.thinker_config)?;
        let stats = load::load_weights(&mut model, config.model_dir)?;
        model.eval()?;

        log::info!(
            "Engine loaded: {}/{} keys, {} quantized ({}bit)",
            stats.loaded,
            stats.total_keys,
            stats.quantized_layers,
            stats.bits,
        );

        let tokenizer = tokenizers::Tokenizer::from_file(config.tokenizer_path)
            .map_err(|e| Exception::custom(format!("load tokenizer: {e}")))?;

        let aligner = ForcedAligner::load(config.aligner_dir, tokenizer.clone())?;

        Ok(Engine {
            model,
            tokenizer,
            aligner,
        })
    }

    /// Create a new transcription session.
    pub fn session(&self, options: SessionOptions) -> Session<'_> {
        let chunk_size_samples = (options.chunk_duration * 16000.0) as usize;

        let lang_header = format!("language {}", options.language.as_str());
        let language_tokens = tokenize_to_i32(&self.tokenizer, &lang_header);
        let asr_text_tokens = tokenize_to_i32(&self.tokenizer, "<asr_text>");

        Session {
            engine: self,
            vad: None, // caller sets this via set_vad()
            buffer: Vec::new(),
            audio: Vec::new(),
            chunk_size_samples,
            chunk_count: 0,
            encoder_cache: EncoderCache::new(),
            mel_extractor: MelExtractor::new(400, 160, 128, 16000),
            token_ids: Vec::new(),
            committed_tokens: Vec::new(),
            committed_alignments: Vec::new(),
            committed_audio_offset: 0.0,
            language_tokens,
            asr_text_tokens,
            options,
            speech_detected: false,
        }
    }
}

// ── Session ─────────────────────────────────────────────────────────────

/// A live transcription session. Borrows the engine immutably.
pub struct Session<'a> {
    engine: &'a Engine,
    vad: Option<bee_vad::SileroVad>,

    // Audio buffering
    buffer: Vec<f32>,
    audio: Vec<f32>,
    chunk_size_samples: usize,
    chunk_count: usize,

    // Encoder state
    encoder_cache: EncoderCache,
    mel_extractor: MelExtractor,

    // Decoder output for the current segment
    token_ids: Vec<u32>,

    // Committed state (accumulated across rotations)
    committed_tokens: Vec<u32>,
    committed_alignments: Vec<AlignedWord>,
    committed_audio_offset: f64,

    // Precomputed prompt tokens
    language_tokens: Vec<i32>,
    asr_text_tokens: Vec<i32>,

    options: SessionOptions,
    speech_detected: bool,
}

impl<'a> Session<'a> {
    /// Attach a VAD instance for speech detection gating.
    pub fn set_vad(&mut self, vad: bee_vad::SileroVad) {
        self.vad = Some(vad);
    }

    /// Feed raw 16kHz mono f32 audio samples.
    ///
    /// Returns `Ok(Some(update))` when new text is available,
    /// `Ok(None)` if the audio was silence or not enough has buffered yet.
    pub fn feed(&mut self, samples: &[f32]) -> Result<Option<Update>, Exception> {
        // VAD gate
        if !self.speech_detected {
            if let Some(ref mut vad) = self.vad {
                let prob = vad.process_audio(samples).unwrap_or(0.0);
                if prob < self.options.vad_threshold {
                    return Ok(None);
                }
            }
            self.speech_detected = true;
        }

        self.buffer.extend_from_slice(samples);

        if self.buffer.len() < self.chunk_size_samples {
            return Ok(None);
        }

        // Drain one chunk
        let chunk: Vec<f32> = self.buffer.drain(..self.chunk_size_samples).collect();
        self.audio.extend_from_slice(&chunk);
        self.chunk_count += 1;

        // Skip decode if this chunk is silence (but keep the audio)
        if self.chunk_count > 1 {
            let is_silence = if let Some(ref mut vad) = self.vad {
                vad.process_audio(&chunk).unwrap_or(0.0) < self.options.vad_threshold
            } else {
                compute_rms(&chunk) < 0.006
            };
            if is_silence {
                return Ok(None);
            }
        }

        // Decode
        self.decode_step(self.options.max_tokens_streaming)?;

        // Check for commit
        self.maybe_commit()?;

        Ok(Some(self.make_update()))
    }

    /// Finalize the session: flush remaining audio with a higher token
    /// budget and return the final transcription.
    pub fn finish(mut self) -> Result<Update, Exception> {
        // Flush remaining buffer
        if !self.buffer.is_empty() {
            self.audio.append(&mut self.buffer);
            self.chunk_count += 1;
        }

        if !self.audio.is_empty() {
            self.decode_step(self.options.max_tokens_final)?;
        }

        // Align any remaining uncommitted text
        let remaining_text = self.engine.tokenizer
            .decode(&self.token_ids, true)
            .unwrap_or_default();
        if !remaining_text.is_empty() && !self.audio.is_empty() {
            if let Ok(items) = self.engine.aligner.align(&self.audio, &remaining_text) {
                let offset = self.committed_audio_offset;
                for item in &items {
                    self.committed_alignments.push(AlignedWord {
                        word: item.word.clone(),
                        start: item.start_time + offset,
                        end: item.end_time + offset,
                    });
                }
            }
        }

        Ok(self.make_update())
    }

    // ── Internal ────────────────────────────────────────────────────

    fn decode_step(&mut self, max_tokens: usize) -> Result<(), Exception> {
        // Mel extraction
        let (mel_data, n_mels, n_frames) = self
            .mel_extractor
            .extract(&self.audio)
            .map_err(|e| Exception::custom(format!("mel: {e}")))?;
        let mel = Array::from_slice(&mel_data, &[n_mels as i32, n_frames as i32]);

        // Encode audio (incremental)
        let audio_features =
            self.engine
                .model
                .encode_incremental(&mel, &mut self.encoder_cache)?;
        let audio_features = mlx_rs::ops::expand_dims(&audio_features, 0)?;
        // Force evaluation so encoder intermediates can be freed
        audio_features.eval()?;

        // Build prompt with prefix rollback
        let prefix_ids = self.compute_prefix();
        let mut prompt = generate::build_initial_prompt(
            audio_features.shape()[1] as usize,
            &self.language_tokens,
            &self.asr_text_tokens,
        );
        if let Some(prefix) = &prefix_ids {
            prompt.extend(prefix.iter().map(|&t| t as i32));
        }

        // Generate
        let mut cache = None;
        let (generated, _) = generate::prefill_and_decode(
            &self.engine.model,
            &prompt,
            &audio_features,
            &mut cache,
            0,
            max_tokens,
        )?;

        // Combine prefix + generated
        let all_ids: Vec<u32> = if let Some(prefix) = prefix_ids {
            let mut combined = prefix;
            combined.extend(generated.iter().map(|&t| t as u32));
            combined
        } else {
            generated.iter().map(|&t| t as u32).collect()
        };

        self.token_ids = all_ids;

        // KV cache dropped here; return freed buffers to the system
        drop(cache);
        clear_mlx_cache();

        Ok(())
    }

    /// Compute the fixed prefix to feed back to the model.
    /// Returns None during the warm-up phase.
    fn compute_prefix(&self) -> Option<Vec<u32>> {
        if self.chunk_count <= 2 || self.token_ids.is_empty() {
            return None;
        }
        let keep = self
            .token_ids
            .len()
            .saturating_sub(self.options.rollback_tokens);
        if keep == 0 {
            return None;
        }
        Some(self.token_ids[..keep].to_vec())
    }

    /// Commit fixed tokens and rotate the session if we have enough.
    ///
    /// The fixed portion is everything except the last `rollback_tokens`.
    /// We commit when that fixed portion >= `commit_token_count`.
    fn maybe_commit(&mut self) -> Result<(), Exception> {
        let fixed_count = self
            .token_ids
            .len()
            .saturating_sub(self.options.rollback_tokens);

        // Wait until we have twice the commit threshold in fixed tokens,
        // then commit half. This keeps a large seed for context.
        if fixed_count < self.options.commit_token_count * 2 {
            return Ok(());
        }

        let commit_count = self.options.commit_token_count;
        let commit_text = self.engine.tokenizer
            .decode(&self.token_ids[..commit_count], true)
            .unwrap_or_default();

        // Run forced aligner to find precise audio boundary
        let items = self.engine.aligner.align(&self.audio, &commit_text)
            .map_err(|e| Exception::custom(format!("aligner: {e}")))?;

        if items.is_empty() {
            return Ok(());
        }

        let last = &items[items.len() - 1];
        let audio_cut_samples = (last.end_time * 16000.0) as usize;

        // Store alignments with absolute timestamps
        let offset = self.committed_audio_offset;
        for item in &items {
            self.committed_alignments.push(AlignedWord {
                word: item.word.clone(),
                start: item.start_time + offset,
                end: item.end_time + offset,
            });
        }

        // Update committed tokens
        self.committed_tokens
            .extend_from_slice(&self.token_ids[..commit_count]);

        // Rotate session
        let cut = audio_cut_samples.min(self.audio.len());
        self.committed_audio_offset += cut as f64 / 16000.0;
        self.audio = self.audio[cut..].to_vec();
        self.encoder_cache = EncoderCache::new();
        self.token_ids = self.token_ids[commit_count..].to_vec();
        // Skip warm-up so prefix rollback kicks in immediately
        self.chunk_count = 3;

        log::info!(
            "Committed {} tokens | kept {:.1}s audio, {} seed tokens",
            self.committed_tokens.len(),
            self.audio.len() as f64 / 16000.0,
            self.token_ids.len(),
        );

        Ok(())
    }

    /// Decode all tokens (committed + current session) as one sequence
    /// so the tokenizer preserves whitespace context across the boundary.
    fn full_token_ids(&self) -> Vec<u32> {
        let mut all = self.committed_tokens.clone();
        all.extend_from_slice(&self.token_ids);
        all
    }

    /// Build the Update to return to the caller.
    ///
    /// Decodes all tokens (committed + current) as one sequence so the
    /// tokenizer preserves whitespace across commit boundaries. The
    /// committed_len is derived from decoding just the committed tokens
    /// within the same full decode.
    fn make_update(&self) -> Update {
        let all_ids = self.full_token_ids();
        let text = self.engine.tokenizer.decode(&all_ids, true).unwrap_or_default();

        // Committed length = decode committed tokens in the same context
        let committed_len = if self.committed_tokens.is_empty() {
            0
        } else {
            // Decode committed portion to find its length in the full string.
            // Since the full decode has the same prefix, the committed part
            // is always the first N characters.
            let committed_text = self.engine.tokenizer
                .decode(&self.committed_tokens, true)
                .unwrap_or_default();
            committed_text.len()
        };

        Update {
            text,
            committed_len,
            alignments: self.committed_alignments.clone(),
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

fn tokenize_to_i32(tokenizer: &tokenizers::Tokenizer, text: &str) -> Vec<i32> {
    tokenizer
        .encode(text, false)
        .map(|enc| enc.get_ids().iter().map(|&id| id as i32).collect())
        .unwrap_or_default()
}

fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

/// Walk backwards from `max_tokens` to find a token count where the
/// decoded text ends at a word boundary (space or punctuation).
fn find_word_boundary(
    token_ids: &[u32],
    max_tokens: usize,
    tokenizer: &tokenizers::Tokenizer,
) -> Option<(usize, String)> {
    let mut n = max_tokens.min(token_ids.len());
    while n > 0 {
        let text = tokenizer.decode(&token_ids[..n], true).unwrap_or_default();
        let trimmed = text.trim_end();
        if trimmed.is_empty()
            || matches!(trimmed.chars().last(), Some(c) if !c.is_alphanumeric())
        {
            return Some((n, text));
        }
        n -= 1;
    }
    None
}

/// Decode WAV bytes to 16kHz mono f32 samples.
pub fn decode_wav(bytes: &[u8]) -> Result<Vec<f32>, mlx_rs::error::Exception> {
    let cursor = Cursor::new(bytes);
    let mut reader = hound::WavReader::new(cursor)
        .map_err(|e| mlx_rs::error::Exception::custom(format!("invalid WAV: {e}")))?;
    let spec = reader.spec();

    if spec.sample_rate != 16_000 {
        return Err(mlx_rs::error::Exception::custom(format!(
            "expected 16kHz WAV, got {}Hz",
            spec.sample_rate
        )));
    }

    let channels = spec.channels.max(1) as usize;
    let mut mono = Vec::new();

    match spec.sample_format {
        hound::SampleFormat::Float => {
            let mut acc = 0.0f32;
            let mut idx = 0usize;
            for sample in reader.samples::<f32>() {
                acc += sample.map_err(|e| mlx_rs::error::Exception::custom(format!("{e}")))?;
                idx += 1;
                if idx == channels {
                    mono.push(acc / channels as f32);
                    acc = 0.0;
                    idx = 0;
                }
            }
        }
        hound::SampleFormat::Int => {
            let scale = if spec.bits_per_sample <= 16 {
                i16::MAX as f32
            } else {
                ((1_i64 << (spec.bits_per_sample - 1)) - 1) as f32
            };
            let mut acc = 0.0f32;
            let mut idx = 0usize;
            for sample in reader.samples::<i32>() {
                acc += sample.map_err(|e| mlx_rs::error::Exception::custom(format!("{e}")))? as f32 / scale;
                idx += 1;
                if idx == channels {
                    mono.push(acc / channels as f32);
                    acc = 0.0;
                    idx = 0;
                }
            }
        }
    }

    if mono.is_empty() {
        return Err(mlx_rs::error::Exception::custom("WAV is empty"));
    }

    Ok(mono)
}

