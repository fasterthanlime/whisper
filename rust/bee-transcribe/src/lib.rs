//! High-level streaming transcription built on `bee-qwen3-asr`.
//!
//! `Engine` holds the loaded model weights and is immutable after construction —
//! multiple `Session`s can borrow it concurrently.
//!
//! Each `Session` processes a single audio stream, producing incremental text
//! updates with word-level timestamps.

use std::path::Path;

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

    /// Number of leading tokens to monitor for stability. When these
    /// tokens stay identical across `commit_stability_rounds` consecutive
    /// decode steps, they are considered final: we run the forced aligner,
    /// trim the audio at the word boundary, and rotate the session.
    /// Default: 12
    pub commit_token_count: usize,

    /// How many consecutive decode steps the leading tokens must be
    /// unchanged before triggering a commit and session rotation.
    /// Default: 3
    pub commit_stability_rounds: usize,

    /// Minimum tokens beyond `commit_token_count` required before
    /// committing. Ensures we don't commit when the model has barely
    /// started generating past the stability window. Default: 6
    pub min_trailing_tokens: usize,

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
            commit_stability_rounds: 3,
            min_trailing_tokens: 6,
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

/// Holds loaded model weights, tokenizer, and forced aligner.
///
/// Immutable after construction — multiple sessions can borrow it
/// concurrently via `&Engine`.
pub struct Engine {
    model: Qwen3ASRModel,
    tokenizer: tokenizers::Tokenizer,
    aligner: Option<ForcedAligner>,
}

impl Engine {
    /// Load an engine from a model directory.
    ///
    /// Looks for `config.json`, `*.safetensors`, and `tokenizer.json`
    /// in the given directory. Optionally loads a forced aligner from
    /// a well-known cache location.
    pub fn load(model_dir: &Path) -> Result<Self, Exception> {
        let config_str = std::fs::read_to_string(model_dir.join("config.json"))
            .map_err(|e| Exception::custom(format!("read config: {e}")))?;
        let config: AsrConfig = serde_json::from_str(&config_str)
            .map_err(|e| Exception::custom(format!("parse config: {e}")))?;

        let mut model = Qwen3ASRModel::new(&config.thinker_config)?;
        let stats = load::load_weights(&mut model, model_dir)?;
        model.eval()?;

        log::info!(
            "Engine loaded: {}/{} keys, {} quantized ({}bit)",
            stats.loaded,
            stats.total_keys,
            stats.quantized_layers,
            stats.bits,
        );

        let tokenizer = find_tokenizer(model_dir)
            .ok_or_else(|| Exception::custom("tokenizer.json not found"))?;

        let aligner = find_aligner_dir().and_then(|dir| {
            match ForcedAligner::load(&dir, tokenizer.clone()) {
                Ok(a) => Some(a),
                Err(e) => {
                    log::warn!("failed to load forced aligner: {e}");
                    None
                }
            }
        });

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
            committed_text: String::new(),
            committed_tokens: Vec::new(),
            committed_alignments: Vec::new(),
            committed_audio_offset: 0.0,
            prefix_tokens: Vec::new(),
            stable_rounds: 0,
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
    committed_text: String,
    committed_tokens: Vec<u32>,
    committed_alignments: Vec<AlignedWord>,
    committed_audio_offset: f64,

    // Stability tracking
    prefix_tokens: Vec<u32>,
    stable_rounds: usize,

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

    /// Check if leading tokens are stable enough to commit,
    /// and if so, perform a session rotation.
    fn maybe_commit(&mut self) -> Result<(), Exception> {
        if self.token_ids.is_empty() {
            return Ok(());
        }

        let n = self
            .options
            .commit_token_count
            .min(self.token_ids.len());
        let current_prefix: Vec<u32> = self.token_ids[..n].to_vec();

        if current_prefix == self.prefix_tokens {
            self.stable_rounds += 1;
        } else {
            self.prefix_tokens = current_prefix;
            self.stable_rounds = 1;
        }

        let should_commit = self.stable_rounds >= self.options.commit_stability_rounds
            && n >= self.options.commit_token_count
            && self.token_ids.len() >= self.options.commit_token_count + self.options.min_trailing_tokens;

        if !should_commit {
            return Ok(());
        }

        // Find a word boundary to commit at
        let Some((commit_count, commit_text)) =
            find_word_boundary(&self.token_ids, n, &self.engine.tokenizer)
        else {
            // No clean word boundary — reset stability and wait
            self.stable_rounds = 0;
            self.prefix_tokens.clear();
            return Ok(());
        };

        // Use forced aligner for precise audio boundary
        let audio_cut_samples = if let Some(ref aligner) = self.engine.aligner {
            match aligner.align(&self.audio, &commit_text) {
                Ok(items) if !items.is_empty() => {
                    let last = &items[items.len() - 1];
                    // Store alignments with absolute timestamps
                    let offset = self.committed_audio_offset;
                    for item in &items {
                        self.committed_alignments.push(AlignedWord {
                            word: item.word.clone(),
                            start: item.start_time + offset,
                            end: item.end_time + offset,
                        });
                    }
                    (last.end_time * 16000.0) as usize
                }
                _ => estimate_audio_boundary(&commit_text, &self.current_text(), self.audio.len()),
            }
        } else {
            estimate_audio_boundary(&commit_text, &self.current_text(), self.audio.len())
        };

        // Update committed state
        if self.committed_text.is_empty() {
            self.committed_text = commit_text;
        } else {
            self.committed_text.push(' ');
            self.committed_text.push_str(commit_text.trim());
        }
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
        self.stable_rounds = 0;
        self.prefix_tokens.clear();

        log::info!(
            "Committed: {:?} | kept {:.1}s audio, {} seed tokens",
            self.committed_text,
            self.audio.len() as f64 / 16000.0,
            self.token_ids.len(),
        );

        Ok(())
    }

    /// Decode current segment tokens to text.
    fn current_text(&self) -> String {
        self.engine
            .tokenizer
            .decode(&self.token_ids, true)
            .unwrap_or_default()
    }

    /// Build the Update to return to the caller.
    fn make_update(&self) -> Update {
        let current = self.current_text();
        let committed_len = self.committed_text.len();
        let text = if self.committed_text.is_empty() {
            current
        } else if current.trim().is_empty() {
            self.committed_text.clone()
        } else {
            format!("{} {}", self.committed_text, current.trim())
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

/// Rough audio boundary estimate when no aligner is available.
/// Uses character-proportional approximation.
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
    let boundary = (fraction * total_audio_samples as f64) as usize;
    boundary.min(total_audio_samples.saturating_sub(16000))
}

fn find_tokenizer(model_dir: &Path) -> Option<tokenizers::Tokenizer> {
    let mut paths = vec![model_dir.join("tokenizer.json")];
    if let Some(home) = dirs::home_dir() {
        paths.push(home.join("Library/Caches/qwen3-asr/Qwen--Qwen3-ASR-1.7B/tokenizer.json"));
        paths.push(home.join("Library/Caches/qwen3-asr/Qwen--Qwen3-ASR-0.6B/tokenizer.json"));
    }
    for path in &paths {
        if path.exists() {
            if let Ok(tokenizer) = tokenizers::Tokenizer::from_file(path) {
                return Some(tokenizer);
            }
        }
    }
    None
}

fn find_aligner_dir() -> Option<std::path::PathBuf> {
    let home = dirs::home_dir()?;
    let base = home.join("Library/Caches/qwen3-asr");
    let candidates = [
        "mlx-community--Qwen3-ForcedAligner-0.6B-4bit",
        "Qwen--Qwen3-ForcedAligner-0.6B",
    ];
    for name in candidates {
        let dir = base.join(name);
        if dir.exists() {
            return Some(dir);
        }
    }
    None
}
