//! High-level streaming transcription built on `bee-qwen3-asr`.

mod asr;
mod mlx_stuff;
use std::collections::HashMap;

use bee_types::Confidence;
use bee_vad::SileroVad;
pub use mlx_stuff::*;
mod types;
use tokenizers::Tokenizer;
pub use types::*;
mod wav_util;
pub use wav_util::decode_wav;

use bee_qwen3_asr::config::AsrConfig;
use bee_qwen3_asr::encoder::EncoderCache;
use bee_qwen3_asr::forced_aligner::ForcedAligner;
use bee_qwen3_asr::generate::TokenLogprob;
use bee_qwen3_asr::mel::MelExtractor;
use bee_qwen3_asr::model::Qwen3ASRModel;
use bee_qwen3_asr::{generate, load};
use mlx_rs::error::Exception;
use mlx_rs::module::ModuleParametersExt;
use mlx_rs::Array;

pub use bee_types::AlignedWord;

use crate::asr::load_tokenizer;

// ── Engine ──────────────────────────────────────────────────────────────

/// Holds loaded model weights, tokenizer, and forced aligner.
///
/// Immutable after construction — multiple sessions can borrow it
/// concurrently via `&Engine`.
pub struct Engine {
    /// Qwen3-compatible tokenizer
    tokenizer: Tokenizer,

    /// Qwen3-ASR
    model: Qwen3ASRModel,

    /// Qwen3-ASR-ForcedAligner
    aligner: ForcedAligner,

    /// Pre-loaded VAD tensors
    vad_tensors: HashMap<String, mlx_rs::Array>,
}

// SAFETY: Engine is immutable after construction. The MLX arrays inside are
// heap-allocated Metal buffers that are safe to read concurrently.
unsafe impl Send for Engine {}
unsafe impl Sync for Engine {}

impl Engine {
    /// Load an engine from explicit paths.
    pub fn load(config: &EngineConfig<'_>) -> Result<Self, Exception> {
        let config_path = config.model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path).map_err(|e| {
            Exception::custom(format!("read config: {e} at {}", config_path.display()))
        })?;
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

        let tokenizer = load_tokenizer(config.tokenizer_dir)?;
        log::info!("Tokenizer loaded");

        let aligner = ForcedAligner::load(config.aligner_dir, tokenizer.clone())?;
        log::info!("Aligner loaded");

        let st_path = config.silero_dir.join("model.safetensors");
        let vad_tensors = mlx_rs::Array::load_safetensors(&st_path)
            .map_err(|e| Exception::custom(format!("vad weights load: {e}")))?;

        Ok(Engine {
            model,
            tokenizer,
            aligner,
            vad_tensors,
        })
    }

    /// Create a new transcription session.
    pub fn session(&self, options: SessionOptions) -> Result<Session<'_>, Exception> {
        let chunk_size_samples = (options.chunk_duration * 16000.0) as usize;

        let vad = SileroVad::from_tensors(&self.vad_tensors)
            .map_err(|e| Exception::custom(format!("vad creation failed: {e}")))?;

        let session = Session {
            engine: self,
            vad,
            buffer: Vec::new(),
            audio: Vec::new(),
            chunk_size_samples,
            chunk_count: 0,
            encoder_cache: EncoderCache::new(),
            mel_extractor: MelExtractor::new(400, 160, 128, 16000),
            token_ids: Vec::new(),
            token_logprobs: Vec::new(),
            committed_tokens: Vec::new(),
            committed_logprobs: Vec::new(),
            committed_alignments: Vec::new(),
            committed_audio_offset: 0.0,
            options,
            speech_detected: false,
            detected_language: String::new(),
        };
        Ok(session)
    }
}

// ── Session ─────────────────────────────────────────────────────────────

/// A live transcription session. Borrows the engine immutably.
pub struct Session<'a> {
    engine: &'a Engine,
    vad: SileroVad,

    // Audio buffering
    buffer: Vec<f32>,
    audio: Vec<f32>,
    chunk_size_samples: usize,
    chunk_count: usize,

    // Encoder state
    encoder_cache: EncoderCache,
    mel_extractor: MelExtractor,

    // Decoder output for the current segment
    token_ids: Vec<TokenId>,
    token_logprobs: Vec<TokenLogprob>,

    // Committed state (accumulated across rotations)
    committed_tokens: Vec<TokenId>,
    committed_logprobs: Vec<TokenLogprob>,
    committed_alignments: Vec<AlignedWord>,
    committed_audio_offset: f64,

    options: SessionOptions,
    speech_detected: bool,
    /// Language detected by the model in auto-detect mode.
    detected_language: String,
}

impl<'a> Session<'a> {
    /// Feed raw 16kHz mono f32 audio samples.
    ///
    /// Returns `Ok(Some(update))` when new text is available,
    /// `Ok(None)` if the audio was silence or not enough has buffered yet.
    pub fn feed(&mut self, samples: &[f32]) -> Result<Option<Update>, Exception> {
        self.buffer.extend_from_slice(samples);

        if self.buffer.len() < self.chunk_size_samples {
            tracing::trace!(
                "feed: buffering {}/{}",
                self.buffer.len(),
                self.chunk_size_samples
            );
            return Ok(None);
        }

        // Drain one chunk
        let chunk: Vec<f32> = self.buffer.drain(..self.chunk_size_samples).collect();

        // VAD gate: run on full chunks for reliable detection
        if !self.speech_detected {
            let prob = self.vad.process_audio(&chunk).unwrap_or(0.0);
            if prob < self.options.vad_threshold {
                tracing::debug!("feed: pre-speech silence (vad={prob:.3})");
                return Ok(None);
            }
            tracing::info!("feed: speech detected (vad={prob:.3})");
            self.speech_detected = true;
        }

        self.audio.extend_from_slice(&chunk);
        self.chunk_count += 1;

        // Skip decode if this chunk is silence (but keep the audio)
        if self.chunk_count > 1 {
            let prob = self.vad.process_audio(&chunk).unwrap_or(0.0);
            let silent = prob < self.options.vad_threshold;
            if silent {
                tracing::debug!("feed: mid-speech silence (vad={prob:.3}), skipping decode");
                return Ok(None);
            }
        }

        // Decode
        tracing::debug!(
            "feed: decoding chunk {} ({} audio samples total)",
            self.chunk_count,
            self.audio.len()
        );
        self.decode_step(self.options.max_tokens_streaming)?;

        // Check for commit
        self.maybe_commit()?;

        let update = self.make_update();
        tracing::debug!(
            "feed: text={:?} committed_len={}",
            &update.text[..update.text.len().min(80)],
            update.committed_len
        );

        Ok(Some(update))
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
        let remaining_text = self
            .engine
            .tokenizer
            .decode(&self.token_ids, true)
            .unwrap_or_default();
        if !remaining_text.is_empty() && !self.audio.is_empty() {
            if let Ok(items) = self.engine.aligner.align(&self.audio, &remaining_text) {
                let wstats = word_logprob_stats(
                    &self.engine.tokenizer,
                    &self.token_ids,
                    &self.token_logprobs,
                    items.len(),
                )
                .map_err(|e| Exception::custom(format!("{e}")))?;
                let offset = self.committed_audio_offset;
                for (i, item) in items.iter().enumerate() {
                    self.committed_alignments.push(AlignedWord {
                        word: item.word.clone(),
                        start: item.start_time + offset,
                        end: item.end_time + offset,
                        confidence: wstats[i].clone(),
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
        let audio_features = self
            .engine
            .model
            .encode_incremental(&mel, &mut self.encoder_cache)?;
        let audio_features = mlx_rs::ops::expand_dims(&audio_features, 0)?;
        // Force evaluation so encoder intermediates can be freed
        audio_features.eval()?;

        // Build prompt with prefix rollback
        let prefix_ids = self.compute_prefix();
        let mut prompt = generate::build_initial_prompt(
            audio_features.shape()[1] as usize,
            self.options.language.as_str(),
            "", // TODO: context support
            &self.engine.tokenizer,
        );
        if let Some(prefix) = &prefix_ids {
            prompt.extend(prefix.iter().map(|&t| t as i32));
        }

        // Generate
        let mut cache = None;
        let (generated, logprobs, _) = generate::prefill_and_decode(
            &self.engine.model,
            &prompt,
            &audio_features,
            &mut cache,
            0,
            max_tokens,
        )?;

        // Combine prefix + generated
        let prefix_len = prefix_ids.as_ref().map_or(0, |p| p.len());
        tracing::debug!(
            "decode_step: generated={} prefix={prefix_len} prompt_len={}",
            generated.len(),
            prompt.len(),
        );

        let (all_ids, all_logprobs): (Vec<TokenId>, Vec<TokenLogprob>) =
            if let Some(prefix) = prefix_ids {
                // Prefix tokens reuse their previous logprobs; generated tokens get fresh ones.
                let prefix_len = prefix.len();
                let prefix_logprobs = if self.token_logprobs.len() >= prefix_len {
                    self.token_logprobs[..prefix_len].to_vec()
                } else {
                    // Fallback: pad with zeros (shouldn't happen in practice)
                    let mut lps = self.token_logprobs.clone();
                    lps.resize(
                        prefix_len,
                        TokenLogprob {
                            token_id: 0,
                            logprob: 0.0,
                            margin: 0.0,
                        },
                    );
                    lps
                };
                let mut ids = prefix;
                ids.extend(generated.iter().map(|&t| t as TokenId));
                let mut lps = prefix_logprobs;
                lps.extend_from_slice(&logprobs);
                (ids, lps)
            } else {
                (generated.iter().map(|&t| t as TokenId).collect(), logprobs)
            };

        tracing::debug!(
            "decode_step: total_ids={} (prefix={prefix_len} + generated={})",
            all_ids.len(),
            generated.len(),
        );

        // If the model returned EOS immediately (no generated tokens and no prefix),
        // preserve the existing transcription rather than wiping it.
        if all_ids.is_empty() && !self.token_ids.is_empty() {
            tracing::debug!(
                "decode_step: EOS with no output, preserving {} existing tokens",
                self.token_ids.len()
            );
        } else {
            self.token_ids = all_ids;
            self.token_logprobs = all_logprobs;
        }

        // KV cache dropped here; return freed buffers to the system
        drop(cache);
        clear_mlx_cache();

        Ok(())
    }

    /// Compute the fixed prefix to feed back to the model.
    /// Returns None during the warm-up phase.
    fn compute_prefix(&self) -> Option<Vec<TokenId>> {
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
        let commit_text = self
            .engine
            .tokenizer
            .decode(&self.token_ids[..commit_count], true)
            .unwrap_or_default();

        // Run forced aligner to find precise audio boundary
        let items = self
            .engine
            .aligner
            .align(&self.audio, &commit_text)
            .map_err(|e| Exception::custom(format!("aligner: {e}")))?;

        if items.is_empty() {
            return Ok(());
        }

        let last = &items[items.len() - 1];
        let audio_cut_samples = (last.end_time * 16000.0) as usize;

        // Compute per-word logprob stats from the tokens being committed
        let commit_logprobs = if self.token_logprobs.len() >= commit_count {
            &self.token_logprobs[..commit_count]
        } else {
            &self.token_logprobs[..]
        };
        let wstats = word_logprob_stats(
            &self.engine.tokenizer,
            &self.token_ids[..commit_count],
            commit_logprobs,
            items.len(),
        )
        .map_err(|e| Exception::custom(format!("{e}")))?;

        // Store alignments with absolute timestamps and logprob stats
        let offset = self.committed_audio_offset;
        for (i, item) in items.iter().enumerate() {
            self.committed_alignments.push(AlignedWord {
                word: item.word.clone(),
                start: item.start_time + offset,
                end: item.end_time + offset,
                confidence: wstats[i].clone(),
            });
        }

        // Update committed tokens and logprobs
        self.committed_tokens
            .extend_from_slice(&self.token_ids[..commit_count]);
        self.committed_logprobs.extend_from_slice(commit_logprobs);

        // Rotate session
        let cut = audio_cut_samples.min(self.audio.len());
        self.committed_audio_offset += cut as f64 / 16000.0;
        self.audio = self.audio[cut..].to_vec();
        self.encoder_cache = EncoderCache::new();
        self.token_ids = self.token_ids[commit_count..].to_vec();
        self.token_logprobs = if self.token_logprobs.len() > commit_count {
            self.token_logprobs[commit_count..].to_vec()
        } else {
            Vec::new()
        };
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
    fn full_token_ids(&self) -> Vec<TokenId> {
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
    fn make_update(&mut self) -> Update {
        let all_ids = self.full_token_ids();

        // Split at the <asr_text> token — metadata before, text after.
        let asr_text_id = generate::TOK_ASR_TEXT;
        let (text, detected_lang) = if let Some(tag_pos) =
            all_ids.iter().position(|&id| id == asr_text_id as TokenId)
        {
            let meta_ids = &all_ids[..tag_pos];
            let text_ids = &all_ids[tag_pos + 1..];
            let meta = self
                .engine
                .tokenizer
                .decode(meta_ids, true)
                .unwrap_or_default();
            let text = self
                .engine
                .tokenizer
                .decode(text_ids, true)
                .unwrap_or_default();

            // Extract language from metadata (e.g. "language English")
            let lang = meta
                .trim()
                .strip_prefix("language ")
                .map(|l| l.trim().to_string())
                .filter(|l| !l.eq_ignore_ascii_case("none"))
                .unwrap_or_default();

            tracing::info!("make_update: lang={lang:?} text={text:?} (split at token {tag_pos}, {meta_ids_len} meta + {text_ids_len} text tokens)",
                meta_ids_len = meta_ids.len(), text_ids_len = text_ids.len());
            (text, lang)
        } else {
            // No <asr_text> token — decode everything as text
            let text = self
                .engine
                .tokenizer
                .decode(&all_ids, true)
                .unwrap_or_default();
            tracing::info!(
                "make_update: no <asr_text> token, {n} tokens, raw_text={text:?}",
                n = all_ids.len()
            );
            (text, String::new())
        };

        if !detected_lang.is_empty() {
            self.detected_language = detected_lang;
        }

        // Committed length = decode committed tokens in the same context
        let committed_len = if self.committed_tokens.is_empty() {
            0
        } else {
            // Decode committed portion to find its length in the full string.
            // Since the full decode has the same prefix, the committed part
            // is always the first N characters.
            let committed_text = self
                .engine
                .tokenizer
                .decode(&self.committed_tokens, true)
                .unwrap_or_default();
            committed_text.len()
        };

        Update {
            text,
            committed_len,
            alignments: self.committed_alignments.clone(),
            detected_language: self.detected_language.clone(),
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Compute per-word logprob statistics by mapping decoder tokens to aligner words.
///
/// Uses a greedy character-accumulation approach: decode tokens one by one,
/// accumulate text, and assign tokens to the next aligner word whose text
/// they contribute to.
fn word_logprob_stats(
    tokenizer: &Tokenizer,
    token_ids: &[TokenId],
    token_logprobs: &[TokenLogprob],
    word_count: usize,
) -> Result<Vec<Confidence>, TranscribeError> {
    if token_logprobs.is_empty() || word_count == 0 {
        return Ok(vec![]);
    }

    // Decode each token to find its text contribution
    let mut per_token_texts: Vec<String> = Vec::with_capacity(token_ids.len());
    for (i, _) in token_ids.iter().enumerate() {
        // Decode [0..=i] minus [0..i] to get just this token's contribution
        let with = tokenizer.decode(&token_ids[..=i], true).unwrap_or_default();
        let without = if i > 0 {
            tokenizer.decode(&token_ids[..i], true).unwrap_or_default()
        } else {
            String::new()
        };
        let contribution = if with.len() >= without.len() {
            with[without.len()..].to_string()
        } else {
            String::new()
        };
        per_token_texts.push(contribution);
    }

    // Assign tokens to words: accumulate non-whitespace runs
    // Each word boundary is roughly a whitespace transition
    let mut word_logprobs: Vec<Vec<&TokenLogprob>> = vec![Vec::new(); word_count];
    let mut word_idx = 0;
    let mut seen_chars_in_word = false;

    for (i, text) in per_token_texts.iter().enumerate() {
        if word_idx >= word_count {
            break;
        }
        let lp = token_logprobs.get(i);

        // Check if this token starts a new word (leading whitespace after we've seen chars)
        let starts_with_space = text.starts_with(' ') || text.starts_with('\n');
        if starts_with_space && seen_chars_in_word && word_idx + 1 < word_count {
            word_idx += 1;
            seen_chars_in_word = false;
        }

        let has_non_ws = text.chars().any(|c| !c.is_whitespace());
        if has_non_ws {
            seen_chars_in_word = true;
        }

        if let Some(lp) = lp {
            word_logprobs[word_idx].push(lp);
        }
    }

    let word_confidences = word_logprobs
        .iter()
        .enumerate()
        .map(|(word_idx, lps)| {
            if lps.is_empty() {
                panic!(
                    "word {word_idx}/{word_count} has no token logprobs (token_ids={}, logprobs={}, per_token_texts={:?})",
                    token_ids.len(),
                    token_logprobs.len(),
                    per_token_texts,
                );
            }
            let n = lps.len() as f32;
            let mean_lp = lps.iter().map(|lp| lp.logprob).sum::<f32>() / n;
            let min_lp = lps
                .iter()
                .map(|lp| lp.logprob)
                .fold(f32::INFINITY, f32::min);
            let mean_m = lps.iter().map(|lp| lp.margin).sum::<f32>() / n;
            let min_m = lps.iter().map(|lp| lp.margin).fold(f32::INFINITY, f32::min);
            Confidence {
                mean_lp,
                min_lp,
                mean_m,
                min_m,
            }
        })
        .collect();
    Ok(word_confidences)
}
