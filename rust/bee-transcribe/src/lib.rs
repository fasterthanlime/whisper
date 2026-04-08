//! High-level streaming transcription built on `bee-qwen3-asr`.

mod aligner;
pub mod audio_buffer;
pub mod audio_filter;
mod asr;
pub mod text_buffer;
pub mod correct;
pub mod corrector;
mod generator;
mod mlx_stuff;
mod speech_gate;
mod structured_output;
mod types;
mod wav_util;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use bee_types::Confidence;
use bee_vad::SileroVad;
pub use mlx_stuff::*;
pub use types::*;
pub use wav_util::decode_wav;

use bee_qwen3_asr::config::AsrConfig;
use bee_qwen3_asr::forced_aligner::ForcedAligner;
use bee_qwen3_asr::generate::TokenLogprob;
use bee_qwen3_asr::model::Qwen3ASRModel;
use bee_qwen3_asr::load;
use mlx_rs::error::Exception;
use mlx_rs::module::ModuleParametersExt;
use tokenizers::Tokenizer;

pub use bee_types::AlignedWord;

/// Shared handle to a correction engine. Cloneable — multiple sessions
/// can reference the same engine (locked during use).
pub type SharedCorrectionEngine = Arc<Mutex<CorrectionEngine>>;

/// Result of `Session::finish()`. Contains the final update and optionally
/// the corrector state (for teach/save).
pub struct FinishResult {
    pub update: Update,
    pub corrector: Option<Corrector>,
}

use crate::aligner::Aligner;
use crate::asr::load_tokenizer;
use crate::correct::CorrectionEngine;
use crate::corrector::Corrector;
use crate::generator::Generator;
use crate::speech_gate::{FeedResult, SpeechGate};
use crate::structured_output::StructuredAsrOutput;

// ── Engine ──────────────────────────────────────────────────────────────

/// Holds loaded model weights, tokenizer, and forced aligner.
///
/// Immutable after construction — multiple sessions can borrow it
/// concurrently via `&Engine`.
pub struct Engine {
    tokenizer: Tokenizer,
    model: Qwen3ASRModel,
    aligner: ForcedAligner,
    vad_tensors: HashMap<String, mlx_rs::Array>,
    correction: Option<SharedCorrectionEngine>,
}

// SAFETY: Engine is immutable after construction. The MLX arrays inside are
// heap-allocated Metal buffers that are safe to read concurrently.
unsafe impl Send for Engine {}
unsafe impl Sync for Engine {}

impl Engine {
    /// Access the shared correction engine, if loaded.
    pub fn correction(&self) -> Option<&SharedCorrectionEngine> {
        self.correction.as_ref()
    }

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

        let correction = if let Some(dataset_dir) = config.correction_dir {
            let cc = crate::correct::CorrectionConfig {
                dataset_dir,
                events_path: config.correction_events_path.clone(),
                gate_threshold: 0.0, // uses default
                ranker_threshold: 0.0, // uses default
            };
            let ce = crate::correct::load_correction_engine(&cc)
                .map_err(|e| Exception::custom(format!("correction engine: {e}")))?;
            log::info!("Correction engine loaded");
            Some(Arc::new(Mutex::new(ce)))
        } else {
            None
        };

        Ok(Engine {
            model,
            tokenizer,
            aligner,
            vad_tensors,
            correction,
        })
    }

    /// Create a new transcription session.
    ///
    /// Create a new transcription session.
    pub fn session(&self, options: SessionOptions) -> Result<Session<'_>, Exception> {
        let chunk_size_samples = (options.chunk_duration * 16000.0) as usize;

        let vad = SileroVad::from_tensors(&self.vad_tensors)
            .map_err(|e| Exception::custom(format!("vad creation failed: {e}")))?;

        let correction = self.correction.as_ref().map(|ce| (ce.clone(), Corrector::new()));

        Ok(Session {
            engine: self,
            speech_gate: SpeechGate::new(vad, chunk_size_samples, options.vad_threshold),
            generator: Generator::new(options.rollback_tokens),
            aligner: Aligner::new(options.commit_token_count),
            correction,
            options,
        })
    }
}

// ── Session ─────────────────────────────────────────────────────────────

/// A live transcription session. Borrows the engine immutably.
pub struct Session<'a> {
    engine: &'a Engine,
    speech_gate: SpeechGate,
    generator: Generator,
    aligner: Aligner,
    correction: Option<(SharedCorrectionEngine, Corrector)>,
    options: SessionOptions,
}

impl<'a> Session<'a> {
    /// Feed raw 16kHz mono f32 audio samples.
    ///
    /// Returns `Ok(Some(update))` when new text is available,
    /// `Ok(None)` if the audio was silence or not enough has buffered yet.
    pub fn feed(&mut self, samples: &[f32]) -> Result<Option<Update>, Exception> {
        // Layer 1: SpeechGate — buffer, chunk, VAD
        let result = match self.speech_gate.feed(samples) {
            Some(r) => r,
            None => return Ok(None),
        };
        if matches!(result, FeedResult::Silence) {
            return Ok(None);
        }

        // Layer 2: Generator — decode
        tracing::debug!(
            "feed: decoding chunk {} ({} audio samples total)",
            self.speech_gate.chunk_count,
            self.speech_gate.audio().len()
        );
        self.generator.decode_step(
            &self.engine.model,
            &self.engine.tokenizer,
            self.speech_gate.audio(),
            self.options.language.as_str(),
            self.speech_gate.chunk_count,
            self.options.max_tokens_streaming,
        )?;

        // Layer 3: StructuredAsrOutput — split metadata from text
        let output = StructuredAsrOutput::from_raw(
            self.generator.raw_token_ids(),
            self.generator.raw_token_logprobs(),
        );
        self.aligner
            .detect_language(&self.engine.tokenizer, output.metadata_ids);

        // Layer 4: Aligner — maybe commit
        if let Some(chunk) = self.aligner.maybe_commit(
            &self.engine.aligner,
            &self.engine.tokenizer,
            self.speech_gate.audio(),
            output.text_ids,
            output.text_logprobs,
            output.metadata_token_count(),
            self.options.rollback_tokens,
        )? {
            // Layer 5: Corrector — run corrections on committed chunk
            self.run_correction(&chunk);

            // Rotate upstream layers
            self.generator.rotate(chunk.rotate.raw_tokens_to_drop);
            self.speech_gate.rotate(chunk.rotate.audio_cut_samples);
            self.speech_gate.skip_warmup();
        }

        let update = self.make_update();
        tracing::debug!(
            "feed: text={:?} asr_committed={}",
            &update.text[..update.text.len().min(80)],
            update.asr_committed_len
        );

        Ok(Some(update))
    }

    /// Finalize the session: flush remaining audio with a higher token
    /// budget and return the final transcription.
    pub fn finish(mut self) -> Result<FinishResult, Exception> {
        self.speech_gate.flush();

        if !self.speech_gate.audio().is_empty() {
            self.generator.decode_step(
                &self.engine.model,
                &self.engine.tokenizer,
                self.speech_gate.audio(),
                self.options.language.as_str(),
                self.speech_gate.chunk_count,
                self.options.max_tokens_final,
            )?;
        }

        // Align remaining uncommitted text
        let output = StructuredAsrOutput::from_raw(
            self.generator.raw_token_ids(),
            self.generator.raw_token_logprobs(),
        );
        self.aligner
            .detect_language(&self.engine.tokenizer, output.metadata_ids);

        if let Some(chunk) = self.aligner.finish_commit(
            &self.engine.aligner,
            &self.engine.tokenizer,
            self.speech_gate.audio(),
            output.text_ids,
            output.text_logprobs,
        )? {
            self.run_correction(&chunk);
            // Rotate generator so make_update doesn't double-count committed tokens
            self.generator.rotate(chunk.rotate.raw_tokens_to_drop);
        }

        let update = self.make_update();
        let corrector = self.correction.take().map(|(_, c)| c);
        Ok(FinishResult { update, corrector })
    }

    // ── Internal ────────────────────────────────────────────────────

    fn run_correction(&mut self, chunk: &crate::aligner::AlignedChunk) {
        if let Some((ref engine_arc, ref mut corrector)) = self.correction {
            let mut engine = engine_arc.lock().expect("correction engine lock poisoned");
            corrector.process_chunk(&mut engine, chunk, self.options.app_id.as_deref());
        }
    }

    fn make_update(&self) -> Update {
        let output = StructuredAsrOutput::from_raw(
            self.generator.raw_token_ids(),
            self.generator.raw_token_logprobs(),
        );

        // Build text from three parts:
        // 1. Correction-committed text (corrected, truly final)
        // 2. ASR-committed text not yet corrected (decoded from tokens)
        // 3. In-progress tail (pending text tokens from generator)
        //
        // When no corrector is active, parts 1+2 collapse to just "ASR committed".

        let correction_text = self
            .correction
            .as_ref()
            .map(|(_, c)| c.committed_text())
            .unwrap_or("");

        // Decode uncommitted text tokens (pending in generator, not yet ASR-committed)
        let pending_text = if output.text_ids.is_empty() {
            String::new()
        } else {
            self.engine
                .tokenizer
                .decode(output.text_ids, true)
                .unwrap_or_default()
        };

        // ASR-committed but not yet correction-committed: decode from aligner tokens
        // that haven't been consumed by the corrector yet.
        let asr_only_text = if self.correction.is_some() {
            // Corrector consumed all committed tokens via process_chunk,
            // so there's no gap between correction-committed and pending.
            String::new()
        } else {
            // No corrector — decode all ASR-committed tokens
            if self.aligner.committed_text_tokens().is_empty() {
                String::new()
            } else {
                self.engine
                    .tokenizer
                    .decode(self.aligner.committed_text_tokens(), true)
                    .unwrap_or_default()
            }
        };

        let text = format!("{correction_text}{asr_only_text}{pending_text}");
        let correction_committed_len = correction_text.len();
        let asr_committed_len = correction_text.len() + asr_only_text.len();

        Update {
            text,
            asr_committed_len,
            correction_committed_len,
            alignments: self.aligner.committed_alignments().to_vec(),
            detected_language: self.aligner.detected_language().to_string(),
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Compute per-word logprob statistics by mapping decoder tokens to aligner words.
fn word_logprob_stats(
    tokenizer: &Tokenizer,
    token_ids: &[TokenId],
    token_logprobs: &[TokenLogprob],
    word_count: usize,
) -> Result<Vec<Confidence>, TranscribeError> {
    if token_logprobs.is_empty() || word_count == 0 {
        return Ok(vec![]);
    }

    let mut per_token_texts: Vec<String> = Vec::with_capacity(token_ids.len());
    for (i, _) in token_ids.iter().enumerate() {
        let with = tokenizer
            .decode(
                &token_ids[..=i],
                true,
            )
            .unwrap_or_default();
        let without = if i > 0 {
            tokenizer
                .decode(
                    &token_ids[..i],
                    true,
                )
                .unwrap_or_default()
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

    let mut word_logprobs: Vec<Vec<&TokenLogprob>> = vec![Vec::new(); word_count];
    let mut word_idx = 0;
    let mut seen_chars_in_word = false;

    for (i, text) in per_token_texts.iter().enumerate() {
        if word_idx >= word_count {
            break;
        }
        let lp = token_logprobs.get(i);

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
