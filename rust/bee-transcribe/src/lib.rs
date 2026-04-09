//! High-level streaming transcription built on `bee-qwen3-asr`.

mod asr;
pub mod audio_buffer;
pub mod audio_filter;
pub mod correct;
pub mod corrector;
pub mod decode_session;
mod mlx_stuff;
pub mod session;
pub mod text_buffer;
mod timing;
mod types;
mod wav_util;
pub mod zipa_align;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use bee_vad::SileroVad;
use bee_zipa_mlx::infer::ZipaInference;
pub use mlx_stuff::*;
pub use types::*;
pub use wav_util::decode_wav;

use bee_qwen3_asr::config::AsrConfig;
use bee_qwen3_asr::forced_aligner::ForcedAligner;
use bee_qwen3_asr::load;
use bee_qwen3_asr::model::Qwen3ASRModel;
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
    pub snapshot: SessionSnapshot,
    pub session_audio: audio_buffer::AudioBuffer,
    pub corrector: Option<Corrector>,
}

use crate::asr::load_tokenizer;
use crate::correct::CorrectionEngine;
use crate::corrector::Corrector;

// ── Engine ──────────────────────────────────────────────────────────────

/// Holds loaded model weights, tokenizer, and forced aligner.
///
/// Immutable after construction — multiple sessions can borrow it
/// concurrently via `&Engine`.
pub struct Engine {
    tokenizer: Tokenizer,
    model: Qwen3ASRModel,
    aligner: ForcedAligner,
    zipa: ZipaInference,
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

        let aligner = ForcedAligner::load(
            config.aligner_dir,
            tokenizer.clone(),
            config.share_aligner_audio_tower.then_some(&model),
        )?;
        log::info!("Aligner loaded");

        let zipa = ZipaInference::load_quantized_bundle_dir(config.zipa_bundle_dir)
            .map_err(|e| Exception::custom(format!("zipa inference: {e}")))?;
        log::info!("ZIPA loaded");

        let st_path = config.silero_dir.join("model.safetensors");
        let vad_tensors = mlx_rs::Array::load_safetensors(&st_path)
            .map_err(|e| Exception::custom(format!("vad weights load: {e}")))?;

        let correction = if let Some(dataset_dir) = config.correction_dir {
            let cc = crate::correct::CorrectionConfig {
                dataset_dir,
                events_path: config.correction_events_path.clone(),
                gate_threshold: 0.0,
                ranker_threshold: 0.0,
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
            zipa,
            vad_tensors,
            correction,
        })
    }

    /// Create a new transcription session.
    pub fn session(&self, options: SessionOptions) -> Result<session::Session<'_>, Exception> {
        let vad = SileroVad::from_tensors(&self.vad_tensors)
            .map_err(|e| Exception::custom(format!("vad creation failed: {e}")))?;
        let correction = self
            .correction
            .as_ref()
            .map(|ce| (ce.clone(), Corrector::new()));
        Ok(session::Session::new(
            &self.model,
            &self.tokenizer,
            &self.aligner,
            &self.zipa,
            vad,
            options,
            correction,
        ))
    }
}
