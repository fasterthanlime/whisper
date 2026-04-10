//! Correction engine: judge, phonetic index, G2P.
//! Moved from bee-ffi to live alongside the transcription pipeline.

use std::path::{Path, PathBuf};

use crate::g2p::CachedEspeakG2p;
use bee_correct::judge::{CorrectionEventSink, TwoStageJudge};
use bee_phonetic::phonetic_verify::CandidateFeatureRow;
use bee_phonetic::{PhoneticIndex, SeedDataset};
use bee_types::{CorrectionEvent, IdentifierFlags, SpanContext, TranscriptSpan};

/// Data stashed from correct_process so correct_teach can call teach_span.
pub struct PendingEdit {
    pub span: TranscriptSpan,
    pub candidates: Vec<(CandidateFeatureRow, IdentifierFlags)>,
    pub ctx: SpanContext,
    pub chosen_alias_id: Option<u32>,
}

pub struct CorrectionEngine {
    pub judge: TwoStageJudge,
    pub index: PhoneticIndex,
    pub g2p: CachedEspeakG2p,
    /// Accumulated events since last save.
    pub event_log: Vec<CorrectionEvent>,
    /// Path to persist events (JSONL).
    pub events_path: Option<PathBuf>,
}

impl CorrectionEventSink for CorrectionEngine {
    fn log_event(&mut self, event: &CorrectionEvent) {
        self.event_log.push(event.clone());
    }
}

/// Configuration for loading a correction engine.
pub struct CorrectionConfig<'a> {
    pub dataset_dir: &'a Path,
    pub events_path: Option<PathBuf>,
    pub gate_threshold: f32,
    pub ranker_threshold: f32,
}

pub fn load_correction_engine(config: &CorrectionConfig<'_>) -> Result<CorrectionEngine, String> {
    let dataset =
        SeedDataset::load(config.dataset_dir).map_err(|e| format!("load dataset: {e}"))?;
    let index = dataset.phonetic_index();

    let g2p = CachedEspeakG2p::english(config.dataset_dir).map_err(|e| format!("init g2p: {e}"))?;

    let gt = if config.gate_threshold > 0.0 {
        config.gate_threshold
    } else {
        0.5
    };
    let rt = if config.ranker_threshold > 0.0 {
        config.ranker_threshold
    } else {
        0.2
    };
    let mut judge = TwoStageJudge::new(gt, rt, Some(config.dataset_dir));

    // Replay persisted events to rebuild TermMemory
    if let Some(ref path) = config.events_path {
        if path.exists() {
            match std::fs::read_to_string(path) {
                Ok(contents) => {
                    let events: Vec<CorrectionEvent> = contents
                        .lines()
                        .filter(|l| !l.trim().is_empty())
                        .filter_map(|line| match facet_json::from_str::<CorrectionEvent>(line) {
                            Ok(e) => Some(e),
                            Err(err) => {
                                tracing::warn!("skipping malformed event line: {err}");
                                None
                            }
                        })
                        .collect();
                    if !events.is_empty() {
                        judge.replay_events(&events);
                    }
                }
                Err(e) => {
                    tracing::warn!("could not read events file {}: {e}", path.display());
                }
            }
        }
    }

    Ok(CorrectionEngine {
        judge,
        index,
        g2p,
        event_log: Vec::new(),
        events_path: config.events_path.clone(),
    })
}
