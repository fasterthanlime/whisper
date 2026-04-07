use std::collections::HashMap;
use std::path::{Path, PathBuf};

use bee_correct::g2p::CachedEspeakG2p;
use bee_correct::judge::{CorrectionEventSink, TwoStageJudge};
use bee_phonetic::phonetic_verify::CandidateFeatureRow;
use bee_phonetic::{PhoneticIndex, SeedDataset};
use bee_types::{CorrectionEvent, IdentifierFlags, SpanContext, TranscriptSpan};

/// Data stashed from correct_process so correct_teach can call teach_span.
pub(crate) struct PendingEdit {
    pub span: TranscriptSpan,
    pub candidates: Vec<(CandidateFeatureRow, IdentifierFlags)>,
    pub ctx: SpanContext,
    pub chosen_alias_id: Option<u32>,
}

pub(crate) struct CorrectionEngine {
    pub(crate) judge: TwoStageJudge,
    pub(crate) index: PhoneticIndex,
    pub(crate) g2p: CachedEspeakG2p,
    /// session_id -> edit_id -> PendingEdit
    pub(crate) pending: HashMap<String, HashMap<String, PendingEdit>>,
    /// Accumulated events since last save.
    pub(crate) event_log: Vec<CorrectionEvent>,
    /// Path to persist events (JSONL).
    pub(crate) events_path: Option<PathBuf>,
}

impl CorrectionEventSink for CorrectionEngine {
    fn log_event(&mut self, event: &CorrectionEvent) {
        self.event_log.push(event.clone());
    }
}

pub(crate) fn load_correction_engine(
    dataset_dir: &Path,
    events_path: Option<PathBuf>,
    gate_threshold: f32,
    ranker_threshold: f32,
) -> Result<CorrectionEngine, String> {
    let dataset = SeedDataset::load(dataset_dir).map_err(|e| format!("load dataset: {e}"))?;
    let index = dataset.phonetic_index();

    let g2p = CachedEspeakG2p::english().map_err(|e| format!("init g2p: {e}"))?;

    let gt = if gate_threshold > 0.0 {
        gate_threshold
    } else {
        0.5
    };
    let rt = if ranker_threshold > 0.0 {
        ranker_threshold
    } else {
        0.2
    };
    let mut judge = TwoStageJudge::new(gt, rt);

    // Replay persisted events to rebuild TermMemory
    let resolved_path = events_path.clone();
    if let Some(ref path) = resolved_path {
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
        pending: HashMap::new(),
        event_log: Vec::new(),
        events_path: resolved_path,
    })
}
