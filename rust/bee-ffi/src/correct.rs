use std::path::{Path, PathBuf};

use bee_correct::g2p::CachedEspeakG2p;
use bee_correct::judge::TwoStageJudge;
use bee_phonetic::{PhoneticIndex, SeedDataset};

pub(crate) struct CorrectionEngine {
    pub(crate) judge: TwoStageJudge,
    pub(crate) index: PhoneticIndex,
    pub(crate) g2p: CachedEspeakG2p,
    pub(crate) events_path: Option<PathBuf>,
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
    let judge = TwoStageJudge::new(gt, rt);

    Ok(CorrectionEngine {
        judge,
        index,
        g2p,
        events_path,
    })
}
