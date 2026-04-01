use facet::Facet;

use bee_asr::forced_aligner::ForcedAlignItem;

#[derive(Clone, Debug, Facet)]
pub struct TranscribeWavResult {
    pub transcript: String,
    pub qwen_words: Vec<ForcedAlignItem>,
}

#[vox::service]
pub trait BeeMl {
    async fn transcribe_wav(&self, wav_bytes: Vec<u8>) -> Result<TranscribeWavResult, String>;
}
