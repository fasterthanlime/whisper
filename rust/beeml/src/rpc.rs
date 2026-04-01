use facet::Facet;

use bee_transcribe::AlignedWord;

#[derive(Clone, Debug, Facet)]
pub struct TranscribeWavResult {
    pub transcript: String,
    pub words: Vec<AlignedWord>,
}

#[vox::service]
pub trait BeeMl {
    async fn transcribe_wav(&self, wav_bytes: Vec<u8>) -> Result<TranscribeWavResult, String>;
}
