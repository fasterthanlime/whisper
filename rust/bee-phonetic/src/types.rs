use facet::Facet;

#[derive(Debug, Clone, Facet, PartialEq, Eq)]
pub struct VocabRow {
    pub id: i64,
    pub term: String,
    pub spoken_auto: String,
    pub spoken_override: Option<String>,
    pub reviewed_ipa: Option<String>,
    pub reviewed: bool,
    pub description: Option<String>,
}

impl VocabRow {
    pub fn spoken(&self) -> &str {
        self.spoken_override.as_deref().unwrap_or(&self.spoken_auto)
    }
}

#[derive(Debug, Clone, Facet, PartialEq, Eq)]
pub struct ReviewedConfusionSurfaceRow {
    pub id: i64,
    pub term: String,
    pub surface_form: String,
    pub reviewed_ipa: Option<String>,
    pub status: String,
    pub source: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Facet, PartialEq, Eq)]
pub struct AuthoredSentenceRow {
    pub term: String,
    pub sentence: String,
    pub kind: String,
    pub surface_form: Option<String>,
}

#[derive(Debug, Clone, Facet, PartialEq, Eq)]
pub struct AuthoredRecordingRow {
    pub term: String,
    pub sentence: String,
    pub take_no: i64,
    pub wav_path: String,
    pub qwen_clean: Option<String>,
}
