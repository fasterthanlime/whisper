use facet::Facet;
use vox::{Rx, Tx};

use bee_transcribe::{AlignedWord, Update};

#[derive(Clone, Debug, Facet)]
pub struct TranscribeWavResult {
    pub transcript: String,
    pub words: Vec<AlignedWord>,
}

#[repr(u8)]
#[derive(Clone, Debug, Facet)]
pub enum AliasSource {
    Canonical,
    Spoken,
    Identifier,
    Confusion,
    G2p,
}

#[repr(u8)]
#[derive(Clone, Debug, Facet)]
pub enum RetrievalIndexView {
    RawIpa2,
    RawIpa3,
    ReducedIpa2,
    ReducedIpa3,
    Feature2,
    Feature3,
    ShortQueryFallback,
}

#[derive(Clone, Debug, Facet)]
pub struct IdentifierFlags {
    pub acronym_like: bool,
    pub has_digits: bool,
    pub snake_like: bool,
    pub camel_like: bool,
    pub symbol_like: bool,
}

#[derive(Clone, Debug, Facet)]
pub struct CorrectionRequest {
    pub transcript: String,
    pub words: Vec<AlignedWord>,
    pub max_span_words: u8,
    pub shortlist_limit: u16,
    pub verify_limit: u16,
    pub reranker_candidate_limit: u16,
    pub include_debug_trace: bool,
}

#[derive(Clone, Debug, Facet)]
pub struct AcceptedEdit {
    pub token_start: u32,
    pub token_end: u32,
    pub char_start: u32,
    pub char_end: u32,
    pub original_text: String,
    pub replacement_text: String,
    pub term: String,
    pub score: f32,
    pub phonetic_score: f32,
}

#[derive(Clone, Debug, Facet)]
pub struct CorrectionResult {
    pub original_transcript: String,
    pub corrected_transcript: String,
    pub accepted_edits: Vec<AcceptedEdit>,
}

#[derive(Clone, Debug, Facet)]
pub struct TermInspectionRequest {
    pub term: String,
}

#[derive(Clone, Debug, Facet)]
pub struct TermAliasView {
    pub alias_text: String,
    pub alias_source: AliasSource,
    pub ipa_tokens: Vec<String>,
    pub reduced_ipa_tokens: Vec<String>,
    pub feature_tokens: Vec<String>,
    pub identifier_flags: IdentifierFlags,
}

#[derive(Clone, Debug, Facet)]
pub struct TermInspectionResult {
    pub term: String,
    pub aliases: Vec<TermAliasView>,
}

#[derive(Clone, Debug, Facet)]
pub struct SpanDebugView {
    pub token_start: u32,
    pub token_end: u32,
    pub char_start: u32,
    pub char_end: u32,
    pub start_sec: f64,
    pub end_sec: f64,
    pub text: String,
    pub ipa_tokens: Vec<String>,
    pub reduced_ipa_tokens: Vec<String>,
    pub feature_tokens: Vec<String>,
}

#[derive(Clone, Debug, Facet)]
pub struct FilterDecision {
    pub name: String,
    pub passed: bool,
    pub detail: String,
}

#[derive(Clone, Debug, Facet)]
pub struct CandidateFeatureDebug {
    pub matched_view: RetrievalIndexView,
    pub qgram_overlap: u16,
    pub total_qgram_overlap: u16,
    pub best_view_score: f32,
    pub cross_view_support: u8,
    pub token_count_match: bool,
    pub phone_count_delta: i16,
    pub token_bonus: f32,
    pub phone_bonus: f32,
    pub extra_length_penalty: f32,
    pub structure_bonus: f32,
    pub coarse_score: f32,
    pub token_distance: u16,
    pub token_weighted_distance: f32,
    pub token_boundary_penalty: f32,
    pub token_max_len: u16,
    pub token_score: f32,
    pub feature_distance: f32,
    pub feature_weighted_distance: f32,
    pub feature_boundary_penalty: f32,
    pub feature_max_len: u16,
    pub feature_score: f32,
    pub feature_bonus: f32,
    pub feature_gate_token_ok: bool,
    pub feature_gate_coarse_ok: bool,
    pub feature_gate_phone_ok: bool,
    pub short_guard_applied: bool,
    pub short_guard_onset_match: bool,
    pub short_guard_passed: bool,
    pub low_content_guard_applied: bool,
    pub low_content_guard_passed: bool,
    pub acceptance_floor_passed: bool,
    pub used_feature_bonus: bool,
    pub phonetic_score: f32,
    pub acceptance_score: f32,
    pub verified: bool,
}

#[derive(Clone, Debug, Facet)]
pub struct RetrievalCandidateDebug {
    pub alias_id: u32,
    pub term: String,
    pub alias_text: String,
    pub alias_source: AliasSource,
    pub alias_ipa_tokens: Vec<String>,
    pub alias_reduced_ipa_tokens: Vec<String>,
    pub alias_feature_tokens: Vec<String>,
    pub identifier_flags: IdentifierFlags,
    pub features: CandidateFeatureDebug,
    pub filter_decisions: Vec<FilterDecision>,
    pub reached_reranker: bool,
    pub accepted: bool,
}

#[derive(Clone, Debug, Facet)]
pub struct SpanDebugTrace {
    pub span: SpanDebugView,
    pub candidates: Vec<RetrievalCandidateDebug>,
    pub judge_options: Vec<JudgeOptionDebug>,
}

#[derive(Clone, Debug, Facet)]
pub struct JudgeOptionDebug {
    pub alias_id: Option<u32>,
    pub term: String,
    pub is_keep_original: bool,
    pub score: f32,
    pub probability: f32,
    pub chosen: bool,
}

#[derive(Clone, Debug, Facet)]
pub struct JudgeStateDebug {
    pub update_count: u32,
    pub learning_rate: f32,
    pub feature_names: Vec<String>,
    pub weights: Vec<f32>,
}

#[derive(Clone, Debug, Facet)]
pub struct RerankerCandidateDebug {
    pub index: u16,
    pub text: String,
    pub is_keep_original: bool,
    pub heuristic_score: f32,
    pub model_score: f32,
    pub chosen: bool,
}

#[derive(Clone, Debug, Facet)]
pub struct RerankerDebugTrace {
    pub region_index: u16,
    pub left_context: String,
    pub original_span: String,
    pub right_context: String,
    pub candidates: Vec<RerankerCandidateDebug>,
}

#[derive(Clone, Debug, Facet)]
pub struct TimingBreakdown {
    pub span_enumeration_ms: u32,
    pub retrieval_ms: u32,
    pub verify_ms: u32,
    pub rerank_ms: u32,
    pub total_ms: u32,
}

#[derive(Clone, Debug, Facet)]
pub struct CorrectionDebugResult {
    pub result: CorrectionResult,
    pub spans: Vec<SpanDebugTrace>,
    pub reranker_regions: Vec<RerankerDebugTrace>,
    pub timings: TimingBreakdown,
}

#[derive(Clone, Debug, Facet)]
pub struct RetrievalPrototypeProbeRequest {
    pub transcript: String,
    pub words: Vec<AlignedWord>,
    pub max_span_words: u8,
    pub shortlist_limit: u16,
    pub verify_limit: u16,
}

#[derive(Clone, Debug, Facet)]
pub struct RetrievalPrototypeProbeResult {
    pub transcript: String,
    pub spans: Vec<SpanDebugTrace>,
    pub timings: TimingBreakdown,
    pub judge_state: JudgeStateDebug,
}

#[derive(Clone, Debug, Facet)]
pub struct TeachRetrievalPrototypeJudgeRequest {
    pub probe: RetrievalPrototypeProbeRequest,
    pub span_token_start: u32,
    pub span_token_end: u32,
    pub choose_keep_original: bool,
    pub chosen_alias_id: Option<u32>,
    pub reject_group: bool,
    pub rejected_group_spans: Vec<RejectedGroupSpan>,
}

#[derive(Clone, Debug, Facet)]
pub struct RejectedGroupSpan {
    pub token_start: u32,
    pub token_end: u32,
}

#[derive(Clone, Debug, Facet)]
pub struct RetrievalPrototypeTeachingDeckRequest {
    pub limit: u32,
    pub include_counterexamples: bool,
}

#[derive(Clone, Debug, Facet)]
pub struct RetrievalPrototypeTeachingCase {
    pub case_id: String,
    pub suite: String,
    pub target_term: String,
    pub source_text: String,
    pub transcript: String,
    pub should_abstain: bool,
    pub take: Option<i64>,
    pub audio_path: Option<String>,
    pub surface_form: Option<String>,
}

#[derive(Clone, Debug, Facet)]
pub struct RetrievalPrototypeTeachingDeckResult {
    pub cases: Vec<RetrievalPrototypeTeachingCase>,
}

#[derive(Clone, Debug, Facet)]
pub struct RetrievalPrototypeEvalRequest {
    pub limit: u32,
    pub max_span_words: u8,
    pub shortlist_limit: u16,
    pub verify_limit: u16,
}

#[derive(Clone, Debug, Facet)]
pub struct RetrievalEvalTermSummary {
    pub term: String,
    pub cases: u32,
    pub top1_hits: u32,
    pub top3_hits: u32,
    pub top10_hits: u32,
}

#[derive(Clone, Debug, Facet)]
pub struct RetrievalEvalMiss {
    pub recording_id: u32,
    pub suite: String,
    pub term: String,
    pub transcript: String,
    pub best_span_text: String,
}

#[derive(Clone, Debug, Facet)]
pub struct JudgeEvalFailure {
    pub case_id: String,
    pub suite: String,
    pub target_term: String,
    pub transcript: String,
    pub expected_action: String,
    pub chosen_action: String,
    pub chosen_span_text: String,
    pub chosen_probability: f32,
}

#[derive(Clone, Debug, Facet)]
pub struct RetrievalPrototypeEvalResult {
    pub evaluated_cases: u32,
    pub top1_hits: u32,
    pub top3_hits: u32,
    pub top10_hits: u32,
    pub judge_correct: u32,
    pub judge_replace_correct: u32,
    pub judge_abstain_correct: u32,
    pub misses: Vec<RetrievalEvalMiss>,
    pub judge_failures: Vec<JudgeEvalFailure>,
    pub per_term: Vec<RetrievalEvalTermSummary>,
}

#[vox::service]
pub trait BeeMl {
    async fn transcribe_wav(&self, wav_bytes: Vec<u8>) -> Result<TranscribeWavResult, String>;

    /// Stream audio chunks (16kHz mono f32) and receive incremental transcription updates.
    async fn stream_transcribe(
        &self,
        audio_in: Rx<Vec<f32>>,
        updates_out: Tx<Update>,
    ) -> Result<(), String>;

    async fn correct_transcript(
        &self,
        request: CorrectionRequest,
    ) -> Result<CorrectionResult, String>;

    async fn debug_correction(
        &self,
        request: CorrectionRequest,
    ) -> Result<CorrectionDebugResult, String>;

    async fn probe_retrieval_prototype(
        &self,
        request: RetrievalPrototypeProbeRequest,
    ) -> Result<RetrievalPrototypeProbeResult, String>;

    async fn teach_retrieval_prototype_judge(
        &self,
        request: TeachRetrievalPrototypeJudgeRequest,
    ) -> Result<RetrievalPrototypeProbeResult, String>;

    async fn load_retrieval_prototype_teaching_deck(
        &self,
        request: RetrievalPrototypeTeachingDeckRequest,
    ) -> Result<RetrievalPrototypeTeachingDeckResult, String>;

    async fn inspect_term(
        &self,
        request: TermInspectionRequest,
    ) -> Result<TermInspectionResult, String>;

    async fn run_retrieval_prototype_eval(
        &self,
        request: RetrievalPrototypeEvalRequest,
    ) -> Result<RetrievalPrototypeEvalResult, String>;
}
