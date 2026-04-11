pub mod alignment;
pub mod dataset;
pub mod feature_view;
pub mod phonetic_index;
pub mod phonetic_lexicon;
pub mod phonetic_verify;
pub mod prototype;
pub mod region_proposal;
pub mod types;
pub mod word_split;

pub use alignment::{
    AlignmentOp, AlignmentOpKind, AlignmentWindowCandidate, ComparisonToken, TokenAlignment,
    align_token_sequences, align_token_sequences_with_left_word_boundaries,
    top_right_anchor_windows,
};
pub use dataset::{
    CounterexampleRecordingRow, RecordingExampleRow, SeedDataset, SeedDatasetError,
    SeedDatasetValidationError, SeedTermRow, SentenceExampleRow,
};
pub use feature_view::{
    FeatureEditKind, FeatureEditOp, FeatureSimilarityDetails, feature_similarity,
    feature_similarity_for_tokens, feature_similarity_from_vectors, feature_tokens_for_ipa,
    feature_vector_for_token, feature_vectors_for_ipa,
};
pub use phonetic_index::{
    IndexView, PhoneticIndex, RetrievalCandidate, RetrievalQuery, build_index, query_index,
};
pub use phonetic_lexicon::{
    AliasSource, IdentifierFlags, LexiconAlias, build_phonetic_lexicon, derive_identifier_flags,
    normalize_ipa_for_comparison, normalize_ipa_for_comparison_with_spans, reduce_ipa_tokens,
};
pub use phonetic_verify::{
    CandidateFeatureRow, VerifiedCandidate, score_shortlist, verify_shortlist,
};
pub use prototype::{
    PhonemeSimilarityDetails, TokenEditKind, TokenEditOp, parse_reviewed_ipa, phoneme_similarity,
    phoneme_similarity_details,
};
pub use region_proposal::{
    TranscriptAlignmentTiming, TranscriptAlignmentToken, TranscriptSpan,
    enumerate_transcript_spans_with,
};
pub use types::{ReviewedConfusionSurfaceRow, VocabRow};
pub use word_split::{
    SentenceWordToken, count_sentence_words, sentence_word_tokens, split_sentence_words,
};
