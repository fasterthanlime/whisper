use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::time::SystemTime;

use bee_phonetic::{sentence_word_tokens, CandidateFeatureRow};
// Re-export types that consumers need
use bee_types::{AliasSource, IdentifierFlags, SentenceWordToken, TranscriptSpan};
pub use bee_types::{CorrectionEvent, SpanContext};

use crate::sparse_ftrl::{Feature, SparseFtrl};

const LOW_CONTENT: &[&str] = &[
    "a", "an", "and", "the", "then", "that", "this", "these", "those", "if", "you", "we", "they",
    "he", "she", "it", "i", "me", "my", "your", "our", "their", "him", "her", "them", "about",
    "not", "sure", "what", "yeah", "well", "oh", "hmm", "uh", "um", "want", "some", "there",
    "here",
];

// ── Dense feature layout ────────────────────────────────────────────

// Each candidate is scored independently: "should this replacement happen?"
// All candidates share the same feature space so learning transfers across cases.
pub const FEATURE_NAMES: &[&str] = &[
    "bias",
    "acceptance_score",
    "phonetic_score",
    "coarse_score",
    "token_score",
    "feature_score",
    "feature_bonus",
    "best_view_score",
    "cross_view_support",
    "qgram_overlap",
    "total_qgram_overlap",
    "token_count_match",
    "phone_closeness",
    "alias_source_spoken",
    "alias_source_identifier",
    "alias_source_confusion",
    "identifier_acronym",
    "identifier_digits",
    "identifier_snake",
    "identifier_camel",
    "identifier_symbol",
    "short_guard_passed",
    "low_content_guard_passed",
    "acceptance_floor_passed",
    "verified",
    "span_token_count",
    "span_phone_count",
    "span_low_content",
    // ASR uncertainty features (indices 28-31)
    "span_mean_logprob",
    "span_min_logprob",
    "span_mean_margin",
    "span_min_margin",
    // Memory features (indices 32-37)
    "candidate_prior_accept_count",
    "candidate_prior_reject_count",
    "candidate_total_count",
    "candidate_recent_accept_count",
    "candidate_session_recency",
    "span_text_prior_correct_count",
];

pub const NUM_DENSE: usize = 38;

/// Offset for sparse hashed features so they don't collide with dense indices 0..31.
pub const SPARSE_OFFSET: u64 = 1000;
/// Hash space for sparse context features.
const SPARSE_BUCKETS: u64 = 1 << 14; // 16384

/// Threshold for the judge to accept a candidate replacement.
/// If no candidate's probability exceeds this, keep original.
const ACCEPT_THRESHOLD: f32 = 0.5;

// ── Hashing ─────────────────────────────────────────────────────────

/// FNV-1a hash of a feature name string, mapped into the sparse bucket space.
fn hash_feature(name: &str) -> u64 {
    let mut hasher = fnv::FnvHasher::default();
    name.hash(&mut hasher);
    SPARSE_OFFSET + (hasher.finish() % SPARSE_BUCKETS)
}

/// Build sparse context features for one candidate.
fn context_features(ctx: &SpanContext, candidate_term: &str) -> Vec<Feature> {
    let mut features = Vec::with_capacity(20);
    let term_lower = candidate_term.to_ascii_lowercase();

    // Left context tokens
    if let Some(l1) = ctx.left_tokens.first() {
        features.push(Feature {
            index: hash_feature(&format!("L1={l1}")),
            value: 1.0,
        });
        features.push(Feature {
            index: hash_feature(&format!("TERM={term_lower}|L1={l1}")),
            value: 1.0,
        });
    }
    if ctx.left_tokens.len() >= 2 {
        let bigram = format!("{}_{}", ctx.left_tokens[0], ctx.left_tokens[1]);
        features.push(Feature {
            index: hash_feature(&format!("L2={bigram}")),
            value: 1.0,
        });
        features.push(Feature {
            index: hash_feature(&format!("TERM={term_lower}|L2={bigram}")),
            value: 1.0,
        });
    }

    // Right context tokens
    if let Some(r1) = ctx.right_tokens.first() {
        features.push(Feature {
            index: hash_feature(&format!("R1={r1}")),
            value: 1.0,
        });
        features.push(Feature {
            index: hash_feature(&format!("TERM={term_lower}|R1={r1}")),
            value: 1.0,
        });
    }
    if ctx.right_tokens.len() >= 2 {
        let bigram = format!("{}_{}", ctx.right_tokens[0], ctx.right_tokens[1]);
        features.push(Feature {
            index: hash_feature(&format!("R2={bigram}")),
            value: 1.0,
        });
        features.push(Feature {
            index: hash_feature(&format!("TERM={term_lower}|R2={bigram}")),
            value: 1.0,
        });
    }

    // Candidate identity
    features.push(Feature {
        index: hash_feature(&format!("TERM={term_lower}")),
        value: 1.0,
    });

    // Context type flags
    if ctx.code_like {
        features.push(Feature {
            index: hash_feature("CTX=code"),
            value: 1.0,
        });
    }
    if ctx.prose_like {
        features.push(Feature {
            index: hash_feature("CTX=prose"),
            value: 1.0,
        });
    }
    if ctx.list_like {
        features.push(Feature {
            index: hash_feature("CTX=list"),
            value: 1.0,
        });
    }
    if ctx.sentence_start {
        features.push(Feature {
            index: hash_feature("CTX=sent_start"),
            value: 1.0,
        });
    }

    // App context
    if let Some(app) = &ctx.app_id {
        features.push(Feature {
            index: hash_feature(&format!("APP={app}")),
            value: 1.0,
        });
        features.push(Feature {
            index: hash_feature(&format!("TERM={term_lower}|APP={app}")),
            value: 1.0,
        });
    }

    features
}

// ── Memory ──────────────────────────────────────────────────────────

/// Per-term memory entry tracking accept/reject history.
#[derive(Clone, Debug, Default)]
struct TermMemoryEntry {
    accept_count: u32,
    reject_count: u32,
    recent_accept_count: u32,
    last_used: Option<SystemTime>,
}

/// Memory lookup passed to feature building.
#[derive(Clone, Debug, Default)]
pub struct TermMemory {
    terms: HashMap<String, TermMemoryEntry>,
    /// How many times a span text was kept as correct (user chose keep_original).
    span_text_correct: HashMap<String, u32>,
}

impl TermMemory {
    fn get(&self, term: &str) -> Option<&TermMemoryEntry> {
        self.terms.get(&term.to_ascii_lowercase())
    }

    fn span_text_count(&self, text: &str) -> u32 {
        self.span_text_correct
            .get(&text.to_ascii_lowercase())
            .copied()
            .unwrap_or(0)
    }

    fn record_accept(&mut self, term: &str, now: SystemTime) {
        let key = term.to_ascii_lowercase();
        let entry = self.terms.entry(key).or_default();
        entry.accept_count += 1;
        entry.recent_accept_count += 1;
        entry.last_used = Some(now);
    }

    fn record_reject(&mut self, term: &str, now: SystemTime) {
        let key = term.to_ascii_lowercase();
        let entry = self.terms.entry(key).or_default();
        entry.reject_count += 1;
        entry.last_used = Some(now);
    }

    fn record_span_correct(&mut self, span_text: &str) {
        let key = span_text.to_ascii_lowercase();
        *self.span_text_correct.entry(key).or_default() += 1;
    }
}

// ── Judge ────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct OnlineJudge {
    model: SparseFtrl,
    update_count: u32,
    memory: TermMemory,
    event_log: Vec<CorrectionEvent>,
}

#[derive(Clone, Debug)]
pub struct JudgeOption {
    pub alias_id: Option<u32>,
    pub term: String,
    pub is_keep_original: bool,
    pub score: f32,
    pub probability: f32,
    pub chosen: bool,
}

#[derive(Clone, Debug)]
pub struct JudgeExample {
    pub alias_id: u32,
    pub term: String,
    pub features: Vec<Feature>,
}

impl Default for OnlineJudge {
    fn default() -> Self {
        let mut model = SparseFtrl::new(1.0, 1.0, 0.0001, 0.001);
        seed_model(&mut model);
        let judge = Self {
            model,
            update_count: 0,
            memory: TermMemory::default(),
            event_log: Vec::new(),
        };
        tracing::info!(
            num_active = judge.model.num_active(),
            "judge initialized with seed weights"
        );
        judge
    }
}

impl OnlineJudge {
    /// Create a new judge without tracing (for batch eval).
    pub fn new_quiet() -> Self {
        let mut model = SparseFtrl::new(1.0, 1.0, 0.0001, 0.001);
        seed_model(&mut model);
        Self {
            model,
            update_count: 0,
            memory: TermMemory::default(),
            event_log: Vec::new(),
        }
    }
}

/// Seed the FTRL model by training on synthetic examples that span the
/// quality range. This teaches the model: high scores → replace, low scores → keep.
fn seed_model(model: &mut SparseFtrl) {
    // Synthetic candidates at various quality levels.
    // (accept, phonetic, coarse, token, feature, verified, target)
    let levels: &[(f64, f64, f64, f64, f64, f64, bool)] = &[
        // Strong matches → should replace
        (0.90, 0.88, 0.85, 0.86, 0.88, 1.0, true),
        (0.82, 0.80, 0.75, 0.78, 0.82, 1.0, true),
        (0.75, 0.72, 0.68, 0.70, 0.74, 1.0, true),
        (0.68, 0.65, 0.60, 0.64, 0.66, 1.0, true),
        (0.60, 0.58, 0.55, 0.56, 0.60, 1.0, true),
        // Borderline → should replace (above threshold)
        (0.55, 0.52, 0.50, 0.50, 0.54, 1.0, true),
        // Weak matches → should NOT replace
        (0.45, 0.42, 0.40, 0.40, 0.44, 1.0, false),
        (0.38, 0.35, 0.32, 0.34, 0.36, 0.0, false),
        (0.30, 0.28, 0.25, 0.26, 0.30, 0.0, false),
        (0.20, 0.18, 0.15, 0.16, 0.20, 0.0, false),
        (0.12, 0.10, 0.08, 0.10, 0.12, 0.0, false),
        (0.05, 0.04, 0.03, 0.04, 0.05, 0.0, false),
    ];

    for epoch in 0..20 {
        for &(accept, phonetic, coarse, token, feature, verified, target) in levels {
            let features =
                dense_features_from_synthetic(accept, phonetic, coarse, token, feature, verified);
            model.update(&features, target);
        }
        // Early epochs use higher alpha for faster convergence
        if epoch == 5 {
            model.alpha = 0.5;
        }
    }
    // Reset to default learning rate for online updates
    model.alpha = 0.5;
}

fn dense_features_from_synthetic(
    accept: f64,
    phonetic: f64,
    coarse: f64,
    token: f64,
    feature: f64,
    verified: f64,
) -> Vec<Feature> {
    let values = [
        1.0,                                    // bias
        accept,                                 // acceptance_score
        phonetic,                               // phonetic_score
        coarse,                                 // coarse_score
        token,                                  // token_score
        feature,                                // feature_score
        (feature - token).max(0.0),             // feature_bonus
        coarse * 0.9,                           // best_view_score
        0.33,                                   // cross_view_support
        coarse * 0.5,                           // qgram_overlap
        coarse * 0.8,                           // total_qgram_overlap
        if accept > 0.5 { 1.0 } else { 0.0 },   // token_count_match
        if accept > 0.5 { 0.80 } else { 0.40 }, // phone_closeness
        0.0,                                    // alias_source_spoken
        0.0,                                    // alias_source_identifier
        0.0,                                    // alias_source_confusion
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,                                    // identifier flags
        if verified > 0.5 { 1.0 } else { 0.0 }, // short_guard_passed
        1.0,                                    // low_content_guard_passed
        if accept > 0.35 { 1.0 } else { 0.0 },  // acceptance_floor_passed
        verified,                               // verified
        0.25,                                   // span_token_count
        0.40,                                   // span_phone_count
        0.0,                                    // span_low_content
        // ASR uncertainty: not available in synthetic data
        0.0,
        0.0,
        0.0,
        0.0,
        // Memory features: not available in synthetic data
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ];
    values
        .iter()
        .enumerate()
        .map(|(i, &v)| Feature {
            index: i as u64,
            value: v,
        })
        .collect()
}

impl OnlineJudge {
    pub fn feature_names(&self) -> Vec<String> {
        FEATURE_NAMES
            .iter()
            .map(|name| (*name).to_string())
            .collect()
    }

    /// Get dense feature weights (indices 0..NUM_DENSE) for debugging.
    pub fn weights(&self) -> Vec<f32> {
        (0..NUM_DENSE)
            .map(|i| self.model.weight_at(i as u64) as f32)
            .collect()
    }

    pub fn update_count(&self) -> u32 {
        self.update_count
    }

    pub fn score_candidates(
        &self,
        span: &TranscriptSpan,
        candidates: &[(CandidateFeatureRow, IdentifierFlags)],
        ctx: &SpanContext,
    ) -> Vec<JudgeOption> {
        let examples = build_examples(span, candidates, ctx, &self.memory);
        score_examples(&self.model, &examples)
    }

    pub fn teach_choice(
        &mut self,
        span: &TranscriptSpan,
        candidates: &[(CandidateFeatureRow, IdentifierFlags)],
        chosen_alias_id: Option<u32>,
        ctx: &SpanContext,
    ) -> Vec<JudgeOption> {
        let examples = build_examples(span, candidates, ctx, &self.memory);
        if examples.is_empty() {
            return vec![keep_original_option()];
        }

        // chosen_alias_id == None means "keep original" => all candidates are false
        for _ in 0..4 {
            for example in &examples {
                let label = Some(example.alias_id) == chosen_alias_id;
                self.model.update(&example.features, label);
            }
        }
        self.update_count += 1;

        // Update memory counters
        let now = SystemTime::now();
        let all_terms: Vec<String> = candidates.iter().map(|(c, _)| c.term.clone()).collect();
        let chosen_term = if let Some(id) = chosen_alias_id {
            let chosen = candidates.iter().find(|(c, _)| c.alias_id == id);
            if let Some((c, _)) = chosen {
                self.memory.record_accept(&c.term, now);
                // Reject all other terms
                for (other, _) in candidates {
                    if other.alias_id != id {
                        self.memory.record_reject(&other.term, now);
                    }
                }
                c.term.clone()
            } else {
                "keep_original".to_string()
            }
        } else {
            // Keep original: reject all candidates, record span text as correct
            for (c, _) in candidates {
                self.memory.record_reject(&c.term, now);
            }
            self.memory.record_span_correct(&span.text);
            "keep_original".to_string()
        };

        // Log the event
        self.event_log.push(CorrectionEvent {
            timestamp_secs: now
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            span_text: span.text.clone(),
            chosen_term,
            all_candidate_terms: all_terms,
            chosen_alias_id,
        });

        tracing::debug!(
            update_count = self.update_count,
            chosen = ?chosen_alias_id,
            num_candidates = examples.len(),
            num_active = self.model.num_active(),
            event_count = self.event_log.len(),
            "judge taught"
        );

        score_examples(&self.model, &examples)
    }

    /// Case-balanced training: each case contributes bounded weight.
    /// For canonical (gold_alias_id = Some): 1 positive + up to hard_neg_cap hard negatives.
    /// For counterexample (gold_alias_id = None): single hardest false positive as negative.
    pub fn train_balanced(
        &mut self,
        span: &TranscriptSpan,
        candidates: &[(CandidateFeatureRow, IdentifierFlags)],
        gold_alias_id: Option<u32>,
        ctx: &SpanContext,
        hard_neg_cap: usize,
    ) {
        let examples = build_examples(span, candidates, ctx, &self.memory);
        if examples.is_empty() {
            return;
        }

        if let Some(gold_id) = gold_alias_id {
            // Canonical: train gold as positive
            if let Some(gold) = examples.iter().find(|e| e.alias_id == gold_id) {
                self.model.update(&gold.features, true);
            }
            // Hard negatives: top-scoring non-gold candidates
            let mut negatives: Vec<&JudgeExample> =
                examples.iter().filter(|e| e.alias_id != gold_id).collect();
            negatives.sort_by(|a, b| {
                let sa = self.model.predict(&a.features);
                let sb = self.model.predict(&b.features);
                sb.total_cmp(&sa)
            });
            for neg in negatives.into_iter().take(hard_neg_cap) {
                self.model.update(&neg.features, false);
            }
        } else {
            // Counterexample: single hardest false positive
            let hardest = examples.iter().max_by(|a, b| {
                let sa = self.model.predict(&a.features);
                let sb = self.model.predict(&b.features);
                sa.total_cmp(&sb)
            });
            if let Some(neg) = hardest {
                self.model.update(&neg.features, false);
            }
        }
    }

    /// Case-wise softmax training: all candidates + keep_original compete.
    /// gold_index is the index into the candidate list that should win,
    /// or None if keep_original should win.
    pub fn train_softmax(
        &mut self,
        span: &TranscriptSpan,
        candidates: &[(CandidateFeatureRow, IdentifierFlags)],
        gold_alias_id: Option<u32>,
        ctx: &SpanContext,
    ) {
        let examples = build_examples(span, candidates, ctx, &self.memory);
        if examples.is_empty() {
            return;
        }

        // Build feature vecs for all candidates + keep_original
        let mut all_features: Vec<Vec<Feature>> =
            examples.iter().map(|e| e.features.clone()).collect();

        // keep_original candidate: bias=1, candidate scores=0, but context/ASR features present
        let keep_features = build_keep_original_features(span, ctx);
        all_features.push(keep_features);

        // Gold index: if gold_alias_id matches a candidate, use that index.
        // Otherwise (keep_original), gold is the last index.
        let gold_index = if let Some(gold_id) = gold_alias_id {
            examples
                .iter()
                .position(|e| e.alias_id == gold_id)
                .unwrap_or(all_features.len() - 1) // fallback to keep_original
        } else {
            all_features.len() - 1 // keep_original
        };

        self.model.update_softmax(
            &all_features
                .iter()
                .map(|f| f.as_slice())
                .collect::<Vec<_>>()
                .iter()
                .map(|f| f.to_vec())
                .collect::<Vec<_>>(),
            gold_index,
        );
    }

    /// Access the underlying model for direct scoring in eval.
    pub fn model(&self) -> &SparseFtrl {
        &self.model
    }

    /// Mutable access to the underlying model for ablated training.
    pub fn model_mut(&mut self) -> &mut SparseFtrl {
        &mut self.model
    }

    /// Freeze the first N dense feature indices so seed weights are preserved.
    pub fn freeze_dense(&mut self, n: usize) {
        self.model.freeze(0..n as u64);
    }

    /// Get a reference to the event log for persistence.
    pub fn event_log(&self) -> &[CorrectionEvent] {
        &self.event_log
    }

    /// Replay events to rebuild memory counters (e.g., on startup from JSONL).
    pub fn replay_events(&mut self, events: Vec<CorrectionEvent>) {
        for event in &events {
            let ts = std::time::UNIX_EPOCH + std::time::Duration::from_secs(event.timestamp_secs);
            if let Some(_alias_id) = event.chosen_alias_id {
                // Accept the chosen term, reject others
                self.memory.record_accept(&event.chosen_term, ts);
                for term in &event.all_candidate_terms {
                    if !term.eq_ignore_ascii_case(&event.chosen_term) {
                        self.memory.record_reject(term, ts);
                    }
                }
            } else {
                // Keep original
                for term in &event.all_candidate_terms {
                    self.memory.record_reject(term, ts);
                }
                self.memory.record_span_correct(&event.span_text);
            }
        }
        tracing::info!(
            events = events.len(),
            terms_tracked = self.memory.terms.len(),
            span_texts_tracked = self.memory.span_text_correct.len(),
            "replayed correction events"
        );
        self.event_log = events;
    }
}

// ── Two-stage judge ─────────────────────────────────────────────────

/// External event sink — caller decides storage.
pub trait CorrectionEventSink {
    fn log_event(&mut self, event: &CorrectionEvent);
}

/// Sink that captures a single event (used by `teach_span_event`).
struct CaptureEventSink(Option<CorrectionEvent>);
impl CorrectionEventSink for CaptureEventSink {
    fn log_event(&mut self, event: &CorrectionEvent) {
        self.0 = Some(event.clone());
    }
}

/// Decision output from the two-stage judge.
#[derive(Clone, Debug)]
pub struct SpanDecision {
    /// Whether the gate opened (span worth correcting).
    pub gate_open: bool,
    /// Gate probability (higher = more likely to correct).
    pub gate_prob: f32,
    /// Chosen candidate, if gate opened and a candidate exceeded ranker threshold.
    pub chosen: Option<CandidateChoice>,
    /// All candidates with ranker scores, sorted descending.
    pub options: Vec<RankedCandidate>,
}

/// A candidate chosen by the ranker.
#[derive(Clone, Debug)]
pub struct CandidateChoice {
    pub alias_id: u32,
    pub term: String,
    pub replacement_text: String,
    pub ranker_prob: f32,
}

/// A candidate scored by the ranker.
#[derive(Clone, Debug)]
pub struct RankedCandidate {
    pub alias_id: u32,
    pub term: String,
    pub ranker_prob: f32,
}

/// Two-stage correction judge using online-learned sparse linear models.
///
/// **Stage A — Gate:** A span-level binary classifier that decides whether a
/// transcript span should be corrected at all. Uses aggregate signals (ASR
/// uncertainty, phonetic similarity of best candidates, memory of past
/// corrections). If `gate_prob < gate_threshold`, the span is left as-is.
///
/// **Stage B — Ranker:** A candidate-level classifier that scores each
/// replacement candidate. Uses per-candidate features (phonetic distance,
/// frequency, identifier flags). The highest-scoring candidate is chosen
/// only if `ranker_prob >= ranker_threshold`.
///
/// Both models are FTRL-Proximal learners ([`SparseFtrl`]) that update online
/// from user feedback via [`teach_span`](Self::teach_span). They start from
/// seeded weights (hand-tuned priors) and adapt to the user's vocabulary.
///
/// A [`TermMemory`] tracks historical accept/reject counts per term, providing
/// memory-based features (e.g. "user accepted 'Kubernetes' 5 times").
#[derive(Clone, Debug)]
pub struct TwoStageJudge {
    /// Stage A: span-level "should we correct?" classifier.
    gate: SparseFtrl,
    /// Stage B: candidate-level "which replacement?" classifier.
    ranker: SparseFtrl,
    /// Historical accept/reject counts per term, used as features.
    memory: TermMemory,
    /// Minimum gate probability to attempt correction (typically 0.3–0.7).
    pub gate_threshold: f32,
    /// Minimum ranker probability to accept a candidate (typically 0.3–0.7).
    pub ranker_threshold: f32,
}

impl TwoStageJudge {
    pub fn new(
        gate_threshold: f32,
        ranker_threshold: f32,
        weights_dir: Option<&std::path::Path>,
    ) -> Self {
        let mut gate = SparseFtrl::new(0.5, 1.0, 0.0001, 0.001);
        seed_gate_model(&mut gate);
        let mut ranker = SparseFtrl::new(0.5, 1.0, 0.0001, 0.001);
        seed_ranker_model(&mut ranker);

        // Load trained weights if available (overrides seed weights)
        if let Some(dir) = weights_dir {
            let gate_path = dir.join("gate_weights.bin");
            if let Ok(mut f) = std::fs::File::open(&gate_path) {
                if let Ok(()) = gate.load_weights(&mut f) {
                    tracing::info!("Loaded gate weights from {}", gate_path.display());
                }
            }
            let ranker_path = dir.join("ranker_weights.bin");
            if let Ok(mut f) = std::fs::File::open(&ranker_path) {
                if let Ok(()) = ranker.load_weights(&mut f) {
                    tracing::info!("Loaded ranker weights from {}", ranker_path.display());
                }
            }
        }

        Self {
            gate,
            ranker,
            memory: TermMemory::default(),
            gate_threshold,
            ranker_threshold,
        }
    }

    /// Score a span: gate decision + ranked candidates.
    pub fn score_span(
        &self,
        span: &TranscriptSpan,
        candidates: &[(CandidateFeatureRow, IdentifierFlags)],
        ctx: &SpanContext,
    ) -> SpanDecision {
        let gate_features = build_gate_features(span, candidates, ctx, &self.memory);
        let gate_prob = self.gate.predict_prob(&gate_features) as f32;

        if gate_prob < self.gate_threshold {
            return SpanDecision {
                gate_open: false,
                gate_prob,
                chosen: None,
                options: Vec::new(),
            };
        }

        let examples = build_ranker_features(span, candidates, &self.memory);
        let mut options: Vec<RankedCandidate> = examples
            .iter()
            .map(|ex| RankedCandidate {
                alias_id: ex.alias_id,
                term: ex.term.clone(),
                ranker_prob: self.ranker.predict_prob(&ex.features) as f32,
            })
            .collect();
        options.sort_by(|a, b| b.ranker_prob.total_cmp(&a.ranker_prob));

        let chosen = options.first().and_then(|best| {
            if best.ranker_prob >= self.ranker_threshold {
                // Find the alias text for the chosen candidate
                let alias_text = candidates
                    .iter()
                    .find(|(c, _)| c.alias_id == best.alias_id)
                    .map(|(c, _)| c.alias_text.clone())
                    .unwrap_or_default();
                Some(CandidateChoice {
                    alias_id: best.alias_id,
                    term: best.term.clone(),
                    replacement_text: alias_text,
                    ranker_prob: best.ranker_prob,
                })
            } else {
                None
            }
        });

        SpanDecision {
            gate_open: true,
            gate_prob,
            chosen,
            options,
        }
    }

    /// Update weights from user feedback and log via sink.
    pub fn teach_span(
        &mut self,
        span: &TranscriptSpan,
        candidates: &[(CandidateFeatureRow, IdentifierFlags)],
        chosen_alias_id: Option<u32>,
        ctx: &SpanContext,
        sink: &mut dyn CorrectionEventSink,
    ) {
        let now = SystemTime::now();
        let is_correction = chosen_alias_id.is_some();

        // Train gate: positive if correcting, negative if keeping original
        let gate_features = build_gate_features(span, candidates, ctx, &self.memory);
        self.gate.update(&gate_features, is_correction);

        // Train ranker: only when correcting
        if let Some(gold_id) = chosen_alias_id {
            let examples = build_ranker_features(span, candidates, &self.memory);
            // Score all to find hard negatives
            let mut scored: Vec<_> = examples
                .iter()
                .map(|ex| (ex, self.ranker.predict_prob(&ex.features)))
                .collect();
            scored.sort_by(|a, b| b.1.total_cmp(&a.1));

            for (ex, _score) in &scored {
                if ex.alias_id == gold_id {
                    self.ranker.update(&ex.features, true);
                } else {
                    self.ranker.update(&ex.features, false);
                }
            }
        }

        // Update memory
        let all_terms: Vec<String> = candidates.iter().map(|(c, _)| c.term.clone()).collect();
        let chosen_term = if let Some(gold_id) = chosen_alias_id {
            candidates
                .iter()
                .find(|(c, _)| c.alias_id == gold_id)
                .map(|(c, _)| c.term.clone())
                .unwrap_or_else(|| "keep_original".to_string())
        } else {
            "keep_original".to_string()
        };

        if is_correction {
            self.memory.record_accept(&chosen_term, now);
            for (c, _) in candidates {
                if !c.term.eq_ignore_ascii_case(&chosen_term) {
                    self.memory.record_reject(&c.term, now);
                }
            }
        } else {
            for (c, _) in candidates {
                self.memory.record_reject(&c.term, now);
            }
            self.memory.record_span_correct(&span.text);
        }

        // Log event
        sink.log_event(&CorrectionEvent {
            timestamp_secs: now
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            span_text: span.text.clone(),
            chosen_term,
            all_candidate_terms: all_terms,
            chosen_alias_id,
        });
    }

    /// Like `teach_span`, but returns the event instead of writing to a sink.
    /// Useful when the caller IS the sink (avoids borrow conflict).
    pub fn teach_span_event(
        &mut self,
        span: &TranscriptSpan,
        candidates: &[(CandidateFeatureRow, IdentifierFlags)],
        chosen_alias_id: Option<u32>,
        ctx: &SpanContext,
    ) -> CorrectionEvent {
        let mut capture = CaptureEventSink(None);
        self.teach_span(span, candidates, chosen_alias_id, ctx, &mut capture);
        capture.0.expect("teach_span always logs one event")
    }

    /// Gate probability for diagnostics.
    pub fn gate_prob(
        &self,
        span: &TranscriptSpan,
        candidates: &[(CandidateFeatureRow, IdentifierFlags)],
        ctx: &SpanContext,
    ) -> f32 {
        let features = build_gate_features(span, candidates, ctx, &self.memory);
        self.gate.predict_prob(&features) as f32
    }

    /// Read access to memory for diagnostics.
    pub fn memory(&self) -> &TermMemory {
        &self.memory
    }

    /// Replay events to rebuild memory counters.
    pub fn replay_events(&mut self, events: &[CorrectionEvent]) {
        for event in events {
            let ts = std::time::UNIX_EPOCH + std::time::Duration::from_secs(event.timestamp_secs);
            if event.chosen_alias_id.is_some() {
                self.memory.record_accept(&event.chosen_term, ts);
                for term in &event.all_candidate_terms {
                    if !term.eq_ignore_ascii_case(&event.chosen_term) {
                        self.memory.record_reject(term, ts);
                    }
                }
            } else {
                for term in &event.all_candidate_terms {
                    self.memory.record_reject(term, ts);
                }
                self.memory.record_span_correct(&event.span_text);
            }
        }
        tracing::info!(
            events = events.len(),
            terms_tracked = self.memory.terms.len(),
            span_texts_tracked = self.memory.span_text_correct.len(),
            "replayed correction events into TwoStageJudge"
        );
    }
}

pub fn build_examples(
    span: &TranscriptSpan,
    candidates: &[(CandidateFeatureRow, IdentifierFlags)],
    ctx: &SpanContext,
    memory: &TermMemory,
) -> Vec<JudgeExample> {
    let span_token_count = (span.token_end - span.token_start) as f64;
    let span_phone_count = span.ipa_tokens.len() as f64;
    let span_low_content = is_low_content_span(&span.text) as u8 as f64;

    // ASR uncertainty (normalized by /5.0 as per design doc)
    let mean_lp = span.mean_logprob.unwrap_or(0.0) as f64 / 5.0;
    let min_lp = span.min_logprob.unwrap_or(0.0) as f64 / 5.0;
    let mean_m = span.mean_margin.unwrap_or(0.0) as f64 / 5.0;
    let min_m = span.min_margin.unwrap_or(0.0) as f64 / 5.0;

    // Span-level memory
    let span_correct_count = memory.span_text_count(&span.text);

    candidates
        .iter()
        .map(|(candidate, flags)| {
            // Per-candidate memory lookup
            let mem = memory.get(&candidate.term);
            let accept_count = mem.map(|m| m.accept_count).unwrap_or(0);
            let reject_count = mem.map(|m| m.reject_count).unwrap_or(0);
            let recent_accept = mem.map(|m| m.recent_accept_count).unwrap_or(0);
            let recency = mem
                .and_then(|m| m.last_used)
                .and_then(|t| t.elapsed().ok())
                .map(|d| {
                    if d.as_secs() < 300 {
                        1.0
                    }
                    // < 5 min
                    else if d.as_secs() < 1800 {
                        0.5
                    }
                    // < 30 min
                    else {
                        0.0
                    }
                })
                .unwrap_or(0.0);

            // Dense features (indices 0..37)
            let mut features: Vec<Feature> = vec![
                Feature {
                    index: 0,
                    value: 1.0,
                }, // bias
                Feature {
                    index: 1,
                    value: candidate.acceptance_score as f64,
                }, // acceptance_score
                Feature {
                    index: 2,
                    value: candidate.phonetic_score as f64,
                }, // phonetic_score
                Feature {
                    index: 3,
                    value: candidate.coarse_score as f64,
                }, // coarse_score
                Feature {
                    index: 4,
                    value: candidate.token_score as f64,
                }, // token_score
                Feature {
                    index: 5,
                    value: candidate.feature_score as f64,
                }, // feature_score
                Feature {
                    index: 6,
                    value: candidate.feature_bonus as f64,
                }, // feature_bonus
                Feature {
                    index: 7,
                    value: candidate.best_view_score as f64,
                }, // best_view_score
                Feature {
                    index: 8,
                    value: candidate.cross_view_support as f64 / 6.0,
                }, // cross_view_support
                Feature {
                    index: 9,
                    value: candidate.qgram_overlap as f64 / 10.0,
                }, // qgram_overlap
                Feature {
                    index: 10,
                    value: candidate.total_qgram_overlap as f64 / 20.0,
                }, // total_qgram_overlap
                Feature {
                    index: 11,
                    value: candidate.token_count_match as u8 as f64,
                }, // token_count_match
                Feature {
                    index: 12,
                    value: 1.0 / (1.0 + candidate.phone_count_delta.abs() as f64),
                }, // phone_closeness
                Feature {
                    index: 13,
                    value: (candidate.alias_source == AliasSource::Spoken) as u8 as f64,
                },
                Feature {
                    index: 14,
                    value: (candidate.alias_source == AliasSource::Identifier) as u8 as f64,
                },
                Feature {
                    index: 15,
                    value: (candidate.alias_source == AliasSource::Confusion) as u8 as f64,
                },
                Feature {
                    index: 16,
                    value: flags.acronym_like as u8 as f64,
                },
                Feature {
                    index: 17,
                    value: flags.has_digits as u8 as f64,
                },
                Feature {
                    index: 18,
                    value: flags.snake_like as u8 as f64,
                },
                Feature {
                    index: 19,
                    value: flags.camel_like as u8 as f64,
                },
                Feature {
                    index: 20,
                    value: flags.symbol_like as u8 as f64,
                },
                Feature {
                    index: 21,
                    value: candidate.short_guard_passed as u8 as f64,
                },
                Feature {
                    index: 22,
                    value: candidate.low_content_guard_passed as u8 as f64,
                },
                Feature {
                    index: 23,
                    value: candidate.acceptance_floor_passed as u8 as f64,
                },
                Feature {
                    index: 24,
                    value: candidate.verified as u8 as f64,
                },
                Feature {
                    index: 25,
                    value: span_token_count / 4.0,
                },
                Feature {
                    index: 26,
                    value: span_phone_count / 12.0,
                },
                Feature {
                    index: 27,
                    value: span_low_content,
                },
                // ASR uncertainty
                Feature {
                    index: 28,
                    value: mean_lp,
                },
                Feature {
                    index: 29,
                    value: min_lp,
                },
                Feature {
                    index: 30,
                    value: mean_m,
                },
                Feature {
                    index: 31,
                    value: min_m,
                },
                // Memory features
                Feature {
                    index: 32,
                    value: (1.0 + accept_count as f64).ln() / 3.0,
                },
                Feature {
                    index: 33,
                    value: (1.0 + reject_count as f64).ln() / 3.0,
                },
                Feature {
                    index: 34,
                    value: (1.0 + (accept_count + reject_count) as f64).ln() / 3.0,
                },
                Feature {
                    index: 35,
                    value: (1.0 + recent_accept as f64).ln() / 3.0,
                },
                Feature {
                    index: 36,
                    value: recency,
                },
                Feature {
                    index: 37,
                    value: (1.0 + span_correct_count as f64).ln() / 3.0,
                },
            ];

            // Sparse context features (hashed into offset range)
            features.extend(context_features(ctx, &candidate.term));

            JudgeExample {
                alias_id: candidate.alias_id,
                term: candidate.term.clone(),
                features,
            }
        })
        .collect()
}

pub fn score_examples(model: &SparseFtrl, examples: &[JudgeExample]) -> Vec<JudgeOption> {
    let mut options: Vec<JudgeOption> = examples
        .iter()
        .map(|example| {
            let score = model.predict(&example.features) as f32;
            let probability = model.predict_prob(&example.features) as f32;
            JudgeOption {
                alias_id: Some(example.alias_id),
                term: example.term.clone(),
                is_keep_original: false,
                score,
                probability,
                chosen: false,
            }
        })
        .collect();

    // Sort by probability descending
    options.sort_by(|lhs, rhs| {
        rhs.probability
            .total_cmp(&lhs.probability)
            .then_with(|| rhs.score.total_cmp(&lhs.score))
    });

    // Best candidate wins only if it exceeds the threshold; otherwise keep original
    let best_exceeds_threshold = options
        .first()
        .is_some_and(|best| best.probability >= ACCEPT_THRESHOLD);

    if best_exceeds_threshold {
        if let Some(first) = options.first_mut() {
            first.chosen = true;
        }
    }

    // Always include a keep_original option so callers can see it
    let mut keep = keep_original_option();
    if !best_exceeds_threshold {
        keep.chosen = true;
    }
    // keep_original probability = 1 - best candidate probability
    keep.probability = 1.0 - options.first().map(|o| o.probability).unwrap_or(0.0);
    keep.score = -options.first().map(|o| o.score).unwrap_or(0.0);
    options.push(keep);

    options
}

fn keep_original_option() -> JudgeOption {
    JudgeOption {
        alias_id: None,
        term: "keep_original".to_string(),
        is_keep_original: true,
        score: 0.0,
        probability: 1.0,
        chosen: false,
    }
}

fn is_low_content_span(text: &str) -> bool {
    let tokens: Vec<SentenceWordToken> = sentence_word_tokens(text);
    !tokens.is_empty()
        && tokens.iter().all(|token| {
            let lower = token.text.to_ascii_lowercase();
            LOW_CONTENT.iter().any(|entry| *entry == lower)
        })
}

/// Extract SpanContext from a transcript and span boundaries.
pub fn extract_span_context(transcript: &str, char_start: usize, char_end: usize) -> SpanContext {
    let before = &transcript[..char_start];
    let after = &transcript[char_end..];

    // Left tokens: split on whitespace, take last 2
    let left_words: Vec<String> = before
        .split_whitespace()
        .rev()
        .take(2)
        .map(|w| w.to_ascii_lowercase())
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();

    // Right tokens: split on whitespace, take first 2
    let right_words: Vec<String> = after
        .split_whitespace()
        .take(2)
        .map(|w| w.to_ascii_lowercase())
        .collect();

    // Code-like: check ±10 chars for code markers
    let window_start = transcript[..char_start]
        .char_indices()
        .rev()
        .nth(10)
        .map(|(i, _)| i)
        .unwrap_or(0);
    let window_end = transcript[char_end..]
        .char_indices()
        .nth(10)
        .map(|(i, _)| char_end + i)
        .unwrap_or(transcript.len());
    let window = &transcript[window_start..window_end];
    let code_markers = ["()", "{}", "::", ".", "_", "->", "=>", "fn ", "let "];
    let code_like = code_markers.iter().any(|m| window.contains(m));

    // List-like: line starts with a list marker
    let line_start = before.rfind('\n').map(|i| i + 1).unwrap_or(0);
    let line_prefix = transcript[line_start..char_start].trim_start();
    let list_like = line_prefix.starts_with('-')
        || line_prefix.starts_with('*')
        || line_prefix
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_digit());

    // Sentence start: span is at the beginning or after sentence-ending punctuation
    let sentence_start = before.is_empty() || before.trim_end().ends_with(['.', '!', '?', '\n']);

    SpanContext {
        left_tokens: left_words,
        right_tokens: right_words,
        code_like,
        prose_like: !code_like,
        list_like,
        sentence_start,
        app_id: None,
    }
}

/// Build features for a synthetic "keep_original" candidate in softmax formulation.
/// Has context + ASR features but no candidate-specific phonetic scores.
pub fn build_keep_original_features(span: &TranscriptSpan, ctx: &SpanContext) -> Vec<Feature> {
    let span_token_count = (span.token_end - span.token_start) as f64;
    let span_phone_count = span.ipa_tokens.len() as f64;
    let span_low_content = is_low_content_span(&span.text) as u8 as f64;
    let mean_lp = span.mean_logprob.unwrap_or(0.0) as f64 / 5.0;
    let min_lp = span.min_logprob.unwrap_or(0.0) as f64 / 5.0;
    let mean_m = span.mean_margin.unwrap_or(0.0) as f64 / 5.0;
    let min_m = span.min_margin.unwrap_or(0.0) as f64 / 5.0;

    let mut features = vec![
        Feature {
            index: 0,
            value: 1.0,
        }, // bias
        // indices 1-24: candidate-specific scores = 0 (no candidate)
        Feature {
            index: 25,
            value: span_token_count / 4.0,
        },
        Feature {
            index: 26,
            value: span_phone_count / 12.0,
        },
        Feature {
            index: 27,
            value: span_low_content,
        },
        Feature {
            index: 28,
            value: mean_lp,
        },
        Feature {
            index: 29,
            value: min_lp,
        },
        Feature {
            index: 30,
            value: mean_m,
        },
        Feature {
            index: 31,
            value: min_m,
        },
        // indices 32-37: memory = 0 (no candidate)
    ];

    // Context features (same as any candidate, but no TERM= features)
    if let Some(l1) = ctx.left_tokens.first() {
        features.push(Feature {
            index: hash_feature(&format!("L1={l1}")),
            value: 1.0,
        });
    }
    if ctx.left_tokens.len() >= 2 {
        let bigram = format!("{}_{}", ctx.left_tokens[0], ctx.left_tokens[1]);
        features.push(Feature {
            index: hash_feature(&format!("L2={bigram}")),
            value: 1.0,
        });
    }
    if let Some(r1) = ctx.right_tokens.first() {
        features.push(Feature {
            index: hash_feature(&format!("R1={r1}")),
            value: 1.0,
        });
    }
    if ctx.right_tokens.len() >= 2 {
        let bigram = format!("{}_{}", ctx.right_tokens[0], ctx.right_tokens[1]);
        features.push(Feature {
            index: hash_feature(&format!("R2={bigram}")),
            value: 1.0,
        });
    }
    if ctx.code_like {
        features.push(Feature {
            index: hash_feature("CTX=code"),
            value: 1.0,
        });
    }
    if ctx.prose_like {
        features.push(Feature {
            index: hash_feature("CTX=prose"),
            value: 1.0,
        });
    }
    if ctx.list_like {
        features.push(Feature {
            index: hash_feature("CTX=list"),
            value: 1.0,
        });
    }
    if ctx.sentence_start {
        features.push(Feature {
            index: hash_feature("CTX=sent_start"),
            value: 1.0,
        });
    }
    if let Some(app) = &ctx.app_id {
        features.push(Feature {
            index: hash_feature(&format!("APP={app}")),
            value: 1.0,
        });
    }

    features
}

// ── Two-stage feature builders ─────────────────────────────────────

/// Dense feature layout for the span gate model (Stage A).
pub const GATE_FEATURE_NAMES: &[&str] = &[
    "bias",                 // 0
    "span_token_count",     // 1
    "span_phone_count",     // 2
    "span_low_content",     // 3
    "span_mean_logprob",    // 4
    "span_min_logprob",     // 5
    "span_mean_margin",     // 6
    "span_min_margin",      // 7
    "span_correct_count",   // 8
    "max_acceptance_score", // 9
    "max_phonetic_score",   // 10
    "any_verified",         // 11
    "any_acceptance_floor", // 12
    "candidate_count",      // 13
];

pub const NUM_GATE_DENSE: usize = 14;

/// Dense feature layout for the candidate ranker model (Stage B).
/// Indices 0-24 are candidate-specific, 25-30 are candidate-relative.
pub const RANKER_FEATURE_NAMES: &[&str] = &[
    "bias",                     // 0
    "acceptance_score",         // 1
    "phonetic_score",           // 2
    "coarse_score",             // 3
    "token_score",              // 4
    "feature_score",            // 5
    "feature_bonus",            // 6
    "best_view_score",          // 7
    "cross_view_support",       // 8
    "qgram_overlap",            // 9
    "total_qgram_overlap",      // 10
    "token_count_match",        // 11
    "phone_closeness",          // 12
    "alias_source_spoken",      // 13
    "alias_source_identifier",  // 14
    "alias_source_confusion",   // 15
    "identifier_acronym",       // 16
    "identifier_digits",        // 17
    "identifier_snake",         // 18
    "identifier_camel",         // 19
    "identifier_symbol",        // 20
    "short_guard_passed",       // 21
    "low_content_guard_passed", // 22
    "acceptance_floor_passed",  // 23
    "verified",                 // 24
    // Candidate-relative features
    "rank_in_span",          // 25
    "margin_to_next",        // 26
    "is_best_verified",      // 27
    "is_only_verified",      // 28
    "normalized_acceptance", // 29
    "normalized_phonetic",   // 30
    // Per-candidate memory
    "prior_accept_count",  // 31
    "prior_reject_count",  // 32
    "total_count",         // 33
    "recent_accept_count", // 34
    "session_recency",     // 35
];

pub const NUM_RANKER_DENSE: usize = 36;

/// Build span-level features for the gate model (Stage A).
/// One feature vector per span — does NOT depend on individual candidates.
pub fn build_gate_features(
    span: &TranscriptSpan,
    candidates: &[(CandidateFeatureRow, IdentifierFlags)],
    ctx: &SpanContext,
    memory: &TermMemory,
) -> Vec<Feature> {
    let span_token_count = (span.token_end - span.token_start) as f64;
    let span_phone_count = span.ipa_tokens.len() as f64;
    let span_low_content = is_low_content_span(&span.text) as u8 as f64;
    let mean_lp = span.mean_logprob.unwrap_or(0.0) as f64 / 5.0;
    let min_lp = span.min_logprob.unwrap_or(0.0) as f64 / 5.0;
    let mean_m = span.mean_margin.unwrap_or(0.0) as f64 / 5.0;
    let min_m = span.min_margin.unwrap_or(0.0) as f64 / 5.0;
    let span_correct_count = memory.span_text_count(&span.text);

    // Summary stats from candidates
    let max_acceptance = candidates
        .iter()
        .map(|(c, _)| c.acceptance_score)
        .fold(0.0f32, f32::max);
    let max_phonetic = candidates
        .iter()
        .map(|(c, _)| c.phonetic_score)
        .fold(0.0f32, f32::max);
    let any_verified = candidates.iter().any(|(c, _)| c.verified) as u8 as f64;
    let any_acceptance_floor =
        candidates.iter().any(|(c, _)| c.acceptance_floor_passed) as u8 as f64;
    let candidate_count = candidates.len() as f64;

    tracing::debug!(
        "gate_features: span={:?} tokens={:.2} phones={:.2} low_content={} mean_lp={:.3} min_lp={:.3} mean_m={:.3} min_m={:.3} correct_count={} max_accept={:.3} max_phonetic={:.3} verified={} accept_floor={} candidates={}",
        span.text,
        span_token_count / 4.0,
        span_phone_count / 12.0,
        span_low_content,
        mean_lp,
        min_lp,
        mean_m,
        min_m,
        span_correct_count,
        max_acceptance,
        max_phonetic,
        any_verified,
        any_acceptance_floor,
        candidate_count,
    );

    let mut features = vec![
        Feature {
            index: 0,
            value: 1.0,
        }, // bias
        Feature {
            index: 1,
            value: span_token_count / 4.0,
        }, // span_token_count
        Feature {
            index: 2,
            value: span_phone_count / 12.0,
        }, // span_phone_count
        Feature {
            index: 3,
            value: span_low_content,
        }, // span_low_content
        Feature {
            index: 4,
            value: mean_lp,
        }, // span_mean_logprob
        Feature {
            index: 5,
            value: min_lp,
        }, // span_min_logprob
        Feature {
            index: 6,
            value: mean_m,
        }, // span_mean_margin
        Feature {
            index: 7,
            value: min_m,
        }, // span_min_margin
        Feature {
            index: 8,
            value: (1.0 + span_correct_count as f64).ln() / 3.0,
        }, // span_correct_count
        Feature {
            index: 9,
            value: max_acceptance as f64,
        }, // max_acceptance_score
        Feature {
            index: 10,
            value: max_phonetic as f64,
        }, // max_phonetic_score
        Feature {
            index: 11,
            value: any_verified,
        }, // any_verified
        Feature {
            index: 12,
            value: any_acceptance_floor,
        }, // any_acceptance_floor
        Feature {
            index: 13,
            value: (1.0 + candidate_count).ln() / 3.0,
        }, // candidate_count
    ];

    // Sparse context features (no TERM= since this is span-level)
    if let Some(l1) = ctx.left_tokens.first() {
        features.push(Feature {
            index: hash_feature(&format!("L1={l1}")),
            value: 1.0,
        });
    }
    if ctx.left_tokens.len() >= 2 {
        let bigram = format!("{}_{}", ctx.left_tokens[0], ctx.left_tokens[1]);
        features.push(Feature {
            index: hash_feature(&format!("L2={bigram}")),
            value: 1.0,
        });
    }
    if let Some(r1) = ctx.right_tokens.first() {
        features.push(Feature {
            index: hash_feature(&format!("R1={r1}")),
            value: 1.0,
        });
    }
    if ctx.right_tokens.len() >= 2 {
        let bigram = format!("{}_{}", ctx.right_tokens[0], ctx.right_tokens[1]);
        features.push(Feature {
            index: hash_feature(&format!("R2={bigram}")),
            value: 1.0,
        });
    }
    if ctx.code_like {
        features.push(Feature {
            index: hash_feature("CTX=code"),
            value: 1.0,
        });
    }
    if ctx.prose_like {
        features.push(Feature {
            index: hash_feature("CTX=prose"),
            value: 1.0,
        });
    }
    if ctx.list_like {
        features.push(Feature {
            index: hash_feature("CTX=list"),
            value: 1.0,
        });
    }
    if ctx.sentence_start {
        features.push(Feature {
            index: hash_feature("CTX=sent_start"),
            value: 1.0,
        });
    }
    if let Some(app) = &ctx.app_id {
        features.push(Feature {
            index: hash_feature(&format!("APP={app}")),
            value: 1.0,
        });
    }

    features
}

/// Build candidate-level features for the ranker model (Stage B).
/// Each candidate gets candidate-specific + candidate-relative features.
/// NO span-level ASR or context features (those belong to Stage A).
pub fn build_ranker_features(
    _span: &TranscriptSpan,
    candidates: &[(CandidateFeatureRow, IdentifierFlags)],
    memory: &TermMemory,
) -> Vec<JudgeExample> {
    if candidates.is_empty() {
        return Vec::new();
    }

    // Pre-compute candidate-relative stats
    let max_acceptance = candidates
        .iter()
        .map(|(c, _)| c.acceptance_score)
        .fold(0.0f32, f32::max);
    let max_phonetic = candidates
        .iter()
        .map(|(c, _)| c.phonetic_score)
        .fold(0.0f32, f32::max);
    let verified_count: usize = candidates.iter().filter(|(c, _)| c.verified).count();

    // Sort indices by acceptance_score desc for rank computation
    let mut sorted_indices: Vec<usize> = (0..candidates.len()).collect();
    sorted_indices.sort_by(|&a, &b| {
        candidates[b]
            .0
            .acceptance_score
            .total_cmp(&candidates[a].0.acceptance_score)
    });
    let mut ranks = vec![0usize; candidates.len()];
    for (rank, &idx) in sorted_indices.iter().enumerate() {
        ranks[idx] = rank;
    }

    // Find best verified candidate (by acceptance_score)
    let best_verified_id = candidates
        .iter()
        .filter(|(c, _)| c.verified)
        .max_by(|a, b| a.0.acceptance_score.total_cmp(&b.0.acceptance_score))
        .map(|(c, _)| c.alias_id);

    // Sorted acceptance scores for margin computation
    let sorted_acceptance: Vec<f32> = sorted_indices
        .iter()
        .map(|&i| candidates[i].0.acceptance_score)
        .collect();

    candidates
        .iter()
        .enumerate()
        .map(|(i, (candidate, flags))| {
            let rank = ranks[i];
            let rank_feature = 1.0 / (1.0 + rank as f64);

            // Margin to next candidate
            let margin = if rank == 0 && sorted_acceptance.len() > 1 {
                (sorted_acceptance[0] - sorted_acceptance[1]) as f64
            } else if rank > 0 {
                (sorted_acceptance[rank - 1] - sorted_acceptance[rank]) as f64
            } else {
                0.0
            };

            let is_best_verified = best_verified_id == Some(candidate.alias_id);
            let is_only_verified = candidate.verified && verified_count == 1;
            let norm_acceptance = if max_acceptance > 0.0 {
                candidate.acceptance_score as f64 / max_acceptance as f64
            } else {
                0.0
            };
            let norm_phonetic = if max_phonetic > 0.0 {
                candidate.phonetic_score as f64 / max_phonetic as f64
            } else {
                0.0
            };

            // Per-candidate memory
            let mem = memory.get(&candidate.term);
            let accept_count = mem.map(|m| m.accept_count).unwrap_or(0);
            let reject_count = mem.map(|m| m.reject_count).unwrap_or(0);
            let recent_accept = mem.map(|m| m.recent_accept_count).unwrap_or(0);
            let recency = mem
                .and_then(|m| m.last_used)
                .and_then(|t| t.elapsed().ok())
                .map(|d| {
                    if d.as_secs() < 300 {
                        1.0
                    } else if d.as_secs() < 1800 {
                        0.5
                    } else {
                        0.0
                    }
                })
                .unwrap_or(0.0);

            let features = vec![
                Feature {
                    index: 0,
                    value: 1.0,
                }, // bias
                Feature {
                    index: 1,
                    value: candidate.acceptance_score as f64,
                }, // acceptance_score
                Feature {
                    index: 2,
                    value: candidate.phonetic_score as f64,
                }, // phonetic_score
                Feature {
                    index: 3,
                    value: candidate.coarse_score as f64,
                }, // coarse_score
                Feature {
                    index: 4,
                    value: candidate.token_score as f64,
                }, // token_score
                Feature {
                    index: 5,
                    value: candidate.feature_score as f64,
                }, // feature_score
                Feature {
                    index: 6,
                    value: candidate.feature_bonus as f64,
                }, // feature_bonus
                Feature {
                    index: 7,
                    value: candidate.best_view_score as f64,
                }, // best_view_score
                Feature {
                    index: 8,
                    value: candidate.cross_view_support as f64 / 6.0,
                }, // cross_view_support
                Feature {
                    index: 9,
                    value: candidate.qgram_overlap as f64 / 10.0,
                }, // qgram_overlap
                Feature {
                    index: 10,
                    value: candidate.total_qgram_overlap as f64 / 20.0,
                }, // total_qgram_overlap
                Feature {
                    index: 11,
                    value: candidate.token_count_match as u8 as f64,
                }, // token_count_match
                Feature {
                    index: 12,
                    value: 1.0 / (1.0 + candidate.phone_count_delta.abs() as f64),
                }, // phone_closeness
                Feature {
                    index: 13,
                    value: (candidate.alias_source == AliasSource::Spoken) as u8 as f64,
                },
                Feature {
                    index: 14,
                    value: (candidate.alias_source == AliasSource::Identifier) as u8 as f64,
                },
                Feature {
                    index: 15,
                    value: (candidate.alias_source == AliasSource::Confusion) as u8 as f64,
                },
                Feature {
                    index: 16,
                    value: flags.acronym_like as u8 as f64,
                },
                Feature {
                    index: 17,
                    value: flags.has_digits as u8 as f64,
                },
                Feature {
                    index: 18,
                    value: flags.snake_like as u8 as f64,
                },
                Feature {
                    index: 19,
                    value: flags.camel_like as u8 as f64,
                },
                Feature {
                    index: 20,
                    value: flags.symbol_like as u8 as f64,
                },
                Feature {
                    index: 21,
                    value: candidate.short_guard_passed as u8 as f64,
                },
                Feature {
                    index: 22,
                    value: candidate.low_content_guard_passed as u8 as f64,
                },
                Feature {
                    index: 23,
                    value: candidate.acceptance_floor_passed as u8 as f64,
                },
                Feature {
                    index: 24,
                    value: candidate.verified as u8 as f64,
                },
                // Candidate-relative features
                Feature {
                    index: 25,
                    value: rank_feature,
                }, // rank_in_span
                Feature {
                    index: 26,
                    value: margin,
                }, // margin_to_next
                Feature {
                    index: 27,
                    value: is_best_verified as u8 as f64,
                }, // is_best_verified
                Feature {
                    index: 28,
                    value: is_only_verified as u8 as f64,
                }, // is_only_verified
                Feature {
                    index: 29,
                    value: norm_acceptance,
                }, // normalized_acceptance
                Feature {
                    index: 30,
                    value: norm_phonetic,
                }, // normalized_phonetic
                // Per-candidate memory
                Feature {
                    index: 31,
                    value: (1.0 + accept_count as f64).ln() / 3.0,
                },
                Feature {
                    index: 32,
                    value: (1.0 + reject_count as f64).ln() / 3.0,
                },
                Feature {
                    index: 33,
                    value: (1.0 + (accept_count + reject_count) as f64).ln() / 3.0,
                },
                Feature {
                    index: 34,
                    value: (1.0 + recent_accept as f64).ln() / 3.0,
                },
                Feature {
                    index: 35,
                    value: recency,
                },
            ];

            JudgeExample {
                alias_id: candidate.alias_id,
                term: candidate.term.clone(),
                features,
            }
        })
        .collect()
}

/// Seed the span gate model with reasonable initial weights.
pub fn seed_gate_model(model: &mut SparseFtrl) {
    // Gate should open when: ASR is uncertain (low logprob, low margin)
    // AND candidates look promising (high acceptance, verified present).
    //
    // Synthetic examples: spans that should be corrected vs. should not.
    // We encode them as (gate_features_dense_values, label).
    //
    // Format: [bias, span_tok, span_phone, low_content, mean_lp, min_lp, mean_m, min_m,
    //          span_correct, max_accept, max_phonetic, any_verified, any_accept_floor, cand_count]

    let examples: &[(&[f64], bool)] = &[
        // Positive: good candidate + uncertain ASR
        (
            &[
                1.0, 0.5, 0.5, 0.0, -0.10, -0.15, 0.05, 0.02, 0.0, 0.70, 0.65, 1.0, 1.0, 0.5,
            ],
            true,
        ),
        (
            &[
                1.0, 0.3, 0.4, 0.0, -0.08, -0.12, 0.08, 0.04, 0.0, 0.55, 0.50, 1.0, 0.0, 0.3,
            ],
            true,
        ),
        (
            &[
                1.0, 0.5, 0.6, 0.0, -0.05, -0.10, 0.10, 0.05, 0.0, 0.80, 0.75, 1.0, 1.0, 0.7,
            ],
            true,
        ),
        // Positive: moderate candidate, low ASR confidence
        (
            &[
                1.0, 0.4, 0.5, 0.0, -0.15, -0.20, 0.03, 0.01, 0.0, 0.50, 0.45, 0.0, 0.0, 0.3,
            ],
            true,
        ),
        // Negative: no good candidates
        (
            &[
                1.0, 0.5, 0.5, 0.0, -0.10, -0.15, 0.05, 0.02, 0.0, 0.20, 0.15, 0.0, 0.0, 0.5,
            ],
            false,
        ),
        (
            &[
                1.0, 0.3, 0.4, 0.0, -0.05, -0.08, 0.12, 0.08, 0.0, 0.15, 0.10, 0.0, 0.0, 0.2,
            ],
            false,
        ),
        // Negative: high ASR confidence (probably transcribed correctly)
        (
            &[
                1.0, 0.5, 0.5, 0.0, -0.02, -0.03, 0.30, 0.25, 0.0, 0.60, 0.55, 1.0, 0.0, 0.5,
            ],
            false,
        ),
        // Negative: low content span
        (
            &[
                1.0, 0.3, 0.3, 1.0, -0.10, -0.15, 0.05, 0.02, 0.0, 0.40, 0.35, 0.0, 0.0, 0.3,
            ],
            false,
        ),
        // Negative: span already corrected before
        (
            &[
                1.0, 0.5, 0.5, 0.0, -0.10, -0.15, 0.05, 0.02, 0.8, 0.50, 0.45, 0.0, 0.0, 0.3,
            ],
            false,
        ),
    ];

    let featurize = |vals: &[f64]| -> Vec<Feature> {
        vals.iter()
            .enumerate()
            .map(|(i, &v)| Feature {
                index: i as u64,
                value: v,
            })
            .collect()
    };

    let orig_alpha = model.alpha;
    model.alpha = 1.0;
    for epoch in 0..20 {
        if epoch == 5 {
            model.alpha = 0.5;
        }
        for &(vals, label) in examples {
            model.update(&featurize(vals), label);
        }
    }
    model.alpha = orig_alpha;
}

/// Seed the candidate ranker model with initial weights.
pub fn seed_ranker_model(model: &mut SparseFtrl) {
    // Ranker should prefer: high acceptance, verified, good rank, good margin.
    // Seed with the same phonetic score structure as the main model,
    // plus positive weight on candidate-relative features.
    let examples: &[(&[f64], bool)] = &[
        // Positive: strong verified candidate, rank 1
        //          [bias, accept, phon, coarse, token, feat, fb, bv, cvs, qg, tqg, tcm, pc,
        //           sp, id, cf, acr, dig, snk, cam, sym, sg, lcg, afp, ver,
        //           rank, margin, bv, ov, na, np, ac, rc, tc, rac, rec]
        (
            &[
                1.0, 0.75, 0.70, 0.60, 0.55, 0.50, 0.30, 0.30, 0.50, 0.40, 0.35, 1.0, 0.80, 1.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.15, 1.0, 1.0, 1.0,
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            true,
        ),
        // Positive: good candidate, rank 1, not only verified
        (
            &[
                1.0, 0.60, 0.55, 0.45, 0.40, 0.40, 0.20, 0.20, 0.33, 0.30, 0.25, 1.0, 0.70, 0.0,
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.10, 1.0, 0.0, 1.0,
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            true,
        ),
        // Negative: weak candidate, low rank
        (
            &[
                1.0, 0.25, 0.20, 0.15, 0.10, 0.15, 0.05, 0.10, 0.17, 0.10, 0.05, 0.0, 0.40, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.25, 0.02, 0.0, 0.0, 0.33,
                0.40, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            false,
        ),
        // Negative: unverified, low scores
        (
            &[
                1.0, 0.15, 0.10, 0.08, 0.05, 0.08, 0.02, 0.05, 0.17, 0.05, 0.02, 0.0, 0.30, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.50, 0.01, 0.0, 0.0, 0.20,
                0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            false,
        ),
        // Negative: decent scores but not verified, rank 2
        (
            &[
                1.0, 0.50, 0.45, 0.35, 0.30, 0.35, 0.15, 0.15, 0.33, 0.25, 0.20, 1.0, 0.60, 1.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.50, 0.05, 0.0, 0.0, 0.67,
                0.75, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            false,
        ),
    ];

    let featurize = |vals: &[f64]| -> Vec<Feature> {
        vals.iter()
            .enumerate()
            .map(|(i, &v)| Feature {
                index: i as u64,
                value: v,
            })
            .collect()
    };

    let orig_alpha = model.alpha;
    model.alpha = 1.0;
    for epoch in 0..20 {
        if epoch == 5 {
            model.alpha = 0.5;
        }
        for &(vals, label) in examples {
            model.update(&featurize(vals), label);
        }
    }
    model.alpha = orig_alpha;
}

/// Feature slice definitions for ablation.
#[derive(Clone, Copy, Debug)]
pub enum FeatureSlice {
    /// Indices 0-27: phonetic/structural only
    PhoneticOnly,
    /// Indices 0-31: + ASR uncertainty
    PlusAsr,
    /// Indices 0-31 + sparse context features
    PlusContext,
    /// All features (0-37 + sparse)
    All,
}

impl FeatureSlice {
    pub fn name(&self) -> &'static str {
        match self {
            Self::PhoneticOnly => "phonetic_only",
            Self::PlusAsr => "+asr",
            Self::PlusContext => "+context",
            Self::All => "all",
        }
    }
}

/// Filter features to only include those allowed by the slice.
pub fn filter_features(features: &[Feature], slice: FeatureSlice) -> Vec<Feature> {
    features
        .iter()
        .copied()
        .filter(|f| match slice {
            FeatureSlice::PhoneticOnly => f.index < 28,
            FeatureSlice::PlusAsr => f.index < 32,
            FeatureSlice::PlusContext => f.index < 32 || f.index >= SPARSE_OFFSET,
            FeatureSlice::All => true,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use bee_phonetic::{AliasSource, IdentifierFlags, IndexView, TranscriptSpan};

    fn span(text: &str) -> TranscriptSpan {
        TranscriptSpan {
            token_start: 0,
            token_end: 2,
            char_start: 0,
            char_end: text.len(),
            start_sec: None,
            end_sec: None,
            text: text.to_string(),
            ipa_tokens: vec!["ɹ".into(), "ɛ".into(), "k".into()],
            reduced_ipa_tokens: vec!["r".into(), "e".into(), "k".into()],
            ..Default::default()
        }
    }

    fn ctx() -> SpanContext {
        SpanContext::default()
    }

    fn candidate(
        alias_id: u32,
        term: &str,
        acceptance: f32,
        phonetic: f32,
        coarse: f32,
        verified: bool,
    ) -> CandidateFeatureRow {
        CandidateFeatureRow {
            alias_id,
            term: term.to_string(),
            alias_text: term.to_string(),
            alias_source: AliasSource::Canonical,
            matched_view: IndexView::RawIpa2,
            qgram_overlap: 4,
            total_qgram_overlap: 7,
            best_view_score: coarse,
            cross_view_support: 3,
            token_count_match: true,
            phone_count_delta: 0,
            token_bonus: 0.0,
            phone_bonus: 0.0,
            extra_length_penalty: 0.0,
            structure_bonus: 0.0,
            coarse_score: coarse,
            token_distance: 1,
            token_weighted_distance: 1.0,
            token_boundary_penalty: 0.0,
            token_max_len: 3,
            token_score: phonetic,
            token_ops: Vec::new(),
            feature_distance: 1.0,
            feature_weighted_distance: 1.0,
            feature_boundary_penalty: 0.0,
            feature_max_len: 3,
            feature_score: phonetic,
            feature_ops: Vec::new(),
            feature_bonus: 0.0,
            feature_gate_token_ok: true,
            feature_gate_coarse_ok: true,
            feature_gate_phone_ok: true,
            short_guard_applied: false,
            short_guard_onset_match: false,
            short_guard_passed: true,
            low_content_guard_applied: false,
            low_content_guard_passed: true,
            acceptance_floor_passed: true,
            used_feature_bonus: false,
            phonetic_score: phonetic,
            acceptance_score: acceptance,
            verified,
        }
    }

    #[test]
    fn score_candidates_includes_keep_original() {
        let judge = OnlineJudge::default();
        let options = judge.score_candidates(
            &span("req west"),
            &[(
                candidate(7, "reqwest", 0.85, 0.82, 0.78, true),
                IdentifierFlags::default(),
            )],
            &ctx(),
        );
        assert!(options.iter().any(|option| option.is_keep_original));
        assert!(options.iter().any(|option| option.alias_id == Some(7)));
    }

    #[test]
    fn teach_choice_promotes_selected_candidate() {
        let mut judge = OnlineJudge::default();
        let span = span("req west");
        let candidates = vec![
            (
                candidate(7, "reqwest", 0.75, 0.74, 0.70, true),
                IdentifierFlags::default(),
            ),
            (
                candidate(8, "request", 0.40, 0.38, 0.35, false),
                IdentifierFlags::default(),
            ),
        ];

        // Teach 5 times so the model has enough signal
        for _ in 0..5 {
            judge.teach_choice(&span, &candidates, Some(7), &ctx());
        }

        let after = judge.score_candidates(&span, &candidates, &ctx());
        let after_reqwest = after
            .iter()
            .find(|option| option.alias_id == Some(7))
            .expect("reqwest option should exist")
            .probability;

        assert_eq!(judge.update_count(), 5);
        // reqwest should still be the chosen candidate and above threshold
        let chosen = after
            .iter()
            .find(|o| o.chosen)
            .expect("should have a chosen option");
        assert_eq!(chosen.alias_id, Some(7), "reqwest should be chosen");
        assert!(
            after_reqwest > ACCEPT_THRESHOLD,
            "reqwest should exceed threshold: prob={after_reqwest}"
        );
        // reqwest should beat request
        let after_request = after
            .iter()
            .find(|o| o.alias_id == Some(8))
            .expect("request option should exist")
            .probability;
        assert!(
            after_reqwest > after_request,
            "reqwest ({after_reqwest}) should beat request ({after_request})"
        );
    }

    #[test]
    fn teach_keep_original_lowers_candidate_scores() {
        let mut judge = OnlineJudge::default();
        let span = span("req west");
        let candidates = vec![(
            candidate(7, "reqwest", 0.75, 0.74, 0.70, true),
            IdentifierFlags::default(),
        )];

        let before = judge.score_candidates(&span, &candidates, &ctx());
        let before_prob = before
            .iter()
            .find(|o| o.alias_id == Some(7))
            .unwrap()
            .probability;

        // Teach keep_original (chosen_alias_id = None) 5 times
        for _ in 0..5 {
            judge.teach_choice(&span, &candidates, None, &ctx());
        }

        let after = judge.score_candidates(&span, &candidates, &ctx());
        let after_prob = after
            .iter()
            .find(|o| o.alias_id == Some(7))
            .unwrap()
            .probability;

        assert!(
            after_prob < before_prob,
            "teaching keep_original should lower candidate probability: before={before_prob} after={after_prob}"
        );
    }

    #[test]
    fn teaching_transfers_to_similar_candidates() {
        let mut judge = OnlineJudge::default();
        let span1 = span("sir day");
        let candidates1 = vec![(
            candidate(1, "serde", 0.80, 0.78, 0.75, true),
            IdentifierFlags::default(),
        )];

        // Teach: "serde" is correct for "sir day"
        for _ in 0..10 {
            judge.teach_choice(&span1, &candidates1, Some(1), &ctx());
        }

        // Now score a DIFFERENT case with similar features
        let span2 = span("toe key oh");
        let candidates2 = vec![(
            candidate(2, "tokio", 0.78, 0.76, 0.72, true),
            IdentifierFlags::default(),
        )];

        let result = judge.score_candidates(&span2, &candidates2, &ctx());
        let tokio_prob = result
            .iter()
            .find(|o| o.alias_id == Some(2))
            .unwrap()
            .probability;

        // The model should have learned that high acceptance/phonetic/coarse => accept
        // so tokio (with similar scores) should also get a high probability
        assert!(
            tokio_prob > 0.5,
            "learning should transfer: tokio_prob={tokio_prob}"
        );
    }

    #[test]
    fn teach_choice_actually_changes_weights() {
        let mut judge = OnlineJudge::default();
        let weights_before = judge.weights();
        let span = span("req west");
        let candidates = vec![
            (
                candidate(7, "reqwest", 0.75, 0.74, 0.70, true),
                IdentifierFlags::default(),
            ),
            (
                candidate(8, "request", 0.40, 0.38, 0.35, false),
                IdentifierFlags::default(),
            ),
        ];

        for _ in 0..10 {
            judge.teach_choice(&span, &candidates, Some(7), &ctx());
        }

        let weights_after = judge.weights();
        assert_eq!(judge.update_count(), 10);

        let max_diff = weights_before
            .iter()
            .zip(&weights_after)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff > 0.01,
            "weights should change meaningfully after 10 teaches, max_diff={max_diff}"
        );
    }

    #[test]
    fn context_features_are_generated() {
        let ctx = SpanContext {
            left_tokens: vec!["the".into()],
            right_tokens: vec!["crate".into()],
            code_like: false,
            prose_like: true,
            list_like: false,
            sentence_start: false,
            app_id: None,
        };
        let features = context_features(&ctx, "serde");
        // Should have: L1=the, TERM=serde|L1=the, R1=crate, TERM=serde|R1=crate,
        //              TERM=serde, CTX=prose = 6 features
        assert!(
            features.len() >= 5,
            "should generate context features, got {}: {features:?}",
            features.len()
        );
        // All should be in the sparse range
        assert!(
            features.iter().all(|f| f.index >= SPARSE_OFFSET),
            "context features should be in sparse range"
        );
    }

    #[test]
    fn extract_context_from_transcript() {
        let ctx = extract_span_context("I'm talking about the serde crate today", 22, 27);
        assert_eq!(ctx.left_tokens, vec!["about", "the"]);
        assert_eq!(ctx.right_tokens, vec!["crate", "today"]);
        assert!(!ctx.sentence_start);
    }

    #[test]
    fn trained_gate_weights_survive_save_load_roundtrip() {
        // Simulate the full production flow:
        //   1. seed gate model
        //   2. train on data
        //   3. save weights
        //   4. seed gate model again (fresh TwoStageJudge)
        //   5. load weights
        //   6. predictions must match step 2, not step 4
        use crate::sparse_ftrl::{Feature, SparseFtrl};

        let featurize = |vals: &[f64]| -> Vec<Feature> {
            vals.iter()
                .enumerate()
                .map(|(i, &v)| Feature {
                    index: i as u64,
                    value: v,
                })
                .collect()
        };

        // Step 1+2: seed then train
        let mut trained = SparseFtrl::new(0.5, 1.0, 0.0001, 0.001);
        seed_gate_model(&mut trained);
        // Simulate offline training with dense + sparse features
        for _ in 0..30 {
            // Positive: good candidate match
            trained.update(
                &{
                    let mut f = featurize(&[
                        1.0, 0.4, 0.5, 0.0, -0.12, -0.18, 0.04, 0.01, 0.0, 0.65, 0.60, 1.0, 1.0,
                        0.4,
                    ]);
                    f.push(Feature {
                        index: 1000,
                        value: 1.0,
                    }); // sparse context
                    f
                },
                true,
            );
            // Negative: no match
            trained.update(
                &{
                    let mut f = featurize(&[
                        1.0, 0.5, 0.5, 0.0, -0.03, -0.04, 0.25, 0.20, 0.0, 0.15, 0.10, 0.0, 0.0,
                        0.3,
                    ]);
                    f.push(Feature {
                        index: 2000,
                        value: 1.0,
                    });
                    f
                },
                false,
            );
        }

        // Collect trained predictions
        let test_cases = [
            featurize(&[
                1.0, 0.4, 0.5, 0.0, -0.10, -0.15, 0.05, 0.02, 0.0, 0.70, 0.65, 1.0, 1.0, 0.5,
            ]),
            featurize(&[
                1.0, 0.5, 0.5, 0.0, -0.02, -0.03, 0.30, 0.25, 0.0, 0.20, 0.15, 0.0, 0.0, 0.5,
            ]),
            featurize(&[
                1.0, 0.3, 0.4, 0.0, -0.08, -0.12, 0.08, 0.04, 0.0, 0.55, 0.50, 1.0, 0.0, 0.3,
            ]),
        ];
        let trained_probs: Vec<f64> = test_cases.iter().map(|f| trained.predict_prob(f)).collect();

        // Step 3: save
        let mut buf = Vec::new();
        trained.save_weights(&mut buf).unwrap();

        // Step 4: fresh seed (simulating TwoStageJudge::new)
        let mut loaded = SparseFtrl::new(0.5, 1.0, 0.0001, 0.001);
        seed_gate_model(&mut loaded);
        let seeded_probs: Vec<f64> = test_cases.iter().map(|f| loaded.predict_prob(f)).collect();

        // Step 5: load
        loaded.load_weights(&mut &buf[..]).unwrap();
        let loaded_probs: Vec<f64> = test_cases.iter().map(|f| loaded.predict_prob(f)).collect();

        // Step 6: verify
        for (i, ((trained_p, loaded_p), seeded_p)) in trained_probs
            .iter()
            .zip(&loaded_probs)
            .zip(&seeded_probs)
            .enumerate()
        {
            assert!(
                (trained_p - loaded_p).abs() < 1e-6,
                "case {i}: loaded prediction {loaded_p:.6} should match trained {trained_p:.6}, not seeded {seeded_p:.6}"
            );
        }

        // Also verify that loaded != seeded (the training actually changed something)
        let any_different = trained_probs
            .iter()
            .zip(&seeded_probs)
            .any(|(t, s)| (t - s).abs() > 0.01);
        assert!(
            any_different,
            "trained predictions should differ from seed-only predictions"
        );
    }

    #[test]
    fn trained_ranker_weights_survive_save_load_roundtrip() {
        use crate::sparse_ftrl::{Feature, SparseFtrl};

        let featurize = |vals: &[f64]| -> Vec<Feature> {
            vals.iter()
                .enumerate()
                .map(|(i, &v)| Feature {
                    index: i as u64,
                    value: v,
                })
                .collect()
        };

        // Seed + train ranker
        let mut trained = SparseFtrl::new(0.5, 1.0, 0.0001, 0.001);
        seed_ranker_model(&mut trained);
        for _ in 0..30 {
            // Gold candidate: strong scores
            trained.update(
                &featurize(&[
                    1.0, 0.75, 0.70, 0.60, 0.55, 0.50, 0.30, 0.30, 0.50, 0.40, 0.35, 1.0, 0.80,
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.15, 1.0,
                    1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ]),
                true,
            );
            // Wrong candidate: weak scores
            trained.update(
                &featurize(&[
                    1.0, 0.25, 0.20, 0.15, 0.10, 0.15, 0.05, 0.10, 0.17, 0.10, 0.05, 0.0, 0.40,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.25, 0.02, 0.0,
                    0.0, 0.33, 0.40, 0.0, 0.0, 0.0, 0.0, 0.0,
                ]),
                false,
            );
        }

        let test_cases = [
            featurize(&[
                1.0, 0.65, 0.60, 0.50, 0.45, 0.45, 0.25, 0.25, 0.33, 0.35, 0.30, 1.0, 0.75, 1.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.12, 1.0, 1.0, 1.0,
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ]),
            featurize(&[
                1.0, 0.30, 0.25, 0.20, 0.15, 0.20, 0.08, 0.12, 0.20, 0.12, 0.08, 0.0, 0.45, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.33, 0.03, 0.0, 0.0, 0.40,
                0.50, 0.0, 0.0, 0.0, 0.0, 0.0,
            ]),
        ];
        let trained_probs: Vec<f64> = test_cases.iter().map(|f| trained.predict_prob(f)).collect();

        // Save + load into fresh seeded model
        let mut buf = Vec::new();
        trained.save_weights(&mut buf).unwrap();

        let mut loaded = SparseFtrl::new(0.5, 1.0, 0.0001, 0.001);
        seed_ranker_model(&mut loaded);
        loaded.load_weights(&mut &buf[..]).unwrap();

        let loaded_probs: Vec<f64> = test_cases.iter().map(|f| loaded.predict_prob(f)).collect();

        for (i, (trained_p, loaded_p)) in trained_probs.iter().zip(&loaded_probs).enumerate() {
            assert!(
                (trained_p - loaded_p).abs() < 1e-6,
                "ranker case {i}: loaded={loaded_p:.6} should match trained={trained_p:.6}"
            );
        }
    }

    #[test]
    fn gate_with_loaded_weights_discriminates_positive_negative() {
        // The exported gate weights should produce high probs for positive
        // cases and low probs for negative cases — not near-zero for everything.
        use crate::sparse_ftrl::{Feature, SparseFtrl};

        let featurize = |vals: &[f64]| -> Vec<Feature> {
            vals.iter()
                .enumerate()
                .map(|(i, &v)| Feature {
                    index: i as u64,
                    value: v,
                })
                .collect()
        };

        let mut trained = SparseFtrl::new(0.5, 1.0, 0.0001, 0.001);
        seed_gate_model(&mut trained);
        for _ in 0..50 {
            trained.update(
                &featurize(&[
                    1.0, 0.4, 0.5, 0.0, -0.12, -0.18, 0.04, 0.01, 0.0, 0.65, 0.60, 1.0, 1.0, 0.4,
                ]),
                true,
            );
            trained.update(
                &featurize(&[
                    1.0, 0.5, 0.5, 0.0, -0.03, -0.04, 0.25, 0.20, 0.0, 0.15, 0.10, 0.0, 0.0, 0.3,
                ]),
                false,
            );
        }

        let mut buf = Vec::new();
        trained.save_weights(&mut buf).unwrap();

        let mut loaded = SparseFtrl::new(0.5, 1.0, 0.0001, 0.001);
        seed_gate_model(&mut loaded);
        loaded.load_weights(&mut &buf[..]).unwrap();

        // Positive-like input
        let pos_prob = loaded.predict_prob(&featurize(&[
            1.0, 0.4, 0.5, 0.0, -0.10, -0.15, 0.05, 0.02, 0.0, 0.70, 0.65, 1.0, 1.0, 0.5,
        ]));
        // Negative-like input
        let neg_prob = loaded.predict_prob(&featurize(&[
            1.0, 0.5, 0.5, 0.0, -0.02, -0.03, 0.30, 0.25, 0.0, 0.20, 0.15, 0.0, 0.0, 0.5,
        ]));

        assert!(
            pos_prob > 0.3,
            "positive case should have gate_prob > 0.3, got {pos_prob:.6}"
        );
        assert!(
            neg_prob < 0.3,
            "negative case should have gate_prob < 0.3, got {neg_prob:.6}"
        );
        assert!(
            pos_prob > neg_prob,
            "positive ({pos_prob:.6}) should score higher than negative ({neg_prob:.6})"
        );
    }
}
