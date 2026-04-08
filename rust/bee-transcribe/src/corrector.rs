//! Correction layer: runs the phonetic correction pipeline on aligned chunks.

use std::collections::HashMap;

use bee_phonetic::phonetic_verify::CandidateFeatureRow;
use bee_phonetic::{
    RetrievalQuery, enumerate_transcript_spans_with, feature_tokens_for_ipa, query_index,
    score_shortlist,
};
use bee_rpc::CorrectionEdit;
use bee_types::{IdentifierFlags, SpanContext};

use bee_types::AlignedWord;

use crate::correct::{CorrectionEngine, PendingEdit};

/// Runs phonetic correction on ASR-committed chunks.
///
/// Accumulates correction-committed text and pending edits for the teach flow.
pub struct Corrector {
    /// Correction-committed text accumulated across chunks.
    committed_text: String,
    /// All edits applied so far in this session.
    committed_edits: Vec<CorrectionEdit>,
    /// Session ID for teach/save.
    session_id: String,
    /// edit_id -> PendingEdit, for the teach flow.
    pending: HashMap<String, PendingEdit>,
    /// Running edit counter.
    edit_counter: usize,
}

impl Corrector {
    pub fn new() -> Self {
        let session_id = format!(
            "{:x}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        );
        Self {
            committed_text: String::new(),
            committed_edits: Vec::new(),
            session_id,
            pending: HashMap::new(),
            edit_counter: 0,
        }
    }

    /// Run correction pipeline on a chunk with explicit neighbor context.
    ///
    /// Only the chunk text/words are editable — left/right neighbors are
    /// context-only for `SpanContext` extraction. Spans never cross chunk
    /// boundaries.
    pub fn process_chunk_with_context(
        &mut self,
        engine: &mut CorrectionEngine,
        text: &str,
        words: &[AlignedWord],
        left_context: &[AlignedWord],
        right_context: &[AlignedWord],
        app_id: Option<&str>,
    ) {
        if text.trim().is_empty() {
            self.committed_text.push_str(text);
            return;
        }

        let app_id_opt = app_id.map(|s| s.to_string());

        for (i, w) in words.iter().enumerate() {
            tracing::debug!(
                "corrector: word[{i}]={:?} logprob=({:.3},{:.3}) margin=({:.3},{:.3})",
                w.word,
                w.confidence.mean_lp,
                w.confidence.min_lp,
                w.confidence.mean_m,
                w.confidence.min_m,
            );
        }

        let spans = enumerate_transcript_spans_with(text, 3, Some(words), |span_text| {
            engine.g2p.ipa_tokens(span_text).ok().flatten()
        });

        tracing::info!(
            "corrector: text={:?} spans={} left_ctx={} right_ctx={}",
            text.chars().take(60).collect::<String>(),
            spans.len(),
            left_context.len(),
            right_context.len(),
        );

        // Phase 1: score all spans, collect candidates
        struct ScoredEdit {
            span: bee_types::TranscriptSpan,
            candidates: Vec<(CandidateFeatureRow, IdentifierFlags)>,
            ctx: SpanContext,
            edit: CorrectionEdit,
            score: f64,
        }

        let mut candidates: Vec<ScoredEdit> = Vec::new();

        for span in &spans {
            tracing::debug!(
                "corrector: span={:?} mean_logprob={:?} min_logprob={:?} mean_margin={:?} min_margin={:?}",
                span.text,
                span.mean_logprob,
                span.min_logprob,
                span.mean_margin,
                span.min_margin,
            );
            let query = RetrievalQuery {
                text: span.text.clone(),
                ipa_tokens: span.ipa_tokens.clone(),
                reduced_ipa_tokens: span.reduced_ipa_tokens.clone(),
                feature_tokens: feature_tokens_for_ipa(&span.ipa_tokens),
                token_count: (span.token_end - span.token_start) as u8,
            };
            let shortlist = query_index(&engine.index, &query, 50);
            if shortlist.is_empty() {
                continue;
            }
            let scored = score_shortlist(span, &shortlist, &engine.index);
            if scored.is_empty() {
                continue;
            }

            let candidates_with_flags: Vec<_> = scored
                .iter()
                .map(|c| {
                    let flags = engine
                        .index
                        .aliases
                        .iter()
                        .find(|a| a.alias_id == c.alias_id)
                        .map(|a| a.identifier_flags.clone())
                        .unwrap_or_default();
                    (c.clone(), flags)
                })
                .collect();

            // Build context from neighboring words (not from string slicing).
            // Left context: last 2 words from left_context + words before span.
            // Right context: words after span + first 2 words from right_context.
            let ctx = extract_span_context_from_words(
                text,
                span.char_start,
                span.char_end,
                words,
                span.token_start,
                span.token_end,
                left_context,
                right_context,
                app_id_opt.clone(),
            );

            let decision = engine.judge.score_span(span, &candidates_with_flags, &ctx);

            tracing::debug!(
                "corrector: span={:?} gate={:.3} chosen={} top_candidate={:?}",
                span.text,
                decision.gate_prob,
                decision.chosen.is_some(),
                scored.first().map(|c| &c.term),
            );

            if let Some(ref chosen) = decision.chosen {
                let score = decision.gate_prob as f64 * chosen.ranker_prob as f64;
                candidates.push(ScoredEdit {
                    span: span.clone(),
                    candidates: candidates_with_flags,
                    ctx,
                    edit: CorrectionEdit {
                        edit_id: String::new(), // assigned after selection
                        span_start: span.char_start as u32,
                        span_end: span.char_end as u32,
                        original: span.text.clone(),
                        replacement: chosen.replacement_text.clone(),
                        term: chosen.term.clone(),
                        alias_id: chosen.alias_id as i32,
                        ranker_prob: chosen.ranker_prob as f64,
                        gate_prob: decision.gate_prob as f64,
                    },
                    score,
                });
            }
        }

        // Phase 2: greedy non-overlapping selection, best score first
        candidates.sort_by(|a, b| b.score.total_cmp(&a.score));

        let mut edits = Vec::new();
        let mut claimed: Vec<(usize, usize)> = Vec::new();

        for mut cand in candidates {
            let cs = cand.span.char_start;
            let ce = cand.span.char_end;
            if claimed.iter().any(|&(s, e)| cs < e && ce > s) {
                continue;
            }
            claimed.push((cs, ce));

            let edit_id = format!("e{}", self.edit_counter);
            self.edit_counter += 1;
            let alias_id = cand.edit.alias_id as u32;
            cand.edit.edit_id = edit_id.clone();
            edits.push(cand.edit);

            self.pending.insert(
                edit_id,
                PendingEdit {
                    span: cand.span,
                    candidates: cand.candidates,
                    ctx: cand.ctx,
                    chosen_alias_id: Some(alias_id),
                },
            );
        }

        // Sort edits by position for correct text composition
        edits.sort_by_key(|e| e.span_start);

        // Apply edits to chunk text
        let mut corrected = text.to_string();
        let mut offset: i64 = 0;
        for edit in &edits {
            let start = edit.span_start as usize;
            let end = edit.span_end as usize;
            let adj_start = (start as i64 + offset) as usize;
            let adj_end = (end as i64 + offset) as usize;
            corrected.replace_range(adj_start..adj_end, &edit.replacement);
            offset += edit.replacement.len() as i64 - (end - start) as i64;
        }

        self.committed_text.push_str(&corrected);
        self.committed_edits.extend(edits);
    }

    pub fn committed_text(&self) -> &str {
        &self.committed_text
    }

    pub fn committed_edits(&self) -> &[CorrectionEdit] {
        &self.committed_edits
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Take pending edits for the teach flow. Drains the map.
    pub fn take_pending(&mut self) -> HashMap<String, PendingEdit> {
        std::mem::take(&mut self.pending)
    }
}

/// Build `SpanContext` using word-based neighbor context instead of string slicing.
///
/// Left context: words before the span within the chunk + neighbor left_context words.
/// Right context: words after the span within the chunk + neighbor right_context words.
/// This avoids char-offset surgery across concatenated strings.
fn extract_span_context_from_words(
    text: &str,
    char_start: usize,
    _char_end: usize,
    chunk_words: &[AlignedWord],
    token_start: usize,
    token_end: usize,
    left_context: &[AlignedWord],
    right_context: &[AlignedWord],
    app_id: Option<String>,
) -> SpanContext {
    // Left tokens: words before span in chunk, then neighbor words, take last 2
    let mut left_tokens: Vec<String> = Vec::new();
    for w in left_context.iter() {
        left_tokens.push(w.word.trim().to_ascii_lowercase());
    }
    for w in chunk_words.iter().take(token_start) {
        left_tokens.push(w.word.trim().to_ascii_lowercase());
    }
    // Keep only last 2
    let skip = left_tokens.len().saturating_sub(2);
    let left_tokens: Vec<String> = left_tokens.into_iter().skip(skip).collect();

    // Right tokens: words after span in chunk, then neighbor words, take first 2
    let mut right_tokens: Vec<String> = Vec::new();
    for w in chunk_words.iter().skip(token_end) {
        right_tokens.push(w.word.trim().to_ascii_lowercase());
    }
    for w in right_context.iter() {
        right_tokens.push(w.word.trim().to_ascii_lowercase());
    }
    right_tokens.truncate(2);

    // Sentence start: span is at beginning or after sentence-ending punctuation
    let before = &text[..char_start];
    let sentence_start = before.is_empty() || before.trim_end().ends_with(['.', '!', '?']);

    SpanContext {
        left_tokens,
        right_tokens,
        code_like: false, // ASR speech is not code
        prose_like: true,
        list_like: false,
        sentence_start,
        app_id,
    }
}
