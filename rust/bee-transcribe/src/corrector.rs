//! Correction layer: runs the phonetic correction pipeline on aligned chunks.

use std::collections::HashMap;

use bee_phonetic::phonetic_verify::CandidateFeatureRow;
use bee_phonetic::{
    enumerate_transcript_spans_with, feature_tokens_for_ipa, query_index, score_shortlist,
    RetrievalQuery,
};
use bee_rpc::CorrectionEdit;
use bee_types::{IdentifierFlags, SpanContext};

use crate::aligner::AlignedChunk;
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

    /// Run correction pipeline on an aligned chunk.
    pub fn process_chunk(
        &mut self,
        engine: &mut CorrectionEngine,
        chunk: &AlignedChunk,
        app_id: Option<&str>,
    ) {
        let text = &chunk.text;
        if text.trim().is_empty() {
            self.committed_text.push_str(text);
            return;
        }

        let app_id_opt = app_id.map(|s| s.to_string());

        for (i, w) in chunk.words.iter().enumerate() {
            tracing::debug!(
                "corrector: word[{i}]={:?} logprob=({:.3},{:.3}) margin=({:.3},{:.3})",
                w.word,
                w.confidence.mean_lp,
                w.confidence.min_lp,
                w.confidence.mean_m,
                w.confidence.min_m,
            );
        }

        let spans =
            enumerate_transcript_spans_with(text, 3, Some(chunk.words.as_slice()), |span_text| {
                engine.g2p.ipa_tokens(span_text).ok().flatten()
            });

        tracing::info!(
            "corrector: text={:?} spans={}",
            text.chars().take(60).collect::<String>(),
            spans.len()
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

            let ctx =
                bee_correct::judge::extract_span_context(text, span.char_start, span.char_end);
            let ctx = SpanContext {
                app_id: app_id_opt.clone(),
                ..ctx
            };

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
        let mut corrected = text.clone();
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
