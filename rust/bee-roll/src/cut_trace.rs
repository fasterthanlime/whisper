//! Structured cut-decision trace export for bee-roll.
//!
//! Enable by setting `BEE_ROLL_CUT_TRACE=/abs/path/cuts.jsonl` before
//! running bee-roll. One JSON object per significant event is appended to
//! the file, one object per line (JSON Lines format).
//!
//! When the env var is not set, `CutTracer` is a zero-cost no-op.

use std::io::{BufWriter, Write as _};

use serde::Serialize;

/// One word span in the decoded transcript.
#[derive(Serialize)]
pub(crate) struct WordSpan {
    /// Start token index in the utterance-global tape (inclusive).
    pub start: usize,
    /// End token index in the utterance-global tape (exclusive).
    pub end: usize,
    pub text: String,
    /// ZIPA-aligned start time in seconds (None if no aligned token in span).
    pub start_secs: Option<f64>,
    /// ZIPA-aligned end time in seconds (None if no aligned token in span).
    pub end_secs: Option<f64>,
    /// Which partition this word falls in: "stable", "carry", or "preview".
    pub region: &'static str,
}

// ── Per-event payload structs ────────────────────────────────────────────────

#[derive(Serialize)]
struct FeedStartEvent {
    event: &'static str,
    feed_index: usize,
    audio_end_secs: f64,
}

#[derive(Serialize)]
struct PlanPreviewDecodeEvent {
    event: &'static str,
    feed_index: usize,
    stable_through: usize,
    preview_from: usize,
    tape_end: usize,
    retained_decoder_position: usize,
    current_kv_position: usize,
}

#[derive(Serialize)]
struct RewritePreviewEvent {
    event: &'static str,
    feed_index: usize,
    rollback_to: usize,
    decoder_position: usize,
    current_kv_position: usize,
}

#[derive(Serialize)]
struct UpdatePreviewFromEvent {
    event: &'static str,
    feed_index: usize,
    stable_through: usize,
    tape_end: usize,
    target: usize,
    preview_from: usize,
}

#[derive(Serialize)]
struct CutCandidateEvent {
    event: &'static str,
    feed_index: usize,
    stable_through: usize,
    preview_from: usize,
    latest_legal_boundary: usize,
    /// `None` when `find_auto_cut_boundary` returned `None` (no legal word boundary found).
    chosen_boundary: Option<usize>,
}

#[derive(Serialize)]
struct CutAppliedEvent {
    event: &'static str,
    feed_index: usize,
    /// Value of `stable_through` *before* this step.
    prev_stable_through: usize,
    prev_preview_from: usize,
    /// The boundary that `apply_cut_if_any` tried to promote.
    new_stable: usize,
    /// The audio sample at which the rotation was made, if any.
    cut_sample: Option<usize>,
    /// The audio cut point in seconds (cut_sample / SAMPLE_RATE), if any.
    cut_sample_secs: Option<f64>,
    /// Whether the cut was actually applied (false if no aligned audio sample was found).
    applied: bool,
    /// Compact debug context string (window of tokens around the boundary).
    context_debug: String,
}

#[derive(Serialize)]
struct FeedEndEvent<'a> {
    event: &'static str,
    feed_index: usize,
    stable_through: usize,
    preview_from: usize,
    tape_end: usize,
    transcript: &'a str,
    word_spans: Vec<WordSpan>,
    /// True when any 2-gram of words in the transcript appears more than once.
    has_repeated_ngrams: bool,
}

// ── CutTracer ────────────────────────────────────────────────────────────────

pub(crate) struct CutTracer {
    writer: Option<BufWriter<std::fs::File>>,
}

impl CutTracer {
    /// Reads `BEE_ROLL_CUT_TRACE` from the environment.
    ///
    /// If the variable is set, opens (or creates and truncates) the file and
    /// returns an active tracer. If not set, returns a silent no-op tracer.
    pub(crate) fn from_env() -> Self {
        let path = match std::env::var("BEE_ROLL_CUT_TRACE") {
            Ok(p) if !p.is_empty() => p,
            _ => return Self { writer: None },
        };
        match std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&path)
        {
            Ok(file) => {
                tracing::info!(path, "bee_roll.cut_trace: writing to file");
                Self {
                    writer: Some(BufWriter::new(file)),
                }
            }
            Err(e) => {
                tracing::warn!(path, error = %e, "bee_roll.cut_trace: failed to open trace file");
                Self { writer: None }
            }
        }
    }

    pub(crate) fn is_active(&self) -> bool {
        self.writer.is_some()
    }

    fn emit<T: Serialize>(&mut self, payload: &T) {
        let Some(w) = self.writer.as_mut() else {
            return;
        };
        match serde_json::to_string(payload) {
            Ok(line) => {
                let _ = w.write_all(line.as_bytes());
                let _ = w.write_all(b"\n");
                let _ = w.flush();
            }
            Err(e) => {
                tracing::warn!(error = %e, "bee_roll.cut_trace: serialization error");
            }
        }
    }

    // ── Emit helpers ─────────────────────────────────────────────────────────

    pub(crate) fn feed_start(&mut self, feed_index: usize, audio_end_secs: f64) {
        self.emit(&FeedStartEvent {
            event: "feed_start",
            feed_index,
            audio_end_secs,
        });
    }

    pub(crate) fn plan_preview_decode(
        &mut self,
        feed_index: usize,
        stable_through: usize,
        preview_from: usize,
        tape_end: usize,
        retained_decoder_position: usize,
        current_kv_position: usize,
    ) {
        self.emit(&PlanPreviewDecodeEvent {
            event: "plan_preview_decode",
            feed_index,
            stable_through,
            preview_from,
            tape_end,
            retained_decoder_position,
            current_kv_position,
        });
    }

    pub(crate) fn rewrite_preview(
        &mut self,
        feed_index: usize,
        rollback_to: usize,
        decoder_position: usize,
        current_kv_position: usize,
    ) {
        self.emit(&RewritePreviewEvent {
            event: "rewrite_preview",
            feed_index,
            rollback_to,
            decoder_position,
            current_kv_position,
        });
    }

    pub(crate) fn update_preview_from(
        &mut self,
        feed_index: usize,
        stable_through: usize,
        tape_end: usize,
        target: usize,
        preview_from: usize,
    ) {
        self.emit(&UpdatePreviewFromEvent {
            event: "update_preview_from",
            feed_index,
            stable_through,
            tape_end,
            target,
            preview_from,
        });
    }

    pub(crate) fn cut_candidate(
        &mut self,
        feed_index: usize,
        stable_through: usize,
        preview_from: usize,
        latest_legal_boundary: usize,
        chosen_boundary: Option<usize>,
    ) {
        self.emit(&CutCandidateEvent {
            event: "cut_candidate",
            feed_index,
            stable_through,
            preview_from,
            latest_legal_boundary,
            chosen_boundary,
        });
    }

    pub(crate) fn cut_applied(
        &mut self,
        feed_index: usize,
        prev_stable_through: usize,
        prev_preview_from: usize,
        new_stable: usize,
        cut_sample: Option<usize>,
        applied: bool,
        context_debug: String,
    ) {
        let cut_sample_secs = cut_sample.map(|s| s as f64 / crate::SAMPLE_RATE as f64);
        self.emit(&CutAppliedEvent {
            event: "cut_applied",
            feed_index,
            prev_stable_through,
            prev_preview_from,
            new_stable,
            cut_sample,
            cut_sample_secs,
            applied,
            context_debug,
        });
    }

    pub(crate) fn feed_end(
        &mut self,
        feed_index: usize,
        stable_through: usize,
        preview_from: usize,
        tape_end: usize,
        transcript: &str,
        word_spans: Vec<WordSpan>,
    ) {
        let has_repeated_ngrams = detect_repeated_bigrams(&word_spans);
        self.emit(&FeedEndEvent {
            event: "feed_end",
            feed_index,
            stable_through,
            preview_from,
            tape_end,
            transcript,
            word_spans,
            has_repeated_ngrams,
        });
    }
}

/// Returns true if any adjacent 2-gram of word texts appears more than once
/// in the span list. Simple O(n²) scan — spans are short.
fn detect_repeated_bigrams(spans: &[WordSpan]) -> bool {
    if spans.len() < 4 {
        return false;
    }
    for i in 0..spans.len().saturating_sub(1) {
        let a = (&spans[i].text, &spans[i + 1].text);
        for j in (i + 2)..spans.len().saturating_sub(1) {
            if (&spans[j].text, &spans[j + 1].text) == a {
                return true;
            }
        }
    }
    false
}
