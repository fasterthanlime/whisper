use bee_correct::g2p::CachedEspeakG2p;
use bee_correct::judge::{CorrectionEventSink, SpanDecision, TwoStageJudge};
use bee_phonetic::{
    enumerate_transcript_spans_with, feature_tokens_for_ipa, query_index, score_shortlist,
    PhoneticIndex, RetrievalQuery, SeedDataset,
};
use bee_types::{CorrectionEvent, SpanContext, TranscriptSpan};

/// Opaque correction engine handle.
pub struct CorrectionEngine {
    judge: TwoStageJudge,
    index: PhoneticIndex,
    g2p: CachedEspeakG2p,
    events_path: Option<PathBuf>,
}

/// Opaque correction result handle.
pub struct CorrectionResult {
    session_id: String,
    best_text: String,
    edits_json: CString,
    spans: Vec<SpanWithDecision>,
}

struct SpanWithDecision {
    span: TranscriptSpan,
    decision: SpanDecision,
}

struct FileEventSink {
    file: Option<std::fs::File>,
}

impl CorrectionEventSink for FileEventSink {
    fn log_event(&mut self, event: &CorrectionEvent) {
        use std::io::Write;
        if let Some(ref mut f) = self.file {
            if let Ok(json) = facet_json::to_string(event) {
                let _ = writeln!(f, "{json}");
            }
        }
    }
}

/// Load a correction engine. Returns NULL on error.
///
/// # Safety
/// All string pointers must be valid nul-terminated UTF-8 or NULL.
#[no_mangle]
pub unsafe extern "C" fn bee_correct_engine_load(
    dataset_dir: *const c_char,
    events_path: *const c_char,
    gate_threshold: c_float,
    ranker_threshold: c_float,
    out_err: *mut *mut c_char,
) -> *mut CorrectionEngine {
    let dataset_dir = match ptr_to_str(dataset_dir) {
        Some(s) => PathBuf::from(s),
        None => {
            set_err(out_err, "dataset_dir is null or invalid");
            return std::ptr::null_mut();
        }
    };

    let events_path = ptr_to_str(events_path).map(PathBuf::from);

    match load_correction_engine(&dataset_dir, events_path, gate_threshold, ranker_threshold) {
        Ok(engine) => Box::into_raw(Box::new(engine)),
        Err(e) => {
            set_err(out_err, &e);
            std::ptr::null_mut()
        }
    }
}

fn ptr_to_str(p: *const c_char) -> Option<&'static str> {
    if p.is_null() {
        return None;
    }
    unsafe { CStr::from_ptr(p) }.to_str().ok()
}

fn load_correction_engine(
    dataset_dir: &Path,
    events_path: Option<PathBuf>,
    gate_threshold: c_float,
    ranker_threshold: c_float,
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

/// Process a transcript and return correction results.
///
/// `transcript_json` is a JSON object: `{ "text": "...", "app_id": "..." }`
///
/// # Safety
/// `engine` must be a valid pointer. `transcript_json` must be valid UTF-8.
#[no_mangle]
pub unsafe extern "C" fn bee_correct_process(
    engine: *mut CorrectionEngine,
    transcript_json: *const c_char,
    out_err: *mut *mut c_char,
) -> *mut CorrectionResult {
    if engine.is_null() || transcript_json.is_null() {
        set_err(out_err, "null engine or transcript");
        return std::ptr::null_mut();
    }

    let engine = unsafe { &mut *engine };
    let json_str = match unsafe { CStr::from_ptr(transcript_json) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_err(out_err, &format!("invalid transcript JSON: {e}"));
            return std::ptr::null_mut();
        }
    };

    match process_transcript(engine, json_str) {
        Ok(result) => Box::into_raw(Box::new(result)),
        Err(e) => {
            set_err(out_err, &e);
            std::ptr::null_mut()
        }
    }
}

fn process_transcript(
    engine: &mut CorrectionEngine,
    json_str: &str,
) -> Result<CorrectionResult, String> {
    let input: serde_json::Value =
        serde_json::from_str(json_str).map_err(|e| format!("parse transcript JSON: {e}"))?;
    let text = input["text"].as_str().ok_or("missing 'text' field")?;
    let app_id = input["app_id"].as_str().map(String::from);

    let spans = enumerate_transcript_spans_with(
        text,
        3,
        None::<&[bee_types::TranscriptAlignmentToken]>,
        |span_text| engine.g2p.ipa_tokens(span_text).ok().flatten(),
    );

    let mut span_decisions = Vec::new();
    let mut edits = Vec::new();
    let session_id = format!(
        "{:x}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    );

    for span in &spans {
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

        let ctx = bee_correct::judge::extract_span_context(text, span.char_start, span.char_end);
        let ctx = SpanContext {
            app_id: app_id.clone(),
            ..ctx
        };

        let decision = engine.judge.score_span(span, &candidates_with_flags, &ctx);

        if let Some(ref chosen) = decision.chosen {
            edits.push(serde_json::json!({
                "edit_id": format!("e{}", edits.len()),
                "span_start": span.char_start,
                "span_end": span.char_end,
                "original": span.text,
                "replacement": chosen.replacement_text,
                "term": chosen.term,
                "alias_id": chosen.alias_id,
                "ranker_prob": chosen.ranker_prob,
                "gate_prob": decision.gate_prob,
            }));
        }

        span_decisions.push(SpanWithDecision {
            span: span.clone(),
            decision,
        });
    }

    // Build best text by applying edits (non-overlapping, sorted by position)
    let mut best_text = text.to_string();
    let mut offset: i64 = 0;
    for edit in &edits {
        let start = edit["span_start"].as_u64().unwrap() as usize;
        let end = edit["span_end"].as_u64().unwrap() as usize;
        let replacement = edit["replacement"].as_str().unwrap();
        let adj_start = (start as i64 + offset) as usize;
        let adj_end = (end as i64 + offset) as usize;
        best_text.replace_range(adj_start..adj_end, replacement);
        offset += replacement.len() as i64 - (end - start) as i64;
    }

    let edits_value = serde_json::json!({
        "session_id": session_id,
        "edits": edits,
    });
    let edits_json = CString::new(edits_value.to_string()).unwrap_or_default();

    Ok(CorrectionResult {
        session_id,
        best_text,
        edits_json,
        spans: span_decisions,
    })
}

/// Get the session ID from a correction result.
#[no_mangle]
pub extern "C" fn bee_correction_result_session_id(result: *const CorrectionResult) -> *mut c_char {
    if result.is_null() {
        return std::ptr::null_mut();
    }
    to_c_string(&unsafe { &*result }.session_id)
}

/// Get the corrected text.
#[no_mangle]
pub extern "C" fn bee_correction_result_best_text(result: *const CorrectionResult) -> *mut c_char {
    if result.is_null() {
        return std::ptr::null_mut();
    }
    to_c_string(&unsafe { &*result }.best_text)
}

/// Get the edits as JSON.
#[no_mangle]
pub extern "C" fn bee_correction_result_json(result: *const CorrectionResult) -> *const c_char {
    if result.is_null() {
        return std::ptr::null();
    }
    unsafe { &*result }.edits_json.as_ptr()
}

/// Get the number of edits.
#[no_mangle]
pub extern "C" fn bee_correction_result_edit_count(result: *const CorrectionResult) -> u32 {
    if result.is_null() {
        return 0;
    }
    unsafe { &*result }
        .spans
        .iter()
        .filter(|s| s.decision.chosen.is_some())
        .count() as u32
}

/// Free a correction result.
#[no_mangle]
pub extern "C" fn bee_correction_result_free(result: *mut CorrectionResult) {
    if !result.is_null() {
        unsafe { drop(Box::from_raw(result)) };
    }
}

/// Teach the judge from user feedback.
///
/// `teach_json`: `{ "edits": [{"edit_id": "e0", "resolution": "accepted"}], ... }`
///
/// # Safety
/// `engine` must be valid. `session_id` and `teach_json` must be valid UTF-8.
#[no_mangle]
pub unsafe extern "C" fn bee_correct_teach(
    engine: *mut CorrectionEngine,
    _session_id: *const c_char,
    _teach_json: *const c_char,
    _out_err: *mut *mut c_char,
) {
    if engine.is_null() {
        return;
    }
    // TODO: implement teaching from stable session IDs
    // For now this is a stub — teaching requires storing the
    // process result keyed by session_id, which needs more design.
}

/// Save judge weights and flush events.
///
/// # Safety
/// `engine` must be a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn bee_correct_save(
    _engine: *mut CorrectionEngine,
    _out_err: *mut *mut c_char,
) {
    // TODO: implement persistence (weights + memory + events)
}

/// Free a correction engine.
#[no_mangle]
pub extern "C" fn bee_correct_engine_free(engine: *mut CorrectionEngine) {
    if !engine.is_null() {
        unsafe { drop(Box::from_raw(engine)) };
    }
}
