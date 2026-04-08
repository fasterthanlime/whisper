use std::hash::{Hash, Hasher};
use std::path::PathBuf;

use anyhow::{Context, Result};
use beeml::rpc::{AliasSource, CandidateFeatureDebug, IdentifierFlags, RetrievalIndexView};

use crate::service::{CounterexampleRecordingRow, EvalCase};

pub(crate) fn randomish_case_key(case: &EvalCase, seed: u64) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    seed.hash(&mut hasher);
    case.case_id.hash(&mut hasher);
    case.target_term.hash(&mut hasher);
    case.transcript.hash(&mut hasher);
    hasher.finish()
}

pub(crate) fn load_correction_events(
    path: &std::path::Path,
) -> anyhow::Result<Vec<beeml::judge::CorrectionEvent>> {
    use std::io::BufRead;
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let mut events = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        match facet_json::from_str::<beeml::judge::CorrectionEvent>(&line) {
            Ok(event) => events.push(event),
            Err(e) => tracing::warn!(error = ?e, "skipping malformed event line"),
        }
    }
    // Cap at 10,000 events (keep most recent)
    if events.len() > 10_000 {
        events = events.split_off(events.len() - 10_000);
    }
    Ok(events)
}

pub(crate) fn save_correction_events(
    path: &std::path::Path,
    events: &[beeml::judge::CorrectionEvent],
) -> anyhow::Result<()> {
    use std::io::Write;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::File::create(path)?;
    // Cap at 10,000 events (keep most recent)
    let start = events.len().saturating_sub(10_000);
    for event in &events[start..] {
        let json = facet_json::to_string(event).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        writeln!(file, "{json}")?;
    }
    Ok(())
}

pub(crate) fn map_alias_source(source: bee_phonetic::AliasSource) -> AliasSource {
    match source {
        bee_phonetic::AliasSource::Canonical => AliasSource::Canonical,
        bee_phonetic::AliasSource::Spoken => AliasSource::Spoken,
        bee_phonetic::AliasSource::Identifier => AliasSource::Identifier,
        bee_phonetic::AliasSource::Confusion => AliasSource::Confusion,
    }
}

pub(crate) fn map_index_view(view: bee_phonetic::IndexView) -> RetrievalIndexView {
    match view {
        bee_phonetic::IndexView::RawIpa2 => RetrievalIndexView::RawIpa2,
        bee_phonetic::IndexView::RawIpa3 => RetrievalIndexView::RawIpa3,
        bee_phonetic::IndexView::ReducedIpa2 => RetrievalIndexView::ReducedIpa2,
        bee_phonetic::IndexView::ReducedIpa3 => RetrievalIndexView::ReducedIpa3,
        bee_phonetic::IndexView::Feature2 => RetrievalIndexView::Feature2,
        bee_phonetic::IndexView::Feature3 => RetrievalIndexView::Feature3,
        bee_phonetic::IndexView::ShortQueryFallback => RetrievalIndexView::ShortQueryFallback,
    }
}

pub(crate) fn map_candidate_features(
    candidate: &bee_phonetic::CandidateFeatureRow,
) -> CandidateFeatureDebug {
    CandidateFeatureDebug {
        matched_view: map_index_view(candidate.matched_view),
        qgram_overlap: candidate.qgram_overlap,
        total_qgram_overlap: candidate.total_qgram_overlap,
        best_view_score: candidate.best_view_score,
        cross_view_support: candidate.cross_view_support,
        token_count_match: candidate.token_count_match,
        phone_count_delta: candidate.phone_count_delta,
        token_bonus: candidate.token_bonus,
        phone_bonus: candidate.phone_bonus,
        extra_length_penalty: candidate.extra_length_penalty,
        structure_bonus: candidate.structure_bonus,
        coarse_score: candidate.coarse_score,
        token_distance: candidate.token_distance,
        token_weighted_distance: candidate.token_weighted_distance,
        token_boundary_penalty: candidate.token_boundary_penalty,
        token_max_len: candidate.token_max_len,
        token_score: candidate.token_score,
        feature_distance: candidate.feature_distance,
        feature_weighted_distance: candidate.feature_weighted_distance,
        feature_boundary_penalty: candidate.feature_boundary_penalty,
        feature_max_len: candidate.feature_max_len,
        feature_score: candidate.feature_score,
        feature_bonus: candidate.feature_bonus,
        feature_gate_token_ok: candidate.feature_gate_token_ok,
        feature_gate_coarse_ok: candidate.feature_gate_coarse_ok,
        feature_gate_phone_ok: candidate.feature_gate_phone_ok,
        short_guard_applied: candidate.short_guard_applied,
        short_guard_onset_match: candidate.short_guard_onset_match,
        short_guard_passed: candidate.short_guard_passed,
        low_content_guard_applied: candidate.low_content_guard_applied,
        low_content_guard_passed: candidate.low_content_guard_passed,
        acceptance_floor_passed: candidate.acceptance_floor_passed,
        used_feature_bonus: candidate.used_feature_bonus,
        phonetic_score: candidate.phonetic_score,
        acceptance_score: candidate.acceptance_score,
        verified: candidate.verified,
    }
}

pub(crate) fn map_identifier_flags(flags: &bee_phonetic::IdentifierFlags) -> IdentifierFlags {
    IdentifierFlags {
        acronym_like: flags.acronym_like,
        has_digits: flags.has_digits,
        snake_like: flags.snake_like,
        camel_like: flags.camel_like,
        symbol_like: flags.symbol_like,
    }
}

pub(crate) fn compare_candidate_rows(
    a: &bee_phonetic::CandidateFeatureRow,
    b: &bee_phonetic::CandidateFeatureRow,
) -> std::cmp::Ordering {
    b.acceptance_score
        .total_cmp(&a.acceptance_score)
        .then_with(|| b.phonetic_score.total_cmp(&a.phonetic_score))
        .then_with(|| b.coarse_score.total_cmp(&a.coarse_score))
        .then_with(|| b.qgram_overlap.cmp(&a.qgram_overlap))
}

pub(crate) fn load_counterexample_recordings() -> Result<Vec<CounterexampleRecordingRow>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../data/phonetic-seed/counterexample_recordings.jsonl");
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let mut rows = Vec::new();
    for (line_idx, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row = facet_json::from_str::<CounterexampleRecordingRow>(line)
            .map_err(|e| anyhow::anyhow!("{e:?}"))
            .with_context(|| format!("parsing {} line {}", path.display(), line_idx + 1))?;
        rows.push(row);
    }
    Ok(rows)
}
