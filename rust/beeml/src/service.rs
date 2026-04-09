use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use bee_phonetic::{
    AlignmentOp, AlignmentOpKind, PhoneticIndex, RetrievalQuery, SeedDataset,
    TranscriptAlignmentToken, align_token_sequences,
    align_token_sequences_with_left_word_boundaries, enumerate_transcript_spans_with,
    feature_similarity, normalize_ipa_for_comparison, normalize_ipa_for_comparison_with_spans,
    phoneme_similarity, query_index, reduce_ipa_tokens, score_shortlist, sentence_word_tokens,
    top_right_anchor_windows,
};
use bee_transcribe::zipa_align::timed_range_for_normalized_range;
use bee_transcribe::{AlignedWord, Engine, SessionSnapshot};
use bee_zipa_mlx::audio::AudioBuffer;
use bee_zipa_mlx::infer::ZipaInference;
use beeml::g2p::CachedEspeakG2p;
use beeml::judge::{OnlineJudge, extract_span_context};
use beeml::kokoro_phonemes::sanitize_for_kokoro;
use beeml::rpc::{
    CorpusAlignmentBucketSummary, CorpusAlignmentEvalJob, CorpusAlignmentEvalJobStatus,
    CorpusAlignmentEvalResult, CorpusAlignmentEvalRow, CorpusAlignmentSelectedSpanRole,
    CorpusCapturePrompt, FilterDecision, JudgeOptionDebug, JudgeStateDebug,
    PhoneticComparisonRequest, PhoneticComparisonResult, PhoneticComparisonRow,
    PhoneticComparisonSummary, RapidFireChoice, RetrievalCandidateDebug,
    RetrievalPrototypeEvalRequest, RetrievalPrototypeProbeRequest, RetrievalPrototypeProbeResult,
    SpanDebugTrace, SpanDebugView, SynthesizePhonemesRequest, SynthesizePhonemesResult,
    TeachRetrievalPrototypeJudgeRequest, TimingBreakdown, TranscribeAsrObservedToken,
    TranscribeAsrTokenAlternative, TranscribePhoneticAlignmentKind, TranscribePhoneticAlignmentOp,
    TranscribePhoneticAnchorConfidence, TranscribePhoneticCandidate, TranscribePhoneticSpan,
    TranscribePhoneticSpanClass, TranscribePhoneticSpanUsefulness, TranscribePhoneticTrace,
    TranscribePhoneticWordAlignment, TranscribeZipaPhoneSpan,
};
use rand::seq::SliceRandom;

use crate::offline_eval::*;
use crate::rapid_fire::build_rapid_fire_decision_set;
use crate::util::*;

#[derive(Clone, Debug, facet::Facet)]
struct KokoroSidecarRequest {
    phonemes: String,
    voice: Option<String>,
    speed: f32,
    lang: String,
    out: String,
}

#[derive(Clone, Debug, facet::Facet)]
struct KokoroSidecarResponse {
    ok: bool,
    resolved_voice: Option<String>,
    sample_rate_hz: Option<u32>,
    wav_path: Option<String>,
    error: Option<String>,
}

pub(crate) struct KokoroSidecar {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    stderr: BufReader<ChildStderr>,
}

struct SpanAlignmentSelection {
    range: std::ops::Range<usize>,
    alignment: bee_phonetic::TokenAlignment,
    zipa_normalized: Vec<String>,
    projected_alignment_score: Option<f32>,
    chosen_alignment_score: f32,
    second_best_alignment_score: Option<f32>,
    alignment_score_gap: Option<f32>,
    alignment_source: &'static str,
}

#[derive(Clone)]
struct WordSegmentCandidate {
    zipa_norm_range: std::ops::Range<usize>,
    zipa_normalized: Vec<String>,
    alignment: bee_phonetic::TokenAlignment,
    local_score: f32,
}

#[derive(Clone)]
pub(crate) struct BeeMlService {
    pub(crate) inner: Arc<BeemlServiceInner>,
}

pub(crate) struct BeemlServiceInner {
    pub(crate) engine: Engine,
    pub(crate) index: PhoneticIndex,
    pub(crate) dataset: SeedDataset,
    pub(crate) counterexamples: Vec<CounterexampleRecordingRow>,
    pub(crate) g2p: Mutex<CachedEspeakG2p>,
    pub(crate) zipa: Mutex<ZipaInference>,
    pub(crate) zipa_wav_dir: PathBuf,
    pub(crate) corpus_dir: PathBuf,
    pub(crate) kokoro_sidecar: Mutex<Option<KokoroSidecar>>,
    pub(crate) corpus_eval_jobs: Mutex<HashMap<u64, CorpusAlignmentEvalJob>>,
    pub(crate) next_corpus_eval_job_id: AtomicU64,
    pub(crate) judge: Mutex<OnlineJudge>,
    pub(crate) event_log_path: PathBuf,
}

pub(crate) use bee_phonetic::CounterexampleRecordingRow;

#[derive(Clone)]
pub(crate) struct EvalCase {
    pub(crate) case_id: String,
    pub(crate) suite: &'static str,
    pub(crate) target_term: String,
    pub(crate) source_text: String,
    pub(crate) transcript: String,
    pub(crate) should_abstain: bool,
    pub(crate) take: Option<i64>,
    pub(crate) audio_path: Option<String>,
    pub(crate) surface_form: Option<String>,
    pub(crate) words: Vec<AlignedWord>,
}

impl BeeMlService {
    pub(crate) fn transcribe_samples_with_options(
        &self,
        samples: &[f32],
        options: bee_transcribe::SessionOptions,
    ) -> Result<bee_transcribe::FinishResult, String> {
        let (result, _) = self.transcribe_samples_with_options_and_history(samples, options)?;
        Ok(result)
    }

    pub(crate) fn transcribe_samples_with_options_and_history(
        &self,
        samples: &[f32],
        options: bee_transcribe::SessionOptions,
    ) -> Result<(bee_transcribe::FinishResult, Vec<SessionSnapshot>), String> {
        let chunk_samples = (options.chunk_duration * 16_000.0) as usize;
        let mut session = self
            .inner
            .engine
            .session(options)
            .map_err(|e| e.to_string())?;
        let mut snapshots = Vec::new();

        let mut offset = 0;
        while offset < samples.len() {
            let end = (offset + chunk_samples).min(samples.len());
            if let Some(snapshot) = session
                .feed(&samples[offset..end])
                .map_err(|e| e.to_string())?
            {
                snapshots.push(snapshot);
            }
            offset = end;
        }

        let result = session.finish().map_err(|e| e.to_string())?;
        if snapshots
            .last()
            .is_none_or(|snapshot| snapshot.revision != result.snapshot.revision)
        {
            snapshots.push(result.snapshot.clone());
        }
        Ok((result, snapshots))
    }

    pub(crate) fn transcribe_samples_chunked(
        &self,
        samples: &[f32],
    ) -> Result<bee_transcribe::FinishResult, String> {
        self.transcribe_samples_with_options(samples, bee_transcribe::SessionOptions::default())
    }

    pub(crate) fn corpus_capture_prompts(&self) -> Vec<beeml::rpc::CorpusCapturePrompt> {
        const PROMPT_SET_PREFIX: &str = "zipa-targeted-v1";
        load_corpus_prompt_rows()
            .unwrap_or_else(|error| {
                tracing::error!(?error, "failed to load ZIPA corpus prompts");
                Vec::new()
            })
            .into_iter()
            .enumerate()
            .map(|(index, row)| {
                let ordinal = index as u32 + 1;
                beeml::rpc::CorpusCapturePrompt {
                    prompt_id: format!("{PROMPT_SET_PREFIX}-{ordinal:03}"),
                    ordinal,
                    bucket: row.bucket,
                    term: row.term,
                    text: row.text,
                    prompt_notes: row.prompt_notes,
                }
            })
            .collect()
    }

    pub(crate) fn build_transcribe_phonetic_trace(
        &self,
        audio: &AudioBuffer,
        snapshot: &SessionSnapshot,
        snapshots: &[SessionSnapshot],
    ) -> Result<TranscribePhoneticTrace, String> {
        let tail_blocks_rescue = !snapshot.pending_text.trim().is_empty()
            || snapshot.pending_token_count > 0
            || snapshot.ambiguity.volatile_token_count > 0;
        let transcript = snapshot.committed_text.trim();
        let words = &snapshot.committed_words;
        let alignments = words
            .iter()
            .map(|word| TranscriptAlignmentToken {
                start_time: word.start,
                end_time: word.end,
                confidence: word.confidence.clone(),
            })
            .collect::<Vec<_>>();

        let mut g2p = self
            .inner
            .g2p
            .lock()
            .map_err(|_| "g2p cache mutex poisoned".to_string())?;
        let zipa = self
            .inner
            .zipa
            .lock()
            .map_err(|_| "zipa mutex poisoned".to_string())?;

        let utterance = zipa.infer_audio(audio).map_err(|e| e.to_string())?;
        let utterance_duration_secs = audio.samples.len() as f64 / audio.sample_rate_hz as f64;
        let utterance_phone_spans = utterance
            .derive_phone_spans(&zipa.tokens, utterance_duration_secs, 0)
            .into_iter()
            .filter(|span| span.token != "▁")
            .collect::<Vec<_>>();
        let utterance_zipa_phone_spans = utterance_phone_spans
            .iter()
            .map(|span| TranscribeZipaPhoneSpan {
                token_id: span.token_id.min(u32::MAX as usize) as u32,
                token: span.token.clone(),
                start_frame: span.start_frame.min(u32::MAX as usize) as u32,
                end_frame: span.end_frame.min(u32::MAX as usize) as u32,
                start_sec: span.start_time_secs,
                end_sec: span.end_time_secs,
            })
            .collect::<Vec<_>>();
        let utterance_zipa_raw = utterance
            .tokens
            .into_iter()
            .filter(|token| token != "▁")
            .collect::<Vec<_>>();
        let utterance_zipa_normalized_with_spans =
            normalize_ipa_for_comparison_with_spans(&utterance_zipa_raw);
        let utterance_zipa_normalized = utterance_zipa_normalized_with_spans
            .iter()
            .map(|token| token.token.clone())
            .collect::<Vec<_>>();
        let transcript_word_raw_ranges = transcript_word_raw_ranges(&mut g2p, transcript)?;
        let utterance_transcript_raw = transcript_word_raw_ranges
            .iter()
            .flat_map(|(_, raw_tokens)| raw_tokens.iter().cloned())
            .collect::<Vec<_>>();
        let transcript_word_normalized_ranges = transcript_word_raw_ranges
            .iter()
            .map(|(range, raw_tokens)| (range.clone(), normalize_ipa_for_comparison(raw_tokens)))
            .collect::<Vec<_>>();
        let utterance_transcript_normalized = transcript_word_normalized_ranges
            .iter()
            .flat_map(|(_, tokens)| tokens.iter().cloned())
            .collect::<Vec<_>>();
        let utterance_word_ids = transcript_word_normalized_ranges
            .iter()
            .enumerate()
            .flat_map(|(word_index, (_, tokens))| std::iter::repeat_n(word_index, tokens.len()))
            .collect::<Vec<_>>();
        let utterance_alignment = align_token_sequences_with_left_word_boundaries(
            &utterance_transcript_normalized,
            &utterance_zipa_normalized,
            &utterance_word_ids,
        );
        let transcript_words = sentence_word_tokens(transcript);
        let transcript_token_ranges = (0..transcript_word_normalized_ranges.len())
            .map(|word_index| {
                transcript_token_range_for_span(
                    &transcript_word_normalized_ranges,
                    word_index,
                    word_index + 1,
                )
            })
            .collect::<Vec<_>>();
        let transcript_word_tokens = transcript_word_normalized_ranges
            .iter()
            .map(|(_, tokens)| tokens.clone())
            .collect::<Vec<_>>();
        let word_windows = select_segmental_word_windows(
            &transcript_word_tokens,
            &transcript_token_ranges,
            &utterance_zipa_normalized,
            &utterance_alignment,
        );
        let word_alignments = transcript_word_normalized_ranges
            .iter()
            .enumerate()
            .filter_map(|(word_index, (_, transcript_normalized))| {
                let window = word_windows.get(word_index)?.as_ref()?;
                let zipa_norm_range = window.zipa_norm_range.clone();
                let zipa_normalized = utterance_zipa_normalized
                    .get(zipa_norm_range.clone())
                    .unwrap_or(&[])
                    .to_vec();
                if zipa_normalized.is_empty() {
                    return None;
                }
                let transcript_raw = transcript_word_raw_ranges
                    .get(word_index)
                    .map(|(_, raw_tokens)| raw_tokens.clone())
                    .unwrap_or_default();
                let timed_zipa_range = timed_range_for_normalized_range(
                    &utterance_zipa_normalized_with_spans,
                    &utterance_phone_spans,
                    zipa_norm_range.clone(),
                );
                let zipa_raw = raw_slice_for_normalized_range(
                    &utterance_zipa_raw,
                    &utterance_zipa_normalized_with_spans,
                    zipa_norm_range.clone(),
                );
                let word_text = transcript_words.get(word_index)?.text.clone();
                let aligned_word = words.get(word_index)?;
                Some(TranscribePhoneticWordAlignment {
                    word_text,
                    token_start: word_index as u32,
                    token_end: (word_index + 1) as u32,
                    start_sec: aligned_word.start,
                    end_sec: aligned_word.end,
                    zipa_raw_phone_start: timed_zipa_range
                        .as_ref()
                        .map(|range| range.raw_phone_range.start.min(u32::MAX as usize) as u32),
                    zipa_raw_phone_end: timed_zipa_range
                        .as_ref()
                        .map(|range| range.raw_phone_range.end.min(u32::MAX as usize) as u32),
                    zipa_start_sec: timed_zipa_range.as_ref().map(|range| range.start_time_secs),
                    zipa_end_sec: timed_zipa_range.as_ref().map(|range| range.end_time_secs),
                    transcript_raw,
                    transcript_normalized: transcript_normalized.clone(),
                    zipa_norm_start: zipa_norm_range.start as u32,
                    zipa_norm_end: zipa_norm_range.end as u32,
                    zipa_raw,
                    zipa_normalized,
                    alignment: map_alignment_ops(&window.ops),
                })
            })
            .collect::<Vec<_>>();

        let spans = enumerate_transcript_spans_with(transcript, 3, Some(&alignments[..]), |text| {
            g2p.ipa_tokens(text).ok().flatten()
        });

        let mut phonetic_spans = Vec::new();
        for span in spans {
            let transcript_normalized = transcript_normalized_for_span(
                &transcript_word_normalized_ranges,
                span.token_start,
                span.token_end,
            );
            if transcript_normalized.is_empty() {
                continue;
            }
            let transcript_token_count = transcript_normalized.len().min(u32::MAX as usize) as u32;
            let transcript_norm_range = transcript_token_range_for_span(
                &transcript_word_normalized_ranges,
                span.token_start,
                span.token_end,
            );
            let Some(selection) = select_span_alignment_range(
                &transcript_normalized,
                &utterance_zipa_normalized,
                utterance_alignment.project_left_range(transcript_norm_range),
            ) else {
                continue;
            };
            let zipa_norm_range = selection.range.clone();
            let zipa_raw = raw_slice_for_normalized_range(
                &utterance_zipa_raw,
                &utterance_zipa_normalized_with_spans,
                zipa_norm_range.clone(),
            );
            let zipa_token_count = selection.zipa_normalized.len().min(u32::MAX as usize) as u32;
            let transcript_similarity =
                phoneme_similarity(&selection.zipa_normalized, &transcript_normalized);
            let transcript_feature_similarity =
                feature_similarity(&selection.zipa_normalized, &transcript_normalized);

            let shortlist = query_index(
                &self.inner.index,
                &RetrievalQuery {
                    text: span.text.clone(),
                    ipa_tokens: span.ipa_tokens.clone(),
                    reduced_ipa_tokens: span.reduced_ipa_tokens.clone(),
                    feature_tokens: bee_phonetic::feature_tokens_for_ipa(&span.ipa_tokens),
                    token_count: (span.token_end - span.token_start) as u8,
                },
                8,
            );
            let mut scored_rows = score_shortlist(&span, &shortlist, &self.inner.index);
            scored_rows.sort_by(|a, b| {
                b.acceptance_score
                    .total_cmp(&a.acceptance_score)
                    .then_with(|| b.phonetic_score.total_cmp(&a.phonetic_score))
            });

            let mut candidates = Vec::new();
            let mut seen_terms = std::collections::HashSet::new();
            for scored in scored_rows {
                let alias = &self.inner.index.aliases[scored.alias_id as usize];
                if !seen_terms.insert(alias.term.to_ascii_lowercase()) {
                    continue;
                }
                let candidate_raw = g2p
                    .ipa_tokens(&alias.alias_text)
                    .map_err(|e| e.to_string())?
                    .ok_or_else(|| {
                        format!("espeak produced no tokens for '{}'", alias.alias_text)
                    })?;
                let candidate_normalized = normalize_ipa_for_comparison(&candidate_raw);
                let feature_similarity =
                    feature_similarity(&selection.zipa_normalized, &candidate_normalized);
                let similarity_delta = match (feature_similarity, transcript_feature_similarity) {
                    (Some(candidate), Some(transcript)) => Some(candidate - transcript),
                    _ => None,
                };
                candidates.push(TranscribePhoneticCandidate {
                    term: alias.term.clone(),
                    alias_text: alias.alias_text.clone(),
                    alias_source: map_alias_source(alias.alias_source),
                    candidate_normalized,
                    feature_similarity,
                    similarity_delta,
                });
                if candidates.len() >= 5 {
                    break;
                }
            }
            candidates.sort_by(|a, b| {
                b.similarity_delta
                    .unwrap_or(f32::NEG_INFINITY)
                    .total_cmp(&a.similarity_delta.unwrap_or(f32::NEG_INFINITY))
            });
            let span_usefulness = span_usefulness(
                &span.text,
                transcript_token_count,
                &selection.zipa_normalized,
            );
            let span_class = span_class(
                &span.text,
                transcript_token_count,
                &selection.zipa_normalized,
            );
            let candidate_plausible = zipa_candidate_plausible(&candidates);
            let anchor_confidence = anchor_confidence(
                selection.projected_alignment_score,
                selection.chosen_alignment_score,
                selection.alignment_score_gap,
                transcript_token_count,
                zipa_token_count,
            );
            let zipa_rescue_eligible =
                matches!(
                    anchor_confidence,
                    TranscribePhoneticAnchorConfidence::Medium
                        | TranscribePhoneticAnchorConfidence::High
                ) && !matches!(span_usefulness, TranscribePhoneticSpanUsefulness::Low)
                    && candidate_plausible
                    && !tail_blocks_rescue;

            phonetic_spans.push(TranscribePhoneticSpan {
                span_text: span.text,
                token_start: span.token_start as u32,
                token_end: span.token_end as u32,
                start_sec: span.start_sec.unwrap_or(0.0),
                end_sec: span.end_sec.unwrap_or(0.0),
                zipa_norm_start: zipa_norm_range.start as u32,
                zipa_norm_end: zipa_norm_range.end as u32,
                zipa_raw,
                zipa_normalized: selection.zipa_normalized,
                transcript_normalized,
                transcript_phone_count: transcript_token_count,
                chosen_zipa_phone_count: zipa_token_count,
                transcript_similarity,
                transcript_feature_similarity,
                projected_alignment_score: selection.projected_alignment_score,
                chosen_alignment_score: Some(selection.chosen_alignment_score),
                second_best_alignment_score: selection.second_best_alignment_score,
                alignment_score_gap: selection.alignment_score_gap,
                alignment_source: selection.alignment_source.to_string(),
                anchor_confidence,
                span_usefulness,
                span_class,
                zipa_rescue_eligible,
                alignment: map_alignment_ops(&selection.alignment.ops),
                candidates,
            });
        }

        phonetic_spans.sort_by(|a, b| {
            best_delta(&b.candidates)
                .total_cmp(&best_delta(&a.candidates))
                .then_with(|| a.token_start.cmp(&b.token_start))
        });

        let worst_raw_span_index = phonetic_spans
            .iter()
            .enumerate()
            .filter_map(|(index, span)| {
                span.transcript_feature_similarity
                    .map(|score| (index.min(u32::MAX as usize) as u32, score))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(index, _)| index);

        let worst_contentful_span_index = phonetic_spans
            .iter()
            .enumerate()
            .filter(|(_, span)| {
                !matches!(span.span_usefulness, TranscribePhoneticSpanUsefulness::Low)
            })
            .filter_map(|(index, span)| {
                span.transcript_feature_similarity
                    .map(|score| (index.min(u32::MAX as usize) as u32, score))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(index, _)| index);

        let best_rescue_span_index = phonetic_spans
            .iter()
            .enumerate()
            .filter(|(_, span)| span.zipa_rescue_eligible)
            .filter_map(|(index, span)| {
                let best_delta = span
                    .candidates
                    .iter()
                    .filter_map(|candidate| candidate.similarity_delta)
                    .max_by(|a, b| a.total_cmp(b))?;
                Some((index.min(u32::MAX as usize) as u32, best_delta))
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(index, _)| index);

        Ok(TranscribePhoneticTrace {
            snapshot_revision: snapshot.revision,
            aligned_transcript: snapshot.committed_text.clone(),
            pending_text: snapshot.pending_text.clone(),
            full_transcript: snapshot.full_text.clone(),
            session_audio_f32: audio.samples.clone(),
            session_audio_sample_rate_hz: audio.sample_rate_hz,
            tail_ambiguity: snapshot.ambiguity.clone(),
            worst_raw_span_index,
            worst_contentful_span_index,
            best_rescue_span_index,
            utterance_similarity: phoneme_similarity(
                &utterance_zipa_raw,
                &utterance_transcript_raw,
            ),
            utterance_feature_similarity: feature_similarity(
                &utterance_zipa_normalized,
                &utterance_transcript_normalized,
            ),
            utterance_zipa_raw,
            utterance_zipa_phone_spans,
            utterance_zipa_normalized,
            utterance_transcript_normalized,
            utterance_alignment: map_alignment_ops(&utterance_alignment.ops),
            asr_alternatives: collect_asr_alternatives(snapshots),
            word_alignments,
            spans: phonetic_spans,
        })
    }

    pub(crate) fn latest_corpus_recordings(
        &self,
        bucket_filter: Option<&str>,
        prompt_id_filter: Option<&str>,
    ) -> Result<Vec<(CorpusCapturePrompt, beeml::rpc::CorpusCaptureRecording)>, String> {
        let prompts = self.corpus_capture_prompts();
        let prompt_map = prompts
            .into_iter()
            .filter(|prompt| bucket_filter.is_none_or(|bucket| prompt.bucket == bucket))
            .filter(|prompt| prompt_id_filter.is_none_or(|prompt_id| prompt.prompt_id == prompt_id))
            .map(|prompt| (prompt.prompt_id.clone(), prompt))
            .collect::<std::collections::HashMap<_, _>>();

        let mut latest =
            std::collections::HashMap::<String, beeml::rpc::CorpusCaptureRecording>::new();
        for recording in
            load_corpus_recordings(&self.inner.corpus_dir).map_err(|e| e.to_string())?
        {
            if !prompt_map.contains_key(&recording.prompt_id) {
                continue;
            }
            let replace = match latest.get(&recording.prompt_id) {
                Some(existing) => {
                    recording.take > existing.take
                        || (recording.take == existing.take
                            && recording.created_at_unix_ms > existing.created_at_unix_ms)
                }
                None => true,
            };
            if replace {
                latest.insert(recording.prompt_id.clone(), recording);
            }
        }

        let mut rows = latest
            .into_iter()
            .filter_map(|(prompt_id, recording)| {
                prompt_map
                    .get(&prompt_id)
                    .cloned()
                    .map(|prompt| (prompt, recording))
            })
            .collect::<Vec<_>>();
        rows.sort_by(|(left_prompt, _), (right_prompt, _)| {
            left_prompt.ordinal.cmp(&right_prompt.ordinal)
        });
        Ok(rows)
    }

    pub(crate) fn eval_corpus_alignment(
        &self,
        limit: usize,
        bucket_filter: Option<&str>,
        randomize: bool,
        prompt_id_filter: Option<&str>,
    ) -> Result<CorpusAlignmentEvalResult, String> {
        let mut recordings = self.latest_corpus_recordings(bucket_filter, prompt_id_filter)?;
        if randomize {
            let mut rng = rand::thread_rng();
            recordings.shuffle(&mut rng);
        }
        let mut rows = Vec::new();

        for (prompt, recording) in recordings.into_iter().take(limit) {
            let wav_bytes = std::fs::read(&recording.wav_path)
                .map_err(|e| format!("reading {}: {e}", recording.wav_path))?;
            let samples = bee_transcribe::decode_wav(&wav_bytes).map_err(|e| e.to_string())?;

            let mut options = bee_transcribe::SessionOptions::default();
            options.language = bee_transcribe::Language("English".to_string());
            let (result, snapshots) =
                self.transcribe_samples_with_options_and_history(&samples, options)?;
            let audio = AudioBuffer {
                samples: result.session_audio.samples().to_vec(),
                sample_rate_hz: result.session_audio.sample_rate().0 as u32,
            };
            let snapshot = result.snapshot;

            match self.build_transcribe_phonetic_trace(&audio, &snapshot, &snapshots) {
                Ok(trace) => {
                    let tail_volatile_token_count = trace.tail_ambiguity.volatile_token_count;
                    let row_rescue_ready =
                        tail_volatile_token_count == 0 && trace.pending_text.trim().is_empty();
                    let positive_span_count = trace
                        .spans
                        .iter()
                        .filter(|span| {
                            span.zipa_rescue_eligible
                                && span.candidates.iter().any(|candidate| {
                                    candidate.similarity_delta.unwrap_or(0.0) > 0.0
                                })
                        })
                        .count()
                        .min(u32::MAX as usize)
                        as u32;
                    let contentful_span_count = trace
                        .spans
                        .iter()
                        .filter(|span| {
                            !matches!(span.span_usefulness, TranscribePhoneticSpanUsefulness::Low)
                        })
                        .count()
                        .min(u32::MAX as usize)
                        as u32;
                    let rescue_eligible_span_count = trace
                        .spans
                        .iter()
                        .filter(|span| span.zipa_rescue_eligible)
                        .count()
                        .min(u32::MAX as usize)
                        as u32;
                    let worst_span_feature_similarity = trace
                        .spans
                        .iter()
                        .filter_map(|span| span.transcript_feature_similarity)
                        .min_by(|a, b| a.total_cmp(b));
                    let best_span_delta = trace
                        .spans
                        .iter()
                        .filter(|span| span.zipa_rescue_eligible)
                        .flat_map(|span| {
                            span.candidates
                                .iter()
                                .filter_map(|candidate| candidate.similarity_delta)
                        })
                        .max_by(|a, b| a.total_cmp(b));
                    let selected_span_role = trace
                        .best_rescue_span_index
                        .map(|_| CorpusAlignmentSelectedSpanRole::BestRescue)
                        .or_else(|| {
                            trace
                                .worst_contentful_span_index
                                .map(|_| CorpusAlignmentSelectedSpanRole::WorstContentful)
                        })
                        .or_else(|| {
                            trace
                                .worst_raw_span_index
                                .map(|_| CorpusAlignmentSelectedSpanRole::WorstRaw)
                        });
                    let selected_span = trace
                        .best_rescue_span_index
                        .or(trace.worst_contentful_span_index)
                        .or(trace.worst_raw_span_index)
                        .and_then(|index| trace.spans.get(index as usize));
                    rows.push(CorpusAlignmentEvalRow {
                        prompt_id: prompt.prompt_id,
                        ordinal: prompt.ordinal,
                        bucket: prompt.bucket,
                        term: prompt.term,
                        prompt_text: prompt.text,
                        prompt_notes: prompt.prompt_notes,
                        take: recording.take,
                        wav_path: recording.wav_path,
                        asr_transcript: snapshot.full_text.clone(),
                        utterance_similarity: trace.utterance_similarity,
                        utterance_feature_similarity: trace.utterance_feature_similarity,
                        tail_volatile_token_count,
                        row_rescue_ready,
                        positive_span_count,
                        contentful_span_count,
                        rescue_eligible_span_count,
                        worst_span_feature_similarity,
                        best_span_delta,
                        selected_span_role,
                        selected_span_text: selected_span.map(|span| span.span_text.clone()),
                        selected_span_feature_similarity: selected_span
                            .and_then(|span| span.transcript_feature_similarity),
                        selected_span_best_delta: selected_span.and_then(|span| {
                            span.candidates
                                .iter()
                                .filter_map(|candidate| candidate.similarity_delta)
                                .max_by(|a, b| a.total_cmp(b))
                        }),
                        trace: Some(trace),
                        error: None,
                    });
                }
                Err(error) => rows.push(CorpusAlignmentEvalRow {
                    prompt_id: prompt.prompt_id,
                    ordinal: prompt.ordinal,
                    bucket: prompt.bucket,
                    term: prompt.term,
                    prompt_text: prompt.text,
                    prompt_notes: prompt.prompt_notes,
                    take: recording.take,
                    wav_path: recording.wav_path,
                    asr_transcript: snapshot.full_text,
                    utterance_similarity: None,
                    utterance_feature_similarity: None,
                    tail_volatile_token_count: 0,
                    row_rescue_ready: false,
                    positive_span_count: 0,
                    contentful_span_count: 0,
                    rescue_eligible_span_count: 0,
                    worst_span_feature_similarity: None,
                    best_span_delta: None,
                    selected_span_role: None,
                    selected_span_text: None,
                    selected_span_feature_similarity: None,
                    selected_span_best_delta: None,
                    trace: None,
                    error: Some(error),
                }),
            }
        }

        rows.sort_by(|a, b| {
            a.utterance_feature_similarity
                .unwrap_or(f32::INFINITY)
                .total_cmp(&b.utterance_feature_similarity.unwrap_or(f32::INFINITY))
                .then_with(|| a.ordinal.cmp(&b.ordinal))
        });

        let mut by_bucket =
            std::collections::BTreeMap::<String, Vec<&CorpusAlignmentEvalRow>>::new();
        for row in &rows {
            by_bucket.entry(row.bucket.clone()).or_default().push(row);
        }
        let bucket_summaries = by_bucket
            .into_iter()
            .map(|(bucket, bucket_rows)| CorpusAlignmentBucketSummary {
                bucket,
                rows: bucket_rows.len().min(u32::MAX as usize) as u32,
                utterance_feature_similarity_mean: mean_option(
                    bucket_rows
                        .iter()
                        .map(|row| row.utterance_feature_similarity),
                ),
                utterance_similarity_mean: mean_option(
                    bucket_rows.iter().map(|row| row.utterance_similarity),
                ),
            })
            .collect();

        Ok(CorpusAlignmentEvalResult {
            rows,
            bucket_summaries,
        })
    }

    pub(crate) fn create_corpus_eval_job(
        &self,
        limit: u32,
        bucket: Option<String>,
        prompt_id: Option<String>,
        _randomize: bool,
    ) -> Result<CorpusAlignmentEvalJob, String> {
        let total_rows = self
            .latest_corpus_recordings(bucket.as_deref(), prompt_id.as_deref())?
            .len()
            .min(limit as usize)
            .min(u32::MAX as usize) as u32;
        let job_id = self
            .inner
            .next_corpus_eval_job_id
            .fetch_add(1, Ordering::Relaxed);
        let job = CorpusAlignmentEvalJob {
            job_id,
            status: CorpusAlignmentEvalJobStatus::Running,
            limit,
            bucket,
            completed_rows: 0,
            total_rows,
            started_at_unix_ms: unix_time_ms(),
            finished_at_unix_ms: None,
            result: None,
            error: None,
        };
        self.inner
            .corpus_eval_jobs
            .lock()
            .map_err(|_| "corpus eval job mutex poisoned".to_string())?
            .insert(job_id, job.clone());
        Ok(job)
    }

    pub(crate) fn get_corpus_eval_job(
        &self,
        job_id: u64,
    ) -> Result<CorpusAlignmentEvalJob, String> {
        self.inner
            .corpus_eval_jobs
            .lock()
            .map_err(|_| "corpus eval job mutex poisoned".to_string())?
            .get(&job_id)
            .cloned()
            .ok_or_else(|| format!("unknown corpus eval job {job_id}"))
    }

    pub(crate) fn update_corpus_eval_job_progress(
        &self,
        job_id: u64,
        completed_rows: u32,
    ) -> Result<(), String> {
        let mut jobs = self
            .inner
            .corpus_eval_jobs
            .lock()
            .map_err(|_| "corpus eval job mutex poisoned".to_string())?;
        let job = jobs
            .get_mut(&job_id)
            .ok_or_else(|| format!("unknown corpus eval job {job_id}"))?;
        job.completed_rows = completed_rows.min(job.total_rows);
        Ok(())
    }

    pub(crate) fn finish_corpus_eval_job(
        &self,
        job_id: u64,
        result: Result<CorpusAlignmentEvalResult, String>,
    ) -> Result<(), String> {
        let mut jobs = self
            .inner
            .corpus_eval_jobs
            .lock()
            .map_err(|_| "corpus eval job mutex poisoned".to_string())?;
        let job = jobs
            .get_mut(&job_id)
            .ok_or_else(|| format!("unknown corpus eval job {job_id}"))?;
        job.finished_at_unix_ms = Some(unix_time_ms());
        match result {
            Ok(result) => {
                job.completed_rows = job.total_rows;
                job.status = CorpusAlignmentEvalJobStatus::Completed;
                job.result = Some(result);
                job.error = None;
            }
            Err(error) => {
                job.status = CorpusAlignmentEvalJobStatus::Failed;
                job.error = Some(error);
                job.result = None;
            }
        }
        Ok(())
    }

    pub(crate) fn run_phonetic_comparison(
        &self,
        request: PhoneticComparisonRequest,
    ) -> Result<PhoneticComparisonResult, String> {
        let mut g2p = self
            .inner
            .g2p
            .lock()
            .map_err(|_| "g2p cache mutex poisoned".to_string())?;
        let zipa = self
            .inner
            .zipa
            .lock()
            .map_err(|_| "zipa mutex poisoned".to_string())?;

        let mut rows = Vec::new();
        for row in &self.inner.dataset.recording_examples {
            if request
                .term
                .as_ref()
                .is_some_and(|term| !row.term.eq_ignore_ascii_case(term))
            {
                continue;
            }
            if request.limit > 0 && rows.len() >= request.limit as usize {
                break;
            }

            let wav_path = self.zipa_wav_path(row)?;
            let zipa_result = zipa.infer_wav(&wav_path).map_err(|e| e.to_string())?;
            let zipa_raw = zipa_result
                .tokens
                .into_iter()
                .filter(|token| token != "▁")
                .collect::<Vec<_>>();
            let zipa_reduced = reduce_ipa_tokens(&zipa_raw);
            let zipa_normalized = normalize_ipa_for_comparison(&zipa_raw);

            let text = if request.use_transcript {
                row.transcript.clone()
            } else {
                row.text.clone()
            };
            let espeak_raw = g2p
                .ipa_tokens(&text)
                .map_err(|e| e.to_string())?
                .ok_or_else(|| format!("espeak produced no tokens for '{text}'"))?;
            let espeak_reduced = reduce_ipa_tokens(&espeak_raw);
            let espeak_normalized = normalize_ipa_for_comparison(&espeak_raw);

            rows.push(PhoneticComparisonRow {
                term: row.term.clone(),
                text,
                wav_path: wav_path.display().to_string(),
                raw_similarity: phoneme_similarity(&zipa_raw, &espeak_raw),
                reduced_similarity: phoneme_similarity(&zipa_reduced, &espeak_reduced),
                normalized_similarity: phoneme_similarity(&zipa_normalized, &espeak_normalized),
                feature_similarity: feature_similarity(&zipa_raw, &espeak_raw),
                normalized_feature_similarity: feature_similarity(
                    &zipa_normalized,
                    &espeak_normalized,
                ),
                reduced_exact: zipa_reduced == espeak_reduced,
                normalized_exact: zipa_normalized == espeak_normalized,
                zipa_raw,
                espeak_raw,
                zipa_reduced,
                espeak_reduced,
                zipa_normalized,
                espeak_normalized,
            });
        }

        if rows.is_empty() {
            return Err("no matching recording examples".to_string());
        }

        Ok(PhoneticComparisonResult {
            summary: PhoneticComparisonSummary {
                rows: rows.len() as u32,
                reduced_exact: rows.iter().filter(|row| row.reduced_exact).count() as u32,
                normalized_exact: rows.iter().filter(|row| row.normalized_exact).count() as u32,
                raw_similarity_mean: mean_option(rows.iter().map(|row| row.raw_similarity)),
                reduced_similarity_mean: mean_option(rows.iter().map(|row| row.reduced_similarity)),
                normalized_similarity_mean: mean_option(
                    rows.iter().map(|row| row.normalized_similarity),
                ),
                feature_similarity_mean: mean_option(rows.iter().map(|row| row.feature_similarity)),
                normalized_feature_similarity_mean: mean_option(
                    rows.iter().map(|row| row.normalized_feature_similarity),
                ),
            },
            rows,
        })
    }

    pub(crate) fn synthesize_phonemes(
        &self,
        request: SynthesizePhonemesRequest,
    ) -> Result<SynthesizePhonemesResult, String> {
        let phonemes = sanitize_for_kokoro(&request.phonemes);
        if phonemes.is_empty() {
            return Err("phonemes must not be empty".to_string());
        }

        let model_path = kokoro_model_path()?;
        let voices_path = kokoro_voices_path()?;
        let pythonpath = kokoro_site_packages_path()?;
        let voice = request
            .voice
            .as_deref()
            .filter(|voice| !voice.trim().is_empty())
            .unwrap_or("af_sarah");
        let speed = request.speed.unwrap_or(1.0).clamp(0.5, 2.0);

        let temp_root = std::env::temp_dir().join("beeml-kokoro");
        fs::create_dir_all(&temp_root)
            .map_err(|e| format!("creating temp dir {}: {e}", temp_root.display()))?;
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| e.to_string())?
            .as_nanos();
        let out_path = temp_root.join(format!("phonemes-{stamp}.wav"));
        let response =
            self.with_kokoro_sidecar(&model_path, &voices_path, &pythonpath, |sidecar| {
                let line = facet_json::to_string(&KokoroSidecarRequest {
                    phonemes: phonemes.to_string(),
                    voice: Some(voice.to_string()),
                    speed,
                    lang: "en-us".to_string(),
                    out: out_path.display().to_string(),
                })
                .map_err(|e| format!("encoding kokoro request: {e}"))?;
                sidecar
                    .stdin
                    .write_all(line.as_bytes())
                    .map_err(|e| format!("writing kokoro request: {e}"))?;
                sidecar
                    .stdin
                    .write_all(b"\n")
                    .map_err(|e| format!("writing kokoro request newline: {e}"))?;
                sidecar
                    .stdin
                    .flush()
                    .map_err(|e| format!("flushing kokoro request: {e}"))?;

                let mut response_line = String::new();
                let read = sidecar
                    .stdout
                    .read_line(&mut response_line)
                    .map_err(|e| format!("reading kokoro response: {e}"))?;
                if read == 0 {
                    let mut stderr = String::new();
                    let _ = sidecar.stderr.read_to_string(&mut stderr);
                    return Err(format!(
                        "kokoro sidecar exited unexpectedly{}",
                        if stderr.trim().is_empty() {
                            String::new()
                        } else {
                            format!(", stderr:\n{stderr}")
                        }
                    ));
                }
                facet_json::from_str::<KokoroSidecarResponse>(response_line.trim())
                    .map_err(|e| format!("parsing kokoro response: {e}"))
            })?;

        if !response.ok {
            return Err(response
                .error
                .unwrap_or_else(|| "kokoro sidecar returned an unknown error".to_string()));
        }

        let wav_bytes =
            fs::read(&out_path).map_err(|e| format!("reading {}: {e}", out_path.display()))?;
        let _ = fs::remove_file(&out_path);

        Ok(SynthesizePhonemesResult {
            wav_bytes,
            sample_rate_hz: response.sample_rate_hz.unwrap_or(24_000),
            resolved_voice: response.resolved_voice.unwrap_or_else(|| voice.to_string()),
        })
    }

    pub(crate) fn load_audio_file(&self, path: &str) -> Result<Vec<u8>, String> {
        fs::read(path).map_err(|e| format!("reading {path}: {e}"))
    }

    pub(crate) fn warm_kokoro_sidecar(&self) -> Result<(), String> {
        let model_path = kokoro_model_path()?;
        let voices_path = kokoro_voices_path()?;
        let pythonpath = kokoro_site_packages_path()?;
        self.with_kokoro_sidecar(&model_path, &voices_path, &pythonpath, |_| Ok(()))
    }

    fn with_kokoro_sidecar<T>(
        &self,
        model_path: &PathBuf,
        voices_path: &PathBuf,
        pythonpath: &str,
        f: impl FnOnce(&mut KokoroSidecar) -> Result<T, String>,
    ) -> Result<T, String> {
        let mut guard = self
            .inner
            .kokoro_sidecar
            .lock()
            .map_err(|_| "kokoro sidecar mutex poisoned".to_string())?;

        if guard.is_none() {
            *guard = Some(start_kokoro_sidecar(model_path, voices_path, pythonpath)?);
        }

        let sidecar = guard
            .as_mut()
            .ok_or_else(|| "kokoro sidecar failed to initialize".to_string())?;
        match f(sidecar) {
            Ok(value) => Ok(value),
            Err(err) => {
                let _ = sidecar.child.kill();
                *guard = None;
                Err(err)
            }
        }
    }

    pub(crate) fn run_probe(
        &self,
        request: RetrievalPrototypeProbeRequest,
        teach: Option<TeachRetrievalPrototypeJudgeRequest>,
    ) -> Result<RetrievalPrototypeProbeResult, String> {
        let expected_source_text = request.expected_source_text.clone();
        let alignments = if request.words.is_empty() {
            None
        } else {
            Some(
                request
                    .words
                    .iter()
                    .map(|word| TranscriptAlignmentToken {
                        start_time: word.start,
                        end_time: word.end,
                        confidence: word.confidence.clone(),
                    })
                    .collect::<Vec<_>>(),
            )
        };

        let spans = {
            let mut g2p = self
                .inner
                .g2p
                .lock()
                .map_err(|_| "g2p cache mutex poisoned".to_string())?;
            enumerate_transcript_spans_with(
                &request.transcript,
                request.max_span_words as usize,
                alignments.as_deref(),
                |text| g2p.ipa_tokens(text).ok().flatten(),
            )
        };

        let mut judge = self
            .inner
            .judge
            .lock()
            .map_err(|_| "judge mutex poisoned".to_string())?;
        let mut taught_span = teach.is_none();
        let selected_component_choices = teach
            .as_ref()
            .map(|teach| teach.selected_component_choices.clone())
            .unwrap_or_default();
        let rejected_group_spans = teach
            .as_ref()
            .filter(|teach| teach.reject_group)
            .map(|teach| {
                teach
                    .rejected_group_spans
                    .iter()
                    .map(|span| (span.token_start as usize, span.token_end as usize))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let traces = spans
            .into_iter()
            .map(|span| {
                let shortlist = query_index(
                    &self.inner.index,
                    &RetrievalQuery {
                        text: span.text.clone(),
                        ipa_tokens: span.ipa_tokens.clone(),
                        reduced_ipa_tokens: span.reduced_ipa_tokens.clone(),
                        feature_tokens: bee_phonetic::feature_tokens_for_ipa(&span.ipa_tokens),
                        token_count: (span.token_end - span.token_start) as u8,
                    },
                    request.shortlist_limit as usize,
                );
                let scored_rows = score_shortlist(&span, &shortlist, &self.inner.index);
                let judge_input = scored_rows
                    .iter()
                    .map(|row| {
                        let alias = &self.inner.index.aliases[row.alias_id as usize];
                        (row.clone(), alias.identifier_flags.clone())
                    })
                    .collect::<Vec<_>>();

                let explicitly_chosen_span = teach.as_ref().is_some_and(|teach| {
                    !teach.reject_group
                        && teach.selected_component_choices.is_empty()
                        && teach.span_token_start as usize == span.token_start
                        && teach.span_token_end as usize == span.token_end
                });
                let selected_component_choice = selected_component_choices.iter().find(|choice| {
                    if choice.choose_keep_original {
                        choice.component_spans.iter().any(|component_span| {
                            component_span.token_start as usize == span.token_start
                                && component_span.token_end as usize == span.token_end
                        })
                    } else {
                        choice
                            .span_token_start
                            .is_some_and(|start| start as usize == span.token_start)
                            && choice
                                .span_token_end
                                .is_some_and(|end| end as usize == span.token_end)
                    }
                });
                let rejected_group_match = rejected_group_spans
                    .iter()
                    .any(|(start, end)| *start == span.token_start && *end == span.token_end);
                let should_teach = explicitly_chosen_span
                    || rejected_group_match
                    || selected_component_choice.is_some();
                if should_teach {
                    taught_span = true;
                }

                let chosen_alias_id = selected_component_choice
                    .and_then(|choice| {
                        if choice.choose_keep_original {
                            None
                        } else {
                            choice.chosen_alias_id
                        }
                    })
                    .or_else(|| {
                        teach.as_ref().and_then(|teach| {
                            if explicitly_chosen_span && !teach.choose_keep_original {
                                teach.chosen_alias_id
                            } else {
                                None
                            }
                        })
                    });
                let span_ctx =
                    extract_span_context(&request.transcript, span.char_start, span.char_end);
                let judge_options = if should_teach {
                    judge.teach_choice(&span, &judge_input, chosen_alias_id, &span_ctx)
                } else {
                    judge.score_candidates(&span, &judge_input, &span_ctx)
                };

                let candidates = scored_rows
                    .iter()
                    .map(|scored| {
                        let alias = &self.inner.index.aliases[scored.alias_id as usize];
                        RetrievalCandidateDebug {
                            alias_id: scored.alias_id,
                            term: scored.term.clone(),
                            alias_text: scored.alias_text.clone(),
                            alias_source: map_alias_source(scored.alias_source),
                            alias_ipa_tokens: alias.ipa_tokens.clone(),
                            alias_reduced_ipa_tokens: alias.reduced_ipa_tokens.clone(),
                            alias_feature_tokens: alias.feature_tokens.clone(),
                            identifier_flags: map_identifier_flags(&alias.identifier_flags),
                            features: map_candidate_features(scored),
                            filter_decisions: vec![
                                FilterDecision {
                                    name: "feature_gate".to_string(),
                                    passed: scored.feature_gate_token_ok
                                        && scored.feature_gate_coarse_ok
                                        && scored.feature_gate_phone_ok,
                                    detail: format!(
                                        "token_ok={} coarse_ok={} phone_ok={}",
                                        scored.feature_gate_token_ok,
                                        scored.feature_gate_coarse_ok,
                                        scored.feature_gate_phone_ok
                                    ),
                                },
                                FilterDecision {
                                    name: "short_guard".to_string(),
                                    passed: scored.short_guard_passed,
                                    detail: format!(
                                        "applied={} onset_match={}",
                                        scored.short_guard_applied, scored.short_guard_onset_match
                                    ),
                                },
                                FilterDecision {
                                    name: "low_content_guard".to_string(),
                                    passed: scored.low_content_guard_passed,
                                    detail: format!("applied={}", scored.low_content_guard_applied),
                                },
                                FilterDecision {
                                    name: "acceptance_floor".to_string(),
                                    passed: scored.acceptance_floor_passed,
                                    detail: format!(
                                        "accept={:.3} phonetic={:.3} coarse={:.3}",
                                        scored.acceptance_score,
                                        scored.phonetic_score,
                                        scored.coarse_score
                                    ),
                                },
                                FilterDecision {
                                    name: "verified".to_string(),
                                    passed: scored.verified,
                                    detail: if scored.verified {
                                        "candidate survives verifier".to_string()
                                    } else {
                                        "candidate rejected by verifier".to_string()
                                    },
                                },
                            ],
                            reached_reranker: false,
                            accepted: scored.verified,
                        }
                    })
                    .collect::<Vec<_>>();

                SpanDebugTrace {
                    span: SpanDebugView {
                        token_start: span.token_start as u32,
                        token_end: span.token_end as u32,
                        char_start: span.char_start as u32,
                        char_end: span.char_end as u32,
                        start_sec: span.start_sec.unwrap_or(0.0),
                        end_sec: span.end_sec.unwrap_or(0.0),
                        text: span.text,
                        feature_tokens: bee_phonetic::feature_tokens_for_ipa(&span.ipa_tokens),
                        ipa_tokens: span.ipa_tokens,
                        reduced_ipa_tokens: span.reduced_ipa_tokens,
                    },
                    candidates,
                    judge_options: judge_options
                        .into_iter()
                        .map(|option| JudgeOptionDebug {
                            alias_id: option.alias_id,
                            term: option.term,
                            is_keep_original: option.is_keep_original,
                            score: option.score,
                            probability: option.probability,
                            chosen: option.chosen,
                        })
                        .collect(),
                }
            })
            .collect::<Vec<_>>();

        if let Some(teach) = &teach {
            let sentence_level_keep = teach.choose_keep_original
                && !teach.reject_group
                && teach.selected_component_choices.is_empty();
            if !taught_span && !sentence_level_keep {
                return Err("requested span was not present in the probe result".to_string());
            }
            if teach.reject_group && rejected_group_spans.is_empty() {
                return Err("reject_group requested without any rejected_group_spans".to_string());
            }
            if !teach.selected_component_choices.is_empty() && !teach.reject_group && !taught_span {
                return Err(
                    "selected_component_choices did not match any spans in the probe result"
                        .to_string(),
                );
            }

            // Persist correction events to disk
            if let Err(e) = save_correction_events(&self.inner.event_log_path, judge.event_log()) {
                tracing::warn!(error = %e, "failed to save correction events");
            }
        }

        let rapid_fire = expected_source_text
            .as_deref()
            .map(|expected| build_rapid_fire_decision_set(&request.transcript, &traces, expected));

        Ok(RetrievalPrototypeProbeResult {
            transcript: request.transcript,
            spans: traces,
            timings: TimingBreakdown {
                span_enumeration_ms: 0,
                retrieval_ms: 0,
                verify_ms: 0,
                rerank_ms: 0,
                total_ms: 0,
            },
            judge_state: JudgeStateDebug {
                update_count: judge.update_count(),
                learning_rate: 0.25,
                feature_names: judge.feature_names(),
                weights: judge.weights(),
            },
            rapid_fire,
        })
    }

    fn zipa_wav_path(&self, row: &bee_phonetic::RecordingExampleRow) -> Result<PathBuf, String> {
        let stem = PathBuf::from(&row.audio_path)
            .file_stem()
            .ok_or_else(|| format!("audio path has no file stem: {}", row.audio_path))?
            .to_string_lossy()
            .into_owned();
        let wav_path = self.inner.zipa_wav_dir.join(format!("{stem}.wav"));
        if !wav_path.is_file() {
            return Err(format!(
                "ZIPA WAV mirror is missing {} for {}",
                wav_path.display(),
                row.audio_path
            ));
        }
        Ok(wav_path)
    }

    pub(crate) fn teaching_cases(
        &self,
        limit: usize,
        include_counterexamples: bool,
    ) -> Vec<EvalCase> {
        let mut cases = self
            .inner
            .dataset
            .recording_examples
            .iter()
            .enumerate()
            .map(|(index, row)| EvalCase {
                case_id: format!("canonical:{index}"),
                suite: "canonical",
                target_term: row.term.clone(),
                source_text: row.text.clone(),
                transcript: row.transcript.clone(),
                should_abstain: false,
                take: Some(row.take),
                audio_path: Some(row.audio_path.clone()),
                surface_form: None,
                words: row
                    .words
                    .iter()
                    .map(|w| AlignedWord {
                        word: w.word.clone(),
                        start: w.start,
                        end: w.end,
                        confidence: w.confidence.clone(),
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();

        if include_counterexamples {
            cases.extend(
                self.inner
                    .counterexamples
                    .iter()
                    .enumerate()
                    .map(|(index, row)| EvalCase {
                        case_id: format!("counterexample:{index}"),
                        suite: "counterexample",
                        target_term: row.term.clone(),
                        source_text: row.text.clone(),
                        transcript: row.transcript.clone(),
                        should_abstain: true,
                        take: Some(row.take),
                        audio_path: Some(row.audio_path.clone()),
                        surface_form: Some(row.surface_form.clone()),
                        words: row
                            .words
                            .iter()
                            .map(|w| AlignedWord {
                                word: w.word.clone(),
                                start: w.start,
                                end: w.end,
                                confidence: w.confidence.clone(),
                            })
                            .collect(),
                    }),
            );
        }

        let random_seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos() as u64)
            .unwrap_or(0);
        cases.sort_by_key(|case| randomish_case_key(case, random_seed));

        if limit == 0 || limit >= cases.len() {
            cases
        } else {
            cases.into_iter().take(limit).collect()
        }
    }

    /// Extract spans + candidates for a case without involving the judge.
    /// Used by offline eval to separate feature extraction from judge training.
    pub(crate) fn probe_case_spans(
        &self,
        case: &EvalCase,
        max_span_words: u8,
        shortlist_limit: u16,
    ) -> Result<ProbedCase, String> {
        let alignments = if case.words.is_empty() {
            None
        } else {
            Some(
                case.words
                    .iter()
                    .map(|word| TranscriptAlignmentToken {
                        start_time: word.start,
                        end_time: word.end,
                        confidence: word.confidence.clone(),
                    })
                    .collect::<Vec<_>>(),
            )
        };

        let spans = {
            let mut g2p = self
                .inner
                .g2p
                .lock()
                .map_err(|_| "g2p cache mutex poisoned".to_string())?;
            enumerate_transcript_spans_with(
                &case.transcript,
                max_span_words as usize,
                alignments.as_deref(),
                |text| g2p.ipa_tokens(text).ok().flatten(),
            )
        };

        let probed_spans: Vec<ProbedSpan> = spans
            .into_iter()
            .map(|span| {
                let shortlist = query_index(
                    &self.inner.index,
                    &RetrievalQuery {
                        text: span.text.clone(),
                        ipa_tokens: span.ipa_tokens.clone(),
                        reduced_ipa_tokens: span.reduced_ipa_tokens.clone(),
                        feature_tokens: bee_phonetic::feature_tokens_for_ipa(&span.ipa_tokens),
                        token_count: (span.token_end - span.token_start) as u8,
                    },
                    shortlist_limit as usize,
                );
                let scored_rows = score_shortlist(&span, &shortlist, &self.inner.index);
                let candidates: Vec<_> = scored_rows
                    .iter()
                    .map(|row| {
                        let alias = &self.inner.index.aliases[row.alias_id as usize];
                        (row.clone(), alias.identifier_flags.clone())
                    })
                    .collect();
                let ctx = extract_span_context(&case.transcript, span.char_start, span.char_end);

                // Find gold alias_id: the best-scoring alias for the target term
                let gold_alias_id = if !case.should_abstain {
                    candidates
                        .iter()
                        .find(|(c, _)| c.term.eq_ignore_ascii_case(&case.target_term))
                        .map(|(c, _)| c.alias_id)
                } else {
                    None
                };

                ProbedSpan {
                    span,
                    candidates,
                    ctx,
                    gold_alias_id,
                }
            })
            .collect();

        Ok(ProbedCase {
            case: case.clone(),
            spans: probed_spans,
        })
    }

    pub(crate) fn evaluate_case(
        &self,
        case: &EvalCase,
        request: &RetrievalPrototypeEvalRequest,
    ) -> Result<CaseEvalResult, String> {
        // Use the same pipeline as run_probe — one path for everything.
        // Pass expected_source_text so rapid fire decision set is built with is_gold flags.
        let probe_result = self.run_probe(
            RetrievalPrototypeProbeRequest {
                transcript: case.transcript.clone(),
                words: case.words.clone(),
                max_span_words: request.max_span_words,
                shortlist_limit: request.shortlist_limit,
                verify_limit: request.verify_limit,
                expected_source_text: Some(case.source_text.clone()),
            },
            None,
        )?;

        // Stage 1: Retrieval — did any span's candidate list contain the target term?
        let mut target_best_rank: Option<usize> = None;
        for span_trace in &probe_result.spans {
            if let Some(rank) = span_trace
                .candidates
                .iter()
                .enumerate()
                .find(|(_, c)| c.term.eq_ignore_ascii_case(&case.target_term))
                .map(|(i, _)| i + 1)
            {
                match target_best_rank {
                    Some(existing) if existing <= rank => {}
                    _ => {
                        target_best_rank = Some(rank);
                    }
                }
            }
        }
        let target_in_shortlist = target_best_rank.is_some();

        // Stage 2: Composition — is the gold sentence in the judge-visible decision set?
        let rapid_fire = probe_result.rapid_fire.as_ref();
        let choices = rapid_fire.map(|rf| &rf.choices[..]).unwrap_or(&[]);
        let decision_set_size = choices.len();
        let replacement_choice_count = choices.iter().filter(|c| !c.choose_keep_original).count();
        let gold_choice_index = choices.iter().position(|c| c.is_gold);
        let gold_reachable = gold_choice_index.is_some();

        // Stage 3: Judge — what would the judge pick?
        // Build a lookup of judge probabilities per (span_start, span_end, alias_id)
        // and per (span_start, span_end) for keep_original, then score each sentence
        // choice by the judge's actual probabilities.
        let mut judge_edit_prob: HashMap<(u32, u32, u32), f32> = HashMap::new();
        let mut judge_keep_prob: HashMap<(u32, u32), f32> = HashMap::new();
        for span_trace in &probe_result.spans {
            let ts = span_trace.span.token_start;
            let te = span_trace.span.token_end;
            for opt in &span_trace.judge_options {
                if opt.is_keep_original {
                    judge_keep_prob.insert((ts, te), opt.probability);
                } else if let Some(alias_id) = opt.alias_id {
                    judge_edit_prob.insert((ts, te, alias_id), opt.probability);
                }
            }
        }

        // Score each choice by the judge: for each component, look up the judge
        // probability for the chosen action (keep or specific edit). The sentence
        // score is the min across components (weakest link).
        // For keep_original sentences that have no component_choices (synthesized
        // fallback), use the min keep probability across all spans with edit candidates.
        let judge_score_for_choice = |choice: &RapidFireChoice| -> f32 {
            if choice.component_choices.is_empty() {
                // Synthesized keep_original with no components — use min keep prob
                // across all spans that have edit candidates
                if judge_keep_prob.is_empty() {
                    return 1.0; // no spans with candidates, keeping is trivially correct
                }
                return judge_keep_prob.values().copied().fold(f32::MAX, f32::min);
            }
            let mut min_prob = f32::MAX;
            for cc in &choice.component_choices {
                if cc.choose_keep_original {
                    // This component keeps original — use keep prob for its spans
                    let prob = cc
                        .component_spans
                        .iter()
                        .filter_map(|s| judge_keep_prob.get(&(s.token_start, s.token_end)))
                        .copied()
                        .fold(0.0f32, f32::max);
                    min_prob = min_prob.min(prob);
                } else if let (Some(start), Some(end), Some(alias_id)) =
                    (cc.span_token_start, cc.span_token_end, cc.chosen_alias_id)
                {
                    let prob = judge_edit_prob
                        .get(&(start, end, alias_id))
                        .copied()
                        .unwrap_or(0.0);
                    min_prob = min_prob.min(prob);
                }
            }
            if min_prob == f32::MAX { 0.0 } else { min_prob }
        };

        // Log judge scores for the first few cases to debug
        if !case.should_abstain {
            let gold = choices.iter().find(|c| c.is_gold);
            let keep = choices.iter().find(|c| c.choose_keep_original);
            tracing::debug!(
                case_id = %case.case_id,
                target = %case.target_term,
                gold_score = gold.map(|c| judge_score_for_choice(c)),
                keep_score = keep.map(|c| judge_score_for_choice(c)),
                gold_reachable,
                num_choices = choices.len(),
                "judge eval debug"
            );
        }

        let judge_pick = choices
            .iter()
            .max_by(|a, b| judge_score_for_choice(a).total_cmp(&judge_score_for_choice(b)));
        let (chosen_kind, chosen_choice_id, chosen_sentence, chosen_edit_count, chosen_probability) =
            if let Some(pick) = judge_pick {
                (
                    if pick.choose_keep_original {
                        EvalChoiceKind::KeepOriginal
                    } else {
                        EvalChoiceKind::SentenceCandidate
                    },
                    Some(pick.option_id.clone()),
                    pick.sentence.clone(),
                    pick.edits.len(),
                    judge_score_for_choice(pick),
                )
            } else {
                (
                    EvalChoiceKind::KeepOriginal,
                    None,
                    case.transcript.clone(),
                    0,
                    0.0,
                )
            };

        let judge_correct = if case.should_abstain {
            matches!(chosen_kind, EvalChoiceKind::KeepOriginal)
        } else {
            judge_pick.is_some_and(|p| p.is_gold)
        };

        // Why is gold unreachable? (canonical cases only)
        let gold_unreachable_reason = if gold_reachable || case.should_abstain {
            None
        } else if !target_in_shortlist {
            Some(GoldUnreachableReason::TargetNotRetrieved)
        } else {
            // Target is in shortlist. Check if any choice contains the target term.
            let any_choice_has_target = choices.iter().any(|c| {
                !c.choose_keep_original
                    && c.edits
                        .iter()
                        .any(|e| e.replacement_text.eq_ignore_ascii_case(&case.target_term))
            });
            if !any_choice_has_target {
                // Target was retrieved but never made it into any composed sentence.
                // Either other required edits were missing, or composition pruned it.
                Some(GoldUnreachableReason::MissingRequiredEdits)
            } else {
                // Target term IS in some choices, but none match the gold sentence.
                // This means the edits were applied but the surface form doesn't match.
                let closest = choices
                    .iter()
                    .filter(|c| !c.choose_keep_original)
                    .filter(|c| {
                        c.edits
                            .iter()
                            .any(|e| e.replacement_text.eq_ignore_ascii_case(&case.target_term))
                    })
                    .min_by_key(|c| {
                        // Simple word-level edit distance to gold
                        let got_words: Vec<&str> = c.sentence.split_whitespace().collect();
                        let want_words: Vec<&str> = case.source_text.split_whitespace().collect();
                        let diff = got_words.len().abs_diff(want_words.len())
                            + got_words
                                .iter()
                                .zip(&want_words)
                                .filter(|(a, b)| a != b)
                                .count();
                        diff
                    });
                Some(GoldUnreachableReason::SurfaceMismatch {
                    closest_sentence: closest.map(|c| c.sentence.clone()).unwrap_or_default(),
                })
            }
        };

        // Attribution: which stage failed first? (canonical cases only)
        let first_failure = if case.should_abstain {
            if !judge_correct {
                Some(EvalFailureStage::Judge)
            } else {
                None
            }
        } else if !target_in_shortlist {
            Some(EvalFailureStage::RetrievalShortlist)
        } else if !gold_reachable {
            Some(EvalFailureStage::Composition)
        } else if !judge_correct {
            Some(EvalFailureStage::Judge)
        } else {
            None
        };

        Ok(CaseEvalResult {
            target_in_shortlist,
            target_best_rank,
            gold_reachable,
            gold_choice_rank: gold_choice_index,
            gold_unreachable_reason,
            decision_set_size,
            replacement_choice_count,
            chosen_kind,
            chosen_choice_id,
            chosen_sentence,
            chosen_edit_count,
            chosen_probability,
            judge_correct,
            first_failure,
        })
    }
}

fn kokoro_model_path() -> Result<PathBuf, String> {
    let path = std::env::var("BEE_KOKORO_MODEL_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("bearcove/whisper/kokoro-v1.0.onnx")
        });
    if path.is_file() {
        Ok(path)
    } else {
        Err(format!(
            "kokoro ONNX model not found at {}. Set BEE_KOKORO_MODEL_PATH to kokoro-v1.0.onnx.",
            path.display()
        ))
    }
}

fn kokoro_voices_path() -> Result<PathBuf, String> {
    let path = std::env::var("BEE_KOKORO_VOICES_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("bearcove/whisper/voices-v1.0.bin")
        });
    if path.is_file() {
        Ok(path)
    } else {
        Err(format!(
            "kokoro voices not found at {}. Set BEE_KOKORO_VOICES_PATH.",
            path.display()
        ))
    }
}

fn kokoro_site_packages_path() -> Result<String, String> {
    if let Ok(path) = std::env::var("BEE_KOKORO_PYTHONPATH") {
        return Ok(path);
    }

    let lib_root = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".local/share/uv/tools/kokoro-tts/lib");
    let entries =
        fs::read_dir(&lib_root).map_err(|e| format!("reading {}: {e}", lib_root.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| e.to_string())?;
        let site_packages = entry.path().join("site-packages");
        if site_packages.join("kokoro_onnx").exists() {
            return Ok(site_packages.display().to_string());
        }
    }
    Err(format!(
        "could not find kokoro_onnx under {}. Set BEE_KOKORO_PYTHONPATH.",
        lib_root.display()
    ))
}

fn start_kokoro_sidecar(
    model_path: &PathBuf,
    voices_path: &PathBuf,
    pythonpath: &str,
) -> Result<KokoroSidecar, String> {
    let script_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts/kokoro_phonemes.py");
    let mut child = Command::new("python3")
        .env("PYTHONPATH", pythonpath)
        .arg(script_path)
        .arg("--server")
        .arg("--model")
        .arg(model_path)
        .arg("--voices")
        .arg(voices_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("starting kokoro sidecar: {e}"))?;

    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| "kokoro sidecar stdin unavailable".to_string())?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "kokoro sidecar stdout unavailable".to_string())?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| "kokoro sidecar stderr unavailable".to_string())?;

    Ok(KokoroSidecar {
        child,
        stdin,
        stdout: BufReader::new(stdout),
        stderr: BufReader::new(stderr),
    })
}

fn mean_option(values: impl Iterator<Item = Option<f32>>) -> Option<f32> {
    let mut total = 0.0f32;
    let mut count = 0usize;
    for value in values.flatten() {
        total += value;
        count += 1;
    }
    (count > 0).then_some(total / count as f32)
}

fn collect_asr_alternatives(snapshots: &[SessionSnapshot]) -> Vec<TranscribeAsrObservedToken> {
    #[derive(Clone)]
    struct AggregatedToken {
        chosen_text: String,
        concentration: f32,
        margin: f32,
        revision: u64,
        alternatives: HashMap<u32, (String, f32)>,
    }

    let mut by_index = std::collections::BTreeMap::<u32, AggregatedToken>::new();

    for snapshot in snapshots {
        for (offset, token) in snapshot.pending_tokens.iter().enumerate() {
            let token_index = snapshot.committed_token_count + offset as u32;
            let entry = by_index
                .entry(token_index)
                .or_insert_with(|| AggregatedToken {
                    chosen_text: token.text.clone(),
                    concentration: token.concentration,
                    margin: token.margin,
                    revision: snapshot.revision,
                    alternatives: HashMap::new(),
                });

            if token.margin < entry.margin
                || (token.margin == entry.margin && token.concentration < entry.concentration)
                || snapshot.revision > entry.revision
            {
                entry.chosen_text = token.text.clone();
                entry.concentration = token.concentration;
                entry.margin = token.margin;
                entry.revision = snapshot.revision;
            }

            for alternative in &token.alternatives {
                entry
                    .alternatives
                    .entry(alternative.token_id)
                    .and_modify(|(_, best_logit)| {
                        if alternative.logit > *best_logit {
                            *best_logit = alternative.logit;
                        }
                    })
                    .or_insert((alternative.text.clone(), alternative.logit));
            }
        }
    }

    by_index
        .into_iter()
        .map(|(token_index, token)| {
            let mut alternatives = token
                .alternatives
                .into_iter()
                .map(|(token_id, (text, logit))| TranscribeAsrTokenAlternative {
                    token_id,
                    text,
                    logit,
                })
                .collect::<Vec<_>>();
            alternatives.sort_by(|a, b| {
                b.logit
                    .total_cmp(&a.logit)
                    .then_with(|| a.token_id.cmp(&b.token_id))
            });

            TranscribeAsrObservedToken {
                token_index,
                chosen_text: token.chosen_text,
                concentration: token.concentration,
                margin: token.margin,
                revision: token.revision,
                alternatives,
            }
        })
        .collect()
}

fn unix_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn transcript_word_raw_ranges(
    g2p: &mut CachedEspeakG2p,
    transcript: &str,
) -> Result<Vec<(std::ops::Range<usize>, Vec<String>)>, String> {
    let words = sentence_word_tokens(transcript);
    let raw_groups = g2p
        .ipa_word_tokens_in_utterance(transcript)
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("espeak produced no tokens for '{transcript}'"))?;

    if raw_groups.len() != words.len() {
        return Err(format!(
            "espeak word-group count mismatch for '{transcript}': transcript has {} words, espeak produced {} groups",
            words.len(),
            raw_groups.len()
        ));
    }

    Ok(words
        .into_iter()
        .zip(raw_groups)
        .map(|(word, raw_tokens)| (word.char_start..word.char_end, raw_tokens))
        .collect())
}

fn transcript_token_range_for_span(
    word_ranges: &[(std::ops::Range<usize>, Vec<String>)],
    token_start: usize,
    token_end: usize,
) -> std::ops::Range<usize> {
    let start = word_ranges
        .iter()
        .take(token_start)
        .map(|(_, tokens)| tokens.len())
        .sum::<usize>();
    let len = word_ranges[token_start..token_end]
        .iter()
        .map(|(_, tokens)| tokens.len())
        .sum::<usize>();
    start..(start + len)
}

fn transcript_normalized_for_span(
    word_ranges: &[(std::ops::Range<usize>, Vec<String>)],
    token_start: usize,
    token_end: usize,
) -> Vec<String> {
    word_ranges[token_start..token_end]
        .iter()
        .flat_map(|(_, tokens)| tokens.iter().cloned())
        .collect()
}

fn raw_slice_for_normalized_range(
    raw_tokens: &[String],
    normalized: &[bee_phonetic::ComparisonToken],
    normalized_range: std::ops::Range<usize>,
) -> Vec<String> {
    let Some(first) = normalized.get(normalized_range.start) else {
        return Vec::new();
    };
    let Some(last) = normalized.get(normalized_range.end.saturating_sub(1)) else {
        return Vec::new();
    };
    raw_tokens
        .get(first.source_start..last.source_end)
        .unwrap_or(&[])
        .to_vec()
}

fn select_span_alignment_range(
    transcript_normalized: &[String],
    utterance_zipa_normalized: &[String],
    projected_range: Option<std::ops::Range<usize>>,
) -> Option<SpanAlignmentSelection> {
    let mut candidate_ranges = Vec::<std::ops::Range<usize>>::new();
    if let Some(range) = &projected_range {
        candidate_ranges.push(range.clone());
    }
    candidate_ranges.extend(
        top_right_anchor_windows(transcript_normalized, utterance_zipa_normalized, 3)
            .into_iter()
            .map(|window| window.right_start as usize..window.right_end as usize),
    );
    normalize_candidate_ranges(&mut candidate_ranges, utterance_zipa_normalized.len());

    let mut scored = Vec::new();
    for (candidate_index, range) in candidate_ranges.into_iter().enumerate() {
        let zipa_normalized = utterance_zipa_normalized
            .get(range.clone())
            .unwrap_or(&[])
            .to_vec();
        if zipa_normalized.is_empty() {
            continue;
        }
        let alignment = align_token_sequences(transcript_normalized, &zipa_normalized);
        let score = alignment_quality_score(
            &alignment.ops,
            transcript_normalized.len(),
            zipa_normalized.len(),
        );
        scored.push((candidate_index, range, score, alignment, zipa_normalized));
    }

    scored.sort_by(|a, b| b.2.total_cmp(&a.2));
    let projected_alignment_score = scored
        .iter()
        .find(|(candidate_index, _, _, _, _)| projected_range.is_some() && *candidate_index == 0)
        .map(|(_, _, score, _, _)| *score);
    let second_best_alignment_score = scored.get(1).map(|(_, _, score, _, _)| *score);
    let (candidate_index, range, chosen_alignment_score, alignment, zipa_normalized) =
        scored.into_iter().next()?;
    let alignment_score_gap =
        second_best_alignment_score.map(|score| chosen_alignment_score - score);
    let alignment_source = if projected_range.is_some() && candidate_index == 0 {
        "projected"
    } else {
        "anchored"
    };

    Some(SpanAlignmentSelection {
        range,
        alignment,
        zipa_normalized,
        projected_alignment_score,
        chosen_alignment_score,
        second_best_alignment_score,
        alignment_score_gap,
        alignment_source,
    })
}

fn select_segmental_word_windows(
    transcript_word_tokens: &[Vec<String>],
    transcript_token_ranges: &[std::ops::Range<usize>],
    utterance_zipa_normalized: &[String],
    utterance_alignment: &bee_phonetic::TokenAlignment,
) -> Vec<Option<WordAlignmentWindow>> {
    if transcript_word_tokens.is_empty() {
        return Vec::new();
    }

    let candidates_per_word = transcript_word_tokens
        .iter()
        .enumerate()
        .map(|(word_index, transcript_tokens)| {
            build_word_segment_candidates(
                transcript_tokens,
                transcript_token_ranges.get(word_index).cloned(),
                utterance_zipa_normalized,
                utterance_alignment,
            )
        })
        .collect::<Vec<_>>();

    if candidates_per_word
        .iter()
        .any(|candidates| candidates.is_empty())
    {
        return std::iter::repeat_with(|| None)
            .take(transcript_word_tokens.len())
            .collect();
    }

    let mut dp = candidates_per_word
        .iter()
        .map(|candidates| vec![f32::NEG_INFINITY; candidates.len()])
        .collect::<Vec<_>>();
    let mut backpointers = candidates_per_word
        .iter()
        .map(|candidates| vec![None; candidates.len()])
        .collect::<Vec<_>>();

    for (candidate_index, candidate) in candidates_per_word[0].iter().enumerate() {
        dp[0][candidate_index] = candidate.local_score
            - gap_penalty(&utterance_zipa_normalized[..candidate.zipa_norm_range.start]);
    }

    for word_index in 1..candidates_per_word.len() {
        for (candidate_index, candidate) in candidates_per_word[word_index].iter().enumerate() {
            let mut best_score = f32::NEG_INFINITY;
            let mut best_prev = None;
            for (prev_index, prev) in candidates_per_word[word_index - 1].iter().enumerate() {
                if prev.zipa_norm_range.end > candidate.zipa_norm_range.start {
                    continue;
                }
                let transition_score = dp[word_index - 1][prev_index]
                    - boundary_gap_penalty(
                        &utterance_zipa_normalized
                            [prev.zipa_norm_range.end..candidate.zipa_norm_range.start],
                        prev,
                        candidate,
                        transcript_word_tokens[word_index - 1].as_slice(),
                        transcript_word_tokens[word_index].as_slice(),
                    )
                    + candidate.local_score;
                if transition_score > best_score {
                    best_score = transition_score;
                    best_prev = Some(prev_index);
                }
            }
            dp[word_index][candidate_index] = best_score;
            backpointers[word_index][candidate_index] = best_prev;
        }
    }

    let last_word_index = candidates_per_word.len() - 1;
    let mut best_last = None;
    let mut best_last_score = f32::NEG_INFINITY;
    for (candidate_index, candidate) in candidates_per_word[last_word_index].iter().enumerate() {
        let total_score = dp[last_word_index][candidate_index]
            - gap_penalty(&utterance_zipa_normalized[candidate.zipa_norm_range.end..]);
        if total_score > best_last_score {
            best_last_score = total_score;
            best_last = Some(candidate_index);
        }
    }

    let Some(mut candidate_index) = best_last else {
        return std::iter::repeat_with(|| None)
            .take(transcript_word_tokens.len())
            .collect();
    };

    let mut chosen = vec![None; transcript_word_tokens.len()];
    for word_index in (0..candidates_per_word.len()).rev() {
        let candidate = candidates_per_word[word_index][candidate_index].clone();
        chosen[word_index] = Some(WordAlignmentWindow {
            zipa_norm_range: candidate.zipa_norm_range,
            ops: candidate.alignment.ops,
        });
        if let Some(prev_index) = backpointers[word_index][candidate_index] {
            candidate_index = prev_index;
        } else if word_index != 0 {
            break;
        }
    }

    chosen
}

fn build_word_segment_candidates(
    transcript_tokens: &[String],
    transcript_token_range: Option<std::ops::Range<usize>>,
    utterance_zipa_normalized: &[String],
    utterance_alignment: &bee_phonetic::TokenAlignment,
) -> Vec<WordSegmentCandidate> {
    if transcript_tokens.is_empty() || utterance_zipa_normalized.is_empty() {
        return Vec::new();
    }

    let mut candidate_ranges = Vec::<std::ops::Range<usize>>::new();
    if let Some(transcript_token_range) = transcript_token_range {
        if let Some(projected_range) =
            utterance_alignment.project_left_range(transcript_token_range)
        {
            candidate_ranges.push(projected_range);
        }
    }
    candidate_ranges.extend(
        top_right_anchor_windows(transcript_tokens, utterance_zipa_normalized, 4)
            .into_iter()
            .map(|window| window.right_start as usize..window.right_end as usize),
    );
    let expanded = candidate_ranges
        .iter()
        .flat_map(|range| {
            let mut variants = vec![range.clone()];
            for shift in 1..=3 {
                variants.push(range.start.saturating_sub(shift)..range.end);
                variants
                    .push(range.start..(range.end + shift).min(utterance_zipa_normalized.len()));
                variants.push(
                    range.start.saturating_sub(shift)
                        ..(range.end + shift).min(utterance_zipa_normalized.len()),
                );
                if range.start + shift < range.end {
                    variants.push((range.start + shift)..range.end);
                }
                if range.end > range.start + shift {
                    variants.push(range.start..(range.end - shift));
                }
                if range.start + shift < range.end && range.end > range.start + shift {
                    variants.push(
                        (range.start + shift)..(range.end - shift).max(range.start + shift + 1),
                    );
                }
            }
            variants
        })
        .collect::<Vec<_>>();
    candidate_ranges.extend(expanded);
    dedup_candidate_ranges(&mut candidate_ranges, utterance_zipa_normalized.len());

    candidate_ranges
        .into_iter()
        .filter_map(|range| {
            let zipa_normalized = utterance_zipa_normalized.get(range.clone())?.to_vec();
            if zipa_normalized.is_empty() {
                return None;
            }
            let alignment = align_token_sequences(transcript_tokens, &zipa_normalized);
            let local_score =
                segment_local_alignment_score(transcript_tokens, &zipa_normalized, &alignment.ops);
            Some(WordSegmentCandidate {
                zipa_norm_range: range,
                zipa_normalized,
                alignment,
                local_score,
            })
        })
        .collect()
}

fn segment_local_alignment_score(
    transcript_tokens: &[String],
    zipa_tokens: &[String],
    ops: &[AlignmentOp],
) -> f32 {
    let mut score = alignment_quality_score(ops, transcript_tokens.len(), zipa_tokens.len());
    let leading_deletes = ops
        .iter()
        .take_while(|op| op.left_index.is_some() && op.right_index.is_none())
        .count();
    let trailing_deletes = ops
        .iter()
        .rev()
        .take_while(|op| op.left_index.is_some() && op.right_index.is_none())
        .count();
    let leading_inserts = ops
        .iter()
        .take_while(|op| op.left_index.is_none() && op.right_index.is_some())
        .count();
    let trailing_inserts = ops
        .iter()
        .rev()
        .take_while(|op| op.left_index.is_none() && op.right_index.is_some())
        .count();
    score -= leading_deletes as f32 * 0.22;
    score -= trailing_deletes as f32 * 0.22;
    score -= insert_run_penalty(
        ops.iter()
            .take(leading_inserts)
            .filter_map(|op| op.right_token.as_deref()),
    );
    score -= insert_run_penalty(
        ops.iter()
            .rev()
            .take(trailing_inserts)
            .filter_map(|op| op.right_token.as_deref()),
    );

    if let Some(first) = transcript_tokens.first() {
        if let Some(first_right) = ops.iter().find_map(|op| op.right_token.as_deref()) {
            score += token_affinity(first, first_right) * 0.18;
        }
    }
    if let Some(last) = transcript_tokens.last() {
        if let Some(last_right) = ops.iter().rev().find_map(|op| op.right_token.as_deref()) {
            score += token_affinity(last, last_right) * 0.18;
        }
    }

    if let Some((first_left, first_right)) = first_aligned_pair(ops) {
        score += token_affinity(first_left, first_right) * 0.15;
    }
    if let Some((last_left, last_right)) = last_aligned_pair(ops) {
        score += token_affinity(last_left, last_right) * 0.15;
    }

    score
}

fn boundary_gap_penalty(
    gap_tokens: &[String],
    prev: &WordSegmentCandidate,
    next: &WordSegmentCandidate,
    prev_transcript: &[String],
    next_transcript: &[String],
) -> f32 {
    let mut penalty = gap_penalty(gap_tokens) * 1.35;
    if let (Some(last_gap), Some(prev_last)) = (gap_tokens.last(), prev_transcript.last()) {
        penalty += token_affinity(last_gap, prev_last) * 0.45;
    }
    if let (Some(first_gap), Some(next_first)) = (gap_tokens.first(), next_transcript.first()) {
        penalty += token_affinity(first_gap, next_first) * 0.75;
    }
    if gap_tokens.len() >= 2 {
        let prefix_affinity = gap_prefix_affinity(gap_tokens, next_transcript);
        let suffix_affinity = gap_suffix_affinity(gap_tokens, prev_transcript);
        penalty += prefix_affinity * 0.55;
        penalty += suffix_affinity * 0.35;
    }
    penalty += range_compression_penalty(prev, next);
    penalty
}

fn range_compression_penalty(prev: &WordSegmentCandidate, next: &WordSegmentCandidate) -> f32 {
    let gap = next
        .zipa_norm_range
        .start
        .saturating_sub(prev.zipa_norm_range.end);
    if gap == 0 { 0.0 } else { (gap as f32) * 0.03 }
}

fn gap_penalty(tokens: &[String]) -> f32 {
    tokens
        .iter()
        .map(|token| {
            if is_weak_vowelish(token) {
                0.12
            } else if is_vowelish(token) {
                0.25
            } else {
                0.55
            }
        })
        .sum()
}

fn insert_run_penalty<'a>(tokens: impl Iterator<Item = &'a str>) -> f32 {
    tokens
        .map(|token| {
            if is_weak_vowelish(token) {
                0.08
            } else if is_vowelish(token) {
                0.22
            } else {
                0.5
            }
        })
        .sum()
}

fn token_affinity(left: &str, right: &str) -> f32 {
    feature_similarity(&[left.to_string()], &[right.to_string()])
        .or_else(|| phoneme_similarity(&[left.to_string()], &[right.to_string()]))
        .unwrap_or(0.0)
        .max(0.0)
}

fn first_aligned_pair<'a>(ops: &'a [AlignmentOp]) -> Option<(&'a str, &'a str)> {
    ops.iter()
        .find_map(|op| Some((op.left_token.as_deref()?, op.right_token.as_deref()?)))
}

fn last_aligned_pair<'a>(ops: &'a [AlignmentOp]) -> Option<(&'a str, &'a str)> {
    ops.iter()
        .rev()
        .find_map(|op| Some((op.left_token.as_deref()?, op.right_token.as_deref()?)))
}

fn gap_prefix_affinity(gap_tokens: &[String], next_transcript: &[String]) -> f32 {
    let take = gap_tokens.len().min(next_transcript.len()).min(2);
    if take == 0 {
        return 0.0;
    }
    feature_similarity(&gap_tokens[..take], &next_transcript[..take])
        .or_else(|| phoneme_similarity(&gap_tokens[..take], &next_transcript[..take]))
        .unwrap_or(0.0)
        .max(0.0)
}

fn gap_suffix_affinity(gap_tokens: &[String], prev_transcript: &[String]) -> f32 {
    let take = gap_tokens.len().min(prev_transcript.len()).min(2);
    if take == 0 {
        return 0.0;
    }
    let gap_slice = &gap_tokens[gap_tokens.len() - take..];
    let prev_slice = &prev_transcript[prev_transcript.len() - take..];
    feature_similarity(gap_slice, prev_slice)
        .or_else(|| phoneme_similarity(gap_slice, prev_slice))
        .unwrap_or(0.0)
        .max(0.0)
}

fn is_weak_vowelish(token: &str) -> bool {
    matches!(token, "ə" | "ɚ" | "ɝ")
}

fn is_vowelish(token: &str) -> bool {
    matches!(
        token,
        "a" | "ɑ" | "ɔ" | "ɛ" | "ə" | "ɪ" | "ʊ" | "i" | "u" | "e" | "o" | "æ" | "ʌ" | "ɚ" | "ɝ"
    )
}

fn slice_alignment_ops(
    ops: &[AlignmentOp],
    transcript_range: std::ops::Range<usize>,
    zipa_range: std::ops::Range<usize>,
) -> Vec<AlignmentOp> {
    ops.iter()
        .filter(|op| {
            let left_ok = op
                .left_index
                .map(|index| {
                    transcript_range.start <= index as usize
                        && (index as usize) < transcript_range.end
                })
                .unwrap_or(false);
            let right_ok = op
                .right_index
                .map(|index| {
                    zipa_range.start <= index as usize && (index as usize) < zipa_range.end
                })
                .unwrap_or(false);
            left_ok || right_ok
        })
        .cloned()
        .collect()
}

fn strict_project_left_range(
    ops: &[AlignmentOp],
    left_range: std::ops::Range<usize>,
) -> Option<std::ops::Range<usize>> {
    if left_range.start >= left_range.end {
        return None;
    }

    let mut matched_right = Vec::new();
    let mut right_before = None;
    let mut right_after = None;

    for op in ops {
        let left_index = op.left_index.map(|index| index as usize);
        let right_index = op.right_index.map(|index| index as usize);

        if let Some(left_index) = left_index {
            if left_range.contains(&left_index) {
                if let Some(right_index) = right_index {
                    matched_right.push(right_index);
                }
                continue;
            }

            if left_index < left_range.start {
                if let Some(right_index) = right_index {
                    right_before = Some(right_index);
                }
            } else if left_index >= left_range.end && right_after.is_none() {
                if let Some(right_index) = right_index {
                    right_after = Some(right_index);
                }
            }
        }
    }

    if let (Some(start), Some(end)) = (matched_right.first(), matched_right.last()) {
        return Some(*start..(*end + 1));
    }

    match (right_before, right_after) {
        (Some(start), Some(end)) if start < end => Some((start + 1)..end),
        (Some(start), Some(end)) => Some(start..(end + 1)),
        _ => None,
    }
}

#[derive(Clone)]
struct WordAlignmentWindow {
    zipa_norm_range: std::ops::Range<usize>,
    ops: Vec<AlignmentOp>,
}

fn partition_word_alignment_windows(
    ops: &[AlignmentOp],
    left_ranges: &[std::ops::Range<usize>],
    transcript_word_tokens: &[Vec<String>],
) -> Vec<Option<WordAlignmentWindow>> {
    if left_ranges.is_empty() {
        return Vec::new();
    }

    let footprints = left_ranges
        .iter()
        .map(|left_range| {
            let positions = ops
                .iter()
                .enumerate()
                .filter_map(|(position, op)| {
                    let left_index = op.left_index.map(|index| index as usize)?;
                    left_range.contains(&left_index).then_some(position)
                })
                .collect::<Vec<_>>();
            positions
                .first()
                .zip(positions.last())
                .map(|(first, last)| *first..(*last + 1))
        })
        .collect::<Vec<_>>();

    let mut op_ranges = footprints.clone();

    let existing = footprints
        .iter()
        .enumerate()
        .filter_map(|(word_index, range)| range.as_ref().map(|range| (word_index, range.clone())))
        .collect::<Vec<_>>();

    if existing.is_empty() {
        return std::iter::repeat_with(|| None)
            .take(left_ranges.len())
            .collect();
    }

    let (first_index, first_range) = &existing[0];
    if first_range.start > 0 {
        op_ranges[*first_index] = Some(0..first_range.end);
    }

    let (last_index, last_range) = &existing[existing.len() - 1];
    if last_range.end < ops.len() {
        op_ranges[*last_index] = Some(last_range.start..ops.len());
    }

    for pair in existing.windows(2) {
        let (left_word_index, left_range) = &pair[0];
        let (right_word_index, right_range) = &pair[1];
        let boundary = choose_word_boundary(
            ops,
            left_range.end,
            right_range.start,
            transcript_word_tokens
                .get(*left_word_index)
                .map(Vec::as_slice)
                .unwrap_or(&[]),
            transcript_word_tokens
                .get(*right_word_index)
                .map(Vec::as_slice)
                .unwrap_or(&[]),
        );

        let left_start = op_ranges[*left_word_index]
            .as_ref()
            .map(|range| range.start)
            .unwrap_or(left_range.start);
        let right_end = op_ranges[*right_word_index]
            .as_ref()
            .map(|range| range.end)
            .unwrap_or(right_range.end);

        op_ranges[*left_word_index] = Some(left_start..boundary);
        op_ranges[*right_word_index] = Some(boundary..right_end);
    }

    op_ranges
        .into_iter()
        .map(|op_range| {
            let op_range = op_range?;
            let word_ops = ops.get(op_range)?.to_vec();
            let mut right_indices = word_ops
                .iter()
                .filter_map(|op| op.right_index.map(|index| index as usize))
                .collect::<Vec<_>>();
            if right_indices.is_empty() {
                return None;
            }
            right_indices.sort_unstable();
            let zipa_norm_range = right_indices[0]..(right_indices[right_indices.len() - 1] + 1);
            Some(WordAlignmentWindow {
                zipa_norm_range,
                ops: word_ops,
            })
        })
        .collect()
}

fn choose_word_boundary(
    ops: &[AlignmentOp],
    left_end: usize,
    right_start: usize,
    prev_word_tokens: &[String],
    next_word_tokens: &[String],
) -> usize {
    if left_end >= right_start {
        return right_start;
    }

    let segment_start = left_end;
    let segment = &ops[segment_start..right_start];
    if segment
        .iter()
        .all(|op| op.left_index.is_none() && op.right_index.is_some())
    {
        return segment_start;
    }

    let mut best_split = 0usize;
    let mut best_score = f32::NEG_INFINITY;

    for split in 0..=segment.len() {
        let left_score = boundary_side_affinity(&segment[..split], prev_word_tokens, false);
        let right_score = boundary_side_affinity(&segment[split..], next_word_tokens, true);
        let score = left_score + right_score;
        if score > best_score || (score == best_score && split < best_split) {
            best_score = score;
            best_split = split;
        }
    }

    segment_start + best_split
}

fn boundary_side_affinity(
    segment: &[AlignmentOp],
    neighbor_tokens: &[String],
    prefix: bool,
) -> f32 {
    let right_tokens = segment
        .iter()
        .filter_map(|op| op.right_token.clone())
        .collect::<Vec<_>>();
    if right_tokens.is_empty() || neighbor_tokens.is_empty() {
        return 0.0;
    }

    let take = right_tokens.len().min(neighbor_tokens.len());
    let neighbor_slice = if prefix {
        &neighbor_tokens[..take]
    } else {
        &neighbor_tokens[neighbor_tokens.len() - take..]
    };
    let right_slice = if prefix {
        &right_tokens[..take]
    } else {
        &right_tokens[right_tokens.len() - take..]
    };

    feature_similarity(right_slice, neighbor_slice)
        .or_else(|| phoneme_similarity(right_slice, neighbor_slice))
        .unwrap_or(0.0)
        .max(0.0)
}

fn normalize_candidate_ranges(ranges: &mut Vec<std::ops::Range<usize>>, utterance_len: usize) {
    for range in ranges.iter_mut() {
        let start = range.start.saturating_sub(1);
        let end = (range.end + 1).min(utterance_len);
        *range = start..end;
    }
    ranges.sort_by(|a, b| a.start.cmp(&b.start).then_with(|| a.end.cmp(&b.end)));
    ranges.dedup_by(|a, b| a.start == b.start && a.end == b.end);
}

fn dedup_candidate_ranges(ranges: &mut Vec<std::ops::Range<usize>>, utterance_len: usize) {
    for range in ranges.iter_mut() {
        range.start = range.start.min(utterance_len);
        range.end = range.end.min(utterance_len);
        if range.end < range.start {
            range.end = range.start;
        }
    }
    ranges.retain(|range| range.start < range.end);
    ranges.sort_by(|a, b| a.start.cmp(&b.start).then_with(|| a.end.cmp(&b.end)));
    ranges.dedup_by(|a, b| a.start == b.start && a.end == b.end);
}

fn alignment_quality_score(ops: &[AlignmentOp], left_len: usize, right_len: usize) -> f32 {
    let denom = left_len.max(right_len).max(1) as f32;
    let total_cost = ops.iter().map(|op| op.cost).sum::<f32>();
    let compression_penalty =
        (left_len.saturating_sub(right_len) as f32 / left_len.max(1) as f32).max(0.0) * 0.35;
    (1.0 - (total_cost / denom) - compression_penalty).clamp(0.0, 1.0)
}

fn anchor_confidence(
    projected: Option<f32>,
    chosen: f32,
    gap: Option<f32>,
    transcript_phone_count: u32,
    chosen_zipa_phone_count: u32,
) -> TranscribePhoneticAnchorConfidence {
    let projected_delta = projected.map(|score| chosen - score).unwrap_or(0.0);
    let gap = gap.unwrap_or(0.0);
    let compression_ratio = if transcript_phone_count == 0 {
        0.0
    } else {
        chosen_zipa_phone_count as f32 / transcript_phone_count as f32
    };
    let sane_length = (0.75..=1.35).contains(&compression_ratio);
    let somewhat_sane_length = (0.5..=1.75).contains(&compression_ratio);

    if projected_delta >= 0.06 && gap >= 0.08 && sane_length {
        TranscribePhoneticAnchorConfidence::High
    } else if projected_delta >= 0.02 && gap >= 0.02 && somewhat_sane_length {
        TranscribePhoneticAnchorConfidence::Medium
    } else {
        TranscribePhoneticAnchorConfidence::Low
    }
}

#[cfg(test)]
mod tests {
    use super::{partition_word_alignment_windows, select_segmental_word_windows};
    use bee_phonetic::{
        AlignmentOp, AlignmentOpKind, align_token_sequences_with_left_word_boundaries,
    };

    fn op(
        kind: AlignmentOpKind,
        left_index: Option<u32>,
        right_index: Option<u32>,
        left_token: Option<&str>,
        right_token: Option<&str>,
    ) -> AlignmentOp {
        AlignmentOp {
            kind,
            left_index,
            right_index,
            left_token: left_token.map(ToOwned::to_owned),
            right_token: right_token.map(ToOwned::to_owned),
            cost: 0.0,
        }
    }

    #[test]
    fn word_alignment_windows_partition_insertions_without_overlap() {
        let ops = vec![
            op(
                AlignmentOpKind::Match,
                Some(0),
                Some(0),
                Some("m"),
                Some("m"),
            ),
            op(AlignmentOpKind::Insert, None, Some(1), None, Some("ɪ")),
            op(
                AlignmentOpKind::Substitute,
                Some(1),
                Some(2),
                Some("ɹ"),
                Some("ə"),
            ),
            op(
                AlignmentOpKind::Match,
                Some(2),
                Some(3),
                Some("ɪ"),
                Some("ɪ"),
            ),
            op(AlignmentOpKind::Insert, None, Some(4), None, Some("n")),
            op(
                AlignmentOpKind::Match,
                Some(3),
                Some(5),
                Some("t"),
                Some("t"),
            ),
        ];

        let windows = partition_word_alignment_windows(
            &ops,
            &[0..2, 2..4],
            &[
                vec!["m".to_string(), "ɹ".to_string()],
                vec!["ɪ".to_string(), "t".to_string()],
            ],
        );
        let first = windows[0].as_ref().expect("first word window");
        let second = windows[1].as_ref().expect("second word window");

        assert_eq!(first.zipa_norm_range, 0..3);
        assert_eq!(second.zipa_norm_range, 3..6);
        assert_eq!(first.ops.len(), 3);
        assert_eq!(second.ops.len(), 3);
        assert_eq!(first.ops[1].kind, AlignmentOpKind::Insert);
        assert_eq!(first.ops[1].right_token.as_deref(), Some("ɪ"));
        assert_eq!(second.ops[0].kind, AlignmentOpKind::Match);
        assert_eq!(second.ops[0].right_token.as_deref(), Some("ɪ"));
    }

    #[test]
    fn word_alignment_windows_assign_leading_and_trailing_inserts_once() {
        let ops = vec![
            op(AlignmentOpKind::Insert, None, Some(0), None, Some("d")),
            op(
                AlignmentOpKind::Match,
                Some(0),
                Some(1),
                Some("m"),
                Some("m"),
            ),
            op(
                AlignmentOpKind::Match,
                Some(1),
                Some(2),
                Some("ɪ"),
                Some("ɪ"),
            ),
            op(AlignmentOpKind::Insert, None, Some(3), None, Some("ə")),
            op(
                AlignmentOpKind::Match,
                Some(2),
                Some(4),
                Some("ɹ"),
                Some("ɹ"),
            ),
        ];

        let windows = partition_word_alignment_windows(
            &ops,
            &[0..2, 2..3],
            &[
                vec!["m".to_string(), "ɪ".to_string()],
                vec!["ɹ".to_string()],
            ],
        );
        let first = windows[0].as_ref().expect("first word window");
        let second = windows[1].as_ref().expect("second word window");

        assert_eq!(first.zipa_norm_range, 0..3);
        assert_eq!(second.zipa_norm_range, 3..5);
        assert_eq!(
            first.ops.first().and_then(|op| op.right_token.as_deref()),
            Some("d")
        );
        assert_eq!(
            second.ops.first().and_then(|op| op.right_token.as_deref()),
            Some("ə")
        );
        assert_eq!(second.ops.len(), 2);
        assert_eq!(
            second.ops.last().and_then(|op| op.right_token.as_deref()),
            Some("ɹ")
        );
    }

    #[test]
    fn word_alignment_windows_assign_boundary_glide_to_following_word() {
        let ops = vec![
            op(
                AlignmentOpKind::Match,
                Some(0),
                Some(0),
                Some("a"),
                Some("a"),
            ),
            op(
                AlignmentOpKind::Match,
                Some(1),
                Some(1),
                Some("ɪ"),
                Some("ɪ"),
            ),
            op(AlignmentOpKind::Insert, None, Some(2), None, Some("j")),
            op(
                AlignmentOpKind::Match,
                Some(2),
                Some(3),
                Some("ʊ"),
                Some("ʊ"),
            ),
            op(
                AlignmentOpKind::Match,
                Some(3),
                Some(4),
                Some("z"),
                Some("z"),
            ),
            op(
                AlignmentOpKind::Match,
                Some(4),
                Some(5),
                Some("d"),
                Some("d"),
            ),
        ];

        let windows = partition_word_alignment_windows(
            &ops,
            &[0..2, 2..5],
            &[
                vec!["a".to_string(), "ɪ".to_string()],
                vec![
                    "j".to_string(),
                    "ʊ".to_string(),
                    "z".to_string(),
                    "d".to_string(),
                ],
            ],
        );
        let first = windows[0].as_ref().expect("first word window");
        let second = windows[1].as_ref().expect("second word window");

        assert_eq!(first.zipa_norm_range, 0..2);
        assert_eq!(second.zipa_norm_range, 2..6);
        assert_eq!(
            second.ops.first().and_then(|op| op.right_token.as_deref()),
            Some("j")
        );
    }

    #[test]
    fn word_alignment_windows_assign_pure_boundary_insert_to_following_word_even_without_onset_hint()
     {
        let ops = vec![
            op(
                AlignmentOpKind::Match,
                Some(0),
                Some(0),
                Some("a"),
                Some("a"),
            ),
            op(
                AlignmentOpKind::Match,
                Some(1),
                Some(1),
                Some("ɪ"),
                Some("ɪ"),
            ),
            op(AlignmentOpKind::Insert, None, Some(2), None, Some("j")),
            op(
                AlignmentOpKind::Match,
                Some(2),
                Some(3),
                Some("ʊ"),
                Some("ʊ"),
            ),
            op(
                AlignmentOpKind::Match,
                Some(3),
                Some(4),
                Some("z"),
                Some("z"),
            ),
            op(
                AlignmentOpKind::Match,
                Some(4),
                Some(5),
                Some("d"),
                Some("d"),
            ),
        ];

        let windows = partition_word_alignment_windows(
            &ops,
            &[0..2, 2..5],
            &[
                vec!["a".to_string(), "ɪ".to_string()],
                vec!["ʊ".to_string(), "z".to_string(), "d".to_string()],
            ],
        );
        let first = windows[0].as_ref().expect("first word window");
        let second = windows[1].as_ref().expect("second word window");

        assert_eq!(first.zipa_norm_range, 0..2);
        assert_eq!(second.zipa_norm_range, 2..6);
        assert_eq!(
            second.ops.first().and_then(|op| op.right_token.as_deref()),
            Some("j")
        );
    }

    #[test]
    fn word_alignment_windows_preserve_left_tail_and_assign_boundary_onset_once() {
        let ops = vec![
            op(
                AlignmentOpKind::Match,
                Some(0),
                Some(0),
                Some("a"),
                Some("ɛ"),
            ),
            op(
                AlignmentOpKind::Match,
                Some(1),
                Some(1),
                Some("k"),
                Some("k"),
            ),
            op(
                AlignmentOpKind::Match,
                Some(2),
                Some(2),
                Some("t"),
                Some("t"),
            ),
            op(
                AlignmentOpKind::Match,
                Some(3),
                Some(3),
                Some("ʃ"),
                Some("ʃ"),
            ),
            op(
                AlignmentOpKind::Substitute,
                Some(4),
                Some(4),
                Some("ʊ"),
                Some("ə"),
            ),
            op(AlignmentOpKind::Delete, Some(5), None, Some("l"), None),
            op(AlignmentOpKind::Delete, Some(6), None, Some("ɪ"), None),
            op(AlignmentOpKind::Insert, None, Some(5), None, Some("m")),
            op(
                AlignmentOpKind::Match,
                Some(7),
                Some(6),
                Some("ɛ"),
                Some("ɪ"),
            ),
            op(
                AlignmentOpKind::Match,
                Some(8),
                Some(7),
                Some("ɹ"),
                Some("ɪ"),
            ),
            op(
                AlignmentOpKind::Match,
                Some(9),
                Some(8),
                Some("ɪ"),
                Some("ɪ"),
            ),
        ];

        let windows = partition_word_alignment_windows(
            &ops,
            &[0..7, 7..10],
            &[
                vec![
                    "a".to_string(),
                    "k".to_string(),
                    "t".to_string(),
                    "ʃ".to_string(),
                    "ʊ".to_string(),
                    "l".to_string(),
                    "ɪ".to_string(),
                ],
                vec![
                    "m".to_string(),
                    "ɛ".to_string(),
                    "ɹ".to_string(),
                    "ɪ".to_string(),
                ],
            ],
        );

        let first = windows[0].as_ref().expect("first word window");
        let second = windows[1].as_ref().expect("second word window");

        assert_eq!(first.ops.len(), 7);
        assert_eq!(
            first.ops[first.ops.len() - 2].left_token.as_deref(),
            Some("l")
        );
        assert_eq!(
            first.ops.last().and_then(|op| op.left_token.as_deref()),
            Some("ɪ")
        );
        assert_eq!(first.zipa_norm_range, 0..5);

        assert_eq!(
            second.ops.first().and_then(|op| op.right_token.as_deref()),
            Some("m")
        );
        assert_eq!(second.zipa_norm_range, 5..9);
    }

    #[test]
    fn segmental_windows_preserve_previous_tail_and_next_onset() {
        let transcript_words = vec![
            vec![
                "a".to_string(),
                "k".to_string(),
                "t".to_string(),
                "ʃ".to_string(),
                "ʊ".to_string(),
                "l".to_string(),
                "ɪ".to_string(),
            ],
            vec![
                "m".to_string(),
                "ɪ".to_string(),
                "ɹ".to_string(),
                "ɪ".to_string(),
            ],
        ];
        let transcript_all = transcript_words
            .iter()
            .flatten()
            .cloned()
            .collect::<Vec<_>>();
        let word_ids = vec![0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1];
        let utterance = vec![
            "ɛ".to_string(),
            "k".to_string(),
            "t".to_string(),
            "ə".to_string(),
            "ʃ".to_string(),
            "w".to_string(),
            "ə".to_string(),
            "l".to_string(),
            "ɪ".to_string(),
            "m".to_string(),
            "ɪ".to_string(),
            "ɪ".to_string(),
        ];
        let utterance_alignment =
            align_token_sequences_with_left_word_boundaries(&transcript_all, &utterance, &word_ids);
        let windows = select_segmental_word_windows(
            &transcript_words,
            &[0..7, 7..11],
            &utterance,
            &utterance_alignment,
        );

        let first = windows[0].as_ref().expect("first word window");
        let second = windows[1].as_ref().expect("second word window");

        assert_eq!(first.zipa_norm_range.end, second.zipa_norm_range.start);
        let first_right = first
            .ops
            .iter()
            .filter_map(|op| op.right_token.as_deref())
            .collect::<Vec<_>>();
        let second_right = second
            .ops
            .iter()
            .filter_map(|op| op.right_token.as_deref())
            .collect::<Vec<_>>();
        assert!(first_right.ends_with(&["l", "ɪ"]));
        assert_eq!(second_right.first().copied(), Some("m"));
    }

    #[test]
    fn segmental_windows_assign_boundary_glide_to_following_word() {
        let transcript_words = vec![
            vec!["a".to_string(), "ɪ".to_string()],
            vec![
                "j".to_string(),
                "ʊ".to_string(),
                "z".to_string(),
                "d".to_string(),
            ],
        ];
        let transcript_all = transcript_words
            .iter()
            .flatten()
            .cloned()
            .collect::<Vec<_>>();
        let word_ids = vec![0, 0, 1, 1, 1, 1];
        let utterance = vec![
            "a".to_string(),
            "ɪ".to_string(),
            "j".to_string(),
            "ʊ".to_string(),
            "z".to_string(),
            "d".to_string(),
        ];
        let utterance_alignment =
            align_token_sequences_with_left_word_boundaries(&transcript_all, &utterance, &word_ids);
        let windows = select_segmental_word_windows(
            &transcript_words,
            &[0..2, 2..6],
            &utterance,
            &utterance_alignment,
        );

        let first = windows[0].as_ref().expect("first word window");
        let second = windows[1].as_ref().expect("second word window");
        let first_right = first
            .ops
            .iter()
            .filter_map(|op| op.right_token.as_deref())
            .collect::<Vec<_>>();
        let second_right = second
            .ops
            .iter()
            .filter_map(|op| op.right_token.as_deref())
            .collect::<Vec<_>>();
        assert_eq!(first_right, vec!["a", "ɪ"]);
        assert_eq!(second_right.first().copied(), Some("j"));
    }

    #[test]
    fn segmental_windows_handle_actual_miri_boundary_shape() {
        let transcript_words = vec![
            vec![
                "a".to_string(),
                "k".to_string(),
                "t".to_string(),
                "ʃ".to_string(),
                "ʊ".to_string(),
                "ɪ".to_string(),
            ],
            vec![
                "m".to_string(),
                "ɛ".to_string(),
                "ɹ".to_string(),
                "ɪ".to_string(),
            ],
        ];
        let transcript_all = transcript_words
            .iter()
            .flatten()
            .cloned()
            .collect::<Vec<_>>();
        let word_ids = vec![0, 0, 0, 0, 0, 0, 1, 1, 1, 1];
        let utterance = vec![
            "ɛ".to_string(),
            "k".to_string(),
            "t".to_string(),
            "ʃ".to_string(),
            "ə".to_string(),
            "w".to_string(),
            "ə".to_string(),
            "l".to_string(),
            "ɪ".to_string(),
            "m".to_string(),
            "ɪ".to_string(),
            "ɪ".to_string(),
        ];
        let utterance_alignment =
            align_token_sequences_with_left_word_boundaries(&transcript_all, &utterance, &word_ids);
        let windows = select_segmental_word_windows(
            &transcript_words,
            &[0..6, 6..10],
            &utterance,
            &utterance_alignment,
        );

        let first = windows[0].as_ref().expect("first word window");
        let second = windows[1].as_ref().expect("second word window");

        assert_eq!(first.zipa_norm_range, 0..9);
        assert_eq!(second.zipa_norm_range.start, 9);
        let second_right = second
            .ops
            .iter()
            .filter_map(|op| op.right_token.as_deref())
            .collect::<Vec<_>>();
        assert_eq!(second_right.first().copied(), Some("m"));
    }
}

fn span_usefulness(
    span_text: &str,
    transcript_phone_count: u32,
    zipa_normalized: &[String],
) -> TranscribePhoneticSpanUsefulness {
    let text = normalized_span_text(span_text);
    let word_count = span_word_count(&text);
    let low_content = is_low_content_text(&text);
    let vowel_onlyish = is_vowel_onlyish(zipa_normalized);

    if low_content || transcript_phone_count <= 2 || (word_count == 1 && vowel_onlyish) {
        TranscribePhoneticSpanUsefulness::Low
    } else if transcript_phone_count >= 5 || word_count >= 2 {
        TranscribePhoneticSpanUsefulness::High
    } else {
        TranscribePhoneticSpanUsefulness::Medium
    }
}

fn span_class(
    span_text: &str,
    transcript_phone_count: u32,
    zipa_normalized: &[String],
) -> TranscribePhoneticSpanClass {
    let text = normalized_span_text(span_text);
    let word_count = span_word_count(&text);
    let token_count = text.split_whitespace().count();
    let repeat = {
        let words = text
            .split_whitespace()
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>();
        words.len() >= 2 && words.windows(2).all(|pair| pair[0] == pair[1])
    };
    let low_content = is_low_content_text(&text);
    let vowel_onlyish = is_vowel_onlyish(zipa_normalized);
    let looks_codey = text.chars().any(|ch| ch.is_ascii_digit())
        || text.contains('-')
        || text.split_whitespace().any(|word| {
            let has_upper = word.chars().any(|ch| ch.is_ascii_uppercase());
            let has_lower = word.chars().any(|ch| ch.is_ascii_lowercase());
            has_upper && has_lower
        })
        || text
            .split_whitespace()
            .any(|word| word.chars().all(|ch| ch.is_ascii_uppercase()) && word.len() >= 2);
    let looks_proper = text
        .split_whitespace()
        .filter(|part| !part.is_empty())
        .all(|word| {
            word.chars()
                .next()
                .is_some_and(|ch| ch.is_ascii_uppercase())
        })
        && token_count > 0;

    if repeat {
        TranscribePhoneticSpanClass::Repeat
    } else if low_content {
        TranscribePhoneticSpanClass::FunctionWord
    } else if looks_codey || transcript_phone_count <= 3 {
        TranscribePhoneticSpanClass::ShortCodeTerm
    } else if vowel_onlyish {
        TranscribePhoneticSpanClass::VowelHeavy
    } else if looks_proper || word_count >= 2 {
        TranscribePhoneticSpanClass::ProperNoun
    } else {
        TranscribePhoneticSpanClass::Ordinary
    }
}

fn normalized_span_text(span_text: &str) -> String {
    span_text
        .chars()
        .filter(|ch| ch.is_alphanumeric() || ch.is_whitespace() || *ch == '-')
        .collect::<String>()
        .trim()
        .to_string()
}

fn span_word_count(normalized_text: &str) -> usize {
    normalized_text
        .split_whitespace()
        .filter(|part| !part.is_empty())
        .count()
}

fn is_low_content_text(normalized_text: &str) -> bool {
    matches!(
        normalized_text.to_ascii_lowercase().trim(),
        "a" | "an"
            | "the"
            | "and"
            | "or"
            | "but"
            | "uh"
            | "um"
            | "oh"
            | "so"
            | "well"
            | "actually"
            | "like"
            | "not"
            | "is"
            | "was"
            | "were"
            | "to"
            | "of"
            | "in"
            | "on"
            | "for"
            | "it"
    )
}

fn is_vowel_onlyish(zipa_normalized: &[String]) -> bool {
    zipa_normalized.iter().all(|token| {
        matches!(
            token.as_str(),
            "a" | "ɑ" | "ɔ" | "ɛ" | "ə" | "ɪ" | "ʊ" | "i" | "u" | "e" | "o" | "æ" | "ʌ"
        )
    })
}

fn zipa_candidate_plausible(candidates: &[TranscribePhoneticCandidate]) -> bool {
    candidates.iter().any(|candidate| {
        candidate.similarity_delta.unwrap_or(0.0) >= 0.04
            && candidate.feature_similarity.unwrap_or(0.0) >= 0.5
    })
}

fn best_delta(candidates: &[TranscribePhoneticCandidate]) -> f32 {
    candidates
        .iter()
        .filter_map(|candidate| candidate.similarity_delta)
        .fold(f32::NEG_INFINITY, f32::max)
}

fn map_alignment_ops(ops: &[AlignmentOp]) -> Vec<TranscribePhoneticAlignmentOp> {
    ops.iter()
        .map(|op| TranscribePhoneticAlignmentOp {
            kind: match op.kind {
                AlignmentOpKind::Match => TranscribePhoneticAlignmentKind::Match,
                AlignmentOpKind::Substitute => TranscribePhoneticAlignmentKind::Substitute,
                AlignmentOpKind::Insert => TranscribePhoneticAlignmentKind::Insert,
                AlignmentOpKind::Delete => TranscribePhoneticAlignmentKind::Delete,
            },
            transcript_index: op.left_index,
            zipa_index: op.right_index,
            transcript_token: op.left_token.clone(),
            zipa_token: op.right_token.clone(),
            cost: op.cost,
        })
        .collect()
}
