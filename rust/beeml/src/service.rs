use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use bee_phonetic::{
    AlignmentOp, AlignmentOpKind, PhoneticIndex, RetrievalQuery, SeedDataset,
    TranscriptAlignmentToken, align_token_sequences, enumerate_transcript_spans_with,
    feature_similarity, normalize_ipa_for_comparison, normalize_ipa_for_comparison_with_spans,
    phoneme_similarity, query_index, reduce_ipa_tokens, score_shortlist, sentence_word_tokens,
    top_right_anchor_windows,
};
use bee_transcribe::{AlignedWord, Engine, SessionSnapshot};
use bee_zipa_mlx::audio::AudioBuffer;
use bee_zipa_mlx::infer::ZipaInference;
use beeml::g2p::CachedEspeakG2p;
use beeml::judge::{OnlineJudge, extract_span_context};
use beeml::rpc::{
    CorpusAlignmentBucketSummary, CorpusAlignmentEvalJob, CorpusAlignmentEvalJobStatus,
    CorpusAlignmentEvalResult, CorpusAlignmentEvalRow, CorpusCapturePrompt, FilterDecision,
    JudgeOptionDebug, JudgeStateDebug, PhoneticComparisonRequest, PhoneticComparisonResult,
    PhoneticComparisonRow, PhoneticComparisonSummary, RapidFireChoice, RetrievalCandidateDebug,
    RetrievalPrototypeEvalRequest, RetrievalPrototypeProbeRequest, RetrievalPrototypeProbeResult,
    SpanDebugTrace, SpanDebugView, TeachRetrievalPrototypeJudgeRequest, TimingBreakdown,
    TranscribePhoneticAlignmentKind, TranscribePhoneticAlignmentOp,
    TranscribePhoneticAnchorConfidence, TranscribePhoneticCandidate, TranscribePhoneticSpan,
    TranscribePhoneticSpanClass, TranscribePhoneticSpanUsefulness, TranscribePhoneticTrace,
};

use crate::offline_eval::*;
use crate::rapid_fire::build_rapid_fire_decision_set;
use crate::util::*;

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
    pub(crate) fn transcribe_samples_chunked(
        &self,
        samples: &[f32],
    ) -> Result<bee_transcribe::FinishResult, String> {
        let options = bee_transcribe::SessionOptions::default();
        let chunk_samples = (options.chunk_duration * 16_000.0) as usize;
        let mut session = self
            .inner
            .engine
            .session(options)
            .map_err(|e| e.to_string())?;

        let mut offset = 0;
        while offset < samples.len() {
            let end = (offset + chunk_samples).min(samples.len());
            session
                .feed(&samples[offset..end])
                .map_err(|e| e.to_string())?;
            offset = end;
        }

        session.finish().map_err(|e| e.to_string())
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
    ) -> Result<TranscribePhoneticTrace, String> {
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
        let utterance_transcript_raw = if transcript.is_empty() {
            Vec::new()
        } else {
            g2p.ipa_tokens(transcript)
                .map_err(|e| e.to_string())?
                .ok_or_else(|| format!("espeak produced no tokens for '{transcript}'"))?
        };
        let transcript_word_normalized_ranges =
            transcript_word_normalized_ranges(&mut g2p, transcript)?;
        let utterance_transcript_normalized = transcript_word_normalized_ranges
            .iter()
            .flat_map(|(_, tokens)| tokens.iter().cloned())
            .collect::<Vec<_>>();
        let utterance_alignment =
            align_token_sequences(&utterance_transcript_normalized, &utterance_zipa_normalized);

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
                    && candidate_plausible;

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
            utterance_zipa_normalized,
            utterance_transcript_normalized,
            utterance_alignment: map_alignment_ops(&utterance_alignment.ops),
            spans: phonetic_spans,
        })
    }

    pub(crate) fn latest_corpus_recordings(
        &self,
        bucket_filter: Option<&str>,
    ) -> Result<Vec<(CorpusCapturePrompt, beeml::rpc::CorpusCaptureRecording)>, String> {
        let prompts = self.corpus_capture_prompts();
        let prompt_map = prompts
            .into_iter()
            .filter(|prompt| bucket_filter.is_none_or(|bucket| prompt.bucket == bucket))
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
    ) -> Result<CorpusAlignmentEvalResult, String> {
        let recordings = self.latest_corpus_recordings(bucket_filter)?;
        let mut rows = Vec::new();

        for (prompt, recording) in recordings.into_iter().take(limit) {
            let wav_bytes = std::fs::read(&recording.wav_path)
                .map_err(|e| format!("reading {}: {e}", recording.wav_path))?;
            let samples = bee_transcribe::decode_wav(&wav_bytes).map_err(|e| e.to_string())?;
            let audio = AudioBuffer {
                samples: samples.clone(),
                sample_rate_hz: 16_000,
            };

            let result = self.transcribe_samples_chunked(&samples)?;
            let snapshot = result.snapshot;

            match self.build_transcribe_phonetic_trace(&audio, &snapshot) {
                Ok(trace) => {
                    let positive_span_count = trace
                        .spans
                        .iter()
                        .filter(|span| {
                            span.candidates
                                .iter()
                                .any(|candidate| candidate.similarity_delta.unwrap_or(0.0) > 0.0)
                        })
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
                        .flat_map(|span| {
                            span.candidates
                                .iter()
                                .filter_map(|candidate| candidate.similarity_delta)
                        })
                        .max_by(|a, b| a.total_cmp(b));
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
                        positive_span_count,
                        worst_span_feature_similarity,
                        best_span_delta,
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
                    positive_span_count: 0,
                    worst_span_feature_similarity: None,
                    best_span_delta: None,
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
    ) -> Result<CorpusAlignmentEvalJob, String> {
        let total_rows = self
            .latest_corpus_recordings(bucket.as_deref())?
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

fn mean_option(values: impl Iterator<Item = Option<f32>>) -> Option<f32> {
    let mut total = 0.0f32;
    let mut count = 0usize;
    for value in values.flatten() {
        total += value;
        count += 1;
    }
    (count > 0).then_some(total / count as f32)
}

fn unix_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn transcript_word_normalized_ranges(
    g2p: &mut CachedEspeakG2p,
    transcript: &str,
) -> Result<Vec<(std::ops::Range<usize>, Vec<String>)>, String> {
    let mut out = Vec::new();
    for word in sentence_word_tokens(transcript) {
        let raw = g2p
            .ipa_tokens(&word.text)
            .map_err(|e| e.to_string())?
            .ok_or_else(|| format!("espeak produced no tokens for '{}'", word.text))?;
        out.push((
            word.char_start..word.char_end,
            normalize_ipa_for_comparison(&raw),
        ));
    }
    Ok(out)
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

fn normalize_candidate_ranges(ranges: &mut Vec<std::ops::Range<usize>>, utterance_len: usize) {
    for range in ranges.iter_mut() {
        let start = range.start.saturating_sub(1);
        let end = (range.end + 1).min(utterance_len);
        *range = start..end;
    }
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
