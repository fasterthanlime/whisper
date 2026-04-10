//! A decode sub-session: owns audio, encoder cache, and tokens together.
//!
//! Rotation = throw away the old DecodeSession and create a new one.
//! The start_time tracks where this sub-session begins in the timeline.

use bee_phonetic::{
    align_token_sequences_with_left_word_boundaries, normalize_ipa_for_comparison,
    normalize_ipa_for_comparison_with_spans, sentence_word_tokens,
};
use bee_qwen3_asr::encoder::EncoderCache;
use bee_qwen3_asr::forced_aligner::ForcedAligner;
use bee_qwen3_asr::generate::{self, ConfidenceMode, TOP_K, TokenConfidence};
use bee_qwen3_asr::mel::MelExtractor;
use bee_qwen3_asr::model::Qwen3ASRModel;
use bee_zipa_mlx::audio::AudioBuffer as ZipaAudioBuffer;
use bee_zipa_mlx::infer::ZipaInference;
use mlx_rs::Array;
use mlx_rs::error::Exception;
use tokenizers::Tokenizer;

use crate::audio_buffer::{AudioBuffer, Seconds};
use crate::g2p::CachedEspeakG2p;
use crate::mlx_stuff::clear_mlx_cache;
use crate::rotation_plan::{RotationTextPlan, plan_rotation};
use crate::text_buffer::{
    self, AlignmentItem, AsrToken, TextBuffer, TokenCount, TokenEntry, TokenId, WordStart,
};
use crate::timing::{log_phase_chunk, phase_start};
use crate::types::{Aligner, RotationCutStrategy};
use crate::zipa_align::{self, timed_range_for_normalized_range};

/// A decode sub-session. Replaced wholesale on rotation.
pub struct DecodeSession {
    /// Audio buffer for this sub-session.
    audio: AudioBuffer,

    /// When this sub-session starts in the session timeline.
    start_time: Seconds,

    /// Encoder cache for incremental encoding.
    encoder_cache: EncoderCache,

    /// Mel spectrogram extractor.
    mel_extractor: MelExtractor,

    /// Current tokens (metadata + text), merged from prefix + generated.
    tokens: Vec<AsrToken>,

    /// Index of the first text token (after `<asr_text>` tag).
    /// Everything before this is metadata.
    metadata_end: usize,

    /// How many recent tokens the model may revise each step.
    rollback_tokens: TokenCount,

    /// How many chunks have been decoded in this sub-session.
    chunk_count: usize,

    /// First index in `tokens` that came from model generation rather than prompt prefix.
    generated_start: usize,

    /// Number of text tokens at the start that are context carried forward from
    /// the previous rotation. These have audio at the beginning of `self.audio`
    /// and must not be counted toward the commit threshold.
    context_token_count: usize,
}

impl DecodeSession {
    /// Create a new decode session.
    pub fn new(audio: AudioBuffer, start_time: Seconds, rollback_tokens: TokenCount) -> Self {
        Self {
            audio,
            start_time,
            encoder_cache: EncoderCache::new(),
            mel_extractor: MelExtractor::new(400, 160, 128, 16000),
            tokens: Vec::new(),
            metadata_end: 0,
            rollback_tokens,
            chunk_count: 0,
            generated_start: 0,
            context_token_count: 0,
        }
    }

    /// Update the rollback window size (for adaptive rollback).
    pub fn set_rollback(&mut self, rollback: TokenCount) {
        self.rollback_tokens = rollback;
    }

    /// Append audio samples to this sub-session.
    pub fn append_audio(&mut self, chunk: &AudioBuffer) {
        self.audio.append(chunk);
    }

    /// Whether this sub-session has any audio.
    pub fn has_audio(&self) -> bool {
        !self.audio.is_empty()
    }

    /// Number of audio samples in this sub-session.
    pub fn audio_len(&self) -> usize {
        self.audio.len()
    }

    /// When this sub-session starts in the absolute timeline.
    pub fn start_time(&self) -> Seconds {
        self.start_time
    }

    /// The text tokens (after `<asr_text>`).
    pub fn text_tokens(&self) -> &[AsrToken] {
        &self.tokens[self.metadata_end..]
    }

    /// The metadata tokens (before `<asr_text>`).
    pub fn metadata_tokens(&self) -> &[AsrToken] {
        if self.metadata_end > 0 {
            // metadata_end includes the <asr_text> tag itself,
            // so metadata tokens are [0..metadata_end-1]
            &self.tokens[..self.metadata_end.saturating_sub(1)]
        } else {
            &[]
        }
    }

    /// Extract detected language from metadata tokens (e.g. "language English").
    /// Returns `None` if no language found.
    pub fn detected_language(&self, tokenizer: &tokenizers::Tokenizer) -> Option<String> {
        let metadata = self.metadata_tokens();
        if metadata.is_empty() {
            return None;
        }
        let ids: Vec<u32> = metadata.iter().map(|t| t.id).collect();
        let meta = tokenizer.decode(&ids, true).ok()?;
        let lang = meta.trim().strip_prefix("language ")?.trim();
        if lang.is_empty() || lang.eq_ignore_ascii_case("none") {
            None
        } else {
            Some(lang.to_string())
        }
    }

    /// Total token count (metadata + text).
    pub fn total_tokens(&self) -> TokenCount {
        TokenCount(self.tokens.len())
    }

    /// How many chunks have been processed.
    pub fn chunk_count(&self) -> usize {
        self.chunk_count
    }

    /// Number of leading text tokens that came from the fixed rollback prefix
    /// rather than the most recent generation pass.
    pub fn generated_text_start(&self) -> usize {
        self.generated_start.saturating_sub(self.metadata_end)
    }

    /// Number of context tokens carried forward from the previous rotation.
    pub fn context_token_count(&self) -> usize {
        self.context_token_count
    }

    /// How many *fresh* text tokens can be committed without splitting a word.
    /// Context tokens from the previous rotation are excluded from the count.
    pub fn committable_text_tokens(&self, tokenizer: &Tokenizer, n: TokenCount) -> TokenCount {
        let all_entries = TextBuffer::from_entries(self.pending_entries(tokenizer));
        let old_ctx = self.context_token_count;
        let stable_total = all_entries.len().saturating_sub(self.rollback_tokens);
        let stable_fresh = stable_total.0.saturating_sub(old_ctx);
        let fresh_entries =
            TextBuffer::from_entries(all_entries.entries()[old_ctx..stable_total.0].to_vec());
        fresh_entries.snap_to_word_boundary(TokenCount(n.0.min(stable_fresh)))
    }

    /// Run one decode step: mel extraction, encoding, generation.
    /// Updates internal tokens.
    pub fn decode_step(
        &mut self,
        model: &Qwen3ASRModel,
        tokenizer: &Tokenizer,
        language: &str,
        max_tokens: usize,
        confidence_mode: ConfidenceMode,
    ) -> Result<(), Exception> {
        self.chunk_count += 1;
        let decode_total_start = phase_start();

        // Mel extraction
        let mel_start = phase_start();
        let (mel_data, n_mels, n_frames) = self
            .mel_extractor
            .extract(self.audio.samples())
            .map_err(|e| Exception::custom(format!("mel: {e}")))?;
        log_phase_chunk("decode_step", "mel_extract", self.chunk_count, mel_start);
        let mel = Array::from_slice(&mel_data, &[n_mels as i32, n_frames as i32]);

        // Encode audio (incremental)
        let encode_start = phase_start();
        let audio_features = model.encode_incremental(&mel, &mut self.encoder_cache)?;
        let audio_features = mlx_rs::ops::expand_dims(&audio_features, 0)?;
        audio_features.eval()?;
        log_phase_chunk(
            "decode_step",
            "encode_incremental",
            self.chunk_count,
            encode_start,
        );

        // Build prompt with prefix rollback
        let prompt_start = phase_start();
        let prefix = self.compute_prefix();
        let mut prompt = generate::build_initial_prompt(
            audio_features.shape()[1] as usize,
            language,
            "",
            tokenizer,
        );
        if let Some(ref prefix_tokens) = prefix {
            prompt.extend(prefix_tokens.iter().map(|t| t.id as i32));
        }
        log_phase_chunk(
            "decode_step",
            "build_prompt",
            self.chunk_count,
            prompt_start,
        );

        // Generate
        let generate_start = phase_start();
        let mut cache = None;
        let (generated, logprobs, _) = generate::prefill_and_decode(
            model,
            &prompt,
            &audio_features,
            &mut cache,
            0,
            max_tokens,
            confidence_mode,
        )?;
        log_phase_chunk(
            "decode_step",
            "prefill_and_decode",
            self.chunk_count,
            generate_start,
        );

        // Merge prefix + generated into a single Vec<AsrToken>
        let merge_start = phase_start();
        let prefix_len = prefix.as_ref().map_or(0, |p| p.len());
        let merged = Self::merge_tokens(prefix, &generated, &logprobs);
        log_phase_chunk("decode_step", "merge_tokens", self.chunk_count, merge_start);

        tracing::debug!(
            "decode_session: chunk={} generated={} prefix={prefix_len} total={}",
            self.chunk_count,
            generated.len(),
            merged.len(),
        );

        if merged.is_empty() && !self.tokens.is_empty() {
            tracing::debug!(
                "decode_session: EOS with no output, preserving {} tokens",
                self.tokens.len()
            );
        } else if !merged.is_empty() {
            self.tokens = merged;
            self.generated_start = prefix_len;
            self.recompute_metadata_boundary();
        }

        drop(cache);
        let clear_start = phase_start();
        clear_mlx_cache();
        log_phase_chunk(
            "decode_step",
            "clear_mlx_cache",
            self.chunk_count,
            clear_start,
        );
        log_phase_chunk("decode_step", "total", self.chunk_count, decode_total_start);
        Ok(())
    }

    /// Build pending TokenEntry values from text tokens (for TextBuffer).
    /// Word boundaries are detected via DecodeStream: if a token's
    /// contribution starts with a space or newline, it begins a new word.
    pub fn pending_entries(&self, tokenizer: &Tokenizer) -> Vec<TokenEntry> {
        let text_tokens = self.text_tokens();
        if text_tokens.is_empty() {
            return Vec::new();
        }

        let mut stream = tokenizer.decode_stream(true);
        let mut entries = Vec::with_capacity(text_tokens.len());

        for (i, token) in text_tokens.iter().enumerate() {
            let chunk = stream.step(token.id).ok().flatten();
            let is_word_start = if i == 0 {
                true
            } else {
                chunk
                    .as_ref()
                    .map_or(false, |c| c.starts_with(' ') || c.starts_with('\n'))
            };

            entries.push(TokenEntry {
                token: *token,
                word: if is_word_start {
                    Some(WordStart { alignment: None })
                } else {
                    None
                },
            });
        }

        entries
    }

    /// Commit a stable prefix of the current decode. The cut is planned from the
    /// full current text state, then timing is derived by aligning the full
    /// current transcript against the full current audio.
    pub fn commit(
        &mut self,
        n: TokenCount,
        context_tokens: usize,
        rotation_cut_strategy: &RotationCutStrategy,
        aligner: &Aligner,
        forced_aligner: Option<&ForcedAligner>,
        zipa: &ZipaInference,
        tokenizer: &Tokenizer,
        g2p: Option<&mut CachedEspeakG2p>,
    ) -> Result<Option<(TextBuffer, AudioBuffer, AudioBuffer)>, Exception> {
        let commit_total_start = phase_start();
        let chunk_index = self.chunk_count;
        let plan_start = phase_start();

        if self.audio.is_empty() {
            tracing::warn!("commit: no audio to align against, skipping");
            return Ok(None);
        }

        let pending_entries = self.pending_entries(tokenizer);
        let entries = TextBuffer::from_entries(pending_entries.clone());
        let Some(plan) = plan_rotation(
            &entries,
            self.context_token_count,
            self.rollback_tokens,
            context_tokens,
            n,
            rotation_cut_strategy,
        ) else {
            tracing::warn!(
                requested_tokens = n.0,
                total_text_tokens = self.text_tokens().len(),
                rollback_tokens = self.rollback_tokens.0,
                old_context_tokens = self.context_token_count,
                rotation_cut_strategy = ?rotation_cut_strategy,
                "commit: no rotatable stable prefix"
            );
            return Ok(None);
        };
        log_phase_chunk("commit", "plan_rotation", chunk_index, plan_start);

        tracing::debug!(
            requested_tokens = n.0,
            total_tokens = plan.total_tokens.0,
            stable_total_tokens = plan.stable_total_tokens.0,
            commit_tokens = plan.commit_tokens.0,
            next_context_tokens = plan.next_context_tokens.0,
            rollback_tokens = plan.rollback_tokens.0,
            old_context_tokens = plan.old_context_tokens.0,
            "commit: planned text partition"
        );

        let align_start = phase_start();
        let mut aligned_full =
            self.align_current_text(&entries, aligner, forced_aligner, zipa, tokenizer, g2p)?;
        log_phase_chunk("commit", "align_current_text", chunk_index, align_start);

        let trim_at = self.trim_time_for_plan(&aligned_full, plan, rotation_cut_strategy)?;
        let committed_word_count = aligned_full.word_count_before(plan.commit_end());

        let split_start = phase_start();
        let aligned_committed = aligned_full.split_off_front(plan.commit_end());
        log_phase_chunk(
            "commit",
            "extract_committed_buffer",
            chunk_index,
            split_start,
        );

        let rotate_start = phase_start();
        tracing::info!(
            "{}",
            render_rotation_debug_report(
                tokenizer,
                self.text_tokens(),
                &aligned_full,
                plan,
                trim_at,
                self.start_time,
            )
        );
        let new_start = self.start_time + trim_at;
        let (committed_audio, remaining) = self.audio.split_at(trim_at);
        let remaining_audio_for_sink = remaining.clone();
        let remaining_text_tokens = self
            .text_tokens()
            .len()
            .saturating_sub(plan.drain_count().0);

        tracing::info!(
            committed_word_count,
            committed_tokens = plan.commit_tokens.0,
            drain_count = plan.drain_count().0,
            next_context_tokens = plan.next_context_tokens.0,
            rollback_tokens = plan.rollback_tokens.0,
            remaining_text_tokens,
            old_start = %format!("{:.3}s", self.start_time.0),
            new_start = %format!("{:.3}s", new_start.0),
            trim_at = %format!("{:.3}s", trim_at.0),
            audio_before = self.audio.len(),
            audio_after = remaining.len(),
            "commit: rotation"
        );

        self.audio = remaining;
        self.start_time = new_start;

        let drop_start = self.metadata_end;
        let drop_end = self.metadata_end + plan.drain_count().0;
        self.tokens.drain(drop_start..drop_end);
        self.context_token_count = plan.next_context_tokens.0;

        self.encoder_cache = EncoderCache::new();
        self.mel_extractor = MelExtractor::new(400, 160, 128, 16000);
        self.generated_start = 0;
        log_phase_chunk("commit", "rotate_reset", chunk_index, rotate_start);
        log_phase_chunk("commit", "total", chunk_index, commit_total_start);

        Ok(Some((
            aligned_committed,
            committed_audio,
            remaining_audio_for_sink,
        )))
    }

    /// Commit all text tokens. Same as `commit` but without a token limit.
    pub fn commit_all(
        &mut self,
        aligner: &Aligner,
        forced_aligner: Option<&ForcedAligner>,
        zipa: &ZipaInference,
        tokenizer: &Tokenizer,
        g2p: Option<&mut CachedEspeakG2p>,
    ) -> Result<Option<TextBuffer>, Exception> {
        let commit_total_start = phase_start();
        let chunk_index = self.chunk_count;

        if self.audio.is_empty() {
            tracing::warn!("commit_all: no audio to align against, skipping");
            return Ok(None);
        }

        let align_start = phase_start();
        let entries = TextBuffer::from_entries(self.pending_entries(tokenizer));
        let aligned =
            self.align_current_text(&entries, aligner, forced_aligner, zipa, tokenizer, g2p)?;
        log_phase_chunk("commit_all", "align_current_text", chunk_index, align_start);

        tracing::info!(
            committed_tokens = self.text_tokens().len(),
            old_ctx = self.context_token_count,
            audio_samples = self.audio.len(),
            aligned_words = aligned.words().count(),
            "commit_all: final commit without rotation"
        );

        self.audio = AudioBuffer::empty(self.audio.sample_rate());
        self.tokens.truncate(self.metadata_end);
        self.encoder_cache = EncoderCache::new();
        self.mel_extractor = MelExtractor::new(400, 160, 128, 16000);
        self.generated_start = 0;
        log_phase_chunk("commit_all", "total", chunk_index, commit_total_start);

        Ok(Some(aligned))
    }

    /// Clear all tokens and reset state.
    pub fn clear(&mut self) {
        self.tokens.clear();
        self.metadata_end = 0;
        self.encoder_cache = EncoderCache::new();
        self.chunk_count = 0;
        self.generated_start = 0;
    }

    // ── Internal ────────────────────────────────────────────────────

    fn align_current_text(
        &self,
        entries: &TextBuffer,
        aligner: &Aligner,
        forced_aligner: Option<&ForcedAligner>,
        zipa: &ZipaInference,
        tokenizer: &Tokenizer,
        g2p: Option<&mut CachedEspeakG2p>,
    ) -> Result<TextBuffer, Exception> {
        let transcript_ids = entries.token_ids();
        let transcript = tokenizer.decode(&transcript_ids, true).unwrap_or_default();
        if transcript.trim().is_empty() {
            return Err(Exception::custom("empty transcript during alignment"));
        }

        let alignment_items: Vec<AlignmentItem> = match aligner {
            Aligner::Zipa => {
                let g2p = g2p.expect(
                    "Aligner::Zipa requires G2P but g2p is None — check that correction_dir / g2p_dir is configured",
                );
                zipa_word_alignments(zipa, &self.audio, &transcript, g2p)
                    .map_err(|e| Exception::custom(format!("zipa alignment: {e}")))?
            }
            Aligner::Qwen => {
                let forced_aligner = forced_aligner.expect(
                    "Aligner::Qwen requires forced aligner but aligner_dir was not provided",
                );
                forced_aligner
                    .align(self.audio.samples(), &transcript)
                    .map_err(|e| Exception::custom(format!("aligner: {e}")))?
                    .into_iter()
                    .map(|item| AlignmentItem {
                        word: item.word,
                        start: Seconds(item.start_time as f64),
                        end: Seconds(item.end_time as f64),
                    })
                    .collect()
            }
        };

        if alignment_items.is_empty() {
            return Err(Exception::custom("alignment returned no items"));
        }

        Ok(text_buffer::align(
            TextBuffer::from_entries(entries.entries().to_vec()),
            &alignment_items,
            &self.audio,
            self.start_time,
        ))
    }

    fn trim_time_for_plan(
        &self,
        aligned_full: &TextBuffer,
        plan: RotationTextPlan,
        rotation_cut_strategy: &RotationCutStrategy,
    ) -> Result<Seconds, Exception> {
        let commit_end = aligned_full
            .last_aligned_word_end_before(plan.commit_end())
            .ok_or_else(|| Exception::custom("no aligned word end at commit boundary"))?;
        let trim_at = match rotation_cut_strategy {
            RotationCutStrategy::Zipa => commit_end - self.start_time,
            RotationCutStrategy::Qwen3
            | RotationCutStrategy::ManualTargetCommittedTextTokens(_)
            | RotationCutStrategy::Uncut => commit_end - self.start_time,
        };
        Ok(trim_at)
    }

    /// Compute the fixed prefix for rollback — text tokens only.
    ///
    /// Metadata is never included here: `build_initial_prompt` re-emits the
    /// language header on every step, so including metadata in the prefix
    /// would duplicate it in the model's context.
    fn compute_prefix(&self) -> Option<Vec<AsrToken>> {
        if self.chunk_count < 2 || self.tokens.is_empty() {
            return None;
        }
        let text_tokens = self.text_tokens();
        let text_keep = text_tokens.len().saturating_sub(self.rollback_tokens.0);
        if text_keep == 0 {
            return None;
        }
        Some(text_tokens[..text_keep].to_vec())
    }

    /// Merge prefix tokens + newly generated tokens into a single Vec<AsrToken>.
    ///
    /// The prefix already carries correct confidence from the previous step.
    /// Generated tokens get fresh confidence from this step.
    fn merge_tokens(
        prefix: Option<Vec<AsrToken>>,
        generated: &[i32],
        logprobs: &[TokenConfidence],
    ) -> Vec<AsrToken> {
        let mut merged: Vec<AsrToken> = prefix.unwrap_or_default();
        for (i, &token_id) in generated.iter().enumerate() {
            let lp = logprobs.get(i);
            merged.push(AsrToken {
                id: token_id as TokenId,
                concentration: lp.map_or(0.0, |l| l.concentration),
                margin: lp.map_or(0.0, |l| l.margin),
                alternative_count: lp.map_or(0, |l| l.alternative_count),
                top_ids: lp.map_or([0; TOP_K], |l| l.top_ids.map(|id| id as TokenId)),
                top_logits: lp.map_or([0.0; TOP_K], |l| l.top_logits),
            });
        }
        merged
    }

    /// Find the <asr_text> boundary in tokens.
    fn recompute_metadata_boundary(&mut self) {
        let asr_text_id = generate::TOK_ASR_TEXT as TokenId;
        self.metadata_end = self
            .tokens
            .iter()
            .position(|t| t.id == asr_text_id)
            .map(|pos| pos + 1) // +1 to skip the tag itself
            .unwrap_or(0);
    }
}

/// Derive per-word timing from ZIPA inference + G2P phoneme alignment.
///
/// Returns `AlignmentItem` per word with ZIPA-derived start/end times.
/// Falls back to an empty vec (caller should use forced aligner) when
/// G2P or ZIPA alignment produces no usable output.
fn zipa_word_alignments(
    zipa: &ZipaInference,
    audio: &AudioBuffer,
    transcript: &str,
    g2p: &mut CachedEspeakG2p,
) -> Result<Vec<AlignmentItem>, String> {
    let zipa_audio = ZipaAudioBuffer {
        samples: audio.samples().to_vec(),
        sample_rate_hz: audio.sample_rate().0,
    };
    tracing::debug!(
        audio_samples = audio.len(),
        audio_secs = audio.duration().0,
        sample_rate = audio.sample_rate().0,
        "zipa_word_alignments: audio received"
    );
    let utterance = zipa.infer_audio(&zipa_audio).map_err(|e| e.to_string())?;
    let duration = audio.duration().0;

    let phone_spans = utterance
        .derive_phone_spans(&zipa.tokens, duration, 0)
        .into_iter()
        .filter(|s| s.token != "▁")
        .collect::<Vec<_>>();

    let zipa_raw = utterance
        .tokens
        .into_iter()
        .filter(|t| t != "▁")
        .collect::<Vec<_>>();
    let zipa_norm_with_spans = normalize_ipa_for_comparison_with_spans(&zipa_raw);
    let zipa_norm = zipa_norm_with_spans
        .iter()
        .map(|t| t.token.clone())
        .collect::<Vec<_>>();

    tracing::debug!(
        phone_spans = phone_spans.len(),
        zipa_tokens = zipa_norm.len(),
        transcript = %transcript.trim(),
        "zipa_word_alignments: inferred"
    );

    let word_raw_ranges = zipa_align::transcript_word_raw_ranges(g2p, transcript)?;
    let word_norm_ranges = word_raw_ranges
        .iter()
        .map(|(range, raw)| (range.clone(), normalize_ipa_for_comparison(raw)))
        .collect::<Vec<_>>();
    let transcript_norm = word_norm_ranges
        .iter()
        .flat_map(|(_, tokens)| tokens.iter().cloned())
        .collect::<Vec<_>>();
    let word_ids = word_norm_ranges
        .iter()
        .enumerate()
        .flat_map(|(wi, (_, tokens))| std::iter::repeat_n(wi, tokens.len()))
        .collect::<Vec<_>>();

    tracing::debug!(
        words = word_norm_ranges.len(),
        transcript_phones = transcript_norm.len(),
        transcript_norm = %transcript_norm.join(" "),
        zipa_norm = %zipa_norm.join(" "),
        "zipa_word_alignments: phoneme sequences"
    );

    let alignment =
        align_token_sequences_with_left_word_boundaries(&transcript_norm, &zipa_norm, &word_ids);

    let transcript_words = sentence_word_tokens(transcript);
    let token_ranges = (0..word_norm_ranges.len())
        .map(|wi| zipa_align::transcript_token_range_for_span(&word_norm_ranges, wi, wi + 1))
        .collect::<Vec<_>>();
    let word_tokens = word_norm_ranges
        .iter()
        .map(|(_, tokens)| tokens.clone())
        .collect::<Vec<_>>();
    let word_windows = zipa_align::select_segmental_word_windows(
        &word_tokens,
        &token_ranges,
        &zipa_norm,
        &alignment,
    );

    let mut items = Vec::new();
    for (wi, _) in word_norm_ranges.iter().enumerate() {
        let word_text = match transcript_words.get(wi) {
            Some(w) => w.text.clone(),
            None => continue,
        };
        let window = match word_windows.get(wi).and_then(|w| w.as_ref()) {
            Some(w) => w,
            None => {
                tracing::debug!(word = %word_text, word_index = wi, "zipa_word_alignments: no window for word");
                continue;
            }
        };
        let timed = timed_range_for_normalized_range(
            &zipa_norm_with_spans,
            &phone_spans,
            window.zipa_norm_range.clone(),
        );
        match timed {
            Some(t) => {
                tracing::debug!(
                    word = %word_text,
                    start = t.start_time_secs,
                    end = t.end_time_secs,
                    "zipa_word_alignments: aligned"
                );
                items.push(AlignmentItem {
                    word: word_text,
                    start: Seconds(t.start_time_secs),
                    end: Seconds(t.end_time_secs),
                });
            }
            None => {
                tracing::debug!(word = %word_text, word_index = wi, "zipa_word_alignments: no timing for word");
            }
        }
    }
    Ok(items)
}

fn render_rotation_debug_report(
    tokenizer: &Tokenizer,
    text_tokens: &[AsrToken],
    aligned_full: &TextBuffer,
    plan: RotationTextPlan,
    trim_at: Seconds,
    start_time: Seconds,
) -> String {
    let total = text_tokens.len();
    let old_ctx_end = plan.old_context_tokens.0.min(total);
    let drain_end = plan.drain_count().0.min(total);
    let stable_end = plan.stable_end().0.min(total);

    let full_text = decode_token_slice(tokenizer, text_tokens);
    let old_context_text = decode_token_slice(tokenizer, &text_tokens[..old_ctx_end]);
    let commit_text = decode_token_slice(tokenizer, &text_tokens[old_ctx_end..drain_end]);
    let next_prefix_text = decode_token_slice(tokenizer, &text_tokens[drain_end..stable_end]);
    let rollback_text = decode_token_slice(tokenizer, &text_tokens[stable_end..]);
    let remaining_text = decode_token_slice(tokenizer, &text_tokens[drain_end..]);

    let audio_cut_word = last_word_before(tokenizer, aligned_full, plan.commit_end())
        .unwrap_or_else(|| "<none>".to_string());
    let first_kept_word = first_word_after(tokenizer, aligned_full, plan.drain_count())
        .unwrap_or_else(|| "<none>".to_string());

    format!(
        concat!(
            "\n🔪 Rotation boundary\n",
            "  📝 full         : {full}\n",
            "  📦 old context  : {old_context}\n",
            "  ✅ commit       : {commit}\n",
            "  ↪️ next prefix  : {next_prefix}\n",
            "  🔁 rollback     : {rollback}\n",
            "  🧵 remaining    : {remaining}\n",
            "  🎯 audio cut    : +{trim:.3}s / @{absolute:.3}s\n",
            "  🔤 cut word     : {audio_cut_word}\n",
            "  ⏭️ first kept    : {first_kept_word}\n"
        ),
        full = sanitize_debug_text(&full_text),
        old_context = sanitize_debug_text(&old_context_text),
        commit = sanitize_debug_text(&commit_text),
        next_prefix = sanitize_debug_text(&next_prefix_text),
        rollback = sanitize_debug_text(&rollback_text),
        remaining = sanitize_debug_text(&remaining_text),
        trim = trim_at.0,
        absolute = (start_time + trim_at).0,
        audio_cut_word = sanitize_debug_text(&audio_cut_word),
        first_kept_word = sanitize_debug_text(&first_kept_word),
    )
}

fn decode_token_slice(tokenizer: &Tokenizer, tokens: &[AsrToken]) -> String {
    let ids = tokens.iter().map(|token| token.id).collect::<Vec<_>>();
    tokenizer
        .decode(&ids, true)
        .unwrap_or_else(|_| "<decode failed>".to_string())
}

fn decode_entries(tokenizer: &Tokenizer, entries: &[TokenEntry]) -> String {
    let ids = entries
        .iter()
        .map(|entry| entry.token.id)
        .collect::<Vec<_>>();
    tokenizer
        .decode(&ids, true)
        .unwrap_or_else(|_| "<decode failed>".to_string())
}

fn sanitize_debug_text(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        "∅".to_string()
    } else {
        trimmed.replace('\n', " ↩ ")
    }
}

fn last_word_before(tokenizer: &Tokenizer, buf: &TextBuffer, n: TokenCount) -> Option<String> {
    let limit = n.0.min(buf.entries().len());
    let mut seen = 0usize;
    let mut result = None;
    for word in buf.words() {
        seen += word.len();
        if seen > limit {
            break;
        }
        result = Some(decode_entries(tokenizer, word));
    }
    result
}

fn first_word_after(tokenizer: &Tokenizer, buf: &TextBuffer, n: TokenCount) -> Option<String> {
    let limit = n.0.min(buf.entries().len());
    let mut seen = 0usize;
    for word in buf.words() {
        let word_start = seen;
        seen += word.len();
        if word_start >= limit {
            return Some(decode_entries(tokenizer, word));
        }
    }
    None
}

#[cfg(test)]
mod tests {}
