//! `Utterance` owns the rollback-oriented streaming ASR model from the README.
//!
//! Desired state:
//! - `stable` is the left prefix that remains alive in retained KV state
//! - `carry` is the bridge slice that is replayed as token IDs in the next prompt
//! - `preview` is the live tail that gets truncated and regenerated
//! - there is one machine:
//!   - zero stable / zero cut is not a special mode
//!   - it is simply the case where nothing has been promoted yet
//!   - the same planning rules still apply
//!
//! Non-negotiable constraints:
//! - token boundaries are canonical; we do not cut or replay in string space
//! - carry replay must use token IDs sliced from the tape, never decoded text
//! - transcript text must exist
//! - G2P must exist
//! - ZIPA must exist
//! - nothing in the real machine happens without those components
//! - cut promotion only belongs in the real rotated-audio path described in the
//!   README; the current full-audio path must not pretend otherwise

use crate::tokens::UtteranceTokenRange;
use crate::{
    AsrTokenAlternative, AsrTokenConfidence, AudioBuffer, ComparisonPhone, FeedOutput, OutputToken,
    SampleCount, SampleOffset, SampleRange, Tape, TimeRange, TimedToken, TokenId, TokenIndex,
    ZipaTiming,
};
use bee_g2p::{BeeG2p, token_piece_phones, transcript_alignment_input};
use bee_qwen3_asr::encoder::EncoderCache;
use bee_qwen3_asr::generate::{self, ConfidenceMode, TokenConfidence};
use bee_qwen3_asr::mel::MelExtractor;
use bee_qwen3_asr::mlx_rs::Array;
use bee_qwen3_asr::mlx_rs::ops;
use bee_qwen3_asr::model::Qwen3ASRModel;
use bee_transcribe::zipa_align::{
    CachedZipaOutput, ComparisonRangeTiming, TranscriptAlignment, infer_cached_zipa_output,
    transcript_comparison_input_from_g2p,
};
use bee_zipa_mlx::audio::AudioBuffer as ZipaAudioBuffer;
use bee_zipa_mlx::infer::ZipaInference;
use compact_str::CompactString;
use std::path::Path;
use std::sync::Arc;

const DEFAULT_PREVIEW_REWRITE_TOKENS: usize = 5;
const DEFAULT_MAX_NEW_TOKENS: usize = 256;
const MIN_PREVIEW_MAX_NEW_TOKENS: usize = 4;
const PREVIEW_MAX_NEW_TOKENS_BASE: usize = 2;
const PREVIEW_TOKENS_PER_SECOND: usize = 8;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Cutting {
    /// Never promote `stable`; this is the effective "cut at 0" case.
    Never,
    /// Choose the cut internally from the current carry/preview geometry.
    Auto,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PreviewDecodePlan {
    /// Token boundary where the current `preview` starts. Everything at or to the
    /// right of this boundary is provisional and may be replaced wholesale.
    rollback_to: TokenIndex,
    /// Decoder position that should remain alive in KV before this decode step.
    ///
    /// In the final three-way split this is the decoder position for
    /// `stable_through`, not for `preview_from`.
    decoder_position: usize,
    /// Budget for how much new text this feed is allowed to produce.
    rewrite_budget_tokens: usize,
}

#[derive(Clone, Debug, PartialEq)]
struct DecodedPreview {
    detected_language: Option<CompactString>,
    tokens: Vec<OutputToken>,
    prompt_end_position: usize,
    next_decoder_position: usize,
}

struct AsrRuntime {
    model: Arc<Qwen3ASRModel>,
    mel_extractor: MelExtractor,
    encoder_cache: EncoderCache,
    language: CompactString,
    max_new_tokens: usize,
}

struct PhoneticRuntime {
    g2p: BeeG2p,
    zipa: ZipaInference,
    lang_code: CompactString,
}

struct PreviewRun {
    transcript: String,
    token_enrichments: Vec<PreviewTokenEnrichment>,
}

#[derive(Clone, Debug)]
struct ZipaPreviewCache {
    audio_end: SampleOffset,
    output: CachedZipaOutput,
}

struct PreviewTokenEnrichment {
    token_index: usize,
    g2p_ipa: CompactString,
    transcript_phones: Vec<ComparisonPhone>,
    zipa_timing: ZipaTiming,
}

impl AsrRuntime {
    fn new(
        model: Arc<Qwen3ASRModel>,
        tokenizer_path: &Path,
        num_mel_bins: usize,
        language: CompactString,
        max_new_tokens: usize,
    ) -> Self {
        crate::init_tokenizer(tokenizer_path);
        Self {
            model,
            mel_extractor: MelExtractor::new(400, 160, num_mel_bins, crate::SAMPLE_RATE),
            encoder_cache: EncoderCache::new(),
            language,
            max_new_tokens,
        }
    }
}

impl PhoneticRuntime {
    fn load(
        g2p_model_dir: &Path,
        tokenizer_path: &Path,
        zipa_bundle_dir: &Path,
        lang_code: CompactString,
    ) -> anyhow::Result<Self> {
        let g2p = BeeG2p::load(g2p_model_dir, tokenizer_path)?;
        let zipa = ZipaInference::load_quantized_bundle_dir(zipa_bundle_dir)
            .map_err(|e| anyhow::anyhow!("loading ZIPA bundle: {e}"))?;
        Ok(Self {
            g2p,
            zipa,
            lang_code,
        })
    }
}

/// Streaming utterance state for the next rollback model.
///
/// Intent:
/// - audio is the only public ingress
/// - utterance audio is append-only and remains anchored at utterance sample 0
/// - utterance-owned token, commit, and ASR state stay synchronized under one owner
/// - inference and chunk construction happen inside this type
/// - cut application happens inside this type
/// - the moving stable/carry/preview boundaries are tracked in token space
/// - cut policy is a small enum, not an external trait
///
/// Required pipeline:
/// - transcript text must exist before we can reason about boundaries
/// - G2P must exist before we can build transcript-side phones
/// - ZIPA must exist before we can project timings back onto tokens
/// - if any of those are missing, the real machine is not "partially running";
///   it is simply not running the intended model yet
pub struct Utterance {
    /// Append-only utterance audio owned in utterance-global sample space.
    ///
    /// Invariant:
    /// - this buffer remains anchored at utterance sample 0
    audio: AudioBuffer,

    /// Token boundary through which the utterance has been promoted into `stable`.
    ///
    /// Invariant:
    /// - zero means no tokens are stable yet
    /// - when this stays at zero, the machine is still the same machine: no
    ///   stable KV is retained, prompt building falls back to the initial prompt,
    ///   and the whole utterance remains live
    /// - tokens before this boundary are `stable`
    /// - any sample/time boundaries are derived from this token boundary, not stored separately
    stable_through: TokenIndex,

    /// Token boundary where the live `preview` region starts.
    ///
    /// Invariant:
    /// - `stable_through <= preview_from <= tape.end()`
    /// - tokens in `[stable_through, preview_from)` are the replay bridge
    /// - tokens in `[preview_from, tape.end())` are the current `preview`
    preview_from: TokenIndex,

    /// Canonical token-aligned output tape plus synchronized ASR rollback state.
    tape: Tape,

    /// How many tail tokens the next feed may rewrite before a new cut is chosen.
    preview_rewrite_tokens: usize,

    /// Number of feed steps processed so far.
    feed_count: usize,

    /// Built-in cut policy.
    cutting: Cutting,

    /// Optional ASR runtime used to decode preview on each feed step.
    asr: Option<AsrRuntime>,

    /// Phonetic/timing runtime used to enrich the current preview.
    ///
    /// This is not optional in the intended machine semantics:
    /// - transcript text
    /// - G2P
    /// - ZIPA
    /// are all required before preview/cut decisions are meaningful.
    ///
    /// It remains an `Option` only because the crate is still being brought into
    /// alignment with the README model incrementally.
    phonetics: Option<PhoneticRuntime>,

    /// Cached ZIPA acoustic output for the most recently inferred audio prefix.
    zipa_cache: Option<ZipaPreviewCache>,
}

impl Utterance {
    /// Creates a new utterance with empty audio and all token boundaries at 0.
    pub fn new(num_layers: usize, cutting: Cutting) -> Self {
        Self {
            audio: AudioBuffer::new(SampleOffset::new(0), Vec::new()),
            stable_through: TokenIndex::new(0),
            preview_from: TokenIndex::new(0),
            tape: Tape::new(num_layers),
            preview_rewrite_tokens: DEFAULT_PREVIEW_REWRITE_TOKENS,
            feed_count: 0,
            cutting,
            asr: None,
            phonetics: None,
            zipa_cache: None,
        }
    }

    pub fn attach_qwen_asr(
        &mut self,
        model: Arc<Qwen3ASRModel>,
        tokenizer_path: &Path,
        num_mel_bins: usize,
        language: impl Into<CompactString>,
    ) {
        self.asr = Some(AsrRuntime::new(
            model,
            tokenizer_path,
            num_mel_bins,
            language.into(),
            DEFAULT_MAX_NEW_TOKENS,
        ));
    }

    pub fn attach_phonetics(
        &mut self,
        g2p_model_dir: &Path,
        tokenizer_path: &Path,
        zipa_bundle_dir: &Path,
        lang_code: impl Into<CompactString>,
    ) -> anyhow::Result<()> {
        self.phonetics = Some(PhoneticRuntime::load(
            g2p_model_dir,
            tokenizer_path,
            zipa_bundle_dir,
            lang_code.into(),
        )?);
        Ok(())
    }

    /// Feeds raw samples into the utterance recording buffer.
    ///
    /// Intent:
    /// - audio is the only public input into utterance state
    /// - callers provide only raw samples, never timed audio buffers
    /// - utterance timing stays internal and is derived from sample position in this append-only buffer
    /// - future implementations will decide internally when enough audio exists
    ///   to run inference and construct transient token slices for cutting
    pub fn feed(&mut self, samples: Vec<f32>) -> FeedOutput<'_> {
        self.audio.extend_samples(samples);
        self.feed_count += 1;

        let plan = self.plan_preview_decode();
        self.rewrite_preview(plan.rollback_to, plan.decoder_position);
        if let Some(decoded) = self
            .decode_preview(&plan)
            .unwrap_or_else(|e| panic!("bee-roll preview decode failed: {e}"))
        {
            self.apply_decoded_preview(decoded);
        }
        if let Some(preview_run) = self
            .build_preview_run()
            .unwrap_or_else(|e| panic!("bee-roll preview enrichment failed: {e}"))
        {
            self.apply_preview_run(&preview_run);
        }
        self.update_preview_from();
        if self.apply_cut_if_any() {
            self.zipa_cache = None;
        }

        FeedOutput::new(self.tape.tokens(), self.tape.detected_language())
    }

    /// Current stable token slice.
    ///
    /// Invariant:
    /// - this is the prefix `[0, stable_through)`
    fn stable_tokens(&self) -> &[OutputToken] {
        self.tape.slice(UtteranceTokenRange::new(
            TokenIndex::new(0),
            self.stable_through,
        ))
    }

    /// Current carry token slice.
    ///
    /// Invariant:
    /// - this is the bridge `[stable_through, preview_from)`
    fn carry_tokens(&self) -> &[OutputToken] {
        self.tape.slice(UtteranceTokenRange::new(
            self.stable_through,
            self.preview_from,
        ))
    }

    /// Current preview token slice.
    ///
    /// Invariant:
    /// - this is the live tail `[preview_from, tape.end())`
    fn preview_tokens(&self) -> &[OutputToken] {
        self.tape
            .slice(UtteranceTokenRange::new(self.preview_from, self.tape.end()))
    }

    /// Replace the current stable/carry/preview partition after a cut.
    ///
    /// Invariant:
    /// - `stable_through <= preview_from <= tape.end()`
    fn set_stable_and_preview(&mut self, stable_through: TokenIndex, preview_from: TokenIndex) {
        assert!(
            stable_through <= preview_from,
            "stable must not exceed preview start"
        );
        assert!(
            preview_from <= self.tape.end(),
            "preview start must lie within the tape"
        );
        self.stable_through = stable_through;
        self.preview_from = preview_from;
    }

    /// Rewind the live tail so the next decode pass can regenerate preview from
    /// the current preview boundary.
    ///
    /// Intent:
    /// - the preview suffix is provisional and can be discarded wholesale
    /// - stable/carry boundaries remain intact
    /// - `stable` stays in KV, `carry` is replayed in the prompt, and only
    ///   `preview` is regenerated
    ///
    /// WARNING:
    /// - do not "simplify" this by rewinding KV to `preview_from`
    /// - do not apply cuts unless audio rotation matches the README model
    /// - the README is the source of truth here, not ad hoc experiments
    ///
    /// Orientation:
    /// - if `stable_through == 0`, then the retained decoder position is also 0
    /// - that is the "no stable prefix yet" case, not a different decode mode
    fn plan_preview_decode(&self) -> PreviewDecodePlan {
        PreviewDecodePlan {
            rollback_to: self.preview_from,
            decoder_position: self.tape.decoder_position_for_boundary(self.stable_through),
            rewrite_budget_tokens: self.preview_rewrite_tokens,
        }
    }

    /// Rewind the live tail so the next decode pass can regenerate preview from
    /// the chosen rollback boundary.
    ///
    /// Invariants:
    /// - `rollback_to` must not precede `preview_from`
    /// - `rollback_to` must lie within the current tape
    fn rewrite_preview(&mut self, rollback_to: TokenIndex, decoder_position: usize) {
        assert!(
            rollback_to >= self.preview_from,
            "preview rewrites must not cut into carry"
        );
        assert!(
            rollback_to <= self.tape.end(),
            "preview rewrite boundary must lie within the tape"
        );
        self.tape.truncate_to(rollback_to, decoder_position);
    }

    /// Apply one decode pass that regenerated preview from the current
    /// rollback boundary.
    ///
    /// Intent:
    /// - token append + KV advance stay coupled inside `Tape`
    /// - detected language is updated at the same point as the decode result
    /// - persistent-KV mode adopts the model-reported next decoder position
    fn apply_decoded_preview(&mut self, decoded: DecodedPreview) {
        self.tape.set_detected_language(decoded.detected_language);
        self.tape.append_decoded(
            decoded.tokens,
            decoded.prompt_end_position,
            decoded.next_decoder_position,
        );
    }

    fn decode_preview(
        &mut self,
        plan: &PreviewDecodePlan,
    ) -> anyhow::Result<Option<DecodedPreview>> {
        // Desired state:
        // - `carry` is replayed as raw token IDs from the tape
        // - no decode/re-tokenize round trip is ever allowed here
        // - the prompt choice below is driven only by the retained decoder
        //   position from `plan`, not by feed count and not by whether a cut
        //   "just happened"
        let carry_prompt_tokens = self
            .carry_tokens()
            .iter()
            .map(|token| token.timed_token().token().as_u32() as i32)
            .collect::<Vec<_>>();
        let Some(asr) = self.asr.as_mut() else {
            return Ok(None);
        };
        if self.audio.is_empty() {
            return Ok(None);
        }

        let (mel_data, n_mels, n_frames) = asr
            .mel_extractor
            .extract(self.audio.samples())
            .map_err(|e| anyhow::anyhow!("extracting mel features: {e}"))?;
        let mel = Array::from_slice(&mel_data, &[n_mels as i32, n_frames as i32]);
        let audio_features = asr
            .model
            .encode_incremental(&mel, &mut asr.encoder_cache)
            .map_err(|e| anyhow::anyhow!("encoding incremental audio: {e}"))?;
        let audio_features = ops::expand_dims(&audio_features, 0)
            .map_err(|e| anyhow::anyhow!("adding batch axis to audio features: {e}"))?;
        audio_features
            .eval()
            .map_err(|e| anyhow::anyhow!("evaluating audio features: {e}"))?;
        let n_audio_tokens = audio_features.shape()[1] as usize;

        let prompt = Self::build_prompt(plan, asr, n_audio_tokens, &carry_prompt_tokens);
        let (generated, confidences, next_decoder_position, _) = self
            .tape
            .prefill_and_decode(
                asr.model.as_ref(),
                &prompt,
                &audio_features,
                Self::preview_max_new_tokens_for_samples(
                    self.audio.sample_count().as_usize(),
                    asr.max_new_tokens,
                ),
                ConfidenceMode::Streaming,
            )
            .map_err(|e| anyhow::anyhow!("prefill/decode preview: {e}"))?;
        let prompt_end_position = next_decoder_position.saturating_sub(generated.len());

        Ok(Some(DecodedPreview {
            detected_language: Some(asr.language.clone()),
            tokens: self.output_tokens_from_generated(&generated, &confidences),
            prompt_end_position,
            next_decoder_position,
        }))
    }

    fn build_prompt(
        plan: &PreviewDecodePlan,
        asr: &AsrRuntime,
        n_audio_tokens: usize,
        carry_prompt_tokens: &[i32],
    ) -> Vec<i32> {
        // Desired state:
        // - `stable` survives in retained KV state
        // - `carry` is replayed as token IDs sliced directly from the tape
        // - `preview` is generated after that prompt
        //
        // Forbidden:
        // - decoding carry to text
        // - re-tokenizing carry text
        // - storing carry as String anywhere in bee-roll
        //
        // Orientation:
        // - retained decoder position 0 => initial prompt
        // - retained decoder position > 0 => follow-up prompt
        // - that rule stays true even when the chosen cut is effectively 0 and
        //   nothing has been promoted yet
        let mut prompt = if plan.decoder_position == 0 {
            // Initial prompt emits metadata and audio structure only.
            // Carry replay is appended below as token IDs so the prompt
            // builder never needs string context.
            generate::build_initial_prompt(
                n_audio_tokens,
                asr.language.as_str(),
                "",
                crate::tokenizer(),
            )
        } else {
            // Follow-up prompt assumes the stable prefix is already in KV.
            // Only the bridge (`carry`) is appended after this base prompt,
            // then preview is regenerated.
            generate::build_followup_prompt(
                n_audio_tokens,
                asr.language.as_str(),
                crate::tokenizer(),
            )
        };
        prompt.extend_from_slice(carry_prompt_tokens);
        prompt
    }

    fn output_tokens_from_generated(
        &self,
        generated: &[i32],
        confidences: &[TokenConfidence],
    ) -> Vec<OutputToken> {
        let start_index = self.tape.end().as_usize();
        let anchor = self.audio.utterance_range().end;
        generated
            .iter()
            .enumerate()
            .map(|(offset, &token_id)| {
                let confidence = confidences.get(offset).map(Self::map_token_confidence);
                OutputToken::new(
                    TimedToken::new(
                        TokenIndex::new(start_index + offset),
                        TokenId::new(token_id as u32),
                        SampleRange::new(anchor, anchor),
                    ),
                    confidence,
                    None,
                    Vec::new(),
                    Vec::new(),
                    crate::ZipaTiming::Invalid,
                )
            })
            .collect()
    }

    fn preview_max_new_tokens_for_samples(sample_count: usize, max_new_tokens: usize) -> usize {
        let scaled = sample_count
            .saturating_mul(PREVIEW_TOKENS_PER_SECOND)
            .saturating_add(crate::SAMPLE_RATE as usize - 1)
            / crate::SAMPLE_RATE as usize;
        let budget = PREVIEW_MAX_NEW_TOKENS_BASE.saturating_add(scaled);
        budget.max(MIN_PREVIEW_MAX_NEW_TOKENS).min(max_new_tokens)
    }

    fn map_token_confidence(confidence: &TokenConfidence) -> AsrTokenConfidence {
        let alternatives = confidence.top_ids[..confidence.alternative_count as usize]
            .iter()
            .zip(confidence.top_logits.iter())
            .map(|(&token, &logit)| AsrTokenAlternative::new(TokenId::new(token as u32), logit))
            .collect();
        AsrTokenConfidence::new(confidence.concentration, confidence.margin, alternatives)
    }

    fn build_preview_run(&mut self) -> anyhow::Result<Option<PreviewRun>> {
        // Hard model invariant from the README:
        // - transcript text must exist
        // - G2P must exist
        // - ZIPA must exist
        // Nothing meaningful happens without those three components.
        //
        // Stable tokens are treated as frozen here. Only the mutable seam
        // (`carry` + `preview`, i.e. `[stable_through, tape.end())`) is
        // re-decoded, re-phoneticized, and re-aligned on each feed.
        let Some(_) = self.phonetics.as_ref() else {
            return Ok(None);
        };
        let seam_start = self.stable_through.as_usize();
        let seam_tokens = self.tape.slice(UtteranceTokenRange::new(
            self.stable_through,
            self.tape.end(),
        ));
        if seam_tokens.is_empty() {
            return Ok(None);
        }
        let token_ids = seam_tokens
            .iter()
            .map(|token| token.timed_token().token())
            .collect::<Vec<_>>();

        let transcript = crate::decode_token_ids(&token_ids)?;
        let analysis = {
            let phonetics = self
                .phonetics
                .as_mut()
                .expect("phonetics presence checked above");
            phonetics
                .g2p
                .analyze_text(&transcript, phonetics.lang_code.as_str())?
        };
        let token_piece_phones = token_piece_phones(&analysis);
        let alignment_input = transcript_alignment_input(&analysis);
        let comparison_input = transcript_comparison_input_from_g2p(&transcript, &alignment_input);
        let current_audio_end = self.audio.utterance_range().end;
        let zipa_output = {
            let audio = &self.audio;
            let cache = (self.stable_through > TokenIndex::new(0))
                .then_some(self.zipa_cache.as_ref())
                .flatten();
            let phonetics = self
                .phonetics
                .as_ref()
                .expect("phonetics presence checked above");
            infer_zipa_output_for_current_audio(audio, cache, &phonetics.zipa, current_audio_end)?
        };
        self.zipa_cache = Some(ZipaPreviewCache {
            audio_end: current_audio_end,
            output: zipa_output.clone(),
        });
        let alignment = TranscriptAlignment::build_from_cached_zipa(comparison_input, zipa_output);
        let timings = alignment.token_piece_timings(&alignment_input.token_pieces);

        let token_enrichments = token_piece_phones
            .into_iter()
            .zip(timings)
            .map(|(phones, timing)| PreviewTokenEnrichment {
                token_index: seam_start + phones.token_index,
                g2p_ipa: CompactString::from(phones.ipa_text),
                transcript_phones: phones
                    .normalized_phones
                    .into_iter()
                    .map(|phone| ComparisonPhone::new(CompactString::from(phone)))
                    .collect(),
                zipa_timing: Self::map_zipa_timing(timing.timing),
            })
            .collect();

        Ok(Some(PreviewRun {
            transcript,
            token_enrichments,
        }))
    }

    fn apply_preview_run(&mut self, preview_run: &PreviewRun) {
        let _ = &preview_run.transcript;
        for enrichment in &preview_run.token_enrichments {
            if let Some(token) = self.tape.token_mut(enrichment.token_index) {
                token.set_g2p_ipa(Some(enrichment.g2p_ipa.clone()));
                token.set_transcript_phones(enrichment.transcript_phones.clone());
                token.set_zipa_timing(enrichment.zipa_timing.clone());
            }
        }
    }

    fn map_zipa_timing(timing: ComparisonRangeTiming) -> ZipaTiming {
        match timing {
            ComparisonRangeTiming::Invalid => ZipaTiming::Invalid,
            ComparisonRangeTiming::Deleted { projected_at } => ZipaTiming::Deleted { projected_at },
            ComparisonRangeTiming::NoTiming { projected_range } => ZipaTiming::Projected {
                normalized_start: projected_range.start,
                normalized_end: projected_range.end,
            },
            ComparisonRangeTiming::Aligned(timed) => ZipaTiming::Aligned(TimeRange::new(
                crate::UtteranceTime::from_secs(timed.start_time_secs),
                crate::UtteranceTime::from_secs(timed.end_time_secs),
            )),
        }
    }

    fn update_preview_from(&mut self) {
        let target = self
            .tape
            .end()
            .as_usize()
            .saturating_sub(self.preview_rewrite_tokens);
        let ids = self
            .tape
            .tokens()
            .iter()
            .map(|token| token.timed_token().token().as_u32())
            .collect::<Vec<_>>();
        self.preview_from = TokenIndex::new(find_word_start_at_or_before(
            &ids,
            self.stable_through.as_usize(),
            target,
        ));
        tracing::trace!(
            stable_through = self.stable_through.as_usize(),
            tape_end = self.tape.end().as_usize(),
            target,
            preview_from = self.preview_from.as_usize(),
            "bee_roll.update_preview_from"
        );
    }

    fn apply_cut_if_any(&mut self) -> bool {
        // Invariant:
        // - boundary 0 is not a special mode
        // - `Cutting::Never` is just the policy that selects boundary 0
        // - once a boundary has been selected, this one cut-application path must
        //   handle 0 and nonzero boundaries alike
        // - the rest of the machine derives naturally from the chosen boundary:
        //   retained decoder position, prompt choice, audio rotation, and tape
        //   partitioning must not branch on "zero means no cut"
        let new_stable = match self.cutting {
            Cutting::Never => TokenIndex::new(0),
            Cutting::Auto => self.find_auto_cut_boundary().unwrap_or(self.stable_through),
        };
        let cut_context = self.cut_context_debug(new_stable);
        tracing::trace!(
            cutting = ?self.cutting,
            stable_through = self.stable_through.as_usize(),
            preview_from = self.preview_from.as_usize(),
            chosen_boundary = new_stable.as_usize(),
            context = %cut_context,
            "bee_roll.apply_cut_if_any.choose"
        );
        if new_stable <= self.stable_through {
            return false;
        }
        let Some(cut_sample) = self.audio_cut_sample_for_boundary(new_stable) else {
            tracing::trace!(
                boundary = new_stable.as_usize(),
                context = %self.cut_context_debug(new_stable),
                "bee_roll.apply_cut_if_any.no_audio_cut_sample"
            );
            return false;
        };
        self.rotate_audio_to(cut_sample);
        self.set_stable_and_preview(new_stable, self.preview_from);
        tracing::trace!(
            stable_through = self.stable_through.as_usize(),
            preview_from = self.preview_from.as_usize(),
            cut_sample = cut_sample.as_usize(),
            context = %self.cut_context_debug(new_stable),
            "bee_roll.apply_cut_if_any.applied"
        );
        true
    }

    fn find_auto_cut_boundary(&self) -> Option<TokenIndex> {
        // Desired strategy:
        // - search in token space only
        // - use the tokenizer's streaming decoder over the actual generated token
        //   IDs to detect word ends
        // - stop as soon as the latest legal cut boundary has been identified
        // - do not decode tokens one-by-one into standalone strings
        // - do not re-tokenize transcript text to rediscover boundaries
        // - do not treat boundary 0 as special; it is just another candidate
        //   boundary and the policy may legitimately return it
        if self.preview_from <= self.stable_through {
            return None;
        }
        let ids = self
            .tape
            .tokens()
            .iter()
            .map(|token| token.timed_token().token().as_u32())
            .collect::<Vec<_>>();
        let latest_legal_boundary = self
            .preview_from
            .as_usize()
            .saturating_sub(self.preview_rewrite_tokens);
        let boundary = find_latest_word_end_at_or_before(
            &ids,
            self.stable_through.as_usize(),
            self.preview_from.as_usize(),
            latest_legal_boundary,
        );
        tracing::trace!(
            stable_through = self.stable_through.as_usize(),
            preview_from = self.preview_from.as_usize(),
            latest_legal_boundary,
            chosen_boundary = boundary,
            "bee_roll.find_auto_cut_boundary"
        );
        (boundary > self.stable_through.as_usize()).then_some(TokenIndex::new(boundary))
    }

    fn audio_cut_sample_for_boundary(&self, boundary: TokenIndex) -> Option<SampleOffset> {
        let end = boundary.as_usize().min(self.tape.tokens().len());
        for token in self.tape.tokens()[..end].iter().rev() {
            if let ZipaTiming::Aligned(range) = token.zipa_timing() {
                let sample = (range.end.as_secs() * crate::SAMPLE_RATE as f64).round() as usize;
                return Some(SampleOffset::new(sample));
            }
        }
        None
    }

    fn rotate_audio_to(&mut self, cut_sample: SampleOffset) {
        let current_start = self.audio.utterance_range().start;
        if cut_sample <= current_start {
            return;
        }
        let drop = SampleCount::new(cut_sample.as_usize() - current_start.as_usize());
        self.audio.drop_front(drop);
        if let Some(asr) = self.asr.as_mut() {
            asr.encoder_cache = EncoderCache::new();
        }
    }

    fn cut_context_debug(&self, boundary: TokenIndex) -> String {
        let tokens = self.tape.tokens();
        let ids = tokens
            .iter()
            .map(|token| token.timed_token().token().as_u32())
            .collect::<Vec<_>>();
        let transcript = crate::decode_token_ids(
            &tokens
                .iter()
                .map(|token| token.timed_token().token())
                .collect::<Vec<_>>(),
        )
        .unwrap_or_else(|e| format!("<decode-error:{e}>"));
        let word_spans = streamed_word_spans(&ids, self.stable_through.as_usize(), tokens.len())
            .into_iter()
            .map(|(start, end, text)| format!("{start}..{end}:{text:?}"))
            .collect::<Vec<_>>()
            .join(" | ");
        let start = boundary.as_usize().saturating_sub(3);
        let end = (boundary.as_usize() + 3).min(tokens.len());
        format!(
            "transcript={transcript:?}; stable={} preview_from={} boundary={} tape_end={}; window=[{}]; words=[{}]",
            self.stable_through.as_usize(),
            self.preview_from.as_usize(),
            boundary.as_usize(),
            tokens.len(),
            debug_token_window(tokens, start, end),
            word_spans,
        )
    }
}

fn infer_zipa_output_for_current_audio(
    audio: &AudioBuffer,
    existing_cache: Option<&ZipaPreviewCache>,
    zipa: &ZipaInference,
    current_audio_end: SampleOffset,
) -> anyhow::Result<CachedZipaOutput> {
    let Some(cache) = existing_cache else {
        let zipa_audio = ZipaAudioBuffer {
            samples: audio.samples().to_vec(),
            sample_rate_hz: crate::SAMPLE_RATE,
        };
        return infer_cached_zipa_output(&zipa_audio, zipa, 0, 0.0)
            .map_err(|e| anyhow::anyhow!("running ZIPA inference: {e}"));
    };

    if cache.audio_end == current_audio_end {
        return Ok(cache.output.clone());
    }
    if cache.audio_end > current_audio_end {
        let zipa_audio = ZipaAudioBuffer {
            samples: audio.samples().to_vec(),
            sample_rate_hz: crate::SAMPLE_RATE,
        };
        return infer_cached_zipa_output(&zipa_audio, zipa, 0, 0.0)
            .map_err(|e| anyhow::anyhow!("running ZIPA inference: {e}"));
    }

    let tail_audio = audio.slice(SampleRange::new(cache.audio_end, current_audio_end));
    let tail_offset_secs =
        tail_audio.utterance_range().start.as_usize() as f64 / crate::SAMPLE_RATE as f64;
    let tail_audio = ZipaAudioBuffer {
        samples: tail_audio.samples().to_vec(),
        sample_rate_hz: crate::SAMPLE_RATE,
    };
    let mut output = cache.output.clone();
    let tail = infer_cached_zipa_output(
        &tail_audio,
        zipa,
        output.raw_token_count(),
        tail_offset_secs,
    )
    .map_err(|e| anyhow::anyhow!("running ZIPA inference on tail: {e}"))?;
    output.append(tail);
    Ok(output)
}

fn debug_token_window(tokens: &[OutputToken], start: usize, end: usize) -> String {
    tokens[start.min(tokens.len())..end.min(tokens.len())]
        .iter()
        .map(|token| debug_token(token))
        .collect::<Vec<_>>()
        .join(" | ")
}

fn debug_token(token: &OutputToken) -> String {
    let timed = token.timed_token();
    let surface = timed
        .token()
        .decode()
        .unwrap_or_else(|e| format!("<decode-error:{e}>"));
    let g2p = token
        .g2p_ipa()
        .map(|ipa| ipa.as_str().to_owned())
        .unwrap_or_else(|| "-".to_owned());
    let phones = if token.transcript_phones().is_empty() {
        "-".to_owned()
    } else {
        token
            .transcript_phones()
            .iter()
            .map(|phone| phone.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    };
    format!(
        "#{} id={} txt={surface:?} g2p={g2p} phones={phones} zipa={}",
        timed.index().as_usize(),
        timed.token().as_u32(),
        format_zipa_timing(token.zipa_timing())
    )
}

fn format_zipa_timing(timing: &ZipaTiming) -> String {
    match timing {
        ZipaTiming::Aligned(range) => {
            format!("{:.3}..{:.3}", range.start.as_secs(), range.end.as_secs())
        }
        ZipaTiming::Deleted { projected_at } => format!("del@{projected_at}"),
        ZipaTiming::Projected {
            normalized_start,
            normalized_end,
        } => format!("proj {normalized_start}..{normalized_end}"),
        ZipaTiming::Invalid => "invalid".to_owned(),
    }
}

fn streamed_word_spans(
    ids: &[u32],
    start_boundary: usize,
    end_boundary: usize,
) -> Vec<(usize, usize, String)> {
    let end_boundary = end_boundary.min(ids.len());
    if start_boundary >= end_boundary {
        return Vec::new();
    }
    let mut stream = crate::tokenizer().decode_stream(true);
    let mut pending_start = start_boundary;
    let mut current_start = start_boundary;
    let mut current_text = String::new();
    let mut out = Vec::new();

    for (i, &id) in ids
        .iter()
        .enumerate()
        .take(end_boundary)
        .skip(start_boundary)
    {
        match stream
            .step(id)
            .unwrap_or_else(|e| panic!("decode stream failed: {e}"))
        {
            None => {}
            Some(chunk) => {
                let starts_new_word = chunk.starts_with(' ') || chunk.starts_with('\n');
                if starts_new_word {
                    if !current_text.is_empty() {
                        out.push((
                            current_start,
                            pending_start,
                            std::mem::take(&mut current_text),
                        ));
                    }
                    current_start = pending_start;
                    current_text.push_str(chunk.trim_start());
                } else {
                    current_text.push_str(&chunk);
                }
                pending_start = i + 1;
            }
        }
    }

    if !current_text.is_empty() {
        out.push((current_start, end_boundary, current_text));
    }

    out
}

fn find_word_start_at_or_before(ids: &[u32], start_boundary: usize, target: usize) -> usize {
    if ids.is_empty() {
        return 0;
    }
    if target <= start_boundary {
        return start_boundary;
    }
    let target = target.min(ids.len());
    let mut stream = crate::tokenizer().decode_stream(true);
    let mut last_word_start = start_boundary;
    let mut pending_start = start_boundary.min(ids.len());

    for (i, &id) in ids.iter().enumerate().skip(start_boundary) {
        match stream
            .step(id)
            .unwrap_or_else(|e| panic!("decode stream failed: {e}"))
        {
            None => {}
            Some(chunk) => {
                if chunk.starts_with(' ') || chunk.starts_with('\n') {
                    tracing::trace!(
                        start_boundary,
                        target,
                        token_index = i,
                        pending_start,
                        chunk = %chunk.escape_debug(),
                        "bee_roll.find_word_start_at_or_before.word_start"
                    );
                    if pending_start <= target {
                        last_word_start = pending_start;
                    } else {
                        break;
                    }
                }
                pending_start = i + 1;
            }
        }
    }

    last_word_start
}

fn find_latest_word_end_at_or_before(
    ids: &[u32],
    start_boundary: usize,
    end_boundary: usize,
    target_boundary: usize,
) -> usize {
    if start_boundary >= end_boundary || ids.is_empty() {
        return start_boundary;
    }
    let end_boundary = end_boundary.min(ids.len());
    let target_boundary = target_boundary.min(end_boundary);
    if target_boundary <= start_boundary {
        return start_boundary;
    }
    let mut stream = crate::tokenizer().decode_stream(true);
    let mut last_word_end = start_boundary;
    let mut pending_start = start_boundary;

    for (i, &id) in ids
        .iter()
        .enumerate()
        .take(end_boundary)
        .skip(start_boundary)
    {
        match stream
            .step(id)
            .unwrap_or_else(|e| panic!("decode stream failed: {e}"))
        {
            None => {}
            Some(chunk) => {
                if chunk.starts_with(' ') || chunk.starts_with('\n') {
                    tracing::trace!(
                        start_boundary,
                        end_boundary,
                        target_boundary,
                        token_index = i,
                        pending_start,
                        chunk = %chunk.escape_debug(),
                        "bee_roll.find_latest_word_end_at_or_before.word_boundary"
                    );
                    if pending_start <= target_boundary {
                        last_word_end = pending_start;
                    } else {
                        break;
                    }
                }
                pending_start = i + 1;
            }
        }
    }

    last_word_end
}

#[cfg(test)]
mod tests {
    use super::{Cutting, DecodedPreview, Utterance};
    use crate::{
        ComparisonPhone, OutputToken, SampleOffset, SampleRange, TimedToken, TokenId, TokenIndex,
        ZipaTiming,
    };
    use compact_str::CompactString;
    use std::path::PathBuf;
    use std::sync::Once;

    static TEST_TOKENIZER: Once = Once::new();

    #[test]
    fn new_utterance_starts_with_empty_stable_carry_preview() {
        let utterance = Utterance::new(2, Cutting::Never);
        assert!(utterance.stable_tokens().is_empty());
        assert!(utterance.carry_tokens().is_empty());
        assert!(utterance.preview_tokens().is_empty());
    }

    #[test]
    fn feed_rewrites_preview_from_preview_boundary() {
        ensure_test_tokenizer();
        let token_ids = encode_ids("I asked Copilot");
        let mut utterance = Utterance::new(2, Cutting::Never);
        utterance.tape.append_decoded(
            token_ids
                .iter()
                .enumerate()
                .map(|(index, &token_id)| dummy_output_token(index, token_id))
                .collect(),
            0,
            4,
        );
        utterance.set_stable_and_preview(TokenIndex::new(1), TokenIndex::new(3));

        let output_len = {
            let output = utterance.feed(vec![0.0; 320]);
            output.tokens().len()
        };

        assert_eq!(utterance.stable_tokens().len(), 1);
        assert_eq!(utterance.carry_tokens().len(), 0);
        assert_eq!(utterance.preview_tokens().len(), 2);
        assert_eq!(output_len, 3);
        assert_eq!(utterance.tape.decoder_position(), 1);
    }

    #[test]
    fn apply_decoded_preview_appends_tokens_and_updates_language() {
        let mut utterance = Utterance::new(2, Cutting::Never);
        utterance
            .tape
            .append_decoded(vec![dummy_output_token(0, 10)], 0, 1);
        utterance.set_stable_and_preview(TokenIndex::new(0), TokenIndex::new(1));
        utterance.rewrite_preview(TokenIndex::new(1), 1);

        utterance.apply_decoded_preview(DecodedPreview {
            detected_language: Some(CompactString::from("English")),
            tokens: vec![dummy_output_token(1, 11), dummy_output_token(2, 12)],
            prompt_end_position: 1,
            next_decoder_position: 3,
        });

        assert_eq!(utterance.tape.tokens().len(), 3);
        assert_eq!(utterance.tape.decoder_position(), 3);
        assert_eq!(utterance.tape.detected_language(), Some("English"));
        assert_eq!(utterance.preview_tokens().len(), 2);
    }

    #[test]
    fn preview_token_budget_scales_with_audio_length() {
        assert_eq!(Utterance::preview_max_new_tokens_for_samples(3_200, 256), 4);
        assert_eq!(
            Utterance::preview_max_new_tokens_for_samples(16_000, 256),
            10
        );
        assert_eq!(
            Utterance::preview_max_new_tokens_for_samples(320_000, 12),
            12
        );
    }

    fn dummy_output_token(index: usize, token_id: u32) -> OutputToken {
        OutputToken::new(
            TimedToken::new(
                TokenIndex::new(index),
                TokenId::new(token_id),
                SampleRange::new(
                    SampleOffset::new(index * 160),
                    SampleOffset::new((index + 1) * 160),
                ),
            ),
            None,
            None,
            Vec::<ComparisonPhone>::new(),
            Vec::new(),
            ZipaTiming::Invalid,
        )
    }

    fn ensure_test_tokenizer() {
        TEST_TOKENIZER.call_once(|| {
            let path =
                PathBuf::from(std::env::var("BEE_TOKENIZER_PATH").unwrap_or_else(|_| {
                    panic!("BEE_TOKENIZER_PATH must be set for bee-roll tests")
                }));
            crate::init_tokenizer(&path);
        });
    }

    fn encode_ids(text: &str) -> Vec<u32> {
        crate::tokenizer()
            .encode(text, false)
            .unwrap_or_else(|e| panic!("encoding {text:?}: {e}"))
            .get_ids()
            .to_vec()
    }
}
