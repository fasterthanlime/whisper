use crate::tokens::UtteranceTokenRange;
use crate::{
    AsrTokenAlternative, AsrTokenConfidence, AudioBuffer, ComparisonPhone, Cut, FeedOutput,
    OutputToken, SampleOffset, SampleRange, Tape, TimeRange, TimedToken, TokenId, TokenIndex,
    ZipaTiming,
};
use bee_g2p::{BeeG2p, token_piece_phones, transcript_alignment_input};
use bee_qwen3_asr::encoder::EncoderCache;
use bee_qwen3_asr::generate::{self, ConfidenceMode, TokenConfidence};
use bee_qwen3_asr::mel::MelExtractor;
use bee_qwen3_asr::mlx_rs::Array;
use bee_qwen3_asr::mlx_rs::error::Exception;
use bee_qwen3_asr::mlx_rs::ops;
use bee_qwen3_asr::model::Qwen3ASRModel;
use bee_transcribe::zipa_align::{
    ComparisonRangeTiming, TranscriptAlignment, transcript_comparison_input_from_g2p,
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
enum DecodeMode {
    RebuildPromptEachFeed,
    PersistentKv,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PreviewDecodePlan {
    rollback_to: TokenIndex,
    decoder_position: usize,
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

/// Policy hook that decides where to cut a ready chunk.
///
/// Intent:
/// - the cutter owns the small amount of business logic that chooses a cut
/// - the cutter never mutates utterance state directly
pub trait Cutter {
    /// Chooses a cut for `tokens`.
    ///
    /// Invariant:
    /// - returned cuts must refer to utterance-global token coordinates
    fn cut(&mut self, tokens: &[OutputToken]) -> Cut;
}

/// Observer hook for utterance lifecycle events.
///
/// Intent:
/// - side effects, logging, debug capture, and inspection live here
/// - event methods are intentionally deferred until the debug/HTML contract exists
pub trait Listener {}

/// Streaming utterance state for the next rollback model.
///
/// Intent:
/// - audio is the only public ingress
/// - utterance audio is append-only and remains anchored at utterance sample 0
/// - utterance-owned token, commit, and ASR state stay synchronized under one owner
/// - inference and chunk construction happen inside this type
/// - cut application happens inside this type
/// - the moving stable/carry/preview boundaries are tracked in token space
/// - external policy is delegated to [`Cutter`]
/// - external observation is delegated to [`Listener`]
///
/// Non-goals:
/// - this scaffold does not yet implement decode scheduling or cut application
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
    /// - tokens before this boundary are `stable`
    /// - any sample/time boundaries are derived from this token boundary, not stored separately
    stable_through: TokenIndex,

    /// Token boundary through which the utterance is retained as `carry`.
    ///
    /// Invariant:
    /// - `stable_through <= carry_through <= tape.end()`
    /// - tokens in `[stable_through, carry_through)` are the replay bridge
    /// - tokens in `[carry_through, tape.end())` are the current `preview`
    carry_through: TokenIndex,

    /// Canonical token-aligned output tape plus synchronized ASR rollback state.
    tape: Tape,

    /// How many tail tokens the next feed may rewrite before a new cut is chosen.
    preview_rewrite_tokens: usize,

    /// Streaming decode behavior for this utterance.
    decode_mode: DecodeMode,

    /// Number of feed steps processed so far.
    feed_count: usize,

    /// Boxed cut policy used by this utterance.
    cutter: Box<dyn Cutter>,

    /// Boxed event sink used by this utterance.
    listener: Box<dyn Listener>,

    /// Optional ASR runtime used to decode preview on each feed step.
    asr: Option<AsrRuntime>,

    /// Optional phonetic/timing runtime used to enrich the current preview.
    phonetics: Option<PhoneticRuntime>,
}

impl Utterance {
    // Boxed trait objects are intentional here. We accept the cost and do not want
    // further reminders to genericize or optimize this construction path.
    /// Creates a new utterance with empty audio and all token boundaries at 0.
    pub fn new(num_layers: usize, cutter: Box<dyn Cutter>, listener: Box<dyn Listener>) -> Self {
        Self {
            audio: AudioBuffer::new(SampleOffset::new(0), Vec::new()),
            stable_through: TokenIndex::new(0),
            carry_through: TokenIndex::new(0),
            tape: Tape::new(num_layers),
            preview_rewrite_tokens: DEFAULT_PREVIEW_REWRITE_TOKENS,
            decode_mode: DecodeMode::PersistentKv,
            feed_count: 0,
            cutter,
            listener,
            asr: None,
            phonetics: None,
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
        // TODO: optionally run cutter and promote stable/carry boundaries

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
    /// - this is the bridge `[stable_through, carry_through)`
    fn carry_tokens(&self) -> &[OutputToken] {
        self.tape.slice(UtteranceTokenRange::new(
            self.stable_through,
            self.carry_through,
        ))
    }

    /// Current preview token slice.
    ///
    /// Invariant:
    /// - this is the live tail `[carry_through, tape.end())`
    fn preview_tokens(&self) -> &[OutputToken] {
        self.tape.slice(UtteranceTokenRange::new(
            self.carry_through,
            self.tape.end(),
        ))
    }

    /// Replace the current carry/preview partition after a cut.
    ///
    /// Invariant:
    /// - `stable_through <= carry_through <= tape.end()`
    fn set_stable_and_carry(&mut self, stable_through: TokenIndex, carry_through: TokenIndex) {
        assert!(
            stable_through <= carry_through,
            "stable must not exceed carry"
        );
        assert!(
            carry_through <= self.tape.end(),
            "carry must lie within the tape"
        );
        self.stable_through = stable_through;
        self.carry_through = carry_through;
    }

    /// Rewind the live tail so the next decode pass can regenerate preview from
    /// the current carry boundary.
    ///
    /// Intent:
    /// - the preview suffix is provisional and can be discarded wholesale
    /// - stable/carry boundaries remain intact
    /// - persistent-KV mode tracks the visible decoder position at the carry cut
    fn plan_preview_decode(&self) -> PreviewDecodePlan {
        PreviewDecodePlan {
            rollback_to: self.carry_through,
            decoder_position: self.tape.decoder_position_for_boundary(self.carry_through),
            rewrite_budget_tokens: self.preview_rewrite_tokens,
        }
    }

    /// Rewind the live tail so the next decode pass can regenerate preview from
    /// the chosen rollback boundary.
    ///
    /// Invariants:
    /// - `rollback_to` must not precede `carry_through`
    /// - `rollback_to` must lie within the current tape
    fn rewrite_preview(&mut self, rollback_to: TokenIndex, decoder_position: usize) {
        assert!(
            rollback_to >= self.carry_through,
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
        let carry_context = if matches!(self.decode_mode, DecodeMode::RebuildPromptEachFeed) {
            Some(
                crate::decode_token_ids(
                    &self
                        .carry_tokens()
                        .iter()
                        .map(|token| token.timed_token().token())
                        .collect::<Vec<_>>(),
                )
                .map_err(|e| anyhow::anyhow!("decoding carry tokens: {e}"))?,
            )
        } else {
            None
        };

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

        let prompt = Self::build_prompt(
            self.decode_mode,
            plan,
            asr,
            n_audio_tokens,
            carry_context.as_deref(),
        )?;
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
        decode_mode: DecodeMode,
        plan: &PreviewDecodePlan,
        asr: &AsrRuntime,
        n_audio_tokens: usize,
        carry_context: Option<&str>,
    ) -> Result<Vec<i32>, Exception> {
        Ok(match decode_mode {
            DecodeMode::PersistentKv => {
                if plan.decoder_position == 0 {
                    generate::build_initial_prompt(
                        n_audio_tokens,
                        asr.language.as_str(),
                        "",
                        crate::tokenizer(),
                    )
                } else {
                    generate::build_followup_prompt(
                        n_audio_tokens,
                        asr.language.as_str(),
                        crate::tokenizer(),
                    )
                }
            }
            DecodeMode::RebuildPromptEachFeed => generate::build_initial_prompt(
                n_audio_tokens,
                asr.language.as_str(),
                carry_context.unwrap_or(""),
                crate::tokenizer(),
            ),
        })
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
        let Some(phonetics) = self.phonetics.as_mut() else {
            return Ok(None);
        };
        let token_ids = self
            .tape
            .tokens()
            .iter()
            .map(|token| token.timed_token().token())
            .collect::<Vec<_>>();
        if token_ids.is_empty() {
            return Ok(None);
        }

        let transcript = crate::decode_token_ids(&token_ids)?;
        let analysis = phonetics
            .g2p
            .analyze_text(&transcript, phonetics.lang_code.as_str())?;
        let token_piece_phones = token_piece_phones(&analysis);
        let alignment_input = transcript_alignment_input(&analysis);
        let comparison_input = transcript_comparison_input_from_g2p(&transcript, &alignment_input);
        let zipa_audio = ZipaAudioBuffer {
            samples: self.audio.samples().to_vec(),
            sample_rate_hz: crate::SAMPLE_RATE,
        };
        let alignment = TranscriptAlignment::build_from_comparison_input(
            comparison_input,
            &zipa_audio,
            &phonetics.zipa,
        )
        .map_err(|e| anyhow::anyhow!("building transcript alignment: {e}"))?;
        let timings = alignment.token_piece_timings(&alignment_input.token_pieces);

        let token_enrichments = token_piece_phones
            .into_iter()
            .zip(timings)
            .map(|(phones, timing)| PreviewTokenEnrichment {
                token_index: phones.token_index,
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
}

#[cfg(test)]
mod tests {
    use super::{Cutter, DecodedPreview, Listener, Utterance};
    use crate::{
        ComparisonPhone, Cut, OutputToken, SampleOffset, SampleRange, TimedToken, TokenId,
        TokenIndex, ZipaTiming,
    };
    use compact_str::CompactString;

    struct NoCut;
    impl Cutter for NoCut {
        fn cut(&mut self, _tokens: &[crate::OutputToken]) -> Cut {
            Cut::NoCut
        }
    }

    struct NullListener;
    impl Listener for NullListener {}

    #[test]
    fn new_utterance_starts_with_empty_stable_carry_preview() {
        let utterance = Utterance::new(2, Box::new(NoCut), Box::new(NullListener));
        assert!(utterance.stable_tokens().is_empty());
        assert!(utterance.carry_tokens().is_empty());
        assert!(utterance.preview_tokens().is_empty());
    }

    #[test]
    fn feed_rewrites_preview_from_carry_boundary() {
        let mut utterance = Utterance::new(2, Box::new(NoCut), Box::new(NullListener));
        utterance.tape.append_decoded(
            vec![
                dummy_output_token(0, 10),
                dummy_output_token(1, 11),
                dummy_output_token(2, 12),
                dummy_output_token(3, 13),
            ],
            0,
            4,
        );
        utterance.set_stable_and_carry(TokenIndex::new(1), TokenIndex::new(3));

        let output_len = {
            let output = utterance.feed(vec![0.0; 320]);
            output.tokens().len()
        };

        assert_eq!(utterance.stable_tokens().len(), 1);
        assert_eq!(utterance.carry_tokens().len(), 2);
        assert_eq!(utterance.preview_tokens().len(), 0);
        assert_eq!(output_len, 3);
        assert_eq!(utterance.tape.decoder_position(), 3);
    }

    #[test]
    fn apply_decoded_preview_appends_tokens_and_updates_language() {
        let mut utterance = Utterance::new(2, Box::new(NoCut), Box::new(NullListener));
        utterance
            .tape
            .append_decoded(vec![dummy_output_token(0, 10)], 0, 1);
        utterance.set_stable_and_carry(TokenIndex::new(0), TokenIndex::new(1));
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
}
