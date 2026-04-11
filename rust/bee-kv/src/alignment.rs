use std::env;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use bee_phonetic::sentence_word_tokens;
use bee_transcribe::g2p::CachedEspeakG2p;
use bee_transcribe::zipa_align::TranscriptAlignment;
use bee_zipa_mlx::audio::AudioBuffer as ZipaAudioBuffer;
use bee_zipa_mlx::infer::ZipaInference;

use bee_qwen3_asr::tokenizers::Tokenizer;

use crate::decode::tokenize_token_ids;
use crate::types::*;
use crate::{KEEP_BOUNDARY_MIN_KEPT_SECS, SAMPLE_RATE};

/// Returns `text` with `prefix` stripped from the front, or unchanged if no match.
pub(crate) fn suffix_after_prefix<'a>(prefix: Option<&str>, text: &'a str) -> &'a str {
    let Some(prefix) = prefix else {
        return text;
    };
    if let Some(suffix) = text.strip_prefix(prefix) {
        suffix
    } else {
        text
    }
}

/// A word from a transcript with character offsets and ZIPA-derived timing.
#[derive(Clone, Debug)]
pub(crate) struct TimedWord {
    /// Character range this word occupies in the combined transcript.
    pub(crate) char_range: std::ops::Range<usize>,
    /// Start time in seconds (relative to the chunk/window start).
    pub(crate) start_secs: f64,
    /// End time in seconds (relative to the chunk/window start).
    pub(crate) end_secs: f64,
}

/// Result of computing a bridge cut: kept prefix plus a bridge to carry forward.
pub(crate) struct TimedGeneratedBridge {
    /// Number of tokens spanning the kept generated words.
    pub(crate) kept_token_count: usize,
    /// Full kept text (including carried prefix).
    pub(crate) kept_text: String,
    /// Bridge tokens and words to carry into the next window's prompt.
    pub(crate) bridge: CarriedBridge,
}

/// Extracts aligned word timings from a ZIPA transcript alignment.
///
/// Stops at the first word that lacks aligned timing (e.g. `NoWindow` or `NoTiming`).
pub(crate) fn timed_aligned_words_for_alignment(
    transcript: &str,
    alignment: &TranscriptAlignment,
) -> Result<Vec<TimedWord>> {
    let word_ranges = sentence_word_tokens(transcript);
    let word_timings = alignment.word_timings();
    if word_ranges.len() != word_timings.len() {
        bail!(
            "alignment word count mismatch: transcript has {} words, alignment has {}",
            word_ranges.len(),
            word_timings.len()
        );
    }

    let mut timed_words = Vec::with_capacity(word_ranges.len());
    for (word_range, word_timing) in word_ranges.iter().zip(word_timings.iter()) {
        let bee_transcribe::zipa_align::AlignmentQuality::Aligned {
            start_secs,
            end_secs,
        } = word_timing.quality
        else {
            break;
        };

        timed_words.push(TimedWord {
            char_range: word_range.char_start..word_range.char_end,
            start_secs,
            end_secs,
        });
    }

    Ok(timed_words)
}

/// Computes a bridge cut: splits the transcript into kept prefix + bridge region.
///
/// The bridge region spans from `keep_until_secs` to `replay_until_secs` and is
/// tokenized into a `CarriedBridge` that gets replayed in the next window's prompt.
pub(crate) fn timed_generated_bridge_for_cuts(
    tokenizer: &Tokenizer,
    combined_transcript: &str,
    replayed_prefix: Option<&CarriedBridge>,
    timed_words: &[TimedWord],
    keep_until_secs: f64,
    replay_until_secs: f64,
    chosen_word: Option<&BoundaryWordDebug>,
) -> Result<TimedGeneratedBridge> {
    if timed_words.is_empty() {
        let bridge = replayed_prefix.cloned().unwrap_or_else(|| CarriedBridge {
            token_ids: Vec::new(),
            text: String::new(),
            words: Vec::new(),
        });
        return Ok(TimedGeneratedBridge {
            kept_token_count: 0,
            kept_text: String::new(),
            bridge,
        });
    }

    let carried_word_count = replayed_prefix
        .map(|prefix| {
            if prefix.words.is_empty() {
                sentence_word_tokens(&prefix.text).len()
            } else {
                prefix.words.len()
            }
        })
        .unwrap_or(0);
    let generated_words = &timed_words[carried_word_count.min(timed_words.len())..];
    let chosen_word_index = chosen_word.map(|chosen| chosen.word_index);
    let full_kept_word_count = chosen_word_index.map(|index| index + 1).unwrap_or_else(|| {
        timed_words
            .iter()
            .take_while(|word| word.end_secs <= keep_until_secs)
            .count()
    });
    let kept_word_count = full_kept_word_count.saturating_sub(carried_word_count);
    let bridge_start_index = full_kept_word_count;
    let bridge_end_index = timed_words.partition_point(|word| word.start_secs < replay_until_secs);
    let bridge_words_slice = if bridge_start_index <= bridge_end_index {
        &timed_words[bridge_start_index..bridge_end_index]
    } else {
        &timed_words[0..0]
    };

    let kept_text = timed_words
        .get(full_kept_word_count.saturating_sub(1))
        .map(|end_word| combined_transcript[..end_word.char_range.end].to_string())
        .unwrap_or_default();

    let bridge = if let (Some(first_bridge_word), Some(last_bridge_word)) =
        (bridge_words_slice.first(), bridge_words_slice.last())
    {
        let mut bridge_char_base = first_bridge_word.char_range.start;
        while bridge_char_base > 0
            && combined_transcript.as_bytes()[bridge_char_base - 1].is_ascii_whitespace()
        {
            bridge_char_base -= 1;
        }
        let text =
            combined_transcript[bridge_char_base..last_bridge_word.char_range.end].to_string();
        let token_ids = tokenize_token_ids(tokenizer, &text)?;
        let bridge_words = bridge_words_slice
            .iter()
            .map(|word| -> Result<CarriedBridgeWord> {
                let relative_start = word.char_range.start.saturating_sub(bridge_char_base);
                let relative_end = word.char_range.end.saturating_sub(bridge_char_base);
                let token_start = tokenize_token_ids(tokenizer, &text[..relative_start])?.len();
                let token_end = tokenize_token_ids(tokenizer, &text[..relative_end])?.len();
                Ok(CarriedBridgeWord {
                    token_range: token_start..token_end,
                    end_secs: (word.end_secs - keep_until_secs).max(0.0),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        CarriedBridge {
            token_ids,
            text,
            words: bridge_words,
        }
    } else if bridge_words_slice.is_empty() {
        CarriedBridge {
            token_ids: Vec::new(),
            text: String::new(),
            words: Vec::new(),
        }
    } else {
        unreachable!()
    };

    let generated_kept_text = if kept_word_count == 0 {
        String::new()
    } else {
        let generated_start = generated_words
            .first()
            .map(|word| word.char_range.start)
            .ok_or_else(|| anyhow::anyhow!("missing generated word start"))?;
        let generated_end = generated_words
            .get(kept_word_count - 1)
            .map(|word| word.char_range.end)
            .ok_or_else(|| anyhow::anyhow!("missing generated kept word range"))?;
        combined_transcript[generated_start..generated_end].to_string()
    };

    let kept_token_count = if generated_kept_text.is_empty() {
        0
    } else {
        tokenizer
            .encode_fast(generated_kept_text.as_str(), false)
            .map_err(|e| anyhow::anyhow!("encoding kept bridge prefix: {e}"))?
            .len()
    };

    Ok(TimedGeneratedBridge {
        kept_token_count,
        kept_text,
        bridge,
    })
}

/// Adjusts the keep boundary based on the selected policy.
///
/// For `NearestWordEnd`, searches ZIPA word timings for the word end closest
/// to `target_keep_until_secs` (without exceeding it) within the valid range.
/// For `Fixed`, returns the target unchanged.
pub(crate) fn adjust_keep_boundary_secs(
    policy: KeepBoundaryPolicy,
    alignment: &TranscriptAlignment,
    target_keep_until_secs: f64,
    replay_until_secs: f64,
) -> Result<(f64, KeepBoundaryDebug)> {
    let fixed_debug = KeepBoundaryDebug {
        earliest_candidate_secs: target_keep_until_secs,
        min_keep_secs: target_keep_until_secs,
        snapped: false,
        chosen_word: None,
    };
    if policy == KeepBoundaryPolicy::Fixed {
        return Ok((target_keep_until_secs, fixed_debug));
    }

    let min_keep_secs = KEEP_BOUNDARY_MIN_KEPT_SECS.min(target_keep_until_secs);
    let earliest_candidate_secs = min_keep_secs;
    let mut best_candidate = None;
    let mut best_distance = f64::INFINITY;

    for (word_index, word_timing) in alignment.word_timings().iter().enumerate() {
        if let bee_transcribe::zipa_align::AlignmentQuality::Aligned {
            start_secs,
            end_secs,
        } = word_timing.quality
        {
            if end_secs <= 0.0 || end_secs >= replay_until_secs {
                continue;
            }
            if end_secs > target_keep_until_secs || end_secs < earliest_candidate_secs {
                continue;
            }
            let distance = (end_secs - target_keep_until_secs).abs();
            if distance < best_distance {
                best_distance = distance;
                best_candidate = Some(BoundaryWordDebug {
                    word_index,
                    text: word_timing.word.to_string(),
                    start_secs,
                    end_secs,
                });
            }
        }
    }

    let chosen_word = best_candidate;
    let keep_until_secs = chosen_word
        .as_ref()
        .map(|word| word.end_secs)
        .unwrap_or(target_keep_until_secs);
    let debug = KeepBoundaryDebug {
        earliest_candidate_secs,
        min_keep_secs,
        snapped: chosen_word.is_some(),
        chosen_word,
    };

    Ok((keep_until_secs, debug))
}

/// Holds ZIPA alignment resources (G2P engine and acoustic model).
pub(crate) struct AlignmentContext {
    /// Cached espeak-based grapheme-to-phoneme engine.
    g2p: CachedEspeakG2p,
    /// ZIPA forced-alignment inference model.
    pub(crate) zipa: ZipaInference,
}

impl AlignmentContext {
    /// Initializes a new alignment context, loading G2P and ZIPA models.
    pub(crate) fn new() -> Result<Self> {
        Ok(Self {
            g2p: CachedEspeakG2p::english(&g2p_base_dir()).context("initializing g2p engine")?,
            zipa: ZipaInference::load_quantized_bundle_dir(&zipa_bundle_dir()?)
                .context("loading ZIPA bundle")?,
        })
    }
}

/// Runs ZIPA forced alignment on a transcript against audio samples.
pub(crate) fn build_transcript_alignment(
    align_ctx: &mut AlignmentContext,
    transcript: &str,
    samples: &[f32],
) -> Result<TranscriptAlignment> {
    let zipa_audio = ZipaAudioBuffer {
        samples: samples.to_vec(),
        sample_rate_hz: SAMPLE_RATE,
    };

    TranscriptAlignment::build(transcript, &zipa_audio, &mut align_ctx.g2p, &align_ctx.zipa)
        .map_err(|error| anyhow::anyhow!(error.to_string()))
}

/// Returns the base directory for espeak G2P data (under the cargo target dir).
pub(crate) fn g2p_base_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target")
}

/// Returns the ZIPA model bundle directory, from `$BEE_ZIPA_BUNDLE_DIR` or a fallback.
pub(crate) fn zipa_bundle_dir() -> Result<PathBuf> {
    if let Ok(path) = env::var("BEE_ZIPA_BUNDLE_DIR") {
        return Ok(PathBuf::from(path));
    }

    let home = env::var("HOME").context("HOME is not set for ZIPA fallback path")?;
    Ok(PathBuf::from(home).join("bearcove/zipa-mlx-hf"))
}
