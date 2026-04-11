fn suffix_after_prefix<'a>(prefix: Option<&str>, text: &'a str) -> &'a str {
    let Some(prefix) = prefix else {
        return text;
    };
    if let Some(suffix) = text.strip_prefix(prefix) {
        suffix
    } else {
        text
    }
}

struct TimedGeneratedPrefix {
    kept_word_count: usize,
    kept_token_count: usize,
}

#[derive(Clone, Debug)]
struct TimedWord {
    text: String,
    char_range: std::ops::Range<usize>,
    start_secs: f64,
    end_secs: f64,
}

struct TimedGeneratedBridge {
    kept_word_count: usize,
    kept_token_count: usize,
    kept_text: String,
    bridge: CarriedBridge,
}

fn timed_aligned_words_for_alignment(
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
            text: word_timing.word.to_string(),
            char_range: word_range.char_start..word_range.char_end,
            start_secs,
            end_secs,
        });
    }

    Ok(timed_words)
}

fn timed_generated_prefix_for_cut(
    align_ctx: &mut AlignmentContext,
    tokenizer: &Tokenizer,
    chunk_run: &ChunkRun,
    chunk_samples: &[f32],
    keep_until_secs: f64,
) -> Result<TimedGeneratedPrefix> {
    let transcript = normalized_transcript(&chunk_run.transcript);
    if transcript.is_empty() {
        return Ok(TimedGeneratedPrefix {
            kept_word_count: 0,
            kept_token_count: 0,
        });
    }

    let alignment = build_transcript_alignment(align_ctx, transcript, chunk_samples)?;
    let timed_words = timed_aligned_words_for_alignment(transcript, &alignment)?;
    let kept_word_count = timed_words
        .iter()
        .take_while(|word| word.end_secs <= keep_until_secs)
        .count();

    let kept_text = if kept_word_count == 0 {
        String::new()
    } else {
        let end = timed_words
            .get(kept_word_count - 1)
            .map(|word| word.char_range.end)
            .ok_or_else(|| anyhow::anyhow!("missing word range for kept prefix"))?;
        transcript[..end].to_string()
    };

    let kept_token_count = if kept_text.is_empty() {
        0
    } else {
        tokenizer
            .encode_fast(kept_text.as_str(), false)
            .map_err(|e| anyhow::anyhow!("encoding kept prefix: {e}"))?
            .len()
    };

    Ok(TimedGeneratedPrefix {
        kept_word_count,
        kept_token_count,
    })
}

fn timed_generated_bridge_for_cuts(
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
            kept_word_count: 0,
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
                    text: word.text.clone(),
                    token_range: token_start..token_end,
                    start_secs: (word.start_secs - keep_until_secs).max(0.0),
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
        kept_word_count,
        kept_token_count,
        kept_text,
        bridge,
    })
}

fn adjust_keep_boundary_secs(
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

struct AlignmentContext {
    g2p: CachedEspeakG2p,
    zipa: ZipaInference,
}

impl AlignmentContext {
    fn new() -> Result<Self> {
        Ok(Self {
            g2p: CachedEspeakG2p::english(&g2p_base_dir()).context("initializing g2p engine")?,
            zipa: ZipaInference::load_quantized_bundle_dir(&zipa_bundle_dir()?)
                .context("loading ZIPA bundle")?,
        })
    }
}

fn build_transcript_alignment(
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

fn format_span_timing(span_timing: SpanTiming) -> String {
    match span_timing {
        SpanTiming::Aligned {
            start_secs,
            end_secs,
        } => format!("{{{start_secs:.2}-{end_secs:.2}s}}"),
        SpanTiming::PartialGap {
            start_secs,
            end_secs,
        } => format!("{{partial {start_secs:.2}-{end_secs:.2}s}}"),
        SpanTiming::NoAlignedWords => "{no-aligned-words}".to_string(),
        SpanTiming::NoTiming => "{no-timing}".to_string(),
    }
}

fn g2p_base_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target")
}

fn zipa_bundle_dir() -> Result<PathBuf> {
    if let Ok(path) = env::var("BEE_ZIPA_BUNDLE_DIR") {
        return Ok(PathBuf::from(path));
    }

    let home = env::var("HOME").context("HOME is not set for ZIPA fallback path")?;
    Ok(PathBuf::from(home).join("bearcove/zipa-mlx-hf"))
}
