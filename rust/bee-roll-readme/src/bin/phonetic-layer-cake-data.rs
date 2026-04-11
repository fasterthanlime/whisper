use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::ops::Range;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use bee_correct::g2p::CachedEspeakG2p;
use bee_phonetic::{
    align_token_sequences_with_left_word_boundaries, normalize_ipa_for_comparison,
    normalize_ipa_for_comparison_with_spans, sentence_word_tokens,
};
use bee_qwen3_asr::tokenizers::Tokenizer;
use bee_transcribe::zipa_align::{
    raw_slice_for_normalized_range, select_segmental_word_windows,
    timed_range_for_normalized_range, transcript_token_range_for_span, transcript_word_raw_ranges,
};
use bee_zipa_mlx::audio::load_wav_mono_f32;
use bee_zipa_mlx::infer::ZipaInference;
use serde_json::{Value, json};

fn main() -> Result<()> {
    let args = parse_args()?;
    let recording = load_recording_examples(&recording_examples_path())?
        .into_iter()
        .find(|row| row.term.eq_ignore_ascii_case(&args.term))
        .ok_or_else(|| anyhow!("no recording example for term '{}'", args.term))?;
    let text = if args.use_transcript {
        recording.transcript.clone()
    } else {
        recording.text.clone()
    };

    let tokenizer_path = env::var("BEE_TOKENIZER_PATH")
        .map(PathBuf::from)
        .context("missing BEE_TOKENIZER_PATH; run with direnv")?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow!("loading {}: {e}", tokenizer_path.display()))?;
    let encoding = tokenizer
        .encode(text.as_str(), false)
        .map_err(|e| anyhow!("encoding text: {e}"))?;

    let tokens = encoding
        .get_ids()
        .iter()
        .zip(encoding.get_tokens())
        .zip(encoding.get_offsets())
        .enumerate()
        .map(|(index, ((id, label), (start, end)))| {
            json!({
                "index": index,
                "id": id,
                "label": label,
                "char_start": start,
                "char_end": end,
                "surface": text.get(*start..*end).unwrap_or_default(),
            })
        })
        .collect::<Vec<_>>();

    let words = sentence_word_tokens(&text);
    let text_segments = text_segments(&text, &words, encoding.get_offsets());

    let inference = ZipaInference::load_reference_small_no_diacritics()?;
    let wav = wav_path_for(&default_wav_dir(), &recording.audio_path);
    let audio = load_wav_mono_f32(&wav)?;
    let utterance = inference.infer_wav_greedy(&wav)?;
    let duration = audio.samples.len() as f64 / audio.sample_rate_hz as f64;
    let phone_spans = utterance
        .derive_phone_spans(&inference.tokens, duration, 0)
        .into_iter()
        .filter(|span| span.token != "▁")
        .collect::<Vec<_>>();
    let zipa_raw = utterance
        .tokens
        .into_iter()
        .filter(|token| token != "▁")
        .collect::<Vec<_>>();
    let zipa_norm_with_spans = normalize_ipa_for_comparison_with_spans(&zipa_raw);
    let zipa_norm = zipa_norm_with_spans
        .iter()
        .map(|token| token.token.clone())
        .collect::<Vec<_>>();

    let mut g2p = CachedEspeakG2p::english(&g2p_base_dir())?;
    let word_raw_ranges =
        transcript_word_raw_ranges(&mut g2p, &text).map_err(|e| anyhow!("g2p word ranges: {e}"))?;
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
        .flat_map(|(word_index, (_, tokens))| std::iter::repeat_n(word_index, tokens.len()))
        .collect::<Vec<_>>();
    let alignment =
        align_token_sequences_with_left_word_boundaries(&transcript_norm, &zipa_norm, &word_ids);
    let transcript_token_ranges = (0..word_norm_ranges.len())
        .map(|word_index| {
            transcript_token_range_for_span(&word_norm_ranges, word_index, word_index + 1)
        })
        .collect::<Vec<_>>();
    let transcript_word_tokens = word_norm_ranges
        .iter()
        .map(|(_, tokens)| tokens.clone())
        .collect::<Vec<_>>();
    let word_windows = select_segmental_word_windows(
        &transcript_word_tokens,
        &transcript_token_ranges,
        &zipa_norm,
        &alignment,
    );

    let words_json = words
        .iter()
        .enumerate()
        .map(|(word_index, word)| {
            let (token_start, token_end) =
                token_span_for_char_range(word.char_start..word.char_end, encoding.get_offsets());
            let g2p_raw = word_raw_ranges
                .get(word_index)
                .map(|(_, raw)| raw.clone())
                .unwrap_or_default();
            let g2p_normalized = word_norm_ranges
                .get(word_index)
                .map(|(_, tokens)| tokens.clone())
                .unwrap_or_default();
            let window = word_windows.get(word_index).and_then(|window| window.as_ref());
            let (zipa_normalized, zipa_raw_word, audio, alignment_ops) = match window {
                Some(window) => {
                    let zipa_normalized = zipa_norm
                        .get(window.zipa_norm_range.clone())
                        .unwrap_or(&[])
                        .to_vec();
                    let zipa_raw_word = raw_slice_for_normalized_range(
                        &zipa_raw,
                        &zipa_norm_with_spans,
                        window.zipa_norm_range.clone(),
                    );
                    let timing = timed_range_for_normalized_range(
                        &zipa_norm_with_spans,
                        &phone_spans,
                        window.zipa_norm_range.clone(),
                    );
                    let audio = timing.map(|timing| {
                        json!({
                            "start": timing.start_time_secs,
                            "end": timing.end_time_secs,
                            "label": format!("{:.2}-{:.2}s", timing.start_time_secs, timing.end_time_secs),
                        })
                    });
                    let alignment_ops = window
                        .ops
                        .iter()
                        .map(|op| {
                            if op.left_index.is_some() && op.right_index.is_some() {
                                if op.left_token == op.right_token { "|" } else { "x" }
                            } else if op.left_index.is_some() {
                                "<"
                            } else {
                                ">"
                            }
                        })
                        .collect::<Vec<_>>()
                        .join("");
                    (zipa_normalized, zipa_raw_word, audio, alignment_ops)
                }
                None => (Vec::new(), Vec::new(), None, String::new()),
            };
            json!({
                "word": word.text,
                "char_start": word.char_start,
                "char_end": word.char_end,
                "token_start": token_start,
                "token_end": token_end,
                "g2p_raw": g2p_raw,
                "g2p_normalized": g2p_normalized,
                "zipa_normalized": zipa_normalized,
                "zipa_raw": zipa_raw_word,
                "audio": audio,
                "alignment": alignment_ops,
            })
        })
        .collect::<Vec<_>>();

    let json = json!({
        "term": recording.term,
        "sentence": text,
        "tokenizer_path": tokenizer_path,
        "wav": wav,
        "tokens": tokens,
        "text_segments": text_segments,
        "words": words_json,
    });
    println!("{}", serde_json::to_string_pretty(&json)?);
    Ok(())
}

struct Args {
    term: String,
    use_transcript: bool,
}

fn parse_args() -> Result<Args> {
    let mut args = env::args().skip(1);
    let mut term = None::<String>;
    let mut use_transcript = true;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--term" => term = Some(args.next().context("--term requires a value")?),
            "--text" => use_transcript = false,
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(anyhow!("unexpected argument: {other}")),
        }
    }

    Ok(Args {
        term: term.unwrap_or_else(|| "serde".to_string()),
        use_transcript,
    })
}

fn print_usage() {
    eprintln!("usage: phonetic-layer-cake-data [--term TERM] [--text]");
}

#[derive(Clone)]
struct RecordingExample {
    term: String,
    text: String,
    audio_path: String,
    transcript: String,
}

fn recording_examples_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../data/phonetic-seed/recording_examples.jsonl")
}

fn default_wav_dir() -> PathBuf {
    let local = PathBuf::from("/Users/amos/bearcove/bee/data/phonetic-seed/audio-wav");
    if local.is_dir() {
        return local;
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/phonetic-seed/audio-wav")
}

fn wav_path_for(wav_dir: &Path, audio_path: &str) -> PathBuf {
    let stem = Path::new(audio_path)
        .file_stem()
        .expect("recording audio path has a stem");
    wav_dir.join(format!("{}.wav", stem.to_string_lossy()))
}

fn g2p_base_dir() -> PathBuf {
    std::env::temp_dir().join("bee-roll-espeak")
}

fn load_recording_examples(path: &Path) -> Result<Vec<RecordingExample>> {
    let file = File::open(path)?;
    let mut rows = Vec::new();
    for (line_idx, line) in BufReader::new(file).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(&line)
            .map_err(|error| anyhow!("parse {} line {}: {error}", path.display(), line_idx + 1))?;
        rows.push(RecordingExample {
            term: value
                .get("term")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("missing term in {} line {}", path.display(), line_idx + 1))?
                .to_string(),
            text: value
                .get("text")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("missing text in {} line {}", path.display(), line_idx + 1))?
                .to_string(),
            audio_path: value
                .get("audio_path")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    anyhow!(
                        "missing audio_path in {} line {}",
                        path.display(),
                        line_idx + 1
                    )
                })?
                .to_string(),
            transcript: value
                .get("transcript")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    anyhow!(
                        "missing transcript in {} line {}",
                        path.display(),
                        line_idx + 1
                    )
                })?
                .to_string(),
        });
    }
    Ok(rows)
}

fn token_span_for_char_range(
    char_range: Range<usize>,
    offsets: &[(usize, usize)],
) -> (usize, usize) {
    let indices = offsets
        .iter()
        .enumerate()
        .filter_map(|(index, (start, end))| {
            (start < &char_range.end && end > &char_range.start).then_some(index)
        })
        .collect::<Vec<_>>();
    match (indices.first(), indices.last()) {
        (Some(start), Some(end)) => (*start, end + 1),
        _ => (0, 0),
    }
}

fn text_segments(
    text: &str,
    words: &[bee_phonetic::SentenceWordToken],
    offsets: &[(usize, usize)],
) -> Vec<Value> {
    let mut out = Vec::new();
    let mut cursor = 0usize;
    for word in words {
        push_gap_segment(&mut out, text, cursor..word.char_start, offsets);
        let (token_start, token_end) =
            token_span_for_char_range(word.char_start..word.char_end, offsets);
        out.push(json!({
            "label": word.text,
            "token_start": token_start,
            "token_end": token_end,
        }));
        cursor = word.char_end;
    }
    push_gap_segment(&mut out, text, cursor..text.len(), offsets);
    out
}

fn push_gap_segment(
    out: &mut Vec<Value>,
    text: &str,
    gap: Range<usize>,
    offsets: &[(usize, usize)],
) {
    if gap.start >= gap.end {
        return;
    }
    let raw = text.get(gap.clone()).unwrap_or_default();
    let Some(first_non_ws) = raw.char_indices().find(|(_, ch)| !ch.is_whitespace()) else {
        return;
    };
    let Some(last_non_ws) = raw.char_indices().rfind(|(_, ch)| !ch.is_whitespace()) else {
        return;
    };
    let trimmed_start = gap.start + first_non_ws.0;
    let trimmed_end = gap.start + last_non_ws.0 + last_non_ws.1.len_utf8();
    let slice = text.get(trimmed_start..trimmed_end).unwrap_or_default();
    if slice.is_empty() {
        return;
    }
    let (token_start, token_end) = token_span_for_char_range(trimmed_start..trimmed_end, offsets);
    out.push(json!({
        "label": slice,
        "token_start": token_start,
        "token_end": token_end,
    }));
}
