use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::ops::Range;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use bee_g2p::{BeeG2p, token_piece_phones, transcript_alignment_input, transcript_words};
use bee_phonetic::{
    align_token_sequences, normalize_ipa_for_comparison_with_spans, sentence_word_tokens,
};
use bee_qwen3_asr::tokenizers::Tokenizer;
use bee_transcribe::zipa_align::{
    ComparisonRangeTiming, TranscriptAlignment, raw_slice_for_normalized_range,
    transcript_comparison_input_from_g2p,
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

    let sentence_words = sentence_word_tokens(&text);
    let text_segments = text_segments(&text, &sentence_words, encoding.get_offsets());

    let inference = ZipaInference::load_reference_small_no_diacritics()?;
    let wav = wav_path_for(&default_wav_dir(), &recording.audio_path);
    let audio = load_wav_mono_f32(&wav)?;
    let utterance = inference.infer_wav_greedy(&wav)?;
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

    let mut g2p = BeeG2p::load_default()?;
    let analysis = g2p.analyze_text(&text, "eng-us")?;
    let chars_input = transcript_alignment_input(&analysis);
    let piece_phones = token_piece_phones(&analysis);
    let alignment = TranscriptAlignment::build_from_comparison_input(
        transcript_comparison_input_from_g2p(&text, &chars_input),
        &audio,
        &inference,
    )
    .map_err(|e| anyhow!("building transcript alignment: {e}"))?;
    let piece_timings = alignment.token_piece_timings(&chars_input.token_pieces);

    if analysis.token_piece_spans.len() != chars_input.token_pieces.len()
        || analysis.token_piece_spans.len() != piece_phones.len()
        || analysis.token_piece_spans.len() != piece_timings.len()
    {
        return Err(anyhow!(
            "token piece pipeline length mismatch: spans={} ranges={} phones={} timings={}",
            analysis.token_piece_spans.len(),
            chars_input.token_pieces.len(),
            piece_phones.len(),
            piece_timings.len()
        ));
    }

    let token_pieces_json = analysis
        .token_piece_spans
        .iter()
        .zip(chars_input.token_pieces.iter())
        .zip(piece_phones.iter())
        .zip(piece_timings.iter())
        .map(|(((piece, piece_range), piece_phone), timing)| {
            let projected = alignment
                .projected_comparison_range(piece_range.comparison_start..piece_range.comparison_end);
            let (zipa_normalized, zipa_raw_piece, audio_json, projection_label) =
                match &timing.timing {
                    ComparisonRangeTiming::Aligned(timed) => {
                        let audio = json!({
                            "start": timed.start_time_secs,
                            "end": timed.end_time_secs,
                            "label": format!("{:.2}-{:.2}s", timed.start_time_secs, timed.end_time_secs),
                        });
                        let zipa_normalized = zipa_norm
                            .get(timed.normalized_range.clone())
                            .unwrap_or(&[])
                            .to_vec();
                        let zipa_raw_piece = raw_slice_for_normalized_range(
                            &zipa_raw,
                            &zipa_norm_with_spans,
                            timed.normalized_range.clone(),
                        );
                        (
                            zipa_normalized,
                            zipa_raw_piece,
                            Some(audio),
                            format!("{}..{}", timed.normalized_range.start, timed.normalized_range.end),
                        )
                    }
                    ComparisonRangeTiming::Deleted { projected_at } => {
                        (Vec::new(), Vec::new(), None, format!("del@{projected_at}"))
                    }
                    ComparisonRangeTiming::NoTiming { projected_range } => {
                        let zipa_normalized = zipa_norm
                            .get(projected_range.clone())
                            .unwrap_or(&[])
                            .to_vec();
                        let zipa_raw_piece = raw_slice_for_normalized_range(
                            &zipa_raw,
                            &zipa_norm_with_spans,
                            projected_range.clone(),
                        );
                        (
                            zipa_normalized,
                            zipa_raw_piece,
                            None,
                            format!("nt:{}..{}", projected_range.start, projected_range.end),
                        )
                    }
                    ComparisonRangeTiming::Invalid => {
                        (Vec::new(), Vec::new(), None, "invalid".to_owned())
                    }
                };
            let ops_label = align_token_sequences(&piece_phone.normalized_phones, &zipa_normalized)
                .ops
                .iter()
                .map(|op| {
                    if op.left_index.is_some() && op.right_index.is_some() {
                        if op.left_token == op.right_token {
                            "|"
                        } else {
                            "x"
                        }
                    } else if op.left_index.is_some() {
                        "<"
                    } else {
                        ">"
                    }
                })
                .collect::<Vec<_>>()
                .join("");
            Ok(json!({
                "word": piece.word_surface,
                "token_index": piece.token_index,
                "token": piece.token,
                "surface": piece.token_surface,
                "char_start": piece.token_char_start,
                "char_end": piece.token_char_end,
                "token_start": piece.token_index,
                "token_end": piece.token_index + 1,
                "comparison_start": piece_range.comparison_start,
                "comparison_end": piece_range.comparison_end,
                "projected_start": projected.as_ref().map(|range| range.start),
                "projected_end": projected.as_ref().map(|range| range.end),
                "g2p_raw": piece_phone.ipa_tokens,
                "g2p_normalized": piece_phone.normalized_phones,
                "zipa_normalized": zipa_normalized,
                "zipa_raw": zipa_raw_piece,
                "audio": audio_json,
                "proj": projection_label,
                "ops": ops_label,
            }))
        })
        .collect::<Result<Vec<_>>>()?;

    let words_json = transcript_words(&text)
        .iter()
        .map(|word| {
            let (token_start, token_end) =
                token_span_for_char_range(word.char_start..word.char_end, encoding.get_offsets());
            json!({
                "word": word.word,
                "char_start": word.char_start,
                "char_end": word.char_end,
                "token_start": token_start,
                "token_end": token_end,
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
        "token_pieces": token_pieces_json,
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
