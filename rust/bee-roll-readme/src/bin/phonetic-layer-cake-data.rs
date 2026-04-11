use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::ops::Range;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use bee_g2p_charsiu::{
    CharsiuSidecarClient, PhonemizeTextRequest, ProbeRequest, TranscriptAlignmentInput,
    TranscriptTokenPieceComparisonRange, TranscriptWordComparisonRange, probe_text_default,
    token_piece_phones, transcript_alignment_input,
};
use bee_phonetic::{
    normalize_ipa_for_comparison, normalize_ipa_for_comparison_with_spans, parse_reviewed_ipa,
    sentence_word_tokens,
};
use bee_qwen3_asr::tokenizers::Tokenizer;
use bee_transcribe::zipa_align::{
    ComparisonRangeTiming, TranscriptAlignment, raw_slice_for_normalized_range,
};
use bee_zipa_mlx::audio::load_wav_mono_f32;
use bee_zipa_mlx::infer::ZipaInference;
use serde_json::{Value, json};

#[derive(Clone)]
struct SentenceTokenMeta {
    index: usize,
    label: String,
    char_start: usize,
    char_end: usize,
    surface: String,
}

#[derive(Clone)]
struct PieceRow {
    word: String,
    token_index: usize,
    token: String,
    surface: String,
    char_start: usize,
    char_end: usize,
    comparison_start: usize,
    comparison_end: usize,
    g2p_raw: Vec<String>,
    g2p_normalized: Vec<String>,
}

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

    let sentence_tokens = encoding
        .get_ids()
        .iter()
        .zip(encoding.get_tokens())
        .zip(encoding.get_offsets())
        .enumerate()
        .map(|(index, ((_id, label), (start, end)))| SentenceTokenMeta {
            index,
            label: format!("{label}"),
            char_start: *start,
            char_end: *end,
            surface: text.get(*start..*end).unwrap_or_default().to_owned(),
        })
        .collect::<Vec<_>>();

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

    let mut charsiu = CharsiuSidecarClient::spawn_default()
        .map_err(|e| anyhow!("spawning Charsiu sidecar: {e}"))?;
    let word_ipas = charsiu
        .phonemize_text(PhonemizeTextRequest {
            text: text.clone(),
            lang_code: "eng-us".to_owned(),
        })
        .map_err(|e| anyhow!("phonemizing text with Charsiu: {e}"))?;

    if word_ipas.word_ipas.len() != words.len() {
        return Err(anyhow!(
            "word count mismatch: sentence_word_tokens={} charsiu={}",
            words.len(),
            word_ipas.word_ipas.len()
        ));
    }

    let mut normalized = Vec::new();
    let mut word_ranges = Vec::new();
    let mut token_piece_ranges = Vec::new();
    let mut piece_rows = Vec::new();

    for (word_index, (word, word_ipa)) in words.iter().zip(word_ipas.word_ipas.iter()).enumerate() {
        let (token_start, token_end) =
            token_span_for_char_range(word.char_start..word.char_end, encoding.get_offsets());
        if token_start >= token_end {
            continue;
        }

        let word_cmp_start = normalized.len();
        if token_end - token_start == 1 {
            let token = sentence_tokens
                .get(token_start)
                .ok_or_else(|| anyhow!("missing sentence token {}", token_start))?;
            let raw = parse_reviewed_ipa(&word_ipa.ipa);
            let norm = normalize_ipa_for_comparison(&raw);
            let comparison_start = normalized.len();
            normalized.extend(norm.iter().cloned());
            let comparison_end = normalized.len();

            token_piece_ranges.push(TranscriptTokenPieceComparisonRange {
                token_index: token.index,
                token: token.label.clone(),
                token_surface: token.surface.clone(),
                token_char_start: token.char_start,
                token_char_end: token.char_end,
                word_index: Some(word_index),
                word_surface: Some(word.text.clone()),
                comparison_start,
                comparison_end,
            });
            piece_rows.push(PieceRow {
                word: word.text.clone(),
                token_index: token.index,
                token: token.label.clone(),
                surface: token.surface.clone(),
                char_start: token.char_start,
                char_end: token.char_end,
                comparison_start,
                comparison_end,
                g2p_raw: raw,
                g2p_normalized: norm,
            });
        } else {
            let probe_start = sentence_tokens
                .get(token_start)
                .map(|token| token.char_start)
                .ok_or_else(|| anyhow!("missing probe start token {}", token_start))?;
            let probe_end = sentence_tokens
                .get(token_end - 1)
                .map(|token| token.char_end)
                .ok_or_else(|| anyhow!("missing probe end token {}", token_end - 1))?;
            let probe_text = text
                .get(probe_start..probe_end)
                .ok_or_else(|| anyhow!("invalid probe slice {probe_start}..{probe_end}"))?
                .to_owned();
            let probe = probe_text_default(ProbeRequest {
                text: probe_text,
                lang_code: "eng-us".to_owned(),
                top_k: 4,
            })
            .map_err(|e| anyhow!("Charsiu probe failed for '{}': {e}", word.text))?;
            let local_input = transcript_alignment_input(&probe);
            let local_phones = token_piece_phones(&probe);
            if local_input.token_pieces.len() != token_end - token_start {
                return Err(anyhow!(
                    "token piece mismatch for '{}': global={} local={}",
                    word.text,
                    token_end - token_start,
                    local_input.token_pieces.len()
                ));
            }
            normalized.extend(local_input.normalized.iter().cloned());

            for (offset, (local_piece, local_phone)) in local_input
                .token_pieces
                .iter()
                .zip(local_phones.iter())
                .enumerate()
            {
                let token = sentence_tokens
                    .get(token_start + offset)
                    .ok_or_else(|| anyhow!("missing global token {}", token_start + offset))?;
                let comparison_start = word_cmp_start + local_piece.comparison_start;
                let comparison_end = word_cmp_start + local_piece.comparison_end;
                token_piece_ranges.push(TranscriptTokenPieceComparisonRange {
                    token_index: token.index,
                    token: token.label.clone(),
                    token_surface: token.surface.clone(),
                    token_char_start: token.char_start,
                    token_char_end: token.char_end,
                    word_index: Some(word_index),
                    word_surface: Some(word.text.clone()),
                    comparison_start,
                    comparison_end,
                });
                piece_rows.push(PieceRow {
                    word: word.text.clone(),
                    token_index: token.index,
                    token: token.label.clone(),
                    surface: token.surface.clone(),
                    char_start: token.char_start,
                    char_end: token.char_end,
                    comparison_start,
                    comparison_end,
                    g2p_raw: local_phone.ipa_tokens.clone(),
                    g2p_normalized: local_phone.normalized_phones.clone(),
                });
            }
        }

        let word_cmp_end = normalized.len();
        word_ranges.push(TranscriptWordComparisonRange {
            word_index,
            word_surface: word.text.clone(),
            char_start: word.char_start,
            char_end: word.char_end,
            comparison_start: word_cmp_start,
            comparison_end: word_cmp_end,
        });
    }

    let chars_input = TranscriptAlignmentInput {
        normalized,
        sequence: bee_g2p_charsiu::TranscriptComparisonSequence {
            tokens: Vec::new(),
            provenance: Vec::new(),
        },
        words: word_ranges,
        token_pieces: token_piece_ranges,
    };
    let alignment = TranscriptAlignment::build_from_comparison_input(
        bee_transcribe::zipa_align::transcript_comparison_input_from_charsiu(&text, &chars_input),
        &audio,
        &inference,
    )
    .map_err(|e| anyhow!("building transcript alignment: {e}"))?;
    let piece_timings = alignment.token_piece_timings(&chars_input.token_pieces);

    let token_pieces_json = piece_rows
        .iter()
        .map(|piece| {
            let timing = piece_timings
                .iter()
                .find(|timing| timing.token_index == piece.token_index)
                .ok_or_else(|| anyhow!("missing token piece timing for {}", piece.token_index))?;
            let projected = alignment
                .projected_comparison_range(piece.comparison_start..piece.comparison_end);
            let (zipa_normalized, zipa_raw_piece, audio_json, alignment_label) =
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
                    ComparisonRangeTiming::Invalid => (Vec::new(), Vec::new(), None, "invalid".to_owned()),
                };
            Ok(json!({
                "word": piece.word,
                "token_index": piece.token_index,
                "token": piece.token,
                "surface": piece.surface,
                "char_start": piece.char_start,
                "char_end": piece.char_end,
                "token_start": piece.token_index,
                "token_end": piece.token_index + 1,
                "comparison_start": piece.comparison_start,
                "comparison_end": piece.comparison_end,
                "projected_start": projected.as_ref().map(|range| range.start),
                "projected_end": projected.as_ref().map(|range| range.end),
                "g2p_raw": piece.g2p_raw,
                "g2p_normalized": piece.g2p_normalized,
                "zipa_normalized": zipa_normalized,
                "zipa_raw": zipa_raw_piece,
                "audio": audio_json,
                "alignment": alignment_label,
            }))
        })
        .collect::<Result<Vec<_>>>()?;

    let words_json = words
        .iter()
        .map(|word| {
            let (token_start, token_end) =
                token_span_for_char_range(word.char_start..word.char_end, encoding.get_offsets());
            json!({
                "word": word.text,
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
