use std::fs;
use std::path::PathBuf;

use anyhow::{Context, bail};
use bee_g2p_charsiu::{
    ProbeRequest, TranscriptAlignmentInput, TranscriptComparisonSequence,
    TranscriptTokenPieceComparisonRange, TranscriptWordComparisonRange, probe_text_default,
    transcript_alignment_input, transcript_words,
};
use bee_transcribe::zipa_align::{TranscriptAlignment, transcript_comparison_input_from_charsiu};
use bee_zipa_mlx::audio::load_wav_mono_f32;
use bee_zipa_mlx::infer::ZipaInference;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let mut wav = None::<PathBuf>;
    let mut text = None::<String>;
    let mut transcript_file = None::<PathBuf>;
    let mut lang_code = "eng-us".to_string();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--wav" => wav = Some(args.next().context("--wav requires a value")?.into()),
            "--text" => text = Some(args.next().context("--text requires a value")?),
            "--transcript-file" => {
                transcript_file = Some(
                    args.next()
                        .context("--transcript-file requires a value")?
                        .into(),
                )
            }
            "--lang-code" => {
                lang_code = args.next().context("--lang-code requires a value")?;
            }
            _ => bail!(
                "usage: charsiu-zipa-harness --wav PATH [--text TEXT | --transcript-file PATH] [--lang-code CODE]"
            ),
        }
    }

    let wav = wav.context("--wav is required")?;
    let text = match (text, transcript_file) {
        (Some(text), None) => text,
        (None, Some(path)) => extract_transcript_text(&path)?,
        (Some(_), Some(_)) => bail!("use only one of --text or --transcript-file"),
        (None, None) => bail!("one of --text or --transcript-file is required"),
    };

    let charsiu_input = build_per_word_alignment_input(&text, &lang_code)?;
    let probe = probe_text_default(ProbeRequest {
        text: text.clone(),
        lang_code,
        top_k: 6,
    })?;
    let align_input = transcript_comparison_input_from_charsiu(&text, &charsiu_input);

    let audio = load_wav_mono_f32(&wav).with_context(|| format!("load wav {}", wav.display()))?;
    let zipa = ZipaInference::load_reference_small_no_diacritics()?;
    let alignment = TranscriptAlignment::build_from_comparison_input(align_input, &audio, &zipa)
        .map_err(|e| anyhow::anyhow!("build alignment: {e}"))?;

    println!("wav\t{}", wav.display());
    println!("text\t{}", text);
    println!("decoded_ipa\t{}", probe.decoded_output);
    println!("normalized\t{}", charsiu_input.normalized.join(" "));

    for word in &charsiu_input.words {
        match alignment.comparison_range_timing(word.comparison_start..word.comparison_end) {
            Some(timing) => println!(
                "word\t{}\tchars={}..{}\tcmp={}..{}\ttime={:.3}..{:.3}",
                word.word_surface,
                word.char_start,
                word.char_end,
                word.comparison_start,
                word.comparison_end,
                timing.start_time_secs,
                timing.end_time_secs
            ),
            None => println!(
                "word\t{}\tchars={}..{}\tcmp={}..{}\ttime=<none>",
                word.word_surface,
                word.char_start,
                word.char_end,
                word.comparison_start,
                word.comparison_end
            ),
        }
    }

    for token in &charsiu_input.token_pieces {
        let projected =
            alignment.projected_comparison_range(token.comparison_start..token.comparison_end);
        match alignment.comparison_range_timing(token.comparison_start..token.comparison_end) {
            Some(timing) => println!(
                "token\t{}\tword={}\tchars={}..{}\tcmp={}..{}\tproj={}\ttime={:.3}..{:.3}",
                token.token_surface,
                token.word_surface.as_deref().unwrap_or_default(),
                token.token_char_start,
                token.token_char_end,
                token.comparison_start,
                token.comparison_end,
                format_projected_range(projected.as_ref()),
                timing.start_time_secs,
                timing.end_time_secs
            ),
            None => println!(
                "token\t{}\tword={}\tchars={}..{}\tcmp={}..{}\tproj={}\ttime=<none>",
                token.token_surface,
                token.word_surface.as_deref().unwrap_or_default(),
                token.token_char_start,
                token.token_char_end,
                token.comparison_start,
                token.comparison_end,
                format_projected_range(projected.as_ref())
            ),
        }
    }

    Ok(())
}

fn format_projected_range(range: Option<&std::ops::Range<usize>>) -> String {
    match range {
        Some(range) => format!("{}..{}", range.start, range.end),
        None => "<none>".to_string(),
    }
}

fn extract_transcript_text(path: &PathBuf) -> anyhow::Result<String> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("read transcript file {}", path.display()))?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("  text: ") {
            return Ok(rest.trim_matches('"').to_string());
        }
    }
    bail!("no final transcript text found in {}", path.display())
}

fn build_per_word_alignment_input(
    text: &str,
    lang_code: &str,
) -> anyhow::Result<TranscriptAlignmentInput> {
    let words = transcript_words(text);
    let mut normalized = Vec::new();
    let mut word_ranges = Vec::new();
    let mut token_pieces = Vec::new();
    let mut comparison_cursor = 0usize;
    let mut token_cursor = 0usize;

    for (word_index, word) in words.iter().enumerate() {
        let probe = probe_text_default(ProbeRequest {
            text: word.word.clone(),
            lang_code: lang_code.to_string(),
            top_k: 6,
        })?;
        let local = transcript_alignment_input(&probe);
        let word_cmp_start = comparison_cursor;
        let word_cmp_end = comparison_cursor + local.normalized.len();

        normalized.extend(local.normalized);
        word_ranges.push(TranscriptWordComparisonRange {
            word_index,
            word_surface: word.word.clone(),
            char_start: word.char_start,
            char_end: word.char_end,
            comparison_start: word_cmp_start,
            comparison_end: word_cmp_end,
        });

        for token in local.token_pieces {
            token_pieces.push(TranscriptTokenPieceComparisonRange {
                token_index: token_cursor,
                token: token.token,
                token_surface: text[word.char_start + token.token_char_start
                    ..word.char_start + token.token_char_end]
                    .to_string(),
                token_char_start: word.char_start + token.token_char_start,
                token_char_end: word.char_start + token.token_char_end,
                word_index: Some(word_index),
                word_surface: Some(word.word.clone()),
                comparison_start: token.comparison_start + word_cmp_start,
                comparison_end: token.comparison_end + word_cmp_start,
            });
            token_cursor += 1;
        }

        comparison_cursor = word_cmp_end;
    }

    Ok(TranscriptAlignmentInput {
        normalized,
        sequence: TranscriptComparisonSequence {
            tokens: Vec::new(),
            provenance: Vec::new(),
        },
        words: word_ranges,
        token_pieces,
    })
}
