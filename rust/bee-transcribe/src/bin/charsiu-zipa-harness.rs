use std::fs;
use std::path::PathBuf;

use anyhow::{Context, bail};
use bee_g2p::{BeeG2p, TranscriptAlignmentInput, transcript_alignment_input};
use bee_transcribe::zipa_align::{
    ComparisonRangeTiming, TranscriptAlignment, transcript_comparison_input_from_g2p,
};
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

    let g2p_input = build_per_word_alignment_input(&text, &lang_code)?;
    let align_input = transcript_comparison_input_from_g2p(&text, &g2p_input);

    let audio = load_wav_mono_f32(&wav).with_context(|| format!("load wav {}", wav.display()))?;
    let zipa = ZipaInference::load_reference_small_no_diacritics()?;
    let alignment = TranscriptAlignment::build_from_comparison_input(align_input, &audio, &zipa)
        .map_err(|e| anyhow::anyhow!("build alignment: {e}"))?;

    println!("wav\t{}", wav.display());
    println!("text\t{}", text);
    println!("normalized\t{}", g2p_input.normalized.join(" "));

    for word in &g2p_input.words {
        match alignment.comparison_range_timing(word.comparison_start..word.comparison_end) {
            ComparisonRangeTiming::Aligned(timing) => println!(
                "word\t{}\tchars={}..{}\tcmp={}..{}\ttime={:.3}..{:.3}",
                word.word_surface,
                word.char_start,
                word.char_end,
                word.comparison_start,
                word.comparison_end,
                timing.start_time_secs,
                timing.end_time_secs
            ),
            ComparisonRangeTiming::Deleted { projected_at } => println!(
                "word\t{}\tchars={}..{}\tcmp={}..{}\ttime=<deleted@{}>",
                word.word_surface,
                word.char_start,
                word.char_end,
                word.comparison_start,
                word.comparison_end,
                projected_at
            ),
            ComparisonRangeTiming::NoTiming { projected_range } => println!(
                "word\t{}\tchars={}..{}\tcmp={}..{}\ttime=<no-timing {}..{}>",
                word.word_surface,
                word.char_start,
                word.char_end,
                word.comparison_start,
                word.comparison_end,
                projected_range.start,
                projected_range.end
            ),
            ComparisonRangeTiming::Invalid => println!(
                "word\t{}\tchars={}..{}\tcmp={}..{}\ttime=<none>",
                word.word_surface,
                word.char_start,
                word.char_end,
                word.comparison_start,
                word.comparison_end
            ),
        }
    }

    for token in &g2p_input.token_pieces {
        let projected =
            alignment.projected_comparison_range(token.comparison_start..token.comparison_end);
        match alignment.comparison_range_timing(token.comparison_start..token.comparison_end) {
            ComparisonRangeTiming::Aligned(timing) => println!(
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
            ComparisonRangeTiming::Deleted { projected_at } => println!(
                "token\t{}\tword={}\tchars={}..{}\tcmp={}..{}\tproj={}\ttime=<deleted@{}>",
                token.token_surface,
                token.word_surface.as_deref().unwrap_or_default(),
                token.token_char_start,
                token.token_char_end,
                token.comparison_start,
                token.comparison_end,
                format_projected_range(projected.as_ref()),
                projected_at
            ),
            ComparisonRangeTiming::NoTiming { projected_range } => println!(
                "token\t{}\tword={}\tchars={}..{}\tcmp={}..{}\tproj={}\ttime=<no-timing {}..{}>",
                token.token_surface,
                token.word_surface.as_deref().unwrap_or_default(),
                token.token_char_start,
                token.token_char_end,
                token.comparison_start,
                token.comparison_end,
                format_projected_range(projected.as_ref()),
                projected_range.start,
                projected_range.end
            ),
            ComparisonRangeTiming::Invalid => println!(
                "token\t{}\tword={}\tchars={}..{}\tcmp={}..{}\tproj={}\ttime=<invalid>",
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
    let mut g2p = BeeG2p::load_default()?;
    let analysis = g2p.analyze_text(text, lang_code)?;
    Ok(transcript_alignment_input(&analysis))
}
