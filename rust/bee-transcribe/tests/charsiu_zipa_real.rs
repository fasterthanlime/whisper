use std::fs;
use std::path::Path;

use bee_g2p_charsiu::{
    ProbeRequest, TranscriptAlignmentInput, TranscriptComparisonSequence,
    TranscriptTokenPieceComparisonRange, TranscriptWordComparisonRange, probe_text_default,
    transcript_alignment_input, transcript_words,
};
use bee_transcribe::zipa_align::{
    ComparisonRangeTiming, TranscriptAlignment, transcript_comparison_input_from_charsiu,
};
use bee_zipa_mlx::audio::load_wav_mono_f32;
use bee_zipa_mlx::infer::ZipaInference;

#[test]
#[ignore = "requires local Charsiu sidecar, ZIPA weights, and real artifact WAV"]
fn real_artifact_recovers_some_token_piece_timings() -> Result<(), Box<dyn std::error::Error>> {
    let wav = Path::new(
        "/Users/amos/bearcove/bee/.artifacts/gpu-hotline/corpus-staging/current/8453579B.wav",
    );
    let transcript_file = Path::new(
        "/Users/amos/bearcove/bee/.artifacts/gpu-hotline/runs/clean-corpus-v1-no-correction-20260409/transcripts/8453579B.txt",
    );
    let text = extract_transcript_text(transcript_file)?;

    let charsiu_input = build_per_word_alignment_input(&text, "eng-us")?;
    let align_input = transcript_comparison_input_from_charsiu(&text, &charsiu_input);
    let audio = load_wav_mono_f32(wav)?;
    let zipa = ZipaInference::load_reference_small_no_diacritics()?;
    let alignment = TranscriptAlignment::build_from_comparison_input(align_input, &audio, &zipa)
        .map_err(|e| format!("build alignment: {e}"))?;

    let tokens = charsiu_input
        .token_pieces
        .iter()
        .map(|token| {
            let timing = match alignment
                .comparison_range_timing(token.comparison_start..token.comparison_end)
            {
                ComparisonRangeTiming::Aligned(timing) => {
                    Some((timing.start_time_secs, timing.end_time_secs))
                }
                _ => None,
            };
            (
                token.token_surface.clone(),
                token.word_surface.clone().unwrap_or_default(),
                timing,
            )
        })
        .collect::<Vec<_>>();

    assert!(
        tokens
            .iter()
            .any(|(surface, _, timing)| surface == "There" && timing.is_some())
    );
    assert!(
        tokens
            .iter()
            .any(|(surface, _, timing)| surface == "work" && timing.is_some())
    );

    let there = tokens
        .iter()
        .find(|(surface, _, _)| surface == "There")
        .and_then(|(_, _, timing)| *timing)
        .expect("There should have timing");
    let work = tokens
        .iter()
        .find(|(surface, _, _)| surface == "work")
        .and_then(|(_, _, timing)| *timing)
        .expect("work should have timing");

    assert!(
        there.0 >= 0.2 && there.1 <= 0.6,
        "There timing out of expected range: {there:?}"
    );
    assert!(
        work.0 >= 1.5 && work.1 <= 2.0,
        "work timing out of expected range: {work:?}"
    );
    assert!(
        there.1 <= work.0,
        "token timings should be ordered: There={there:?}, work={work:?}"
    );

    Ok(())
}

fn extract_transcript_text(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let text = fs::read_to_string(path)?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("  text: ") {
            return Ok(rest.trim_matches('"').to_string());
        }
    }
    Err(format!("no final transcript text found in {}", path.display()).into())
}

fn build_per_word_alignment_input(
    text: &str,
    lang_code: &str,
) -> Result<TranscriptAlignmentInput, Box<dyn std::error::Error>> {
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
