use std::fs;
use std::path::Path;

use bee_g2p::{BeeG2p, TranscriptAlignmentInput, transcript_alignment_input};
use bee_transcribe::zipa_align::{
    ComparisonRangeTiming, TranscriptAlignment, transcript_comparison_input_from_g2p,
};
use bee_zipa_mlx::audio::load_wav_mono_f32;
use bee_zipa_mlx::infer::ZipaInference;

#[test]
#[ignore = "requires local bee-g2p model, ZIPA weights, and real artifact WAV"]
fn real_artifact_recovers_some_token_piece_timings() -> Result<(), Box<dyn std::error::Error>> {
    let wav = Path::new(
        "/Users/amos/bearcove/bee/.artifacts/gpu-hotline/corpus-staging/current/8453579B.wav",
    );
    let transcript_file = Path::new(
        "/Users/amos/bearcove/bee/.artifacts/gpu-hotline/runs/clean-corpus-v1-no-correction-20260409/transcripts/8453579B.txt",
    );
    let text = extract_transcript_text(transcript_file)?;

    let g2p_input = build_per_word_alignment_input(&text, "eng-us")?;
    let align_input = transcript_comparison_input_from_g2p(&text, &g2p_input);
    let audio = load_wav_mono_f32(wav)?;
    let zipa = ZipaInference::load_reference_small_no_diacritics()?;
    let alignment = TranscriptAlignment::build_from_comparison_input(align_input, &audio, &zipa)
        .map_err(|e| format!("build alignment: {e}"))?;

    let tokens = g2p_input
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
            .any(|(surface, _, timing)| surface.trim() == "There" && timing.is_some())
    );
    assert!(
        tokens
            .iter()
            .any(|(surface, _, timing)| surface.trim() == "work" && timing.is_some())
    );

    let there = tokens
        .iter()
        .find(|(surface, _, _)| surface.trim() == "There")
        .and_then(|(_, _, timing)| *timing)
        .expect("There should have timing");
    let work = tokens
        .iter()
        .find(|(surface, _, _)| surface.trim() == "work")
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
    let mut g2p = BeeG2p::load_default()?;
    let analysis = g2p.analyze_text(text, lang_code)?;
    Ok(transcript_alignment_input(&analysis))
}
