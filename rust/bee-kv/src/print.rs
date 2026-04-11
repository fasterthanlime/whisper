use anyhow::Result;
use bee_phonetic::sentence_word_tokens;

use crate::alignment::{AlignmentContext, build_transcript_alignment, format_span_timing};
use crate::decode::combine_transcripts;
use crate::types::*;
use crate::{
    ANSI_BLUE, ANSI_BOLD, ANSI_RESET, DEFAULT_BRIDGE_MS, DEFAULT_CHUNK_MS, DEFAULT_LANGUAGE,
    DEFAULT_MAX_NEW_TOKENS, DEFAULT_ROLLBACK_MS, MAX_BRIDGE_WINDOWS, SAMPLE_RATE,
};

/// Prints CLI usage information to stderr.
pub(crate) fn print_usage() {
    eprintln!(
        "usage: bee-kv [--mode MODE] [--context TEXT] [--chunk-ms N] [--bridge-ms N] [--rollback-ms N] [wav-path] [language] [max-new-tokens]\n\
         defaults:\n\
           wav-path = {}\n\
           language = {DEFAULT_LANGUAGE}\n\
           max-new-tokens = {DEFAULT_MAX_NEW_TOKENS}\n\
           mode = initial\n\
           chunk-ms = {DEFAULT_CHUNK_MS}\n\
           bridge-ms = {DEFAULT_BRIDGE_MS}\n\
           max-bridge-windows = {MAX_BRIDGE_WINDOWS}\n\
           rollback-ms = {DEFAULT_ROLLBACK_MS}\n\
         modes:\n\
           initial\n\
           followup-fresh\n\
           system-compare\n\
           chunked-followup\n\
           prefix-rerun\n\
           truncate-replay\n\
           sliding-window-timed-rollback\n\
           sliding-window-full-replay\n\
           sliding-window-bridge-replay\n\
           dual-lane-followup\n\
           chunk-segment-merge-rollback\n\
           chunk-segment-merge-boundary-sweep\n\
         rollback policies for truncate-replay:\n\
           text-suffix\n\
           chunk-segment\n\
         environment:\n\
           BEE_ASR_MODEL_DIR\n\
           BEE_TOKENIZER_PATH (optional; defaults to $BEE_ASR_MODEL_DIR/tokenizer.json)",
        default_wav_path().display()
    );
}

/// Prints results of a chunked follow-up experiment.
pub(crate) fn print_chunked_experiment(experiment: &ChunkedExperimentResult) {
    println!("=== chunked-followup ===");
    println!("chunk_ms={}", experiment.chunk_ms);
    println!();
    for chunk in &experiment.chunk_results {
        println!("--- {} ---", chunk.label);
        println!(
            "samples={} prompt_tokens={} generated_tokens={}",
            chunk.sample_count, chunk.prompt_tokens, chunk.generated_tokens
        );
        println!("{}", chunk.transcript);
        println!();
    }
    println!("=== combined ===");
    println!("{}", experiment.combined_transcript);
    println!();
}

/// Prints results of a prefix-rerun experiment.
pub(crate) fn print_prefix_rerun_experiment(experiment: &PrefixRerunExperimentResult) {
    println!("=== prefix-rerun ===");
    println!("chunk_ms={}", experiment.chunk_ms);
    println!();
    for prefix in &experiment.prefix_results {
        println!("--- {} ---", prefix.label);
        println!(
            "samples={} prompt_tokens={} generated_tokens={}",
            prefix.sample_count, prefix.prompt_tokens, prefix.generated_tokens
        );
        println!("{}", prefix.transcript);
        println!();
    }
}

/// Prints results of a truncate-replay experiment with baseline vs replay comparison.
pub(crate) fn print_truncate_replay_experiment(experiment: &TruncateReplayExperimentResult) {
    println!("=== truncate-replay ===");
    println!("chunk_ms={}", experiment.chunk_ms);
    println!("rollback_policy={}", experiment.rollback_policy.as_str());
    println!("replay_chunk_index={}", experiment.replay_chunk_index);
    println!(
        "replay_from_chunk_index={}",
        experiment.replay_from_chunk_index
    );
    println!("rollback_position={}", experiment.rollback_position);
    println!(
        "requested_truncate_tokens={} applied_truncate_tokens={}",
        experiment.requested_truncate_tokens, experiment.applied_truncate_tokens
    );
    println!();

    println!("=== baseline ===");
    for chunk in &experiment.baseline_runs {
        print_chunk_run(chunk);
    }
    println!("=== baseline combined ===");
    println!("{}", combine_transcripts(&experiment.baseline_runs));
    println!();

    println!("=== replay path ===");
    for chunk in &experiment.replay_runs {
        print_chunk_run(chunk);
    }
    println!("=== replay combined ===");
    println!("{}", combine_transcripts(&experiment.replay_runs));
    println!();
}

/// Prints results of a sliding-window timed-rollback experiment.
pub(crate) fn print_sliding_window_timed_rollback_experiment(
    experiment: &SlidingWindowTimedRollbackExperimentResult,
) {
    println!("=== {} ===", experiment.mode_label);
    println!("chunk_ms={}", experiment.chunk_ms);
    println!("rollback_ms={}", experiment.rollback_ms);
    println!("stride_ms={}", experiment.stride_ms);
    println!("interrupted_early={}", experiment.interrupted_early);
    println!();

    for run in &experiment.window_runs {
        print_chunk_run(&run.chunk_run);
        if let Some(prefix) = &run.replayed_prefix {
            println!("replayed_prefix_text={}", prefix.text);
        }
        if let Some(rollback) = &run.rollback {
            println!(
                "rollback keep_until={:?} kept_words={} kept_tokens={} bridge_tokens={} rollback_position={}",
                rollback.keep_until_secs,
                rollback.kept_word_count,
                rollback.kept_token_count,
                rollback.bridge_token_ids.len(),
                rollback.rollback_position
            );
            println!("kept_text={}", rollback.kept_text);
            if rollback.keep_until_secs.is_some() && !rollback.kept_text.is_empty() {
                println!(
                    "{ANSI_BLUE}{ANSI_BOLD}==================== KEPT WORDS ===================={ANSI_RESET}"
                );
                println!("{ANSI_BLUE}{ANSI_BOLD}{}{}", rollback.kept_text, ANSI_RESET);
                println!(
                    "{ANSI_BLUE}{ANSI_BOLD}===================================================={ANSI_RESET}"
                );
            }
            if let Some(replay_until_secs) = rollback.replay_until_secs {
                println!("replay_until={:.3}s", replay_until_secs);
            }
            if let Some(bridge_text) = &rollback.bridge_text {
                println!("bridge_text={}", bridge_text);
            }
        } else {
            println!("rollback final-window");
        }
        println!();
    }
    println!("html_path={}", experiment.html_path.display());
    if let Some(committed_timeline_path) = &experiment.committed_timeline_path {
        println!(
            "committed_timeline_path={}",
            committed_timeline_path.display()
        );
    }
    println!();
}

/// Prints results of a dual-lane follow-up experiment.
pub(crate) fn print_dual_lane_followup_experiment(experiment: &DualLaneFollowupExperimentResult) {
    println!("=== dual-lane-followup ===");
    println!("lane_a_chunk_ms={}", experiment.chunk_ms);
    println!("lane_b_first_chunk_ms={}", experiment.lane_b_first_chunk_ms);
    println!();

    println!("=== lane-a row ===");
    print_lane_row("A", &experiment.lane_a_runs);
    println!("=== lane-b row ===");
    print_lane_row("B", &experiment.lane_b_runs);
    println!();

    println!("=== lane-a chunks ===");
    for chunk in &experiment.lane_a_runs {
        print_chunk_run(chunk);
    }
    println!("=== lane-b chunks ===");
    for chunk in &experiment.lane_b_runs {
        print_chunk_run(chunk);
    }
}

/// Prints results of a chunk-segment merge rollback experiment.
pub(crate) fn print_chunk_segment_merge_rollback_experiment(
    experiment: &ChunkSegmentMergeRollbackExperimentResult,
) {
    println!("=== chunk-segment-merge-rollback ===");
    println!("chunk_ms={}", experiment.chunk_ms);
    println!();

    println!("=== baseline first three ===");
    for chunk in &experiment.baseline_runs {
        print_chunk_run(chunk);
    }
    println!(
        "baseline_bracketed={}",
        bracketed_chunks(&experiment.baseline_runs)
    );
    println!("baseline_annotated={}", experiment.baseline_annotated);
    println!();

    println!("=== replay after rollback ===");
    print_chunk_run(&experiment.replay_chunk0);
    print_chunk_run(&experiment.merged_chunk);
    println!(
        "replay_bracketed={}",
        bracketed_chunks(&[
            clone_chunk_run(&experiment.replay_chunk0),
            clone_chunk_run(&experiment.merged_chunk),
        ])
    );
    println!("replay_annotated={}", experiment.replay_annotated);
    println!("html_path={}", experiment.html_path.display());
    println!();
}

/// Prints results of a boundary sweep experiment.
pub(crate) fn print_chunk_segment_merge_boundary_sweep_experiment(
    experiment: &ChunkSegmentMergeBoundarySweepExperimentResult,
) {
    println!("=== chunk-segment-merge-boundary-sweep ===");
    println!("chunk_ms={}", experiment.chunk_ms);
    println!("exact_boundary={}", experiment.exact_boundary);
    println!(
        "baseline_bracketed={}",
        bracketed_chunks(&experiment.baseline_runs)
    );
    println!();
    for result in &experiment.sweep_results {
        println!(
            "offset={:+} rollback_position={} replay_bracketed=[{}] [{}]",
            result.offset,
            result.rollback_position,
            experiment.baseline_runs[0].transcript,
            result.merged_chunk.transcript
        );
    }
    println!();
}

/// Creates a deep clone of a `ChunkRun` (since `ChunkRun` doesn't derive `Clone`).
pub(crate) fn clone_chunk_run(chunk: &ChunkRun) -> ChunkRun {
    ChunkRun {
        label: chunk.label.clone(),
        prompt_tokens: chunk.prompt_tokens,
        generated_tokens: chunk.generated_tokens,
        generated_token_ids: chunk.generated_token_ids.clone(),
        transcript: chunk.transcript.clone(),
        sample_count: chunk.sample_count,
        decode_ms: chunk.decode_ms,
        stop_reason: chunk.stop_reason,
        start_position: chunk.start_position,
        end_position: chunk.end_position,
        start_sample: chunk.start_sample,
        end_sample: chunk.end_sample,
    }
}

/// Formats chunk transcripts as `[chunk1] [chunk2] ...` for display.
pub(crate) fn bracketed_chunks(chunks: &[ChunkRun]) -> String {
    chunks
        .iter()
        .map(|chunk| format!("[{}]", chunk.transcript))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Annotates chunk transcripts with ZIPA-derived word timing spans.
pub(crate) fn annotate_chunk_runs(chunks: &[ChunkRun], samples: &[f32]) -> Result<String> {
    let combined_transcript = combine_transcripts(chunks);
    let mut align_ctx = AlignmentContext::new()?;
    let alignment = build_transcript_alignment(&mut align_ctx, &combined_transcript, samples)?;
    let mut word_start = 0usize;
    let mut annotated = Vec::with_capacity(chunks.len());

    for chunk in chunks {
        let word_count = sentence_word_tokens(&chunk.transcript).len();
        let word_end = word_start + word_count;
        annotated.push(format!(
            "[{}]{}",
            chunk.transcript,
            format_span_timing(alignment.span_timing(word_start, word_end))
        ));
        word_start = word_end;
    }

    Ok(annotated.join(" "))
}

/// Prints a one-line summary per chunk in a lane (label, seam position, token count).
pub(crate) fn print_lane_row(label: &str, chunks: &[ChunkRun]) {
    for chunk in chunks {
        let seam_ms = (chunk.end_sample * 1000) / SAMPLE_RATE as usize;
        println!(
            "{} seam@{}ms [{} tok]: {}",
            label, seam_ms, chunk.generated_tokens, chunk.transcript
        );
    }
    println!();
}

/// Prints detailed info for a single chunk run.
pub(crate) fn print_chunk_run(chunk: &ChunkRun) {
    println!("--- {} ---", chunk.label);
    println!(
        "samples={} audio={}..{}ms prompt_tokens={} generated_tokens={} decode_ms={:.1} stop_reason={} start_position={} end_position={}",
        chunk.sample_count,
        (chunk.start_sample * 1000) / SAMPLE_RATE as usize,
        (chunk.end_sample * 1000) / SAMPLE_RATE as usize,
        chunk.prompt_tokens,
        chunk.generated_tokens,
        chunk.decode_ms,
        chunk.stop_reason.as_str(),
        chunk.start_position,
        chunk.end_position
    );
    println!("{}", chunk.transcript);
    println!();
}

/// Prints a bold blue "FINALIZING" banner.
pub(crate) fn print_finalizing_banner() {
    println!(
        "{ANSI_BLUE}{ANSI_BOLD}===================== FINALIZING ====================={ANSI_RESET}"
    );
}

/// Prints model/audio summary info.
pub(crate) fn print_summary(summary: &RunSummary<'_>) {
    println!("wav: {}", summary.wav_path.display());
    println!("model_dir: {}", summary.model_dir.display());
    println!("tokenizer: {}", summary.tokenizer_path.display());
    println!("language: {}", summary.language);
    println!(
        "weights: loaded={} total_keys={} quantized_layers={} bits={} group_size={}",
        summary.load_stats.loaded,
        summary.load_stats.total_keys,
        summary.load_stats.quantized_layers,
        summary.load_stats.bits,
        summary.load_stats.group_size
    );
    println!(
        "audio: samples={} mel_frames={} audio_tokens={}",
        summary.sample_count, summary.mel_frames, summary.audio_tokens
    );
    println!();
}

/// Prints results of a single full-audio decode experiment.
pub(crate) fn print_experiment(experiment: &ExperimentResult) {
    println!("=== {} ===", experiment.label);
    println!(
        "prompt_tokens={} generated_tokens={}",
        experiment.prompt_tokens, experiment.generated_tokens
    );
    println!("{}", experiment.transcript);
    println!();
}
