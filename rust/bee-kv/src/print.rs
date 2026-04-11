use crate::types::default_wav_path;
use crate::types::{ChunkRun, RunSummary, SlidingWindowTimedRollbackExperimentResult};
use crate::{
    ANSI_BLUE, ANSI_BOLD, ANSI_RESET, DEFAULT_BRIDGE_MS, DEFAULT_CHUNK_MS, DEFAULT_LANGUAGE,
    DEFAULT_MAX_NEW_TOKENS, DEFAULT_ROLLBACK_MS, MAX_BRIDGE_WINDOWS, SAMPLE_RATE,
};

/// Prints CLI usage information to stderr.
pub(crate) fn print_usage() {
    eprintln!(
        "usage: bee-kv --mode sliding-window-bridge-replay [--context TEXT] [--chunk-ms N] [--bridge-ms N] [--rollback-ms N] [wav-path] [language] [max-new-tokens]\n\
         defaults:\n\
           wav-path = {}\n\
           language = {DEFAULT_LANGUAGE}\n\
           max-new-tokens = {DEFAULT_MAX_NEW_TOKENS}\n\
           mode = sliding-window-bridge-replay\n\
           chunk-ms = {DEFAULT_CHUNK_MS}\n\
           bridge-ms = {DEFAULT_BRIDGE_MS}\n\
           max-bridge-windows = {MAX_BRIDGE_WINDOWS}\n\
           rollback-ms = {DEFAULT_ROLLBACK_MS}\n\
         environment:\n\
           BEE_ASR_MODEL_DIR\n\
           BEE_TOKENIZER_PATH (optional; defaults to $BEE_ASR_MODEL_DIR/tokenizer.json)",
        default_wav_path().display()
    );
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

/// Prints a bold blue "FINALIZING" banner.
pub(crate) fn print_finalizing_banner() {
    println!(
        "{ANSI_BLUE}{ANSI_BOLD}===================== FINALIZING ====================={ANSI_RESET}"
    );
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
