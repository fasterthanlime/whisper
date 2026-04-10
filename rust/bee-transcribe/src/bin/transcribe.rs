use std::path::{Path, PathBuf};
use std::time::Instant;
use std::{env, io::IsTerminal};

use bee_transcribe::text_buffer::TokenEntry;
use bee_transcribe::{EngineConfig, RotationCutStrategy, SessionOptions, SessionSnapshot};
use tokenizers::Tokenizer;
use tracing_subscriber::EnvFilter;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let args: Vec<String> = std::env::args().collect();

    // Subcommand: transcribe compare <audio.wav> [--out <path>]
    if args.get(1).map(|s| s.as_str()) == Some("compare") {
        let rest = &args[2..];

        fn flag_val<'a>(rest: &'a [String], flag: &str) -> Option<&'a str> {
            rest.windows(2)
                .find(|w| w[0] == flag)
                .map(|w| w[1].as_str())
        }
        let out_path = flag_val(rest, "--out").map(PathBuf::from);
        let chunk_ms: f32 = flag_val(rest, "--chunk-ms")
            .and_then(|v| v.parse().ok())
            .unwrap_or(400.0);
        let commit_tokens: usize = flag_val(rest, "--commit-tokens")
            .and_then(|v| v.parse().ok())
            .unwrap_or(SessionOptions::default().commit_token_count);
        let rollback_tokens: usize = flag_val(rest, "--rollback-tokens")
            .and_then(|v| v.parse().ok())
            .unwrap_or(SessionOptions::default().rollback_tokens);
        let context_tokens: usize = flag_val(rest, "--context-tokens")
            .and_then(|v| v.parse().ok())
            .unwrap_or(SessionOptions::default().context_tokens);
        let modes_filter: Option<Vec<&str>> =
            flag_val(rest, "--modes").map(|v| v.split(',').map(|s| s.trim()).collect());

        let known_flags = [
            "--out",
            "--chunk-ms",
            "--commit-tokens",
            "--rollback-tokens",
            "--context-tokens",
            "--modes",
        ];
        let positional: Vec<&str> = rest
            .iter()
            .filter(|a| {
                if a.starts_with("--") {
                    return false;
                }
                !known_flags
                    .iter()
                    .any(|f| flag_val(rest, f) == Some(a.as_str()))
            })
            .map(|s| s.as_str())
            .collect();
        if positional.is_empty() {
            eprintln!(
                "Usage: transcribe compare <audio> [--out report.html] [--chunk-ms 400] [--commit-tokens 12] [--rollback-tokens 5] [--context-tokens 0] [--modes uncut,qwen3,zipa,raw]"
            );
            std::process::exit(1);
        }
        let mut base_options = SessionOptions::default();
        base_options.chunk_duration = chunk_ms / 1000.0;
        base_options.commit_token_count = commit_tokens;
        base_options.rollback_tokens = rollback_tokens;
        base_options.context_tokens = context_tokens;
        let modes_filter: Option<Vec<String>> =
            modes_filter.map(|v| v.iter().map(|s| s.to_string()).collect());
        return cmd_compare(
            positional[0],
            out_path.as_deref(),
            base_options,
            modes_filter.as_deref(),
        );
    }

    let show_alts = args.iter().any(|a| a == "--alternatives" || a == "--alts");
    let positional: Vec<&str> = args[1..]
        .iter()
        .filter(|a| !a.starts_with("--"))
        .map(|s| s.as_str())
        .collect();
    if positional.is_empty() {
        eprintln!("Usage: transcribe [--alternatives] <audio.wav>");
        eprintln!("       transcribe compare <audio.wav> [--out report.html]");
        std::process::exit(1);
    }
    let audio_path = positional[0];

    let model_dir = std::env::var("BEE_ASR_MODEL_DIR")
        .map_err(|_| anyhow::anyhow!("BEE_ASR_MODEL_DIR not set"))?;
    let tokenizer_dir = std::env::var("BEE_TOKENIZER_DIR").unwrap_or_else(|_| model_dir.clone());
    let aligner_dir =
        std::env::var("BEE_ALIGNER_DIR").map_err(|_| anyhow::anyhow!("BEE_ALIGNER_DIR not set"))?;
    let vad_dir =
        std::env::var("BEE_VAD_DIR").map_err(|_| anyhow::anyhow!("BEE_VAD_DIR not set"))?;
    let zipa_bundle_dir = std::env::var("BEE_ZIPA_BUNDLE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("bearcove/zipa-mlx-hf")
        });
    let share_aligner_audio_tower = std::env::var("BEE_SHARE_ALIGNER_AUDIO_TOWER")
        .map(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false);

    // Correction engine: look in group container (same as install-bee.sh)
    let disable_correction = std::env::var("BEE_DISABLE_CORRECTION")
        .map(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false);
    let group_container: PathBuf = dirs::home_dir()
        .unwrap()
        .join("Library/Group Containers/B2N6FSRTPV.group.fasterthanlime.bee");
    let correction_dir_path = group_container.join("phonetic-seed");
    let correction_dir: Option<&Path> = if disable_correction {
        println!("Correction disabled via BEE_DISABLE_CORRECTION");
        None
    } else if correction_dir_path.exists() {
        println!("Correction dataset: {}", correction_dir_path.display());
        Some(&correction_dir_path)
    } else {
        println!(
            "Correction dataset not found at {}",
            correction_dir_path.display()
        );
        None
    };
    let correction_events_path = correction_dir.map(|d| d.join("events.jsonl"));

    // Load engine
    let t0 = Instant::now();
    let engine = bee_transcribe::Engine::load(&EngineConfig {
        model_dir: Path::new(&model_dir),
        tokenizer_dir: Path::new(&tokenizer_dir),
        aligner_dir: Path::new(&aligner_dir),
        share_aligner_audio_tower,
        silero_dir: Path::new(&vad_dir),
        correction_dir,
        correction_events_path,
        zipa_bundle_dir: &zipa_bundle_dir,
    })?;
    println!("Engine loaded in {:.0}ms", t0.elapsed().as_millis());

    // Load audio
    let t0 = Instant::now();
    let samples = bee_qwen3_asr::mel::load_audio_wav(audio_path, 16000)?;
    let duration = samples.len() as f64 / 16000.0;
    println!(
        "Audio: {:.1}s ({} samples) loaded in {:.0}ms",
        duration,
        samples.len(),
        t0.elapsed().as_millis()
    );

    // Create session with env var overrides
    let mut options = SessionOptions::default();
    if let Ok(v) = std::env::var("BEE_CHUNK_DURATION") {
        options.chunk_duration = v.parse().unwrap();
    }
    if let Ok(v) = std::env::var("BEE_VAD_THRESHOLD") {
        options.vad_threshold = v.parse().unwrap();
    }
    if let Ok(v) = std::env::var("BEE_ROLLBACK_TOKENS") {
        options.rollback_tokens = v.parse().unwrap();
    }
    if let Ok(v) = std::env::var("BEE_COMMIT_TOKENS") {
        options.commit_token_count = v.parse().unwrap();
    }
    if let Ok(v) = std::env::var("BEE_MAX_TOKENS_STREAMING") {
        options.max_tokens_streaming = v.parse().unwrap();
    }
    if let Ok(v) = std::env::var("BEE_MAX_TOKENS_FINAL") {
        options.max_tokens_final = v.parse().unwrap();
    }
    if let Ok(v) = std::env::var("BEE_ROTATION_CUT_MODE") {
        options.rotation_cut_strategy = match v.to_ascii_lowercase().as_str() {
            "uncut" | "never" => RotationCutStrategy::Uncut,
            "qwen3" | "qwen" => RotationCutStrategy::Qwen3,
            "zipa" => RotationCutStrategy::Zipa,
            "manual" => {
                let target: u32 = std::env::var("BEE_ROTATION_TARGET_COMMITTED_TOKENS")
                    .unwrap_or_else(|_| "12".to_string())
                    .parse()
                    .unwrap();
                RotationCutStrategy::ManualTargetCommittedTextTokens(target)
            }
            other => {
                anyhow::bail!(
                    "invalid BEE_ROTATION_CUT_MODE={other}; expected one of: uncut|qwen3|zipa|manual"
                );
            }
        };
    } else if let Ok(v) = std::env::var("BEE_ROTATION_TARGET_COMMITTED_TOKENS") {
        let target: u32 = v.parse().unwrap();
        options.rotation_cut_strategy =
            RotationCutStrategy::ManualTargetCommittedTextTokens(target);
    }
    let chunk_samples = (options.chunk_duration * 16000.0) as usize;

    println!(
        "\n--- Streaming (chunk={:.0}ms) ---\n",
        chunk_samples as f64 / 16.0
    );

    let mut session = engine.session(options)?;

    let t_total = Instant::now();
    let mut chunk_idx = 0;
    let mut offset = 0;
    let mut last_text = String::new();

    while offset < samples.len() {
        let end = (offset + chunk_samples).min(samples.len());
        let chunk = &samples[offset..end];
        offset = end;
        chunk_idx += 1;

        let t0 = Instant::now();
        let result = session.feed(chunk)?;
        let ms = t0.elapsed().as_millis();

        match result {
            Some(snapshot) => {
                if snapshot.full_text != last_text {
                    print_update(chunk_idx, ms, &snapshot);
                    last_text = snapshot.full_text.clone();
                } else {
                    println!("  chunk {chunk_idx}: {ms:.0}ms (unchanged)");
                }
                if show_alts {
                    print_alternatives(session.tokenizer(), session.pending_entries());
                }
            }
            None => {
                println!("  chunk {chunk_idx}: {ms:.0}ms (silence/buffering)");
            }
        }
    }

    let t0 = Instant::now();
    let result = session.finish()?;
    let finish_ms = t0.elapsed().as_millis();
    println!(
        "\n--- Final ({finish_ms:.0}ms, total {:.0}ms) ---",
        t_total.elapsed().as_millis()
    );
    println!("  text: {:?}", result.snapshot.full_text);

    if !result.snapshot.committed_words.is_empty() {
        println!("\nAlignments:");
        for w in &result.snapshot.committed_words {
            println!("  [{:.3}s - {:.3}s] {}", w.start, w.end, w.word);
        }
    }

    Ok(())
}

fn print_update(chunk: usize, ms: u128, snapshot: &SessionSnapshot) {
    let rendered_text = render_colored_text(snapshot);
    println!(
        "  chunk {chunk}: {ms:.0}ms | rev={} committed={} pending={} volatile={} | {}",
        snapshot.revision,
        snapshot.committed_token_count,
        snapshot.pending_token_count,
        snapshot.ambiguity.volatile_token_count,
        rendered_text,
    );
}

fn render_colored_text(snapshot: &SessionSnapshot) -> String {
    let mut text = String::new();
    let use_color = should_use_color();
    if use_color && !snapshot.committed_text.is_empty() {
        text.push_str("\x1b[38;5;120m");
        text.push_str(&snapshot.committed_text);
        text.push_str("\x1b[0m");
    } else {
        text.push_str(&snapshot.committed_text);
    }
    text.push_str(&snapshot.pending_text);
    text
}

fn should_use_color() -> bool {
    if env::var_os("NO_COLOR").is_some() {
        return false;
    }
    if let Some(v) = env::var_os("CLICOLOR_FORCE") {
        let forced = v.to_string_lossy();
        if forced != "0" {
            return true;
        }
    }
    if let Some(v) = env::var_os("FORCE_COLOR") {
        let forced = v.to_string_lossy();
        if forced != "0" {
            return true;
        }
    }
    std::io::stdout().is_terminal()
}

fn load_engine(disable_correction: bool) -> anyhow::Result<bee_transcribe::Engine> {
    let model_dir = std::env::var("BEE_ASR_MODEL_DIR")
        .map_err(|_| anyhow::anyhow!("BEE_ASR_MODEL_DIR not set"))?;
    let tokenizer_dir = std::env::var("BEE_TOKENIZER_DIR").unwrap_or_else(|_| model_dir.clone());
    let aligner_dir =
        std::env::var("BEE_ALIGNER_DIR").map_err(|_| anyhow::anyhow!("BEE_ALIGNER_DIR not set"))?;
    let vad_dir =
        std::env::var("BEE_VAD_DIR").map_err(|_| anyhow::anyhow!("BEE_VAD_DIR not set"))?;
    let zipa_bundle_dir = std::env::var("BEE_ZIPA_BUNDLE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("bearcove/zipa-mlx-hf")
        });
    let share_aligner_audio_tower = std::env::var("BEE_SHARE_ALIGNER_AUDIO_TOWER")
        .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false);
    let group_container: PathBuf = dirs::home_dir()
        .unwrap()
        .join("Library/Group Containers/B2N6FSRTPV.group.fasterthanlime.bee");
    let correction_dir_path = group_container.join("phonetic-seed");
    let correction_dir: Option<&Path> = if !disable_correction && correction_dir_path.exists() {
        Some(&correction_dir_path)
    } else {
        None
    };
    let correction_events_path = correction_dir.map(|d| d.join("events.jsonl"));
    Ok(bee_transcribe::Engine::load(&EngineConfig {
        model_dir: Path::new(&model_dir),
        tokenizer_dir: Path::new(&tokenizer_dir),
        aligner_dir: Path::new(&aligner_dir),
        share_aligner_audio_tower,
        silero_dir: Path::new(&vad_dir),
        correction_dir,
        correction_events_path,
        zipa_bundle_dir: &zipa_bundle_dir,
    })?)
}

fn run_mode(
    engine: &bee_transcribe::Engine,
    samples: &[f32],
    cut_mode: RotationCutStrategy,
    bypass_audio_filters: bool,
    label: &str,
    base_options: SessionOptions,
) -> anyhow::Result<(
    String,
    Vec<bee_transcribe::CutEvent>,
    Vec<bee_transcribe::ChunkEvent>,
)> {
    println!("--- Running {label} ---");
    let mut options = base_options;
    // For uncut, use a very large chunk so it processes as one pass
    if matches!(cut_mode, RotationCutStrategy::Uncut) {
        options.chunk_duration = 600.0;
    }
    options.rotation_cut_strategy = cut_mode;
    options.bypass_audio_filters = bypass_audio_filters;
    let chunk_samples = (options.chunk_duration * 16000.0) as usize;

    let cuts = std::rc::Rc::new(std::cell::RefCell::new(
        Vec::<bee_transcribe::CutEvent>::new(),
    ));
    let cuts_sink = cuts.clone();
    let cut_sink: bee_transcribe::CutSink = Box::new(move |event: bee_transcribe::CutEvent| {
        cuts_sink.borrow_mut().push(event);
    });

    let chunks = std::rc::Rc::new(std::cell::RefCell::new(
        Vec::<bee_transcribe::ChunkEvent>::new(),
    ));
    let chunks_sink = chunks.clone();
    let chunk_sink: bee_transcribe::ChunkSink =
        Box::new(move |event: bee_transcribe::ChunkEvent| {
            chunks_sink.borrow_mut().push(event);
        });

    let mut session = engine.session_with_sinks(options, Some(cut_sink), Some(chunk_sink))?;
    let mut offset = 0;
    while offset < samples.len() {
        let end = (offset + chunk_samples).min(samples.len());
        session.feed(&samples[offset..end])?;
        offset = end;
    }
    let result = session.finish()?;
    let text = result.snapshot.full_text.clone();
    let cuts = std::rc::Rc::try_unwrap(cuts).ok().unwrap().into_inner();
    let chunks = std::rc::Rc::try_unwrap(chunks).ok().unwrap().into_inner();
    println!(
        "  {label}: {:?} ({} cuts, {} chunks)",
        text,
        cuts.len(),
        chunks.len()
    );
    Ok((text, cuts, chunks))
}

/// Downsample `samples` to `n` [lo, hi] pairs for waveform display.
fn downsample_waveform(samples: &[f32], n: usize) -> Vec<[f32; 2]> {
    if samples.is_empty() {
        return vec![[0.0, 0.0]; n];
    }
    let len = samples.len();
    (0..n)
        .map(|i| {
            let start = (i * len / n).min(len);
            let end = ((i + 1) * len / n).min(len);
            if start >= end {
                return [0.0, 0.0];
            }
            let mut lo = f32::INFINITY;
            let mut hi = f32::NEG_INFINITY;
            for &s in &samples[start..end] {
                if s < lo {
                    lo = s;
                }
                if s > hi {
                    hi = s;
                }
            }
            [lo, hi]
        })
        .collect()
}

fn write_wav(path: &Path, audio: &bee_transcribe::audio_buffer::AudioBuffer) -> anyhow::Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: audio.sample_rate().0,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &s in audio.samples() {
        writer.write_sample((s * i16::MAX as f32) as i16)?;
    }
    writer.finalize()?;
    Ok(())
}

fn normalize_word(w: &str) -> String {
    w.chars()
        .filter(|c| c.is_alphanumeric())
        .collect::<String>()
        .to_lowercase()
}

/// Align two sequences using LCS, returning gapped versions (None = gap).
fn align_pair<T: Clone + Eq + std::hash::Hash + Ord>(
    a: &[T],
    b: &[T],
) -> (Vec<Option<T>>, Vec<Option<T>>) {
    use similar::{Algorithm, DiffOp, capture_diff_slices};
    let ops = capture_diff_slices(Algorithm::Myers, a, b);
    let mut ga: Vec<Option<T>> = Vec::new();
    let mut gb: Vec<Option<T>> = Vec::new();
    for op in ops {
        match op {
            DiffOp::Equal {
                old_index,
                new_index,
                len,
            } => {
                for i in 0..len {
                    ga.push(Some(a[old_index + i].clone()));
                    gb.push(Some(b[new_index + i].clone()));
                }
            }
            DiffOp::Delete {
                old_index, old_len, ..
            } => {
                for i in 0..old_len {
                    ga.push(Some(a[old_index + i].clone()));
                    gb.push(None);
                }
            }
            DiffOp::Insert {
                new_index, new_len, ..
            } => {
                for _ in 0..new_len {
                    ga.push(None);
                }
                for i in 0..new_len {
                    gb.push(Some(b[new_index + i].clone()));
                }
            }
            DiffOp::Replace {
                old_index,
                old_len,
                new_index,
                new_len,
                ..
            } => {
                let n = old_len.max(new_len);
                for i in 0..n {
                    ga.push(if i < old_len {
                        Some(a[old_index + i].clone())
                    } else {
                        None
                    });
                    gb.push(if i < new_len {
                        Some(b[new_index + i].clone())
                    } else {
                        None
                    });
                }
            }
        }
    }
    (ga, gb)
}

fn expand_gaps(gapped: &[Option<String>], mask: &[Option<String>]) -> Vec<Option<String>> {
    // mask has one non-None entry per element of gapped (in order).
    // For each non-None in mask, emit the next element of gapped (which may itself be None).
    // For each None in mask, emit None (new gap introduced by the third alignment pass).
    let mut src = gapped.iter();
    mask.iter()
        .map(|m| {
            if m.is_none() {
                None
            } else {
                src.next().cloned().unwrap()
            }
        })
        .collect()
}

/// Load audio from any format, converting via ffmpeg if not already WAV.
fn load_audio_any(path: &str) -> anyhow::Result<Vec<f32>> {
    let is_wav = path.to_lowercase().ends_with(".wav");
    if is_wav {
        return Ok(bee_qwen3_asr::mel::load_audio_wav(path, 16000)?);
    }
    // Convert to a temp WAV via ffmpeg
    let tmp = std::env::temp_dir().join("bee-compare-input.wav");
    println!("Converting to WAV via ffmpeg → {}", tmp.display());
    let status = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-i",
            path,
            "-ar",
            "16000",
            "-ac",
            "1",
            "-sample_fmt",
            "s16",
            tmp.to_str().unwrap(),
        ])
        .status()
        .map_err(|e| anyhow::anyhow!("ffmpeg not found: {e}"))?;
    anyhow::ensure!(status.success(), "ffmpeg failed with {status}");
    Ok(bee_qwen3_asr::mel::load_audio_wav(
        tmp.to_str().unwrap(),
        16000,
    )?)
}

fn cmd_compare(
    audio_path: &str,
    out_path: Option<&Path>,
    base_options: SessionOptions,
    modes_filter: Option<&[String]>,
) -> anyhow::Result<()> {
    let t0 = Instant::now();
    let engine = load_engine(true)?;
    println!("Engine loaded in {:.0}ms", t0.elapsed().as_millis());

    let samples = load_audio_any(audio_path)?;
    let duration = samples.len() as f64 / 16000.0;
    println!("Audio: {:.1}s ({} samples)\n", duration, samples.len());

    // (label, cut_strategy, bypass_audio_filters)
    let all_modes: &[(&str, RotationCutStrategy, bool)] = &[
        ("uncut", RotationCutStrategy::Uncut, false),
        ("qwen3", RotationCutStrategy::Qwen3, false),
        ("zipa", RotationCutStrategy::Zipa, false),
        ("raw", RotationCutStrategy::Uncut, true),
    ];
    let modes: Vec<(&str, RotationCutStrategy, bool)> = all_modes
        .iter()
        .filter(|(label, _, _)| {
            modes_filter
                .map(|f| f.iter().any(|m| m == label))
                .unwrap_or(true)
        })
        .cloned()
        .collect();
    if modes.is_empty() {
        let valid: Vec<&str> = all_modes.iter().map(|(l, _, _)| *l).collect();
        anyhow::bail!(
            "--modes filter matched nothing; valid: {}",
            valid.join(", ")
        );
    }

    let mut transcripts: Vec<(
        &str,
        String,
        Vec<bee_transcribe::CutEvent>,
        Vec<bee_transcribe::ChunkEvent>,
    )> = Vec::new();
    for (label, cut_mode, bypass) in modes {
        let (text, cuts, chunks) = run_mode(
            &engine,
            &samples,
            cut_mode.clone(),
            bypass,
            label,
            base_options.clone(),
        )?;
        transcripts.push((label, text, cuts, chunks));
    }

    // LCS 3-way alignment on normalized words
    let words: Vec<Vec<String>> = transcripts
        .iter()
        .map(|(_, t, _, _)| t.split_whitespace().map(|w| w.to_string()).collect())
        .collect();
    let norm: Vec<Vec<String>> = words
        .iter()
        .map(|ws| ws.iter().map(|w| normalize_word(w)).collect())
        .collect();

    // N-way greedy LCS alignment: repeatedly align the running consensus against
    // the next transcript, expanding all previous gap sequences each time.
    fn resolve_words(
        norm_seq: &[Option<String>],
        orig: &[String],
    ) -> (Vec<Option<String>>, Vec<usize>) {
        let mut it = orig.iter();
        let mut word_to_col = Vec::new();
        let aligned = norm_seq
            .iter()
            .enumerate()
            .map(|(col, n)| {
                if n.is_none() {
                    None
                } else {
                    word_to_col.push(col);
                    Some(it.next().unwrap().clone())
                }
            })
            .collect();
        (aligned, word_to_col)
    }

    let mut all_gapped: Vec<Vec<Option<String>>> =
        vec![norm[0].iter().map(|w| Some(w.clone())).collect()];
    let mut consensus_flat: Vec<String> = norm[0].clone();

    for i in 1..norm.len() {
        let (gc, gni) = align_pair(&consensus_flat, &norm[i]);
        // Expand all previous sequences to the new column count
        for seq in &mut all_gapped {
            *seq = expand_gaps(seq, &gc);
        }
        all_gapped.push(gni);
        // Rebuild flat consensus from all columns
        let ncols = all_gapped[0].len();
        consensus_flat = (0..ncols)
            .map(|col| {
                all_gapped
                    .iter()
                    .find_map(|seq| seq[col].clone())
                    .unwrap_or_default()
            })
            .collect();
    }

    // Map gapped norm sequences back to original display words + col index per word
    let resolved: Vec<(Vec<Option<String>>, Vec<usize>)> = all_gapped
        .iter()
        .enumerate()
        .map(|(i, gapped)| resolve_words(gapped, &words[i]))
        .collect();
    let aligned: Vec<Vec<Option<String>>> = resolved.iter().map(|(a, _)| a.clone()).collect();
    let word_to_col: Vec<Vec<usize>> = resolved.iter().map(|(_, w)| w.clone()).collect();

    // Compute cut-after column sets: after the last word of each CutEvent.
    let cut_after_cols: Vec<std::collections::HashSet<usize>> = transcripts
        .iter()
        .enumerate()
        .map(|(row, (_, _, cuts, _))| {
            let mut set = std::collections::HashSet::new();
            let mut word_idx: usize = 0;
            for cut in cuts {
                word_idx += cut.committed_words.len();
                // word_idx is now the index of the first word AFTER this cut;
                // the last word before the cut is word_idx - 1.
                if word_idx > 0 {
                    let last = word_idx - 1;
                    if let Some(&col) = word_to_col[row].get(last) {
                        set.insert(col);
                    }
                }
            }
            set
        })
        .collect();

    let ncols = aligned[0].len();

    // Determine cell class per column
    let nrows = aligned.len();
    let cell_classes: Vec<&str> = (0..ncols)
        .map(|i| {
            let vals: Vec<Option<String>> = aligned
                .iter()
                .map(|row| row[i].as_ref().map(|w| normalize_word(w)))
                .collect();
            let non_null: Vec<&String> = vals.iter().filter_map(|v| v.as_ref()).collect();
            if non_null.len() < 2 {
                return "diff";
            }
            if non_null.iter().all(|v| *v == non_null[0]) {
                return "same";
            }
            // "partial" if any two rows agree
            let any_pair_agrees = (0..nrows).any(|a| {
                (a + 1..nrows).any(|b| vals[a].is_some() && vals[b].is_some() && vals[a] == vals[b])
            });
            if any_pair_agrees { "partial" } else { "diff" }
        })
        .collect();

    // Generate HTML
    let chunk_size = 15;
    let mut body = String::new();
    for chunk_start in (0..ncols).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(ncols);
        body.push_str("<div class=\"chunk\"><table>\n<tr>");
        body.push_str("<td class=\"label\" style=\"background:transparent\"></td>");
        for i in chunk_start..chunk_end {
            body.push_str(&format!("<td class=\"idx\">{i}</td>"));
        }
        body.push_str("</tr>\n");
        for (row_idx, (label, _, _, _)) in transcripts.iter().enumerate() {
            body.push_str(&format!(
                "<tr><td class=\"label\">{}</td>",
                label.to_uppercase()
            ));
            for i in chunk_start..chunk_end {
                let is_cut = cut_after_cols[row_idx].contains(&i);
                let cut_style = if is_cut {
                    " style=\"border-right: 2px solid #cba6f7;\""
                } else {
                    ""
                };
                match &aligned[row_idx][i] {
                    None => body.push_str(&format!("<td class=\"gap\"{cut_style}>·</td>")),
                    Some(w) => {
                        let cls = cell_classes[i];
                        body.push_str(&format!("<td class=\"{cls}\"{cut_style}>{w}</td>"));
                    }
                }
            }
            body.push_str("</tr>\n");
        }
        body.push_str("</table></div>\n");
    }

    let wav_name = Path::new(audio_path)
        .file_name()
        .unwrap_or_default()
        .to_string_lossy();
    let chunk_ms = (base_options.chunk_duration * 1000.0).round() as u32;
    let commit_tokens = base_options.commit_token_count;
    let rollback_tokens = base_options.rollback_tokens;
    let context_tokens = base_options.context_tokens;

    // Compute output path early so audio files can sit next to the HTML
    let out = out_path.map(|p| p.to_path_buf()).unwrap_or_else(|| {
        std::env::temp_dir().join(format!(
            "bee-compare-{}.html",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        ))
    });
    let audio_dir = out.with_extension("audio");
    std::fs::create_dir_all(&audio_dir)?;

    // Build cuts detail section
    let mut cuts_detail =
        String::from("<h2 style=\"color:#cba6f7;margin-top:2em\">Cut Details</h2>\n");
    for (label, _, cuts, _) in &transcripts {
        cuts_detail.push_str(&format!(
            "<h3 style=\"color:#89b4fa;margin-bottom:0.5em\">{} — {} cut(s)</h3>\n",
            label.to_uppercase(),
            cuts.len()
        ));
        if cuts.is_empty() {
            cuts_detail.push_str("<p style=\"color:#6c7086\">No rotations.</p>\n");
            continue;
        }
        cuts_detail.push_str(
            "<table style=\"border-collapse:collapse;margin-bottom:1.5em;width:100%\">\n",
        );
        cuts_detail.push_str("<tr style=\"color:#45475a;font-size:11px\">");
        cuts_detail.push_str("<td style=\"padding:4px 8px\">#</td>");
        cuts_detail.push_str("<td style=\"padding:4px 8px\">time</td>");
        cuts_detail.push_str("<td style=\"padding:4px 8px\">committed</td>");
        cuts_detail.push_str("<td style=\"padding:4px 8px\">retained context</td>");
        cuts_detail.push_str("<td style=\"padding:4px 8px\">audio</td>");
        cuts_detail.push_str("</tr>\n");

        let mut cumulative_words: Vec<String> = Vec::new();
        for (i, cut) in cuts.iter().enumerate() {
            let t_start = cut.committed_words.first().map(|w| w.start).unwrap_or(0.0);
            let t_end = cut.committed_words.last().map(|w| w.end).unwrap_or(0.0);
            let committed_text: String = cut
                .committed_words
                .iter()
                .map(|w| w.word.trim())
                .collect::<Vec<_>>()
                .join(" ");

            cumulative_words.extend(
                cut.committed_words
                    .iter()
                    .map(|w| w.word.trim().to_string()),
            );
            let retained_start = cumulative_words.len().saturating_sub(context_tokens);
            let retained: String = cumulative_words[retained_start..].join(" ");

            // Write audio files
            let audio_stem = format!("{label}-cut{:02}", i + 1);
            let committed_path = audio_dir.join(format!("{audio_stem}-committed.wav"));
            let remaining_path = audio_dir.join(format!("{audio_stem}-remaining.wav"));
            let audio_dir_name = audio_dir.file_name().unwrap_or_default().to_string_lossy();
            if let Err(e) = write_wav(&committed_path, &cut.committed_audio) {
                eprintln!("warn: failed to write {}: {e}", committed_path.display());
            }
            if let Err(e) = write_wav(&remaining_path, &cut.remaining_audio) {
                eprintln!("warn: failed to write {}: {e}", remaining_path.display());
            }

            let row_bg = if i % 2 == 0 { "#181825" } else { "#1e1e2e" };
            cuts_detail.push_str(&format!(
                "<tr style=\"background:{row_bg}\">\
                <td style=\"padding:5px 8px;color:#45475a\">{n}</td>\
                <td style=\"padding:5px 8px;color:#6c7086;white-space:nowrap\">{t_start:.2}s–{t_end:.2}s</td>\
                <td style=\"padding:5px 8px;color:#cdd6f4\">{committed_text}</td>\
                <td style=\"padding:5px 8px;color:#f9e2af\">{retained}</td>\
                <td style=\"padding:5px 8px\">\
                  <div style=\"font-size:10px;color:#6c7086;margin-bottom:2px\">committed</div>\
                  <audio controls style=\"height:24px;width:220px\"><source src=\"{audio_dir_name}/{audio_stem}-committed.wav\" type=\"audio/wav\"></audio>\
                  <div style=\"font-size:10px;color:#6c7086;margin:4px 0 2px\">remaining (fed back)</div>\
                  <audio controls style=\"height:24px;width:220px\"><source src=\"{audio_dir_name}/{audio_stem}-remaining.wav\" type=\"audio/wav\"></audio>\
                </td>\
                </tr>\n",
                n = i + 1,
            ));
        }
        cuts_detail.push_str("</table>\n");
    }
    const WAVEFORM_POINTS: usize = 2000;

    // Stitch per-mode before/after WAVs, downsample for embedded waveform data.
    let mut waveforms_js = String::from("window.WAVEFORMS = {\n");
    let mut chunk_timeline =
        String::from("<h2 style=\"color:#cba6f7;margin-top:2em\">VAD Chunk Timeline</h2>\n");
    let audio_dir_name = audio_dir.file_name().unwrap_or_default().to_string_lossy();

    fn fmt_waveform(pts: &[[f32; 2]]) -> String {
        let inner: Vec<String> = pts
            .iter()
            .map(|[lo, hi]| format!("[{:.4},{:.4}]", lo, hi))
            .collect();
        format!("[{}]", inner.join(","))
    }

    for (label, _, _, chunks) in &transcripts {
        if chunks.is_empty() {
            continue;
        }
        let sr = chunks[0].raw_audio.sample_rate();

        // Stitch: before = all raw. after = filtered if passed, silence if dropped.
        let mut before_samples: Vec<f32> = Vec::new();
        let mut after_samples: Vec<f32> = Vec::new();
        for chunk in chunks.iter() {
            before_samples.extend_from_slice(chunk.raw_audio.samples());
            match &chunk.filtered_audio {
                Some(f) => after_samples.extend_from_slice(f.samples()),
                None => after_samples.extend(std::iter::repeat(0.0f32).take(chunk.raw_audio.len())),
            }
        }

        let before_buf = bee_transcribe::audio_buffer::AudioBuffer::new(before_samples.clone(), sr);
        let after_buf = bee_transcribe::audio_buffer::AudioBuffer::new(after_samples.clone(), sr);

        let before_path = audio_dir.join(format!("{label}-all-before.wav"));
        let after_path = audio_dir.join(format!("{label}-all-after.wav"));
        if let Err(e) = write_wav(&before_path, &before_buf) {
            eprintln!("warn: {e}");
        }
        if let Err(e) = write_wav(&after_path, &after_buf) {
            eprintln!("warn: {e}");
        }
        let before_name = before_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy();
        let after_name = after_path.file_name().unwrap_or_default().to_string_lossy();

        // Downsample for JS embedding
        let before_pts = downsample_waveform(&before_samples, WAVEFORM_POINTS);
        let after_pts = downsample_waveform(&after_samples, WAVEFORM_POINTS);
        waveforms_js.push_str(&format!(
            "  \"{label}\": {{ before: {}, after: {} }},\n",
            fmt_waveform(&before_pts),
            fmt_waveform(&after_pts),
        ));

        chunk_timeline.push_str(&format!(
            "<h3 style=\"color:#89b4fa;margin-bottom:0.5em\">{label_up}</h3>\
             <div style=\"margin-bottom:0.5em;display:flex;gap:0.8em;align-items:center\">\
               <button onclick=\"playAudio('{label}-before')\" \
                 style=\"background:#1a3a1a;color:#a6e3a1;border:1px solid #a6e3a1;\
                         border-radius:4px;padding:4px 12px;cursor:pointer;font-family:monospace\">▶ Before</button>\
               <button onclick=\"playAudio('{label}-after')\" \
                 style=\"background:#1e2a3a;color:#89b4fa;border:1px solid #89b4fa;\
                         border-radius:4px;padding:4px 12px;cursor:pointer;font-family:monospace\">▶ After</button>\
               <button onclick=\"stopAll()\" \
                 style=\"background:#181825;color:#6c7086;border:1px solid #45475a;\
                         border-radius:4px;padding:4px 12px;cursor:pointer;font-family:monospace\">■ Stop</button>\
               <span style=\"color:#6c7086;font-size:10px\">{n_chunks} chunks · {dur_s:.1}s</span>\
             </div>\
             <audio id=\"{label}-before\" preload=\"auto\" src=\"{audio_dir_name}/{before_name}\"></audio>\
             <audio id=\"{label}-after\"  preload=\"auto\" src=\"{audio_dir_name}/{after_name}\"></audio>\
             <div class=\"wf-wrapper\" data-audio-before=\"{label}-before\" data-audio-after=\"{label}-after\" \
               style=\"position:relative;cursor:crosshair;margin-bottom:1.5em;border-radius:6px;overflow:hidden\">\
               <canvas data-label=\"{label}\" width=\"2000\" height=\"240\" \
                 style=\"width:100%;height:120px;display:block;background:#11111b\"></canvas>\
               <div class=\"wf-cursor\" style=\"display:none;position:absolute;top:0;width:2px;height:100%;background:#f5c2e7;pointer-events:none\"></div>\
             </div>\n",
            label_up = label.to_uppercase(),
            n_chunks = chunks.len(),
            dur_s = before_buf.len() as f32 / sr.0 as f32,
        ));
    }
    waveforms_js.push_str("};\n");

    let html = format!(
        r#"<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Cut Mode Comparison — {wav_name}</title>
<style>
  body {{ font-family: monospace; background: #1e1e2e; color: #cdd6f4; padding: 2em; font-size: 13px; }}
  h1 {{ color: #cba6f7; margin-bottom: 0.2em; }}
  p.subtitle {{ color: #6c7086; margin-top: 0; margin-bottom: 1em; }}
  .settings {{ color: #6c7086; margin-bottom: 1.5em; font-size: 0.9em; }}
  .settings span {{ color: #a6e3a1; margin-right: 1.5em; }}
  .legend {{ display: flex; gap: 1.5em; margin-bottom: 1.5em; }}
  .legend-item {{ display: flex; align-items: center; gap: 0.5em; }}
  .swatch {{ display: inline-block; width: 12px; height: 12px; border-radius: 2px; }}
  .chunk {{ margin-bottom: 1.5em; }}
  table {{ border-collapse: separate; border-spacing: 3px 3px; }}
  td {{ padding: 5px 8px; border-radius: 4px; text-align: center; white-space: nowrap; }}
  td.label {{ text-align: right; background: #181825; color: #cba6f7; font-weight: bold; padding-right: 12px; width: 3.5em; }}
  td.idx {{ background: transparent; color: #45475a; font-size: 10px; padding: 2px 8px; }}
  td.same    {{ background: #2a2a3e; color: #cdd6f4; }}
  td.partial {{ background: #3a2f10; color: #f9e2af; }}
  td.diff    {{ background: #3d1a1a; color: #f38ba8; }}
  td.gap     {{ background: transparent; color: #313244; }}
</style>
<script>
{waveforms_js}

let currentAudio = null;

function stopAll() {{
  document.querySelectorAll('audio').forEach(a => {{ a.pause(); a.currentTime = 0; }});
  currentAudio = null;
}}
function playAudio(id) {{
  stopAll();
  const a = document.getElementById(id);
  currentAudio = a;
  a.play();
}}

function drawWaveform(canvas) {{
  const label = canvas.dataset.label;
  const data  = window.WAVEFORMS[label];
  if (!data) {{ console.warn('no waveform data for', label); return; }}

  const W = canvas.width;
  const H = canvas.height;
  const c = canvas.getContext('2d');
  c.clearRect(0, 0, W, H);

  function drawPts(pts, color) {{
    const n   = pts.length;
    const mid = H / 2;
    c.beginPath();
    c.strokeStyle = color;
    c.lineWidth = 1;
    for (let x = 0; x < W; x++) {{
      const idx = Math.floor(x * n / W);
      const [lo, hi] = pts[idx] || [0, 0];
      c.moveTo(x + 0.5, mid + lo * mid * 0.95);
      c.lineTo(x + 0.5, mid + hi * mid * 0.95);
    }}
    c.stroke();
  }}

  drawPts(data.before, 'rgba(137,180,250,0.4)');
  drawPts(data.after,  'rgba(166,227,161,0.9)');
}}

// Cursor: one per waveform wrapper, driven by whichever audio is playing.
function setupWaveformWrapper(wrapper) {{
  const canvas = wrapper.querySelector('canvas');
  const cursor = wrapper.querySelector('.wf-cursor');
  const audioIds = [wrapper.dataset.audioBefore, wrapper.dataset.audioAfter];

  // Click to seek
  wrapper.addEventListener('click', e => {{
    const rect = wrapper.getBoundingClientRect();
    const pct  = (e.clientX - rect.left) / rect.width;
    // find which audio belongs to this wrapper and is current, or just seek both
    audioIds.forEach(id => {{
      const a = document.getElementById(id);
      if (a && isFinite(a.duration)) a.currentTime = pct * a.duration;
    }});
    // play the "before" one if nothing is current for this wrapper
    if (!currentAudio || !audioIds.includes(currentAudio.id)) {{
      playAudio(audioIds[0]);
      const a = document.getElementById(audioIds[0]);
      if (a && isFinite(a.duration)) a.currentTime = pct * a.duration;
    }}
  }});

  function tick() {{
    // Show cursor if one of our audio elements is the current one
    const active = audioIds.map(id => document.getElementById(id)).find(a => a === currentAudio);
    if (active && isFinite(active.duration) && active.duration > 0) {{
      const pct = active.currentTime / active.duration;
      cursor.style.left = (pct * 100).toFixed(3) + '%';
      cursor.style.display = 'block';
    }} else if (currentAudio === null) {{
      cursor.style.display = 'none';
    }}
    requestAnimationFrame(tick);
  }}
  tick();
}}

document.addEventListener('DOMContentLoaded', () => {{
  document.querySelectorAll('canvas[data-label]').forEach(drawWaveform);
  document.querySelectorAll('.wf-wrapper').forEach(setupWaveformWrapper);
}});
</script>
</head>
<body>
<h1>Cut Mode Comparison — {wav_name}</h1>
<p class="subtitle">LCS-aligned · 15 word positions per block · gaps (·) where a mode has no word · purple border = cut point</p>
<p class="settings">
  <span>chunk {chunk_ms}ms</span>
  <span>commit-tokens {commit_tokens}</span>
  <span>rollback-tokens {rollback_tokens}</span>
  <span>context-tokens {context_tokens}</span>
</p>
<div class="legend">
  <div class="legend-item"><span class="swatch" style="background:#2a2a3e;border:1px solid #45475a"></span> All agree</div>
  <div class="legend-item"><span class="swatch" style="background:#3a2f10"></span> Two agree</div>
  <div class="legend-item"><span class="swatch" style="background:#3d1a1a"></span> All differ</div>
</div>
{body}
{chunk_timeline}
{cuts_detail}
</body>
</html>
"#
    );

    std::fs::write(&out, &html)?;
    println!("\nReport written to: {}", out.display());
    // Try to open in browser
    let _ = std::process::Command::new("open").arg(&out).status();
    Ok(())
}

/// Print per-word alternatives with confidence info.
///
/// Groups tokens by word boundaries (WordStart markers), decodes each
/// alternative token, and shows concentration/margin for the chosen token.
fn print_alternatives(tokenizer: &Tokenizer, entries: &[TokenEntry]) {
    if entries.is_empty() {
        return;
    }

    // Group entries into words
    let mut words: Vec<&[TokenEntry]> = Vec::new();
    let mut word_start = None;
    for (i, entry) in entries.iter().enumerate() {
        if entry.word.is_some() {
            if let Some(start) = word_start {
                words.push(&entries[start..i]);
            }
            word_start = Some(i);
        }
    }
    if let Some(start) = word_start {
        words.push(&entries[start..]);
    }

    println!("    ┌─ alternatives ─────────────────────────────────");
    for word_entries in &words {
        // Decode the chosen word
        let word_ids: Vec<u32> = word_entries.iter().map(|e| e.token.id).collect();
        let word_text = tokenizer.decode(&word_ids, true).unwrap_or_default();

        // Average confidence across tokens in this word
        let n = word_entries.len() as f32;
        let avg_conc: f32 = word_entries
            .iter()
            .map(|e| e.token.concentration)
            .sum::<f32>()
            / n;
        let avg_margin: f32 = word_entries.iter().map(|e| e.token.margin).sum::<f32>() / n;

        // Collect alternatives for each token position
        let alt_count = word_entries
            .iter()
            .map(|entry| entry.token.alternative_count as usize)
            .min()
            .unwrap_or(0);
        let mut alt_columns: Vec<Vec<String>> = vec![Vec::new(); alt_count.max(1)];
        for entry in *word_entries {
            for k in 0..alt_count {
                let alt_text = tokenizer
                    .decode(&[entry.token.top_ids[k]], true)
                    .unwrap_or_default();
                alt_columns[k].push(alt_text);
            }
        }

        // Format: chosen word, then alternatives
        let alts: Vec<String> = (1..alt_count)
            .map(|k| {
                let alt_word: String = alt_columns[k].join("");
                let alt_word = alt_word.trim();
                if alt_word.is_empty() || alt_word == word_text.trim() {
                    return String::new();
                }
                // Show the logit delta from top-1
                let avg_delta: f32 = word_entries
                    .iter()
                    .map(|e| e.token.top_logits[0] - e.token.top_logits[k])
                    .sum::<f32>()
                    / n;
                format!("{alt_word}(-{avg_delta:.1})")
            })
            .filter(|s| !s.is_empty())
            .collect();

        let alts_str = if alts.is_empty() {
            String::new()
        } else {
            format!("  alts: {}", alts.join(", "))
        };

        println!(
            "    │ {:>20}  conc={:5.1} margin={:5.1}{alts_str}",
            word_text.trim(),
            avg_conc,
            avg_margin,
        );
    }
    println!("    └──────────────────────────────────────────────");
}
