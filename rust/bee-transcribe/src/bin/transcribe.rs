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
        let out_path = rest
            .windows(2)
            .find(|w| w[0] == "--out")
            .map(|w| PathBuf::from(&w[1]));
        let positional: Vec<&str> = rest
            .iter()
            .filter(|a| {
                !a.starts_with("--")
                    && !rest
                        .windows(2)
                        .any(|w| w[0] == "--out" && &w[1] == a.as_str())
            })
            .map(|s| s.as_str())
            .collect();
        if positional.is_empty() {
            eprintln!("Usage: transcribe compare <audio.wav> [--out report.html]");
            std::process::exit(1);
        }
        return cmd_compare(positional[0], out_path.as_deref());
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
    label: &str,
) -> anyhow::Result<(String, Vec<bee_transcribe::CutEvent>)> {
    println!("--- Running {label} ---");
    let mut options = SessionOptions::default();
    // For uncut, use a very large chunk so it processes as one pass
    if matches!(cut_mode, RotationCutStrategy::Uncut) {
        options.chunk_duration = 600.0;
    }
    options.rotation_cut_strategy = cut_mode;
    let chunk_samples = (options.chunk_duration * 16000.0) as usize;

    let cuts = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let cuts_sink = cuts.clone();
    let sink: bee_transcribe::CutSink = Box::new(move |event: bee_transcribe::CutEvent| {
        cuts_sink.borrow_mut().push(event);
    });
    let mut session = engine.session_with_sink(options, Some(sink))?;
    let mut offset = 0;
    while offset < samples.len() {
        let end = (offset + chunk_samples).min(samples.len());
        session.feed(&samples[offset..end])?;
        offset = end;
    }
    let result = session.finish()?;
    let text = result.snapshot.full_text.clone();
    let cuts = std::rc::Rc::try_unwrap(cuts).ok().unwrap().into_inner();
    println!("  {label}: {:?} ({} cuts)", text, cuts.len());
    Ok((text, cuts))
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
            DiffOp::Equal { old_index, len, .. } => {
                for i in 0..len {
                    ga.push(Some(a[old_index + i].clone()));
                    gb.push(Some(b[old_index + i].clone()));
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
    let mut src = gapped.iter().filter(|v| v.is_some());
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

fn cmd_compare(audio_path: &str, out_path: Option<&Path>) -> anyhow::Result<()> {
    let t0 = Instant::now();
    let engine = load_engine(true)?;
    println!("Engine loaded in {:.0}ms", t0.elapsed().as_millis());

    let samples = bee_qwen3_asr::mel::load_audio_wav(audio_path, 16000)?;
    let duration = samples.len() as f64 / 16000.0;
    println!("Audio: {:.1}s ({} samples)\n", duration, samples.len());

    let modes: &[(&str, RotationCutStrategy)] = &[
        ("uncut", RotationCutStrategy::Uncut),
        ("qwen3", RotationCutStrategy::Qwen3),
        ("zipa", RotationCutStrategy::Zipa),
    ];

    let mut transcripts: Vec<(&str, String, Vec<bee_transcribe::CutEvent>)> = Vec::new();
    for (label, cut_mode) in modes {
        let (text, cuts) = run_mode(&engine, &samples, cut_mode.clone(), label)?;
        transcripts.push((label, text, cuts));
    }

    // LCS 3-way alignment on normalized words
    let words: Vec<Vec<String>> = transcripts
        .iter()
        .map(|(_, t, _)| t.split_whitespace().map(|w| w.to_string()).collect())
        .collect();
    let norm: Vec<Vec<String>> = words
        .iter()
        .map(|ws| ws.iter().map(|w| normalize_word(w)).collect())
        .collect();

    // Step 1: align norm[0] vs norm[1]
    let (gn0, gn1) = align_pair(&norm[0], &norm[1]);
    // consensus: first non-None wins
    let consensus: Vec<String> = gn0
        .iter()
        .zip(gn1.iter())
        .map(|(a, b)| a.clone().or_else(|| b.clone()).unwrap_or_default())
        .collect();
    // Step 2: align consensus vs norm[2]
    let (gc, gn2) = align_pair(&consensus, &norm[2]);
    // Step 3: expand gn0/gn1 to match gc's new gaps
    let gn0e = expand_gaps(&gn0, &gc);
    let gn1e = expand_gaps(&gn1, &gc);

    // Map norm gaps back to original words, also returning col index per original word.
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
    let (al0, w2c0) = resolve_words(&gn0e, &words[0]);
    let (al1, w2c1) = resolve_words(&gn1e, &words[1]);
    let (al2, w2c2) = resolve_words(&gn2, &words[2]);
    let aligned = [al0, al1, al2];
    let word_to_col = [w2c0, w2c1, w2c2];

    // Compute cut-after column sets: after the last word of each CutEvent.
    let cut_after_cols: Vec<std::collections::HashSet<usize>> = transcripts
        .iter()
        .enumerate()
        .map(|(row, (_, _, cuts))| {
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
            let pairs = [(0, 1), (0, 2), (1, 2)];
            if pairs
                .iter()
                .any(|(a, b)| vals[*a].is_some() && vals[*b].is_some() && vals[*a] == vals[*b])
            {
                return "partial";
            }
            "diff"
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
        for (row_idx, (label, _, _)) in transcripts.iter().enumerate() {
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
    let html = format!(
        r#"<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Cut Mode Comparison — {wav_name}</title>
<style>
  body {{ font-family: monospace; background: #1e1e2e; color: #cdd6f4; padding: 2em; font-size: 13px; }}
  h1 {{ color: #cba6f7; margin-bottom: 0.2em; }}
  p.subtitle {{ color: #6c7086; margin-top: 0; margin-bottom: 1.5em; }}
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
</head>
<body>
<h1>Cut Mode Comparison — {wav_name}</h1>
<p class="subtitle">LCS-aligned. 15 word positions per block. Gaps (·) where a mode has no word.</p>
<div class="legend">
  <div class="legend-item"><span class="swatch" style="background:#2a2a3e;border:1px solid #45475a"></span> All agree</div>
  <div class="legend-item"><span class="swatch" style="background:#3a2f10"></span> Two agree</div>
  <div class="legend-item"><span class="swatch" style="background:#3d1a1a"></span> All differ</div>
</div>
{body}
</body>
</html>
"#
    );

    let out = out_path.map(|p| p.to_path_buf()).unwrap_or_else(|| {
        std::env::temp_dir().join(format!(
            "bee-compare-{}.html",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        ))
    });
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
