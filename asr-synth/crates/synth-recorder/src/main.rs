use anyhow::{Context, Result};
use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::terminal;
use ratatui::prelude::*;
use ratatui::widgets::*;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

#[derive(Parser)]
#[command(about = "Record eval sentences with mic profile support")]
struct Args {
    /// Mic profile name (e.g. "desk-sm7b", "laptop-bed", "airpods")
    #[arg(short, long)]
    profile: String,

    /// Path to eval sentences JSONL
    #[arg(short, long, default_value = "data/eval_sentences.jsonl")]
    sentences: String,

    /// Output directory for recordings
    #[arg(short, long, default_value = "data/eval_recordings")]
    output: String,

    /// Start from sentence N (0-indexed, for resuming)
    #[arg(long, default_value = "0")]
    start: usize,
}

#[derive(serde::Deserialize)]
struct EvalSentence {
    text: String,
    spoken: String,
    vocab_terms: Vec<String>,
}

#[derive(serde::Serialize)]
struct RecordingManifest {
    sentence_index: usize,
    text: String,
    spoken: String,
    vocab_terms: Vec<String>,
    profile: String,
    wav_path: String,
    sample_rate: u32,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Load sentences
    let sentences: Vec<EvalSentence> = std::fs::read_to_string(&args.sentences)
        .context("reading sentences file")?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).context("parsing sentence"))
        .collect::<Result<Vec<_>>>()?;

    if args.start >= sentences.len() {
        println!(
            "Nothing to record (start={} >= total={})",
            args.start,
            sentences.len()
        );
        return Ok(());
    }

    // Create output directory
    let profile_dir = PathBuf::from(&args.output).join(&args.profile);
    std::fs::create_dir_all(&profile_dir)?;

    // Open manifest file (append mode for resuming)
    let manifest_path = profile_dir.join("manifest.jsonl");
    let mut manifest = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&manifest_path)?;

    // Set up audio — always recording
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .context("no input device found")?;
    let device_name = device.name().unwrap_or_else(|_| "unknown".to_string());
    let config = device.default_input_config()?;
    let sample_rate = config.sample_rate().0;
    let channels = config.channels() as usize;

    let buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let buf_clone = buffer.clone();

    let stream = device.build_input_stream(
        &config.into(),
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let mut buf = buf_clone.lock().unwrap();
            if channels == 1 {
                buf.extend_from_slice(data);
            } else {
                for chunk in data.chunks(channels) {
                    let mono: f32 = chunk.iter().sum::<f32>() / channels as f32;
                    buf.push(mono);
                }
            }
        },
        |err| eprintln!("Audio error: {err}"),
        None,
    )?;
    stream.play()?;

    // Set up terminal
    terminal::enable_raw_mode()?;
    let mut terminal = Terminal::new(CrosstermBackend::new(std::io::stdout()))?;
    terminal.clear()?;

    let mut i = args.start;
    let mut status_msg = String::new();
    let mut recorded_count = 0usize;

    loop {
        if i >= sentences.len() {
            break;
        }
        let sentence = &sentences[i];

        // Compute current audio level for VU meter
        let (buf_len, peak) = {
            let buf = buffer.lock().unwrap();
            let len = buf.len();
            let peak = buf
                .iter()
                .rev()
                .take(1600)
                .fold(0.0f32, |m, &s| m.max(s.abs()));
            (len, peak)
        };
        let buf_secs = buf_len as f32 / sample_rate as f32;

        // Draw UI
        terminal.draw(|frame| {
            let area = frame.area();

            let chunks = Layout::vertical([
                Constraint::Length(3), // header
                Constraint::Min(8),    // sentence
                Constraint::Length(3), // audio meter + status
                Constraint::Length(3), // controls
            ])
            .split(area);

            // Header
            let header = Paragraph::new(Line::from(vec![
                Span::styled(
                    format!(" {} ", args.profile),
                    Style::default().fg(Color::Black).bg(Color::Cyan).bold(),
                ),
                Span::raw("  "),
                Span::styled(&device_name, Style::default().fg(Color::DarkGray)),
                Span::raw(format!("  {} Hz", sample_rate)),
                Span::raw("  "),
                Span::styled(
                    format!("{} recorded", recorded_count),
                    Style::default().fg(Color::Green),
                ),
            ]))
            .block(Block::bordered().title(format!(
                " Sentence {}/{} ",
                i + 1,
                sentences.len()
            )));
            frame.render_widget(header, chunks[0]);

            // Sentence content
            let lines = vec![
                Line::raw(""),
                Line::from(Span::styled(
                    &sentence.text,
                    Style::default().fg(Color::White).bold(),
                )),
                Line::raw(""),
                Line::from(vec![
                    Span::styled("say: ", Style::default().fg(Color::DarkGray)),
                    Span::styled(&sentence.spoken, Style::default().fg(Color::Yellow)),
                ]),
                Line::raw(""),
                Line::from(vec![
                    Span::styled("vocab: ", Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        sentence.vocab_terms.join(", "),
                        Style::default().fg(Color::Green),
                    ),
                ]),
            ];
            let content = Paragraph::new(lines)
                .block(Block::bordered())
                .wrap(Wrap { trim: false });
            frame.render_widget(content, chunks[1]);

            // Audio meter + status
            let meter_width = (chunks[2].width.saturating_sub(4)) as usize;
            let level = (peak * meter_width as f32).min(meter_width as f32) as usize;
            let meter_bar: String =
                "█".repeat(level) + &"░".repeat(meter_width.saturating_sub(level));
            let meter_style = if peak > 0.05 {
                Color::Red
            } else {
                Color::DarkGray
            };

            let status_line = if status_msg.is_empty() {
                Line::from(vec![
                    Span::styled(" ● ", Style::default().fg(Color::Red)),
                    Span::styled(&meter_bar, Style::default().fg(meter_style)),
                    Span::styled(
                        format!(" {:.1}s", buf_secs),
                        Style::default().fg(Color::DarkGray),
                    ),
                ])
            } else {
                Line::from(Span::styled(
                    format!("  {}", &status_msg),
                    Style::default().fg(Color::Green),
                ))
            };
            let status = Paragraph::new(status_line).block(Block::bordered());
            frame.render_widget(status, chunks[2]);

            // Controls
            let ctrl = Paragraph::new(Line::from(Span::styled(
                " Space: save & next    R: redo    S: skip    Q: quit ",
                Style::default().fg(Color::Cyan),
            )))
            .block(Block::bordered());
            frame.render_widget(ctrl, chunks[3]);
        })?;

        // Poll with short timeout for VU meter updates
        if !event::poll(std::time::Duration::from_millis(50))? {
            continue;
        }

        if let Event::Key(KeyEvent {
            code, modifiers, ..
        }) = event::read()?
        {
            match code {
                KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc => break,
                KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => break,

                // Save current recording & advance
                KeyCode::Char(' ') | KeyCode::Enter => {
                    let samples: Vec<f32> = {
                        let mut buf = buffer.lock().unwrap();
                        let s = buf.clone();
                        buf.clear();
                        s
                    };

                    let duration = samples.len() as f32 / sample_rate as f32;
                    if duration < 0.5 {
                        status_msg = format!("Too short ({:.1}s), speak first", duration);
                        continue;
                    }

                    let wav_name = format!("{:04}.wav", i);
                    let wav_path = profile_dir.join(&wav_name);
                    write_wav_16k(&wav_path, &samples, sample_rate)?;

                    let entry = RecordingManifest {
                        sentence_index: i,
                        text: sentence.text.clone(),
                        spoken: sentence.spoken.clone(),
                        vocab_terms: sentence.vocab_terms.clone(),
                        profile: args.profile.clone(),
                        wav_path: wav_name,
                        sample_rate: 16000,
                    };
                    serde_json::to_writer(&mut manifest, &entry)?;
                    manifest.write_all(b"\n")?;
                    manifest.flush()?;

                    recorded_count += 1;
                    status_msg = format!("Saved {:.1}s", duration);
                    i += 1;
                    // Clear buffer for next sentence
                    buffer.lock().unwrap().clear();
                }

                // Redo — clear buffer, stay on same sentence
                KeyCode::Char('r') | KeyCode::Char('R') => {
                    buffer.lock().unwrap().clear();
                    status_msg = "Buffer cleared, speak again".to_string();
                }

                // Skip
                KeyCode::Char('s') | KeyCode::Char('S') => {
                    buffer.lock().unwrap().clear();
                    status_msg.clear();
                    i += 1;
                }

                _ => {}
            }
        }
    }

    // Cleanup
    drop(stream);
    drop(terminal);
    terminal::disable_raw_mode()?;

    println!(
        "\nDone! {} recordings in {}",
        recorded_count,
        profile_dir.display()
    );
    Ok(())
}

fn write_wav_16k(path: &Path, samples: &[f32], source_rate: u32) -> Result<()> {
    let samples_16k = if source_rate == 16000 {
        samples.to_vec()
    } else {
        let ratio = 16000.0 / source_rate as f64;
        let out_len = (samples.len() as f64 * ratio) as usize;
        let mut out = Vec::with_capacity(out_len);
        for i in 0..out_len {
            let src_pos = i as f64 / ratio;
            let idx = src_pos as usize;
            let frac = src_pos - idx as f64;
            let s0 = samples.get(idx).copied().unwrap_or(0.0);
            let s1 = samples.get(idx + 1).copied().unwrap_or(s0);
            out.push(s0 + (s1 - s0) * frac as f32);
        }
        out
    };

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &s in &samples_16k {
        writer.write_sample((s * 32767.0f32).clamp(-32768.0, 32767.0) as i16)?;
    }
    writer.finalize()?;
    Ok(())
}
