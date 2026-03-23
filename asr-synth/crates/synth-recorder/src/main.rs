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
use std::time::Instant;

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

#[derive(PartialEq)]
enum State {
    Waiting,
    Recording,
    Recorded,
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
        println!("Nothing to record (start={} >= total={})", args.start, sentences.len());
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

    // Set up audio
    let host = cpal::default_host();
    let device = host.default_input_device().context("no input device found")?;
    let device_name = device.name().unwrap_or_else(|_| "unknown".to_string());
    let config = device.default_input_config()?;
    let sample_rate = config.sample_rate().0;
    let channels = config.channels() as usize;

    // Set up terminal
    terminal::enable_raw_mode()?;
    let mut terminal = Terminal::new(CrosstermBackend::new(std::io::stdout()))?;
    terminal.clear()?;

    let mut i = args.start;
    let mut state = State::Waiting;
    let mut recording_start: Option<Instant> = None;
    let mut last_duration = 0.0f32;
    let mut status_msg = String::new();
    let buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let mut active_stream: Option<cpal::Stream> = None;

    loop {
        if i >= sentences.len() {
            break;
        }
        let sentence = &sentences[i];

        // Draw UI
        let recording_secs = recording_start
            .map(|s| s.elapsed().as_secs_f32())
            .unwrap_or(0.0);

        terminal.draw(|frame| {
            let area = frame.area();

            let chunks = Layout::vertical([
                Constraint::Length(3),  // header
                Constraint::Min(6),    // sentence
                Constraint::Length(3), // status
                Constraint::Length(3),  // controls
            ])
            .split(area);

            // Header
            let header = Paragraph::new(Line::from(vec![
                Span::styled(
                    format!(" {} ", args.profile),
                    Style::default().fg(Color::Black).bg(Color::Cyan).bold(),
                ),
                Span::raw("  "),
                Span::styled(
                    format!("{}", device_name),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::raw(format!("  {} Hz  {} ch", sample_rate, channels)),
            ]))
            .block(Block::bordered().title(format!(
                " Sentence {}/{} ",
                i + 1,
                sentences.len()
            )));
            frame.render_widget(header, chunks[0]);

            // Sentence content
            let mut lines = vec![
                Line::raw(""),
                Line::from(Span::styled(
                    &sentence.text,
                    Style::default().fg(Color::White).bold(),
                )),
                Line::raw(""),
                Line::from(vec![
                    Span::styled("say: ", Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        &sentence.spoken,
                        Style::default().fg(Color::Yellow),
                    ),
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

            // Status
            let status_line = match state {
                State::Waiting => Line::from(Span::styled(
                    "  Ready to record",
                    Style::default().fg(Color::DarkGray),
                )),
                State::Recording => Line::from(vec![
                    Span::styled(" ● REC ", Style::default().fg(Color::White).bg(Color::Red).bold()),
                    Span::raw(format!("  {:.1}s", recording_secs)),
                ]),
                State::Recorded => {
                    let msg = if status_msg.is_empty() {
                        format!("  Saved ({:.1}s)", last_duration)
                    } else {
                        format!("  {}", status_msg)
                    };
                    Line::from(Span::styled(
                        msg,
                        Style::default().fg(Color::Green),
                    ))
                }
            };
            let status = Paragraph::new(status_line).block(Block::bordered().title(" Status "));
            frame.render_widget(status, chunks[2]);

            // Controls
            let controls = match state {
                State::Waiting => " Space: record  S: skip  Q: quit ",
                State::Recording => " Space: stop recording ",
                State::Recorded => " Space: next  R: redo  Q: quit ",
            };
            let ctrl = Paragraph::new(Line::from(Span::styled(
                controls,
                Style::default().fg(Color::Cyan),
            )))
            .block(Block::bordered());
            frame.render_widget(ctrl, chunks[3]);
        })?;

        // Poll events (with timeout for recording animation)
        let timeout = if state == State::Recording {
            std::time::Duration::from_millis(100)
        } else {
            std::time::Duration::from_secs(60)
        };

        if !event::poll(timeout)? {
            continue;
        }

        if let Event::Key(KeyEvent { code, modifiers, .. }) = event::read()? {
            match (&state, code) {
                // Quit
                (_, KeyCode::Char('q')) | (_, KeyCode::Char('Q')) | (_, KeyCode::Esc) => break,
                (_, KeyCode::Char('c')) if modifiers.contains(KeyModifiers::CONTROL) => break,

                // Start recording
                (State::Waiting, KeyCode::Char(' ') | KeyCode::Enter) => {
                    buffer.lock().unwrap().clear();
                    let buf_clone = buffer.clone();
                    let ch = channels;

                    let stream = device.build_input_stream(
                        &config.clone().into(),
                        move |data: &[f32], _: &cpal::InputCallbackInfo| {
                            let mut buf = buf_clone.lock().unwrap();
                            if ch == 1 {
                                buf.extend_from_slice(data);
                            } else {
                                for chunk in data.chunks(ch) {
                                    let mono: f32 = chunk.iter().sum::<f32>() / ch as f32;
                                    buf.push(mono);
                                }
                            }
                        },
                        |err| eprintln!("Audio error: {err}"),
                        None,
                    )?;
                    stream.play()?;
                    active_stream = Some(stream);
                    recording_start = Some(Instant::now());
                    state = State::Recording;
                }

                // Stop recording
                (State::Recording, KeyCode::Char(' ') | KeyCode::Enter) => {
                    drop(active_stream.take());
                    let duration = recording_start.take().map(|s| s.elapsed().as_secs_f32()).unwrap_or(0.0);
                    let samples = buffer.lock().unwrap().clone();

                    if samples.is_empty() || duration < 0.3 {
                        status_msg = format!("Too short ({:.1}s), try again", duration);
                        state = State::Waiting;
                        continue;
                    }

                    // Write WAV
                    let wav_name = format!("{:04}.wav", i);
                    let wav_path = profile_dir.join(&wav_name);
                    write_wav_16k(&wav_path, &samples, sample_rate)?;

                    // Write manifest
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

                    last_duration = duration;
                    status_msg.clear();
                    state = State::Recorded;
                }

                // Skip
                (State::Waiting, KeyCode::Char('s') | KeyCode::Char('S')) => {
                    i += 1;
                    state = State::Waiting;
                    status_msg.clear();
                }

                // Next sentence
                (State::Recorded, KeyCode::Char(' ') | KeyCode::Enter) => {
                    i += 1;
                    state = State::Waiting;
                    status_msg.clear();
                }

                // Redo
                (State::Recorded, KeyCode::Char('r') | KeyCode::Char('R')) => {
                    state = State::Waiting;
                    status_msg.clear();
                }

                _ => {}
            }
        }
    }

    // Cleanup terminal
    drop(terminal);
    terminal::disable_raw_mode()?;
    crossterm::execute!(std::io::stdout(), crossterm::terminal::LeaveAlternateScreen)?;

    println!("Done! Recordings in {}", profile_dir.display());
    println!("Manifest: {}", manifest_path.display());
    Ok(())
}

/// Write samples to a 16kHz mono WAV, resampling if needed.
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
