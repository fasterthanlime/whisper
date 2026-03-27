use anyhow::{Context, Result};
use clap::Parser;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::terminal;
use ratatui::prelude::*;
use ratatui::widgets::*;
use std::io::Write;

#[derive(Parser)]
#[command(about = "Review and correct ASR-tainted dictation messages")]
struct Args {
    /// Path to review pairs JSONL
    #[arg(short, long, default_value = "data/review_pairs.jsonl")]
    input: String,

    /// Output path for reviewed pairs
    #[arg(short, long, default_value = "data/reviewed_pairs.jsonl")]
    output: String,

    /// Start from item N (for resuming)
    #[arg(long, default_value = "0")]
    start: usize,
}

#[derive(serde::Deserialize, serde::Serialize, Clone)]
struct ReviewPair {
    id: usize,
    project: String,
    original: String,
    corrected: String,
    status: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let mut pairs: Vec<ReviewPair> = std::fs::read_to_string(&args.input)
        .context("reading input")?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).context("parsing pair"))
        .collect::<Result<Vec<_>>>()?;

    if args.start >= pairs.len() {
        println!("Nothing to review");
        return Ok(());
    }

    // Load existing reviews to preserve progress
    let mut reviewed: Vec<ReviewPair> = if std::path::Path::new(&args.output).exists() {
        std::fs::read_to_string(&args.output)?
            .lines()
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect()
    } else {
        Vec::new()
    };

    terminal::enable_raw_mode()?;
    let mut terminal = Terminal::new(CrosstermBackend::new(std::io::stdout()))?;
    terminal.clear()?;

    let mut i = args.start;
    let mut editing = false;
    let mut edit_buf = String::new();
    let mut cursor_pos = 0usize;
    let mut stats = (0usize, 0usize, 0usize); // accepted, rejected, edited

    // Count existing stats
    for r in &reviewed {
        match r.status.as_str() {
            "accepted" => stats.0 += 1,
            "rejected" => stats.1 += 1,
            "edited" => stats.2 += 1,
            _ => {}
        }
    }

    loop {
        if i >= pairs.len() {
            break;
        }
        let pair = &pairs[i];

        terminal.draw(|frame| {
            let area = frame.area();

            let chunks = Layout::vertical([
                Constraint::Length(3), // header
                Constraint::Length(5), // original
                Constraint::Min(5),    // corrected / edit
                Constraint::Length(3), // controls
            ])
            .split(area);

            // Header
            let header = Paragraph::new(Line::from(vec![
                Span::styled(
                    format!(" {} ", pair.project),
                    Style::default().fg(Color::Black).bg(Color::Cyan).bold(),
                ),
                Span::raw("  "),
                Span::styled(
                    format!("✓{} ✗{} ✎{}", stats.0, stats.1, stats.2),
                    Style::default().fg(Color::DarkGray),
                ),
            ]))
            .block(Block::bordered().title(format!(
                " Review {}/{} ",
                i + 1,
                pairs.len()
            )));
            frame.render_widget(header, chunks[0]);

            // Original (ASR output)
            let orig = Paragraph::new(pair.original.as_str())
                .block(
                    Block::bordered()
                        .title(" Original (ASR) ")
                        .border_style(Style::default().fg(Color::DarkGray)),
                )
                .wrap(Wrap { trim: false });
            frame.render_widget(orig, chunks[1]);

            // Corrected / Edit
            if editing {
                let edit_text = format!("{}_", &edit_buf[..cursor_pos]);
                let edit_para = Paragraph::new(edit_buf.as_str())
                    .block(
                        Block::bordered()
                            .title(" Editing (Enter=save, Esc=cancel) ")
                            .border_style(Style::default().fg(Color::Yellow)),
                    )
                    .wrap(Wrap { trim: false });
                frame.render_widget(edit_para, chunks[2]);
            } else {
                // Highlight differences
                let corr = Paragraph::new(pair.corrected.as_str())
                    .block(
                        Block::bordered()
                            .title(" Corrected ")
                            .border_style(Style::default().fg(Color::Green)),
                    )
                    .wrap(Wrap { trim: false });
                frame.render_widget(corr, chunks[2]);
            }

            // Controls
            let controls = if editing {
                " Type to edit │ Enter: save │ Esc: cancel "
            } else if pair.original == pair.corrected {
                " Y: accept (no change needed) │ E: edit │ N: reject │ S: skip │ Q: quit "
            } else {
                " Y: accept correction │ E: edit │ N: reject │ S: skip │ Q: quit "
            };
            let ctrl = Paragraph::new(Line::from(Span::styled(
                controls,
                Style::default().fg(Color::Cyan),
            )))
            .block(Block::bordered());
            frame.render_widget(ctrl, chunks[3]);
        })?;

        if !event::poll(std::time::Duration::from_secs(60))? {
            continue;
        }

        if let Event::Key(KeyEvent {
            code, modifiers, ..
        }) = event::read()?
        {
            if editing {
                match code {
                    KeyCode::Enter => {
                        pairs[i].corrected = edit_buf.clone();
                        pairs[i].status = "edited".to_string();
                        reviewed.push(pairs[i].clone());
                        stats.2 += 1;
                        editing = false;
                        i += 1;
                        save_reviewed(&args.output, &reviewed)?;
                    }
                    KeyCode::Esc => {
                        editing = false;
                    }
                    KeyCode::Backspace => {
                        if cursor_pos > 0 {
                            edit_buf.remove(cursor_pos - 1);
                            cursor_pos -= 1;
                        }
                    }
                    KeyCode::Left => {
                        cursor_pos = cursor_pos.saturating_sub(1);
                    }
                    KeyCode::Right => {
                        cursor_pos = (cursor_pos + 1).min(edit_buf.len());
                    }
                    KeyCode::Char(c) => {
                        edit_buf.insert(cursor_pos, c);
                        cursor_pos += 1;
                    }
                    _ => {}
                }
            } else {
                match code {
                    KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc => break,
                    KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => break,

                    KeyCode::Char('y') | KeyCode::Char('Y') => {
                        pairs[i].status = "accepted".to_string();
                        reviewed.push(pairs[i].clone());
                        stats.0 += 1;
                        i += 1;
                        save_reviewed(&args.output, &reviewed)?;
                    }
                    KeyCode::Char('n') | KeyCode::Char('N') => {
                        pairs[i].status = "rejected".to_string();
                        reviewed.push(pairs[i].clone());
                        stats.1 += 1;
                        i += 1;
                        save_reviewed(&args.output, &reviewed)?;
                    }
                    KeyCode::Char('e') | KeyCode::Char('E') => {
                        edit_buf = pairs[i].corrected.clone();
                        cursor_pos = edit_buf.len();
                        editing = true;
                    }
                    KeyCode::Char('s') | KeyCode::Char('S') => {
                        i += 1;
                    }
                    _ => {}
                }
            }
        }
    }

    drop(terminal);
    terminal::disable_raw_mode()?;

    println!("\nDone! ✓{} ✗{} ✎{}", stats.0, stats.1, stats.2);
    println!("Saved to {}", args.output);
    Ok(())
}

fn save_reviewed(path: &str, reviewed: &[ReviewPair]) -> Result<()> {
    let mut f = std::fs::File::create(path)?;
    for r in reviewed {
        serde_json::to_writer(&mut f, r)?;
        f.write_all(b"\n")?;
    }
    f.flush()?;
    Ok(())
}
