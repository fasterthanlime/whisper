use std::collections::VecDeque;
use std::io;
use std::io::IsTerminal;

use crossterm::cursor::{Hide, Show};
use crossterm::execute;
use crossterm::terminal::{EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph, Wrap};

/// Full-screen TUI for live exercise progress (committed text, draft, event log).
pub(crate) struct ExerciseTui {
    /// Whether the TUI is currently active.
    enabled: bool,
    /// Terminal handle used for drawing when the TUI is enabled.
    terminal: Option<Terminal<CrosstermBackend<io::Stdout>>>,
    /// Current phase label shown in the header.
    phase: String,
    /// Current chunk index shown in the header.
    chunk_index: usize,
    /// Text that has been committed so far.
    committed: String,
    /// Text currently held as draft output.
    draft: String,
    /// Recent status messages shown in the event log.
    logs: VecDeque<String>,
}

impl ExerciseTui {
    /// Creates and enters the alternate screen TUI (no-op if stdout is not a terminal).
    pub(crate) fn new() -> Self {
        let enabled = std::io::stdout().is_terminal();
        let terminal = if enabled {
            let mut stdout = std::io::stdout();
            execute!(stdout, EnterAlternateScreen, Hide).ok();
            let backend = CrosstermBackend::new(stdout);
            Terminal::new(backend).ok()
        } else {
            None
        };
        let mut tui = Self {
            enabled,
            terminal,
            phase: "Starting".to_string(),
            chunk_index: 0,
            committed: String::new(),
            draft: String::new(),
            logs: VecDeque::new(),
        };
        tui.render();
        tui
    }

    /// Leaves the alternate screen and disables further rendering.
    pub(crate) fn clear(&mut self) {
        if !self.enabled {
            return;
        }
        if let Some(terminal) = self.terminal.as_mut() {
            let _ = terminal.clear();
        }
        let _ = self.terminal.take();
        let mut stdout = std::io::stdout();
        let _ = execute!(stdout, Show, LeaveAlternateScreen);
        self.enabled = false;
    }

    /// Appends a message to the event log and re-renders.
    pub(crate) fn log(&mut self, message: impl Into<String>) {
        if !self.enabled {
            return;
        }
        self.logs.push_back(message.into());
        while self.logs.len() > 8 {
            self.logs.pop_front();
        }
        self.render();
    }

    /// Updates all TUI fields and re-renders.
    fn update(&mut self, phase: &str, chunk_index: usize, committed: &str, draft: &str) {
        if !self.enabled {
            return;
        }
        self.phase.clear();
        self.phase.push_str(phase);
        self.chunk_index = chunk_index;
        self.committed.clear();
        self.committed.push_str(committed);
        self.draft.clear();
        self.draft.push_str(draft);
        self.render();
    }

    /// Draws the current state to the terminal.
    pub(crate) fn render(&mut self) {
        if !self.enabled {
            return;
        }
        let phase = self.phase.clone();
        let chunk_index = self.chunk_index;
        let committed = if self.committed.trim().is_empty() {
            "[empty]".to_string()
        } else {
            self.committed.clone()
        };
        let draft = if self.draft.trim().is_empty() {
            "[empty]".to_string()
        } else {
            self.draft.clone()
        };
        let logs = self.logs.iter().cloned().collect::<Vec<_>>();
        if let Some(terminal) = self.terminal.as_mut() {
            let _ = terminal.draw(|frame| {
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Length(3),
                        Constraint::Min(6),
                        Constraint::Min(6),
                        Constraint::Length(10),
                    ])
                    .split(frame.area());

                let header = Paragraph::new(Line::from(vec![
                    Span::styled(
                        "Bee Exercise",
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw("  "),
                    Span::styled(
                        phase.to_string(),
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw("  chunk "),
                    Span::styled(format!("{chunk_index}"), Style::default().fg(Color::Green)),
                ]))
                .block(Block::default().borders(Borders::ALL).title("Status"));

                let committed_panel = Paragraph::new(committed)
                    .style(Style::default().fg(Color::White))
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .title("Committed")
                            .border_style(Style::default().fg(Color::Green)),
                    )
                    .wrap(Wrap { trim: false });

                let draft_panel = Paragraph::new(draft)
                    .style(Style::default().fg(Color::Yellow))
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .title("Draft")
                            .border_style(Style::default().fg(Color::Yellow)),
                    )
                    .wrap(Wrap { trim: false });

                let log_items = if logs.is_empty() {
                    vec![ListItem::new(Line::from(Span::styled(
                        "No events yet",
                        Style::default().fg(Color::DarkGray),
                    )))]
                } else {
                    logs.into_iter()
                        .map(|entry| {
                            ListItem::new(Line::from(Span::styled(
                                entry,
                                Style::default().fg(Color::LightYellow),
                            )))
                        })
                        .collect()
                };
                let event_log = List::new(log_items).block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title("Event Log")
                        .border_style(Style::default().fg(Color::DarkGray)),
                );

                frame.render_widget(header, chunks[0]);
                frame.render_widget(committed_panel, chunks[1]);
                frame.render_widget(draft_panel, chunks[2]);
                frame.render_widget(event_log, chunks[3]);
            });
        }
    }
}

/// Updates the exercise TUI with new phase, chunk index, committed, and draft text.
pub(crate) fn update_exercise_progress(
    tui: &mut ExerciseTui,
    phase: &str,
    chunk_index: usize,
    committed: &str,
    draft: &str,
) {
    tui.update(phase, chunk_index, committed, draft);
}

/// Appends `text` to `target`, inserting a space if needed to separate words.
pub(crate) fn append_display_delta(target: &mut String, text: &str) {
    if text.is_empty() {
        return;
    }
    let needs_space = !target.is_empty()
        && !target.chars().last().is_some_and(char::is_whitespace)
        && !text.chars().next().is_some_and(char::is_whitespace);
    if needs_space {
        target.push(' ');
    }
    target.push_str(text);
}
