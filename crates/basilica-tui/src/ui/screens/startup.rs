//! Startup/welcome screen with mode selection

use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Paragraph},
    Frame,
};

use crate::ui::Theme;

/// Selected mode on startup screen
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StartupSelection {
    #[default]
    User,
    Miner,
}

impl StartupSelection {
    pub fn toggle(&mut self) {
        *self = match self {
            StartupSelection::User => StartupSelection::Miner,
            StartupSelection::Miner => StartupSelection::User,
        };
    }
}

/// Render the startup screen
pub fn render_startup(frame: &mut Frame, selection: StartupSelection, theme: &Theme) {
    let area = frame.area();

    // Background
    let bg = Block::default().style(Style::default().bg(theme.bg));
    frame.render_widget(bg, area);

    // Center vertically
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(40),
            Constraint::Length(7),
            Constraint::Percentage(40),
        ])
        .split(area);

    let content_area = vertical[1];

    // Content layout
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2), // Title
            Constraint::Length(1), // Spacing
            Constraint::Length(1), // Options
            Constraint::Length(1), // Spacing
            Constraint::Length(1), // Hints
        ])
        .split(content_area);

    // Title
    let title = Paragraph::new(vec![
        Line::from(Span::styled(
            "⛪  B A S I L I C A  ⛪",
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(
            "Sacred Compute",
            Style::default().fg(theme.fg_muted),
        )),
    ])
    .alignment(Alignment::Center);
    frame.render_widget(title, rows[0]);

    // Mode selection - simple inline
    let user_style = if selection == StartupSelection::User {
        Style::default()
            .fg(theme.accent)
            .add_modifier(Modifier::BOLD | Modifier::UNDERLINED)
    } else {
        Style::default().fg(theme.fg_muted)
    };

    let miner_style = if selection == StartupSelection::Miner {
        Style::default()
            .fg(theme.accent)
            .add_modifier(Modifier::BOLD | Modifier::UNDERLINED)
    } else {
        Style::default().fg(theme.fg_muted)
    };

    let options = Paragraph::new(Line::from(vec![
        Span::styled("[ ", Style::default().fg(theme.fg_muted)),
        Span::styled("👤 User", user_style),
        Span::styled(" ]          [ ", Style::default().fg(theme.fg_muted)),
        Span::styled("⛏  Miner", miner_style),
        Span::styled(" ]", Style::default().fg(theme.fg_muted)),
    ]))
    .alignment(Alignment::Center);
    frame.render_widget(options, rows[2]);

    // Hints
    let hints = Paragraph::new(Line::from(vec![
        Span::styled("Tab", Style::default().fg(theme.fg_muted)),
        Span::raw(" switch  "),
        Span::styled("Enter", Style::default().fg(theme.fg_muted)),
        Span::raw(" select  "),
        Span::styled("q", Style::default().fg(theme.fg_muted)),
        Span::raw(" quit"),
    ]))
    .alignment(Alignment::Center)
    .style(Style::default().fg(theme.border));
    frame.render_widget(hints, rows[4]);
}
