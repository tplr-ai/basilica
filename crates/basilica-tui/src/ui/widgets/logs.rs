//! Log viewer widget

use ratatui::{
    layout::Rect,
    style::Style,
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap},
    Frame,
};

use crate::ui::Theme;

/// A log entry with level and message
#[derive(Debug, Clone)]
pub struct LogEntry {
    pub timestamp: String,
    pub level: LogLevel,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Debug,
    Info,
    Warn,
    Error,
}

impl LogLevel {
    pub fn style(&self, theme: &Theme) -> Style {
        match self {
            LogLevel::Debug => theme.text_muted(),
            LogLevel::Info => theme.text(),
            LogLevel::Warn => theme.text_warning(),
            LogLevel::Error => theme.text_error(),
        }
    }

    pub fn prefix(&self) -> &'static str {
        match self {
            LogLevel::Debug => "DBG",
            LogLevel::Info => "INF",
            LogLevel::Warn => "WRN",
            LogLevel::Error => "ERR",
        }
    }
}

/// Render a log viewer
pub fn render_logs(
    frame: &mut Frame,
    area: Rect,
    title: &str,
    logs: &[LogEntry],
    theme: &Theme,
    auto_scroll: bool,
) {
    let items: Vec<ListItem> = logs
        .iter()
        .map(|log| {
            let line = Line::from(vec![
                Span::styled(format!("{} ", log.timestamp), theme.text_muted()),
                Span::styled(format!("[{}] ", log.level.prefix()), log.level.style(theme)),
                Span::styled(&log.message, theme.text()),
            ]);
            ListItem::new(line)
        })
        .collect();

    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border())
                .title(title)
                .title_style(theme.block_title()),
        )
        .style(theme.text());

    let mut state = ListState::default();
    if auto_scroll && !logs.is_empty() {
        state.select(Some(logs.len() - 1));
    }

    frame.render_stateful_widget(list, area, &mut state);
}

/// Render raw log text (for streaming)
pub fn render_raw_logs(frame: &mut Frame, area: Rect, title: &str, content: &str, theme: &Theme) {
    let paragraph = Paragraph::new(content)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border())
                .title(title)
                .title_style(theme.block_title()),
        )
        .style(theme.text())
        .wrap(Wrap { trim: false });

    frame.render_widget(paragraph, area);
}
