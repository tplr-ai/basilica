//! Miner logs screen - aggregated log viewer

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph},
    Frame,
};

use crate::app::{App, RenderContext};
use crate::ui::components::{footer, header};

/// Render the miner logs screen
pub fn render(frame: &mut Frame, app: &App) {
    let theme = &app.theme;
    let area = frame.area();

    // Layout: header, content, footer
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(header::header_height()),
            Constraint::Min(0),
            Constraint::Length(footer::footer_height()),
        ])
        .split(area);

    header::render_header(frame, app, chunks[0]);
    footer::render_footer(frame, app, chunks[2]);

    let content = chunks[1];

    // Split into filter bar and logs
    let log_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Filter bar
            Constraint::Min(0),    // Logs
        ])
        .split(content);

    render_filter_bar(frame, app, log_chunks[0]);
    render_logs(frame, app, log_chunks[1]);
}

fn render_filter_bar(frame: &mut Frame, app: &App, area: Rect) {
    let theme = &app.theme;

    let content = Line::from(vec![
        Span::styled(" Level: ", theme.text_muted()),
        Span::styled("[All]", theme.text_accent()),
        Span::raw(" INFO "),
        Span::raw(" WARN "),
        Span::raw(" ERROR "),
        Span::styled("  │  ", theme.text_muted()),
        Span::styled(" Source: ", theme.text_muted()),
        Span::styled("[All]", theme.text_accent()),
        Span::raw(" node-001 "),
        Span::raw(" node-002 "),
        Span::raw(" ... "),
        Span::styled("  │  ", theme.text_muted()),
        Span::styled(" [/] Search ", theme.text_muted()),
    ]);

    let paragraph = Paragraph::new(content)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border())
                .title(Span::styled(" Filters ", theme.block_title())),
        )
        .style(theme.text());

    frame.render_widget(paragraph, area);
}

fn render_logs(frame: &mut Frame, app: &App, area: Rect) {
    let theme = &app.theme;

    // Sample log entries
    let logs = vec![
        ("14:32:15", "INFO", "node-001", "Health check passed"),
        ("14:32:14", "INFO", "node-002", "Health check passed"),
        ("14:32:10", "INFO", "validator", "Validator discovery completed, found 3 validators"),
        ("14:32:05", "INFO", "node-003", "Health check passed"),
        ("14:32:00", "WARN", "node-004", "High memory utilization: 95%"),
        ("14:31:55", "INFO", "node-005", "Health check passed"),
        ("14:31:50", "ERROR", "node-006", "SSH connection failed: Connection refused"),
        ("14:31:45", "INFO", "node-007", "Health check passed"),
        ("14:31:40", "INFO", "node-008", "Health check passed"),
        ("14:31:30", "INFO", "validator", "Received assignment request from Val-001"),
        ("14:31:25", "INFO", "node-001", "GPU 0-3 assigned to Val-001"),
        ("14:31:20", "INFO", "metrics", "Metrics collection completed"),
        ("14:31:15", "INFO", "node-002", "Container started for rental abc123"),
        ("14:31:10", "INFO", "validator", "Node assignment successful"),
        ("14:31:05", "INFO", "bittensor", "Chain registration verified"),
        ("14:31:00", "INFO", "main", "Miner tick completed"),
        ("14:30:55", "WARN", "node-004", "GPU 2 temperature high: 82°C"),
        ("14:30:50", "INFO", "node-003", "Container logs rotated"),
        ("14:30:45", "INFO", "validator", "Heartbeat sent to Val-002"),
        ("14:30:40", "INFO", "metrics", "Prometheus scrape completed"),
    ];

    let items: Vec<ListItem> = logs
        .iter()
        .map(|(time, level, source, msg)| {
            let level_style = match *level {
                "INFO" => theme.text_info(),
                "WARN" => theme.text_warning(),
                "ERROR" => theme.text_error(),
                "DEBUG" => theme.text_muted(),
                _ => theme.text(),
            };

            let line = Line::from(vec![
                Span::styled(format!("{} ", time), theme.text_muted()),
                Span::styled(format!("[{:5}] ", level), level_style),
                Span::styled(format!("{:12} ", source), theme.text_accent()),
                Span::styled(*msg, theme.text()),
            ]);

            ListItem::new(line)
        })
        .collect();

    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border_selected())
                .title(Span::styled(" Logs (following) ", theme.block_title())),
        )
        .style(theme.text());

    // Auto-scroll to bottom
    let mut state = ListState::default();
    if !logs.is_empty() {
        state.select(Some(logs.len() - 1));
    }

    frame.render_stateful_widget(list, area, &mut state);
}

/// Render with context
pub fn render_with_ctx(frame: &mut Frame, ctx: &RenderContext) {
    let theme = ctx.theme;
    let area = frame.area();

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(header::header_height()),
            Constraint::Min(0),
            Constraint::Length(footer::footer_height()),
        ])
        .split(area);

    header::render_header_ctx(frame, ctx, chunks[0]);
    footer::render_footer_ctx(frame, ctx, chunks[2]);

    let content = chunks[1];
    let log_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(0)])
        .split(content);

    render_filter_bar_ctx(frame, theme, log_chunks[0]);
    render_logs_ctx(frame, theme, log_chunks[1]);
}

fn render_filter_bar_ctx(frame: &mut Frame, theme: &crate::ui::Theme, area: Rect) {
    let content = Line::from(vec![
        Span::styled(" Level: ", theme.text_muted()),
        Span::styled("[All]", theme.text_accent()),
        Span::raw(" INFO "),
        Span::raw(" WARN "),
        Span::raw(" ERROR "),
        Span::styled("  │  ", theme.text_muted()),
        Span::styled(" Source: ", theme.text_muted()),
        Span::styled("[All]", theme.text_accent()),
        Span::raw(" node-001 "),
        Span::raw(" node-002 "),
        Span::raw(" ... "),
        Span::styled("  │  ", theme.text_muted()),
        Span::styled(" [/] Search ", theme.text_muted()),
    ]);

    frame.render_widget(
        Paragraph::new(content)
            .block(Block::default().borders(Borders::ALL).border_style(theme.border()).title(Span::styled(" Filters ", theme.block_title())))
            .style(theme.text()),
        area,
    );
}

fn render_logs_ctx(frame: &mut Frame, theme: &crate::ui::Theme, area: Rect) {
    let logs = vec![
        ("14:32:15", "INFO", "node-001", "Health check passed"),
        ("14:32:14", "INFO", "node-002", "Health check passed"),
        ("14:32:10", "INFO", "validator", "Validator discovery completed, found 3 validators"),
        ("14:32:05", "INFO", "node-003", "Health check passed"),
        ("14:32:00", "WARN", "node-004", "High memory utilization: 95%"),
        ("14:31:55", "INFO", "node-005", "Health check passed"),
        ("14:31:50", "ERROR", "node-006", "SSH connection failed: Connection refused"),
        ("14:31:45", "INFO", "node-007", "Health check passed"),
        ("14:31:40", "INFO", "node-008", "Health check passed"),
        ("14:31:30", "INFO", "validator", "Received assignment request from Val-001"),
        ("14:31:25", "INFO", "node-001", "GPU 0-3 assigned to Val-001"),
        ("14:31:20", "INFO", "metrics", "Metrics collection completed"),
        ("14:31:15", "INFO", "node-002", "Container started for rental abc123"),
        ("14:31:10", "INFO", "validator", "Node assignment successful"),
        ("14:31:05", "INFO", "bittensor", "Chain registration verified"),
        ("14:31:00", "INFO", "main", "Miner tick completed"),
        ("14:30:55", "WARN", "node-004", "GPU 2 temperature high: 82°C"),
        ("14:30:50", "INFO", "node-003", "Container logs rotated"),
        ("14:30:45", "INFO", "validator", "Heartbeat sent to Val-002"),
        ("14:30:40", "INFO", "metrics", "Prometheus scrape completed"),
    ];

    let items: Vec<ListItem> = logs
        .iter()
        .map(|(time, level, source, msg)| {
            let level_style = match *level {
                "INFO" => theme.text_info(),
                "WARN" => theme.text_warning(),
                "ERROR" => theme.text_error(),
                "DEBUG" => theme.text_muted(),
                _ => theme.text(),
            };

            let line = Line::from(vec![
                Span::styled(format!("{} ", time), theme.text_muted()),
                Span::styled(format!("[{:5}] ", level), level_style),
                Span::styled(format!("{:12} ", source), theme.text_accent()),
                Span::styled(*msg, theme.text()),
            ]);

            ListItem::new(line)
        })
        .collect();

    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border_selected())
                .title(Span::styled(" Logs (following) ", theme.block_title())),
        )
        .style(theme.text());

    let mut state = ListState::default();
    if !logs.is_empty() {
        state.select(Some(logs.len() - 1));
    }

    frame.render_stateful_widget(list, area, &mut state);
}

