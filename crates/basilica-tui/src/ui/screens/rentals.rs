//! Rentals screen - list and manage active GPU rentals

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    text::Span,
    widgets::{Block, Borders, Cell, Paragraph, Row, Table, TableState, Wrap},
    Frame,
};

use crate::app::{App, RenderContext};
use crate::ui::components::{footer, header};

/// Render the rentals screen
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

    // If showing logs, split the view
    if app.screens.rentals.show_logs {
        let split = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(content);

        render_rentals_table(frame, app, split[0]);
        render_logs_panel(frame, app, split[1]);
    } else {
        render_rentals_table(frame, app, content);
    }
}

fn render_rentals_table(frame: &mut Frame, app: &App, area: Rect) {
    let theme = &app.theme;

    // Sample data - would come from app.user_data
    let rentals = vec![
        ("abc123", "H100 x 1", "Running", "2h 15m", "pytorch:latest", "$4.20"),
        ("def456", "A100 x 4", "Running", "45m", "vllm/vllm:v0.4", "$12.00"),
        ("ghi789", "H200 x 2", "Starting", "0m", "nvidia/cuda:12.0", "$0.00"),
        ("jkl012", "RTX 4090 x 1", "Running", "1h 30m", "ubuntu:22.04", "$1.50"),
    ];

    let rows: Vec<Row> = rentals
        .iter()
        .enumerate()
        .map(|(i, (id, gpu, status, uptime, image, cost))| {
            let status_style = match *status {
                "Running" => theme.status_running(),
                "Starting" => theme.status_pending(),
                "Stopped" => theme.status_stopped(),
                _ => theme.text(),
            };

            let row_style = if i == app.selected_index {
                theme.selected_row()
            } else {
                theme.text()
            };

            Row::new(vec![
                Cell::from(*id),
                Cell::from(*gpu),
                Cell::from(format!("● {}", status)).style(status_style),
                Cell::from(*uptime),
                Cell::from(*image),
                Cell::from(*cost),
            ])
            .style(row_style)
        })
        .collect();

    let header = Row::new(vec!["ID", "GPU", "Status", "Uptime", "Image", "Cost"])
        .style(theme.header());

    let table = Table::new(
        rows,
        [
            Constraint::Length(8),
            Constraint::Percentage(15),
            Constraint::Percentage(12),
            Constraint::Length(10),
            Constraint::Percentage(30),
            Constraint::Length(10),
        ],
    )
    .header(header)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(if app.screens.rentals.show_logs {
                theme.border()
            } else {
                theme.border_selected()
            })
            .title(Span::styled(" Rentals ", theme.block_title())),
    )
    .row_highlight_style(theme.selected_row())
    .highlight_symbol("▶ ");

    let mut state = TableState::default();
    state.select(Some(app.selected_index));

    frame.render_stateful_widget(table, area, &mut state);
}

fn render_logs_panel(frame: &mut Frame, app: &App, area: Rect) {
    let theme = &app.theme;

    // Sample log content
    let logs = r#"2024-01-15 14:32:01 [INFO] Container started successfully
2024-01-15 14:32:02 [INFO] GPU 0: NVIDIA H100 80GB HBM3 detected
2024-01-15 14:32:02 [INFO] CUDA 12.2 initialized
2024-01-15 14:32:03 [INFO] Loading model weights...
2024-01-15 14:32:15 [INFO] Model loaded successfully
2024-01-15 14:32:15 [INFO] Starting inference server on port 8000
2024-01-15 14:32:16 [INFO] Server ready, accepting connections"#;

    let title = if app.screens.rentals.log_follow {
        " Logs (following) [Press l to toggle] "
    } else {
        " Logs [Press l to toggle] "
    };

    let paragraph = Paragraph::new(logs)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border_selected())
                .title(Span::styled(title, theme.block_title())),
        )
        .style(theme.text())
        .wrap(Wrap { trim: false });

    frame.render_widget(paragraph, area);
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

    if ctx.screens.rentals.show_logs {
        let split = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(content);

        render_rentals_table_ctx(frame, ctx, split[0]);
        render_logs_panel_ctx(frame, ctx, split[1]);
    } else {
        render_rentals_table_ctx(frame, ctx, content);
    }
}

fn render_rentals_table_ctx(frame: &mut Frame, ctx: &RenderContext, area: Rect) {
    let theme = ctx.theme;

    let rentals = vec![
        ("abc123", "H100 x 1", "Running", "2h 15m", "pytorch:latest", "$4.20"),
        ("def456", "A100 x 4", "Running", "45m", "vllm/vllm:v0.4", "$12.00"),
        ("ghi789", "H200 x 2", "Starting", "0m", "nvidia/cuda:12.0", "$0.00"),
        ("jkl012", "RTX 4090 x 1", "Running", "1h 30m", "ubuntu:22.04", "$1.50"),
    ];

    let rows: Vec<Row> = rentals
        .iter()
        .enumerate()
        .map(|(i, (id, gpu, status, uptime, image, cost))| {
            let status_style = match *status {
                "Running" => theme.status_running(),
                "Starting" => theme.status_pending(),
                "Stopped" => theme.status_stopped(),
                _ => theme.text(),
            };

            let row_style = if i == ctx.selected_index {
                theme.selected_row()
            } else {
                theme.text()
            };

            Row::new(vec![
                Cell::from(*id),
                Cell::from(*gpu),
                Cell::from(format!("● {}", status)).style(status_style),
                Cell::from(*uptime),
                Cell::from(*image),
                Cell::from(*cost),
            ])
            .style(row_style)
        })
        .collect();

    let header_row = Row::new(vec!["ID", "GPU", "Status", "Uptime", "Image", "Cost"])
        .style(theme.header());

    let table = Table::new(
        rows,
        [
            Constraint::Length(8),
            Constraint::Percentage(15),
            Constraint::Percentage(12),
            Constraint::Length(10),
            Constraint::Percentage(30),
            Constraint::Length(10),
        ],
    )
    .header(header_row)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(if ctx.screens.rentals.show_logs {
                theme.border()
            } else {
                theme.border_selected()
            })
            .title(Span::styled(" Rentals ", theme.block_title())),
    )
    .row_highlight_style(theme.selected_row())
    .highlight_symbol("▶ ");

    let mut state = TableState::default();
    state.select(Some(ctx.selected_index));

    frame.render_stateful_widget(table, area, &mut state);
}

fn render_logs_panel_ctx(frame: &mut Frame, ctx: &RenderContext, area: Rect) {
    let theme = ctx.theme;

    let logs = r#"2024-01-15 14:32:01 [INFO] Container started successfully
2024-01-15 14:32:02 [INFO] GPU 0: NVIDIA H100 80GB HBM3 detected
2024-01-15 14:32:02 [INFO] CUDA 12.2 initialized
2024-01-15 14:32:03 [INFO] Loading model weights...
2024-01-15 14:32:15 [INFO] Model loaded successfully
2024-01-15 14:32:15 [INFO] Starting inference server on port 8000
2024-01-15 14:32:16 [INFO] Server ready, accepting connections"#;

    let title = if ctx.screens.rentals.log_follow {
        " Logs (following) [Press l to toggle] "
    } else {
        " Logs [Press l to toggle] "
    };

    let paragraph = Paragraph::new(logs)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border_selected())
                .title(Span::styled(title, theme.block_title())),
        )
        .style(theme.text())
        .wrap(Wrap { trim: false });

    frame.render_widget(paragraph, area);
}

