//! Deployments screen - manage application deployments

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Paragraph, Row, Table, TableState, Wrap},
    Frame,
};

use crate::app::{App, RenderContext};
use crate::ui::components::{footer, header};

/// Render the deployments screen
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
    if app.screens.deployments.show_logs {
        let split = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(content);

        render_deployments_table(frame, app, split[0]);
        render_deployment_logs(frame, app, split[1]);
    } else {
        render_deployments_table(frame, app, content);
    }
}

fn render_deployments_table(frame: &mut Frame, app: &App, area: Rect) {
    let theme = &app.theme;

    // Sample deployments
    let deployments = vec![
        ("qwen-chat", "vllm", "Running", "1/1", "H100 x 1", "https://qwen-chat.basilica.ai"),
        ("llama-api", "sglang", "Running", "2/2", "A100 x 2", "https://llama-api.basilica.ai"),
        ("whisper-svc", "custom", "Scaling", "1/3", "L40S x 1", "https://whisper.basilica.ai"),
        ("train-job", "pytorch", "Running", "1/1", "H100 x 4", "N/A (job)"),
    ];

    let rows: Vec<Row> = deployments
        .iter()
        .enumerate()
        .map(|(i, (name, type_, status, replicas, gpu, url))| {
            let status_style = match *status {
                "Running" => theme.status_running(),
                "Scaling" => theme.status_pending(),
                "Failed" => theme.status_error(),
                _ => theme.text(),
            };

            let row_style = if i == app.selected_index {
                theme.selected_row()
            } else {
                theme.text()
            };

            Row::new(vec![
                Cell::from(*name).style(theme.text_accent()),
                Cell::from(*type_),
                Cell::from(format!("● {}", status)).style(status_style),
                Cell::from(*replicas),
                Cell::from(*gpu),
                Cell::from(*url).style(theme.text_muted()),
            ])
            .style(row_style)
        })
        .collect();

    let header = Row::new(vec!["Name", "Type", "Status", "Replicas", "GPU", "URL"])
        .style(theme.header());

    let table = Table::new(
        rows,
        [
            Constraint::Length(15),
            Constraint::Length(10),
            Constraint::Length(12),
            Constraint::Length(10),
            Constraint::Length(12),
            Constraint::Min(20),
        ],
    )
    .header(header)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(theme.border_selected())
            .title(Span::styled(" Deployments ", theme.block_title())),
    )
    .row_highlight_style(theme.selected_row())
    .highlight_symbol("▶ ");

    let mut state = TableState::default();
    state.select(Some(app.selected_index));

    frame.render_stateful_widget(table, area, &mut state);
}

fn render_deployment_logs(frame: &mut Frame, app: &App, area: Rect) {
    let theme = &app.theme;

    let logs = r#"2024-01-15 14:00:00 [INFO] Deployment qwen-chat created
2024-01-15 14:00:05 [INFO] Pulling image vllm/vllm-openai:v0.4.0
2024-01-15 14:00:45 [INFO] Image pulled successfully
2024-01-15 14:00:46 [INFO] Creating pod qwen-chat-abc123
2024-01-15 14:00:48 [INFO] Pod scheduled on node gpu-h100-01
2024-01-15 14:01:00 [INFO] Container starting...
2024-01-15 14:01:15 [INFO] vLLM server starting with model Qwen/Qwen2.5-7B
2024-01-15 14:02:30 [INFO] Model loaded, warming up...
2024-01-15 14:02:45 [INFO] Health check passed
2024-01-15 14:02:45 [INFO] Deployment ready at https://qwen-chat.basilica.ai"#;

    let paragraph = Paragraph::new(logs)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border())
                .title(Span::styled(" Deployment Logs ", theme.block_title())),
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

    if ctx.screens.deployments.show_logs {
        let split = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(content);

        render_deployments_table_ctx(frame, ctx, split[0]);
        render_deployment_logs_ctx(frame, theme, split[1]);
    } else {
        render_deployments_table_ctx(frame, ctx, content);
    }
}

fn render_deployments_table_ctx(frame: &mut Frame, ctx: &RenderContext, area: Rect) {
    let theme = ctx.theme;

    let deployments = vec![
        ("qwen-chat", "vllm", "Running", "1/1", "H100 x 1", "https://qwen-chat.basilica.ai"),
        ("llama-api", "sglang", "Running", "2/2", "A100 x 2", "https://llama-api.basilica.ai"),
        ("whisper-svc", "custom", "Scaling", "1/3", "L40S x 1", "https://whisper.basilica.ai"),
        ("train-job", "pytorch", "Running", "1/1", "H100 x 4", "N/A (job)"),
    ];

    let rows: Vec<Row> = deployments
        .iter()
        .enumerate()
        .map(|(i, (name, type_, status, replicas, gpu, url))| {
            let status_style = match *status {
                "Running" => theme.status_running(),
                "Scaling" => theme.status_pending(),
                "Failed" => theme.status_error(),
                _ => theme.text(),
            };

            let row_style = if i == ctx.selected_index {
                theme.selected_row()
            } else {
                theme.text()
            };

            Row::new(vec![
                Cell::from(*name).style(theme.text_accent()),
                Cell::from(*type_),
                Cell::from(format!("● {}", status)).style(status_style),
                Cell::from(*replicas),
                Cell::from(*gpu),
                Cell::from(*url).style(theme.text_muted()),
            ])
            .style(row_style)
        })
        .collect();

    let header_row = Row::new(vec!["Name", "Type", "Status", "Replicas", "GPU", "URL"])
        .style(theme.header());

    let table = Table::new(
        rows,
        [
            Constraint::Length(15),
            Constraint::Length(10),
            Constraint::Length(12),
            Constraint::Length(10),
            Constraint::Length(12),
            Constraint::Min(20),
        ],
    )
    .header(header_row)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(theme.border_selected())
            .title(Span::styled(" Deployments ", theme.block_title())),
    )
    .row_highlight_style(theme.selected_row())
    .highlight_symbol("▶ ");

    let mut state = TableState::default();
    state.select(Some(ctx.selected_index));

    frame.render_stateful_widget(table, area, &mut state);
}

fn render_deployment_logs_ctx(frame: &mut Frame, theme: &crate::ui::Theme, area: Rect) {
    let logs = r#"2024-01-15 14:00:00 [INFO] Deployment qwen-chat created
2024-01-15 14:00:05 [INFO] Pulling image vllm/vllm-openai:v0.4.0
2024-01-15 14:00:45 [INFO] Image pulled successfully
2024-01-15 14:00:46 [INFO] Creating pod qwen-chat-abc123
2024-01-15 14:00:48 [INFO] Pod scheduled on node gpu-h100-01
2024-01-15 14:01:00 [INFO] Container starting...
2024-01-15 14:01:15 [INFO] vLLM server starting with model Qwen/Qwen2.5-7B
2024-01-15 14:02:30 [INFO] Model loaded, warming up...
2024-01-15 14:02:45 [INFO] Health check passed
2024-01-15 14:02:45 [INFO] Deployment ready at https://qwen-chat.basilica.ai"#;

    let paragraph = Paragraph::new(logs)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border())
                .title(Span::styled(" Deployment Logs ", theme.block_title())),
        )
        .style(theme.text())
        .wrap(Wrap { trim: false });

    frame.render_widget(paragraph, area);
}

