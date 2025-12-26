//! Marketplace screen - browse and rent GPUs

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Paragraph, Row, Table, TableState},
    Frame,
};
use crate::ui::Theme;

use crate::app::{App, RenderContext};
use crate::ui::components::{footer, header};

/// Render the marketplace screen
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

    // Split into filters and GPU list
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(25), Constraint::Min(0)])
        .split(content);

    render_filters(frame, app, columns[0]);
    render_gpu_list(frame, app, columns[1]);
}

fn render_filters(frame: &mut Frame, app: &App, area: Rect) {
    let theme = &app.theme;

    let content = vec![
        Line::from(Span::styled("GPU Type", theme.text_bold())),
        Line::from(""),
        Line::from(vec![
            Span::raw("  "),
            Span::styled("●", theme.text_accent()),
            Span::raw(" All"),
        ]),
        Line::from("    ○ H100"),
        Line::from("    ○ H200"),
        Line::from("    ○ A100"),
        Line::from("    ○ RTX 4090"),
        Line::from("    ○ L40S"),
        Line::from(""),
        Line::from(Span::styled("GPU Count", theme.text_bold())),
        Line::from(""),
        Line::from("    ○ 1x"),
        Line::from("    ○ 2x"),
        Line::from("    ○ 4x"),
        Line::from("    ○ 8x"),
        Line::from(""),
        Line::from(Span::styled("Source", theme.text_bold())),
        Line::from(""),
        Line::from(vec![
            Span::raw("  "),
            Span::styled("●", theme.text_accent()),
            Span::raw(" All"),
        ]),
        Line::from("    ○ Secure Cloud"),
        Line::from("    ○ Community"),
    ];

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

fn render_gpu_list(frame: &mut Frame, app: &App, area: Rect) {
    let theme = &app.theme;

    // Sample GPU offerings
    let gpus = vec![
        ("H100", "1x", "80GB", "$2.50/hr", "Secure", "4", "Training, inference"),
        ("H100", "4x", "320GB", "$9.00/hr", "Secure", "2", "Large model training"),
        ("H200", "1x", "141GB", "$3.50/hr", "Secure", "1", "Next-gen inference"),
        ("A100", "1x", "80GB", "$1.50/hr", "Community", "12", "General ML"),
        ("A100", "4x", "320GB", "$5.50/hr", "Community", "3", "Distributed training"),
        ("A100", "8x", "640GB", "$10.00/hr", "Secure", "1", "Large scale training"),
        ("RTX 4090", "1x", "24GB", "$0.45/hr", "Community", "25", "Development"),
        ("L40S", "1x", "48GB", "$0.90/hr", "Community", "8", "Inference"),
        ("RTX 4090", "2x", "48GB", "$0.85/hr", "Community", "5", "Multi-GPU dev"),
    ];

    let rows: Vec<Row> = gpus
        .iter()
        .enumerate()
        .map(|(i, (gpu, count, mem, price, source, avail, use_case))| {
            let source_style = match *source {
                "Secure" => theme.text_success(),
                "Community" => theme.text_info(),
                _ => theme.text(),
            };

            let row_style = if i == app.selected_index {
                theme.selected_row()
            } else {
                theme.text()
            };

            Row::new(vec![
                Cell::from(*gpu),
                Cell::from(*count),
                Cell::from(*mem),
                Cell::from(*price).style(theme.text_accent()),
                Cell::from(*source).style(source_style),
                Cell::from(*avail),
                Cell::from(*use_case).style(theme.text_muted()),
            ])
            .style(row_style)
        })
        .collect();

    let header = Row::new(vec!["GPU", "Count", "Memory", "Price", "Source", "Avail", "Best For"])
        .style(theme.header());

    let table = Table::new(
        rows,
        [
            Constraint::Length(10),
            Constraint::Length(6),
            Constraint::Length(8),
            Constraint::Length(10),
            Constraint::Length(10),
            Constraint::Length(6),
            Constraint::Min(15),
        ],
    )
    .header(header)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(theme.border_selected())
            .title(Span::styled(" Available GPUs ", theme.block_title())),
    )
    .row_highlight_style(theme.selected_row())
    .highlight_symbol("▶ ");

    let mut state = TableState::default();
    state.select(Some(app.selected_index));

    frame.render_stateful_widget(table, area, &mut state);
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
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(25), Constraint::Min(0)])
        .split(content);

    render_filters_ctx(frame, theme, columns[0]);
    render_gpu_list_ctx(frame, ctx, columns[1]);
}

fn render_filters_ctx(frame: &mut Frame, theme: &crate::ui::Theme, area: Rect) {
    let content = vec![
        Line::from(Span::styled("GPU Type", theme.text_bold())),
        Line::from(""),
        Line::from(vec![
            Span::raw("  "),
            Span::styled("●", theme.text_accent()),
            Span::raw(" All"),
        ]),
        Line::from("    ○ H100"),
        Line::from("    ○ H200"),
        Line::from("    ○ A100"),
        Line::from("    ○ RTX 4090"),
        Line::from("    ○ L40S"),
        Line::from(""),
        Line::from(Span::styled("GPU Count", theme.text_bold())),
        Line::from(""),
        Line::from("    ○ 1x"),
        Line::from("    ○ 2x"),
        Line::from("    ○ 4x"),
        Line::from("    ○ 8x"),
        Line::from(""),
        Line::from(Span::styled("Source", theme.text_bold())),
        Line::from(""),
        Line::from(vec![
            Span::raw("  "),
            Span::styled("●", theme.text_accent()),
            Span::raw(" All"),
        ]),
        Line::from("    ○ Secure Cloud"),
        Line::from("    ○ Community"),
    ];

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

fn render_gpu_list_ctx(frame: &mut Frame, ctx: &RenderContext, area: Rect) {
    let theme = ctx.theme;

    let gpus = vec![
        ("H100", "1x", "80GB", "$2.50/hr", "Secure", "4", "Training, inference"),
        ("H100", "4x", "320GB", "$9.00/hr", "Secure", "2", "Large model training"),
        ("H200", "1x", "141GB", "$3.50/hr", "Secure", "1", "Next-gen inference"),
        ("A100", "1x", "80GB", "$1.50/hr", "Community", "12", "General ML"),
        ("A100", "4x", "320GB", "$5.50/hr", "Community", "3", "Distributed training"),
        ("A100", "8x", "640GB", "$10.00/hr", "Secure", "1", "Large scale training"),
        ("RTX 4090", "1x", "24GB", "$0.45/hr", "Community", "25", "Development"),
        ("L40S", "1x", "48GB", "$0.90/hr", "Community", "8", "Inference"),
        ("RTX 4090", "2x", "48GB", "$0.85/hr", "Community", "5", "Multi-GPU dev"),
    ];

    let rows: Vec<Row> = gpus
        .iter()
        .enumerate()
        .map(|(i, (gpu, count, mem, price, source, avail, use_case))| {
            let source_style = match *source {
                "Secure" => theme.text_success(),
                "Community" => theme.text_info(),
                _ => theme.text(),
            };

            let row_style = if i == ctx.selected_index {
                theme.selected_row()
            } else {
                theme.text()
            };

            Row::new(vec![
                Cell::from(*gpu),
                Cell::from(*count),
                Cell::from(*mem),
                Cell::from(*price).style(theme.text_accent()),
                Cell::from(*source).style(source_style),
                Cell::from(*avail),
                Cell::from(*use_case).style(theme.text_muted()),
            ])
            .style(row_style)
        })
        .collect();

    let header_row = Row::new(vec!["GPU", "Count", "Memory", "Price", "Source", "Avail", "Best For"])
        .style(theme.header());

    let table = Table::new(
        rows,
        [
            Constraint::Length(10),
            Constraint::Length(6),
            Constraint::Length(8),
            Constraint::Length(10),
            Constraint::Length(10),
            Constraint::Length(6),
            Constraint::Min(15),
        ],
    )
    .header(header_row)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(theme.border_selected())
            .title(Span::styled(" Available GPUs ", theme.block_title())),
    )
    .row_highlight_style(theme.selected_row())
    .highlight_symbol("▶ ");

    let mut state = TableState::default();
    state.select(Some(ctx.selected_index));

    frame.render_stateful_widget(table, area, &mut state);
}

