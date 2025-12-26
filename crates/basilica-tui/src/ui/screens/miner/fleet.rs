//! Fleet dashboard - overview of all nodes and GPUs

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::Modifier,
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Paragraph, Row, Table, TableState},
    Frame,
};

use crate::app::{App, RenderContext};
use crate::ui::components::{footer, header};
use crate::ui::widgets::gauge::mini_bar;

/// Render the fleet dashboard
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

    // Split into stats bar and node grid
    let main_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5), // Stats bar
            Constraint::Min(0),    // Node grid
        ])
        .split(content);

    render_stats_bar(frame, app, main_chunks[0]);
    render_node_grid(frame, app, main_chunks[1]);
}

fn render_stats_bar(frame: &mut Frame, app: &App, area: Rect) {
    let theme = &app.theme;

    // Split into stat boxes
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(20),
            Constraint::Percentage(20),
            Constraint::Percentage(20),
            Constraint::Percentage(20),
            Constraint::Percentage(20),
        ])
        .split(area);

    // Total Nodes
    render_stat_box(
        frame,
        app,
        chunks[0],
        "Nodes",
        "8",
        Some("4 healthy"),
        theme.text_success(),
    );

    // Total GPUs
    render_stat_box(
        frame,
        app,
        chunks[1],
        "GPUs",
        "24",
        Some("20 active"),
        theme.text_accent(),
    );

    // Current Earnings
    render_stat_box(
        frame,
        app,
        chunks[2],
        "Earning",
        "$12.50/hr",
        None,
        theme.text_success(),
    );

    // Today's Revenue
    render_stat_box(
        frame,
        app,
        chunks[3],
        "Today",
        "$145.20",
        None,
        theme.text_accent(),
    );

    // Validator Status
    render_stat_box(
        frame,
        app,
        chunks[4],
        "Validators",
        "3",
        Some("assigned"),
        theme.text_info(),
    );
}

fn render_stat_box(
    frame: &mut Frame,
    app: &App,
    area: Rect,
    title: &str,
    value: &str,
    subtitle: Option<&str>,
    value_style: ratatui::style::Style,
) {
    let theme = &app.theme;

    let mut lines = vec![
        Line::from(""),
        Line::from(vec![Span::styled(
            format!(" {}", value),
            value_style.add_modifier(Modifier::BOLD),
        )]),
    ];

    if let Some(sub) = subtitle {
        lines.push(Line::from(vec![Span::styled(
            format!(" {}", sub),
            theme.text_muted(),
        )]));
    }

    let paragraph = Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border())
                .title(Span::styled(format!(" {} ", title), theme.block_title())),
        )
        .style(theme.text());

    frame.render_widget(paragraph, area);
}

fn render_node_grid(frame: &mut Frame, app: &App, area: Rect) {
    let theme = &app.theme;

    // Sample node data
    let nodes = [(
            "node-001",
            "192.168.1.10",
            "H100 x 4",
            "Healthy",
            85.0,
            72.0,
            "3/4 assigned",
        ),
        (
            "node-002",
            "192.168.1.11",
            "H100 x 4",
            "Healthy",
            92.0,
            88.0,
            "4/4 assigned",
        ),
        (
            "node-003",
            "192.168.1.12",
            "A100 x 8",
            "Healthy",
            45.0,
            52.0,
            "4/8 assigned",
        ),
        (
            "node-004",
            "192.168.1.13",
            "A100 x 8",
            "Warning",
            78.0,
            95.0,
            "6/8 assigned",
        ),
        (
            "node-005",
            "192.168.1.14",
            "RTX 4090 x 4",
            "Healthy",
            30.0,
            25.0,
            "1/4 assigned",
        ),
        (
            "node-006",
            "192.168.1.15",
            "RTX 4090 x 4",
            "Offline",
            0.0,
            0.0,
            "0/4 assigned",
        ),
        (
            "node-007",
            "192.168.1.16",
            "L40S x 2",
            "Healthy",
            60.0,
            55.0,
            "2/2 assigned",
        ),
        (
            "node-008",
            "192.168.1.17",
            "L40S x 2",
            "Healthy",
            40.0,
            35.0,
            "1/2 assigned",
        )];

    let rows: Vec<Row> = nodes
        .iter()
        .enumerate()
        .map(
            |(i, (name, ip, gpu, status, gpu_util, mem_util, assigned))| {
                let status_style = match *status {
                    "Healthy" => theme.status_running(),
                    "Warning" => theme.status_pending(),
                    "Offline" => theme.status_error(),
                    _ => theme.text(),
                };

                let gpu_bar = mini_bar(*gpu_util, 8);
                let mem_bar = mini_bar(*mem_util, 8);

                let row_style = if i == app.selected_index {
                    theme.selected_row()
                } else {
                    theme.text()
                };

                Row::new(vec![
                    Cell::from(*name).style(theme.text_accent()),
                    Cell::from(*ip),
                    Cell::from(*gpu),
                    Cell::from(format!("● {}", status)).style(status_style),
                    Cell::from(format!("{} {:>3.0}%", gpu_bar, gpu_util)),
                    Cell::from(format!("{} {:>3.0}%", mem_bar, mem_util)),
                    Cell::from(*assigned),
                ])
                .style(row_style)
            },
        )
        .collect();

    let header = Row::new(vec![
        "Node", "IP", "GPUs", "Status", "GPU Util", "Memory", "Assigned",
    ])
    .style(theme.header());

    let table = Table::new(
        rows,
        [
            Constraint::Length(12),
            Constraint::Length(14),
            Constraint::Length(14),
            Constraint::Length(12),
            Constraint::Length(14),
            Constraint::Length(14),
            Constraint::Min(12),
        ],
    )
    .header(header)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(theme.border_selected())
            .title(Span::styled(" Node Fleet ", theme.block_title())),
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
    let main_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(5), Constraint::Min(0)])
        .split(content);

    render_stats_bar_ctx(frame, theme, main_chunks[0]);
    render_node_grid_ctx(frame, ctx, main_chunks[1]);
}

fn render_stats_bar_ctx(frame: &mut Frame, theme: &crate::ui::Theme, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(20),
            Constraint::Percentage(20),
            Constraint::Percentage(20),
            Constraint::Percentage(20),
            Constraint::Percentage(20),
        ])
        .split(area);

    render_stat_box_ctx(
        frame,
        theme,
        chunks[0],
        "Nodes",
        "8",
        Some("4 healthy"),
        theme.text_success(),
    );
    render_stat_box_ctx(
        frame,
        theme,
        chunks[1],
        "GPUs",
        "24",
        Some("20 active"),
        theme.text_accent(),
    );
    render_stat_box_ctx(
        frame,
        theme,
        chunks[2],
        "Earning",
        "$12.50/hr",
        None,
        theme.text_success(),
    );
    render_stat_box_ctx(
        frame,
        theme,
        chunks[3],
        "Today",
        "$145.20",
        None,
        theme.text_accent(),
    );
    render_stat_box_ctx(
        frame,
        theme,
        chunks[4],
        "Validators",
        "3",
        Some("assigned"),
        theme.text_info(),
    );
}

fn render_stat_box_ctx(
    frame: &mut Frame,
    theme: &crate::ui::Theme,
    area: Rect,
    title: &str,
    value: &str,
    subtitle: Option<&str>,
    value_style: ratatui::style::Style,
) {
    let mut lines = vec![
        Line::from(""),
        Line::from(vec![Span::styled(
            format!(" {}", value),
            value_style.add_modifier(Modifier::BOLD),
        )]),
    ];

    if let Some(sub) = subtitle {
        lines.push(Line::from(vec![Span::styled(
            format!(" {}", sub),
            theme.text_muted(),
        )]));
    }

    let paragraph = Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border())
                .title(Span::styled(format!(" {} ", title), theme.block_title())),
        )
        .style(theme.text());

    frame.render_widget(paragraph, area);
}

fn render_node_grid_ctx(frame: &mut Frame, ctx: &RenderContext, area: Rect) {
    let theme = ctx.theme;

    let nodes = [(
            "node-001",
            "192.168.1.10",
            "H100 x 4",
            "Healthy",
            85.0,
            72.0,
            "3/4 assigned",
        ),
        (
            "node-002",
            "192.168.1.11",
            "H100 x 4",
            "Healthy",
            92.0,
            88.0,
            "4/4 assigned",
        ),
        (
            "node-003",
            "192.168.1.12",
            "A100 x 8",
            "Healthy",
            45.0,
            52.0,
            "4/8 assigned",
        ),
        (
            "node-004",
            "192.168.1.13",
            "A100 x 8",
            "Warning",
            78.0,
            95.0,
            "6/8 assigned",
        ),
        (
            "node-005",
            "192.168.1.14",
            "RTX 4090 x 4",
            "Healthy",
            30.0,
            25.0,
            "1/4 assigned",
        ),
        (
            "node-006",
            "192.168.1.15",
            "RTX 4090 x 4",
            "Offline",
            0.0,
            0.0,
            "0/4 assigned",
        ),
        (
            "node-007",
            "192.168.1.16",
            "L40S x 2",
            "Healthy",
            60.0,
            55.0,
            "2/2 assigned",
        ),
        (
            "node-008",
            "192.168.1.17",
            "L40S x 2",
            "Healthy",
            40.0,
            35.0,
            "1/2 assigned",
        )];

    let rows: Vec<Row> = nodes
        .iter()
        .enumerate()
        .map(
            |(i, (name, ip, gpu, status, gpu_util, mem_util, assigned))| {
                let status_style = match *status {
                    "Healthy" => theme.status_running(),
                    "Warning" => theme.status_pending(),
                    "Offline" => theme.status_error(),
                    _ => theme.text(),
                };

                let gpu_bar = mini_bar(*gpu_util, 8);
                let mem_bar = mini_bar(*mem_util, 8);

                let row_style = if i == ctx.selected_index {
                    theme.selected_row()
                } else {
                    theme.text()
                };

                Row::new(vec![
                    Cell::from(*name).style(theme.text_accent()),
                    Cell::from(*ip),
                    Cell::from(*gpu),
                    Cell::from(format!("● {}", status)).style(status_style),
                    Cell::from(format!("{} {:>3.0}%", gpu_bar, gpu_util)),
                    Cell::from(format!("{} {:>3.0}%", mem_bar, mem_util)),
                    Cell::from(*assigned),
                ])
                .style(row_style)
            },
        )
        .collect();

    let header_row = Row::new(vec![
        "Node", "IP", "GPUs", "Status", "GPU Util", "Memory", "Assigned",
    ])
    .style(theme.header());

    let table = Table::new(
        rows,
        [
            Constraint::Length(12),
            Constraint::Length(14),
            Constraint::Length(14),
            Constraint::Length(12),
            Constraint::Length(14),
            Constraint::Length(14),
            Constraint::Min(12),
        ],
    )
    .header(header_row)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(theme.border_selected())
            .title(Span::styled(" Node Fleet ", theme.block_title())),
    )
    .row_highlight_style(theme.selected_row())
    .highlight_symbol("▶ ");

    let mut state = TableState::default();
    state.select(Some(ctx.selected_index));

    frame.render_stateful_widget(table, area, &mut state);
}
