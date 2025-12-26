//! Nodes screen - manage individual nodes

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::Modifier,
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Paragraph, Row, Table, TableState},
    Frame,
};

use crate::app::{App, RenderContext};
use crate::ui::components::{footer, header};

/// Render the nodes management screen
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

    // Split into node list and details panel
    if app.screens.fleet.show_details {
        let columns = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
            .split(content);

        render_node_list(frame, app, columns[0]);
        render_node_details(frame, app, columns[1]);
    } else {
        render_node_list(frame, app, content);
    }
}

fn render_node_list(frame: &mut Frame, app: &App, area: Rect) {
    let theme = &app.theme;

    // Sample node configuration data
    let nodes = vec![
        ("node-001", "192.168.1.10", "22", "root", "H100 x 4", "Healthy"),
        ("node-002", "192.168.1.11", "22", "root", "H100 x 4", "Healthy"),
        ("node-003", "192.168.1.12", "22", "admin", "A100 x 8", "Healthy"),
        ("node-004", "192.168.1.13", "2222", "ubuntu", "A100 x 8", "Warning"),
        ("node-005", "192.168.1.14", "22", "root", "RTX 4090 x 4", "Healthy"),
        ("node-006", "192.168.1.15", "22", "root", "RTX 4090 x 4", "Offline"),
        ("node-007", "192.168.1.16", "22", "gpu-user", "L40S x 2", "Healthy"),
        ("node-008", "192.168.1.17", "22", "gpu-user", "L40S x 2", "Healthy"),
    ];

    let rows: Vec<Row> = nodes
        .iter()
        .enumerate()
        .map(|(i, (name, ip, port, user, gpu, status))| {
            let status_style = match *status {
                "Healthy" => theme.status_running(),
                "Warning" => theme.status_pending(),
                "Offline" => theme.status_error(),
                _ => theme.text(),
            };

            let row_style = if i == app.selected_index {
                theme.selected_row()
            } else {
                theme.text()
            };

            Row::new(vec![
                Cell::from(*name).style(theme.text_accent()),
                Cell::from(format!("{}@{}:{}", user, ip, port)),
                Cell::from(*gpu),
                Cell::from(format!("● {}", status)).style(status_style),
            ])
            .style(row_style)
        })
        .collect();

    let header = Row::new(vec!["Node ID", "SSH Connection", "GPUs", "Status"])
        .style(theme.header());

    let table = Table::new(
        rows,
        [
            Constraint::Length(12),
            Constraint::Length(28),
            Constraint::Length(16),
            Constraint::Min(12),
        ],
    )
    .header(header)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(theme.border_selected())
            .title(Span::styled(" Nodes ", theme.block_title())),
    )
    .row_highlight_style(theme.selected_row())
    .highlight_symbol("▶ ");

    let mut state = TableState::default();
    state.select(Some(app.selected_index));

    frame.render_stateful_widget(table, area, &mut state);
}

fn render_node_details(frame: &mut Frame, app: &App, area: Rect) {
    let theme = &app.theme;

    // Sample details for selected node
    let details = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("  Node ID: ", theme.text_muted()),
            Span::styled("node-001", theme.text_accent().add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(Span::styled("  Connection", theme.text_bold())),
        Line::from(vec![
            Span::styled("    Host: ", theme.text_muted()),
            Span::styled("192.168.1.10", theme.text()),
        ]),
        Line::from(vec![
            Span::styled("    Port: ", theme.text_muted()),
            Span::styled("22", theme.text()),
        ]),
        Line::from(vec![
            Span::styled("    User: ", theme.text_muted()),
            Span::styled("root", theme.text()),
        ]),
        Line::from(""),
        Line::from(Span::styled("  Hardware", theme.text_bold())),
        Line::from(vec![
            Span::styled("    GPUs: ", theme.text_muted()),
            Span::styled("4x NVIDIA H100 80GB", theme.text()),
        ]),
        Line::from(vec![
            Span::styled("    CPU: ", theme.text_muted()),
            Span::styled("AMD EPYC 7763 (128 cores)", theme.text()),
        ]),
        Line::from(vec![
            Span::styled("    RAM: ", theme.text_muted()),
            Span::styled("512 GB", theme.text()),
        ]),
        Line::from(""),
        Line::from(Span::styled("  Status", theme.text_bold())),
        Line::from(vec![
            Span::styled("    Health: ", theme.text_muted()),
            Span::styled("● Healthy", theme.status_running()),
        ]),
        Line::from(vec![
            Span::styled("    Uptime: ", theme.text_muted()),
            Span::styled("15d 4h 32m", theme.text()),
        ]),
        Line::from(vec![
            Span::styled("    Assigned: ", theme.text_muted()),
            Span::styled("3/4 GPUs to Val-001", theme.text()),
        ]),
        Line::from(""),
        Line::from(Span::styled("  Actions", theme.text_bold())),
        Line::from(vec![
            Span::styled("    [t] Test SSH  [r] Restart  [d] Remove", theme.text_muted()),
        ]),
    ];

    let paragraph = Paragraph::new(details)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border())
                .title(Span::styled(" Node Details ", theme.block_title())),
        )
        .style(theme.text());

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

    if ctx.screens.fleet.show_details {
        let columns = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
            .split(content);

        render_node_list_ctx(frame, ctx, columns[0]);
        render_node_details_ctx(frame, theme, columns[1]);
    } else {
        render_node_list_ctx(frame, ctx, content);
    }
}

fn render_node_list_ctx(frame: &mut Frame, ctx: &RenderContext, area: Rect) {
    let theme = ctx.theme;

    let nodes = vec![
        ("node-001", "192.168.1.10", "22", "root", "H100 x 4", "Healthy"),
        ("node-002", "192.168.1.11", "22", "root", "H100 x 4", "Healthy"),
        ("node-003", "192.168.1.12", "22", "admin", "A100 x 8", "Healthy"),
        ("node-004", "192.168.1.13", "2222", "ubuntu", "A100 x 8", "Warning"),
        ("node-005", "192.168.1.14", "22", "root", "RTX 4090 x 4", "Healthy"),
        ("node-006", "192.168.1.15", "22", "root", "RTX 4090 x 4", "Offline"),
        ("node-007", "192.168.1.16", "22", "gpu-user", "L40S x 2", "Healthy"),
        ("node-008", "192.168.1.17", "22", "gpu-user", "L40S x 2", "Healthy"),
    ];

    let rows: Vec<Row> = nodes
        .iter()
        .enumerate()
        .map(|(i, (name, ip, port, user, gpu, status))| {
            let status_style = match *status {
                "Healthy" => theme.status_running(),
                "Warning" => theme.status_pending(),
                "Offline" => theme.status_error(),
                _ => theme.text(),
            };

            let row_style = if i == ctx.selected_index {
                theme.selected_row()
            } else {
                theme.text()
            };

            Row::new(vec![
                Cell::from(*name).style(theme.text_accent()),
                Cell::from(format!("{}@{}:{}", user, ip, port)),
                Cell::from(*gpu),
                Cell::from(format!("● {}", status)).style(status_style),
            ])
            .style(row_style)
        })
        .collect();

    let header_row = Row::new(vec!["Node ID", "SSH Connection", "GPUs", "Status"])
        .style(theme.header());

    let table = Table::new(
        rows,
        [
            Constraint::Length(12),
            Constraint::Length(28),
            Constraint::Length(16),
            Constraint::Min(12),
        ],
    )
    .header(header_row)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(theme.border_selected())
            .title(Span::styled(" Nodes ", theme.block_title())),
    )
    .row_highlight_style(theme.selected_row())
    .highlight_symbol("▶ ");

    let mut state = TableState::default();
    state.select(Some(ctx.selected_index));

    frame.render_stateful_widget(table, area, &mut state);
}

fn render_node_details_ctx(frame: &mut Frame, theme: &crate::ui::Theme, area: Rect) {
    let details = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("  Node ID: ", theme.text_muted()),
            Span::styled("node-001", theme.text_accent().add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(Span::styled("  Connection", theme.text_bold())),
        Line::from(vec![
            Span::styled("    Host: ", theme.text_muted()),
            Span::styled("192.168.1.10", theme.text()),
        ]),
        Line::from(vec![
            Span::styled("    Port: ", theme.text_muted()),
            Span::styled("22", theme.text()),
        ]),
        Line::from(vec![
            Span::styled("    User: ", theme.text_muted()),
            Span::styled("root", theme.text()),
        ]),
        Line::from(""),
        Line::from(Span::styled("  Hardware", theme.text_bold())),
        Line::from(vec![
            Span::styled("    GPUs: ", theme.text_muted()),
            Span::styled("4x NVIDIA H100 80GB", theme.text()),
        ]),
        Line::from(vec![
            Span::styled("    CPU: ", theme.text_muted()),
            Span::styled("AMD EPYC 7763 (128 cores)", theme.text()),
        ]),
        Line::from(vec![
            Span::styled("    RAM: ", theme.text_muted()),
            Span::styled("512 GB", theme.text()),
        ]),
        Line::from(""),
        Line::from(Span::styled("  Status", theme.text_bold())),
        Line::from(vec![
            Span::styled("    Health: ", theme.text_muted()),
            Span::styled("● Healthy", theme.status_running()),
        ]),
        Line::from(vec![
            Span::styled("    Uptime: ", theme.text_muted()),
            Span::styled("15d 4h 32m", theme.text()),
        ]),
        Line::from(vec![
            Span::styled("    Assigned: ", theme.text_muted()),
            Span::styled("3/4 GPUs to Val-001", theme.text()),
        ]),
        Line::from(""),
        Line::from(Span::styled("  Actions", theme.text_bold())),
        Line::from(vec![
            Span::styled("    [t] Test SSH  [r] Restart  [d] Remove", theme.text_muted()),
        ]),
    ];

    let paragraph = Paragraph::new(details)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border())
                .title(Span::styled(" Node Details ", theme.block_title())),
        )
        .style(theme.text());

    frame.render_widget(paragraph, area);
}

