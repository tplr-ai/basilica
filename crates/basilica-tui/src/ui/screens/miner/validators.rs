//! Validators screen - validator assignments and status

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::Modifier,
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Paragraph, Row, Table, TableState},
    Frame,
};

use crate::app::{App, RenderContext};
use crate::ui::components::{footer, header};

/// Render the validators screen
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

    // Split into assignment info and validator list
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(30), Constraint::Percentage(70)])
        .split(content);

    render_assignment_info(frame, app, columns[0]);
    render_validator_list(frame, app, columns[1]);
}

fn render_assignment_info(frame: &mut Frame, app: &App, area: Rect) {
    let theme = &app.theme;

    // Split into sections
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8),  // Miner info
            Constraint::Length(10), // Assignment strategy
            Constraint::Min(0),     // Discovery status
        ])
        .split(area);

    // Miner info
    let miner_info = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("  UID: ", theme.text_muted()),
            Span::styled("42", theme.text_accent().add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled("  Hotkey: ", theme.text_muted()),
            Span::styled("5Dq8x...kFz3", theme.text()),
        ]),
        Line::from(vec![
            Span::styled("  Stake: ", theme.text_muted()),
            Span::styled("1,250 TAO", theme.text_success()),
        ]),
        Line::from(vec![
            Span::styled("  Netuid: ", theme.text_muted()),
            Span::styled("39", theme.text()),
        ]),
    ];

    let miner_block = Paragraph::new(miner_info)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border())
                .title(Span::styled(" Miner ", theme.block_title())),
        )
        .style(theme.text());

    frame.render_widget(miner_block, chunks[0]);

    // Assignment strategy
    let strategy_info = vec![
        Line::from(""),
        Line::from(vec![Span::styled("  Strategy: ", theme.text_muted())]),
        Line::from(vec![Span::styled("  highest_stake", theme.text_accent())]),
        Line::from(""),
        Line::from(vec![
            Span::styled("  Assigned to: ", theme.text_muted()),
            Span::styled("3", theme.text_success().add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![Span::styled("  validators", theme.text_muted())]),
    ];

    let strategy_block = Paragraph::new(strategy_info)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border())
                .title(Span::styled(" Assignment ", theme.block_title())),
        )
        .style(theme.text());

    frame.render_widget(strategy_block, chunks[1]);

    // Discovery status
    let discovery_info = vec![
        Line::from(""),
        Line::from(vec![Span::styled("  Last discovery:", theme.text_muted())]),
        Line::from(vec![Span::styled("  2 min ago", theme.text())]),
        Line::from(""),
        Line::from(vec![
            Span::styled("  Next in: ", theme.text_muted()),
            Span::styled("8 min", theme.text()),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("  ● ", theme.status_running()),
            Span::styled("Discovery active", theme.text()),
        ]),
    ];

    let discovery_block = Paragraph::new(discovery_info)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border())
                .title(Span::styled(" Discovery ", theme.block_title())),
        )
        .style(theme.text());

    frame.render_widget(discovery_block, chunks[2]);
}

fn render_validator_list(frame: &mut Frame, app: &App, area: Rect) {
    let theme = &app.theme;

    // Sample validator data
    let validators = [(
            "Val-001",
            "5Abc...xyz",
            "1,500,000",
            "Active",
            "12",
            "node-001, node-002",
        ),
        (
            "Val-002",
            "5Def...uvw",
            "850,000",
            "Active",
            "8",
            "node-003",
        ),
        (
            "Val-003",
            "5Ghi...rst",
            "720,000",
            "Active",
            "4",
            "node-005, node-007",
        ),
        ("Val-004", "5Jkl...opq", "500,000", "Pending", "0", "-"),
        ("Val-005", "5Mno...lmn", "350,000", "Inactive", "0", "-")];

    let rows: Vec<Row> = validators
        .iter()
        .enumerate()
        .map(|(i, (name, hotkey, stake, status, gpus, nodes))| {
            let status_style = match *status {
                "Active" => theme.status_running(),
                "Pending" => theme.status_pending(),
                "Inactive" => theme.status_stopped(),
                _ => theme.text(),
            };

            let row_style = if i == app.selected_index {
                theme.selected_row()
            } else {
                theme.text()
            };

            Row::new(vec![
                Cell::from(*name).style(theme.text_accent()),
                Cell::from(*hotkey),
                Cell::from(format!("{} τ", stake)),
                Cell::from(format!("● {}", status)).style(status_style),
                Cell::from(*gpus),
                Cell::from(*nodes).style(theme.text_muted()),
            ])
            .style(row_style)
        })
        .collect();

    let header = Row::new(vec![
        "Validator",
        "Hotkey",
        "Stake",
        "Status",
        "GPUs",
        "Nodes",
    ])
    .style(theme.header());

    let table = Table::new(
        rows,
        [
            Constraint::Length(10),
            Constraint::Length(14),
            Constraint::Length(14),
            Constraint::Length(12),
            Constraint::Length(6),
            Constraint::Min(20),
        ],
    )
    .header(header)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(theme.border_selected())
            .title(Span::styled(" Validators ", theme.block_title())),
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
        .constraints([Constraint::Percentage(30), Constraint::Percentage(70)])
        .split(content);

    render_assignment_info_ctx(frame, theme, columns[0]);
    render_validator_list_ctx(frame, ctx, columns[1]);
}

fn render_assignment_info_ctx(frame: &mut Frame, theme: &crate::ui::Theme, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8),
            Constraint::Length(10),
            Constraint::Min(0),
        ])
        .split(area);

    let miner_info = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("  UID: ", theme.text_muted()),
            Span::styled("42", theme.text_accent().add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled("  Hotkey: ", theme.text_muted()),
            Span::styled("5Dq8x...kFz3", theme.text()),
        ]),
        Line::from(vec![
            Span::styled("  Stake: ", theme.text_muted()),
            Span::styled("1,250 TAO", theme.text_success()),
        ]),
        Line::from(vec![
            Span::styled("  Netuid: ", theme.text_muted()),
            Span::styled("39", theme.text()),
        ]),
    ];

    frame.render_widget(
        Paragraph::new(miner_info)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(theme.border())
                    .title(Span::styled(" Miner ", theme.block_title())),
            )
            .style(theme.text()),
        chunks[0],
    );

    let strategy_info = vec![
        Line::from(""),
        Line::from(vec![Span::styled("  Strategy: ", theme.text_muted())]),
        Line::from(vec![Span::styled("  highest_stake", theme.text_accent())]),
        Line::from(""),
        Line::from(vec![
            Span::styled("  Assigned to: ", theme.text_muted()),
            Span::styled("3", theme.text_success().add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![Span::styled("  validators", theme.text_muted())]),
    ];

    frame.render_widget(
        Paragraph::new(strategy_info)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(theme.border())
                    .title(Span::styled(" Assignment ", theme.block_title())),
            )
            .style(theme.text()),
        chunks[1],
    );

    let discovery_info = vec![
        Line::from(""),
        Line::from(vec![Span::styled("  Last discovery:", theme.text_muted())]),
        Line::from(vec![Span::styled("  2 min ago", theme.text())]),
        Line::from(""),
        Line::from(vec![
            Span::styled("  Next in: ", theme.text_muted()),
            Span::styled("8 min", theme.text()),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("  ● ", theme.status_running()),
            Span::styled("Discovery active", theme.text()),
        ]),
    ];

    frame.render_widget(
        Paragraph::new(discovery_info)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(theme.border())
                    .title(Span::styled(" Discovery ", theme.block_title())),
            )
            .style(theme.text()),
        chunks[2],
    );
}

fn render_validator_list_ctx(frame: &mut Frame, ctx: &RenderContext, area: Rect) {
    let theme = ctx.theme;

    let validators = [(
            "Val-001",
            "5Abc...xyz",
            "1,500,000",
            "Active",
            "12",
            "node-001, node-002",
        ),
        (
            "Val-002",
            "5Def...uvw",
            "850,000",
            "Active",
            "8",
            "node-003",
        ),
        (
            "Val-003",
            "5Ghi...rst",
            "720,000",
            "Active",
            "4",
            "node-005, node-007",
        ),
        ("Val-004", "5Jkl...opq", "500,000", "Pending", "0", "-"),
        ("Val-005", "5Mno...lmn", "350,000", "Inactive", "0", "-")];

    let rows: Vec<Row> = validators
        .iter()
        .enumerate()
        .map(|(i, (name, hotkey, stake, status, gpus, nodes))| {
            let status_style = match *status {
                "Active" => theme.status_running(),
                "Pending" => theme.status_pending(),
                "Inactive" => theme.status_stopped(),
                _ => theme.text(),
            };

            let row_style = if i == ctx.selected_index {
                theme.selected_row()
            } else {
                theme.text()
            };

            Row::new(vec![
                Cell::from(*name).style(theme.text_accent()),
                Cell::from(*hotkey),
                Cell::from(format!("{} τ", stake)),
                Cell::from(format!("● {}", status)).style(status_style),
                Cell::from(*gpus),
                Cell::from(*nodes).style(theme.text_muted()),
            ])
            .style(row_style)
        })
        .collect();

    let header_row = Row::new(vec![
        "Validator",
        "Hotkey",
        "Stake",
        "Status",
        "GPUs",
        "Nodes",
    ])
    .style(theme.header());

    let table = Table::new(
        rows,
        [
            Constraint::Length(10),
            Constraint::Length(14),
            Constraint::Length(14),
            Constraint::Length(12),
            Constraint::Length(6),
            Constraint::Min(20),
        ],
    )
    .header(header_row)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(theme.border_selected())
            .title(Span::styled(" Validators ", theme.block_title())),
    )
    .row_highlight_style(theme.selected_row())
    .highlight_symbol("▶ ");

    let mut state = TableState::default();
    state.select(Some(ctx.selected_index));

    frame.render_stateful_widget(table, area, &mut state);
}
