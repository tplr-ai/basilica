//! Billing screen - balance, deposits, transaction history

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::Modifier,
    text::{Line, Span},
    widgets::{Block, Borders, Cell, List, ListItem, Paragraph, Row, Table},
    Frame,
};

use crate::app::{App, RenderContext};
use crate::ui::components::{footer, header};
use crate::ui::widgets::sparkline::text_sparkline;

/// Render the billing screen
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

    // Split into balance info and transaction history
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(35), Constraint::Percentage(65)])
        .split(content);

    // Left column: Balance and deposit info
    let left_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(10), // Balance
            Constraint::Length(8),  // Spending chart
            Constraint::Min(0),     // Deposit info
        ])
        .split(columns[0]);

    render_balance_info(frame, app, left_chunks[0]);
    render_spending_chart(frame, app, left_chunks[1]);
    render_deposit_info(frame, app, left_chunks[2]);

    // Right column: Transaction history
    render_transactions(frame, app, columns[1]);
}

fn render_balance_info(frame: &mut Frame, app: &App, area: Rect) {
    let theme = &app.theme;

    let content = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("  Available Balance", theme.text_muted()),
        ]),
        Line::from(vec![
            Span::styled("  ", theme.text()),
            Span::styled(
                "12.50 TAO",
                theme.text_accent().add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("  ≈ $125.00 USD", theme.text_muted()),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("  Active spend: ", theme.text_muted()),
            Span::styled("$2.50/hr", theme.text_warning()),
        ]),
    ];

    let paragraph = Paragraph::new(content)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border())
                .title(Span::styled(" Balance ", theme.block_title())),
        )
        .style(theme.text());

    frame.render_widget(paragraph, area);
}

fn render_spending_chart(frame: &mut Frame, app: &App, area: Rect) {
    let theme = &app.theme;

    // Sample spending data for last 7 days
    let spending_data: Vec<f64> = vec![2.5, 4.2, 3.8, 5.1, 4.0, 3.2, 2.8];
    let sparkline = text_sparkline(&spending_data);

    let content = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("  Last 7 days: ", theme.text_muted()),
        ]),
        Line::from(vec![
            Span::styled("  ", theme.text()),
            Span::styled(sparkline, theme.text_accent()),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("  Total: 25.6 TAO", theme.text()),
        ]),
    ];

    let paragraph = Paragraph::new(content)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border())
                .title(Span::styled(" Spending ", theme.block_title())),
        )
        .style(theme.text());

    frame.render_widget(paragraph, area);
}

fn render_deposit_info(frame: &mut Frame, app: &App, area: Rect) {
    let theme = &app.theme;

    let content = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("  Deposit Address", theme.text_muted()),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("  5Dq8xG...", theme.text_accent()),
        ]),
        Line::from(vec![
            Span::styled("  kFz3YnB4", theme.text_accent()),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("  Network: ", theme.text_muted()),
            Span::styled("Finney", theme.text()),
        ]),
        Line::from(vec![
            Span::styled("  Netuid:  ", theme.text_muted()),
            Span::styled("39", theme.text()),
        ]),
    ];

    let paragraph = Paragraph::new(content)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border())
                .title(Span::styled(" Deposit ", theme.block_title())),
        )
        .style(theme.text());

    frame.render_widget(paragraph, area);
}

fn render_transactions(frame: &mut Frame, app: &App, area: Rect) {
    let theme = &app.theme;

    // Sample transactions
    let transactions = vec![
        ("2024-01-15 14:32", "Rental", "H100 x 1", "-0.42 TAO", theme.text_error()),
        ("2024-01-15 12:00", "Deposit", "From wallet", "+5.00 TAO", theme.text_success()),
        ("2024-01-14 23:45", "Rental", "A100 x 4 (ended)", "-8.50 TAO", theme.text_error()),
        ("2024-01-14 15:00", "Deploy", "qwen-chat", "-2.30 TAO", theme.text_error()),
        ("2024-01-13 10:00", "Deposit", "From wallet", "+10.00 TAO", theme.text_success()),
        ("2024-01-12 18:30", "Rental", "RTX 4090 x 1", "-0.90 TAO", theme.text_error()),
        ("2024-01-12 09:00", "Deposit", "From wallet", "+20.00 TAO", theme.text_success()),
    ];

    let rows: Vec<Row> = transactions
        .iter()
        .enumerate()
        .map(|(i, (time, type_, desc, amount, style))| {
            let row_style = if i == app.selected_index {
                theme.selected_row()
            } else {
                theme.text()
            };

            Row::new(vec![
                Cell::from(*time).style(theme.text_muted()),
                Cell::from(*type_),
                Cell::from(*desc),
                Cell::from(*amount).style(*style),
            ])
            .style(row_style)
        })
        .collect();

    let header = Row::new(vec!["Time", "Type", "Description", "Amount"])
        .style(theme.header());

    let table = Table::new(
        rows,
        [
            Constraint::Length(18),
            Constraint::Length(10),
            Constraint::Min(20),
            Constraint::Length(12),
        ],
    )
    .header(header)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(theme.border())
            .title(Span::styled(" Transaction History ", theme.block_title())),
    )
    .row_highlight_style(theme.selected_row());

    frame.render_widget(table, area);
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
        .constraints([Constraint::Percentage(35), Constraint::Percentage(65)])
        .split(content);

    let left_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(10),
            Constraint::Length(8),
            Constraint::Min(0),
        ])
        .split(columns[0]);

    render_balance_info_ctx(frame, theme, left_chunks[0]);
    render_spending_chart_ctx(frame, theme, left_chunks[1]);
    render_deposit_info_ctx(frame, theme, left_chunks[2]);
    render_transactions_ctx(frame, ctx, columns[1]);
}

fn render_balance_info_ctx(frame: &mut Frame, theme: &crate::ui::Theme, area: Rect) {
    let content = vec![
        Line::from(""),
        Line::from(vec![Span::styled("  Available Balance", theme.text_muted())]),
        Line::from(vec![
            Span::styled("  ", theme.text()),
            Span::styled("12.50 TAO", theme.text_accent().add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![Span::styled("  ≈ $125.00 USD", theme.text_muted())]),
        Line::from(""),
        Line::from(vec![
            Span::styled("  Active spend: ", theme.text_muted()),
            Span::styled("$2.50/hr", theme.text_warning()),
        ]),
    ];

    let paragraph = Paragraph::new(content)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border())
                .title(Span::styled(" Balance ", theme.block_title())),
        )
        .style(theme.text());

    frame.render_widget(paragraph, area);
}

fn render_spending_chart_ctx(frame: &mut Frame, theme: &crate::ui::Theme, area: Rect) {
    let spending_data: Vec<f64> = vec![2.5, 4.2, 3.8, 5.1, 4.0, 3.2, 2.8];
    let sparkline = text_sparkline(&spending_data);

    let content = vec![
        Line::from(""),
        Line::from(vec![Span::styled("  Last 7 days: ", theme.text_muted())]),
        Line::from(vec![
            Span::styled("  ", theme.text()),
            Span::styled(sparkline, theme.text_accent()),
        ]),
        Line::from(""),
        Line::from(vec![Span::styled("  Total: 25.6 TAO", theme.text())]),
    ];

    let paragraph = Paragraph::new(content)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border())
                .title(Span::styled(" Spending ", theme.block_title())),
        )
        .style(theme.text());

    frame.render_widget(paragraph, area);
}

fn render_deposit_info_ctx(frame: &mut Frame, theme: &crate::ui::Theme, area: Rect) {
    let content = vec![
        Line::from(""),
        Line::from(vec![Span::styled("  Deposit Address", theme.text_muted())]),
        Line::from(""),
        Line::from(vec![Span::styled("  5Dq8xG...", theme.text_accent())]),
        Line::from(vec![Span::styled("  kFz3YnB4", theme.text_accent())]),
        Line::from(""),
        Line::from(vec![
            Span::styled("  Network: ", theme.text_muted()),
            Span::styled("Finney", theme.text()),
        ]),
        Line::from(vec![
            Span::styled("  Netuid:  ", theme.text_muted()),
            Span::styled("39", theme.text()),
        ]),
    ];

    let paragraph = Paragraph::new(content)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border())
                .title(Span::styled(" Deposit ", theme.block_title())),
        )
        .style(theme.text());

    frame.render_widget(paragraph, area);
}

fn render_transactions_ctx(frame: &mut Frame, ctx: &RenderContext, area: Rect) {
    let theme = ctx.theme;

    let transactions = vec![
        ("2024-01-15 14:32", "Rental", "H100 x 1", "-0.42 TAO", theme.text_error()),
        ("2024-01-15 12:00", "Deposit", "From wallet", "+5.00 TAO", theme.text_success()),
        ("2024-01-14 23:45", "Rental", "A100 x 4 (ended)", "-8.50 TAO", theme.text_error()),
        ("2024-01-14 15:00", "Deploy", "qwen-chat", "-2.30 TAO", theme.text_error()),
        ("2024-01-13 10:00", "Deposit", "From wallet", "+10.00 TAO", theme.text_success()),
        ("2024-01-12 18:30", "Rental", "RTX 4090 x 1", "-0.90 TAO", theme.text_error()),
        ("2024-01-12 09:00", "Deposit", "From wallet", "+20.00 TAO", theme.text_success()),
    ];

    let rows: Vec<Row> = transactions
        .iter()
        .enumerate()
        .map(|(i, (time, type_, desc, amount, style))| {
            let row_style = if i == ctx.selected_index {
                theme.selected_row()
            } else {
                theme.text()
            };

            Row::new(vec![
                Cell::from(*time).style(theme.text_muted()),
                Cell::from(*type_),
                Cell::from(*desc),
                Cell::from(*amount).style(*style),
            ])
            .style(row_style)
        })
        .collect();

    let header_row = Row::new(vec!["Time", "Type", "Description", "Amount"])
        .style(theme.header());

    let table = Table::new(
        rows,
        [
            Constraint::Length(18),
            Constraint::Length(10),
            Constraint::Min(20),
            Constraint::Length(12),
        ],
    )
    .header(header_row)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(theme.border())
            .title(Span::styled(" Transaction History ", theme.block_title())),
    )
    .row_highlight_style(theme.selected_row());

    frame.render_widget(table, area);
}

