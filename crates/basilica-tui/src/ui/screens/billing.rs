//! Billing screen - balance, deposits, transaction history

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::Modifier,
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Paragraph, Row, Table},
    Frame,
};

use crate::app::RenderContext;
use crate::ui::components::{footer, header};
use crate::ui::widgets::sparkline::text_sparkline;

/// Render with context
pub fn render_with_ctx(frame: &mut Frame, ctx: &RenderContext) {
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

    render_balance_info(frame, ctx, left_chunks[0]);
    render_spending_chart(frame, ctx, left_chunks[1]);
    render_deposit_info(frame, ctx, left_chunks[2]);
    render_transactions(frame, ctx, columns[1]);
}

fn render_balance_info(frame: &mut Frame, ctx: &RenderContext, area: Rect) {
    let theme = ctx.theme;
    let user_data = ctx.user_data;

    let content = if let Some(balance) = &user_data.balance {
        vec![
            Line::from(""),
            Line::from(vec![Span::styled(
                "  Available Balance",
                theme.text_muted(),
            )]),
            Line::from(vec![
                Span::styled("  ", theme.text()),
                Span::styled(
                    format!("{:.4} TAO", balance.available_tao),
                    theme.text_accent().add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(""),
            Line::from(vec![Span::styled(
                format!("  ≈ ${:.2} USD", balance.available_usd),
                theme.text_muted(),
            )]),
            Line::from(""),
            Line::from(vec![
                Span::styled("  Active spend: ", theme.text_muted()),
                Span::styled(
                    format!("${:.2}/hr", balance.active_spend_rate),
                    theme.text_warning(),
                ),
            ]),
        ]
    } else {
        vec![
            Line::from(""),
            Line::from(vec![Span::styled(
                if ctx.connected {
                    "  Loading..."
                } else {
                    "  Not connected"
                },
                theme.text_muted(),
            )]),
        ]
    };

    let paragraph = Paragraph::new(content)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border())
                .title(Span::styled(" 💰 Balance ", theme.block_title())),
        )
        .style(theme.text());

    frame.render_widget(paragraph, area);
}

fn render_spending_chart(frame: &mut Frame, ctx: &RenderContext, area: Rect) {
    let theme = ctx.theme;

    // TODO: Get real spending data from API
    let spending_data: Vec<f64> = vec![2.5, 4.2, 3.8, 5.1, 4.0, 3.2, 2.8];
    let sparkline = text_sparkline(&spending_data);

    let total: f64 = spending_data.iter().sum();

    let content = vec![
        Line::from(""),
        Line::from(vec![Span::styled("  Last 7 days: ", theme.text_muted())]),
        Line::from(vec![
            Span::styled("  ", theme.text()),
            Span::styled(sparkline, theme.text_accent()),
        ]),
        Line::from(""),
        Line::from(vec![Span::styled(
            format!("  Total: {:.1} TAO", total),
            theme.text(),
        )]),
    ];

    let paragraph = Paragraph::new(content)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border())
                .title(Span::styled(" 📊 Spending ", theme.block_title())),
        )
        .style(theme.text());

    frame.render_widget(paragraph, area);
}

fn render_deposit_info(frame: &mut Frame, ctx: &RenderContext, area: Rect) {
    let theme = ctx.theme;

    // TODO: Get deposit address from API
    let content = vec![
        Line::from(""),
        Line::from(vec![Span::styled("  Deposit Address", theme.text_muted())]),
        Line::from(""),
        Line::from(vec![Span::styled(
            "  Run 'basilica deposit'",
            theme.text_accent(),
        )]),
        Line::from(vec![Span::styled("  for address", theme.text_accent())]),
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
                .title(Span::styled(" 💳 Deposit ", theme.block_title())),
        )
        .style(theme.text());

    frame.render_widget(paragraph, area);
}

fn render_transactions(frame: &mut Frame, ctx: &RenderContext, area: Rect) {
    let theme = ctx.theme;
    let user_data = ctx.user_data;

    let rows: Vec<Row> = if user_data.transactions.is_empty() {
        vec![Row::new(vec![Cell::from(
            "No transactions yet. Top up your balance to get started.",
        )])
        .style(theme.text_muted())]
    } else {
        user_data
            .transactions
            .iter()
            .enumerate()
            .map(|(i, tx)| {
                let amount_style = if tx.is_credit {
                    theme.text_success()
                } else {
                    theme.text_error()
                };

                let row_style = if i == ctx.selected_index {
                    theme.selected_row()
                } else {
                    theme.text()
                };

                let amount_str = if tx.is_credit {
                    format!("+{:.4} TAO", tx.amount)
                } else {
                    format!("-{:.4} TAO", tx.amount)
                };

                Row::new(vec![
                    Cell::from(tx.timestamp.clone()).style(theme.text_muted()),
                    Cell::from(tx.transaction_type.clone()),
                    Cell::from(tx.description.clone()),
                    Cell::from(amount_str).style(amount_style),
                ])
                .style(row_style)
            })
            .collect()
    };

    let header_row = Row::new(vec!["Time", "Type", "Description", "Amount"]).style(theme.header());

    let title = format!(" Transaction History ({}) ", user_data.transactions.len());

    let table = Table::new(
        rows,
        [
            Constraint::Length(18),
            Constraint::Length(10),
            Constraint::Min(20),
            Constraint::Length(14),
        ],
    )
    .header(header_row)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(theme.border())
            .title(Span::styled(title, theme.block_title())),
    )
    .row_highlight_style(theme.selected_row());

    frame.render_widget(table, area);
}
