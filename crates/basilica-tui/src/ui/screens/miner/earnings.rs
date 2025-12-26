//! Earnings screen - revenue and payment history

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::Modifier,
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Paragraph, Row, Table},
    Frame,
};

use crate::app::{App, RenderContext};
use crate::ui::components::{footer, header};
use crate::ui::widgets::sparkline::text_sparkline;

/// Render the earnings screen
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

    // Split into summary and history
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(35), Constraint::Percentage(65)])
        .split(content);

    // Left column: Summary stats
    let left_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8),  // Current earnings
            Constraint::Length(10), // Revenue chart
            Constraint::Min(0),     // Projections
        ])
        .split(columns[0]);

    render_earnings_summary(frame, app, left_chunks[0]);
    render_revenue_chart(frame, app, left_chunks[1]);
    render_projections(frame, app, left_chunks[2]);

    // Right column: Payment history
    render_payment_history(frame, app, columns[1]);
}

fn render_earnings_summary(frame: &mut Frame, app: &App, area: Rect) {
    let theme = &app.theme;

    let content = vec![
        Line::from(""),
        Line::from(vec![Span::styled("  Current Rate", theme.text_muted())]),
        Line::from(vec![Span::styled(
            "  $12.50/hr",
            theme.text_success().add_modifier(Modifier::BOLD),
        )]),
        Line::from(""),
        Line::from(vec![
            Span::styled("  Today: ", theme.text_muted()),
            Span::styled("$145.20", theme.text_accent()),
        ]),
        Line::from(vec![
            Span::styled("  This Week: ", theme.text_muted()),
            Span::styled("$1,234.50", theme.text()),
        ]),
    ];

    let paragraph = Paragraph::new(content)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border())
                .title(Span::styled(" Earnings ", theme.block_title())),
        )
        .style(theme.text());

    frame.render_widget(paragraph, area);
}

fn render_revenue_chart(frame: &mut Frame, app: &App, area: Rect) {
    let theme = &app.theme;

    // Sample revenue data for last 14 days
    let revenue_data: Vec<f64> = vec![
        120.0, 145.0, 132.0, 158.0, 142.0, 168.0, 155.0, 175.0, 162.0, 180.0, 170.0, 195.0, 145.0,
        145.0,
    ];
    let sparkline = text_sparkline(&revenue_data);

    let content = vec![
        Line::from(""),
        Line::from(vec![Span::styled("  Last 14 days:", theme.text_muted())]),
        Line::from(vec![
            Span::styled("  ", theme.text()),
            Span::styled(sparkline, theme.text_accent()),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("  Avg: ", theme.text_muted()),
            Span::styled("$156.64/day", theme.text()),
        ]),
        Line::from(vec![
            Span::styled("  Total: ", theme.text_muted()),
            Span::styled("$2,193.00", theme.text_success()),
        ]),
    ];

    let paragraph = Paragraph::new(content)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border())
                .title(Span::styled(" Revenue ", theme.block_title())),
        )
        .style(theme.text());

    frame.render_widget(paragraph, area);
}

fn render_projections(frame: &mut Frame, app: &App, area: Rect) {
    let theme = &app.theme;

    let content = vec![
        Line::from(""),
        Line::from(Span::styled("  Projections", theme.text_bold())),
        Line::from(""),
        Line::from(vec![
            Span::styled("  This Month: ", theme.text_muted()),
            Span::styled("~$4,700", theme.text()),
        ]),
        Line::from(vec![
            Span::styled("  Annual: ", theme.text_muted()),
            Span::styled("~$56,400", theme.text_accent()),
        ]),
        Line::from(""),
        Line::from(Span::styled("  GPU Utilization", theme.text_bold())),
        Line::from(""),
        Line::from(vec![
            Span::styled("  Average: ", theme.text_muted()),
            Span::styled("78%", theme.text_success()),
        ]),
        Line::from(vec![
            Span::styled("  Peak: ", theme.text_muted()),
            Span::styled("95%", theme.text()),
        ]),
    ];

    let paragraph = Paragraph::new(content)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border())
                .title(Span::styled(" Projections ", theme.block_title())),
        )
        .style(theme.text());

    frame.render_widget(paragraph, area);
}

fn render_payment_history(frame: &mut Frame, app: &App, area: Rect) {
    let theme = &app.theme;

    // Sample payment data
    let payments = [(
            "2024-01-15",
            "Val-001",
            "24h rental",
            "+$280.00",
            "Completed",
        ),
        (
            "2024-01-14",
            "Val-002",
            "12h rental",
            "+$135.00",
            "Completed",
        ),
        ("2024-01-14", "Val-001", "8h rental", "+$95.00", "Completed"),
        ("2024-01-13", "Val-003", "4h rental", "+$45.00", "Completed"),
        (
            "2024-01-13",
            "Val-001",
            "16h rental",
            "+$190.00",
            "Completed",
        ),
        (
            "2024-01-12",
            "Val-002",
            "24h rental",
            "+$310.00",
            "Completed",
        ),
        (
            "2024-01-11",
            "Val-001",
            "12h rental",
            "+$140.00",
            "Completed",
        ),
        ("2024-01-10", "Val-003", "8h rental", "+$92.00", "Completed")];

    let rows: Vec<Row> = payments
        .iter()
        .enumerate()
        .map(|(i, (date, validator, desc, amount, status))| {
            let row_style = if i == app.selected_index {
                theme.selected_row()
            } else {
                theme.text()
            };

            Row::new(vec![
                Cell::from(*date).style(theme.text_muted()),
                Cell::from(*validator).style(theme.text_accent()),
                Cell::from(*desc),
                Cell::from(*amount).style(theme.text_success()),
                Cell::from(*status).style(theme.text_muted()),
            ])
            .style(row_style)
        })
        .collect();

    let header = Row::new(vec!["Date", "Validator", "Description", "Amount", "Status"])
        .style(theme.header());

    let table = Table::new(
        rows,
        [
            Constraint::Length(12),
            Constraint::Length(10),
            Constraint::Min(15),
            Constraint::Length(12),
            Constraint::Length(12),
        ],
    )
    .header(header)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(theme.border())
            .title(Span::styled(" Payment History ", theme.block_title())),
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
            Constraint::Length(8),
            Constraint::Length(10),
            Constraint::Min(0),
        ])
        .split(columns[0]);

    render_earnings_summary_ctx(frame, theme, left_chunks[0]);
    render_revenue_chart_ctx(frame, theme, left_chunks[1]);
    render_projections_ctx(frame, theme, left_chunks[2]);
    render_payment_history_ctx(frame, ctx, columns[1]);
}

fn render_earnings_summary_ctx(frame: &mut Frame, theme: &crate::ui::Theme, area: Rect) {
    let content = vec![
        Line::from(""),
        Line::from(vec![Span::styled("  Current Rate", theme.text_muted())]),
        Line::from(vec![Span::styled(
            "  $12.50/hr",
            theme.text_success().add_modifier(Modifier::BOLD),
        )]),
        Line::from(""),
        Line::from(vec![
            Span::styled("  Today: ", theme.text_muted()),
            Span::styled("$145.20", theme.text_accent()),
        ]),
        Line::from(vec![
            Span::styled("  This Week: ", theme.text_muted()),
            Span::styled("$1,234.50", theme.text()),
        ]),
    ];

    frame.render_widget(
        Paragraph::new(content)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(theme.border())
                    .title(Span::styled(" Earnings ", theme.block_title())),
            )
            .style(theme.text()),
        area,
    );
}

fn render_revenue_chart_ctx(frame: &mut Frame, theme: &crate::ui::Theme, area: Rect) {
    let revenue_data: Vec<f64> = vec![
        120.0, 145.0, 132.0, 158.0, 142.0, 168.0, 155.0, 175.0, 162.0, 180.0, 170.0, 195.0, 145.0,
        145.0,
    ];
    let sparkline = text_sparkline(&revenue_data);

    let content = vec![
        Line::from(""),
        Line::from(vec![Span::styled("  Last 14 days:", theme.text_muted())]),
        Line::from(vec![
            Span::styled("  ", theme.text()),
            Span::styled(sparkline, theme.text_accent()),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("  Avg: ", theme.text_muted()),
            Span::styled("$156.64/day", theme.text()),
        ]),
        Line::from(vec![
            Span::styled("  Total: ", theme.text_muted()),
            Span::styled("$2,193.00", theme.text_success()),
        ]),
    ];

    frame.render_widget(
        Paragraph::new(content)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(theme.border())
                    .title(Span::styled(" Revenue ", theme.block_title())),
            )
            .style(theme.text()),
        area,
    );
}

fn render_projections_ctx(frame: &mut Frame, theme: &crate::ui::Theme, area: Rect) {
    let content = vec![
        Line::from(""),
        Line::from(Span::styled("  Projections", theme.text_bold())),
        Line::from(""),
        Line::from(vec![
            Span::styled("  This Month: ", theme.text_muted()),
            Span::styled("~$4,700", theme.text()),
        ]),
        Line::from(vec![
            Span::styled("  Annual: ", theme.text_muted()),
            Span::styled("~$56,400", theme.text_accent()),
        ]),
        Line::from(""),
        Line::from(Span::styled("  GPU Utilization", theme.text_bold())),
        Line::from(""),
        Line::from(vec![
            Span::styled("  Average: ", theme.text_muted()),
            Span::styled("78%", theme.text_success()),
        ]),
        Line::from(vec![
            Span::styled("  Peak: ", theme.text_muted()),
            Span::styled("95%", theme.text()),
        ]),
    ];

    frame.render_widget(
        Paragraph::new(content)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(theme.border())
                    .title(Span::styled(" Projections ", theme.block_title())),
            )
            .style(theme.text()),
        area,
    );
}

fn render_payment_history_ctx(frame: &mut Frame, ctx: &RenderContext, area: Rect) {
    let theme = ctx.theme;

    let payments = [(
            "2024-01-15",
            "Val-001",
            "24h rental",
            "+$280.00",
            "Completed",
        ),
        (
            "2024-01-14",
            "Val-002",
            "12h rental",
            "+$135.00",
            "Completed",
        ),
        ("2024-01-14", "Val-001", "8h rental", "+$95.00", "Completed"),
        ("2024-01-13", "Val-003", "4h rental", "+$45.00", "Completed"),
        (
            "2024-01-13",
            "Val-001",
            "16h rental",
            "+$190.00",
            "Completed",
        ),
        (
            "2024-01-12",
            "Val-002",
            "24h rental",
            "+$310.00",
            "Completed",
        ),
        (
            "2024-01-11",
            "Val-001",
            "12h rental",
            "+$140.00",
            "Completed",
        ),
        ("2024-01-10", "Val-003", "8h rental", "+$92.00", "Completed")];

    let rows: Vec<Row> = payments
        .iter()
        .enumerate()
        .map(|(i, (date, validator, desc, amount, status))| {
            let row_style = if i == ctx.selected_index {
                theme.selected_row()
            } else {
                theme.text()
            };

            Row::new(vec![
                Cell::from(*date).style(theme.text_muted()),
                Cell::from(*validator).style(theme.text_accent()),
                Cell::from(*desc),
                Cell::from(*amount).style(theme.text_success()),
                Cell::from(*status).style(theme.text_muted()),
            ])
            .style(row_style)
        })
        .collect();

    let header_row = Row::new(vec!["Date", "Validator", "Description", "Amount", "Status"])
        .style(theme.header());

    let table = Table::new(
        rows,
        [
            Constraint::Length(12),
            Constraint::Length(10),
            Constraint::Min(15),
            Constraint::Length(12),
            Constraint::Length(12),
        ],
    )
    .header(header_row)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(theme.border())
            .title(Span::styled(" Payment History ", theme.block_title())),
    )
    .row_highlight_style(theme.selected_row());

    frame.render_widget(table, area);
}
