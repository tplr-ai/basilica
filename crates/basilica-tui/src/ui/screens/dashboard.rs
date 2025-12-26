//! Dashboard screen - overview of active rentals and balance

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::Modifier,
    text::{Line, Span},
    widgets::{Block, Borders, Cell, List, ListItem, Paragraph, Row, Table},
    Frame,
};

use crate::app::{App, RenderContext};
use crate::ui::components::{footer, header};

/// Render the dashboard screen
pub fn render(frame: &mut Frame, app: &App) {
    let theme = &app.theme;
    let area = frame.area();

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
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(content);

    render_rentals_overview_impl(frame, theme, app.selected_index, columns[0]);

    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(7), Constraint::Min(0)])
        .split(columns[1]);

    render_balance_widget_impl(frame, theme, right_chunks[0]);
    render_activity_widget_impl(frame, theme, right_chunks[1]);
}

/// Render with context (for standalone render function)
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
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(content);

    render_rentals_overview_impl(frame, theme, ctx.selected_index, columns[0]);

    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(7), Constraint::Min(0)])
        .split(columns[1]);

    render_balance_widget_impl(frame, theme, right_chunks[0]);
    render_activity_widget_impl(frame, theme, right_chunks[1]);
}

fn render_rentals_overview_impl(frame: &mut Frame, theme: &crate::ui::Theme, selected_index: usize, area: Rect) {

    // Sample rental data (would come from app.user_data in real implementation)
    let rentals = vec![
        ("H100 x 1", "Running", "2h 15m", "$4.20"),
        ("A100 x 4", "Running", "45m", "$12.00"),
        ("H200 x 2", "Starting...", "0m", "$0.00"),
    ];

    let rows: Vec<Row> = rentals
        .iter()
        .enumerate()
        .map(|(i, (gpu, status, time, cost))| {
            let status_style = match *status {
                "Running" => theme.status_running(),
                "Starting..." => theme.status_pending(),
                _ => theme.text(),
            };

            let row_style = if i == selected_index {
                theme.selected_row()
            } else {
                theme.text()
            };

            Row::new(vec![
                Cell::from(*gpu),
                Cell::from(format!("● {}", status)).style(status_style),
                Cell::from(*time),
                Cell::from(*cost),
            ])
            .style(row_style)
        })
        .collect();

    let header = Row::new(vec!["GPU", "Status", "Uptime", "Cost"])
        .style(theme.header());

    let table = Table::new(
        rows,
        [
            Constraint::Percentage(30),
            Constraint::Percentage(25),
            Constraint::Percentage(20),
            Constraint::Percentage(25),
        ],
    )
    .header(header)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(theme.border())
            .title(Span::styled(
                format!(" Active Rentals ({}) ", 3),
                theme.block_title(),
            )),
    )
    .row_highlight_style(theme.selected_row())
    .highlight_symbol("▶ ");

    frame.render_widget(table, area);
}

fn render_balance_widget_impl(frame: &mut Frame, theme: &crate::ui::Theme, area: Rect) {

    let content = vec![
        Line::from(vec![
            Span::styled("Balance: ", theme.text_muted()),
            Span::styled("12.5 TAO", theme.text_accent().add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Spent Today: ", theme.text_muted()),
            Span::styled("0.82 TAO", theme.text()),
        ]),
        Line::from(vec![
            Span::styled("This Month:  ", theme.text_muted()),
            Span::styled("24.5 TAO", theme.text()),
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

fn render_activity_widget_impl(frame: &mut Frame, theme: &crate::ui::Theme, area: Rect) {

    let activities = vec![
        ("14:32", "✓", "Rental started (H100 x 1)", theme.text_success()),
        ("14:15", "↗", "Deploy scaled to 2 replicas", theme.text_info()),
        ("13:45", "$", "Payment received (5 TAO)", theme.text_success()),
        ("12:00", "✓", "Deployment ready", theme.text_success()),
        ("11:30", "⚠", "Rental approaching limit", theme.text_warning()),
    ];

    let items: Vec<ListItem> = activities
        .iter()
        .map(|(time, icon, msg, style)| {
            ListItem::new(Line::from(vec![
                Span::styled(format!("{} ", time), theme.text_muted()),
                Span::styled(format!("{} ", icon), *style),
                Span::styled(*msg, theme.text()),
            ]))
        })
        .collect();

    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border())
                .title(Span::styled(" Recent Activity ", theme.block_title())),
        )
        .style(theme.text());

    frame.render_widget(list, area);
}

