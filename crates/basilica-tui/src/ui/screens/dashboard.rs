//! Dashboard screen - overview of active rentals and balance

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::Modifier,
    text::{Line, Span},
    widgets::{Block, Borders, Cell, List, ListItem, Paragraph, Row, Table},
    Frame,
};

use crate::app::RenderContext;
use crate::ui::components::{footer, header};

/// Render with context
pub fn render_with_ctx(frame: &mut Frame, ctx: &RenderContext) {
    let _theme = ctx.theme; // Used by helper functions via ctx
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

    render_rentals_overview(frame, ctx, columns[0]);

    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(8), Constraint::Min(0)])
        .split(columns[1]);

    render_balance_widget(frame, ctx, right_chunks[0]);
    render_status_widget(frame, ctx, right_chunks[1]);
}

fn render_rentals_overview(frame: &mut Frame, ctx: &RenderContext, area: Rect) {
    let theme = ctx.theme;
    let user_data = ctx.user_data;

    let rows: Vec<Row> = if user_data.rentals.is_empty() {
        vec![Row::new(vec![Cell::from(
            "No active rentals. Press 'm' for marketplace.",
        )])
        .style(theme.text_muted())]
    } else {
        user_data
            .rentals
            .iter()
            .enumerate()
            .map(|(i, rental)| {
                let status_style = match rental.status.as_str() {
                    "Running" | "Active" => theme.status_running(),
                    "Starting" | "Pending" => theme.status_pending(),
                    "Stopped" => theme.status_stopped(),
                    _ => theme.text(),
                };

                let row_style = if i == ctx.selected_index {
                    theme.selected_row()
                } else {
                    theme.text()
                };

                // Format uptime
                let uptime = format_uptime(rental.uptime_minutes);

                // Format GPU
                let gpu = format!("{} x {}", rental.gpu_type, rental.gpu_count);

                Row::new(vec![
                    Cell::from(gpu),
                    Cell::from(format!("● {}", rental.status)).style(status_style),
                    Cell::from(uptime),
                    Cell::from(format!("${:.2}", rental.cost)),
                ])
                .style(row_style)
            })
            .collect()
    };

    let header = Row::new(vec!["GPU", "Status", "Uptime", "Cost"]).style(theme.header());

    let title = format!(" Active Rentals ({}) ", user_data.rentals.len());

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
            .title(Span::styled(title, theme.block_title())),
    )
    .row_highlight_style(theme.selected_row())
    .highlight_symbol("▶ ");

    frame.render_widget(table, area);
}

fn render_balance_widget(frame: &mut Frame, ctx: &RenderContext, area: Rect) {
    let theme = ctx.theme;
    let user_data = ctx.user_data;

    let content = if let Some(balance) = &user_data.balance {
        vec![
            Line::from(vec![
                Span::styled("Balance: ", theme.text_muted()),
                Span::styled(
                    format!("{:.4} TAO", balance.available_tao),
                    theme.text_accent().add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(vec![
                Span::styled("         ", theme.text_muted()),
                Span::styled(
                    format!("(${:.2})", balance.available_usd),
                    theme.text_muted(),
                ),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("Spent Today: ", theme.text_muted()),
                Span::styled(format!("{:.4} TAO", balance.spent_today), theme.text()),
            ]),
            Line::from(vec![
                Span::styled("This Month:  ", theme.text_muted()),
                Span::styled(format!("{:.4} TAO", balance.spent_this_month), theme.text()),
            ]),
        ]
    } else {
        vec![
            Line::from(Span::styled("Loading balance...", theme.text_muted())),
            Line::from(""),
            Line::from(Span::styled(
                if ctx.connected {
                    "Fetching from API..."
                } else {
                    "Not connected"
                },
                theme.text_muted(),
            )),
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

fn render_status_widget(frame: &mut Frame, ctx: &RenderContext, area: Rect) {
    let theme = ctx.theme;
    let user_data = ctx.user_data;

    let active_rentals = user_data.active_rentals_count();
    let total_gpus = user_data.total_gpus();
    let deployments = user_data.deployments.len();
    let offerings = user_data.offerings.len();

    let status_icon = if ctx.connected { "🟢" } else { "🔴" };
    let status_text = if ctx.connected {
        "Connected"
    } else {
        "Offline"
    };

    let items = vec![
        ListItem::new(Line::from(vec![
            Span::styled(format!("{} ", status_icon), theme.text()),
            Span::styled(
                status_text,
                if ctx.connected {
                    theme.text_success()
                } else {
                    theme.text_error()
                },
            ),
        ])),
        ListItem::new(Line::from("")),
        ListItem::new(Line::from(vec![
            Span::styled("📊 ", theme.text()),
            Span::styled(format!("{} active rentals", active_rentals), theme.text()),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("🖥️  ", theme.text()),
            Span::styled(format!("{} GPUs in use", total_gpus), theme.text()),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("🚀 ", theme.text()),
            Span::styled(format!("{} deployments", deployments), theme.text()),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("🏪 ", theme.text()),
            Span::styled(format!("{} GPUs available", offerings), theme.text()),
        ])),
    ];

    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border())
                .title(Span::styled(" Status ", theme.block_title())),
        )
        .style(theme.text());

    frame.render_widget(list, area);
}

/// Format minutes into human-readable uptime
fn format_uptime(minutes: u64) -> String {
    if minutes < 60 {
        format!("{}m", minutes)
    } else if minutes < 1440 {
        let hours = minutes / 60;
        let mins = minutes % 60;
        format!("{}h {}m", hours, mins)
    } else {
        let days = minutes / 1440;
        let hours = (minutes % 1440) / 60;
        format!("{}d {}h", days, hours)
    }
}
