//! Rentals screen - list and manage active GPU rentals

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Paragraph, Row, Table, TableState, Wrap},
    Frame,
};

use crate::app::RenderContext;
use crate::ui::components::{footer, header};

/// Render with context
pub fn render_with_ctx(frame: &mut Frame, ctx: &RenderContext) {
    let area = frame.area();

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(header::header_height()),
            Constraint::Length(3), // Action bar
            Constraint::Min(0),
            Constraint::Length(footer::footer_height()),
        ])
        .split(area);

    header::render_header_ctx(frame, ctx, chunks[0]);
    render_action_bar(frame, ctx, chunks[1]);
    footer::render_footer_ctx(frame, ctx, chunks[3]);

    let content = chunks[2];

    if ctx.screens.rentals.show_logs {
        let split = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(content);

        render_rentals_table(frame, ctx, split[0]);
        render_logs_panel(frame, ctx, split[1]);
    } else {
        render_rentals_table(frame, ctx, content);
    }
}

fn render_action_bar(frame: &mut Frame, ctx: &RenderContext, area: Rect) {
    let theme = ctx.theme;
    let has_selection = !ctx.user_data.rentals.is_empty();

    let actions = Line::from(vec![
        Span::styled(" Actions: ", theme.text_muted()),
        Span::styled("[s] ", if has_selection { theme.keybind() } else { theme.text_muted() }),
        Span::styled("SSH  ", theme.text_muted()),
        Span::styled("[e] ", if has_selection { theme.keybind() } else { theme.text_muted() }),
        Span::styled("Exec  ", theme.text_muted()),
        Span::styled("[c] ", if has_selection { theme.keybind() } else { theme.text_muted() }),
        Span::styled("Copy  ", theme.text_muted()),
        Span::styled("[r] ", if has_selection { theme.keybind() } else { theme.text_muted() }),
        Span::styled("Restart  ", theme.text_muted()),
        Span::styled("[d] ", if has_selection { theme.keybind() } else { theme.text_muted() }),
        Span::styled("Down  ", theme.text_muted()),
        Span::styled("[l] ", theme.keybind()),
        Span::styled("Logs  ", theme.text_muted()),
        Span::raw("│ "),
        Span::styled("[f] ", theme.keybind()),
        Span::styled("Filter  ", theme.text_muted()),
        Span::styled("[h] ", theme.keybind()),
        Span::styled("History", theme.text_muted()),
    ]);

    let paragraph = Paragraph::new(actions)
        .block(Block::default().borders(Borders::ALL).border_style(theme.border()));

    frame.render_widget(paragraph, area);
}

fn render_rentals_table(frame: &mut Frame, ctx: &RenderContext, area: Rect) {
    let theme = ctx.theme;
    let user_data = ctx.user_data;

    let rows: Vec<Row> = if user_data.rentals.is_empty() {
        vec![Row::new(vec![Cell::from(
            "No active rentals. Go to Marketplace (Tab/3) to rent GPUs.",
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

                // GPU info
                let gpu = format!("{} x {}", rental.gpu_type, rental.gpu_count);

                // Truncate ID
                let id_short = if rental.id.len() > 8 {
                    &rental.id[..8]
                } else {
                    &rental.id
                };

                // Truncate image
                let image = if rental.container_image.len() > 25 {
                    format!("{}...", &rental.container_image[..22])
                } else {
                    rental.container_image.clone()
                };

                Row::new(vec![
                    Cell::from(id_short.to_string()),
                    Cell::from(gpu),
                    Cell::from(format!("● {}", rental.status)).style(status_style),
                    Cell::from(uptime),
                    Cell::from(image),
                    Cell::from(format!("${:.2}", rental.cost)),
                ])
                .style(row_style)
            })
            .collect()
    };

    let header_row =
        Row::new(vec!["ID", "GPU", "Status", "Uptime", "Image", "Cost"]).style(theme.header());

    let title = format!(" Rentals ({}) ", user_data.rentals.len());

    let table = Table::new(
        rows,
        [
            Constraint::Length(10),
            Constraint::Percentage(15),
            Constraint::Percentage(12),
            Constraint::Length(10),
            Constraint::Percentage(28),
            Constraint::Length(10),
        ],
    )
    .header(header_row)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(if ctx.screens.rentals.show_logs {
                theme.border()
            } else {
                theme.border_selected()
            })
            .title(Span::styled(title, theme.block_title())),
    )
    .row_highlight_style(theme.selected_row())
    .highlight_symbol("▶ ");

    let mut state = TableState::default();
    if !user_data.rentals.is_empty() {
        state.select(Some(
            ctx.selected_index
                .min(user_data.rentals.len().saturating_sub(1)),
        ));
    }

    frame.render_stateful_widget(table, area, &mut state);
}

fn render_logs_panel(frame: &mut Frame, ctx: &RenderContext, area: Rect) {
    let theme = ctx.theme;
    let user_data = ctx.user_data;

    // Get logs for selected rental
    let logs = if let Some(rental) = user_data.rentals.get(ctx.selected_index) {
        format!(
            "Logs for rental: {}\n\n\
             [Streaming logs not yet connected]\n\n\
             SSH into this rental:\n  ssh {}@{} -p {}\n\n\
             Press 's' to SSH, 'x' to stop rental",
            rental.id,
            rental.ssh_user.as_deref().unwrap_or("root"),
            rental.ssh_host.as_deref().unwrap_or("unknown"),
            rental.ssh_port.unwrap_or(22)
        )
    } else {
        "No rental selected".to_string()
    };

    let title = if ctx.screens.rentals.log_follow {
        " Logs (following) [Press l to toggle] "
    } else {
        " Logs [Press l to toggle] "
    };

    let paragraph = Paragraph::new(logs)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border_selected())
                .title(Span::styled(title, theme.block_title())),
        )
        .style(theme.text())
        .wrap(Wrap { trim: false });

    frame.render_widget(paragraph, area);
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
