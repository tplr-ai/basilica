//! Marketplace screen - browse and rent GPUs

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Paragraph, Row, Table, TableState},
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

    render_filters(frame, ctx, columns[0]);
    render_gpu_list(frame, ctx, columns[1]);
}

fn render_filters(frame: &mut Frame, ctx: &RenderContext, area: Rect) {
    let theme = ctx.theme;
    let user_data = ctx.user_data;

    // Count GPUs by type
    let mut gpu_types: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for offering in &user_data.offerings {
        *gpu_types.entry(&offering.gpu_type).or_default() += 1;
    }

    let mut content = vec![
        Line::from(Span::styled("GPU Type", theme.text_bold())),
        Line::from(""),
        Line::from(vec![
            Span::raw("  "),
            Span::styled("●", theme.text_accent()),
            Span::styled(
                format!(" All ({})", user_data.offerings.len()),
                theme.text(),
            ),
        ]),
    ];

    // Add GPU type options
    for (gpu_type, count) in gpu_types.iter() {
        content.push(Line::from(format!("    ○ {} ({})", gpu_type, count)));
    }

    content.extend(vec![
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
        Line::from(""),
        Line::from(Span::styled("Press Enter", theme.text_muted())),
        Line::from(Span::styled("to rent GPU", theme.text_muted())),
    ]);

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

fn render_gpu_list(frame: &mut Frame, ctx: &RenderContext, area: Rect) {
    let theme = ctx.theme;
    let user_data = ctx.user_data;

    let rows: Vec<Row> = if user_data.offerings.is_empty() {
        if user_data.loading.offerings {
            vec![Row::new(vec![Cell::from("Loading available GPUs...")]).style(theme.text_muted())]
        } else if !ctx.connected {
            vec![Row::new(vec![Cell::from(
                "Not connected. Run 'basilica login' first.",
            )])
            .style(theme.text_warning())]
        } else {
            vec![
                Row::new(vec![Cell::from("No GPUs available at this time.")])
                    .style(theme.text_muted()),
            ]
        }
    } else {
        user_data
            .offerings
            .iter()
            .enumerate()
            .map(|(i, offering)| {
                let source_style = match offering.source.as_str() {
                    "secure" | "Secure" => theme.text_success(),
                    "community" | "Community" | "basilica" => theme.text_info(),
                    _ => theme.text(),
                };

                let row_style = if i == ctx.selected_index {
                    theme.selected_row()
                } else {
                    theme.text()
                };

                Row::new(vec![
                    Cell::from(offering.gpu_type.clone()),
                    Cell::from(format!("{}x", offering.gpu_count)),
                    Cell::from(format!("{}GB", offering.memory_gb)),
                    Cell::from(format!("${:.2}/hr", offering.price_per_hour))
                        .style(theme.text_accent()),
                    Cell::from(offering.source.clone()).style(source_style),
                    Cell::from(format!("{}", offering.available)),
                ])
                .style(row_style)
            })
            .collect()
    };

    let header_row =
        Row::new(vec!["GPU", "Count", "Memory", "Price", "Source", "Avail"]).style(theme.header());

    let title = format!(" Available GPUs ({}) ", user_data.offerings.len());

    let table = Table::new(
        rows,
        [
            Constraint::Percentage(20),
            Constraint::Length(8),
            Constraint::Length(10),
            Constraint::Length(12),
            Constraint::Percentage(20),
            Constraint::Length(8),
        ],
    )
    .header(header_row)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(theme.border_selected())
            .title(Span::styled(title, theme.block_title())),
    )
    .row_highlight_style(theme.selected_row())
    .highlight_symbol("▶ ");

    let mut state = TableState::default();
    if !user_data.offerings.is_empty() {
        state.select(Some(
            ctx.selected_index
                .min(user_data.offerings.len().saturating_sub(1)),
        ));
    }

    frame.render_stateful_widget(table, area, &mut state);
}
