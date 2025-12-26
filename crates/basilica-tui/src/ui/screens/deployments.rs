//! Deployments screen - manage application deployments

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    text::Span,
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
            Constraint::Min(0),
            Constraint::Length(footer::footer_height()),
        ])
        .split(area);

    header::render_header_ctx(frame, ctx, chunks[0]);
    footer::render_footer_ctx(frame, ctx, chunks[2]);

    let content = chunks[1];

    if ctx.screens.deployments.show_logs {
        let split = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(content);

        render_deployments_table(frame, ctx, split[0]);
        render_deployment_logs(frame, ctx, split[1]);
    } else {
        render_deployments_table(frame, ctx, content);
    }
}

fn render_deployments_table(frame: &mut Frame, ctx: &RenderContext, area: Rect) {
    let theme = ctx.theme;
    let user_data = ctx.user_data;

    let rows: Vec<Row> = if user_data.deployments.is_empty() {
        vec![Row::new(vec![Cell::from(
            "No deployments. Create one with 'basilica deploy'.",
        )])
        .style(theme.text_muted())]
    } else {
        user_data
            .deployments
            .iter()
            .enumerate()
            .map(|(i, deploy)| {
                let status_style = match deploy.status.as_str() {
                    "Running" | "Active" => theme.status_running(),
                    "Scaling" | "Pending" => theme.status_pending(),
                    "Failed" | "Error" => theme.status_error(),
                    _ => theme.text(),
                };

                let row_style = if i == ctx.selected_index {
                    theme.selected_row()
                } else {
                    theme.text()
                };

                let replicas = format!("{}/{}", deploy.replicas_ready, deploy.replicas_desired);
                let gpu = if deploy.gpu_count > 0 {
                    format!("{} x {}", deploy.gpu_type, deploy.gpu_count)
                } else {
                    "CPU".to_string()
                };

                Row::new(vec![
                    Cell::from(deploy.name.clone()).style(theme.text_accent()),
                    Cell::from(deploy.deployment_type.clone()),
                    Cell::from(format!("● {}", deploy.status)).style(status_style),
                    Cell::from(replicas),
                    Cell::from(gpu),
                    Cell::from(deploy.url.clone().unwrap_or_else(|| "N/A".to_string()))
                        .style(theme.text_muted()),
                ])
                .style(row_style)
            })
            .collect()
    };

    let header_row =
        Row::new(vec!["Name", "Type", "Status", "Replicas", "GPU", "URL"]).style(theme.header());

    let title = format!(" Deployments ({}) ", user_data.deployments.len());

    let table = Table::new(
        rows,
        [
            Constraint::Length(15),
            Constraint::Length(10),
            Constraint::Length(14),
            Constraint::Length(10),
            Constraint::Length(14),
            Constraint::Min(20),
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
    if !user_data.deployments.is_empty() {
        state.select(Some(
            ctx.selected_index
                .min(user_data.deployments.len().saturating_sub(1)),
        ));
    }

    frame.render_stateful_widget(table, area, &mut state);
}

fn render_deployment_logs(frame: &mut Frame, ctx: &RenderContext, area: Rect) {
    let theme = ctx.theme;
    let user_data = ctx.user_data;

    let logs = if let Some(deploy) = user_data.deployments.get(ctx.selected_index) {
        format!(
            "Logs for deployment: {}\n\n\
             [Streaming logs not yet connected]\n\n\
             URL: {}\n\
             Status: {}\n\
             Replicas: {}/{}\n\n\
             Press 'l' to toggle logs panel",
            deploy.name,
            deploy.url.as_deref().unwrap_or("N/A"),
            deploy.status,
            deploy.replicas_ready,
            deploy.replicas_desired
        )
    } else {
        "No deployment selected".to_string()
    };

    let paragraph = Paragraph::new(logs)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border())
                .title(Span::styled(" Deployment Logs ", theme.block_title())),
        )
        .style(theme.text())
        .wrap(Wrap { trim: false });

    frame.render_widget(paragraph, area);
}
