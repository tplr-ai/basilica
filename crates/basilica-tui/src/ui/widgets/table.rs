//! Enhanced table widget

use ratatui::{
    layout::Constraint,
    style::Style,
    widgets::{Block, Borders, Cell, Row, Table, TableState},
};

use crate::ui::Theme;

/// Create a styled table
pub fn styled_table<'a>(
    headers: Vec<&'a str>,
    rows: Vec<Vec<String>>,
    widths: Vec<Constraint>,
    theme: &Theme,
    title: &'a str,
) -> (Table<'a>, TableState) {
    let header_cells: Vec<Cell> = headers
        .into_iter()
        .map(|h| Cell::from(h).style(theme.header()))
        .collect();

    let header = Row::new(header_cells).height(1);

    let table_rows: Vec<Row> = rows
        .into_iter()
        .map(|row| {
            let cells: Vec<Cell> = row
                .into_iter()
                .map(|c| Cell::from(c).style(theme.text()))
                .collect();
            Row::new(cells).height(1)
        })
        .collect();

    let table = Table::new(table_rows, widths)
        .header(header)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border())
                .title(title)
                .title_style(theme.block_title()),
        )
        .row_highlight_style(theme.selected_row())
        .highlight_symbol("▶ ");

    let state = TableState::default();

    (table, state)
}

/// Status cell with color
pub fn status_cell<'a>(status: &'a str, theme: &Theme) -> Cell<'a> {
    let style = match status.to_lowercase().as_str() {
        "running" | "active" | "healthy" => theme.status_running(),
        "pending" | "starting" | "provisioning" => theme.status_pending(),
        "stopped" | "terminated" | "inactive" => theme.status_stopped(),
        "error" | "failed" | "unhealthy" => theme.status_error(),
        _ => theme.text(),
    };

    Cell::from(format!("● {}", status)).style(style)
}

