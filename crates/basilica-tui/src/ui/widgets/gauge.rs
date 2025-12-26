//! Gauge widget for displaying percentages

use ratatui::{
    layout::Rect,
    style::Style,
    symbols,
    widgets::{Block, Borders, Gauge},
    Frame,
};

use crate::ui::Theme;

/// Render a colored gauge based on percentage
pub fn render_gauge(
    frame: &mut Frame,
    area: Rect,
    title: &str,
    percent: f64,
    theme: &Theme,
) {
    let percent_u16 = (percent.clamp(0.0, 100.0) as u16).min(100);

    // Color based on value
    let gauge_style = if percent < 50.0 {
        theme.gauge_low()
    } else if percent < 80.0 {
        theme.gauge_medium()
    } else {
        theme.gauge_high()
    };

    let gauge = Gauge::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border())
                .title(title)
                .title_style(theme.block_title()),
        )
        .gauge_style(gauge_style)
        .percent(percent_u16)
        .label(format!("{:.1}%", percent));

    frame.render_widget(gauge, area);
}

/// Render a mini gauge (single line, no border)
pub fn render_mini_gauge(
    frame: &mut Frame,
    area: Rect,
    percent: f64,
    theme: &Theme,
) {
    let percent_u16 = (percent.clamp(0.0, 100.0) as u16).min(100);

    let gauge_style = if percent < 50.0 {
        theme.gauge_low()
    } else if percent < 80.0 {
        theme.gauge_medium()
    } else {
        theme.gauge_high()
    };

    let gauge = Gauge::default()
        .gauge_style(gauge_style)
        .percent(percent_u16)
        .label(format!("{:3}%", percent as u8));

    frame.render_widget(gauge, area);
}

/// Create a text-based mini bar (for compact displays)
pub fn mini_bar(percent: f64, width: usize) -> String {
    let filled = ((percent / 100.0) * width as f64) as usize;
    let empty = width.saturating_sub(filled);

    format!(
        "{}{}",
        "█".repeat(filled),
        "░".repeat(empty)
    )
}

