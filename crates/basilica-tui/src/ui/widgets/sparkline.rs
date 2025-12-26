//! Sparkline widget for inline charts

use ratatui::{
    layout::Rect,
    style::Style,
    symbols,
    widgets::{Block, Borders, Sparkline as RatatuiSparkline},
    Frame,
};

use crate::ui::Theme;

/// Render a sparkline chart
pub fn render_sparkline(
    frame: &mut Frame,
    area: Rect,
    title: &str,
    data: &[u64],
    theme: &Theme,
) {
    let sparkline = RatatuiSparkline::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border())
                .title(title)
                .title_style(theme.block_title()),
        )
        .data(data)
        .style(theme.text_accent());

    frame.render_widget(sparkline, area);
}

/// Render an inline sparkline (no border)
pub fn render_inline_sparkline(
    frame: &mut Frame,
    area: Rect,
    data: &[u64],
    theme: &Theme,
) {
    let sparkline = RatatuiSparkline::default()
        .data(data)
        .style(theme.text_accent());

    frame.render_widget(sparkline, area);
}

/// Create a text-based mini sparkline (for very compact displays)
pub fn text_sparkline(data: &[f64]) -> String {
    const CHARS: &[char] = &['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

    if data.is_empty() {
        return String::new();
    }

    let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;

    if range == 0.0 {
        return CHARS[4].to_string().repeat(data.len());
    }

    data.iter()
        .map(|&v| {
            let normalized = (v - min) / range;
            let index = ((normalized * 7.0) as usize).min(7);
            CHARS[index]
        })
        .collect()
}

