//! Header component with tabs

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Tabs},
    Frame,
};

use crate::app::{App, AppMode, MinerScreen, RenderContext, UserScreen};

/// Render the header with tabs
pub fn render_header(frame: &mut Frame, app: &App, area: Rect) {
    let theme = &app.theme;

    // Split header into tabs and mode indicator
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(0), Constraint::Length(16)])
        .split(area);

    // Render tabs based on mode
    let (titles, selected): (Vec<&str>, usize) = match app.mode {
        AppMode::User => (
            UserScreen::all().iter().map(|s| s.label()).collect(),
            app.user_screen.index(),
        ),
        AppMode::Miner => (
            MinerScreen::all().iter().map(|s| s.label()).collect(),
            app.miner_screen.index(),
        ),
    };

    let tabs: Vec<Line> = titles
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let style = if i == selected {
                theme.tab_active()
            } else {
                theme.tab_inactive()
            };
            Line::from(Span::styled(format!(" {} ", t), style))
        })
        .collect();

    let tabs_widget = Tabs::new(tabs)
        .block(
            Block::default()
                .borders(Borders::BOTTOM)
                .border_style(theme.border())
                .title(Span::styled(
                    " ⛪ Basilica ",
                    Style::default()
                        .fg(theme.accent)
                        .add_modifier(Modifier::BOLD),
                )),
        )
        .select(selected)
        .divider(Span::raw("│"));

    frame.render_widget(tabs_widget, chunks[0]);

    // Render mode indicator
    let mode_label = match app.mode {
        AppMode::User => "User",
        AppMode::Miner => "Miner",
    };

    let mode_block = Block::default()
        .borders(Borders::BOTTOM | Borders::LEFT)
        .border_style(theme.border())
        .title(Span::styled(
            format!(" {} ", mode_label),
            theme.text_accent(),
        ));

    frame.render_widget(mode_block, chunks[1]);
}

/// Render header with context
pub fn render_header_ctx(frame: &mut Frame, ctx: &RenderContext, area: Rect) {
    let theme = ctx.theme;

    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(0), Constraint::Length(16)])
        .split(area);

    let (titles, selected): (Vec<&str>, usize) = match ctx.mode {
        AppMode::User => (
            UserScreen::all().iter().map(|s| s.label()).collect(),
            ctx.user_screen.index(),
        ),
        AppMode::Miner => (
            MinerScreen::all().iter().map(|s| s.label()).collect(),
            ctx.miner_screen.index(),
        ),
    };

    let tabs: Vec<Line> = titles
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let style = if i == selected {
                theme.tab_active()
            } else {
                theme.tab_inactive()
            };
            Line::from(Span::styled(format!(" {} ", t), style))
        })
        .collect();

    let tabs_widget = Tabs::new(tabs)
        .block(
            Block::default()
                .borders(Borders::BOTTOM)
                .border_style(theme.border())
                .title(Span::styled(
                    " ⛪ Basilica ",
                    Style::default()
                        .fg(theme.accent)
                        .add_modifier(Modifier::BOLD),
                )),
        )
        .select(selected)
        .divider(Span::raw("│"));

    frame.render_widget(tabs_widget, chunks[0]);

    let mode_label = match ctx.mode {
        AppMode::User => "User",
        AppMode::Miner => "Miner",
    };

    let mode_block = Block::default()
        .borders(Borders::BOTTOM | Borders::LEFT)
        .border_style(theme.border())
        .title(Span::styled(
            format!(" {} ", mode_label),
            theme.text_accent(),
        ));

    frame.render_widget(mode_block, chunks[1]);
}

/// Get the height of the header
pub fn header_height() -> u16 {
    2
}

