//! Footer component with keybinding hints

use ratatui::{
    layout::Rect,
    text::{Line, Span},
    widgets::Paragraph,
    Frame,
};

use crate::app::{App, AppMode, MinerScreen, RenderContext, UserScreen};

/// Render the footer with keybinding hints
pub fn render_footer(frame: &mut Frame, app: &App, area: Rect) {
    let theme = &app.theme;

    // Build context-aware hints
    let mut hints = vec![
        key_hint("?", "Help", theme),
        key_hint("Tab", "Next", theme),
        key_hint("q", "Quit", theme),
        key_hint("m", "Mode", theme),
    ];

    // Add screen-specific hints
    match app.mode {
        AppMode::User => match app.user_screen {
            UserScreen::Dashboard => {
                hints.insert(0, key_hint("u", "New rental", theme));
            }
            UserScreen::Rentals => {
                hints.insert(0, key_hint("s", "SSH", theme));
                hints.insert(1, key_hint("l", "Logs", theme));
                hints.insert(2, key_hint("d", "Stop", theme));
            }
            UserScreen::Marketplace => {
                hints.insert(0, key_hint("Enter", "Rent", theme));
                hints.insert(1, key_hint("/", "Filter", theme));
            }
            UserScreen::Deployments => {
                hints.insert(0, key_hint("l", "Logs", theme));
                hints.insert(1, key_hint("d", "Delete", theme));
            }
            UserScreen::Billing => {
                hints.insert(0, key_hint("r", "Refresh", theme));
            }
        },
        AppMode::Miner => match app.miner_screen {
            MinerScreen::Fleet => {
                hints.insert(0, key_hint("Enter", "Details", theme));
            }
            MinerScreen::Nodes => {
                hints.insert(0, key_hint("a", "Add", theme));
                hints.insert(1, key_hint("d", "Remove", theme));
            }
            _ => {}
        },
    }

    let spans: Vec<Span> = hints
        .into_iter()
        .flat_map(|h| vec![h, Span::raw("  ")])
        .collect();

    let footer = Paragraph::new(Line::from(spans)).style(theme.text_muted());

    frame.render_widget(footer, area);
}

/// Create a key hint span
fn key_hint<'a>(key: &'a str, desc: &'a str, theme: &crate::ui::Theme) -> Span<'a> {
    Span::styled(
        format!("[{}] {}", key, desc),
        theme.text_muted(),
    )
}

/// Render footer with context
pub fn render_footer_ctx(frame: &mut Frame, ctx: &RenderContext, area: Rect) {
    let theme = ctx.theme;

    let mut hints = vec![
        key_hint("?", "Help", theme),
        key_hint("Tab", "Next", theme),
        key_hint("q", "Quit", theme),
        key_hint("m", "Mode", theme),
    ];

    match ctx.mode {
        AppMode::User => match ctx.user_screen {
            UserScreen::Dashboard => {
                hints.insert(0, key_hint("u", "New rental", theme));
            }
            UserScreen::Rentals => {
                hints.insert(0, key_hint("s", "SSH", theme));
                hints.insert(1, key_hint("l", "Logs", theme));
                hints.insert(2, key_hint("d", "Stop", theme));
            }
            UserScreen::Marketplace => {
                hints.insert(0, key_hint("Enter", "Rent", theme));
                hints.insert(1, key_hint("/", "Filter", theme));
            }
            UserScreen::Deployments => {
                hints.insert(0, key_hint("l", "Logs", theme));
                hints.insert(1, key_hint("d", "Delete", theme));
            }
            UserScreen::Billing => {
                hints.insert(0, key_hint("r", "Refresh", theme));
            }
        },
        AppMode::Miner => match ctx.miner_screen {
            MinerScreen::Fleet => {
                hints.insert(0, key_hint("Enter", "Details", theme));
            }
            MinerScreen::Nodes => {
                hints.insert(0, key_hint("a", "Add", theme));
                hints.insert(1, key_hint("d", "Remove", theme));
            }
            _ => {}
        },
    }

    let spans: Vec<Span> = hints
        .into_iter()
        .flat_map(|h| vec![h, Span::raw("  ")])
        .collect();

    let footer = Paragraph::new(Line::from(spans)).style(theme.text_muted());

    frame.render_widget(footer, area);
}

/// Get the height of the footer
pub fn footer_height() -> u16 {
    1
}

