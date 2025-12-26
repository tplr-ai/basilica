//! Help overlay component

use ratatui::{
    layout::{Alignment, Constraint, Flex, Layout, Rect},
    style::Modifier,
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph, Wrap},
    Frame,
};

use crate::app::{App, AppMode, RenderContext};

/// Render the help overlay
pub fn render_help_overlay(frame: &mut Frame, app: &App) {
    let theme = &app.theme;
    let area = frame.area();

    // Center the help popup
    let popup_area = centered_rect(60, 70, area);

    // Clear the area behind the popup
    frame.render_widget(Clear, popup_area);

    let title = match app.mode {
        AppMode::User => " Help - User Mode ",
        AppMode::Miner => " Help - Miner Mode ",
    };

    let block = Block::default()
        .title(Span::styled(title, theme.block_title()))
        .title_alignment(Alignment::Center)
        .borders(Borders::ALL)
        .border_style(theme.border_selected())
        .style(theme.text());

    // Build help content
    let mut lines = vec![
        Line::from(Span::styled("Global Keybindings", theme.text_bold())),
        Line::from(""),
        help_line("q, Ctrl+c", "Quit application"),
        help_line("?", "Toggle this help"),
        help_line("m", "Switch User/Miner mode"),
        help_line("Tab", "Next screen"),
        help_line("Shift+Tab", "Previous screen"),
        help_line("1-5", "Go to screen by number"),
        help_line("j/↓", "Select next item"),
        help_line("k/↑", "Select previous item"),
        help_line("Enter", "Confirm/activate"),
        help_line("Esc", "Cancel/back"),
        help_line("r", "Refresh data"),
        Line::from(""),
    ];

    // Add mode-specific help
    match app.mode {
        AppMode::User => {
            lines.extend(vec![
                Line::from(Span::styled("User Mode Actions", theme.text_bold())),
                Line::from(""),
                help_line("u", "Start new rental"),
                help_line("s", "SSH into rental"),
                help_line("l", "View logs"),
                help_line("d", "Stop rental / Delete deployment"),
                help_line("/", "Filter/search"),
            ]);
        }
        AppMode::Miner => {
            lines.extend(vec![
                Line::from(Span::styled("Miner Mode Actions", theme.text_bold())),
                Line::from(""),
                help_line("a", "Add new node"),
                help_line("d", "Remove node"),
                help_line("Enter", "View node details"),
            ]);
        }
    }

    lines.extend(vec![
        Line::from(""),
        Line::from(Span::styled(
            "Press ? or Esc to close",
            theme.text_muted(),
        )),
    ]);

    let paragraph = Paragraph::new(lines)
        .block(block)
        .alignment(Alignment::Left)
        .wrap(Wrap { trim: true });

    frame.render_widget(paragraph, popup_area);
}

/// Create a help line with key and description
fn help_line<'a>(key: &'a str, desc: &'a str) -> Line<'a> {
    Line::from(vec![
        Span::styled(
            format!("{:>12}", key),
            ratatui::style::Style::default().add_modifier(Modifier::BOLD),
        ),
        Span::raw("  "),
        Span::raw(desc),
    ])
}

/// Create a centered rect with percentage of parent
fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::vertical([Constraint::Percentage(percent_y)])
        .flex(Flex::Center)
        .split(r);

    Layout::horizontal([Constraint::Percentage(percent_x)])
        .flex(Flex::Center)
        .split(popup_layout[0])[0]
}

/// Render help overlay with context
pub fn render_help_overlay_ctx(frame: &mut Frame, ctx: &RenderContext) {
    let theme = ctx.theme;
    let area = frame.area();

    let popup_area = centered_rect(60, 70, area);
    frame.render_widget(Clear, popup_area);

    let title = match ctx.mode {
        AppMode::User => " Help - User Mode ",
        AppMode::Miner => " Help - Miner Mode ",
    };

    let block = Block::default()
        .title(Span::styled(title, theme.block_title()))
        .title_alignment(Alignment::Center)
        .borders(Borders::ALL)
        .border_style(theme.border_selected())
        .style(theme.text());

    let mut lines = vec![
        Line::from(Span::styled("Global Keybindings", theme.text_bold())),
        Line::from(""),
        help_line("q, Ctrl+c", "Quit application"),
        help_line("?", "Toggle this help"),
        help_line("m", "Switch User/Miner mode"),
        help_line("Tab", "Next screen"),
        help_line("Shift+Tab", "Previous screen"),
        help_line("1-5", "Go to screen by number"),
        help_line("j/↓", "Select next item"),
        help_line("k/↑", "Select previous item"),
        help_line("Enter", "Confirm/activate"),
        help_line("Esc", "Cancel/back"),
        help_line("r", "Refresh data"),
        Line::from(""),
    ];

    match ctx.mode {
        AppMode::User => {
            lines.extend(vec![
                Line::from(Span::styled("User Mode Actions", theme.text_bold())),
                Line::from(""),
                help_line("u", "Start new rental"),
                help_line("s", "SSH into rental"),
                help_line("l", "View logs"),
                help_line("d", "Stop rental / Delete deployment"),
                help_line("/", "Filter/search"),
            ]);
        }
        AppMode::Miner => {
            lines.extend(vec![
                Line::from(Span::styled("Miner Mode Actions", theme.text_bold())),
                Line::from(""),
                help_line("a", "Add new node"),
                help_line("d", "Remove node"),
                help_line("Enter", "View node details"),
            ]);
        }
    }

    lines.extend(vec![
        Line::from(""),
        Line::from(Span::styled(
            "Press ? or Esc to close",
            theme.text_muted(),
        )),
    ]);

    let paragraph = Paragraph::new(lines)
        .block(block)
        .alignment(Alignment::Left)
        .wrap(Wrap { trim: true });

    frame.render_widget(paragraph, popup_area);
}

