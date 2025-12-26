//! Notification toast component

use ratatui::{
    layout::{Alignment, Rect},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph},
    Frame,
};

use crate::app::{App, NotificationLevel, RenderContext};

/// Render notifications as toasts in the top-right corner
pub fn render_notifications(frame: &mut Frame, app: &App) {
    if app.notifications.is_empty() {
        return;
    }

    let theme = &app.theme;
    let area = frame.area();

    // Position toasts in top-right corner
    let toast_width = 40u16;
    let toast_height = 3u16;
    let margin = 2u16;

    let max_toasts = 3;
    let notifications: Vec<_> = app.notifications.iter().rev().take(max_toasts).collect();

    for (i, notification) in notifications.iter().enumerate() {
        let y = margin + (i as u16 * (toast_height + 1));
        let x = area.width.saturating_sub(toast_width + margin);

        if y + toast_height > area.height {
            break;
        }

        let toast_area = Rect::new(x, y, toast_width, toast_height);

        // Clear the area
        frame.render_widget(Clear, toast_area);

        // Determine style based on level
        let (icon, border_style) = match notification.level {
            NotificationLevel::Info => ("ℹ", theme.text_info()),
            NotificationLevel::Success => ("✓", theme.text_success()),
            NotificationLevel::Warning => ("⚠", theme.text_warning()),
            NotificationLevel::Error => ("✗", theme.text_error()),
        };

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(border_style)
            .style(theme.text());

        let content = Line::from(vec![
            Span::styled(format!("{} ", icon), border_style),
            Span::raw(&notification.message),
        ]);

        let paragraph = Paragraph::new(content)
            .block(block)
            .alignment(Alignment::Left);

        frame.render_widget(paragraph, toast_area);
    }
}

/// Render notifications with context
pub fn render_notifications_ctx(frame: &mut Frame, ctx: &RenderContext) {
    if ctx.notifications.is_empty() {
        return;
    }

    let theme = ctx.theme;
    let area = frame.area();

    let toast_width = 40u16;
    let toast_height = 3u16;
    let margin = 2u16;

    let max_toasts = 3;
    let notifications: Vec<_> = ctx.notifications.iter().rev().take(max_toasts).collect();

    for (i, notification) in notifications.iter().enumerate() {
        let y = margin + (i as u16 * (toast_height + 1));
        let x = area.width.saturating_sub(toast_width + margin);

        if y + toast_height > area.height {
            break;
        }

        let toast_area = Rect::new(x, y, toast_width, toast_height);
        frame.render_widget(Clear, toast_area);

        let (icon, border_style) = match notification.level {
            NotificationLevel::Info => ("ℹ", theme.text_info()),
            NotificationLevel::Success => ("✓", theme.text_success()),
            NotificationLevel::Warning => ("⚠", theme.text_warning()),
            NotificationLevel::Error => ("✗", theme.text_error()),
        };

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(border_style)
            .style(theme.text());

        let content = Line::from(vec![
            Span::styled(format!("{} ", icon), border_style),
            Span::raw(&notification.message),
        ]);

        let paragraph = Paragraph::new(content)
            .block(block)
            .alignment(Alignment::Left);

        frame.render_widget(paragraph, toast_area);
    }
}
