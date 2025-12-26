//! Theme and color definitions
#![allow(dead_code)]

use ratatui::style::{Color, Modifier, Style};

/// Application theme
#[derive(Debug, Clone)]
pub struct Theme {
    /// Background color
    pub bg: Color,
    /// Primary foreground color
    pub fg: Color,
    /// Muted/secondary text color
    pub fg_muted: Color,
    /// Accent color (highlights, active items)
    pub accent: Color,
    /// Secondary accent
    pub accent_secondary: Color,
    /// Success color
    pub success: Color,
    /// Warning color
    pub warning: Color,
    /// Error color
    pub error: Color,
    /// Info color
    pub info: Color,
    /// Border color
    pub border: Color,
    /// Selected item background
    pub selection_bg: Color,
    /// Header background
    pub header_bg: Color,
}

impl Theme {
    /// Create a dark theme (default)
    pub fn dark() -> Self {
        Self {
            bg: Color::Rgb(17, 17, 27),                  // Deep dark blue-black
            fg: Color::Rgb(205, 214, 244),               // Light lavender
            fg_muted: Color::Rgb(108, 112, 134),         // Muted gray
            accent: Color::Rgb(137, 180, 250),           // Blue accent
            accent_secondary: Color::Rgb(203, 166, 247), // Mauve
            success: Color::Rgb(166, 227, 161),          // Green
            warning: Color::Rgb(249, 226, 175),          // Yellow/peach
            error: Color::Rgb(243, 139, 168),            // Red/pink
            info: Color::Rgb(148, 226, 213),             // Teal
            border: Color::Rgb(69, 71, 90),              // Surface border
            selection_bg: Color::Rgb(49, 50, 68),        // Surface selection
            header_bg: Color::Rgb(30, 30, 46),           // Slightly lighter bg
        }
    }

    /// Create a light theme
    pub fn light() -> Self {
        Self {
            bg: Color::Rgb(239, 241, 245),              // Light gray
            fg: Color::Rgb(76, 79, 105),                // Dark text
            fg_muted: Color::Rgb(140, 143, 161),        // Muted text
            accent: Color::Rgb(30, 102, 245),           // Blue
            accent_secondary: Color::Rgb(136, 57, 239), // Purple
            success: Color::Rgb(64, 160, 43),           // Green
            warning: Color::Rgb(223, 142, 29),          // Orange
            error: Color::Rgb(210, 15, 57),             // Red
            info: Color::Rgb(23, 146, 153),             // Teal
            border: Color::Rgb(172, 176, 190),          // Border
            selection_bg: Color::Rgb(204, 208, 218),    // Selection
            header_bg: Color::Rgb(220, 224, 232),       // Header
        }
    }

    // Style helpers

    /// Default text style
    pub fn text(&self) -> Style {
        Style::default().fg(self.fg).bg(self.bg)
    }

    /// Muted text style
    pub fn text_muted(&self) -> Style {
        Style::default().fg(self.fg_muted).bg(self.bg)
    }

    /// Bold text style
    pub fn text_bold(&self) -> Style {
        Style::default()
            .fg(self.fg)
            .bg(self.bg)
            .add_modifier(Modifier::BOLD)
    }

    /// Accent text style
    pub fn text_accent(&self) -> Style {
        Style::default().fg(self.accent).bg(self.bg)
    }

    /// Success text style
    pub fn text_success(&self) -> Style {
        Style::default().fg(self.success).bg(self.bg)
    }

    /// Warning text style
    pub fn text_warning(&self) -> Style {
        Style::default().fg(self.warning).bg(self.bg)
    }

    /// Error text style
    pub fn text_error(&self) -> Style {
        Style::default().fg(self.error).bg(self.bg)
    }

    /// Info text style
    pub fn text_info(&self) -> Style {
        Style::default().fg(self.info).bg(self.bg)
    }

    /// Block title style
    pub fn block_title(&self) -> Style {
        Style::default()
            .fg(self.accent)
            .add_modifier(Modifier::BOLD)
    }

    /// Border style
    pub fn border(&self) -> Style {
        Style::default().fg(self.border)
    }

    /// Selected border style
    pub fn border_selected(&self) -> Style {
        Style::default().fg(self.accent)
    }

    /// Selected row style
    pub fn selected_row(&self) -> Style {
        Style::default()
            .bg(self.selection_bg)
            .add_modifier(Modifier::BOLD)
    }

    /// Header style
    pub fn header(&self) -> Style {
        Style::default()
            .fg(self.fg)
            .bg(self.header_bg)
            .add_modifier(Modifier::BOLD)
    }

    /// Tab active style
    pub fn tab_active(&self) -> Style {
        Style::default()
            .fg(self.accent)
            .bg(self.bg)
            .add_modifier(Modifier::BOLD)
    }

    /// Tab inactive style
    pub fn tab_inactive(&self) -> Style {
        Style::default().fg(self.fg_muted).bg(self.bg)
    }

    /// Status indicator styles
    pub fn status_running(&self) -> Style {
        Style::default().fg(self.success)
    }

    pub fn status_pending(&self) -> Style {
        Style::default().fg(self.warning)
    }

    pub fn status_stopped(&self) -> Style {
        Style::default().fg(self.fg_muted)
    }

    pub fn status_error(&self) -> Style {
        Style::default().fg(self.error)
    }

    /// Gauge styles
    pub fn gauge_low(&self) -> Style {
        Style::default().fg(self.success)
    }

    pub fn gauge_medium(&self) -> Style {
        Style::default().fg(self.warning)
    }

    pub fn gauge_high(&self) -> Style {
        Style::default().fg(self.error)
    }
}

impl Default for Theme {
    fn default() -> Self {
        Self::dark()
    }
}
