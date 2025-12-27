//! Settings screen - auth, tokens, SSH keys, theme management

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::Modifier,
    text::{Line, Span},
    widgets::{Block, Borders, Cell, List, ListItem, Paragraph, Row, Table, Tabs},
    Frame,
};

use crate::app::RenderContext;
use crate::ui::components::{footer, header};

/// Settings subsection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SettingsSection {
    #[default]
    Auth,
    Tokens,
    SshKeys,
    Theme,
}

impl SettingsSection {
    pub fn all() -> &'static [SettingsSection] {
        &[
            SettingsSection::Auth,
            SettingsSection::Tokens,
            SettingsSection::SshKeys,
            SettingsSection::Theme,
        ]
    }

    pub fn label(&self) -> &'static str {
        match self {
            SettingsSection::Auth => "Auth",
            SettingsSection::Tokens => "API Tokens",
            SettingsSection::SshKeys => "SSH Keys",
            SettingsSection::Theme => "Theme",
        }
    }

    pub fn next(&self) -> Self {
        match self {
            SettingsSection::Auth => SettingsSection::Tokens,
            SettingsSection::Tokens => SettingsSection::SshKeys,
            SettingsSection::SshKeys => SettingsSection::Theme,
            SettingsSection::Theme => SettingsSection::Auth,
        }
    }

    pub fn prev(&self) -> Self {
        match self {
            SettingsSection::Auth => SettingsSection::Theme,
            SettingsSection::Tokens => SettingsSection::Auth,
            SettingsSection::SshKeys => SettingsSection::Tokens,
            SettingsSection::Theme => SettingsSection::SshKeys,
        }
    }
}

/// Settings screen state
#[derive(Debug, Clone, Default)]
pub struct SettingsState {
    pub section: SettingsSection,
    pub selected_token: usize,
    pub selected_ssh_key: usize,
    pub tokens: Vec<TokenInfo>,
    pub ssh_keys: Vec<SshKeyInfo>,
    pub auth_status: AuthStatus,
}

#[derive(Debug, Clone, Default)]
pub struct AuthStatus {
    pub logged_in: bool,
    pub user_email: Option<String>,
    pub token_expiry: Option<String>,
}

#[derive(Debug, Clone)]
pub struct TokenInfo {
    pub name: String,
    pub created_at: String,
    pub last_used: Option<String>,
    pub prefix: String,
}

#[derive(Debug, Clone)]
pub struct SshKeyInfo {
    pub name: String,
    pub fingerprint: String,
    pub added_at: String,
}

/// Render with context
pub fn render_with_ctx(frame: &mut Frame, ctx: &RenderContext) {
    let area = frame.area();

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(header::header_height()),
            Constraint::Length(3), // Tabs
            Constraint::Min(0),
            Constraint::Length(footer::footer_height()),
        ])
        .split(area);

    header::render_header_ctx(frame, ctx, chunks[0]);
    render_section_tabs(frame, ctx, chunks[1]);
    render_section_content(frame, ctx, chunks[2]);
    render_settings_footer(frame, ctx, chunks[3]);
}

fn render_section_tabs(frame: &mut Frame, ctx: &RenderContext, area: Rect) {
    let theme = ctx.theme;
    let section = ctx.settings_state.section;

    let titles: Vec<Line> = SettingsSection::all()
        .iter()
        .map(|s| {
            let style = if *s == section {
                theme.tab_selected()
            } else {
                theme.tab_inactive()
            };
            Line::from(Span::styled(s.label(), style))
        })
        .collect();

    let tabs = Tabs::new(titles)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border())
                .title(Span::styled(" Settings ", theme.block_title())),
        )
        .select(section as usize)
        .style(theme.text())
        .highlight_style(theme.tab_selected());

    frame.render_widget(tabs, area);
}

fn render_section_content(frame: &mut Frame, ctx: &RenderContext, area: Rect) {
    let section = ctx.settings_state.section;

    match section {
        SettingsSection::Auth => render_auth_section(frame, ctx, area),
        SettingsSection::Tokens => render_tokens_section(frame, ctx, area),
        SettingsSection::SshKeys => render_ssh_keys_section(frame, ctx, area),
        SettingsSection::Theme => render_theme_section(frame, ctx, area),
    }
}

fn render_auth_section(frame: &mut Frame, ctx: &RenderContext, area: Rect) {
    let theme = ctx.theme;
    let auth = &ctx.settings_state.auth_status;

    let content = if auth.logged_in {
        vec![
            Line::from(""),
            Line::from(vec![
                Span::styled("Status: ", theme.text_muted()),
                Span::styled("● Logged In", theme.text_success().add_modifier(Modifier::BOLD)),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("Email: ", theme.text_muted()),
                Span::styled(
                    auth.user_email.clone().unwrap_or_else(|| "Unknown".to_string()),
                    theme.text(),
                ),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("Token Expires: ", theme.text_muted()),
                Span::styled(
                    auth.token_expiry.clone().unwrap_or_else(|| "Unknown".to_string()),
                    theme.text(),
                ),
            ]),
            Line::from(""),
            Line::from(""),
            Line::from(Span::styled(
                "Press [o] to logout",
                theme.text_muted(),
            )),
        ]
    } else {
        vec![
            Line::from(""),
            Line::from(vec![
                Span::styled("Status: ", theme.text_muted()),
                Span::styled("● Not Logged In", theme.text_error().add_modifier(Modifier::BOLD)),
            ]),
            Line::from(""),
            Line::from(""),
            Line::from(Span::styled(
                "Press [l] to login via browser",
                theme.text_accent(),
            )),
            Line::from(""),
            Line::from(Span::styled(
                "Press [d] for device code flow (headless)",
                theme.text_muted(),
            )),
        ]
    };

    let paragraph = Paragraph::new(content)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border())
                .title(Span::styled(" 🔐 Authentication ", theme.block_title())),
        )
        .style(theme.text());

    frame.render_widget(paragraph, area);
}

fn render_tokens_section(frame: &mut Frame, ctx: &RenderContext, area: Rect) {
    let theme = ctx.theme;
    let state = &ctx.settings_state;

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(3)])
        .split(area);

    // Tokens table
    let rows: Vec<Row> = if state.tokens.is_empty() {
        vec![Row::new(vec![Cell::from("No API tokens. Press [a] to create one.")])
            .style(theme.text_muted())]
    } else {
        state
            .tokens
            .iter()
            .enumerate()
            .map(|(i, token)| {
                let row_style = if i == state.selected_token {
                    theme.selected_row()
                } else {
                    theme.text()
                };

                Row::new(vec![
                    Cell::from(token.name.clone()),
                    Cell::from(format!("{}...", token.prefix)),
                    Cell::from(token.created_at.clone()),
                    Cell::from(token.last_used.clone().unwrap_or_else(|| "Never".to_string())),
                ])
                .style(row_style)
            })
            .collect()
    };

    let header = Row::new(vec!["Name", "Prefix", "Created", "Last Used"]).style(theme.header());

    let table = Table::new(
        rows,
        [
            Constraint::Percentage(30),
            Constraint::Percentage(20),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
        ],
    )
    .header(header)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(theme.border())
            .title(Span::styled(
                format!(" 🔑 API Tokens ({}) ", state.tokens.len()),
                theme.block_title(),
            )),
    )
    .row_highlight_style(theme.selected_row())
    .highlight_symbol("▶ ");

    frame.render_widget(table, chunks[0]);

    // Actions bar
    let actions = Paragraph::new(Line::from(vec![
        Span::styled(" [a] ", theme.keybind()),
        Span::styled("Create  ", theme.text_muted()),
        Span::styled("[d] ", theme.keybind()),
        Span::styled("Revoke  ", theme.text_muted()),
        Span::styled("[Enter] ", theme.keybind()),
        Span::styled("Copy Token", theme.text_muted()),
    ]))
    .block(Block::default().borders(Borders::ALL).border_style(theme.border()));

    frame.render_widget(actions, chunks[1]);
}

fn render_ssh_keys_section(frame: &mut Frame, ctx: &RenderContext, area: Rect) {
    let theme = ctx.theme;
    let state = &ctx.settings_state;

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(3)])
        .split(area);

    // SSH keys table
    let rows: Vec<Row> = if state.ssh_keys.is_empty() {
        vec![Row::new(vec![Cell::from(
            "No SSH keys registered. Press [a] to add one.",
        )])
        .style(theme.text_muted())]
    } else {
        state
            .ssh_keys
            .iter()
            .enumerate()
            .map(|(i, key)| {
                let row_style = if i == state.selected_ssh_key {
                    theme.selected_row()
                } else {
                    theme.text()
                };

                Row::new(vec![
                    Cell::from(key.name.clone()),
                    Cell::from(key.fingerprint.clone()),
                    Cell::from(key.added_at.clone()),
                ])
                .style(row_style)
            })
            .collect()
    };

    let header = Row::new(vec!["Name", "Fingerprint", "Added"]).style(theme.header());

    let table = Table::new(
        rows,
        [
            Constraint::Percentage(30),
            Constraint::Percentage(45),
            Constraint::Percentage(25),
        ],
    )
    .header(header)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(theme.border())
            .title(Span::styled(
                format!(" 🔐 SSH Keys ({}) ", state.ssh_keys.len()),
                theme.block_title(),
            )),
    )
    .row_highlight_style(theme.selected_row())
    .highlight_symbol("▶ ");

    frame.render_widget(table, chunks[0]);

    // Actions bar
    let actions = Paragraph::new(Line::from(vec![
        Span::styled(" [a] ", theme.keybind()),
        Span::styled("Add Key  ", theme.text_muted()),
        Span::styled("[d] ", theme.keybind()),
        Span::styled("Delete  ", theme.text_muted()),
    ]))
    .block(Block::default().borders(Borders::ALL).border_style(theme.border()));

    frame.render_widget(actions, chunks[1]);
}

fn render_theme_section(frame: &mut Frame, ctx: &RenderContext, area: Rect) {
    let theme = ctx.theme;

    let themes = [
        ("Dark", "Default dark theme", true),
        ("Light", "Light theme (coming soon)", false),
        ("Dracula", "Dracula color scheme (coming soon)", false),
        ("Nord", "Nord color scheme (coming soon)", false),
    ];

    let items: Vec<ListItem> = themes
        .iter()
        .map(|(name, desc, selected)| {
            let indicator = if *selected { "● " } else { "○ " };
            let style = if *selected {
                theme.text_accent().add_modifier(Modifier::BOLD)
            } else {
                theme.text()
            };
            ListItem::new(Line::from(vec![
                Span::styled(indicator, style),
                Span::styled(*name, style),
                Span::styled(format!(" - {}", desc), theme.text_muted()),
            ]))
        })
        .collect();

    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border())
                .title(Span::styled(" 🎨 Theme ", theme.block_title())),
        )
        .style(theme.text());

    frame.render_widget(list, area);
}

fn render_settings_footer(frame: &mut Frame, ctx: &RenderContext, area: Rect) {
    let theme = ctx.theme;

    let content = Line::from(vec![
        Span::styled(" [Tab] ", theme.keybind()),
        Span::styled("Switch Section  ", theme.text_muted()),
        Span::styled("[j/k] ", theme.keybind()),
        Span::styled("Navigate  ", theme.text_muted()),
        Span::styled("[?] ", theme.keybind()),
        Span::styled("Help  ", theme.text_muted()),
        Span::styled("[q] ", theme.keybind()),
        Span::styled("Quit", theme.text_muted()),
    ]);

    let paragraph = Paragraph::new(content)
        .block(Block::default().borders(Borders::TOP).border_style(theme.border()));

    frame.render_widget(paragraph, area);
}

