//! Input handling utilities
#![allow(dead_code)]

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

/// Key action that can be performed
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyAction {
    Quit,
    Help,
    ToggleMode,
    NextTab,
    PrevTab,
    GoToTab(usize),
    SelectNext,
    SelectPrev,
    Enter,
    Back,
    Refresh,
    Search,
    // Screen-specific actions
    StartRental,
    StopRental,
    SshConnect,
    ViewLogs,
    AddNode,
    RemoveNode,
}

/// Parse a key event into an action
pub fn parse_key_action(key: KeyEvent) -> Option<KeyAction> {
    match (key.modifiers, key.code) {
        (KeyModifiers::CONTROL, KeyCode::Char('c')) => Some(KeyAction::Quit),
        (_, KeyCode::Char('q')) => Some(KeyAction::Quit),
        (_, KeyCode::Char('?')) => Some(KeyAction::Help),
        (_, KeyCode::Char('m')) => Some(KeyAction::ToggleMode),
        (_, KeyCode::Tab) => Some(KeyAction::NextTab),
        (KeyModifiers::SHIFT, KeyCode::BackTab) => Some(KeyAction::PrevTab),
        (_, KeyCode::Char('j')) | (_, KeyCode::Down) => Some(KeyAction::SelectNext),
        (_, KeyCode::Char('k')) | (_, KeyCode::Up) => Some(KeyAction::SelectPrev),
        (_, KeyCode::Enter) => Some(KeyAction::Enter),
        (_, KeyCode::Esc) => Some(KeyAction::Back),
        (_, KeyCode::Char('r')) => Some(KeyAction::Refresh),
        (_, KeyCode::Char('/')) => Some(KeyAction::Search),
        (_, KeyCode::Char('u')) => Some(KeyAction::StartRental),
        (_, KeyCode::Char('d')) => Some(KeyAction::StopRental),
        (_, KeyCode::Char('s')) => Some(KeyAction::SshConnect),
        (_, KeyCode::Char('l')) => Some(KeyAction::ViewLogs),
        (_, KeyCode::Char('a')) => Some(KeyAction::AddNode),
        (_, KeyCode::Char(c)) if c.is_ascii_digit() => {
            c.to_digit(10).map(|d| KeyAction::GoToTab(d as usize))
        }
        _ => None,
    }
}

/// Format a key binding for display
pub fn format_key(key: &str) -> String {
    format!("[{}]", key)
}
