//! Reusable dialog/popup components for forms and confirmations

use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::Modifier,
    text::{Line, Span},
    widgets::{Block, Borders, Clear, List, ListItem, Paragraph, Wrap},
    Frame,
};

use crate::ui::Theme;

/// Dialog type
#[derive(Debug, Clone)]
pub enum DialogKind {
    /// Confirmation dialog (Yes/No)
    Confirm {
        title: String,
        message: String,
        confirm_label: String,
        cancel_label: String,
    },
    /// Input dialog (single field)
    Input {
        title: String,
        prompt: String,
        value: String,
        cursor_pos: usize,
        placeholder: Option<String>,
    },
    /// Selection dialog (choose from list)
    Select {
        title: String,
        items: Vec<SelectItem>,
        selected: usize,
    },
    /// Form dialog (multiple fields)
    Form {
        title: String,
        fields: Vec<FormField>,
        selected_field: usize,
    },
    /// Info/alert dialog
    Alert {
        title: String,
        message: String,
        level: AlertLevel,
    },
}

/// Alert level for styling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlertLevel {
    Info,
    Success,
    Warning,
    Error,
}

/// Item in a selection dialog
#[derive(Debug, Clone)]
pub struct SelectItem {
    pub id: String,
    pub label: String,
    pub description: Option<String>,
    pub disabled: bool,
}

/// Field in a form dialog
#[derive(Debug, Clone)]
pub struct FormField {
    pub id: String,
    pub label: String,
    pub value: String,
    pub placeholder: Option<String>,
    pub required: bool,
    pub field_type: FormFieldType,
}

/// Type of form field
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormFieldType {
    Text,
    Password,
    Number,
    Select(Vec<String>),
    Toggle,
}

/// Dialog state
#[derive(Debug, Clone, Default)]
pub struct DialogState {
    pub active: bool,
    pub kind: Option<DialogKind>,
    pub result: Option<DialogResult>,
}

/// Result from a dialog
#[derive(Debug, Clone)]
pub enum DialogResult {
    Confirmed,
    Cancelled,
    Input(String),
    Selected(String),
    Form(Vec<(String, String)>),
}

impl DialogState {
    /// Open a confirmation dialog
    pub fn confirm(title: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            active: true,
            kind: Some(DialogKind::Confirm {
                title: title.into(),
                message: message.into(),
                confirm_label: "Yes".to_string(),
                cancel_label: "No".to_string(),
            }),
            result: None,
        }
    }

    /// Open a confirmation with custom labels
    pub fn confirm_custom(
        title: impl Into<String>,
        message: impl Into<String>,
        confirm: impl Into<String>,
        cancel: impl Into<String>,
    ) -> Self {
        Self {
            active: true,
            kind: Some(DialogKind::Confirm {
                title: title.into(),
                message: message.into(),
                confirm_label: confirm.into(),
                cancel_label: cancel.into(),
            }),
            result: None,
        }
    }

    /// Open an input dialog
    pub fn input(title: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            active: true,
            kind: Some(DialogKind::Input {
                title: title.into(),
                prompt: prompt.into(),
                value: String::new(),
                cursor_pos: 0,
                placeholder: None,
            }),
            result: None,
        }
    }

    /// Open a selection dialog
    pub fn select(title: impl Into<String>, items: Vec<SelectItem>) -> Self {
        Self {
            active: true,
            kind: Some(DialogKind::Select {
                title: title.into(),
                items,
                selected: 0,
            }),
            result: None,
        }
    }

    /// Open an alert dialog
    pub fn alert(title: impl Into<String>, message: impl Into<String>, level: AlertLevel) -> Self {
        Self {
            active: true,
            kind: Some(DialogKind::Alert {
                title: title.into(),
                message: message.into(),
                level,
            }),
            result: None,
        }
    }

    /// Open a form dialog
    pub fn form(title: impl Into<String>, fields: Vec<FormField>) -> Self {
        Self {
            active: true,
            kind: Some(DialogKind::Form {
                title: title.into(),
                fields,
                selected_field: 0,
            }),
            result: None,
        }
    }

    /// Close the dialog
    pub fn close(&mut self) {
        self.active = false;
        self.kind = None;
    }

    /// Set confirmed result
    pub fn confirm_result(&mut self) {
        self.result = Some(DialogResult::Confirmed);
        self.close();
    }

    /// Set cancelled result
    pub fn cancel_result(&mut self) {
        self.result = Some(DialogResult::Cancelled);
        self.close();
    }

    /// Take the result (consuming it)
    pub fn take_result(&mut self) -> Option<DialogResult> {
        self.result.take()
    }
}

/// Render the dialog overlay
pub fn render_dialog(frame: &mut Frame, dialog: &DialogState, theme: &Theme) {
    if !dialog.active {
        return;
    }

    let Some(kind) = &dialog.kind else {
        return;
    };

    let area = frame.area();
    let dialog_area = centered_rect(60, 40, area);

    // Clear the area behind the dialog
    frame.render_widget(Clear, dialog_area);

    match kind {
        DialogKind::Confirm {
            title,
            message,
            confirm_label,
            cancel_label,
        } => {
            render_confirm_dialog(
                frame,
                dialog_area,
                title,
                message,
                confirm_label,
                cancel_label,
                theme,
            );
        }
        DialogKind::Input {
            title,
            prompt,
            value,
            cursor_pos,
            placeholder,
        } => {
            render_input_dialog(
                frame,
                dialog_area,
                title,
                prompt,
                value,
                *cursor_pos,
                placeholder.as_deref(),
                theme,
            );
        }
        DialogKind::Select {
            title,
            items,
            selected,
        } => {
            render_select_dialog(frame, dialog_area, title, items, *selected, theme);
        }
        DialogKind::Form {
            title,
            fields,
            selected_field,
        } => {
            render_form_dialog(frame, dialog_area, title, fields, *selected_field, theme);
        }
        DialogKind::Alert {
            title,
            message,
            level,
        } => {
            render_alert_dialog(frame, dialog_area, title, message, *level, theme);
        }
    }
}

fn render_confirm_dialog(
    frame: &mut Frame,
    area: Rect,
    title: &str,
    message: &str,
    confirm_label: &str,
    cancel_label: &str,
    theme: &Theme,
) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(3), Constraint::Length(3)])
        .margin(1)
        .split(area);

    // Message
    let msg = Paragraph::new(message)
        .style(theme.text())
        .wrap(Wrap { trim: true })
        .alignment(Alignment::Center)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border())
                .title(Span::styled(format!(" {} ", title), theme.block_title())),
        );
    frame.render_widget(msg, area);

    // Buttons
    let buttons = Line::from(vec![
        Span::styled(
            format!(" [Enter] {} ", confirm_label),
            theme.text_accent().add_modifier(Modifier::BOLD),
        ),
        Span::raw("  "),
        Span::styled(format!("[Esc] {} ", cancel_label), theme.text_muted()),
    ]);
    let buttons_para = Paragraph::new(buttons).alignment(Alignment::Center);
    frame.render_widget(buttons_para, chunks[1]);
}

#[allow(clippy::too_many_arguments)]
fn render_input_dialog(
    frame: &mut Frame,
    area: Rect,
    title: &str,
    prompt: &str,
    value: &str,
    _cursor_pos: usize,
    placeholder: Option<&str>,
    theme: &Theme,
) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),
            Constraint::Length(3),
            Constraint::Length(2),
        ])
        .margin(1)
        .split(area);

    // Dialog block
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(theme.border())
        .title(Span::styled(format!(" {} ", title), theme.block_title()));
    frame.render_widget(block, area);

    // Prompt
    let prompt_para = Paragraph::new(prompt).style(theme.text());
    frame.render_widget(prompt_para, chunks[0]);

    // Input field
    let display_value = if value.is_empty() {
        placeholder.unwrap_or("").to_string()
    } else {
        value.to_string()
    };
    let input_style = if value.is_empty() {
        theme.text_muted()
    } else {
        theme.text()
    };
    let input = Paragraph::new(format!("{}_", display_value))
        .style(input_style)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border_selected()),
        );
    frame.render_widget(input, chunks[1]);

    // Hints
    let hints = Line::from(vec![
        Span::styled("[Enter] ", theme.keybind()),
        Span::styled("Submit  ", theme.text_muted()),
        Span::styled("[Esc] ", theme.keybind()),
        Span::styled("Cancel", theme.text_muted()),
    ]);
    let hints_para = Paragraph::new(hints).alignment(Alignment::Center);
    frame.render_widget(hints_para, chunks[2]);
}

fn render_select_dialog(
    frame: &mut Frame,
    area: Rect,
    title: &str,
    items: &[SelectItem],
    selected: usize,
    theme: &Theme,
) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(3), Constraint::Length(2)])
        .margin(1)
        .split(area);

    // List items
    let list_items: Vec<ListItem> = items
        .iter()
        .enumerate()
        .map(|(i, item)| {
            let style = if item.disabled {
                theme.text_muted()
            } else if i == selected {
                theme.selected_row()
            } else {
                theme.text()
            };
            let prefix = if i == selected { "▶ " } else { "  " };
            let mut spans = vec![Span::raw(prefix), Span::styled(&item.label, style)];
            if let Some(desc) = &item.description {
                spans.push(Span::styled(format!(" - {}", desc), theme.text_muted()));
            }
            ListItem::new(Line::from(spans))
        })
        .collect();

    let list = List::new(list_items).block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(theme.border())
            .title(Span::styled(format!(" {} ", title), theme.block_title())),
    );
    frame.render_widget(list, area);

    // Hints
    let hints = Line::from(vec![
        Span::styled("[j/k] ", theme.keybind()),
        Span::styled("Navigate  ", theme.text_muted()),
        Span::styled("[Enter] ", theme.keybind()),
        Span::styled("Select  ", theme.text_muted()),
        Span::styled("[Esc] ", theme.keybind()),
        Span::styled("Cancel", theme.text_muted()),
    ]);
    let hints_para = Paragraph::new(hints).alignment(Alignment::Center);
    frame.render_widget(hints_para, chunks[1]);
}

fn render_form_dialog(
    frame: &mut Frame,
    area: Rect,
    title: &str,
    fields: &[FormField],
    selected_field: usize,
    theme: &Theme,
) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(3), Constraint::Length(2)])
        .margin(1)
        .split(area);

    // Block
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(theme.border())
        .title(Span::styled(format!(" {} ", title), theme.block_title()));
    frame.render_widget(block, area);

    // Form fields - create constraints for each field
    let field_constraints: Vec<Constraint> = fields.iter().map(|_| Constraint::Length(3)).collect();

    let inner = Layout::default()
        .direction(Direction::Vertical)
        .constraints(field_constraints)
        .margin(1)
        .split(chunks[0]);

    for (i, field) in fields.iter().enumerate() {
        if i >= inner.len() {
            break;
        }
        let is_selected = i == selected_field;
        let border_style = if is_selected {
            theme.border_selected()
        } else {
            theme.border()
        };

        let label_style = if field.required {
            theme.text_accent()
        } else {
            theme.text()
        };

        let display_value = if field.value.is_empty() {
            field.placeholder.clone().unwrap_or_default()
        } else if field.field_type == FormFieldType::Password {
            "*".repeat(field.value.len())
        } else {
            field.value.clone()
        };

        let value_style = if field.value.is_empty() {
            theme.text_muted()
        } else {
            theme.text()
        };

        let label = if field.required {
            format!("{}*", field.label)
        } else {
            field.label.clone()
        };

        let content = if is_selected {
            format!("{}_", display_value)
        } else {
            display_value
        };

        let field_widget = Paragraph::new(content).style(value_style).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(border_style)
                .title(Span::styled(format!(" {} ", label), label_style)),
        );
        frame.render_widget(field_widget, inner[i]);
    }

    // Hints
    let hints = Line::from(vec![
        Span::styled("[Tab] ", theme.keybind()),
        Span::styled("Next  ", theme.text_muted()),
        Span::styled("[Enter] ", theme.keybind()),
        Span::styled("Submit  ", theme.text_muted()),
        Span::styled("[Esc] ", theme.keybind()),
        Span::styled("Cancel", theme.text_muted()),
    ]);
    let hints_para = Paragraph::new(hints).alignment(Alignment::Center);
    frame.render_widget(hints_para, chunks[1]);
}

fn render_alert_dialog(
    frame: &mut Frame,
    area: Rect,
    title: &str,
    message: &str,
    level: AlertLevel,
    theme: &Theme,
) {
    let title_style = match level {
        AlertLevel::Info => theme.text_info(),
        AlertLevel::Success => theme.text_success(),
        AlertLevel::Warning => theme.text_warning(),
        AlertLevel::Error => theme.text_error(),
    };

    let icon = match level {
        AlertLevel::Info => "ℹ️ ",
        AlertLevel::Success => "✅ ",
        AlertLevel::Warning => "⚠️ ",
        AlertLevel::Error => "❌ ",
    };

    let msg = Paragraph::new(message)
        .style(theme.text())
        .wrap(Wrap { trim: true })
        .alignment(Alignment::Center)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border())
                .title(Span::styled(
                    format!(" {}{} ", icon, title),
                    title_style.add_modifier(Modifier::BOLD),
                )),
        );
    frame.render_widget(msg, area);
}

/// Helper function to create a centered rect
fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(area);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}

/// Handle key events for dialogs
/// Returns true if the key was handled
pub fn handle_dialog_key(dialog: &mut DialogState, key: crossterm::event::KeyEvent) -> bool {
    use crossterm::event::KeyCode;

    if !dialog.active {
        return false;
    }

    let Some(kind) = &mut dialog.kind else {
        return false;
    };

    match key.code {
        KeyCode::Esc => {
            dialog.cancel_result();
            true
        }
        KeyCode::Enter => {
            match kind {
                DialogKind::Confirm { .. } | DialogKind::Alert { .. } => {
                    dialog.confirm_result();
                }
                DialogKind::Input { value, .. } => {
                    dialog.result = Some(DialogResult::Input(value.clone()));
                    dialog.close();
                }
                DialogKind::Select {
                    items, selected, ..
                } => {
                    if let Some(item) = items.get(*selected) {
                        if !item.disabled {
                            dialog.result = Some(DialogResult::Selected(item.id.clone()));
                            dialog.close();
                        }
                    }
                }
                DialogKind::Form { fields, .. } => {
                    let values: Vec<(String, String)> = fields
                        .iter()
                        .map(|f| (f.id.clone(), f.value.clone()))
                        .collect();
                    dialog.result = Some(DialogResult::Form(values));
                    dialog.close();
                }
            }
            true
        }
        KeyCode::Up => match kind {
            DialogKind::Select {
                selected, items, ..
            } => {
                if *selected > 0 {
                    *selected -= 1;
                } else {
                    *selected = items.len().saturating_sub(1);
                }
                true
            }
            _ => false,
        },
        KeyCode::Down => match kind {
            DialogKind::Select {
                selected, items, ..
            } => {
                if *selected < items.len().saturating_sub(1) {
                    *selected += 1;
                } else {
                    *selected = 0;
                }
                true
            }
            _ => false,
        },
        KeyCode::Backspace => match kind {
            DialogKind::Input {
                value, cursor_pos, ..
            } => {
                if *cursor_pos > 0 {
                    *cursor_pos -= 1;
                    value.remove(*cursor_pos);
                }
                true
            }
            DialogKind::Form {
                fields,
                selected_field,
                ..
            } => {
                if let Some(field) = fields.get_mut(*selected_field) {
                    field.value.pop();
                }
                true
            }
            _ => false,
        },
        KeyCode::Char('k') => {
            // k for up navigation in select dialogs
            match kind {
                DialogKind::Select {
                    selected, items, ..
                } => {
                    if *selected > 0 {
                        *selected -= 1;
                    } else {
                        *selected = items.len().saturating_sub(1);
                    }
                    true
                }
                // For input/form dialogs, treat 'k' as a character
                DialogKind::Input {
                    value, cursor_pos, ..
                } => {
                    value.insert(*cursor_pos, 'k');
                    *cursor_pos += 1;
                    true
                }
                DialogKind::Form {
                    fields,
                    selected_field,
                    ..
                } => {
                    if let Some(field) = fields.get_mut(*selected_field) {
                        field.value.push('k');
                    }
                    true
                }
                _ => false,
            }
        }
        KeyCode::Char('j') => {
            // j for down navigation in select dialogs
            match kind {
                DialogKind::Select {
                    selected, items, ..
                } => {
                    if *selected < items.len().saturating_sub(1) {
                        *selected += 1;
                    } else {
                        *selected = 0;
                    }
                    true
                }
                // For input/form dialogs, treat 'j' as a character
                DialogKind::Input {
                    value, cursor_pos, ..
                } => {
                    value.insert(*cursor_pos, 'j');
                    *cursor_pos += 1;
                    true
                }
                DialogKind::Form {
                    fields,
                    selected_field,
                    ..
                } => {
                    if let Some(field) = fields.get_mut(*selected_field) {
                        field.value.push('j');
                    }
                    true
                }
                _ => false,
            }
        }
        KeyCode::Char(c) => match kind {
            DialogKind::Input {
                value, cursor_pos, ..
            } => {
                value.insert(*cursor_pos, c);
                *cursor_pos += 1;
                true
            }
            DialogKind::Form {
                fields,
                selected_field,
                ..
            } => {
                if let Some(field) = fields.get_mut(*selected_field) {
                    field.value.push(c);
                }
                true
            }
            _ => false,
        },
        KeyCode::Tab => match kind {
            DialogKind::Form {
                fields,
                selected_field,
                ..
            } => {
                if *selected_field < fields.len().saturating_sub(1) {
                    *selected_field += 1;
                } else {
                    *selected_field = 0;
                }
                true
            }
            _ => false,
        },
        KeyCode::BackTab => match kind {
            DialogKind::Form {
                fields,
                selected_field,
                ..
            } => {
                if *selected_field > 0 {
                    *selected_field -= 1;
                } else {
                    *selected_field = fields.len().saturating_sub(1);
                }
                true
            }
            _ => false,
        },
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    fn key_event(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::empty())
    }

    #[test]
    fn test_dialog_state_default() {
        let dialog = DialogState::default();
        assert!(!dialog.active);
        assert!(dialog.kind.is_none());
        assert!(dialog.result.is_none());
    }

    #[test]
    fn test_confirm_dialog_creation() {
        let dialog = DialogState::confirm("Delete?", "Are you sure you want to delete this?");
        assert!(dialog.active);
        assert!(matches!(dialog.kind, Some(DialogKind::Confirm { .. })));
    }

    #[test]
    fn test_confirm_custom_labels() {
        let dialog = DialogState::confirm_custom("Delete?", "Are you sure?", "Delete", "Keep");
        if let Some(DialogKind::Confirm {
            confirm_label,
            cancel_label,
            ..
        }) = &dialog.kind
        {
            assert_eq!(confirm_label, "Delete");
            assert_eq!(cancel_label, "Keep");
        } else {
            panic!("Expected Confirm dialog");
        }
    }

    #[test]
    fn test_input_dialog_creation() {
        let dialog = DialogState::input("Name", "Enter your name:");
        assert!(dialog.active);
        if let Some(DialogKind::Input { title, prompt, .. }) = &dialog.kind {
            assert_eq!(title, "Name");
            assert_eq!(prompt, "Enter your name:");
        } else {
            panic!("Expected Input dialog");
        }
    }

    #[test]
    fn test_select_dialog_creation() {
        let items = vec![
            SelectItem {
                id: "a".to_string(),
                label: "Option A".to_string(),
                description: None,
                disabled: false,
            },
            SelectItem {
                id: "b".to_string(),
                label: "Option B".to_string(),
                description: Some("Description".to_string()),
                disabled: true,
            },
        ];
        let dialog = DialogState::select("Choose", items);
        assert!(dialog.active);
        if let Some(DialogKind::Select {
            items, selected, ..
        }) = &dialog.kind
        {
            assert_eq!(items.len(), 2);
            assert_eq!(*selected, 0);
        } else {
            panic!("Expected Select dialog");
        }
    }

    #[test]
    fn test_alert_dialog_creation() {
        let dialog = DialogState::alert("Error", "Something went wrong", AlertLevel::Error);
        assert!(dialog.active);
        if let Some(DialogKind::Alert { level, .. }) = &dialog.kind {
            assert_eq!(*level, AlertLevel::Error);
        } else {
            panic!("Expected Alert dialog");
        }
    }

    #[test]
    fn test_form_dialog_creation() {
        let fields = vec![
            FormField {
                id: "name".to_string(),
                label: "Name".to_string(),
                value: String::new(),
                placeholder: Some("Enter name".to_string()),
                required: true,
                field_type: FormFieldType::Text,
            },
            FormField {
                id: "pass".to_string(),
                label: "Password".to_string(),
                value: String::new(),
                placeholder: None,
                required: true,
                field_type: FormFieldType::Password,
            },
        ];
        let dialog = DialogState::form("Login", fields);
        assert!(dialog.active);
        if let Some(DialogKind::Form { fields, .. }) = &dialog.kind {
            assert_eq!(fields.len(), 2);
        } else {
            panic!("Expected Form dialog");
        }
    }

    #[test]
    fn test_dialog_close() {
        let mut dialog = DialogState::confirm("Test", "Test message");
        assert!(dialog.active);
        dialog.close();
        assert!(!dialog.active);
        assert!(dialog.kind.is_none());
    }

    #[test]
    fn test_dialog_confirm_result() {
        let mut dialog = DialogState::confirm("Test", "Test message");
        dialog.confirm_result();
        assert!(!dialog.active);
        assert!(matches!(dialog.result, Some(DialogResult::Confirmed)));
    }

    #[test]
    fn test_dialog_cancel_result() {
        let mut dialog = DialogState::confirm("Test", "Test message");
        dialog.cancel_result();
        assert!(!dialog.active);
        assert!(matches!(dialog.result, Some(DialogResult::Cancelled)));
    }

    #[test]
    fn test_dialog_take_result() {
        let mut dialog = DialogState::confirm("Test", "Test message");
        dialog.confirm_result();
        let result = dialog.take_result();
        assert!(matches!(result, Some(DialogResult::Confirmed)));
        assert!(dialog.result.is_none()); // Consumed
    }

    #[test]
    fn test_handle_esc_cancels_dialog() {
        let mut dialog = DialogState::confirm("Test", "Test message");
        let handled = handle_dialog_key(&mut dialog, key_event(KeyCode::Esc));
        assert!(handled);
        assert!(!dialog.active);
        assert!(matches!(dialog.result, Some(DialogResult::Cancelled)));
    }

    #[test]
    fn test_handle_enter_confirms_dialog() {
        let mut dialog = DialogState::confirm("Test", "Test message");
        let handled = handle_dialog_key(&mut dialog, key_event(KeyCode::Enter));
        assert!(handled);
        assert!(!dialog.active);
        assert!(matches!(dialog.result, Some(DialogResult::Confirmed)));
    }

    #[test]
    fn test_input_dialog_typing() {
        let mut dialog = DialogState::input("Name", "Enter name:");
        handle_dialog_key(&mut dialog, key_event(KeyCode::Char('H')));
        handle_dialog_key(&mut dialog, key_event(KeyCode::Char('i')));

        if let Some(DialogKind::Input { value, .. }) = &dialog.kind {
            assert_eq!(value, "Hi");
        } else {
            panic!("Expected Input dialog");
        }
    }

    #[test]
    fn test_input_dialog_backspace() {
        let mut dialog = DialogState::input("Name", "Enter name:");
        handle_dialog_key(&mut dialog, key_event(KeyCode::Char('A')));
        handle_dialog_key(&mut dialog, key_event(KeyCode::Char('B')));
        handle_dialog_key(&mut dialog, key_event(KeyCode::Backspace));

        if let Some(DialogKind::Input { value, .. }) = &dialog.kind {
            assert_eq!(value, "A");
        } else {
            panic!("Expected Input dialog");
        }
    }

    #[test]
    fn test_input_dialog_submit() {
        let mut dialog = DialogState::input("Name", "Enter name:");
        handle_dialog_key(&mut dialog, key_event(KeyCode::Char('T')));
        handle_dialog_key(&mut dialog, key_event(KeyCode::Char('e')));
        handle_dialog_key(&mut dialog, key_event(KeyCode::Char('s')));
        handle_dialog_key(&mut dialog, key_event(KeyCode::Char('t')));
        handle_dialog_key(&mut dialog, key_event(KeyCode::Enter));

        assert!(!dialog.active);
        if let Some(DialogResult::Input(value)) = &dialog.result {
            assert_eq!(value, "Test");
        } else {
            panic!("Expected Input result");
        }
    }

    #[test]
    fn test_select_dialog_navigation() {
        let items = vec![
            SelectItem {
                id: "a".to_string(),
                label: "A".to_string(),
                description: None,
                disabled: false,
            },
            SelectItem {
                id: "b".to_string(),
                label: "B".to_string(),
                description: None,
                disabled: false,
            },
            SelectItem {
                id: "c".to_string(),
                label: "C".to_string(),
                description: None,
                disabled: false,
            },
        ];
        let mut dialog = DialogState::select("Choose", items);

        // Initial selection is 0
        if let Some(DialogKind::Select { selected, .. }) = &dialog.kind {
            assert_eq!(*selected, 0);
        }

        // Move down
        handle_dialog_key(&mut dialog, key_event(KeyCode::Down));
        if let Some(DialogKind::Select { selected, .. }) = &dialog.kind {
            assert_eq!(*selected, 1);
        }

        // Move down with j
        handle_dialog_key(&mut dialog, key_event(KeyCode::Char('j')));
        if let Some(DialogKind::Select { selected, .. }) = &dialog.kind {
            assert_eq!(*selected, 2);
        }

        // Wrap around
        handle_dialog_key(&mut dialog, key_event(KeyCode::Down));
        if let Some(DialogKind::Select { selected, .. }) = &dialog.kind {
            assert_eq!(*selected, 0);
        }

        // Move up with k
        handle_dialog_key(&mut dialog, key_event(KeyCode::Char('k')));
        if let Some(DialogKind::Select { selected, .. }) = &dialog.kind {
            assert_eq!(*selected, 2); // Wraps to end
        }
    }

    #[test]
    fn test_select_dialog_submit() {
        let items = vec![
            SelectItem {
                id: "opt1".to_string(),
                label: "Option 1".to_string(),
                description: None,
                disabled: false,
            },
            SelectItem {
                id: "opt2".to_string(),
                label: "Option 2".to_string(),
                description: None,
                disabled: false,
            },
        ];
        let mut dialog = DialogState::select("Choose", items);
        handle_dialog_key(&mut dialog, key_event(KeyCode::Down));
        handle_dialog_key(&mut dialog, key_event(KeyCode::Enter));

        assert!(!dialog.active);
        if let Some(DialogResult::Selected(id)) = &dialog.result {
            assert_eq!(id, "opt2");
        } else {
            panic!("Expected Selected result");
        }
    }

    #[test]
    fn test_select_disabled_item_not_selectable() {
        let items = vec![SelectItem {
            id: "disabled".to_string(),
            label: "Disabled".to_string(),
            description: None,
            disabled: true,
        }];
        let mut dialog = DialogState::select("Choose", items);
        handle_dialog_key(&mut dialog, key_event(KeyCode::Enter));

        // Should still be active because item is disabled
        assert!(dialog.active);
    }

    #[test]
    fn test_form_dialog_tab_navigation() {
        let fields = vec![
            FormField {
                id: "f1".to_string(),
                label: "Field 1".to_string(),
                value: String::new(),
                placeholder: None,
                required: false,
                field_type: FormFieldType::Text,
            },
            FormField {
                id: "f2".to_string(),
                label: "Field 2".to_string(),
                value: String::new(),
                placeholder: None,
                required: false,
                field_type: FormFieldType::Text,
            },
        ];
        let mut dialog = DialogState::form("Form", fields);

        if let Some(DialogKind::Form { selected_field, .. }) = &dialog.kind {
            assert_eq!(*selected_field, 0);
        }

        handle_dialog_key(&mut dialog, key_event(KeyCode::Tab));
        if let Some(DialogKind::Form { selected_field, .. }) = &dialog.kind {
            assert_eq!(*selected_field, 1);
        }

        // Tab wraps
        handle_dialog_key(&mut dialog, key_event(KeyCode::Tab));
        if let Some(DialogKind::Form { selected_field, .. }) = &dialog.kind {
            assert_eq!(*selected_field, 0);
        }

        // BackTab
        handle_dialog_key(&mut dialog, key_event(KeyCode::BackTab));
        if let Some(DialogKind::Form { selected_field, .. }) = &dialog.kind {
            assert_eq!(*selected_field, 1);
        }
    }

    #[test]
    fn test_form_dialog_submit() {
        let fields = vec![
            FormField {
                id: "name".to_string(),
                label: "Name".to_string(),
                value: String::new(),
                placeholder: None,
                required: false,
                field_type: FormFieldType::Text,
            },
            FormField {
                id: "email".to_string(),
                label: "Email".to_string(),
                value: String::new(),
                placeholder: None,
                required: false,
                field_type: FormFieldType::Text,
            },
        ];
        let mut dialog = DialogState::form("Form", fields);

        // Type in first field
        handle_dialog_key(&mut dialog, key_event(KeyCode::Char('J')));
        handle_dialog_key(&mut dialog, key_event(KeyCode::Char('o')));
        handle_dialog_key(&mut dialog, key_event(KeyCode::Char('h')));
        handle_dialog_key(&mut dialog, key_event(KeyCode::Char('n')));

        // Move to second field
        handle_dialog_key(&mut dialog, key_event(KeyCode::Tab));
        handle_dialog_key(&mut dialog, key_event(KeyCode::Char('a')));
        handle_dialog_key(&mut dialog, key_event(KeyCode::Char('@')));
        handle_dialog_key(&mut dialog, key_event(KeyCode::Char('b')));

        // Submit
        handle_dialog_key(&mut dialog, key_event(KeyCode::Enter));

        assert!(!dialog.active);
        if let Some(DialogResult::Form(values)) = &dialog.result {
            assert_eq!(values.len(), 2);
            assert_eq!(values[0], ("name".to_string(), "John".to_string()));
            assert_eq!(values[1], ("email".to_string(), "a@b".to_string()));
        } else {
            panic!("Expected Form result");
        }
    }

    #[test]
    fn test_inactive_dialog_ignores_keys() {
        let mut dialog = DialogState::default();
        let handled = handle_dialog_key(&mut dialog, key_event(KeyCode::Enter));
        assert!(!handled);
    }

    #[test]
    fn test_centered_rect() {
        let area = Rect::new(0, 0, 100, 50);
        let centered = centered_rect(60, 40, area);

        // Should be roughly centered (percentage-based so not exact)
        assert!(centered.x > 0);
        assert!(centered.y > 0);
        assert!(centered.width < area.width);
        assert!(centered.height < area.height);
    }
}
