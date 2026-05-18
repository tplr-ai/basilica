//! Interactivity gate. Every interactive prompt MUST route through this.
//! In non-interactive mode the helpers return structured errors instead of hanging.

use crate::error::CliError;
use dialoguer::{theme::ColorfulTheme, Confirm, Input, Select};
use std::io::IsTerminal;
use std::sync::OnceLock;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Interactivity {
    Interactive,
    NonInteractive,
}

static CURRENT: OnceLock<Interactivity> = OnceLock::new();

pub fn current() -> Interactivity {
    *CURRENT.get_or_init(detect)
}

fn detect() -> Interactivity {
    if !std::io::stdin().is_terminal() || std::env::var("BASILICA_NON_INTERACTIVE").is_ok() {
        Interactivity::NonInteractive
    } else {
        Interactivity::Interactive
    }
}

#[cfg(test)]
pub fn set_for_test(v: Interactivity) {
    let _ = CURRENT.set(v);
}

pub fn ask_text(field: &str, default: Option<&str>, hint: &str) -> Result<String, CliError> {
    match current() {
        Interactivity::NonInteractive => {
            if let Some(d) = default {
                return Ok(d.to_string());
            }
            Err(CliError::MissingInput {
                field: field.to_string(),
                hint: hint.to_string(),
            })
        }
        Interactivity::Interactive => {
            let theme = ColorfulTheme::default();
            let mut input = Input::<String>::with_theme(&theme).with_prompt(field);
            if let Some(d) = default {
                input = input.default(d.to_string());
            }
            input
                .interact_text()
                .map_err(|e| CliError::Internal(color_eyre::eyre::eyre!(e)))
        }
    }
}

pub fn ask_select(field: &str, labels: &[&str], hint: &str) -> Result<usize, CliError> {
    match current() {
        Interactivity::NonInteractive => Err(CliError::MissingInput {
            field: field.to_string(),
            hint: hint.to_string(),
        }),
        Interactivity::Interactive => {
            let theme = ColorfulTheme::default();
            Select::with_theme(&theme)
                .with_prompt(field)
                .items(labels)
                .default(0)
                .interact()
                .map_err(|e| CliError::Internal(color_eyre::eyre::eyre!(e)))
        }
    }
}

pub fn ask_confirm(field: &str, default: bool, hint: &str) -> Result<bool, CliError> {
    match current() {
        Interactivity::NonInteractive => Err(CliError::MissingInput {
            field: field.to_string(),
            hint: hint.to_string(),
        }),
        Interactivity::Interactive => {
            let theme = ColorfulTheme::default();
            Confirm::with_theme(&theme)
                .with_prompt(field)
                .default(default)
                .interact()
                .map_err(|e| CliError::Internal(color_eyre::eyre::eyre!(e)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::CliError;
    use serial_test::serial;

    #[test]
    #[serial]
    fn ask_text_non_interactive_no_default_errors() {
        set_for_test(Interactivity::NonInteractive);
        let err = ask_text("name", None, "Pass --name").unwrap_err();
        match err {
            CliError::MissingInput { field, hint } => {
                assert_eq!(field, "name");
                assert!(hint.contains("--name"));
            }
            other => panic!("expected MissingInput, got {other:?}"),
        }
    }

    #[test]
    #[serial]
    fn ask_text_non_interactive_with_default_returns_default() {
        set_for_test(Interactivity::NonInteractive);
        let v = ask_text("name", Some("auto-name"), "irrelevant").unwrap();
        assert_eq!(v, "auto-name");
    }

    #[test]
    #[serial]
    fn ask_select_non_interactive_errors() {
        set_for_test(Interactivity::NonInteractive);
        let labels = ["alpha", "beta"];
        let err = ask_select("offering", &labels, "Pass --offering-id").unwrap_err();
        match err {
            CliError::MissingInput { field, hint } => {
                assert_eq!(field, "offering");
                assert!(hint.contains("--offering-id"));
            }
            other => panic!("expected MissingInput, got {other:?}"),
        }
    }

    #[test]
    #[serial]
    fn ask_confirm_non_interactive_errors() {
        set_for_test(Interactivity::NonInteractive);
        let err = ask_confirm("replace", false, "Pass --force").unwrap_err();
        match err {
            CliError::MissingInput { field, hint } => {
                assert_eq!(field, "replace");
                assert!(hint.contains("--force"));
            }
            other => panic!("expected MissingInput, got {other:?}"),
        }
    }
}
