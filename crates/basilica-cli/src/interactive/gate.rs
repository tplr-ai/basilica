//! Interactivity gate. Every interactive prompt MUST route through this.
//! In non-interactive mode the helpers return structured errors instead of hanging.

use crate::error::CliError;
use dialoguer::{theme::ColorfulTheme, Confirm, Input, Select};
use std::io::IsTerminal;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Interactivity {
    Interactive,
    NonInteractive,
}

pub fn current() -> Interactivity {
    if !std::io::stdin().is_terminal() || std::env::var("BASILICA_NON_INTERACTIVE").is_ok() {
        Interactivity::NonInteractive
    } else {
        Interactivity::Interactive
    }
}

// Helpers below take three concerns explicitly:
// - `field`: stable machine identifier (shows up as `"field":` in JSON errors).
// - `prompt`: human-readable question shown interactively via dialoguer.
// - `hint`: how-to-skip guidance shown in non-interactive errors.

pub fn ask_text(
    field: &str,
    prompt: &str,
    default: Option<&str>,
    hint: &str,
) -> Result<String, CliError> {
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
            let mut input = Input::<String>::with_theme(&theme).with_prompt(prompt);
            if let Some(d) = default {
                input = input.default(d.to_string());
            }
            input
                .interact_text()
                .map_err(|e| CliError::Internal(color_eyre::eyre::eyre!(e)))
        }
    }
}

pub fn ask_select(
    field: &str,
    prompt: &str,
    labels: &[&str],
    hint: &str,
) -> Result<usize, CliError> {
    match current() {
        Interactivity::NonInteractive => Err(CliError::MissingInput {
            field: field.to_string(),
            hint: hint.to_string(),
        }),
        Interactivity::Interactive => {
            let theme = ColorfulTheme::default();
            Select::with_theme(&theme)
                .with_prompt(prompt)
                .items(labels)
                .default(0)
                .interact()
                .map_err(|e| CliError::Internal(color_eyre::eyre::eyre!(e)))
        }
    }
}

pub fn ask_confirm(field: &str, prompt: &str, default: bool, hint: &str) -> Result<bool, CliError> {
    match current() {
        Interactivity::NonInteractive => Err(CliError::MissingInput {
            field: field.to_string(),
            hint: hint.to_string(),
        }),
        Interactivity::Interactive => {
            let theme = ColorfulTheme::default();
            Confirm::with_theme(&theme)
                .with_prompt(prompt)
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

    /// RAII guard: sets `BASILICA_NON_INTERACTIVE` for the duration of the test
    /// and clears it on drop so neighboring tests aren't affected.
    struct NonInteractiveEnv;

    impl NonInteractiveEnv {
        fn set() -> Self {
            std::env::set_var("BASILICA_NON_INTERACTIVE", "1");
            Self
        }
    }

    impl Drop for NonInteractiveEnv {
        fn drop(&mut self) {
            std::env::remove_var("BASILICA_NON_INTERACTIVE");
        }
    }

    #[test]
    #[serial]
    fn ask_text_non_interactive_no_default_errors() {
        let _env = NonInteractiveEnv::set();
        let err = ask_text("name", "Name?", None, "Pass --name").unwrap_err();
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
        let _env = NonInteractiveEnv::set();
        let v = ask_text("name", "Name?", Some("auto-name"), "irrelevant").unwrap();
        assert_eq!(v, "auto-name");
    }

    #[test]
    #[serial]
    fn ask_select_non_interactive_errors() {
        let _env = NonInteractiveEnv::set();
        let labels = ["alpha", "beta"];
        let err = ask_select(
            "offering",
            "Choose an offering",
            &labels,
            "Pass --offering-id",
        )
        .unwrap_err();
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
        let _env = NonInteractiveEnv::set();
        let err = ask_confirm("replace", "Replace?", false, "Pass --force").unwrap_err();
        match err {
            CliError::MissingInput { field, hint } => {
                assert_eq!(field, "replace");
                assert!(hint.contains("--force"));
            }
            other => panic!("expected MissingInput, got {other:?}"),
        }
    }
}
