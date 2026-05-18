//! Render CliError to stderr. Two modes: Human (color-eyre style) and Json
//! (single object per error). Lives outside of `error.rs` so the error type
//! itself stays renderer-agnostic and does not need to implement `Serialize`.

use crate::error::CliError;
use std::io::Write;

#[derive(Copy, Clone, Debug)]
pub enum RenderMode {
    Human,
    Json,
}

pub fn render_error(err: &CliError, mode: RenderMode, w: &mut dyn Write) -> std::io::Result<()> {
    match mode {
        RenderMode::Json => render_json(err, w),
        RenderMode::Human => render_human(err, w),
    }
}

fn render_json(err: &CliError, w: &mut dyn Write) -> std::io::Result<()> {
    let payload = match err {
        CliError::MissingInput { field, hint } => serde_json::json!({
            "error": "missing_input",
            "field": field,
            "hint": hint,
        }),
        CliError::MissingPrerequisite { field, hint } => serde_json::json!({
            "error": "missing_prerequisite",
            "field": field,
            "hint": hint,
        }),
        other => serde_json::json!({
            "error": "cli_error",
            "message": other.to_string(),
        }),
    };
    writeln!(w, "{}", payload)
}

fn render_human(err: &CliError, w: &mut dyn Write) -> std::io::Result<()> {
    match err {
        CliError::MissingInput { field, hint } => {
            writeln!(w, "error: missing input for '{}'", field)?;
            writeln!(w, "hint: {}", hint)
        }
        CliError::MissingPrerequisite { field, hint } => {
            writeln!(w, "error: missing prerequisite for '{}'", field)?;
            writeln!(w, "hint: {}", hint)
        }
        CliError::Internal(report) => writeln!(w, "Error: {:?}", report),
        other => writeln!(w, "Error: {}", other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_render_missing_input() {
        let err = CliError::MissingInput {
            field: "offering_id".into(),
            hint: "Pass --offering-id".into(),
        };
        let mut buf = Vec::new();
        render_error(&err, RenderMode::Json, &mut buf).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&buf).unwrap();
        assert_eq!(v["error"], "missing_input");
        assert_eq!(v["field"], "offering_id");
        assert_eq!(v["hint"], "Pass --offering-id");
    }

    #[test]
    fn json_render_missing_prerequisite() {
        let err = CliError::MissingPrerequisite {
            field: "ssh_key".into(),
            hint: "Run `basilica ssh-keys add` first.".into(),
        };
        let mut buf = Vec::new();
        render_error(&err, RenderMode::Json, &mut buf).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&buf).unwrap();
        assert_eq!(v["error"], "missing_prerequisite");
        assert_eq!(v["field"], "ssh_key");
    }

    #[test]
    fn human_render_missing_input() {
        let err = CliError::MissingInput {
            field: "rental".into(),
            hint: "Pass <rental-name-or-id>.".into(),
        };
        let mut buf = Vec::new();
        render_error(&err, RenderMode::Human, &mut buf).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.contains("missing input"));
        assert!(s.contains("rental"));
        assert!(s.contains("<rental-name-or-id>"));
    }
}
