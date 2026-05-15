//! Render CliError to stderr. Two modes: Human (color-eyre style) and Json
//! (single object per error). Lives outside of `error.rs` so the error type
//! itself stays renderer-agnostic and does not need to implement `Serialize`.

use crate::error::CliError;
use crate::interactive::gate::Choices;
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
        CliError::MissingInput {
            field,
            hint,
            choices,
        } => serde_json::json!({
            "schema_version": 1,
            "error": "missing_input",
            "field": field,
            "hint": hint,
            "choices": choices_to_json(choices),
        }),
        CliError::MissingPrerequisite { field, hint } => serde_json::json!({
            "schema_version": 1,
            "error": "missing_prerequisite",
            "field": field,
            "hint": hint,
        }),
        other => serde_json::json!({
            "schema_version": 1,
            "error": "cli_error",
            "message": other.to_string(),
        }),
    };
    writeln!(w, "{}", payload)
}

fn choices_to_json(choices: &Choices) -> Vec<serde_json::Value> {
    choices
        .0
        .iter()
        .map(|c| {
            let mut obj = serde_json::Map::new();
            obj.insert("id".into(), serde_json::Value::String(c.id.clone()));
            obj.insert("label".into(), serde_json::Value::String(c.label.clone()));
            for (k, v) in &c.meta {
                obj.insert(k.clone(), v.clone());
            }
            serde_json::Value::Object(obj)
        })
        .collect()
}

fn render_human(err: &CliError, w: &mut dyn Write) -> std::io::Result<()> {
    match err {
        CliError::MissingInput {
            field,
            hint,
            choices,
        } => {
            writeln!(w, "error: missing input for '{}'", field)?;
            writeln!(w, "  hint: {}", hint)?;
            if !choices.0.is_empty() {
                writeln!(w, "  choices:")?;
                for c in &choices.0 {
                    writeln!(w, "    - {} ({})", c.label, c.id)?;
                }
            }
            Ok(())
        }
        CliError::MissingPrerequisite { field, hint } => {
            writeln!(w, "error: missing prerequisite for '{}'", field)?;
            writeln!(w, "  hint: {}", hint)
        }
        CliError::Internal(report) => writeln!(w, "Error: {:?}", report),
        other => writeln!(w, "Error: {}", other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interactive::gate::Choice;

    #[test]
    fn json_render_missing_input_with_choices() {
        let err = CliError::MissingInput {
            field: "offering_id".into(),
            hint: "Pass --offering-id".into(),
            choices: Choices(vec![Choice {
                id: "off_a".into(),
                label: "8x H100".into(),
                meta: {
                    let mut m = serde_json::Map::new();
                    m.insert("price_per_hr_usd".into(), serde_json::json!(18.40));
                    m
                },
            }]),
        };
        let mut buf = Vec::new();
        render_error(&err, RenderMode::Json, &mut buf).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&buf).unwrap();
        assert_eq!(v["error"], "missing_input");
        assert_eq!(v["field"], "offering_id");
        assert_eq!(v["choices"][0]["price_per_hr_usd"], 18.40);
        assert_eq!(v["schema_version"], 1);
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
        assert_eq!(v["schema_version"], 1);
    }

    #[test]
    fn human_render_missing_input_lists_choices() {
        let err = CliError::MissingInput {
            field: "rental".into(),
            hint: "Pass <rental-name-or-id>.".into(),
            choices: Choices(vec![Choice {
                id: "rent_1".into(),
                label: "alpha".into(),
                meta: Default::default(),
            }]),
        };
        let mut buf = Vec::new();
        render_error(&err, RenderMode::Human, &mut buf).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.contains("missing input"));
        assert!(s.contains("rental"));
        assert!(s.contains("alpha"));
        assert!(s.contains("rent_1"));
    }
}
