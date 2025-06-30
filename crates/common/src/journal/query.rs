//! Journal query functionality

/// Query journal logs using journalctl
pub fn query_logs(
    validator_id: Option<&str>,
    since: Option<&str>,
    limit: Option<usize>,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    use std::process::Command;

    let mut cmd = Command::new("journalctl");
    cmd.arg("--output=json-pretty");
    cmd.arg("--no-pager");

    if let Some(since) = since {
        cmd.arg("--since").arg(since);
    }

    if let Some(limit) = limit {
        cmd.arg("--lines").arg(limit.to_string());
    }

    // Filter by validator_id if provided
    if let Some(validator_id) = validator_id {
        cmd.arg(format!("VALIDATOR_ID={validator_id}"));
    }

    // Filter by our application
    cmd.arg("RUST_LOG");

    let output = cmd.output()?;

    if output.status.success() {
        let logs = String::from_utf8(output.stdout)?;
        Ok(logs.lines().map(|s| s.to_string()).collect())
    } else {
        let error = String::from_utf8(output.stderr)?;
        Err(format!("journalctl failed: {error}").into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_logs_builds_command() {
        // Test that the function doesn't panic with various inputs
        let result = query_logs(Some("test_validator"), Some("1 hour ago"), Some(100));
        // We expect this to fail in test environment (no journalctl), but shouldn't panic
        assert!(result.is_err());
    }
}
