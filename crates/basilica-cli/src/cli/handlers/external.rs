//! Network component delegation handlers

use crate::error::CliError;
use std::path::PathBuf;
use std::process::Command;
use tracing::debug;

#[cfg(unix)]
use std::os::unix::process::CommandExt;

/// Helper to make extension handling testable across platforms
fn resolve_binary_name_with_ext(binary_name: &str) -> String {
    if cfg!(windows) {
        match std::path::Path::new(binary_name)
            .extension()
            .and_then(|s| s.to_str())
        {
            Some(ext) if ext.eq_ignore_ascii_case("exe") => binary_name.to_string(),
            _ => format!("{binary_name}.exe"),
        }
    } else {
        binary_name.to_string()
    }
}

/// Handle validator delegation
pub fn handle_validator(args: Vec<String>) -> Result<(), CliError> {
    debug!("Delegating to basilica-validator with args: {:?}", args);
    delegate_to_binary("basilica-validator", args)
}

/// Handle miner delegation
pub fn handle_miner(args: Vec<String>) -> Result<(), CliError> {
    debug!("Delegating to basilica-miner with args: {:?}", args);
    delegate_to_binary("basilica-miner", args)
}

/// Delegate execution to another binary
fn delegate_to_binary(binary_name: &str, args: Vec<String>) -> Result<(), CliError> {
    // Handle platform-specific executable extension
    let binary_name_with_ext = resolve_binary_name_with_ext(binary_name);

    // First check if the binary exists in the same directory as the current executable
    let binary_path = if let Ok(current_exe) = std::env::current_exe() {
        if let Some(exe_dir) = current_exe.parent() {
            let local_path = exe_dir.join(&binary_name_with_ext);
            if local_path.exists() {
                debug!(
                    "Found binary in same directory as basilica CLI: {:?}",
                    local_path
                );
                local_path
            } else {
                debug!("Binary not found in same directory as basilica CLI, using PATH lookup");
                PathBuf::from(&binary_name_with_ext)
            }
        } else {
            debug!("Could not determine basilica CLI directory, using PATH lookup");
            PathBuf::from(&binary_name_with_ext)
        }
    } else {
        debug!("Could not determine current executable path, using PATH lookup");
        PathBuf::from(&binary_name_with_ext)
    };

    delegate_to_binary_impl(binary_path, args, binary_name)
}

#[cfg(unix)]
fn delegate_to_binary_impl(
    binary_path: PathBuf,
    args: Vec<String>,
    binary_name: &str,
) -> Result<(), CliError> {
    // On Unix systems, use exec() to replace the current process
    // This provides seamless handoff with no overhead
    let error = Command::new(&binary_path).args(&args).exec(); // This replaces the current process

    // If we reach this point, exec() failed
    Err(CliError::DelegationComponent(
        std::io::Error::other(
            format!("Failed to execute {}: {}. Make sure {} is installed in the same directory as basilica or in PATH.",
                binary_path.display(),
                error,
                binary_name
            )
        )
    ))
}

#[cfg(windows)]
fn delegate_to_binary_impl(
    binary_path: PathBuf,
    args: Vec<String>,
    binary_name: &str,
) -> Result<(), CliError> {
    // On Windows, spawn the child process and wait for completion
    let mut child = Command::new(&binary_path)
        .args(&args)
        .spawn()
        .map_err(|error| {
            CliError::DelegationComponent(
                std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to execute {}: {}. Make sure {} is installed in the same directory as basilica or in PATH.",
                        binary_path.display(),
                        error,
                        binary_name
                    )
                )
            )
        })?;

    // Wait for the child process to complete and get its exit status
    let exit_status = child.wait().map_err(|error| {
        CliError::DelegationComponent(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Failed to wait for {} completion: {}", binary_name, error),
        ))
    })?;

    // Exit with the same code as the child process
    std::process::exit(exit_status.code().unwrap_or(1));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validator_delegation_fails_gracefully() {
        // This should fail since basilica-validator is not in PATH during tests
        let result = handle_validator(vec!["--help".to_string()]);
        assert!(result.is_err());

        if let Err(CliError::DelegationComponent(err)) = result {
            assert!(err.to_string().contains("basilica-validator"));
        } else {
            panic!("Expected DelegationComponent error");
        }
    }

    #[test]
    fn test_binary_name_with_extension() {
        // Test executable extension handling
        if cfg!(windows) {
            assert_eq!(
                resolve_binary_name_with_ext("basilica-validator"),
                "basilica-validator.exe"
            );
            assert_eq!(
                resolve_binary_name_with_ext("basilica-validator.exe"),
                "basilica-validator.exe"
            );
            assert_eq!(
                resolve_binary_name_with_ext("BASILICA-VALIDATOR.EXE"),
                "BASILICA-VALIDATOR.EXE"
            );
        } else {
            assert_eq!(
                resolve_binary_name_with_ext("basilica-validator"),
                "basilica-validator"
            );
        }
    }
}
