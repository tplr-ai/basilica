//! Command execution and actions
//!
//! Provides functions for spawning external processes that require
//! the TUI to be suspended temporarily (SSH, exec, scp).
#![allow(dead_code)]

use anyhow::Result;
use std::process::{Command, Stdio};

/// SSH connection result
pub struct SpawnResult {
    pub success: bool,
    pub message: String,
}

/// Execute SSH connection to a rental
///
/// This spawns an interactive SSH session. The caller must suspend the TUI first.
pub fn ssh_connect_sync(host: &str, port: u16, user: &str) -> Result<SpawnResult> {
    tracing::info!("Connecting via SSH to {}@{}:{}", user, host, port);

    let status = Command::new("ssh")
        .args(["-p", &port.to_string(), &format!("{}@{}", user, host)])
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()?;

    Ok(SpawnResult {
        success: status.success(),
        message: if status.success() {
            "SSH session ended".to_string()
        } else {
            format!("SSH exited with code: {:?}", status.code())
        },
    })
}

/// Execute SSH connection (async wrapper for compatibility)
pub async fn ssh_connect(host: &str, port: u16, user: &str) -> Result<()> {
    // Build SSH command
    let ssh_cmd = format!("ssh -p {} {}@{}", port, user, host);
    tracing::info!("SSH command: {}", ssh_cmd);

    // Note: For interactive SSH, caller must suspend TUI and use ssh_connect_sync
    Ok(())
}

/// Execute a command on a rental via SSH (interactive mode)
///
/// The caller must suspend the TUI first for interactive commands.
pub fn ssh_exec_sync(host: &str, port: u16, user: &str, command: &str) -> Result<SpawnResult> {
    tracing::info!("Executing on {}@{}:{}: {}", user, host, port, command);

    let status = Command::new("ssh")
        .args([
            "-p",
            &port.to_string(),
            "-t", // Force pseudo-terminal allocation for interactive commands
            &format!("{}@{}", user, host),
            command,
        ])
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()?;

    Ok(SpawnResult {
        success: status.success(),
        message: if status.success() {
            "Command completed".to_string()
        } else {
            format!("Command exited with code: {:?}", status.code())
        },
    })
}

/// Execute a command on a rental via SSH (non-interactive, captures output)
pub async fn ssh_exec(host: &str, port: u16, user: &str, command: &str) -> Result<String> {
    let output = Command::new("ssh")
        .args([
            "-p",
            &port.to_string(),
            &format!("{}@{}", user, host),
            command,
        ])
        .output()?;

    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        Err(anyhow::anyhow!(
            "SSH command failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ))
    }
}

/// Copy files to/from a rental (interactive, shows progress)
///
/// The caller must suspend the TUI first.
pub fn scp_copy_sync(
    source: &str,
    destination: &str,
    host: &str,
    port: u16,
    user: &str,
    to_remote: bool,
) -> Result<SpawnResult> {
    let (src, dst) = if to_remote {
        (
            source.to_string(),
            format!("{}@{}:{}", user, host, destination),
        )
    } else {
        (
            format!("{}@{}:{}", user, host, source),
            destination.to_string(),
        )
    };

    tracing::info!("Copying {} -> {}", src, dst);

    let status = Command::new("scp")
        .args(["-P", &port.to_string(), "-r", &src, &dst])
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()?;

    Ok(SpawnResult {
        success: status.success(),
        message: if status.success() {
            "Copy completed".to_string()
        } else {
            format!("SCP exited with code: {:?}", status.code())
        },
    })
}

/// Copy files to/from a rental (non-interactive)
pub async fn scp_copy(
    source: &str,
    destination: &str,
    host: &str,
    port: u16,
    user: &str,
    to_remote: bool,
) -> Result<()> {
    let (src, dst) = if to_remote {
        (
            source.to_string(),
            format!("{}@{}:{}", user, host, destination),
        )
    } else {
        (
            format!("{}@{}:{}", user, host, source),
            destination.to_string(),
        )
    };

    let output = Command::new("scp")
        .args(["-P", &port.to_string(), "-r", &src, &dst])
        .output()?;

    if output.status.success() {
        Ok(())
    } else {
        Err(anyhow::anyhow!(
            "SCP failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ))
    }
}

/// Open URL in browser
pub fn open_url(url: &str) -> Result<()> {
    #[cfg(target_os = "macos")]
    {
        Command::new("open").arg(url).spawn()?;
    }
    #[cfg(target_os = "linux")]
    {
        Command::new("xdg-open").arg(url).spawn()?;
    }
    #[cfg(target_os = "windows")]
    {
        Command::new("cmd").args(["/c", "start", url]).spawn()?;
    }
    Ok(())
}

/// Copy text to clipboard
pub fn copy_to_clipboard(text: &str) -> Result<()> {
    #[cfg(target_os = "macos")]
    {
        let mut child = Command::new("pbcopy")
            .stdin(std::process::Stdio::piped())
            .spawn()?;
        if let Some(stdin) = child.stdin.as_mut() {
            use std::io::Write;
            stdin.write_all(text.as_bytes())?;
        }
        child.wait()?;
    }
    #[cfg(target_os = "linux")]
    {
        let mut child = Command::new("xclip")
            .args(["-selection", "clipboard"])
            .stdin(std::process::Stdio::piped())
            .spawn()?;
        if let Some(stdin) = child.stdin.as_mut() {
            use std::io::Write;
            stdin.write_all(text.as_bytes())?;
        }
        child.wait()?;
    }
    Ok(())
}
