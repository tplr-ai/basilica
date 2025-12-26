//! Command execution and actions
//!
//! These functions are scaffolding for future TUI features.
#![allow(dead_code)]

use anyhow::Result;
use std::process::Command;

/// Execute SSH connection to a rental
pub async fn ssh_connect(host: &str, port: u16, user: &str) -> Result<()> {
    // Build SSH command
    let ssh_cmd = format!("ssh -p {} {}@{}", port, user, host);

    tracing::info!("Connecting via SSH: {}", ssh_cmd);

    // Note: In a real TUI, we'd need to:
    // 1. Exit the TUI temporarily
    // 2. Run SSH
    // 3. Re-enter the TUI when SSH exits
    // For now, just log the intent

    Ok(())
}

/// Execute a command on a rental via SSH
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

/// Copy files to/from a rental
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
