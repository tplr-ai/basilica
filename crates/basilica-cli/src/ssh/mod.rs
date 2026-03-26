//! SSH operations module

mod key_matcher;

pub use key_matcher::{find_local_public_key_path, find_private_key_for_public_key};

use crate::config::SshConfig;
use crate::error::{CliError, Result};
use basilica_common::ssh::{
    SshConnectionConfig, SshConnectionDetails, SshConnectionManager, SshFileTransferManager,
    StandardSshClient,
};
use basilica_sdk::types::{RentalStatusResponse, SshAccess};
use color_eyre::eyre::eyre;
use color_eyre::Section;
use std::path::Path;
use std::time::Duration;
use tracing::{debug, info};

/// SSH client for rental operations
pub struct SshClient {
    client: StandardSshClient,
    config: SshConfig,
}

/// Result of a non-interactive SSH readiness probe.
pub enum SshProbeStatus {
    /// SSH is reachable and key auth succeeded without interaction.
    Ready,
    /// SSH is reachable but requires interactive auth (e.g. encrypted key without agent).
    ReadyAuthRequired,
    /// SSH is not yet reachable or rejected the connection for other reasons.
    NotReady(String),
}

impl SshClient {
    /// Create new SSH client
    pub fn new(config: &SshConfig) -> Result<Self> {
        // Create SSH connection config using configured timeout
        let connection_timeout = if config.connection_timeout > 0 {
            Duration::from_secs(config.connection_timeout)
        } else {
            Duration::from_secs(30) // Default fallback
        };

        let ssh_config = SshConnectionConfig {
            connection_timeout,
            execution_timeout: Duration::from_secs(3600),
            retry_attempts: 3,
            max_transfer_size: 1000 * 1024 * 1024, // 1000MB
            cleanup_remote_files: false,
            strict_host_key_checking: false,
            known_hosts_file: None,
        };

        Ok(Self {
            client: StandardSshClient::with_config(ssh_config),
            config: config.clone(),
        })
    }

    /// Convert SSH access info to connection details
    fn ssh_access_to_connection_details(
        &self,
        ssh_access: &SshAccess,
        private_key_path: std::path::PathBuf,
    ) -> Result<SshConnectionDetails> {
        if !private_key_path.exists() {
            return Err(eyre!(
                "SSH private key not found at: {}",
                private_key_path.display()
            )
            .suggestion("Generate SSH keys with 'basilica ssh-keys generate' or 'ssh-keygen -t ed25519 -f ~/.ssh/basilica_ed25519'")
            .into());
        }

        Ok(SshConnectionDetails {
            host: ssh_access.host.clone(),
            port: ssh_access.port,
            username: ssh_access.username.clone(),
            private_key_path,
            timeout: if self.config.connection_timeout > 0 {
                Duration::from_secs(self.config.connection_timeout)
            } else {
                Duration::from_secs(30) // Default fallback
            },
        })
    }

    /// Execute a command via SSH
    pub async fn execute_command(
        &self,
        ssh_access: &SshAccess,
        command: &str,
        private_key_path: std::path::PathBuf,
    ) -> Result<()> {
        let details = self.ssh_access_to_connection_details(ssh_access, private_key_path)?;

        debug!(
            "Executing command via SSH: ssh -i {} -p {} {}@{} '{}'",
            details.private_key_path.display(),
            details.port,
            details.username,
            details.host,
            command
        );

        let output = self
            .client
            .execute_command(&details, command, true)
            .await
            .map_err(|e| {
                eyre!("Command execution failed: {}", e)
                    .suggestion("Check if the rental is still active and SSH port is exposed")
                    .note("Run 'basilica status <rental-id>' to check rental status")
            })?;

        println!("{}", output);
        Ok(())
    }

    /// Execute a command with rental status (for backward compatibility)
    pub async fn execute_command_with_rental(
        &self,
        _rental: &RentalStatusResponse,
        _command: &str,
    ) -> Result<()> {
        Err(eyre!(
            "SSH access details must be provided separately - use execute_command with SshAccess"
        )
        .into())
    }

    /// Test SSH connectivity without starting an interactive session.
    /// Returns Ok(()) if connection succeeds, Err with the error message if it fails.
    /// This method captures stderr to avoid printing raw SSH error messages.
    ///
    /// Uses a timeout wrapper to prevent hanging if the SSH process doesn't exit cleanly.
    pub async fn test_connection(
        &self,
        ssh_access: &SshAccess,
        private_key_path: std::path::PathBuf,
    ) -> Result<()> {
        let details = self.ssh_access_to_connection_details(ssh_access, private_key_path)?;
        let timeout_secs = details.timeout.as_secs();

        debug!(
            "Testing SSH connectivity to {}@{}:{}",
            details.username, details.host, details.port
        );

        // Build SSH command for connectivity test
        let mut cmd = std::process::Command::new("ssh");
        cmd.arg("-i")
            .arg(details.private_key_path.display().to_string())
            .arg("-p")
            .arg(details.port.to_string())
            .arg("-o")
            .arg("StrictHostKeyChecking=no")
            .arg("-o")
            .arg("UserKnownHostsFile=/dev/null")
            .arg("-o")
            .arg("LogLevel=error")
            .arg("-o")
            .arg("BatchMode=yes")
            .arg("-o")
            .arg(format!("ConnectTimeout={}", timeout_secs))
            .arg(format!("{}@{}", details.username, details.host))
            .arg("exit")
            .arg("0");

        // Run with timeout to prevent hanging on stream close
        // The extra 5 seconds accounts for SSH connection overhead beyond ConnectTimeout
        let timeout_duration = Duration::from_secs(timeout_secs + 5);
        let result = tokio::time::timeout(timeout_duration, async {
            tokio::task::spawn_blocking(move || cmd.output())
                .await
                .map_err(|e| eyre!("Task join error: {}", e))?
                .map_err(|e| eyre!("Failed to run SSH command: {}", e))
        })
        .await;

        match result {
            Ok(Ok(output)) => {
                if output.status.success() {
                    Ok(())
                } else {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    Err(eyre!("SSH connection test failed: {}", stderr.trim()).into())
                }
            }
            Ok(Err(e)) => Err(e.into()),
            Err(_) => Err(eyre!("SSH connection test timed out").into()),
        }
    }

    /// Probe SSH readiness without triggering any interactive prompts.
    /// Returns Ready if SSH accepts a non-interactive key, ReadyAuthRequired if SSH is up
    /// but interactive auth is required (e.g. encrypted key without agent), or NotReady otherwise.
    pub async fn try_connect_silently(
        &self,
        ssh_access: &SshAccess,
        private_key_path: std::path::PathBuf,
    ) -> Result<SshProbeStatus> {
        let details = self.ssh_access_to_connection_details(ssh_access, private_key_path)?;
        let timeout_secs = std::cmp::min(details.timeout.as_secs(), 10); // Quick timeout for retry

        debug!(
            "Trying silent SSH connection to {}@{}:{}",
            details.username, details.host, details.port
        );

        let mut cmd = std::process::Command::new("ssh");
        cmd.arg("-i")
            .arg(details.private_key_path.display().to_string())
            .arg("-p")
            .arg(details.port.to_string())
            .arg("-o")
            .arg("StrictHostKeyChecking=no")
            .arg("-o")
            .arg("UserKnownHostsFile=/dev/null")
            .arg("-o")
            .arg("LogLevel=error")
            .arg("-o")
            .arg("BatchMode=yes")
            .arg("-o")
            .arg("PreferredAuthentications=publickey")
            .arg("-o")
            .arg("PasswordAuthentication=no")
            .arg("-o")
            .arg("KbdInteractiveAuthentication=no")
            .arg("-o")
            .arg(format!("ConnectTimeout={}", timeout_secs))
            .arg(format!("{}@{}", details.username, details.host))
            .arg("true"); // Quick command that exits immediately

        // Disable stdin to prevent passphrase prompts; capture stderr for classification
        cmd.stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped());

        let output = cmd
            .output()
            .map_err(|e| -> CliError { eyre!("Failed to run SSH: {}", e).into() })?;

        if output.status.success() {
            return Ok(SshProbeStatus::Ready);
        }

        let stderr = String::from_utf8_lossy(&output.stderr);
        let stderr_lc = stderr.to_lowercase();

        if stderr_lc.contains("permission denied")
            || stderr_lc.contains("no supported authentication methods")
            || stderr_lc.contains("authentication failed")
        {
            return Ok(SshProbeStatus::ReadyAuthRequired);
        }

        Ok(SshProbeStatus::NotReady(stderr.trim().to_string()))
    }

    /// Open interactive SSH session
    pub async fn interactive_session(
        &self,
        ssh_access: &SshAccess,
        private_key_path: std::path::PathBuf,
    ) -> Result<()> {
        let details = self.ssh_access_to_connection_details(ssh_access, private_key_path)?;

        info!(
            "Opening SSH session to {}@{}",
            ssh_access.username, ssh_access.host
        );

        debug!(
            "Running interactive SSH to {}@{}:{}",
            details.username, details.host, details.port
        );

        // Use SSH command directly with proper arguments for TTY support
        let mut cmd = std::process::Command::new("ssh");
        cmd.arg("-i")
            .arg(details.private_key_path.display().to_string())
            .arg("-p")
            .arg(details.port.to_string())
            .arg("-o")
            .arg("StrictHostKeyChecking=no")
            .arg("-o")
            .arg("UserKnownHostsFile=/dev/null")
            .arg("-o")
            .arg("LogLevel=error")
            .arg(format!("{}@{}", details.username, details.host));

        debug!(
            "Executing SSH command: ssh -i {} -p {} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=error {}@{}",
            details.private_key_path.display(),
            details.port,
            details.username,
            details.host
        );

        let status = cmd.status().map_err(|e| -> CliError {
            eyre!("Failed to start SSH session: {}", e)
                .suggestion("Check your SSH key permissions and network connectivity")
                .note("Ensure the rental is active and accessible")
                .into()
        })?;

        // Only treat exit code 255 as an SSH error (SSH's own error code)
        // Other exit codes are from the remote command
        if status.code() == Some(255) {
            return Err(eyre!("SSH connection failed")
                .suggestion("Check if the rental is still active and SSH port is exposed")
                .note("Run 'basilica status <rental-id>' to check rental status")
                .into());
        }

        Ok(())
    }

    /// Parse port forward specification into components
    fn parse_port_forward_spec<'a>(
        spec: &'a str,
        forward_type: &str,
    ) -> Result<(u16, &'a str, u16)> {
        // Use splitn for more efficient parsing - stops after finding 3 parts
        let mut parts = spec.splitn(3, ':');

        let port1_str = parts.next().ok_or_else(|| -> crate::error::CliError {
            eyre!(
                "Invalid {} forward specification: {}. Expected format: port:host:port",
                forward_type,
                spec
            )
            .into()
        })?;

        let host = parts.next().ok_or_else(|| -> crate::error::CliError {
            eyre!(
                "Invalid {} forward specification: {}. Expected format: port:host:port",
                forward_type,
                spec
            )
            .into()
        })?;

        let port2_str = parts.next().ok_or_else(|| -> crate::error::CliError {
            eyre!(
                "Invalid {} forward specification: {}. Expected format: port:host:port",
                forward_type,
                spec
            )
            .into()
        })?;

        // Parse and validate port numbers
        let port1 = port1_str
            .parse::<u16>()
            .map_err(|_| -> crate::error::CliError {
                eyre!(
                    "Invalid port number '{}' in {} forward spec: {}",
                    port1_str,
                    forward_type,
                    spec
                )
                .into()
            })?;

        let port2 = port2_str
            .parse::<u16>()
            .map_err(|_| -> crate::error::CliError {
                eyre!(
                    "Invalid port number '{}' in {} forward spec: {}",
                    port2_str,
                    forward_type,
                    spec
                )
                .into()
            })?;

        Ok((port1, host, port2))
    }

    /// Open interactive SSH session with port forwarding options
    pub async fn interactive_session_with_options(
        &self,
        ssh_access: &SshAccess,
        options: &crate::cli::commands::SshOptions,
        private_key_path: std::path::PathBuf,
    ) -> Result<()> {
        let details = self.ssh_access_to_connection_details(ssh_access, private_key_path)?;

        info!(
            "Opening SSH session to {}@{}",
            ssh_access.username, ssh_access.host
        );

        if !options.local_forward.is_empty() {
            info!("Local port forwarding enabled");
        }
        if !options.remote_forward.is_empty() {
            info!("Remote port forwarding enabled");
        }

        debug!(
            "Running interactive SSH to {}@{}:{}",
            details.username, details.host, details.port
        );

        // Use SSH command directly with proper arguments for TTY support
        let mut cmd = std::process::Command::new("ssh");
        cmd.arg("-i")
            .arg(details.private_key_path.display().to_string())
            .arg("-p")
            .arg(details.port.to_string())
            .arg("-o")
            .arg("StrictHostKeyChecking=no")
            .arg("-o")
            .arg("UserKnownHostsFile=/dev/null")
            .arg("-o")
            .arg("LogLevel=error");

        // Add local port forwarding arguments
        for forward_spec in &options.local_forward {
            // Validate format: local_port:remote_host:remote_port
            let (_local_port, _host, _remote_port) =
                Self::parse_port_forward_spec(forward_spec, "local")?;

            cmd.arg("-L").arg(forward_spec);
            debug!("Added local port forward: {}", forward_spec);
        }

        // Add remote port forwarding arguments
        for forward_spec in &options.remote_forward {
            // Validate format: remote_port:local_host:local_port
            let (_remote_port, _host, _local_port) =
                Self::parse_port_forward_spec(forward_spec, "remote")?;

            cmd.arg("-R").arg(forward_spec);
            debug!("Added remote port forward: {}", forward_spec);
        }

        // Add the target host
        cmd.arg(format!("{}@{}", details.username, details.host));

        // Log the complete command
        let mut cmd_str = format!(
            "ssh -i {} -p {} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=error",
            details.private_key_path.display(),
            details.port
        );
        for forward_spec in &options.local_forward {
            cmd_str.push_str(&format!(" -L {}", forward_spec));
        }
        for forward_spec in &options.remote_forward {
            cmd_str.push_str(&format!(" -R {}", forward_spec));
        }
        cmd_str.push_str(&format!(" {}@{}", details.username, details.host));
        debug!("Executing SSH command with options: {}", cmd_str);

        let status = cmd.status().map_err(|e| -> CliError {
            eyre!("Failed to start SSH session: {}", e)
                .suggestion("Check your SSH key permissions and network connectivity")
                .note("Ensure the rental is active and accessible")
                .into()
        })?;

        // Only treat exit code 255 as an SSH error (SSH's own error code)
        // Other exit codes are from the remote command and should be ignored
        if status.code() == Some(255) {
            return Err(eyre!("SSH connection failed")
                .suggestion("Check if the rental is still active and SSH port is exposed")
                .note("Run 'basilica status <rental-id>' to check rental status")
                .into());
        }

        Ok(())
    }

    /// Upload file via SSH
    pub async fn upload_file(
        &self,
        ssh_access: &SshAccess,
        local_path: &str,
        remote_path: &str,
        private_key_path: std::path::PathBuf,
    ) -> Result<()> {
        let details = self.ssh_access_to_connection_details(ssh_access, private_key_path)?;
        let local = Path::new(local_path);

        info!("Uploading {} to {}", local_path, ssh_access.host);

        self.client
            .upload_file(&details, local, remote_path)
            .await
            .map_err(|e| {
                eyre!("File upload failed: {}", e)
                    .suggestion("Check file permissions and available disk space on the rental")
                    .note("Ensure the local file exists and is readable")
            })?;

        info!("Upload completed successfully");
        Ok(())
    }

    /// Download file via SSH
    pub async fn download_file(
        &self,
        ssh_access: &SshAccess,
        remote_path: &str,
        local_path: &str,
        private_key_path: std::path::PathBuf,
    ) -> Result<()> {
        let details = self.ssh_access_to_connection_details(ssh_access, private_key_path)?;
        let local = Path::new(local_path);

        info!("Downloading {} from {}", remote_path, ssh_access.host);

        self.client
            .download_file(&details, remote_path, local)
            .await
            .map_err(|e| {
                eyre!("File download failed: {}", e)
                    .suggestion("Check that the remote file exists and you have read permissions")
                    .note("Ensure the destination directory is writable")
            })?;

        info!("Download completed successfully");
        Ok(())
    }
}

/// Parse SSH credentials string into components
pub fn parse_ssh_credentials(credentials: &str) -> Result<(String, u16, String)> {
    debug!("Parsing SSH credentials: {}", credentials);
    // Expected format: "ssh user@host -p port" or "user@host:port" or "host:port"

    // Try to parse "ssh user@host -p port" format
    if credentials.starts_with("ssh ") {
        let parts: Vec<&str> = credentials.split_whitespace().collect();
        if parts.len() >= 4 && parts[2] == "-p" {
            let user_host = parts[1];
            let port = parts[3]
                .parse::<u16>()
                .map_err(|_| eyre!("Invalid port in SSH credentials"))?;

            let (user, host) = if let Some((user, host)) = user_host.split_once('@') {
                (user.to_string(), host.to_string())
            } else {
                ("root".to_string(), user_host.to_string())
            };

            return Ok((host, port, user));
        }
    }

    // Strip "ssh " prefix if present for remaining formats
    let credentials_without_prefix = credentials.trim_start_matches("ssh ");

    // Try to parse "user@host:port" or "host:port" format
    if let Some((left_part, port_str)) = credentials_without_prefix.rsplit_once(':') {
        let port = port_str
            .parse::<u16>()
            .map_err(|_| eyre!("Invalid port in SSH credentials"))?;

        let (user, host) = if let Some((user, host)) = left_part.split_once('@') {
            (user.to_string(), host.to_string())
        } else {
            ("root".to_string(), left_part.to_string())
        };

        return Ok((host, port, user));
    }

    // Try to parse "user@host" or just "host" format (default port 22)
    let (user, host) = if let Some((user, host)) = credentials_without_prefix.split_once('@') {
        (user.to_string(), host.to_string())
    } else {
        ("root".to_string(), credentials_without_prefix.to_string())
    };

    Ok((host, 22, user))
}
