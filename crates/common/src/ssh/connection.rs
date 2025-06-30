//! SSH Connection Management
//!
//! Provides core SSH connection functionality that can be reused across
//! different crates in the Basilica project.

use anyhow::Result;
use async_trait::async_trait;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Duration;
use tokio::time::timeout;
use tracing::{debug, error, info, warn};

/// SSH connection configuration
#[derive(Debug, Clone)]
pub struct SshConnectionConfig {
    /// Connection timeout
    pub connection_timeout: Duration,
    /// Command execution timeout
    pub execution_timeout: Duration,
    /// Maximum file transfer size in bytes
    pub max_transfer_size: u64,
    /// Number of retry attempts
    pub retry_attempts: u32,
    /// Whether to cleanup remote files after operations
    pub cleanup_remote_files: bool,
}

impl Default for SshConnectionConfig {
    fn default() -> Self {
        Self {
            connection_timeout: Duration::from_secs(30),
            execution_timeout: Duration::from_secs(300),
            max_transfer_size: 100 * 1024 * 1024, // 100MB
            retry_attempts: 3,
            cleanup_remote_files: true,
        }
    }
}

/// SSH connection details
#[derive(Debug, Clone)]
pub struct SshConnectionDetails {
    /// Target hostname or IP address
    pub host: String,
    /// SSH username
    pub username: String,
    /// SSH port
    pub port: u16,
    /// Path to private key file
    pub private_key_path: std::path::PathBuf,
    /// Connection timeout
    pub timeout: Duration,
}

/// SSH connection manager trait
#[async_trait]
pub trait SshConnectionManager: Send + Sync {
    /// Test SSH connection
    async fn test_connection(&self, details: &SshConnectionDetails) -> Result<()>;

    /// Execute command on remote host
    async fn execute_command(
        &self,
        details: &SshConnectionDetails,
        command: &str,
        capture_output: bool,
    ) -> Result<String>;

    /// Execute command with retry logic
    async fn execute_command_with_retry(
        &self,
        details: &SshConnectionDetails,
        command: &str,
        capture_output: bool,
    ) -> Result<String>;
}

/// SSH file transfer manager trait
#[async_trait]
pub trait SshFileTransferManager: Send + Sync {
    /// Upload file to remote host
    async fn upload_file(
        &self,
        details: &SshConnectionDetails,
        local_path: &Path,
        remote_path: &str,
    ) -> Result<()>;

    /// Download file from remote host
    async fn download_file(
        &self,
        details: &SshConnectionDetails,
        remote_path: &str,
        local_path: &Path,
    ) -> Result<()>;

    /// Clean up remote files
    async fn cleanup_remote_files(
        &self,
        details: &SshConnectionDetails,
        file_paths: &[String],
    ) -> Result<()>;
}

/// Standard SSH client implementation
pub struct StandardSshClient {
    config: SshConnectionConfig,
}

impl StandardSshClient {
    /// Create a new SSH client with default configuration
    pub fn new() -> Self {
        Self {
            config: SshConnectionConfig::default(),
        }
    }

    /// Create a new SSH client with custom configuration
    pub fn with_config(config: SshConnectionConfig) -> Self {
        Self { config }
    }

    /// Get client configuration
    pub fn config(&self) -> &SshConnectionConfig {
        &self.config
    }

    /// Validate SSH connection details
    fn validate_connection_details(&self, details: &SshConnectionDetails) -> Result<()> {
        if details.host.is_empty() {
            return Err(anyhow::anyhow!("Host cannot be empty"));
        }

        if details.username.is_empty() {
            return Err(anyhow::anyhow!("Username cannot be empty"));
        }

        if !details.private_key_path.exists() {
            return Err(anyhow::anyhow!(
                "Private key not found: {}",
                details.private_key_path.display()
            ));
        }

        Ok(())
    }

    /// Internal SSH command execution
    async fn execute_ssh_command(
        &self,
        details: &SshConnectionDetails,
        command: &str,
        capture_output: bool,
    ) -> Result<String> {
        let mut cmd = Command::new("ssh");
        cmd.arg("-i")
            .arg(&details.private_key_path)
            .arg("-p")
            .arg(details.port.to_string())
            .arg("-o")
            .arg("StrictHostKeyChecking=no")
            .arg("-o")
            .arg("UserKnownHostsFile=/dev/null")
            .arg("-o")
            .arg("BatchMode=yes")
            .arg("-o")
            .arg(format!(
                "ConnectTimeout={}",
                self.config.connection_timeout.as_secs()
            ))
            .arg(format!("{}@{}", details.username, details.host))
            .arg(command);

        if !capture_output {
            cmd.stdout(Stdio::null()).stderr(Stdio::null());
        }

        debug!("Executing SSH command: {:?}", cmd);

        let output = cmd
            .output()
            .map_err(|e| anyhow::anyhow!("Failed to execute SSH command: {}", e))?;

        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            debug!("Command executed successfully");
            Ok(stdout)
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            error!("SSH command failed: {}", stderr);
            Err(anyhow::anyhow!("SSH command failed: {}", stderr))
        }
    }
}

impl Default for StandardSshClient {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SshConnectionManager for StandardSshClient {
    async fn test_connection(&self, details: &SshConnectionDetails) -> Result<()> {
        info!(
            "Testing SSH connection to {}@{}",
            details.username, details.host
        );

        self.validate_connection_details(details)?;

        let result = timeout(
            self.config.connection_timeout,
            self.execute_ssh_command(details, "echo 'connection_test'", true),
        )
        .await;

        match result {
            Ok(Ok(output)) => {
                if output.trim() == "connection_test" {
                    info!("SSH connection test successful");
                    Ok(())
                } else {
                    Err(anyhow::anyhow!("Unexpected response from connection test"))
                }
            }
            Ok(Err(e)) => {
                error!("SSH connection test failed: {}", e);
                Err(e)
            }
            Err(_) => {
                error!("SSH connection test timed out");
                Err(anyhow::anyhow!("Connection test timed out"))
            }
        }
    }

    async fn execute_command(
        &self,
        details: &SshConnectionDetails,
        command: &str,
        capture_output: bool,
    ) -> Result<String> {
        info!("Executing command: {}", command);

        self.validate_connection_details(details)?;

        let result = timeout(
            self.config.execution_timeout,
            self.execute_ssh_command(details, command, capture_output),
        )
        .await;

        match result {
            Ok(result) => result,
            Err(_) => {
                error!("Command execution timed out");
                Err(anyhow::anyhow!("Command execution timed out"))
            }
        }
    }

    async fn execute_command_with_retry(
        &self,
        details: &SshConnectionDetails,
        command: &str,
        capture_output: bool,
    ) -> Result<String> {
        let mut last_error = None;

        for attempt in 1..=self.config.retry_attempts {
            debug!(
                "Command execution attempt {} of {}",
                attempt, self.config.retry_attempts
            );

            match self.execute_command(details, command, capture_output).await {
                Ok(output) => return Ok(output),
                Err(e) => {
                    warn!("Command execution attempt {} failed: {}", attempt, e);
                    last_error = Some(e);

                    if attempt < self.config.retry_attempts {
                        tokio::time::sleep(Duration::from_secs(2)).await;
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("All retry attempts failed")))
    }
}

#[async_trait]
impl SshFileTransferManager for StandardSshClient {
    async fn upload_file(
        &self,
        details: &SshConnectionDetails,
        local_path: &Path,
        remote_path: &str,
    ) -> Result<()> {
        info!(
            "Uploading file {} to {}@{} at {}",
            local_path.display(),
            details.username,
            details.host,
            remote_path
        );

        self.validate_connection_details(details)?;

        if !local_path.exists() {
            return Err(anyhow::anyhow!(
                "Local file not found: {}",
                local_path.display()
            ));
        }

        let file_size = std::fs::metadata(local_path)?.len();
        if file_size > self.config.max_transfer_size {
            return Err(anyhow::anyhow!(
                "File size {} exceeds maximum transfer size {}",
                file_size,
                self.config.max_transfer_size
            ));
        }

        let mut cmd = Command::new("scp");
        cmd.arg("-i")
            .arg(&details.private_key_path)
            .arg("-P")
            .arg(details.port.to_string())
            .arg("-o")
            .arg("StrictHostKeyChecking=no")
            .arg("-o")
            .arg("UserKnownHostsFile=/dev/null")
            .arg("-o")
            .arg(format!(
                "ConnectTimeout={}",
                self.config.connection_timeout.as_secs()
            ))
            .arg(local_path)
            .arg(format!(
                "{}@{}:{}",
                details.username, details.host, remote_path
            ));

        debug!("Executing SCP command: {:?}", cmd);

        let result = timeout(self.config.execution_timeout, async {
            let output = cmd.output()?;
            if output.status.success() {
                Ok(())
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                Err(anyhow::anyhow!("SCP upload failed: {}", stderr))
            }
        })
        .await;

        match result {
            Ok(Ok(())) => {
                info!("File upload successful");
                Ok(())
            }
            Ok(Err(e)) => {
                error!("File upload failed: {}", e);
                Err(e)
            }
            Err(_) => {
                error!("File upload timed out");
                Err(anyhow::anyhow!("File upload timed out"))
            }
        }
    }

    async fn download_file(
        &self,
        details: &SshConnectionDetails,
        remote_path: &str,
        local_path: &Path,
    ) -> Result<()> {
        info!(
            "Downloading file {} from {}@{} to {}",
            remote_path,
            details.username,
            details.host,
            local_path.display()
        );

        self.validate_connection_details(details)?;

        let mut cmd = Command::new("scp");
        cmd.arg("-i")
            .arg(&details.private_key_path)
            .arg("-P")
            .arg(details.port.to_string())
            .arg("-o")
            .arg("StrictHostKeyChecking=no")
            .arg("-o")
            .arg("UserKnownHostsFile=/dev/null")
            .arg("-o")
            .arg(format!(
                "ConnectTimeout={}",
                self.config.connection_timeout.as_secs()
            ))
            .arg(format!(
                "{}@{}:{}",
                details.username, details.host, remote_path
            ))
            .arg(local_path);

        debug!("Executing SCP download command: {:?}", cmd);

        let result = timeout(self.config.execution_timeout, async {
            let output = cmd.output()?;
            if output.status.success() {
                Ok(())
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                Err(anyhow::anyhow!("SCP download failed: {}", stderr))
            }
        })
        .await;

        match result {
            Ok(Ok(())) => {
                info!("File download successful");
                Ok(())
            }
            Ok(Err(e)) => {
                error!("File download failed: {}", e);
                Err(e)
            }
            Err(_) => {
                error!("File download timed out");
                Err(anyhow::anyhow!("File download timed out"))
            }
        }
    }

    async fn cleanup_remote_files(
        &self,
        details: &SshConnectionDetails,
        file_paths: &[String],
    ) -> Result<()> {
        if !self.config.cleanup_remote_files || file_paths.is_empty() {
            return Ok(());
        }

        info!("Cleaning up {} remote files", file_paths.len());

        let rm_command = format!("rm -f {}", file_paths.join(" "));

        match self.execute_command(details, &rm_command, false).await {
            Ok(_) => {
                info!("Remote file cleanup successful");
                Ok(())
            }
            Err(e) => {
                warn!("Remote file cleanup failed: {}", e);
                // Don't fail the entire operation for cleanup errors
                Ok(())
            }
        }
    }
}
