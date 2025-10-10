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
    /// Enable strict host key checking
    pub strict_host_key_checking: bool,
    /// Path to known_hosts file (only used when strict_host_key_checking is true)
    pub known_hosts_file: Option<std::path::PathBuf>,
}

impl Default for SshConnectionConfig {
    fn default() -> Self {
        Self {
            connection_timeout: Duration::from_secs(30),
            execution_timeout: Duration::from_secs(300),
            max_transfer_size: 100 * 1024 * 1024, // 100MB
            retry_attempts: 3,
            cleanup_remote_files: true,
            strict_host_key_checking: false,
            known_hosts_file: None,
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

        if details
            .host
            .contains(&[';', '&', '|', '$', '`', '\n', '\r'][..])
        {
            return Err(anyhow::anyhow!("Host contains invalid characters"));
        }

        if details.username.is_empty() {
            return Err(anyhow::anyhow!("Username cannot be empty"));
        }

        if details
            .username
            .contains(&[';', '&', '|', '$', '`', '\n', '\r', '@'][..])
        {
            return Err(anyhow::anyhow!("Username contains invalid characters"));
        }

        if !details.private_key_path.exists() {
            return Err(anyhow::anyhow!(
                "Private key not found: {}",
                details.private_key_path.display()
            ));
        }

        Ok(())
    }

    /// Format host specification for ssh-keygen and known_hosts operations
    fn format_host_spec(host: &str, port: u16) -> String {
        if port == 22 {
            return host.to_string();
        }

        let is_ipv6 = host.contains(':') && !host.starts_with('[');
        if is_ipv6 {
            format!("[{}]:{}", host, port)
        } else if host.starts_with('[') {
            format!("{}:{}", host, port)
        } else {
            format!("[{}]:{}", host, port)
        }
    }

    /// Remove host key from known_hosts file
    pub async fn remove_host_key(&self, details: &SshConnectionDetails) -> Result<()> {
        let host_spec = Self::format_host_spec(&details.host, details.port);
        let known_hosts_path = self.get_known_hosts_path()?;

        debug!(
            "Removing host key for {} from {}",
            host_spec,
            known_hosts_path.display()
        );

        let output = std::process::Command::new("ssh-keygen")
            .arg("-R")
            .arg(&host_spec)
            .arg("-f")
            .arg(&known_hosts_path)
            .output()
            .map_err(|e| anyhow::anyhow!("Failed to execute ssh-keygen: {}", e))?;

        if output.status.success() {
            debug!("Successfully removed host key for {}", host_spec);
            return Ok(());
        }

        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("not found") || stderr.is_empty() {
            debug!("Host key not found in known_hosts for {}", host_spec);
            return Ok(());
        }

        Err(anyhow::anyhow!(
            "Failed to remove host key for {}: {}",
            host_spec,
            stderr
        ))
    }

    /// Check if host key exists in known_hosts
    fn host_key_exists(&self, details: &SshConnectionDetails) -> Result<bool> {
        let host_spec = Self::format_host_spec(&details.host, details.port);
        let known_hosts_path = self.get_known_hosts_path()?;

        let output = std::process::Command::new("ssh-keygen")
            .arg("-F")
            .arg(&host_spec)
            .arg("-f")
            .arg(&known_hosts_path)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map_err(|e| anyhow::anyhow!("Failed to check host key: {}", e))?;

        Ok(output.success())
    }

    /// Extract SHA256 fingerprints from ssh-keygen output
    fn extract_sha256_fingerprints(output: &str) -> Vec<String> {
        output
            .lines()
            .filter_map(|line| {
                line.find("SHA256:").and_then(|pos| {
                    let rest = &line[pos + 7..];
                    rest.find(' ').map(|end| rest[..end].to_string())
                })
            })
            .collect()
    }

    /// Get current host key fingerprints from remote host
    async fn get_remote_host_fingerprints(
        &self,
        details: &SshConnectionDetails,
    ) -> Result<Vec<String>> {
        let mut cmd = tokio::process::Command::new("ssh-keyscan");
        cmd.arg("-p")
            .arg(details.port.to_string())
            .arg("-T")
            .arg("5")
            .arg("-t")
            .arg("rsa,ed25519,ecdsa")
            .arg(&details.host)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null());

        let output = timeout(Duration::from_secs(10), cmd.output())
            .await
            .map_err(|_| anyhow::anyhow!("Host key scan timeout"))?
            .map_err(|e| anyhow::anyhow!("Failed to scan host key: {}", e))?;

        if !output.status.success() || output.stdout.is_empty() {
            return Err(anyhow::anyhow!("Failed to retrieve remote host key"));
        }

        let mut fingerprint_cmd = std::process::Command::new("ssh-keygen");
        fingerprint_cmd
            .arg("-lf")
            .arg("-")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null());

        let mut child = fingerprint_cmd
            .spawn()
            .map_err(|e| anyhow::anyhow!("Failed to spawn ssh-keygen: {}", e))?;

        if let Some(mut stdin) = child.stdin.take() {
            use std::io::Write;
            stdin
                .write_all(&output.stdout)
                .map_err(|e| anyhow::anyhow!("Failed to write to ssh-keygen stdin: {}", e))?;
            stdin
                .flush()
                .map_err(|e| anyhow::anyhow!("Failed to flush ssh-keygen stdin: {}", e))?;
            drop(stdin);
        }

        let fp_output = child
            .wait_with_output()
            .map_err(|e| anyhow::anyhow!("Failed to get fingerprint: {}", e))?;

        let fingerprint_output = String::from_utf8_lossy(&fp_output.stdout);
        let fingerprints = Self::extract_sha256_fingerprints(&fingerprint_output);

        if fingerprints.is_empty() {
            return Err(anyhow::anyhow!(
                "No fingerprints extracted from remote host"
            ));
        }

        Ok(fingerprints)
    }

    /// Get existing host key fingerprints from known_hosts
    fn get_known_host_fingerprints(&self, details: &SshConnectionDetails) -> Result<Vec<String>> {
        let host_spec = Self::format_host_spec(&details.host, details.port);
        let known_hosts_path = self.get_known_hosts_path()?;

        let output = std::process::Command::new("ssh-keygen")
            .arg("-F")
            .arg(&host_spec)
            .arg("-f")
            .arg(&known_hosts_path)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .output()
            .map_err(|e| anyhow::anyhow!("Failed to check known host: {}", e))?;

        if !output.status.success() {
            return Err(anyhow::anyhow!("Host not found in known_hosts"));
        }

        let known_host_entry = String::from_utf8_lossy(&output.stdout);

        let mut fingerprint_cmd = std::process::Command::new("ssh-keygen");
        fingerprint_cmd
            .arg("-lf")
            .arg("-")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null());

        let mut child = fingerprint_cmd
            .spawn()
            .map_err(|e| anyhow::anyhow!("Failed to spawn ssh-keygen: {}", e))?;

        if let Some(mut stdin) = child.stdin.take() {
            use std::io::Write;
            stdin
                .write_all(known_host_entry.as_bytes())
                .map_err(|e| anyhow::anyhow!("Failed to write to ssh-keygen stdin: {}", e))?;
            stdin
                .flush()
                .map_err(|e| anyhow::anyhow!("Failed to flush ssh-keygen stdin: {}", e))?;
            drop(stdin);
        }

        let fp_output = child
            .wait_with_output()
            .map_err(|e| anyhow::anyhow!("Failed to get fingerprint: {}", e))?;

        let fingerprint_output = String::from_utf8_lossy(&fp_output.stdout);
        let fingerprints = Self::extract_sha256_fingerprints(&fingerprint_output);

        if fingerprints.is_empty() {
            return Err(anyhow::anyhow!(
                "No fingerprints extracted from known_hosts"
            ));
        }

        Ok(fingerprints)
    }

    /// Refresh host key only if it's mismatched or missing
    pub async fn refresh_host_key(&self, details: &SshConnectionDetails) -> Result<()> {
        debug!("Checking host key for {}:{}", details.host, details.port);

        let exists = self.host_key_exists(details)?;

        if !exists {
            debug!(
                "No existing host key for {}:{}, adding new key",
                details.host, details.port
            );
            return self.ensure_host_key_available(details).await;
        }

        let remote_fps = self.get_remote_host_fingerprints(details).await?;
        let known_fps = self.get_known_host_fingerprints(details)?;

        let has_matching_key = remote_fps.iter().any(|rfp| known_fps.contains(rfp));

        if has_matching_key {
            debug!(
                "Host key for {}:{} has matching fingerprints, no refresh needed",
                details.host, details.port
            );
            debug!("Remote fingerprints: {:?}", remote_fps);
            debug!("Known fingerprints: {:?}", known_fps);
            return Ok(());
        }

        warn!(
            "Host key mismatch detected for {}:{}, refreshing",
            details.host, details.port
        );
        debug!("Remote fingerprints: {:?}", remote_fps);
        debug!("Known fingerprints: {:?}", known_fps);

        self.remove_host_key(details).await?;
        self.ensure_host_key_available(details).await?;

        debug!(
            "Successfully refreshed host key for {}:{}",
            details.host, details.port
        );
        Ok(())
    }

    /// Ensure SSH host key is available
    pub async fn ensure_host_key_available(&self, details: &SshConnectionDetails) -> Result<()> {
        debug!(
            "Ensuring host key available for {}:{}",
            details.host, details.port
        );

        let known_hosts_path = self.get_known_hosts_path()?;
        self.ensure_ssh_directory(&known_hosts_path)?;

        let mut cmd = tokio::process::Command::new("ssh-keyscan");
        cmd.arg("-p")
            .arg(details.port.to_string())
            .arg("-T")
            .arg("5")
            .arg("-t")
            .arg("rsa,ed25519,ecdsa")
            .arg(&details.host)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null());

        let output = timeout(Duration::from_secs(10), cmd.output())
            .await
            .map_err(|_| anyhow::anyhow!("Host key scan timeout after 10s"))?
            .map_err(|e| anyhow::anyhow!("Failed to execute ssh-keyscan: {}", e))?;

        if !output.status.success() || output.stdout.is_empty() {
            return Err(anyhow::anyhow!("ssh-keyscan failed or returned no keys"));
        }

        use std::fs::OpenOptions;
        use std::io::Write;

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&known_hosts_path)
            .map_err(|e| anyhow::anyhow!("Failed to open known_hosts: {}", e))?;

        file.write_all(&output.stdout)
            .map_err(|e| anyhow::anyhow!("Failed to write to known_hosts: {}", e))?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&known_hosts_path, std::fs::Permissions::from_mode(0o600))
                .map_err(|e| anyhow::anyhow!("Failed to set known_hosts permissions: {}", e))?;
        }

        debug!(
            "Successfully added host keys for {}:{} to {}",
            details.host,
            details.port,
            known_hosts_path.display()
        );

        Ok(())
    }

    /// Get the path to known_hosts file
    fn get_known_hosts_path(&self) -> Result<std::path::PathBuf> {
        if let Some(ref path) = self.config.known_hosts_file {
            return Ok(path.clone());
        }

        match std::env::var("HOME") {
            Ok(home) => Ok(std::path::PathBuf::from(home)
                .join(".ssh")
                .join("known_hosts")),
            Err(_) => {
                warn!("HOME environment variable not set, using /tmp/known_hosts");
                Ok(std::path::PathBuf::from("/tmp/known_hosts"))
            }
        }
    }

    /// Ensure .ssh directory exists with proper permissions
    fn ensure_ssh_directory(&self, known_hosts_path: &std::path::Path) -> Result<()> {
        if let Some(ssh_dir) = known_hosts_path.parent() {
            std::fs::create_dir_all(ssh_dir)
                .map_err(|e| anyhow::anyhow!("Failed to create .ssh directory: {}", e))?;

            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(ssh_dir, std::fs::Permissions::from_mode(0o700)).map_err(
                    |e| anyhow::anyhow!("Failed to set .ssh directory permissions: {}", e),
                )?;
            }
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
            .arg(details.port.to_string());

        // Configure host key checking based on settings
        if self.config.strict_host_key_checking {
            cmd.arg("-o").arg("StrictHostKeyChecking=yes");
            if let Some(ref known_hosts) = self.config.known_hosts_file {
                cmd.arg("-o")
                    .arg(format!("UserKnownHostsFile={}", known_hosts.display()));
            }
        } else {
            cmd.arg("-o").arg("StrictHostKeyChecking=no");
            cmd.arg("-o").arg("UserKnownHostsFile=/dev/null");
        }

        cmd.arg("-o")
            .arg("IdentitiesOnly=yes")
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
            .arg(details.port.to_string());

        // Configure host key checking based on settings
        if self.config.strict_host_key_checking {
            cmd.arg("-o").arg("StrictHostKeyChecking=yes");
            if let Some(ref known_hosts) = self.config.known_hosts_file {
                cmd.arg("-o")
                    .arg(format!("UserKnownHostsFile={}", known_hosts.display()));
            }
        } else {
            cmd.arg("-o").arg("StrictHostKeyChecking=no");
            cmd.arg("-o").arg("UserKnownHostsFile=/dev/null");
        }

        cmd.arg("-o")
            .arg("IdentitiesOnly=yes")
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
            .arg(details.port.to_string());

        // Configure host key checking based on settings
        if self.config.strict_host_key_checking {
            cmd.arg("-o").arg("StrictHostKeyChecking=yes");
            if let Some(ref known_hosts) = self.config.known_hosts_file {
                cmd.arg("-o")
                    .arg(format!("UserKnownHostsFile={}", known_hosts.display()));
            }
        } else {
            cmd.arg("-o").arg("StrictHostKeyChecking=no");
            cmd.arg("-o").arg("UserKnownHostsFile=/dev/null");
        }

        cmd.arg("-o")
            .arg("IdentitiesOnly=yes")
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

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_ssh_details() -> SshConnectionDetails {
        SshConnectionDetails {
            host: "test.example.com".to_string(),
            username: "testuser".to_string(),
            port: 22,
            private_key_path: std::path::PathBuf::from("/tmp/test_key"),
            timeout: Duration::from_secs(30),
        }
    }

    #[test]
    fn test_host_spec_formatting_standard_port() {
        let details = create_test_ssh_details();

        let host_spec = StandardSshClient::format_host_spec(&details.host, details.port);

        assert_eq!(host_spec, "test.example.com");
    }

    #[test]
    fn test_host_spec_formatting_custom_port() {
        let mut details = create_test_ssh_details();
        details.port = 2222;

        let host_spec = StandardSshClient::format_host_spec(&details.host, details.port);

        assert_eq!(host_spec, "[test.example.com]:2222");
    }

    #[test]
    fn test_ssh_connection_config_default() {
        let config = SshConnectionConfig::default();
        assert_eq!(config.connection_timeout, Duration::from_secs(30));
        assert_eq!(config.execution_timeout, Duration::from_secs(300));
        assert_eq!(config.max_transfer_size, 100 * 1024 * 1024);
        assert_eq!(config.retry_attempts, 3);
        assert!(config.cleanup_remote_files);
        assert!(!config.strict_host_key_checking);
        assert!(config.known_hosts_file.is_none());
    }

    #[test]
    fn test_connection_details_validation() {
        let client = StandardSshClient::new();

        let mut details = create_test_ssh_details();
        details.host = String::new();

        let result = client.validate_connection_details(&details);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Host cannot be empty"));

        let mut details = create_test_ssh_details();
        details.username = String::new();

        let result = client.validate_connection_details(&details);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Username cannot be empty"));
    }

    #[test]
    fn test_extract_sha256_fingerprints() {
        let output = "3072 SHA256:uNiVztksCsDhcc0u9e8BujQXVUpKZIDTMczCvj3tD2s github.com (RSA)\n\
                      256 SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU github.com (ED25519)\n\
                      256 SHA256:p2QAMXNIC1TJYWeIOttrVc98/R1BUFWu3/LiyKgUfQM |1|hash| (ECDSA)";

        let fingerprints = StandardSshClient::extract_sha256_fingerprints(output);

        assert_eq!(fingerprints.len(), 3);
        assert!(fingerprints.contains(&"uNiVztksCsDhcc0u9e8BujQXVUpKZIDTMczCvj3tD2s".to_string()));
        assert!(fingerprints.contains(&"+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU".to_string()));
        assert!(fingerprints.contains(&"p2QAMXNIC1TJYWeIOttrVc98/R1BUFWu3/LiyKgUfQM".to_string()));
    }

    #[test]
    fn test_extract_sha256_fingerprints_empty() {
        let output = "No SHA256 fingerprints here";
        let fingerprints = StandardSshClient::extract_sha256_fingerprints(output);
        assert!(fingerprints.is_empty());
    }

    #[test]
    fn test_extract_sha256_fingerprints_mixed() {
        let output = "# Host github.com found: line 1\n\
                      256 SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU |1|hash| (ED25519)\n\
                      # Comment line\n\
                      3072 SHA256:uNiVztksCsDhcc0u9e8BujQXVUpKZIDTMczCvj3tD2s |1|hash| (RSA)";

        let fingerprints = StandardSshClient::extract_sha256_fingerprints(output);

        assert_eq!(fingerprints.len(), 2);
        assert!(fingerprints.contains(&"+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU".to_string()));
        assert!(fingerprints.contains(&"uNiVztksCsDhcc0u9e8BujQXVUpKZIDTMczCvj3tD2s".to_string()));
    }

    #[test]
    fn test_extract_sha256_fingerprints_real_scenario() {
        let output =
            "256 SHA256:ZRvRYFEFyp5VGOwzrIhCEYHTQI4Gk6z0by/qD8bIAFE 31.22.104.140 (ED25519)\n\
                      3072 SHA256:wJjqSeEKT4m8Oz9lM7l1I6GMlJKFfh3ozKM9W5g/mVQ 31.22.104.140 (RSA)";

        let fingerprints = StandardSshClient::extract_sha256_fingerprints(output);

        assert_eq!(fingerprints.len(), 2);
        assert!(fingerprints.contains(&"ZRvRYFEFyp5VGOwzrIhCEYHTQI4Gk6z0by/qD8bIAFE".to_string()));
        assert!(fingerprints.contains(&"wJjqSeEKT4m8Oz9lM7l1I6GMlJKFfh3ozKM9W5g/mVQ".to_string()));
    }

    #[test]
    fn test_host_spec_port_22_no_brackets() {
        let details = SshConnectionDetails {
            host: "31.22.104.140".to_string(),
            username: "ubuntu".to_string(),
            port: 22,
            private_key_path: std::path::PathBuf::from("/tmp/key"),
            timeout: Duration::from_secs(30),
        };

        let host_spec = StandardSshClient::format_host_spec(&details.host, details.port);

        assert_eq!(host_spec, "31.22.104.140");
    }

    #[test]
    fn test_host_spec_custom_port_with_brackets() {
        let details = SshConnectionDetails {
            host: "31.22.104.140".to_string(),
            username: "ubuntu".to_string(),
            port: 2222,
            private_key_path: std::path::PathBuf::from("/tmp/key"),
            timeout: Duration::from_secs(30),
        };

        let host_spec = StandardSshClient::format_host_spec(&details.host, details.port);

        assert_eq!(host_spec, "[31.22.104.140]:2222");
    }

    #[test]
    fn test_host_spec_ipv6_port_22() {
        let host_spec = StandardSshClient::format_host_spec("2001:db8::1", 22);
        assert_eq!(host_spec, "2001:db8::1");
    }

    #[test]
    fn test_host_spec_ipv6_custom_port() {
        let host_spec = StandardSshClient::format_host_spec("2001:db8::1", 2222);
        assert_eq!(host_spec, "[2001:db8::1]:2222");
    }

    #[test]
    fn test_host_spec_ipv6_already_bracketed() {
        let host_spec = StandardSshClient::format_host_spec("[2001:db8::1]", 2222);
        assert_eq!(host_spec, "[2001:db8::1]:2222");
    }
}
