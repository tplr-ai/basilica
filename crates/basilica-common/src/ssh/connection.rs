//! SSH Connection Management
//!
//! Provides core SSH connection functionality that can be reused across
//! different crates in the Basilica project.

use anyhow::Result;
use async_trait::async_trait;
use std::path::Path;
use std::process::{ExitStatus, Stdio};
use std::time::Duration;
use tokio::io::AsyncReadExt;
use tokio::process::{Child, Command};
use tokio::time::timeout;
use tracing::{debug, info, warn};

/// SSH connection configuration
#[derive(Debug, Clone)]
pub struct SshConnectionConfig {
    /// Connection timeout
    pub connection_timeout: Duration,
    /// Optional command and file-transfer execution timeout
    pub execution_timeout: Option<Duration>,
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
            execution_timeout: None,
            max_transfer_size: 100 * 1024 * 1024, // 100MB
            retry_attempts: 3,
            cleanup_remote_files: true,
            strict_host_key_checking: false,
            known_hosts_file: None,
        }
    }
}

/// Outcome of an SSH command whose output is passed through to the caller.
#[derive(Debug)]
pub enum SshCommandStatus {
    /// The SSH subprocess exited normally with this status.
    Exited(ExitStatus),
    /// The configured execution timeout expired and the SSH subprocess was reaped.
    TimedOut,
}

#[derive(Debug)]
struct CapturedCommandOutput {
    status: SshCommandStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
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

    fn host_key_options(&self) -> Result<Vec<String>> {
        let mut options = Vec::new();

        if self.config.strict_host_key_checking {
            options.push("StrictHostKeyChecking=yes".to_string());
            let known_hosts = self.get_known_hosts_path()?;
            options.push(format!("UserKnownHostsFile={}", known_hosts.display()));
        } else {
            options.push("StrictHostKeyChecking=no".to_string());
            options.push("UserKnownHostsFile=/dev/null".to_string());
        }

        Ok(options)
    }

    fn connection_options(&self, include_batch_mode: bool) -> Result<Vec<String>> {
        let mut options = self.host_key_options()?;
        options.push("LogLevel=ERROR".to_string());
        options.push("IdentitiesOnly=yes".to_string());
        if include_batch_mode {
            options.push("BatchMode=yes".to_string());
        }
        options.push(format!(
            "ConnectTimeout={}",
            self.config.connection_timeout.as_secs()
        ));
        options.push("ServerAliveInterval=15".to_string());
        options.push("ServerAliveCountMax=3".to_string());
        Ok(options)
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

    /// Execute command and return streaming Child process for real-time output
    /// This is useful for long-running commands like log streaming
    pub async fn execute_command_streaming(
        &self,
        details: &SshConnectionDetails,
        command: &str,
    ) -> Result<tokio::process::Child> {
        self.validate_connection_details(details)?;

        let mut cmd = tokio::process::Command::new("ssh");
        cmd.arg("-i")
            .arg(&details.private_key_path)
            .arg("-p")
            .arg(details.port.to_string());

        for option in self.connection_options(true)? {
            cmd.arg("-o").arg(option);
        }

        cmd.arg(format!("{}@{}", details.username, details.host))
            .arg(command);

        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        debug!("Spawning SSH streaming command");

        cmd.kill_on_drop(true);
        cmd.spawn()
            .map_err(|e| anyhow::anyhow!("Failed to spawn SSH streaming command: {}", e))
    }

    async fn wait_for_child(&self, child: &mut Child) -> Result<SshCommandStatus> {
        let Some(execution_timeout) = self.config.execution_timeout else {
            return child
                .wait()
                .await
                .map(SshCommandStatus::Exited)
                .map_err(Into::into);
        };

        match timeout(execution_timeout, child.wait()).await {
            Ok(status) => Ok(SshCommandStatus::Exited(status?)),
            Err(_) => {
                child.start_kill().map_err(|error| {
                    anyhow::anyhow!("Failed to kill timed-out subprocess: {}", error)
                })?;
                child.wait().await.map_err(|error| {
                    anyhow::anyhow!("Failed to reap timed-out subprocess: {}", error)
                })?;
                Ok(SshCommandStatus::TimedOut)
            }
        }
    }

    async fn execute_process(
        &self,
        command: &mut Command,
        capture_output: bool,
    ) -> Result<CapturedCommandOutput> {
        command.stdin(Stdio::null());
        if capture_output {
            command.stdout(Stdio::piped()).stderr(Stdio::piped());
        } else {
            command.stdout(Stdio::null()).stderr(Stdio::null());
        }
        command.kill_on_drop(true);

        let mut child = command.spawn()?;
        let stdout_task = child.stdout.take().map(|mut stdout| {
            tokio::spawn(async move {
                let mut output = Vec::new();
                stdout.read_to_end(&mut output).await?;
                Ok::<_, std::io::Error>(output)
            })
        });
        let stderr_task = child.stderr.take().map(|mut stderr| {
            tokio::spawn(async move {
                let mut output = Vec::new();
                stderr.read_to_end(&mut output).await?;
                Ok::<_, std::io::Error>(output)
            })
        });

        let status = self.wait_for_child(&mut child).await?;
        let stdout = match stdout_task {
            Some(task) => task.await??,
            None => Vec::new(),
        };
        let stderr = match stderr_task {
            Some(task) => task.await??,
            None => Vec::new(),
        };

        Ok(CapturedCommandOutput {
            status,
            stdout,
            stderr,
        })
    }

    /// Execute an SSH command with byte-for-byte stdout and stderr pass-through.
    pub async fn execute_command_passthrough(
        &self,
        details: &SshConnectionDetails,
        command: &str,
    ) -> Result<SshCommandStatus> {
        self.execute_command_passthrough_with_program(details, command, Path::new("ssh"))
            .await
    }

    async fn execute_command_passthrough_with_program(
        &self,
        details: &SshConnectionDetails,
        command: &str,
        program: &Path,
    ) -> Result<SshCommandStatus> {
        self.execute_command_with_program_and_stdio(
            details,
            command,
            program,
            Stdio::inherit(),
            Stdio::inherit(),
        )
        .await
    }

    async fn execute_command_with_program_and_stdio(
        &self,
        details: &SshConnectionDetails,
        command: &str,
        program: &Path,
        stdout: Stdio,
        stderr: Stdio,
    ) -> Result<SshCommandStatus> {
        self.validate_connection_details(details)?;

        let mut cmd = Command::new(program);
        cmd.arg("-i")
            .arg(&details.private_key_path)
            .arg("-p")
            .arg(details.port.to_string());

        for option in self.connection_options(true)? {
            cmd.arg("-o").arg(option);
        }

        cmd.arg(format!("{}@{}", details.username, details.host))
            .arg(command)
            .stdin(Stdio::null())
            .stdout(stdout)
            .stderr(stderr);

        debug!("Executing SSH command with pass-through output");

        cmd.kill_on_drop(true);
        let mut child = cmd
            .spawn()
            .map_err(|e| anyhow::anyhow!("Failed to spawn SSH command: {}", e))?;
        self.wait_for_child(&mut child).await
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

        for option in self.connection_options(true)? {
            cmd.arg("-o").arg(option);
        }

        cmd.arg(format!("{}@{}", details.username, details.host))
            .arg(command);

        debug!("Executing SSH command");

        let output = self
            .execute_process(&mut cmd, capture_output)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to execute SSH command: {}", e))?;

        match output.status {
            SshCommandStatus::Exited(status) if status.success() => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                debug!("Command executed successfully");
                Ok(stdout)
            }
            SshCommandStatus::Exited(_) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                debug!("SSH command failed: {}", stderr);
                Err(anyhow::anyhow!("SSH command failed: {}", stderr))
            }
            SshCommandStatus::TimedOut => {
                debug!("Command execution timed out");
                Err(anyhow::anyhow!("Command execution timed out"))
            }
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

        match self
            .execute_ssh_command(details, "echo 'connection_test'", true)
            .await
        {
            Ok(output) => {
                if output.trim() == "connection_test" {
                    info!("SSH connection test successful");
                    Ok(())
                } else {
                    Err(anyhow::anyhow!("Unexpected response from connection test"))
                }
            }
            Err(e) => {
                debug!("SSH connection test failed: {}", e);
                Err(e)
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

        self.execute_ssh_command(details, command, capture_output)
            .await
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

        for option in self.connection_options(false)? {
            cmd.arg("-o").arg(option);
        }

        cmd.arg(local_path).arg(format!(
            "{}@{}:{}",
            details.username, details.host, remote_path
        ));

        debug!("Executing SCP upload command");

        let output = self.execute_process(&mut cmd, true).await?;

        match output.status {
            SshCommandStatus::Exited(status) if status.success() => {
                info!("File upload successful");
                Ok(())
            }
            SshCommandStatus::Exited(_) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                debug!("File upload failed: {}", stderr);
                Err(anyhow::anyhow!("SCP upload failed: {}", stderr))
            }
            SshCommandStatus::TimedOut => {
                debug!("File upload timed out");
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

        for option in self.connection_options(false)? {
            cmd.arg("-o").arg(option);
        }

        cmd.arg(format!(
            "{}@{}:{}",
            details.username, details.host, remote_path
        ))
        .arg(local_path);

        debug!("Executing SCP download command");

        let output = self.execute_process(&mut cmd, true).await?;

        match output.status {
            SshCommandStatus::Exited(status) if status.success() => {
                info!("File download successful");
                Ok(())
            }
            SshCommandStatus::Exited(_) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                debug!("File download failed: {}", stderr);
                Err(anyhow::anyhow!("SCP download failed: {}", stderr))
            }
            SshCommandStatus::TimedOut => {
                debug!("File download timed out");
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

    fn default_known_hosts_path_for_test() -> std::path::PathBuf {
        match std::env::var("HOME") {
            Ok(home) => std::path::PathBuf::from(home)
                .join(".ssh")
                .join("known_hosts"),
            Err(_) => std::path::PathBuf::from("/tmp/known_hosts"),
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
        assert_eq!(config.execution_timeout, None);
        assert_eq!(config.max_transfer_size, 100 * 1024 * 1024);
        assert_eq!(config.retry_attempts, 3);
        assert!(config.cleanup_remote_files);
        assert!(!config.strict_host_key_checking);
        assert!(config.known_hosts_file.is_none());
    }

    #[tokio::test]
    async fn process_execution_captures_streams_and_nonzero_status() {
        let client = StandardSshClient::new();
        let mut command = Command::new("sh");
        command
            .arg("-c")
            .arg("printf stdout-bytes; printf stderr-bytes >&2; exit 3");

        let output = client.execute_process(&mut command, true).await.unwrap();

        match output.status {
            SshCommandStatus::Exited(status) => assert_eq!(status.code(), Some(3)),
            SshCommandStatus::TimedOut => panic!("command unexpectedly timed out"),
        }
        assert_eq!(output.stdout, b"stdout-bytes");
        assert_eq!(output.stderr, b"stderr-bytes");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn execution_timeout_kills_and_reaps_child() {
        let client = StandardSshClient::with_config(SshConnectionConfig {
            execution_timeout: Some(Duration::from_millis(50)),
            ..SshConnectionConfig::default()
        });
        let mut child = Command::new("sleep").arg("30").spawn().unwrap();
        let pid = child.id().unwrap();

        let status = client.wait_for_child(&mut child).await.unwrap();

        assert!(matches!(status, SshCommandStatus::TimedOut));
        assert_eq!(unsafe { libc::kill(pid as i32, 0) }, -1);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn passthrough_preserves_streams_status_timeout_and_cancellation() {
        use std::os::unix::fs::PermissionsExt;

        let temp_dir = tempfile::tempdir().unwrap();
        let fake_ssh = temp_dir.path().join("fake-ssh");
        let private_key = temp_dir.path().join("key");
        let cancellation_pid_file = temp_dir.path().join("cancel-pid");
        let stdout_file = temp_dir.path().join("stdout");
        let stderr_file = temp_dir.path().join("stderr");
        std::fs::write(
            &fake_ssh,
            r#"#!/bin/sh
for last_arg do :; done
case "$last_arg" in
  success) printf 'stdout-bytes'; printf 'stderr-bytes' >&2; exit 0 ;;
  exit-3) exit 3 ;;
  hang) exec sleep 30 ;;
  cancel:*) printf '%s' "$$" > "${last_arg#cancel:}"; exec sleep 30 ;;
esac
exit 255
"#,
        )
        .unwrap();
        std::fs::set_permissions(&fake_ssh, std::fs::Permissions::from_mode(0o755)).unwrap();
        std::fs::write(&private_key, "test-key").unwrap();
        let details = SshConnectionDetails {
            host: "example.test".to_string(),
            username: "testuser".to_string(),
            port: 22,
            private_key_path: private_key,
            timeout: Duration::from_secs(30),
        };
        let client = StandardSshClient::new();

        let success = client
            .execute_command_with_program_and_stdio(
                &details,
                "success",
                &fake_ssh,
                Stdio::from(std::fs::File::create(&stdout_file).unwrap()),
                Stdio::from(std::fs::File::create(&stderr_file).unwrap()),
            )
            .await
            .unwrap();
        let failure = client
            .execute_command_with_program_and_stdio(
                &details,
                "exit-3",
                &fake_ssh,
                Stdio::null(),
                Stdio::null(),
            )
            .await
            .unwrap();

        assert!(matches!(
            success,
            SshCommandStatus::Exited(status) if status.success()
        ));
        assert_eq!(std::fs::read(&stdout_file).unwrap(), b"stdout-bytes");
        assert_eq!(std::fs::read(&stderr_file).unwrap(), b"stderr-bytes");
        assert!(matches!(
            failure,
            SshCommandStatus::Exited(status) if status.code() == Some(3)
        ));

        let timeout_client = StandardSshClient::with_config(SshConnectionConfig {
            execution_timeout: Some(Duration::from_millis(50)),
            ..SshConnectionConfig::default()
        });
        let timeout_status = timeout_client
            .execute_command_passthrough_with_program(&details, "hang", &fake_ssh)
            .await
            .unwrap();

        assert!(matches!(timeout_status, SshCommandStatus::TimedOut));

        {
            let cancellation_command = format!("cancel:{}", cancellation_pid_file.display());
            let cancellation = client.execute_command_with_program_and_stdio(
                &details,
                &cancellation_command,
                &fake_ssh,
                Stdio::null(),
                Stdio::null(),
            );
            tokio::pin!(cancellation);
            let wait_until_started = async {
                while !cancellation_pid_file.exists() {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            };

            tokio::time::timeout(Duration::from_secs(5), async {
                tokio::select! {
                    result = &mut cancellation => {
                        panic!("fake SSH exited before cancellation: {result:?}")
                    }
                    () = wait_until_started => {}
                }
            })
            .await
            .unwrap();
        }

        let cancelled_pid: i32 = std::fs::read_to_string(&cancellation_pid_file)
            .unwrap()
            .parse()
            .unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            while unsafe { libc::kill(cancelled_pid, 0) } == 0 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
    }

    #[test]
    fn test_connection_options_non_strict_include_log_level_error() {
        let client = StandardSshClient::new();

        let options = client.connection_options(true).unwrap();

        assert_eq!(
            options,
            vec![
                "StrictHostKeyChecking=no".to_string(),
                "UserKnownHostsFile=/dev/null".to_string(),
                "LogLevel=ERROR".to_string(),
                "IdentitiesOnly=yes".to_string(),
                "BatchMode=yes".to_string(),
                "ConnectTimeout=30".to_string(),
                "ServerAliveInterval=15".to_string(),
                "ServerAliveCountMax=3".to_string(),
            ]
        );
    }

    #[test]
    fn test_connection_options_strict_include_known_hosts_and_log_level_error() {
        let client = StandardSshClient::with_config(SshConnectionConfig {
            strict_host_key_checking: true,
            known_hosts_file: Some(std::path::PathBuf::from("/tmp/basilica_known_hosts")),
            connection_timeout: Duration::from_secs(7),
            ..SshConnectionConfig::default()
        });

        let options = client.connection_options(false).unwrap();

        assert_eq!(
            options,
            vec![
                "StrictHostKeyChecking=yes".to_string(),
                "UserKnownHostsFile=/tmp/basilica_known_hosts".to_string(),
                "LogLevel=ERROR".to_string(),
                "IdentitiesOnly=yes".to_string(),
                "ConnectTimeout=7".to_string(),
                "ServerAliveInterval=15".to_string(),
                "ServerAliveCountMax=3".to_string(),
            ]
        );
    }

    #[test]
    fn test_connection_options_strict_use_default_known_hosts_path() {
        let client = StandardSshClient::with_config(SshConnectionConfig {
            strict_host_key_checking: true,
            known_hosts_file: None,
            connection_timeout: Duration::from_secs(7),
            ..SshConnectionConfig::default()
        });
        let expected_known_hosts = default_known_hosts_path_for_test();

        let options = client.connection_options(false).unwrap();

        assert_eq!(
            options,
            vec![
                "StrictHostKeyChecking=yes".to_string(),
                format!("UserKnownHostsFile={}", expected_known_hosts.display()),
                "LogLevel=ERROR".to_string(),
                "IdentitiesOnly=yes".to_string(),
                "ConnectTimeout=7".to_string(),
                "ServerAliveInterval=15".to_string(),
                "ServerAliveCountMax=3".to_string(),
            ]
        );
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
