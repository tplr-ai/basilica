//! Sandbox module for Basilica SDK
//!
//! Provides Daytona-compatible API for running code in isolated sandboxes.
//!
//! # Usage
//!
//! ```rust,no_run
//! use basilica_sdk::{BasilicaClient, ClientBuilder};
//! use basilica_sdk::sandbox::{Sandbox, SandboxConfig};
//!
//! # async fn example() -> basilica_sdk::Result<()> {
//! let client = ClientBuilder::default()
//!     .base_url("https://api.basilica.ai")
//!     .with_tokens("access_token", "refresh_token")
//!     .build()?;
//!
//! // Create a Python sandbox
//! let sandbox = Sandbox::create(&client, SandboxConfig::new("python")).await?;
//!
//! // Run code
//! let result = sandbox.run("print('Hello, World!')").await?;
//! println!("Output: {}", result.stdout);
//!
//! // Execute commands
//! let result = sandbox.exec(&["ls", "-la"]).await?;
//!
//! // File operations
//! sandbox.write_file("/workspace/app.py", "print('Hello')").await?;
//! let content = sandbox.read_file("/workspace/app.py").await?;
//!
//! // Cleanup
//! sandbox.delete().await?;
//! # Ok(())
//! # }
//! ```

use crate::error::{ApiError, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// State of a sandbox
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum SandboxState {
    Creating,
    Initializing,
    Ready,
    Executing,
    Snapshotting,
    Terminating,
    Terminated,
    Failed,
}

impl SandboxState {
    /// Check if sandbox is in a terminal state
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Terminated | Self::Failed)
    }

    /// Check if sandbox is ready for execution
    pub fn is_ready(&self) -> bool {
        matches!(self, Self::Ready | Self::Executing)
    }
}

/// Network isolation mode
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum NetworkIsolation {
    /// Full network access
    #[default]
    None,
    /// Egress only (can reach internet, cannot be reached)
    Egress,
    /// No network access
    Full,
}

/// GPU requirements for sandbox
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GpuSpec {
    /// Number of GPUs
    pub count: u32,
    /// GPU model requirements (e.g., ["A100", "H100"])
    #[serde(default)]
    pub model: Vec<String>,
    /// Minimum CUDA version
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_cuda_version: Option<String>,
    /// Minimum GPU memory in GB
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_gpu_memory_gb: Option<u32>,
}

/// Resource requirements for sandbox
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourceSpec {
    /// CPU allocation (e.g., "500m", "1", "2")
    #[serde(default = "default_cpu")]
    pub cpu: String,
    /// Memory allocation (e.g., "512Mi", "1Gi")
    #[serde(default = "default_memory")]
    pub memory: String,
    /// GPU requirements
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpus: Option<GpuSpec>,
}

impl Default for ResourceSpec {
    fn default() -> Self {
        Self {
            cpu: default_cpu(),
            memory: default_memory(),
            gpus: None,
        }
    }
}

fn default_cpu() -> String {
    "500m".to_string()
}

fn default_memory() -> String {
    "512Mi".to_string()
}

/// Configuration for creating a sandbox
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SandboxConfig {
    /// Programming language (python, javascript, bash, etc.)
    pub language: String,
    /// Custom container image (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,
    /// Resource requirements
    #[serde(default)]
    pub resources: ResourceSpec,
    /// Environment variables
    #[serde(default)]
    pub env: Vec<EnvVar>,
    /// Timeout in seconds (default: 3600)
    #[serde(default = "default_timeout")]
    pub timeout_seconds: u32,
    /// Idle timeout in seconds (default: 600)
    #[serde(default = "default_idle_timeout")]
    pub idle_timeout_seconds: u32,
    /// Auto-snapshot on termination
    #[serde(default)]
    pub auto_snapshot: bool,
    /// Restore from snapshot ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub restore_from: Option<String>,
    /// Network isolation mode
    #[serde(default)]
    pub network_isolation: NetworkIsolation,
}

fn default_timeout() -> u32 {
    3600
}

fn default_idle_timeout() -> u32 {
    600
}

impl SandboxConfig {
    /// Create a new sandbox config with defaults
    pub fn new(language: impl Into<String>) -> Self {
        Self {
            language: language.into(),
            image: None,
            resources: ResourceSpec::default(),
            env: Vec::new(),
            timeout_seconds: default_timeout(),
            idle_timeout_seconds: default_idle_timeout(),
            auto_snapshot: false,
            restore_from: None,
            network_isolation: NetworkIsolation::default(),
        }
    }

    /// Set custom container image
    pub fn with_image(mut self, image: impl Into<String>) -> Self {
        self.image = Some(image.into());
        self
    }

    /// Set resource requirements
    pub fn with_resources(mut self, resources: ResourceSpec) -> Self {
        self.resources = resources;
        self
    }

    /// Add environment variable
    pub fn with_env(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.push(EnvVar {
            name: name.into(),
            value: value.into(),
        });
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, seconds: u32) -> Self {
        self.timeout_seconds = seconds;
        self
    }

    /// Enable auto-snapshot
    pub fn with_auto_snapshot(mut self) -> Self {
        self.auto_snapshot = true;
        self
    }

    /// Restore from snapshot
    pub fn with_restore_from(mut self, snapshot_id: impl Into<String>) -> Self {
        self.restore_from = Some(snapshot_id.into());
        self
    }

    /// Set network isolation
    pub fn with_network_isolation(mut self, isolation: NetworkIsolation) -> Self {
        self.network_isolation = isolation;
        self
    }
}

/// Environment variable
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EnvVar {
    pub name: String,
    pub value: String,
}

/// Result of executing a command in sandbox
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExecResult {
    /// Standard output
    pub stdout: String,
    /// Standard error
    pub stderr: String,
    /// Exit code
    pub exit_code: i32,
    /// Duration in milliseconds
    #[serde(default)]
    pub duration_ms: u64,
}

impl ExecResult {
    /// Check if command succeeded (exit code 0)
    pub fn success(&self) -> bool {
        self.exit_code == 0
    }
}

/// Information about a file in the sandbox
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FileInfo {
    pub name: String,
    pub path: String,
    pub is_dir: bool,
    #[serde(default)]
    pub size: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modified_at: Option<String>,
}

/// Snapshot information
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SnapshotInfo {
    pub snapshot_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub created_at: String,
    #[serde(default)]
    pub size_bytes: u64,
}

/// Sandbox status response
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SandboxStatus {
    pub sandbox_id: String,
    pub state: SandboxState,
    pub language: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_activity_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pod_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_name: Option<String>,
    pub websocket_url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snapshot_id: Option<String>,
}

/// Create sandbox response
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateSandboxResponse {
    pub sandbox_id: String,
    pub state: String,
    pub websocket_url: String,
}

/// Internal HTTP client wrapper for sandbox operations
#[derive(Clone)]
struct SandboxHttpClient {
    client: Client,
    base_url: String,
    auth_token: Option<String>,
}

impl SandboxHttpClient {
    async fn get<T: serde::de::DeserializeOwned>(&self, path: &str) -> Result<T> {
        let url = format!("{}{}", self.base_url, path);
        let mut request = self.client.get(&url);

        if let Some(ref token) = self.auth_token {
            request = request.header("Authorization", format!("Bearer {}", token));
        }

        let response = request.send().await.map_err(ApiError::HttpClient)?;

        if response.status().is_success() {
            response.json().await.map_err(ApiError::HttpClient)
        } else {
            let status = response.status().as_u16();
            let message = response.text().await.unwrap_or_default();
            Err(ApiError::ApiResponse { status, message })
        }
    }

    async fn post<T: serde::de::DeserializeOwned, B: Serialize>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T> {
        let url = format!("{}{}", self.base_url, path);
        let mut request = self.client.post(&url).json(body);

        if let Some(ref token) = self.auth_token {
            request = request.header("Authorization", format!("Bearer {}", token));
        }

        let response = request.send().await.map_err(ApiError::HttpClient)?;

        if response.status().is_success() {
            // Handle empty responses (for delete, write, etc.)
            let text = response.text().await.map_err(ApiError::HttpClient)?;
            if text.is_empty() {
                // Return default value for unit type responses
                serde_json::from_str("null").map_err(|e| ApiError::Internal {
                    message: format!("Failed to parse response: {}", e),
                })
            } else {
                serde_json::from_str(&text).map_err(|e| ApiError::Internal {
                    message: format!("Failed to parse response: {}", e),
                })
            }
        } else {
            let status = response.status().as_u16();
            let message = response.text().await.unwrap_or_default();
            Err(ApiError::ApiResponse { status, message })
        }
    }

    async fn delete(&self, path: &str) -> Result<()> {
        let url = format!("{}{}", self.base_url, path);
        let mut request = self.client.delete(&url);

        if let Some(ref token) = self.auth_token {
            request = request.header("Authorization", format!("Bearer {}", token));
        }

        let response = request.send().await.map_err(ApiError::HttpClient)?;

        if response.status().is_success() {
            Ok(())
        } else {
            let status = response.status().as_u16();
            let message = response.text().await.unwrap_or_default();
            Err(ApiError::ApiResponse { status, message })
        }
    }
}

/// A sandbox execution environment
#[derive(Clone)]
pub struct Sandbox {
    http: SandboxHttpClient,
    sandbox_id: String,
    language: String,
}

impl Sandbox {
    /// Create a new sandbox with the given configuration
    ///
    /// # Arguments
    /// * `base_url` - Base URL of the Basilica API (e.g., "https://api.basilica.ai")
    /// * `auth_token` - Bearer token for authentication
    /// * `config` - Sandbox configuration
    pub async fn create(
        base_url: impl Into<String>,
        auth_token: Option<String>,
        config: SandboxConfig,
    ) -> Result<Self> {
        let http = SandboxHttpClient {
            client: Client::new(),
            base_url: base_url.into(),
            auth_token,
        };

        let language = config.language.clone();
        let response: CreateSandboxResponse = http.post("/sandboxes", &config).await?;

        Ok(Self {
            http,
            sandbox_id: response.sandbox_id,
            language,
        })
    }

    /// Create a sandbox from an existing ID (for reconnecting)
    pub fn from_id(
        base_url: impl Into<String>,
        auth_token: Option<String>,
        sandbox_id: impl Into<String>,
    ) -> Self {
        Self {
            http: SandboxHttpClient {
                client: Client::new(),
                base_url: base_url.into(),
                auth_token,
            },
            sandbox_id: sandbox_id.into(),
            language: String::new(),
        }
    }

    /// Get the sandbox ID
    pub fn id(&self) -> &str {
        &self.sandbox_id
    }

    /// Get the sandbox language
    pub fn language(&self) -> &str {
        &self.language
    }

    /// Get sandbox status
    pub async fn status(&self) -> Result<SandboxStatus> {
        self.http
            .get(&format!("/sandboxes/{}", self.sandbox_id))
            .await
    }

    /// Wait until sandbox is ready
    pub async fn wait_until_ready(&self, timeout: Duration) -> Result<SandboxStatus> {
        let start = std::time::Instant::now();
        let poll_interval = Duration::from_secs(1);

        loop {
            let status = self.status().await?;

            if status.state.is_ready() {
                return Ok(status);
            }

            if status.state.is_terminal() {
                return Err(ApiError::Internal {
                    message: format!(
                        "Sandbox entered terminal state: {:?} - {}",
                        status.state,
                        status.message.unwrap_or_default()
                    ),
                });
            }

            if start.elapsed() > timeout {
                return Err(ApiError::Timeout);
            }

            tokio::time::sleep(poll_interval).await;
        }
    }

    /// Execute a command in the sandbox
    pub async fn exec(&self, command: &[&str]) -> Result<ExecResult> {
        #[derive(Serialize)]
        struct ExecRequest {
            command: Vec<String>,
        }

        self.http
            .post(
                &format!("/sandboxes/{}/exec", self.sandbox_id),
                &ExecRequest {
                    command: command.iter().map(|s| s.to_string()).collect(),
                },
            )
            .await
    }

    /// Execute a command with options
    pub async fn exec_with_options(
        &self,
        command: &[&str],
        stdin: Option<&str>,
        workdir: Option<&str>,
    ) -> Result<ExecResult> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct ExecRequest {
            command: Vec<String>,
            #[serde(skip_serializing_if = "Option::is_none")]
            stdin: Option<String>,
            #[serde(skip_serializing_if = "Option::is_none")]
            workdir: Option<String>,
        }

        self.http
            .post(
                &format!("/sandboxes/{}/exec", self.sandbox_id),
                &ExecRequest {
                    command: command.iter().map(|s| s.to_string()).collect(),
                    stdin: stdin.map(String::from),
                    workdir: workdir.map(String::from),
                },
            )
            .await
    }

    /// Run code in the sandbox
    pub async fn run(&self, code: &str) -> Result<ExecResult> {
        #[derive(Serialize)]
        struct RunRequest {
            code: String,
        }

        self.http
            .post(
                &format!("/sandboxes/{}/run", self.sandbox_id),
                &RunRequest {
                    code: code.to_string(),
                },
            )
            .await
    }

    /// Run code with arguments
    pub async fn run_with_args(&self, code: &str, args: &[&str]) -> Result<ExecResult> {
        #[derive(Serialize)]
        struct RunRequest {
            code: String,
            args: Vec<String>,
        }

        self.http
            .post(
                &format!("/sandboxes/{}/run", self.sandbox_id),
                &RunRequest {
                    code: code.to_string(),
                    args: args.iter().map(|s| s.to_string()).collect(),
                },
            )
            .await
    }

    /// Read a file from the sandbox
    pub async fn read_file(&self, path: &str) -> Result<String> {
        #[derive(Serialize)]
        struct ReadRequest {
            path: String,
        }

        #[derive(Deserialize)]
        struct ReadResponse {
            content: String,
        }

        let response: ReadResponse = self
            .http
            .post(
                &format!("/sandboxes/{}/files/read", self.sandbox_id),
                &ReadRequest {
                    path: path.to_string(),
                },
            )
            .await?;

        Ok(response.content)
    }

    /// Write a file to the sandbox
    pub async fn write_file(&self, path: &str, content: &str) -> Result<()> {
        #[derive(Serialize)]
        struct WriteRequest {
            path: String,
            content: String,
        }

        #[derive(Deserialize)]
        struct EmptyResponse {}

        let _: EmptyResponse = self
            .http
            .post(
                &format!("/sandboxes/{}/files/write", self.sandbox_id),
                &WriteRequest {
                    path: path.to_string(),
                    content: content.to_string(),
                },
            )
            .await?;

        Ok(())
    }

    /// List files in a directory
    pub async fn list_files(&self, path: &str) -> Result<Vec<FileInfo>> {
        #[derive(Serialize)]
        struct ListRequest {
            path: String,
        }

        #[derive(Deserialize)]
        struct ListResponse {
            files: Vec<FileInfo>,
        }

        let response: ListResponse = self
            .http
            .post(
                &format!("/sandboxes/{}/files/list", self.sandbox_id),
                &ListRequest {
                    path: path.to_string(),
                },
            )
            .await?;

        Ok(response.files)
    }

    /// Create a snapshot of the sandbox
    pub async fn create_snapshot(&self, name: Option<&str>) -> Result<SnapshotInfo> {
        #[derive(Serialize)]
        struct SnapshotRequest {
            #[serde(skip_serializing_if = "Option::is_none")]
            name: Option<String>,
        }

        self.http
            .post(
                &format!("/sandboxes/{}/snapshot", self.sandbox_id),
                &SnapshotRequest {
                    name: name.map(String::from),
                },
            )
            .await
    }

    /// Delete the sandbox
    pub async fn delete(self) -> Result<()> {
        self.http
            .delete(&format!("/sandboxes/{}", self.sandbox_id))
            .await
    }

    /// Get WebSocket URL for streaming
    pub fn websocket_url(&self) -> String {
        format!("{}/sandboxes/{}/ws", self.http.base_url, self.sandbox_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sandbox_config_new() {
        let config = SandboxConfig::new("python");
        assert_eq!(config.language, "python");
        assert!(config.image.is_none());
        assert_eq!(config.timeout_seconds, 3600);
    }

    #[test]
    fn test_sandbox_config_builder() {
        let config = SandboxConfig::new("python")
            .with_image("custom:latest")
            .with_env("KEY", "value")
            .with_timeout(7200)
            .with_auto_snapshot()
            .with_network_isolation(NetworkIsolation::Egress);

        assert_eq!(config.language, "python");
        assert_eq!(config.image, Some("custom:latest".to_string()));
        assert_eq!(config.env.len(), 1);
        assert_eq!(config.env[0].name, "KEY");
        assert_eq!(config.timeout_seconds, 7200);
        assert!(config.auto_snapshot);
        assert_eq!(config.network_isolation, NetworkIsolation::Egress);
    }

    #[test]
    fn test_sandbox_state_is_terminal() {
        assert!(!SandboxState::Creating.is_terminal());
        assert!(!SandboxState::Ready.is_terminal());
        assert!(SandboxState::Terminated.is_terminal());
        assert!(SandboxState::Failed.is_terminal());
    }

    #[test]
    fn test_sandbox_state_is_ready() {
        assert!(!SandboxState::Creating.is_terminal());
        assert!(SandboxState::Ready.is_ready());
        assert!(SandboxState::Executing.is_ready());
        assert!(!SandboxState::Terminated.is_ready());
    }

    #[test]
    fn test_exec_result_success() {
        let result = ExecResult {
            stdout: "output".to_string(),
            stderr: "".to_string(),
            exit_code: 0,
            duration_ms: 100,
        };
        assert!(result.success());

        let failed = ExecResult {
            stdout: "".to_string(),
            stderr: "error".to_string(),
            exit_code: 1,
            duration_ms: 50,
        };
        assert!(!failed.success());
    }

    #[test]
    fn test_resource_spec_default() {
        let spec = ResourceSpec::default();
        assert_eq!(spec.cpu, "500m");
        assert_eq!(spec.memory, "512Mi");
        assert!(spec.gpus.is_none());
    }

    #[test]
    fn test_config_serialization() {
        let config = SandboxConfig::new("python")
            .with_env("TEST", "value");

        let json = serde_json::to_string(&config).expect("should serialize");
        assert!(json.contains("\"language\":\"python\""));
        assert!(json.contains("\"env\":["));
    }
}

