//! Sandbox module for Basilica SDK
//!
//! Provides API for running code in isolated sandboxes.
//!
//! # Usage
//!
//! ```rust,no_run
//! use basilica_sdk::sandbox::{Sandbox, SandboxConfig};
//!
//! # async fn example() -> basilica_sdk::Result<()> {
//! // Create a Python sandbox
//! let sandbox = Sandbox::create(
//!     "https://api.basilica.ai",
//!     Some("your-api-token".to_string()),
//!     SandboxConfig::new("python"),
//! ).await?;
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

// ============================================================================
// Git Types
// ============================================================================

/// Result of a git clone operation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GitCloneResult {
    /// Whether the clone succeeded
    pub success: bool,
    /// Path where the repository was cloned
    pub path: String,
    /// Branch that was checked out
    pub branch: String,
    /// Commit hash of HEAD
    pub commit: String,
    /// Error message if clone failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Result of git status
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GitStatusResult {
    /// Whether the status command succeeded
    pub success: bool,
    /// Current branch
    pub branch: String,
    /// Whether the working tree is clean
    pub clean: bool,
    /// Staged files
    #[serde(default)]
    pub staged: Vec<String>,
    /// Modified files
    #[serde(default)]
    pub modified: Vec<String>,
    /// Untracked files
    #[serde(default)]
    pub untracked: Vec<String>,
    /// Error message if status failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Result of git commit
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GitCommitResult {
    /// Whether the commit succeeded
    pub success: bool,
    /// Commit hash
    pub commit_hash: String,
    /// Commit message
    pub message: String,
    /// Error message if commit failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Result of git push
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GitPushResult {
    /// Whether the push succeeded
    pub success: bool,
    /// Remote name
    pub remote: String,
    /// Branch that was pushed
    pub branch: String,
    /// Error message if push failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Result of git pull
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GitPullResult {
    /// Whether the pull succeeded
    pub success: bool,
    /// Remote name
    pub remote: String,
    /// Branch that was pulled
    pub branch: String,
    /// Number of commits pulled
    #[serde(default)]
    pub commits_pulled: u32,
    /// Error message if pull failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

// ============================================================================
// LSP Types
// ============================================================================

/// LSP server capabilities
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LspCapabilities {
    /// Whether completion is supported
    #[serde(default)]
    pub completion_provider: bool,
    /// Whether hover is supported
    #[serde(default)]
    pub hover_provider: bool,
    /// Whether go-to-definition is supported
    #[serde(default)]
    pub definition_provider: bool,
    /// Whether find-references is supported
    #[serde(default)]
    pub references_provider: bool,
    /// Whether document symbols are supported
    #[serde(default)]
    pub document_symbol_provider: bool,
    /// Raw capabilities from server
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw: Option<serde_json::Value>,
}

/// LSP error response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspError {
    /// Error code
    pub code: i32,
    /// Error message
    pub message: String,
    /// Additional data
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

/// LSP completion item
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompletionItem {
    /// The label of this completion item
    pub label: String,
    /// The kind of this completion item (1=Text, 2=Method, 3=Function, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<u32>,
    /// A human-readable string with additional information
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    /// Documentation for this item
    #[serde(skip_serializing_if = "Option::is_none")]
    pub documentation: Option<String>,
    /// Text to be inserted when this item is selected
    #[serde(skip_serializing_if = "Option::is_none")]
    pub insert_text: Option<String>,
    /// A string to sort this item relative to others
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sort_text: Option<String>,
}

/// LSP hover result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HoverResult {
    /// Hover contents (markdown or plaintext)
    pub contents: String,
    /// Range start position
    #[serde(skip_serializing_if = "Option::is_none")]
    pub range_start: Option<Position>,
    /// Range end position
    #[serde(skip_serializing_if = "Option::is_none")]
    pub range_end: Option<Position>,
}

/// LSP position in a document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    /// Line number (0-indexed)
    pub line: u32,
    /// Character offset (0-indexed)
    pub character: u32,
}

/// LSP diagnostic (error/warning)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Diagnostic {
    /// The diagnostic message
    pub message: String,
    /// Severity: 1=Error, 2=Warning, 3=Info, 4=Hint
    pub severity: u32,
    /// Line number (0-indexed)
    pub line: u32,
    /// Character offset (0-indexed)
    pub character: u32,
    /// End line (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_line: Option<u32>,
    /// End character (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_character: Option<u32>,
    /// Source of the diagnostic
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Diagnostic code
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

/// LSP location (file + position)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Location {
    /// File URI
    pub uri: String,
    /// Line number (0-indexed)
    pub line: u32,
    /// Character offset (0-indexed)
    pub character: u32,
}

/// LSP initialization response
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LspInitResponse {
    /// Language the server was initialized for
    pub language: String,
    /// Server capabilities
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capabilities: Option<serde_json::Value>,
}

/// LSP request response
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LspRequestResponse {
    /// Request ID
    pub id: u64,
    /// Result if successful
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    /// Error if failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<LspError>,
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

    /// Create a sandbox handle from an existing ID with explicit language.
    pub fn from_id_with_language(
        base_url: impl Into<String>,
        auth_token: Option<String>,
        sandbox_id: impl Into<String>,
        language: impl Into<String>,
    ) -> Self {
        Self {
            http: SandboxHttpClient {
                client: Client::new(),
                base_url: base_url.into(),
                auth_token,
            },
            sandbox_id: sandbox_id.into(),
            language: language.into(),
        }
    }

    /// Create a sandbox from an existing ID (for reconnecting)
    ///
    /// Note: language is discovered lazily for LSP operations if unknown.
    pub fn from_id(
        base_url: impl Into<String>,
        auth_token: Option<String>,
        sandbox_id: impl Into<String>,
    ) -> Self {
        Self::from_id_with_language(base_url, auth_token, sandbox_id, String::new())
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

        let _: serde_json::Value = self
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

    /// List snapshots for this sandbox
    pub async fn list_snapshots(&self) -> Result<Vec<SnapshotInfo>> {
        #[derive(Deserialize)]
        struct ListSnapshotsResponse {
            snapshots: Vec<SnapshotInfo>,
        }

        let response: ListSnapshotsResponse = self
            .http
            .get(&format!("/sandboxes/{}/snapshots", self.sandbox_id))
            .await?;
        Ok(response.snapshots)
    }

    /// Get a specific snapshot for this sandbox
    pub async fn get_snapshot(&self, snapshot_id: &str) -> Result<SnapshotInfo> {
        self.http
            .get(&format!(
                "/sandboxes/{}/snapshots/{}",
                self.sandbox_id, snapshot_id
            ))
            .await
    }

    /// Delete a specific snapshot metadata record for this sandbox
    pub async fn delete_snapshot(&self, snapshot_id: &str) -> Result<()> {
        self.http
            .delete(&format!(
                "/sandboxes/{}/snapshots/{}",
                self.sandbox_id, snapshot_id
            ))
            .await
    }

    // =========================================================================
    // Git Operations
    // =========================================================================

    /// Clone a git repository into the sandbox
    ///
    /// # Arguments
    /// * `url` - Repository URL (HTTPS or SSH)
    /// * `path` - Target path (defaults to /workspace/<repo_name>)
    /// * `branch` - Branch to clone (defaults to default branch)
    /// * `depth` - Clone depth for shallow clone (defaults to 1)
    ///
    /// # Example
    /// ```rust,no_run
    /// # use basilica_sdk::sandbox::Sandbox;
    /// # async fn example(sandbox: &Sandbox) -> basilica_sdk::Result<()> {
    /// let result = sandbox.git_clone("https://github.com/user/repo.git", None, None, None).await?;
    /// println!("Cloned to {} at commit {}", result.path, result.commit);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn git_clone(
        &self,
        url: &str,
        path: Option<&str>,
        branch: Option<&str>,
        depth: Option<u32>,
    ) -> Result<GitCloneResult> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct GitCloneRequest {
            url: String,
            #[serde(skip_serializing_if = "Option::is_none")]
            path: Option<String>,
            #[serde(skip_serializing_if = "Option::is_none")]
            branch: Option<String>,
            #[serde(skip_serializing_if = "Option::is_none")]
            depth: Option<u32>,
        }

        self.http
            .post(
                &format!("/sandboxes/{}/git/clone", self.sandbox_id),
                &GitCloneRequest {
                    url: url.to_string(),
                    path: path.map(String::from),
                    branch: branch.map(String::from),
                    depth,
                },
            )
            .await
    }

    /// Get git status in the sandbox
    ///
    /// # Arguments
    /// * `path` - Path to the git repository (defaults to /workspace)
    ///
    /// # Example
    /// ```rust,no_run
    /// # use basilica_sdk::sandbox::Sandbox;
    /// # async fn example(sandbox: &Sandbox) -> basilica_sdk::Result<()> {
    /// let status = sandbox.git_status(None).await?;
    /// if !status.clean {
    ///     println!("Modified files: {:?}", status.modified);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn git_status(&self, path: Option<&str>) -> Result<GitStatusResult> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct GitStatusRequest {
            #[serde(skip_serializing_if = "Option::is_none")]
            path: Option<String>,
        }

        self.http
            .post(
                &format!("/sandboxes/{}/git/status", self.sandbox_id),
                &GitStatusRequest {
                    path: path.map(String::from),
                },
            )
            .await
    }

    /// Commit changes in the sandbox
    ///
    /// Stages all changes and creates a commit.
    ///
    /// # Arguments
    /// * `message` - Commit message
    /// * `path` - Path to the git repository (defaults to /workspace)
    /// * `author` - Author in format "Name <email>" (optional)
    ///
    /// # Example
    /// ```rust,no_run
    /// # use basilica_sdk::sandbox::Sandbox;
    /// # async fn example(sandbox: &Sandbox) -> basilica_sdk::Result<()> {
    /// let result = sandbox.git_commit("Add new feature", None, None).await?;
    /// println!("Created commit: {}", result.commit_hash);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn git_commit(
        &self,
        message: &str,
        path: Option<&str>,
        author: Option<&str>,
    ) -> Result<GitCommitResult> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct GitCommitRequest {
            message: String,
            #[serde(skip_serializing_if = "Option::is_none")]
            path: Option<String>,
            #[serde(skip_serializing_if = "Option::is_none")]
            author: Option<String>,
        }

        self.http
            .post(
                &format!("/sandboxes/{}/git/commit", self.sandbox_id),
                &GitCommitRequest {
                    message: message.to_string(),
                    path: path.map(String::from),
                    author: author.map(String::from),
                },
            )
            .await
    }

    /// Push commits to remote
    ///
    /// # Arguments
    /// * `path` - Path to the git repository (defaults to /workspace)
    /// * `remote` - Remote name (defaults to "origin")
    /// * `branch` - Branch to push (defaults to current branch)
    ///
    /// # Example
    /// ```rust,no_run
    /// # use basilica_sdk::sandbox::Sandbox;
    /// # async fn example(sandbox: &Sandbox) -> basilica_sdk::Result<()> {
    /// let result = sandbox.git_push(None, Some("origin"), None).await?;
    /// if result.success {
    ///     println!("Pushed to {}/{}", result.remote, result.branch);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn git_push(
        &self,
        path: Option<&str>,
        remote: Option<&str>,
        branch: Option<&str>,
    ) -> Result<GitPushResult> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct GitPushRequest {
            #[serde(skip_serializing_if = "Option::is_none")]
            path: Option<String>,
            #[serde(skip_serializing_if = "Option::is_none")]
            remote: Option<String>,
            #[serde(skip_serializing_if = "Option::is_none")]
            branch: Option<String>,
        }

        self.http
            .post(
                &format!("/sandboxes/{}/git/push", self.sandbox_id),
                &GitPushRequest {
                    path: path.map(String::from),
                    remote: remote.map(String::from),
                    branch: branch.map(String::from),
                },
            )
            .await
    }

    /// Pull changes from remote
    ///
    /// # Arguments
    /// * `path` - Path to the git repository (defaults to /workspace)
    /// * `remote` - Remote name (defaults to "origin")
    /// * `branch` - Branch to pull (defaults to current branch)
    ///
    /// # Example
    /// ```rust,no_run
    /// # use basilica_sdk::sandbox::Sandbox;
    /// # async fn example(sandbox: &Sandbox) -> basilica_sdk::Result<()> {
    /// let result = sandbox.git_pull(None, Some("origin"), None).await?;
    /// println!("Pulled {} commits", result.commits_pulled);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn git_pull(
        &self,
        path: Option<&str>,
        remote: Option<&str>,
        branch: Option<&str>,
    ) -> Result<GitPullResult> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct GitPullRequest {
            #[serde(skip_serializing_if = "Option::is_none")]
            path: Option<String>,
            #[serde(skip_serializing_if = "Option::is_none")]
            remote: Option<String>,
            #[serde(skip_serializing_if = "Option::is_none")]
            branch: Option<String>,
        }

        self.http
            .post(
                &format!("/sandboxes/{}/git/pull", self.sandbox_id),
                &GitPullRequest {
                    path: path.map(String::from),
                    remote: remote.map(String::from),
                    branch: branch.map(String::from),
                },
            )
            .await
    }

    // =========================================================================
    // LSP (Language Server Protocol) Operations
    // =========================================================================

    /// Initialize LSP server for code intelligence
    ///
    /// Starts a language server for the specified language, enabling
    /// code completion, hover documentation, diagnostics, and more.
    ///
    /// # Arguments
    /// * `language` - Programming language (defaults to sandbox's language)
    /// * `root_path` - Root path for the workspace
    ///
    /// # Example
    /// ```rust,no_run
    /// # use basilica_sdk::sandbox::Sandbox;
    /// # async fn example(sandbox: &Sandbox) -> basilica_sdk::Result<()> {
    /// let capabilities = sandbox.lsp_init(None, "/workspace").await?;
    /// if capabilities.completion_provider {
    ///     println!("Code completion available!");
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn lsp_init(
        &self,
        language: Option<&str>,
        root_path: &str,
    ) -> Result<LspCapabilities> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct LspInitRequest {
            language: String,
            root_path: String,
        }

        let selected_language = if let Some(lang) = language {
            lang.to_string()
        } else if !self.language.is_empty() {
            self.language.clone()
        } else {
            // Reconnected sandboxes may not know language locally; fetch status.
            self.status().await?.language
        };

        let response: LspInitResponse = self
            .http
            .post(
                &format!("/sandboxes/{}/lsp/init", self.sandbox_id),
                &LspInitRequest {
                    language: selected_language,
                    root_path: root_path.to_string(),
                },
            )
            .await?;

        // Parse capabilities from raw response
        let caps = response.capabilities.unwrap_or_default();
        Ok(LspCapabilities {
            completion_provider: caps
                .get("completionProvider")
                .map(|v| !v.is_null())
                .unwrap_or(false),
            hover_provider: caps
                .get("hoverProvider")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            definition_provider: caps
                .get("definitionProvider")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            references_provider: caps
                .get("referencesProvider")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            document_symbol_provider: caps
                .get("documentSymbolProvider")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            raw: Some(caps),
        })
    }

    /// Get code completions at a position
    ///
    /// # Arguments
    /// * `file` - File path (relative to workspace)
    /// * `line` - Line number (0-indexed)
    /// * `character` - Character position (0-indexed)
    ///
    /// # Example
    /// ```rust,no_run
    /// # use basilica_sdk::sandbox::Sandbox;
    /// # async fn example(sandbox: &Sandbox) -> basilica_sdk::Result<()> {
    /// let completions = sandbox.lsp_completion("main.py", 10, 5).await?;
    /// for item in completions.iter().take(5) {
    ///     println!("{}: {:?}", item.label, item.detail);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn lsp_completion(
        &self,
        file: &str,
        line: u32,
        character: u32,
    ) -> Result<Vec<CompletionItem>> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct LspRequest {
            method: String,
            params: serde_json::Value,
        }

        let response: LspRequestResponse = self
            .http
            .post(
                &format!("/sandboxes/{}/lsp/request", self.sandbox_id),
                &LspRequest {
                    method: "textDocument/completion".to_string(),
                    params: serde_json::json!({
                        "textDocument": { "uri": format!("file://{}", file) },
                        "position": { "line": line, "character": character }
                    }),
                },
            )
            .await?;

        if let Some(err) = response.error {
            return Err(ApiError::Internal {
                message: format!("LSP error: {}", err.message),
            });
        }

        let result = response.result.unwrap_or_default();
        let items = if let Some(items) = result.get("items") {
            items.clone()
        } else if result.is_array() {
            result
        } else {
            return Ok(Vec::new());
        };

        let completions: Vec<CompletionItem> = serde_json::from_value(items).unwrap_or_default();
        Ok(completions)
    }

    /// Get hover information at a position
    ///
    /// # Arguments
    /// * `file` - File path (relative to workspace)
    /// * `line` - Line number (0-indexed)
    /// * `character` - Character position (0-indexed)
    ///
    /// # Example
    /// ```rust,no_run
    /// # use basilica_sdk::sandbox::Sandbox;
    /// # async fn example(sandbox: &Sandbox) -> basilica_sdk::Result<()> {
    /// if let Some(hover) = sandbox.lsp_hover("main.py", 10, 5).await? {
    ///     println!("{}", hover.contents);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn lsp_hover(
        &self,
        file: &str,
        line: u32,
        character: u32,
    ) -> Result<Option<HoverResult>> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct LspRequest {
            method: String,
            params: serde_json::Value,
        }

        let response: LspRequestResponse = self
            .http
            .post(
                &format!("/sandboxes/{}/lsp/request", self.sandbox_id),
                &LspRequest {
                    method: "textDocument/hover".to_string(),
                    params: serde_json::json!({
                        "textDocument": { "uri": format!("file://{}", file) },
                        "position": { "line": line, "character": character }
                    }),
                },
            )
            .await?;

        if response.error.is_some() {
            return Ok(None);
        }

        let result = match response.result {
            Some(r) if !r.is_null() => r,
            _ => return Ok(None),
        };

        // Extract contents from various formats
        let contents = if let Some(contents) = result.get("contents") {
            if let Some(s) = contents.as_str() {
                s.to_string()
            } else if let Some(obj) = contents.as_object() {
                obj.get("value")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string()
            } else if let Some(arr) = contents.as_array() {
                arr.iter()
                    .filter_map(|v| {
                        if let Some(s) = v.as_str() {
                            Some(s.to_string())
                        } else if let Some(obj) = v.as_object() {
                            obj.get("value").and_then(|v| v.as_str()).map(String::from)
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            } else {
                return Ok(None);
            }
        } else {
            return Ok(None);
        };

        let range = result.get("range");
        let range_start = range.and_then(|r| r.get("start")).and_then(|s| {
            Some(Position {
                line: s.get("line")?.as_u64()? as u32,
                character: s.get("character")?.as_u64()? as u32,
            })
        });
        let range_end = range.and_then(|r| r.get("end")).and_then(|e| {
            Some(Position {
                line: e.get("line")?.as_u64()? as u32,
                character: e.get("character")?.as_u64()? as u32,
            })
        });

        Ok(Some(HoverResult {
            contents,
            range_start,
            range_end,
        }))
    }

    /// Get definition location for symbol at position
    ///
    /// # Arguments
    /// * `file` - File path (relative to workspace)
    /// * `line` - Line number (0-indexed)
    /// * `character` - Character position (0-indexed)
    ///
    /// # Example
    /// ```rust,no_run
    /// # use basilica_sdk::sandbox::Sandbox;
    /// # async fn example(sandbox: &Sandbox) -> basilica_sdk::Result<()> {
    /// let locations = sandbox.lsp_definition("main.py", 10, 5).await?;
    /// for loc in locations {
    ///     println!("{}:{}:{}", loc.uri, loc.line, loc.character);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn lsp_definition(
        &self,
        file: &str,
        line: u32,
        character: u32,
    ) -> Result<Vec<Location>> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct LspRequest {
            method: String,
            params: serde_json::Value,
        }

        let response: LspRequestResponse = self
            .http
            .post(
                &format!("/sandboxes/{}/lsp/request", self.sandbox_id),
                &LspRequest {
                    method: "textDocument/definition".to_string(),
                    params: serde_json::json!({
                        "textDocument": { "uri": format!("file://{}", file) },
                        "position": { "line": line, "character": character }
                    }),
                },
            )
            .await?;

        if response.error.is_some() {
            return Ok(Vec::new());
        }

        let result = match response.result {
            Some(r) if !r.is_null() => r,
            _ => return Ok(Vec::new()),
        };

        // Can be a single location or array
        let locations_data = if result.is_array() {
            result.as_array().cloned().unwrap_or_default()
        } else {
            vec![result]
        };

        let mut locations = Vec::new();
        for loc in locations_data {
            if let (Some(uri), Some(range)) = (loc.get("uri"), loc.get("range")) {
                if let (Some(uri_str), Some(start)) = (uri.as_str(), range.get("start")) {
                    locations.push(Location {
                        uri: uri_str.to_string(),
                        line: start.get("line").and_then(|l| l.as_u64()).unwrap_or(0) as u32,
                        character: start
                            .get("character")
                            .and_then(|c| c.as_u64())
                            .unwrap_or(0) as u32,
                    });
                }
            }
        }

        Ok(locations)
    }

    /// Notify LSP server that a file was opened
    ///
    /// # Arguments
    /// * `file` - File path (relative to workspace)
    /// * `content` - File content
    pub async fn lsp_did_open(&self, file: &str, content: &str) -> Result<()> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct LspNotification {
            method: String,
            params: serde_json::Value,
        }

        let _: serde_json::Value = self
            .http
            .post(
                &format!("/sandboxes/{}/lsp/notify", self.sandbox_id),
                &LspNotification {
                    method: "textDocument/didOpen".to_string(),
                    params: serde_json::json!({
                        "textDocument": {
                            "uri": format!("file://{}", file),
                            "languageId": self.language,
                            "version": 1,
                            "text": content
                        }
                    }),
                },
            )
            .await
            .unwrap_or_default();

        Ok(())
    }

    /// Notify LSP server that a file was changed
    ///
    /// # Arguments
    /// * `file` - File path (relative to workspace)
    /// * `content` - New file content
    pub async fn lsp_did_change(&self, file: &str, content: &str) -> Result<()> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct LspNotification {
            method: String,
            params: serde_json::Value,
        }

        let _: serde_json::Value = self
            .http
            .post(
                &format!("/sandboxes/{}/lsp/notify", self.sandbox_id),
                &LspNotification {
                    method: "textDocument/didChange".to_string(),
                    params: serde_json::json!({
                        "textDocument": { "uri": format!("file://{}", file), "version": 2 },
                        "contentChanges": [{ "text": content }]
                    }),
                },
            )
            .await
            .unwrap_or_default();

        Ok(())
    }

    /// Shutdown the LSP server
    pub async fn lsp_shutdown(&self) -> Result<()> {
        let _: serde_json::Value = self
            .http
            .post(
                &format!("/sandboxes/{}/lsp/shutdown", self.sandbox_id),
                &serde_json::json!({}),
            )
            .await
            .unwrap_or_default();

        Ok(())
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

    // Git tests
    #[test]
    fn test_git_clone_result_deserialization() {
        let json = r#"{"success":true,"path":"/workspace/repo","branch":"main","commit":"abc1234"}"#;
        let result: GitCloneResult = serde_json::from_str(json).expect("should deserialize");
        
        assert!(result.success);
        assert_eq!(result.path, "/workspace/repo");
        assert_eq!(result.branch, "main");
        assert_eq!(result.commit, "abc1234");
        assert!(result.error.is_none());
    }

    #[test]
    fn test_git_clone_result_with_error() {
        let json = r#"{"success":false,"path":"","branch":"","commit":"","error":"Repository not found"}"#;
        let result: GitCloneResult = serde_json::from_str(json).expect("should deserialize");
        
        assert!(!result.success);
        assert_eq!(result.error, Some("Repository not found".to_string()));
    }

    #[test]
    fn test_git_status_result_deserialization() {
        let json = r#"{"success":true,"branch":"feature","clean":false,"staged":["file1.txt"],"modified":["file2.txt"],"untracked":["file3.txt"]}"#;
        let result: GitStatusResult = serde_json::from_str(json).expect("should deserialize");
        
        assert!(result.success);
        assert_eq!(result.branch, "feature");
        assert!(!result.clean);
        assert_eq!(result.staged, vec!["file1.txt"]);
        assert_eq!(result.modified, vec!["file2.txt"]);
        assert_eq!(result.untracked, vec!["file3.txt"]);
    }

    #[test]
    fn test_git_status_clean() {
        let json = r#"{"success":true,"branch":"main","clean":true,"staged":[],"modified":[],"untracked":[]}"#;
        let result: GitStatusResult = serde_json::from_str(json).expect("should deserialize");
        
        assert!(result.success);
        assert!(result.clean);
        assert!(result.staged.is_empty());
        assert!(result.modified.is_empty());
        assert!(result.untracked.is_empty());
    }

    #[test]
    fn test_git_commit_result_deserialization() {
        let json = r#"{"success":true,"commitHash":"def5678","message":"Test commit"}"#;
        let result: GitCommitResult = serde_json::from_str(json).expect("should deserialize");
        
        assert!(result.success);
        assert_eq!(result.commit_hash, "def5678");
        assert_eq!(result.message, "Test commit");
    }

    #[test]
    fn test_git_push_result_deserialization() {
        let json = r#"{"success":true,"remote":"origin","branch":"main"}"#;
        let result: GitPushResult = serde_json::from_str(json).expect("should deserialize");
        
        assert!(result.success);
        assert_eq!(result.remote, "origin");
        assert_eq!(result.branch, "main");
    }

    #[test]
    fn test_git_pull_result_deserialization() {
        let json = r#"{"success":true,"remote":"origin","branch":"main","commitsPulled":3}"#;
        let result: GitPullResult = serde_json::from_str(json).expect("should deserialize");
        
        assert!(result.success);
        assert_eq!(result.remote, "origin");
        assert_eq!(result.branch, "main");
        assert_eq!(result.commits_pulled, 3);
    }

    #[test]
    fn test_git_clone_result_serialization() {
        let result = GitCloneResult {
            success: true,
            path: "/workspace/repo".to_string(),
            branch: "main".to_string(),
            commit: "abc123".to_string(),
            error: None,
        };
        
        let json = serde_json::to_string(&result).expect("should serialize");
        assert!(json.contains("\"success\":true"));
        assert!(json.contains("\"branch\":\"main\""));
        // error should be omitted when None
        assert!(!json.contains("\"error\""));
    }
}

