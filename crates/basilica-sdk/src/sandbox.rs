//! Sandbox SDK module.
//!
//! Provides control-plane operations (create, list, get, delete) via the API,
//! and data-plane connectivity directly to sandbox domains.
//!
//! Architecture:
//! - Control plane: SDK → basilica-api → BasilicaSandbox CRD
//! - Data plane: SDK → <sandbox-id>.sandboxes.basilica.ai (direct)
//!
//! H1: The API is control-plane only. No exec/ws/file relay through the API.

use serde::{Deserialize, Serialize};

fn default_network_isolation() -> String {
    "egress".to_string()
}

/// Request to create a new sandbox.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateSandboxRequest {
    /// Container image (must be in the server's allowlist).
    pub image: String,

    /// CPU resources (default: "1").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu: Option<String>,

    /// Memory resources (default: "2Gi").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory: Option<String>,

    /// User-supplied environment variables.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub env: Vec<SandboxEnvVar>,

    /// Optional TTL in seconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl_seconds: Option<u32>,

    /// Network isolation level: "egress" (default) or "full".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub network_isolation: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SandboxEnvVar {
    pub name: String,
    pub value: String,
}

/// Response from creating a sandbox.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateSandboxResponse {
    pub sandbox_id: String,
    /// Direct data-plane domain (e.g. "sb-a1b2c3d4.sandboxes.basilica.ai").
    pub domain: String,
    pub status: String,
    /// Exec-agent secret for data-plane authentication.
    /// Use as `Authorization: Bearer <secret>` when calling sandbox domain endpoints.
    pub exec_agent_secret: String,
}

/// Response from rotating a sandbox exec-agent secret.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RotateSandboxSecretResponse {
    pub sandbox_id: String,
    pub exec_agent_secret: String,
}

/// Response from listing sandboxes.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SandboxListResponse {
    pub sandboxes: Vec<SandboxSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SandboxSummary {
    pub sandbox_id: String,
    pub image: String,
    pub status: String,
    pub domain: Option<String>,
    pub created_at: Option<String>,
    pub ttl_seconds: Option<u32>,
    #[serde(default = "default_network_isolation")]
    pub network_isolation: String,
    pub ready_at: Option<String>,
    pub expires_at: Option<String>,
    #[serde(default)]
    pub from_warm_pool: bool,
}

/// Detailed sandbox info.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SandboxDetail {
    pub sandbox_id: String,
    pub image: String,
    pub cpu: String,
    pub memory: String,
    pub status: String,
    pub domain: Option<String>,
    pub created_at: Option<String>,
    pub ttl_seconds: Option<u32>,
    #[serde(default = "default_network_isolation")]
    pub network_isolation: String,
    pub ready_at: Option<String>,
    pub expires_at: Option<String>,
    #[serde(default)]
    pub from_warm_pool: bool,
}

// ============================================================================
// Data-plane types
// ============================================================================

/// Request to execute a command in the sandbox.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExecRequest {
    pub command: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stdin: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workdir: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout_secs: Option<u64>,
}

/// Response from command execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExecResponse {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

/// Request to run code in the sandbox.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RunRequest {
    pub code: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[serde(default)]
    pub args: Vec<String>,
}

/// Request to write a file in the sandbox.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FileWriteRequest {
    pub path: String,
    pub content: String,
}

/// Response from file write.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FileWriteResponse {
    pub path: String,
}

/// Request to read a file from the sandbox.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FileReadRequest {
    pub path: String,
}

/// Response from file read.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FileReadResponse {
    pub content: String,
    pub path: String,
}

/// Request to list files in the sandbox.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FileListRequest {
    pub path: String,
}

/// Response from file list.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FileListResponse {
    pub files: Vec<FileEntry>,
}

/// A file entry in a directory listing.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FileEntry {
    pub name: String,
    #[serde(default)]
    pub is_dir: bool,
    #[serde(default)]
    pub size: Option<u64>,
}

// ============================================================================
// Snapshot types
// ============================================================================

/// Request to create a filesystem snapshot.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateSnapshotRequest {
    /// Optional subdirectory within the workspace to snapshot.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
}

/// Response from creating a snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SnapshotResponse {
    pub snapshot_id: String,
    pub status: String,
    #[serde(default)]
    pub message: Option<String>,
    #[serde(default)]
    pub archive_path: Option<String>,
    #[serde(default)]
    pub archive_size_bytes: Option<u64>,
}

/// Request to upload a snapshot archive to object storage.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SnapshotUploadRequest {
    pub snapshot_id: String,
    pub presigned_url: String,
}

/// Response from uploading a snapshot archive.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SnapshotUploadResponse {
    pub status: String,
    pub bytes_uploaded: u64,
}

/// Request to restore a snapshot archive from object storage.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SnapshotRestoreRequest {
    pub snapshot_id: String,
    pub presigned_url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
}

/// Response from restoring a snapshot archive.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SnapshotRestoreResponse {
    pub snapshot_id: String,
    pub status: String,
    pub restored_path: String,
    pub bytes_downloaded: u64,
}

/// Response from snapshot status.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SnapshotStatusResponse {
    pub snapshot_id: Option<String>,
    pub status: String,
    #[serde(default)]
    pub message: Option<String>,
    #[serde(default)]
    pub archive_size_bytes: Option<u64>,
    #[serde(default)]
    pub bytes_uploaded: Option<u64>,
    #[serde(default)]
    pub bytes_downloaded: Option<u64>,
}

// ============================================================================
// File operation types (extended)
// ============================================================================

/// Request to delete a file.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FileDeleteRequest {
    pub path: String,
}

/// Request to create a directory.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FileMkdirRequest {
    pub path: String,
    #[serde(default)]
    pub recursive: bool,
}

/// Request to stat a file.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FileStatRequest {
    pub path: String,
}

/// Response from file stat.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FileStatResponse {
    pub size: u64,
    pub is_file: bool,
    pub is_dir: bool,
    pub modified: Option<String>,
}

// ============================================================================
// Sandbox handle with data-plane client
// ============================================================================

/// A sandbox handle returned after creation.
/// Provides both URL helpers and working data-plane client methods.
///
/// Data-plane operations go directly to the sandbox domain, NOT through the API.
/// Authentication uses `Authorization: Bearer <exec_agent_secret>`.
#[derive(Debug, Clone)]
pub struct Sandbox {
    pub sandbox_id: String,
    pub domain: String,
    pub status: String,
    /// Exec-agent secret for data-plane auth. Only present after creation.
    exec_agent_secret: Option<String>,
    /// HTTP client for data-plane requests.
    http_client: reqwest::Client,
    /// Override the data-plane base URL (e.g. "http://localhost:12345" for K3d testing).
    /// When set, data-plane requests use this instead of `https://{domain}`.
    data_plane_base_url: Option<String>,
}

impl Sandbox {
    /// Get the exec-agent secret for data-plane authentication, if available.
    pub fn exec_agent_secret(&self) -> Option<&str> {
        self.exec_agent_secret.as_deref()
    }

    /// Override the data-plane base URL for local/test connectivity.
    ///
    /// In production, data-plane requests go to `https://{domain}`.
    /// For local K3d testing via port-forward, set this to e.g. `http://localhost:12345`.
    pub fn with_data_plane_url(mut self, url: String) -> Self {
        self.data_plane_base_url = Some(url);
        self
    }

    /// Override the exec-agent secret (e.g. when retrieved from K8s Secret).
    pub fn with_exec_agent_secret(mut self, secret: String) -> Self {
        self.exec_agent_secret = Some(secret);
        self
    }

    /// Resolve the base URL for data-plane operations.
    fn resolve_data_plane_base(&self) -> String {
        self.data_plane_base_url
            .clone()
            .unwrap_or_else(|| format!("https://{}", self.domain))
    }

    /// Get the base URL for data-plane operations on this sandbox.
    pub fn data_plane_url(&self) -> String {
        self.resolve_data_plane_base()
    }

    /// Get the WebSocket URL for terminal access.
    ///
    /// The exec-agent authenticates WebSocket upgrades via HTTP headers
    /// (same as all other data-plane requests):
    /// - `Authorization: Bearer <exec_agent_secret>`, or
    /// - `X-Exec-Secret: <exec_agent_secret>`
    ///
    /// Set the header on the HTTP upgrade request. Query-string auth is
    /// deliberately NOT supported (S-4).
    ///
    /// For convenience, use [`ws_connect_info`] to get both the URL and
    /// the auth header value together.
    pub fn ws_url(&self) -> String {
        let base = self.resolve_data_plane_base();
        let scheme = if base.starts_with("https://") { "wss" } else { "ws" };
        let host = base
            .trim_start_matches("https://")
            .trim_start_matches("http://");
        format!("{scheme}://{host}/ws")
    }

    /// Get the WebSocket URL and auth header for terminal access.
    ///
    /// Returns `(url, header_name, header_value)` ready to pass to a
    /// WebSocket client library (e.g. `tokio-tungstenite`).
    ///
    /// Returns `None` if the exec-agent secret is not available.
    pub fn ws_connect_info(&self) -> Option<(String, &'static str, String)> {
        let secret = self.exec_agent_secret.as_deref()?;
        Some((
            self.ws_url(),
            "Authorization",
            format!("Bearer {secret}"),
        ))
    }

    /// Get the exec endpoint URL.
    pub fn exec_url(&self) -> String {
        format!("{}/exec", self.resolve_data_plane_base())
    }

    /// Execute a command in the sandbox.
    ///
    /// Sends POST to `https://<domain>/exec` with Bearer auth.
    pub async fn exec(
        &self,
        command: Vec<String>,
    ) -> std::result::Result<ExecResponse, crate::error::ApiError> {
        self.exec_with_options(command, None, None, None).await
    }

    /// Execute a command in the sandbox with full request options.
    pub async fn exec_with_options(
        &self,
        command: Vec<String>,
        workdir: Option<String>,
        stdin: Option<String>,
        timeout_secs: Option<u64>,
    ) -> std::result::Result<ExecResponse, crate::error::ApiError> {
        let req = ExecRequest {
            command,
            stdin,
            workdir,
            timeout_secs,
        };
        self.data_plane_post("/exec", &req).await
    }

    /// Run code in the sandbox.
    ///
    /// Sends POST to `https://<domain>/run` directly to the sandbox data-plane.
    /// This is an authenticated HTTP request to the exec-agent, not an API
    /// relay and not a WebSocket-only path.
    pub async fn run(
        &self,
        code: &str,
    ) -> std::result::Result<ExecResponse, crate::error::ApiError> {
        let req = RunRequest {
            code: code.to_string(),
            language: None,
            args: vec![],
        };
        self.data_plane_post("/run", &req).await
    }

    /// Get a file operations handle.
    pub fn files(&self) -> SandboxFiles<'_> {
        SandboxFiles { sandbox: self }
    }

    /// Create a filesystem snapshot of the sandbox.
    pub async fn snapshot_create(
        &self,
        path: Option<String>,
    ) -> std::result::Result<SnapshotResponse, crate::error::ApiError> {
        self.data_plane_post("/snapshot/create", &CreateSnapshotRequest { path })
            .await
    }

    /// Upload a filesystem snapshot archive to object storage.
    pub async fn snapshot_upload(
        &self,
        snapshot_id: String,
        presigned_url: String,
    ) -> std::result::Result<SnapshotUploadResponse, crate::error::ApiError> {
        self.data_plane_post(
            "/snapshot/upload",
            &SnapshotUploadRequest {
                snapshot_id,
                presigned_url,
            },
        )
        .await
    }

    /// Restore a filesystem snapshot archive from object storage.
    pub async fn snapshot_restore(
        &self,
        snapshot_id: String,
        presigned_url: String,
        path: Option<String>,
    ) -> std::result::Result<SnapshotRestoreResponse, crate::error::ApiError> {
        self.data_plane_post(
            "/snapshot/restore",
            &SnapshotRestoreRequest {
                snapshot_id,
                presigned_url,
                path,
            },
        )
        .await
    }

    /// Get the status of the current/last snapshot.
    pub async fn snapshot_status(
        &self,
    ) -> std::result::Result<SnapshotStatusResponse, crate::error::ApiError> {
        self.data_plane_get("/snapshot/status").await
    }

    /// Make an authenticated POST request to the sandbox data-plane.
    async fn data_plane_post<B: Serialize, T: serde::de::DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> std::result::Result<T, crate::error::ApiError> {
        let secret = self.exec_agent_secret.as_deref().ok_or_else(|| {
            crate::error::ApiError::InvalidRequest {
                message:
                    "exec_agent_secret not available — sandbox was not created through this SDK"
                        .to_string(),
            }
        })?;

        let base = self.resolve_data_plane_base();
        let url = format!("{}{}", base, path);
        let response = self
            .http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", secret))
            .json(body)
            .send()
            .await
            .map_err(crate::error::ApiError::HttpClient)?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body = response.text().await.unwrap_or_default();
            return Err(crate::error::ApiError::ApiResponse {
                status,
                message: body,
            });
        }

        response
            .json()
            .await
            .map_err(crate::error::ApiError::HttpClient)
    }

    /// Make an authenticated GET request to the sandbox data-plane.
    async fn data_plane_get<T: serde::de::DeserializeOwned>(
        &self,
        path: &str,
    ) -> std::result::Result<T, crate::error::ApiError> {
        let secret = self.exec_agent_secret.as_deref().ok_or_else(|| {
            crate::error::ApiError::InvalidRequest {
                message: "exec_agent_secret not available".to_string(),
            }
        })?;

        let base = self.resolve_data_plane_base();
        let url = format!("{}{}", base, path);
        let response = self
            .http_client
            .get(&url)
            .header("Authorization", format!("Bearer {}", secret))
            .send()
            .await
            .map_err(crate::error::ApiError::HttpClient)?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body = response.text().await.unwrap_or_default();
            return Err(crate::error::ApiError::ApiResponse {
                status,
                message: body,
            });
        }

        response
            .json()
            .await
            .map_err(crate::error::ApiError::HttpClient)
    }
}

/// File operations on a sandbox.
pub struct SandboxFiles<'a> {
    sandbox: &'a Sandbox,
}

impl<'a> SandboxFiles<'a> {
    /// Write a file to the sandbox.
    pub async fn write(
        &self,
        path: &str,
        content: &str,
    ) -> std::result::Result<FileWriteResponse, crate::error::ApiError> {
        let req = FileWriteRequest {
            path: path.to_string(),
            content: content.to_string(),
        };
        self.sandbox.data_plane_post("/files/write", &req).await
    }

    /// Read a file from the sandbox.
    pub async fn read(
        &self,
        path: &str,
    ) -> std::result::Result<FileReadResponse, crate::error::ApiError> {
        let req = FileReadRequest {
            path: path.to_string(),
        };
        self.sandbox.data_plane_post("/files/read", &req).await
    }

    /// List files in a directory in the sandbox.
    pub async fn list(
        &self,
        path: &str,
    ) -> std::result::Result<FileListResponse, crate::error::ApiError> {
        let req = FileListRequest {
            path: path.to_string(),
        };
        self.sandbox.data_plane_post("/files/list", &req).await
    }

    /// Delete a file or directory in the sandbox.
    pub async fn delete(
        &self,
        path: &str,
    ) -> std::result::Result<(), crate::error::ApiError> {
        let req = FileDeleteRequest {
            path: path.to_string(),
        };
        let _: serde_json::Value = self.sandbox.data_plane_post("/files/delete", &req).await?;
        Ok(())
    }

    /// Create a directory in the sandbox.
    pub async fn mkdir(
        &self,
        path: &str,
        recursive: bool,
    ) -> std::result::Result<(), crate::error::ApiError> {
        let req = FileMkdirRequest {
            path: path.to_string(),
            recursive,
        };
        let _: serde_json::Value = self.sandbox.data_plane_post("/files/mkdir", &req).await?;
        Ok(())
    }

    /// Get file/directory metadata.
    pub async fn stat(
        &self,
        path: &str,
    ) -> std::result::Result<FileStatResponse, crate::error::ApiError> {
        let req = FileStatRequest {
            path: path.to_string(),
        };
        self.sandbox.data_plane_post("/files/stat", &req).await
    }
}

impl From<CreateSandboxResponse> for Sandbox {
    fn from(resp: CreateSandboxResponse) -> Self {
        Self {
            sandbox_id: resp.sandbox_id,
            domain: resp.domain,
            status: resp.status,
            exec_agent_secret: Some(resp.exec_agent_secret),
            http_client: reqwest::Client::new(),
            data_plane_base_url: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sandbox_data_plane_url() {
        let sandbox = Sandbox {
            sandbox_id: "sb-abc123".to_string(),
            domain: "sb-abc123.sandboxes.basilica.ai".to_string(),
            status: "Running".to_string(),
            exec_agent_secret: Some("test-secret".to_string()),
            http_client: reqwest::Client::new(),
            data_plane_base_url: None,
        };
        assert_eq!(
            sandbox.data_plane_url(),
            "https://sb-abc123.sandboxes.basilica.ai"
        );
        assert_eq!(sandbox.ws_url(), "wss://sb-abc123.sandboxes.basilica.ai/ws");
        assert_eq!(
            sandbox.exec_url(),
            "https://sb-abc123.sandboxes.basilica.ai/exec"
        );
    }

    #[test]
    fn test_sandbox_data_plane_url_with_override() {
        let sandbox = Sandbox {
            sandbox_id: "sb-abc123".to_string(),
            domain: "sb-abc123.sandboxes.basilica.ai".to_string(),
            status: "Running".to_string(),
            exec_agent_secret: Some("test-secret".to_string()),
            http_client: reqwest::Client::new(),
            data_plane_base_url: Some("http://localhost:12345".to_string()),
        };
        assert_eq!(sandbox.data_plane_url(), "http://localhost:12345");
        assert_eq!(sandbox.ws_url(), "ws://localhost:12345/ws");
        assert_eq!(sandbox.exec_url(), "http://localhost:12345/exec");
    }

    #[test]
    fn test_ws_connect_info_with_secret() {
        let sandbox = Sandbox {
            sandbox_id: "sb-abc123".to_string(),
            domain: "sb-abc123.sandboxes.basilica.ai".to_string(),
            status: "Running".to_string(),
            exec_agent_secret: Some("test-secret".to_string()),
            http_client: reqwest::Client::new(),
            data_plane_base_url: Some("http://localhost:12345".to_string()),
        };
        assert_eq!(
            sandbox.ws_connect_info(),
            Some((
                "ws://localhost:12345/ws".to_string(),
                "Authorization",
                "Bearer test-secret".to_string()
            ))
        );
    }

    #[test]
    fn test_ws_connect_info_without_secret() {
        let sandbox = Sandbox {
            sandbox_id: "sb-abc123".to_string(),
            domain: "sb-abc123.sandboxes.basilica.ai".to_string(),
            status: "Running".to_string(),
            exec_agent_secret: None,
            http_client: reqwest::Client::new(),
            data_plane_base_url: None,
        };
        assert_eq!(sandbox.ws_connect_info(), None);
    }

    #[test]
    fn test_create_request_serialization() {
        let req = CreateSandboxRequest {
            image: "registry.basilica.ai/sandbox/python:3.11".to_string(),
            cpu: None,
            memory: None,
            env: vec![],
            ttl_seconds: Some(3600),
            network_isolation: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("python:3.11"));
        assert!(json.contains("3600"));
        // env should be omitted when empty
        assert!(!json.contains("env"));
    }

    #[test]
    fn test_create_request_serialization_with_network_isolation() {
        let req = CreateSandboxRequest {
            image: "registry.basilica.ai/sandbox/python:3.11".to_string(),
            cpu: None,
            memory: None,
            env: vec![],
            ttl_seconds: None,
            network_isolation: Some("full".to_string()),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"networkIsolation\":\"full\""));
    }

    #[test]
    fn test_exec_request_serialization_with_all_fields() {
        let req = ExecRequest {
            command: vec!["pwd".to_string()],
            stdin: Some("hello".to_string()),
            workdir: Some("/tmp".to_string()),
            timeout_secs: Some(30),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"command\":[\"pwd\"]"));
        assert!(json.contains("\"stdin\":\"hello\""));
        assert!(json.contains("\"workdir\":\"/tmp\""));
        assert!(json.contains("\"timeoutSecs\":30"));
    }

    #[test]
    fn test_sandbox_from_response() {
        let resp = CreateSandboxResponse {
            sandbox_id: "sb-test".to_string(),
            domain: "sb-test.sandboxes.basilica.ai".to_string(),
            status: "Pending".to_string(),
            exec_agent_secret: "secret-123".to_string(),
        };
        let sandbox = Sandbox::from(resp);
        assert_eq!(sandbox.sandbox_id, "sb-test");
        assert_eq!(sandbox.domain, "sb-test.sandboxes.basilica.ai");
        assert_eq!(sandbox.exec_agent_secret.as_deref(), Some("secret-123"));
    }
}
