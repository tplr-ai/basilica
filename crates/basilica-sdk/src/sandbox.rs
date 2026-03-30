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
#[derive(Debug, Clone, Deserialize)]
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
#[derive(Debug, Clone, Deserialize)]
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
#[derive(Debug, Clone, Deserialize)]
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
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FileListResponse {
    pub files: Vec<FileEntry>,
}

/// A file entry in a directory listing.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FileEntry {
    pub name: String,
    #[serde(default)]
    pub is_dir: bool,
    #[serde(default)]
    pub size: Option<u64>,
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
        format!("https://{}", self.domain)
    }

    /// Get the WebSocket URL for terminal access.
    pub fn ws_url(&self) -> String {
        format!("wss://{}/ws", self.domain)
    }

    /// Get the exec endpoint URL.
    pub fn exec_url(&self) -> String {
        format!("https://{}/exec", self.domain)
    }

    /// Execute a command in the sandbox.
    ///
    /// Sends POST to `https://<domain>/exec` with Bearer auth.
    pub async fn exec(
        &self,
        command: Vec<String>,
    ) -> std::result::Result<ExecResponse, crate::error::ApiError> {
        let req = ExecRequest {
            command,
            stdin: None,
            workdir: None,
            timeout_secs: None,
        };
        self.data_plane_post("/exec", &req).await
    }

    /// Run code in the sandbox.
    ///
    /// Sends POST to `https://<domain>/run` via the websocket exec-agent.
    /// Note: The exec-agent handles code run over websocket; this sends an
    /// HTTP request that the agent may support depending on configuration.
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
    #[allow(dead_code)]
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
    fn test_create_request_serialization() {
        let req = CreateSandboxRequest {
            image: "registry.basilica.ai/sandbox/python:3.11".to_string(),
            cpu: None,
            memory: None,
            env: vec![],
            ttl_seconds: Some(3600),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("python:3.11"));
        assert!(json.contains("3600"));
        // env should be omitted when empty
        assert!(!json.contains("env"));
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
