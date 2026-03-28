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
}

/// Response from listing sandboxes.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SandboxListResponse {
    pub sandboxes: Vec<SandboxSummary>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SandboxSummary {
    pub sandbox_id: String,
    pub image: String,
    pub status: String,
    pub domain: Option<String>,
    pub created_at: Option<String>,
}

/// Detailed sandbox info.
#[derive(Debug, Clone, Deserialize)]
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

/// A sandbox handle returned after creation.
/// Provides the sandbox domain for direct data-plane access.
#[derive(Debug, Clone)]
pub struct Sandbox {
    pub sandbox_id: String,
    pub domain: String,
    pub status: String,
}

impl Sandbox {
    /// Get the base URL for data-plane operations on this sandbox.
    /// Data-plane traffic goes directly to the sandbox domain, NOT through the API.
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
}

impl From<CreateSandboxResponse> for Sandbox {
    fn from(resp: CreateSandboxResponse) -> Self {
        Self {
            sandbox_id: resp.sandbox_id,
            domain: resp.domain,
            status: resp.status,
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
        };
        assert_eq!(
            sandbox.data_plane_url(),
            "https://sb-abc123.sandboxes.basilica.ai"
        );
        assert_eq!(
            sandbox.ws_url(),
            "wss://sb-abc123.sandboxes.basilica.ai/ws"
        );
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
        };
        let sandbox = Sandbox::from(resp);
        assert_eq!(sandbox.sandbox_id, "sb-test");
        assert_eq!(sandbox.domain, "sb-test.sandboxes.basilica.ai");
    }
}
