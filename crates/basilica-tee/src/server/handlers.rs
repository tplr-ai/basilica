//! HTTP Request Handlers for Attestation Server

use std::sync::Arc;

use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::Json,
};
use serde::{Deserialize, Serialize};
use tracing::{error, info};

use super::ServerState;
use crate::service::TeeAttestationResult;
use crate::types::GpuDeviceInfo;

/// Attestation response
#[derive(Debug, Serialize, Deserialize)]
pub struct AttestationResponse {
    /// Base64-encoded TDX quote
    pub tdx_quote: Option<String>,
    /// JSON string containing NVIDIA trust evidence
    pub nvtrust_evidence: Option<String>,
    /// Nonce used for attestation
    pub nonce_hex: String,
    /// Hostname of the attesting node
    pub hostname: String,
}

impl From<TeeAttestationResult> for AttestationResponse {
    fn from(result: TeeAttestationResult) -> Self {
        Self {
            tdx_quote: result.tdx_quote,
            nvtrust_evidence: result.gpu_evidence,
            nonce_hex: result.nonce_hex,
            hostname: result.hostname,
        }
    }
}

/// Legacy attestation response (for backward compatibility)
#[derive(Debug, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct LegacyAttestationResponse {
    /// Base64-encoded TDX quote
    pub tdx_quote: String,
    /// JSON string containing NVIDIA trust evidence
    pub nvtrust_evidence: String,
}

/// Query parameters for attestation
#[derive(Debug, Deserialize)]
pub struct AttestationQuery {
    /// Nonce to include in the quote (hex string)
    pub nonce: String,
    /// Optional GPU IDs to filter (comma-separated)
    pub gpu_ids: Option<String>,
}

/// Query parameters for devices
#[derive(Debug, Deserialize)]
pub struct DevicesQuery {
    /// Optional GPU IDs to filter (comma-separated)
    pub gpu_ids: Option<String>,
}

/// Query parameters for TDX quote
#[derive(Debug, Deserialize)]
pub struct TdxQuoteQuery {
    /// Nonce to include in the quote (hex string)
    pub nonce: String,
}

/// Query parameters for NVTrust evidence
#[derive(Debug, Deserialize)]
pub struct EvidenceQuery {
    /// Node name
    pub name: Option<String>,
    /// Nonce to include
    pub nonce: Option<String>,
    /// Optional GPU IDs to filter (comma-separated)
    pub gpu_ids: Option<String>,
}

/// Error response
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

/// Attestation request handlers
pub struct AttestationHandlers;

impl AttestationHandlers {
    /// Health check endpoint
    pub async fn health() -> &'static str {
        "pong"
    }

    /// Combined attestation endpoint
    ///
    /// Returns both TDX quote and NVTrust evidence using TeeService
    pub async fn attest(
        State(state): State<Arc<ServerState>>,
        Query(params): Query<AttestationQuery>,
    ) -> Result<Json<AttestationResponse>, (StatusCode, Json<ErrorResponse>)> {
        info!("[Attestation] Generating attestation with nonce");

        let gpu_ids = params.gpu_ids.map(|s| {
            s.split(',')
                .map(|id| id.trim().to_string())
                .collect::<Vec<_>>()
        });

        let result = state
            .service
            .attest(&params.nonce, gpu_ids.as_deref())
            .await
            .map_err(|e| {
                error!("[Attestation] Failed to generate attestation: {}", e);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: format!("Attestation failed: {}", e),
                    }),
                )
            })?;

        Ok(Json(AttestationResponse::from(result)))
    }

    /// Get GPU devices
    pub async fn get_devices(
        State(state): State<Arc<ServerState>>,
        Query(params): Query<DevicesQuery>,
    ) -> Result<Json<Vec<GpuDeviceInfo>>, (StatusCode, Json<ErrorResponse>)> {
        let gpu_ids = params.gpu_ids.map(|s| {
            s.split(',')
                .map(|id| id.trim().to_string())
                .collect::<Vec<_>>()
        });

        let devices = state
            .gpu_provider
            .get_device_info(gpu_ids.as_deref())
            .map_err(|e| {
                error!("[Devices] Failed to get device info: {}", e);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: format!("Failed to get device info: {}", e),
                    }),
                )
            })?;

        Ok(Json(devices))
    }

    /// Get TDX quote only
    pub async fn get_tdx_quote(
        State(state): State<Arc<ServerState>>,
        Query(params): Query<TdxQuoteQuery>,
    ) -> Result<String, (StatusCode, Json<ErrorResponse>)> {
        let result = state
            .service
            .attest(&params.nonce, None)
            .await
            .map_err(|e| {
                error!("[TDX] Failed to generate quote: {}", e);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: format!("TDX quote generation failed: {}", e),
                    }),
                )
            })?;

        result.tdx_quote.ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: "TDX quote not available".to_string(),
                }),
            )
        })
    }

    /// Get NVTrust evidence only
    pub async fn get_nvtrust_evidence(
        State(state): State<Arc<ServerState>>,
        Query(params): Query<EvidenceQuery>,
    ) -> Result<String, (StatusCode, Json<ErrorResponse>)> {
        let gpu_ids = params.gpu_ids.map(|s| {
            s.split(',')
                .map(|id| id.trim().to_string())
                .collect::<Vec<_>>()
        });

        let nonce = params.nonce.as_deref().unwrap_or("");

        let result = state
            .service
            .attest(nonce, gpu_ids.as_deref())
            .await
            .map_err(|e| {
                error!("[NVTrust] Failed to generate evidence: {}", e);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: format!("NVTrust evidence generation failed: {}", e),
                    }),
                )
            })?;

        result.gpu_evidence.ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: "NVTrust evidence not available".to_string(),
                }),
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_handler() {
        let result = AttestationHandlers::health().await;
        assert_eq!(result, "pong");
    }

    #[test]
    fn test_attestation_response_serialization() {
        let response = AttestationResponse {
            tdx_quote: Some("base64data".to_string()),
            nvtrust_evidence: Some(r#"{"gpu": "data"}"#.to_string()),
            nonce_hex: "deadbeef".to_string(),
            hostname: "test-host".to_string(),
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("base64data"));
        assert!(json.contains("nvtrust_evidence"));
    }

    #[test]
    fn test_error_response_serialization() {
        let error = ErrorResponse {
            error: "Test error".to_string(),
        };

        let json = serde_json::to_string(&error).unwrap();
        assert!(json.contains("Test error"));
    }

    #[test]
    fn test_from_tee_attestation_result() {
        let tee_result = TeeAttestationResult {
            tdx_quote: Some("quote".to_string()),
            gpu_evidence: Some("evidence".to_string()),
            nonce_hex: "nonce".to_string(),
            hostname: "host".to_string(),
        };

        let response = AttestationResponse::from(tee_result);
        assert_eq!(response.tdx_quote, Some("quote".to_string()));
        assert_eq!(response.hostname, "host");
    }
}
