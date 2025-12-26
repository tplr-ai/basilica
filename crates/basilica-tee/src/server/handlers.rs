//! HTTP Request Handlers for Attestation Server

use super::ServerState;
use crate::types::GpuDeviceInfo;
use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::Json,
};
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{error, info};

/// Attestation response
#[derive(Debug, Serialize, Deserialize)]
pub struct AttestationResponse {
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
    /// Returns both TDX quote and NVTrust evidence
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

        // Generate TDX quote
        let quote_content = state
            .tdx_provider
            .get_quote(&params.nonce)
            .await
            .map_err(|e| {
                error!("[Attestation] Failed to generate TDX quote: {}", e);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: format!("TDX quote generation failed: {}", e),
                    }),
                )
            })?;

        // Generate NVTrust evidence
        let nvtrust_evidence = state
            .nv_provider
            .get_evidence(&state.hostname, &params.nonce, gpu_ids.as_deref())
            .await
            .map_err(|e| {
                error!("[Attestation] Failed to generate NVTrust evidence: {}", e);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: format!("NVTrust evidence generation failed: {}", e),
                    }),
                )
            })?;

        Ok(Json(AttestationResponse {
            tdx_quote: BASE64_STANDARD.encode(&quote_content),
            nvtrust_evidence,
        }))
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
        let quote = state
            .tdx_provider
            .get_quote(&params.nonce)
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

        Ok(BASE64_STANDARD.encode(&quote))
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

        let evidence = state
            .nv_provider
            .get_evidence(
                params.name.as_deref().unwrap_or("unknown"),
                params.nonce.as_deref().unwrap_or(""),
                gpu_ids.as_deref(),
            )
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

        Ok(evidence)
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
            tdx_quote: "base64data".to_string(),
            nvtrust_evidence: r#"{"gpu": "data"}"#.to_string(),
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
}

