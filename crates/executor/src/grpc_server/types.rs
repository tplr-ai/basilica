//! gRPC service types and common structures

use crate::ExecutorState;
use anyhow::Result;
use std::sync::Arc;
use tonic::{Request, Response, Status};

/// Common result type for gRPC operations
pub type GrpcResult<T> = Result<T>;

/// Shared state for all gRPC services
pub type SharedExecutorState = Arc<ExecutorState>;

/// Error conversion trait for gRPC errors
pub trait GrpcErrorHandler {
    // TODO: Uncomment when gRPC service is implemented
    // fn to_grpc_error<E: BasilcaError>(error: E) -> Status {
    //     error!("gRPC error: {}", error);
    //     Status::internal(error.to_string())
    // }
}

/// Placeholder request/response types until protobuf generation is working
#[derive(Debug)]
pub struct PlaceholderRequest<T> {
    pub data: T,
}

#[derive(Debug)]
pub struct PlaceholderResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
}

impl<T> PlaceholderResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
        }
    }

    pub fn error(message: String) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(message),
        }
    }
}

// =====================================================================
// Validator Access gRPC Message Types
// (These would normally be generated from protobuf)
// =====================================================================

#[derive(Debug, Clone)]
pub struct ValidatorAccessRequest {
    pub hotkey: String,
    pub access_type: String,
    pub duration_seconds: u64,
    pub auth_token: Option<String>,
    pub ssh_public_key: Option<String>,
    // Enhanced fields for hotkey verification
    pub signature: Option<String>,
    pub public_key: Option<String>,
    pub challenge_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ValidatorAccessResponse {
    pub success: bool,
    pub message: String,
    pub ssh_public_key: Option<String>,
    pub ssh_username: Option<String>,
    pub access_token: Option<String>,
    pub expires_at: u64,
}

#[derive(Debug, Clone)]
pub struct ValidatorRevokeRequest {
    pub hotkey: String,
    pub auth_token: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ValidatorRevokeResponse {
    pub success: bool,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct ValidatorListRequest {
    pub admin_token: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ValidatorListResponse {
    pub success: bool,
    pub message: String,
    pub access_entries: Vec<ValidatorAccessInfo>,
}

#[derive(Debug, Clone)]
pub struct ValidatorAccessInfo {
    pub hotkey: String,
    pub access_type: String,
    pub granted_at: u64,
    pub expires_at: u64,
    pub has_ssh_access: bool,
    pub has_token_access: bool,
}

#[derive(Debug, Clone)]
pub struct ValidatorHealthRequest {}

#[derive(Debug, Clone)]
pub struct ValidatorHealthResponse {
    pub healthy: bool,
    pub message: String,
    pub active_sessions: u32,
    pub total_sessions: u32,
    pub ssh_sessions: u32,
    pub token_sessions: u32,
    pub log_entries: u32,
    pub uptime_seconds: u64,
}

/// Validator service trait for gRPC implementations
#[tonic::async_trait]
pub trait ValidatorServiceTrait {
    async fn request_access(
        &self,
        request: Request<ValidatorAccessRequest>,
    ) -> Result<Response<ValidatorAccessResponse>, Status>;

    async fn revoke_access(
        &self,
        request: Request<ValidatorRevokeRequest>,
    ) -> Result<Response<ValidatorRevokeResponse>, Status>;

    async fn list_access(
        &self,
        request: Request<ValidatorListRequest>,
    ) -> Result<Response<ValidatorListResponse>, Status>;

    async fn health_check(
        &self,
        request: Request<ValidatorHealthRequest>,
    ) -> Result<Response<ValidatorHealthResponse>, Status>;
}
