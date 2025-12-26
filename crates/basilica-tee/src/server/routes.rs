//! HTTP Routes for Attestation Server

use super::handlers::AttestationHandlers;
use super::ServerState;
use axum::{
    routing::get,
    Router,
};
use std::sync::Arc;

/// Create the attestation server router
pub fn create_router(state: Arc<ServerState>) -> Router {
    Router::new()
        .route("/health", get(AttestationHandlers::health))
        .route("/attest", get(AttestationHandlers::attest))
        .route("/devices", get(AttestationHandlers::get_devices))
        .route("/tdx/quote", get(AttestationHandlers::get_tdx_quote))
        .route("/nvtrust/evidence", get(AttestationHandlers::get_nvtrust_evidence))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::{GpuDeviceProvider, NvEvidenceProvider};
    use crate::tdx::TdxQuoteProvider;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    fn create_test_state() -> Arc<ServerState> {
        Arc::new(ServerState {
            hostname: "test-host".to_string(),
            tdx_provider: TdxQuoteProvider::new(),
            nv_provider: NvEvidenceProvider::new(),
            gpu_provider: GpuDeviceProvider::default(),
        })
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        let state = create_test_state();
        let router = create_router(state);

        let response = router
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_attest_requires_nonce() {
        let state = create_test_state();
        let router = create_router(state);

        // Without nonce parameter
        let response = router
            .oneshot(Request::builder().uri("/attest").body(Body::empty()).unwrap())
            .await
            .unwrap();

        // Should fail due to missing nonce
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_devices_endpoint() {
        let state = create_test_state();
        let router = create_router(state);

        let response = router
            .oneshot(Request::builder().uri("/devices").body(Body::empty()).unwrap())
            .await
            .unwrap();

        // May succeed or fail depending on NVML availability
        // Just check it doesn't panic
        assert!(response.status() == StatusCode::OK || response.status() == StatusCode::INTERNAL_SERVER_ERROR);
    }
}

