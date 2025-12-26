//! Attestation HTTP Server
//!
//! Provides an HTTP API for generating and verifying attestation evidence.
//! Requires the `server` feature to be enabled.

mod handlers;
mod routes;

pub use handlers::AttestationHandlers;
pub use routes::create_router;

use crate::config::TeeConfig;
use crate::error::TeeResult;
use crate::gpu::{GpuDeviceProvider, NvEvidenceProvider};
use crate::tdx::TdxQuoteProvider;
use std::net::SocketAddr;
use std::sync::Arc;
use tracing::info;

/// Attestation Server
///
/// HTTP server providing attestation endpoints for TDX quotes and GPU evidence.
pub struct AttestationServer {
    /// Server hostname for identification
    hostname: String,
    /// TDX quote provider
    tdx_provider: TdxQuoteProvider,
    /// NVIDIA evidence provider
    nv_provider: NvEvidenceProvider,
    /// GPU device provider
    gpu_provider: GpuDeviceProvider,
    /// Server bind address
    bind_addr: SocketAddr,
}

impl AttestationServer {
    /// Create a new attestation server
    pub fn new(hostname: String, config: &TeeConfig) -> TeeResult<Self> {
        let bind_addr = if let Some(ref server_config) = config.attestation_server {
            format!("{}:{}", server_config.bind_host, server_config.bind_port)
                .parse()
                .map_err(|e| crate::error::TeeError::InvalidConfig(format!("Invalid bind address: {}", e)))?
        } else {
            "0.0.0.0:8443".parse().unwrap()
        };

        Ok(Self {
            hostname,
            tdx_provider: TdxQuoteProvider::from_config(&config.tdx),
            nv_provider: NvEvidenceProvider::from_config(&config.gpu),
            gpu_provider: GpuDeviceProvider::new()?,
            bind_addr,
        })
    }

    /// Create a server with custom providers (for testing)
    pub fn with_providers(
        hostname: String,
        tdx_provider: TdxQuoteProvider,
        nv_provider: NvEvidenceProvider,
        gpu_provider: GpuDeviceProvider,
        bind_addr: SocketAddr,
    ) -> Self {
        Self {
            hostname,
            tdx_provider,
            nv_provider,
            gpu_provider,
            bind_addr,
        }
    }

    /// Get the server's hostname
    pub fn hostname(&self) -> &str {
        &self.hostname
    }

    /// Get the bind address
    pub fn bind_addr(&self) -> SocketAddr {
        self.bind_addr
    }

    /// Create the Axum router
    pub fn router(self) -> axum::Router {
        let state = Arc::new(ServerState {
            hostname: self.hostname,
            tdx_provider: self.tdx_provider,
            nv_provider: self.nv_provider,
            gpu_provider: self.gpu_provider,
        });

        create_router(state)
    }

    /// Run the server
    pub async fn run(self) -> TeeResult<()> {
        let bind_addr = self.bind_addr;
        let router = self.router();

        info!("Starting attestation server on {}", bind_addr);

        let listener = tokio::net::TcpListener::bind(bind_addr)
            .await
            .map_err(|e| crate::error::TeeError::Io(e))?;

        axum::serve(listener, router)
            .await
            .map_err(|e| crate::error::TeeError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

        Ok(())
    }
}

/// Shared server state
pub struct ServerState {
    /// Server hostname
    pub hostname: String,
    /// TDX quote provider
    pub tdx_provider: TdxQuoteProvider,
    /// NVIDIA evidence provider
    pub nv_provider: NvEvidenceProvider,
    /// GPU device provider
    pub gpu_provider: GpuDeviceProvider,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_creation() {
        let config = TeeConfig::default();
        let result = AttestationServer::new("test-host".to_string(), &config);
        assert!(result.is_ok());

        let server = result.unwrap();
        assert_eq!(server.hostname(), "test-host");
    }

    #[test]
    fn test_default_bind_addr() {
        let config = TeeConfig::default();
        let server = AttestationServer::new("test".to_string(), &config).unwrap();
        assert_eq!(server.bind_addr().port(), 8443);
    }
}

