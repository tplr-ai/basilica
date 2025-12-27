//! Attestation HTTP Server
//!
//! Provides an HTTP API for generating and verifying attestation evidence.
//! Requires the `server` feature to be enabled.

mod handlers;
mod routes;

pub use handlers::AttestationHandlers;
pub use routes::create_router;

use std::net::SocketAddr;
use std::sync::Arc;
use tracing::info;

use crate::config::TeeConfig;
use crate::error::TeeResult;
use crate::gpu::GpuDeviceProvider;
use crate::service::TeeService;

/// Attestation Server
///
/// HTTP server providing attestation endpoints for TDX quotes and GPU evidence.
/// Uses [`TeeService`] for attestation operations.
pub struct AttestationServer {
    /// TEE service for attestation operations
    service: Arc<TeeService>,
    /// GPU device provider for device info
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
                .map_err(|e| {
                    crate::error::TeeError::InvalidConfig(format!("Invalid bind address: {}", e))
                })?
        } else {
            "0.0.0.0:8443".parse().unwrap()
        };

        let service = TeeService::from_tee_config(config, hostname)?;

        Ok(Self {
            service: Arc::new(service),
            gpu_provider: GpuDeviceProvider::new()?,
            bind_addr,
        })
    }

    /// Create a server with a custom TeeService (for testing)
    pub fn with_service(
        service: Arc<TeeService>,
        gpu_provider: GpuDeviceProvider,
        bind_addr: SocketAddr,
    ) -> Self {
        Self {
            service,
            gpu_provider,
            bind_addr,
        }
    }

    /// Get the server's hostname
    pub fn hostname(&self) -> &str {
        self.service.hostname()
    }

    /// Get the bind address
    pub fn bind_addr(&self) -> SocketAddr {
        self.bind_addr
    }

    /// Get a reference to the TEE service
    pub fn service(&self) -> &Arc<TeeService> {
        &self.service
    }

    /// Create the Axum router
    pub fn router(self) -> axum::Router {
        let state = Arc::new(ServerState {
            service: self.service,
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
            .map_err(crate::error::TeeError::Io)?;

        axum::serve(listener, router)
            .await
            .map_err(|e| crate::error::TeeError::Io(std::io::Error::other(e)))?;

        Ok(())
    }
}

/// Shared server state
pub struct ServerState {
    /// TEE service for attestation operations
    pub service: Arc<TeeService>,
    /// GPU device provider for device info
    pub gpu_provider: GpuDeviceProvider,
}

impl ServerState {
    /// Get the hostname from the service
    pub fn hostname(&self) -> &str {
        self.service.hostname()
    }
}

// Deprecated constructors for backward compatibility
impl AttestationServer {
    /// Create with custom providers (deprecated, use with_service instead)
    #[deprecated(note = "Use with_service instead")]
    pub fn with_providers(
        hostname: String,
        _tdx_provider: crate::tdx::TdxQuoteProvider,
        _nv_provider: crate::gpu::NvEvidenceProvider,
        gpu_provider: GpuDeviceProvider,
        bind_addr: SocketAddr,
    ) -> Self {
        // Create a service with default providers
        let service = TeeService::new(hostname).unwrap();
        Self {
            service: Arc::new(service),
            gpu_provider,
            bind_addr,
        }
    }
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

    #[test]
    fn test_with_service() {
        let service = TeeService::builder("test-host".to_string())
            .enable_tdx(false)
            .enable_gpu(false)
            .build()
            .unwrap();

        let server = AttestationServer::with_service(
            Arc::new(service),
            GpuDeviceProvider::default(),
            "127.0.0.1:8080".parse().unwrap(),
        );

        assert_eq!(server.hostname(), "test-host");
        assert_eq!(server.bind_addr().port(), 8080);
    }
}
