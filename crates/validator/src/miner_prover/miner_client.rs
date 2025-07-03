//! # Miner Client
//!
//! gRPC client for communicating with miners' MinerDiscovery service.
//! Handles authentication, executor discovery, and SSH session initialization.

use anyhow::{Context, Result};
use std::sync::Arc;
use std::time::Duration;
use tonic::transport::Channel;
use tracing::{debug, info, warn};

use common::identity::Hotkey;
use protocol::miner_discovery::{
    miner_discovery_client::MinerDiscoveryClient, ExecutorConnectionDetails, LeaseRequest,
    SessionInitRequest, ValidatorAuthRequest,
};

/// Configuration for the miner client
#[derive(Debug, Clone)]
pub struct MinerClientConfig {
    /// Timeout for gRPC calls
    pub timeout: Duration,
    /// Number of retry attempts
    pub max_retries: u32,
    /// Offset from axon port to gRPC port (default: gRPC port is 8080)
    pub grpc_port_offset: Option<u16>,
    /// Whether to use TLS for gRPC connections
    pub use_tls: bool,
}

impl Default for MinerClientConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            max_retries: 3,
            grpc_port_offset: None, // Will use default port 8080
            use_tls: false,
        }
    }
}

/// Client for communicating with a miner's gRPC service
pub struct MinerClient {
    config: MinerClientConfig,
    validator_hotkey: Hotkey,
    /// Optional signer for creating signatures
    /// In production, this should be provided by the validator's key management
    signer: Option<Box<dyn ValidatorSigner>>,
}

/// Trait for validator signing operations
pub trait ValidatorSigner: Send + Sync {
    /// Sign data with the validator's key
    fn sign(&self, data: &[u8]) -> Result<Vec<u8>>;
}

/// Bittensor service-based signer implementation
pub struct BittensorServiceSigner {
    service: Arc<bittensor::Service>,
}

impl BittensorServiceSigner {
    /// Create a new signer using a Bittensor service
    pub fn new(service: Arc<bittensor::Service>) -> Self {
        Self { service }
    }
}

impl ValidatorSigner for BittensorServiceSigner {
    fn sign(&self, data: &[u8]) -> Result<Vec<u8>> {
        let signature_hex = self
            .service
            .sign_data(data)
            .map_err(|e| anyhow::anyhow!("Failed to sign data: {}", e))?;

        hex::decode(signature_hex).map_err(|e| anyhow::anyhow!("Failed to decode signature: {}", e))
    }
}

impl MinerClient {
    /// Create a new miner client
    pub fn new(config: MinerClientConfig, validator_hotkey: Hotkey) -> Self {
        Self {
            config,
            validator_hotkey,
            signer: None,
        }
    }

    /// Create a new miner client with a signer
    pub fn with_signer(
        config: MinerClientConfig,
        validator_hotkey: Hotkey,
        signer: Box<dyn ValidatorSigner>,
    ) -> Self {
        Self {
            config,
            validator_hotkey,
            signer: Some(signer),
        }
    }

    /// Create a validator signature for authentication
    fn create_validator_signature(&self, nonce: &str) -> Result<String> {
        if let Some(ref signer) = self.signer {
            // Use the provided signer
            let signature_bytes = signer
                .sign(nonce.as_bytes())
                .unwrap_or_else(|e| panic!("Failed to create validator signature: {e}"));
            Ok(hex::encode(signature_bytes))
        } else {
            panic!("No signer provided for validator signature creation");
        }
    }

    /// Extract gRPC endpoint from axon endpoint
    ///
    /// Converts axon endpoint (e.g., "http://1.2.3.4:8091") to gRPC endpoint
    /// using configured port mapping or default port 8080
    pub fn axon_to_grpc_endpoint(&self, axon_endpoint: &str) -> Result<String> {
        // Parse the axon endpoint
        let url = url::Url::parse(axon_endpoint)
            .with_context(|| format!("Failed to parse axon endpoint: {axon_endpoint}"))?;

        let host = url
            .host_str()
            .ok_or_else(|| anyhow::anyhow!("No host in axon endpoint"))?;

        // Determine gRPC port
        let grpc_port = if let Some(offset) = self.config.grpc_port_offset {
            let axon_port = url
                .port()
                .ok_or_else(|| anyhow::anyhow!("No port in axon endpoint"))?;
            axon_port + offset
        } else {
            // Default gRPC port for miners (same as HTTP port)
            8080
        };

        // Build gRPC endpoint
        let scheme = if self.config.use_tls { "https" } else { "http" };
        Ok(format!("{scheme}://{host}:{grpc_port}"))
    }

    /// Connect to a miner and authenticate
    pub async fn connect_and_authenticate(
        &self,
        axon_endpoint: &str,
    ) -> Result<AuthenticatedMinerConnection> {
        let grpc_endpoint = self.axon_to_grpc_endpoint(axon_endpoint)?;
        info!(
            "Connecting to miner gRPC service at {} (from axon: {})",
            grpc_endpoint, axon_endpoint
        );

        // Create channel with timeout
        let channel = Channel::from_shared(grpc_endpoint.clone())
            .with_context(|| format!("Invalid gRPC endpoint: {grpc_endpoint}"))?
            .connect_timeout(self.config.timeout)
            .timeout(self.config.timeout)
            .connect()
            .await
            .with_context(|| format!("Failed to connect to miner at {grpc_endpoint}"))?;

        // Generate authentication request
        let nonce = uuid::Uuid::new_v4().to_string();
        let _timestamp = chrono::Utc::now();

        // Create signature for authentication
        // The signature needs to be created using the validator's keypair
        // Since we have a Hotkey, we need to sign the nonce with it
        // In production, this would use the actual validator's signing key

        // For Bittensor compatibility, we expect the signature to be a hex-encoded string
        // The miner will verify this using verify_bittensor_signature
        let signature = self.create_validator_signature(&nonce)?;

        let auth_request = ValidatorAuthRequest {
            validator_hotkey: self.validator_hotkey.to_string(),
            signature,
            nonce,
            timestamp: Some(protocol::common::Timestamp {
                value: None, // Handle timestamp conversion properly with matching prost versions
            }),
        };

        debug!(
            "Authenticating with miner as validator {}",
            self.validator_hotkey
        );

        // Authenticate with retry logic
        let auth_response = self
            .retry_grpc_call(|| {
                let channel = channel.clone();
                let auth_request = auth_request.clone();
                async move {
                    let mut client = MinerDiscoveryClient::new(channel);
                    client
                        .authenticate_validator(auth_request)
                        .await
                        .map_err(|e| anyhow::anyhow!("Authentication failed: {}", e))
                }
            })
            .await?;

        let auth_response = auth_response.into_inner();

        if !auth_response.authenticated {
            let error_msg = auth_response
                .error
                .map(|e| e.message)
                .unwrap_or_else(|| "Unknown error".to_string());
            return Err(anyhow::anyhow!("Authentication failed: {}", error_msg));
        }

        let session_token = auth_response.session_token;
        info!("Successfully authenticated with miner");

        Ok(AuthenticatedMinerConnection {
            client: MinerDiscoveryClient::new(channel),
            session_token,
            grpc_endpoint,
        })
    }

    /// Retry a gRPC call with exponential backoff
    async fn retry_grpc_call<F, Fut, T>(&self, mut call: F) -> Result<T>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let mut attempt = 0;
        let mut backoff = Duration::from_millis(100);

        loop {
            match call().await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    attempt += 1;
                    if attempt >= self.config.max_retries {
                        return Err(e);
                    }

                    warn!(
                        "gRPC call failed (attempt {}/{}): {}. Retrying in {:?}",
                        attempt, self.config.max_retries, e, backoff
                    );

                    tokio::time::sleep(backoff).await;
                    backoff = (backoff * 2).min(Duration::from_secs(5));
                }
            }
        }
    }
}

/// Authenticated connection to a miner
pub struct AuthenticatedMinerConnection {
    client: MinerDiscoveryClient<Channel>,
    session_token: String,
    /// The gRPC endpoint used for this connection (useful for debugging/logging)
    #[allow(dead_code)]
    grpc_endpoint: String,
}

impl AuthenticatedMinerConnection {
    /// Request available executors from the miner
    pub async fn request_executors(
        &mut self,
        requirements: Option<protocol::common::ResourceLimits>,
        lease_duration: Duration,
    ) -> Result<Vec<ExecutorConnectionDetails>> {
        info!("Requesting available executors from miner");

        let request = LeaseRequest {
            validator_hotkey: String::new(), // Will be extracted from token by miner
            session_token: self.session_token.clone(),
            requirements,
            lease_duration_seconds: lease_duration.as_secs(),
        };

        let response = self
            .client
            .request_executor_lease(request)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to request executors: {}", e))?;

        let response = response.into_inner();

        if let Some(error) = response.error {
            return Err(anyhow::anyhow!(
                "Executor request failed: {}",
                error.message
            ));
        }

        info!(
            "Received {} available executors from miner",
            response.available_executors.len()
        );

        Ok(response.available_executors)
    }

    /// Initiate SSH session with a specific executor
    pub async fn initiate_ssh_session(
        &mut self,
        executor_id: &str,
        session_type: &str,
    ) -> Result<SshSessionInfo> {
        info!(
            "Initiating {} session with executor {}",
            session_type, executor_id
        );

        let request = SessionInitRequest {
            validator_hotkey: String::new(), // Will be extracted from token by miner
            session_token: self.session_token.clone(),
            executor_id: executor_id.to_string(),
            session_type: session_type.to_string(),
        };

        let response = self
            .client
            .initiate_executor_session(request)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to initiate session: {}", e))?;

        let response = response.into_inner();

        if !response.success {
            let error_msg = response
                .error
                .map(|e| e.message)
                .unwrap_or_else(|| "Unknown error".to_string());
            return Err(anyhow::anyhow!("Session initiation failed: {}", error_msg));
        }

        info!(
            "Successfully initiated session {} with executor {}",
            response.session_id, executor_id
        );

        Ok(SshSessionInfo {
            session_id: response.session_id,
            access_credentials: response.access_credentials,
        })
    }
}

/// Information about an SSH session
#[derive(Debug, Clone)]
pub struct SshSessionInfo {
    pub session_id: String,
    pub access_credentials: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_axon_to_grpc_endpoint_default() {
        let config = MinerClientConfig::default();
        let client = MinerClient::new(
            config,
            Hotkey::new("5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string()).unwrap(),
        );

        let axon = "http://192.168.1.100:8091";
        let grpc = client.axon_to_grpc_endpoint(axon).unwrap();
        assert_eq!(grpc, "http://192.168.1.100:8080");
    }

    #[test]
    fn test_axon_to_grpc_endpoint_with_offset() {
        let config = MinerClientConfig {
            grpc_port_offset: Some(1000),
            ..Default::default()
        };
        let client = MinerClient::new(
            config,
            Hotkey::new("5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string()).unwrap(),
        );

        let axon = "http://10.0.0.1:8091";
        let grpc = client.axon_to_grpc_endpoint(axon).unwrap();
        assert_eq!(grpc, "http://10.0.0.1:9091");
    }

    #[test]
    fn test_axon_to_grpc_endpoint_with_tls() {
        let config = MinerClientConfig {
            use_tls: true,
            ..Default::default()
        };
        let client = MinerClient::new(
            config,
            Hotkey::new("5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string()).unwrap(),
        );

        let axon = "http://example.com:8091";
        let grpc = client.axon_to_grpc_endpoint(axon).unwrap();
        assert_eq!(grpc, "https://example.com:8080");
    }
}
