//! # Miner Client
//!
//! gRPC client for communicating with miners' MinerDiscovery service.
//! Handles authentication, node discovery, and SSH session initialization.

use anyhow::{Context, Result};
use std::sync::Arc;
use std::time::Duration;
use tonic::transport::Channel;
use tracing::{debug, error, info, warn};

use basilica_common::identity::Hotkey;
use basilica_protocol::miner_discovery::{
    miner_discovery_client::MinerDiscoveryClient, DiscoverNodesRequest, NodeConnectionDetails,
    ValidatorAuthRequest,
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
    /// Rental session duration in seconds (0 = no predetermined duration)
    pub rental_session_duration: u64,
    /// Whether to require miner signature verification
    pub require_miner_signature: bool,
}

impl Default for MinerClientConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(120), // Increased to 120s for better reliability with slow/distant miners
            max_retries: 3,
            grpc_port_offset: None, // Will use default port 8080
            use_tls: false,
            rental_session_duration: 0, // No predetermined duration by default
            require_miner_signature: true, // Default to requiring signatures for security
        }
    }
}

/// Client for communicating with a miner's gRPC service
pub struct MinerClient {
    config: MinerClientConfig,
    validator_hotkey: Hotkey,
    /// Signer for creating cryptographic signatures using validator's key
    signer: Option<Arc<dyn ValidatorSigner>>,
    /// Validator's SSH public key for node access
    validator_ssh_public_key: Option<String>,
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
            validator_ssh_public_key: None,
        }
    }

    /// Create a new miner client with a signer
    pub fn with_signer(
        config: MinerClientConfig,
        validator_hotkey: Hotkey,
        signer: Arc<dyn ValidatorSigner>,
    ) -> Self {
        Self {
            config,
            validator_hotkey,
            signer: Some(signer),
            validator_ssh_public_key: None,
        }
    }

    /// Set the validator's SSH public key
    pub fn with_ssh_public_key(mut self, ssh_public_key: String) -> Self {
        self.validator_ssh_public_key = Some(ssh_public_key);
        self
    }

    /// Get the configured rental session duration
    pub fn get_rental_session_duration(&self) -> u64 {
        self.config.rental_session_duration
    }

    /// Create a validator signature for authentication (required for initial auth)
    fn create_signature(&self, payload: &str) -> Result<String> {
        self.signer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No signer provided for validator signature creation"))?
            .sign(payload.as_bytes())
            .map(hex::encode)
            .map_err(|e| anyhow::anyhow!("Failed to create validator signature: {e}"))
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
            // Use the same port as the axon endpoint when no offset is configured
            // This handles cases where the miner is behind NAT/proxy and advertises external ports
            url.port()
                .ok_or_else(|| anyhow::anyhow!("No port in axon endpoint"))?
        };

        // Build gRPC endpoint
        let scheme = if self.config.use_tls { "https" } else { "http" };
        Ok(format!("{scheme}://{host}:{grpc_port}"))
    }

    /// Connect to a miner and authenticate
    pub async fn connect_and_authenticate(
        &self,
        miner_uid: u16,
        axon_endpoint: &str,
        target_miner_hotkey: &str,
    ) -> Result<AuthenticatedMinerConnection> {
        let grpc_endpoint = self.axon_to_grpc_endpoint(axon_endpoint)?;
        info!(
            miner_uid = miner_uid,
            "Connecting to miner gRPC service at {} (from axon: {})", grpc_endpoint, axon_endpoint
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

        // Create current timestamp
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| anyhow::anyhow!("Failed to get system time: {}", e))?;

        let timestamp = prost_types::Timestamp {
            seconds: now.as_secs() as i64,
            nanos: now.subsec_nanos() as i32,
        };

        const AUTH_PREFIX: &str = "BASILICA_AUTH_V1";
        let signature_payload = format!(
            "{}:{}:{}:{}",
            AUTH_PREFIX, nonce, target_miner_hotkey, timestamp.seconds
        );
        let signature = self.create_signature(&signature_payload)?;

        debug!(
            miner_uid = miner_uid,
            "Creating canonical auth signature with timestamp {} for target miner {}",
            timestamp.seconds,
            target_miner_hotkey
        );

        let auth_request = ValidatorAuthRequest {
            validator_hotkey: self.validator_hotkey.to_string(),
            signature,
            nonce,
            timestamp: Some(basilica_protocol::common::Timestamp {
                value: Some(timestamp),
            }),
            target_miner_hotkey: target_miner_hotkey.to_string(),
        };

        debug!(
            miner_uid = miner_uid,
            "Authenticating with miner as validator {}", self.validator_hotkey
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

        // Verify miner's signature
        if !auth_response.miner_hotkey.is_empty() && !auth_response.miner_signature.is_empty() {
            debug!(
                miner_uid = miner_uid,
                "Verifying miner signature from hotkey: {}", auth_response.miner_hotkey
            );

            // Parse miner hotkey
            let miner_hotkey = Hotkey::new(auth_response.miner_hotkey.clone())
                .map_err(|e| anyhow::anyhow!("Invalid miner hotkey: {}", e))?;

            // Create canonical data that miner signed
            let validator_hotkey = &self.validator_hotkey;
            let response_nonce = &auth_response.response_nonce;
            let session_token = &auth_response.session_token;
            let canonical_data =
                format!("MINER_AUTH_RESPONSE:{validator_hotkey}:{response_nonce}:{session_token}");

            // Verify miner's signature
            if let Err(e) = bittensor::utils::verify_bittensor_signature(
                &miner_hotkey,
                &auth_response.miner_signature,
                canonical_data.as_bytes(),
            ) {
                warn!(
                    miner_uid = miner_uid,
                    "Miner signature verification failed for {}: {}", auth_response.miner_hotkey, e
                );
                return Err(anyhow::anyhow!(
                    "Miner signature verification failed: {}",
                    e
                ));
            }

            if auth_response.miner_hotkey != target_miner_hotkey {
                error!(
                    miner_uid = miner_uid,
                    "Miner hotkey mismatch! Expected: {}, Got: {}. Possible MITM attack.",
                    target_miner_hotkey,
                    auth_response.miner_hotkey
                );
                return Err(anyhow::anyhow!(
                    "Security violation: Miner hotkey mismatch. Expected {}, but got {}",
                    target_miner_hotkey,
                    auth_response.miner_hotkey
                ));
            }
            debug!(
                miner_uid = miner_uid,
                "Miner hotkey matches expected target, proceeding with signature verification"
            );

            info!(
                miner_uid = miner_uid,
                "Successfully verified miner signature from {}", auth_response.miner_hotkey
            );
        } else if self.config.require_miner_signature {
            // Signature is required but not provided
            error!(
                miner_uid = miner_uid,
                "Miner did not provide required signature for verification"
            );
            return Err(anyhow::anyhow!(
                "Miner authentication response missing required signature"
            ));
        } else {
            // Signature not required and not provided
            warn!(
                miner_uid = miner_uid,
                "Miner did not provide signature for verification (not required by config)"
            );
        }

        let session_token = auth_response.session_token;
        info!(
            miner_uid = miner_uid,
            "Successfully authenticated with miner"
        );

        Ok(AuthenticatedMinerConnection {
            client: MinerDiscoveryClient::new(channel),
            _session_token: session_token,
            validator_hotkey: self.validator_hotkey.clone(),
            signer: self.signer.clone(),
            validator_ssh_public_key: self.validator_ssh_public_key.clone(),
            miner_uid,
            target_miner_hotkey: target_miner_hotkey.to_string(),
        })
    }

    /// Retry a gRPC call with exponential backoff
    async fn retry_grpc_call<F, Fut, T>(&self, mut call: F) -> Result<T>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let mut attempt = 0;
        let mut backoff = Duration::from_millis(500); // Increased initial backoff

        loop {
            match call().await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    attempt += 1;
                    if attempt >= self.config.max_retries {
                        return Err(e.context(format!(
                            "Failed after {} attempts with exponential backoff",
                            self.config.max_retries
                        )));
                    }

                    warn!(
                        "gRPC call failed (attempt {}/{}): {}. Retrying in {:?}",
                        attempt, self.config.max_retries, e, backoff
                    );

                    tokio::time::sleep(backoff).await;
                    backoff = (backoff * 2).min(Duration::from_secs(10)); // Increased max backoff
                }
            }
        }
    }
}

/// Authenticated connection to a miner
pub struct AuthenticatedMinerConnection {
    client: MinerDiscoveryClient<Channel>,
    _session_token: String,
    /// Validator's actual hotkey
    validator_hotkey: Hotkey,
    /// Optional signer for creating signatures
    signer: Option<Arc<dyn ValidatorSigner>>,
    /// Validator's SSH public key
    validator_ssh_public_key: Option<String>,
    /// Miner UID
    miner_uid: u16,
    /// Target miner hotkey
    target_miner_hotkey: String,
}

impl AuthenticatedMinerConnection {
    /// Create a validator signature for a payload, returning empty string if unavailable
    fn create_signature(&self, payload: &str) -> String {
        self.signer
            .as_ref()
            .and_then(|signer| {
                signer
                    .sign(payload.as_bytes())
                    .map(hex::encode)
                    .map_err(|e| {
                        debug!("Failed to create signature: {}", e);
                        e
                    })
                    .ok()
            })
            .unwrap_or_default()
    }

    /// Request available nodes from the miner
    pub async fn request_nodes(&mut self) -> Result<Vec<NodeConnectionDetails>> {
        info!(
            miner_uid = self.miner_uid,
            "Requesting available nodes from miner"
        );

        // Create request with authentication fields
        let now = chrono::Utc::now();
        let timestamp = basilica_protocol::common::Timestamp {
            value: Some(prost_types::Timestamp {
                seconds: now.timestamp(),
                nanos: now.timestamp_subsec_nanos() as i32,
            }),
        };

        let nonce = uuid::Uuid::new_v4().to_string();

        // Create signature if signer is available
        const DISCOVER_PREFIX: &str = "BASILICA_DISCOVER_V1";
        let signature_payload = format!(
            "{}:{}:{}:{}",
            DISCOVER_PREFIX,
            self.validator_hotkey,
            nonce,
            timestamp.value.as_ref().map(|t| t.seconds).unwrap_or(0)
        );

        // Create signature using helper method
        let signature = self.create_signature(&signature_payload);

        let request = DiscoverNodesRequest {
            validator_hotkey: self.validator_hotkey.to_string(),
            signature,
            nonce,
            validator_public_key: self.validator_ssh_public_key.clone().unwrap_or_default(),
            timestamp: Some(timestamp),
            target_miner_hotkey: self.target_miner_hotkey.to_string(),
        };

        let response = self
            .client
            .discover_nodes(request)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to discover nodes: {}", e))?;

        let response = response.into_inner();

        info!(
            miner_uid = self.miner_uid,
            "Received {} available nodes from miner",
            response.nodes.len()
        );

        Ok(response.nodes)
    }
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
        assert_eq!(grpc, "http://192.168.1.100:8091");
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
        assert_eq!(grpc, "https://example.com:8091");
    }

    #[test]
    fn test_miner_signature_verification_config() {
        // Test default config requires signature
        let config = MinerClientConfig::default();
        assert!(config.require_miner_signature);

        // Test custom config without signature requirement
        let config_no_sig = MinerClientConfig {
            require_miner_signature: false,
            ..Default::default()
        };
        assert!(!config_no_sig.require_miner_signature);
    }

    #[test]
    fn test_canonical_data_format_for_miner_response() {
        // Test that canonical data format matches between miner and validator
        let validator_hotkey = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY";
        let response_nonce = "test-nonce-123";
        let session_token = "test-session-token";

        let canonical_data =
            format!("MINER_AUTH_RESPONSE:{validator_hotkey}:{response_nonce}:{session_token}");

        // Verify format
        assert!(canonical_data.starts_with("MINER_AUTH_RESPONSE:"));
        assert!(canonical_data.contains(validator_hotkey));
        assert!(canonical_data.contains(response_nonce));
        assert!(canonical_data.contains(session_token));

        // Verify no extra colons or formatting issues
        let parts: Vec<&str> = canonical_data.split(':').collect();
        assert_eq!(parts.len(), 4);
        assert_eq!(parts[0], "MINER_AUTH_RESPONSE");
        assert_eq!(parts[1], validator_hotkey);
        assert_eq!(parts[2], response_nonce);
        assert_eq!(parts[3], session_token);
    }
}
