//! # Verification Engine
//!
//! Handles the actual verification of miners and their executors.
//! Implements Single Responsibility Principle by focusing only on verification logic.

use super::miner_client::{MinerClient, MinerClientConfig};
use super::types::{ExecutorInfo, ExecutorStatus, MinerInfo};
use crate::config::VerificationConfig;
use crate::ssh::{ExecutorSshDetails, ValidatorSshClient};
use crate::validation::validator::HardwareValidator;
use anyhow::{Context, Result};
use common::identity::{ExecutorId, Hotkey, MinerUid};
use common::ssh::SshConnectionDetails;
use std::collections::HashMap;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

#[derive(Clone)]
pub struct VerificationEngine {
    config: VerificationConfig,
    miner_client_config: MinerClientConfig,
    validator_hotkey: Hotkey,
    ssh_client: Arc<ValidatorSshClient>,
    hardware_validator: Option<Arc<HardwareValidator>>,
    /// Whether to use dynamic discovery or fall back to static config
    use_dynamic_discovery: bool,
    /// SSH key path for executor access (fallback)
    ssh_key_path: Option<PathBuf>,
    /// Cache of miner endpoints for reconnection
    miner_endpoints: Arc<RwLock<HashMap<MinerUid, String>>>,
    /// Optional Bittensor service for signing
    bittensor_service: Option<Arc<bittensor::Service>>,
}

impl VerificationEngine {
    pub fn new(config: VerificationConfig) -> Self {
        warn!("Creating VerificationEngine without validator hotkey - dynamic discovery will not be available");
        Self {
            config: config.clone(),
            miner_client_config: MinerClientConfig {
                timeout: config.discovery_timeout,
                grpc_port_offset: config.grpc_port_offset,
                ..Default::default()
            },
            validator_hotkey: Hotkey::new(
                "5DAAnrj7VHTznn2AWBemMuyBwZWs6FNFjdyVXUeYum3PTXFy".to_string(),
            )
            .unwrap(),
            ssh_client: Arc::new(ValidatorSshClient::new()),
            hardware_validator: None,
            use_dynamic_discovery: false, // Disabled without proper initialization
            ssh_key_path: None,
            miner_endpoints: Arc::new(RwLock::new(HashMap::new())),
            bittensor_service: None,
        }
    }

    /// Create with full configuration for dynamic discovery
    pub fn with_validator_context(
        config: VerificationConfig,
        validator_hotkey: Hotkey,
        ssh_client: Arc<ValidatorSshClient>,
        hardware_validator: Option<Arc<HardwareValidator>>,
        ssh_key_path: Option<PathBuf>,
    ) -> Self {
        let miner_client_config = MinerClientConfig {
            timeout: config.discovery_timeout,
            max_retries: 3,
            grpc_port_offset: config.grpc_port_offset,
            use_tls: false,
        };

        Self {
            config: config.clone(),
            miner_client_config,
            validator_hotkey,
            ssh_client,
            hardware_validator,
            use_dynamic_discovery: config.use_dynamic_discovery,
            ssh_key_path,
            miner_endpoints: Arc::new(RwLock::new(HashMap::new())),
            bittensor_service: None,
        }
    }

    /// Create with full configuration including Bittensor service for signing
    pub fn with_bittensor_service(
        config: VerificationConfig,
        bittensor_service: Arc<bittensor::Service>,
        ssh_client: Arc<ValidatorSshClient>,
        hardware_validator: Option<Arc<HardwareValidator>>,
        ssh_key_path: Option<PathBuf>,
    ) -> Self {
        let validator_hotkey = bittensor::account_id_to_hotkey(bittensor_service.get_account_id())
            .expect("Failed to convert account ID to hotkey");

        let miner_client_config = MinerClientConfig {
            timeout: config.discovery_timeout,
            max_retries: 3,
            grpc_port_offset: config.grpc_port_offset,
            use_tls: false,
        };

        Self {
            config: config.clone(),
            miner_client_config,
            validator_hotkey,
            ssh_client,
            hardware_validator,
            use_dynamic_discovery: config.use_dynamic_discovery,
            ssh_key_path,
            miner_endpoints: Arc::new(RwLock::new(HashMap::new())),
            bittensor_service: Some(bittensor_service),
        }
    }

    /// Verify all executors for a specific miner
    pub async fn verify_miner(&self, miner: MinerInfo) -> Result<f64> {
        info!(
            "Starting executor verification for miner {}",
            miner.uid.as_u16()
        );

        self.connect_to_miner(&miner).await?;

        // Cache the miner endpoint for later use
        {
            let mut endpoints = self.miner_endpoints.write().await;
            endpoints.insert(miner.uid, miner.endpoint.clone());
        }

        let executors = self.request_executor_lease(&miner).await?;

        if executors.is_empty() {
            warn!("No executors available from miner {}", miner.uid.as_u16());
            return Ok(0.0);
        }

        let scores = self.verify_executors(&executors).await;
        let final_score = self.calculate_final_score(&scores);

        info!(
            "Miner {} final verification score: {:.4} (from {} executors)",
            miner.uid.as_u16(),
            final_score,
            scores.len()
        );

        Ok(final_score)
    }

    async fn connect_to_miner(&self, miner: &MinerInfo) -> Result<()> {
        if !self.use_dynamic_discovery {
            info!(
                "Dynamic discovery disabled, using static configuration for miner {}",
                miner.uid.as_u16()
            );
            return Ok(());
        }

        info!(
            "Attempting to connect to miner {} at axon endpoint {}",
            miner.uid.as_u16(),
            miner.endpoint
        );

        // Create miner client with proper signer if available
        let client = if let Some(ref bittensor_service) = self.bittensor_service {
            let signer = Box::new(super::miner_client::BittensorServiceSigner::new(
                bittensor_service.clone(),
            ));
            MinerClient::with_signer(
                self.miner_client_config.clone(),
                self.validator_hotkey.clone(),
                signer,
            )
        } else {
            MinerClient::new(
                self.miner_client_config.clone(),
                self.validator_hotkey.clone(),
            )
        };

        // Test connection by attempting authentication
        match client.connect_and_authenticate(&miner.endpoint).await {
            Ok(_conn) => {
                info!(
                    "Successfully connected and authenticated with miner {} at {}",
                    miner.uid.as_u16(),
                    miner.endpoint
                );
                Ok(())
            }
            Err(e) => {
                if self.config.fallback_to_static {
                    warn!(
                        "Failed to connect to miner {} at {}: {}. Falling back to static config",
                        miner.uid.as_u16(),
                        miner.endpoint,
                        e
                    );
                    Ok(())
                } else {
                    Err(e).context(format!(
                        "Failed to connect to miner {} at {}",
                        miner.uid.as_u16(),
                        miner.endpoint
                    ))
                }
            }
        }
    }

    async fn request_executor_lease(&self, miner: &MinerInfo) -> Result<Vec<ExecutorInfo>> {
        if !self.use_dynamic_discovery {
            // Fallback to static configuration
            return self.get_static_executor_info(miner).await;
        }

        info!(
            "Requesting executor lease from miner {} via dynamic discovery",
            miner.uid.as_u16()
        );

        // Create miner client with proper signer if available
        let client = if let Some(ref bittensor_service) = self.bittensor_service {
            let signer = Box::new(super::miner_client::BittensorServiceSigner::new(
                bittensor_service.clone(),
            ));
            MinerClient::with_signer(
                self.miner_client_config.clone(),
                self.validator_hotkey.clone(),
                signer,
            )
        } else {
            MinerClient::new(
                self.miner_client_config.clone(),
                self.validator_hotkey.clone(),
            )
        };

        // Connect and authenticate
        let mut connection = match client.connect_and_authenticate(&miner.endpoint).await {
            Ok(conn) => conn,
            Err(e) => {
                if self.config.fallback_to_static {
                    warn!(
                        "Failed to connect for executor discovery: {}. Using static config",
                        e
                    );
                    return self.get_static_executor_info(miner).await;
                } else {
                    return Err(e).context("Failed to connect to miner for executor discovery");
                }
            }
        };

        // Request executors with requirements
        let requirements = protocol::common::ResourceLimits {
            max_cpu_cores: 4,
            max_memory_mb: 8192,
            max_storage_mb: 10240,
            max_containers: 1,
            max_bandwidth_mbps: 100.0,
            max_gpus: 1,
        };

        let lease_duration = Duration::from_secs(3600); // 1 hour lease

        match connection
            .request_executors(Some(requirements), lease_duration)
            .await
        {
            Ok(executor_details) => {
                let executors: Vec<ExecutorInfo> = executor_details
                    .into_iter()
                    .map(|details| ExecutorInfo {
                        id: ExecutorId::from_str(&details.executor_id)
                            .unwrap_or_else(|_| ExecutorId::new()),
                        miner_uid: miner.uid,
                        grpc_endpoint: details.grpc_endpoint,
                        last_verified: None,
                        verification_status: ExecutorStatus::Available,
                    })
                    .collect();

                info!(
                    "Received {} executors from miner {}",
                    executors.len(),
                    miner.uid.as_u16()
                );
                Ok(executors)
            }
            Err(e) => {
                if self.config.fallback_to_static {
                    warn!("Failed to request executors: {}. Using static config", e);
                    self.get_static_executor_info(miner).await
                } else {
                    Err(e).context("Failed to request executors from miner")
                }
            }
        }
    }

    /// Get static executor info (fallback method)
    async fn get_static_executor_info(&self, miner: &MinerInfo) -> Result<Vec<ExecutorInfo>> {
        // This would normally load from configuration or database
        // For now, return empty to indicate no static config available
        warn!(
            "No static executor configuration available for miner {}",
            miner.uid.as_u16()
        );
        Ok(vec![])
    }

    async fn verify_executors(&self, executors: &[ExecutorInfo]) -> Vec<f64> {
        let mut scores = Vec::new();

        for executor in executors {
            match self.verify_single_executor(executor).await {
                Ok(score) => {
                    scores.push(score);
                    info!("Executor {} verified with score: {:.4}", executor.id, score);
                }
                Err(e) => {
                    scores.push(0.0);
                    warn!("Executor {} verification failed: {}", executor.id, e);
                }
            }
        }

        scores
    }

    async fn verify_single_executor(&self, executor: &ExecutorInfo) -> Result<f64> {
        info!("Verifying executor {}", executor.id);

        // If we have dynamic discovery enabled, we need to get SSH credentials
        if self.use_dynamic_discovery {
            return self.verify_executor_dynamic(executor).await;
        }

        // Fallback to static verification (placeholder for now)
        warn!(
            "Static verification not implemented for executor {}",
            executor.id
        );
        Ok(0.0)
    }

    /// Verify executor using dynamic SSH discovery
    async fn verify_executor_dynamic(&self, executor: &ExecutorInfo) -> Result<f64> {
        info!(
            "Using dynamic discovery to verify executor {} from miner {}",
            executor.id,
            executor.miner_uid.as_u16()
        );

        // Get miner endpoint from cache
        let miner_endpoint = {
            let endpoints = self.miner_endpoints.read().await;
            endpoints.get(&executor.miner_uid).cloned().ok_or_else(|| {
                anyhow::anyhow!(
                    "Miner endpoint not found in cache for miner {}",
                    executor.miner_uid.as_u16()
                )
            })?
        };

        // Create miner client with proper signer if available
        let client = if let Some(ref bittensor_service) = self.bittensor_service {
            let signer = Box::new(super::miner_client::BittensorServiceSigner::new(
                bittensor_service.clone(),
            ));
            MinerClient::with_signer(
                self.miner_client_config.clone(),
                self.validator_hotkey.clone(),
                signer,
            )
        } else {
            MinerClient::new(
                self.miner_client_config.clone(),
                self.validator_hotkey.clone(),
            )
        };

        // Connect and authenticate
        let mut connection = client
            .connect_and_authenticate(&miner_endpoint)
            .await
            .context("Failed to reconnect to miner for SSH session")?;

        // Request SSH session for the executor
        let session_info = connection
            .initiate_ssh_session(&executor.id.to_string(), "verification")
            .await
            .context("Failed to initiate SSH session")?;

        info!(
            "Received SSH credentials for executor {}: session_id={}",
            executor.id, session_info.session_id
        );

        // Parse SSH connection details from the access_credentials
        // The format is expected to be: "username@host:port"
        let ssh_details = self.parse_ssh_credentials(&session_info.access_credentials)?;

        // Create ExecutorSshDetails
        let executor_ssh_details = ExecutorSshDetails::new(
            executor.id.clone(),
            ssh_details.host,
            ssh_details.username,
            Some(ssh_details.port),
            self.ssh_key_path
                .clone()
                .unwrap_or_else(|| PathBuf::from("/tmp/validator_key")),
            Some(self.config.challenge_timeout),
        );

        // Use hardware validator if available
        if self.hardware_validator.is_some() {
            // For now, skip hardware validation as it requires mutable access
            // which is not compatible with Arc<HardwareValidator>
            // This needs to be refactored to use Arc<Mutex<HardwareValidator>>
            warn!(
                "Hardware validation skipped for executor {} - requires refactoring for Arc compatibility",
                executor.id
            );

            // Simplified verification without hardware validator
            match self
                .ssh_client
                .test_connection(&executor_ssh_details.connection)
                .await
            {
                Ok(_) => {
                    info!(
                        "SSH connection test successful for executor {}",
                        executor.id
                    );
                    Ok(0.8) // Partial score for successful connection
                }
                Err(e) => {
                    error!(
                        "SSH connection test failed for executor {}: {}",
                        executor.id, e
                    );
                    Ok(0.0)
                }
            }
        } else {
            // Simplified verification without hardware validator
            match self
                .ssh_client
                .test_connection(&executor_ssh_details.connection)
                .await
            {
                Ok(_) => {
                    info!(
                        "SSH connection test successful for executor {}",
                        executor.id
                    );
                    Ok(0.8) // Partial score for successful connection
                }
                Err(e) => {
                    error!(
                        "SSH connection test failed for executor {}: {}",
                        executor.id, e
                    );
                    Ok(0.0)
                }
            }
        }
    }

    /// Parse SSH credentials string into connection details
    pub fn parse_ssh_credentials(&self, credentials: &str) -> Result<SshConnectionDetails> {
        // Expected format: "username@host:port" or just "username@host"
        let parts: Vec<&str> = credentials.split('@').collect();
        if parts.len() != 2 {
            return Err(anyhow::anyhow!(
                "Invalid SSH credentials format: expected username@host[:port]"
            ));
        }

        let username = parts[0].to_string();
        let host_port = parts[1];

        let (host, port) = if let Some(colon_pos) = host_port.rfind(':') {
            let host = host_port[..colon_pos].to_string();
            let port = host_port[colon_pos + 1..]
                .parse::<u16>()
                .context("Invalid port number")?;
            (host, port)
        } else {
            (host_port.to_string(), 22)
        };

        Ok(SshConnectionDetails {
            host,
            port,
            username,
            private_key_path: self
                .ssh_key_path
                .clone()
                .unwrap_or_else(|| PathBuf::from("/tmp/validator_key")),
            timeout: self.config.challenge_timeout,
        })
    }

    fn calculate_final_score(&self, scores: &[f64]) -> f64 {
        if scores.is_empty() {
            return 0.0;
        }

        scores.iter().sum::<f64>() / scores.len() as f64
    }
}
