//! # Verification Engine
//!
//! Handles the actual verification of miners and their executors.
//! Implements Single Responsibility Principle by focusing only on verification logic.

use super::miner_client::{MinerClient, MinerClientConfig};
use super::types::{ExecutorInfo, ExecutorStatus, MinerInfo};
use crate::config::VerificationConfig;
use crate::ssh::{ExecutorSshDetails, ValidatorSshClient, ValidatorSshKeyManager};
use crate::validation::types::AttestationResult;
use crate::validation::validator::HardwareValidator;
use anyhow::{Context, Result};
use common::identity::{ExecutorId, Hotkey, MinerUid};
use common::ssh::SshConnectionDetails;
use protocol::miner_discovery::{
    CloseSshSessionRequest, InitiateSshSessionRequest, SshSessionStatus,
};
use std::collections::HashMap;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tracing::{error, info, warn};
use uuid::Uuid;

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
    /// SSH key manager for session keys
    ssh_key_manager: Option<Arc<ValidatorSshKeyManager>>,
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
            ssh_key_manager: None,
        }
    }

    /// Initiate automated SSH session setup with miner during discovery handshake
    pub async fn initiate_discovery_ssh_handshake(
        &self,
        miner: &MinerInfo,
        executors: &[ExecutorInfo],
    ) -> Result<()> {
        if !self.use_dynamic_discovery {
            info!(
                "Dynamic discovery disabled, skipping SSH handshake for miner {}",
                miner.uid.as_u16()
            );
            return Ok(());
        }

        let ssh_key_manager = match &self.ssh_key_manager {
            Some(manager) => manager,
            None => {
                warn!(
                    "No SSH key manager available for discovery handshake with miner {}",
                    miner.uid.as_u16()
                );
                return Ok(());
            }
        };

        info!(
            "Initiating SSH discovery handshake with miner {} for {} executors",
            miner.uid.as_u16(),
            executors.len()
        );

        // Create authenticated miner client
        let client = self.create_authenticated_client()?;
        let mut connection = client
            .connect_and_authenticate(&miner.endpoint)
            .await
            .context("Failed to connect to miner for SSH handshake")?;

        // Process each executor for SSH key distribution
        for executor in executors {
            if let Err(e) = self
                .setup_executor_ssh_access(&mut connection, executor, ssh_key_manager)
                .await
            {
                error!(
                    "Failed to setup SSH access for executor {}: {}",
                    executor.id, e
                );
            }
        }

        info!(
            "Completed SSH discovery handshake with miner {}",
            miner.uid.as_u16()
        );
        Ok(())
    }

    /// Setup SSH access for a specific executor during discovery
    async fn setup_executor_ssh_access(
        &self,
        connection: &mut super::miner_client::AuthenticatedMinerConnection,
        executor: &ExecutorInfo,
        ssh_key_manager: &ValidatorSshKeyManager,
    ) -> Result<()> {
        // Generate ephemeral SSH keypair for this executor
        let session_id = Uuid::new_v4().to_string();
        let (_, public_key, _) = ssh_key_manager
            .generate_session_keypair(&session_id)
            .await
            .context("Failed to generate SSH keypair for executor")?;

        let public_key_openssh = ValidatorSshKeyManager::get_public_key_openssh(&public_key)
            .context("Failed to convert public key to OpenSSH format")?;

        // Request SSH session setup with validator's public key
        let session_request = InitiateSshSessionRequest {
            validator_hotkey: self.validator_hotkey.to_string(),
            executor_id: executor.id.to_string(),
            purpose: "discovery_handshake".to_string(),
            validator_public_key: public_key_openssh,
            session_duration_secs: 86400, // 24 hours for discovery sessions
            session_metadata: serde_json::json!({
                "validator_version": env!("CARGO_PKG_VERSION"),
                "handshake_type": "discovery",
                "session_id": session_id
            })
            .to_string(),
        };

        match connection.initiate_ssh_session_v2(session_request).await {
            Ok(response) => {
                if response.status() == SshSessionStatus::Active {
                    info!(
                        "Successfully setup SSH access for executor {} (session: {})",
                        executor.id, response.session_id
                    );
                } else {
                    warn!(
                        "SSH session setup incomplete for executor {}: status={:?}",
                        executor.id, response.status
                    );
                }
            }
            Err(e) => {
                error!(
                    "Failed SSH session setup for executor {}: {}",
                    executor.id, e
                );
                // Clean up generated key on failure
                if let Err(cleanup_err) = ssh_key_manager.cleanup_session_keys(&session_id).await {
                    warn!(
                        "Failed to cleanup SSH keys after setup failure: {}",
                        cleanup_err
                    );
                }
                return Err(e);
            }
        }

        Ok(())
    }

    /// Check if this verification engine supports batch processing
    pub fn supports_batch_processing(&self) -> bool {
        // Automated SSH verification supports batch processing when properly configured
        self.use_dynamic_discovery && self.ssh_key_manager.is_some()
    }

    /// Execute automated verification workflow for discovered miners
    pub async fn execute_automated_verification_workflow(
        &self,
        miners: &[MinerInfo],
    ) -> Result<HashMap<MinerUid, f64>> {
        let mut results = HashMap::new();

        info!(
            "Starting automated verification workflow for {} miners",
            miners.len()
        );

        for miner in miners {
            match self.verify_miner_with_ssh_automation(miner.clone()).await {
                Ok(score) => {
                    results.insert(miner.uid, score);
                    info!(
                        "Automated verification completed for miner {} with score: {:.4}",
                        miner.uid.as_u16(),
                        score
                    );
                }
                Err(e) => {
                    results.insert(miner.uid, 0.0);
                    error!(
                        "Automated verification failed for miner {}: {}",
                        miner.uid.as_u16(),
                        e
                    );
                }
            }
        }

        info!(
            "Completed automated verification workflow for {} miners",
            miners.len()
        );
        Ok(results)
    }

    /// Verify miner with full SSH automation including discovery handshake
    async fn verify_miner_with_ssh_automation(&self, miner: MinerInfo) -> Result<f64> {
        info!(
            "Starting SSH-automated verification for miner {}",
            miner.uid.as_u16()
        );

        // Step 1: Connect to miner and discover executors
        self.connect_to_miner(&miner).await?;
        let executors = self.request_executor_lease(&miner).await?;

        if executors.is_empty() {
            warn!("No executors available from miner {}", miner.uid.as_u16());
            return Ok(0.0);
        }

        // Step 2: Perform SSH discovery handshake
        self.initiate_discovery_ssh_handshake(&miner, &executors)
            .await?;

        // Step 3: Execute verification on each executor
        let scores = self.verify_executors_with_automation(&executors).await;
        let final_score = self.calculate_final_score(&scores);

        info!(
            "SSH-automated verification completed for miner {} with final score: {:.4}",
            miner.uid.as_u16(),
            final_score
        );

        Ok(final_score)
    }

    /// Verify executors with enhanced automation
    async fn verify_executors_with_automation(&self, executors: &[ExecutorInfo]) -> Vec<f64> {
        let mut scores = Vec::new();

        for executor in executors {
            match self
                .verify_executor_with_enhanced_automation(executor)
                .await
            {
                Ok(score) => {
                    scores.push(score);
                    info!("Enhanced automation verification completed for executor {} with score: {:.4}", 
                          executor.id, score);
                }
                Err(e) => {
                    scores.push(0.0);
                    warn!(
                        "Enhanced automation verification failed for executor {}: {}",
                        executor.id, e
                    );
                }
            }
        }

        scores
    }

    /// Verify executor with enhanced automation including rate limiting and retry logic
    async fn verify_executor_with_enhanced_automation(
        &self,
        executor: &ExecutorInfo,
    ) -> Result<f64> {
        let max_retries = 3;
        let mut retry_count = 0;

        while retry_count < max_retries {
            match self.verify_executor_dynamic(executor).await {
                Ok(score) => return Ok(score),
                Err(e) if retry_count < max_retries - 1 => {
                    retry_count += 1;
                    let delay = Duration::from_millis(1000 * retry_count as u64);
                    warn!(
                        "Verification attempt {} failed for executor {}: {}. Retrying in {:?}",
                        retry_count, executor.id, e, delay
                    );
                    tokio::time::sleep(delay).await;
                }
                Err(e) => {
                    error!(
                        "All verification attempts failed for executor {}: {}",
                        executor.id, e
                    );
                    return Err(e);
                }
            }
        }

        unreachable!()
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
            ssh_key_manager: None,
        }
    }

    /// Create with full configuration including SSH key manager
    pub async fn with_ssh_key_manager(
        config: VerificationConfig,
        validator_hotkey: Hotkey,
        ssh_client: Arc<ValidatorSshClient>,
        hardware_validator: Option<Arc<HardwareValidator>>,
        ssh_key_manager: Arc<ValidatorSshKeyManager>,
    ) -> Result<Self> {
        let miner_client_config = MinerClientConfig {
            timeout: config.discovery_timeout,
            max_retries: 3,
            grpc_port_offset: config.grpc_port_offset,
            use_tls: false,
        };

        Ok(Self {
            config: config.clone(),
            miner_client_config,
            validator_hotkey,
            ssh_client,
            hardware_validator,
            use_dynamic_discovery: config.use_dynamic_discovery,
            ssh_key_path: None,
            miner_endpoints: Arc::new(RwLock::new(HashMap::new())),
            bittensor_service: None,
            ssh_key_manager: Some(ssh_key_manager),
        })
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
            ssh_key_manager: None,
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

        // Step 1: Generate ephemeral SSH keypair if we have key manager
        let (session_id, public_key_openssh, key_path) = if let Some(ref key_manager) =
            self.ssh_key_manager
        {
            let session_id = Uuid::new_v4().to_string();
            let (_, public_key, key_path) = key_manager
                .generate_session_keypair(&session_id)
                .await
                .context("Failed to generate SSH keypair")?;

            let public_key_openssh = ValidatorSshKeyManager::get_public_key_openssh(&public_key)
                .context("Failed to convert public key to OpenSSH format")?;

            (session_id, public_key_openssh, key_path)
        } else {
            // Fallback to legacy mode without key generation
            warn!("No SSH key manager available, using legacy SSH session mode");
            let session_id = Uuid::new_v4().to_string();
            let fallback_key_path = self
                .ssh_key_path
                .clone()
                .unwrap_or_else(|| PathBuf::from("/tmp/validator_key"));
            (session_id, String::new(), fallback_key_path)
        };

        // Get miner endpoint from cache
        let miner_endpoint = self.get_miner_endpoint(&executor.miner_uid).await?;

        // Create miner client with proper signer if available
        let client = self.create_authenticated_client()?;

        // Connect and authenticate
        let mut connection = client
            .connect_and_authenticate(&miner_endpoint)
            .await
            .context("Failed to reconnect to miner for SSH session")?;

        // Step 3: Request SSH session with public key
        let session_request = InitiateSshSessionRequest {
            validator_hotkey: self.validator_hotkey.to_string(),
            executor_id: executor.id.to_string(),
            purpose: "hardware_attestation".to_string(),
            validator_public_key: public_key_openssh.clone(),
            session_duration_secs: 300, // 5 minutes
            session_metadata: serde_json::json!({
                "validator_version": env!("CARGO_PKG_VERSION"),
                "verification_type": "hardware_attestation"
            })
            .to_string(),
        };

        let session_info = connection
            .initiate_ssh_session_v2(session_request)
            .await
            .context("Failed to initiate SSH session")?;

        // Check if session was successfully created
        if session_info.status() != SshSessionStatus::Active {
            error!(
                "SSH session creation failed for executor {}: status={:?}",
                executor.id, session_info.status
            );
            return Ok(0.0);
        }

        info!(
            "SSH session created for executor {}: session_id={}, expires_at={}",
            executor.id, session_info.session_id, session_info.expires_at
        );

        // Step 4: Parse SSH credentials and create connection details
        let ssh_details = self.parse_ssh_credentials(&session_info.access_credentials)?;
        let executor_ssh_details = ExecutorSshDetails::new(
            executor.id.clone(),
            ssh_details.host,
            ssh_details.username,
            Some(ssh_details.port),
            key_path.clone(),
            Some(self.config.challenge_timeout),
        );

        // Step 5: Perform hardware validation
        let verification_result = if let Some(ref hardware_validator) = self.hardware_validator {
            // Perform full hardware validation
            let result = self
                .perform_hardware_validation(&executor_ssh_details, hardware_validator)
                .await?;
            self.calculate_attestation_score(&result)
        } else {
            // Fallback to connection test
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
                    0.8 // Partial score for successful connection
                }
                Err(e) => {
                    error!(
                        "SSH connection test failed for executor {}: {}",
                        executor.id, e
                    );
                    0.0
                }
            }
        };

        // Step 6: Close SSH session
        let close_request = CloseSshSessionRequest {
            session_id: session_info.session_id.clone(),
            validator_hotkey: self.validator_hotkey.to_string(),
            reason: "verification_complete".to_string(),
        };

        if let Err(e) = connection.close_ssh_session(close_request).await {
            warn!(
                "Failed to close SSH session {}: {}",
                session_info.session_id, e
            );
        }

        // Step 7: Cleanup local keys
        if let Some(ref key_manager) = self.ssh_key_manager {
            if let Err(e) = key_manager.cleanup_session_keys(&session_id).await {
                warn!(
                    "Failed to cleanup SSH keys for session {}: {}",
                    session_id, e
                );
            }
        }

        Ok(verification_result)
    }

    /// Get miner endpoint from cache or error
    async fn get_miner_endpoint(&self, miner_uid: &MinerUid) -> Result<String> {
        let endpoints = self.miner_endpoints.read().await;
        endpoints.get(miner_uid).cloned().ok_or_else(|| {
            anyhow::anyhow!(
                "Miner endpoint not found in cache for miner {}",
                miner_uid.as_u16()
            )
        })
    }

    /// Create authenticated miner client
    fn create_authenticated_client(&self) -> Result<MinerClient> {
        Ok(
            if let Some(ref bittensor_service) = self.bittensor_service {
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
            },
        )
    }

    /// Perform hardware validation (placeholder for Arc compatibility)
    async fn perform_hardware_validation(
        &self,
        _executor_ssh_details: &ExecutorSshDetails,
        _hardware_validator: &Arc<HardwareValidator>,
    ) -> Result<AttestationResult> {
        // This is a placeholder since the actual hardware validator needs refactoring
        // to work with Arc<HardwareValidator> instead of mutable access
        warn!("Hardware validation not yet implemented for Arc<HardwareValidator>");

        Ok(AttestationResult {
            executor_id: _executor_ssh_details.executor_id.clone(),
            validated_at: SystemTime::now(),
            is_valid: true,
            hardware_specs: None,
            signature: None,
            error_message: None,
            validation_duration: Duration::from_secs(0),
        })
    }

    /// Calculate score from attestation result
    fn calculate_attestation_score(&self, result: &AttestationResult) -> f64 {
        if !result.is_valid {
            return 0.0;
        }

        // Basic scoring logic
        let mut score: f64 = 0.5; // Base score for valid attestation

        // Add score for hardware specs if available
        if result.hardware_specs.is_some() {
            score += 0.3;
        }

        // Add score for signature if available
        if result.signature.is_some() {
            score += 0.2;
        }

        score.min(1.0)
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
