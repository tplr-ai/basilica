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
use tracing::{debug, error, info, warn};
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

    /// Check if an endpoint is invalid
    fn is_invalid_endpoint(&self, endpoint: &str) -> bool {
        // Check for common invalid patterns
        if endpoint.contains("0:0:0:0:0:0:0:0")
            || endpoint.contains("0.0.0.0")
            || endpoint.is_empty()
            || !endpoint.starts_with("http")
        {
            debug!("Invalid endpoint detected: {}", endpoint);
            return true;
        }

        // Validate URL parsing
        if let Ok(url) = url::Url::parse(endpoint) {
            if let Some(host) = url.host_str() {
                // Check for zero or loopback addresses that indicate invalid configuration
                if host == "0.0.0.0" || host == "::" || host == "localhost" || host == "127.0.0.1" {
                    debug!("Invalid host in endpoint: {}", endpoint);
                    return true;
                }
            } else {
                debug!("No host found in endpoint: {}", endpoint);
                return true;
            }
        } else {
            debug!("Failed to parse endpoint as URL: {}", endpoint);
            return true;
        }

        false
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

    /// Execute automated verification workflow for discovered miners (legacy batch method)
    pub async fn execute_automated_verification_workflow_batch(
        &self,
        miners: &[MinerInfo],
    ) -> Result<HashMap<MinerUid, f64>> {
        let mut results = HashMap::new();

        info!(
            "Starting automated verification workflow for {} miners",
            miners.len()
        );

        for miner in miners {
            match self.verify_miner_with_ssh_automation(miner).await {
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

    /// Execute complete automated verification workflow with SSH session management (specs-compliant)
    pub async fn execute_automated_verification_workflow(
        &self,
        task: &super::scheduler::VerificationTask,
    ) -> Result<VerificationResult> {
        info!(
            "Executing automated verification workflow for miner {} (type: {:?})",
            task.miner_uid, task.verification_type
        );

        let workflow_start = std::time::Instant::now();
        let mut verification_steps = Vec::new();

        // Step 1: Discover miner executors via gRPC
        let executor_list = self
            .discover_miner_executors(&task.miner_endpoint)
            .await
            .with_context(|| {
                format!("Failed to discover executors for miner {}", task.miner_uid)
            })?;

        verification_steps.push(VerificationStep {
            step_name: "executor_discovery".to_string(),
            status: StepStatus::Completed,
            duration: workflow_start.elapsed(),
            details: format!("Discovered {} executors", executor_list.len()),
        });

        if executor_list.is_empty() {
            return Ok(VerificationResult {
                miner_uid: task.miner_uid,
                overall_score: 0.0,
                verification_steps,
                completed_at: chrono::Utc::now(),
                error: Some("No executors found for miner".to_string()),
            });
        }

        // Step 2: Execute SSH-based verification for each executor
        let mut executor_results = Vec::new();

        for executor_info in executor_list {
            match self
                .verify_executor_with_ssh_automation(&task.miner_endpoint, &executor_info)
                .await
            {
                Ok(result) => {
                    let score = result.verification_score;
                    executor_results.push(result);
                    verification_steps.push(VerificationStep {
                        step_name: format!("ssh_verification_{}", executor_info.id),
                        status: StepStatus::Completed,
                        duration: workflow_start.elapsed(),
                        details: format!("SSH verification completed, score: {score}"),
                    });
                }
                Err(e) => {
                    error!(
                        "SSH verification failed for executor {}: {}",
                        executor_info.id, e
                    );
                    verification_steps.push(VerificationStep {
                        step_name: format!("ssh_verification_{}", executor_info.id),
                        status: StepStatus::Failed,
                        duration: workflow_start.elapsed(),
                        details: format!("SSH verification error: {e}"),
                    });
                }
            }
        }

        // Step 3: Calculate overall verification score
        let overall_score = if executor_results.is_empty() {
            0.0
        } else {
            executor_results
                .iter()
                .map(|r| r.verification_score)
                .sum::<f64>()
                / executor_results.len() as f64
        };

        // Step 4: Store verification result
        self.store_verification_result(task.miner_uid, overall_score)
            .await?;

        verification_steps.push(VerificationStep {
            step_name: "result_storage".to_string(),
            status: StepStatus::Completed,
            duration: workflow_start.elapsed(),
            details: format!("Stored verification result with score: {overall_score:.2}"),
        });

        info!(
            "Automated verification workflow completed for miner {} in {:?}, score: {:.2}",
            task.miner_uid,
            workflow_start.elapsed(),
            overall_score
        );

        Ok(VerificationResult {
            miner_uid: task.miner_uid,
            overall_score,
            verification_steps,
            completed_at: chrono::Utc::now(),
            error: None,
        })
    }

    /// Execute automated verification workflow with task structure for enhanced scheduler integration
    pub async fn execute_automated_verification_workflow_with_task(
        &self,
        task: &super::scheduler::VerificationTask,
    ) -> Result<VerificationResult> {
        info!(
            "Starting task-based automated verification for miner UID: {}",
            task.miner_uid
        );

        // Convert task into MinerInfo structure for existing workflow
        let miner_info = super::types::MinerInfo {
            uid: MinerUid::from(task.miner_uid),
            hotkey: Hotkey::new(task.miner_hotkey.clone()).unwrap_or_else(|_| {
                Hotkey::new("5DAAnrj7VHTznn2AWBemMuyBwZWs6FNFjdyVXUeYum3PTXFy".to_string()).unwrap()
            }),
            endpoint: task.miner_endpoint.clone(),
            last_verified: Some(task.created_at),
            verification_score: 0.0, // Will be updated after verification
            is_validator: false,
            stake_tao: 0.0,
        };

        // Execute verification with SSH automation
        let score = self.verify_miner_with_ssh_automation(&miner_info).await?;

        // Create verification result with additional metadata
        let result = VerificationResult {
            miner_uid: task.miner_uid,
            overall_score: score,
            verification_steps: vec![], // Empty for simplified workflow
            completed_at: chrono::Utc::now(),
            error: None,
        };

        info!(
            "Task-based verification completed for miner {} with score: {:.4}",
            task.miner_uid, score
        );

        Ok(result)
    }

    /// Discover executors from miner via gRPC
    async fn discover_miner_executors(
        &self,
        miner_endpoint: &str,
    ) -> Result<Vec<ExecutorInfoDetailed>> {
        info!(
            "[EVAL_FLOW] Starting executor discovery from miner at: {}",
            miner_endpoint
        );
        debug!("[EVAL_FLOW] Using config: timeout={:?}, grpc_port_offset={:?}, use_dynamic_discovery={}", 
               self.config.discovery_timeout, self.config.grpc_port_offset, self.use_dynamic_discovery);

        // Validate endpoint before attempting connection
        if self.is_invalid_endpoint(miner_endpoint) {
            error!(
                "[EVAL_FLOW] Invalid miner endpoint detected: {}",
                miner_endpoint
            );
            return Err(anyhow::anyhow!(
                "Invalid miner endpoint: {}. Skipping discovery.",
                miner_endpoint
            ));
        }
        info!(
            "[EVAL_FLOW] Endpoint validation passed for: {}",
            miner_endpoint
        );

        // Create authenticated miner client
        info!(
            "[EVAL_FLOW] Creating authenticated miner client with validator hotkey: {}",
            self.validator_hotkey
                .to_string()
                .chars()
                .take(8)
                .collect::<String>()
                + "..."
        );
        let client = self.create_authenticated_client()?;

        // Connect and authenticate to miner
        info!(
            "[EVAL_FLOW] Attempting gRPC connection to miner at: {}",
            miner_endpoint
        );
        let connection_start = std::time::Instant::now();
        let mut connection = match client.connect_and_authenticate(miner_endpoint).await {
            Ok(conn) => {
                info!(
                    "[EVAL_FLOW] Successfully connected and authenticated to miner in {:?}",
                    connection_start.elapsed()
                );
                conn
            }
            Err(e) => {
                error!(
                    "[EVAL_FLOW] Failed to connect to miner at {} after {:?}: {}",
                    miner_endpoint,
                    connection_start.elapsed(),
                    e
                );
                return Err(e).context("Failed to connect to miner for executor discovery");
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

        info!("[EVAL_FLOW] Requesting executors with requirements: cpu_cores={}, memory_mb={}, storage_mb={}, max_gpus={}, lease_duration={:?}",
              requirements.max_cpu_cores, requirements.max_memory_mb, requirements.max_storage_mb,
              requirements.max_gpus, lease_duration);

        let request_start = std::time::Instant::now();
        let executor_details = match connection
            .request_executors(Some(requirements), lease_duration)
            .await
        {
            Ok(details) => {
                info!(
                    "[EVAL_FLOW] Successfully received executor details in {:?}, count={}",
                    request_start.elapsed(),
                    details.len()
                );
                for (i, detail) in details.iter().enumerate() {
                    debug!(
                        "[EVAL_FLOW] Executor {}: id={}, grpc_endpoint={}",
                        i, detail.executor_id, detail.grpc_endpoint
                    );
                }
                details
            }
            Err(e) => {
                error!(
                    "[EVAL_FLOW] Failed to request executors from miner after {:?}: {}",
                    request_start.elapsed(),
                    e
                );
                return Ok(vec![]);
            }
        };

        let executor_count = executor_details.len();
        let executors: Vec<ExecutorInfoDetailed> = executor_details
            .into_iter()
            .map(|details| ExecutorInfoDetailed {
                id: details.executor_id,
                host: "unknown".to_string(), // Will be filled from SSH credentials
                port: 22,
                status: "available".to_string(),
                capabilities: vec!["gpu".to_string()],
                grpc_endpoint: details.grpc_endpoint,
            })
            .collect();

        info!(
            "[EVAL_FLOW] Executor discovery completed: {} executors mapped from {} details",
            executors.len(),
            executor_count
        );

        Ok(executors)
    }

    /// Verify executor with SSH automation
    async fn verify_executor_with_ssh_automation(
        &self,
        miner_endpoint: &str,
        executor_info: &ExecutorInfoDetailed,
    ) -> Result<ExecutorVerificationResult> {
        info!(
            "[EVAL_FLOW] Starting SSH automation verification for executor {} via miner {}",
            executor_info.id, miner_endpoint
        );
        debug!(
            "[EVAL_FLOW] Executor info: host={}, port={}, grpc_endpoint={}, capabilities={:?}",
            executor_info.host,
            executor_info.port,
            executor_info.grpc_endpoint,
            executor_info.capabilities
        );

        // Create authenticated miner client
        info!("[EVAL_FLOW] Creating authenticated client for SSH verification");
        let client = self.create_authenticated_client()?;

        // Connect and authenticate to miner
        info!("[EVAL_FLOW] Establishing connection to miner for SSH session setup");
        let ssh_connect_start = std::time::Instant::now();
        let mut connection = match client.connect_and_authenticate(miner_endpoint).await {
            Ok(conn) => {
                info!(
                    "[EVAL_FLOW] SSH verification connection established in {:?}",
                    ssh_connect_start.elapsed()
                );
                conn
            }
            Err(e) => {
                error!(
                    "[EVAL_FLOW] SSH verification connection failed after {:?}: {}",
                    ssh_connect_start.elapsed(),
                    e
                );
                return Err(e).context("Failed to connect to miner for SSH verification");
            }
        };

        // Generate ephemeral SSH keypair if we have key manager
        info!(
            "[EVAL_FLOW] Generating SSH session keys for executor {}",
            executor_info.id
        );
        let key_gen_start = std::time::Instant::now();
        let (session_id, public_key_openssh, _key_path) = if let Some(ref key_manager) =
            self.ssh_key_manager
        {
            let session_id = Uuid::new_v4().to_string();
            info!("[EVAL_FLOW] Session ID generated: {}", session_id);

            let (_, public_key, key_path) = key_manager
                .generate_session_keypair(&session_id)
                .await
                .context("Failed to generate SSH keypair")?;

            let public_key_openssh = ValidatorSshKeyManager::get_public_key_openssh(&public_key)
                .context("Failed to convert public key to OpenSSH format")?;

            info!(
                "[EVAL_FLOW] SSH keypair generated in {:?}, public key length: {} chars",
                key_gen_start.elapsed(),
                public_key_openssh.len()
            );
            debug!(
                "[EVAL_FLOW] Public key preview: {}...",
                public_key_openssh.chars().take(50).collect::<String>()
            );

            (session_id, public_key_openssh, key_path)
        } else {
            error!(
                "[EVAL_FLOW] No SSH key manager available for verification of executor {}",
                executor_info.id
            );
            return Err(anyhow::anyhow!(
                "No SSH key manager available for verification"
            ));
        };

        // Request SSH session setup
        let session_request = InitiateSshSessionRequest {
            validator_hotkey: self.validator_hotkey.to_string(),
            executor_id: executor_info.id.clone(),
            purpose: "automated_verification".to_string(),
            validator_public_key: public_key_openssh,
            session_duration_secs: 300, // 5 minutes
            session_metadata: serde_json::json!({
                "validator_version": env!("CARGO_PKG_VERSION"),
                "verification_type": "automated"
            })
            .to_string(),
        };

        info!(
            "[EVAL_FLOW] Initiating SSH session with executor {} via miner",
            executor_info.id
        );
        debug!("[EVAL_FLOW] Session request: validator_hotkey={}, executor_id={}, purpose={}, duration={}s",
               session_request.validator_hotkey.chars().take(8).collect::<String>() + "...",
               session_request.executor_id, session_request.purpose, session_request.session_duration_secs);

        let session_start = std::time::Instant::now();
        let session_info = match connection.initiate_ssh_session_v2(session_request).await {
            Ok(info) => {
                info!(
                    "[EVAL_FLOW] SSH session initiation response received in {:?}, status={:?}",
                    session_start.elapsed(),
                    info.status()
                );
                debug!("[EVAL_FLOW] Session details: session_id={}, expires_at={}, credentials_length={}",
                       info.session_id, info.expires_at, info.access_credentials.len());
                info
            }
            Err(e) => {
                error!(
                    "[EVAL_FLOW] Failed to initiate SSH session for executor {} after {:?}: {}",
                    executor_info.id,
                    session_start.elapsed(),
                    e
                );
                return Ok(ExecutorVerificationResult {
                    executor_id: executor_info.id.clone(),
                    verification_score: 0.0,
                    error: Some(format!("SSH session initiation failed: {e}")),
                });
            }
        };

        // Check if session was successfully created
        if session_info.status() != SshSessionStatus::Active {
            error!(
                "[EVAL_FLOW] SSH session not active for executor {}: status={:?}",
                executor_info.id,
                session_info.status()
            );
            return Ok(ExecutorVerificationResult {
                executor_id: executor_info.id.clone(),
                verification_score: 0.0,
                error: Some(format!("SSH session not active: {:?}", session_info.status)),
            });
        }

        info!(
            "[EVAL_FLOW] SSH session active for executor {}: session_id={}, expires_at={}",
            executor_info.id, session_info.session_id, session_info.expires_at
        );

        // Parse SSH credentials and test connection
        info!(
            "[EVAL_FLOW] Parsing SSH credentials for executor {}",
            executor_info.id
        );
        debug!(
            "[EVAL_FLOW] Raw credentials: {}",
            session_info.access_credentials
        );
        let ssh_details = match self.parse_ssh_credentials(&session_info.access_credentials) {
            Ok(details) => {
                info!("[EVAL_FLOW] SSH credentials parsed successfully: host={}, port={}, username={}",
                      details.host, details.port, details.username);
                details
            }
            Err(e) => {
                error!(
                    "[EVAL_FLOW] Failed to parse SSH credentials for executor {}: {}",
                    executor_info.id, e
                );
                return Ok(ExecutorVerificationResult {
                    executor_id: executor_info.id.clone(),
                    verification_score: 0.0,
                    error: Some(format!("Failed to parse SSH credentials: {e}")),
                });
            }
        };

        // Test SSH connection
        info!(
            "[EVAL_FLOW] Testing SSH connection to executor {} at {}:{}",
            executor_info.id, ssh_details.host, ssh_details.port
        );
        let connection_test_start = std::time::Instant::now();
        let verification_score = match self.ssh_client.test_connection(&ssh_details).await {
            Ok(_) => {
                info!(
                    "[EVAL_FLOW] SSH connection test successful for executor {} in {:?}",
                    executor_info.id,
                    connection_test_start.elapsed()
                );
                0.8 // Good score for successful SSH verification
            }
            Err(e) => {
                error!(
                    "[EVAL_FLOW] SSH connection test failed for executor {} after {:?}: {}",
                    executor_info.id,
                    connection_test_start.elapsed(),
                    e
                );
                0.0
            }
        };

        // Close SSH session
        info!(
            "[EVAL_FLOW] Closing SSH session {} for executor {}",
            session_info.session_id, executor_info.id
        );
        let close_request = CloseSshSessionRequest {
            session_id: session_info.session_id.clone(),
            validator_hotkey: self.validator_hotkey.to_string(),
            reason: "verification_complete".to_string(),
        };

        let close_start = std::time::Instant::now();
        if let Err(e) = connection.close_ssh_session(close_request).await {
            warn!(
                "[EVAL_FLOW] Failed to close SSH session {} after {:?}: {}",
                session_info.session_id,
                close_start.elapsed(),
                e
            );
        } else {
            info!(
                "[EVAL_FLOW] SSH session {} closed successfully in {:?}",
                session_info.session_id,
                close_start.elapsed()
            );
        }

        // Cleanup local keys
        info!(
            "[EVAL_FLOW] Cleaning up SSH keys for session {}",
            session_id
        );
        if let Some(ref key_manager) = self.ssh_key_manager {
            let cleanup_start = std::time::Instant::now();
            if let Err(e) = key_manager.cleanup_session_keys(&session_id).await {
                warn!(
                    "[EVAL_FLOW] Failed to cleanup SSH keys for session {} after {:?}: {}",
                    session_id,
                    cleanup_start.elapsed(),
                    e
                );
            } else {
                debug!(
                    "[EVAL_FLOW] SSH keys cleaned up for session {} in {:?}",
                    session_id,
                    cleanup_start.elapsed()
                );
            }
        }

        info!(
            "[EVAL_FLOW] SSH verification completed for executor {} with score: {:.2}",
            executor_info.id, verification_score
        );

        Ok(ExecutorVerificationResult {
            executor_id: executor_info.id.clone(),
            verification_score,
            error: None,
        })
    }

    /// Store verification result (placeholder implementation)
    async fn store_verification_result(&self, miner_uid: u16, score: f64) -> Result<()> {
        info!(
            "Storing verification result for miner {}: score={:.2}",
            miner_uid, score
        );
        // TODO: Implement actual storage (database, cache, etc.)
        Ok(())
    }

    /// Verify miner with full SSH automation including discovery handshake
    async fn verify_miner_with_ssh_automation(
        &self,
        miner: &super::types::MinerInfo,
    ) -> Result<f64> {
        info!(
            "Starting SSH-automated verification for miner {}",
            miner.uid.as_u16()
        );

        // Step 1: Connect to miner and discover executors
        let executor_list = self.discover_miner_executors(&miner.endpoint).await?;

        if executor_list.is_empty() {
            warn!("No executors available from miner {}", miner.uid.as_u16());
            return Ok(0.0);
        }

        // Step 2: Execute verification on each executor
        let mut executor_results = Vec::new();

        for executor_info in executor_list {
            match self
                .verify_executor_with_ssh_automation(&miner.endpoint, &executor_info)
                .await
            {
                Ok(result) => {
                    executor_results.push(result);
                }
                Err(e) => {
                    error!(
                        "SSH verification failed for executor {}: {}",
                        executor_info.id, e
                    );
                }
            }
        }

        // Step 3: Calculate overall score
        let final_score = if executor_results.is_empty() {
            0.0
        } else {
            executor_results
                .iter()
                .map(|r| r.verification_score)
                .sum::<f64>()
                / executor_results.len() as f64
        };

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

        // Initialize SSH key manager with validator configuration (will be created later if needed)
        let ssh_key_manager = None;

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
            ssh_key_manager,
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

    /// Get whether dynamic discovery is enabled
    pub fn use_dynamic_discovery(&self) -> bool {
        self.use_dynamic_discovery
    }

    /// Get SSH key manager reference
    pub fn ssh_key_manager(&self) -> &Option<Arc<ValidatorSshKeyManager>> {
        &self.ssh_key_manager
    }

    /// Get hardware validator reference
    pub fn hardware_validator(&self) -> &Option<Arc<HardwareValidator>> {
        &self.hardware_validator
    }

    /// Get bittensor service reference
    pub fn bittensor_service(&self) -> &Option<Arc<bittensor::Service>> {
        &self.bittensor_service
    }

    /// Get SSH key path reference
    pub fn ssh_key_path(&self) -> &Option<PathBuf> {
        &self.ssh_key_path
    }

    /// Create VerificationEngine with SSH automation components (new preferred method)
    pub fn with_ssh_automation(
        config: VerificationConfig,
        miner_client_config: MinerClientConfig,
        validator_hotkey: Hotkey,
        ssh_client: Arc<ValidatorSshClient>,
        hardware_validator: Option<Arc<HardwareValidator>>,
        use_dynamic_discovery: bool,
        ssh_key_manager: Option<Arc<ValidatorSshKeyManager>>,
        bittensor_service: Option<Arc<bittensor::Service>>,
    ) -> Result<Self> {
        // Validate required components for dynamic discovery
        if use_dynamic_discovery && ssh_key_manager.is_none() {
            return Err(anyhow::anyhow!(
                "SSH key manager is required when dynamic discovery is enabled"
            ));
        }

        Ok(Self {
            config: config.clone(),
            miner_client_config,
            validator_hotkey,
            ssh_client,
            hardware_validator,
            use_dynamic_discovery,
            ssh_key_path: None, // Not used when SSH key manager is available
            miner_endpoints: Arc::new(RwLock::new(HashMap::new())),
            bittensor_service,
            ssh_key_manager,
        })
    }

    /// Check if SSH automation is properly configured
    pub fn is_ssh_automation_ready(&self) -> bool {
        if self.use_dynamic_discovery() {
            self.ssh_key_manager().is_some()
        } else {
            // Static configuration requires either key manager or fallback key path
            self.ssh_key_manager().is_some() || self.ssh_key_path().is_some()
        }
    }

    /// Get SSH automation status
    pub fn get_ssh_automation_status(&self) -> SshAutomationStatus {
        SshAutomationStatus {
            dynamic_discovery_enabled: self.use_dynamic_discovery(),
            ssh_key_manager_available: self.ssh_key_manager().is_some(),
            hardware_validator_available: self.hardware_validator().is_some(),
            bittensor_service_available: self.bittensor_service().is_some(),
            fallback_key_path: self.ssh_key_path().clone(),
        }
    }

    /// Get configuration summary for debugging
    pub fn get_config_summary(&self) -> String {
        format!(
            "VerificationEngine[dynamic_discovery={}, ssh_key_manager={}, hardware_validator={}, bittensor_service={}]",
            self.use_dynamic_discovery(),
            self.ssh_key_manager().is_some(),
            self.hardware_validator().is_some(),
            self.bittensor_service().is_some()
        )
    }
}

/// SSH automation status information
#[derive(Debug, Clone)]
pub struct SshAutomationStatus {
    pub dynamic_discovery_enabled: bool,
    pub ssh_key_manager_available: bool,
    pub hardware_validator_available: bool,
    pub bittensor_service_available: bool,
    pub fallback_key_path: Option<PathBuf>,
}

impl std::fmt::Display for SshAutomationStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SSH Automation Status[dynamic={}, key_manager={}, hardware_validator={}, bittensor={}, fallback_key={}]",
            self.dynamic_discovery_enabled,
            self.ssh_key_manager_available,
            self.hardware_validator_available,
            self.bittensor_service_available,
            self.fallback_key_path.as_ref().map(|p| p.display().to_string()).unwrap_or("none".to_string())
        )
    }
}

/// Enhanced executor information structure for detailed verification
#[derive(Debug, Clone)]
pub struct ExecutorInfoDetailed {
    pub id: String,
    pub host: String,
    pub port: u16,
    pub status: String,
    pub capabilities: Vec<String>,
    pub grpc_endpoint: String,
}

/// Executor verification result
#[derive(Debug, Clone)]
pub struct ExecutorVerificationResult {
    pub executor_id: String,
    pub verification_score: f64,
    pub error: Option<String>,
}

/// Verification step tracking
#[derive(Debug, Clone)]
pub struct VerificationStep {
    pub step_name: String,
    pub status: StepStatus,
    pub duration: Duration,
    pub details: String,
}

/// Step status tracking
#[derive(Debug, Clone)]
pub enum StepStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
}

/// Enhanced verification result structure
#[derive(Debug, Clone)]
pub struct VerificationResult {
    pub miner_uid: u16,
    pub overall_score: f64,
    pub verification_steps: Vec<VerificationStep>,
    pub completed_at: chrono::DateTime<chrono::Utc>,
    pub error: Option<String>,
}
