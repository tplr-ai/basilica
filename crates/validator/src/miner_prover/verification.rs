//! # Verification Engine
//!
//! Handles the actual verification of miners and their executors.
//! Implements Single Responsibility Principle by focusing only on verification logic.

use super::miner_client::{MinerClient, MinerClientConfig};
use super::types::{ExecutorInfo, ExecutorStatus, MinerInfo};
use crate::config::VerificationConfig;
use crate::ssh::{ExecutorSshDetails, ValidatorSshClient, ValidatorSshKeyManager};
use anyhow::{Context, Result};
use common::identity::{ExecutorId, Hotkey, MinerUid};
use common::ssh::SshConnectionDetails;
use protocol::miner_discovery::{
    CloseSshSessionRequest, InitiateSshSessionRequest, SshSessionStatus,
};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

#[derive(Clone)]
pub struct VerificationEngine {
    config: VerificationConfig,
    miner_client_config: MinerClientConfig,
    validator_hotkey: Hotkey,
    ssh_client: Arc<ValidatorSshClient>,
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
    /// Active SSH sessions per executor to prevent concurrent sessions
    active_ssh_sessions: Arc<Mutex<HashSet<String>>>,
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
            use_dynamic_discovery: false, // Disabled without proper initialization
            ssh_key_path: None,
            miner_endpoints: Arc::new(RwLock::new(HashMap::new())),
            bittensor_service: None,
            ssh_key_manager: None,
            active_ssh_sessions: Arc::new(Mutex::new(HashSet::new())),
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
        // Use persistent SSH key instead of generating new session keys
        let session_id = Uuid::new_v4().to_string();
        let (public_key_openssh, _key_path) = match ssh_key_manager.get_persistent_key() {
            Some((public_key, private_key_path)) => {
                info!("Using persistent SSH key for executor {}", executor.id);
                (public_key.clone(), private_key_path.clone())
            }
            None => {
                error!(
                    "No persistent SSH key available for executor {}",
                    executor.id
                );
                return Err(anyhow::anyhow!("No persistent SSH key available"));
            }
        };

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
                .verify_executor_with_ssh_automation_enhanced(&task.miner_endpoint, &executor_info)
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

    /// Helper function to clean up active SSH session for an executor
    async fn cleanup_active_session(&self, executor_id: &str) {
        let mut active_sessions = self.active_ssh_sessions.lock().await;
        let before_count = active_sessions.len();
        let removed = active_sessions.remove(executor_id);
        let after_count = active_sessions.len();
        
        if removed {
            info!(
                "[EVAL_FLOW] SSH session cleanup successful for executor {} - Active sessions: {} -> {} (removed: {})",
                executor_id, before_count, after_count, executor_id
            );
        } else {
            warn!(
                "[EVAL_FLOW] SSH session cleanup attempted for executor {} but no active session found - Active sessions: {} (current: {:?})",
                executor_id, before_count, active_sessions.iter().collect::<Vec<_>>()
            );
        }
        
        // Log remaining active sessions for transparency
        if !active_sessions.is_empty() {
            debug!(
                "[EVAL_FLOW] Remaining active SSH sessions after cleanup: {:?}",
                active_sessions.iter().collect::<Vec<_>>()
            );
        }
    }

    /// Verify executor with SSH automation (enhanced with binary validation)
    async fn verify_executor_with_ssh_automation(
        &self,
        miner_endpoint: &str,
        executor_info: &ExecutorInfoDetailed,
    ) -> Result<ExecutorVerificationResult> {
        // Direct call to enhanced method
        self.verify_executor_with_ssh_automation_enhanced(miner_endpoint, executor_info)
            .await
    }

    async fn store_verification_result(&self, miner_uid: u16, score: f64) -> Result<()> {
        info!(
            "Storing verification result for miner {}: score={:.2}",
            miner_uid, score
        );

        // Store verification result with timestamp and miner info
        let verification_entry = serde_json::json!({
            "miner_uid": miner_uid,
            "score": score,
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "verification_method": "dynamic_discovery"
        });

        // Write to verification results file for persistence
        let results_file = "/tmp/basilica_verification_results.json";
        let mut results = if let Ok(content) = tokio::fs::read_to_string(results_file).await {
            serde_json::from_str::<Vec<serde_json::Value>>(&content).unwrap_or_default()
        } else {
            Vec::new()
        };

        results.push(verification_entry);

        let results_json = serde_json::to_string_pretty(&results)?;
        tokio::fs::write(results_file, results_json).await?;

        info!(
            "Verification result stored for miner {}: score={:.2}",
            miner_uid, score
        );

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
                .verify_executor_with_ssh_automation_enhanced(&miner.endpoint, &executor_info)
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

        loop {
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
    }

    /// Create with full configuration for dynamic discovery
    pub fn with_validator_context(
        config: VerificationConfig,
        validator_hotkey: Hotkey,
        ssh_client: Arc<ValidatorSshClient>,
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
            use_dynamic_discovery: config.use_dynamic_discovery,
            ssh_key_path,
            miner_endpoints: Arc::new(RwLock::new(HashMap::new())),
            bittensor_service: None,
            ssh_key_manager: None,
            active_ssh_sessions: Arc::new(Mutex::new(HashSet::new())),
        }
    }

    /// Create with full configuration including SSH key manager
    pub async fn with_ssh_key_manager(
        config: VerificationConfig,
        validator_hotkey: Hotkey,
        ssh_client: Arc<ValidatorSshClient>,
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
            use_dynamic_discovery: config.use_dynamic_discovery,
            ssh_key_path: None,
            miner_endpoints: Arc::new(RwLock::new(HashMap::new())),
            bittensor_service: None,
            ssh_key_manager: Some(ssh_key_manager),
            active_ssh_sessions: Arc::new(Mutex::new(HashSet::new())),
        })
    }

    /// Create with full configuration including Bittensor service for signing
    pub fn with_bittensor_service(
        config: VerificationConfig,
        bittensor_service: Arc<bittensor::Service>,
        ssh_client: Arc<ValidatorSshClient>,
        ssh_key_path: Option<PathBuf>,
    ) -> anyhow::Result<Self> {
        let validator_hotkey = bittensor::account_id_to_hotkey(bittensor_service.get_account_id())
            .map_err(|e| anyhow::anyhow!("Failed to convert account ID to hotkey: {}", e))?;

        let miner_client_config = MinerClientConfig {
            timeout: config.discovery_timeout,
            max_retries: 3,
            grpc_port_offset: config.grpc_port_offset,
            use_tls: false,
        };

        // Initialize SSH key manager with validator configuration (will be created later if needed)
        let ssh_key_manager = None;

        Ok(Self {
            config: config.clone(),
            miner_client_config,
            validator_hotkey,
            ssh_client,
            use_dynamic_discovery: config.use_dynamic_discovery,
            ssh_key_path,
            miner_endpoints: Arc::new(RwLock::new(HashMap::new())),
            bittensor_service: Some(bittensor_service),
            ssh_key_manager,
            active_ssh_sessions: Arc::new(Mutex::new(HashSet::new())),
        })
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

        // Step 1: Use persistent SSH key if we have key manager
        let (_session_id, public_key_openssh, key_path) =
            if let Some(ref key_manager) = self.ssh_key_manager {
                let session_id = Uuid::new_v4().to_string();
                let (public_key_openssh, key_path) = match key_manager.get_persistent_key() {
                    Some((public_key, private_key_path)) => {
                        info!(
                            "Using persistent SSH key for executor {} dynamic verification",
                            executor.id
                        );
                        (public_key.clone(), private_key_path.clone())
                    }
                    None => {
                        error!(
                            "No persistent SSH key available for executor {} dynamic verification",
                            executor.id
                        );
                        return Err(anyhow::anyhow!("No persistent SSH key available"));
                    }
                };

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
        let ssh_details =
            self.parse_ssh_credentials(&session_info.access_credentials, Some(key_path.clone()))?;
        let executor_ssh_details = ExecutorSshDetails::new(
            executor.id.clone(),
            ssh_details.host,
            ssh_details.username,
            Some(ssh_details.port),
            key_path.clone(),
            Some(self.config.challenge_timeout),
        );

        // Step 5: Perform SSH connection test
        let verification_result = match self
            .ssh_client
            .test_connection(&executor_ssh_details.connection)
            .await
        {
            Ok(_) => {
                info!(
                    "SSH connection test successful for executor {}",
                    executor.id
                );
                0.8 // Score for successful connection
            }
            Err(e) => {
                error!(
                    "SSH connection test failed for executor {}: {}",
                    executor.id, e
                );
                0.0
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

    /// Parse SSH credentials string into connection details
    pub fn parse_ssh_credentials(
        &self,
        credentials: &str,
        key_path: Option<PathBuf>,
    ) -> Result<SshConnectionDetails> {
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
            private_key_path: key_path
                .or_else(|| self.ssh_key_path.clone())
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
            use_dynamic_discovery,
            ssh_key_path: None, // Not used when SSH key manager is available
            miner_endpoints: Arc::new(RwLock::new(HashMap::new())),
            bittensor_service,
            ssh_key_manager,
            active_ssh_sessions: Arc::new(Mutex::new(HashSet::new())),
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
            bittensor_service_available: self.bittensor_service().is_some(),
            fallback_key_path: self.ssh_key_path().clone(),
        }
    }

    /// Get configuration summary for debugging
    pub fn get_config_summary(&self) -> String {
        format!(
            "VerificationEngine[dynamic_discovery={}, ssh_key_manager={}, bittensor_service={}]",
            self.use_dynamic_discovery(),
            self.ssh_key_manager().is_some(),
            self.bittensor_service().is_some()
        )
    }

    // ====================================================================
    // Binary Validation Methods
    // ====================================================================

    /// Execute binary validation using validator-binary
    async fn execute_binary_validation(
        &self,
        ssh_details: &SshConnectionDetails,
        _session_info: &protocol::miner_discovery::InitiateSshSessionResponse,
    ) -> Result<crate::validation::types::ValidatorBinaryOutput> {
        info!("[EVAL_FLOW] Starting binary validation process");

        let binary_config = &self.config.binary_validation;

        // Execute validator-binary locally (it will handle executor binary upload)
        let execution_start = std::time::Instant::now();
        let binary_output = self
            .execute_validator_binary_locally(ssh_details, binary_config)
            .await?;
        let execution_duration = execution_start.elapsed();

        info!(
            "[EVAL_FLOW] Validator binary executed in {:?}",
            execution_duration
        );

        // Parse and validate output
        let validation_result = self.parse_validator_binary_output(&binary_output)?;

        // Calculate validation score
        let validation_score = self.calculate_binary_validation_score(&validation_result)?;

        Ok(crate::validation::types::ValidatorBinaryOutput {
            success: validation_result.success,
            executor_result: validation_result.executor_result,
            error_message: validation_result.error_message,
            execution_time_ms: execution_duration.as_millis() as u64,
            validation_score,
        })
    }

    /// Execute validator-binary locally with SSH parameters
    async fn execute_validator_binary_locally(
        &self,
        ssh_details: &SshConnectionDetails,
        binary_config: &crate::config::BinaryValidationConfig,
    ) -> Result<Vec<u8>> {
        info!("[EVAL_FLOW] Executing validator binary locally");

        let mut command = tokio::process::Command::new(&binary_config.validator_binary_path);

        // Configure SSH parameters and executor binary path
        command
            .arg("--ssh-host")
            .arg(&ssh_details.host)
            .arg("--ssh-port")
            .arg(ssh_details.port.to_string())
            .arg("--ssh-user")
            .arg(&ssh_details.username)
            .arg("--ssh-key")
            .arg(&ssh_details.private_key_path)
            .arg("--executor-path")
            .arg(&binary_config.executor_binary_path)
            .arg("--output-format")
            .arg(&binary_config.output_format)
            .arg("--timeout")
            .arg(binary_config.execution_timeout_secs.to_string());

        // Set timeout for entire process
        let timeout_duration = Duration::from_secs(binary_config.execution_timeout_secs + 10);

        // Debug: log the complete command being executed
        debug!("[EVAL_FLOW] Executing command: {:?}", command);
        info!("[EVAL_FLOW] Command args: validator_binary_path={:?}, ssh_host={}, ssh_port={}, ssh_user={}, ssh_key={:?}, executor_binary_path={:?}, output_format={}, timeout={}",
              binary_config.validator_binary_path, ssh_details.host, ssh_details.port, ssh_details.username,
              ssh_details.private_key_path, binary_config.executor_binary_path, binary_config.output_format, binary_config.execution_timeout_secs);

        info!(
            "[EVAL_FLOW] Starting validator binary execution with timeout {}s",
            timeout_duration.as_secs()
        );
        let start_time = std::time::Instant::now();

        let output = tokio::time::timeout(timeout_duration, command.output())
            .await
            .map_err(|_| {
                error!(
                    "[EVAL_FLOW] Validator binary execution timed out after {}s",
                    timeout_duration.as_secs()
                );
                anyhow::anyhow!(
                    "Validator binary execution timeout after {}s",
                    timeout_duration.as_secs()
                )
            })?
            .map_err(|e| {
                error!(
                    "[EVAL_FLOW] Failed to execute validator binary process: {}",
                    e
                );
                anyhow::anyhow!("Failed to execute validator binary: {}", e)
            })?;

        let execution_time = start_time.elapsed();
        info!(
            "[EVAL_FLOW] Validator binary execution completed in {:.2}s",
            execution_time.as_secs_f64()
        );

        // Log stdout and stderr regardless of status
        let stdout_str = String::from_utf8_lossy(&output.stdout);
        let stderr_str = String::from_utf8_lossy(&output.stderr);

        if !stdout_str.is_empty() {
            info!("[EVAL_FLOW] Validator binary stdout: {}", stdout_str);
        }

        if !stderr_str.is_empty() {
            if output.status.success() {
                warn!(
                    "[EVAL_FLOW] Validator binary stderr (non-fatal): {}",
                    stderr_str
                );
            } else {
                error!("[EVAL_FLOW] Validator binary stderr: {}", stderr_str);
            }
        }

        if !output.status.success() {
            let exit_code = output.status.code().unwrap_or(-1);
            error!(
                "[EVAL_FLOW] Validator binary execution failed with exit code: {}",
                exit_code
            );
            return Err(anyhow::anyhow!(
                "Validator binary execution failed with exit code {}: {}",
                exit_code,
                stderr_str
            ));
        }

        info!(
            "[EVAL_FLOW] Validator binary execution successful, processing output ({} bytes)",
            output.stdout.len()
        );
        Ok(output.stdout)
    }

    /// Parse validator binary output
    fn parse_validator_binary_output(
        &self,
        output: &[u8],
    ) -> Result<crate::validation::types::ValidatorBinaryOutput> {
        let output_str = String::from_utf8_lossy(output);

        info!(
            "[EVAL_FLOW] Parsing validator binary output ({} bytes)",
            output.len()
        );
        debug!("[EVAL_FLOW] Raw output: {}", output_str);

        // Parse JSON output
        let parsed_output: crate::validation::types::ValidatorBinaryOutput =
            serde_json::from_str(&output_str).map_err(|e| {
                error!(
                    "[EVAL_FLOW] Failed to parse validator binary JSON output: {}",
                    e
                );
                error!(
                    "[EVAL_FLOW] Raw output that failed to parse: {}",
                    output_str
                );
                anyhow::anyhow!("Failed to parse validator binary JSON output: {}", e)
            })?;

        info!("[EVAL_FLOW] Successfully parsed binary output - success: {}, execution_time: {}ms, validation_score: {:.3}", 
              parsed_output.success, parsed_output.execution_time_ms, parsed_output.validation_score);

        if let Some(ref executor_result) = parsed_output.executor_result {
            info!("[EVAL_FLOW] Executor hardware details - CPU cores: {}, Memory: {:.1}GB, Network interfaces: {}", 
                  executor_result.cpu_info.cores, executor_result.memory_info.total_gb, 
                  executor_result.network_info.interfaces.len());

            if !executor_result.gpu_name.is_empty() {
                info!(
                    "[EVAL_FLOW] GPU Details: {} (UUID: {}), SMs: {}/{}, Memory bandwidth: {:.1} GB/s",
                    executor_result.gpu_name, executor_result.gpu_uuid, 
                    executor_result.active_sms, executor_result.total_sms, 
                    executor_result.memory_bandwidth_gbps
                );
            } else {
                warn!("[EVAL_FLOW] No GPU information found in executor result");
            }

            info!("[EVAL_FLOW] Binary validation metrics - Matrix computation: {:.2}ms, SM utilization: max={:.1}%, avg={:.1}%", 
                  executor_result.computation_time_ns as f64 / 1_000_000.0,
                  executor_result.sm_utilization.max_utilization, 
                  executor_result.sm_utilization.avg_utilization);
        } else {
            warn!("[EVAL_FLOW] No executor result found in binary output");
        }

        if let Some(ref error_msg) = parsed_output.error_message {
            error!("[EVAL_FLOW] Binary validation error message: {}", error_msg);
        }

        // Validate structure
        if parsed_output.success && parsed_output.executor_result.is_none() {
            error!("[EVAL_FLOW] Validator binary reported success but no executor result provided");
            return Err(anyhow::anyhow!(
                "Validator binary reported success but no executor result provided"
            ));
        }

        Ok(parsed_output)
    }

    /// Calculate binary validation score based on executor result
    fn calculate_binary_validation_score(
        &self,
        validation_result: &crate::validation::types::ValidatorBinaryOutput,
    ) -> Result<f64> {
        info!("[EVAL_FLOW] Starting binary validation score calculation");

        if !validation_result.success {
            error!("[EVAL_FLOW] Binary validation failed, returning score: 0.0");
            return Ok(0.0);
        }

        let executor_result = validation_result.executor_result.as_ref().ok_or_else(|| {
            error!("[EVAL_FLOW] No executor result available for scoring");
            anyhow::anyhow!("No executor result available for scoring")
        })?;

        let mut score: f64 = 0.0;
        let mut score_breakdown = Vec::new();

        // Base score for successful execution
        score += 0.3;
        score_breakdown.push(("base_execution", 0.3));
        info!(
            "[EVAL_FLOW] Score component - Base execution: +0.3 (total: {:.3})",
            score
        );

        // Anti-debug check score
        if executor_result.anti_debug_passed {
            score += 0.2;
            score_breakdown.push(("anti_debug", 0.2));
            info!(
                "[EVAL_FLOW] Score component - Anti-debug passed: +0.2 (total: {:.3})",
                score
            );
        } else {
            warn!(
                "[EVAL_FLOW] Score component - Anti-debug failed: +0.0 (total: {:.3})",
                score
            );
        }

        // SM utilization score (higher utilization = better score)
        let avg_utilization = executor_result.sm_utilization.avg_utilization;
        let sm_score = if avg_utilization > 0.8 {
            0.2
        } else if avg_utilization > 0.6 {
            0.1
        } else {
            0.0
        };
        score += sm_score;
        score_breakdown.push(("sm_utilization", sm_score));
        info!(
            "[EVAL_FLOW] Score component - SM utilization ({:.1}%): +{:.3} (total: {:.3})",
            avg_utilization * 100.0,
            sm_score,
            score
        );

        // GPU resource score
        let gpu_efficiency = executor_result.active_sms as f64 / executor_result.total_sms as f64;
        let gpu_score = if gpu_efficiency > 0.9 {
            0.15
        } else if gpu_efficiency > 0.7 {
            0.1
        } else {
            0.0
        };
        score += gpu_score;
        score_breakdown.push(("gpu_efficiency", gpu_score));
        info!(
            "[EVAL_FLOW] Score component - GPU efficiency ({:.1}%, {}/{}): +{:.3} (total: {:.3})",
            gpu_efficiency * 100.0,
            executor_result.active_sms,
            executor_result.total_sms,
            gpu_score,
            score
        );

        // Memory bandwidth score
        let bandwidth_score = if executor_result.memory_bandwidth_gbps > 500.0 {
            0.1
        } else if executor_result.memory_bandwidth_gbps > 200.0 {
            0.05
        } else {
            0.0
        };
        score += bandwidth_score;
        score_breakdown.push(("memory_bandwidth", bandwidth_score));
        info!(
            "[EVAL_FLOW] Score component - Memory bandwidth ({:.1} GB/s): +{:.3} (total: {:.3})",
            executor_result.memory_bandwidth_gbps, bandwidth_score, score
        );

        // Computation time score (reasonable timing)
        let computation_time_ms = executor_result.computation_time_ns / 1_000_000;
        let timing_score = if computation_time_ms > 10 && computation_time_ms < 5000 {
            0.05
        } else {
            0.0
        };
        score += timing_score;
        score_breakdown.push(("computation_timing", timing_score));
        info!(
            "[EVAL_FLOW] Score component - Computation timing ({}ms): +{:.3} (total: {:.3})",
            computation_time_ms, timing_score, score
        );

        // Final score clamping and summary
        let final_score = score.clamp(0.0, 1.0);
        info!(
            "[EVAL_FLOW] Binary validation score calculation complete: {:.3}/1.0",
            final_score
        );
        info!("[EVAL_FLOW] Score breakdown: {:?}", score_breakdown);

        Ok(final_score)
    }

    /// Calculate combined verification score from SSH and binary validation
    fn calculate_combined_verification_score(
        &self,
        ssh_score: f64,
        binary_score: f64,
        ssh_successful: bool,
        binary_successful: bool,
    ) -> f64 {
        let binary_config = &self.config.binary_validation;

        info!("[EVAL_FLOW] Starting combined score calculation - SSH: {:.3} (success: {}), Binary: {:.3} (success: {})", 
              ssh_score, ssh_successful, binary_score, binary_successful);

        // If SSH fails, total score is 0
        if !ssh_successful {
            error!("[EVAL_FLOW] SSH validation failed, returning combined score: 0.0");
            return 0.0;
        }

        // If binary validation is disabled, use SSH score only
        if !binary_config.enabled {
            info!(
                "[EVAL_FLOW] Binary validation disabled, using SSH score only: {:.3}",
                ssh_score
            );
            return ssh_score;
        }

        // If binary validation is enabled but failed, penalize but don't zero
        if !binary_successful {
            let penalized_score = ssh_score * 0.5;
            warn!("[EVAL_FLOW] Binary validation failed, applying 50% penalty to SSH score: {:.3} -> {:.3}", 
                  ssh_score, penalized_score);
            return penalized_score;
        }

        // Calculate weighted combination
        let ssh_weight = 1.0 - binary_config.score_weight;
        let binary_weight = binary_config.score_weight;

        let combined_score = (ssh_score * ssh_weight) + (binary_score * binary_weight);

        info!(
            "[EVAL_FLOW] Combined score calculation: ({:.3} × {:.3}) + ({:.3} × {:.3}) = {:.3}",
            ssh_score, ssh_weight, binary_score, binary_weight, combined_score
        );

        // Ensure score is within bounds
        combined_score.clamp(0.0, 1.0)
    }

    /// Cleanup SSH session after validation
    async fn cleanup_ssh_session(
        &self,
        session_info: &protocol::miner_discovery::InitiateSshSessionResponse,
    ) {
        info!(
            "[EVAL_FLOW] Cleaning up SSH session {}",
            session_info.session_id
        );

        let close_request = protocol::miner_discovery::CloseSshSessionRequest {
            session_id: session_info.session_id.clone(),
            validator_hotkey: self.validator_hotkey.to_string(),
            reason: "binary_validation_complete".to_string(),
        };

        // Attempt to close session gracefully
        if let Err(e) = self.close_ssh_session_gracefully(close_request).await {
            warn!("[EVAL_FLOW] Failed to close SSH session gracefully: {}", e);
        }
    }

    /// Helper method for closing SSH sessions gracefully
    async fn close_ssh_session_gracefully(
        &self,
        _close_request: protocol::miner_discovery::CloseSshSessionRequest,
    ) -> Result<()> {
        // Create a miner client
        let _client = self.create_authenticated_client()?;

        // Find the miner endpoint - this is a simplified approach
        // In a real implementation, you'd need to determine which miner this session belongs to
        // For now, we'll just log the attempt
        warn!("SSH session cleanup not fully implemented - session will timeout naturally");
        Ok(())
    }

    /// Test SSH connection with the given details
    async fn test_ssh_connection(&self, ssh_details: &SshConnectionDetails) -> Result<()> {
        self.ssh_client.test_connection(ssh_details).await
    }

    /// Establish SSH session (existing implementation helper)
    async fn establish_ssh_session(
        &self,
        miner_endpoint: &str,
        executor_info: &ExecutorInfoDetailed,
    ) -> Result<(
        SshConnectionDetails,
        protocol::miner_discovery::InitiateSshSessionResponse,
    )> {
        // Create authenticated client
        let client = self.create_authenticated_client()?;
        let mut connection = client.connect_and_authenticate(miner_endpoint).await?;

        // Get SSH key for session
        let (private_key_path, public_key_content) =
            if let Some(ref key_manager) = self.ssh_key_manager {
                if let Some((public_key, private_key_path)) = key_manager.get_persistent_key() {
                    (private_key_path.clone(), public_key.clone())
                } else {
                    return Err(anyhow::anyhow!("No persistent SSH key available"));
                }
            } else {
                return Err(anyhow::anyhow!("SSH key manager not available"));
            };

        // Generate unique session ID
        let _session_id = Uuid::new_v4().to_string();

        // Create SSH session request
        let ssh_request = protocol::miner_discovery::InitiateSshSessionRequest {
            validator_hotkey: self.validator_hotkey.to_string(),
            executor_id: executor_info.id.clone(),
            purpose: "binary_validation".to_string(),
            validator_public_key: public_key_content,
            session_duration_secs: 300, // 5 minutes
            session_metadata: "binary_validation_session".to_string(),
        };

        // Initiate SSH session
        let session_info = connection.initiate_ssh_session_v2(ssh_request).await?;

        // Parse SSH credentials
        let ssh_details =
            self.parse_ssh_credentials(&session_info.access_credentials, Some(private_key_path))?;

        Ok((ssh_details, session_info))
    }

    /// Enhanced verify executor with SSH automation and binary validation
    async fn verify_executor_with_ssh_automation_enhanced(
        &self,
        miner_endpoint: &str,
        executor_info: &ExecutorInfoDetailed,
    ) -> Result<ExecutorVerificationResult> {
        info!(
            "[EVAL_FLOW] Starting enhanced SSH automation verification for executor {} via miner {}",
            executor_info.id, miner_endpoint
        );

        let total_start = std::time::Instant::now();
        let mut validation_details = crate::validation::types::ValidationDetails {
            ssh_test_duration: Duration::from_secs(0),
            binary_upload_duration: Duration::from_secs(0),
            binary_execution_duration: Duration::from_secs(0),
            total_validation_duration: Duration::from_secs(0),
            ssh_score: 0.0,
            binary_score: 0.0,
            combined_score: 0.0,
        };

        // Check for active SSH session and register new session
        {
            let mut active_sessions = self.active_ssh_sessions.lock().await;
            let before_count = active_sessions.len();
            let all_active: Vec<String> = active_sessions.iter().cloned().collect();

            info!("[EVAL_FLOW] SSH session lifecycle check for executor {} - Current state: {} active sessions {:?}", 
                  executor_info.id, before_count, all_active);

            if active_sessions.contains(&executor_info.id) {
                error!(
                    "[EVAL_FLOW] SSH session collision detected for executor {}, rejecting concurrent verification. Active sessions: {:?}",
                    executor_info.id, all_active
                );
                return Ok(ExecutorVerificationResult {
                    executor_id: executor_info.id.clone(),
                    verification_score: 0.0,
                    ssh_connection_successful: false,
                    binary_validation_successful: false,
                    executor_result: None,
                    error: Some(
                        format!("Concurrent SSH session already active for this executor. Active sessions: {:?}", all_active),
                    ),
                    execution_time: Duration::from_secs(0),
                    validation_details,
                });
            }
            
            // Register new SSH session
            let inserted = active_sessions.insert(executor_info.id.clone());
            let after_count = active_sessions.len();
            
            if inserted {
                info!("[EVAL_FLOW] SSH session registered successfully for executor {} - Sessions: {} -> {} (added: {})", 
                      executor_info.id, before_count, after_count, executor_info.id);
            } else {
                warn!("[EVAL_FLOW] SSH session already existed for executor {} during registration - this should not happen", 
                      executor_info.id);
            }
            
            debug!("[EVAL_FLOW] Current active SSH sessions after registration: {:?}", 
                   active_sessions.iter().collect::<Vec<_>>());
        }

        // Establish SSH session (existing implementation)
        let ssh_session_result = self
            .establish_ssh_session(miner_endpoint, executor_info)
            .await;
        let (ssh_details, session_info) = match ssh_session_result {
            Ok(details) => details,
            Err(e) => {
                self.cleanup_active_session(&executor_info.id).await;
                return Ok(ExecutorVerificationResult {
                    executor_id: executor_info.id.clone(),
                    verification_score: 0.0,
                    ssh_connection_successful: false,
                    binary_validation_successful: false,
                    executor_result: None,
                    error: Some(format!("SSH session establishment failed: {}", e)),
                    execution_time: total_start.elapsed(),
                    validation_details,
                });
            }
        };

        // Phase 1: SSH Connection Test (existing implementation)
        info!(
            "[EVAL_FLOW] Phase 1: SSH connection test for executor {}",
            executor_info.id
        );
        let ssh_test_start = std::time::Instant::now();

        let ssh_connection_successful = match self.test_ssh_connection(&ssh_details).await {
            Ok(_) => {
                info!(
                    "[EVAL_FLOW] SSH connection test successful for executor {}",
                    executor_info.id
                );
                true
            }
            Err(e) => {
                error!(
                    "[EVAL_FLOW] SSH connection test failed for executor {}: {}",
                    executor_info.id, e
                );
                false
            }
        };

        validation_details.ssh_test_duration = ssh_test_start.elapsed();
        validation_details.ssh_score = if ssh_connection_successful { 0.8 } else { 0.0 };

        // Phase 2: Binary Validation (NEW)
        let mut binary_validation_successful = false;
        let mut executor_result = None;
        let mut binary_score = 0.0;

        info!(
            "[EVAL_FLOW] Binary validation config check for executor {}: ssh_successful={}, enabled={}, validator_binary_path={:?}",
            executor_info.id, ssh_connection_successful, self.config.binary_validation.enabled, self.config.binary_validation.validator_binary_path
        );

        if ssh_connection_successful && self.config.binary_validation.enabled {
            info!(
                "[EVAL_FLOW] Phase 2: Binary validation for executor {}",
                executor_info.id
            );

            match self
                .execute_binary_validation(&ssh_details, &session_info)
                .await
            {
                Ok(binary_result) => {
                    binary_validation_successful = binary_result.success;
                    executor_result = binary_result.executor_result;
                    binary_score = binary_result.validation_score;
                    validation_details.binary_upload_duration = Duration::from_secs(0); // Upload handled by validator binary
                    validation_details.binary_execution_duration =
                        Duration::from_millis(binary_result.execution_time_ms);

                    info!(
                        "[EVAL_FLOW] Binary validation completed for executor {} - success: {}, score: {:.2}",
                        executor_info.id, binary_validation_successful, binary_score
                    );
                }
                Err(e) => {
                    error!(
                        "[EVAL_FLOW] Binary validation failed for executor {}: {}",
                        executor_info.id, e
                    );
                    binary_validation_successful = false;
                    binary_score = 0.0;
                }
            }
        } else if !self.config.binary_validation.enabled {
            info!(
                "[EVAL_FLOW] Binary validation disabled for executor {}",
                executor_info.id
            );
            binary_validation_successful = true; // Not required
            binary_score = 0.8; // Default score when disabled
        }

        // Phase 3: Calculate Combined Score
        let combined_score = self.calculate_combined_verification_score(
            validation_details.ssh_score,
            binary_score,
            ssh_connection_successful,
            binary_validation_successful,
        );

        validation_details.combined_score = combined_score;
        validation_details.binary_score = binary_score;
        validation_details.total_validation_duration = total_start.elapsed();

        // Phase 4: Session and Resource Cleanup
        info!("[EVAL_FLOW] Phase 4: Starting cleanup for executor {} - Duration: {:.2}s", 
              executor_info.id, total_start.elapsed().as_secs_f64());
        
        self.cleanup_ssh_session(&session_info).await;
        self.cleanup_active_session(&executor_info.id).await;

        info!(
            "[EVAL_FLOW] Enhanced verification completed for executor {} - SSH: {}, Binary: {}, Combined: {:.2}, Duration: {:.2}s",
            executor_info.id, ssh_connection_successful, binary_validation_successful, combined_score, total_start.elapsed().as_secs_f64()
        );

        Ok(ExecutorVerificationResult {
            executor_id: executor_info.id.clone(),
            verification_score: combined_score,
            ssh_connection_successful,
            binary_validation_successful,
            executor_result,
            error: None,
            execution_time: total_start.elapsed(),
            validation_details,
        })
    }
}

/// SSH automation status information
#[derive(Debug, Clone)]
pub struct SshAutomationStatus {
    pub dynamic_discovery_enabled: bool,
    pub ssh_key_manager_available: bool,
    pub bittensor_service_available: bool,
    pub fallback_key_path: Option<PathBuf>,
}

impl std::fmt::Display for SshAutomationStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SSH Automation Status[dynamic={}, key_manager={}, bittensor={}, fallback_key={}]",
            self.dynamic_discovery_enabled,
            self.ssh_key_manager_available,
            self.bittensor_service_available,
            self.fallback_key_path
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or("none".to_string())
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
    pub ssh_connection_successful: bool,
    pub binary_validation_successful: bool,
    pub executor_result: Option<crate::validation::types::ExecutorResult>,
    pub error: Option<String>,
    pub execution_time: Duration,
    pub validation_details: crate::validation::types::ValidationDetails,
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
