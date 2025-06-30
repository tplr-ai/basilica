//! # Validator Communications
//!
//! Simplified gRPC server for handling validator requests according to SPEC v1.6.
//! Primary responsibilities:
//! - Authenticate validators
//! - List available executors
//! - Coordinate SSH access to executors

use anyhow::Result;
use std::net::SocketAddr;
use std::sync::Arc;
use uuid::Uuid;

use crate::executor_manager::AvailableExecutor;

use tonic::{transport::Server, Request, Response, Status};
use tonic_health::server::health_reporter;
use tracing::{debug, error, info, warn};

use common::identity::Hotkey;
use protocol::miner_discovery::{
    miner_discovery_server::{MinerDiscovery, MinerDiscoveryServer},
    ExecutorConnectionDetails, LeaseOfferResponse, LeaseRequest, MinerAuthResponse,
    SessionInitRequest, SessionInitResponse, ValidatorAuthRequest,
};

use crate::auth::JwtAuthService;
use crate::config::{SecurityConfig, ValidatorCommsConfig};
use crate::executor_manager::ExecutorManager;
use crate::persistence::RegistrationDb;
use crate::ssh::ValidatorAccessService;
use crate::validator_discovery::ValidatorDiscovery;

/// Validator communications server
#[derive(Clone)]
pub struct ValidatorCommsServer {
    config: ValidatorCommsConfig,
    security_config: SecurityConfig,
    executor_manager: Arc<ExecutorManager>,
    db: RegistrationDb,
    ssh_access_service: ValidatorAccessService,
    pub jwt_service: Arc<JwtAuthService>,
    validator_discovery: Option<Arc<ValidatorDiscovery>>,
}

impl ValidatorCommsServer {
    /// Create a new validator communications server
    pub async fn new(
        config: ValidatorCommsConfig,
        security_config: SecurityConfig,
        executor_manager: Arc<ExecutorManager>,
        db: RegistrationDb,
        ssh_access_service: ValidatorAccessService,
        validator_discovery: Option<Arc<ValidatorDiscovery>>,
    ) -> Result<Self> {
        info!("Initializing validator communications server");

        // Initialize JWT service
        let jwt_service = Arc::new(JwtAuthService::new(
            &security_config.jwt_secret,
            "basilica-miner".to_string(),
            "basilica-miner".to_string(),
            chrono::Duration::seconds(security_config.token_expiration.as_secs() as i64),
        )?);

        Ok(Self {
            config,
            security_config,
            executor_manager,
            db,
            ssh_access_service,
            jwt_service,
            validator_discovery,
        })
    }

    /// Start serving gRPC requests
    pub async fn serve(&self, addr: SocketAddr) -> Result<()> {
        info!("Starting validator communications server on {}", addr);

        let miner_discovery_service = MinerDiscoveryService {
            _config: self.config.clone(),
            security_config: self.security_config.clone(),
            executor_manager: self.executor_manager.clone(),
            db: self.db.clone(),
            ssh_access_service: self.ssh_access_service.clone(),
            jwt_service: self.jwt_service.clone(),
            validator_discovery: self.validator_discovery.clone(),
        };

        // Create health reporter
        let (mut health_reporter, health_service) = health_reporter();

        // Set the service as serving
        health_reporter
            .set_serving::<MinerDiscoveryServer<MinerDiscoveryService>>()
            .await;

        let server = Server::builder()
            .add_service(health_service)
            .add_service(MinerDiscoveryServer::new(miner_discovery_service))
            .serve(addr);

        info!("Validator communications server started successfully");

        if let Err(e) = server.await {
            error!("Validator communications server error: {}", e);
            return Err(e.into());
        }

        Ok(())
    }
}

/// Simplified gRPC service implementation for miner discovery
#[derive(Clone)]
struct MinerDiscoveryService {
    _config: ValidatorCommsConfig,
    security_config: SecurityConfig,
    executor_manager: Arc<ExecutorManager>,
    db: RegistrationDb,
    ssh_access_service: ValidatorAccessService,
    jwt_service: Arc<JwtAuthService>,
    validator_discovery: Option<Arc<ValidatorDiscovery>>,
}

#[tonic::async_trait]
impl MinerDiscovery for MinerDiscoveryService {
    /// Authenticate a validator using Bittensor signature
    async fn authenticate_validator(
        &self,
        request: Request<ValidatorAuthRequest>,
    ) -> Result<Response<MinerAuthResponse>, Status> {
        let auth_request = request.into_inner();

        debug!(
            "Received authentication request from validator: {}",
            auth_request.validator_hotkey
        );

        // Verify the signature if enabled
        if self.security_config.verify_signatures {
            // Parse validator hotkey
            let validator_hotkey = Hotkey::new(auth_request.validator_hotkey.clone())
                .map_err(|e| Status::invalid_argument(format!("Invalid hotkey: {e}")))?;

            // Verify signature using bittensor crate
            if let Err(e) = bittensor::utils::verify_bittensor_signature(
                &validator_hotkey,
                &auth_request.signature,
                auth_request.nonce.as_bytes(),
            ) {
                warn!(
                    "Signature verification failed for validator {}: {}",
                    auth_request.validator_hotkey, e
                );
                return Err(Status::unauthenticated("Invalid signature"));
            }
        }

        // Check if validator is in allowlist (if configured)
        if !self.security_config.allowed_validators.is_empty() {
            let validator_hotkey = Hotkey::new(auth_request.validator_hotkey.clone())
                .map_err(|e| Status::invalid_argument(format!("Invalid hotkey: {e}")))?;

            if !self
                .security_config
                .allowed_validators
                .contains(&validator_hotkey)
            {
                warn!(
                    "Validator {} not in allowlist",
                    auth_request.validator_hotkey
                );
                return Err(Status::permission_denied("Validator not authorized"));
            }
        }

        // Record validator interaction
        if let Err(e) = self
            .db
            .update_validator_interaction(&auth_request.validator_hotkey, true)
            .await
        {
            error!("Failed to record validator interaction: {}", e);
        }

        // Parse validator hotkey for JWT
        let validator_hotkey = Hotkey::new(auth_request.validator_hotkey.clone())
            .map_err(|e| Status::invalid_argument(format!("Invalid hotkey: {e}")))?;

        // Generate session ID
        let session_id = format!("session_{}", uuid::Uuid::new_v4());

        // Define validator permissions
        let permissions = vec![
            "executor.list".to_string(),
            "executor.access".to_string(),
            "executor.lease".to_string(),
        ];

        // Extract IP address from request metadata if available
        // Note: In production, you would extract this from the transport layer
        let ip_address = None;

        // Generate JWT token
        let session_token = self
            .jwt_service
            .generate_token(&validator_hotkey, &session_id, permissions, ip_address)
            .await
            .map_err(|e| {
                error!("Failed to generate JWT token: {}", e);
                Status::internal("Failed to generate authentication token")
            })?;

        info!(
            "Successfully authenticated validator: {} with session: {}",
            auth_request.validator_hotkey, session_id
        );

        // Calculate expiration time
        let expires_at = chrono::Utc::now()
            + chrono::Duration::seconds(self.security_config.token_expiration.as_secs() as i64);

        let response = MinerAuthResponse {
            authenticated: true,
            session_token,
            expires_at: Some(protocol::common::Timestamp {
                value: Some(prost_types::Timestamp::from(std::time::SystemTime::from(
                    expires_at,
                ))),
            }),
            error: None,
        };

        Ok(Response::new(response))
    }

    /// Request available executor leases from miner (adapted to list executors)
    async fn request_executor_lease(
        &self,
        request: Request<LeaseRequest>,
    ) -> Result<Response<LeaseOfferResponse>, Status> {
        let lease_request = request.into_inner();

        debug!("Received executor lease request");

        // Validate JWT token
        let claims = self
            .jwt_service
            .validate_token(&lease_request.session_token)
            .await
            .map_err(|e| {
                debug!("Token validation failed: {}", e);
                Status::unauthenticated("Invalid or expired session token")
            })?;

        // Check if validator has permission to list executors
        if !claims.permissions.contains(&"executor.list".to_string()) {
            return Err(Status::permission_denied("Insufficient permissions"));
        }

        debug!("Validated lease request from validator: {}", claims.sub);

        // Check if validator discovery is enabled and has assignments for this validator
        let executors = if let Some(ref discovery) = self.validator_discovery {
            // Get assigned executor IDs for this validator
            if let Some(assigned_executor_ids) =
                discovery.get_validator_assignments(&claims.sub).await
            {
                debug!(
                    "Found {} assigned executors for validator {}",
                    assigned_executor_ids.len(),
                    claims.sub
                );

                // Get all available executors
                let all_executors = self
                    .executor_manager
                    .list_available()
                    .await
                    .map_err(|e| Status::internal(format!("Failed to list executors: {e}")))?;

                // Filter to only assigned executors
                all_executors
                    .into_iter()
                    .filter(|exec| assigned_executor_ids.contains(&exec.id))
                    .collect()
            } else {
                // No assignments for this validator
                warn!("No executor assignments found for validator {}", claims.sub);
                Vec::new()
            }
        } else {
            // Validator discovery disabled - return all available executors (original behavior)
            debug!("Validator discovery disabled, returning all available executors");
            self.executor_manager
                .list_available()
                .await
                .map_err(|e| Status::internal(format!("Failed to list executors: {e}")))?
        };

        // Convert to ExecutorConnectionDetails
        let executor_details: Vec<ExecutorConnectionDetails> = executors
            .into_iter()
            .map(|exec| {
                let gpu_spec = create_gpu_spec_from_executor(&exec);
                ExecutorConnectionDetails {
                    executor_id: exec.id,
                    grpc_endpoint: exec.grpc_address,
                    gpu_spec,
                    available_resources: exec.resources.map(|r| protocol::common::ResourceLimits {
                        max_cpu_cores: r.cpu_percent as u32,
                        max_memory_mb: r.memory_mb,
                        max_storage_mb: 0, // Not provided in ResourceUsageStats
                        max_containers: 1,
                        max_bandwidth_mbps: 0.0,
                        max_gpus: exec.gpu_count,
                    }),
                    status: "available".to_string(),
                }
            })
            .collect();

        info!("Returning {} available executors", executor_details.len());

        let response = LeaseOfferResponse {
            available_executors: executor_details,
            error: None,
        };

        Ok(Response::new(response))
    }

    /// Initiate session with specific executor (adapted for SSH access)
    async fn initiate_executor_session(
        &self,
        request: Request<SessionInitRequest>,
    ) -> Result<Response<SessionInitResponse>, Status> {
        let session_request = request.into_inner();

        debug!(
            "Received session init request for executor {}",
            session_request.executor_id
        );

        // Validate JWT token
        let claims = self
            .jwt_service
            .validate_token(&session_request.session_token)
            .await
            .map_err(|e| {
                debug!("Token validation failed: {}", e);
                Status::unauthenticated("Invalid or expired session token")
            })?;

        // Check if validator has permission to access executors
        if !claims.permissions.contains(&"executor.access".to_string()) {
            return Err(Status::permission_denied("Insufficient permissions"));
        }

        // Verify the validator hotkey matches
        if claims.sub != session_request.validator_hotkey {
            return Err(Status::permission_denied("Token validator mismatch"));
        }

        debug!(
            "Validated session init request from validator: {}",
            claims.sub
        );

        // Create SSH session for the validator to access the executor
        let connection_string = match self
            .ssh_access_service
            .provision_validator_access(
                &session_request.validator_hotkey,
                &session_request.executor_id,
                None, // Use default timeout
            )
            .await
        {
            Ok(connection) => connection,
            Err(e) => {
                error!("Failed to provision SSH access: {}", e);
                return Err(Status::internal(format!(
                    "Failed to create SSH access: {e}"
                )));
            }
        };

        let session_id = format!("session_{}", Uuid::new_v4().simple());

        // Record the session initiation
        if let Err(e) = self
            .db
            .record_validator_interaction(
                &session_request.validator_hotkey,
                "session_init",
                true,
                Some(
                    serde_json::json!({
                        "executor_id": session_request.executor_id,
                        "session_type": session_request.session_type,
                    })
                    .to_string(),
                ),
            )
            .await
        {
            error!("Failed to record session initiation: {}", e);
        }

        let response = SessionInitResponse {
            success: true,
            session_id,
            access_credentials: connection_string,
            error: None,
        };

        Ok(Response::new(response))
    }
}

/// Create GPU spec from available executor information
fn create_gpu_spec_from_executor(exec: &AvailableExecutor) -> Option<protocol::common::GpuSpec> {
    if exec.gpu_count == 0 {
        return None;
    }

    // Use available runtime information from ResourceUsageStats if present
    if let Some(ref resources) = exec.resources {
        // Take first GPU metrics as representative (could be extended to handle multiple GPUs)
        let gpu_utilization = resources.gpu_utilization.first().copied().unwrap_or(0.0);
        let gpu_memory_mb = resources.gpu_memory_mb.first().copied().unwrap_or(0);
        let memory_utilization = if gpu_memory_mb > 0 {
            (gpu_memory_mb as f64 / (gpu_memory_mb as f64 * 1.2)).min(100.0)
        } else {
            0.0
        };

        Some(protocol::common::GpuSpec {
            model: format!("GPU-{}", &exec.id[..8.min(exec.id.len())]), // Placeholder based on executor ID
            memory_mb: gpu_memory_mb,
            uuid: format!("gpu-{}-{}", exec.id, 0), // Generate deterministic UUID
            driver_version: "unknown".to_string(),
            cuda_version: "unknown".to_string(),
            utilization_percent: gpu_utilization,
            memory_utilization_percent: memory_utilization,
            temperature_celsius: 0.0, // Not available from current data
            power_watts: 0.0,         // Not available from current data
            core_clock_mhz: 0,
            memory_clock_mhz: 0,
            compute_capability: "unknown".to_string(),
        })
    } else {
        // Fallback for executors without resource stats
        Some(protocol::common::GpuSpec {
            model: format!("GPU-{}", &exec.id[..8.min(exec.id.len())]),
            memory_mb: 0,
            uuid: format!("gpu-{}-{}", exec.id, 0),
            driver_version: "unknown".to_string(),
            cuda_version: "unknown".to_string(),
            utilization_percent: 0.0,
            memory_utilization_percent: 0.0,
            temperature_celsius: 0.0,
            power_watts: 0.0,
            core_clock_mhz: 0,
            memory_clock_mhz: 0,
            compute_capability: "unknown".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ExecutorConfig;

    #[tokio::test]
    async fn test_validator_auth_with_production_verification() {
        let config = ValidatorCommsConfig::default();
        let security_config = SecurityConfig {
            verify_signatures: true,
            ..Default::default()
        };

        let miner_config = crate::config::MinerConfig {
            executor_management: crate::config::ExecutorManagementConfig {
                executors: vec![ExecutorConfig {
                    id: "test-executor".to_string(),
                    grpc_address: "127.0.0.1:50051".to_string(),
                    name: None,
                    metadata: None,
                }],
                ..Default::default()
            },
            database: common::config::DatabaseConfig {
                url: "sqlite::memory:".to_string(),
                ..Default::default()
            },
            ..Default::default()
        };

        let db = RegistrationDb::new(&miner_config.database).await.unwrap();
        let executor_manager = Arc::new(
            ExecutorManager::new(&miner_config, db.clone())
                .await
                .unwrap(),
        );

        // Create SSH access service for testing
        let ssh_config = crate::ssh::MinerSshConfig {
            key_directory: std::path::PathBuf::from("/tmp/test_ssh_keys"),
            ..crate::ssh::MinerSshConfig::default()
        };
        let ssh_service = std::sync::Arc::new(
            common::ssh::manager::DefaultSshService::new(ssh_config.clone()).unwrap(),
        );
        let ssh_access_service = crate::ssh::ValidatorAccessService::new(
            ssh_config,
            ssh_service,
            executor_manager.clone(),
            db.clone(),
        )
        .await
        .unwrap();

        // Create JWT service for testing
        let jwt_service = Arc::new(
            JwtAuthService::new(
                "test_secret_key_that_is_long_enough_for_security",
                "test-miner".to_string(),
                "test-miner".to_string(),
                chrono::Duration::hours(1),
            )
            .unwrap(),
        );

        let service = MinerDiscoveryService {
            _config: config,
            security_config,
            executor_manager,
            db,
            ssh_access_service,
            jwt_service,
            validator_discovery: None,
        };

        // Test with production-level verification enabled
        let test_hotkey = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY";
        let nonce = "test-nonce";

        // Use a signature that should fail verification (testing the verification path)
        let invalid_signature = "deadbeef".repeat(16); // 64-byte hex string but invalid signature

        let request = ValidatorAuthRequest {
            validator_hotkey: test_hotkey.to_string(),
            signature: invalid_signature,
            nonce: nonce.to_string(),
            timestamp: None,
        };

        // This should fail authentication due to invalid signature
        let result = service.authenticate_validator(Request::new(request)).await;

        // Verify that the authentication fails with proper signature verification
        assert!(
            result.is_err(),
            "Authentication should fail with invalid signature"
        );
        if let Err(status) = result {
            assert_eq!(status.code(), tonic::Code::Unauthenticated);
            assert!(status.message().contains("Invalid signature"));
        }
    }

    #[tokio::test]
    async fn test_validator_auth_signature_verification_path() {
        let config = ValidatorCommsConfig::default();
        let security_config = SecurityConfig {
            verify_signatures: true,
            ..Default::default()
        };

        let miner_config = crate::config::MinerConfig {
            executor_management: crate::config::ExecutorManagementConfig {
                executors: vec![ExecutorConfig {
                    id: "test-executor".to_string(),
                    grpc_address: "127.0.0.1:50051".to_string(),
                    name: None,
                    metadata: None,
                }],
                ..Default::default()
            },
            database: common::config::DatabaseConfig {
                url: "sqlite::memory:".to_string(),
                ..Default::default()
            },
            ..Default::default()
        };

        let db = RegistrationDb::new(&miner_config.database).await.unwrap();
        let executor_manager = Arc::new(
            ExecutorManager::new(&miner_config, db.clone())
                .await
                .unwrap(),
        );

        // Create SSH access service for testing
        let ssh_config = crate::ssh::MinerSshConfig {
            key_directory: std::path::PathBuf::from("/tmp/test_ssh_keys"),
            ..crate::ssh::MinerSshConfig::default()
        };
        let ssh_service = std::sync::Arc::new(
            common::ssh::manager::DefaultSshService::new(ssh_config.clone()).unwrap(),
        );
        let ssh_access_service = crate::ssh::ValidatorAccessService::new(
            ssh_config,
            ssh_service,
            executor_manager.clone(),
            db.clone(),
        )
        .await
        .unwrap();

        // Create JWT service for testing
        let jwt_service = Arc::new(
            JwtAuthService::new(
                "test_secret_key_that_is_long_enough_for_security",
                "test-miner".to_string(),
                "test-miner".to_string(),
                chrono::Duration::hours(1),
            )
            .unwrap(),
        );

        let service = MinerDiscoveryService {
            _config: config,
            security_config,
            executor_manager,
            db,
            ssh_access_service,
            jwt_service,
            validator_discovery: None,
        };

        // Test various invalid signature scenarios to ensure production verification works

        // Test 1: Empty signature
        let request = ValidatorAuthRequest {
            validator_hotkey: "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string(),
            signature: "".to_string(),
            nonce: "test-nonce".to_string(),
            timestamp: None,
        };

        let result = service.authenticate_validator(Request::new(request)).await;
        assert!(result.is_err(), "Empty signature should fail");

        // Test 2: Invalid hex signature
        let request = ValidatorAuthRequest {
            validator_hotkey: "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string(),
            signature: "invalid_hex_!@#$".to_string(),
            nonce: "test-nonce".to_string(),
            timestamp: None,
        };

        let result = service.authenticate_validator(Request::new(request)).await;
        assert!(result.is_err(), "Invalid hex signature should fail");

        // Test 3: Wrong length signature
        let request = ValidatorAuthRequest {
            validator_hotkey: "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string(),
            signature: "deadbeef".to_string(), // Too short
            nonce: "test-nonce".to_string(),
            timestamp: None,
        };

        let result = service.authenticate_validator(Request::new(request)).await;
        assert!(result.is_err(), "Wrong length signature should fail");
    }
}
