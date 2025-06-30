//! gRPC server implementation for the Basilca Executor
//!
//! Provides the ExecutorControl service interface for miner communication.
//! Modularized following SOLID principles for maintainability and testability.

pub mod container_operations;
pub mod executor_management;
pub mod health_check;
pub mod system_profile;
pub mod types;
pub mod validator_access;

// Re-export gRPC types for external use
pub use types::{
    GrpcResult, SharedExecutorState, ValidatorAccessInfo, ValidatorAccessRequest,
    ValidatorAccessResponse, ValidatorHealthRequest, ValidatorHealthResponse, ValidatorListRequest,
    ValidatorListResponse, ValidatorRevokeRequest, ValidatorRevokeResponse, ValidatorServiceTrait,
};

use crate::ExecutorState;
use anyhow::Result;
use container_operations::ContainerOperationsService;
use health_check::{HealthCheckService, HealthStatus};
use std::net::SocketAddr;
use std::sync::Arc;
use system_profile::SystemProfileService;
use tracing::info;
use validator_access::ValidatorAccessService;

use protocol::common::{ChallengeResult, LogEntry};
use protocol::executor_control::{
    executor_control_server::{ExecutorControl, ExecutorControlServer},
    BenchmarkRequest, BenchmarkResponse, ChallengeRequest, ChallengeResponse, ContainerOpRequest,
    ContainerOpResponse, HealthCheckRequest, HealthCheckResponse, LogSubscriptionRequest,
    ProvisionAccessRequest, ProvisionAccessResponse, SystemProfileRequest, SystemProfileResponse,
};
use tokio_stream::wrappers::ReceiverStream;

/// gRPC server for executor control
pub struct ExecutorServer {
    state: SharedExecutorState,
}

impl ExecutorServer {
    /// Create new executor server
    pub fn new(state: ExecutorState) -> Self {
        Self {
            state: Arc::new(state),
        }
    }

    /// Get the state reference
    pub fn state(&self) -> &SharedExecutorState {
        &self.state
    }

    /// Start serving gRPC requests
    pub async fn serve(self, addr: SocketAddr) -> Result<()> {
        info!("Starting gRPC server on {}", addr);

        let control_service = ExecutorControlService::new(self.state.clone());
        let management_service = executor_management::ExecutorManagementService::new(self.state);

        tonic::transport::Server::builder()
            .add_service(ExecutorControlServer::new(control_service))
            .add_service(protocol::executor_management::executor_management_server::ExecutorManagementServer::new(management_service))
            .serve(addr)
            .await
            .map_err(|e| anyhow::anyhow!("gRPC server error: {}", e))?;

        Ok(())
    }
}

/// Implementation of the ExecutorControl gRPC service
/// Composed of focused service modules following SOLID principles
pub struct ExecutorControlService {
    state: SharedExecutorState,
    validator_access: ValidatorAccessService,
    system_profile: SystemProfileService,
    container_operations: ContainerOperationsService,
    health_check: HealthCheckService,
}

impl ExecutorControlService {
    /// Create new executor control service
    pub fn new(state: SharedExecutorState) -> Self {
        Self {
            state: state.clone(),
            validator_access: ValidatorAccessService::new(state.clone()),
            system_profile: SystemProfileService::new(state.clone()),
            container_operations: ContainerOperationsService::new(state.clone()),
            health_check: HealthCheckService::new(state),
        }
    }

    /// Perform comprehensive health check
    pub async fn perform_health_check(&self) -> Result<HealthStatus> {
        self.health_check.health_check().await
    }

    /// Placeholder implementations for development (matching original functionality)
    ///
    /// Provision validator access placeholder
    pub async fn provision_validator_access_placeholder(
        &self,
        validator_hotkey: &str,
        ssh_public_key: Option<String>,
    ) -> Result<()> {
        // Require SSH public key for simplified approach
        let ssh_key = ssh_public_key
            .ok_or_else(|| anyhow::anyhow!("SSH public key required for simplified access"))?;

        self.validator_access
            .provision_ssh_access(validator_hotkey, ssh_key)
            .await
    }

    /// Handle full gRPC validator access request
    pub async fn handle_validator_access_request(
        &self,
        request: tonic::Request<types::ValidatorAccessRequest>,
    ) -> Result<tonic::Response<types::ValidatorAccessResponse>, tonic::Status> {
        self.validator_access.handle_access_request(request).await
    }

    /// Handle gRPC validator access revocation
    pub async fn handle_validator_revoke_request(
        &self,
        request: tonic::Request<types::ValidatorRevokeRequest>,
    ) -> Result<tonic::Response<types::ValidatorRevokeResponse>, tonic::Status> {
        self.validator_access.handle_revoke_request(request).await
    }

    /// Handle gRPC validator access list request
    pub async fn handle_validator_list_request(
        &self,
        request: tonic::Request<types::ValidatorListRequest>,
    ) -> Result<tonic::Response<types::ValidatorListResponse>, tonic::Status> {
        self.validator_access.handle_list_request(request).await
    }

    /// Handle gRPC validator health check
    pub async fn handle_validator_health_check(
        &self,
        request: tonic::Request<types::ValidatorHealthRequest>,
    ) -> Result<tonic::Response<types::ValidatorHealthResponse>, tonic::Status> {
        self.validator_access.handle_health_check(request).await
    }

    /// Execute system profile placeholder
    pub async fn execute_system_profile_placeholder(&self) -> Result<String> {
        self.system_profile.execute_system_profile().await
    }

    /// Container operations placeholder
    pub async fn container_operations_placeholder(&self) -> Result<()> {
        self.container_operations
            .container_operations_health_check()
            .await
    }
}

#[tonic::async_trait]
impl ExecutorControl for ExecutorControlService {
    async fn provision_validator_access(
        &self,
        request: tonic::Request<ProvisionAccessRequest>,
    ) -> Result<tonic::Response<ProvisionAccessResponse>, tonic::Status> {
        let req = request.into_inner();

        if req.validator_hotkey.is_empty() {
            return Err(tonic::Status::invalid_argument("Validator hotkey required"));
        }

        let state = &*self.state;
        let validation_service = match &state.validation_service {
            Some(service) => service,
            None => {
                return Err(tonic::Status::unavailable("Validation service not enabled"));
            }
        };

        let validator_id =
            crate::validation_session::types::ValidatorId::new(req.validator_hotkey.clone());

        // Generate ephemeral SSH keypair for this session
        let (private_key_pem, public_key_openssh) =
            common::crypto::generate_ephemeral_ed25519_keypair();

        // Grant SSH access using the generated public key
        validation_service
            .grant_ssh_access(&validator_id, &public_key_openssh)
            .await
            .map_err(|e| {
                tracing::error!("Failed to provision SSH access: {}", e);
                tonic::Status::internal("Failed to provision access")
            })?;

        // Create SSH credentials in JSON format
        let credentials = serde_json::json!({
            "ssh_private_key": private_key_pem,
            "ssh_username": format!("validator_{}", req.validator_hotkey),
            "ssh_host": "executor.local",
            "ssh_port": 22
        })
        .to_string();

        // Set expiration to 1 hour from now
        let expires_at = std::time::SystemTime::now() + std::time::Duration::from_secs(3600);

        Ok(tonic::Response::new(ProvisionAccessResponse {
            success: true,
            connection_endpoint: "executor.local".to_string(),
            credentials,
            expires_at: Some(protocol::common::Timestamp {
                value: Some(prost_types::Timestamp::from(expires_at)),
            }),
            error: None,
        }))
    }

    async fn execute_system_profile(
        &self,
        request: tonic::Request<SystemProfileRequest>,
    ) -> Result<tonic::Response<SystemProfileResponse>, tonic::Status> {
        let req = request.into_inner();

        if req.validator_hotkey.is_empty() {
            return Err(tonic::Status::invalid_argument("Validator hotkey required"));
        }

        // Get system information from the system monitor
        let system_info = self
            .state
            .system_monitor
            .get_system_info()
            .await
            .map_err(|e| {
                tracing::error!("Failed to get system info: {}", e);
                tonic::Status::internal("Failed to collect system profile")
            })?;

        // Create comprehensive system profile
        let profile_data = serde_json::json!({
            "timestamp": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
            "validator_hotkey": req.validator_hotkey,
            "executor_id": self.state.id.to_string(),
            "system": {
                "hostname": system_info.system.hostname,
                "os_name": system_info.system.os_name,
                "os_version": system_info.system.os_version,
                "kernel": system_info.system.kernel_version,
                "uptime_seconds": system_info.system.uptime_seconds
            },
            "cpu": {
                "model": system_info.cpu.model,
                "cores": system_info.cpu.cores,
                "usage_percent": system_info.cpu.usage_percent,
                "frequency_mhz": system_info.cpu.frequency_mhz
            },
            "memory": {
                "total_bytes": system_info.memory.total_bytes,
                "available_bytes": system_info.memory.available_bytes,
                "usage_percent": system_info.memory.usage_percent
            },
            "gpu": system_info.gpu.iter().map(|gpu| {
                serde_json::json!({
                    "index": gpu.index,
                    "name": gpu.name,
                    "memory_total_bytes": gpu.memory_total_bytes,
                    "memory_used_bytes": gpu.memory_used_bytes,
                    "utilization_percent": gpu.utilization_percent,
                    "temperature_celsius": gpu.temperature_celsius,
                    "driver_version": gpu.driver_version,
                    "cuda_version": gpu.cuda_version
                })
            }).collect::<Vec<_>>(),
            "disk": system_info.disk.iter().map(|disk| {
                serde_json::json!({
                    "name": disk.name,
                    "total_bytes": disk.total_bytes,
                    "available_bytes": disk.available_bytes,
                    "usage_percent": disk.usage_percent
                })
            }).collect::<Vec<_>>(),
            "network": {
                "interfaces": system_info.network.interfaces.iter().map(|iface| {
                    serde_json::json!({
                        "name": iface.name,
                        "bytes_sent": iface.bytes_sent,
                        "bytes_received": iface.bytes_received
                    })
                }).collect::<Vec<_>>()
            }
        });

        let profile_json = profile_data.to_string();
        let profile_bytes = profile_json.as_bytes();

        // Generate encryption key from GPU information for deterministic encryption
        let gpu_info_str = system_info
            .gpu
            .iter()
            .map(|gpu| {
                format!(
                    "{},{},{},{}",
                    gpu.index, gpu.name, gpu.memory_total_bytes, gpu.driver_version
                )
            })
            .collect::<Vec<_>>()
            .join("|");

        let encryption_key = common::crypto::derive_key_from_gpu_info(&gpu_info_str);

        // Encrypt the profile data
        let encrypted_data = common::crypto::symmetric_encrypt(&encryption_key, profile_bytes)
            .map_err(|e| {
                tracing::error!("Failed to encrypt system profile: {}", e);
                tonic::Status::internal("Failed to encrypt system profile")
            })?;

        // Extract nonce (first 12 bytes) and ciphertext
        let (nonce_bytes, ciphertext_bytes) =
            encrypted_data.split_at(common::crypto::AES_NONCE_SIZE);
        let encrypted_profile = hex::encode(ciphertext_bytes);
        let encryption_nonce = hex::encode(nonce_bytes);

        // Generate profile hash for integrity verification
        let profile_hash = common::crypto::hash_blake3_string(profile_bytes);

        let collected_at = std::time::SystemTime::now();

        Ok(tonic::Response::new(SystemProfileResponse {
            encrypted_profile,
            encryption_nonce,
            collected_at: Some(protocol::common::Timestamp {
                value: Some(prost_types::Timestamp::from(collected_at)),
            }),
            profile_hash,
            error: None,
        }))
    }

    async fn execute_computational_challenge(
        &self,
        request: tonic::Request<ChallengeRequest>,
    ) -> Result<tonic::Response<ChallengeResponse>, tonic::Status> {
        let req = request.into_inner();
        info!(
            "Computational challenge requested by validator: {}",
            req.validator_hotkey
        );

        let params = req
            .parameters
            .ok_or_else(|| tonic::Status::invalid_argument("Challenge parameters required"))?;

        match params.challenge_type.as_str() {
            "vdf" => {
                // Parse VDF parameters from JSON
                let vdf_params: serde_json::Value = serde_json::from_str(&params.parameters_json)
                    .map_err(|e| {
                    tonic::Status::invalid_argument(format!("Invalid VDF parameters: {e}"))
                })?;

                let iterations = vdf_params["iterations"].as_u64().unwrap_or(1000000) as u32;

                // Create VDF challenge
                let vdf_params = gpu_attestor::vdf::VdfParameters {
                    modulus: vec![0u8; 256], // Placeholder RSA modulus
                    generator: vec![2u8],    // Simple generator
                    difficulty: iterations as u64,
                    challenge_seed: params.seed.clone().into_bytes(),
                };

                let challenge = gpu_attestor::vdf::VdfChallenge {
                    parameters: vdf_params,
                    expected_computation_time_ms: 1000,
                    max_allowed_time_ms: 60000,
                    min_required_time_ms: 100,
                };

                // Track active challenge
                self.state
                    .active_challenges
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                // Compute VDF proof
                let start_time = std::time::Instant::now();
                let vdf_proof = gpu_attestor::vdf::compute_vdf_proof(
                    &challenge,
                    gpu_attestor::vdf::VdfAlgorithm::Wesolowski,
                )
                .map_err(|e| {
                    self.state
                        .active_challenges
                        .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                    tonic::Status::internal(format!("VDF computation failed: {e}"))
                })?;
                let computation_time = start_time.elapsed();

                // Challenge completed
                self.state
                    .active_challenges
                    .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);

                // Create challenge result
                let result = ChallengeResult {
                    solution: serde_json::to_string(&vdf_proof).map_err(|e| {
                        tonic::Status::internal(format!("Failed to serialize VDF proof: {e}"))
                    })?,
                    execution_time_ms: computation_time.as_millis() as u64,
                    gpu_utilization: vec![100.0], // VDF uses full GPU
                    memory_usage_mb: 1024,        // Approximate
                    error_message: String::new(),
                    metadata_json: serde_json::json!({
                        "challenge_type": "vdf",
                        "algorithm": "wesolowski",
                        "iterations": iterations,
                        "validator": req.validator_hotkey,
                        "nonce": req.nonce
                    })
                    .to_string(),
                };

                Ok(tonic::Response::new(ChallengeResponse {
                    result: Some(result),
                    metadata: Default::default(),
                    error: None,
                }))
            }
            "hardware_attestation" => {
                // Generate hardware attestation report
                let attestation =
                    gpu_attestor::attestation::AttestationBuilder::new(self.state.id.to_string())
                        .build();

                // Attestation is already complete

                let result = ChallengeResult {
                    solution: serde_json::to_string(&attestation)
                        .map_err(|e| tonic::Status::internal(format!("Failed to serialize attestation: {e}")))?,
                    execution_time_ms: 0, // Attestation is not time-based
                    gpu_utilization: vec![],
                    memory_usage_mb: 0,
                    error_message: String::new(),
                    metadata_json: serde_json::json!({
                        "challenge_type": "hardware_attestation",
                        "validator": req.validator_hotkey,
                        "nonce": req.nonce,
                        "timestamp": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()
                    }).to_string(),
                };

                Ok(tonic::Response::new(ChallengeResponse {
                    result: Some(result),
                    metadata: Default::default(),
                    error: None,
                }))
            }
            _ => Err(tonic::Status::invalid_argument(format!(
                "Unknown challenge type: {}",
                params.challenge_type
            ))),
        }
    }

    async fn execute_benchmark(
        &self,
        request: tonic::Request<BenchmarkRequest>,
    ) -> Result<tonic::Response<BenchmarkResponse>, tonic::Status> {
        let req = request.into_inner();
        info!(
            "Benchmark requested by validator: {} for type: {}",
            req.validator_hotkey, req.benchmark_type
        );

        let state = self.state.clone();
        let start_time = std::time::Instant::now();

        // Get system info once
        let system_info = state
            .system_monitor
            .get_system_info()
            .await
            .map_err(|e| tonic::Status::internal(format!("Failed to get system info: {e}")))?;

        let (score, metrics) = match req.benchmark_type.as_str() {
            "gpu" => {
                if system_info.gpu.is_empty() {
                    return Err(tonic::Status::failed_precondition("No GPU found"));
                }

                // Run a simple GPU compute benchmark
                let mut metrics = std::collections::HashMap::new();
                metrics.insert("gpu_count".to_string(), system_info.gpu.len().to_string());
                metrics.insert("gpu_model".to_string(), system_info.gpu[0].name.clone());
                let memory_mb = system_info.gpu[0].memory_total_bytes / (1024 * 1024);
                metrics.insert("gpu_memory_mb".to_string(), memory_mb.to_string());

                // Score based on GPU capabilities
                let score = (memory_mb as f64 / 1024.0) * 10.0; // Simple scoring based on memory

                (score, metrics)
            }
            "cpu" => {
                // CPU benchmark
                let mut metrics = std::collections::HashMap::new();
                metrics.insert("cpu_cores".to_string(), system_info.cpu.cores.to_string());
                metrics.insert("cpu_model".to_string(), system_info.cpu.model.clone());
                metrics.insert(
                    "cpu_usage_percent".to_string(),
                    system_info.cpu.usage_percent.to_string(),
                );

                // Simple CPU score based on cores
                let score = (system_info.cpu.cores as f64) * 100.0;

                (score, metrics)
            }
            "memory" => {
                // Memory benchmark
                let mut metrics = std::collections::HashMap::new();
                let total_mb = system_info.memory.total_bytes / (1024 * 1024);
                let available_mb = system_info.memory.available_bytes / (1024 * 1024);
                metrics.insert("total_memory_mb".to_string(), total_mb.to_string());
                metrics.insert("available_memory_mb".to_string(), available_mb.to_string());

                // Score based on total memory
                let score = total_mb as f64 / 1024.0; // GB

                (score, metrics)
            }
            "network" => {
                // Network benchmark
                let mut metrics = std::collections::HashMap::new();
                metrics.insert(
                    "network_interfaces".to_string(),
                    system_info.network.interfaces.len().to_string(),
                );

                // Simple network score
                let score = 100.0; // Placeholder

                (score, metrics)
            }
            "disk" => {
                // Disk benchmark
                let mut metrics = std::collections::HashMap::new();
                metrics.insert(
                    "total_disks".to_string(),
                    system_info.disk.len().to_string(),
                );
                if !system_info.disk.is_empty() {
                    let total_gb = system_info.disk[0].total_bytes / (1024 * 1024 * 1024);
                    metrics.insert("primary_disk_total_gb".to_string(), total_gb.to_string());
                }

                // Score based on total disk space
                let score = if !system_info.disk.is_empty() {
                    system_info.disk[0].total_bytes as f64 / (1024.0 * 1024.0 * 1024.0 * 1024.0)
                // TB
                } else {
                    0.0
                };

                (score, metrics)
            }
            _ => {
                return Err(tonic::Status::invalid_argument(format!(
                    "Unknown benchmark type: {}",
                    req.benchmark_type
                )));
            }
        };

        let duration_ms = start_time.elapsed().as_millis() as u64;

        Ok(tonic::Response::new(BenchmarkResponse {
            results: metrics,
            score,
            execution_details: format!(
                "Benchmark type: {}, Execution time: {}ms, Validator: {}",
                req.benchmark_type, duration_ms, req.validator_hotkey
            ),
            error: None,
        }))
    }

    async fn manage_container(
        &self,
        request: tonic::Request<ContainerOpRequest>,
    ) -> Result<tonic::Response<ContainerOpResponse>, tonic::Status> {
        let req = request.into_inner();
        info!("Container operation requested: {}", req.operation);

        let container_ops = ContainerOperationsService::new(self.state.clone());

        match req.operation.as_str() {
            "create" => {
                if let Some(spec) = req.container_spec {
                    let container_id = container_ops
                        .create_container(&spec.image, &spec.command)
                        .await
                        .map_err(|e| {
                            tonic::Status::internal(format!("Failed to create container: {e}"))
                        })?;

                    Ok(tonic::Response::new(ContainerOpResponse {
                        success: true,
                        container_id: container_id.clone(),
                        status: Some(protocol::ContainerStatus {
                            container_id: container_id.clone(),
                            status: "created".to_string(),
                            status_message: format!(
                                "Container {container_id} created successfully"
                            ),
                            created_at: Some(protocol::common::Timestamp {
                                value: Some(prost_types::Timestamp::from(
                                    std::time::SystemTime::now(),
                                )),
                            }),
                            started_at: None,
                            finished_at: None,
                            exit_code: 0,
                            resource_usage: None,
                        }),
                        details: "Container created successfully".to_string(),
                        error: None,
                    }))
                } else {
                    Err(tonic::Status::invalid_argument(
                        "Container spec required for create operation",
                    ))
                }
            }
            "delete" | "destroy" => {
                let force = req
                    .parameters
                    .get("force")
                    .map(|v| v == "true")
                    .unwrap_or(false);

                container_ops
                    .destroy_container(&req.container_id, force)
                    .await
                    .map_err(|e| {
                        tonic::Status::internal(format!("Failed to destroy container: {e}"))
                    })?;

                Ok(tonic::Response::new(ContainerOpResponse {
                    success: true,
                    container_id: req.container_id.clone(),
                    status: Some(protocol::ContainerStatus {
                        container_id: req.container_id.clone(),
                        status: "deleted".to_string(),
                        status_message: format!(
                            "Container {} deleted successfully",
                            req.container_id
                        ),
                        created_at: None,
                        started_at: None,
                        finished_at: Some(protocol::common::Timestamp {
                            value: Some(prost_types::Timestamp::from(std::time::SystemTime::now())),
                        }),
                        exit_code: 0,
                        resource_usage: None,
                    }),
                    details: "Container deleted successfully".to_string(),
                    error: None,
                }))
            }
            "get_status" => {
                let status = container_ops
                    .get_container_status(&req.container_id)
                    .await
                    .map_err(|e| {
                        tonic::Status::internal(format!("Failed to get container status: {e}"))
                    })?;

                Ok(tonic::Response::new(ContainerOpResponse {
                    success: true,
                    container_id: req.container_id.clone(),
                    status: Some(protocol::ContainerStatus {
                        container_id: req.container_id.clone(),
                        status,
                        status_message: format!("Container {} status retrieved", req.container_id),
                        created_at: None,
                        started_at: None,
                        finished_at: None,
                        exit_code: 0,
                        resource_usage: None,
                    }),
                    details: "Status retrieved successfully".to_string(),
                    error: None,
                }))
            }
            "add_key" => {
                // Handle SSH key addition through validation session
                let state = self.state.clone();
                // Create ValidatorId from hotkey
                let validator_id = crate::validation_session::types::ValidatorId::new(
                    req.validator_hotkey.clone(),
                );

                state
                    .validation_session
                    .grant_ssh_access(&validator_id, &req.ssh_public_key)
                    .await
                    .map_err(|e| tonic::Status::internal(format!("Failed to add SSH key: {e}")))?;

                Ok(tonic::Response::new(ContainerOpResponse {
                    success: true,
                    container_id: req.container_id.clone(),
                    status: Some(protocol::ContainerStatus {
                        container_id: req.container_id.clone(),
                        status: "key_added".to_string(),
                        status_message: format!(
                            "SSH key added for validator {}",
                            req.validator_hotkey
                        ),
                        created_at: None,
                        started_at: None,
                        finished_at: None,
                        exit_code: 0,
                        resource_usage: None,
                    }),
                    details: "SSH key added successfully".to_string(),
                    error: None,
                }))
            }
            _ => Err(tonic::Status::invalid_argument(format!(
                "Unknown operation: {}",
                req.operation
            ))),
        }
    }

    type StreamLogsStream = tokio_stream::wrappers::ReceiverStream<Result<LogEntry, tonic::Status>>;

    async fn stream_logs(
        &self,
        request: tonic::Request<LogSubscriptionRequest>,
    ) -> Result<tonic::Response<Self::StreamLogsStream>, tonic::Status> {
        let req = request.into_inner();
        info!(
            "Log streaming requested for container: {}",
            req.container_id
        );

        let state = self.state.clone();
        let (tx, rx) = tokio::sync::mpsc::channel(100);

        // Spawn a task to stream logs
        let container_manager = state.container_manager.clone();
        let container_id = req.container_id.clone();
        let follow = req.follow;
        let tail_lines = if req.tail_lines > 0 {
            Some(req.tail_lines as i32)
        } else {
            None
        };

        tokio::spawn(async move {
            match container_manager
                .stream_logs(&container_id, follow, tail_lines)
                .await
            {
                Ok(mut log_stream) => {
                    use futures_util::StreamExt;
                    while let Some(log_entry) = log_stream.next().await {
                        let proto_log_entry = LogEntry {
                            timestamp: Some(protocol::common::Timestamp {
                                value: Some(prost_types::Timestamp {
                                    seconds: log_entry.timestamp,
                                    nanos: 0,
                                }),
                            }),
                            level: match log_entry.level {
                                crate::container_manager::types::LogLevel::Info => {
                                    "INFO".to_string()
                                }
                                crate::container_manager::types::LogLevel::Warning => {
                                    "WARN".to_string()
                                }
                                crate::container_manager::types::LogLevel::Error => {
                                    "ERROR".to_string()
                                }
                                crate::container_manager::types::LogLevel::Debug => {
                                    "DEBUG".to_string()
                                }
                            },
                            source: "container".to_string(),
                            message: log_entry.message,
                            metadata: Default::default(),
                        };

                        if tx.send(Ok(proto_log_entry)).await.is_err() {
                            break; // Client disconnected
                        }
                    }
                }
                Err(e) => {
                    let _ = tx
                        .send(Err(tonic::Status::internal(format!(
                            "Failed to stream logs: {e}"
                        ))))
                        .await;
                }
            }
        });

        Ok(tonic::Response::new(ReceiverStream::new(rx)))
    }

    async fn health_check(
        &self,
        request: tonic::Request<HealthCheckRequest>,
    ) -> Result<tonic::Response<HealthCheckResponse>, tonic::Status> {
        let req = request.into_inner();
        info!("Health check requested by: {}", req.requester);

        match self.health_check.health_check().await {
            Ok(health_status) => {
                let mut resource_status = std::collections::HashMap::new();
                resource_status.insert(
                    "cpu_percent".to_string(),
                    health_status.details.cpu_usage_percent.to_string(),
                );
                resource_status.insert(
                    "memory_percent".to_string(),
                    health_status.details.memory_usage_percent.to_string(),
                );
                resource_status.insert(
                    "disk_percent".to_string(),
                    health_status.details.disk_usage_percent.to_string(),
                );

                Ok(tonic::Response::new(HealthCheckResponse {
                    status: health_status.status,
                    resource_status,
                    docker_status: "running".to_string(),
                    uptime_seconds: health_status.details.uptime_seconds,
                    last_update: Some(protocol::common::Timestamp {
                        value: Some(prost_types::Timestamp::from(std::time::SystemTime::now())),
                    }),
                    metrics: std::collections::HashMap::new(),
                }))
            }
            Err(e) => Err(tonic::Status::internal(e.to_string())),
        }
    }
}
