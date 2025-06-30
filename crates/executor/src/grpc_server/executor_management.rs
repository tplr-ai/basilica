//! ExecutorManagement service implementation for miner-executor communication

use super::types::SharedExecutorState;
use protocol::executor_management::{
    executor_management_server::ExecutorManagement, HealthCheckRequest, HealthCheckResponse,
    SshKeyUpdate, SshKeyUpdateResponse, StatusRequest, StatusResponse,
};
use std::collections::HashMap;
use tonic::{Request, Response, Status};
use tracing::info;

/// ExecutorManagement service implementation
pub struct ExecutorManagementService {
    state: SharedExecutorState,
}

impl ExecutorManagementService {
    pub fn new(state: SharedExecutorState) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl ExecutorManagement for ExecutorManagementService {
    async fn health_check(
        &self,
        request: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        let req = request.into_inner();
        info!("Health check requested by miner: {}", req.miner_hotkey);

        // Verify miner is our configured miner
        if req.miner_hotkey != self.state.config.managing_miner_hotkey.to_string() {
            return Err(Status::permission_denied("Not authorized miner"));
        }

        let mut resource_status = HashMap::new();

        // Get system resource status
        let system_info = self
            .state
            .system_monitor
            .get_system_info()
            .await
            .map_err(|e| Status::internal(format!("Failed to get system info: {e}")))?;

        resource_status.insert(
            "cpu_usage".to_string(),
            system_info.cpu.usage_percent.to_string(),
        );
        let used_mb = system_info.memory.used_bytes / (1024 * 1024);
        resource_status.insert("memory_mb".to_string(), used_mb.to_string());
        resource_status.insert("disk_read_bytes".to_string(), "0".to_string());
        resource_status.insert("disk_write_bytes".to_string(), "0".to_string());
        resource_status.insert("network_rx_bytes".to_string(), "0".to_string());
        resource_status.insert("network_tx_bytes".to_string(), "0".to_string());

        // Get GPU info if available
        if !system_info.gpu.is_empty() {
            let gpu_utils: Vec<String> = system_info
                .gpu
                .iter()
                .map(|g| g.utilization_percent.to_string())
                .collect();
            let gpu_mems: Vec<String> = system_info
                .gpu
                .iter()
                .map(|g| (g.memory_used_bytes / (1024 * 1024)).to_string())
                .collect();

            resource_status.insert("gpu_utilization".to_string(), gpu_utils.join(","));
            resource_status.insert("gpu_memory_mb".to_string(), gpu_mems.join(","));
        }

        // Check Docker status
        let docker_status = match self.state.container_manager.health_check().await {
            Ok(_) => "healthy".to_string(),
            Err(e) => format!("unhealthy: {e}"),
        };

        // Get container count
        let container_count = match self.state.container_manager.list_containers().await {
            Ok(containers) => containers.len() as u32,
            Err(_) => 0,
        };

        let active_challenges = self
            .state
            .active_challenges
            .load(std::sync::atomic::Ordering::Relaxed);

        Ok(Response::new(HealthCheckResponse {
            status: "healthy".to_string(),
            resource_status,
            docker_status,
            uptime_seconds: system_info.system.uptime_seconds,
            container_count,
            active_challenges,
        }))
    }

    async fn update_ssh_keys(
        &self,
        request: Request<SshKeyUpdate>,
    ) -> Result<Response<SshKeyUpdateResponse>, Status> {
        let req = request.into_inner();
        info!(
            "SSH key update requested by miner: {} for validator: {}",
            req.miner_hotkey, req.validator_hotkey
        );

        // Verify miner is our configured miner
        if req.miner_hotkey != self.state.config.managing_miner_hotkey.to_string() {
            return Err(Status::permission_denied("Not authorized miner"));
        }

        // Perform the SSH key operation
        match req.operation.as_str() {
            "add" => {
                // Create ValidatorId from hotkey
                let validator_id = crate::validation_session::types::ValidatorId::new(
                    req.validator_hotkey.clone(),
                );

                self.state
                    .validation_session
                    .grant_ssh_access(&validator_id, &req.ssh_public_key)
                    .await
                    .map_err(|e| Status::internal(format!("Failed to add SSH key: {e}")))?;

                Ok(Response::new(SshKeyUpdateResponse {
                    success: true,
                    message: format!("SSH key added for validator {}", req.validator_hotkey),
                    error: None,
                }))
            }
            "remove" => {
                // Create ValidatorId from hotkey
                let validator_id = crate::validation_session::types::ValidatorId::new(
                    req.validator_hotkey.clone(),
                );

                self.state
                    .validation_session
                    .revoke_ssh_access(&validator_id)
                    .await
                    .map_err(|e| Status::internal(format!("Failed to remove SSH key: {e}")))?;

                Ok(Response::new(SshKeyUpdateResponse {
                    success: true,
                    message: format!("SSH key removed for validator {}", req.validator_hotkey),
                    error: None,
                }))
            }
            _ => Err(Status::invalid_argument(format!(
                "Unknown operation: {}",
                req.operation
            ))),
        }
    }

    async fn get_status(
        &self,
        request: Request<StatusRequest>,
    ) -> Result<Response<StatusResponse>, Status> {
        let req = request.into_inner();
        info!("Status requested with detailed={}", req.detailed);

        let system_info = self
            .state
            .system_monitor
            .get_system_info()
            .await
            .map_err(|e| Status::internal(format!("Failed to get system info: {e}")))?;

        // Build machine info
        let machine_info = protocol::common::MachineInfo {
            gpus: system_info
                .gpu
                .iter()
                .map(|g| protocol::common::GpuSpec {
                    model: g.name.clone(),
                    memory_mb: g.memory_total_bytes / (1024 * 1024),
                    uuid: String::new(), // Not tracked in system monitor
                    driver_version: g.driver_version.clone(),
                    cuda_version: g.cuda_version.clone().unwrap_or_default(),
                    utilization_percent: g.utilization_percent as f64,
                    memory_utilization_percent: g.memory_usage_percent as f64,
                    temperature_celsius: g.temperature_celsius as f64,
                    power_watts: g.power_usage_watts as f64,
                    core_clock_mhz: 0,
                    memory_clock_mhz: 0,
                    compute_capability: String::new(),
                })
                .collect(),
            cpu: Some(protocol::common::CpuSpec {
                model: system_info.cpu.model.clone(),
                physical_cores: system_info.cpu.cores as u32,
                logical_cores: system_info.cpu.cores as u32,
                base_frequency_mhz: system_info.cpu.frequency_mhz as u32,
                max_frequency_mhz: system_info.cpu.frequency_mhz as u32,
                vendor: system_info.cpu.vendor.clone(),
                architecture: String::new(), // Not tracked in BasicSystemInfo
                l1_cache_kb: 0,
                l2_cache_kb: 0,
                l3_cache_kb: 0,
                utilization_percent: system_info.cpu.usage_percent as f64,
                temperature_celsius: system_info.cpu.temperature_celsius.unwrap_or(0.0) as f64,
            }),
            memory: Some(protocol::common::MemorySpec {
                total_mb: system_info.memory.total_bytes / (1024 * 1024),
                available_mb: system_info.memory.available_bytes / (1024 * 1024),
                used_mb: system_info.memory.used_bytes / (1024 * 1024),
                speed_mhz: 0,
                memory_type: "Unknown".to_string(),
            }),
            fingerprint: format!("executor-{}", self.state.id),
            os_info: Some(protocol::common::OsInfo {
                name: system_info.system.os_name.clone(),
                version: system_info.system.os_version.clone(),
                kernel_version: system_info.system.kernel_version.clone(),
                distribution: system_info.system.os_name.clone(),
                architecture: String::new(), // Not tracked in BasicSystemInfo
                uptime_seconds: system_info.system.uptime_seconds,
                hostname: system_info.system.hostname.clone(),
            }),
        };

        // Get container statistics
        let containers = self
            .state
            .container_manager
            .list_containers()
            .await
            .unwrap_or_default();
        let active_containers = containers.iter().filter(|c| c.state == "running").count() as u32;

        // Get validator count from validation session
        let active_validators = self
            .state
            .validation_session
            .list_active_access()
            .await
            .unwrap_or_default()
            .len() as u32;

        // Build OS info
        let _os_info = protocol::common::OsInfo {
            name: system_info.system.os_name.clone(),
            version: system_info.system.os_version.clone(),
            kernel_version: system_info.system.kernel_version.clone(),
            distribution: system_info.system.os_name.clone(),
            architecture: String::new(), // Not tracked in BasicSystemInfo
            uptime_seconds: system_info.system.uptime_seconds,
            hostname: system_info.system.hostname.clone(),
        };

        Ok(Response::new(StatusResponse {
            executor_id: self.state.id.to_string(),
            status: "operational".to_string(),
            machine_info: Some(machine_info),
            resource_usage: None, // Could be populated if needed
            total_containers: containers.len() as u32,
            active_containers,
            active_validators,
            uptime_seconds: system_info.system.uptime_seconds,
        }))
    }
}
