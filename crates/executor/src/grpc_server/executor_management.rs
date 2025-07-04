//! ExecutorManagement service implementation for miner-executor communication

use super::types::SharedExecutorState;
use chrono::{DateTime, Utc};
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
            "add" => self.add_validator_ssh_key(&req).await,
            "remove" => self.remove_validator_ssh_key(&req).await,
            "cleanup_expired" => self.cleanup_expired_ssh_keys(&req).await,
            "list_sessions" => self.list_validator_ssh_sessions(&req).await,
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

impl ExecutorManagementService {
    /// Add validator SSH key with session management
    async fn add_validator_ssh_key(
        &self,
        req: &SshKeyUpdate,
    ) -> Result<Response<SshKeyUpdateResponse>, Status> {
        info!("Adding SSH key for validator {}", req.validator_hotkey);

        // Create ValidatorId from hotkey
        let validator_id =
            crate::validation_session::types::ValidatorId::new(req.validator_hotkey.clone());

        // Set default expiration time (24 hours from now)
        let expires_at = Some(Utc::now() + chrono::Duration::hours(24));

        // Add key with session metadata
        self.add_key_with_session_metadata(&validator_id, &req.ssh_public_key, None, expires_at)
            .await
            .map_err(|e| Status::internal(format!("Failed to add SSH key: {e}")))?;

        // Log audit event
        info!(
            target: "ssh_audit",
            executor_id = %self.state.id,
            validator_hotkey = %req.validator_hotkey,
            action = "ssh_key_added",
            success = true,
            "SSH key added for validator"
        );

        Ok(Response::new(SshKeyUpdateResponse {
            success: true,
            message: format!("SSH key added for validator {}", req.validator_hotkey),
            error: None,
        }))
    }

    /// Remove validator SSH key with session management
    async fn remove_validator_ssh_key(
        &self,
        req: &SshKeyUpdate,
    ) -> Result<Response<SshKeyUpdateResponse>, Status> {
        info!("Removing SSH key for validator {}", req.validator_hotkey);

        // Create ValidatorId from hotkey
        let validator_id =
            crate::validation_session::types::ValidatorId::new(req.validator_hotkey.clone());

        // Remove all keys for validator
        self.state
            .validation_session
            .revoke_ssh_access(&validator_id)
            .await
            .map_err(|e| Status::internal(format!("Failed to remove SSH key: {e}")))?;

        // Log audit event
        info!(
            target: "ssh_audit",
            executor_id = %self.state.id,
            validator_hotkey = %req.validator_hotkey,
            action = "ssh_key_removed",
            success = true,
            "SSH key removed for validator"
        );

        Ok(Response::new(SshKeyUpdateResponse {
            success: true,
            message: format!("SSH key removed for validator {}", req.validator_hotkey),
            error: None,
        }))
    }

    /// Cleanup expired SSH keys
    async fn cleanup_expired_ssh_keys(
        &self,
        _req: &SshKeyUpdate,
    ) -> Result<Response<SshKeyUpdateResponse>, Status> {
        info!(
            "Cleaning up expired SSH keys for executor {}",
            self.state.id
        );

        let cleaned_count = self
            .cleanup_expired_keys()
            .await
            .map_err(|e| Status::internal(format!("Failed to cleanup expired keys: {e}")))?;

        // Log audit event
        info!(
            target: "ssh_audit",
            executor_id = %self.state.id,
            action = "ssh_keys_cleanup",
            success = true,
            cleaned_count = %cleaned_count,
            "Expired SSH keys cleaned up"
        );

        Ok(Response::new(SshKeyUpdateResponse {
            success: true,
            message: format!("Cleaned up {cleaned_count} expired SSH keys"),
            error: None,
        }))
    }

    /// List active validator SSH sessions
    async fn list_validator_ssh_sessions(
        &self,
        _req: &SshKeyUpdate,
    ) -> Result<Response<SshKeyUpdateResponse>, Status> {
        let sessions = self
            .list_active_ssh_sessions()
            .await
            .map_err(|e| Status::internal(format!("Failed to list SSH sessions: {e}")))?;

        let session_info = sessions
            .iter()
            .map(|s| {
                format!(
                    "{}:{}:{}",
                    s.validator_hotkey,
                    s.session_id.as_ref().unwrap_or(&"unknown".to_string()),
                    s.expires_at
                        .map(|e| e.to_rfc3339())
                        .unwrap_or("never".to_string())
                )
            })
            .collect::<Vec<_>>()
            .join(",");

        Ok(Response::new(SshKeyUpdateResponse {
            success: true,
            message: format!("Active SSH sessions: [{session_info}]"),
            error: None,
        }))
    }

    /// Add SSH key with session metadata to authorized_keys
    async fn add_key_with_session_metadata(
        &self,
        validator_id: &crate::validation_session::types::ValidatorId,
        public_key: &str,
        session_id: Option<&String>,
        expires_at: Option<DateTime<Utc>>,
    ) -> anyhow::Result<()> {
        use std::fs::OpenOptions;
        use std::io::Write;

        // Ensure SSH directory exists
        let ssh_dir =
            std::env::var("HOME").unwrap_or_else(|_| "/home/executor".to_string()) + "/.ssh";
        std::fs::create_dir_all(&ssh_dir)?;

        let authorized_keys_path = format!("{ssh_dir}/authorized_keys");

        // Create authorized_keys entry with session metadata
        let key_entry = if let Some(session_id) = session_id {
            format!(
                "{} validator-session-{} validator={} executor={} expires={} miner-managed\n",
                public_key.trim(),
                session_id,
                validator_id.hotkey,
                self.state.id,
                expires_at
                    .map(|e| e.to_rfc3339())
                    .unwrap_or("never".to_string())
            )
        } else {
            format!(
                "{} validator={} executor={} legacy\n",
                public_key.trim(),
                validator_id.hotkey,
                self.state.id
            )
        };

        // Append to authorized_keys file
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&authorized_keys_path)?;
        file.write_all(key_entry.as_bytes())?;
        file.flush()?;

        // Set proper permissions
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&ssh_dir, std::fs::Permissions::from_mode(0o700))?;
            std::fs::set_permissions(
                &authorized_keys_path,
                std::fs::Permissions::from_mode(0o600),
            )?;
        }

        info!(
            "Added SSH key for validator {} to {}",
            validator_id.hotkey, authorized_keys_path
        );

        Ok(())
    }

    /// Remove SSH key by session ID
    async fn remove_key_by_session(
        &self,
        validator_id: &crate::validation_session::types::ValidatorId,
        session_id: &str,
    ) -> anyhow::Result<()> {
        let ssh_dir =
            std::env::var("HOME").unwrap_or_else(|_| "/home/executor".to_string()) + "/.ssh";
        let authorized_keys_path = format!("{ssh_dir}/authorized_keys");

        if !std::path::Path::new(&authorized_keys_path).exists() {
            return Ok(());
        }

        let content = std::fs::read_to_string(&authorized_keys_path)?;
        let filtered_content: String = content
            .lines()
            .filter(|line| !line.contains(&format!("validator-session-{session_id}")))
            .map(|line| format!("{line}\n"))
            .collect();

        std::fs::write(&authorized_keys_path, filtered_content)?;

        info!(
            "Removed SSH key for validator {} from {}",
            validator_id.hotkey, authorized_keys_path
        );

        Ok(())
    }

    /// Cleanup expired SSH keys from authorized_keys
    async fn cleanup_expired_keys(&self) -> anyhow::Result<usize> {
        let ssh_dir =
            std::env::var("HOME").unwrap_or_else(|_| "/home/executor".to_string()) + "/.ssh";
        let authorized_keys_path = format!("{ssh_dir}/authorized_keys");

        if !std::path::Path::new(&authorized_keys_path).exists() {
            return Ok(0);
        }

        let content = std::fs::read_to_string(&authorized_keys_path)?;
        let now = Utc::now();
        let mut removed_count = 0;

        let filtered_content: String = content
            .lines()
            .filter_map(|line| {
                if line.contains("expires=") && line.contains("validator-session-") {
                    // Extract expiration time
                    if let Some(expires_start) = line.find("expires=") {
                        let expires_str = &line[expires_start + 8..];
                        if let Some(expires_end) = expires_str.find(' ') {
                            let expires_str = &expires_str[..expires_end];

                            if let Ok(expires_at) = DateTime::parse_from_rfc3339(expires_str) {
                                if expires_at.with_timezone(&Utc) < now {
                                    removed_count += 1;
                                    return None; // Filter out expired key
                                }
                            }
                        }
                    }
                }
                Some(format!("{line}\n"))
            })
            .collect();

        if removed_count > 0 {
            std::fs::write(&authorized_keys_path, filtered_content)?;
            info!(
                "Removed {} expired SSH keys from {}",
                removed_count, authorized_keys_path
            );
        }

        Ok(removed_count)
    }

    /// List active SSH sessions from authorized_keys
    async fn list_active_ssh_sessions(&self) -> anyhow::Result<Vec<SshSessionInfo>> {
        let ssh_dir =
            std::env::var("HOME").unwrap_or_else(|_| "/home/executor".to_string()) + "/.ssh";
        let authorized_keys_path = format!("{ssh_dir}/authorized_keys");

        if !std::path::Path::new(&authorized_keys_path).exists() {
            return Ok(Vec::new());
        }

        let content = std::fs::read_to_string(&authorized_keys_path)?;
        let now = Utc::now();
        let mut sessions = Vec::new();

        for line in content.lines() {
            if line.contains("validator=") {
                let mut session_info = SshSessionInfo {
                    validator_hotkey: String::new(),
                    session_id: None,
                    expires_at: None,
                    is_active: true,
                };

                // Extract validator hotkey
                if let Some(validator_start) = line.find("validator=") {
                    let validator_part = &line[validator_start + 10..];
                    if let Some(validator_end) = validator_part.find(' ') {
                        session_info.validator_hotkey = validator_part[..validator_end].to_string();
                    }
                }

                // Extract session ID if present
                if let Some(session_start) = line.find("validator-session-") {
                    let session_part = &line[session_start + 18..];
                    if let Some(session_end) = session_part.find(' ') {
                        session_info.session_id = Some(session_part[..session_end].to_string());
                    }
                }

                // Extract expiration time if present
                if let Some(expires_start) = line.find("expires=") {
                    let expires_str = &line[expires_start + 8..];
                    if let Some(expires_end) = expires_str.find(' ') {
                        let expires_str = &expires_str[..expires_end];

                        if expires_str != "never" {
                            if let Ok(expires_at) = DateTime::parse_from_rfc3339(expires_str) {
                                let expires_utc = expires_at.with_timezone(&Utc);
                                session_info.expires_at = Some(expires_utc);
                                session_info.is_active = expires_utc > now;
                            }
                        }
                    }
                }

                if !session_info.validator_hotkey.is_empty() {
                    sessions.push(session_info);
                }
            }
        }

        Ok(sessions)
    }
}

/// SSH session information
#[derive(Debug, Clone)]
struct SshSessionInfo {
    validator_hotkey: String,
    session_id: Option<String>,
    expires_at: Option<DateTime<Utc>>,
    is_active: bool,
}
