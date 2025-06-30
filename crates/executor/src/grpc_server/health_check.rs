//! Health check service

use super::types::{GrpcResult, SharedExecutorState};
use tracing::{info, warn};

/// Health check handler
pub struct HealthCheckService {
    state: SharedExecutorState,
}

impl HealthCheckService {
    /// Create new health check service
    pub fn new(state: SharedExecutorState) -> Self {
        Self { state }
    }

    /// Perform comprehensive health check
    pub async fn health_check(&self) -> GrpcResult<HealthStatus> {
        info!("Health check requested");

        let health_status = match self.state.health_check().await {
            Ok(_) => HealthStatus {
                status: "healthy".to_string(),
                timestamp: chrono::Utc::now().timestamp(),
                executor_id: self.state.id.to_string(),
                details: self.get_health_details().await?,
            },
            Err(e) => {
                warn!("Health check failed: {}", e);
                HealthStatus {
                    status: format!("unhealthy: {e}"),
                    timestamp: chrono::Utc::now().timestamp(),
                    executor_id: self.state.id.to_string(),
                    details: HealthDetails::default(),
                }
            }
        };

        Ok(health_status)
    }

    /// Get detailed health information
    async fn get_health_details(&self) -> GrpcResult<HealthDetails> {
        let system_info = self.state.system_monitor.get_system_info().await?;

        // Get container count
        let container_count = match self.state.container_manager.list_containers().await {
            Ok(containers) => containers.len() as u32,
            Err(_) => 0,
        };

        // Get active challenges from state (if tracking)
        let active_challenges = self
            .state
            .active_challenges
            .load(std::sync::atomic::Ordering::Relaxed);

        Ok(HealthDetails {
            cpu_usage_percent: system_info.cpu.usage_percent,
            memory_usage_percent: system_info.memory.usage_percent,
            disk_usage_percent: system_info
                .disk
                .first()
                .map(|d| d.usage_percent)
                .unwrap_or(0.0),
            container_count,
            active_challenges,
            uptime_seconds: system_info.system.uptime_seconds,
        })
    }

    /// Quick health check (lighter version)
    pub async fn quick_health_check(&self) -> GrpcResult<bool> {
        info!("Quick health check requested");

        // Basic checks without full system scan
        match self.state.container_manager.health_check().await {
            Ok(_) => Ok(true),
            Err(e) => {
                warn!("Quick health check failed: {}", e);
                Ok(false)
            }
        }
    }

    /// Get service readiness
    pub async fn readiness_check(&self) -> GrpcResult<bool> {
        info!("Readiness check requested");

        // Check if all services are ready to handle requests
        let container_ready = self.state.container_manager.health_check().await.is_ok();
        let monitor_ready = self.state.system_monitor.health_check().await.is_ok();

        Ok(container_ready && monitor_ready)
    }

    /// Get service liveness
    pub async fn liveness_check(&self) -> GrpcResult<bool> {
        info!("Liveness check requested");

        // Basic check to ensure service is alive
        Ok(true)
    }
}

/// Health status response
#[derive(Debug)]
pub struct HealthStatus {
    pub status: String,
    pub timestamp: i64,
    pub executor_id: String,
    pub details: HealthDetails,
}

/// Detailed health information
#[derive(Debug, Default)]
pub struct HealthDetails {
    pub cpu_usage_percent: f32,
    pub memory_usage_percent: f32,
    pub disk_usage_percent: f32,
    pub container_count: u32,
    pub active_challenges: u32,
    pub uptime_seconds: u64,
}
