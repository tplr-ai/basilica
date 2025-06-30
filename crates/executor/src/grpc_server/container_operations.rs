//! Container operations service

use super::types::{GrpcResult, SharedExecutorState};
use tracing::info;

/// Container operations handler
pub struct ContainerOperationsService {
    state: SharedExecutorState,
}

impl ContainerOperationsService {
    /// Create new container operations service
    pub fn new(state: SharedExecutorState) -> Self {
        Self { state }
    }

    /// Create container
    pub async fn create_container(&self, image: &str, command: &[String]) -> GrpcResult<String> {
        info!("Creating container with image: {}", image);

        let state = self.state.clone();
        let container_id = state
            .container_manager
            .create_container(image, command, None)
            .await
            .map_err(|e| tonic::Status::internal(format!("Failed to create container: {e}")))?;

        Ok(container_id)
    }

    /// Execute command in container
    pub async fn execute_container_command(
        &self,
        container_id: &str,
        command: &str,
        timeout_secs: Option<u32>,
    ) -> GrpcResult<(i32, String, String)> {
        info!(
            "Executing command in container {}: {}",
            container_id, command
        );

        let state = self.state.clone();
        let result = state
            .container_manager
            .execute_command(container_id, command, timeout_secs.map(|s| s as u64))
            .await
            .map_err(|e| tonic::Status::internal(format!("Failed to execute command: {e}")))?;

        Ok((result.exit_code, result.stdout, result.stderr))
    }

    /// Destroy container
    pub async fn destroy_container(&self, container_id: &str, force: bool) -> GrpcResult<()> {
        info!("Destroying container: {} (force: {})", container_id, force);

        let state = self.state.clone();
        state
            .container_manager
            .destroy_container(container_id, force)
            .await
            .map_err(|e| tonic::Status::internal(format!("Failed to destroy container: {e}")))?;

        Ok(())
    }

    /// List containers
    pub async fn list_containers(&self) -> GrpcResult<Vec<String>> {
        info!("Listing containers");

        let state = self.state.clone();
        let containers = state
            .container_manager
            .list_containers()
            .await
            .map_err(|e| tonic::Status::internal(format!("Failed to list containers: {e}")))?;

        Ok(containers.into_iter().map(|c| c.id).collect())
    }

    /// Get container status
    pub async fn get_container_status(&self, container_id: &str) -> GrpcResult<String> {
        info!("Getting status for container: {}", container_id);

        let state = self.state.clone();
        let status = state
            .container_manager
            .get_container_status(container_id)
            .await
            .map_err(|e| tonic::Status::internal(format!("Failed to get container status: {e}")))?;

        match status {
            Some(s) => Ok(s.state),
            None => Err(anyhow::anyhow!("Container {} not found", container_id)),
        }
    }

    /// Stream container logs with real-time streaming capability
    pub async fn stream_container_logs(
        &self,
        container_id: &str,
        follow: bool,
        tail_lines: Option<u32>,
    ) -> GrpcResult<impl tokio_stream::Stream<Item = Result<String, anyhow::Error>>> {
        info!(
            "Streaming logs for container: {} (follow: {}, tail: {:?})",
            container_id, follow, tail_lines
        );

        let state = self.state.clone();
        let log_stream = state
            .container_manager
            .stream_logs(container_id, follow, tail_lines.map(|n| n as i32))
            .await
            .map_err(|e| tonic::Status::internal(format!("Failed to create log stream: {e}")))?;

        // Convert container log entries to formatted strings with enhanced metadata
        let formatted_stream = tokio_stream::StreamExt::map(log_stream, |log_entry| {
            let formatted_log = format!(
                "[{}] [{}] [{}] {}",
                chrono::DateTime::<chrono::Utc>::from_timestamp(log_entry.timestamp, 0)
                    .unwrap_or_else(chrono::Utc::now)
                    .format("%Y-%m-%d %H:%M:%S%.3f UTC"),
                match log_entry.level {
                    crate::container_manager::types::LogLevel::Info => "INFO",
                    crate::container_manager::types::LogLevel::Warning => "WARN",
                    crate::container_manager::types::LogLevel::Error => "ERROR",
                    crate::container_manager::types::LogLevel::Debug => "DEBUG",
                },
                log_entry.container_id,
                log_entry.message
            );
            Ok(formatted_log)
        });

        Ok(formatted_stream)
    }

    /// Get container resource usage
    pub async fn get_container_resources(&self, container_id: &str) -> GrpcResult<String> {
        info!("Getting resource usage for container: {}", container_id);

        let state = self.state.clone();
        let stats = state
            .container_manager
            .get_container_stats(container_id)
            .await
            .map_err(|e| tonic::Status::internal(format!("Failed to get container stats: {e}")))?;

        match stats {
            Some(usage) => {
                let json_stats = serde_json::json!({
                    "container_id": container_id,
                    "cpu_usage_percent": usage.cpu_usage_percent,
                    "memory_usage_bytes": usage.memory_usage_bytes,
                    "memory_limit_bytes": usage.memory_limit_bytes,
                    "network_io_bytes": usage.network_io_bytes,
                    "block_io_bytes": usage.block_io_bytes,
                    "memory_usage_mb": usage.memory_usage_bytes / (1024 * 1024),
                    "timestamp": chrono::Utc::now().timestamp()
                });
                Ok(json_stats.to_string())
            }
            None => Err(anyhow::anyhow!(
                "Container {} not found or not running",
                container_id
            )),
        }
    }

    /// Get comprehensive container environment analysis
    pub async fn get_container_environment_analysis(
        &self,
        container_id: &str,
    ) -> GrpcResult<String> {
        info!("Analyzing container environment for: {}", container_id);

        let state = self.state.clone();

        // Get container status and stats
        let status = state
            .container_manager
            .get_container_status(container_id)
            .await
            .map_err(|e| tonic::Status::internal(format!("Failed to get container status: {e}")))?;

        let stats = state
            .container_manager
            .get_container_stats(container_id)
            .await
            .map_err(|e| tonic::Status::internal(format!("Failed to get container stats: {e}")))?;

        // Get system information for context
        let system_info = state
            .system_monitor
            .get_system_info()
            .await
            .map_err(|e| tonic::Status::internal(format!("Failed to get system info: {e}")))?;

        match (status, stats) {
            (Some(container_status), Some(resource_stats)) => {
                let analysis = serde_json::json!({
                    "container_analysis": {
                        "container_id": container_id,
                        "name": container_status.name,
                        "image": container_status.image,
                        "state": container_status.state,
                        "uptime_seconds": container_status.started
                            .map(|started| chrono::Utc::now().timestamp() - started)
                            .unwrap_or(0)
                    },
                    "resource_profile": {
                        "cpu_intensive": resource_stats.cpu_usage_percent > 50.0,
                        "memory_intensive": (resource_stats.memory_usage_bytes as f64 /
                            resource_stats.memory_limit_bytes.max(1) as f64) > 0.5,
                        "io_intensive": resource_stats.block_io_bytes > (100 * 1024 * 1024),
                        "network_intensive": resource_stats.network_io_bytes > (50 * 1024 * 1024)
                    },
                    "system_context": {
                        "host_cpu_cores": system_info.cpu.cores,
                        "host_memory_total_gb": system_info.memory.total_bytes / (1024 * 1024 * 1024),
                        "container_cpu_share": resource_stats.cpu_usage_percent / 100.0 * system_info.cpu.cores as f64,
                        "container_memory_share": (resource_stats.memory_usage_bytes as f64 /
                            system_info.memory.total_bytes as f64) * 100.0
                    },
                    "performance_recommendations": {
                        "cpu_optimization": if resource_stats.cpu_usage_percent > 80.0 {
                            "Consider CPU limit adjustment or code optimization"
                        } else if resource_stats.cpu_usage_percent < 10.0 {
                            "Container appears CPU underutilized"
                        } else {
                            "CPU usage within normal range"
                        },
                        "memory_optimization": get_memory_optimization_advice(
                            resource_stats.memory_usage_bytes,
                            resource_stats.memory_limit_bytes
                        ),
                        "scaling_advice": get_scaling_advice(
                            resource_stats.cpu_usage_percent,
                            resource_stats.memory_usage_bytes,
                            resource_stats.memory_limit_bytes
                        )
                    },
                    "analysis_timestamp": chrono::Utc::now().to_rfc3339()
                });

                Ok(analysis.to_string())
            }
            _ => Err(anyhow::anyhow!(
                "Container {} not found or insufficient data for analysis",
                container_id
            )),
        }
    }

    /// Get historical resource usage trends for a container
    pub async fn get_container_resource_history(
        &self,
        container_id: &str,
        duration_minutes: Option<u32>,
    ) -> GrpcResult<String> {
        info!(
            "Getting resource usage history for container: {} over {} minutes",
            container_id,
            duration_minutes.unwrap_or(60)
        );

        // For now, simulate historical data with current stats
        // In production, this would query a time-series database
        let current_stats = self.get_container_resources(container_id).await?;
        let current_data: serde_json::Value = serde_json::from_str(&current_stats)
            .map_err(|e| tonic::Status::internal(format!("Failed to parse current stats: {e}")))?;

        let duration = duration_minutes.unwrap_or(60);
        let mut historical_data = Vec::new();
        let now = chrono::Utc::now();

        // Generate sample historical data points (in production, this would be real data)
        for i in 0..std::cmp::min(duration, 60) {
            let timestamp = now - chrono::Duration::minutes(i as i64);
            let variance = (i as f64 * 0.1) % 10.0; // Simulate variance

            let historical_point = serde_json::json!({
                "timestamp": timestamp.to_rfc3339(),
                "cpu_usage_percent": current_data["resource_usage"]["cpu"]["usage_percent"]
                    .as_f64().unwrap_or(0.0) + variance - 5.0,
                "memory_usage_percent": current_data["resource_usage"]["memory"]["usage_percent"]
                    .as_f64().unwrap_or(0.0) + variance - 3.0,
                "network_io_rate_mbps": current_data["resource_usage"]["network"]["avg_throughput_mbps"]
                    .as_f64().unwrap_or(0.0) + variance,
                "disk_io_rate_mbps": current_data["resource_usage"]["disk"]["avg_throughput_mbps"]
                    .as_f64().unwrap_or(0.0) + variance
            });
            historical_data.push(historical_point);
        }

        let history_response = serde_json::json!({
            "container_id": container_id,
            "duration_minutes": duration,
            "data_points": historical_data.len(),
            "current_snapshot": current_data,
            "historical_data": historical_data,
            "trends": {
                "cpu_trend": "stable", // In production: analyze actual trends
                "memory_trend": "stable",
                "network_trend": "stable",
                "disk_trend": "stable"
            },
            "generated_at": now.to_rfc3339()
        });

        Ok(history_response.to_string())
    }

    /// Container operations health check
    pub async fn container_operations_health_check(&self) -> GrpcResult<()> {
        info!("Container operations health check");

        // Check if Docker is available
        self.state
            .container_manager
            .health_check()
            .await
            .map_err(|e| tonic::Status::internal(format!("Docker health check failed: {e}")))?;

        Ok(())
    }
}

/// Helper function to get memory optimization advice
fn get_memory_optimization_advice(
    memory_usage_bytes: u64,
    memory_limit_bytes: u64,
) -> &'static str {
    let memory_percent = (memory_usage_bytes as f64 / memory_limit_bytes.max(1) as f64) * 100.0;
    if memory_percent > 90.0 {
        "Critical: Memory usage very high, consider increasing limit"
    } else if memory_percent > 75.0 {
        "Warning: High memory usage detected"
    } else if memory_percent < 25.0 {
        "Memory limit may be over-provisioned"
    } else {
        "Memory usage within optimal range"
    }
}

/// Helper function to get scaling advice
fn get_scaling_advice(
    cpu_usage_percent: f64,
    memory_usage_bytes: u64,
    memory_limit_bytes: u64,
) -> &'static str {
    let cpu_high = cpu_usage_percent > 70.0;
    let memory_high = (memory_usage_bytes as f64 / memory_limit_bytes.max(1) as f64) > 0.7;

    if cpu_high && memory_high {
        "Consider vertical scaling (more CPU and memory)"
    } else if cpu_high {
        "Consider increasing CPU allocation"
    } else if memory_high {
        "Consider increasing memory allocation"
    } else {
        "Current resource allocation appears sufficient"
    }
}
