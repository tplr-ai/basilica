//! Container operations and lifecycle management

use super::config_builder::ContainerConfigBuilder;
use super::types::{ContainerExecutionResult, ContainerResourceUsage, ContainerStatus};
use crate::config::{ContainerResourceLimits, DockerConfig};
use anyhow::Result;
use bollard::{
    container::{
        CreateContainerOptions, KillContainerOptions, RemoveContainerOptions,
        RestartContainerOptions, StartContainerOptions, StatsOptions, StopContainerOptions,
        WaitContainerOptions,
    },
    exec::{CreateExecOptions, StartExecResults},
    image::CreateImageOptions,
    Docker,
};
use futures_util::stream::StreamExt;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

#[derive(Debug, Clone)]
pub struct ContainerOperations {
    docker: Docker,
    config: DockerConfig,
    active_containers: Arc<RwLock<HashMap<String, ContainerStatus>>>,
    config_builder: ContainerConfigBuilder,
    lifecycle: ContainerLifecycle,
}

impl ContainerOperations {
    pub fn new(
        docker: Docker,
        config: DockerConfig,
        active_containers: Arc<RwLock<HashMap<String, ContainerStatus>>>,
    ) -> Self {
        let config_builder = ContainerConfigBuilder::new(config.clone());
        let lifecycle = ContainerLifecycle::new(docker.clone(), active_containers.clone());

        Self {
            docker,
            config,
            active_containers,
            config_builder,
            lifecycle,
        }
    }

    pub async fn create_container(
        &self,
        image: &str,
        command: &[String],
        resource_limits: Option<ContainerResourceLimits>,
    ) -> Result<String> {
        info!(
            "Creating container with image: {} and command: {:?}",
            image, command
        );

        self.ensure_image_available(image).await?;

        let uuid_str = uuid::Uuid::new_v4().to_string();
        let container_name = format!("basilca-{}", &uuid_str[..8]);

        let limits = resource_limits.unwrap_or_else(|| self.config.resource_limits.clone());
        let container_config = self.config_builder.build(image, command, &limits)?;

        let create_options = CreateContainerOptions {
            name: container_name.clone(),
            platform: None,
        };

        let container = self
            .docker
            .create_container(Some(create_options), container_config)
            .await?;

        let container_id = container.id;
        info!("Created container: {} ({})", container_name, container_id);

        self.docker
            .start_container(&container_id, None::<StartContainerOptions<String>>)
            .await?;

        info!("Started container: {}", container_id);

        self.lifecycle
            .track_container(&container_id, &container_name, image)
            .await;

        Ok(container_id)
    }

    pub async fn execute_command(
        &self,
        container_id: &str,
        command: &str,
        _timeout_secs: Option<u64>,
    ) -> Result<ContainerExecutionResult> {
        info!(
            "Executing command in container {}: {}",
            container_id, command
        );

        let start_time = std::time::Instant::now();

        let exec_config = CreateExecOptions {
            cmd: Some(vec!["/bin/sh", "-c", command]),
            attach_stdout: Some(true),
            attach_stderr: Some(true),
            ..Default::default()
        };

        let exec = self.docker.create_exec(container_id, exec_config).await?;

        let start_exec = bollard::exec::StartExecOptions {
            detach: false,
            ..Default::default()
        };

        let output = self.docker.start_exec(&exec.id, Some(start_exec)).await?;

        let (stdout, stderr) = self.process_exec_output(output).await;

        let exec_inspect = self.docker.inspect_exec(&exec.id).await?;
        let exit_code = exec_inspect.exit_code.unwrap_or(-1) as i32;
        let duration_ms = start_time.elapsed().as_millis() as u64;

        Ok(ContainerExecutionResult {
            exit_code,
            stdout,
            stderr,
            duration_ms,
        })
    }

    async fn process_exec_output(&self, output: StartExecResults) -> (String, String) {
        let mut stdout = String::new();
        let mut stderr = String::new();

        match output {
            StartExecResults::Attached { output, input: _ } => {
                let mut stream = output;
                while let Some(chunk) = stream.next().await {
                    match chunk {
                        Ok(log_output) => {
                            let bytes = log_output.into_bytes();
                            let text = String::from_utf8_lossy(&bytes);
                            stdout.push_str(&text);
                        }
                        Err(e) => {
                            error!("Error reading exec output: {}", e);
                            stderr.push_str(&format!("Error: {e}"));
                            break;
                        }
                    }
                }
            }
            StartExecResults::Detached => {
                info!("Command executed in detached mode");
            }
        }

        (stdout, stderr)
    }

    /// Stop a running container gracefully
    pub async fn stop_container(
        &self,
        container_id: &str,
        timeout_secs: Option<u64>,
    ) -> Result<()> {
        info!(
            "Stopping container: {} (timeout: {:?}s)",
            container_id, timeout_secs
        );

        let stop_options = StopContainerOptions {
            t: timeout_secs.unwrap_or(10) as i64,
        };

        self.docker
            .stop_container(container_id, Some(stop_options))
            .await?;

        self.update_container_state(container_id, "stopped").await;
        info!("Container stopped: {}", container_id);
        Ok(())
    }

    /// Restart a container
    pub async fn restart_container(
        &self,
        container_id: &str,
        timeout_secs: Option<u64>,
    ) -> Result<()> {
        info!(
            "Restarting container: {} (timeout: {:?}s)",
            container_id, timeout_secs
        );

        let restart_options = RestartContainerOptions {
            t: timeout_secs.unwrap_or(10) as isize,
        };

        self.docker
            .restart_container(container_id, Some(restart_options))
            .await?;

        self.update_container_state(container_id, "running").await;
        info!("Container restarted: {}", container_id);
        Ok(())
    }

    /// Kill a container forcibly
    pub async fn kill_container(&self, container_id: &str, signal: Option<&str>) -> Result<()> {
        info!("Killing container: {} (signal: {:?})", container_id, signal);

        let kill_options = KillContainerOptions {
            signal: signal.unwrap_or("SIGKILL"),
        };

        self.docker
            .kill_container(container_id, Some(kill_options))
            .await?;

        self.update_container_state(container_id, "killed").await;
        info!("Container killed: {}", container_id);
        Ok(())
    }

    /// Wait for container to finish and return exit code
    pub async fn wait_for_container(&self, container_id: &str) -> Result<i32> {
        info!("Waiting for container to finish: {}", container_id);

        let wait_options = WaitContainerOptions {
            condition: "not-running",
        };

        let mut wait_stream = self.docker.wait_container(container_id, Some(wait_options));

        if let Some(wait_result) = wait_stream.next().await {
            match wait_result {
                Ok(result) => {
                    let exit_code = result.status_code;
                    info!(
                        "Container {} finished with exit code: {}",
                        container_id, exit_code
                    );
                    self.update_container_exit_code(container_id, exit_code as i32)
                        .await;
                    return Ok(exit_code as i32);
                }
                Err(e) => {
                    error!("Error waiting for container {}: {}", container_id, e);
                    return Err(e.into());
                }
            }
        }

        Err(anyhow::anyhow!("Container wait stream ended unexpectedly"))
    }

    /// Get real-time resource usage statistics for a container
    pub async fn get_container_stats(
        &self,
        container_id: &str,
    ) -> Result<Option<ContainerResourceUsage>> {
        debug!(
            "Getting resource statistics for container: {}",
            container_id
        );

        let stats_options = StatsOptions {
            stream: false,
            one_shot: true,
        };

        let mut stats_stream = self.docker.stats(container_id, Some(stats_options));

        if let Some(stats_result) = stats_stream.next().await {
            match stats_result {
                Ok(stats) => {
                    let resource_usage = self.convert_docker_stats_to_usage(&stats)?;
                    return Ok(Some(resource_usage));
                }
                Err(e) => {
                    warn!("Failed to get stats for container {}: {}", container_id, e);
                    return Ok(None);
                }
            }
        }

        Ok(None)
    }

    /// Remove a container (destroy)
    pub async fn destroy_container(&self, container_id: &str, force: bool) -> Result<()> {
        info!("Destroying container: {} (force: {})", container_id, force);

        // Try to stop gracefully first unless forcing
        if !force {
            if let Err(e) = self.stop_container(container_id, Some(10)).await {
                warn!(
                    "Failed to stop container gracefully: {}, forcing removal",
                    e
                );
            }
        }

        let remove_options = RemoveContainerOptions {
            force,
            v: true, // Remove volumes
            link: false,
        };

        self.docker
            .remove_container(container_id, Some(remove_options))
            .await?;

        self.active_containers.write().await.remove(container_id);
        info!("Container destroyed: {}", container_id);
        Ok(())
    }

    pub async fn get_container_status(
        &self,
        container_id: &str,
    ) -> Result<Option<ContainerStatus>> {
        debug!("Getting status for container: {}", container_id);

        if let Some(status) = self.active_containers.read().await.get(container_id) {
            return Ok(Some(status.clone()));
        }

        match self.docker.inspect_container(container_id, None).await {
            Ok(container) => {
                let status = self
                    .lifecycle
                    .build_status_from_inspect(container_id, container);
                Ok(Some(status))
            }
            Err(_) => Ok(None),
        }
    }

    /// List all containers with optional filtering
    pub async fn list_containers(
        &self,
        all: bool,
        filters: Option<HashMap<String, Vec<String>>>,
    ) -> Result<Vec<ContainerStatus>> {
        debug!("Listing containers (all: {}, filters: {:?})", all, filters);

        let list_options = bollard::container::ListContainersOptions {
            all,
            filters: filters.unwrap_or_default(),
            ..Default::default()
        };

        let containers = self.docker.list_containers(Some(list_options)).await?;
        let mut container_statuses = Vec::new();

        for container in containers {
            if let Some(id) = &container.id {
                if let Ok(Some(status)) = self.get_container_status(id).await {
                    container_statuses.push(status);
                }
            }
        }

        Ok(container_statuses)
    }

    /// Get resource usage for all active containers
    pub async fn get_all_container_stats(&self) -> Result<HashMap<String, ContainerResourceUsage>> {
        debug!("Getting stats for all active containers");

        let mut stats_map = HashMap::new();
        let active_containers = self.active_containers.read().await;

        for container_id in active_containers.keys() {
            if let Ok(Some(stats)) = self.get_container_stats(container_id).await {
                stats_map.insert(container_id.clone(), stats);
            }
        }

        Ok(stats_map)
    }

    /// Enforce resource limits on containers
    pub async fn enforce_resource_limits(&self) -> Result<()> {
        debug!("Enforcing resource limits on active containers");

        let stats_map = self.get_all_container_stats().await?;
        let mut violations = Vec::new();

        for (container_id, stats) in stats_map {
            // Check memory limit violation
            if stats.memory_limit_bytes > 0 && stats.memory_usage_bytes > stats.memory_limit_bytes {
                warn!(
                    "Container {} exceeds memory limit: {} > {}",
                    container_id, stats.memory_usage_bytes, stats.memory_limit_bytes
                );
                violations.push((container_id.clone(), "memory"));
            }

            // Check CPU usage (if extremely high)
            if stats.cpu_usage_percent > 95.0 {
                warn!(
                    "Container {} has very high CPU usage: {:.2}%",
                    container_id, stats.cpu_usage_percent
                );
            }
        }

        // Take action on violations (for now, just log - could implement kill/throttle)
        for (container_id, violation_type) in violations {
            warn!(
                "Resource limit violation in container {}: {}",
                container_id, violation_type
            );
        }

        Ok(())
    }

    pub async fn cleanup_inactive_containers(&self) -> Result<()> {
        info!("Cleaning up inactive containers");

        let containers = self.docker.list_containers::<String>(None).await?;
        let mut cleanup_count = 0;

        for container in containers {
            if let Some(state) = &container.state {
                if state == "exited" || state == "dead" {
                    if let Some(id) = &container.id {
                        info!("Cleaning up inactive container: {}", id);
                        if let Err(e) = self.destroy_container(id, true).await {
                            warn!("Failed to cleanup container {}: {}", id, e);
                        } else {
                            cleanup_count += 1;
                        }
                    }
                }
            }
        }

        info!("Cleaned up {} inactive containers", cleanup_count);
        Ok(())
    }

    /// Update container state in active containers tracking
    async fn update_container_state(&self, container_id: &str, state: &str) {
        if let Some(container) = self.active_containers.write().await.get_mut(container_id) {
            container.state = state.to_string();
            container.status = state.to_string();

            if state == "stopped" || state == "killed" {
                container.finished = Some(chrono::Utc::now().timestamp());
            }
        }
    }

    /// Update container exit code in tracking
    async fn update_container_exit_code(&self, container_id: &str, exit_code: i32) {
        if let Some(container) = self.active_containers.write().await.get_mut(container_id) {
            container.exit_code = Some(exit_code);
            container.finished = Some(chrono::Utc::now().timestamp());
        }
    }

    /// Get container logs
    pub async fn get_container_logs(
        &self,
        container_id: &str,
        follow: bool,
        tail_lines: Option<u32>,
    ) -> Result<String> {
        info!(
            "Getting logs for container: {} (follow: {}, tail: {:?})",
            container_id, follow, tail_lines
        );

        let logs_options = bollard::container::LogsOptions::<String> {
            follow,
            stdout: true,
            stderr: true,
            tail: tail_lines
                .map(|n| n.to_string())
                .unwrap_or_else(|| "100".to_string()),
            ..Default::default()
        };

        let mut log_stream = self.docker.logs(container_id, Some(logs_options));
        let mut logs = String::new();
        let mut line_count = 0;
        let max_lines = tail_lines.unwrap_or(100);

        while let Some(chunk) = log_stream.next().await {
            match chunk {
                Ok(log_output) => {
                    let log_bytes = log_output.into_bytes();
                    let log_text = String::from_utf8_lossy(&log_bytes);

                    // Add each line with timestamp if it doesn't have one
                    for line in log_text.lines() {
                        if line_count >= max_lines && !follow {
                            break;
                        }

                        let formatted_line = if line.contains('[') && line.contains(']') {
                            // Line already has timestamp-like format
                            line.to_string()
                        } else {
                            // Add timestamp
                            format!(
                                "[{}] {}",
                                chrono::Utc::now().format("%Y-%m-%d %H:%M:%S"),
                                line
                            )
                        };

                        logs.push_str(&formatted_line);
                        logs.push('\n');
                        line_count += 1;
                    }

                    // If not following, stop after getting the requested lines
                    if !follow && line_count >= max_lines {
                        break;
                    }
                }
                Err(e) => {
                    error!("Error reading container logs: {}", e);
                    logs.push_str(&format!("[ERROR] Failed to read logs: {e}\n"));
                    break;
                }
            }
        }

        if logs.is_empty() {
            logs = format!(
                "[{}] No logs available for container {}\n",
                chrono::Utc::now().format("%Y-%m-%d %H:%M:%S"),
                container_id
            );
        }

        Ok(logs)
    }

    /// Convert Docker stats to our internal resource usage format
    fn convert_docker_stats_to_usage(
        &self,
        stats: &bollard::container::Stats,
    ) -> Result<ContainerResourceUsage> {
        let cpu_usage_percent = self.calculate_cpu_percentage(stats);
        let memory_usage_bytes = stats.memory_stats.usage.unwrap_or(0);
        let memory_limit_bytes = stats.memory_stats.limit.unwrap_or(0);

        // Calculate network I/O
        let network_io_bytes = stats
            .networks
            .as_ref()
            .map(|networks| {
                networks
                    .values()
                    .map(|net| net.rx_bytes + net.tx_bytes)
                    .sum()
            })
            .unwrap_or(0);

        // Calculate block I/O
        let block_io_bytes = stats
            .blkio_stats
            .io_service_bytes_recursive
            .as_ref()
            .map(|blkio| blkio.iter().map(|io| io.value).sum())
            .unwrap_or(0);

        Ok(ContainerResourceUsage {
            cpu_usage_percent: cpu_usage_percent as f64,
            memory_usage_bytes,
            memory_limit_bytes,
            network_io_bytes,
            block_io_bytes,
        })
    }

    /// Calculate CPU usage percentage from Docker stats
    fn calculate_cpu_percentage(&self, stats: &bollard::container::Stats) -> f32 {
        let cpu_delta =
            stats.cpu_stats.cpu_usage.total_usage - stats.precpu_stats.cpu_usage.total_usage;
        let system_delta = stats.cpu_stats.system_cpu_usage.unwrap_or(0)
            - stats.precpu_stats.system_cpu_usage.unwrap_or(0);

        if system_delta > 0 && cpu_delta > 0 {
            let cpu_count = stats.cpu_stats.online_cpus.unwrap_or(1) as f64;
            ((cpu_delta as f64 / system_delta as f64) * cpu_count * 100.0) as f32
        } else {
            0.0
        }
    }

    /// Validate image against allowed registries
    fn validate_image_registry(&self, image: &str) -> Result<()> {
        let allowed_registries = &self.config.registry.allowed_registries;

        // Extract registry from image name
        let registry = if image.contains('/') {
            image.split('/').next().unwrap_or("")
        } else {
            "docker.io" // Default registry
        };

        if !allowed_registries
            .iter()
            .any(|allowed| registry.contains(allowed))
        {
            return Err(anyhow::anyhow!(
                "Image registry '{}' not in allowed list: {:?}",
                registry,
                allowed_registries
            ));
        }

        Ok(())
    }

    async fn ensure_image_available(&self, image: &str) -> Result<()> {
        debug!("Ensuring image is available: {}", image);

        // Validate image registry
        self.validate_image_registry(image)?;

        match self.docker.inspect_image(image).await {
            Ok(_) => {
                debug!("Image {} already available locally", image);
                return Ok(());
            }
            Err(_) => {
                info!("Image {} not found locally, pulling...", image);
            }
        }

        let create_image_options = CreateImageOptions {
            from_image: image,
            ..Default::default()
        };

        let mut pull_stream = self
            .docker
            .create_image(Some(create_image_options), None, None);

        while let Some(pull_result) = pull_stream.next().await {
            match pull_result {
                Ok(info) => {
                    if let Some(status) = info.status {
                        debug!("Pull status: {}", status);
                    }
                }
                Err(e) => {
                    error!("Failed to pull image {}: {}", image, e);
                    return Err(e.into());
                }
            }
        }

        info!("Successfully pulled image: {}", image);
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ContainerLifecycle {
    active_containers: Arc<RwLock<HashMap<String, ContainerStatus>>>,
}

impl ContainerLifecycle {
    pub fn new(
        _docker: Docker,
        active_containers: Arc<RwLock<HashMap<String, ContainerStatus>>>,
    ) -> Self {
        Self { active_containers }
    }

    pub async fn track_container(&self, container_id: &str, container_name: &str, image: &str) {
        let status = ContainerStatus {
            id: container_id.to_string(),
            name: container_name.to_string(),
            image: image.to_string(),
            state: "running".to_string(),
            status: "started".to_string(),
            created: chrono::Utc::now().timestamp(),
            started: Some(chrono::Utc::now().timestamp()),
            finished: None,
            exit_code: None,
            resource_usage: None,
        };

        self.active_containers
            .write()
            .await
            .insert(container_id.to_string(), status);
    }

    pub fn build_status_from_inspect(
        &self,
        container_id: &str,
        container: bollard::models::ContainerInspectResponse,
    ) -> ContainerStatus {
        let container_state = container.state.clone();
        ContainerStatus {
            id: container_id.to_string(),
            name: container.name.unwrap_or_default(),
            image: container.image.unwrap_or_default(),
            state: container_state
                .as_ref()
                .and_then(|s| s.status.as_ref())
                .map(|s| format!("{s:?}"))
                .unwrap_or_default(),
            status: container_state
                .as_ref()
                .and_then(|s| s.status.as_ref())
                .map(|s| format!("{s:?}"))
                .unwrap_or_default(),
            created: container
                .created
                .and_then(|c| c.parse::<chrono::DateTime<chrono::Utc>>().ok())
                .map(|d| d.timestamp())
                .unwrap_or(0),
            started: container_state
                .as_ref()
                .and_then(|s| s.started_at.as_ref())
                .and_then(|s| s.parse::<chrono::DateTime<chrono::Utc>>().ok())
                .map(|d| d.timestamp()),
            finished: container_state
                .as_ref()
                .and_then(|s| s.finished_at.as_ref())
                .and_then(|s| s.parse::<chrono::DateTime<chrono::Utc>>().ok())
                .map(|d| d.timestamp()),
            exit_code: container_state.and_then(|s| s.exit_code).map(|c| c as i32),
            resource_usage: None,
        }
    }
}
