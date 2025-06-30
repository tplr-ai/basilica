//! Container management module
//!
//! Provides modular container management functionality with separation of concerns.

pub mod config_builder;
pub mod health;
pub mod logs;
pub mod operations;
pub mod types;

use health::HealthChecker;
use logs::LogStreamer;
use operations::ContainerOperations;
pub use types::*;

use crate::config::{ContainerResourceLimits, DockerConfig};
use anyhow::Result;
use bollard::Docker;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

#[derive(Debug, Clone)]
pub struct ContainerManager {
    active_containers: Arc<RwLock<HashMap<String, ContainerStatus>>>,
    operations: ContainerOperations,
    log_streamer: LogStreamer,
    health_checker: HealthChecker,
}

impl ContainerManager {
    pub async fn new(config: DockerConfig) -> Result<Self> {
        info!(
            "Initializing container manager with Docker socket: {}",
            config.socket_path
        );

        let docker = if config.socket_path.starts_with("unix://") {
            Docker::connect_with_unix(&config.socket_path, 120, bollard::API_DEFAULT_VERSION)?
        } else {
            Docker::connect_with_socket_defaults()?
        };

        let version = docker.version().await?;
        info!(
            "Connected to Docker daemon version: {}",
            version.version.unwrap_or_default()
        );

        let active_containers = Arc::new(RwLock::new(HashMap::new()));
        let operations =
            ContainerOperations::new(docker.clone(), config.clone(), active_containers.clone());
        let log_streamer = LogStreamer::new(docker.clone());
        let health_checker = HealthChecker::new(docker.clone());

        Ok(Self {
            active_containers,
            operations,
            log_streamer,
            health_checker,
        })
    }

    pub async fn create_container(
        &self,
        image: &str,
        command: &[String],
        resource_limits: Option<ContainerResourceLimits>,
    ) -> Result<String> {
        self.operations
            .create_container(image, command, resource_limits)
            .await
    }

    pub async fn execute_command(
        &self,
        container_id: &str,
        command: &str,
        timeout_secs: Option<u64>,
    ) -> Result<ContainerExecutionResult> {
        self.operations
            .execute_command(container_id, command, timeout_secs)
            .await
    }

    pub async fn destroy_container(&self, container_id: &str, force: bool) -> Result<()> {
        self.operations.destroy_container(container_id, force).await
    }

    pub async fn stream_logs(
        &self,
        container_id: &str,
        follow: bool,
        tail_lines: Option<i32>,
    ) -> Result<impl futures_util::Stream<Item = ContainerLogEntry>> {
        self.log_streamer
            .stream_logs(container_id, follow, tail_lines)
            .await
    }

    pub async fn get_container_status(
        &self,
        container_id: &str,
    ) -> Result<Option<ContainerStatus>> {
        self.operations.get_container_status(container_id).await
    }

    pub async fn list_containers(&self) -> Result<Vec<ContainerStatus>> {
        let containers = self.active_containers.read().await;
        Ok(containers.values().cloned().collect())
    }

    pub async fn cleanup_inactive_containers(&self) -> Result<()> {
        self.operations.cleanup_inactive_containers().await
    }

    pub async fn get_container_logs(
        &self,
        container_id: &str,
        follow: bool,
        tail_lines: Option<u32>,
    ) -> Result<String> {
        self.operations
            .get_container_logs(container_id, follow, tail_lines)
            .await
    }

    pub async fn get_container_stats(
        &self,
        container_id: &str,
    ) -> Result<Option<ContainerResourceUsage>> {
        self.operations.get_container_stats(container_id).await
    }

    pub async fn health_check(&self) -> Result<()> {
        self.health_checker.health_check().await
    }
}
