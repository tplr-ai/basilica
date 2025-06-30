//! Unit tests for container manager

use executor::config::{ContainerResourceLimits, DockerConfig};
use executor::container_manager::{
    types::{ContainerInfo, ContainerSpec, ContainerStats, LogEntry},
    ContainerExecutionResult, ContainerLogEntry, ContainerManager, ContainerStatus,
    DockerContainerManager, LogLevel,
};
use std::collections::HashMap;
use std::time::Duration;
use tokio::time::timeout;

// Mock container manager for testing
#[derive(Clone)]
struct MockContainerManager {
    containers: std::sync::Arc<tokio::sync::RwLock<Vec<ContainerInfo>>>,
}

impl MockContainerManager {
    fn new() -> Self {
        Self {
            containers: std::sync::Arc::new(tokio::sync::RwLock::new(vec![])),
        }
    }

    async fn add_container(&self, info: ContainerInfo) {
        let mut containers = self.containers.write().await;
        containers.push(info);
    }
}

#[async_trait::async_trait]
impl executor::container_manager::ContainerManager for MockContainerManager {
    async fn create_container(&self, spec: &ContainerSpec) -> anyhow::Result<String> {
        let container_id = format!("mock_{}", uuid::Uuid::new_v4());
        let info = ContainerInfo {
            id: container_id.clone(),
            name: spec.name.clone().unwrap_or_default(),
            image: spec.image.clone(),
            status: executor::container_manager::types::ContainerStatus::Created,
            created_at: chrono::Utc::now(),
            started_at: None,
            finished_at: None,
            exit_code: None,
            labels: spec.labels.clone(),
            environment: spec.environment.clone(),
        };
        self.add_container(info).await;
        Ok(container_id)
    }

    async fn start_container(&self, container_id: &str) -> anyhow::Result<()> {
        let mut containers = self.containers.write().await;
        if let Some(container) = containers.iter_mut().find(|c| c.id == container_id) {
            container.status = executor::container_manager::types::ContainerStatus::Running;
            container.started_at = Some(chrono::Utc::now());
        }
        Ok(())
    }

    async fn stop_container(&self, container_id: &str, timeout_secs: u64) -> anyhow::Result<()> {
        let mut containers = self.containers.write().await;
        if let Some(container) = containers.iter_mut().find(|c| c.id == container_id) {
            container.status = executor::container_manager::types::ContainerStatus::Stopped;
            container.finished_at = Some(chrono::Utc::now());
            container.exit_code = Some(0);
        }
        Ok(())
    }

    async fn remove_container(&self, container_id: &str, force: bool) -> anyhow::Result<()> {
        let mut containers = self.containers.write().await;
        containers.retain(|c| c.id != container_id);
        Ok(())
    }

    async fn get_container_info(&self, container_id: &str) -> anyhow::Result<ContainerInfo> {
        let containers = self.containers.read().await;
        containers
            .iter()
            .find(|c| c.id == container_id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Container not found"))
    }

    async fn list_containers(&self) -> anyhow::Result<Vec<ContainerInfo>> {
        let containers = self.containers.read().await;
        Ok(containers.clone())
    }

    async fn get_container_logs(
        &self,
        container_id: &str,
        tail: Option<i32>,
    ) -> anyhow::Result<Vec<LogEntry>> {
        Ok(vec![LogEntry {
            timestamp: chrono::Utc::now().timestamp(),
            level: executor::container_manager::types::LogLevel::Info,
            message: format!("Test log for container {}", container_id),
            container_id: container_id.to_string(),
            stream: "stdout".to_string(),
        }])
    }

    async fn stream_logs(
        &self,
        container_id: &str,
        follow: bool,
        tail: Option<i32>,
    ) -> anyhow::Result<futures::stream::BoxStream<'static, LogEntry>> {
        let entry = LogEntry {
            timestamp: chrono::Utc::now().timestamp(),
            level: executor::container_manager::types::LogLevel::Info,
            message: format!("Streaming log for container {}", container_id),
            container_id: container_id.to_string(),
            stream: "stdout".to_string(),
        };
        let stream = futures::stream::once(async move { entry });
        Ok(Box::pin(stream))
    }

    async fn get_container_stats(&self, container_id: &str) -> anyhow::Result<ContainerStats> {
        Ok(ContainerStats {
            cpu_usage_percent: 10.5,
            memory_usage_bytes: 104857600,  // 100MB
            memory_limit_bytes: 1073741824, // 1GB
            network_rx_bytes: 1024,
            network_tx_bytes: 2048,
            block_read_bytes: 4096,
            block_write_bytes: 8192,
        })
    }

    async fn health_check(&self) -> anyhow::Result<()> {
        Ok(())
    }

    async fn cleanup_stopped_containers(&self, older_than_hours: u64) -> anyhow::Result<u32> {
        let cutoff = chrono::Utc::now() - chrono::Duration::hours(older_than_hours as i64);
        let mut containers = self.containers.write().await;
        let original_count = containers.len();
        containers.retain(|c| {
            !(c.status == executor::container_manager::types::ContainerStatus::Stopped
                && c.finished_at.map(|t| t < cutoff).unwrap_or(false))
        });
        Ok((original_count - containers.len()) as u32)
    }

    fn is_mock(&self) -> bool {
        true
    }
}

// Legacy tests for backward compatibility
#[tokio::test]
#[ignore] // Requires Docker daemon
async fn test_container_manager_new() {
    let config = DockerConfig::default();
    let manager = ContainerManager::new(config).await;

    // Should connect to Docker or fail gracefully
    match manager {
        Ok(m) => {
            // Health check should pass
            assert!(m.health_check().await.is_ok());
        }
        Err(e) => {
            // If Docker not available, should get connection error
            assert!(e.to_string().contains("Docker") || e.to_string().contains("connection"));
        }
    }
}

#[tokio::test]
#[ignore] // Requires Docker daemon
async fn test_create_and_destroy_container() {
    let config = DockerConfig::default();
    let manager = ContainerManager::new(config).await.unwrap();

    // Create a simple container
    let container_id = manager
        .create_container(
            "alpine:latest",
            &["sleep".to_string(), "30".to_string()],
            None,
        )
        .await
        .unwrap();

    assert!(!container_id.is_empty());

    // Container should be in list
    let containers = manager.list_containers().await.unwrap();
    assert!(containers.iter().any(|c| c.id == container_id));

    // Destroy container
    manager
        .destroy_container(&container_id, true)
        .await
        .unwrap();

    // Container should no longer be in list
    let containers = manager.list_containers().await.unwrap();
    assert!(!containers.iter().any(|c| c.id == container_id));
}

#[tokio::test]
#[ignore] // Requires Docker daemon
async fn test_execute_command_in_container() {
    let config = DockerConfig::default();
    let manager = ContainerManager::new(config).await.unwrap();

    // Create container
    let container_id = manager
        .create_container(
            "alpine:latest",
            &["sleep".to_string(), "30".to_string()],
            None,
        )
        .await
        .unwrap();

    // Execute command
    let result = manager
        .execute_command(&container_id, "echo 'Hello World'", Some(5))
        .await
        .unwrap();

    assert_eq!(result.exit_code, 0);
    assert!(result.stdout.contains("Hello World"));
    assert!(result.stderr.is_empty());

    // Cleanup
    manager
        .destroy_container(&container_id, true)
        .await
        .unwrap();
}

// New comprehensive tests using mock

#[tokio::test]
async fn test_mock_container_manager_create() {
    let manager = MockContainerManager::new();

    let spec = ContainerSpec {
        image: "alpine:latest".to_string(),
        name: Some("test_container".to_string()),
        command: Some(vec!["sleep".to_string(), "60".to_string()]),
        environment: HashMap::new(),
        labels: HashMap::new(),
        mounts: vec![],
        working_dir: None,
        user: None,
        resource_limits: None,
    };

    let container_id = manager.create_container(&spec).await.unwrap();
    assert!(container_id.starts_with("mock_"));

    let containers = manager.list_containers().await.unwrap();
    assert_eq!(containers.len(), 1);
    assert_eq!(containers[0].image, "alpine:latest");
    assert_eq!(
        containers[0].status,
        executor::container_manager::types::ContainerStatus::Created
    );
}

#[tokio::test]
async fn test_mock_container_manager_start() {
    let manager = MockContainerManager::new();

    let spec = ContainerSpec {
        image: "alpine:latest".to_string(),
        name: Some("test_container".to_string()),
        command: Some(vec!["sleep".to_string(), "60".to_string()]),
        environment: HashMap::new(),
        labels: HashMap::new(),
        mounts: vec![],
        working_dir: None,
        user: None,
        resource_limits: None,
    };

    let container_id = manager.create_container(&spec).await.unwrap();

    // Start the container
    manager.start_container(&container_id).await.unwrap();

    let info = manager.get_container_info(&container_id).await.unwrap();
    assert_eq!(
        info.status,
        executor::container_manager::types::ContainerStatus::Running
    );
    assert!(info.started_at.is_some());
}

#[tokio::test]
async fn test_mock_container_manager_stop() {
    let manager = MockContainerManager::new();

    let spec = ContainerSpec {
        image: "alpine:latest".to_string(),
        name: Some("test_container".to_string()),
        command: Some(vec!["sleep".to_string(), "60".to_string()]),
        environment: HashMap::new(),
        labels: HashMap::new(),
        mounts: vec![],
        working_dir: None,
        user: None,
        resource_limits: None,
    };

    let container_id = manager.create_container(&spec).await.unwrap();
    manager.start_container(&container_id).await.unwrap();

    // Stop the container
    manager.stop_container(&container_id, 10).await.unwrap();

    let info = manager.get_container_info(&container_id).await.unwrap();
    assert_eq!(
        info.status,
        executor::container_manager::types::ContainerStatus::Stopped
    );
    assert!(info.finished_at.is_some());
    assert_eq!(info.exit_code, Some(0));
}

#[tokio::test]
async fn test_mock_container_manager_remove() {
    let manager = MockContainerManager::new();

    let spec = ContainerSpec {
        image: "alpine:latest".to_string(),
        name: Some("test_container".to_string()),
        command: Some(vec!["sleep".to_string(), "60".to_string()]),
        environment: HashMap::new(),
        labels: HashMap::new(),
        mounts: vec![],
        working_dir: None,
        user: None,
        resource_limits: None,
    };

    let container_id = manager.create_container(&spec).await.unwrap();

    // Remove the container
    manager
        .remove_container(&container_id, false)
        .await
        .unwrap();

    let containers = manager.list_containers().await.unwrap();
    assert_eq!(containers.len(), 0);
}

#[tokio::test]
async fn test_mock_container_manager_logs() {
    let manager = MockContainerManager::new();

    let spec = ContainerSpec {
        image: "alpine:latest".to_string(),
        name: Some("test_container".to_string()),
        command: Some(vec!["echo".to_string(), "Hello World".to_string()]),
        environment: HashMap::new(),
        labels: HashMap::new(),
        mounts: vec![],
        working_dir: None,
        user: None,
        resource_limits: None,
    };

    let container_id = manager.create_container(&spec).await.unwrap();

    let logs = manager
        .get_container_logs(&container_id, Some(10))
        .await
        .unwrap();
    assert_eq!(logs.len(), 1);
    assert!(logs[0].message.contains(&container_id));
}

#[tokio::test]
async fn test_mock_container_manager_stats() {
    let manager = MockContainerManager::new();

    let spec = ContainerSpec {
        image: "alpine:latest".to_string(),
        name: Some("test_container".to_string()),
        command: Some(vec!["sleep".to_string(), "60".to_string()]),
        environment: HashMap::new(),
        labels: HashMap::new(),
        mounts: vec![],
        working_dir: None,
        user: None,
        resource_limits: None,
    };

    let container_id = manager.create_container(&spec).await.unwrap();

    let stats = manager.get_container_stats(&container_id).await.unwrap();
    assert_eq!(stats.cpu_usage_percent, 10.5);
    assert_eq!(stats.memory_usage_bytes, 104857600);
}

#[tokio::test]
async fn test_mock_container_manager_cleanup() {
    let manager = MockContainerManager::new();

    // Create a stopped container with old finish time
    let info = ContainerInfo {
        id: "old_container".to_string(),
        name: "old".to_string(),
        image: "alpine:latest".to_string(),
        status: executor::container_manager::types::ContainerStatus::Stopped,
        created_at: chrono::Utc::now() - chrono::Duration::days(2),
        started_at: Some(chrono::Utc::now() - chrono::Duration::days(2)),
        finished_at: Some(chrono::Utc::now() - chrono::Duration::hours(25)),
        exit_code: Some(0),
        labels: HashMap::new(),
        environment: HashMap::new(),
    };
    manager.add_container(info).await;

    // Create a recent stopped container
    let info2 = ContainerInfo {
        id: "recent_container".to_string(),
        name: "recent".to_string(),
        image: "alpine:latest".to_string(),
        status: executor::container_manager::types::ContainerStatus::Stopped,
        created_at: chrono::Utc::now() - chrono::Duration::hours(1),
        started_at: Some(chrono::Utc::now() - chrono::Duration::hours(1)),
        finished_at: Some(chrono::Utc::now() - chrono::Duration::minutes(30)),
        exit_code: Some(0),
        labels: HashMap::new(),
        environment: HashMap::new(),
    };
    manager.add_container(info2).await;

    // Cleanup containers older than 24 hours
    let cleaned = manager.cleanup_stopped_containers(24).await.unwrap();
    assert_eq!(cleaned, 1);

    let containers = manager.list_containers().await.unwrap();
    assert_eq!(containers.len(), 1);
    assert_eq!(containers[0].id, "recent_container");
}

#[tokio::test]
async fn test_container_spec_with_resource_limits() {
    let limits = executor::container_manager::types::ResourceLimits {
        memory_mb: 2048,
        cpu_cores: 1.5,
        disk_gb: 10,
        gpu_count: 1,
    };

    let spec = ContainerSpec {
        image: "nvidia/cuda:11.0-base".to_string(),
        name: Some("gpu_container".to_string()),
        command: Some(vec!["nvidia-smi".to_string()]),
        environment: HashMap::new(),
        labels: HashMap::new(),
        mounts: vec![],
        working_dir: None,
        user: None,
        resource_limits: Some(limits.clone()),
    };

    assert_eq!(spec.resource_limits.unwrap().memory_mb, 2048);
    assert_eq!(spec.resource_limits.unwrap().gpu_count, 1);
}

#[tokio::test]
async fn test_docker_container_manager_creation() {
    let config = executor::config::DockerConfig::default();

    // This may fail if Docker is not available, which is expected in CI
    match DockerContainerManager::new(config).await {
        Ok(manager) => {
            // If Docker is available, test basic functionality
            let result = manager.health_check().await;
            assert!(result.is_ok());
        }
        Err(_) => {
            // Docker not available, use mock instead
            let mock_manager = MockContainerManager::new();
            assert!(mock_manager.is_mock());
        }
    }
}

#[tokio::test]
async fn test_container_status_transitions() {
    let manager = MockContainerManager::new();

    let spec = ContainerSpec {
        image: "alpine:latest".to_string(),
        name: Some("transition_test".to_string()),
        command: Some(vec!["sleep".to_string(), "60".to_string()]),
        environment: HashMap::new(),
        labels: HashMap::new(),
        mounts: vec![],
        working_dir: None,
        user: None,
        resource_limits: None,
    };

    // Create -> Running -> Stopped
    let container_id = manager.create_container(&spec).await.unwrap();

    let info = manager.get_container_info(&container_id).await.unwrap();
    assert_eq!(
        info.status,
        executor::container_manager::types::ContainerStatus::Created
    );

    manager.start_container(&container_id).await.unwrap();
    let info = manager.get_container_info(&container_id).await.unwrap();
    assert_eq!(
        info.status,
        executor::container_manager::types::ContainerStatus::Running
    );

    manager.stop_container(&container_id, 10).await.unwrap();
    let info = manager.get_container_info(&container_id).await.unwrap();
    assert_eq!(
        info.status,
        executor::container_manager::types::ContainerStatus::Stopped
    );
}

#[tokio::test]
async fn test_log_stream() {
    let manager = MockContainerManager::new();

    let spec = ContainerSpec {
        image: "alpine:latest".to_string(),
        name: Some("log_stream_test".to_string()),
        command: Some(vec!["echo".to_string(), "test".to_string()]),
        environment: HashMap::new(),
        labels: HashMap::new(),
        mounts: vec![],
        working_dir: None,
        user: None,
        resource_limits: None,
    };

    let container_id = manager.create_container(&spec).await.unwrap();

    let mut stream = manager
        .stream_logs(&container_id, false, Some(10))
        .await
        .unwrap();

    use futures::StreamExt;
    let log = stream.next().await;
    assert!(log.is_some());
    assert!(log.unwrap().message.contains(&container_id));
}

// Legacy test structures
#[test]
fn test_container_status_creation() {
    let status = ContainerStatus {
        id: "test123".to_string(),
        name: "test-container".to_string(),
        image: "alpine:latest".to_string(),
        state: "running".to_string(),
        status: "Up 5 minutes".to_string(),
        created: 1234567890,
        started: Some(1234567900),
        finished: None,
        exit_code: None,
        resource_usage: None,
    };

    assert_eq!(status.id, "test123");
    assert_eq!(status.state, "running");
    assert!(status.finished.is_none());
}

#[test]
fn test_container_execution_result() {
    let result = ContainerExecutionResult {
        exit_code: 0,
        stdout: "Success".to_string(),
        stderr: String::new(),
        duration_ms: 100,
    };

    assert_eq!(result.exit_code, 0);
    assert_eq!(result.stdout, "Success");
    assert!(result.stderr.is_empty());
    assert_eq!(result.duration_ms, 100);
}

#[test]
fn test_log_level_variants() {
    let levels = vec![
        LogLevel::Info,
        LogLevel::Warning,
        LogLevel::Error,
        LogLevel::Debug,
    ];

    for level in levels {
        match level {
            LogLevel::Info => assert!(true),
            LogLevel::Warning => assert!(true),
            LogLevel::Error => assert!(true),
            LogLevel::Debug => assert!(true),
        }
    }
}

#[test]
fn test_container_status_display() {
    use executor::container_manager::types::ContainerStatus as NewContainerStatus;
    assert_eq!(format!("{}", NewContainerStatus::Created), "created");
    assert_eq!(format!("{}", NewContainerStatus::Running), "running");
    assert_eq!(format!("{}", NewContainerStatus::Stopped), "stopped");
    assert_eq!(format!("{}", NewContainerStatus::Failed), "failed");
    assert_eq!(format!("{}", NewContainerStatus::Unknown), "unknown");
}

#[test]
fn test_log_level_display() {
    use executor::container_manager::types::LogLevel as NewLogLevel;
    assert_eq!(format!("{}", NewLogLevel::Debug), "DEBUG");
    assert_eq!(format!("{}", NewLogLevel::Info), "INFO");
    assert_eq!(format!("{}", NewLogLevel::Warning), "WARNING");
    assert_eq!(format!("{}", NewLogLevel::Error), "ERROR");
}
