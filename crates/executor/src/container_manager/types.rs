//! Container management types and data structures

use serde::{Deserialize, Serialize};

/// Container execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerExecutionResult {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub duration_ms: u64,
}

/// Container log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerLogEntry {
    pub timestamp: i64,
    pub level: LogLevel,
    pub message: String,
    pub container_id: String,
}

/// Log level enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogLevel {
    Info,
    Warning,
    Error,
    Debug,
}

/// Container status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerStatus {
    pub id: String,
    pub name: String,
    pub image: String,
    pub state: String,
    pub status: String,
    pub created: i64,
    pub started: Option<i64>,
    pub finished: Option<i64>,
    pub exit_code: Option<i32>,
    pub resource_usage: Option<ContainerResourceUsage>,
}

/// Container resource usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerResourceUsage {
    pub cpu_usage_percent: f64,
    pub memory_usage_bytes: u64,
    pub memory_limit_bytes: u64,
    pub network_io_bytes: u64,
    pub block_io_bytes: u64,
}

/// Container operation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContainerOperation {
    Create {
        image: String,
        command: Vec<String>,
        resource_limits: Option<crate::config::ContainerResourceLimits>,
    },
    Start {
        container_id: String,
    },
    Stop {
        container_id: String,
        timeout_secs: Option<u64>,
    },
    Restart {
        container_id: String,
        timeout_secs: Option<u64>,
    },
    Kill {
        container_id: String,
        signal: Option<String>,
    },
    Remove {
        container_id: String,
        force: bool,
    },
    Execute {
        container_id: String,
        command: String,
        timeout_secs: Option<u64>,
    },
    GetStatus {
        container_id: String,
    },
    GetStats {
        container_id: String,
    },
    ListContainers {
        all: bool,
        filters: Option<std::collections::HashMap<String, Vec<String>>>,
    },
}

/// Container operation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContainerOperationResult {
    Created(String), // container_id
    Started,
    Stopped,
    Restarted,
    Killed,
    Removed,
    ExecutionResult(ContainerExecutionResult),
    Status(Option<ContainerStatus>),
    Stats(Option<ContainerResourceUsage>),
    ContainerList(Vec<ContainerStatus>),
    Error(String),
}

/// Container lifecycle state
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ContainerLifecycleState {
    Created,
    Running,
    Stopped,
    Paused,
    Restarting,
    Removing,
    Dead,
    Exited,
}

/// Container network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerNetworkConfig {
    pub network_mode: String,
    pub port_mappings: Vec<ContainerPortMapping>,
    pub dns_servers: Vec<String>,
    pub enable_internet: bool,
}

/// Container port mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerPortMapping {
    pub host_port: u16,
    pub container_port: u16,
    pub protocol: String, // tcp, udp
    pub host_ip: Option<String>,
}

/// Container volume configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerVolumeConfig {
    pub host_path: Option<String>,
    pub container_path: String,
    pub read_only: bool,
    pub volume_type: VolumeType,
}

/// Volume type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VolumeType {
    Bind,
    Volume,
    Tmpfs {
        size_bytes: Option<u64>,
        mode: Option<u32>,
    },
}

/// Container security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerSecurityConfig {
    pub user: Option<String>,
    pub capabilities_add: Vec<String>,
    pub capabilities_drop: Vec<String>,
    pub no_new_privileges: bool,
    pub readonly_rootfs: bool,
    pub security_opt: Vec<String>,
}

/// GPU allocation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuAllocationConfig {
    pub enable_gpu: bool,
    pub gpu_count: Option<i32>, // -1 = all available
    pub gpu_device_ids: Vec<String>,
    pub gpu_memory_limit: Option<u64>,
    pub gpu_capabilities: Vec<String>,
}

impl Default for ContainerNetworkConfig {
    fn default() -> Self {
        Self {
            network_mode: "bridge".to_string(),
            port_mappings: vec![],
            dns_servers: vec!["8.8.8.8".to_string(), "8.8.4.4".to_string()],
            enable_internet: false,
        }
    }
}

impl Default for ContainerSecurityConfig {
    fn default() -> Self {
        Self {
            user: Some("1000:1000".to_string()),
            capabilities_add: vec![
                "CHOWN".to_string(),
                "SETUID".to_string(),
                "SETGID".to_string(),
            ],
            capabilities_drop: vec!["ALL".to_string()],
            no_new_privileges: true,
            readonly_rootfs: false,
            security_opt: vec!["no-new-privileges:true".to_string()],
        }
    }
}

impl Default for GpuAllocationConfig {
    fn default() -> Self {
        Self {
            enable_gpu: true,
            gpu_count: Some(-1), // All available
            gpu_device_ids: vec![],
            gpu_memory_limit: Some(4 * 1024 * 1024 * 1024), // 4GB
            gpu_capabilities: vec![
                "gpu".to_string(),
                "compute".to_string(),
                "utility".to_string(),
            ],
        }
    }
}
