//! Docker configuration types and validation

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Docker configuration
///
/// Handles all Docker-related settings following the Single Responsibility Principle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockerConfig {
    /// Docker socket path
    pub socket_path: String,

    /// Default container image for user tasks
    pub default_image: String,

    /// Container resource limits
    pub resource_limits: ContainerResourceLimits,

    /// Network configuration
    pub network_config: ContainerNetworkConfig,

    /// Container timeout
    pub container_timeout: Duration,

    /// Maximum concurrent containers
    pub max_concurrent_containers: u32,

    /// Enable GPU passthrough
    pub enable_gpu_passthrough: bool,

    /// Container registry configuration
    pub registry: ContainerRegistryConfig,
}

/// Container resource limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerResourceLimits {
    /// Maximum memory in bytes
    pub memory_bytes: u64,

    /// Maximum CPU cores (fractional)
    pub cpu_cores: f64,

    /// Maximum GPU memory in bytes
    pub gpu_memory_bytes: Option<u64>,

    /// Maximum disk I/O bandwidth in bytes/sec
    pub disk_io_bps: Option<u64>,

    /// Maximum network bandwidth in bytes/sec
    pub network_bps: Option<u64>,
}

/// Container network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerNetworkConfig {
    /// Enable network isolation
    pub enable_isolation: bool,

    /// Allow internet access
    pub allow_internet: bool,

    /// Custom DNS servers
    pub dns_servers: Vec<String>,

    /// Port mapping rules
    pub port_mappings: Vec<PortMapping>,
}

/// Port mapping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortMapping {
    /// Host port
    pub host_port: u16,

    /// Container port
    pub container_port: u16,

    /// Protocol (tcp/udp)
    pub protocol: String,
}

/// Container registry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerRegistryConfig {
    /// Registry URL
    pub url: String,

    /// Username for authentication
    pub username: Option<String>,

    /// Password for authentication
    pub password: Option<String>,

    /// Enable image signature verification
    pub verify_signatures: bool,

    /// Allowed image registries
    pub allowed_registries: Vec<String>,
}

impl Default for DockerConfig {
    fn default() -> Self {
        Self {
            socket_path: "/var/run/docker.sock".to_string(),
            default_image: "ubuntu:22.04".to_string(),
            resource_limits: ContainerResourceLimits::default(),
            network_config: ContainerNetworkConfig::default(),
            container_timeout: Duration::from_secs(3600), // 1 hour
            max_concurrent_containers: 10,
            enable_gpu_passthrough: true,
            registry: ContainerRegistryConfig::default(),
        }
    }
}

impl Default for ContainerResourceLimits {
    fn default() -> Self {
        Self {
            memory_bytes: 8 * 1024 * 1024 * 1024, // 8GB
            cpu_cores: 4.0,
            gpu_memory_bytes: Some(4 * 1024 * 1024 * 1024), // 4GB
            disk_io_bps: Some(100 * 1024 * 1024),           // 100MB/s
            network_bps: Some(100 * 1024 * 1024),           // 100MB/s
        }
    }
}

impl Default for ContainerNetworkConfig {
    fn default() -> Self {
        Self {
            enable_isolation: true,
            allow_internet: false,
            dns_servers: vec!["8.8.8.8".to_string(), "8.8.4.4".to_string()],
            port_mappings: vec![],
        }
    }
}

impl Default for ContainerRegistryConfig {
    fn default() -> Self {
        Self {
            url: "docker.io".to_string(),
            username: None,
            password: None,
            verify_signatures: true,
            allowed_registries: vec![
                "docker.io".to_string(),
                "ghcr.io".to_string(),
                "quay.io".to_string(),
            ],
        }
    }
}

/// Docker configuration validation trait
pub trait DockerConfigValidation {
    fn validate_resource_limits(&self) -> Result<(), String>;
    fn validate_network_settings(&self) -> Result<(), String>;
    fn validate_registry_settings(&self) -> Result<(), String>;
    fn docker_warnings(&self) -> Vec<String>;
}

impl DockerConfigValidation for DockerConfig {
    fn validate_resource_limits(&self) -> Result<(), String> {
        if self.max_concurrent_containers == 0 {
            return Err("Must allow at least 1 concurrent container".to_string());
        }

        if self.resource_limits.memory_bytes == 0 {
            return Err("Memory limit must be greater than 0".to_string());
        }

        if self.resource_limits.cpu_cores <= 0.0 {
            return Err("CPU cores must be greater than 0".to_string());
        }

        Ok(())
    }

    fn validate_network_settings(&self) -> Result<(), String> {
        for mapping in &self.network_config.port_mappings {
            if mapping.host_port == 0 || mapping.container_port == 0 {
                return Err("Port mappings must use valid port numbers".to_string());
            }

            if !["tcp", "udp"].contains(&mapping.protocol.as_str()) {
                return Err("Port mapping protocol must be 'tcp' or 'udp'".to_string());
            }
        }

        Ok(())
    }

    fn validate_registry_settings(&self) -> Result<(), String> {
        if self.registry.allowed_registries.is_empty() {
            return Err("At least one registry must be allowed".to_string());
        }

        Ok(())
    }

    fn docker_warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        if self.network_config.allow_internet {
            warnings.push("Container internet access is enabled - may be insecure".to_string());
        }

        if !self.registry.verify_signatures {
            warnings.push("Image signature verification is disabled - may be insecure".to_string());
        }

        if self.max_concurrent_containers > 50 {
            warnings
                .push("Very high concurrent container limit may affect performance".to_string());
        }

        warnings
    }
}
