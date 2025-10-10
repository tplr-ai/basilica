//! Types for rental operations

use chrono::{DateTime, Utc};
use core::fmt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Rental request from validator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RentalRequest {
    pub validator_hotkey: String,
    pub miner_id: String,
    pub node_id: String,
    pub container_spec: ContainerSpec,
    pub ssh_public_key: String,
    pub metadata: HashMap<String, String>,
}

/// Container specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerSpec {
    pub image: String,
    pub environment: HashMap<String, String>,
    pub ports: Vec<PortMapping>,
    pub resources: ResourceRequirements,
    #[serde(default)]
    pub entrypoint: Vec<String>,
    pub command: Vec<String>,
    pub volumes: Vec<VolumeMount>,
    pub labels: HashMap<String, String>,
    pub capabilities: Vec<String>,
    pub network: NetworkConfig,
}

/// Port mapping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortMapping {
    pub container_port: u32,
    pub host_port: u32,
    pub protocol: String,
}

/// Resource requirements
// TODO: make this type compatible with the one in basilica-api
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub cpu_cores: f64,
    pub memory_mb: i64,
    pub storage_mb: i64,
    pub gpu_count: u32,
    pub gpu_types: Vec<String>,
}

/// Volume mount configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeMount {
    pub host_path: String,
    pub container_path: String,
    pub read_only: bool,
}

/// Network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub mode: String,
    pub dns: Vec<String>,
    pub extra_hosts: HashMap<String, String>,
}

/// Rental response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RentalResponse {
    pub rental_id: String,
    pub ssh_credentials: Option<String>,
    pub container_info: ContainerInfo,
}

/// Container information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerInfo {
    pub container_id: String,
    pub container_name: String,
    #[serde(default)]
    pub mapped_ports: Vec<PortMapping>,
    pub status: String,
    #[serde(default)]
    pub labels: HashMap<String, String>,
}

/// Rental state
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
pub enum RentalState {
    Provisioning,
    Active,
    Stopping,
    Stopped,
    Failed,
}

impl fmt::Display for RentalState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

/// Rental information stored in memory and persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RentalInfo {
    pub rental_id: String,
    pub validator_hotkey: String,
    pub node_id: String,
    pub container_id: String,
    pub ssh_session_id: String,
    pub ssh_credentials: String, // Validator SSH access to node
    pub state: RentalState,
    pub created_at: DateTime<Utc>,
    pub container_spec: ContainerSpec,
    pub miner_id: String,
    pub node_details: crate::api::types::NodeDetails,
    pub metadata: HashMap<String, String>,
}

/// Rental status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RentalStatus {
    pub rental_id: String,
    pub state: RentalState,
    pub container_status: ContainerStatus,
    pub created_at: DateTime<Utc>,
    pub resource_usage: ResourceUsage,
}

/// Container status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerStatus {
    pub container_id: String,
    pub state: String,
    pub exit_code: Option<i32>,
    pub health: String,
    pub started_at: Option<DateTime<Utc>>,
    pub finished_at: Option<DateTime<Utc>>,
}

/// Resource usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub cpu_percent: f64,
    pub memory_mb: i64,
    pub disk_read_bytes: i64,
    pub disk_write_bytes: i64,
    pub network_rx_bytes: i64,
    pub network_tx_bytes: i64,
    pub gpu_usage: Vec<GpuUsage>,
}

/// GPU usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuUsage {
    pub gpu_index: u32,
    pub utilization_percent: f64,
    pub memory_mb: i64,
    pub temperature_celsius: f64,
}

/// Log entry from container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub timestamp: DateTime<Utc>,
    pub stream: String,
    pub message: String,
    pub container_id: String,
}
