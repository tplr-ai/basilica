use crate::{types::LocationProfile, utils::PortMapping as UtilityPortMapping};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fmt};

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct RentCapacityRequest {
    pub gpu_requirements: GpuRequirements,
    pub ssh_public_key: String,
    pub docker_image: String,
    pub env_vars: Option<HashMap<String, String>>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct GpuRequirements {
    pub min_memory_gb: u32,
    pub gpu_type: Option<String>,
    pub gpu_count: u32,
}

impl Default for GpuRequirements {
    fn default() -> Self {
        Self {
            min_memory_gb: 0,
            gpu_type: Some("b200".to_string()),
            gpu_count: 0,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RentCapacityResponse {
    pub rental_id: String,
    pub node: NodeDetails,
    pub ssh_access: SshAccess,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkSpeedInfo {
    pub download_mbps: Option<f64>,
    pub upload_mbps: Option<f64>,
    pub test_timestamp: Option<chrono::DateTime<chrono::Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeDetails {
    pub id: String,
    pub gpu_specs: Vec<GpuSpec>,
    pub cpu_specs: CpuSpec,
    pub location: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub network_speed: Option<NetworkSpeedInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hourly_rate_cents: Option<i32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GpuSpec {
    pub name: String,
    pub memory_gb: u32,
    pub compute_capability: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CpuSpec {
    pub cores: u32,
    pub model: String,
    pub memory_gb: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SshAccess {
    pub host: String,
    pub port: u16,
    pub username: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TerminateRentalRequest {
    pub reason: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RentalStatusResponse {
    pub rental_id: String,
    pub status: RentalStatus,
    pub node: NodeDetails,
    pub miner_uid: u16,
    pub miner_hotkey: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "lowercase")]
pub enum RentalStatus {
    Pending,
    Active,
    Terminated,
    Failed,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ListAvailableNodesResponse {
    pub available_nodes: Vec<AvailableNode>,
    pub total_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvailableNode {
    pub node: NodeDetails,
    pub availability: AvailabilityInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvailabilityInfo {
    pub available_until: Option<chrono::DateTime<chrono::Utc>>,
}

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct ListAvailableNodesQuery {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub available: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_gpu_memory: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_gpu_count: Option<u32>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub location: Option<LocationProfile>,
}

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct LogQuery {
    pub follow: Option<bool>,
    pub tail: Option<u32>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct StartRentalRequest {
    pub gpu_category: String,
    pub gpu_count: u32,
    #[serde(default)]
    pub min_memory_gb: Option<u32>,
    pub max_hourly_rate_cents: u32,
    pub container_image: String,
    pub ssh_public_key: String,
    #[serde(default)]
    pub environment: HashMap<String, String>,
    #[serde(default)]
    pub ports: Vec<PortMappingRequest>,
    #[serde(default)]
    pub resources: ResourceRequirementsRequest,
    #[serde(default = "default_command")]
    pub command: Vec<String>,
    #[serde(default)]
    pub volumes: Vec<VolumeMountRequest>,
}

fn default_command() -> Vec<String> {
    vec!["/bin/bash".to_string()]
}

impl Default for StartRentalRequest {
    fn default() -> Self {
        Self {
            gpu_category: String::new(),
            gpu_count: 1,
            min_memory_gb: None,
            max_hourly_rate_cents: 0,
            container_image: "nvidia/cuda:12.2.0-base-ubuntu22.04".to_string(),
            ssh_public_key: String::new(),
            environment: HashMap::new(),
            ports: Vec::new(),
            resources: ResourceRequirementsRequest::default(),
            command: default_command(),
            volumes: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PortMappingRequest {
    pub container_port: u32,
    pub host_port: u32,
    #[serde(default = "default_protocol")]
    pub protocol: String,
}

fn default_protocol() -> String {
    "tcp".to_string()
}

impl Default for PortMappingRequest {
    fn default() -> Self {
        Self {
            container_port: 0,
            host_port: 0,
            protocol: default_protocol(),
        }
    }
}

impl From<UtilityPortMapping> for PortMappingRequest {
    fn from(mapping: UtilityPortMapping) -> Self {
        Self {
            container_port: mapping.container_port,
            host_port: mapping.host_port,
            protocol: mapping.protocol,
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ResourceRequirementsRequest {
    pub cpu_cores: f64,
    pub memory_mb: i64,
    pub storage_mb: i64,
    pub gpu_count: u32,
    #[serde(default)]
    pub gpu_types: Vec<String>,
}

impl Default for ResourceRequirementsRequest {
    fn default() -> Self {
        Self {
            cpu_cores: 0.0,
            memory_mb: 0,
            storage_mb: 0,
            gpu_count: 0,
            gpu_types: Vec::new(),
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct VolumeMountRequest {
    pub host_path: String,
    pub container_path: String,
    #[serde(default)]
    pub read_only: bool,
}

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct RentalStatusQuery {
    pub include_resource_usage: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RentalState {
    Provisioning,
    Active,
    Restarting,
    Stopping,
    Stopped,
    Failed,
}

impl fmt::Display for RentalState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct ListRentalsQuery {
    pub state: Option<RentalState>,
    pub list_type: Option<String>,
    pub min_gpu_memory: Option<u32>,
    pub gpu_type: Option<String>,
    pub min_gpu_count: Option<u32>,
    pub max_cost_per_hour: Option<f64>,
}

pub type ValidatorListRentalsQuery = ListRentalsQuery;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RentalListItem {
    pub rental_id: String,
    pub node_id: String,
    pub container_id: String,
    pub state: RentalState,
    pub created_at: String,
    pub miner_id: String,
    pub container_image: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_specs: Option<Vec<GpuSpec>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_specs: Option<CpuSpec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub network_speed: Option<NetworkSpeedInfo>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ListRentalsResponse {
    pub rentals: Vec<RentalListItem>,
    pub total_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortMapping {
    pub container_port: u32,
    pub host_port: u32,
    pub protocol: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RentalResponse {
    pub rental_id: String,
    pub ssh_credentials: Option<String>,
    pub container_info: ContainerInfo,
}

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RentalRestartResponse {
    pub rental_id: String,
    pub status: RentalState,
    pub message: String,
    pub operation_duration_ms: u64,
}
