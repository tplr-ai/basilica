use basilica_common::types::GpuCategory;
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

/// Provider identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "TEXT")]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    DataCrunch,
    Hyperstack,
    Lambda,
    HydraHost,
}

impl Provider {
    pub fn as_str(&self) -> &'static str {
        match self {
            Provider::DataCrunch => "datacrunch",
            Provider::Hyperstack => "hyperstack",
            Provider::Lambda => "lambda",
            Provider::HydraHost => "hydrahost",
        }
    }
}

impl std::fmt::Display for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::str::FromStr for Provider {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "datacrunch" => Ok(Provider::DataCrunch),
            "hyperstack" => Ok(Provider::Hyperstack),
            "lambda" => Ok(Provider::Lambda),
            "hydrahost" => Ok(Provider::HydraHost),
            _ => Err(format!("Unknown provider: {}", s)),
        }
    }
}

/// Unified GPU offering structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuOffering {
    pub id: String,
    pub provider: Provider,
    pub gpu_type: GpuCategory,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_memory_gb_per_gpu: Option<u32>, // GPU memory per single GPU card in GB (NULL if provider doesn't specify)
    pub gpu_count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interconnect: Option<String>, // GPU interconnect type (SXM4, SXM5, PCIe, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub storage: Option<String>, // Storage capacity (raw provider data, no unit conversion)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deployment_type: Option<String>, // Deployment type (vm, bare-metal, container, etc.)
    pub system_memory_gb: u32, // System RAM
    pub vcpu_count: u32,
    pub region: String,
    #[serde(with = "rust_decimal::serde::str")]
    pub hourly_rate_per_gpu: Decimal, // Price per GPU per hour (multiply by gpu_count for total instance cost)
    pub availability: bool,
    pub fetched_at: DateTime<Utc>,
    #[serde(skip)] // Never expose in API, skip both serializing and deserializing
    pub raw_metadata: serde_json::Value,
}

/// Provider health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderHealth {
    pub provider: Provider,
    pub is_healthy: bool,
    pub last_success_at: Option<DateTime<Utc>>,
    pub last_error: Option<String>,
}

/// Deployment status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeploymentStatus {
    Pending,
    Provisioning,
    Running,
    Error,
    Deleted,
}

impl DeploymentStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            DeploymentStatus::Pending => "pending",
            DeploymentStatus::Provisioning => "provisioning",
            DeploymentStatus::Running => "running",
            DeploymentStatus::Error => "error",
            DeploymentStatus::Deleted => "deleted",
        }
    }
}

impl std::fmt::Display for DeploymentStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::str::FromStr for DeploymentStatus {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "pending" => Ok(DeploymentStatus::Pending),
            "provisioning" => Ok(DeploymentStatus::Provisioning),
            "running" => Ok(DeploymentStatus::Running),
            "error" => Ok(DeploymentStatus::Error),
            "deleted" => Ok(DeploymentStatus::Deleted),
            _ => Err(format!("Unknown deployment status: {}", s)),
        }
    }
}

/// Deployment tracking model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Deployment {
    pub id: String,
    pub user_id: String,
    pub provider: Provider,
    pub provider_instance_id: Option<String>,
    pub offering_id: String,
    pub instance_type: String,
    pub location_code: Option<String>,
    pub status: DeploymentStatus,
    pub hostname: String,
    pub ssh_public_key: Option<String>,
    pub ip_address: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub connection_info: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_response: Option<serde_json::Value>,
    pub error_message: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// User SSH key
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SshKey {
    pub id: String,
    pub user_id: String,
    pub name: String,
    pub public_key: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Provider-specific SSH key mapping (lazy registration)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderSshKey {
    pub id: String,
    pub ssh_key_id: String,
    pub provider: Provider,
    pub provider_key_id: String,
    pub created_at: DateTime<Utc>,
    /// Provider-specific metadata (e.g., Hyperstack stores key name here)
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

/// Request to register SSH key
#[derive(Debug, Deserialize)]
pub struct RegisterSshKeyRequest {
    pub name: String,
    pub public_key: String,
}

/// SSH key response (excludes public_key for security)
#[derive(Debug, Serialize)]
pub struct SshKeyResponse {
    pub id: String,
    pub user_id: String,
    pub name: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl From<SshKey> for SshKeyResponse {
    fn from(key: SshKey) -> Self {
        Self {
            id: key.id,
            user_id: key.user_id,
            name: key.name,
            created_at: key.created_at,
            updated_at: key.updated_at,
        }
    }
}
