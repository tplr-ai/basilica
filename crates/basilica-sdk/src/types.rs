//! Type definitions for the Basilica SDK

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Re-export types from basilica-validator that are used by the client
pub use basilica_validator::api::types::{
    AvailabilityInfo, AvailableNode, CpuSpec, GpuRequirements, GpuSpec, ListAvailableNodesQuery,
    ListAvailableNodesResponse, LogQuery, NetworkSpeedInfo, NodeDetails, RentCapacityRequest,
    RentCapacityResponse, RentalListItem, RentalStatus,
    RentalStatusResponse as ValidatorRentalStatusResponse, SshAccess, TerminateRentalRequest,
};

// Re-export LocationProfile for SDK consumers
pub use basilica_common::LocationProfile;

// Re-export rental-specific types from validator
pub use basilica_validator::api::routes::rentals::{
    PortMappingRequest, ResourceRequirementsRequest, StartRentalRequest, VolumeMountRequest,
};

// Re-export RentalState from validator for SDK consumers
pub use basilica_validator::rental::types::RentalState;

// SDK-specific types

/// Health check response
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct HealthCheckResponse {
    /// Service status
    pub status: String,

    /// Service version
    pub version: String,

    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Healthy validators count
    pub healthy_validators: usize,

    /// Total validators count
    pub total_validators: usize,
}

/// List rentals query
#[derive(Debug, Deserialize, Serialize, Default)]
pub struct ListRentalsQuery {
    /// Status filter
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<RentalState>,

    /// GPU type filter
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_type: Option<String>,

    /// Minimum GPU count
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_gpu_count: Option<u32>,
}

/// Rental status response (alias for compatibility)
pub type RentalStatusResponse = ValidatorRentalStatusResponse;

/// API rental list item with GPU information
#[derive(Debug, Serialize, Deserialize)]
pub struct ApiRentalListItem {
    pub rental_id: String,
    pub node_id: String,
    pub container_id: String,
    pub state: RentalState,
    pub created_at: String,
    pub miner_id: String,
    pub container_image: String,
    /// GPU specifications for this rental
    pub gpu_specs: Vec<GpuSpec>,
    /// Whether SSH credentials are available for this rental
    pub has_ssh: bool,
    /// Optional CPU specifications for detailed view
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_specs: Option<CpuSpec>,
    /// Optional location for detailed view
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<String>,
    /// Optional network speed information
    #[serde(skip_serializing_if = "Option::is_none")]
    pub network_speed: Option<NetworkSpeedInfo>,
    /// Port mappings for this rental
    #[serde(skip_serializing_if = "Option::is_none")]
    pub port_mappings: Option<Vec<basilica_validator::rental::PortMapping>>,
    /// Hourly cost rate for this rental (includes markup)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hourly_cost: Option<f64>,
    /// Accumulated cost from billing service
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accumulated_cost: Option<String>,
}

/// API list rentals response with GPU information
#[derive(Debug, Serialize, Deserialize)]
pub struct ApiListRentalsResponse {
    pub rentals: Vec<ApiRentalListItem>,
    pub total_count: usize,
}

/// Historical rental item from billing service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalRentalItem {
    pub rental_id: String,
    pub node_id: Option<String>,
    pub status: String,
    pub total_cost: String,
    pub hourly_rate: Option<f64>,
    pub started_at: chrono::DateTime<chrono::Utc>,
    pub stopped_at: chrono::DateTime<chrono::Utc>,
    pub duration_seconds: i64,
    pub gpu_count: u32,
    pub cloud_type: String, // "community" or "secure"
}

/// API response for historical rentals
#[derive(Debug, Serialize, Deserialize)]
pub struct HistoricalRentalsResponse {
    pub rentals: Vec<HistoricalRentalItem>,
    pub total_count: usize,
    pub total_cost: String,
}

/// Rental status query parameters
#[derive(Debug, Deserialize, Serialize)]
pub struct RentalStatusQuery {
    #[allow(dead_code)]
    pub include_resource_usage: Option<bool>,
}

/// Log streaming query parameters
#[derive(Debug, Deserialize, Serialize)]
pub struct LogStreamQuery {
    pub follow: Option<bool>,
    pub tail: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub since_seconds: Option<u32>,
}

/// Node selection strategy for rental requests
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum NodeSelection {
    /// Select a specific node by ID
    NodeId { node_id: String },
    /// Select node with exact GPU configuration (exact count match)
    ExactGpuConfiguration { gpu_requirements: GpuRequirements },
}

/// Start rental request with flexible node selection
#[derive(Debug, Serialize, Deserialize)]
pub struct StartRentalApiRequest {
    /// How to select the node for this rental
    pub node_selection: NodeSelection,

    /// Container image to run
    pub container_image: String,

    /// Environment variables
    #[serde(default)]
    pub environment: std::collections::HashMap<String, String>,

    /// Port mappings
    #[serde(default)]
    pub ports: Vec<PortMappingRequest>,

    /// Resource requirements
    #[serde(default)]
    pub resources: ResourceRequirementsRequest,

    /// Command to run
    #[serde(default)]
    pub command: Vec<String>,

    /// Volume mounts
    #[serde(default)]
    pub volumes: Vec<VolumeMountRequest>,

    /// Disable SSH
    #[serde(default)]
    pub no_ssh: bool,
}

/// Extended rental status response that includes SSH credentials from the database
#[derive(Debug, Serialize, Deserialize)]
pub struct RentalStatusWithSshResponse {
    /// Rental ID
    pub rental_id: String,

    /// Current rental status
    pub status: RentalStatus,

    /// Node details
    pub node: NodeDetails,

    /// SSH credentials (from database, not validator)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ssh_credentials: Option<String>,

    /// Port mappings (from database)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub port_mappings: Option<Vec<basilica_validator::rental::PortMapping>>,

    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,

    /// Last update timestamp
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

impl RentalStatusWithSshResponse {
    /// Create from validator response, database SSH credentials, and port mappings
    pub fn from_validator_response(
        response: ValidatorRentalStatusResponse,
        ssh_credentials: Option<String>,
        port_mappings: Option<Vec<basilica_validator::rental::PortMapping>>,
    ) -> Self {
        Self {
            rental_id: response.rental_id,
            status: response.status,
            node: response.node,
            ssh_credentials,
            port_mappings,
            created_at: response.created_at,
            updated_at: response.updated_at,
        }
    }
}

// API Key Management Types

/// Request to create a new API key
#[derive(Debug, Serialize, Deserialize)]
pub struct CreateApiKeyRequest {
    /// Name for the API key
    pub name: String,

    /// Optional scopes for the API key
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scopes: Option<Vec<String>>,
}

/// Response after creating a new API key
#[derive(Debug, Serialize, Deserialize)]
pub struct ApiKeyResponse {
    /// Name of the key
    pub name: String,

    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,

    /// The full API key token (only returned once at creation)
    pub token: String,
}

/// API key information (without the secret)
#[derive(Debug, Serialize, Deserialize)]
pub struct ApiKeyInfo {
    /// Key identifier (kid)
    pub kid: String,

    /// Name of the key
    pub name: String,

    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,

    /// Last usage timestamp
    pub last_used_at: Option<chrono::DateTime<chrono::Utc>>,
}

// SSH Key Management Types

/// Request to register an SSH key
#[derive(Debug, Serialize, Deserialize)]
pub struct RegisterSshKeyRequest {
    /// Name for the SSH key
    pub name: String,

    /// SSH public key content
    pub public_key: String,
}

/// SSH key response
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SshKeyResponse {
    /// Key identifier
    pub id: String,

    /// User identifier
    pub user_id: String,

    /// Name of the key
    pub name: String,

    /// SSH public key content (needed for local key matching)
    pub public_key: String,

    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,

    /// Last update timestamp
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

// ============================================================================
// Secure Cloud Rental Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartSecureCloudRentalRequest {
    /// Offering ID from list_gpu_prices endpoint
    pub offering_id: String,

    /// User's registered SSH key ID (NOT the public key string)
    /// Must be a key owned by the authenticated user
    pub ssh_public_key_id: String,

    /// Docker container image (optional)
    pub container_image: Option<String>,

    /// Environment variables for the container
    #[serde(default)]
    pub environment: HashMap<String, String>,

    /// Port mappings
    #[serde(default)]
    pub ports: Vec<PortMappingRequest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecureCloudRentalResponse {
    /// Rental ID (for API tracking)
    pub rental_id: String,

    /// Deployment ID (aggregator service ID)
    pub deployment_id: String,

    /// Provider name
    pub provider: String,

    /// Deployment status
    pub status: String,

    /// IP address of the instance (if available)
    pub ip_address: Option<String>,

    /// Ready-to-use SSH command
    pub ssh_command: Option<String>,

    /// Hourly cost in USD (base_price × gpu_count × (1 + markup%/100))
    pub hourly_cost: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StopSecureCloudRentalResponse {
    /// Rental ID
    pub rental_id: String,

    /// Final status
    pub status: String,

    /// Total rental duration in hours
    pub duration_hours: f64,

    /// Total cost charged
    pub total_cost: f64,
}

// Payment Management Types

/// Deposit account response from API
#[derive(Debug, Serialize, Deserialize)]
pub struct DepositAccountResponse {
    pub user_id: String,
    pub address: String,
    pub exists: bool,
}

/// Response after creating a deposit account
#[derive(Debug, Serialize, Deserialize)]
pub struct CreateDepositAccountResponse {
    pub user_id: String,
    pub address: String,
}

/// Deposit status
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DepositStatus {
    Pending,
    Finalized,
    Credited,
    Failed,
}

/// Individual deposit record
#[derive(Debug, Serialize, Deserialize)]
pub struct DepositRecord {
    pub tx_hash: String,
    pub block_number: u64,
    pub event_index: u32,
    pub from_address: String,
    pub to_address: String,
    pub amount_tao: String,
    pub status: DepositStatus,
    pub observed_at: chrono::DateTime<chrono::Utc>,
    pub finalized_at: Option<chrono::DateTime<chrono::Utc>>,
    pub credited_at: Option<chrono::DateTime<chrono::Utc>>,
}

/// List deposits response
#[derive(Debug, Serialize, Deserialize)]
pub struct ListDepositsResponse {
    pub deposits: Vec<DepositRecord>,
    pub total_count: usize,
}

/// Query parameters for listing deposits
#[derive(Debug, Serialize, Deserialize, Default)]
pub struct ListDepositsQuery {
    #[serde(default)]
    pub limit: u32,
    #[serde(default)]
    pub offset: u32,
}

// Billing Management Types

/// Balance response from billing service
#[derive(Debug, Serialize, Deserialize)]
pub struct BalanceResponse {
    pub balance: String,
    pub last_updated: String,
}

// Usage History Types

/// Individual rental usage record
#[derive(Debug, Serialize, Deserialize)]
pub struct RentalUsageRecord {
    pub rental_id: String,
    pub node_id: String,
    pub status: String,
    pub hourly_rate: String,
    pub current_cost: String,
    pub start_time: chrono::DateTime<chrono::Utc>,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// Usage history response
#[derive(Debug, Serialize, Deserialize)]
pub struct UsageHistoryResponse {
    pub rentals: Vec<RentalUsageRecord>,
    pub total_count: u64,
}

/// Time-series usage data point
#[derive(Debug, Serialize, Deserialize)]
pub struct UsageDataPoint {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub cpu_percent: f64,
    pub memory_mb: u64,
    pub cost: String,
}

/// Aggregated usage summary
#[derive(Debug, Serialize, Deserialize)]
pub struct UsageSummary {
    pub avg_cpu_percent: f64,
    pub avg_memory_mb: u64,
    pub total_network_bytes: u64,
    pub total_disk_bytes: u64,
    pub avg_gpu_utilization: f64,
    pub duration_secs: u64,
}

/// Detailed rental usage response
#[derive(Debug, Serialize, Deserialize)]
pub struct RentalUsageResponse {
    pub rental_id: String,
    pub data_points: Vec<UsageDataPoint>,
    pub summary: Option<UsageSummary>,
    pub total_cost: String,
}

// Secure Cloud (GPU Aggregator) Types

// Re-export ComputeCategory from basilica-common
pub use basilica_common::types::ComputeCategory;

// Re-export GpuOffering from basilica-aggregator
pub use basilica_aggregator::GpuOffering;

/// Secure cloud rental list item for PS command display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecureCloudRentalListItem {
    /// Rental ID
    pub rental_id: String,

    /// Provider name (datacrunch, hyperstack, lambda, hydrahost)
    pub provider: String,

    /// Provider's instance ID
    pub provider_instance_id: Option<String>,

    /// GPU type (e.g., "h100", "a100")
    pub gpu_type: String,

    /// Number of GPUs
    pub gpu_count: u32,

    /// Instance type identifier
    pub instance_type: String,

    /// Region/location code
    pub location_code: Option<String>,

    /// Deployment status
    pub status: String,

    /// IP address
    pub ip_address: Option<String>,

    /// Hourly cost per hour (total price charged to user)
    pub hourly_cost: f64,

    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,

    /// Stop timestamp
    pub stopped_at: Option<chrono::DateTime<chrono::Utc>>,

    /// SSH connection info
    pub ssh_command: Option<String>,

    /// SSH public key associated with this rental (for local key matching)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ssh_public_key: Option<String>,

    /// Number of vCPU cores
    pub vcpu_count: Option<u32>,

    /// System memory in GB
    pub system_memory_gb: Option<u32>,

    /// Accumulated cost from billing service (actual tracked cost)
    /// None if billing service is unavailable
    pub accumulated_cost: Option<String>,
}

/// List secure cloud rentals response
#[derive(Debug, Serialize, Deserialize)]
pub struct ListSecureCloudRentalsResponse {
    pub rentals: Vec<SecureCloudRentalListItem>,
    pub total_count: usize,
}

/// List secure cloud GPUs response from aggregator
#[derive(Debug, Serialize, Deserialize)]
pub struct ListSecureCloudGpusResponse {
    pub nodes: Vec<GpuOffering>,
    pub count: usize,
}

/// Environment variable for container deployments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvVar {
    pub name: String,
    pub value: String,
}

/// Resource requirements for container deployments
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourceRequirements {
    pub cpu: String,
    pub memory: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpus: Option<GpuRequirementsSpec>,
}

/// GPU requirements specification
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GpuRequirementsSpec {
    pub count: u32,
    pub model: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_cuda_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_gpu_memory_gb: Option<u32>,
}

/// Storage specification
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StorageSpec {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub persistent: Option<PersistentStorageSpec>,
}

/// Persistent storage specification
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PersistentStorageSpec {
    pub enabled: bool,
    pub backend: StorageBackend,
    pub bucket: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub credentials_secret: Option<String>,
    #[serde(default = "default_sync_interval")]
    pub sync_interval_ms: u64,
    #[serde(default = "default_cache_size")]
    pub cache_size_mb: usize,
    #[serde(default = "default_mount_path")]
    pub mount_path: String,
}

/// Storage backend types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StorageBackend {
    R2,
    S3,
    GCS,
}

fn default_sync_interval() -> u64 {
    1000
}

fn default_cache_size() -> usize {
    2048
}

fn default_mount_path() -> String {
    "/data".to_string()
}

fn default_public() -> bool {
    true
}

/// Create deployment request
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateDeploymentRequest {
    pub instance_name: String,
    pub image: String,
    pub replicas: u32,
    pub port: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub args: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub env: Option<std::collections::HashMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resources: Option<ResourceRequirements>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl_seconds: Option<u32>,
    #[serde(default = "default_public")]
    pub public: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub storage: Option<StorageSpec>,
    #[serde(default = "default_enable_billing")]
    pub enable_billing: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub queue_name: Option<String>,
    #[serde(default)]
    pub suspended: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub priority: Option<String>,
}

fn default_enable_billing() -> bool {
    true
}

/// Replica status for deployments
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReplicaStatus {
    pub desired: u32,
    pub ready: u32,
}

/// Pod information
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PodInfo {
    pub name: String,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node: Option<String>,
}

/// Deployment response
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeploymentResponse {
    pub instance_name: String,
    pub user_id: String,
    pub namespace: String,
    pub state: String,
    pub url: String,
    pub replicas: ReplicaStatus,
    pub created_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pods: Option<Vec<PodInfo>>,
}

/// Deployment summary for list responses
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeploymentSummary {
    pub instance_name: String,
    pub state: String,
    pub url: String,
    pub replicas: ReplicaStatus,
    pub created_at: String,
}

/// List deployments response
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeploymentListResponse {
    pub deployments: Vec<DeploymentSummary>,
    pub total: usize,
}

/// Delete deployment response
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeleteDeploymentResponse {
    pub instance_name: String,
    pub state: String,
    pub message: String,
}
