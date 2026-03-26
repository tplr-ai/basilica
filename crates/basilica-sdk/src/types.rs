//! Type definitions for the Basilica SDK

use serde::{Deserialize, Serialize};

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

    /// SSH public key associated with this rental (for local key matching)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ssh_public_key: Option<String>,
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
    pub compute_type: String,
    pub vcpu_count: Option<u32>,
    pub system_memory_gb: Option<u32>,
    pub provider: Option<String>,
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

/// Start rental request with GPU-based node selection
#[derive(Debug, Serialize, Deserialize)]
pub struct StartRentalApiRequest {
    /// GPU category: "H100", "A100", "B200", etc. (required)
    pub gpu_category: String,

    /// Number of GPUs required (required)
    pub gpu_count: u32,

    /// Minimum GPU memory in GB (e.g., 80 for 80GB)
    #[serde(default)]
    pub min_memory_gb: Option<u32>,

    /// Maximum acceptable cents/GPU-hour
    pub max_hourly_rate_cents: u32,

    /// Container image to run
    pub container_image: String,

    /// SSH public key
    pub ssh_public_key: String,

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

    /// SSH public key used at rental creation (for local key matching)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ssh_public_key: Option<String>,

    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,

    /// Last update timestamp
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

impl RentalStatusWithSshResponse {
    /// Create from validator response, database SSH credentials, port mappings, and public key
    pub fn from_validator_response(
        response: ValidatorRentalStatusResponse,
        ssh_credentials: Option<String>,
        port_mappings: Option<Vec<basilica_validator::rental::PortMapping>>,
        ssh_public_key: Option<String>,
    ) -> Self {
        Self {
            rental_id: response.rental_id,
            status: response.status,
            node: response.node,
            ssh_credentials,
            port_mappings,
            ssh_public_key,
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

    /// Whether this rental is a spot/preemptible instance
    #[serde(default)]
    pub is_spot: bool,
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

// Re-export ComputeCategory and GpuOffering from basilica-common
pub use basilica_common::types::{ComputeCategory, GpuOffering};

/// Query parameters for filtering GPU price listings
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GpuPriceQuery {
    /// Filter by interconnect type (e.g., "SXM", "SXM5", "PCIe")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interconnect: Option<String>,
    /// Filter by region - accepts geo codes (US, CA, EU, APAC) or region substrings
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    /// Show only spot offerings
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spot_only: Option<bool>,
    /// Exclude spot offerings
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exclude_spot: Option<bool>,
}

/// Secure cloud rental list item for PS command display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecureCloudRentalListItem {
    /// Rental ID
    pub rental_id: String,

    /// Provider name (verda, hyperstack, lambda, hydrahost)
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

    /// Whether this is a VIP rental (managed machine, cannot be stopped by user)
    #[serde(default)]
    pub is_vip: bool,

    /// Whether this rental is a spot/preemptible instance
    #[serde(default)]
    pub is_spot: bool,
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
    pub cpu_request: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_request: Option<String>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interconnect: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub geo: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spot: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub infiniband: Option<bool>,
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

/// Pod spreading mode for controlling how pods are distributed across topology domains.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum SpreadMode {
    /// Best-effort spreading using TopologySpreadConstraints with ScheduleAnyway.
    /// Pods prefer spreading but can be co-located if necessary.
    #[default]
    Preferred,
    /// Strict spreading using TopologySpreadConstraints with DoNotSchedule.
    /// Pods will not schedule if spreading constraints cannot be satisfied.
    Required,
    /// Hard one-pod-per-node using podAntiAffinity with requiredDuringScheduling.
    /// Guarantees each pod runs on a unique node (for unique IP requirements).
    UniqueNodes,
}

fn default_max_skew() -> i32 {
    1
}

fn default_topology_key() -> String {
    "kubernetes.io/hostname".to_string()
}

/// Configuration for pod topology spreading.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct TopologySpreadConfig {
    /// Spreading mode: preferred, required, or unique_nodes.
    #[serde(default)]
    pub mode: SpreadMode,

    /// Maximum allowed difference in pod count between topology domains.
    /// Only used for Preferred and Required modes (ignored for UniqueNodes).
    /// Range: 1-10, default: 1.
    #[serde(default = "default_max_skew")]
    pub max_skew: i32,

    /// Topology key for spreading (default: kubernetes.io/hostname).
    #[serde(default = "default_topology_key")]
    pub topology_key: String,
}

impl Default for TopologySpreadConfig {
    fn default() -> Self {
        Self {
            mode: SpreadMode::default(),
            max_skew: default_max_skew(),
            topology_key: default_topology_key(),
        }
    }
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub health_check: Option<HealthCheckConfig>,
    #[serde(default = "default_enable_billing")]
    pub enable_billing: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub queue_name: Option<String>,
    #[serde(default)]
    pub suspended: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub priority: Option<String>,
    /// Optional topology spreading configuration.
    /// Controls how pod replicas are distributed across nodes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub topology_spread: Option<TopologySpreadConfig>,
    /// Optional WebSocket configuration for long-lived connections.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub websocket: Option<WebSocketConfig>,
    /// Opt-in to exposing non-sensitive metadata publicly for validator verification.
    #[serde(default)]
    pub public_metadata: bool,
}

fn default_enable_billing() -> bool {
    true
}

fn default_ws_idle_timeout() -> u32 {
    1800
}

/// WebSocket configuration for deployments.
/// `idle_timeout_seconds` valid range: 60-3600 (1 minute to 1 hour).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct WebSocketConfig {
    pub enabled: bool,
    #[serde(default = "default_ws_idle_timeout")]
    pub idle_timeout_seconds: u32,
}

impl Default for WebSocketConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            idle_timeout_seconds: default_ws_idle_timeout(),
        }
    }
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress: Option<DeploymentProgress>,
    /// Share token for private deployments (only returned on creation).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub share_token: Option<String>,
    /// Shareable URL with token query parameter for private deployments.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub share_url: Option<String>,
    /// WebSocket configuration if enabled for this deployment.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub websocket: Option<WebSocketConfig>,
    /// Whether public metadata enrollment is enabled for this deployment.
    #[serde(default)]
    pub public_metadata: bool,
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
    /// Whether deployment is publicly accessible (no token required).
    #[serde(default = "default_public")]
    pub public: bool,
    /// WebSocket configuration if enabled for this deployment.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub websocket: Option<WebSocketConfig>,
    /// Whether public metadata enrollment is enabled for this deployment.
    #[serde(default)]
    pub public_metadata: bool,
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

/// Response for POST /deployments/{name}/share-token
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct RegenerateShareTokenResponse {
    /// Raw token value. Only returned once, cannot be retrieved later.
    pub token: String,
    /// Full shareable URL with token as query parameter.
    pub share_url: String,
}

/// Response for GET /deployments/{name}/share-token
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ShareTokenStatusResponse {
    /// Whether a share token exists for this deployment.
    pub exists: bool,
}

/// Response for DELETE /deployments/{name}/share-token
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct DeleteShareTokenResponse {
    /// Whether a token was revoked.
    pub revoked: bool,
}

/// Request to enroll or unenroll a deployment in public metadata exposure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrollMetadataRequest {
    pub enabled: bool,
}

/// Response for metadata enrollment status.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EnrollMetadataResponse {
    pub public_metadata: bool,
}

/// Public deployment metadata visible without authentication.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PublicDeploymentMetadataResponse {
    pub instance_name: String,
    pub image: String,
    pub image_tag: String,
    pub id: String,
    pub uptime_seconds: u64,
    pub replicas: ReplicaStatus,
    pub state: String,
}

/// Deployment event from Kubernetes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentEvent {
    #[serde(rename = "type")]
    pub event_type: String,
    pub reason: String,
    pub message: String,
    pub count: Option<i32>,
    pub last_timestamp: Option<String>,
}

/// Deployment events response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentEventsResponse {
    pub events: Vec<DeploymentEvent>,
}

/// Health check configuration for deployments
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HealthCheckConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub liveness: Option<ProbeConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub readiness: Option<ProbeConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub startup: Option<ProbeConfig>,
}

/// HTTP probe configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProbeConfig {
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub port: Option<u16>,
    #[serde(default = "default_initial_delay")]
    pub initial_delay_seconds: u32,
    #[serde(default = "default_period")]
    pub period_seconds: u32,
    #[serde(default = "default_timeout")]
    pub timeout_seconds: u32,
    #[serde(default = "default_failure_threshold")]
    pub failure_threshold: u32,
}

fn default_initial_delay() -> u32 {
    30
}

fn default_period() -> u32 {
    10
}

fn default_timeout() -> u32 {
    5
}

fn default_failure_threshold() -> u32 {
    3
}

/// Scale deployment request
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScaleDeploymentRequest {
    pub replicas: u32,
}

/// Deployment progress information
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeploymentProgress {
    pub current_step: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub percentage: Option<f64>,
    pub elapsed_seconds: u64,
}

/// Result of waiting for a deployment to become ready
#[derive(Debug, Clone)]
pub enum WaitResult {
    /// Deployment is ready with all replicas running
    Ready(Box<DeploymentResponse>),
    /// Deployment failed with an error message
    Failed { reason: String },
    /// Wait timed out before deployment became ready
    Timeout {
        last_state: String,
        last_phase: Option<String>,
    },
}

/// Options for waiting on a deployment
#[derive(Debug, Clone)]
pub struct WaitOptions {
    /// Maximum time to wait in seconds (default: 300)
    pub timeout_secs: u64,
    /// Interval between status checks in seconds (default: 5)
    pub poll_interval_secs: u64,
}

impl Default for WaitOptions {
    fn default() -> Self {
        Self {
            timeout_secs: 300,
            poll_interval_secs: 5,
        }
    }
}

impl WaitOptions {
    /// Create wait options with a specific timeout
    pub fn with_timeout(timeout_secs: u64) -> Self {
        Self {
            timeout_secs,
            ..Default::default()
        }
    }
}

// ============================================================================
// CPU-Only Secure Cloud Types
// ============================================================================

/// CPU-only offering from secure cloud providers (no GPU)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuOffering {
    /// Unique offering identifier
    pub id: String,

    /// Provider name (e.g., "hyperstack")
    pub provider: String,

    /// Number of vCPU cores
    pub vcpu_count: u32,

    /// System memory in GB
    pub system_memory_gb: u32,

    /// Storage in GB
    pub storage_gb: u32,

    /// Region/location code
    pub region: String,

    /// Hourly rate in USD (flat rate, not per-GPU)
    pub hourly_rate: String,

    /// Whether the offering is currently available
    pub availability: bool,

    /// When this offering data was fetched
    pub fetched_at: chrono::DateTime<chrono::Utc>,
}

/// Response for listing CPU-only offerings
#[derive(Debug, Serialize, Deserialize)]
pub struct ListCpuOfferingsResponse {
    /// List of available CPU offerings
    pub nodes: Vec<CpuOffering>,

    /// Total count of offerings
    pub count: usize,
}

// ============================================================================
// Volume Management Types
// ============================================================================

/// Volume status enum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VolumeStatus {
    Pending,
    Available,
    Attached,
    Deleting,
    Error,
}

impl std::fmt::Display for VolumeStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VolumeStatus::Pending => write!(f, "Pending"),
            VolumeStatus::Available => write!(f, "Available"),
            VolumeStatus::Attached => write!(f, "Attached"),
            VolumeStatus::Deleting => write!(f, "Deleting"),
            VolumeStatus::Error => write!(f, "Error"),
        }
    }
}

/// Volume response from API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeResponse {
    /// Unique volume identifier
    pub volume_id: String,

    /// User-friendly volume name
    pub name: String,

    /// Optional description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// Cloud provider (e.g., "hyperstack")
    pub provider: String,

    /// Provider's internal volume ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_volume_id: Option<String>,

    /// Volume size in GB
    pub size_gb: u32,

    /// Volume type (e.g., "ssd")
    pub volume_type: String,

    /// Region code (e.g., "US-1", "CANADA-1")
    pub region: String,

    /// Current volume status
    pub status: VolumeStatus,

    /// Rental ID if attached
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rental_id: Option<String>,

    /// Estimated hourly cost in USD
    pub estimated_hourly_cost: Option<f64>,

    /// Accumulated cost from billing service (actual tracked cost)
    /// None if billing service is unavailable
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accumulated_cost: Option<String>,

    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// List volumes response
#[derive(Debug, Serialize, Deserialize)]
pub struct ListVolumesResponse {
    /// List of volumes
    pub volumes: Vec<VolumeResponse>,

    /// Total count
    pub total_count: u32,
}

/// Create volume request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateVolumeRequest {
    /// Volume name (unique per user, case-insensitive)
    pub name: String,

    /// Optional description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// Size in GB (1-10240)
    pub size_gb: u32,

    /// Cloud provider (e.g., "hyperstack")
    pub provider: String,

    /// Region code (e.g., "US-1", "CANADA-1")
    pub region: String,
}

/// Attach volume request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttachVolumeRequest {
    /// Rental ID to attach the volume to
    pub rental_id: String,
}

/// Response for volume attach/detach operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeOperationResponse {
    /// Volume ID
    pub volume_id: String,

    /// New volume status
    pub status: VolumeStatus,

    /// Human-readable message
    pub message: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spread_mode_default() {
        assert_eq!(SpreadMode::default(), SpreadMode::Preferred);
    }

    #[test]
    fn test_spread_mode_serialization() {
        assert_eq!(
            serde_json::to_string(&SpreadMode::Preferred).unwrap(),
            "\"preferred\""
        );
        assert_eq!(
            serde_json::to_string(&SpreadMode::Required).unwrap(),
            "\"required\""
        );
        assert_eq!(
            serde_json::to_string(&SpreadMode::UniqueNodes).unwrap(),
            "\"unique_nodes\""
        );
    }

    #[test]
    fn test_spread_mode_deserialization() {
        assert_eq!(
            serde_json::from_str::<SpreadMode>("\"preferred\"").unwrap(),
            SpreadMode::Preferred
        );
        assert_eq!(
            serde_json::from_str::<SpreadMode>("\"required\"").unwrap(),
            SpreadMode::Required
        );
        assert_eq!(
            serde_json::from_str::<SpreadMode>("\"unique_nodes\"").unwrap(),
            SpreadMode::UniqueNodes
        );
    }

    #[test]
    fn test_topology_spread_config_default() {
        let config = TopologySpreadConfig::default();
        assert_eq!(config.mode, SpreadMode::Preferred);
        assert_eq!(config.max_skew, 1);
        assert_eq!(config.topology_key, "kubernetes.io/hostname");
    }

    #[test]
    fn test_topology_spread_config_serialization() {
        let config = TopologySpreadConfig {
            mode: SpreadMode::UniqueNodes,
            max_skew: 2,
            topology_key: "topology.kubernetes.io/zone".to_string(),
        };
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"mode\":\"unique_nodes\""));
        assert!(json.contains("\"maxSkew\":2"));
        assert!(json.contains("\"topologyKey\":\"topology.kubernetes.io/zone\""));
    }

    #[test]
    fn test_topology_spread_config_deserialization() {
        let json = r#"{"mode":"unique_nodes","maxSkew":3,"topologyKey":"kubernetes.io/hostname"}"#;
        let config: TopologySpreadConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.mode, SpreadMode::UniqueNodes);
        assert_eq!(config.max_skew, 3);
        assert_eq!(config.topology_key, "kubernetes.io/hostname");
    }

    #[test]
    fn test_topology_spread_config_deserialization_with_defaults() {
        let json = r#"{}"#;
        let config: TopologySpreadConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.mode, SpreadMode::Preferred);
        assert_eq!(config.max_skew, 1);
        assert_eq!(config.topology_key, "kubernetes.io/hostname");
    }

    #[test]
    fn test_create_deployment_request_without_topology_spread() {
        let request = CreateDeploymentRequest {
            instance_name: "test".to_string(),
            image: "nginx:latest".to_string(),
            replicas: 1,
            port: 80,
            command: None,
            args: None,
            env: None,
            resources: None,
            ttl_seconds: None,
            public: true,
            storage: None,
            health_check: None,
            enable_billing: true,
            queue_name: None,
            suspended: false,
            priority: None,
            topology_spread: None,
            websocket: None,
            public_metadata: false,
        };
        let json = serde_json::to_string(&request).unwrap();
        assert!(!json.contains("topologySpread"));
    }

    #[test]
    fn test_create_deployment_request_with_topology_spread() {
        let request = CreateDeploymentRequest {
            instance_name: "test".to_string(),
            image: "nginx:latest".to_string(),
            replicas: 3,
            port: 80,
            command: None,
            args: None,
            env: None,
            resources: None,
            ttl_seconds: None,
            public: true,
            storage: None,
            health_check: None,
            enable_billing: true,
            queue_name: None,
            suspended: false,
            priority: None,
            topology_spread: Some(TopologySpreadConfig {
                mode: SpreadMode::UniqueNodes,
                max_skew: 1,
                topology_key: "kubernetes.io/hostname".to_string(),
            }),
            websocket: None,
            public_metadata: false,
        };
        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"topologySpread\""));
        assert!(json.contains("\"mode\":\"unique_nodes\""));
    }

    #[test]
    fn test_spread_mode_deserialization_invalid_value() {
        let result = serde_json::from_str::<SpreadMode>("\"invalid_mode\"");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("unknown variant"));
    }

    #[test]
    fn test_spread_mode_deserialization_wrong_type() {
        let result = serde_json::from_str::<SpreadMode>("123");
        assert!(result.is_err());
    }

    #[test]
    fn test_topology_spread_config_deserialization_invalid_mode() {
        let json = r#"{"mode":"bad_mode","maxSkew":1,"topologyKey":"kubernetes.io/hostname"}"#;
        let result = serde_json::from_str::<TopologySpreadConfig>(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_topology_spread_config_equality() {
        let config1 = TopologySpreadConfig {
            mode: SpreadMode::UniqueNodes,
            max_skew: 1,
            topology_key: "kubernetes.io/hostname".to_string(),
        };
        let config2 = TopologySpreadConfig {
            mode: SpreadMode::UniqueNodes,
            max_skew: 1,
            topology_key: "kubernetes.io/hostname".to_string(),
        };
        let config3 = TopologySpreadConfig {
            mode: SpreadMode::Required,
            max_skew: 1,
            topology_key: "kubernetes.io/hostname".to_string(),
        };
        assert_eq!(config1, config2);
        assert_ne!(config1, config3);
    }

    #[test]
    fn test_spread_mode_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(SpreadMode::Preferred);
        set.insert(SpreadMode::Required);
        set.insert(SpreadMode::UniqueNodes);
        assert_eq!(set.len(), 3);
        assert!(set.contains(&SpreadMode::Preferred));
    }

    // Share Token Tests

    #[test]
    fn test_regenerate_share_token_response_serialization() {
        let response = RegenerateShareTokenResponse {
            token: "abc123def456".to_string(),
            share_url: "https://api.example.com/d/my-app?token=abc123def456".to_string(),
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("token"));
        assert!(json.contains("shareUrl")); // camelCase
    }

    #[test]
    fn test_regenerate_share_token_response_deserialization() {
        let json = r#"{"token":"abc123","shareUrl":"https://example.com"}"#;
        let response: RegenerateShareTokenResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.token, "abc123");
        assert_eq!(response.share_url, "https://example.com");
    }

    #[test]
    fn test_share_token_status_response() {
        let exists = ShareTokenStatusResponse { exists: true };
        let not_exists = ShareTokenStatusResponse { exists: false };

        assert!(exists.exists);
        assert!(!not_exists.exists);
        assert_ne!(exists, not_exists);
    }

    #[test]
    fn test_share_token_status_response_serialization() {
        let response = ShareTokenStatusResponse { exists: true };
        let json = serde_json::to_string(&response).unwrap();
        assert_eq!(json, r#"{"exists":true}"#);
    }

    #[test]
    fn test_delete_share_token_response() {
        let revoked = DeleteShareTokenResponse { revoked: true };
        let not_revoked = DeleteShareTokenResponse { revoked: false };

        assert!(revoked.revoked);
        assert!(!not_revoked.revoked);
        assert_ne!(revoked, not_revoked);
    }

    #[test]
    fn test_delete_share_token_response_serialization() {
        let response = DeleteShareTokenResponse { revoked: true };
        let json = serde_json::to_string(&response).unwrap();
        assert_eq!(json, r#"{"revoked":true}"#);
    }

    #[test]
    fn test_deployment_summary_public_field_default() {
        // Test that public field deserializes with default when missing
        let json = r#"{
            "instanceName": "my-app",
            "state": "Running",
            "url": "https://example.com",
            "replicas": {"desired": 1, "ready": 1},
            "createdAt": "2024-01-01T00:00:00Z"
        }"#;
        let summary: DeploymentSummary = serde_json::from_str(json).unwrap();
        assert!(summary.public); // default_public() returns true
    }

    #[test]
    fn test_deployment_summary_public_field_explicit() {
        let json = r#"{
            "instanceName": "my-app",
            "state": "Running",
            "url": "https://example.com",
            "replicas": {"desired": 1, "ready": 1},
            "createdAt": "2024-01-01T00:00:00Z",
            "public": false
        }"#;
        let summary: DeploymentSummary = serde_json::from_str(json).unwrap();
        assert!(!summary.public);
    }

    #[test]
    fn test_deployment_response_share_token_fields() {
        let json = r#"{
            "instanceName": "my-app",
            "userId": "user123",
            "namespace": "u-user123",
            "state": "Running",
            "url": "https://example.com",
            "replicas": {"desired": 1, "ready": 1},
            "createdAt": "2024-01-01T00:00:00Z",
            "shareToken": "abc123",
            "shareUrl": "https://example.com?token=abc123"
        }"#;
        let response: DeploymentResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.share_token, Some("abc123".to_string()));
        assert_eq!(
            response.share_url,
            Some("https://example.com?token=abc123".to_string())
        );
    }

    #[test]
    fn test_deployment_response_share_token_fields_optional() {
        let json = r#"{
            "instanceName": "my-app",
            "userId": "user123",
            "namespace": "u-user123",
            "state": "Running",
            "url": "https://example.com",
            "replicas": {"desired": 1, "ready": 1},
            "createdAt": "2024-01-01T00:00:00Z"
        }"#;
        let response: DeploymentResponse = serde_json::from_str(json).unwrap();
        assert!(response.share_token.is_none());
        assert!(response.share_url.is_none());
    }

    #[test]
    fn test_websocket_config_default_idle_timeout() {
        let json = r#"{"enabled":true}"#;
        let config: WebSocketConfig = serde_json::from_str(json).unwrap();
        assert!(config.enabled);
        assert_eq!(config.idle_timeout_seconds, 1800);
    }

    #[test]
    fn test_websocket_config_default() {
        let config = WebSocketConfig::default();
        assert!(config.enabled);
        assert_eq!(config.idle_timeout_seconds, 1800);
    }

    #[test]
    fn test_websocket_config_custom_idle_timeout() {
        let config = WebSocketConfig {
            enabled: true,
            idle_timeout_seconds: 3600,
        };
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"enabled\":true"));
        assert!(json.contains("\"idleTimeoutSeconds\":3600"));
    }

    #[test]
    fn test_websocket_config_serialization_roundtrip() {
        let config = WebSocketConfig {
            enabled: true,
            idle_timeout_seconds: 120,
        };
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: WebSocketConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, deserialized);
    }

    #[test]
    fn test_create_deployment_request_without_websocket() {
        let request = CreateDeploymentRequest {
            instance_name: "test".to_string(),
            image: "nginx:latest".to_string(),
            replicas: 1,
            port: 80,
            command: None,
            args: None,
            env: None,
            resources: None,
            ttl_seconds: None,
            public: true,
            storage: None,
            health_check: None,
            enable_billing: true,
            queue_name: None,
            suspended: false,
            priority: None,
            topology_spread: None,
            websocket: None,
            public_metadata: false,
        };
        let json = serde_json::to_string(&request).unwrap();
        assert!(!json.contains("\"websocket\""));
    }

    #[test]
    fn test_create_deployment_request_with_websocket() {
        let request = CreateDeploymentRequest {
            instance_name: "ws-app".to_string(),
            image: "node:18".to_string(),
            replicas: 1,
            port: 3000,
            command: None,
            args: None,
            env: None,
            resources: None,
            ttl_seconds: None,
            public: true,
            storage: None,
            health_check: None,
            enable_billing: true,
            queue_name: None,
            suspended: false,
            priority: None,
            topology_spread: None,
            websocket: Some(WebSocketConfig {
                enabled: true,
                idle_timeout_seconds: 1800,
            }),
            public_metadata: false,
        };
        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"websocket\""));
        assert!(json.contains("\"enabled\":true"));
        assert!(json.contains("\"idleTimeoutSeconds\":1800"));
    }

    #[test]
    fn test_deployment_response_websocket_optional() {
        let json = r#"{
            "instanceName": "my-app",
            "userId": "user123",
            "namespace": "u-user123",
            "state": "Running",
            "url": "https://example.com",
            "replicas": {"desired": 1, "ready": 1},
            "createdAt": "2024-01-01T00:00:00Z"
        }"#;
        let response: DeploymentResponse = serde_json::from_str(json).unwrap();
        assert!(response.websocket.is_none());
    }

    #[test]
    fn test_deployment_summary_websocket_optional() {
        let json = r#"{
            "instanceName": "my-app",
            "state": "Running",
            "url": "https://example.com",
            "replicas": {"desired": 1, "ready": 1},
            "createdAt": "2024-01-01T00:00:00Z"
        }"#;
        let summary: DeploymentSummary = serde_json::from_str(json).unwrap();
        assert!(summary.websocket.is_none());
    }

    // =========================================================================
    // GPU Flavour Preferences - Serde contract tests
    // =========================================================================

    #[test]
    fn test_gpu_price_query_serde_roundtrip() {
        let query = GpuPriceQuery {
            interconnect: Some("SXM5".to_string()),
            region: Some("EU".to_string()),
            spot_only: None,
            exclude_spot: Some(true),
        };
        let json = serde_json::to_string(&query).unwrap();
        let deserialized: GpuPriceQuery = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.interconnect.as_deref(), Some("SXM5"));
        assert_eq!(deserialized.region.as_deref(), Some("EU"));
        assert!(deserialized.spot_only.is_none());
        assert_eq!(deserialized.exclude_spot, Some(true));
    }

    #[test]
    fn test_gpu_price_query_serde_skip_none_fields() {
        let query = GpuPriceQuery {
            interconnect: Some("PCIe".to_string()),
            region: None,
            spot_only: None,
            exclude_spot: None,
        };
        let json = serde_json::to_string(&query).unwrap();
        assert!(json.contains("\"interconnect\":\"PCIe\""));
        assert!(!json.contains("\"region\""));
        assert!(!json.contains("\"spot_only\""));
        assert!(!json.contains("\"exclude_spot\""));
    }

    #[test]
    fn test_gpu_price_query_deserialize_empty_object() {
        let json = "{}";
        let query: GpuPriceQuery = serde_json::from_str(json).unwrap();
        assert!(query.interconnect.is_none());
        assert!(query.region.is_none());
        assert!(query.spot_only.is_none());
        assert!(query.exclude_spot.is_none());
    }

    #[test]
    fn test_gpu_requirements_spec_with_flavour_fields() {
        let spec = GpuRequirementsSpec {
            count: 2,
            model: vec!["H100".to_string()],
            min_cuda_version: None,
            min_gpu_memory_gb: None,
            interconnect: Some("SXM".to_string()),
            geo: Some("US".to_string()),
            spot: Some(true),
            infiniband: None,
        };
        let json = serde_json::to_string(&spec).unwrap();
        assert!(json.contains("\"interconnect\":\"SXM\""));
        assert!(json.contains("\"geo\":\"US\""));
        assert!(json.contains("\"spot\":true"));
        assert!(!json.contains("\"infiniband\""));
    }

    #[test]
    fn test_gpu_requirements_spec_serde_roundtrip() {
        let spec = GpuRequirementsSpec {
            count: 1,
            model: vec!["A100".to_string()],
            min_cuda_version: Some("12.0".to_string()),
            min_gpu_memory_gb: Some(80),
            interconnect: Some("SXM5".to_string()),
            geo: Some("EU".to_string()),
            spot: Some(false),
            infiniband: Some(true),
        };
        let json = serde_json::to_string(&spec).unwrap();
        let deserialized: GpuRequirementsSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.count, 1);
        assert_eq!(deserialized.model, vec!["A100"]);
        assert_eq!(deserialized.interconnect.as_deref(), Some("SXM5"));
        assert_eq!(deserialized.geo.as_deref(), Some("EU"));
        assert_eq!(deserialized.spot, Some(false));
        assert_eq!(deserialized.infiniband, Some(true));
    }

    #[test]
    fn test_gpu_requirements_spec_backward_compat() {
        // Existing JSON without flavour fields should deserialize with None
        let json = r#"{
            "count": 4,
            "model": ["H100"],
            "minCudaVersion": "12.0",
            "minGpuMemoryGb": 80
        }"#;
        let spec: GpuRequirementsSpec = serde_json::from_str(json).unwrap();
        assert_eq!(spec.count, 4);
        assert_eq!(spec.model, vec!["H100"]);
        assert!(spec.interconnect.is_none());
        assert!(spec.geo.is_none());
        assert!(spec.spot.is_none());
        assert!(spec.infiniband.is_none());
    }

    #[test]
    fn test_gpu_requirements_spec_camel_case_serialization() {
        let spec = GpuRequirementsSpec {
            count: 1,
            model: vec!["H100".to_string()],
            min_cuda_version: None,
            min_gpu_memory_gb: Some(40),
            interconnect: Some("PCIe".to_string()),
            geo: None,
            spot: None,
            infiniband: None,
        };
        let json = serde_json::to_string(&spec).unwrap();
        // Should use camelCase (minGpuMemoryGb, not min_gpu_memory_gb)
        assert!(json.contains("\"minGpuMemoryGb\":40"));
        assert!(json.contains("\"interconnect\":\"PCIe\""));
    }
}
