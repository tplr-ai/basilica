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

/// Response from starting a rental through the API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RentalResponse {
    /// User-facing rental name
    pub name: String,

    /// Internal rental ID
    pub rental_id: String,

    /// SSH credentials when available
    pub ssh_credentials: Option<String>,

    /// Container details
    pub container_info: basilica_validator::rental::ContainerInfo,
}

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
    /// User-facing rental name
    pub name: String,

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
    /// User-facing rental name
    pub name: Option<String>,

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
    /// Optional user-facing rental name
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

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
    /// User-facing rental name
    pub name: String,

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
        name: String,
        ssh_credentials: Option<String>,
        port_mappings: Option<Vec<basilica_validator::rental::PortMapping>>,
        ssh_public_key: Option<String>,
    ) -> Self {
        Self {
            name,
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
    /// Optional user-facing rental name
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Offering ID from list_gpu_prices endpoint
    pub offering_id: String,

    /// User's registered SSH key ID (NOT the public key string)
    /// Must be a key owned by the authenticated user
    pub ssh_public_key_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecureCloudRentalResponse {
    /// User-facing rental name
    pub name: String,

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
    /// User-facing rental name
    pub name: Option<String>,

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

// Card payment types
//
// Amounts are `u64` in the SDK: the server uses `i64` because Postgres
// BIGINT is signed, but payments are never negative. Keeping the SDK
// unsigned rejects a negative wire value at the deserialize boundary
// instead of silently passing it up the stack.

/// Request body for POST /card-payments/purchases.
#[derive(Debug, Serialize)]
pub struct CreateCardPurchaseRequest {
    pub amount_cents: u64,
}

/// Response from POST /card-payments/purchases.
#[derive(Debug, Serialize, Deserialize)]
pub struct CardPurchaseResponse {
    pub id: String,
    pub checkout_url: String,
    pub requested_amount_cents: u64,
    pub status: CardPurchaseStatus,
}

/// Purchase summary shared by GET /card-payments/purchases/{id} and
/// /card-payments/purchases.
#[derive(Debug, Serialize, Deserialize)]
pub struct CardPurchaseSummary {
    pub id: String,
    pub status: CardPurchaseStatus,
    pub requested_amount_cents: u64,
    pub paid_amount_cents: Option<u64>,
    pub checkout_url: String,
    pub created_at: Option<chrono::DateTime<chrono::Utc>>,
    pub completed_at: Option<chrono::DateTime<chrono::Utc>>,
    pub expires_at: Option<chrono::DateTime<chrono::Utc>>,
    /// Stripe-hosted receipt page for the underlying charge. Present once
    /// `charge.succeeded` has landed for this session.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub receipt_url: Option<String>,
    /// Stripe Invoice id (`in_...`). Present only when the backend was
    /// configured with `[stripe.invoice_creation]` at the time the session
    /// was created.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub invoice_id: Option<String>,
    /// Sequenced human-readable invoice number (e.g. `INV-0001`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub invoice_number: Option<String>,
    /// Stripe-hosted invoice page URL.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hosted_invoice_url: Option<String>,
    /// Direct URL to the invoice PDF.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub invoice_pdf: Option<String>,
}

/// Response from GET /card-payments/purchases.
#[derive(Debug, Serialize, Deserialize)]
pub struct ListCardPurchasesResponse {
    pub purchases: Vec<CardPurchaseSummary>,
}

/// Lifecycle state of a card purchase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CardPurchaseStatus {
    Unspecified,
    Pending,
    Completed,
    Expired,
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
    /// User-facing rental name
    pub name: String,

    /// Rental ID
    pub rental_id: String,

    /// Public availability-zone root (e.g., "cyan", "plum", "opal")
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

/// Phase 4 (ADR-ISSUE-783-NVCT-CDI-ROBUSTNESS §7.2): per-pod per-container
/// state snapshot mirroring kubelet's `Pod.status.containerStatuses[]`.
/// Surfaces the underlying CrashLoopBackOff / ImagePullBackOff / Error
/// state so the CLI / Python SDK can render an honest
/// `Phase: starting (waiting: CrashLoopBackOff x3 restarts)` instead of
/// the misleading `Phase: starting / Elapsed: 0s` while the underlying
/// pods are unhealthy.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ContainerStatusSnapshot {
    pub pod_name: String,
    pub container_name: String,
    /// One of `"running"`, `"waiting"`, `"terminated"`. Empty string for
    /// the unobserved-state edge case (a container whose kubelet has not
    /// yet written any state) -- consumers should treat empty the same as
    /// waiting for display purposes.
    #[serde(default)]
    pub state: String,
    /// K8s waiting / terminated reason verbatim
    /// (e.g. `"CrashLoopBackOff"`, `"ImagePullBackOff"`,
    /// `"OOMKilled"`, `"Completed"`). `None` for running containers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Kubelet human-readable detail. `None` when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    /// Kubelet monotonic restart count for this container.
    #[serde(default)]
    pub restart_count: i32,
}

/// Deployment response
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeploymentResponse {
    pub instance_name: String,
    #[serde(default)]
    pub friendly_name: String,
    pub user_id: String,
    pub namespace: String,
    /// Container image. Populated by basilica-api from the underlying
    /// `UserDeployment.spec.image`.
    #[serde(default)]
    pub image: String,
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
    /// Read-only mirror of `status.distributed` from the `UserDeployment`
    /// CR (issue #431, exposed end-to-end via #449). `None` for
    /// non-distributed UDs; the JSON key is omitted entirely thanks to
    /// `skip_serializing_if` so older API responses without `distributed`
    /// continue to deserialize correctly. Populated by basilica-api's
    /// `extract_distributed_status` helper from
    /// `crates/basilica-api/src/api/routes/deployments/distributed_status.rs`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distributed: Option<DistributedStatus>,
    /// Phase 4 (ADR §7.2): per-pod per-container state snapshot.
    /// Empty when the operator has not yet observed pods (first
    /// reconcile) and when the workload is scaled to zero. Omitted from
    /// the wire when empty (additive, backwards-compatible).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub container_statuses: Vec<ContainerStatusSnapshot>,
    /// Phase 4 (ADR §7.2): sum of container `restartCount` across every
    /// pod. `0` when no restarts have occurred or no pods exist.
    #[serde(default)]
    pub phase_progress: i32,
}

/// Deployment summary for list responses
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeploymentSummary {
    pub instance_name: String,
    #[serde(default)]
    pub friendly_name: String,
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
    #[serde(default)]
    pub friendly_name: String,
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

// =============================================================================
// Distributed-training wire types (SDK arch § 4 / § 8 / § 12).
//
// These mirror the operator's CRD shape exactly. The serde rename rules
// produce the JSON the API expects (camelCase + kebab-case enum tokens,
// matching `crates/basilica-operator/src/crd/user_deployment.rs`). Tests at
// the bottom of this module pin the wire shape so SDK and operator cannot
// drift silently.
//
// Naming convention: Rust types are `Distributed*`-prefixed to keep this
// surface unambiguous in the SDK's flat namespace. The user-facing Python
// dataclasses (in `distributed.py`) drop the prefix per SDK arch § 8.
// =============================================================================

/// Scale-distributed request — patches `spec.distributed.worldSize.target`
/// only via JSON merge-patch on the operator-side CR. Bounds (`min ≤ target
/// ≤ max`) are enforced by the basilica-api endpoint and the operator's
/// admission. SDK arch § 12.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ScaleDistributedRequest {
    pub target: u32,
}

/// Rendezvous backend for torchelastic. `etcd-v2` is the default — the only
/// backend with full elasticity. `c10d` and `static` are escape hatches for
/// users who explicitly opt out of elasticity. SDK arch § 4 footnote on
/// auto-torchrun wrapping.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum DistributedRendezvousBackend {
    #[default]
    EtcdV2,
    C10d,
    Static,
}

/// Rendezvous Pod configuration. `port=None` lets the operator pick the
/// default for the chosen backend (2379 for `etcd-v2`, 29400 for c10d/static).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct DistributedRendezvousSpec {
    #[serde(default)]
    pub backend: DistributedRendezvousBackend,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub port: Option<u16>,
}

/// Availability-zone filter for worker scheduling. Empty `include` = any
/// availability zone root. Match is on Basilica public availability zone root
/// names (e.g. `cyan`, `plum`, `opal`).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct DistributedProviderFilter {
    #[serde(default)]
    pub include: Vec<String>,
    #[serde(default)]
    pub exclude: Vec<String>,
}

impl DistributedProviderFilter {
    /// Print the temporary migration warning for legacy provider input.
    pub fn warn_if_legacy_secure_cloud_providers(&self) {
        for provider in self.include.iter().chain(self.exclude.iter()) {
            warn_if_legacy_secure_cloud_provider(provider);
        }
    }
}

/// Topology spread strategy for ranks-to-nodes assignment. SDK arch § 4.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum DistributedTopologySpreadStrategy {
    Pack,
    #[default]
    ProviderAware,
    RegionAware,
    None,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct DistributedTopologySpread {
    #[serde(default)]
    pub strategy: DistributedTopologySpreadStrategy,
}

/// User-supplied NCCL env merged on top of operator defaults. User values
/// win on collision. SDK arch § 4 / platform doc § 8.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct DistributedNcclSpec {
    #[serde(default)]
    pub env: std::collections::BTreeMap<String, String>,
}

/// Per-UD bench probe mode. `OnStart` schedules a 2-rank NCCL
/// `all_reduce_perf` Job in the user's namespace alongside the worker
/// StatefulSet. Counts against the namespace rank budget. Result lands on
/// `status.distributed.bench` (read via `DistributedTraining.bench`).
/// SDK arch § 7 (tenancy invariant — bench is per-UD, never cross-tenant).
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum DistributedBenchMode {
    #[default]
    Off,
    OnStart,
}

/// Architecture doc § 11.1 placement knob: bench Pod node-placement mode.
///
/// - `Preferred` (default): the bench Pod prefers the worker pair's
///   nodes but falls back to any worker-eligible GPU node when the pair
///   has no spare GPU. Bench always schedules; the resulting BenchResult
///   may not measure the worker pair's link if it falls back.
/// - `Strict`: bench measures the worker pair's link or stays Pending —
///   never silently mismeasures. Architecturally correct for honest
///   measurement on multi-GPU/node hardware.
///
/// Default is `Preferred` (operator-side default; `None` on the wire is
/// treated identically). Wire token is lowercase: `"preferred"` |
/// `"strict"`, matching the operator's serde rename.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum DistributedBenchPlacement {
    #[default]
    Preferred,
    Strict,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct DistributedBenchSpec {
    #[serde(default)]
    pub mode: DistributedBenchMode,
    /// Optional placement override. `None` is treated as `Preferred`
    /// operator-side. Field omitted from the wire when `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub placement: Option<DistributedBenchPlacement>,
}

/// World-size triple: `min ≤ target ≤ max`. The operator clamps the worker
/// StatefulSet replica count to `target` and reconciles toward it.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct DistributedWorldSize {
    pub min: u32,
    pub target: u32,
    pub max: u32,
}

/// `spec.distributed` block on the UserDeployment CR. SDK arch § 4.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DistributedSpec {
    #[serde(default = "default_distributed_enabled")]
    pub enabled: bool,
    pub world_size: DistributedWorldSize,
    #[serde(default)]
    pub rendezvous: DistributedRendezvousSpec,
    #[serde(default)]
    pub provider_filter: DistributedProviderFilter,
    #[serde(default)]
    pub topology_spread: DistributedTopologySpread,
    #[serde(default)]
    pub nccl: DistributedNcclSpec,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bench: Option<DistributedBenchSpec>,
    #[serde(default = "default_distributed_command")]
    pub command: String,
}

fn default_distributed_enabled() -> bool {
    true
}

fn default_distributed_command() -> String {
    "auto".to_string()
}

/// Distributed-training deployment creation request. Serializes to the same
/// `POST /deployments` body as `CreateDeploymentRequest` but `distributed`
/// is required (not optional). Use this when the workload should be a
/// distributed StatefulSet (rather than a single-replica Deployment).
/// SDK arch § 12.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateDistributedDeploymentRequest {
    pub instance_name: String,
    pub image: String,
    /// Ignored when distributed mode is enabled (operator uses
    /// `worldSize.target`); kept for wire compatibility with the API's
    /// CreateDeploymentRequest. Set to `worldSize.target` for clarity.
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
    #[serde(default = "default_enable_billing")]
    pub enable_billing: bool,
    /// Distributed-training spec. Required for this request type.
    pub distributed: DistributedSpec,
}

/// Per-rank pod observation. Read-only, populated on
/// `status.distributed.ranks` by the operator. Mirrors the operator's
/// `RankStatus` (architecture doc § 17.1).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct DistributedRankStatus {
    pub rank: u32,
    pub pod_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node_name: Option<String>,
    /// Basilica public availability zone root name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// `basilica.ai/region` node label value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    /// `Pending | Running | Failed | Succeeded`.
    pub phase: String,
    pub restarts: u32,
}

/// World-size observation. Read-only, populated on
/// `status.distributed.worldSize` by the operator.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct DistributedWorldStatus {
    pub ready: u32,
    pub target: u32,
    pub min: u32,
    pub max: u32,
    pub below_minimum: bool,
}

/// Per-UD bench probe result, read from `status.distributed.bench.result`
/// once the bench Job completes. SDK arch § 8. All bandwidth fields are
/// in GB/s (1 GB = 10^9 bytes), matching the NCCL paper-canonical units.
/// Latency is microseconds at the smallest swept message size.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct DistributedBenchResult {
    /// RFC3339 wall-clock time the probe rank-0 wrote the result.
    pub measured_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub busbw_gbps_p10: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub busbw_gbps_p50: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub busbw_gbps_p90: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub algbw_gbps_p50: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency_us_at_1mib: Option<f64>,
    /// Message sizes swept in bytes.
    #[serde(default)]
    pub size_bytes_swept: Vec<u64>,
    pub probe_node_a: String,
    pub probe_node_b: String,
}

/// Phase 5b (#445) per-rank exit diagnostics, populated when a distributed
/// UD reaches a terminal state (`Succeeded` / `Failed` / `Cancelled`).
/// Mirrors the K8s container `terminated` block so the SDK can surface
/// per-rank exit codes after the operator scales the worker StatefulSet
/// to `replicas: 0` (and the worker pods are no longer queryable). Source:
/// `crd::user_deployment::RankExit`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct DistributedRankExit {
    pub rank: u32,
    pub exit_code: i32,
    /// `Completed` | `Error` | `OOMKilled` | ... or `None` when kubelet did
    /// not record a reason (rare; usually means the container terminated
    /// before kubelet could attribute a cause).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub termination_reason: Option<String>,
    /// Container `restartCount` at the moment the operator observed the
    /// terminated state. `0` for a clean first-iteration exit; non-zero
    /// when kubelet's `restartPolicy=Always` already restarted the
    /// container at least once before the operator caught it.
    pub restart_count: u32,
}

/// Bench probe state. `result` is populated only after a successful run.
/// `last_attempt_outcome` is one of `success | error | timeout` (stable
/// wire tokens defined in the operator).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct DistributedBenchStatus {
    pub mode: DistributedBenchMode,
    /// PR #517 lifecycle phase: `Skipped | Pending | Running | Succeeded | Failed | TimedOut`.
    /// Read by the SDK's `wait_until_bench_complete` opt-in waiter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<DistributedBenchResult>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_attempt_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_attempt_outcome: Option<String>,
}

/// Structured condition on a distributed UD's status. Operator source:
/// `crd::user_deployment::DistributedCondition`. Mirror of
/// `basilica-api::DistributedCondition` (issue #449).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct DistributedCondition {
    /// Stable token, e.g. `Admitted`, `BelowMinimum`, `MeshCapable`.
    #[serde(rename = "type")]
    pub type_: String,
    /// `True` | `False` | `Unknown`.
    pub status: String,
    /// Stable machine-readable token, e.g. `QuotaExceeded`,
    /// `DistributedShapeInvalid`, `RanksReady`.
    pub reason: String,
    pub message: String,
    pub last_transition_time: String,
}

/// Read-only mirror of what the operator rendered for the rendezvous Pod.
/// Operator source: `crd::user_deployment::RendezvousStatus`. Mirror of
/// `basilica-api::RendezvousStatus` (issue #449).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct DistributedRendezvousStatus {
    /// `etcd-v2` | `c10d` | `static`.
    pub backend: String,
    pub port: u16,
    /// Container image rendered for the rendezvous Pod. Empty when
    /// `backend=c10d`.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub image: String,
    /// Monotonic count of rendezvous-Pod restart events the operator has
    /// observed.
    #[serde(default)]
    pub restart_count: u32,
    /// Phase 2.1: RFC3339 timestamp of the last operator-mediated rendezvous
    /// reset (resize, auto-recovery, or rank-loss).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_reset_at: Option<String>,
}

/// A single transition in `worldSize.target`. Operator source:
/// `crd::user_deployment::WorldSizeTransition`. Mirror of
/// `basilica-api::WorldSizeTransition` (issue #449).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct DistributedWorldSizeTransition {
    pub timestamp: String,
    pub target: u32,
    pub ready: u32,
}

/// Record of the most recent `worldSize.target` resize. Operator source:
/// `crd::user_deployment::ResizeRecord`. Mirror of
/// `basilica-api::ResizeRecord` (issue #449).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct DistributedResizeRecord {
    pub from_target: u32,
    pub to_target: u32,
    pub timestamp: String,
    pub reason: String,
}

/// One-shot lifecycle event recorded by the operator. Operator source:
/// `crd::user_deployment::Milestone`. Mirror of `basilica-api::Milestone`
/// (issue #449).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct DistributedMilestone {
    pub name: String,
    pub timestamp: String,
}

/// Phase 2.1 record of one operator-mediated rendezvous reset triggered by
/// stuck-restarting worker ranks. Operator source:
/// `crd::user_deployment::RankLossReset`. Mirror of
/// `basilica-api::RankLossReset` (issue #449).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct DistributedRankLossReset {
    pub timestamp: String,
    pub ranks: Vec<u32>,
}

/// Phase 4b preflight band the operator pulled from the bench collector.
/// Operator source: `crd::user_deployment::PreflightEstimate`. Mirror of
/// `basilica-api::PreflightEstimate` (issue #449).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct DistributedPreflightEstimate {
    pub freshness: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub busbw_gbps_p10: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub busbw_gbps_p50: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub busbw_gbps_p90: Option<f64>,
    #[serde(default)]
    pub sample_count: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_sample_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_queried_at: Option<String>,
}

/// Read-only mirror of `status.distributed`. Populated by the operator
/// after first reconcile. Architecture doc § 17.1. Mirror of
/// `basilica-api::DistributedStatus` -- field-for-field, byte-equal at the
/// JSON layer (issue #449).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct DistributedStatus {
    pub world_size: DistributedWorldStatus,
    #[serde(default)]
    pub ranks: Vec<DistributedRankStatus>,
    /// Structured operator conditions (e.g. `Admitted`, `RanksReady`).
    /// Populated by the operator's reconciler.
    #[serde(default)]
    pub conditions: Vec<DistributedCondition>,
    /// `hub-relay` (Phase 1) | `direct-mesh` | `mixed` (Tier 2+). Reserved
    /// shape; consult only as a hint.
    pub transport: String,
    /// Read-only mirror of the rendezvous Pod the operator rendered.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rendezvous: Option<DistributedRendezvousStatus>,
    /// Append-only history of `worldSize.target` transitions.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub world_size_history: Vec<DistributedWorldSizeTransition>,
    /// Most recent `worldSize.target` resize.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_resize: Option<DistributedResizeRecord>,
    /// One-shot lifecycle events recorded by the operator.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub milestones: Vec<DistributedMilestone>,
    /// Original `worldSize.max` at admission; preserved across resizes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub original_max: Option<u32>,
    /// Phase 2.1: operator-mediated rendezvous resets due to stuck ranks.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rank_loss_resets: Vec<DistributedRankLossReset>,
    /// Phase 4b preflight band pulled from the bench collector.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preflight: Option<DistributedPreflightEstimate>,
    /// Phase 5a per-UD NCCL bench probe state. `None` when bench is off
    /// or the probe has not yet completed its first attempt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bench: Option<DistributedBenchStatus>,
    /// Phase 5a deprecation flag for the legacy preflight surface.
    #[serde(default, skip_serializing_if = "is_false_status")]
    pub preflight_deprecation_warned: bool,
    /// Phase 5b (#445): per-rank exit diagnostics. Empty while the UD is
    /// non-terminal. On transition to `Succeeded` / `Failed` the operator
    /// snapshots each worker pod's container `terminated` state and
    /// persists it here so the SDK can surface them after the worker
    /// StatefulSet has been scaled to `replicas: 0`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rank_exits: Vec<DistributedRankExit>,
}

#[inline]
fn is_false_status(b: &bool) -> bool {
    !*b
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

    /// Public availability-zone root (e.g., "cyan")
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

    /// Availability-zone root (e.g., "cyan")
    pub provider: String,

    /// Provider's internal volume ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_volume_id: Option<String>,

    /// Volume size in GB
    pub size_gb: u32,

    /// Volume type (e.g., "ssd")
    pub volume_type: String,

    /// Basilica region segment (e.g., "us-texas-1", "ca-quebec-1")
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

    /// Availability-zone root (e.g., "cyan")
    pub provider: String,

    /// Basilica region segment (e.g., "us-texas-1", "ca-quebec-1")
    pub region: String,
}

/// Returns true when `provider` is a legacy secure-cloud provider tag that is
/// no longer accepted by the V2 secure-cloud volume API.
pub fn is_legacy_secure_cloud_provider(provider: &str) -> bool {
    matches!(
        provider.trim().to_ascii_lowercase().as_str(),
        "datacrunch"
            | "denvr"
            | "hydrahost"
            | "hyperstack"
            | "lambda"
            | "masscompute"
            | "shadeform"
            | "verda"
    )
}

/// Print the temporary migration warning for explicit legacy provider input.
pub fn warn_if_legacy_secure_cloud_provider(provider: &str) {
    if is_legacy_secure_cloud_provider(provider) {
        eprintln!(
            "Warning: '{}' is a legacy secure-cloud provider tag. Basilica secure-cloud \
             V2 uses public availability-zone names instead; update provider/region \
             inputs to public availability-zone values.",
            provider.trim()
        );
    }
}

/// Attach volume request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttachVolumeRequest {
    /// Rental name to attach the volume to
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rental_name: Option<String>,

    /// Rental ID to attach the volume to
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rental_id: Option<String>,
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
    fn test_create_volume_request_uses_public_az_names() {
        let request = CreateVolumeRequest {
            name: "cache".to_string(),
            description: None,
            size_gb: 100,
            provider: "cyan".to_string(),
            region: "us-texas-1".to_string(),
        };

        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(json["provider"], "cyan");
        assert_eq!(json["region"], "us-texas-1");
        assert!(!json.to_string().contains("hyperstack"));
        assert!(!json.to_string().contains("US-1"));
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

    // -------------------------------------------------------------------------
    // Distributed-training wire-shape tests (SDK arch § 12).
    //
    // These pin the JSON shape that the operator's CRD admission accepts
    // (see `crates/basilica-operator/src/crd/user_deployment.rs::DistributedSpec`).
    // The Phase 5b precursor PR `feat(api): wire spec.distributed through
    // CreateDeploymentRequest` (basilica-backend #421) verified the shape
    // end-to-end against the live cluster on 2026-05-02.
    // -------------------------------------------------------------------------

    fn full_distributed_spec() -> DistributedSpec {
        DistributedSpec {
            enabled: true,
            world_size: DistributedWorldSize {
                min: 4,
                target: 8,
                max: 16,
            },
            rendezvous: DistributedRendezvousSpec {
                backend: DistributedRendezvousBackend::EtcdV2,
                port: None,
            },
            provider_filter: DistributedProviderFilter {
                include: vec!["cyan".to_string(), "plum".to_string()],
                exclude: vec![],
            },
            topology_spread: DistributedTopologySpread {
                strategy: DistributedTopologySpreadStrategy::ProviderAware,
            },
            nccl: DistributedNcclSpec {
                env: {
                    let mut m = std::collections::BTreeMap::new();
                    m.insert("NCCL_DEBUG".to_string(), "INFO".to_string());
                    m
                },
            },
            bench: Some(DistributedBenchSpec {
                mode: DistributedBenchMode::OnStart,
                placement: None,
            }),
            command: "auto".to_string(),
        }
    }

    #[test]
    fn test_distributed_spec_camelcase_full_shape() {
        let spec = full_distributed_spec();
        let v: serde_json::Value = serde_json::to_value(&spec).unwrap();

        assert_eq!(v["enabled"], true);
        assert_eq!(v["command"], "auto");
        assert_eq!(v["worldSize"]["min"], 4);
        assert_eq!(v["worldSize"]["target"], 8);
        assert_eq!(v["worldSize"]["max"], 16);
        assert_eq!(v["rendezvous"]["backend"], "etcd-v2");
        assert!(
            v["rendezvous"].get("port").is_none(),
            "port omitted when None"
        );
        assert_eq!(v["providerFilter"]["include"][0], "cyan");
        assert_eq!(v["topologySpread"]["strategy"], "provider-aware");
        assert_eq!(v["nccl"]["env"]["NCCL_DEBUG"], "INFO");
        assert_eq!(v["bench"]["mode"], "on-start");
    }

    #[test]
    fn test_distributed_rendezvous_backend_kebab_case_tokens() {
        for (variant, token) in [
            (DistributedRendezvousBackend::EtcdV2, "etcd-v2"),
            (DistributedRendezvousBackend::C10d, "c10d"),
            (DistributedRendezvousBackend::Static, "static"),
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            assert_eq!(json, format!("\"{}\"", token));
            let back: DistributedRendezvousBackend = serde_json::from_str(&json).unwrap();
            assert_eq!(back, variant);
        }
    }

    #[test]
    fn test_distributed_topology_strategy_kebab_case_tokens() {
        for (variant, token) in [
            (DistributedTopologySpreadStrategy::Pack, "pack"),
            (
                DistributedTopologySpreadStrategy::ProviderAware,
                "provider-aware",
            ),
            (
                DistributedTopologySpreadStrategy::RegionAware,
                "region-aware",
            ),
            (DistributedTopologySpreadStrategy::None, "none"),
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            assert_eq!(json, format!("\"{}\"", token));
        }
    }

    #[test]
    fn test_distributed_bench_mode_kebab_case_tokens() {
        assert_eq!(
            serde_json::to_string(&DistributedBenchMode::Off).unwrap(),
            "\"off\""
        );
        assert_eq!(
            serde_json::to_string(&DistributedBenchMode::OnStart).unwrap(),
            "\"on-start\""
        );
    }

    #[test]
    fn test_distributed_bench_placement_lowercase_tokens() {
        // Architecture doc § 11.1: wire tokens are lowercase, matching
        // the operator's `BenchPlacement` serde rename. Round-trip via
        // BenchSpec to lock the field name (`"placement"`) too.
        assert_eq!(
            serde_json::to_string(&DistributedBenchPlacement::Preferred).unwrap(),
            "\"preferred\""
        );
        assert_eq!(
            serde_json::to_string(&DistributedBenchPlacement::Strict).unwrap(),
            "\"strict\""
        );

        let spec = DistributedBenchSpec {
            mode: DistributedBenchMode::OnStart,
            placement: Some(DistributedBenchPlacement::Strict),
        };
        let json = serde_json::to_string(&spec).unwrap();
        assert!(
            json.contains("\"placement\":\"strict\""),
            "BenchSpec JSON must contain placement=strict, got: {json}"
        );
        assert!(
            json.contains("\"mode\":\"on-start\""),
            "BenchSpec JSON must keep mode=on-start, got: {json}"
        );
        // Round-trip back.
        let back: DistributedBenchSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back, spec);
    }

    #[test]
    fn test_distributed_bench_spec_placement_omitted_when_none() {
        // Architecture doc § 11.1 wire-compat: pre-placement SDKs emit
        // BenchSpec without the field. Locks `skip_serializing_if =
        // Option::is_none` so a wire upgrade does not break older
        // operators (`None` is interpreted as Preferred operator-side).
        let spec = DistributedBenchSpec {
            mode: DistributedBenchMode::OnStart,
            placement: None,
        };
        let v: serde_json::Value = serde_json::to_value(&spec).unwrap();
        assert_eq!(v["mode"], "on-start");
        assert!(
            v.get("placement").is_none(),
            "placement omitted when None, got: {v}"
        );
        // And accepts an incoming JSON that lacks the field.
        let parsed: DistributedBenchSpec = serde_json::from_str(r#"{"mode":"on-start"}"#).unwrap();
        assert_eq!(parsed.mode, DistributedBenchMode::OnStart);
        assert!(parsed.placement.is_none());
    }

    #[test]
    fn test_distributed_spec_default_command_is_auto() {
        // `command` defaults to "auto" when not present in incoming JSON
        // (matches operator's `DEFAULT_DISTRIBUTED_COMMAND`).
        let json = r#"{
            "enabled": true,
            "worldSize": { "min": 1, "target": 1, "max": 1 }
        }"#;
        let spec: DistributedSpec = serde_json::from_str(json).unwrap();
        assert_eq!(spec.command, "auto");
        assert!(spec.bench.is_none(), "bench is Option, default None");
        assert_eq!(
            spec.rendezvous.backend,
            DistributedRendezvousBackend::EtcdV2
        );
    }

    #[test]
    fn test_distributed_provider_filter_empty_arrays_kept() {
        let spec = DistributedSpec {
            enabled: true,
            world_size: DistributedWorldSize {
                min: 1,
                target: 1,
                max: 1,
            },
            rendezvous: DistributedRendezvousSpec::default(),
            provider_filter: DistributedProviderFilter::default(),
            topology_spread: DistributedTopologySpread::default(),
            nccl: DistributedNcclSpec::default(),
            bench: None,
            command: "auto".to_string(),
        };
        let v = serde_json::to_value(&spec).unwrap();
        // Empty vectors must serialize as empty arrays, not omitted —
        // matches the operator's CRD field defaults.
        assert_eq!(v["providerFilter"]["include"], serde_json::json!([]));
        assert_eq!(v["providerFilter"]["exclude"], serde_json::json!([]));
    }

    #[test]
    fn test_legacy_secure_cloud_provider_detection() {
        assert!(is_legacy_secure_cloud_provider("hyperstack"));
        assert!(is_legacy_secure_cloud_provider(" MASSCOMPUTE "));
        assert!(!is_legacy_secure_cloud_provider("cyan"));
    }

    #[test]
    fn test_create_distributed_deployment_request_round_trip() {
        let req = CreateDistributedDeploymentRequest {
            instance_name: "dlc-test".to_string(),
            image: "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime".to_string(),
            replicas: 8,
            port: 18789,
            command: None,
            args: None,
            env: None,
            resources: None,
            ttl_seconds: Some(86400),
            enable_billing: true,
            distributed: full_distributed_spec(),
        };
        let v = serde_json::to_value(&req).unwrap();
        assert_eq!(v["instanceName"], "dlc-test");
        assert_eq!(v["distributed"]["worldSize"]["target"], 8);
        assert_eq!(v["distributed"]["bench"]["mode"], "on-start");
        // Round-trip
        let back: CreateDistributedDeploymentRequest = serde_json::from_value(v).unwrap();
        assert_eq!(back.instance_name, req.instance_name);
        assert_eq!(back.distributed.world_size.target, 8);
    }

    #[test]
    fn test_scale_distributed_request_shape() {
        let req = ScaleDistributedRequest { target: 12 };
        let json = serde_json::to_string(&req).unwrap();
        assert_eq!(json, r#"{"target":12}"#);
        // Round-trip
        let back: ScaleDistributedRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.target, 12);
    }

    #[test]
    fn test_distributed_bench_result_matches_operator_status_shape() {
        // The exact JSON the operator writes to `status.distributed.bench.result`
        // (per operator's BenchResult struct). Round-tripping this guarantees
        // SDK reads the operator's writes losslessly.
        let json = r#"{
            "measuredAt": "2026-05-02T10:00:00Z",
            "busbwGbpsP10": 0.045,
            "busbwGbpsP50": 0.063,
            "busbwGbpsP90": 0.072,
            "algbwGbpsP50": 0.058,
            "latencyUsAt1mib": 1850.0,
            "sizeBytesSwept": [1048576, 16777216, 268435456],
            "probeNodeA": "basilica-verda-fin-03",
            "probeNodeB": "basilica-verda-fin-04"
        }"#;
        let result: DistributedBenchResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.busbw_gbps_p50, Some(0.063));
        assert_eq!(result.size_bytes_swept.len(), 3);
        assert_eq!(result.probe_node_a, "basilica-verda-fin-03");
        assert_eq!(result.probe_node_b, "basilica-verda-fin-04");
    }

    #[test]
    fn test_distributed_status_camelcase() {
        let status = DistributedStatus {
            world_size: DistributedWorldStatus {
                ready: 6,
                target: 8,
                min: 4,
                max: 16,
                below_minimum: false,
            },
            ranks: vec![DistributedRankStatus {
                rank: 0,
                pod_name: "dlc-test-0".to_string(),
                node_name: Some("basilica-verda-fin-03".to_string()),
                provider: Some("cyan".to_string()),
                region: Some("us-texas-1".to_string()),
                phase: "Running".to_string(),
                restarts: 0,
            }],
            transport: "hub-relay".to_string(),
            ..Default::default()
        };
        let v = serde_json::to_value(&status).unwrap();
        assert_eq!(v["worldSize"]["belowMinimum"], false);
        assert_eq!(v["ranks"][0]["podName"], "dlc-test-0");
        assert_eq!(v["transport"], "hub-relay");
    }

    // -------------------------------------------------------------------------
    // Issue #449 regression tests: status.distributed wired into
    // DeploymentResponse end-to-end. The 4-test floor is the load-bearing
    // gate that keeps the PyO3 deserializer from silently dropping the
    // distributed block again.
    // -------------------------------------------------------------------------

    fn _sample_deployment_with_distributed() -> DeploymentResponse {
        DeploymentResponse {
            instance_name: "dlc-449".to_string(),
            friendly_name: "dlc-449".to_string(),
            user_id: "u-test".to_string(),
            namespace: "u-test".to_string(),
            image: "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime".to_string(),
            state: "running".to_string(),
            url: "https://dlc-449.deployments.basilica.ai".to_string(),
            replicas: ReplicaStatus {
                desired: 2,
                ready: 2,
            },
            created_at: "2026-05-02T10:00:00Z".to_string(),
            updated_at: Some("2026-05-02T10:05:00Z".to_string()),
            pods: None,
            phase: Some("Running".to_string()),
            message: None,
            progress: None,
            share_token: None,
            share_url: None,
            websocket: None,
            public_metadata: false,
            distributed: Some(DistributedStatus {
                world_size: DistributedWorldStatus {
                    ready: 2,
                    target: 2,
                    min: 2,
                    max: 3,
                    below_minimum: false,
                },
                ranks: vec![
                    DistributedRankStatus {
                        rank: 0,
                        pod_name: "dlc-449-0".to_string(),
                        node_name: Some("basilica-verda-fin-03".to_string()),
                        provider: Some("cyan".to_string()),
                        region: Some("us-texas-1".to_string()),
                        phase: "Running".to_string(),
                        restarts: 0,
                    },
                    DistributedRankStatus {
                        rank: 1,
                        pod_name: "dlc-449-1".to_string(),
                        node_name: Some("basilica-verda-fin-04".to_string()),
                        provider: Some("cyan".to_string()),
                        region: Some("us-texas-1".to_string()),
                        phase: "Running".to_string(),
                        restarts: 0,
                    },
                ],
                conditions: vec![],
                transport: "hub-relay".to_string(),
                rendezvous: None,
                world_size_history: vec![],
                last_resize: None,
                milestones: vec![],
                original_max: Some(3),
                rank_loss_resets: vec![],
                preflight: None,
                bench: Some(DistributedBenchStatus {
                    mode: DistributedBenchMode::OnStart,
                    phase: None,
                    started_at: None,
                    completed_at: None,
                    message: None,
                    result: Some(DistributedBenchResult {
                        measured_at: "2026-05-02T10:00:30Z".to_string(),
                        busbw_gbps_p50: Some(0.00897),
                        size_bytes_swept: vec![1_048_576, 16_777_216],
                        probe_node_a: "basilica-verda-fin-03".to_string(),
                        probe_node_b: "basilica-verda-fin-04".to_string(),
                        ..Default::default()
                    }),
                    last_attempt_at: Some("2026-05-02T10:00:35Z".to_string()),
                    last_attempt_outcome: Some("success".to_string()),
                }),
                preflight_deprecation_warned: false,
                rank_exits: vec![],
            }),
            container_statuses: Vec::new(),
            phase_progress: 0,
        }
    }

    #[test]
    fn test_deployment_response_with_distributed_round_trips() {
        // Issue #449: full round-trip with the distributed block populated.
        // This is the load-bearing test: a regression that drops `distributed`
        // from the SDK's serde_json round-trip would fail this assertion.
        let original = _sample_deployment_with_distributed();
        let json = serde_json::to_string(&original).unwrap();
        let parsed: DeploymentResponse = serde_json::from_str(&json).unwrap();
        assert!(parsed.distributed.is_some());
        let d = parsed.distributed.unwrap();
        assert_eq!(d.world_size.ready, 2);
        assert_eq!(d.world_size.target, 2);
        assert_eq!(d.world_size.min, 2);
        assert_eq!(d.world_size.max, 3);
        assert!(!d.world_size.below_minimum);
        assert_eq!(d.ranks.len(), 2);
        assert_eq!(d.ranks[0].pod_name, "dlc-449-0");
        assert_eq!(d.ranks[1].provider.as_deref(), Some("cyan"));
        assert_eq!(d.transport, "hub-relay");
        assert_eq!(d.original_max, Some(3));
        let bench = d.bench.expect("bench present");
        assert_eq!(bench.mode, DistributedBenchMode::OnStart);
        let result = bench.result.expect("bench.result present");
        assert_eq!(result.busbw_gbps_p50, Some(0.00897));
        assert_eq!(result.probe_node_a, "basilica-verda-fin-03");
    }

    #[test]
    fn test_deployment_response_without_distributed_omits_key() {
        // Issue #449 backwards-compat guard: when `distributed` is None, the
        // JSON output must NOT include the `distributed` key. Older API
        // responses without this field continue to deserialize correctly.
        let mut resp = _sample_deployment_with_distributed();
        resp.distributed = None;
        let v = serde_json::to_value(&resp).unwrap();
        assert!(
            v.get("distributed").is_none(),
            "distributed key must be omitted when None (skip_serializing_if), \
             else older API responses without this field break: {v}"
        );
        // And the remaining fields still round-trip cleanly.
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: DeploymentResponse = serde_json::from_str(&json).unwrap();
        assert!(parsed.distributed.is_none());
        assert_eq!(
            parsed.image,
            "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime"
        );
    }

    #[test]
    fn test_distributed_status_json_is_camelcase() {
        // Issue #449 wire-shape lock: the operator emits camelCase
        // (`belowMinimum`, `worldSize`, `lastResize`, `originalMax`,
        // `worldSizeHistory`, `rankLossResets`, `preflightDeprecationWarned`).
        // The SDK's Python facade reads camelCase. A regression toward
        // snake_case would silently break every read property.
        let resp = _sample_deployment_with_distributed();
        let v = serde_json::to_value(&resp).unwrap();
        let d = &v["distributed"];
        // Top-level: required `worldSize`, `ranks`, `transport`.
        assert!(d.get("worldSize").is_some(), "worldSize key missing: {v}");
        assert!(d.get("ranks").is_some(), "ranks key missing: {v}");
        assert!(d.get("transport").is_some(), "transport key missing: {v}");
        // Nested: `belowMinimum` (camelCase, not snake_case).
        assert_eq!(d["worldSize"]["belowMinimum"], false);
        // Optional but present here: `originalMax`. `lastResize`,
        // `worldSizeHistory`, `rankLossResets`, `milestones`,
        // `preflightDeprecationWarned` are skipped when empty/false.
        assert_eq!(d["originalMax"], 3);
        assert!(
            d.get("lastResize").is_none(),
            "lastResize must be omitted when None (got: {d})"
        );
        assert!(
            d.get("worldSizeHistory").is_none(),
            "worldSizeHistory must be omitted when empty (got: {d})"
        );
        assert!(
            d.get("rankLossResets").is_none(),
            "rankLossResets must be omitted when empty (got: {d})"
        );
        assert!(
            d.get("milestones").is_none(),
            "milestones must be omitted when empty (got: {d})"
        );
        assert!(
            d.get("preflightDeprecationWarned").is_none(),
            "preflightDeprecationWarned must be omitted when false (got: {d})"
        );
        // Rank fields: `podName`, `nodeName`.
        assert_eq!(d["ranks"][0]["podName"], "dlc-449-0");
        assert_eq!(d["ranks"][0]["nodeName"], "basilica-verda-fin-03");
        // Bench: `lastAttemptAt`, `lastAttemptOutcome`, `probeNodeA`/`B`.
        let bench = &d["bench"];
        assert_eq!(bench["mode"], "on-start"); // kebab-case enum
        assert_eq!(bench["lastAttemptOutcome"], "success");
        assert_eq!(bench["result"]["busbwGbpsP50"], 0.00897);
        assert_eq!(bench["result"]["probeNodeA"], "basilica-verda-fin-03");
        assert_eq!(bench["result"]["probeNodeB"], "basilica-verda-fin-04");
        assert_eq!(bench["result"]["measuredAt"], "2026-05-02T10:00:30Z");
    }

    #[test]
    fn test_distributed_status_bench_result_round_trips() {
        // Issue #449: bench sub-shape locks the wire contract for
        // `BenchResult`. The SDK's Python facade calls
        // `BenchResult.from_status_dict(raw)` with this exact JSON; a drift
        // in field naming silently produces zero values on the Python side.
        let json = r#"{
            "worldSize": {"ready": 2, "target": 2, "min": 2, "max": 2, "belowMinimum": false},
            "transport": "hub-relay",
            "bench": {
                "mode": "on-start",
                "result": {
                    "measuredAt": "2026-05-02T11:00:00Z",
                    "busbwGbpsP10": 0.0042,
                    "busbwGbpsP50": 0.00897,
                    "busbwGbpsP90": 0.012,
                    "algbwGbpsP50": 0.0085,
                    "latencyUsAt1mib": 1820.5,
                    "sizeBytesSwept": [1048576, 16777216, 268435456],
                    "probeNodeA": "basilica-verda-fin-03",
                    "probeNodeB": "basilica-verda-fin-04"
                },
                "lastAttemptAt": "2026-05-02T11:00:05Z",
                "lastAttemptOutcome": "success"
            }
        }"#;
        let status: DistributedStatus = serde_json::from_str(json).unwrap();
        let bench = status.bench.expect("bench present");
        let result = bench.result.expect("bench.result present");
        assert_eq!(result.busbw_gbps_p50, Some(0.00897));
        assert_eq!(result.busbw_gbps_p10, Some(0.0042));
        assert_eq!(result.latency_us_at_1mib, Some(1820.5));
        assert_eq!(
            result.size_bytes_swept,
            vec![1_048_576, 16_777_216, 268_435_456]
        );
        assert_eq!(result.probe_node_a, "basilica-verda-fin-03");
        assert_eq!(bench.last_attempt_outcome.as_deref(), Some("success"));
    }
}
