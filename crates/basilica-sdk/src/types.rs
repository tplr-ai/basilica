//! Type definitions for the Basilica SDK

use serde::{Deserialize, Serialize};

// Re-export types from basilica-validator that are used by the client
pub use basilica_validator::api::types::{
    AvailabilityInfo, AvailableExecutor, CpuSpec, ExecutorDetails, GpuRequirements, GpuSpec,
    ListAvailableExecutorsQuery, ListAvailableExecutorsResponse, LogQuery, NetworkSpeedInfo,
    RentCapacityRequest, RentCapacityResponse, RentalListItem, RentalStatus,
    RentalStatusResponse as ValidatorRentalStatusResponse, SshAccess, TerminateRentalRequest,
};

// Re-export LocationProfile for SDK consumers
pub use basilica_common::LocationProfile;

// Re-export rental-specific types from validator
pub use basilica_validator::api::rental_routes::{
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
    pub executor_id: String,
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
}

/// API list rentals response with GPU information
#[derive(Debug, Serialize, Deserialize)]
pub struct ApiListRentalsResponse {
    pub rentals: Vec<ApiRentalListItem>,
    pub total_count: usize,
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
}

/// Executor selection strategy for rental requests
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ExecutorSelection {
    /// Select a specific executor by ID
    ExecutorId { executor_id: String },
    /// Select executor with exact GPU configuration (exact count match)
    ExactGpuConfiguration { gpu_requirements: GpuRequirements },
}

/// Start rental request with flexible executor selection
#[derive(Debug, Serialize, Deserialize)]
pub struct StartRentalApiRequest {
    /// How to select the executor for this rental
    pub executor_selection: ExecutorSelection,

    /// Container image to run
    pub container_image: String,

    /// SSH public key for authentication
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

    /// Executor details
    pub executor: ExecutorDetails,

    /// SSH credentials (from database, not validator)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ssh_credentials: Option<String>,

    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,

    /// Last update timestamp
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

impl RentalStatusWithSshResponse {
    /// Create from validator response and database SSH credentials
    pub fn from_validator_response(
        response: ValidatorRentalStatusResponse,
        ssh_credentials: Option<String>,
    ) -> Self {
        Self {
            rental_id: response.rental_id,
            status: response.status,
            executor: response.executor,
            ssh_credentials,
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
