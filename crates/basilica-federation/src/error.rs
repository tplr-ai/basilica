use thiserror::Error;

/// Federation error types
#[derive(Debug, Error)]
pub enum FederationError {
    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),
    
    /// Cluster not found
    #[error("Cluster not found: {0}")]
    ClusterNotFound(String),
    
    /// Kubernetes API error
    #[error("Kubernetes API error: {0}")]
    Kube(#[from] kube::Error),
    
    /// HTTP error
    #[error("HTTP error: {0}")]
    Http(#[from] http::Error),
    
    /// Request error
    #[error("Request error: {0}")]
    Request(#[from] reqwest::Error),
    
    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    /// Service discovery error
    #[error("Service discovery error: {0}")]
    Discovery(String),
    
    /// Health check error
    #[error("Health check error: {0}")]
    Health(String),
    
    /// Load balancing error
    #[error("Load balancing error: {0}")]
    LoadBalancing(String),
    
    /// Resource management error
    #[error("Resource management error: {0}")]
    ResourceManagement(String),
    
    /// Timeout error
    #[error("Operation timed out: {0}")]
    Timeout(String),
    
    /// Invalid cluster state
    #[error("Invalid cluster state: {0}")]
    InvalidState(String),
}

/// Result type for federation operations
pub type Result<T> = std::result::Result<T, FederationError>;

