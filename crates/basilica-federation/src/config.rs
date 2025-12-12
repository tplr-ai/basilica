use figment::{providers::Format, Figment};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Configuration for the federation system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationConfig {
    /// Federation name/identifier
    pub name: String,
    
    /// List of federated clusters
    pub clusters: Vec<ClusterConfig>,
    
    /// API gateway configuration
    pub gateway: GatewayConfig,
    
    /// Service discovery configuration
    pub discovery: DiscoveryConfig,
    
    /// Health check configuration
    pub health: HealthConfig,
    
    /// Load balancing configuration
    pub load_balancer: LoadBalancerConfig,
    
    /// Resource management configuration
    pub resource_manager: ResourceManagerConfig,
}

/// Configuration for a single cluster in the federation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterConfig {
    /// Unique cluster identifier
    pub id: String,
    
    /// Human-readable cluster name
    pub name: String,
    
    /// Cluster region/zone
    pub region: String,
    
    /// Kubeconfig path or content
    pub kubeconfig: String,
    
    /// API server endpoint
    pub api_server: String,
    
    /// Cluster priority (higher = preferred)
    pub priority: u32,
    
    /// Cluster tags/labels
    pub tags: HashMap<String, String>,
    
    /// Whether cluster is enabled
    pub enabled: bool,
    
    /// Cluster capacity limits
    pub capacity: Option<ClusterCapacity>,
}

/// Cluster capacity limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterCapacity {
    /// Maximum CPU cores
    pub max_cpu: Option<u64>,
    
    /// Maximum memory in bytes
    pub max_memory: Option<u64>,
    
    /// Maximum GPU count
    pub max_gpu: Option<u32>,
    
    /// Maximum pods
    pub max_pods: Option<u32>,
}

/// API gateway configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayConfig {
    /// Gateway listen address
    pub listen_addr: String,
    
    /// Gateway port
    pub port: u16,
    
    /// Request timeout
    pub request_timeout: Duration,
    
    /// Maximum concurrent requests
    pub max_concurrent_requests: usize,
    
    /// Enable request logging
    pub enable_logging: bool,
    
    /// Rate limiting configuration
    pub rate_limit: Option<RateLimitConfig>,
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Requests per second per client
    pub requests_per_second: u32,
    
    /// Burst size
    pub burst_size: u32,
}

/// Service discovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryConfig {
    /// Discovery refresh interval
    pub refresh_interval: Duration,
    
    /// Service cache TTL
    pub cache_ttl: Duration,
    
    /// Enable cross-cluster discovery
    pub enable_cross_cluster: bool,
    
    /// Service selector labels
    pub service_labels: HashMap<String, String>,
}

/// Health check configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthConfig {
    /// Health check interval
    pub check_interval: Duration,
    
    /// Health check timeout
    pub check_timeout: Duration,
    
    /// Number of consecutive failures before marking unhealthy
    pub failure_threshold: u32,
    
    /// Number of consecutive successes before marking healthy
    pub success_threshold: u32,
    
    /// Enable detailed health metrics
    pub enable_metrics: bool,
}

/// Load balancer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancerConfig {
    /// Load balancing algorithm
    pub algorithm: LoadBalancingAlgorithm,
    
    /// Enable health-aware routing
    pub health_aware: bool,
    
    /// Enable region-aware routing
    pub region_aware: bool,
    
    /// Sticky session configuration
    pub sticky_sessions: Option<StickySessionConfig>,
}

/// Load balancing algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingAlgorithm {
    /// Round-robin
    RoundRobin,
    
    /// Least connections
    LeastConnections,
    
    /// Weighted round-robin
    WeightedRoundRobin,
    
    /// Random
    Random,
    
    /// Geographic proximity
    Geographic,
}

/// Sticky session configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StickySessionConfig {
    /// Session timeout
    pub timeout: Duration,
    
    /// Cookie name for sticky sessions
    pub cookie_name: String,
}

/// Resource manager configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceManagerConfig {
    /// Resource sync interval
    pub sync_interval: Duration,
    
    /// Enable automatic resource distribution
    pub auto_distribute: bool,
    
    /// Resource distribution policy
    pub distribution_policy: DistributionPolicy,
    
    /// Enable resource quotas
    pub enable_quotas: bool,
}

/// Resource distribution policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributionPolicy {
    /// Distribute evenly across clusters
    Even,
    
    /// Distribute based on cluster capacity
    CapacityBased,
    
    /// Distribute based on cluster priority
    PriorityBased,
    
    /// Distribute based on geographic proximity
    Geographic,
}

impl Default for FederationConfig {
    fn default() -> Self {
        Self {
            name: "basilica-federation".to_string(),
            clusters: Vec::new(),
            gateway: GatewayConfig {
                listen_addr: "0.0.0.0".to_string(),
                port: 8080,
                request_timeout: Duration::from_secs(30),
                max_concurrent_requests: 1000,
                enable_logging: true,
                rate_limit: Some(RateLimitConfig {
                    requests_per_second: 100,
                    burst_size: 200,
                }),
            },
            discovery: DiscoveryConfig {
                refresh_interval: Duration::from_secs(30),
                cache_ttl: Duration::from_secs(60),
                enable_cross_cluster: true,
                service_labels: HashMap::new(),
            },
            health: HealthConfig {
                check_interval: Duration::from_secs(10),
                check_timeout: Duration::from_secs(5),
                failure_threshold: 3,
                success_threshold: 2,
                enable_metrics: true,
            },
            load_balancer: LoadBalancerConfig {
                algorithm: LoadBalancingAlgorithm::RoundRobin,
                health_aware: true,
                region_aware: false,
                sticky_sessions: None,
            },
            resource_manager: ResourceManagerConfig {
                sync_interval: Duration::from_secs(60),
                auto_distribute: false,
                distribution_policy: DistributionPolicy::Even,
                enable_quotas: true,
            },
        }
    }
}

impl FederationConfig {
    /// Load configuration from file or environment
    pub fn load() -> Result<Self, figment::Error> {
        Figment::new()
            .merge(figment::providers::Toml::file("federation.toml"))
            .merge(figment::providers::Env::prefixed("FEDERATION_"))
            .extract()
    }
    
    /// Get enabled clusters
    pub fn enabled_clusters(&self) -> Vec<&ClusterConfig> {
        self.clusters.iter().filter(|c| c.enabled).collect()
    }
    
    /// Get cluster by ID
    pub fn get_cluster(&self, id: &str) -> Option<&ClusterConfig> {
        self.clusters.iter().find(|c| c.id == id)
    }
}

