//! Cluster health aggregation

use crate::config::FederationConfig;
use crate::error::{FederationError, Result};
use crate::utils::create_kube_client;
use kube::Client;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use tracing::{debug, info, warn};

/// Health aggregator for federated clusters
pub struct HealthAggregator {
    config: Arc<FederationConfig>,
    clients: HashMap<String, Client>,
    health_status: Arc<dashmap::DashMap<String, ClusterHealth>>,
    failure_counts: Arc<dashmap::DashMap<String, AtomicU32>>,
}

/// Cluster health status
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ClusterHealth {
    pub cluster_id: String,
    pub status: HealthStatus,
    pub last_check: chrono::DateTime<chrono::Utc>,
    pub nodes: NodeHealth,
    pub components: HashMap<String, ComponentHealth>,
}

/// Health status
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

/// Node health information
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NodeHealth {
    pub total: u32,
    pub ready: u32,
    pub not_ready: u32,
}

/// Component health
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ComponentHealth {
    pub status: HealthStatus,
    pub message: String,
}

impl HealthAggregator {
    /// Create a new health aggregator
    pub async fn new(config: Arc<FederationConfig>) -> Result<Self> {
        let mut clients = HashMap::new();
        
        // Initialize Kubernetes clients for each cluster
        for cluster in config.enabled_clusters() {
            match create_kube_client(&cluster.kubeconfig).await {
                Ok(client) => {
                    clients.insert(cluster.id.clone(), client);
                    info!(cluster_id = %cluster.id, "Initialized health check client");
                }
                Err(e) => {
                    warn!(
                        cluster_id = %cluster.id,
                        error = %e,
                        "Failed to initialize health check client"
                    );
                }
            }
        }
        
        let health_status = Arc::new(dashmap::DashMap::new());
        let failure_counts = Arc::new(dashmap::DashMap::new());
        
        Ok(Self {
            config,
            clients,
            health_status,
            failure_counts,
        })
    }
    
    /// Start health checking
    pub async fn start(&self) {
        let config = self.config.clone();
        let clients = self.clients.clone();
        let health_status = self.health_status.clone();
        let failure_counts = self.failure_counts.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.health.check_interval);
            
            loop {
                interval.tick().await;
                
                for cluster in config.enabled_clusters() {
                    if let Some(client) = clients.get(&cluster.id) {
                        match Self::check_cluster_health(client, &cluster.id).await {
                            Ok(health) => {
                                health_status.insert(cluster.id.clone(), health);
                                
                                // Reset failure count on success
                                if let Some(count) = failure_counts.get(&cluster.id) {
                                    count.store(0, Ordering::Relaxed);
                                }
                            }
                            Err(e) => {
                                warn!(
                                    cluster_id = %cluster.id,
                                    error = %e,
                                    "Health check failed"
                                );
                                
                                // Increment failure count
                                let count = failure_counts
                                    .entry(cluster.id.clone())
                                    .or_insert_with(|| AtomicU32::new(0));
                                count.fetch_add(1, Ordering::Relaxed);
                                
                                // Mark as unhealthy if threshold exceeded
                                if count.load(Ordering::Relaxed) >= config.health.failure_threshold {
                                    let health = ClusterHealth {
                                        cluster_id: cluster.id.clone(),
                                        status: HealthStatus::Unhealthy,
                                        last_check: chrono::Utc::now(),
                                        nodes: NodeHealth {
                                            total: 0,
                                            ready: 0,
                                            not_ready: 0,
                                        },
                                        components: HashMap::new(),
                                    };
                                    health_status.insert(cluster.id.clone(), health);
                                }
                            }
                        }
                    }
                }
            }
        });
    }
    
    /// Check cluster health
    async fn check_cluster_health(
        client: &Client,
        cluster_id: &str,
    ) -> Result<ClusterHealth> {
        use k8s_openapi::api::core::v1::Node;
        use kube::api::{Api, ListParams};
        
        let nodes_api: Api<Node> = Api::all(client.clone());
        let nodes = nodes_api.list(&ListParams::default()).await?;
        
        let mut total = 0;
        let mut ready = 0;
        let mut not_ready = 0;
        
        for node in nodes {
            total += 1;
            
            if let Some(status) = node.status {
                if let Some(conditions) = status.conditions {
                    let is_ready = conditions.iter().any(|c| {
                        c.type_ == "Ready" && c.status == "True"
                    });
                    
                    if is_ready {
                        ready += 1;
                    } else {
                        not_ready += 1;
                    }
                }
            }
        }
        
        let status = if not_ready == 0 && total > 0 {
            HealthStatus::Healthy
        } else if not_ready < total / 2 {
            HealthStatus::Degraded
        } else {
            HealthStatus::Unhealthy
        };
        
        Ok(ClusterHealth {
            cluster_id: cluster_id.to_string(),
            status,
            last_check: chrono::Utc::now(),
            nodes: NodeHealth {
                total,
                ready,
                not_ready,
            },
            components: HashMap::new(),
        })
    }
    
    /// Aggregate health from all clusters
    pub async fn aggregate_health(&self) -> Result<HashMap<String, serde_json::Value>> {
        let mut aggregated = HashMap::new();
        
        for cluster in self.config.enabled_clusters() {
            if let Some(health) = self.health_status.get(&cluster.id) {
                aggregated.insert(
                    cluster.id.clone(),
                    serde_json::json!({
                        "status": format!("{:?}", health.status),
                        "last_check": health.last_check,
                        "nodes": health.nodes,
                        "components": health.components,
                    }),
                );
            }
        }
        
        Ok(aggregated)
    }
    
    /// Get health for a specific cluster
    pub async fn get_cluster_health(&self, cluster_id: &str) -> Option<serde_json::Value> {
        self.health_status.get(cluster_id).map(|health| {
            serde_json::json!({
                "cluster_id": health.cluster_id,
                "status": format!("{:?}", health.status),
                "last_check": health.last_check,
                "nodes": health.nodes,
                "components": health.components,
            })
        })
    }
}

