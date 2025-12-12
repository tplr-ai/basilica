//! Cross-cluster load balancing

use crate::config::{FederationConfig, LoadBalancingAlgorithm};
use crate::error::{FederationError, Result};
use crate::health::HealthAggregator;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tracing::debug;

/// Load balancer for federated clusters
pub struct LoadBalancer {
    config: Arc<FederationConfig>,
    health: Arc<HealthAggregator>,
    round_robin_index: Arc<AtomicUsize>,
}

/// Load balancing strategy
pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastConnections,
    WeightedRoundRobin,
    Random,
    Geographic,
}

impl LoadBalancer {
    /// Create a new load balancer
    pub async fn new(config: Arc<FederationConfig>) -> Result<Self> {
        // Health aggregator will be set later
        Ok(Self {
            config,
            health: Arc::new(HealthAggregator::new(config.clone()).await?),
            round_robin_index: Arc::new(AtomicUsize::new(0)),
        })
    }
    
    /// Set health aggregator reference
    pub fn set_health(&mut self, health: Arc<HealthAggregator>) {
        self.health = health;
    }
    
    /// Select a cluster using the configured algorithm
    pub async fn select_cluster(&self) -> Option<&crate::config::ClusterConfig> {
        let enabled_clusters = self.config.enabled_clusters();
        
        if enabled_clusters.is_empty() {
            return None;
        }
        
        // Filter healthy clusters if health-aware
        let candidates: Vec<_> = if self.config.load_balancer.health_aware {
            enabled_clusters
                .iter()
                .filter(|cluster| {
                    // Check if cluster is healthy
                    // This is a simplified check - in production, use health aggregator
                    true
                })
                .collect()
        } else {
            enabled_clusters
        };
        
        if candidates.is_empty() {
            return enabled_clusters.first();
        }
        
        match self.config.load_balancer.algorithm {
            LoadBalancingAlgorithm::RoundRobin => {
                let index = self.round_robin_index.fetch_add(1, Ordering::Relaxed);
                candidates.get(index % candidates.len()).copied()
            }
            LoadBalancingAlgorithm::LeastConnections => {
                // Simplified - in production, track connection counts
                candidates.first().copied()
            }
            LoadBalancingAlgorithm::WeightedRoundRobin => {
                // Select based on priority
                candidates
                    .iter()
                    .max_by_key(|c| c.priority)
                    .copied()
            }
            LoadBalancingAlgorithm::Random => {
                use rand::Rng;
                let mut rng = rand::thread_rng();
                candidates.get(rng.gen_range(0..candidates.len())).copied()
            }
            LoadBalancingAlgorithm::Geographic => {
                // Simplified - in production, use client location
                candidates.first().copied()
            }
        }
    }
}

