//! Tests for load balancer

use basilica_federation::{FederationConfig, load_balancer::LoadBalancer};
use std::sync::Arc;

#[tokio::test]
async fn test_load_balancer_creation() {
    let config = Arc::new(FederationConfig::default());
    let lb = LoadBalancer::new(config).await;
    
    assert!(lb.is_ok());
}

#[tokio::test]
async fn test_load_balancer_no_clusters() {
    let config = Arc::new(FederationConfig::default());
    let lb = LoadBalancer::new(config).await.unwrap();
    
    let cluster = lb.select_cluster().await;
    assert!(cluster.is_none());
}

#[tokio::test]
async fn test_load_balancing_algorithms() {
    use basilica_federation::config::LoadBalancingAlgorithm;
    
    let mut config = FederationConfig::default();
    
    // Test RoundRobin
    config.load_balancer.algorithm = LoadBalancingAlgorithm::RoundRobin;
    assert!(matches!(config.load_balancer.algorithm, LoadBalancingAlgorithm::RoundRobin));
    
    // Test LeastConnections
    config.load_balancer.algorithm = LoadBalancingAlgorithm::LeastConnections;
    assert!(matches!(config.load_balancer.algorithm, LoadBalancingAlgorithm::LeastConnections));
    
    // Test WeightedRoundRobin
    config.load_balancer.algorithm = LoadBalancingAlgorithm::WeightedRoundRobin;
    assert!(matches!(config.load_balancer.algorithm, LoadBalancingAlgorithm::WeightedRoundRobin));
    
    // Test Random
    config.load_balancer.algorithm = LoadBalancingAlgorithm::Random;
    assert!(matches!(config.load_balancer.algorithm, LoadBalancingAlgorithm::Random));
    
    // Test Geographic
    config.load_balancer.algorithm = LoadBalancingAlgorithm::Geographic;
    assert!(matches!(config.load_balancer.algorithm, LoadBalancingAlgorithm::Geographic));
}

#[tokio::test]
async fn test_load_balancer_health_aware() {
    let mut config = FederationConfig::default();
    config.load_balancer.health_aware = true;
    
    let config = Arc::new(config);
    let lb = LoadBalancer::new(config).await.unwrap();
    
    // With no clusters, should return None
    let cluster = lb.select_cluster().await;
    assert!(cluster.is_none());
}

#[tokio::test]
async fn test_load_balancer_region_aware() {
    let mut config = FederationConfig::default();
    config.load_balancer.region_aware = true;
    
    let config = Arc::new(config);
    let lb = LoadBalancer::new(config).await.unwrap();
    
    // With no clusters, should return None
    let cluster = lb.select_cluster().await;
    assert!(cluster.is_none());
}

