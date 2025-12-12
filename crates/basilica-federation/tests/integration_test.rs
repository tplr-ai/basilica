//! Integration tests for federation

use basilica_federation::{FederationConfig, FederationApi};
use std::collections::HashMap;

#[tokio::test]
async fn test_federation_config_loading() {
    let config = FederationConfig::default();
    assert_eq!(config.name, "basilica-federation");
    assert!(config.clusters.is_empty());
}

#[tokio::test]
async fn test_federation_config_with_clusters() {
    let mut config = FederationConfig::default();
    
    config.clusters.push(basilica_federation::config::ClusterConfig {
        id: "test-cluster".to_string(),
        name: "Test Cluster".to_string(),
        region: "us-east-1".to_string(),
        kubeconfig: "/tmp/kubeconfig".to_string(),
        api_server: "https://test.example.com:6443".to_string(),
        priority: 100,
        tags: HashMap::new(),
        enabled: true,
        capacity: None,
    });
    
    assert_eq!(config.clusters.len(), 1);
    assert_eq!(config.enabled_clusters().len(), 1);
}

#[tokio::test]
async fn test_federation_config_get_cluster() {
    let mut config = FederationConfig::default();
    
    config.clusters.push(basilica_federation::config::ClusterConfig {
        id: "cluster-1".to_string(),
        name: "Cluster 1".to_string(),
        region: "us-east-1".to_string(),
        kubeconfig: "/tmp/kubeconfig".to_string(),
        api_server: "https://test.example.com:6443".to_string(),
        priority: 100,
        tags: HashMap::new(),
        enabled: true,
        capacity: None,
    });
    
    let cluster = config.get_cluster("cluster-1");
    assert!(cluster.is_some());
    assert_eq!(cluster.unwrap().id, "cluster-1");
    
    let cluster_not_found = config.get_cluster("nonexistent");
    assert!(cluster_not_found.is_none());
}

#[tokio::test]
async fn test_federation_config_enabled_clusters() {
    let mut config = FederationConfig::default();
    
    config.clusters.push(basilica_federation::config::ClusterConfig {
        id: "cluster-1".to_string(),
        name: "Cluster 1".to_string(),
        region: "us-east-1".to_string(),
        kubeconfig: "/tmp/kubeconfig".to_string(),
        api_server: "https://test.example.com:6443".to_string(),
        priority: 100,
        tags: HashMap::new(),
        enabled: true,
        capacity: None,
    });
    
    config.clusters.push(basilica_federation::config::ClusterConfig {
        id: "cluster-2".to_string(),
        name: "Cluster 2".to_string(),
        region: "us-west-1".to_string(),
        kubeconfig: "/tmp/kubeconfig2".to_string(),
        api_server: "https://test2.example.com:6443".to_string(),
        priority: 90,
        tags: HashMap::new(),
        enabled: false,
        capacity: None,
    });
    
    let enabled = config.enabled_clusters();
    assert_eq!(enabled.len(), 1);
    assert_eq!(enabled[0].id, "cluster-1");
}

#[tokio::test]
async fn test_load_balancing_algorithm_config() {
    use basilica_federation::config::LoadBalancingAlgorithm;
    
    let mut config = FederationConfig::default();
    config.load_balancer.algorithm = LoadBalancingAlgorithm::RoundRobin;
    
    match config.load_balancer.algorithm {
        LoadBalancingAlgorithm::RoundRobin => assert!(true),
        _ => assert!(false, "Algorithm should be RoundRobin"),
    }
}

#[tokio::test]
async fn test_health_config_defaults() {
    let config = FederationConfig::default();
    
    assert_eq!(config.health.failure_threshold, 3);
    assert_eq!(config.health.success_threshold, 2);
    assert!(config.health.enable_metrics);
}

#[tokio::test]
async fn test_discovery_config_defaults() {
    let config = FederationConfig::default();
    
    assert_eq!(config.discovery.refresh_interval.as_secs(), 30);
    assert_eq!(config.discovery.cache_ttl.as_secs(), 60);
    assert!(config.discovery.enable_cross_cluster);
}

#[tokio::test]
async fn test_gateway_config_defaults() {
    let config = FederationConfig::default();
    
    assert_eq!(config.gateway.port, 8080);
    assert_eq!(config.gateway.listen_addr, "0.0.0.0");
    assert_eq!(config.gateway.request_timeout.as_secs(), 30);
    assert_eq!(config.gateway.max_concurrent_requests, 1000);
}

#[tokio::test]
async fn test_resource_manager_config_defaults() {
    let config = FederationConfig::default();
    
    assert_eq!(config.resource_manager.sync_interval.as_secs(), 60);
    assert!(!config.resource_manager.auto_distribute);
    assert!(config.resource_manager.enable_quotas);
}

