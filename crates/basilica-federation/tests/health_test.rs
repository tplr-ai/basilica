//! Tests for health aggregation

use basilica_federation::{FederationConfig, health::HealthAggregator};
use std::sync::Arc;

#[tokio::test]
async fn test_health_aggregator_creation() {
    let config = Arc::new(FederationConfig::default());
    let aggregator = HealthAggregator::new(config).await;
    
    assert!(aggregator.is_ok());
}

#[tokio::test]
async fn test_health_status_enum() {
    use basilica_federation::health::HealthStatus;
    
    let healthy = HealthStatus::Healthy;
    let degraded = HealthStatus::Degraded;
    let unhealthy = HealthStatus::Unhealthy;
    let unknown = HealthStatus::Unknown;
    
    // Test that all variants exist
    match healthy {
        HealthStatus::Healthy => assert!(true),
        _ => assert!(false),
    }
    
    match degraded {
        HealthStatus::Degraded => assert!(true),
        _ => assert!(false),
    }
    
    match unhealthy {
        HealthStatus::Unhealthy => assert!(true),
        _ => assert!(false),
    }
    
    match unknown {
        HealthStatus::Unknown => assert!(true),
        _ => assert!(false),
    }
}

#[tokio::test]
async fn test_node_health_structure() {
    use basilica_federation::health::NodeHealth;
    
    let node_health = NodeHealth {
        total: 5,
        ready: 4,
        not_ready: 1,
    };
    
    assert_eq!(node_health.total, 5);
    assert_eq!(node_health.ready, 4);
    assert_eq!(node_health.not_ready, 1);
}

#[tokio::test]
async fn test_health_aggregation_empty_clusters() {
    let config = Arc::new(FederationConfig::default());
    let aggregator = HealthAggregator::new(config).await.unwrap();
    
    let health = aggregator.aggregate_health().await.unwrap();
    assert_eq!(health.len(), 0);
}

#[tokio::test]
async fn test_get_cluster_health_nonexistent() {
    let config = Arc::new(FederationConfig::default());
    let aggregator = HealthAggregator::new(config).await.unwrap();
    
    let health = aggregator.get_cluster_health("nonexistent").await;
    assert!(health.is_none());
}

