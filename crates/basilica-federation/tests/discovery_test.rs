//! Tests for service discovery

use basilica_federation::{FederationConfig, discovery::ServiceDiscovery};
use std::collections::HashMap;
use std::sync::Arc;

#[tokio::test]
async fn test_service_discovery_creation() {
    let config = Arc::new(FederationConfig::default());
    let discovery = ServiceDiscovery::new(config).await;
    
    // Should succeed even with no clusters configured
    assert!(discovery.is_ok());
}

#[tokio::test]
async fn test_service_discovery_empty_clusters() {
    let config = Arc::new(FederationConfig::default());
    let discovery = ServiceDiscovery::new(config).await.unwrap();
    
    let services = discovery.discover_services(None).await.unwrap();
    assert_eq!(services.len(), 0);
}

#[tokio::test]
async fn test_service_discovery_namespace_filtering() {
    let config = Arc::new(FederationConfig::default());
    let discovery = ServiceDiscovery::new(config).await.unwrap();
    
    let namespace = Some("default".to_string());
    let services = discovery.discover_services(namespace.as_ref()).await.unwrap();
    
    // With no clusters, should return empty
    assert_eq!(services.len(), 0);
}

#[tokio::test]
async fn test_service_info_structure() {
    use basilica_federation::discovery::ServiceInfo;
    
    let service = ServiceInfo {
        name: "test-service".to_string(),
        namespace: "default".to_string(),
        cluster_id: "cluster-1".to_string(),
        endpoints: vec!["10.0.0.1".to_string()],
        labels: HashMap::new(),
        annotations: HashMap::new(),
    };
    
    assert_eq!(service.name, "test-service");
    assert_eq!(service.namespace, "default");
    assert_eq!(service.cluster_id, "cluster-1");
    assert_eq!(service.endpoints.len(), 1);
}

