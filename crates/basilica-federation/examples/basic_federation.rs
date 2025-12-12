//! Basic federation example

use basilica_federation::{FederationApi, FederationConfig};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    // Create federation configuration
    let mut config = FederationConfig::default();
    config.name = "example-federation".to_string();
    
    // Add clusters
    config.clusters.push(basilica_federation::config::ClusterConfig {
        id: "cluster-1".to_string(),
        name: "Cluster 1".to_string(),
        region: "us-east-1".to_string(),
        kubeconfig: "/path/to/kubeconfig1".to_string(),
        api_server: "https://cluster1.example.com:6443".to_string(),
        priority: 100,
        tags: {
            let mut tags = HashMap::new();
            tags.insert("environment".to_string(), "production".to_string());
            tags
        },
        enabled: true,
        capacity: None,
    });
    
    config.clusters.push(basilica_federation::config::ClusterConfig {
        id: "cluster-2".to_string(),
        name: "Cluster 2".to_string(),
        region: "us-west-1".to_string(),
        kubeconfig: "/path/to/kubeconfig2".to_string(),
        api_server: "https://cluster2.example.com:6443".to_string(),
        priority: 90,
        tags: HashMap::new(),
        enabled: true,
        capacity: None,
    });
    
    // Create federation API
    let federation = FederationApi::new(config).await?;
    
    // Start federation
    println!("Starting federation...");
    federation.start().await?;
    
    Ok(())
}

