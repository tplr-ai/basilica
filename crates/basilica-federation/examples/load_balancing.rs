//! Load balancing example

use basilica_federation::{FederationConfig, load_balancer::LoadBalancer};
use basilica_federation::config::LoadBalancingAlgorithm;
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    
    let mut config = FederationConfig::default();
    
    // Configure load balancing algorithm
    config.load_balancer.algorithm = LoadBalancingAlgorithm::RoundRobin;
    config.load_balancer.health_aware = true;
    config.load_balancer.region_aware = false;
    
    // Add clusters
    // ... (cluster configuration)
    
    let config = Arc::new(config);
    let load_balancer = LoadBalancer::new(config).await?;
    
    // Select cluster
    for i in 0..10 {
        if let Some(cluster) = load_balancer.select_cluster().await {
            println!("Request {} routed to cluster: {}", i + 1, cluster.id);
        }
    }
    
    Ok(())
}

