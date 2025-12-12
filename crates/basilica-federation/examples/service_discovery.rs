//! Service discovery example

use basilica_federation::{FederationConfig, discovery::ServiceDiscovery};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    
    let config = Arc::new(FederationConfig::default());
    let discovery = ServiceDiscovery::new(config).await?;
    
    // Discover services
    let services = discovery.discover_services(None).await?;
    println!("Discovered {} services", services.len());
    
    // Discover services in specific namespace
    let namespace = Some("default".to_string());
    let namespace_services = discovery.discover_services(namespace.as_ref()).await?;
    println!("Discovered {} services in default namespace", namespace_services.len());
    
    // Get specific service
    if let Ok(service) = discovery.get_service("my-service", None).await {
        println!("Service details: {:?}", service);
    }
    
    Ok(())
}

