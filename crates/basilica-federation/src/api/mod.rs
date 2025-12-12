//! Multi-cluster API gateway implementation

mod gateway;
pub mod handler;
pub mod router;

pub use gateway::FederationGateway;
pub use handler::FederationHandler;
pub use router::FederationRouter;

use crate::config::FederationConfig;
use crate::discovery::ServiceDiscovery;
use crate::health::HealthAggregator;
use crate::load_balancer::LoadBalancer;
use crate::resource_manager::ResourceManager;
use crate::utils::create_kube_client;
use std::sync::Arc;

/// Federation API server
pub struct FederationApi {
    config: Arc<FederationConfig>,
    gateway: Arc<FederationGateway>,
    discovery: Arc<ServiceDiscovery>,
    health: Arc<HealthAggregator>,
    load_balancer: Arc<LoadBalancer>,
    resource_manager: Arc<ResourceManager>,
}

impl FederationApi {
    /// Create a new federation API server
    pub async fn new(config: FederationConfig) -> crate::Result<Self> {
        let config = Arc::new(config);
        
        let discovery = Arc::new(ServiceDiscovery::new(config.clone()).await?);
        let health = Arc::new(HealthAggregator::new(config.clone()).await?);
        let mut load_balancer = LoadBalancer::new(config.clone()).await?;
        load_balancer.set_health(health.clone());
        let load_balancer = Arc::new(load_balancer);
        let resource_manager = Arc::new(ResourceManager::new(config.clone()).await?);
        
        let gateway = Arc::new(
            FederationGateway::new(
                config.clone(),
                discovery.clone(),
                health.clone(),
                load_balancer.clone(),
                resource_manager.clone(),
            )
            .await?,
        );
        
        Ok(Self {
            config,
            gateway,
            discovery,
            health,
            load_balancer,
            resource_manager,
        })
    }
    
    /// Start the federation API server
    pub async fn start(&self) -> crate::Result<()> {
        tracing::info!(
            federation = %self.config.name,
            "Starting federation API server"
        );
        
        // Start background tasks
        let discovery = self.discovery.clone();
        let health = self.health.clone();
        let resource_manager = self.resource_manager.clone();
        
        tokio::spawn(async move {
            discovery.start().await;
        });
        
        tokio::spawn(async move {
            health.start().await;
        });
        
        tokio::spawn(async move {
            resource_manager.start().await;
        });
        
        // Start gateway
        self.gateway.start().await?;
        
        Ok(())
    }
    
    /// Get gateway reference
    pub fn gateway(&self) -> &Arc<FederationGateway> {
        &self.gateway
    }
    
    /// Get discovery reference
    pub fn discovery(&self) -> &Arc<ServiceDiscovery> {
        &self.discovery
    }
    
    /// Get health aggregator reference
    pub fn health(&self) -> &Arc<HealthAggregator> {
        &self.health
    }
    
    /// Get load balancer reference
    pub fn load_balancer(&self) -> &Arc<LoadBalancer> {
        &self.load_balancer
    }
    
    /// Get resource manager reference
    pub fn resource_manager(&self) -> &Arc<ResourceManager> {
        &self.resource_manager
    }
}

