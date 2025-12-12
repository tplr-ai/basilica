//! Cross-cluster service discovery

use crate::config::FederationConfig;
use crate::error::{FederationError, Result};
use crate::utils::create_kube_client;
use kube::Client;
use moka::future::Cache;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, info, warn};

/// Service discovery for federated clusters
pub struct ServiceDiscovery {
    config: Arc<FederationConfig>,
    clients: HashMap<String, Client>,
    cache: Cache<String, Vec<ServiceInfo>>,
}

/// Service information
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ServiceInfo {
    pub name: String,
    pub namespace: String,
    pub cluster_id: String,
    pub endpoints: Vec<String>,
    pub labels: HashMap<String, String>,
    pub annotations: HashMap<String, String>,
}

impl ServiceDiscovery {
    /// Create a new service discovery instance
    pub async fn new(config: Arc<FederationConfig>) -> Result<Self> {
        let mut clients = HashMap::new();
        
        // Initialize Kubernetes clients for each cluster
        for cluster in config.enabled_clusters() {
            match create_kube_client(&cluster.kubeconfig).await {
                Ok(client) => {
                    clients.insert(cluster.id.clone(), client);
                    info!(cluster_id = %cluster.id, "Initialized K8s client");
                }
                Err(e) => {
                    warn!(
                        cluster_id = %cluster.id,
                        error = %e,
                        "Failed to initialize K8s client"
                    );
                }
            }
        }
        
        let cache = Cache::builder()
            .time_to_live(config.discovery.cache_ttl)
            .build();
        
        Ok(Self {
            config,
            clients,
            cache,
        })
    }
    
    /// Create Kubernetes client from kubeconfig
    async fn create_kube_client(kubeconfig: &str) -> Result<Client> {
        // Try to load from path first
        if std::path::Path::new(kubeconfig).exists() {
            let config = kube::Config::from_kubeconfig(&kube::config::KubeconfigOptions {
                cluster: None,
                user: None,
                context: None,
            })
            .await?;
            return Ok(Client::try_from(config)?);
        }
        
        // Try to parse as kubeconfig content
        let kubeconfig_data: kube::config::Kubeconfig = serde_yaml::from_str(kubeconfig)
            .map_err(|e| FederationError::Config(format!("Invalid kubeconfig: {}", e)))?;
        
        let config = kube::Config::from_custom_kubeconfig(kubeconfig_data, &kube::config::KubeconfigOptions {
            cluster: None,
            user: None,
            context: None,
        })
        .await?;
        
        Ok(Client::try_from(config)?)
    }
    
    /// Start the discovery service
    pub async fn start(&self) {
        let config = self.config.clone();
        let clients = self.clients.clone();
        let cache = self.cache.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.discovery.refresh_interval);
            
            loop {
                interval.tick().await;
                
                if let Err(e) = Self::refresh_services(&config, &clients, &cache).await {
                    warn!(error = %e, "Failed to refresh services");
                }
            }
        });
    }
    
    /// Refresh service cache
    async fn refresh_services(
        config: &FederationConfig,
        clients: &HashMap<String, Client>,
        cache: &Cache<String, Vec<ServiceInfo>>,
    ) -> Result<()> {
        for cluster in config.enabled_clusters() {
            if let Some(client) = clients.get(&cluster.id) {
                match Self::discover_cluster_services_internal(client, &cluster.id).await {
                    Ok(services) => {
                        cache.insert(cluster.id.clone(), services).await;
                    }
                    Err(e) => {
                        warn!(
                            cluster_id = %cluster.id,
                            error = %e,
                            "Failed to discover services"
                        );
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Discover services in a cluster
    async fn discover_cluster_services_internal(
        client: &Client,
        cluster_id: &str,
    ) -> Result<Vec<ServiceInfo>> {
        use k8s_openapi::api::core::v1::Service;
        use kube::api::{Api, ListParams};
        
        let services_api: Api<Service> = Api::all(client.clone());
        let services = services_api.list(&ListParams::default()).await?;
        
        let mut service_infos = Vec::new();
        
        for service in services {
            let name = service.metadata.name.clone().unwrap_or_default();
            let namespace = service.metadata.namespace.clone().unwrap_or_default();
            
            let endpoints = service
                .spec
                .as_ref()
                .and_then(|spec| spec.cluster_ip.as_ref())
                .map(|ip| vec![ip.clone()])
                .unwrap_or_default();
            
            let labels = service.metadata.labels.clone().unwrap_or_default();
            let annotations = service.metadata.annotations.clone().unwrap_or_default();
            
            service_infos.push(ServiceInfo {
                name,
                namespace,
                cluster_id: cluster_id.to_string(),
                endpoints,
                labels,
                annotations,
            });
        }
        
        Ok(service_infos)
    }
    
    /// Discover services across all clusters
    pub async fn discover_services(&self, namespace: Option<&String>) -> Result<Vec<ServiceInfo>> {
        let mut all_services = Vec::new();
        
        for cluster in self.config.enabled_clusters() {
            if let Some(cached) = self.cache.get(&cluster.id).await {
                let mut cluster_services = cached;
                
                // Filter by namespace if specified
                if let Some(ns) = namespace {
                    cluster_services.retain(|s| &s.namespace == ns);
                }
                
                all_services.extend(cluster_services);
            }
        }
        
        Ok(all_services)
    }
    
    /// Get a specific service
    pub async fn get_service(
        &self,
        service_name: &str,
        namespace: Option<&String>,
    ) -> Result<serde_json::Value> {
        let services = self.discover_services(namespace).await?;
        
        let service = services
            .iter()
            .find(|s| s.name == service_name)
            .ok_or_else(|| FederationError::Discovery(format!("Service {} not found", service_name)))?;
        
        Ok(serde_json::json!({
            "name": service.name,
            "namespace": service.namespace,
            "cluster_id": service.cluster_id,
            "endpoints": service.endpoints,
            "labels": service.labels,
            "annotations": service.annotations,
        }))
    }
}

