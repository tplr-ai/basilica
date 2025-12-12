//! Request handlers for federation API

use crate::config::FederationConfig;
use crate::discovery::ServiceDiscovery;
use crate::error::{FederationError, Result};
use crate::health::HealthAggregator;
use crate::load_balancer::LoadBalancer;
use crate::resource_manager::ResourceManager;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::Json,
};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, error, info};

/// Federation API handler
pub struct FederationHandler {
    config: Arc<FederationConfig>,
    discovery: Arc<ServiceDiscovery>,
    health: Arc<HealthAggregator>,
    load_balancer: Arc<LoadBalancer>,
    resource_manager: Arc<ResourceManager>,
}

impl FederationHandler {
    /// Create a new handler
    pub fn new(
        config: Arc<FederationConfig>,
        discovery: Arc<ServiceDiscovery>,
        health: Arc<HealthAggregator>,
        load_balancer: Arc<LoadBalancer>,
        resource_manager: Arc<ResourceManager>,
    ) -> Self {
        Self {
            config,
            discovery,
            health,
            load_balancer,
            resource_manager,
        }
    }
    
    /// Handle cluster list request
    pub async fn handle_list_clusters(&self) -> Result<Json<serde_json::Value>> {
        let clusters: Vec<_> = self.config.enabled_clusters()
            .iter()
            .map(|c| {
                serde_json::json!({
                    "id": c.id,
                    "name": c.name,
                    "region": c.region,
                    "priority": c.priority,
                    "tags": c.tags,
                    "enabled": c.enabled,
                })
            })
            .collect();
        
        Ok(Json(serde_json::json!({
            "clusters": clusters,
            "total": clusters.len(),
        })))
    }
    
    /// Handle cluster details request
    pub async fn handle_get_cluster(&self, cluster_id: &str) -> Result<Json<serde_json::Value>, StatusCode> {
        let cluster = self.config.get_cluster(cluster_id)
            .ok_or_else(|| {
                error!(cluster_id = %cluster_id, "Cluster not found");
                StatusCode::NOT_FOUND
            })?;
        
        let health = self.health.get_cluster_health(cluster_id).await
            .unwrap_or_else(|| serde_json::json!({
                "status": "Unknown",
                "last_check": null,
            }));
        
        Ok(Json(serde_json::json!({
            "id": cluster.id,
            "name": cluster.name,
            "region": cluster.region,
            "priority": cluster.priority,
            "tags": cluster.tags,
            "enabled": cluster.enabled,
            "api_server": cluster.api_server,
            "capacity": cluster.capacity,
            "health": health,
        })))
    }
    
    /// Handle service discovery request
    pub async fn handle_discover_services(
        &self,
        namespace: Option<&String>,
        labels: Option<&HashMap<String, String>>,
    ) -> Result<Json<serde_json::Value>> {
        let services = self.discovery.discover_services(namespace).await?;
        
        let filtered_services: Vec<_> = if let Some(filter_labels) = labels {
            services.into_iter()
                .filter(|s| {
                    filter_labels.iter().all(|(k, v)| {
                        s.labels.get(k).map(|sv| sv == v).unwrap_or(false)
                    })
                })
                .map(|s| serde_json::json!({
                    "name": s.name,
                    "namespace": s.namespace,
                    "cluster_id": s.cluster_id,
                    "endpoints": s.endpoints,
                    "labels": s.labels,
                }))
                .collect()
        } else {
            services.into_iter()
                .map(|s| serde_json::json!({
                    "name": s.name,
                    "namespace": s.namespace,
                    "cluster_id": s.cluster_id,
                    "endpoints": s.endpoints,
                    "labels": s.labels,
                }))
                .collect()
        };
        
        Ok(Json(serde_json::json!({
            "services": filtered_services,
            "total": filtered_services.len(),
        })))
    }
    
    /// Handle resource list request
    pub async fn handle_list_resources(
        &self,
        resource_type: &str,
        namespace: Option<&str>,
    ) -> Result<Json<serde_json::Value>, StatusCode> {
        match resource_type {
            "pods" => {
                let namespace = namespace.unwrap_or("default");
                match self.resource_manager.list_pods(namespace).await {
                    Ok(pods) => Ok(Json(serde_json::json!({
                        "resources": pods,
                        "type": "pods",
                        "namespace": namespace,
                    }))),
                    Err(e) => {
                        error!(error = %e, "Failed to list pods");
                        Err(StatusCode::INTERNAL_SERVER_ERROR)
                    }
                }
            }
            _ => {
                error!(resource_type = %resource_type, "Unsupported resource type");
                Err(StatusCode::BAD_REQUEST)
            }
        }
    }
    
    /// Handle cluster selection for load balancing
    pub async fn handle_select_cluster(
        &self,
        prefer_region: Option<&str>,
    ) -> Result<Json<serde_json::Value>, StatusCode> {
        let cluster = self.load_balancer.select_cluster().await
            .ok_or_else(|| {
                error!("No available clusters");
                StatusCode::SERVICE_UNAVAILABLE
            })?;
        
        // If region preference specified, try to match
        if let Some(pref_region) = prefer_region {
            if let Some(region_cluster) = self.config.enabled_clusters()
                .iter()
                .find(|c| c.region == pref_region && c.enabled) {
                return Ok(Json(serde_json::json!({
                    "cluster_id": region_cluster.id,
                    "name": region_cluster.name,
                    "region": region_cluster.region,
                    "api_server": region_cluster.api_server,
                })));
            }
        }
        
        Ok(Json(serde_json::json!({
            "cluster_id": cluster.id,
            "name": cluster.name,
            "region": cluster.region,
            "api_server": cluster.api_server,
        })))
    }
    
    /// Handle federation statistics
    pub async fn handle_statistics(&self) -> Result<Json<serde_json::Value>> {
        let clusters = self.config.enabled_clusters();
        let health_status = self.health.aggregate_health().await?;
        
        let mut healthy_count = 0;
        let mut degraded_count = 0;
        let mut unhealthy_count = 0;
        
        for (_, status) in &health_status {
            if let Some(status_str) = status.get("status").and_then(|s| s.as_str()) {
                match status_str {
                    s if s.contains("Healthy") => healthy_count += 1,
                    s if s.contains("Degraded") => degraded_count += 1,
                    s if s.contains("Unhealthy") => unhealthy_count += 1,
                    _ => {}
                }
            }
        }
        
        Ok(Json(serde_json::json!({
            "total_clusters": clusters.len(),
            "healthy_clusters": healthy_count,
            "degraded_clusters": degraded_count,
            "unhealthy_clusters": unhealthy_count,
            "clusters": health_status,
        })))
    }
}

