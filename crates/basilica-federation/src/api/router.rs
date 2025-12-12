//! Router configuration for federation API

use crate::api::gateway::GatewayState;
use crate::api::handler::FederationHandler;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use std::collections::HashMap;
use std::sync::Arc;

/// Federation API router builder
pub struct FederationRouter;

impl FederationRouter {
    /// Build the complete API router
    pub fn build(
        state: GatewayState,
    ) -> Router {
        Router::new()
            // Health and status endpoints
            .route("/health", get(Self::health))
            .route("/ready", get(Self::ready))
            .route("/live", get(Self::live))
            .route("/statistics", get(Self::statistics))
            
            // Cluster management endpoints
            .route("/api/v1/clusters", get(Self::list_clusters))
            .route("/api/v1/clusters/:id", get(Self::get_cluster))
            .route("/api/v1/clusters/:id/health", get(Self::cluster_health))
            .route("/api/v1/clusters/:id/select", post(Self::select_cluster))
            
            // Service discovery endpoints
            .route("/api/v1/services", get(Self::list_services))
            .route("/api/v1/services/:name", get(Self::get_service))
            .route("/api/v1/services/:name/endpoints", get(Self::get_service_endpoints))
            
            // Resource endpoints
            .route("/api/v1/resources/:type", get(Self::list_resources))
            .route("/api/v1/namespaces/:namespace/:type", get(Self::list_namespace_resources))
            .route("/api/v1/namespaces/:namespace/pods/:pod", get(Self::get_pod))
            
            // Load balancer endpoints
            .route("/api/v1/loadbalancer/select", post(Self::select_cluster_for_request))
            .route("/api/v1/loadbalancer/stats", get(Self::loadbalancer_stats))
            
            // Metrics endpoint
            .route("/metrics", get(Self::metrics))
            
            .with_state(state)
    }
    
    /// Health check endpoint
    async fn health(State(state): State<GatewayState>) -> Result<Json<serde_json::Value>, StatusCode> {
        match state.health.aggregate_health().await {
            Ok(health) => Ok(Json(serde_json::json!({
                "status": "healthy",
                "clusters": health,
            }))),
            Err(_) => Err(StatusCode::SERVICE_UNAVAILABLE),
        }
    }
    
    /// Readiness check endpoint
    async fn ready(State(state): State<GatewayState>) -> Result<Json<serde_json::Value>, StatusCode> {
        let enabled_clusters = state.config.enabled_clusters();
        if enabled_clusters.is_empty() {
            return Err(StatusCode::SERVICE_UNAVAILABLE);
        }
        Ok(Json(serde_json::json!({
            "status": "ready",
            "clusters_configured": enabled_clusters.len(),
        })))
    }
    
    /// Liveness check endpoint
    async fn live() -> Json<serde_json::Value> {
        Json(serde_json::json!({
            "status": "alive",
        }))
    }
    
    /// Statistics endpoint
    async fn statistics(State(state): State<GatewayState>) -> Result<Json<serde_json::Value>, StatusCode> {
        let handler = FederationHandler::new(
            state.config.clone(),
            state.discovery.clone(),
            state.health.clone(),
            state.load_balancer.clone(),
            state.resource_manager.clone(),
        );
        handler.handle_statistics().await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
    }
    
    /// List clusters endpoint
    async fn list_clusters(State(state): State<GatewayState>) -> Result<Json<serde_json::Value>, StatusCode> {
        let handler = FederationHandler::new(
            state.config.clone(),
            state.discovery.clone(),
            state.health.clone(),
            state.load_balancer.clone(),
            state.resource_manager.clone(),
        );
        handler.handle_list_clusters().await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
    }
    
    /// Get cluster endpoint
    async fn get_cluster(
        State(state): State<GatewayState>,
        Path(cluster_id): Path<String>,
    ) -> Result<Json<serde_json::Value>, StatusCode> {
        let handler = FederationHandler::new(
            state.config.clone(),
            state.discovery.clone(),
            state.health.clone(),
            state.load_balancer.clone(),
            state.resource_manager.clone(),
        );
        handler.handle_get_cluster(&cluster_id).await
    }
    
    /// Get cluster health endpoint
    async fn cluster_health(
        State(state): State<GatewayState>,
        Path(cluster_id): Path<String>,
    ) -> Result<Json<serde_json::Value>, StatusCode> {
        let health = state.health.get_cluster_health(&cluster_id).await
            .ok_or_else(|| StatusCode::NOT_FOUND)?;
        Ok(Json(health))
    }
    
    /// Select cluster endpoint
    async fn select_cluster(
        State(state): State<GatewayState>,
        Path(cluster_id): Path<String>,
    ) -> Result<Json<serde_json::Value>, StatusCode> {
        let handler = FederationHandler::new(
            state.config.clone(),
            state.discovery.clone(),
            state.health.clone(),
            state.load_balancer.clone(),
            state.resource_manager.clone(),
        );
        handler.handle_select_cluster(None).await
    }
    
    /// List services endpoint
    async fn list_services(
        State(state): State<GatewayState>,
        Query(params): Query<HashMap<String, String>>,
    ) -> Result<Json<serde_json::Value>, StatusCode> {
        let namespace = params.get("namespace");
        let handler = FederationHandler::new(
            state.config.clone(),
            state.discovery.clone(),
            state.health.clone(),
            state.load_balancer.clone(),
            state.resource_manager.clone(),
        );
        handler.handle_discover_services(namespace, None).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
    }
    
    /// Get service endpoint
    async fn get_service(
        State(state): State<GatewayState>,
        Path(service_name): Path<String>,
        Query(params): Query<HashMap<String, String>>,
    ) -> Result<Json<serde_json::Value>, StatusCode> {
        let namespace = params.get("namespace");
        match state.discovery.get_service(&service_name, namespace).await {
            Ok(service) => Ok(Json(service)),
            Err(_) => Err(StatusCode::NOT_FOUND),
        }
    }
    
    /// Get service endpoints
    async fn get_service_endpoints(
        State(state): State<GatewayState>,
        Path(service_name): Path<String>,
        Query(params): Query<HashMap<String, String>>,
    ) -> Result<Json<serde_json::Value>, StatusCode> {
        let namespace = params.get("namespace");
        match state.discovery.get_service(&service_name, namespace).await {
            Ok(service) => {
                if let Some(endpoints) = service.get("endpoints") {
                    Ok(Json(serde_json::json!({
                        "service": service_name,
                        "endpoints": endpoints,
                    })))
                } else {
                    Err(StatusCode::NOT_FOUND)
                }
            }
            Err(_) => Err(StatusCode::NOT_FOUND),
        }
    }
    
    /// List resources endpoint
    async fn list_resources(
        State(state): State<GatewayState>,
        Path(resource_type): Path<String>,
        Query(params): Query<HashMap<String, String>>,
    ) -> Result<Json<serde_json::Value>, StatusCode> {
        let namespace = params.get("namespace");
        let handler = FederationHandler::new(
            state.config.clone(),
            state.discovery.clone(),
            state.health.clone(),
            state.load_balancer.clone(),
            state.resource_manager.clone(),
        );
        handler.handle_list_resources(&resource_type, namespace.as_deref()).await
    }
    
    /// List namespace resources endpoint
    async fn list_namespace_resources(
        State(state): State<GatewayState>,
        Path((namespace, resource_type)): Path<(String, String)>,
    ) -> Result<Json<serde_json::Value>, StatusCode> {
        let handler = FederationHandler::new(
            state.config.clone(),
            state.discovery.clone(),
            state.health.clone(),
            state.load_balancer.clone(),
            state.resource_manager.clone(),
        );
        handler.handle_list_resources(&resource_type, Some(&namespace)).await
    }
    
    /// Get pod endpoint
    async fn get_pod(
        State(state): State<GatewayState>,
        Path((namespace, pod_name)): Path<(String, String)>,
    ) -> Result<Json<serde_json::Value>, StatusCode> {
        match state.resource_manager.get_pod(&namespace, &pod_name).await {
            Ok(pod) => Ok(Json(pod)),
            Err(_) => Err(StatusCode::NOT_FOUND),
        }
    }
    
    /// Select cluster for request endpoint
    async fn select_cluster_for_request(
        State(state): State<GatewayState>,
        Query(params): Query<HashMap<String, String>>,
    ) -> Result<Json<serde_json::Value>, StatusCode> {
        let prefer_region = params.get("region");
        let handler = FederationHandler::new(
            state.config.clone(),
            state.discovery.clone(),
            state.health.clone(),
            state.load_balancer.clone(),
            state.resource_manager.clone(),
        );
        handler.handle_select_cluster(prefer_region.as_deref()).await
    }
    
    /// Load balancer statistics endpoint
    async fn loadbalancer_stats(State(_state): State<GatewayState>) -> Json<serde_json::Value> {
        Json(serde_json::json!({
            "algorithm": "configured",
            "health_aware": true,
        }))
    }
    
    /// Metrics endpoint
    async fn metrics(State(_state): State<GatewayState>) -> String {
        "# Federation metrics\n".to_string()
    }
}

