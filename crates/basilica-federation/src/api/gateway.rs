//! Multi-cluster API gateway

use crate::config::FederationConfig;
use crate::discovery::ServiceDiscovery;
use crate::error::{FederationError, Result};
use crate::health::HealthAggregator;
use crate::load_balancer::LoadBalancer;
use crate::resource_manager::ResourceManager;
use axum::{
    extract::{Path, Query, State},
    http::{HeaderMap, Method, StatusCode},
    response::Response,
    routing::{get, post, put, delete},
    Router,
};
use std::collections::HashMap;
use std::sync::Arc;
use tower::ServiceBuilder;
use tower_http::{
    cors::CorsLayer,
    timeout::TimeoutLayer,
    trace::TraceLayer,
};
use tracing::{debug, error, info, warn};

/// Federation API gateway
pub struct FederationGateway {
    config: Arc<FederationConfig>,
    discovery: Arc<ServiceDiscovery>,
    health: Arc<HealthAggregator>,
    load_balancer: Arc<LoadBalancer>,
    router: Router,
}

impl FederationGateway {
    /// Create a new federation gateway
    pub async fn new(
        config: Arc<FederationConfig>,
        discovery: Arc<ServiceDiscovery>,
        health: Arc<HealthAggregator>,
        load_balancer: Arc<LoadBalancer>,
        resource_manager: Arc<ResourceManager>,
    ) -> Result<Self> {
        let http_client = reqwest::Client::builder()
            .timeout(config.gateway.request_timeout)
            .build()
            .map_err(|e| FederationError::Config(format!("Failed to create HTTP client: {}", e)))?;
        
        let router = Self::build_router(
            config.clone(),
            discovery.clone(),
            health.clone(),
            load_balancer.clone(),
            resource_manager.clone(),
            http_client.clone(),
        );
        
        Ok(Self {
            config,
            discovery,
            health,
            load_balancer,
            router,
        })
    }
    
    /// Build the API router
    fn build_router(
        config: Arc<FederationConfig>,
        discovery: Arc<ServiceDiscovery>,
        health: Arc<HealthAggregator>,
        load_balancer: Arc<LoadBalancer>,
        resource_manager: Arc<ResourceManager>,
        http_client: reqwest::Client,
    ) -> Router {
        Router::new()
            .route("/health", get(Self::health_handler))
            .route("/clusters", get(Self::list_clusters))
            .route("/clusters/:cluster_id", get(Self::get_cluster))
            .route("/clusters/:cluster_id/health", get(Self::cluster_health))
            .route("/services", get(Self::list_services))
            .route("/services/:service_name", get(Self::get_service))
            .route("/proxy/*path", get(Self::proxy_get).post(Self::proxy_post).put(Self::proxy_put).delete(Self::proxy_delete))
            .route("/api/v1/namespaces/:namespace/pods", get(Self::list_pods))
            .route("/api/v1/namespaces/:namespace/pods/:pod_name", get(Self::get_pod))
            .route("/metrics", get(Self::metrics_handler))
            .layer(
                ServiceBuilder::new()
                    .layer(TimeoutLayer::new(config.gateway.request_timeout))
                    .layer(TraceLayer::new_for_http())
                    .layer(CorsLayer::permissive())
            )
            .with_state(GatewayState {
                config,
                discovery,
                health,
                load_balancer,
                resource_manager,
                http_client,
            })
    }
    
    /// Start the gateway server
    pub async fn start(&self) -> Result<()> {
        let addr = format!("{}:{}", self.config.gateway.listen_addr, self.config.gateway.port);
        let listener = tokio::net::TcpListener::bind(&addr).await
            .map_err(|e| FederationError::Config(format!("Failed to bind to {}: {}", addr, e)))?;
        
        info!(
            address = %addr,
            "Federation gateway listening"
        );
        
        axum::serve(listener, self.router.clone())
            .await
            .map_err(|e| FederationError::Config(format!("Server error: {}", e)))?;
        
        Ok(())
    }
    
    /// Health check handler
    async fn health_handler(State(state): State<GatewayState>) -> Result<axum::Json<serde_json::Value>, StatusCode> {
        match state.health.aggregate_health().await {
            Ok(health) => Ok(axum::Json(serde_json::json!({
                "status": "healthy",
                "clusters": health,
            }))),
            Err(e) => {
                error!(error = %e, "Health check failed");
                Err(StatusCode::SERVICE_UNAVAILABLE)
            }
        }
    }
    
    /// List all clusters
    async fn list_clusters(State(state): State<GatewayState>) -> Result<axum::Json<serde_json::Value>> {
        let clusters: Vec<_> = state.config.enabled_clusters()
            .iter()
            .map(|c| serde_json::json!({
                "id": c.id,
                "name": c.name,
                "region": c.region,
                "priority": c.priority,
                "tags": c.tags,
            }))
            .collect();
        
        Ok(axum::Json(serde_json::json!({
            "clusters": clusters,
        })))
    }
    
    /// Get cluster details
    async fn get_cluster(
        State(state): State<GatewayState>,
        Path(cluster_id): Path<String>,
    ) -> Result<axum::Json<serde_json::Value>, StatusCode> {
        let cluster = state.config.get_cluster(&cluster_id)
            .ok_or_else(|| StatusCode::NOT_FOUND)?;
        
        let health = state.health.get_cluster_health(&cluster_id).await
            .unwrap_or_default();
        
        Ok(axum::Json(serde_json::json!({
            "id": cluster.id,
            "name": cluster.name,
            "region": cluster.region,
            "priority": cluster.priority,
            "tags": cluster.tags,
            "health": health,
        })))
    }
    
    /// Get cluster health
    async fn cluster_health(
        State(state): State<GatewayState>,
        Path(cluster_id): Path<String>,
    ) -> Result<axum::Json<serde_json::Value>, StatusCode> {
        let health = state.health.get_cluster_health(&cluster_id).await
            .ok_or_else(|| StatusCode::NOT_FOUND)?;
        
        Ok(axum::Json(health))
    }
    
    /// List services across all clusters
    async fn list_services(
        State(state): State<GatewayState>,
        Query(params): Query<HashMap<String, String>>,
    ) -> Result<axum::Json<serde_json::Value>> {
        let namespace = params.get("namespace");
        let services = state.discovery.discover_services(namespace).await?;
        
        Ok(axum::Json(serde_json::json!({
            "services": services,
        })))
    }
    
    /// Get service details
    async fn get_service(
        State(state): State<GatewayState>,
        Path(service_name): Path<String>,
        Query(params): Query<HashMap<String, String>>,
    ) -> Result<axum::Json<serde_json::Value>, StatusCode> {
        let namespace = params.get("namespace");
        let service = state.discovery.get_service(&service_name, namespace).await
            .map_err(|_| StatusCode::NOT_FOUND)?;
        
        Ok(axum::Json(service))
    }
    
    /// Proxy GET request
    async fn proxy_get(
        State(state): State<GatewayState>,
        Path(path): Path<String>,
        Query(params): Query<HashMap<String, String>>,
        headers: HeaderMap,
    ) -> Result<Response, StatusCode> {
        Self::proxy_request(
            state,
            Method::GET,
            path,
            params,
            headers,
            None,
        ).await
    }
    
    /// Proxy POST request
    async fn proxy_post(
        State(state): State<GatewayState>,
        Path(path): Path<String>,
        Query(params): Query<HashMap<String, String>>,
        headers: HeaderMap,
        body: axum::body::Body,
    ) -> Result<Response, StatusCode> {
        let max_body_size = state.config.gateway.max_body_size;
        let body_bytes = axum::body::to_bytes(body, max_body_size).await
            .map_err(|_| StatusCode::from_u16(413).unwrap_or(StatusCode::BAD_REQUEST))?;
        
        Self::proxy_request(
            state,
            Method::POST,
            path,
            params,
            headers,
            Some(body_bytes.to_vec()),
        ).await
    }
    
    /// Proxy PUT request
    async fn proxy_put(
        State(state): State<GatewayState>,
        Path(path): Path<String>,
        Query(params): Query<HashMap<String, String>>,
        headers: HeaderMap,
        body: axum::body::Body,
    ) -> Result<Response, StatusCode> {
        let max_body_size = state.config.gateway.max_body_size;
        let body_bytes = axum::body::to_bytes(body, max_body_size).await
            .map_err(|_| StatusCode::from_u16(413).unwrap_or(StatusCode::BAD_REQUEST))?;
        
        Self::proxy_request(
            state,
            Method::PUT,
            path,
            params,
            headers,
            Some(body_bytes.to_vec()),
        ).await
    }
    
    /// Proxy DELETE request
    async fn proxy_delete(
        State(state): State<GatewayState>,
        Path(path): Path<String>,
        Query(params): Query<HashMap<String, String>>,
        headers: HeaderMap,
    ) -> Result<Response, StatusCode> {
        Self::proxy_request(
            state,
            Method::DELETE,
            path,
            params,
            headers,
            None,
        ).await
    }
    
    /// Proxy request to target cluster
    async fn proxy_request(
        state: GatewayState,
        method: Method,
        path: String,
        params: HashMap<String, String>,
        headers: HeaderMap,
        body: Option<Vec<u8>>,
    ) -> Result<Response, StatusCode> {
        // Select target cluster using load balancer
        let cluster = state.load_balancer.select_cluster().await
            .ok_or_else(|| StatusCode::SERVICE_UNAVAILABLE)?;
        
        // Build target URL with query parameters
        let mut target_url = format!("{}/{}", cluster.api_server, path);
        if !params.is_empty() {
            let query_string: Vec<String> = params
                .iter()
                .map(|(k, v)| format!("{}={}", urlencoding::encode(k), urlencoding::encode(v)))
                .collect();
            target_url.push('?');
            target_url.push_str(&query_string.join("&"));
        }
        
        // Parse as reqwest::Url (not http::Uri)
        let url: reqwest::Url = target_url.parse()
            .map_err(|e| {
                error!(error = %e, "Invalid target URL");
                StatusCode::BAD_REQUEST
            })?;
        
        // Convert http::Method to reqwest::Method
        let reqwest_method = match method.as_str() {
            "GET" => reqwest::Method::GET,
            "POST" => reqwest::Method::POST,
            "PUT" => reqwest::Method::PUT,
            "DELETE" => reqwest::Method::DELETE,
            _ => {
                error!(method = %method, "Unsupported HTTP method");
                return Err(StatusCode::METHOD_NOT_ALLOWED);
            }
        };
        
        // Create request builder
        let mut request_builder = state.http_client.request(reqwest_method, url);
        
        // Copy headers (excluding host and hop-by-hop headers)
        let hop_by_hop_headers = [
            "connection", "keep-alive", "proxy-authenticate",
            "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade",
        ];
        
        for (key, value) in headers.iter() {
            let header_name_lower = key.as_str().to_lowercase();
            if header_name_lower != "host" && !hop_by_hop_headers.contains(&header_name_lower.as_str()) {
                if let Ok(header_value) = value.to_str() {
                    request_builder = request_builder.header(key.as_str(), header_value);
                }
            }
        }
        
        // Set body if provided
        if let Some(body_data) = body {
            request_builder = request_builder.body(body_data);
        }
        
        // Execute request
        let response = request_builder.send().await
            .map_err(|e| {
                error!(error = %e, "Proxy request failed");
                StatusCode::BAD_GATEWAY
            })?;
        
        // Build response
        let status = StatusCode::from_u16(response.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        
        let mut response_builder = Response::builder()
            .status(status);
        
        // Copy response headers (excluding hop-by-hop headers)
        for (key, value) in response.headers() {
            let header_name_lower = key.as_str().to_lowercase();
            if !hop_by_hop_headers.contains(&header_name_lower.as_str()) {
                if let Ok(header_name) = http::HeaderName::from_bytes(key.as_str().as_bytes()) {
                    if let Ok(header_value) = value.to_str() {
                        response_builder = response_builder.header(header_name, header_value);
                    }
                }
            }
        }
        
        let body_bytes = response.bytes().await
            .map_err(|_| StatusCode::BAD_GATEWAY)?;
        
        Ok(response_builder.body(axum::body::Body::from(body_bytes.to_vec()))
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?)
    }
    
    /// List pods
    async fn list_pods(
        State(state): State<GatewayState>,
        Path(namespace): Path<String>,
    ) -> Result<axum::Json<serde_json::Value>, StatusCode> {
        match state.resource_manager.list_pods(&namespace).await {
            Ok(pods) => Ok(axum::Json(serde_json::json!({
                "pods": pods,
            }))),
            Err(e) => {
                error!(error = %e, "Failed to list pods");
                Err(StatusCode::INTERNAL_SERVER_ERROR)
            }
        }
    }
    
    /// Get pod details
    async fn get_pod(
        State(state): State<GatewayState>,
        Path((namespace, pod_name)): Path<(String, String)>,
    ) -> Result<axum::Json<serde_json::Value>, StatusCode> {
        match state.resource_manager.get_pod(&namespace, &pod_name).await {
            Ok(pod) => Ok(axum::Json(pod)),
            Err(_) => Err(StatusCode::NOT_FOUND),
        }
    }
    
    /// Metrics handler
    async fn metrics_handler(State(_state): State<GatewayState>) -> String {
        // Return Prometheus metrics
        "# Federation metrics\n".to_string()
    }
}

/// Gateway state
#[derive(Clone)]
struct GatewayState {
    config: Arc<FederationConfig>,
    discovery: Arc<ServiceDiscovery>,
    health: Arc<HealthAggregator>,
    load_balancer: Arc<LoadBalancer>,
    resource_manager: Arc<ResourceManager>,
    http_client: reqwest::Client,
}

