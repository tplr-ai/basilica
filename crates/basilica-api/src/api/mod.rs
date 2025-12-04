//! API module for the Basilica API Gateway

pub mod auth;
pub mod extractors;
pub mod middleware;
pub mod query;
pub mod routes;

use crate::server::AppState;
use axum::{
    routing::{delete, get, post},
    Router,
};

/// Create all API routes
pub fn routes(state: AppState) -> Router<AppState> {
    // Unprotected routes (for health checks, metrics, etc.)
    let public_routes = Router::new()
        // Health endpoint - no authentication required for ALB health checks
        .route("/health", get(routes::health::health_check))
        .route("/health/k3s", get(routes::health::k3s_health_check))
        // Metrics endpoint - no authentication required for Prometheus scraping
        .route("/metrics", get(routes::metrics::metrics_handler));

    // Protected routes with unified authentication and scope validation
    let protected_routes = Router::new()
        .route("/jobs", post(routes::jobs::create_job))
        .route(
            "/jobs/:id",
            get(routes::jobs::get_job_status).delete(routes::jobs::delete_job),
        )
        .route("/jobs/:id/logs", get(routes::jobs::get_job_logs))
        .route("/jobs/:id/read-file", post(routes::jobs::read_job_file))
        .route("/jobs/:id/suspend", post(routes::jobs::suspend_job))
        .route("/jobs/:id/resume", post(routes::jobs::resume_job))
        // Rentals endpoints
        .route(
            "/rentals",
            get(routes::rentals::list_rentals_validator).post(routes::rentals::start_rental),
        )
        .route(
            "/rentals/history",
            get(routes::rentals::list_rental_history),
        )
        .route(
            "/rentals/:id",
            get(routes::rentals::get_rental_status).delete(routes::rentals::stop_rental),
        )
        .route(
            "/rentals/:id/restart",
            post(routes::rentals::restart_rental),
        )
        .route(
            "/rentals/:id/logs",
            get(routes::rentals::stream_rental_logs),
        )
        .route("/nodes", get(routes::rentals::list_available_nodes))
        // API key management endpoints (JWT auth only)
        .route(
            "/api-keys",
            post(routes::api_keys::create_key).get(routes::api_keys::list_keys),
        )
        .route("/api-keys/:name", delete(routes::api_keys::revoke_key))
        // SSH key management endpoints (JWT auth only)
        .route(
            "/ssh-keys",
            post(routes::ssh_keys::register_ssh_key)
                .get(routes::ssh_keys::get_ssh_key)
                .delete(routes::ssh_keys::delete_ssh_key),
        )
        // User deployment management endpoints
        .route(
            "/deployments",
            post(routes::deployments::create_deployment).get(routes::deployments::list_deployments),
        )
        .route(
            "/deployments/:instance_name",
            get(routes::deployments::get_deployment).delete(routes::deployments::delete_deployment),
        )
        .route(
            "/deployments/:instance_name/logs",
            get(routes::deployments::stream_deployment_logs),
        )
        // GPU node registration endpoints
        .route(
            "/v1/gpu-nodes/register",
            post(routes::gpu_nodes::register_gpu_node),
        )
        .route(
            "/v1/gpu-nodes/revoke",
            post(routes::gpu_nodes::revoke_gpu_node),
        )
        .route(
            "/v1/gpu-nodes/:node_id/wireguard-key",
            post(routes::gpu_nodes::register_wireguard_key),
        );

    // Add payment and billing service endpoints
    let protected_routes = protected_routes
        .nest("/payments", routes::payments::routes())
        .nest("/billing", routes::billing::routes())
        // Secure Cloud endpoints (GPU aggregator) - proxied through API
        .route(
            "/secure-cloud/gpu-prices",
            get(routes::secure_cloud::list_gpu_prices),
        )
        .route(
            "/secure-cloud/rentals",
            get(routes::secure_cloud::list_secure_cloud_rentals),
        )
        .route(
            "/secure-cloud/rentals/start",
            post(routes::secure_cloud::start_secure_cloud_rental),
        )
        .route(
            "/secure-cloud/rentals/:rental_id/stop",
            post(routes::secure_cloud::stop_secure_cloud_rental),
        )
        // Apply middleware layers
        // Apply scope validation AFTER auth middleware
        .layer(axum::middleware::from_fn(
            middleware::scope_validation_middleware,
        ))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            middleware::auth_middleware,
        ));

    // Build the router with both public and protected routes
    let router = Router::new()
        .merge(public_routes)
        .merge(protected_routes)
        .with_state(state.clone());

    // Apply general middleware
    middleware::apply_middleware(router, state)
}
