//! API module for the Basilica API Gateway

pub mod auth;
pub mod extractors;
pub mod middleware;
pub mod routes;

use crate::server::AppState;
use axum::{
    routing::{delete, get, post},
    Router,
};

/// Create all API routes
pub fn routes(state: AppState) -> Router<AppState> {
    // Unprotected routes (for health checks, etc.)
    let public_routes = Router::new()
        // Health endpoint - no authentication required for ALB health checks
        .route("/health", get(routes::health::health_check));

    // Protected routes with unified authentication and scope validation
    let protected_routes = Router::new()
        .route("/rentals", get(routes::rentals::list_rentals_validator))
        .route("/rentals", post(routes::rentals::start_rental))
        .route("/rentals/:id", get(routes::rentals::get_rental_status))
        .route("/rentals/:id", delete(routes::rentals::stop_rental))
        .route(
            "/rentals/:id/logs",
            get(routes::rentals::stream_rental_logs),
        )
        .route("/executors", get(routes::rentals::list_available_executors))
        // API key management endpoints (JWT auth only)
        .route(
            "/api-keys",
            post(routes::api_keys::create_key).get(routes::api_keys::list_keys),
        )
        .route("/api-keys/:name", delete(routes::api_keys::revoke_key))
        // Apply scope validation AFTER auth middleware
        .layer(axum::middleware::from_fn(
            middleware::scope_validation_middleware,
        ))
        // Apply unified authentication first
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
