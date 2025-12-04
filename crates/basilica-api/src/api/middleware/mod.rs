//! API middleware stack

mod auth;
mod auth0;
mod balance;
mod metrics;
mod rate_limit;
mod scope;

pub use auth::{auth_middleware, get_auth_context, AuthContext, AuthDetails};
pub use auth0::{auth0_middleware, get_auth0_claims, Auth0Claims};
pub use balance::validate_balance_for_rental;
pub use metrics::metrics_middleware;
pub use rate_limit::RateLimitMiddleware;
pub use scope::scope_validation_middleware;

use crate::server::AppState;
use axum::{
    body::Body,
    extract::State,
    http::{Request, StatusCode},
    middleware::Next,
    response::Response,
    Router,
};
use tower_http::{
    cors::{Any, CorsLayer},
    timeout::TimeoutLayer,
};

/// Apply middleware to a router
pub fn apply_middleware(router: Router<AppState>, state: AppState) -> Router<AppState> {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    router
        // Add timeout
        .layer(TimeoutLayer::new(state.config.request_timeout()))
        // Add CORS
        .layer(cors)
        // Add metrics middleware (first, to track all requests)
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            metrics_middleware,
        ))
        // Add rate limit middleware
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            rate_limit_handler,
        ))
}

/// Rate limit handler function
async fn rate_limit_handler(
    State(state): axum::extract::State<AppState>,
    req: Request<Body>,
    next: Next,
) -> Result<Response<Body>, crate::error::ApiError> {
    // Create rate limit storage
    let storage = std::sync::Arc::new(rate_limit::RateLimitStorage::new(std::sync::Arc::new(
        state.config.rate_limit.clone(),
    )));

    // Check rate limit
    match rate_limit::rate_limit_middleware(storage, req, next).await {
        Ok(response) => Ok(response),
        Err(StatusCode::TOO_MANY_REQUESTS) => {
            // Record rate limit event
            if let Some(metrics) = &state.metrics {
                let api_metrics = metrics.api_metrics();
                tokio::spawn(async move {
                    api_metrics.record_rate_limited("unknown").await;
                });
            }
            Err(crate::error::ApiError::RateLimitExceeded)
        }
        Err(_) => Err(crate::error::ApiError::Internal {
            message: "Rate limit check failed".to_string(),
        }),
    }
}
