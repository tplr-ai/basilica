//! API middleware stack

mod auth;
mod cache;
mod rate_limit;

pub use auth::AuthMiddleware;
pub use cache::CacheMiddleware;
pub use rate_limit::RateLimitMiddleware;

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
    trace::{DefaultMakeSpan, TraceLayer},
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
        // Add tracing
        .layer(TraceLayer::new_for_http().make_span_with(DefaultMakeSpan::default()))
        // Add custom middleware layers
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            rate_limit_handler,
        ))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            auth_handler,
        ))
        .layer(axum::middleware::from_fn_with_state(state, cache_handler))
}

/// Rate limit handler function
async fn rate_limit_handler(
    State(state): axum::extract::State<AppState>,
    req: Request<Body>,
    next: Next,
) -> Result<Response<Body>, crate::error::Error> {
    // Create rate limit storage
    let storage = std::sync::Arc::new(rate_limit::RateLimitStorage::new(std::sync::Arc::new(
        state.config.rate_limit.clone(),
    )));

    // Check rate limit
    match rate_limit::rate_limit_middleware(storage, req, next).await {
        Ok(response) => Ok(response),
        Err(StatusCode::TOO_MANY_REQUESTS) => Err(crate::error::Error::RateLimitExceeded),
        Err(_) => Err(crate::error::Error::Internal {
            message: "Rate limit check failed".to_string(),
        }),
    }
}

/// Auth handler function
async fn auth_handler(
    State(state): axum::extract::State<AppState>,
    req: Request<Body>,
    next: Next,
) -> Result<Response<Body>, crate::error::Error> {
    // Create auth middleware and handle request
    let _auth = auth::AuthMiddleware::new(state.clone());
    auth::AuthMiddleware::handle(State(state), req, next).await
}

/// Cache handler function
async fn cache_handler(
    State(state): axum::extract::State<AppState>,
    req: Request<Body>,
    next: Next,
) -> Result<Response<Body>, crate::error::Error> {
    // Create cache storage
    let storage = std::sync::Arc::new(cache::CacheStorage::new(std::sync::Arc::new(
        state.config.cache.clone(),
    )));

    // Handle caching
    match cache::cache_middleware(storage, state.config.clone(), req, next).await {
        Ok(response) => Ok(response),
        Err(_) => Err(crate::error::Error::Internal {
            message: "Cache middleware failed".to_string(),
        }),
    }
}
