//! # Validator API Module
//!
//! Clean, modular HTTP/REST API server for external services to interact with the Validator.
//! Follows SOLID principles with separation of concerns.

pub mod routes;
pub mod types;

use crate::config::ApiConfig;
use anyhow::Result;
use axum::{
    routing::{delete, get, post, put},
    Router,
};
use std::sync::Arc;
use tokio::net::TcpListener;
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use tracing::info;

/// API server state shared across handlers
#[derive(Clone)]
pub struct ApiState {
    config: ApiConfig,
    persistence: Arc<crate::persistence::SimplePersistence>,
    #[allow(dead_code)]
    storage: common::MemoryStorage,
}

impl ApiState {
    pub fn new(
        config: ApiConfig,
        persistence: Arc<crate::persistence::SimplePersistence>,
        storage: common::MemoryStorage,
    ) -> Self {
        Self {
            config,
            persistence,
            storage,
        }
    }
}

/// Main API server implementation following Single Responsibility Principle
pub struct ApiHandler {
    state: ApiState,
}

impl ApiHandler {
    /// Create a new API handler
    pub fn new(
        config: ApiConfig,
        persistence: Arc<crate::persistence::SimplePersistence>,
        storage: common::MemoryStorage,
    ) -> Self {
        Self {
            state: ApiState::new(config, persistence, storage),
        }
    }

    /// Start the API server
    pub async fn start(&self) -> Result<()> {
        let app = self.create_router();

        let listener = TcpListener::bind(&self.state.config.bind_address).await?;
        info!("API server listening on {}", self.state.config.bind_address);

        axum::serve(listener, app).await?;

        Ok(())
    }

    /// Create the Axum router with all endpoints
    /// Follows Open/Closed Principle - easy to extend with new routes
    fn create_router(&self) -> Router {
        Router::new()
            .route("/capacity/available", get(routes::list_available_capacity))
            .route("/rentals", post(routes::rent_capacity))
            .route("/rentals/:rental_id", delete(routes::terminate_rental))
            .route("/rentals/:rental_id/status", get(routes::get_rental_status))
            .route("/rentals/:rental_id/logs", get(routes::stream_rental_logs))
            .route("/miners", get(routes::list_miners))
            .route("/miners/register", post(routes::register_miner))
            .route("/miners/:miner_id", get(routes::get_miner))
            .route("/miners/:miner_id", put(routes::update_miner))
            .route("/miners/:miner_id", delete(routes::remove_miner))
            .route("/miners/:miner_id/health", get(routes::get_miner_health))
            .route(
                "/miners/:miner_id/verify",
                post(routes::trigger_miner_verification),
            )
            .route(
                "/miners/:miner_id/executors",
                get(routes::list_miner_executors),
            )
            .route("/health", get(routes::health_check))
            .layer(TraceLayer::new_for_http())
            .layer(CorsLayer::permissive())
            .with_state(self.state.clone())
    }
}
