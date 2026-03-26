//! # Validator API Module
//!
//! Clean, modular HTTP/REST API server for external services to interact with the Validator.
//! Follows SOLID principles with separation of concerns.

#[cfg(feature = "client")]
pub mod client;

pub mod routes;
pub mod types;

use crate::config::ApiConfig;
use crate::rental;
use anyhow::Result;
use axum::{
    routing::{delete, get, post},
    Router,
};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::TcpListener;
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use tracing::info;

/// API server state shared across handlers
#[derive(Clone)]
pub struct ApiState {
    config: ApiConfig,
    persistence: Arc<crate::persistence::SimplePersistence>,
    gpu_profile_repo: Arc<crate::persistence::gpu_profile_repository::GpuProfileRepository>,
    #[allow(dead_code)]
    storage: basilica_common::MemoryStorage,
    validator_config: crate::config::ValidatorConfig,
    #[allow(dead_code)]
    rental_manager: Option<Arc<rental::RentalManager>>,
    #[allow(dead_code)]
    validator_hotkey: basilica_common::identity::Hotkey,
    pub evidence_storage_path: PathBuf,
}

impl ApiState {
    pub fn new(
        config: ApiConfig,
        persistence: Arc<crate::persistence::SimplePersistence>,
        gpu_profile_repo: Arc<crate::persistence::gpu_profile_repository::GpuProfileRepository>,
        storage: basilica_common::MemoryStorage,
        validator_config: crate::config::ValidatorConfig,
        validator_hotkey: basilica_common::identity::Hotkey,
    ) -> Self {
        let evidence_storage_path = validator_config
            .collateral
            .as_ref()
            .map(|c| c.evidence_storage_path.clone())
            .unwrap_or_else(|| PathBuf::from("./evidence"));
        Self {
            config,
            persistence,
            gpu_profile_repo,
            storage,
            validator_config,
            rental_manager: None,
            validator_hotkey,
            evidence_storage_path,
        }
    }

    pub fn with_rental_manager(mut self, rental_manager: Arc<rental::RentalManager>) -> Self {
        self.rental_manager = Some(rental_manager);
        self
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
        gpu_profile_repo: Arc<crate::persistence::gpu_profile_repository::GpuProfileRepository>,
        storage: basilica_common::MemoryStorage,
        validator_config: crate::config::ValidatorConfig,
        validator_hotkey: basilica_common::identity::Hotkey,
    ) -> Self {
        Self {
            state: ApiState::new(
                config,
                persistence,
                gpu_profile_repo,
                storage,
                validator_config,
                validator_hotkey,
            ),
        }
    }

    /// Set rental manager
    pub fn with_rental_manager(mut self, rental_manager: Arc<rental::RentalManager>) -> Self {
        self.state = self.state.with_rental_manager(rental_manager);
        self
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
            .route("/rentals", get(routes::list_rentals))
            .route("/rentals", post(routes::start_rental))
            .route("/rentals/:id", get(routes::get_rental_status))
            .route("/rentals/:id", delete(routes::stop_rental))
            .route("/rentals/:id/logs", get(routes::stream_rental_logs))
            .route("/evidence/:file_name", get(routes::get_evidence))
            .route("/nodes", get(routes::list_available_nodes))
            // Existing miner routes
            .route("/miners", get(routes::list_miners))
            .route("/miners/:miner_id", get(routes::get_miner))
            .route("/miners/:miner_id/health", get(routes::get_miner_health))
            .route("/miners/:miner_id/nodes", get(routes::list_miner_nodes))
            .route("/health", get(routes::health_check))
            // new
            .route("/gpu-profiles", get(routes::list_gpu_profiles))
            .route(
                "/gpu-profiles/:category",
                get(routes::list_gpu_profiles_by_category),
            )
            .route("/gpu-categories", get(routes::list_gpu_categories))
            .route(
                "/verification/active",
                get(routes::list_active_verifications),
            )
            .route(
                "/verification/results/:miner_id",
                get(routes::get_verification_results),
            )
            .route("/config", get(routes::get_config))
            .route("/config/verification", get(routes::get_verification_config))
            .route("/config/emission", get(routes::get_emission_config))
            .layer(TraceLayer::new_for_http())
            .layer(CorsLayer::permissive())
            .with_state(self.state.clone())
    }
}
