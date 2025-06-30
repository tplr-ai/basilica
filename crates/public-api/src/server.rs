//! Main server implementation for the Public API Gateway

use crate::{
    api,
    config::Config,
    discovery::ValidatorDiscovery,
    error::{Error, Result},
    load_balancer::LoadBalancer,
};
use axum::Router;
use std::sync::Arc;
use tokio::{signal, sync::RwLock};
use tower::ServiceBuilder;
use tower_http::{
    cors::{Any, CorsLayer},
    timeout::TimeoutLayer,
    trace::TraceLayer,
};
use tracing::{info, warn};

/// Main server structure
pub struct Server {
    config: Arc<Config>,
    app: Router,
}

/// Shared application state
#[derive(Clone)]
pub struct AppState {
    /// Application configuration
    pub config: Arc<Config>,

    /// Validator discovery service
    pub discovery: Arc<ValidatorDiscovery>,

    /// Load balancer
    pub load_balancer: Arc<RwLock<LoadBalancer>>,

    /// HTTP client for validator requests
    pub http_client: reqwest::Client,
}

impl Server {
    /// Create a new server instance
    pub async fn new(config: Config) -> Result<Self> {
        info!("Initializing Public API Gateway server");

        let config = Arc::new(config);

        // Initialize Bittensor service for validator discovery
        let bittensor_config = config.to_bittensor_config();
        let bittensor_service = bittensor::Service::new(bittensor_config).await?;

        // Initialize validator discovery
        let discovery = Arc::new(ValidatorDiscovery::new(
            Arc::new(bittensor_service),
            config.clone(),
        ));

        // Start discovery task
        let discovery_clone = discovery.clone();
        tokio::spawn(async move {
            discovery_clone.start_discovery_loop().await;
        });

        // Initialize load balancer
        let load_balancer = Arc::new(RwLock::new(LoadBalancer::new(
            config.load_balancer.strategy.clone(),
        )));

        // Create HTTP client for validator communication
        let http_client = reqwest::Client::builder()
            .timeout(config.request_timeout())
            .connect_timeout(config.connection_timeout())
            .pool_max_idle_per_host(10)
            .build()
            .map_err(Error::HttpClient)?;

        // Create application state
        let state = AppState {
            config: config.clone(),
            discovery,
            load_balancer,
            http_client,
        };

        // Build the application router
        let app = Self::build_router(state)?;

        Ok(Self { config, app })
    }

    /// Build the application router with all routes and middleware
    fn build_router(state: AppState) -> Result<Router> {
        let cors = CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any);

        let middleware = ServiceBuilder::new()
            .layer(TraceLayer::new_for_http())
            .layer(TimeoutLayer::new(state.config.request_timeout()))
            .layer(cors);

        let app = Router::new()
            .nest("/api/v1", api::routes(state.clone()))
            .merge(api::docs_routes())
            .layer(middleware)
            .with_state(state);

        Ok(app)
    }

    /// Run the server until shutdown signal
    pub async fn run(self) -> Result<()> {
        let addr = self.config.server.bind_address;

        info!("Starting HTTP server on {}", addr);

        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .map_err(|e| Error::Internal {
                message: format!("Failed to bind to address {addr}: {e}"),
            })?;

        info!("Public API Gateway listening on {}", addr);

        axum::serve(listener, self.app)
            .with_graceful_shutdown(shutdown_signal())
            .await
            .map_err(|e| Error::Internal {
                message: format!("Server error: {e}"),
            })?;

        Ok(())
    }
}

/// Shutdown signal handler
async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            warn!("Received Ctrl+C, shutting down");
        },
        _ = terminate => {
            warn!("Received terminate signal, shutting down");
        },
    }
}
