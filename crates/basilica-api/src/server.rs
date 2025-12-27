//! Main server implementation for the Basilica API Gateway

use crate::k8s_client::{ApiK8sClient, K8sClient};
use crate::{
    api,
    api::extractors::ownership::archive_rental_ownership,
    config::Config,
    error::{ApiError, Result},
};
use axum::Router;
use basilica_aggregator::{
    vip::{CsvDataSource, MockVipDataSource, VipCache, VipCsvRow, VipPoller, VipPollerTask},
    AggregatorService, Database as AggregatorDatabase,
};
use basilica_billing::BillingClient;
use basilica_payments::client::PaymentsClient;
use basilica_protocol::billing::{GpuUsage, ResourceUsage, TelemetryData};
use basilica_validator::{api::types::RentalStatus, ValidatorClient};
use sqlx::{postgres::PgPoolOptions, PgPool};
use std::sync::Arc;
use std::time::SystemTime;
use tokio::signal;
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
    cancel_token: tokio_util::sync::CancellationToken,
    background_tasks: Option<tokio::task::JoinSet<()>>,
}

/// Shared application state
#[derive(Clone)]
pub struct AppState {
    /// Application configuration
    pub config: Arc<Config>,

    /// Validator client for making requests
    pub validator_client: Arc<ValidatorClient>,

    /// Validator endpoint for reference
    pub validator_endpoint: String,

    /// Validator UID in the subnet
    pub validator_uid: u16,

    /// Validator hotkey (SS58 address)
    pub validator_hotkey: String,

    /// HTTP client for validator requests
    pub http_client: reqwest::Client,

    /// Database pool for user rental tracking
    pub db: PgPool,

    /// Optional K8s client seam for Jobs/Rentals backed by K3s
    pub k8s: Option<Arc<dyn crate::k8s_client::ApiK8sClient + Send + Sync>>,

    /// Payments service client
    pub payments_client: Option<Arc<PaymentsClient>>,

    /// Billing service client
    pub billing_client: Option<Arc<BillingClient>>,

    /// DNS provider for public deployments
    pub dns_provider: Option<Arc<dyn crate::dns::DnsProvider>>,

    /// Metrics system
    pub metrics: Option<Arc<crate::metrics::ApiMetricsSystem>>,

    /// GPU Aggregator service (Secure Cloud)
    pub aggregator_service: Arc<AggregatorService>,

    /// GPU Aggregator configuration (for webhook validation)
    pub aggregator_config: basilica_aggregator::config::Config,

    /// Pricing configuration (marketplace markups)
    pub pricing_config: crate::config::PricingConfig,

    /// SSH client for K3s token generation
    pub ssh_client: Arc<crate::ssh::K3sSshClient>,
}

/// Process health check for a single rental
async fn process_rental_health_check(
    rental_id: &str,
    validator_client: &ValidatorClient,
    db: &PgPool,
) {
    match validator_client.get_rental_status(rental_id).await {
        Ok(status_response) => {
            // Check if rental is in a terminated state
            if matches!(
                status_response.status,
                RentalStatus::Terminated | RentalStatus::Failed
            ) {
                // Archive the rental to terminated_user_rentals table
                if let Err(e) = archive_rental_ownership(
                    db,
                    rental_id,
                    Some("Health check: rental no longer accessible"),
                )
                .await
                {
                    tracing::error!("Failed to archive stopped rental {}: {}", rental_id, e);
                } else {
                    tracing::info!(
                        "Health check: Archived {:?} rental {}",
                        status_response.status,
                        rental_id
                    );
                }
            }
        }
        Err(e) => {
            // Validator unavailable or having issues - don't change rental state
            // The validator is the source of truth and will report actual status when available
            tracing::warn!(
                "Health check failed for rental {} (validator may be unavailable): {}",
                rental_id,
                e
            );
        }
    }
}

/// Send synthetic telemetry for active secure cloud rentals to billing service
async fn process_secure_cloud_billing(billing_client: &BillingClient, db: &PgPool) {
    // Query active secure cloud rentals with their GPU count from offerings
    let rentals: Vec<(String, i32)> = match sqlx::query_as(
        r#"
        SELECT r.id, COALESCE(o.gpu_count, 1) as gpu_count
        FROM secure_cloud_rentals r
        LEFT JOIN gpu_offerings o ON r.offering_id = o.id
        WHERE r.status = 'running'
        "#,
    )
    .fetch_all(db)
    .await
    {
        Ok(records) => records,
        Err(e) => {
            tracing::error!("Failed to query secure cloud rentals for billing: {}", e);
            return;
        }
    };

    if rentals.is_empty() {
        return;
    }

    // Build telemetry batch for all active rentals
    let telemetry_batch: Vec<TelemetryData> = rentals
        .iter()
        .map(|(rental_id, gpu_count)| {
            let gpu_count = (*gpu_count).max(1) as u32;

            TelemetryData {
                rental_id: rental_id.clone(),
                node_id: format!("secure-cloud-{}", rental_id),
                timestamp: Some(prost_types::Timestamp::from(SystemTime::now())),
                resource_usage: Some(ResourceUsage {
                    cpu_percent: 100.0,
                    memory_mb: 0,
                    network_rx_bytes: 0,
                    network_tx_bytes: 0,
                    disk_read_bytes: 0,
                    disk_write_bytes: 0,
                    gpu_usage: (0..gpu_count)
                        .map(|i| GpuUsage {
                            index: i,
                            utilization_percent: 100.0,
                            memory_used_mb: 0,
                            temperature_celsius: 0.0,
                            power_watts: 0,
                        })
                        .collect(),
                }),
                custom_metrics: std::collections::HashMap::new(),
            }
        })
        .collect();

    let batch_size = telemetry_batch.len();

    // Send telemetry batch to billing service
    match billing_client.ingest_telemetry(telemetry_batch).await {
        Ok(response) => {
            tracing::trace!(
                "Secure cloud billing: sent {} telemetry records (received: {}, processed: {}, failed: {})",
                batch_size,
                response.events_received,
                response.events_processed,
                response.events_failed
            );
        }
        Err(e) => {
            tracing::error!("Failed to send secure cloud billing telemetry: {}", e);
        }
    }
}

/// Process credit exhaustion check for active rentals.
/// Polls billing service for rental status and terminates rentals when credits are exhausted.
/// Reuses the same stop logic as user-initiated stops to ensure consistency.
async fn process_credit_exhaustion_check(
    billing_client: &BillingClient,
    validator_client: &ValidatorClient,
    aggregator_service: &basilica_aggregator::service::AggregatorService,
    db: &PgPool,
) {
    use basilica_protocol::billing::RentalStatus as BillingRentalStatus;

    // Query all active community cloud rentals
    let community_rentals: Vec<(String,)> =
        match sqlx::query_as("SELECT rental_id FROM user_rentals")
            .fetch_all(db)
            .await
        {
            Ok(records) => records,
            Err(e) => {
                tracing::error!("Failed to query community rentals for credit check: {}", e);
                return;
            }
        };

    // Query all active secure cloud rentals
    let secure_rentals: Vec<(String,)> =
        match sqlx::query_as("SELECT id FROM secure_cloud_rentals WHERE status = 'running'")
            .fetch_all(db)
            .await
        {
            Ok(records) => records,
            Err(e) => {
                tracing::error!(
                    "Failed to query secure cloud rentals for credit check: {}",
                    e
                );
                return;
            }
        };

    // Check community cloud rentals
    for (rental_id,) in community_rentals {
        match billing_client.get_rental_status(&rental_id).await {
            Ok(response) => {
                if let Ok(status) = BillingRentalStatus::try_from(response.status) {
                    // Stop rental if billing reports any terminal/failure status
                    let should_stop = !matches!(
                        status,
                        BillingRentalStatus::Unspecified
                            | BillingRentalStatus::Pending
                            | BillingRentalStatus::Active
                            | BillingRentalStatus::Stopping
                    );

                    if should_stop {
                        let stop_reason = match status {
                            BillingRentalStatus::FailedInsufficientCredits => {
                                "insufficient_credits"
                            }
                            BillingRentalStatus::Stopped => "billing_stopped",
                            _ => "billing_failed",
                        };
                        tracing::info!(
                            "Rental {} reported {:?} by billing, terminating",
                            rental_id,
                            status
                        );

                        // Use the shared stop logic to terminate infrastructure and archive
                        // Skip billing finalize since billing already handled it
                        match api::routes::rentals::stop_community_rental_internal(
                            validator_client,
                            Some(billing_client),
                            db,
                            &rental_id,
                            stop_reason,
                            status,
                            true, // Skip billing finalize - already done by billing service
                        )
                        .await
                        {
                            Ok(_total_cost) => {
                                tracing::info!(
                                    "Successfully stopped rental {} due to billing status {:?}",
                                    rental_id,
                                    status
                                );
                            }
                            Err(e) => {
                                tracing::error!(
                                    "Failed to stop rental {} due to billing status: {}",
                                    rental_id,
                                    e
                                );
                            }
                        }
                    }
                }
            }
            Err(e) => {
                // If billing no longer has a record for this rental, assume unsafe to keep running
                let not_found = e
                    .chain()
                    .find_map(|cause| cause.downcast_ref::<tonic::Status>())
                    .map(|s| s.code() == tonic::Code::NotFound)
                    .unwrap_or(false);

                if not_found {
                    tracing::warn!(
                        "Billing missing rental {}; stopping to avoid orphaned compute",
                        rental_id
                    );

                    if let Err(stop_err) = api::routes::rentals::stop_community_rental_internal(
                        validator_client,
                        Some(billing_client),
                        db,
                        &rental_id,
                        "billing_not_found",
                        BillingRentalStatus::Failed,
                        true, // Skip billing finalize - billing has no record
                    )
                    .await
                    {
                        tracing::error!(
                            "Failed to stop rental {} after billing missing: {}",
                            rental_id,
                            stop_err
                        );
                    }
                } else {
                    tracing::debug!(
                        "Failed to get billing status for rental {}: {}",
                        rental_id,
                        e
                    );
                }
            }
        }
    }

    // Check secure cloud rentals
    for (rental_id,) in secure_rentals {
        match billing_client.get_rental_status(&rental_id).await {
            Ok(response) => {
                if let Ok(status) = BillingRentalStatus::try_from(response.status) {
                    // Stop rental if billing reports any terminal/failure status
                    let should_stop = !matches!(
                        status,
                        BillingRentalStatus::Unspecified
                            | BillingRentalStatus::Pending
                            | BillingRentalStatus::Active
                            | BillingRentalStatus::Stopping
                    );

                    if should_stop {
                        let stop_reason = match status {
                            BillingRentalStatus::FailedInsufficientCredits => {
                                "insufficient_credits"
                            }
                            BillingRentalStatus::Stopped => "billing_stopped",
                            _ => "billing_failed",
                        };
                        tracing::info!(
                            "Secure cloud rental {} reported {:?} by billing, terminating",
                            rental_id,
                            status
                        );

                        // Use the shared stop logic to delete deployment and archive
                        // Skip billing finalize since billing already handled it
                        match api::routes::secure_cloud::stop_secure_cloud_rental_internal(
                            aggregator_service,
                            Some(billing_client),
                            db,
                            &rental_id,
                            stop_reason,
                            status,
                            true, // Skip billing finalize - already done by billing service
                        )
                        .await
                        {
                            Ok(_total_cost) => {
                                tracing::info!(
                                    "Successfully stopped secure cloud rental {} due to billing status {:?}",
                                    rental_id,
                                    status
                                );
                            }
                            Err(e) => {
                                tracing::error!(
                                    "Failed to stop secure cloud rental {} due to billing status: {}",
                                    rental_id,
                                    e
                                );
                            }
                        }
                    }
                }
            }
            Err(e) => {
                let not_found = e
                    .chain()
                    .find_map(|cause| cause.downcast_ref::<tonic::Status>())
                    .map(|s| s.code() == tonic::Code::NotFound)
                    .unwrap_or(false);

                if not_found {
                    tracing::warn!(
                        "Billing missing secure cloud rental {}; stopping to avoid orphaned compute",
                        rental_id
                    );

                    if let Err(stop_err) =
                        api::routes::secure_cloud::stop_secure_cloud_rental_internal(
                            aggregator_service,
                            Some(billing_client),
                            db,
                            &rental_id,
                            "billing_not_found",
                            BillingRentalStatus::Failed,
                            true,
                        )
                        .await
                    {
                        tracing::error!(
                            "Failed to stop secure cloud rental {} after billing missing: {}",
                            rental_id,
                            stop_err
                        );
                    }
                } else {
                    tracing::debug!(
                        "Failed to get billing status for secure cloud rental {}: {}",
                        rental_id,
                        e
                    );
                }
            }
        }
    }
}

/// Process health check for a single secure cloud rental
async fn process_secure_cloud_health_check(
    rental_id: &str,
    aggregator_service: &basilica_aggregator::service::AggregatorService,
    db: &PgPool,
    billing_client: Option<&BillingClient>,
) {
    match aggregator_service.get_deployment(rental_id).await {
        Ok(deployment) => {
            use basilica_aggregator::models::DeploymentStatus;

            // Check if deployment is in a terminal state
            if matches!(
                deployment.status,
                DeploymentStatus::Deleted | DeploymentStatus::Error
            ) {
                // Archive the rental to terminated_secure_cloud_rentals table
                if let Err(e) = crate::api::extractors::ownership::archive_secure_cloud_rental(
                    db,
                    rental_id,
                    Some("Health check: deployment no longer accessible"),
                    Some("deleted"),
                )
                .await
                {
                    tracing::error!(
                        "Failed to archive stopped secure cloud rental {}: {}",
                        rental_id,
                        e
                    );
                } else {
                    tracing::info!(
                        "Health check: Archived {:?} secure cloud rental {}",
                        deployment.status,
                        rental_id
                    );
                }
            }
        }
        Err(basilica_aggregator::error::AggregatorError::NotFound(_)) => {
            // VM was deleted externally (404 from provider)
            tracing::info!(
                "VM {} not found at provider (deleted externally), finalizing billing and archiving rental",
                rental_id
            );

            // Finalize billing before archiving
            if let Some(billing_client) = billing_client {
                use basilica_protocol::billing::{FinalizeRentalRequest, RentalStatus};
                use prost_types::Timestamp;

                let now = chrono::Utc::now();
                let end_timestamp = Timestamp {
                    seconds: now.timestamp(),
                    nanos: now.timestamp_subsec_nanos() as i32,
                };

                let finalize_request = FinalizeRentalRequest {
                    rental_id: rental_id.to_string(),
                    end_time: Some(end_timestamp),
                    termination_reason: "vm_deleted_externally".to_string(),
                    target_status: RentalStatus::Stopped.into(),
                };

                if let Err(e) = billing_client.finalize_rental(finalize_request).await {
                    tracing::error!(
                        "Failed to finalize billing for externally deleted rental {}: {}",
                        rental_id,
                        e
                    );
                } else {
                    tracing::info!(
                        "Finalized billing for externally deleted rental {}",
                        rental_id
                    );
                }
            }

            if let Err(e) = crate::api::extractors::ownership::archive_secure_cloud_rental(
                db,
                rental_id,
                Some("Health check: VM not found at provider (deleted externally)"),
                Some("deleted"),
            )
            .await
            {
                tracing::error!(
                    "Failed to archive rental {} after detecting external deletion: {}",
                    rental_id,
                    e
                );
            } else {
                tracing::info!(
                    "Successfully archived rental {} after detecting external deletion",
                    rental_id
                );
            }
        }
        Err(e) => {
            // Provider unavailable or having issues - don't change rental state
            // The provider is the source of truth and will report actual status when available
            tracing::warn!(
                "Health check failed for secure cloud rental {} (temporary error): {}",
                rental_id,
                e
            );
        }
    }
}

impl Server {
    /// Create a new server instance
    pub async fn new(config: Config) -> Result<Self> {
        info!("Initializing Basilica API Gateway server");

        let config = Arc::new(config);

        // Discover validator (or use mock in dev mode)
        let (validator_client, validator_endpoint, validator_uid, validator_hotkey) =
            if config.dev_mode {
                Self::init_dev_mode(&config)?
            } else {
                Self::init_production_mode(&config).await?
            };

        // Create HTTP client for validator communication
        let http_client = reqwest::Client::builder()
            .timeout(config.request_timeout())
            .connect_timeout(config.connection_timeout())
            .pool_max_idle_per_host(10)
            .build()
            .map_err(ApiError::HttpClient)?;

        // Initialize database connection
        info!("Initializing database connection");

        let db = PgPoolOptions::new()
            .max_connections(config.database.max_connections)
            .connect(&config.database.url)
            .await
            .map_err(|e| ApiError::Internal {
                message: format!("Failed to connect to database: {}", e),
            })?;

        // Run migrations
        info!("Running database migrations");
        sqlx::migrate!("./migrations")
            .run(&db)
            .await
            .map_err(|e| ApiError::Internal {
                message: format!("Failed to run migrations: {}", e),
            })?;

        // Initialize Kubernetes client (optional)
        let k8s: Option<Arc<dyn ApiK8sClient + Send + Sync>> = match K8sClient::try_default().await
        {
            Ok(c) => {
                info!("Initialized Kubernetes client for API integration");
                Some(Arc::new(c))
            }
            Err(e) => {
                warn!(
                    "K8s client unavailable: {} (continuing without K8s integration)",
                    e
                );
                None
            }
        };

        // Initialize payments service client if enabled
        let payments_client = if config.payments.enabled {
            info!(
                "Initializing payments service client at {}",
                config.payments.endpoint
            );
            match PaymentsClient::new(&config.payments.endpoint).await {
                Ok(client) => {
                    info!("Successfully connected to payments service");
                    Some(Arc::new(client))
                }
                Err(e) => {
                    warn!(
                        "Failed to connect to payments service: {}. Payments features will be disabled.",
                        e
                    );
                    None
                }
            }
        } else {
            info!("Payments service integration is disabled");
            None
        };

        // Initialize billing service client if enabled
        let billing_client = if config.billing.enabled {
            info!(
                "Initializing billing service client at {}",
                config.billing.endpoint
            );
            match BillingClient::new(&config.billing.endpoint).await {
                Ok(client) => {
                    info!("Successfully connected to billing service");
                    Some(Arc::new(client))
                }
                Err(e) => {
                    warn!(
                        "Failed to connect to billing service: {}. Billing features will be disabled.",
                        e
                    );
                    None
                }
            }
        } else {
            info!("Billing service integration is disabled");
            None
        };

        // Initialize metrics system if enabled
        let metrics = if config.metrics.enabled {
            info!("Initializing metrics system");
            match crate::metrics::ApiMetricsSystem::new(config.metrics.clone()) {
                Ok(system) => {
                    info!("Successfully initialized metrics system");
                    Some(Arc::new(system))
                }
                Err(e) => {
                    warn!(
                        "Failed to initialize metrics system: {}. Metrics will be disabled.",
                        e
                    );
                    None
                }
            }
        } else {
            info!("Metrics collection is disabled");
            None
        };

        // Initialize GPU Aggregator service (Secure Cloud)
        info!("Initializing GPU Aggregator service (Secure Cloud)");

        // Debug: Show provider configuration
        tracing::debug!(
            "DataCrunch config: client_id={:?}, has_secret={}",
            config.aggregator.providers.datacrunch.client_id,
            config
                .aggregator
                .providers
                .datacrunch
                .client_secret
                .is_some()
        );

        let aggregator_db = Arc::new(
            AggregatorDatabase::new(&config.database.url)
                .await
                .map_err(|e| ApiError::Internal {
                    message: format!("Failed to initialize aggregator database: {}", e),
                })?,
        );

        let aggregator_config = config.to_aggregator_config();
        let aggregator_service = Arc::new(
            AggregatorService::new(aggregator_db, aggregator_config.clone()).map_err(|e| {
                ApiError::Internal {
                    message: format!("Failed to initialize aggregator service: {}", e),
                }
            })?,
        );
        info!("GPU Aggregator service initialized successfully");

        // Initialize DNS provider for public deployments
        let dns_provider: Option<Arc<dyn crate::dns::DnsProvider>> = if config.dns.enabled {
            let dns_config = config.dns.from_env();
            if let Err(e) = dns_config.validate() {
                warn!(
                    "DNS configuration invalid: {}. Public deployments will be disabled.",
                    e
                );
                None
            } else {
                match crate::dns::cloudflare::CloudflareDnsManager::new(
                    crate::dns::cloudflare::CloudflareConfig {
                        api_token: dns_config.api_token.unwrap(),
                        zone_id: dns_config.zone_id.unwrap(),
                        domain: dns_config.domain.clone(),
                        proxy: dns_config.proxy,
                    },
                ) {
                    Ok(manager) => {
                        info!(
                            "DNS provider initialized successfully for domain: {}",
                            dns_config.domain
                        );
                        Some(Arc::new(manager) as Arc<dyn crate::dns::DnsProvider>)
                    }
                    Err(e) => {
                        warn!("Failed to initialize DNS provider: {}. Public deployments will be disabled.", e);
                        None
                    }
                }
            }
        } else {
            info!("DNS management is disabled");
            None
        };

        // Write SSH private key to disk if provided via environment variable
        if let Ok(ssh_key_content) = std::env::var("SSH_PRIVATE_KEY") {
            let key_dir = std::path::PathBuf::from("/tmp/.ssh");
            let key_path = key_dir.join("k3s_key");

            std::fs::create_dir_all(&key_dir).map_err(|e| ApiError::Internal {
                message: format!("Failed to create SSH key directory: {}", e),
            })?;

            std::fs::write(&key_path, ssh_key_content).map_err(|e| ApiError::Internal {
                message: format!("Failed to write SSH key: {}", e),
            })?;

            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&key_path, std::fs::Permissions::from_mode(0o600))
                    .map_err(|e| ApiError::Internal {
                        message: format!("Failed to set SSH key permissions: {}", e),
                    })?;
            }

            info!("SSH private key written to {}", key_path.display());
        }

        // Initialize SSH client for K3s token generation
        let ssh_client = Arc::new(
            crate::ssh::K3sSshClient::new(&config.k3s_ssh)
                .map_err(|e| {
                    warn!(
                        "Failed to initialize SSH client: {}. SSH token creation will be disabled.",
                        e
                    );
                    e
                })
                .unwrap_or_else(|_| crate::ssh::K3sSshClient::disabled()),
        );

        if ssh_client.is_enabled() {
            info!("SSH client initialized for K3s token generation");
        } else {
            info!("SSH token generation is disabled");
        }

        // Create application state
        let state = AppState {
            config: config.clone(),
            validator_client: validator_client.clone(),
            validator_endpoint: validator_endpoint.clone(),
            validator_uid,
            validator_hotkey,
            http_client: http_client.clone(),
            db,
            k8s,
            payments_client,
            billing_client,
            dns_provider,
            metrics,
            aggregator_service,
            aggregator_config,
            pricing_config: config.pricing.clone(),
            ssh_client,
        };

        // Start health check tasks (skip in dev mode - no real validator)
        if !config.dev_mode {
            // Start optional health check task using HTTP client
            let health_http_client = http_client;
            let health_endpoint = validator_endpoint.clone();
            let health_interval = config.health_check_interval();
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(health_interval);
                loop {
                    interval.tick().await;
                    let health_url = format!("{health_endpoint}/health");
                    match health_http_client.get(&health_url).send().await {
                        Ok(response) if response.status().is_success() => {
                            tracing::trace!("Validator health check passed");
                        }
                        Ok(response) => {
                            tracing::warn!(
                                "Validator health check returned status: {}",
                                response.status()
                            );
                        }
                        Err(e) => {
                            tracing::warn!(
                                "Validator health check failed for {}: {}",
                                health_endpoint,
                                e
                            );
                        }
                    }
                }
            });

            // Start rental health check task
            let rental_health_client = validator_client.clone();
            let rental_health_db = state.db.clone();
            let rental_health_interval = config.rental_health_check_interval();

            tokio::spawn(async move {
                let mut interval = tokio::time::interval(rental_health_interval);
                loop {
                    interval.tick().await;

                    // Query all active rentals from the database
                    match sqlx::query_as::<_, (String,)>("SELECT rental_id FROM user_rentals")
                        .fetch_all(&rental_health_db)
                        .await
                    {
                        Ok(rental_records) => {
                            // TODO: Consider batching these API calls for better performance
                            // Currently making individual API calls for each rental
                            for record in &rental_records {
                                let rental_id = &record.0;
                                process_rental_health_check(
                                    rental_id,
                                    &rental_health_client,
                                    &rental_health_db,
                                )
                                .await;
                            }

                            if !rental_records.is_empty() {
                                tracing::debug!(
                                    "Rental health check completed for {} rentals",
                                    rental_records.len()
                                );
                            }
                        }
                        Err(e) => {
                            tracing::error!(
                                "Failed to query active rentals for health check: {}",
                                e
                            );
                        }
                    }
                }
            });

            tracing::info!(
                "Started rental health check task (interval: {} seconds)",
                rental_health_interval.as_secs()
            );
        } else {
            tracing::info!("🔧 DEV MODE: Skipping validator health check tasks");
        }

        // Start GPU offerings refresh task (Secure Cloud)
        let refresh_aggregator_service = state.aggregator_service.clone();
        let refresh_interval = std::time::Duration::from_secs(60); // Refresh every 60 seconds

        tokio::spawn(async move {
            // Do initial refresh immediately on startup
            tracing::info!("Starting initial GPU offerings refresh...");
            match refresh_aggregator_service.refresh_all_providers().await {
                Ok(count) => {
                    tracing::info!("Initial GPU offerings refresh: fetched {} offerings", count);
                }
                Err(e) => {
                    tracing::error!("Failed initial GPU offerings refresh: {}", e);
                }
            }

            // Then start periodic refresh
            let mut interval = tokio::time::interval(refresh_interval);
            loop {
                interval.tick().await;

                tracing::trace!("Running periodic GPU offerings refresh...");
                match refresh_aggregator_service.refresh_all_providers().await {
                    Ok(count) => {
                        if count > 0 {
                            tracing::trace!("GPU offerings refresh: fetched {} offerings", count);
                        } else {
                            tracing::trace!(
                                "GPU offerings refresh: no new offerings (cooldown or no providers enabled)"
                            );
                        }
                    }
                    Err(e) => {
                        tracing::error!("Failed to refresh GPU offerings: {}", e);
                    }
                }
            }
        });

        tracing::info!(
            "Started GPU offerings refresh task (interval: {} seconds)",
            refresh_interval.as_secs()
        );

        // Start VIP poller task if configured
        tracing::info!(
            vip_configured = config.aggregator.vip.is_some(),
            vip_mock_user_id = ?config.aggregator.vip.as_ref().and_then(|v| v.mock_user_id.clone()),
            vip_is_configured = config.aggregator.vip.as_ref().is_some_and(|v| v.is_configured()),
            vip_is_mock_mode = config.aggregator.vip.as_ref().is_some_and(|v| v.is_mock_mode()),
            "VIP configuration check"
        );
        if let Some(vip_config) = config.aggregator.vip.as_ref().filter(|v| v.is_configured()) {
            let vip_config = vip_config.clone();
            let vip_db = state.db.clone();

            if vip_config.is_mock_mode() {
                // Mock mode - use fake VIP data
                let mock_user_id = vip_config.mock_user_id.clone().unwrap();
                tracing::info!(user_id = %mock_user_id, "Starting VIP poller in MOCK mode");

                let mock_data = create_mock_vip_data(&mock_user_id);
                let mock_source = MockVipDataSource::new(mock_data);

                let vip_cache = Arc::new(VipCache::new());
                if let Err(e) = vip_cache.rebuild_from_db(&vip_db).await {
                    tracing::error!(error = %e, "Failed to rebuild VIP cache from database");
                }

                let poller = Arc::new(VipPoller::new(
                    vip_config.clone(),
                    mock_source,
                    vip_cache,
                    vip_db,
                    config.pricing.secure_cloud_markup_percent,
                    state.billing_client.clone(),
                ));

                tracing::info!("Starting initial VIP poll (mock mode)...");
                match poller.poll_once().await {
                    Ok(stats) => {
                        tracing::info!(
                            created = stats.created,
                            updated = stats.updated,
                            removed = stats.removed,
                            "Initial VIP poll completed (mock mode)"
                        );
                    }
                    Err(e) => {
                        tracing::error!(error = %e, "Initial VIP poll failed (mock mode)");
                    }
                }

                let task = VipPollerTask::new(poller, vip_config.poll_interval_secs);
                task.start();

                tracing::info!(
                    interval_secs = vip_config.poll_interval_secs,
                    "VIP poller task started (mock mode)"
                );
            } else {
                // Real mode - use CSV data source (S3 or local file)
                let data_source_result = if vip_config.has_s3_source() {
                    let bucket = vip_config.s3_bucket.as_ref().unwrap();
                    let key = vip_config.s3_key.as_ref().unwrap();
                    let region = vip_config.s3_region.clone();
                    tracing::info!(bucket = %bucket, key = %key, region = ?region, "VIP: Using S3 CSV source");
                    CsvDataSource::from_s3(bucket.clone(), key.clone(), region).await
                } else if vip_config.has_local_csv() {
                    let file_path = vip_config.csv_file_path.as_ref().unwrap();
                    tracing::info!(file_path = %file_path, "VIP: Using local CSV file");
                    Ok(CsvDataSource::from_local(file_path.clone()))
                } else {
                    tracing::warn!("VIP: No CSV source configured - VIP polling disabled");
                    Err(basilica_aggregator::vip::DataSourceError::FileRead(
                        std::io::Error::new(
                            std::io::ErrorKind::NotFound,
                            "No CSV source configured",
                        ),
                    ))
                };

                match data_source_result {
                    Ok(data_source) => {
                        let vip_cache = Arc::new(VipCache::new());

                        // Rebuild cache from DB on startup
                        if let Err(e) = vip_cache.rebuild_from_db(&vip_db).await {
                            tracing::error!(error = %e, "Failed to rebuild VIP cache from database");
                        }

                        let poller = Arc::new(VipPoller::new(
                            vip_config.clone(),
                            data_source,
                            vip_cache,
                            vip_db,
                            config.pricing.secure_cloud_markup_percent,
                            state.billing_client.clone(),
                        ));

                        // Do initial poll immediately
                        tracing::info!("Starting initial VIP poll...");
                        match poller.poll_once().await {
                            Ok(stats) => {
                                tracing::info!(
                                    created = stats.created,
                                    updated = stats.updated,
                                    removed = stats.removed,
                                    "Initial VIP poll completed"
                                );
                            }
                            Err(e) => {
                                tracing::error!(error = %e, "Initial VIP poll failed");
                            }
                        }

                        // Start periodic polling
                        let task = VipPollerTask::new(poller, vip_config.poll_interval_secs);
                        task.start();

                        tracing::info!(
                            interval_secs = vip_config.poll_interval_secs,
                            "VIP poller task started"
                        );
                    }
                    Err(e) => {
                        tracing::error!(
                            error = %e,
                            "Failed to initialize CSV data source for VIP - VIP polling disabled"
                        );
                    }
                }
            }
        } else {
            tracing::debug!("VIP poller not configured, skipping");
        }

        // Start Secure Cloud health check task
        let health_check_aggregator_service = state.aggregator_service.clone();
        let health_check_db = state.db.clone();
        let health_check_billing_client = state.billing_client.clone();
        let health_check_interval = config.rental_health_check_interval();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(health_check_interval);
            loop {
                interval.tick().await;

                // Query all active secure cloud rentals from the database (excluding VIP rentals
                // which don't need health checks - they're manually managed and assumed always up,
                // and excluding Hyperstack rentals which use webhook callbacks instead of polling)
                match sqlx::query_as::<_, (String,)>(
                    "SELECT id FROM secure_cloud_rentals WHERE is_vip = FALSE AND provider != 'hyperstack'",
                )
                .fetch_all(&health_check_db)
                .await
                {
                    Ok(rental_records) => {
                        // Process rentals sequentially to avoid overwhelming provider APIs
                        for record in &rental_records {
                            let rental_id = &record.0;
                            process_secure_cloud_health_check(
                                rental_id,
                                &health_check_aggregator_service,
                                &health_check_db,
                                health_check_billing_client.as_ref().map(|c| c.as_ref()),
                            )
                            .await;
                        }

                        if !rental_records.is_empty() {
                            tracing::trace!(
                                "Secure Cloud health check completed for {} rentals",
                                rental_records.len()
                            );
                        }
                    }
                    Err(e) => {
                        tracing::error!(
                            "Failed to query secure cloud rentals for health check: {}",
                            e
                        );
                    }
                }
            }
        });

        tracing::info!(
            "Started Secure Cloud health check task (interval: {} seconds)",
            health_check_interval.as_secs()
        );

        // Start Secure Cloud billing task (sends synthetic telemetry for active rentals)
        if let Some(billing_client) = state.billing_client.clone() {
            let billing_task_db = state.db.clone();
            let billing_interval = std::time::Duration::from_secs(30); // Same as validator telemetry interval

            tokio::spawn(async move {
                let mut interval = tokio::time::interval(billing_interval);
                loop {
                    interval.tick().await;
                    process_secure_cloud_billing(&billing_client, &billing_task_db).await;
                }
            });

            tracing::info!(
                "Started Secure Cloud billing task (interval: {} seconds)",
                billing_interval.as_secs()
            );
        } else {
            tracing::info!("Secure Cloud billing task not started (billing service disabled)");
        }

        // Start credit exhaustion check task (polls billing for insufficient credits)
        if let Some(billing_client) = state.billing_client.clone() {
            let credit_check_db = state.db.clone();
            let credit_check_validator = state.validator_client.clone();
            let credit_check_aggregator = state.aggregator_service.clone();
            let credit_check_billing = billing_client.clone();
            let credit_check_interval = std::time::Duration::from_secs(30);

            tokio::spawn(async move {
                let mut interval = tokio::time::interval(credit_check_interval);
                loop {
                    interval.tick().await;
                    process_credit_exhaustion_check(
                        &credit_check_billing,
                        &credit_check_validator,
                        &credit_check_aggregator,
                        &credit_check_db,
                    )
                    .await;
                }
            });

            tracing::info!(
                "Started credit exhaustion check task (interval: {} seconds)",
                credit_check_interval.as_secs()
            );
        }

        // Start node token cleanup task
        let cleanup_db = state.db.clone();
        let cleanup_interval = config.node_token_cleanup_interval();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(cleanup_interval);
            loop {
                interval.tick().await;

                match crate::k8s::cleanup_expired_cluster_tokens(&cleanup_db).await {
                    Ok(count) if count > 0 => {
                        tracing::info!("Node token cleanup: removed {} expired tokens", count);
                    }
                    Ok(_) => {
                        tracing::debug!("Node token cleanup: no expired tokens found");
                    }
                    Err(e) => {
                        tracing::error!("Node token cleanup failed: {}", e);
                    }
                }
            }
        });

        tracing::info!(
            "Started node token cleanup task (interval: {} seconds)",
            cleanup_interval.as_secs()
        );

        // Build the application router
        let app = Self::build_router(state.clone())?;

        let cancel_token = tokio_util::sync::CancellationToken::new();
        let background_tasks = tokio::task::JoinSet::new();

        Ok(Self {
            config,
            app,
            cancel_token,
            background_tasks: Some(background_tasks),
        })
    }

    /// Initialize dev mode - bypasses Bittensor with mock validator
    fn init_dev_mode(config: &Config) -> Result<(Arc<ValidatorClient>, String, u16, String)> {
        use crate::mock::{DEV_VALIDATOR_ENDPOINT, DEV_VALIDATOR_HOTKEY, DEV_VALIDATOR_UID};

        info!("🔧 DEV MODE: Using mock validator (Bittensor bypassed)");

        // Create a mock validator client pointing to localhost
        // In dev mode, we don't actually need a real validator running
        let validator_endpoint = DEV_VALIDATOR_ENDPOINT.to_string();
        let validator_uid = DEV_VALIDATOR_UID;
        let validator_hotkey = DEV_VALIDATOR_HOTKEY.to_string();

        // Create validator client (it won't be used for actual calls in dev mode)
        let validator_client = Arc::new(
            ValidatorClient::new(&validator_endpoint, config.request_timeout()).map_err(|e| {
                ApiError::Internal {
                    message: format!("Failed to create validator client: {e}"),
                }
            })?,
        );

        info!(
            "🔧 DEV MODE: Mock validator configured (uid={}, endpoint={})",
            validator_uid, validator_endpoint
        );

        Ok((
            validator_client,
            validator_endpoint,
            validator_uid,
            validator_hotkey,
        ))
    }

    /// Initialize production mode - discovers validator via Bittensor
    async fn init_production_mode(
        config: &Config,
    ) -> Result<(Arc<ValidatorClient>, String, u16, String)> {
        // Validate configuration
        if config.bittensor.validator_hotkey.is_empty() {
            return Err(ApiError::ConfigError(
                "validator_hotkey must be configured in bittensor section".to_string(),
            ));
        }

        // Initialize Bittensor service to find validator endpoint
        info!("Connecting to Bittensor network to discover validator endpoint");
        let bittensor_config = config.to_bittensor_config();
        let bittensor_service = bittensor::Service::new(bittensor_config).await?;

        // Query metagraph to find validator by hotkey
        info!(
            "Looking up validator with hotkey: {}",
            config.bittensor.validator_hotkey
        );
        let metagraph = bittensor_service
            .get_metagraph(config.bittensor.netuid)
            .await?;

        // Use NeuronDiscovery to find validator
        let discovery = bittensor::NeuronDiscovery::new(&metagraph);
        let validator_info = discovery
            .find_neuron_by_hotkey(&config.bittensor.validator_hotkey)
            .ok_or_else(|| {
                ApiError::ConfigError(format!(
                    "Validator with hotkey {} not found in subnet {}",
                    config.bittensor.validator_hotkey, config.bittensor.netuid
                ))
            })?;

        // Verify it's actually a validator (has validator_permit)
        if !validator_info.is_validator {
            return Err(ApiError::ConfigError(format!(
                "Hotkey {} exists but does not have validator permit in subnet {}",
                config.bittensor.validator_hotkey, config.bittensor.netuid
            )));
        }

        let validator_uid = validator_info.uid;

        // Get axon info from the validator info
        let axon_info = validator_info.axon_info.ok_or_else(|| {
            ApiError::ConfigError(format!("No axon info found for validator {validator_uid}"))
        })?;

        let validator_endpoint = format!("http://{}:{}", axon_info.ip, axon_info.port);
        info!(
            "Found validator {} at endpoint {}",
            validator_uid, validator_endpoint
        );

        // Create validator client
        let validator_client = Arc::new(
            ValidatorClient::new(&validator_endpoint, config.request_timeout()).map_err(|e| {
                ApiError::Internal {
                    message: format!("Failed to create validator client: {e}"),
                }
            })?,
        );

        Ok((
            validator_client,
            validator_endpoint,
            validator_uid,
            config.bittensor.validator_hotkey.clone(),
        ))
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
            .merge(api::routes(state.clone()))
            .layer(middleware)
            .with_state(state);

        Ok(app)
    }

    /// Run the server until shutdown signal
    pub async fn run(mut self) -> Result<()> {
        let addr = self.config.server.bind_address;

        info!("Starting HTTP server on {}", addr);

        let listener =
            tokio::net::TcpListener::bind(addr)
                .await
                .map_err(|e| ApiError::Internal {
                    message: format!("Failed to bind to address {addr}: {e}"),
                })?;

        info!("Basilica API Gateway listening on {}", addr);

        let cancel_token = self.cancel_token.clone();
        let shutdown_signal = async move {
            shutdown_signal().await;
            cancel_token.cancel();
        };

        axum::serve(listener, self.app)
            .with_graceful_shutdown(shutdown_signal)
            .await
            .map_err(|e| ApiError::Internal {
                message: format!("Server error: {e}"),
            })?;

        if let Some(mut tasks) = self.background_tasks.take() {
            info!("Waiting for background tasks to complete");
            while let Some(result) = tasks.join_next().await {
                if let Err(e) = result {
                    tracing::error!("Background task error: {}", e);
                }
            }
            info!("All background tasks completed");
        }

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

/// Create mock VIP data for testing
fn create_mock_vip_data(user_id: &str) -> Vec<VipCsvRow> {
    use rust_decimal::Decimal;

    vec![
        VipCsvRow {
            vip_machine_id: "mock-vip-h100-01".to_string(),
            assigned_user: user_id.to_string(),
            active: true,
            ssh_host: "mock-h100-01.basilica.dev".to_string(),
            ssh_port: 22,
            ssh_user: "ubuntu".to_string(),
            gpu_type: "H100".to_string(),
            gpu_count: 8,
            region: "us-west-2".to_string(),
            hourly_rate: Decimal::new(2500, 2), // $25.00/hr
            vcpu_count: 128,
            system_memory_gb: 500,
            notes: Some("Mock H100 cluster for testing".to_string()),
        },
        VipCsvRow {
            vip_machine_id: "mock-vip-a100-01".to_string(),
            assigned_user: user_id.to_string(),
            active: true,
            ssh_host: "mock-a100-01.basilica.dev".to_string(),
            ssh_port: 22,
            ssh_user: "ubuntu".to_string(),
            gpu_type: "A100-80GB".to_string(),
            gpu_count: 4,
            region: "us-east-1".to_string(),
            hourly_rate: Decimal::new(1200, 2), // $12.00/hr
            vcpu_count: 64,
            system_memory_gb: 256,
            notes: Some("Mock A100 for testing".to_string()),
        },
    ]
}
