use basilica_payments::{
    blockchain::{local_treasury::LocalTreasury, monitor::ChainMonitor},
    config::PaymentsConfig,
    domain::price::PriceConverter,
    grpc::payments_service::PaymentsServer as GrpcPaymentsServer,
    price_oracle::{PriceOracle, PriceOracleConfig},
    processor::{billing_client::GrpcBillingClient, dispatcher::OutboxDispatcher},
    server::PaymentsServer,
    storage::PgRepos,
};

use basilica_common::crypto::Aead;

use anyhow::{Context, Result};
use basilica_protocol::payments::payments_service_server::PaymentsServiceServer;
use clap::Parser;
use clap_verbosity_flag::{InfoLevel, Verbosity};
use sqlx::postgres::PgPoolOptions;
use std::{net::SocketAddr, path::PathBuf, sync::Arc};
use tokio::signal;
use tonic::transport::Server;
use tracing::{info, warn};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
#[command(name = "basilica-payments")]
#[command(
    about = "Basilica Payments Service - Blockchain payment processing and treasury management"
)]
struct Args {
    #[arg(short, long, help = "Path to configuration file")]
    config: Option<PathBuf>,

    #[arg(long, help = "Generate sample configuration file")]
    gen_config: bool,

    #[arg(long, help = "Dry run mode (validate config without starting)")]
    dry_run: bool,

    #[command(flatten)]
    verbosity: Verbosity<InfoLevel>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging using the unified system
    let binary_name = env!("CARGO_BIN_NAME").replace("-", "_");
    let base_filter = format!("basilica_protocol=info,{}", binary_name);
    let default_filter = format!("basilica_protocol=info,{}=info", binary_name);
    basilica_common::logging::init_logging(&args.verbosity, &base_filter, &default_filter)?;

    if args.gen_config {
        let config = PaymentsConfig::default();
        let toml = toml::to_string_pretty(&config)?;
        println!("{}", toml);
        return Ok(());
    }

    let cfg = PaymentsConfig::load(args.config).context("Failed to load configuration")?;
    cfg.validate().context("Configuration validation failed")?;

    let warnings = cfg.warnings();
    for warning in warnings {
        warn!("{}", warning);
    }

    if args.dry_run {
        info!("Configuration loaded successfully (dry-run mode)");
        info!(
            "gRPC listen address: {}:{}",
            cfg.grpc.listen_address, cfg.grpc.port
        );
        info!(
            "HTTP listen address: {}:{}",
            cfg.http.listen_address, cfg.http.port
        );
        info!("Substrate websocket: {}", cfg.blockchain.websocket_url);
        info!("Billing gRPC endpoint: {}", cfg.billing.grpc_endpoint);
        let db_display = match cfg.database.url.rsplit_once('@') {
            Some((_, rest)) => rest,
            None => &cfg.database.url,
        };
        info!("Database URL: {}", db_display);
        info!("Configuration validated successfully (dry-run mode)");
        return Ok(());
    }

    info!("Starting basilica-payments service");
    info!(
        "gRPC listen address: {}:{}",
        cfg.grpc.listen_address, cfg.grpc.port
    );
    info!(
        "HTTP listen address: {}:{}",
        cfg.http.listen_address, cfg.http.port
    );
    info!("Substrate websocket: {}", cfg.blockchain.websocket_url);
    if !cfg.blockchain.fallback_websocket_urls.is_empty() {
        info!(
            "Using {} fallback endpoint(s)",
            cfg.blockchain.fallback_websocket_urls.len()
        );
    }
    info!("Billing gRPC endpoint: {}", cfg.billing.grpc_endpoint);

    let db_display = match cfg.database.url.rsplit_once('@') {
        Some((_, rest)) => rest,
        None => &cfg.database.url,
    };
    info!("Connecting to database: {}", db_display);
    let pool = PgPoolOptions::new()
        .max_connections(cfg.database.max_connections)
        .min_connections(cfg.database.min_connections)
        .acquire_timeout(cfg.acquire_timeout())
        .idle_timeout(cfg.idle_timeout())
        .max_lifetime(cfg.max_lifetime())
        .connect(&cfg.database.url)
        .await
        .context("Failed to connect to database")?;

    // Run migrations
    info!("Running database migrations");
    sqlx::migrate!("./migrations")
        .run(&pool)
        .await
        .context("Failed to run database migrations")?;
    info!("Database migrations completed successfully");

    let repos = PgRepos::new(pool.clone());

    let aead = Arc::new(
        Aead::new(&cfg.treasury.aead_key_hex).context("Failed to initialize AEAD encryption")?,
    );
    let treasury = Arc::new(LocalTreasury::new(cfg.blockchain.ss58_prefix));

    info!(
        "Connecting to billing service at: {}",
        cfg.billing.grpc_endpoint
    );
    let billing = GrpcBillingClient::connect(&cfg.billing.grpc_endpoint)
        .await
        .context("Failed to connect to billing service")?;

    let oracle_config = PriceOracleConfig {
        update_interval: cfg.price_oracle.update_interval_seconds,
        max_price_age: cfg.price_oracle.max_price_age_seconds,
        request_timeout: cfg.price_oracle.request_timeout_seconds,
    };

    let oracle = Arc::new(PriceOracle::new(oracle_config));
    let oracle_for_updates = Arc::clone(&oracle);
    oracle_for_updates.run().await;

    let price = PriceConverter::new(oracle, cfg.treasury.tao_decimals);

    let grpc_svc = GrpcPaymentsServer::new(repos.clone(), treasury, aead).into_service();
    let dispatcher = OutboxDispatcher::new(repos.clone(), billing, price);

    info!(
        "Connecting to substrate node at: {}",
        cfg.blockchain.websocket_url
    );
    let mut endpoints = vec![cfg.blockchain.websocket_url.clone()];
    endpoints.extend(cfg.blockchain.fallback_websocket_urls.clone());
    let monitor = ChainMonitor::new(repos.clone(), endpoints)
        .await
        .context("Failed to initialize blockchain monitor")?;

    let (mut health_reporter, health_service) = tonic_health::server::health_reporter();
    health_reporter
        .set_serving::<PaymentsServiceServer<GrpcPaymentsServer<LocalTreasury>>>()
        .await;

    let grpc_bind: SocketAddr = format!("{}:{}", cfg.grpc.listen_address, cfg.grpc.port)
        .parse()
        .context("Failed to parse gRPC bind address")?;

    info!("Starting gRPC server on {}", grpc_bind);

    // Start HTTP server
    let http_server = PaymentsServer::new(cfg.clone(), Arc::new(pool));
    let http_handle = tokio::spawn(async move { http_server.serve(shutdown_signal()).await });

    tokio::select! {
        r = Server::builder()
            .add_service(health_service)
            .add_service(grpc_svc)
            .serve(grpc_bind) => {
            r.context("gRPC server failed")?;
        },
        r = dispatcher.run() => {
            r.context("Outbox dispatcher failed")?;
        },
        r = monitor.run() => {
            r.context("Blockchain monitor failed")?;
        },
        r = http_handle => {
            r.context("HTTP server task failed")?.context("HTTP server failed")?;
        },
        _ = shutdown_signal() => {
            warn!("Received shutdown signal");
        }
    }

    info!("Basilica payments service shutting down gracefully");
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("Failed to install signal handler")
            .recv()
            .await;
    };

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
