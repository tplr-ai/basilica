//! Main entry point for the Basilica API Gateway

use axum::{routing::get, Router};
use basilica_api::{config::Config, server::Server, Result};
use clap::Parser;
use clap_verbosity_flag::{InfoLevel, Verbosity};
use std::path::PathBuf;
use tracing::{error, info};

#[derive(Parser)]
#[command(name = "basilica-api", about = "Basilica API Gateway", version, author)]
struct Args {
    /// Path to configuration file
    #[arg(short, long, value_name = "FILE")]
    config: Option<PathBuf>,

    /// Generate example configuration file
    #[arg(long)]
    gen_config: bool,

    /// Dev mode - bypasses Bittensor/Validator with mock data
    /// Useful for local development and TUI testing
    #[arg(long)]
    dev: bool,

    #[command(flatten)]
    verbosity: Verbosity<InfoLevel>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging using the unified system
    let binary_name = env!("CARGO_BIN_NAME").replace("-", "_");
    let default_filter = format!("{}=info,basilica_aggregator=debug", binary_name);
    basilica_common::logging::init_logging(&args.verbosity, &binary_name, &default_filter)?;

    // Install Prometheus metrics recorder and expose /metrics on a separate listener
    let handle = metrics_exporter_prometheus::PrometheusBuilder::new()
        .install_recorder()
        .expect("install prometheus recorder");
    let metrics_handle = handle.clone();
    let metrics_app = Router::new().route(
        "/metrics",
        get(move || {
            let h = metrics_handle.clone();
            async move {
                let body = h.render();
                Ok::<_, std::convert::Infallible>(
                    axum::response::Response::builder()
                        .header(
                            axum::http::header::CONTENT_TYPE,
                            "text/plain; version=0.0.4",
                        )
                        .body(axum::body::Body::from(body))
                        .unwrap(),
                )
            }
        }),
    );
    let metrics_bind =
        std::env::var("BASILICA_API_METRICS_ADDR").unwrap_or_else(|_| "0.0.0.0:9401".into());
    let metrics_listener = tokio::net::TcpListener::bind(&metrics_bind)
        .await
        .expect("bind API metrics addr");
    tokio::spawn(async move {
        axum::serve(metrics_listener, metrics_app)
            .await
            .expect("serve API metrics");
    });

    info!("Starting Basilica API Gateway v{}", basilica_api::VERSION);

    // Handle config generation
    if args.gen_config {
        let example_config = Config::generate_example()?;
        println!("{example_config}");
        return Ok(());
    }

    // Load configuration
    let mut config = Config::load(args.config)?;

    // CLI flag overrides config file
    if args.dev {
        config.dev_mode = true;
    }

    if config.dev_mode {
        info!("🔧 Running in DEV MODE - Bittensor/Validator bypassed with mock data");
    }

    info!(
        "Configuration loaded, binding to {}",
        config.server.bind_address
    );

    // Create and run server
    let server = Server::new(config).await?;

    info!("Basilica API Gateway initialized successfully");

    // Run until shutdown signal
    match server.run().await {
        Ok(()) => {
            info!("Basilica API Gateway shut down gracefully");
            Ok(())
        }
        Err(e) => {
            error!("Basilica API Gateway error: {}", e);
            Err(e)
        }
    }
}
