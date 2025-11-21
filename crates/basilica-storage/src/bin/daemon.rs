//! Basilica Storage Daemon
//!
//! FUSE filesystem daemon that provides transparent object storage for stateful workloads.
//! Automatically syncs file I/O to object storage (R2/S3) with lazy loading and continuous background sync.

use anyhow::{Context, Result};
use basilica_storage::{
    backend::{S3Backend, StorageBackend},
    config::StorageConfig,
    fuse::{BasilicaFS, SharedBasilicaFS},
    http::HttpServer,
    StorageMetrics,
};
use clap::Parser;
use fuser::MountOption;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;

/// Basilica Storage Daemon - FUSE filesystem with object storage backend
#[derive(Parser, Debug)]
#[command(name = "basilica-storage-daemon")]
#[command(about = "FUSE filesystem daemon with transparent object storage", long_about = None)]
struct Args {
    /// Kubernetes namespace (used as prefix in object storage)
    #[arg(short = 'n', long, env = "NAMESPACE")]
    namespace: String,

    /// Experiment ID (used as prefix in object storage)
    #[arg(short, long, env = "EXPERIMENT_ID")]
    experiment_id: String,

    /// Mount point for the FUSE filesystem
    #[arg(short, long, default_value = "/data", env = "MOUNT_POINT")]
    mount_point: PathBuf,

    /// Storage backend type (r2, s3, gcs)
    #[arg(short = 'b', long, default_value = "r2", env = "STORAGE_BACKEND")]
    backend: String,

    /// S3/R2 bucket name
    #[arg(long, env = "STORAGE_BUCKET")]
    bucket: String,

    /// S3/R2 region
    #[arg(long, env = "STORAGE_REGION")]
    region: Option<String>,

    /// S3/R2 endpoint (required for R2, optional for S3)
    #[arg(long, env = "STORAGE_ENDPOINT")]
    endpoint: Option<String>,

    /// S3/R2 access key ID
    #[arg(long, env = "STORAGE_ACCESS_KEY_ID")]
    access_key_id: Option<String>,

    /// S3/R2 secret access key
    #[arg(long, env = "STORAGE_SECRET_ACCESS_KEY")]
    secret_access_key: Option<String>,

    /// Background sync interval in milliseconds
    #[arg(long, default_value = "1000", env = "SYNC_INTERVAL_MS")]
    sync_interval_ms: u64,

    /// Cache size in MB
    #[arg(long, default_value = "2048", env = "CACHE_SIZE_MB")]
    cache_size_mb: usize,

    /// Storage quota in GB
    #[arg(long, default_value = "100", env = "QUOTA_GB")]
    quota_gb: u64,

    /// Allow other users to access the filesystem
    #[arg(long)]
    allow_other: bool,

    /// Automatically unmount on process exit
    #[arg(long, default_value = "false")]
    auto_unmount: bool,

    /// HTTP server port for health and metrics endpoints
    #[arg(long, default_value = "9090", env = "HTTP_PORT")]
    http_port: u16,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let args = Args::parse();

    info!(
        "Starting Basilica Storage Daemon for experiment: {}",
        args.experiment_id
    );
    info!("Mount point: {}", args.mount_point.display());
    info!("Storage backend: {}", args.backend);
    info!("Bucket: {}", args.bucket);
    info!("Sync interval: {}ms", args.sync_interval_ms);
    info!("Cache size: {}MB", args.cache_size_mb);

    // Create storage backend
    let config = match args.backend.as_str() {
        "r2" => {
            // Extract account_id from endpoint URL
            // Expected format: https://<account_id>.r2.cloudflarestorage.com
            let account_id = args
                .endpoint
                .as_ref()
                .and_then(|e| {
                    // Remove https:// or http:// prefix
                    let url_without_scheme = e
                        .strip_prefix("https://")
                        .or_else(|| e.strip_prefix("http://"))
                        .unwrap_or(e);
                    // Split by '.' and take first part (account_id)
                    url_without_scheme.split('.').next()
                })
                .context("R2 requires account_id from endpoint")?;

            let access_key = args.access_key_id.context("R2 requires access_key_id")?;
            let secret_key = args
                .secret_access_key
                .context("R2 requires secret_access_key")?;

            StorageConfig::r2(account_id, &access_key, &secret_key, &args.bucket)
        }
        "s3" => {
            let region = args.region.as_deref().unwrap_or("us-east-1");
            let access_key = args.access_key_id.context("S3 requires access_key_id")?;
            let secret_key = args
                .secret_access_key
                .context("S3 requires secret_access_key")?;

            StorageConfig::s3(region, &access_key, &secret_key, &args.bucket)
        }
        _ => {
            anyhow::bail!("Unsupported storage backend: {}", args.backend);
        }
    };

    let storage: Arc<dyn StorageBackend> = Arc::new(
        S3Backend::from_config(&config)
            .await
            .context("Failed to create storage backend")?,
    );

    info!("Storage backend initialized successfully");

    // Create metrics
    let metrics = Arc::new(StorageMetrics::new());

    // Convert quota from GB to bytes (with overflow protection)
    let quota_bytes = args.quota_gb.saturating_mul(1024 * 1024 * 1024);

    // Create FUSE filesystem
    let fs = BasilicaFS::new(
        args.namespace.clone(),
        args.experiment_id.clone(),
        storage,
        args.sync_interval_ms,
        args.cache_size_mb,
        quota_bytes,
        metrics.clone(),
    );

    // Start background sync worker
    info!("Starting background sync worker...");
    fs.start_sync_worker().await;

    // Wrap filesystem for sharing
    let shared_fs = SharedBasilicaFS::new(fs);

    // Keep reference for shutdown
    let shared_fs_for_shutdown = shared_fs.clone();

    // Start HTTP server for health and metrics endpoints
    let http_addr = format!("0.0.0.0:{}", args.http_port);
    info!("Starting HTTP server at: {}", http_addr);

    let http_server = HttpServer::new(metrics, shared_fs.arc());
    let http_addr_clone = http_addr.clone();
    tokio::spawn(async move {
        if let Err(e) = http_server.serve(&http_addr_clone).await {
            eprintln!("HTTP server error: {}", e);
        }
    });

    // Prepare mount options
    let mut mount_options = vec![
        MountOption::FSName("basilica-storage".to_string()),
        MountOption::RW,
    ];

    if args.allow_other {
        mount_options.push(MountOption::AllowOther);
    }

    if args.auto_unmount {
        mount_options.push(MountOption::AutoUnmount);
    }

    // Ensure mount point exists
    if !args.mount_point.exists() {
        std::fs::create_dir_all(&args.mount_point).context("Failed to create mount point")?;
    }

    info!("Mounting filesystem at: {}", args.mount_point.display());

    // Mount the filesystem
    // Note: This is a blocking call that will run until unmounted
    let session = fuser::spawn_mount2(shared_fs, &args.mount_point, &mount_options)
        .context("Failed to mount filesystem")?;

    info!("Filesystem mounted successfully");

    // Create readiness marker file to signal main container
    let ready_file = args.mount_point.join(".fuse_ready");
    if let Err(e) = std::fs::write(&ready_file, "ready") {
        eprintln!("Warning: Failed to create readiness marker: {}", e);
    } else {
        info!("Created readiness marker at: {}", ready_file.display());
    }

    info!("Press Ctrl+C to unmount and exit");

    // Wait for shutdown signal (SIGINT or SIGTERM)
    let shutdown = async {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{signal, SignalKind};
            let mut sigterm =
                signal(SignalKind::terminate()).expect("Failed to register SIGTERM handler");

            tokio::select! {
                _ = tokio::signal::ctrl_c() => {
                    info!("Received SIGINT");
                }
                _ = sigterm.recv() => {
                    info!("Received SIGTERM");
                }
            }
        }

        #[cfg(not(unix))]
        {
            tokio::signal::ctrl_c()
                .await
                .expect("Failed to listen for Ctrl+C");
            info!("Received SIGINT");
        }
    };

    shutdown.await;
    info!("Received shutdown signal, flushing dirty pages...");

    // Flush dirty pages before unmounting
    if let Err(e) = shared_fs_for_shutdown.shutdown().await {
        eprintln!("Warning: Failed to flush dirty pages: {}", e);
    } else {
        info!("All dirty pages flushed successfully");
    }

    info!("Unmounting filesystem...");

    // Unmount
    drop(session);

    info!("Filesystem unmounted successfully");
    info!("Basilica Storage Daemon stopped");

    Ok(())
}
