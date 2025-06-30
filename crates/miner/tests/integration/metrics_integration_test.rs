use anyhow::Result;
use miner::config::{MetricsConfig, MinerConfig};
use miner::metrics::{MetricsCollector, MetricsRegistry};
use miner::persistence::registration_db::RegistrationDb;
use prometheus::{Counter, Gauge, Histogram, Registry};
use sqlx::SqlitePool;
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;
use tokio::sync::RwLock;
use tokio::time::sleep;
use warp::Filter;

#[tokio::test]
async fn test_metrics_registry_initialization() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("test_miner.db");
    let db_url = format!("sqlite:{}", db_path.display());

    let pool = SqlitePool::connect(&db_url).await?;
    sqlx::migrate!("../../migrations").run(&pool).await?;

    let metrics_config = MetricsConfig {
        enabled: true,
        port: 9092,
        path: "/metrics".to_string(),
        update_interval: Duration::from_secs(5),
        custom_labels: std::collections::HashMap::from([
            ("service".to_string(), "miner".to_string()),
            ("environment".to_string(), "test".to_string()),
        ]),
    };

    let config = MinerConfig {
        metrics: metrics_config,
        database: miner::config::DatabaseConfig {
            url: db_url,
            ..Default::default()
        },
        ..Default::default()
    };

    let registry = MetricsRegistry::new(&config)?;

    // Verify standard metrics are registered
    let metric_families = registry.gather();
    assert!(!metric_families.is_empty(), "Metrics should be registered");

    // Check for specific metric types
    let metric_names: Vec<_> = metric_families.iter().map(|f| f.get_name()).collect();

    // Should have executor-related metrics
    assert!(
        metric_names.iter().any(|n| n.contains("executor")),
        "Should have executor metrics"
    );

    // Should have chain-related metrics
    assert!(
        metric_names.iter().any(|n| n.contains("chain")),
        "Should have chain metrics"
    );

    // Should have grpc-related metrics
    assert!(
        metric_names.iter().any(|n| n.contains("grpc")),
        "Should have gRPC metrics"
    );

    Ok(())
}

#[tokio::test]
async fn test_executor_fleet_metrics() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("test_miner.db");
    let db_url = format!("sqlite:{}", db_path.display());

    let pool = SqlitePool::connect(&db_url).await?;
    sqlx::migrate!("../../migrations").run(&pool).await?;

    let config = MinerConfig {
        database: miner::config::DatabaseConfig {
            url: db_url.clone(),
            ..Default::default()
        },
        ..Default::default()
    };

    let db = Arc::new(RwLock::new(RegistrationDb::new(pool.clone())));
    let collector = MetricsCollector::new(config.clone(), db.clone())?;

    // Register test executors
    {
        let mut db_write = db.write().await;
        db_write
            .register_executor("test-executor-1", "127.0.0.1:50051", Some("Test 1"))
            .await?;
        db_write
            .register_executor("test-executor-2", "127.0.0.1:50052", Some("Test 2"))
            .await?;
        db_write
            .register_executor("test-executor-3", "127.0.0.1:50053", Some("Test 3"))
            .await?;
    }

    // Update fleet metrics
    collector.update_executor_fleet_metrics().await?;

    // Verify metrics
    let total_executors = collector.get_metric_value("miner_executors_total").await?;
    assert_eq!(total_executors, 3.0, "Should have 3 total executors");

    let healthy_executors = collector
        .get_metric_value("miner_executors_healthy")
        .await?;
    assert_eq!(
        healthy_executors, 0.0,
        "Should have 0 healthy executors initially"
    );

    // Simulate health check updates
    {
        let mut db_write = db.write().await;
        db_write
            .update_executor_health("test-executor-1", true)
            .await?;
        db_write
            .update_executor_health("test-executor-2", true)
            .await?;
    }

    // Update metrics again
    collector.update_executor_fleet_metrics().await?;

    let healthy_executors = collector
        .get_metric_value("miner_executors_healthy")
        .await?;
    assert_eq!(
        healthy_executors, 2.0,
        "Should have 2 healthy executors after update"
    );

    Ok(())
}

#[tokio::test]
async fn test_grpc_request_metrics() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("test_miner.db");
    let db_url = format!("sqlite:{}", db_path.display());

    let pool = SqlitePool::connect(&db_url).await?;
    sqlx::migrate!("../../migrations").run(&pool).await?;

    let config = MinerConfig {
        database: miner::config::DatabaseConfig {
            url: db_url,
            ..Default::default()
        },
        ..Default::default()
    };

    let db = Arc::new(RwLock::new(RegistrationDb::new(pool.clone())));
    let collector = MetricsCollector::new(config, db)?;

    // Simulate gRPC requests
    for i in 0..10 {
        collector
            .record_grpc_request("SubmitTask", Duration::from_millis(100 + i * 10), true)
            .await?;
    }

    for i in 0..5 {
        collector
            .record_grpc_request("GetStatus", Duration::from_millis(50 + i * 5), true)
            .await?;
    }

    // Record some failures
    collector
        .record_grpc_request("SubmitTask", Duration::from_millis(200), false)
        .await?;
    collector
        .record_grpc_request("GetStatus", Duration::from_millis(100), false)
        .await?;

    // Verify request counts
    let submit_success = collector
        .get_counter_value(
            "miner_grpc_requests_total",
            &[("method", "SubmitTask"), ("status", "success")],
        )
        .await?;
    assert_eq!(
        submit_success, 10.0,
        "Should have 10 successful SubmitTask requests"
    );

    let submit_failure = collector
        .get_counter_value(
            "miner_grpc_requests_total",
            &[("method", "SubmitTask"), ("status", "failure")],
        )
        .await?;
    assert_eq!(
        submit_failure, 1.0,
        "Should have 1 failed SubmitTask request"
    );

    let status_success = collector
        .get_counter_value(
            "miner_grpc_requests_total",
            &[("method", "GetStatus"), ("status", "success")],
        )
        .await?;
    assert_eq!(
        status_success, 5.0,
        "Should have 5 successful GetStatus requests"
    );

    // Verify latency metrics recorded
    let latency_count = collector
        .get_histogram_count(
            "miner_grpc_request_duration_seconds",
            &[("method", "SubmitTask")],
        )
        .await?;
    assert_eq!(
        latency_count, 11,
        "Should have 11 total SubmitTask latency recordings"
    );

    Ok(())
}

#[tokio::test]
async fn test_chain_sync_metrics() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("test_miner.db");
    let db_url = format!("sqlite:{}", db_path.display());

    let pool = SqlitePool::connect(&db_url).await?;
    sqlx::migrate!("../../migrations").run(&pool).await?;

    let config = MinerConfig {
        database: miner::config::DatabaseConfig {
            url: db_url,
            ..Default::default()
        },
        bittensor: miner::config::BittensorConfig {
            uid: Some(42),
            ..Default::default()
        },
        ..Default::default()
    };

    let db = Arc::new(RwLock::new(RegistrationDb::new(pool.clone())));
    let collector = MetricsCollector::new(config, db)?;

    // Simulate chain sync events
    collector.record_chain_sync_attempt(true).await?;
    collector.record_chain_sync_attempt(true).await?;
    collector.record_chain_sync_attempt(false).await?;
    collector.record_chain_sync_attempt(true).await?;

    // Update chain state metrics
    collector.update_chain_metrics(1000, 42, true).await?;

    // Verify metrics
    let sync_success = collector
        .get_counter_value("miner_chain_sync_total", &[("status", "success")])
        .await?;
    assert_eq!(sync_success, 3.0, "Should have 3 successful syncs");

    let sync_failure = collector
        .get_counter_value("miner_chain_sync_total", &[("status", "failure")])
        .await?;
    assert_eq!(sync_failure, 1.0, "Should have 1 failed sync");

    let block_height = collector
        .get_gauge_value("miner_chain_block_height")
        .await?;
    assert_eq!(block_height, 1000.0, "Should have block height 1000");

    let miner_uid = collector.get_gauge_value("miner_chain_uid").await?;
    assert_eq!(miner_uid, 42.0, "Should have UID 42");

    let is_registered = collector.get_gauge_value("miner_chain_registered").await?;
    assert_eq!(is_registered, 1.0, "Should be registered (1.0)");

    Ok(())
}

#[tokio::test]
async fn test_database_metrics() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("test_miner.db");
    let db_url = format!("sqlite:{}", db_path.display());

    let pool = SqlitePool::connect(&db_url).await?;
    sqlx::migrate!("../../migrations").run(&pool).await?;

    let config = MinerConfig {
        database: miner::config::DatabaseConfig {
            url: db_url.clone(),
            max_connections: 10,
            ..Default::default()
        },
        ..Default::default()
    };

    let db = Arc::new(RwLock::new(RegistrationDb::new(pool.clone())));
    let collector = MetricsCollector::new(config, db.clone())?;

    // Perform database operations
    for i in 0..20 {
        let mut db_write = db.write().await;
        db_write
            .register_executor(
                &format!("exec-{}", i),
                &format!("127.0.0.1:{}", 50000 + i),
                None,
            )
            .await?;
    }

    // Update database metrics
    collector.update_database_metrics().await?;

    // Verify pool metrics
    let pool_size = collector.get_gauge_value("miner_db_pool_size").await?;
    assert!(pool_size > 0.0, "Pool size should be positive");

    let pool_idle = collector
        .get_gauge_value("miner_db_pool_idle_connections")
        .await?;
    assert!(pool_idle >= 0.0, "Idle connections should be non-negative");

    // Test query timing
    let start = std::time::Instant::now();
    {
        let db_read = db.read().await;
        let _ = db_read.list_executors().await?;
    }
    let duration = start.elapsed();

    collector
        .record_database_query("list_executors", duration, true)
        .await?;

    let query_count = collector
        .get_counter_value(
            "miner_db_queries_total",
            &[("query", "list_executors"), ("status", "success")],
        )
        .await?;
    assert_eq!(
        query_count, 1.0,
        "Should have 1 successful list_executors query"
    );

    Ok(())
}

#[tokio::test]
async fn test_metrics_http_endpoint() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("test_miner.db");
    let db_url = format!("sqlite:{}", db_path.display());

    let pool = SqlitePool::connect(&db_url).await?;
    sqlx::migrate!("../../migrations").run(&pool).await?;

    let metrics_config = MetricsConfig {
        enabled: true,
        port: 0, // Use any available port
        path: "/metrics".to_string(),
        update_interval: Duration::from_secs(5),
        custom_labels: std::collections::HashMap::new(),
    };

    let config = MinerConfig {
        metrics: metrics_config,
        database: miner::config::DatabaseConfig {
            url: db_url,
            ..Default::default()
        },
        ..Default::default()
    };

    let db = Arc::new(RwLock::new(RegistrationDb::new(pool.clone())));
    let collector = Arc::new(MetricsCollector::new(config.clone(), db)?);

    // Create metrics endpoint
    let registry = collector.registry();
    let metrics_route = warp::path("metrics").and(warp::get()).map(move || {
        use prometheus::Encoder;
        let encoder = prometheus::TextEncoder::new();
        let metric_families = registry.gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer).unwrap();
        String::from_utf8(buffer).unwrap()
    });

    // Start server on random port
    let (tx, rx) = tokio::sync::oneshot::channel();
    let server =
        warp::serve(metrics_route).bind_with_graceful_shutdown(([127, 0, 0, 1], 0), async {
            rx.await.ok();
        });

    let (addr, server_future) = server;
    tokio::spawn(server_future);

    // Wait for server to start
    sleep(Duration::from_millis(100)).await;

    // Test metrics endpoint
    let client = reqwest::Client::new();
    let response = client
        .get(format!("http://{}/metrics", addr))
        .send()
        .await?;

    assert_eq!(response.status(), 200, "Metrics endpoint should return 200");

    let body = response.text().await?;
    assert!(
        body.contains("# HELP"),
        "Should contain Prometheus help text"
    );
    assert!(
        body.contains("# TYPE"),
        "Should contain Prometheus type declarations"
    );
    assert!(
        body.contains("miner_"),
        "Should contain miner-specific metrics"
    );

    // Shutdown server
    let _ = tx.send(());

    Ok(())
}

#[tokio::test]
async fn test_custom_metrics_labels() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("test_miner.db");
    let db_url = format!("sqlite:{}", db_path.display());

    let pool = SqlitePool::connect(&db_url).await?;

    let metrics_config = MetricsConfig {
        enabled: true,
        port: 9092,
        path: "/metrics".to_string(),
        update_interval: Duration::from_secs(5),
        custom_labels: std::collections::HashMap::from([
            ("service".to_string(), "miner".to_string()),
            ("environment".to_string(), "production".to_string()),
            ("region".to_string(), "us-east-1".to_string()),
            ("node_id".to_string(), "miner-001".to_string()),
        ]),
    };

    let config = MinerConfig {
        metrics: metrics_config,
        database: miner::config::DatabaseConfig {
            url: db_url,
            ..Default::default()
        },
        ..Default::default()
    };

    let registry = MetricsRegistry::new(&config)?;

    // Create a test metric with custom labels
    let test_counter = Counter::new("test_metric", "Test metric with custom labels")?;
    registry.register(Box::new(test_counter.clone()))?;
    test_counter.inc();

    // Export metrics and verify labels
    use prometheus::Encoder;
    let encoder = prometheus::TextEncoder::new();
    let metric_families = registry.gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer)?;
    let output = String::from_utf8(buffer)?;

    // Verify custom labels are present
    assert!(
        output.contains("service=\"miner\""),
        "Should contain service label"
    );
    assert!(
        output.contains("environment=\"production\""),
        "Should contain environment label"
    );
    assert!(
        output.contains("region=\"us-east-1\""),
        "Should contain region label"
    );
    assert!(
        output.contains("node_id=\"miner-001\""),
        "Should contain node_id label"
    );

    Ok(())
}

#[tokio::test]
async fn test_metrics_aggregation() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("test_miner.db");
    let db_url = format!("sqlite:{}", db_path.display());

    let pool = SqlitePool::connect(&db_url).await?;
    sqlx::migrate!("../../migrations").run(&pool).await?;

    let config = MinerConfig {
        database: miner::config::DatabaseConfig {
            url: db_url,
            ..Default::default()
        },
        ..Default::default()
    };

    let db = Arc::new(RwLock::new(RegistrationDb::new(pool.clone())));
    let collector = MetricsCollector::new(config, db.clone())?;

    // Register multiple executors with different states
    {
        let mut db_write = db.write().await;
        for i in 0..10 {
            db_write
                .register_executor(
                    &format!("healthy-{}", i),
                    &format!("127.0.0.1:{}", 50000 + i),
                    None,
                )
                .await?;
            db_write
                .update_executor_health(&format!("healthy-{}", i), true)
                .await?;
        }

        for i in 0..5 {
            db_write
                .register_executor(
                    &format!("unhealthy-{}", i),
                    &format!("127.0.0.1:{}", 51000 + i),
                    None,
                )
                .await?;
            db_write
                .update_executor_health(&format!("unhealthy-{}", i), false)
                .await?;
        }
    }

    // Record various metrics
    for _ in 0..100 {
        collector
            .record_grpc_request(
                "SubmitTask",
                Duration::from_millis(rand::random::<u64>() % 500),
                true,
            )
            .await?;
    }

    for _ in 0..10 {
        collector
            .record_grpc_request(
                "SubmitTask",
                Duration::from_millis(rand::random::<u64>() % 1000),
                false,
            )
            .await?;
    }

    // Update aggregated metrics
    collector.update_all_metrics().await?;

    // Verify aggregated results
    let total_executors = collector.get_metric_value("miner_executors_total").await?;
    assert_eq!(total_executors, 15.0, "Should have 15 total executors");

    let healthy_ratio = collector
        .get_metric_value("miner_executors_healthy_ratio")
        .await?;
    assert!(
        (healthy_ratio - 0.666).abs() < 0.01,
        "Healthy ratio should be ~66.6%"
    );

    let request_success_rate = collector
        .get_metric_value("miner_grpc_success_rate")
        .await?;
    assert!(
        (request_success_rate - 0.909).abs() < 0.01,
        "Success rate should be ~90.9%"
    );

    // Verify percentiles
    let p50_latency = collector
        .get_histogram_quantile(
            "miner_grpc_request_duration_seconds",
            &[("method", "SubmitTask")],
            0.5,
        )
        .await?;
    let p99_latency = collector
        .get_histogram_quantile(
            "miner_grpc_request_duration_seconds",
            &[("method", "SubmitTask")],
            0.99,
        )
        .await?;

    assert!(p50_latency < p99_latency, "P50 should be less than P99");
    assert!(p99_latency < 1.0, "P99 should be less than 1 second");

    Ok(())
}
