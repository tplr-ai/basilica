use anyhow::Result;
use common::metrics::{MetricsRecorder, SystemMetricsProvider};
use executor::config::{ExecutorConfig, MetricsConfig, MonitoringConfig};
use executor::monitoring::{
    AlertLevel, AlertType, ExecutorMonitor, HealthChecker, MetricThreshold, ResourceMonitor,
    ResourceUsage, SystemMonitor,
};
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;
use tokio::sync::RwLock;
use tokio::time::{sleep, timeout};

#[tokio::test]
async fn test_system_metrics_collection() -> Result<()> {
    let config = MonitoringConfig {
        enable_metrics: true,
        metrics_interval: Duration::from_millis(100),
        resource_check_interval: Duration::from_millis(100),
        alert_thresholds: MetricThreshold {
            cpu_percent: 80.0,
            memory_percent: 90.0,
            disk_percent: 95.0,
            gpu_memory_percent: 85.0,
            temperature_celsius: 85.0,
        },
    };

    let monitor = SystemMonitor::new(config)?;

    // Start monitoring
    monitor.start().await?;

    // Wait for some metrics to be collected
    sleep(Duration::from_millis(500)).await;

    // Get current metrics
    let metrics = monitor.get_current_metrics().await?;

    // Verify basic metrics are present
    assert!(metrics.cpu_usage >= 0.0 && metrics.cpu_usage <= 100.0);
    assert!(metrics.memory_used_mb > 0);
    assert!(metrics.memory_total_mb > metrics.memory_used_mb);
    assert!(metrics.disk_free_gb >= 0.0);
    assert!(!metrics.load_average.is_empty());

    // Get historical metrics
    let history = monitor.get_metrics_history(Duration::from_secs(1)).await?;
    assert!(!history.is_empty());

    // Stop monitoring
    monitor.stop().await?;

    Ok(())
}

#[tokio::test]
async fn test_gpu_monitoring() -> Result<()> {
    let config = MonitoringConfig {
        enable_metrics: true,
        enable_gpu_monitoring: true,
        gpu_check_interval: Duration::from_millis(200),
        ..Default::default()
    };

    let monitor = SystemMonitor::new(config)?;

    // Check if GPU is available
    if !monitor.has_gpu_support().await {
        println!("No GPU detected, skipping GPU monitoring test");
        return Ok(());
    }

    monitor.start().await?;

    // Wait for GPU metrics
    sleep(Duration::from_millis(500)).await;

    let gpu_metrics = monitor.get_gpu_metrics().await?;

    if !gpu_metrics.is_empty() {
        for gpu in gpu_metrics {
            assert!(gpu.index >= 0);
            assert!(!gpu.name.is_empty());
            assert!(gpu.utilization >= 0.0 && gpu.utilization <= 100.0);
            assert!(gpu.memory_used_mb >= 0);
            assert!(gpu.memory_total_mb > 0);
            assert!(gpu.temperature_c >= 0.0);
        }
    }

    monitor.stop().await?;

    Ok(())
}

#[tokio::test]
async fn test_resource_alerts() -> Result<()> {
    let config = MonitoringConfig {
        enable_metrics: true,
        enable_alerts: true,
        alert_thresholds: MetricThreshold {
            cpu_percent: 0.1, // Very low threshold to trigger alert
            memory_percent: 0.1,
            disk_percent: 99.9,
            gpu_memory_percent: 85.0,
            temperature_celsius: 200.0,
        },
        ..Default::default()
    };

    let alerts = Arc::new(RwLock::new(Vec::new()));
    let alerts_clone = alerts.clone();

    let monitor = SystemMonitor::new_with_alert_handler(
        config,
        Box::new(move |alert| {
            let alerts = alerts_clone.clone();
            Box::pin(async move {
                alerts.write().await.push(alert);
            })
        }),
    )?;

    monitor.start().await?;

    // Wait for alerts to be triggered
    sleep(Duration::from_secs(1)).await;

    let collected_alerts = alerts.read().await;
    assert!(
        !collected_alerts.is_empty(),
        "Should have triggered some alerts"
    );

    // Check alert structure
    for alert in collected_alerts.iter() {
        assert!(!alert.message.is_empty());
        assert!(matches!(
            alert.level,
            AlertLevel::Warning | AlertLevel::Critical
        ));
        assert!(matches!(
            alert.alert_type,
            AlertType::HighCpuUsage
                | AlertType::HighMemoryUsage
                | AlertType::LowDiskSpace
                | AlertType::HighGpuMemory
                | AlertType::HighTemperature
        ));
    }

    monitor.stop().await?;

    Ok(())
}

#[tokio::test]
async fn test_health_checker() -> Result<()> {
    let temp_dir = TempDir::new()?;

    let config = ExecutorConfig {
        working_dir: temp_dir.path().to_path_buf(),
        ..Default::default()
    };

    let health_checker = HealthChecker::new(config)?;

    // Perform health check
    let health_status = health_checker.check_health().await?;

    assert!(health_status.is_healthy);
    assert!(!health_status.services.is_empty());

    // Check individual services
    for (service, status) in &health_status.services {
        println!(
            "Service {}: {}",
            service,
            if *status { "healthy" } else { "unhealthy" }
        );
    }

    // Test specific checks
    assert!(health_checker.check_disk_space().await?);
    assert!(health_checker.check_memory_available().await?);
    assert!(health_checker.check_network_connectivity().await?);

    // Check container runtime
    let docker_available = health_checker.check_docker_runtime().await?;
    let podman_available = health_checker.check_podman_runtime().await?;

    assert!(
        docker_available || podman_available,
        "At least one container runtime should be available"
    );

    Ok(())
}

#[tokio::test]
async fn test_resource_usage_tracking() -> Result<()> {
    let monitor = ResourceMonitor::new()?;

    // Track initial usage
    let initial = monitor.get_current_usage().await?;

    // Simulate some work
    let mut data = vec![0u8; 10 * 1024 * 1024]; // 10MB
    for i in 0..data.len() {
        data[i] = (i % 256) as u8;
    }

    // Get usage after allocation
    let after_alloc = monitor.get_current_usage().await?;

    // Memory should have increased
    assert!(after_alloc.memory_mb > initial.memory_mb);

    // Drop the data
    drop(data);

    // Force garbage collection if possible
    sleep(Duration::from_millis(100)).await;

    // Track usage history
    let history = monitor.get_usage_history(Duration::from_secs(1)).await?;
    assert!(!history.is_empty());

    // Calculate average usage
    let avg_usage = monitor.get_average_usage(Duration::from_secs(1)).await?;
    assert!(avg_usage.cpu_percent >= 0.0);
    assert!(avg_usage.memory_mb > 0);

    Ok(())
}

#[tokio::test]
async fn test_executor_specific_monitoring() -> Result<()> {
    let temp_dir = TempDir::new()?;

    let config = ExecutorConfig {
        working_dir: temp_dir.path().to_path_buf(),
        max_concurrent_sessions: 10,
        ..Default::default()
    };

    let executor_monitor = ExecutorMonitor::new(config)?;

    // Start monitoring
    executor_monitor.start().await?;

    // Simulate session activity
    executor_monitor.record_session_start("session-1").await?;
    executor_monitor.record_session_start("session-2").await?;

    sleep(Duration::from_millis(100)).await;

    executor_monitor
        .record_session_complete("session-1", true)
        .await?;

    // Get executor metrics
    let metrics = executor_monitor.get_executor_metrics().await?;

    assert_eq!(metrics.active_sessions, 1);
    assert_eq!(metrics.total_sessions, 2);
    assert_eq!(metrics.successful_sessions, 1);
    assert_eq!(metrics.failed_sessions, 0);
    assert!(metrics.average_session_duration_ms > 0);

    // Record container metrics
    executor_monitor
        .record_container_start("container-1")
        .await?;
    executor_monitor
        .record_container_resource_usage(
            "container-1",
            ResourceUsage {
                cpu_percent: 25.0,
                memory_mb: 512,
                disk_io_mb: 100,
                network_io_mb: 50,
            },
        )
        .await?;

    let container_metrics = executor_monitor.get_container_metrics().await?;
    assert_eq!(container_metrics.active_containers, 1);
    assert!(container_metrics.total_cpu_percent > 0.0);
    assert!(container_metrics.total_memory_mb > 0);

    executor_monitor.stop().await?;

    Ok(())
}

#[tokio::test]
async fn test_prometheus_metrics_export() -> Result<()> {
    let metrics_config = MetricsConfig {
        enable_prometheus: true,
        prometheus_port: 0, // Use random port
        metrics_path: "/metrics".to_string(),
        ..Default::default()
    };

    let monitor = SystemMonitor::new_with_metrics(MonitoringConfig::default(), metrics_config)?;

    monitor.start().await?;

    // Wait for metrics server to start
    sleep(Duration::from_millis(500)).await;

    // Get metrics endpoint
    let metrics_url = monitor.get_metrics_endpoint()?;

    // Verify metrics are being exported
    let response = reqwest::get(&metrics_url).await?;
    assert_eq!(response.status(), 200);

    let metrics_text = response.text().await?;

    // Check for standard metrics
    assert!(metrics_text.contains("executor_cpu_usage"));
    assert!(metrics_text.contains("executor_memory_usage_bytes"));
    assert!(metrics_text.contains("executor_active_sessions"));
    assert!(metrics_text.contains("executor_container_count"));

    monitor.stop().await?;

    Ok(())
}

#[tokio::test]
async fn test_monitoring_persistence() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let metrics_file = temp_dir.path().join("metrics.db");

    let config = MonitoringConfig {
        enable_metrics: true,
        persist_metrics: true,
        metrics_db_path: Some(metrics_file.clone()),
        metrics_retention_days: 7,
        ..Default::default()
    };

    // First run - collect metrics
    {
        let monitor = SystemMonitor::new(config.clone())?;
        monitor.start().await?;

        sleep(Duration::from_secs(1)).await;

        monitor.stop().await?;
    }

    // Verify metrics were persisted
    assert!(metrics_file.exists());

    // Second run - load historical metrics
    {
        let monitor = SystemMonitor::new(config)?;

        let historical = monitor.get_metrics_history(Duration::from_days(1)).await?;
        assert!(
            !historical.is_empty(),
            "Should have loaded historical metrics"
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_custom_metric_collection() -> Result<()> {
    let monitor = SystemMonitor::new(MonitoringConfig::default())?;

    // Register custom metrics
    monitor
        .register_custom_metric(
            "basilica_executor_custom_metric",
            "Custom metric for testing",
            vec!["label1", "label2"],
        )
        .await?;

    // Record custom metric values
    for i in 0..5 {
        monitor
            .record_custom_metric(
                "basilica_executor_custom_metric",
                i as f64,
                vec![("label1", "value1"), ("label2", &format!("value{}", i))],
            )
            .await?;

        sleep(Duration::from_millis(100)).await;
    }

    // Get custom metric values
    let values = monitor
        .get_custom_metric_values("basilica_executor_custom_metric", Duration::from_secs(1))
        .await?;

    assert_eq!(values.len(), 5);

    Ok(())
}

#[tokio::test]
async fn test_monitoring_graceful_shutdown() -> Result<()> {
    let config = MonitoringConfig {
        enable_metrics: true,
        metrics_interval: Duration::from_millis(50),
        ..Default::default()
    };

    let monitor = Arc::new(SystemMonitor::new(config)?);
    let monitor_clone = monitor.clone();

    // Start monitoring in background task
    let handle = tokio::spawn(async move {
        monitor_clone.start().await.unwrap();
        monitor_clone.run_forever().await
    });

    // Let it run for a bit
    sleep(Duration::from_millis(500)).await;

    // Trigger graceful shutdown
    monitor.shutdown_gracefully(Duration::from_secs(5)).await?;

    // Wait for background task to complete
    let result = timeout(Duration::from_secs(10), handle).await;
    assert!(result.is_ok(), "Monitor should shut down gracefully");

    Ok(())
}
