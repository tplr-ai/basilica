use anyhow::Result;
use miner::config::{
    DatabaseConfig, ExecutorManagementConfig, MinerConfig, RemoteExecutorDeploymentConfig,
    RemoteMachine, SshConfig,
};
use miner::executor_manager::ExecutorFleetManager;
use miner::persistence::registration_db::RegistrationDb;
use sqlx::SqlitePool;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;
use tokio::sync::RwLock;
use tokio::time::timeout;

#[tokio::test]
async fn test_static_executor_configuration() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("test_miner.db");
    let db_url = format!("sqlite:{}", db_path.display());

    let pool = SqlitePool::connect(&db_url).await?;
    sqlx::migrate!("../../migrations").run(&pool).await?;

    let mut config = MinerConfig::default();
    config.database.url = db_url.clone();
    config.executor_management.executors = vec![
        miner::config::StaticExecutor {
            id: "test-executor-1".to_string(),
            grpc_address: "127.0.0.1:50051".to_string(),
            name: Some("Test Executor 1".to_string()),
        },
        miner::config::StaticExecutor {
            id: "test-executor-2".to_string(),
            grpc_address: "127.0.0.1:50052".to_string(),
            name: Some("Test Executor 2".to_string()),
        },
    ];

    let db = Arc::new(RwLock::new(RegistrationDb::new(pool.clone())));
    let manager = ExecutorFleetManager::new(config.clone(), db.clone());

    // Initialize fleet from static config
    manager.initialize_from_config().await?;

    // Verify executors were registered
    let executors = manager.list_executors().await?;
    assert_eq!(executors.len(), 2);
    assert!(executors.iter().any(|e| e.id == "test-executor-1"));
    assert!(executors.iter().any(|e| e.id == "test-executor-2"));

    // Check initial health status
    for executor in &executors {
        assert!(!executor.is_healthy);
        assert_eq!(executor.health_check_failures, 0);
    }

    Ok(())
}

#[tokio::test]
async fn test_executor_health_monitoring() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("test_miner.db");
    let db_url = format!("sqlite:{}", db_path.display());

    let pool = SqlitePool::connect(&db_url).await?;
    sqlx::migrate!("../../migrations").run(&pool).await?;

    let mut config = MinerConfig::default();
    config.database.url = db_url.clone();
    config.executor_management.health_check_interval = Duration::from_millis(100);
    config.executor_management.health_check_timeout = Duration::from_millis(50);
    config.executor_management.max_retry_attempts = 3;

    let db = Arc::new(RwLock::new(RegistrationDb::new(pool.clone())));
    let manager = ExecutorFleetManager::new(config.clone(), db.clone());

    // Register a test executor
    manager
        .register_executor(
            "health-test-executor",
            "127.0.0.1:60000", // Non-existent port
        )
        .await?;

    // Run health check - should fail due to connection error
    let health_status = manager.check_executor_health().await?;
    assert_eq!(health_status.len(), 1);
    assert!(!health_status[0].is_healthy);

    // Verify failure count increases
    let executors = manager.list_executors().await?;
    assert_eq!(executors[0].health_check_failures, 1);

    Ok(())
}

#[tokio::test]
async fn test_remote_deployment_configuration() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("test_miner.db");
    let db_url = format!("sqlite:{}", db_path.display());

    let pool = SqlitePool::connect(&db_url).await?;
    sqlx::migrate!("../../migrations").run(&pool).await?;

    let mut config = MinerConfig::default();
    config.database.url = db_url.clone();
    config.remote_executor_deployment = Some(RemoteExecutorDeploymentConfig {
        remote_machines: vec![
            RemoteMachine {
                id: "remote-gpu-1".to_string(),
                name: "Remote GPU Server 1".to_string(),
                gpu_count: 4,
                executor_port: 50051,
                ssh: SshConfig {
                    host: "gpu1.example.com".to_string(),
                    port: 22,
                    username: "ubuntu".to_string(),
                    private_key_path: PathBuf::from("/home/user/.ssh/id_rsa"),
                },
            },
            RemoteMachine {
                id: "remote-gpu-2".to_string(),
                name: "Remote GPU Server 2".to_string(),
                gpu_count: 2,
                executor_port: 50051,
                ssh: SshConfig {
                    host: "gpu2.example.com".to_string(),
                    port: 22,
                    username: "admin".to_string(),
                    private_key_path: PathBuf::from("/home/user/.ssh/id_rsa"),
                },
            },
        ],
        local_executor_binary: PathBuf::from("./target/release/executor"),
        executor_config_template: None,
        health_check_interval: Some(Duration::from_secs(60)),
        auto_deploy: false,
        auto_start: false,
    });

    let db = Arc::new(RwLock::new(RegistrationDb::new(pool.clone())));
    let manager = ExecutorFleetManager::new(config.clone(), db.clone());

    // Test deployment info generation
    let deployment_info = manager.get_deployment_info().await?;
    assert_eq!(deployment_info.len(), 2);
    assert!(deployment_info.iter().any(|d| d.id == "remote-gpu-1"));
    assert!(deployment_info.iter().any(|d| d.id == "remote-gpu-2"));

    // Verify SSH configuration
    for info in &deployment_info {
        assert!(!info.ssh_config.private_key_path.as_os_str().is_empty());
        assert!(!info.ssh_config.host.is_empty());
        assert!(info.ssh_config.port > 0);
    }

    Ok(())
}

#[tokio::test]
async fn test_executor_auto_recovery() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("test_miner.db");
    let db_url = format!("sqlite:{}", db_path.display());

    let pool = SqlitePool::connect(&db_url).await?;
    sqlx::migrate!("../../migrations").run(&pool).await?;

    let mut config = MinerConfig::default();
    config.database.url = db_url.clone();
    config.executor_management.auto_recovery = true;
    config.executor_management.max_retry_attempts = 3;
    config.executor_management.health_check_interval = Duration::from_millis(100);

    let db = Arc::new(RwLock::new(RegistrationDb::new(pool.clone())));
    let manager = ExecutorFleetManager::new(config.clone(), db.clone());

    // Register executor with invalid address for recovery testing
    manager
        .register_executor("recovery-test", "127.0.0.1:60001")
        .await?;

    // Simulate multiple health check failures
    for _ in 0..3 {
        let _ = manager.check_executor_health().await?;
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    // Check that executor is marked for recovery
    let executors = manager.list_executors().await?;
    assert_eq!(executors[0].health_check_failures, 3);

    // Test recovery attempt (should fail but not panic)
    let recovery_result = manager.attempt_executor_recovery(&executors[0].id).await;
    assert!(recovery_result.is_err() || !recovery_result?);

    Ok(())
}

#[tokio::test]
async fn test_systemd_service_generation() -> Result<()> {
    let service_content = miner::executor_manager::generate_systemd_service(
        "/opt/basilica/bin/executor",
        "/opt/basilica/config/executor.toml",
        "basilica",
        50051,
    );

    // Verify service content
    assert!(service_content.contains("[Unit]"));
    assert!(service_content.contains("Description=Basilica Executor Service"));
    assert!(service_content.contains("[Service]"));
    assert!(service_content.contains("Type=simple"));
    assert!(service_content.contains("User=basilica"));
    assert!(service_content.contains("ExecStart=/opt/basilica/bin/executor"));
    assert!(service_content.contains("--config /opt/basilica/config/executor.toml"));
    assert!(service_content.contains("Restart=always"));
    assert!(service_content.contains("[Install]"));
    assert!(service_content.contains("WantedBy=multi-user.target"));

    // Verify security settings
    assert!(service_content.contains("NoNewPrivileges=yes"));
    assert!(service_content.contains("PrivateTmp=yes"));
    assert!(service_content.contains("ProtectSystem=strict"));

    Ok(())
}

#[tokio::test]
async fn test_concurrent_executor_management() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("test_miner.db");
    let db_url = format!("sqlite:{}", db_path.display());

    let pool = SqlitePool::connect(&db_url).await?;
    sqlx::migrate!("../../migrations").run(&pool).await?;

    let mut config = MinerConfig::default();
    config.database.url = db_url.clone();

    let db = Arc::new(RwLock::new(RegistrationDb::new(pool.clone())));
    let manager = Arc::new(ExecutorFleetManager::new(config.clone(), db.clone()));

    // Test concurrent executor registration
    let mut handles = vec![];
    for i in 0..5 {
        let manager_clone = manager.clone();
        let handle = tokio::spawn(async move {
            manager_clone
                .register_executor(
                    &format!("concurrent-executor-{}", i),
                    &format!("127.0.0.1:5005{}", i),
                )
                .await
        });
        handles.push(handle);
    }

    // Wait for all registrations
    for handle in handles {
        handle.await??;
    }

    // Verify all executors were registered
    let executors = manager.list_executors().await?;
    assert_eq!(executors.len(), 5);

    // Test concurrent health checks
    let health_result = timeout(Duration::from_secs(5), manager.check_executor_health()).await;
    assert!(health_result.is_ok());

    Ok(())
}
