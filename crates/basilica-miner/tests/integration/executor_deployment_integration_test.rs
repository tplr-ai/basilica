use anyhow::Result;
use basilica_miner::config::MinerConfig;
use basilica_miner::node_manager::NodeFleetManager;
use basilica_miner::persistence::registration_db::RegistrationDb;
use sqlx::SqlitePool;
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;
use tokio::sync::RwLock;
use tokio::time::timeout;

#[tokio::test]
async fn test_static_node_configuration() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("test_miner.db");
    let db_url = format!("sqlite:{}", db_path.display());

    let pool = SqlitePool::connect(&db_url).await?;
    sqlx::migrate!("../../migrations").run(&pool).await?;

    let mut config = MinerConfig::default();
    config.database.url = db_url.clone();
    config.node_management.nodes = vec![
        basilica_miner::config::StaticNode {
            id: "test-node-1".to_string(),
            grpc_address: "127.0.0.1:50051".to_string(),
            name: Some("Test Node 1".to_string()),
        },
        basilica_miner::config::StaticNode {
            id: "test-node-2".to_string(),
            grpc_address: "127.0.0.1:50052".to_string(),
            name: Some("Test Node 2".to_string()),
        },
    ];

    let db = Arc::new(RwLock::new(RegistrationDb::new(pool.clone())));
    let manager = NodeFleetManager::new(config.clone(), db.clone());

    // Initialize fleet from static config
    manager.initialize_from_config().await?;

    // Verify nodes were registered
    let nodes = manager.list_nodes().await?;
    assert_eq!(nodes.len(), 2);
    assert!(nodes.iter().any(|e| e.id == "test-node-1"));
    assert!(nodes.iter().any(|e| e.id == "test-node-2"));

    // Check initial health status
    for node in &nodes {
        assert!(!node.is_healthy);
        assert_eq!(node.health_check_failures, 0);
    }

    Ok(())
}

#[tokio::test]
async fn test_node_health_monitoring() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("test_miner.db");
    let db_url = format!("sqlite:{}", db_path.display());

    let pool = SqlitePool::connect(&db_url).await?;
    sqlx::migrate!("../../migrations").run(&pool).await?;

    let mut config = MinerConfig::default();
    config.database.url = db_url.clone();
    config.node_management.health_check_interval = Duration::from_millis(100);
    config.node_management.health_check_timeout = Duration::from_millis(50);
    config.node_management.max_retry_attempts = 3;

    let db = Arc::new(RwLock::new(RegistrationDb::new(pool.clone())));
    let manager = NodeFleetManager::new(config.clone(), db.clone());

    // Register a test node
    manager
        .register_node(
            "health-test-node",
            "127.0.0.1:60000", // Non-existent port
        )
        .await?;

    // Run health check - should fail due to connection error
    let health_status = manager.check_node_health().await?;
    assert_eq!(health_status.len(), 1);
    assert!(!health_status[0].is_healthy);

    // Verify failure count increases
    let nodes = manager.list_nodes().await?;
    assert_eq!(nodes[0].health_check_failures, 1);

    Ok(())
}

#[tokio::test]
async fn test_node_auto_recovery() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("test_miner.db");
    let db_url = format!("sqlite:{}", db_path.display());

    let pool = SqlitePool::connect(&db_url).await?;
    sqlx::migrate!("../../migrations").run(&pool).await?;

    let mut config = MinerConfig::default();
    config.database.url = db_url.clone();
    config.node_management.auto_recovery = true;
    config.node_management.max_retry_attempts = 3;
    config.node_management.health_check_interval = Duration::from_millis(100);

    let db = Arc::new(RwLock::new(RegistrationDb::new(pool.clone())));
    let manager = NodeFleetManager::new(config.clone(), db.clone());

    // Register node with invalid address for recovery testing
    manager
        .register_node("recovery-test", "127.0.0.1:60001")
        .await?;

    // Simulate multiple health check failures
    for _ in 0..3 {
        let _ = manager.check_node_health().await?;
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    // Check that node is marked for recovery
    let nodes = manager.list_nodes().await?;
    assert_eq!(nodes[0].health_check_failures, 3);

    // Test recovery attempt (should fail but not panic)
    let recovery_result = manager.attempt_node_recovery(&nodes[0].id).await;
    assert!(recovery_result.is_err() || !recovery_result?);

    Ok(())
}

#[tokio::test]
async fn test_systemd_service_generation() -> Result<()> {
    let service_content = basilica_miner::node_manager::generate_systemd_service(
        "/opt/basilica/bin/node",
        "/opt/basilica/config/node.toml",
        "basilica",
        50051,
    );

    // Verify service content
    assert!(service_content.contains("[Unit]"));
    assert!(service_content.contains("Description=Basilica Node Service"));
    assert!(service_content.contains("[Service]"));
    assert!(service_content.contains("Type=simple"));
    assert!(service_content.contains("User=basilica"));
    assert!(service_content.contains("ExecStart=/opt/basilica/bin/node"));
    assert!(service_content.contains("--config /opt/basilica/config/node.toml"));
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
async fn test_concurrent_node_management() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("test_miner.db");
    let db_url = format!("sqlite:{}", db_path.display());

    let pool = SqlitePool::connect(&db_url).await?;
    sqlx::migrate!("../../migrations").run(&pool).await?;

    let mut config = MinerConfig::default();
    config.database.url = db_url.clone();

    let db = Arc::new(RwLock::new(RegistrationDb::new(pool.clone())));
    let manager = Arc::new(NodeFleetManager::new(config.clone(), db.clone()));

    // Test concurrent node registration
    let mut handles = vec![];
    for i in 0..5 {
        let manager_clone = manager.clone();
        let handle = tokio::spawn(async move {
            manager_clone
                .register_node(
                    &format!("concurrent-node-{}", i),
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

    // Verify all nodes were registered
    let nodes = manager.list_nodes().await?;
    assert_eq!(nodes.len(), 5);

    // Test concurrent health checks
    let health_result = timeout(Duration::from_secs(5), manager.check_node_health()).await;
    assert!(health_result.is_ok());

    Ok(())
}
