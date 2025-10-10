use anyhow::Result;
use basilica_miner::config::{
    AuthConfig, BittensorConfig, NodeConfig, NodeManagementConfig, MinerConfig,
    RateLimitConfig, SecurityConfig, ValidatorCommsConfig,
};
use basilica_miner::{MinerService, ServiceManager};
use basilica_protocol::basilica::basilica_miner::v1::{
    miner_service_client::MinerServiceClient, AuthenticateRequest, GrantSshAccessRequest,
    ListNodesRequest,
};
use sqlx::SqlitePool;
use std::net::SocketAddr;
use std::time::Duration;
use tempfile::TempDir;
use tokio::net::TcpListener;
use tokio::time::{sleep, timeout};
use tonic::transport::Channel;

#[tokio::test]
async fn test_complete_validator_miner_node_workflow() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("test_miner.db");
    let db_url = format!("sqlite:{}", db_path.display());

    // Create database pool
    let pool = SqlitePool::connect(&db_url).await?;
    sqlx::migrate!("../../migrations").run(&pool).await?;

    // Configure miner with test nodes
    let nodes = vec![
        NodeConfig {
            id: "test-node-1".to_string(),
            name: "Test Node 1".to_string(),
            grpc_address: "127.0.0.1:60001".to_string(),
        },
        NodeConfig {
            id: "test-node-2".to_string(),
            name: "Test Node 2".to_string(),
            grpc_address: "127.0.0.1:60002".to_string(),
        },
    ];

    // Find available port for miner
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let miner_addr = listener.local_addr()?;
    drop(listener);

    let config = MinerConfig {
        bittensor: BittensorConfig {
            wallet_name: "test_wallet".to_string(),
            hotkey_name: "test_hotkey".to_string(),
            network: "test".to_string(),
            netuid: 999,
            chain_endpoint: "wss://test-endpoint.invalid:443".to_string(),
            weight_interval_secs: 300,
            uid: Some(42),
            coldkey_name: "test_coldkey".to_string(),
            axon_port: 8091,
            external_ip: Some("127.0.0.1".to_string()),
            max_weight_uids: 256,
        },
        server: basilica_miner::config::ServerConfig {
            host: miner_addr.ip().to_string(),
            port: miner_addr.port(),
            max_connections: 100,
            tls_enabled: false,
            request_timeout: Duration::from_secs(30),
        },
        node_management: NodeManagementConfig {
            nodes,
            health_check_interval: Duration::from_secs(300),
            health_check_timeout: Duration::from_secs(10),
            max_retry_attempts: 3,
            auto_recovery: true,
        },
        validator_comms: ValidatorCommsConfig {
            host: "0.0.0.0".to_string(),
            port: 50051,
            request_timeout: Duration::from_secs(30),
        },
        security: SecurityConfig {
            verify_signatures: false, // Disable for E2E test
            ..Default::default()
        },
        ..Default::default()
    };

    // Create and start miner service
    let service = MinerService::new(config.clone(), pool.clone()).await?;
    let service_manager = ServiceManager::new(service);

    // Start service in background
    let service_handle = tokio::spawn(async move { service_manager.run().await });

    // Wait for service to start
    sleep(Duration::from_secs(2)).await;

    // Create gRPC client
    let channel = Channel::from_shared(format!("http://{}", miner_addr))?
        .connect()
        .await?;

    let mut client = MinerServiceClient::new(channel);

    // Step 1: Authenticate as validator
    let auth_response = client
        .authenticate(AuthenticateRequest {
            validator_hotkey: "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string(),
            signature: "test-signature".to_string(),
            timestamp: chrono::Utc::now().timestamp(),
            nonce: "test-nonce-e2e".to_string(),
        })
        .await?;

    let session_token = auth_response.into_inner().session_token;
    assert!(!session_token.is_empty(), "Should receive session token");

    // Step 2: List available nodes
    let mut list_request = tonic::Request::new(ListNodesRequest {
        include_unhealthy: true,
        limit: 100,
    });
    list_request.metadata_mut().insert(
        "authorization",
        format!("Bearer {}", session_token).parse()?,
    );

    let list_response = client.list_nodes(list_request).await?;
    let nodes = list_response.into_inner().nodes;

    assert_eq!(nodes.len(), 2, "Should have 2 nodes");
    assert!(nodes.iter().any(|e| e.id == "test-node-1"));
    assert!(nodes.iter().any(|e| e.id == "test-node-2"));

    // Step 3: Request SSH access to an node
    let mut ssh_request = tonic::Request::new(GrantSshAccessRequest {
        node_id: "test-node-1".to_string(),
        validator_hotkey: "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string(),
        ssh_public_key: "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITestKeyForE2E test@e2e.com"
            .to_string(),
        duration_seconds: 3600,
        purpose: "e2e_testing".to_string(),
    });
    ssh_request.metadata_mut().insert(
        "authorization",
        format!("Bearer {}", session_token).parse()?,
    );

    let ssh_response = client.grant_ssh_access(ssh_request).await;

    // SSH grant might fail without real node, but structure should work
    match ssh_response {
        Ok(response) => {
            let grant = response.into_inner();
            assert!(!grant.access_token.is_empty(), "Should have access token");
            assert_eq!(grant.node_id, "test-node-1");
        }
        Err(status) => {
            // Expected if node not actually running
            assert!(
                status.code() == tonic::Code::NotFound || status.code() == tonic::Code::Internal,
                "Expected node-related error, got: {:?}",
                status
            );
        }
    }

    // Verify database state
    let session_count = sqlx::query!("SELECT COUNT(*) as count FROM validator_sessions")
        .fetch_one(&pool)
        .await?;

    assert!(
        session_count.count > 0,
        "Should have validator session in database"
    );

    // Cleanup: Stop service
    service_handle.abort();

    Ok(())
}

#[tokio::test]
async fn test_concurrent_validator_sessions() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("test_miner.db");
    let db_url = format!("sqlite:{}", db_path.display());

    let pool = SqlitePool::connect(&db_url).await?;
    sqlx::migrate!("../../migrations").run(&pool).await?;

    // Find available port
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let miner_addr = listener.local_addr()?;
    drop(listener);

    let config = MinerConfig {
        server: basilica_miner::config::ServerConfig {
            host: miner_addr.ip().to_string(),
            port: miner_addr.port(),
            max_connections: 100,
            tls_enabled: false,
            request_timeout: Duration::from_secs(30),
        },
        validator_comms: ValidatorCommsConfig {
            host: "0.0.0.0".to_string(),
            port: miner_addr.port(),
            request_timeout: Duration::from_secs(30),
        },
        security: SecurityConfig {
            verify_signatures: false,
            ..Default::default()
        },
        ..Default::default()
    };

    let service = MinerService::new(config, pool.clone()).await?;
    let service_manager = ServiceManager::new(service);

    let _service_handle = tokio::spawn(async move { service_manager.run().await });

    sleep(Duration::from_secs(1)).await;

    // Create multiple concurrent clients
    let mut handles = vec![];

    for i in 0..10 {
        let addr = miner_addr.clone();
        let handle = tokio::spawn(async move {
            let channel = Channel::from_shared(format!("http://{}", addr))?
                .connect()
                .await?;

            let mut client = MinerServiceClient::new(channel);

            let result = client
                .authenticate(AuthenticateRequest {
                    validator_hotkey: format!("validator-{}", i),
                    signature: "test-sig".to_string(),
                    timestamp: chrono::Utc::now().timestamp(),
                    nonce: format!("nonce-{}", i),
                })
                .await;

            Ok::<_, anyhow::Error>(result.is_ok())
        });

        handles.push(handle);
    }

    // Wait for all requests to complete
    let results: Vec<bool> = futures::future::try_join_all(handles)
        .await?
        .into_iter()
        .collect::<Result<Vec<_>>>()?;

    // At least max_concurrent_sessions should succeed
    let successful = results.iter().filter(|&&r| r).count();
    assert!(
        successful >= 5,
        "At least 5 concurrent sessions should succeed"
    );

    Ok(())
}

#[tokio::test]
async fn test_node_failover_scenario() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("test_miner.db");
    let db_url = format!("sqlite:{}", db_path.display());

    let pool = SqlitePool::connect(&db_url).await?;
    sqlx::migrate!("../../migrations").run(&pool).await?;

    // Configure with multiple nodes
    let nodes = vec![
        NodeConfig {
            id: "primary-node".to_string(),
            name: "Primary Node".to_string(),
            grpc_address: "127.0.0.1:60001".to_string(),
        },
        NodeConfig {
            id: "backup-node".to_string(),
            name: "Backup Node".to_string(),
            grpc_address: "127.0.0.1:60002".to_string(),
        },
    ];

    let config = MinerConfig {
        node_management: NodeManagementConfig {
            nodes,
            health_check_interval: Duration::from_secs(1), // Fast for testing
            health_check_timeout: Duration::from_secs(1),
            max_retry_attempts: 2,
            auto_recovery: true,
        },
        ..Default::default()
    };

    let service = MinerService::new(config, pool.clone()).await?;

    // Mark primary as unhealthy
    sqlx::query!(
        r#"
        INSERT INTO nodes (id, name, grpc_address, is_healthy, last_health_check, failure_count)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6)
        ON CONFLICT(id) DO UPDATE SET
            is_healthy = excluded.is_healthy,
            failure_count = excluded.failure_count
        "#,
        "primary-node",
        "Primary Node",
        "127.0.0.1:60001",
        false,
        chrono::Utc::now(),
        5
    )
    .execute(&pool)
    .await?;

    // Mark backup as healthy
    sqlx::query!(
        r#"
        INSERT INTO nodes (id, name, grpc_address, is_healthy, last_health_check)
        VALUES (?1, ?2, ?3, ?4, ?5)
        ON CONFLICT(id) DO UPDATE SET
            is_healthy = excluded.is_healthy
        "#,
        "backup-node",
        "Backup Node",
        "127.0.0.1:60002",
        true,
        chrono::Utc::now()
    )
    .execute(&pool)
    .await?;

    // Request healthy nodes
    let healthy_nodes = service.get_healthy_nodes().await?;

    assert_eq!(
        healthy_nodes.len(),
        1,
        "Should have one healthy node"
    );
    assert_eq!(
        healthy_nodes[0].id, "backup-node",
        "Backup should be available"
    );

    // Simulate recovery of primary
    service
        .attempt_node_recovery("primary-node")
        .await
        .ok();

    // Check recovery was attempted
    let primary_status = sqlx::query!(
        "SELECT recovery_attempts FROM nodes WHERE id = ?",
        "primary-node"
    )
    .fetch_one(&pool)
    .await?;

    assert!(
        primary_status.recovery_attempts.unwrap_or(0) > 0,
        "Should have attempted recovery"
    );

    Ok(())
}

#[tokio::test]
async fn test_audit_logging() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("test_miner.db");
    let db_url = format!("sqlite:{}", db_path.display());

    let pool = SqlitePool::connect(&db_url).await?;
    sqlx::migrate!("../../migrations").run(&pool).await?;

    let config = MinerConfig::default();
    let service = MinerService::new(config, pool.clone()).await?;

    // Log various audit events
    service
        .log_validator_interaction(
            "test-validator-1",
            "authenticate",
            true,
            Some("Session created".to_string()),
        )
        .await?;

    service
        .log_validator_interaction("test-validator-2", "list_nodes", true, None)
        .await?;

    service
        .log_validator_interaction(
            "test-validator-1",
            "grant_ssh_access",
            false,
            Some("Node not found".to_string()),
        )
        .await?;

    // Verify audit logs
    let logs =
        sqlx::query!("SELECT * FROM validator_interactions ORDER BY timestamp DESC LIMIT 10")
            .fetch_all(&pool)
            .await?;

    assert_eq!(logs.len(), 3, "Should have 3 audit log entries");

    // Check specific log entries
    let auth_log = logs.iter().find(|l| l.action == "authenticate").unwrap();
    assert_eq!(auth_log.validator_hotkey, "test-validator-1");
    assert!(auth_log.success);

    let ssh_log = logs
        .iter()
        .find(|l| l.action == "grant_ssh_access")
        .unwrap();
    assert!(!ssh_log.success);
    assert!(ssh_log.details.as_ref().unwrap().contains("not found"));

    Ok(())
}
