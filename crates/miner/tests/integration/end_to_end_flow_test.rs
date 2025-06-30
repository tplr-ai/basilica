use anyhow::Result;
use miner::config::{
    AuthConfig, BittensorConfig, ExecutorConfig, ExecutorManagementConfig, MinerConfig,
    RateLimitConfig, SecurityConfig, ValidatorCommsConfig,
};
use miner::{MinerService, ServiceManager};
use protocol::basilica::miner::v1::{
    miner_service_client::MinerServiceClient, AuthenticateRequest, GrantSshAccessRequest,
    ListExecutorsRequest,
};
use sqlx::SqlitePool;
use std::net::SocketAddr;
use std::time::Duration;
use tempfile::TempDir;
use tokio::net::TcpListener;
use tokio::time::{sleep, timeout};
use tonic::transport::Channel;

#[tokio::test]
async fn test_complete_validator_miner_executor_workflow() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("test_miner.db");
    let db_url = format!("sqlite:{}", db_path.display());

    // Create database pool
    let pool = SqlitePool::connect(&db_url).await?;
    sqlx::migrate!("../../migrations").run(&pool).await?;

    // Configure miner with test executors
    let executors = vec![
        ExecutorConfig {
            id: "test-executor-1".to_string(),
            name: "Test Executor 1".to_string(),
            grpc_address: "127.0.0.1:60001".to_string(),
        },
        ExecutorConfig {
            id: "test-executor-2".to_string(),
            name: "Test Executor 2".to_string(),
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
        server: miner::config::ServerConfig {
            host: miner_addr.ip().to_string(),
            port: miner_addr.port(),
            max_connections: 100,
            tls_enabled: false,
            request_timeout: Duration::from_secs(30),
        },
        executor_management: ExecutorManagementConfig {
            executors,
            health_check_interval: Duration::from_secs(300),
            health_check_timeout: Duration::from_secs(10),
            max_retry_attempts: 3,
            auto_recovery: true,
        },
        validator_comms: ValidatorCommsConfig {
            max_concurrent_sessions: 10,
            session_timeout: Duration::from_secs(3600),
            auth: AuthConfig {
                enabled: true,
                method: "bittensor_signature".to_string(),
            },
            request_timeout: Duration::from_secs(30),
            rate_limit: RateLimitConfig {
                enabled: true,
                requests_per_second: 100,
                burst_capacity: 200,
            },
        },
        security: SecurityConfig {
            enable_mtls: false,
            jwt_secret: "test-secret-key-for-e2e-testing".to_string(),
            allowed_validators: vec![],
            verify_signatures: false, // Disable for E2E test
            token_expiration: Duration::from_secs(3600),
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

    // Step 2: List available executors
    let mut list_request = tonic::Request::new(ListExecutorsRequest {
        include_unhealthy: true,
        limit: 100,
    });
    list_request.metadata_mut().insert(
        "authorization",
        format!("Bearer {}", session_token).parse()?,
    );

    let list_response = client.list_executors(list_request).await?;
    let executors = list_response.into_inner().executors;

    assert_eq!(executors.len(), 2, "Should have 2 executors");
    assert!(executors.iter().any(|e| e.id == "test-executor-1"));
    assert!(executors.iter().any(|e| e.id == "test-executor-2"));

    // Step 3: Request SSH access to an executor
    let mut ssh_request = tonic::Request::new(GrantSshAccessRequest {
        executor_id: "test-executor-1".to_string(),
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

    // SSH grant might fail without real executor, but structure should work
    match ssh_response {
        Ok(response) => {
            let grant = response.into_inner();
            assert!(!grant.access_token.is_empty(), "Should have access token");
            assert_eq!(grant.executor_id, "test-executor-1");
        }
        Err(status) => {
            // Expected if executor not actually running
            assert!(
                status.code() == tonic::Code::NotFound || status.code() == tonic::Code::Internal,
                "Expected executor-related error, got: {:?}",
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
        server: miner::config::ServerConfig {
            host: miner_addr.ip().to_string(),
            port: miner_addr.port(),
            max_connections: 100,
            tls_enabled: false,
            request_timeout: Duration::from_secs(30),
        },
        validator_comms: ValidatorCommsConfig {
            max_concurrent_sessions: 5, // Low limit for testing
            session_timeout: Duration::from_secs(3600),
            auth: AuthConfig {
                enabled: true,
                method: "bittensor_signature".to_string(),
            },
            request_timeout: Duration::from_secs(30),
            rate_limit: RateLimitConfig {
                enabled: false, // Disable for concurrent test
                requests_per_second: 100,
                burst_capacity: 200,
            },
        },
        security: SecurityConfig {
            enable_mtls: false,
            jwt_secret: "test-secret".to_string(),
            allowed_validators: vec![],
            verify_signatures: false,
            token_expiration: Duration::from_secs(3600),
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
async fn test_executor_failover_scenario() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("test_miner.db");
    let db_url = format!("sqlite:{}", db_path.display());

    let pool = SqlitePool::connect(&db_url).await?;
    sqlx::migrate!("../../migrations").run(&pool).await?;

    // Configure with multiple executors
    let executors = vec![
        ExecutorConfig {
            id: "primary-executor".to_string(),
            name: "Primary Executor".to_string(),
            grpc_address: "127.0.0.1:60001".to_string(),
        },
        ExecutorConfig {
            id: "backup-executor".to_string(),
            name: "Backup Executor".to_string(),
            grpc_address: "127.0.0.1:60002".to_string(),
        },
    ];

    let config = MinerConfig {
        executor_management: ExecutorManagementConfig {
            executors,
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
        INSERT INTO executors (id, name, grpc_address, is_healthy, last_health_check, failure_count)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6)
        ON CONFLICT(id) DO UPDATE SET
            is_healthy = excluded.is_healthy,
            failure_count = excluded.failure_count
        "#,
        "primary-executor",
        "Primary Executor",
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
        INSERT INTO executors (id, name, grpc_address, is_healthy, last_health_check)
        VALUES (?1, ?2, ?3, ?4, ?5)
        ON CONFLICT(id) DO UPDATE SET
            is_healthy = excluded.is_healthy
        "#,
        "backup-executor",
        "Backup Executor",
        "127.0.0.1:60002",
        true,
        chrono::Utc::now()
    )
    .execute(&pool)
    .await?;

    // Request healthy executors
    let healthy_executors = service.get_healthy_executors().await?;

    assert_eq!(
        healthy_executors.len(),
        1,
        "Should have one healthy executor"
    );
    assert_eq!(
        healthy_executors[0].id, "backup-executor",
        "Backup should be available"
    );

    // Simulate recovery of primary
    service
        .attempt_executor_recovery("primary-executor")
        .await
        .ok();

    // Check recovery was attempted
    let primary_status = sqlx::query!(
        "SELECT recovery_attempts FROM executors WHERE id = ?",
        "primary-executor"
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
        .log_validator_interaction("test-validator-2", "list_executors", true, None)
        .await?;

    service
        .log_validator_interaction(
            "test-validator-1",
            "grant_ssh_access",
            false,
            Some("Executor not found".to_string()),
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
