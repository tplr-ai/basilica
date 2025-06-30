//! Unit tests for persistence layer

use chrono::Utc;
use common::config::DatabaseConfig;
use common::identity::Hotkey;
use miner::persistence::{RegistrationDb, ValidatorSession};
use std::time::Duration;

#[tokio::test]
async fn test_registration_db_new() {
    let db = create_test_db().await;

    // Should be able to create DB and run migrations
    assert!(db.health_check().await.is_ok());
}

#[tokio::test]
async fn test_executor_registration() {
    let db = create_test_db().await;

    // Register executor
    let result = db
        .register_executor(
            "exec1",
            "127.0.0.1:50051",
            serde_json::json!({"gpu": "RTX 4090"}),
        )
        .await;

    assert!(result.is_ok());

    // Get executor
    let executor = db.get_executor("exec1").await.unwrap();
    assert!(executor.is_some());

    let executor = executor.unwrap();
    assert_eq!(executor.id, "exec1");
    assert_eq!(executor.grpc_address, "127.0.0.1:50051");
    assert!(executor.is_active);
}

#[tokio::test]
async fn test_executor_health_update() {
    let db = create_test_db().await;

    // Register executor
    db.register_executor("exec1", "127.0.0.1:50051", serde_json::json!({}))
        .await
        .unwrap();

    // Update health status
    let result = db.update_executor_health("exec1", false).await;
    assert!(result.is_ok());

    // Verify update
    let executor = db.get_executor("exec1").await.unwrap().unwrap();
    assert!(!executor.is_healthy);
}

#[tokio::test]
async fn test_list_active_executors() {
    let db = create_test_db().await;

    // Register multiple executors
    db.register_executor("exec1", "127.0.0.1:50051", serde_json::json!({}))
        .await
        .unwrap();
    db.register_executor("exec2", "127.0.0.1:50052", serde_json::json!({}))
        .await
        .unwrap();
    db.register_executor("exec3", "127.0.0.1:50053", serde_json::json!({}))
        .await
        .unwrap();

    // Deactivate one
    db.update_executor_active("exec2", false).await.unwrap();

    // List active executors
    let active = db.list_active_executors().await.unwrap();
    assert_eq!(active.len(), 2);
    assert!(active.iter().any(|e| e.id == "exec1"));
    assert!(active.iter().any(|e| e.id == "exec3"));
    assert!(!active.iter().any(|e| e.id == "exec2"));
}

#[tokio::test]
async fn test_validator_session_creation() {
    let db = create_test_db().await;

    let validator_hotkey = Hotkey("5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string());
    let session_token = "test_token_123";

    // Create session
    let result = db
        .create_validator_session(&validator_hotkey, session_token, 3600)
        .await;

    assert!(result.is_ok());

    // Get session
    let session = db.get_validator_session(session_token).await.unwrap();
    assert!(session.is_some());

    let session = session.unwrap();
    assert_eq!(session.validator_hotkey, validator_hotkey.to_string());
    assert_eq!(session.session_token, session_token);
    assert!(session.is_active);
}

#[tokio::test]
async fn test_validator_session_expiry() {
    let db = create_test_db().await;

    let validator_hotkey = Hotkey("5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string());
    let session_token = "expired_token";

    // Create session with 0 duration (already expired)
    db.create_validator_session(&validator_hotkey, session_token, 0)
        .await
        .unwrap();

    // Session should exist but be expired
    let session = db.get_validator_session(session_token).await.unwrap();
    assert!(session.is_some());

    // Clean expired sessions
    let cleaned = db.cleanup_expired_sessions().await.unwrap();
    assert_eq!(cleaned, 1);

    // Session should now be inactive
    let session = db.get_validator_session(session_token).await.unwrap();
    assert!(session.is_none() || !session.unwrap().is_active);
}

#[tokio::test]
async fn test_record_ssh_access() {
    let db = create_test_db().await;

    let validator_hotkey = Hotkey("5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string());

    // Record SSH access
    let result = db
        .record_ssh_access(
            &validator_hotkey,
            &["exec1".to_string(), "exec2".to_string()],
            "ssh-rsa AAAAB3NzaC1yc2E...",
        )
        .await;

    assert!(result.is_ok());
}

#[tokio::test]
async fn test_get_ssh_access_by_validator() {
    let db = create_test_db().await;

    let validator_hotkey = Hotkey("5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string());

    // Record SSH access
    db.record_ssh_access(
        &validator_hotkey,
        &["exec1".to_string()],
        "ssh-rsa AAAAB3NzaC1yc2E...",
    )
    .await
    .unwrap();

    // Get SSH access records
    let records = db
        .get_ssh_access_by_validator(&validator_hotkey)
        .await
        .unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].executor_id, "exec1");
    assert!(records[0].is_active);
}

#[tokio::test]
async fn test_health_check() {
    let db = create_test_db().await;

    let result = db.health_check().await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_cleanup_operations() {
    let db = create_test_db().await;

    // Create expired session
    let validator_hotkey = Hotkey("5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string());
    db.create_validator_session(&validator_hotkey, "expired", 0)
        .await
        .unwrap();

    // Create active session
    db.create_validator_session(&validator_hotkey, "active", 3600)
        .await
        .unwrap();

    // Cleanup
    let cleaned = db.cleanup_expired_sessions().await.unwrap();
    assert_eq!(cleaned, 1);

    // Verify active session still exists
    let session = db.get_validator_session("active").await.unwrap();
    assert!(session.is_some());
    assert!(session.unwrap().is_active);
}

// Helper functions

async fn create_test_db() -> RegistrationDb {
    let db_config = DatabaseConfig {
        url: "sqlite::memory:".to_string(),
        max_connections: 5,
        min_connections: 1,
        connection_timeout: Duration::from_secs(10),
        idle_timeout: Duration::from_secs(300),
        max_lifetime: Duration::from_secs(3600),
    };

    RegistrationDb::new(&db_config).await.unwrap()
}
