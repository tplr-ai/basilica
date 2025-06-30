use anyhow::Result;
use common::identity::{ExecutorId, Hotkey};
use executor::config::{ContainerConfig, ExecutorConfig, GrpcConfig, SecurityConfig};
use executor::persistence::{ExecutorPersistence, ValidationSession, ValidationSessionRepository};
use executor::session::{SessionManager, SessionState, SessionValidator};
use sqlx::SqlitePool;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::TempDir;
use tokio::sync::RwLock;
use tokio::time::{sleep, timeout, Duration};
use uuid::Uuid;

#[tokio::test]
async fn test_validation_session_lifecycle() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("executor_sessions.db");
    let db_url = format!("sqlite:{}", db_path.display());

    let pool = SqlitePool::connect(&db_url).await?;
    sqlx::migrate!("../../migrations").run(&pool).await?;

    let config = ExecutorConfig {
        executor_id: ExecutorId::new("test-executor-sessions"),
        database_url: db_url.clone(),
        working_dir: temp_dir.path().to_path_buf(),
        ..Default::default()
    };

    let persistence = Arc::new(ExecutorPersistence::new(pool.clone()));
    let session_manager = SessionManager::new(config.clone(), persistence.clone());

    // Create new session
    let validator_hotkey = Hotkey::new("5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY")?;
    let session_id = session_manager
        .create_session(
            validator_hotkey.clone(),
            Duration::from_secs(3600),
            vec!["hardware_attestation".to_string()],
        )
        .await?;

    // Verify session created
    let session = session_manager.get_session(&session_id).await?;
    assert!(session.is_some());
    let session = session.unwrap();
    assert_eq!(session.state, SessionState::Active);
    assert_eq!(session.validator_hotkey, validator_hotkey.to_string());

    // Validate session
    let validator = SessionValidator::new(Arc::new(RwLock::new(vec![validator_hotkey.clone()])));
    let is_valid = validator
        .validate_session(&session_id, &validator_hotkey)
        .await?;
    assert!(is_valid);

    // Update session progress
    session_manager
        .update_session_progress(&session_id, "Starting hardware attestation")
        .await?;

    // Complete session
    session_manager
        .complete_session(
            &session_id,
            serde_json::json!({
                "attestation": "successful",
                "score": 0.95
            }),
        )
        .await?;

    // Verify session completed
    let completed = session_manager.get_session(&session_id).await?.unwrap();
    assert_eq!(completed.state, SessionState::Completed);
    assert!(completed.result.is_some());

    // Test session expiration
    let expired_session_id = session_manager
        .create_session(
            validator_hotkey.clone(),
            Duration::from_millis(100), // Very short expiration
            vec!["test".to_string()],
        )
        .await?;

    sleep(Duration::from_millis(200)).await;

    let expired = session_manager.get_session(&expired_session_id).await?;
    assert!(expired.is_none() || expired.unwrap().state == SessionState::Expired);

    Ok(())
}

#[tokio::test]
async fn test_concurrent_session_management() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("executor_concurrent.db");
    let db_url = format!("sqlite:{}", db_path.display());

    let pool = SqlitePool::connect(&db_url).await?;
    sqlx::migrate!("../../migrations").run(&pool).await?;

    let config = ExecutorConfig {
        executor_id: ExecutorId::new("test-concurrent-executor"),
        database_url: db_url.clone(),
        working_dir: temp_dir.path().to_path_buf(),
        ..Default::default()
    };

    let persistence = Arc::new(ExecutorPersistence::new(pool.clone()));
    let session_manager = Arc::new(SessionManager::new(config, persistence));

    // Create multiple concurrent sessions
    let mut handles = vec![];

    for i in 0..10 {
        let manager = session_manager.clone();
        let handle = tokio::spawn(async move {
            let validator_hotkey = Hotkey::new(&format!("validator-{}", i))?;
            let session_id = manager
                .create_session(
                    validator_hotkey,
                    Duration::from_secs(600),
                    vec!["test".to_string()],
                )
                .await?;

            // Simulate work
            sleep(Duration::from_millis(50)).await;

            manager
                .update_session_progress(&session_id, &format!("Processing task {}", i))
                .await?;

            // Simulate more work
            sleep(Duration::from_millis(50)).await;

            manager
                .complete_session(
                    &session_id,
                    serde_json::json!({ "task": i, "status": "success" }),
                )
                .await?;

            Ok::<String, anyhow::Error>(session_id)
        });

        handles.push(handle);
    }

    // Wait for all sessions to complete
    let session_ids: Vec<String> = futures::future::try_join_all(handles)
        .await?
        .into_iter()
        .collect::<Result<Vec<_>>>()?;

    // Verify all sessions completed successfully
    for session_id in session_ids {
        let session = session_manager.get_session(&session_id).await?.unwrap();
        assert_eq!(session.state, SessionState::Completed);
        assert!(session.result.is_some());
    }

    // Check active sessions count
    let active_count = session_manager.get_active_session_count().await?;
    assert_eq!(active_count, 0);

    Ok(())
}

#[tokio::test]
async fn test_session_cleanup_and_recovery() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("executor_cleanup.db");
    let db_url = format!("sqlite:{}", db_path.display());

    let pool = SqlitePool::connect(&db_url).await?;
    sqlx::migrate!("../../migrations").run(&pool).await?;

    // Create stale sessions directly in database
    let stale_time = chrono::Utc::now() - chrono::Duration::hours(2);

    sqlx::query!(
        r#"
        INSERT INTO validation_sessions (id, validator_hotkey, state, created_at, expires_at)
        VALUES (?1, ?2, ?3, ?4, ?5)
        "#,
        Uuid::new_v4().to_string(),
        "stale-validator-1",
        "active",
        stale_time,
        stale_time + chrono::Duration::hours(1)
    )
    .execute(&pool)
    .await?;

    sqlx::query!(
        r#"
        INSERT INTO validation_sessions (id, validator_hotkey, state, created_at, expires_at)
        VALUES (?1, ?2, ?3, ?4, ?5)
        "#,
        Uuid::new_v4().to_string(),
        "stale-validator-2",
        "in_progress",
        stale_time,
        stale_time + chrono::Duration::hours(1)
    )
    .execute(&pool)
    .await?;

    let config = ExecutorConfig {
        executor_id: ExecutorId::new("test-cleanup-executor"),
        database_url: db_url.clone(),
        working_dir: temp_dir.path().to_path_buf(),
        session_cleanup_interval: Duration::from_millis(100),
        ..Default::default()
    };

    let persistence = Arc::new(ExecutorPersistence::new(pool.clone()));
    let session_manager = SessionManager::new(config, persistence.clone());

    // Start cleanup task
    session_manager.start_cleanup_task();

    // Wait for cleanup to run
    sleep(Duration::from_millis(200)).await;

    // Verify stale sessions were cleaned up
    let repo = persistence.validation_sessions();
    let all_sessions = repo.list(100, 0).await?;

    for session in all_sessions {
        assert_ne!(session.state, SessionState::Active.to_string());
        assert_ne!(session.state, SessionState::InProgress.to_string());
    }

    Ok(())
}

#[tokio::test]
async fn test_session_resource_limits() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("executor_limits.db");
    let db_url = format!("sqlite:{}", db_path.display());

    let pool = SqlitePool::connect(&db_url).await?;
    sqlx::migrate!("../../migrations").run(&pool).await?;

    let config = ExecutorConfig {
        executor_id: ExecutorId::new("test-limits-executor"),
        database_url: db_url.clone(),
        working_dir: temp_dir.path().to_path_buf(),
        max_concurrent_sessions: 3,
        ..Default::default()
    };

    let persistence = Arc::new(ExecutorPersistence::new(pool.clone()));
    let session_manager = SessionManager::new(config, persistence);

    // Create sessions up to limit
    let mut session_ids = vec![];
    for i in 0..3 {
        let validator_hotkey = Hotkey::new(&format!("validator-{}", i))?;
        let session_id = session_manager
            .create_session(
                validator_hotkey,
                Duration::from_secs(300),
                vec!["test".to_string()],
            )
            .await?;
        session_ids.push(session_id);
    }

    // Try to create one more - should fail
    let validator_hotkey = Hotkey::new("validator-overflow")?;
    let result = session_manager
        .create_session(
            validator_hotkey,
            Duration::from_secs(300),
            vec!["test".to_string()],
        )
        .await;

    assert!(result.is_err(), "Should fail when exceeding session limit");

    // Complete one session
    session_manager
        .complete_session(
            &session_ids[0],
            serde_json::json!({ "status": "completed" }),
        )
        .await?;

    // Now should be able to create new session
    let validator_hotkey = Hotkey::new("validator-new")?;
    let new_session = session_manager
        .create_session(
            validator_hotkey,
            Duration::from_secs(300),
            vec!["test".to_string()],
        )
        .await;

    assert!(new_session.is_ok(), "Should succeed after freeing a slot");

    Ok(())
}

#[tokio::test]
async fn test_session_persistence_across_restarts() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("executor_restart.db");
    let db_url = format!("sqlite:{}", db_path.display());

    let pool = SqlitePool::connect(&db_url).await?;
    sqlx::migrate!("../../migrations").run(&pool).await?;

    let session_id: String;
    let validator_hotkey = Hotkey::new("persistent-validator")?;

    // First instance - create session
    {
        let config = ExecutorConfig {
            executor_id: ExecutorId::new("test-restart-executor"),
            database_url: db_url.clone(),
            working_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let persistence = Arc::new(ExecutorPersistence::new(pool.clone()));
        let session_manager = SessionManager::new(config, persistence);

        session_id = session_manager
            .create_session(
                validator_hotkey.clone(),
                Duration::from_secs(3600),
                vec!["hardware_attestation".to_string()],
            )
            .await?;

        session_manager
            .update_session_progress(&session_id, "Work in progress")
            .await?;
    }

    // Simulate restart - create new instance
    {
        let new_pool = SqlitePool::connect(&db_url).await?;
        let config = ExecutorConfig {
            executor_id: ExecutorId::new("test-restart-executor"),
            database_url: db_url.clone(),
            working_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let persistence = Arc::new(ExecutorPersistence::new(new_pool));
        let session_manager = SessionManager::new(config, persistence);

        // Should be able to recover session
        let recovered = session_manager.get_session(&session_id).await?;
        assert!(recovered.is_some());

        let session = recovered.unwrap();
        assert_eq!(session.validator_hotkey, validator_hotkey.to_string());
        assert_eq!(session.state, SessionState::InProgress);

        // Should be able to complete recovered session
        session_manager
            .complete_session(
                &session_id,
                serde_json::json!({ "status": "recovered and completed" }),
            )
            .await?;
    }

    Ok(())
}

#[tokio::test]
async fn test_session_authorization_and_security() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("executor_auth.db");
    let db_url = format!("sqlite:{}", db_path.display());

    let pool = SqlitePool::connect(&db_url).await?;
    sqlx::migrate!("../../migrations").run(&pool).await?;

    let config = ExecutorConfig {
        executor_id: ExecutorId::new("test-auth-executor"),
        database_url: db_url.clone(),
        working_dir: temp_dir.path().to_path_buf(),
        ..Default::default()
    };

    let persistence = Arc::new(ExecutorPersistence::new(pool.clone()));
    let session_manager = SessionManager::new(config, persistence);

    // Set up authorized validators
    let authorized_validator = Hotkey::new("authorized-validator")?;
    let unauthorized_validator = Hotkey::new("unauthorized-validator")?;

    let validator =
        SessionValidator::new(Arc::new(RwLock::new(vec![authorized_validator.clone()])));

    // Create session with authorized validator
    let valid_session = session_manager
        .create_session(
            authorized_validator.clone(),
            Duration::from_secs(600),
            vec!["test".to_string()],
        )
        .await?;

    // Validation should succeed
    assert!(
        validator
            .validate_session(&valid_session, &authorized_validator)
            .await?
    );

    // Validation with unauthorized validator should fail
    assert!(
        !validator
            .validate_session(&valid_session, &unauthorized_validator)
            .await?
    );

    // Test session hijacking prevention
    let result = session_manager
        .update_session_progress_with_auth(
            &valid_session,
            "Malicious update",
            &unauthorized_validator,
        )
        .await;

    assert!(
        result.is_err(),
        "Should prevent unauthorized session updates"
    );

    Ok(())
}
