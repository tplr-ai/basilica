use anyhow::Result;
use chrono::{DateTime, Utc};
use common::errors::DatabaseError;
use sqlx::{migrate::MigrateDatabase, SqlitePool};
use std::path::PathBuf;
use tempfile::TempDir;
use uuid::Uuid;
use validator::persistence::{
    ChallengeResult, ChallengeResultRepository, EnvironmentValidation,
    EnvironmentValidationRepository, SimplePersistence, VerificationLog,
    VerificationLogRepository, VerificationService,
};

#[tokio::test]
async fn test_database_initialization_and_migrations() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("test_validator.db");
    let db_url = format!("sqlite:{}", db_path.display());
    
    // Ensure database doesn't exist
    assert!(!sqlx::Sqlite::database_exists(&db_url).await?);
    
    // Create and connect
    let pool = SqlitePool::connect(&db_url).await?;
    
    // Run migrations
    sqlx::migrate!("../../migrations").run(&pool).await?;
    
    // Verify tables exist
    let tables = sqlx::query!("SELECT name FROM sqlite_master WHERE type='table'")
        .fetch_all(&pool)
        .await?;
    
    let table_names: Vec<String> = tables.iter().map(|r| r.name.clone()).collect();
    
    assert!(table_names.contains(&"verification_logs".to_string()));
    assert!(table_names.contains(&"challenge_results".to_string()));
    assert!(table_names.contains(&"environment_validations".to_string()));
    assert!(table_names.contains(&"_sqlx_migrations".to_string()));
    
    Ok(())
}

#[tokio::test]
async fn test_verification_log_crud_operations() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("test_validator.db");
    let db_url = format!("sqlite:{}", db_path.display());
    
    let pool = SqlitePool::connect(&db_url).await?;
    sqlx::migrate!("../../migrations").run(&pool).await?;
    
    let persistence = SimplePersistence::new(pool.clone());
    let repo = persistence.verification_logs();
    
    // Create
    let log = VerificationLog {
        id: Uuid::new_v4().to_string(),
        executor_id: "test-executor-1".to_string(),
        miner_uid: 42,
        validator_hotkey: "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string(),
        verification_type: "hardware_attestation".to_string(),
        status: "success".to_string(),
        score: Some(0.95),
        error_message: None,
        attestation_data: Some(serde_json::json!({
            "cpu_count": 16,
            "memory_gb": 64,
            "gpu_count": 2
        })),
        started_at: Utc::now(),
        completed_at: Some(Utc::now()),
        created_at: Utc::now(),
    };
    
    repo.create(&log).await?;
    
    // Read
    let retrieved = repo.find_by_id(&log.id).await?;
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().executor_id, "test-executor-1");
    
    // Update
    let mut updated_log = log.clone();
    updated_log.score = Some(0.98);
    repo.update(&updated_log).await?;
    
    let updated = repo.find_by_id(&log.id).await?.unwrap();
    assert_eq!(updated.score, Some(0.98));
    
    // List
    let all_logs = repo.list(100, 0).await?;
    assert_eq!(all_logs.len(), 1);
    
    // List by executor
    let executor_logs = repo.list_by_executor("test-executor-1", 10).await?;
    assert_eq!(executor_logs.len(), 1);
    
    // List by miner
    let miner_logs = repo.list_by_miner(42, 10).await?;
    assert_eq!(miner_logs.len(), 1);
    
    // Delete
    repo.delete(&log.id).await?;
    let deleted = repo.find_by_id(&log.id).await?;
    assert!(deleted.is_none());
    
    Ok(())
}

#[tokio::test]
async fn test_challenge_result_operations() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("test_validator.db");
    let db_url = format!("sqlite:{}", db_path.display());
    
    let pool = SqlitePool::connect(&db_url).await?;
    sqlx::migrate!("../../migrations").run(&pool).await?;
    
    let persistence = SimplePersistence::new(pool.clone());
    let repo = persistence.challenge_results();
    
    // Create multiple challenge results
    let challenges = vec![
        ChallengeResult {
            id: Uuid::new_v4().to_string(),
            verification_id: "verification-1".to_string(),
            challenge_type: "compute_task".to_string(),
            input_data: serde_json::json!({"task": "matrix_multiply", "size": 1000}),
            expected_output: Some(serde_json::json!({"checksum": "abc123"})),
            actual_output: Some(serde_json::json!({"checksum": "abc123"})),
            passed: true,
            execution_time_ms: 1250,
            created_at: Utc::now(),
        },
        ChallengeResult {
            id: Uuid::new_v4().to_string(),
            verification_id: "verification-1".to_string(),
            challenge_type: "memory_bandwidth".to_string(),
            input_data: serde_json::json!({"size_gb": 10}),
            expected_output: None,
            actual_output: Some(serde_json::json!({"bandwidth_gbps": 450.5})),
            passed: true,
            execution_time_ms: 5000,
            created_at: Utc::now(),
        },
        ChallengeResult {
            id: Uuid::new_v4().to_string(),
            verification_id: "verification-2".to_string(),
            challenge_type: "gpu_compute".to_string(),
            input_data: serde_json::json!({"kernel": "reduction"}),
            expected_output: Some(serde_json::json!({"result": 42})),
            actual_output: Some(serde_json::json!({"result": 41})),
            passed: false,
            execution_time_ms: 3000,
            created_at: Utc::now(),
        },
    ];
    
    for challenge in &challenges {
        repo.create(challenge).await?;
    }
    
    // List by verification
    let verification_1_results = repo.list_by_verification("verification-1").await?;
    assert_eq!(verification_1_results.len(), 2);
    
    // Get passed challenges
    let passed = repo.get_passed_challenges("verification-1").await?;
    assert_eq!(passed.len(), 2);
    
    // Get average execution time
    let avg_time = repo.get_average_execution_time("verification-1").await?;
    assert!(avg_time.is_some());
    assert_eq!(avg_time.unwrap(), 3125.0); // (1250 + 5000) / 2
    
    Ok(())
}

#[tokio::test]
async fn test_environment_validation_persistence() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("test_validator.db");
    let db_url = format!("sqlite:{}", db_path.display());
    
    let pool = SqlitePool::connect(&db_url).await?;
    sqlx::migrate!("../../migrations").run(&pool).await?;
    
    let persistence = SimplePersistence::new(pool.clone());
    let repo = persistence.environment_validations();
    
    let env_validation = EnvironmentValidation {
        id: Uuid::new_v4().to_string(),
        verification_id: "verification-env-1".to_string(),
        os_name: "Ubuntu".to_string(),
        os_version: "22.04 LTS".to_string(),
        kernel_version: "5.15.0-88-generic".to_string(),
        docker_installed: true,
        docker_version: Some("24.0.7".to_string()),
        nvidia_driver_version: Some("535.129.03".to_string()),
        cuda_version: Some("12.2".to_string()),
        python_version: Some("3.10.12".to_string()),
        disk_space_gb: 450,
        network_bandwidth_mbps: Some(1000.0),
        security_patches_updated: true,
        firewall_configured: true,
        created_at: Utc::now(),
    };
    
    repo.create(&env_validation).await?;
    
    // Find by verification
    let found = repo.find_by_verification("verification-env-1").await?;
    assert!(found.is_some());
    assert_eq!(found.unwrap().os_name, "Ubuntu");
    
    // Get latest by executor
    let latest = repo.get_latest_by_executor("test-executor").await?;
    assert!(latest.is_none()); // No direct executor link in this table
    
    // Check Docker environments
    let docker_envs = repo.get_environments_with_docker().await?;
    assert_eq!(docker_envs.len(), 1);
    assert!(docker_envs[0].docker_installed);
    
    Ok(())
}

#[tokio::test]
async fn test_verification_service_statistics() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("test_validator.db");
    let db_url = format!("sqlite:{}", db_path.display());
    
    let pool = SqlitePool::connect(&db_url).await?;
    sqlx::migrate!("../../migrations").run(&pool).await?;
    
    let service = VerificationService::new(pool.clone());
    
    // Create test data
    let now = Utc::now();
    let logs = vec![
        VerificationLog {
            id: Uuid::new_v4().to_string(),
            executor_id: "executor-1".to_string(),
            miner_uid: 10,
            validator_hotkey: "validator-1".to_string(),
            verification_type: "hardware_attestation".to_string(),
            status: "success".to_string(),
            score: Some(0.9),
            error_message: None,
            attestation_data: None,
            started_at: now - chrono::Duration::hours(2),
            completed_at: Some(now - chrono::Duration::hours(1)),
            created_at: now - chrono::Duration::hours(2),
        },
        VerificationLog {
            id: Uuid::new_v4().to_string(),
            executor_id: "executor-2".to_string(),
            miner_uid: 10,
            validator_hotkey: "validator-1".to_string(),
            verification_type: "hardware_attestation".to_string(),
            status: "failed".to_string(),
            score: None,
            error_message: Some("Connection timeout".to_string()),
            attestation_data: None,
            started_at: now - chrono::Duration::minutes(30),
            completed_at: Some(now - chrono::Duration::minutes(25)),
            created_at: now - chrono::Duration::minutes(30),
        },
        VerificationLog {
            id: Uuid::new_v4().to_string(),
            executor_id: "executor-1".to_string(),
            miner_uid: 20,
            validator_hotkey: "validator-1".to_string(),
            verification_type: "hardware_attestation".to_string(),
            status: "success".to_string(),
            score: Some(0.95),
            error_message: None,
            attestation_data: None,
            started_at: now - chrono::Duration::minutes(10),
            completed_at: Some(now - chrono::Duration::minutes(5)),
            created_at: now - chrono::Duration::minutes(10),
        },
    ];
    
    for log in &logs {
        service.log_verification(log.clone()).await?;
    }
    
    // Get success rate
    let success_rate = service.get_success_rate(chrono::Duration::hours(24)).await?;
    assert_eq!(success_rate, 0.67); // 2 success out of 3
    
    // Get average score
    let avg_score = service.get_average_score_by_miner(10).await?;
    assert_eq!(avg_score, Some(0.9)); // Only one successful verification for miner 10
    
    // Get recent verifications
    let recent = service.get_recent_verifications(10).await?;
    assert_eq!(recent.len(), 3);
    
    // Get executor statistics
    let executor_stats = service.get_executor_statistics("executor-1").await?;
    assert_eq!(executor_stats.total_verifications, 2);
    assert_eq!(executor_stats.successful_verifications, 2);
    assert_eq!(executor_stats.average_score, Some(0.925)); // (0.9 + 0.95) / 2
    
    Ok(())
}

#[tokio::test]
async fn test_database_cleanup_operations() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("test_validator.db");
    let db_url = format!("sqlite:{}", db_path.display());
    
    let pool = SqlitePool::connect(&db_url).await?;
    sqlx::migrate!("../../migrations").run(&pool).await?;
    
    let persistence = SimplePersistence::new(pool.clone());
    let log_repo = persistence.verification_logs();
    
    // Create old and new logs
    let old_date = Utc::now() - chrono::Duration::days(40);
    let recent_date = Utc::now() - chrono::Duration::days(5);
    
    let old_log = VerificationLog {
        id: Uuid::new_v4().to_string(),
        executor_id: "old-executor".to_string(),
        miner_uid: 1,
        validator_hotkey: "validator".to_string(),
        verification_type: "hardware_attestation".to_string(),
        status: "success".to_string(),
        score: Some(0.8),
        error_message: None,
        attestation_data: None,
        started_at: old_date,
        completed_at: Some(old_date),
        created_at: old_date,
    };
    
    let recent_log = VerificationLog {
        id: Uuid::new_v4().to_string(),
        executor_id: "recent-executor".to_string(),
        miner_uid: 2,
        validator_hotkey: "validator".to_string(),
        verification_type: "hardware_attestation".to_string(),
        status: "success".to_string(),
        score: Some(0.9),
        error_message: None,
        attestation_data: None,
        started_at: recent_date,
        completed_at: Some(recent_date),
        created_at: recent_date,
    };
    
    log_repo.create(&old_log).await?;
    log_repo.create(&recent_log).await?;
    
    // Cleanup old records (older than 30 days)
    let deleted = log_repo.cleanup_old_records(30).await?;
    assert_eq!(deleted, 1, "Should delete 1 old record");
    
    // Verify only recent log remains
    let remaining = log_repo.list(100, 0).await?;
    assert_eq!(remaining.len(), 1);
    assert_eq!(remaining[0].executor_id, "recent-executor");
    
    Ok(())
}

#[tokio::test]
async fn test_concurrent_database_operations() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("test_validator.db");
    let db_url = format!("sqlite:{}", db_path.display());
    
    let pool = SqlitePool::connect(&db_url).await?;
    sqlx::migrate!("../../migrations").run(&pool).await?;
    
    // Launch concurrent writes
    let mut handles = vec![];
    
    for i in 0..10 {
        let pool_clone = pool.clone();
        let handle = tokio::spawn(async move {
            let persistence = SimplePersistence::new(pool_clone);
            let repo = persistence.verification_logs();
            
            let log = VerificationLog {
                id: Uuid::new_v4().to_string(),
                executor_id: format!("executor-{}", i),
                miner_uid: i as u16,
                validator_hotkey: "validator".to_string(),
                verification_type: "hardware_attestation".to_string(),
                status: "success".to_string(),
                score: Some(0.8 + (i as f64 * 0.01)),
                error_message: None,
                attestation_data: None,
                started_at: Utc::now(),
                completed_at: Some(Utc::now()),
                created_at: Utc::now(),
            };
            
            repo.create(&log).await
        });
        
        handles.push(handle);
    }
    
    // Wait for all operations to complete
    let results = futures::future::try_join_all(handles).await?;
    
    // All should succeed
    for result in results {
        assert!(result.is_ok(), "Concurrent write should succeed");
    }
    
    // Verify all records were created
    let persistence = SimplePersistence::new(pool);
    let repo = persistence.verification_logs();
    let all_logs = repo.list(100, 0).await?;
    
    assert_eq!(all_logs.len(), 10, "Should have all 10 records");
    
    Ok(())
}