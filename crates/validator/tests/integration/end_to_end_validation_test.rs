use anyhow::Result;
use common::identity::{ExecutorId, Hotkey, MinerUid};
use p256::ecdsa::SigningKey;
use p256::elliptic_curve::rand_core::OsRng;
use sqlx::SqlitePool;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::TempDir;
use tokio::time::{sleep, timeout, Duration};
use uuid::Uuid;
use validator::{
    api::{create_router, ApiContext},
    bittensor_core::WeightSetter,
    config::{BittensorConfig, DatabaseConfig, ValidationConfig, ValidatorConfig, VerificationConfig},
    persistence::{SimplePersistence, VerificationLog, VerificationService},
    ssh::SshClient,
    validation::{
        AttestationData, HardwareInfo, HardwareValidator, ValidationContext,
        ValidationOptions, ValidatorFactory,
    },
};

#[tokio::test]
async fn test_complete_validation_workflow() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("test_e2e.db");
    let db_url = format!("sqlite:{}", db_path.display());
    
    // Initialize database
    let pool = SqlitePool::connect(&db_url).await?;
    sqlx::migrate!("../../migrations").run(&pool).await?;
    
    // Create validator configuration
    let config = ValidatorConfig {
        database: DatabaseConfig {
            url: db_url.clone(),
            max_connections: 10,
            min_connections: 1,
            connect_timeout: Duration::from_secs(30),
            max_lifetime: Some(Duration::from_secs(3600)),
            run_migrations: true,
        },
        validation: ValidationConfig {
            gpu_attestor_path: PathBuf::from("/usr/local/bin/gpu-attestor"),
            remote_work_dir: PathBuf::from("/tmp/basilica_validation"),
            execution_timeout: Duration::from_secs(120),
            cleanup_on_success: true,
            cleanup_on_failure: false,
            max_attestation_size: 10 * 1024 * 1024,
            allowed_algorithms: vec!["simple".to_string(), "advanced".to_string()],
        },
        verification: VerificationConfig {
            max_concurrent_verifications: 5,
            min_score_threshold: 0.1,
            min_stake_threshold: 100.0,
            max_miners_per_round: 20,
            verification_interval: Duration::from_secs(600),
            challenge_timeout: Duration::from_secs(120),
            min_verification_interval: Some(Duration::from_secs(1800)),
            netuid: 42,
        },
        ..Default::default()
    };
    
    // Create services
    let persistence = Arc::new(SimplePersistence::new(pool.clone()));
    let verification_service = Arc::new(VerificationService::new(pool.clone()));
    let factory = ValidatorFactory::new(config.clone());
    
    // Step 1: Simulate executor registration
    let executor_id = ExecutorId::new("test-executor-e2e");
    let miner_uid = MinerUid::new(42);
    let validator_hotkey = Hotkey::new("5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY")?;
    
    // Step 2: Create validation context
    let context = ValidationContext {
        executor_id: executor_id.to_string(),
        miner_uid: miner_uid.as_u16(),
        validator_hotkey: validator_hotkey.to_string(),
        timestamp: chrono::Utc::now(),
    };
    
    // Step 3: Start validation (will fail without real SSH)
    let validator = factory.create_hardware_validator()?;
    let options = ValidationOptions {
        skip_network_benchmark: false,
        skip_hardware_collection: false,
        skip_vdf: false,
        vdf_difficulty: 1000,
        vdf_algorithm: "simple".to_string(),
        custom_args: vec![],
    };
    
    // Create verification log entry
    let verification_id = Uuid::new_v4().to_string();
    let verification_log = VerificationLog {
        id: verification_id.clone(),
        executor_id: executor_id.to_string(),
        miner_uid: miner_uid.as_u16(),
        validator_hotkey: validator_hotkey.to_string(),
        verification_type: "hardware_attestation".to_string(),
        status: "in_progress".to_string(),
        score: None,
        error_message: None,
        attestation_data: None,
        started_at: chrono::Utc::now(),
        completed_at: None,
        created_at: chrono::Utc::now(),
    };
    
    verification_service.log_verification(verification_log.clone()).await?;
    
    // Simulate validation attempt
    let validation_result = timeout(
        Duration::from_secs(5),
        validator.validate_executor(
            "localhost",
            "test_user",
            "/tmp/test_key",
            22,
            context,
            options
        )
    ).await;
    
    // Update verification log based on result
    let mut updated_log = verification_log.clone();
    updated_log.completed_at = Some(chrono::Utc::now());
    
    match validation_result {
        Ok(Ok(result)) => {
            updated_log.status = "success".to_string();
            updated_log.score = Some(calculate_score(&result.attestation_data)?);
            updated_log.attestation_data = Some(serde_json::to_value(&result.attestation_data)?);
        }
        _ => {
            updated_log.status = "failed".to_string();
            updated_log.error_message = Some("SSH connection failed in test environment".to_string());
        }
    }
    
    let repo = persistence.verification_logs();
    repo.update(&updated_log).await?;
    
    // Step 4: Verify database state
    let stored_log = repo.find_by_id(&verification_id).await?;
    assert!(stored_log.is_some());
    assert_eq!(stored_log.unwrap().status, "failed"); // Expected in test
    
    // Step 5: Check statistics
    let success_rate = verification_service.get_success_rate(chrono::Duration::hours(1)).await?;
    assert_eq!(success_rate, 0.0); // Failed validation
    
    let executor_stats = verification_service.get_executor_statistics(&executor_id.to_string()).await?;
    assert_eq!(executor_stats.total_verifications, 1);
    assert_eq!(executor_stats.successful_verifications, 0);
    
    Ok(())
}

#[tokio::test]
async fn test_concurrent_miner_verifications() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("test_concurrent.db");
    let db_url = format!("sqlite:{}", db_path.display());
    
    let pool = SqlitePool::connect(&db_url).await?;
    sqlx::migrate!("../../migrations").run(&pool).await?;
    
    let config = ValidatorConfig {
        database: DatabaseConfig {
            url: db_url.clone(),
            max_connections: 20,
            min_connections: 5,
            ..Default::default()
        },
        verification: VerificationConfig {
            max_concurrent_verifications: 10,
            ..Default::default()
        },
        ..Default::default()
    };
    
    let verification_service = Arc::new(VerificationService::new(pool.clone()));
    
    // Launch concurrent verifications
    let mut handles = vec![];
    
    for i in 0..10 {
        let service = verification_service.clone();
        
        let handle = tokio::spawn(async move {
            let log = VerificationLog {
                id: Uuid::new_v4().to_string(),
                executor_id: format!("executor-{}", i),
                miner_uid: (i % 5) as u16, // 5 different miners
                validator_hotkey: "test-validator".to_string(),
                verification_type: "hardware_attestation".to_string(),
                status: if i % 3 == 0 { "failed" } else { "success" }.to_string(),
                score: if i % 3 != 0 { Some(0.8 + (i as f64 * 0.01)) } else { None },
                error_message: if i % 3 == 0 { Some("Test failure".to_string()) } else { None },
                attestation_data: None,
                started_at: chrono::Utc::now(),
                completed_at: Some(chrono::Utc::now()),
                created_at: chrono::Utc::now(),
            };
            
            service.log_verification(log).await
        });
        
        handles.push(handle);
    }
    
    // Wait for all verifications
    for handle in handles {
        handle.await??;
    }
    
    // Verify results
    let recent = verification_service.get_recent_verifications(20).await?;
    assert_eq!(recent.len(), 10);
    
    // Check success rate (7 success out of 10)
    let success_rate = verification_service.get_success_rate(chrono::Duration::hours(1)).await?;
    assert!((success_rate - 0.7).abs() < 0.01); // ~70% success rate
    
    Ok(())
}

#[tokio::test]
async fn test_weight_calculation_and_setting() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("test_weights.db");
    let db_url = format!("sqlite:{}", db_path.display());
    
    let pool = SqlitePool::connect(&db_url).await?;
    sqlx::migrate!("../../migrations").run(&pool).await?;
    
    // Create test verification data
    let persistence = SimplePersistence::new(pool.clone());
    let repo = persistence.verification_logs();
    
    // Miner 1: High performance
    for i in 0..5 {
        let log = VerificationLog {
            id: Uuid::new_v4().to_string(),
            executor_id: format!("miner1-executor-{}", i),
            miner_uid: 1,
            validator_hotkey: "test-validator".to_string(),
            verification_type: "hardware_attestation".to_string(),
            status: "success".to_string(),
            score: Some(0.95),
            error_message: None,
            attestation_data: Some(serde_json::json!({
                "gpu_count": 8,
                "gpu_memory_total_gb": 640
            })),
            started_at: chrono::Utc::now() - chrono::Duration::hours(i as i64),
            completed_at: Some(chrono::Utc::now() - chrono::Duration::hours(i as i64)),
            created_at: chrono::Utc::now() - chrono::Duration::hours(i as i64),
        };
        repo.create(&log).await?;
    }
    
    // Miner 2: Medium performance
    for i in 0..3 {
        let log = VerificationLog {
            id: Uuid::new_v4().to_string(),
            executor_id: format!("miner2-executor-{}", i),
            miner_uid: 2,
            validator_hotkey: "test-validator".to_string(),
            verification_type: "hardware_attestation".to_string(),
            status: "success".to_string(),
            score: Some(0.70),
            error_message: None,
            attestation_data: Some(serde_json::json!({
                "gpu_count": 4,
                "gpu_memory_total_gb": 192
            })),
            started_at: chrono::Utc::now() - chrono::Duration::hours(i as i64),
            completed_at: Some(chrono::Utc::now() - chrono::Duration::hours(i as i64)),
            created_at: chrono::Utc::now() - chrono::Duration::hours(i as i64),
        };
        repo.create(&log).await?;
    }
    
    // Miner 3: Poor performance
    for i in 0..2 {
        let log = VerificationLog {
            id: Uuid::new_v4().to_string(),
            executor_id: format!("miner3-executor-{}", i),
            miner_uid: 3,
            validator_hotkey: "test-validator".to_string(),
            verification_type: "hardware_attestation".to_string(),
            status: if i == 0 { "success" } else { "failed" }.to_string(),
            score: if i == 0 { Some(0.40) } else { None },
            error_message: if i == 1 { Some("GPU error".to_string()) } else { None },
            attestation_data: None,
            started_at: chrono::Utc::now() - chrono::Duration::hours(i as i64),
            completed_at: Some(chrono::Utc::now() - chrono::Duration::hours(i as i64)),
            created_at: chrono::Utc::now() - chrono::Duration::hours(i as i64),
        };
        repo.create(&log).await?;
    }
    
    // Calculate weights
    let bittensor_config = BittensorConfig {
        wallet_name: "test_wallet".to_string(),
        hotkey_name: "test_hotkey".to_string(),
        network: "test".to_string(),
        netuid: 42,
        chain_endpoint: "wss://test.invalid:443".to_string(),
        weight_interval_secs: 300,
    };
    
    let weight_setter = WeightSetter::new(
        bittensor_config,
        Arc::new(VerificationService::new(pool.clone()))
    );
    
    let weights = weight_setter.calculate_weights().await?;
    
    // Verify weight ordering
    assert!(!weights.is_empty());
    
    // Find miners in weights
    let miner1_weight = weights.iter().find(|w| w.uid == 1);
    let miner2_weight = weights.iter().find(|w| w.uid == 2);
    let miner3_weight = weights.iter().find(|w| w.uid == 3);
    
    // Miner 1 should have highest weight if present
    if let (Some(w1), Some(w2)) = (miner1_weight, miner2_weight) {
        assert!(w1.weight > w2.weight, "Miner 1 should have higher weight than Miner 2");
    }
    
    if let (Some(w2), Some(w3)) = (miner2_weight, miner3_weight) {
        assert!(w2.weight > w3.weight, "Miner 2 should have higher weight than Miner 3");
    }
    
    Ok(())
}

#[tokio::test]
async fn test_api_integration_with_validation_data() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("test_api_integration.db");
    let db_url = format!("sqlite:{}", db_path.display());
    
    let pool = SqlitePool::connect(&db_url).await?;
    sqlx::migrate!("../../migrations").run(&pool).await?;
    
    // Create test data
    let persistence = SimplePersistence::new(pool.clone());
    let repo = persistence.verification_logs();
    
    // Create verifications with GPU data
    let verifications = vec![
        VerificationLog {
            id: Uuid::new_v4().to_string(),
            executor_id: "gpu-executor-1".to_string(),
            miner_uid: 10,
            validator_hotkey: "validator-1".to_string(),
            verification_type: "hardware_attestation".to_string(),
            status: "success".to_string(),
            score: Some(0.92),
            error_message: None,
            attestation_data: Some(serde_json::json!({
                "gpu_info": [
                    {
                        "name": "NVIDIA A100-SXM4-80GB",
                        "memory_mb": 81920,
                        "compute_capability": "8.0"
                    },
                    {
                        "name": "NVIDIA A100-SXM4-80GB",
                        "memory_mb": 81920,
                        "compute_capability": "8.0"
                    }
                ]
            })),
            started_at: chrono::Utc::now(),
            completed_at: Some(chrono::Utc::now()),
            created_at: chrono::Utc::now(),
        },
        VerificationLog {
            id: Uuid::new_v4().to_string(),
            executor_id: "gpu-executor-2".to_string(),
            miner_uid: 11,
            validator_hotkey: "validator-1".to_string(),
            verification_type: "hardware_attestation".to_string(),
            status: "success".to_string(),
            score: Some(0.88),
            error_message: None,
            attestation_data: Some(serde_json::json!({
                "gpu_info": [
                    {
                        "name": "NVIDIA RTX 4090",
                        "memory_mb": 24576,
                        "compute_capability": "8.9"
                    }
                ]
            })),
            started_at: chrono::Utc::now(),
            completed_at: Some(chrono::Utc::now()),
            created_at: chrono::Utc::now(),
        },
    ];
    
    for log in verifications {
        repo.create(&log).await?;
    }
    
    // Create API context and router
    let config = ValidatorConfig::default();
    let context = ApiContext::new(config, pool.clone());
    let app = create_router(context);
    
    // Test capacity endpoint
    let response = app
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .uri("/api/v1/capacity")
                .body(axum::body::Body::empty())?
        )
        .await?;
    
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    
    let body = hyper::body::to_bytes(response.into_body()).await?;
    let capacity: serde_json::Value = serde_json::from_slice(&body)?;
    
    assert_eq!(capacity["total_gpus"], 3); // 2 A100s + 1 RTX 4090
    assert_eq!(capacity["gpu_types"]["NVIDIA A100-SXM4-80GB"], 2);
    assert_eq!(capacity["gpu_types"]["NVIDIA RTX 4090"], 1);
    
    Ok(())
}

fn calculate_score(attestation_data: &AttestationData) -> Result<f64> {
    let mut score = 0.0;
    
    // Hardware score (40%)
    let cpu_score = (attestation_data.hardware_info.cpu_count as f64 / 64.0).min(1.0);
    let memory_score = (attestation_data.hardware_info.memory_gb as f64 / 256.0).min(1.0);
    let hardware_score = (cpu_score + memory_score) / 2.0;
    score += hardware_score * 0.4;
    
    // GPU score (40%)
    if !attestation_data.gpu_info.is_empty() {
        let gpu_count_score = (attestation_data.gpu_info.len() as f64 / 8.0).min(1.0);
        let gpu_memory_score = attestation_data.gpu_info.iter()
            .map(|gpu| gpu.memory_mb as f64 / 81920.0) // A100 80GB as reference
            .sum::<f64>() / attestation_data.gpu_info.len() as f64;
        let gpu_score = (gpu_count_score + gpu_memory_score.min(1.0)) / 2.0;
        score += gpu_score * 0.4;
    }
    
    // VDF score (10%)
    if let Some(vdf) = &attestation_data.vdf_result {
        let vdf_score = if vdf.iterations >= vdf.difficulty as u64 { 1.0 } else { 0.5 };
        score += vdf_score * 0.1;
    }
    
    // Environment score (10%)
    let has_docker = attestation_data.docker_info.is_some();
    let env_score = if has_docker { 1.0 } else { 0.5 };
    score += env_score * 0.1;
    
    Ok(score)
}

#[tokio::test]
async fn test_validation_session_persistence() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("test_session.db");
    let db_url = format!("sqlite:{}", db_path.display());
    
    let pool = SqlitePool::connect(&db_url).await?;
    sqlx::migrate!("../../migrations").run(&pool).await?;
    
    // Test session state persistence across restarts
    let session_id = Uuid::new_v4().to_string();
    
    // Create initial session
    sqlx::query!(
        r#"
        INSERT INTO validation_sessions (id, executor_id, validator_hotkey, state, created_at)
        VALUES (?1, ?2, ?3, ?4, ?5)
        "#,
        session_id,
        "test-executor",
        "test-validator", 
        "active",
        chrono::Utc::now()
    )
    .execute(&pool)
    .await?;
    
    // Simulate restart by creating new pool
    drop(pool);
    let new_pool = SqlitePool::connect(&db_url).await?;
    
    // Verify session persisted
    let session = sqlx::query!(
        "SELECT * FROM validation_sessions WHERE id = ?",
        session_id
    )
    .fetch_optional(&new_pool)
    .await?;
    
    assert!(session.is_some());
    assert_eq!(session.unwrap().state, "active");
    
    Ok(())
}