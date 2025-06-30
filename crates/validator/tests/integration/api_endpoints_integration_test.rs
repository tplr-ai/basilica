use anyhow::Result;
use axum::{
    body::Body,
    http::{Request, StatusCode},
    Router,
};
use serde_json::json;
use sqlx::SqlitePool;
use std::net::SocketAddr;
use tempfile::TempDir;
use tower::ServiceExt;
use uuid::Uuid;
use validator::api::{
    create_router, ApiContext,
    types::{
        CapacityQuery, CapacityResponse, CreateRentalRequest, CreateRentalResponse,
        GpuCapacity, HealthResponse, LogQuery, PaginatedLogs, RentalStatus,
        UpdateRentalRequest, VerificationLogResponse,
    },
};
use validator::config::{ApiConfig, ValidatorConfig};
use validator::persistence::{SimplePersistence, VerificationLog};

async fn setup_test_api() -> Result<(Router, SqlitePool)> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("test_api.db");
    let db_url = format!("sqlite:{}", db_path.display());
    
    let pool = SqlitePool::connect(&db_url).await?;
    sqlx::migrate!("../../migrations").run(&pool).await?;
    
    let api_config = ApiConfig {
        api_key: Some("test-api-key-12345".to_string()),
        max_body_size: 1024 * 1024, // 1MB
        bind_address: "127.0.0.1:0".parse()?,
    };
    
    let config = ValidatorConfig {
        api: api_config,
        ..Default::default()
    };
    
    let context = ApiContext::new(config, pool.clone());
    let router = create_router(context);
    
    Ok((router, pool))
}

#[tokio::test]
async fn test_health_endpoint() -> Result<()> {
    let (app, _pool) = setup_test_api().await?;
    
    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())?
        )
        .await?;
    
    assert_eq!(response.status(), StatusCode::OK);
    
    let body = hyper::body::to_bytes(response.into_body()).await?;
    let health: HealthResponse = serde_json::from_slice(&body)?;
    
    assert_eq!(health.status, "healthy");
    assert!(health.version.starts_with("0."));
    assert!(health.uptime_seconds >= 0);
    
    Ok(())
}

#[tokio::test]
async fn test_capacity_endpoint() -> Result<()> {
    let (app, pool) = setup_test_api().await?;
    
    // Insert test data
    sqlx::query!(
        r#"
        INSERT INTO verification_logs (id, executor_id, miner_uid, validator_hotkey, 
            verification_type, status, score, attestation_data, started_at, created_at)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
        "#,
        Uuid::new_v4().to_string(),
        "executor-gpu-1",
        10,
        "validator-1",
        "hardware_attestation",
        "success",
        0.95,
        json!({
            "gpu_info": [{
                "name": "NVIDIA RTX 4090",
                "memory_mb": 24576,
                "compute_capability": "8.9"
            }]
        }).to_string(),
        chrono::Utc::now(),
        chrono::Utc::now()
    )
    .execute(&pool)
    .await?;
    
    // Test without filters
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/v1/capacity")
                .header("X-API-Key", "test-api-key-12345")
                .body(Body::empty())?
        )
        .await?;
    
    assert_eq!(response.status(), StatusCode::OK);
    
    let body = hyper::body::to_bytes(response.into_body()).await?;
    let capacity: CapacityResponse = serde_json::from_slice(&body)?;
    
    assert!(capacity.total_gpus > 0);
    assert!(capacity.available_gpus >= 0);
    assert!(!capacity.gpu_types.is_empty());
    
    // Test with GPU type filter
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/v1/capacity?gpu_type=RTX%204090")
                .header("X-API-Key", "test-api-key-12345")
                .body(Body::empty())?
        )
        .await?;
    
    assert_eq!(response.status(), StatusCode::OK);
    
    Ok(())
}

#[tokio::test]
async fn test_verification_logs_endpoint() -> Result<()> {
    let (app, pool) = setup_test_api().await?;
    
    // Insert test logs
    let persistence = SimplePersistence::new(pool.clone());
    let repo = persistence.verification_logs();
    
    for i in 0..5 {
        let log = VerificationLog {
            id: Uuid::new_v4().to_string(),
            executor_id: format!("executor-{}", i),
            miner_uid: (i * 10) as u16,
            validator_hotkey: "validator-1".to_string(),
            verification_type: "hardware_attestation".to_string(),
            status: if i % 2 == 0 { "success" } else { "failed" }.to_string(),
            score: if i % 2 == 0 { Some(0.9 + (i as f64 * 0.01)) } else { None },
            error_message: if i % 2 == 1 { Some("Test error".to_string()) } else { None },
            attestation_data: None,
            started_at: chrono::Utc::now() - chrono::Duration::hours(i as i64),
            completed_at: Some(chrono::Utc::now() - chrono::Duration::hours(i as i64)),
            created_at: chrono::Utc::now() - chrono::Duration::hours(i as i64),
        };
        
        repo.create(&log).await?;
    }
    
    // Test basic listing
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/v1/logs")
                .header("X-API-Key", "test-api-key-12345")
                .body(Body::empty())?
        )
        .await?;
    
    assert_eq!(response.status(), StatusCode::OK);
    
    let body = hyper::body::to_bytes(response.into_body()).await?;
    let paginated: PaginatedLogs = serde_json::from_slice(&body)?;
    
    assert_eq!(paginated.logs.len(), 5);
    assert_eq!(paginated.total, 5);
    assert_eq!(paginated.page, 1);
    
    // Test with filters
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/v1/logs?status=success&limit=2")
                .header("X-API-Key", "test-api-key-12345")
                .body(Body::empty())?
        )
        .await?;
    
    assert_eq!(response.status(), StatusCode::OK);
    
    let body = hyper::body::to_bytes(response.into_body()).await?;
    let filtered: PaginatedLogs = serde_json::from_slice(&body)?;
    
    assert!(filtered.logs.len() <= 2);
    assert!(filtered.logs.iter().all(|log| log.status == "success"));
    
    Ok(())
}

#[tokio::test]
async fn test_rentals_crud_endpoints() -> Result<()> {
    let (app, _pool) = setup_test_api().await?;
    
    // Create rental
    let create_request = CreateRentalRequest {
        gpu_type: "NVIDIA RTX 4090".to_string(),
        quantity: 2,
        duration_hours: 24,
        max_price_per_hour: 1.5,
        requirements: Some(json!({
            "min_memory_gb": 24,
            "cuda_version": "12.0"
        })),
    };
    
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/v1/rentals")
                .method("POST")
                .header("X-API-Key", "test-api-key-12345")
                .header("Content-Type", "application/json")
                .body(Body::from(serde_json::to_string(&create_request)?))?
        )
        .await?;
    
    assert_eq!(response.status(), StatusCode::CREATED);
    
    let body = hyper::body::to_bytes(response.into_body()).await?;
    let created: CreateRentalResponse = serde_json::from_slice(&body)?;
    
    assert!(!created.rental_id.is_empty());
    assert_eq!(created.status, RentalStatus::Pending);
    
    let rental_id = created.rental_id;
    
    // Get rental
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(&format!("/api/v1/rentals/{}", rental_id))
                .header("X-API-Key", "test-api-key-12345")
                .body(Body::empty())?
        )
        .await?;
    
    assert_eq!(response.status(), StatusCode::OK);
    
    // Update rental
    let update_request = UpdateRentalRequest {
        status: Some(RentalStatus::Active),
        assigned_executors: Some(vec!["executor-1".to_string(), "executor-2".to_string()]),
    };
    
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(&format!("/api/v1/rentals/{}", rental_id))
                .method("PUT")
                .header("X-API-Key", "test-api-key-12345")
                .header("Content-Type", "application/json")
                .body(Body::from(serde_json::to_string(&update_request)?))?
        )
        .await?;
    
    assert_eq!(response.status(), StatusCode::OK);
    
    // Cancel rental
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(&format!("/api/v1/rentals/{}/cancel", rental_id))
                .method("POST")
                .header("X-API-Key", "test-api-key-12345")
                .body(Body::empty())?
        )
        .await?;
    
    assert_eq!(response.status(), StatusCode::OK);
    
    Ok(())
}

#[tokio::test]
async fn test_api_authentication() -> Result<()> {
    let (app, _pool) = setup_test_api().await?;
    
    // Test without API key
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/v1/capacity")
                .body(Body::empty())?
        )
        .await?;
    
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    
    // Test with wrong API key
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/v1/capacity")
                .header("X-API-Key", "wrong-api-key")
                .body(Body::empty())?
        )
        .await?;
    
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    
    // Test with correct API key
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/v1/capacity")
                .header("X-API-Key", "test-api-key-12345")
                .body(Body::empty())?
        )
        .await?;
    
    assert_eq!(response.status(), StatusCode::OK);
    
    Ok(())
}

#[tokio::test]
async fn test_pagination() -> Result<()> {
    let (app, pool) = setup_test_api().await?;
    
    // Create many logs
    let persistence = SimplePersistence::new(pool);
    let repo = persistence.verification_logs();
    
    for i in 0..25 {
        let log = VerificationLog {
            id: Uuid::new_v4().to_string(),
            executor_id: format!("executor-{}", i),
            miner_uid: i as u16,
            validator_hotkey: "validator-1".to_string(),
            verification_type: "hardware_attestation".to_string(),
            status: "success".to_string(),
            score: Some(0.9),
            error_message: None,
            attestation_data: None,
            started_at: chrono::Utc::now(),
            completed_at: Some(chrono::Utc::now()),
            created_at: chrono::Utc::now(),
        };
        
        repo.create(&log).await?;
    }
    
    // Test first page
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/v1/logs?page=1&limit=10")
                .header("X-API-Key", "test-api-key-12345")
                .body(Body::empty())?
        )
        .await?;
    
    let body = hyper::body::to_bytes(response.into_body()).await?;
    let page1: PaginatedLogs = serde_json::from_slice(&body)?;
    
    assert_eq!(page1.logs.len(), 10);
    assert_eq!(page1.page, 1);
    assert_eq!(page1.total, 25);
    assert!(page1.has_next);
    
    // Test second page
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/v1/logs?page=2&limit=10")
                .header("X-API-Key", "test-api-key-12345")
                .body(Body::empty())?
        )
        .await?;
    
    let body = hyper::body::to_bytes(response.into_body()).await?;
    let page2: PaginatedLogs = serde_json::from_slice(&body)?;
    
    assert_eq!(page2.logs.len(), 10);
    assert_eq!(page2.page, 2);
    
    // Test last page
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/v1/logs?page=3&limit=10")
                .header("X-API-Key", "test-api-key-12345")
                .body(Body::empty())?
        )
        .await?;
    
    let body = hyper::body::to_bytes(response.into_body()).await?;
    let page3: PaginatedLogs = serde_json::from_slice(&body)?;
    
    assert_eq!(page3.logs.len(), 5);
    assert_eq!(page3.page, 3);
    assert!(!page3.has_next);
    
    Ok(())
}

#[tokio::test]
async fn test_error_responses() -> Result<()> {
    let (app, _pool) = setup_test_api().await?;
    
    // Test 404 - Not Found
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/v1/rentals/non-existent-id")
                .header("X-API-Key", "test-api-key-12345")
                .body(Body::empty())?
        )
        .await?;
    
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    
    // Test 400 - Bad Request (invalid JSON)
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/v1/rentals")
                .method("POST")
                .header("X-API-Key", "test-api-key-12345")
                .header("Content-Type", "application/json")
                .body(Body::from("{invalid json"))?
        )
        .await?;
    
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    
    // Test 422 - Unprocessable Entity (validation error)
    let invalid_request = CreateRentalRequest {
        gpu_type: "".to_string(), // Empty GPU type
        quantity: 0, // Invalid quantity
        duration_hours: 0, // Invalid duration
        max_price_per_hour: -1.0, // Negative price
        requirements: None,
    };
    
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/v1/rentals")
                .method("POST")
                .header("X-API-Key", "test-api-key-12345")
                .header("Content-Type", "application/json")
                .body(Body::from(serde_json::to_string(&invalid_request)?))?
        )
        .await?;
    
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    
    Ok(())
}