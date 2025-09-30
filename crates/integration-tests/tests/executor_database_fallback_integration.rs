use anyhow::Result;
use basilica_common::identity::ExecutorId;
use basilica_validator::miner_prover::types::ExecutorInfoDetailed;
use basilica_validator::persistence::SimplePersistence;
use sqlx::SqlitePool;
use std::str::FromStr;

/// Integration test for executor database fallback functionality
/// Tests the complete flow: database query -> data conversion -> executor list combination
#[tokio::test]
async fn test_get_known_executors_for_miner_with_test_data() -> Result<()> {
    // Create in-memory test database
    let pool = SqlitePool::connect("sqlite::memory:").await?;

    // Create the miner_executors table schema
    sqlx::query(
        r#"
        CREATE TABLE miner_executors (
            id TEXT PRIMARY KEY,
            miner_id TEXT NOT NULL,
            executor_id TEXT NOT NULL,
            grpc_address TEXT NOT NULL,
            gpu_count INTEGER NOT NULL,
            gpu_specs TEXT NOT NULL,
            cpu_specs TEXT NOT NULL,
            location TEXT,
            status TEXT DEFAULT 'unknown',
            last_health_check TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            gpu_uuids TEXT
        )
    "#,
    )
    .execute(&pool)
    .await?;

    // Insert test data mimicking production structure
    sqlx::query(r#"
        INSERT INTO miner_executors
        (id, miner_id, executor_id, grpc_address, gpu_count, gpu_specs, cpu_specs, location, status, last_health_check, created_at, updated_at, gpu_uuids)
        VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    "#)
    .bind("miner_171_test_id")
    .bind("miner_171")
    .bind("miner171__b273a70c-cdcb-448c-8a0b-3945c83cfa40")
    .bind("84.32.59.107:50051")
    .bind(2)
    .bind("{}")
    .bind("{}")
    .bind("discovered")
    .bind("verified")
    .bind(chrono::Utc::now().format("%Y-%m-%d %H:%M:%S").to_string()) // Current timestamp
    .bind("2025-08-26 10:11:50")
    .bind("2025-08-27 10:00:00")
    .bind("")
    .execute(&pool).await?;

    // Insert more test data for miner_124
    sqlx::query(r#"
        INSERT INTO miner_executors
        (id, miner_id, executor_id, grpc_address, gpu_count, gpu_specs, cpu_specs, location, status, last_health_check, created_at, updated_at, gpu_uuids)
        VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?),
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    "#)
    .bind("miner_124_test_id_1")
    .bind("miner_124")
    .bind("miner124__642ecf9d-6cb0-4c79-8ef0-18dc45d545f7")
    .bind("204.12.162.228:50051")
    .bind(1)
    .bind("{}")
    .bind("{}")
    .bind("discovered")
    .bind("verified")
    .bind(chrono::Utc::now().format("%Y-%m-%d %H:%M:%S").to_string())
    .bind("2025-08-26 10:11:50")
    .bind(chrono::Utc::now().format("%Y-%m-%d %H:%M:%S").to_string())
    .bind("")
    .bind("miner_124_test_id_2")
    .bind("miner_124")
    .bind("miner124__534fe257-dbd5-4ac5-86ab-ad597314c9ab")
    .bind("216.81.248.61:50051")
    .bind(1)
    .bind("{}")
    .bind("{}")
    .bind("discovered")
    .bind("verified")
    .bind(chrono::Utc::now().format("%Y-%m-%d %H:%M:%S").to_string())
    .bind("2025-08-26 10:11:50")
    .bind(chrono::Utc::now().format("%Y-%m-%d %H:%M:%S").to_string())
    .bind("")
    .execute(&pool).await?;

    // Insert offline executor that should be filtered out
    sqlx::query(r#"
        INSERT INTO miner_executors
        (id, miner_id, executor_id, grpc_address, gpu_count, gpu_specs, cpu_specs, location, status, last_health_check, created_at, updated_at, gpu_uuids)
        VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    "#)
    .bind("miner_214_test_id")
    .bind("miner_214")
    .bind("miner214__offline-executor")
    .bind("192.168.1.1:50051")
    .bind(0)
    .bind("{}")
    .bind("{}")
    .bind("discovered")
    .bind("offline")
    .bind("2025-08-27 09:00:00")
    .bind("2025-08-26 10:11:50")
    .bind("2025-08-27 09:00:00")
    .bind("")
    .execute(&pool).await?;

    // Now test with SimplePersistence method using the test database
    let persistence = SimplePersistence::with_pool(pool);

    // Test with miner_171 (should have 1 verified executor)
    let result = persistence.get_known_executors_for_miner(171).await?;

    println!("Miner 171 results: {:?}", result);

    // Validate the result structure
    assert_eq!(result.len(), 1, "Miner 171 should have 1 verified executor");

    let (executor_id, grpc_address, gpu_count, status) = &result[0];
    assert_eq!(*gpu_count, 2, "Executor should have 2 GPUs");
    assert_eq!(status, "verified", "Executor should be verified");
    assert_eq!(
        grpc_address, "84.32.59.107:50051",
        "gRPC address should match expected value"
    );
    assert_eq!(
        executor_id, "miner171__b273a70c-cdcb-448c-8a0b-3945c83cfa40",
        "Executor ID should match expected value"
    );

    // Test with miner_124 (should have 2 verified executors)
    let result_124 = persistence.get_known_executors_for_miner(124).await?;

    println!("Miner 124 results: {:?}", result_124);

    // Should have 2 verified executors (offline ones filtered out)
    assert_eq!(
        result_124.len(),
        2,
        "Miner 124 should have 2 verified executors"
    );

    for (executor_id, grpc_address, gpu_count, status) in &result_124 {
        assert_eq!(
            status, "verified",
            "All returned executors should be verified"
        );
        assert_eq!(*gpu_count, 1, "Each executor should have 1 GPU");
        assert!(
            grpc_address.contains(":50051"),
            "gRPC address should have port 50051"
        );
        assert!(
            executor_id.starts_with("miner124__"),
            "Executor ID should have miner124 prefix"
        );
    }

    // Test with non-existent miner
    let result_empty = persistence.get_known_executors_for_miner(999).await?;
    assert_eq!(
        result_empty.len(),
        0,
        "Non-existent miner should return empty result"
    );

    // Test with miner that has only offline executors
    let result_offline = persistence.get_known_executors_for_miner(214).await?;
    assert_eq!(
        result_offline.len(),
        0,
        "Miner with only offline executors should return empty result"
    );

    println!("✅ Database persistence integration tests passed!");
    Ok(())
}

/// Test the complete data conversion pipeline from database tuples to ExecutorInfoDetailed
#[tokio::test]
async fn test_executor_data_conversion_pipeline() -> Result<()> {
    // Test data based on actual production database results
    let db_data = vec![
        (
            "miner124__642ecf9d-6cb0-4c79-8ef0-18dc45d545f7".to_string(),
            "204.12.162.228:50051".to_string(),
            1,
            "verified".to_string(),
        ),
        (
            "miner124__534fe257-dbd5-4ac5-86ab-ad597314c9ab".to_string(),
            "216.81.248.61:50051".to_string(),
            1,
            "verified".to_string(),
        ),
    ];

    // Test the conversion logic that exists in VerificationEngine
    let miner_uid = 124u16;
    let prefix = format!("miner{}__", miner_uid);
    let mut converted_executors = Vec::new();

    for (executor_id, grpc_address, gpu_count, status) in &db_data {
        // Test prefix stripping logic
        let clean_executor_id = if executor_id.starts_with(&prefix) {
            executor_id.strip_prefix(&prefix).unwrap_or(executor_id)
        } else {
            executor_id
        };

        println!(
            "Original ID: {}, Clean ID: {}",
            executor_id, clean_executor_id
        );

        // Test ExecutorId parsing
        let executor_id_parsed = ExecutorId::from_str(clean_executor_id)?;

        // Verify it's a valid UUID format
        assert_eq!(clean_executor_id.len(), 36, "Should be a valid UUID length");
        assert!(
            clean_executor_id.contains("-"),
            "Should contain UUID hyphens"
        );

        // Test ExecutorInfoDetailed creation
        let executor_info = ExecutorInfoDetailed {
            id: executor_id_parsed,
            status: status.clone(),
            capabilities: if *gpu_count > 0 {
                vec!["gpu".to_string()]
            } else {
                vec![]
            },
            grpc_endpoint: grpc_address.clone(),
        };

        // Validate converted data
        assert_eq!(executor_info.status, "verified");
        assert_eq!(executor_info.capabilities, vec!["gpu"]);
        assert_eq!(executor_info.grpc_endpoint, *grpc_address);

        converted_executors.push(executor_info);
    }

    assert_eq!(
        converted_executors.len(),
        2,
        "Should have converted 2 executors"
    );

    println!("✅ Executor data conversion pipeline tests passed!");
    Ok(())
}

/// Test the executor list combination logic for deduplication
#[test]
fn test_executor_list_combination_deduplication() -> Result<()> {
    // Create test executor with same ID but different sources
    let executor_id = ExecutorId::from_str("642ecf9d-6cb0-4c79-8ef0-18dc45d545f7")?;

    let discovered_executor = ExecutorInfoDetailed {
        id: executor_id.clone(),
        status: "online".to_string(),
        capabilities: vec!["gpu".to_string()],
        grpc_endpoint: "192.168.1.1:50051".to_string(),
    };

    let known_executor = ExecutorInfoDetailed {
        id: executor_id.clone(),
        status: "verified".to_string(),
        capabilities: vec!["gpu".to_string()],
        grpc_endpoint: "192.168.1.2:50051".to_string(),
    };

    // Test combination logic (discovered executors have priority)
    let discovered = vec![discovered_executor.clone()];
    let known = vec![known_executor];

    let mut combined = Vec::new();
    let mut seen_ids = std::collections::HashSet::new();

    // Add discovered first (priority)
    for executor in discovered {
        if seen_ids.insert(executor.id.to_string()) {
            combined.push(executor);
        }
    }

    // Add known that weren't discovered
    for executor in known {
        if seen_ids.insert(executor.id.to_string()) {
            combined.push(executor);
        }
    }

    // Validate deduplication worked
    assert_eq!(combined.len(), 1, "Should have deduplicated to 1 executor");
    assert_eq!(
        combined[0].status, "online",
        "Should keep discovered executor status"
    );
    assert_eq!(
        combined[0].grpc_endpoint, "192.168.1.1:50051",
        "Should keep discovered executor endpoint"
    );

    println!("✅ Executor list combination deduplication tests passed!");
    Ok(())
}

/// Test edge cases and error conditions
#[tokio::test]
async fn test_edge_cases_and_error_conditions() -> Result<()> {
    // Test with invalid database path (should fail gracefully)
    let invalid_db_result = SqlitePool::connect("sqlite:///nonexistent/database.db").await;
    assert!(
        invalid_db_result.is_err(),
        "Should fail with invalid database path"
    );

    // Test ExecutorId parsing with invalid UUID
    let invalid_uuid_result = ExecutorId::from_str("invalid-uuid");
    assert!(
        invalid_uuid_result.is_err(),
        "Should fail with invalid UUID format"
    );

    // Test ExecutorId parsing with valid UUID
    let valid_uuid_result = ExecutorId::from_str("642ecf9d-cdcb-448c-8a0b-3945c83cfa40");
    assert!(
        valid_uuid_result.is_ok(),
        "Should succeed with valid UUID format"
    );

    println!("✅ Edge cases and error condition tests passed!");
    Ok(())
}
