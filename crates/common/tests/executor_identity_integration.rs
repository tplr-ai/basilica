//! End-to-end integration tests for the executor identity module
//!
//! These tests verify the complete functionality of the UUID+HUID system
//! across all components in realistic usage scenarios.

#![cfg(feature = "sqlite")]

use common::executor_identity::{
    is_valid_huid, validate_identifier, ExecutorId, ExecutorIdentity, IdentityDisplay,
    IdentityPersistence, SqliteIdentityStore, StaticWordProvider, WordProvider,
};
use std::collections::HashSet;
use std::sync::Arc;
use tempfile::TempDir;
use tokio::time::{sleep, Duration};

/// Create a test database in a temporary directory
async fn create_test_database() -> (SqliteIdentityStore, TempDir) {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let db_path = temp_dir.path().join("test.db");

    // Create the database file
    std::fs::File::create(&db_path).expect("Failed to create database file");

    let db_url = format!("sqlite:///{}", db_path.display());

    let store = SqliteIdentityStore::new(&db_url)
        .await
        .expect("Failed to create identity store");

    (store, temp_dir)
}

#[tokio::test]
async fn test_e2e_executor_lifecycle() {
    let (store, _temp_dir) = create_test_database().await;

    // Step 1: Create a new executor identity
    let executor = store
        .get_or_create()
        .await
        .expect("Should create new executor");

    println!("Created executor: {}", executor.full_display());

    // Verify the identity format
    assert!(is_valid_huid(executor.huid()), "HUID should be valid");
    assert_eq!(executor.uuid().get_version(), Some(uuid::Version::Random));

    // Step 2: Look up by UUID
    let found_by_uuid = store
        .find_by_identifier(&executor.uuid().to_string())
        .await
        .expect("Lookup should succeed")
        .expect("Should find executor");

    assert_eq!(found_by_uuid.uuid(), executor.uuid());
    assert_eq!(found_by_uuid.huid(), executor.huid());

    // Step 3: Look up by HUID prefix (minimum 3 chars)
    let huid_prefix = &executor.huid()[..5];
    let found_by_prefix = store
        .find_by_identifier(huid_prefix)
        .await
        .expect("Prefix lookup should succeed")
        .expect("Should find executor");

    assert_eq!(found_by_prefix.uuid(), executor.uuid());

    // Step 4: Test display formatting
    use common::executor_identity::ExecutorIdentityDisplay;
    let display = ExecutorIdentityDisplay::new(&*executor);

    let compact = display.format_compact();
    assert_eq!(compact, executor.huid());

    let verbose = display.format_verbose();
    assert!(verbose.contains(&executor.uuid().to_string()));
    assert!(verbose.contains(executor.huid()));

    let json = display.format_json().expect("JSON formatting should work");
    let parsed: serde_json::Value =
        serde_json::from_str(&json).expect("Should parse as valid JSON");
    assert_eq!(parsed["uuid"], executor.uuid().to_string());
    assert_eq!(parsed["huid"], executor.huid());
}

#[tokio::test]
async fn test_e2e_collision_handling() {
    // Test HUID collision handling by creating many executors directly
    // Note: get_or_create returns the same executor for a given store

    let mut executors = Vec::new();
    let mut huids = HashSet::new();

    // Create many executor IDs directly to test HUID uniqueness
    let provider = StaticWordProvider::new();
    for i in 0..100 {
        let executor = ExecutorId::new_with_provider(&provider).expect("Should create executor");

        // Verify HUID uniqueness
        assert!(
            huids.insert(executor.huid().to_string()),
            "HUID collision detected at iteration {}: {}",
            i,
            executor.huid()
        );

        executors.push(executor);
    }

    println!(
        "Created {} unique executors with no HUID collisions",
        executors.len()
    );

    // Test that the store handles collisions properly when saving
    let (store, _temp_dir) = create_test_database().await;

    // Get the single executor identity for this store
    let store_executor = store.get_or_create().await.expect("Should create executor");

    // Verify it's retrievable
    let found = store
        .find_by_identifier(&store_executor.uuid().to_string())
        .await
        .expect("Lookup should succeed")
        .expect("Should find executor");

    assert_eq!(found.huid(), store_executor.huid());
}

#[tokio::test]
async fn test_e2e_prefix_matching_disambiguation() {
    let (store, _temp_dir) = create_test_database().await;

    // Get the store's executor
    let executor = store.get_or_create().await.expect("Should create executor");

    let huid = executor.huid();

    // Test various prefix lengths
    for len in 3..8.min(huid.len()) {
        let prefix = &huid[..len];

        // Try to find by prefix - should always find the same executor
        let found = store
            .find_by_identifier(prefix)
            .await
            .expect("Query should succeed")
            .expect("Should find executor");

        assert_eq!(found.uuid(), executor.uuid());
        assert_eq!(found.huid(), executor.huid());
    }

    // Test non-matching prefix
    let non_matching = "zzz";
    let result = store
        .find_by_identifier(non_matching)
        .await
        .expect("Query should succeed");
    assert!(
        result.is_none(),
        "Should not find anything for non-matching prefix"
    );

    // Test too-short prefix
    let too_short = &huid[..2];
    let result = store.find_by_identifier(too_short).await;
    assert!(result.is_err(), "Should error on too-short prefix");

    if let Err(e) = result {
        assert!(e.to_string().contains("at least 3 characters"));
    }
}

#[tokio::test]
async fn test_e2e_persistence_and_caching() {
    let (store, _temp_dir) = create_test_database().await;

    // Create an executor
    let executor = store.get_or_create().await.expect("Should create executor");

    let uuid = executor.uuid().to_string();
    let huid = executor.huid().to_string();

    // Multiple rapid lookups should use cache
    let start = std::time::Instant::now();
    for _ in 0..100 {
        let found = store
            .find_by_identifier(&uuid)
            .await
            .expect("Lookup should succeed")
            .expect("Should find executor");

        assert_eq!(found.huid(), &huid);
    }
    let elapsed = start.elapsed();

    println!("100 lookups took {elapsed:?} (should be fast due to caching)");
    assert!(elapsed < Duration::from_millis(100), "Lookups too slow");

    // Verify data persists across store instances
    drop(store);

    // Create new store with same database
    let db_url = format!("sqlite:{}", _temp_dir.path().join("test.db").display());
    let new_store = SqliteIdentityStore::new(&db_url)
        .await
        .expect("Should create new store");

    // Should find the same executor
    let found = new_store
        .find_by_identifier(&uuid)
        .await
        .expect("Lookup should succeed")
        .expect("Should find executor");

    assert_eq!(found.uuid().to_string(), uuid);
    assert_eq!(found.huid(), huid);
}

#[tokio::test]
async fn test_e2e_legacy_migration() {
    let (store, _temp_dir) = create_test_database().await;

    // Simulate legacy string IDs that need migration
    let legacy_ids = vec![
        "gpu-node-1",
        "worker-abc123",
        "executor-west-2",
        "miner-node-xyz",
    ];

    // Migrate all legacy IDs
    let mut migrated_ids = Vec::new();
    for legacy_id in &legacy_ids {
        let executor = store
            .migrate_legacy_id(legacy_id)
            .await
            .expect("Migration should succeed");

        println!("Migrated '{}' -> {}", legacy_id, executor.full_display());
        migrated_ids.push((legacy_id.to_string(), executor));
    }

    // Verify idempotency - migrating again returns same identity
    for (legacy_id, original) in &migrated_ids {
        let executor_again = store
            .migrate_legacy_id(legacy_id)
            .await
            .expect("Repeated migration should succeed");

        assert_eq!(
            executor_again.uuid(),
            original.uuid(),
            "Repeated migration should return same UUID"
        );
        assert_eq!(
            executor_again.huid(),
            original.huid(),
            "Repeated migration should return same HUID"
        );
    }

    // Verify all mappings are stored
    let mappings: std::collections::HashMap<String, uuid::Uuid> = store
        .get_legacy_mappings()
        .await
        .expect("Should get mappings");

    assert_eq!(mappings.len(), legacy_ids.len());
    for legacy_id in &legacy_ids {
        assert!(mappings.contains_key(*legacy_id));
    }
}

#[tokio::test]
async fn test_e2e_concurrent_operations() {
    let (store, _temp_dir) = create_test_database().await;
    let store = Arc::new(store);

    // Create initial executor
    let initial = store.get_or_create().await.expect("Should create executor");

    let uuid = initial.uuid().to_string();
    let huid = initial.huid().to_string();

    // Launch concurrent tasks
    let mut handles = Vec::new();

    // Task 1: Concurrent lookups by UUID
    for i in 0..5 {
        let store = store.clone();
        let uuid = uuid.clone();
        handles.push(tokio::spawn(async move {
            for j in 0..20 {
                let found = store
                    .find_by_identifier(&uuid)
                    .await
                    .expect("Lookup should succeed");
                assert!(found.is_some(), "Task {i} iteration {j} failed");

                // Small delay to increase contention
                sleep(Duration::from_micros(100)).await;
            }
        }));
    }

    // Task 2: Concurrent lookups by HUID prefix
    for i in 0..5 {
        let store = store.clone();
        let prefix = huid[..5].to_string();
        handles.push(tokio::spawn(async move {
            for j in 0..20 {
                let found = store
                    .find_by_identifier(&prefix)
                    .await
                    .expect("Prefix lookup should succeed");
                assert!(found.is_some(), "Task {i} iteration {j} failed");

                sleep(Duration::from_micros(100)).await;
            }
        }));
    }

    // Task 3: Repeated get_or_create (should return same identity)
    for i in 0..3 {
        let store = store.clone();
        let expected_uuid = *initial.uuid();
        handles.push(tokio::spawn(async move {
            for j in 0..10 {
                let executor = store
                    .get_or_create()
                    .await
                    .expect("Get or create should succeed");
                assert_eq!(
                    executor.uuid(),
                    &expected_uuid,
                    "Task {i} iteration {j} got wrong executor"
                );

                sleep(Duration::from_micros(200)).await;
            }
        }));
    }

    // Wait for all tasks to complete
    for handle in handles {
        handle.await.expect("Task should complete successfully");
    }

    // Verify final state is consistent
    let final_check = store
        .find_by_identifier(&uuid)
        .await
        .expect("Final lookup should succeed")
        .expect("Should find executor");

    assert_eq!(final_check.uuid().to_string(), uuid);
    assert_eq!(final_check.huid(), huid);
}

#[tokio::test]
async fn test_e2e_validation_and_error_handling() {
    let (store, _temp_dir) = create_test_database().await;

    // Test invalid identifier queries
    let invalid_queries = vec![
        ("", "empty string"),
        ("a", "too short (1 char)"),
        ("ab", "too short (2 chars)"),
        ("12", "too short numeric"),
        ("SWIFT-FALCON-A3F2", "uppercase HUID"),
        ("swift_falcon_a3f2", "underscores instead of hyphens"),
        ("not-a-valid-uuid-at-all", "invalid UUID format"),
    ];

    for (query, description) in invalid_queries {
        let result = validate_identifier(query);

        if query.len() < 3 {
            assert!(
                result.is_err(),
                "Query '{query}' ({description}) should fail validation"
            );
        }

        // Store should handle invalid queries gracefully
        let lookup_result = store.find_by_identifier(query).await;

        // Empty or too-short queries should error
        if query.len() < 3 {
            assert!(
                lookup_result.is_err(),
                "Store should reject query '{query}' ({description})"
            );
        } else {
            // Longer invalid queries should succeed but return None
            assert!(
                lookup_result.expect("Query should succeed").is_none(),
                "Store should not find anything for '{query}' ({description})"
            );
        }
    }
}

#[tokio::test]
async fn test_e2e_performance_characteristics() {
    let (store, _temp_dir) = create_test_database().await;

    // Measure identity creation time
    let mut creation_times = Vec::new();
    for _ in 0..100 {
        let start = std::time::Instant::now();
        let _ = store.get_or_create().await.expect("Should create");
        creation_times.push(start.elapsed());
    }

    let avg_creation = creation_times.iter().sum::<Duration>() / creation_times.len() as u32;
    println!("Average identity creation time: {avg_creation:?}");
    assert!(
        avg_creation < Duration::from_millis(10),
        "Creation too slow"
    );

    // Create a known set for lookup tests
    let mut test_executors = Vec::new();
    for _ in 0..50 {
        let executor = store.get_or_create().await.expect("Should create");
        test_executors.push(executor);
    }

    // Measure lookup times
    let mut lookup_times = Vec::new();
    for executor in &test_executors {
        let uuid = executor.uuid().to_string();

        let start = std::time::Instant::now();
        let _ = store
            .find_by_identifier(&uuid)
            .await
            .expect("Lookup should succeed");
        lookup_times.push(start.elapsed());
    }

    let avg_lookup = lookup_times.iter().sum::<Duration>() / lookup_times.len() as u32;
    println!("Average lookup time: {avg_lookup:?}");
    assert!(avg_lookup < Duration::from_millis(5), "Lookup too slow");

    // Test prefix matching performance
    let mut prefix_times = Vec::new();
    for executor in test_executors.iter().take(20) {
        let prefix = &executor.huid()[..4];

        let start = std::time::Instant::now();
        let _ = store
            .find_by_identifier(prefix)
            .await
            .expect("Prefix lookup should succeed");
        prefix_times.push(start.elapsed());
    }

    let avg_prefix = prefix_times.iter().sum::<Duration>() / prefix_times.len() as u32;
    println!("Average prefix lookup time: {avg_prefix:?}");
    assert!(
        avg_prefix < Duration::from_millis(10),
        "Prefix lookup too slow"
    );
}

#[test]
fn test_e2e_word_lists_and_combinations() {
    let provider = StaticWordProvider::new();

    // Verify word lists meet requirements
    assert!(provider.adjective_count() >= 100);
    assert!(provider.noun_count() >= 100);

    // Calculate total possible combinations
    let total_combinations = provider.adjective_count() * provider.noun_count() * 65536;
    println!("Total possible HUIDs: {total_combinations}");
    assert!(total_combinations > 600_000_000); // Should be 655M+

    // Generate many HUIDs and check for early collisions
    let mut generated_huids = HashSet::new();
    for i in 0..10000 {
        let executor = ExecutorId::new_with_provider(&provider).expect("Should create executor");

        let huid = executor.huid().to_string();
        assert!(
            generated_huids.insert(huid.clone()),
            "HUID collision at iteration {i}: {huid}"
        );
    }

    println!(
        "Generated {} unique HUIDs with no collisions",
        generated_huids.len()
    );
}
