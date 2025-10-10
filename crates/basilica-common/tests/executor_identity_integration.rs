//! End-to-end integration tests for the node identity module
//!
//! These tests verify the complete functionality of the UUID system
//! across all components in realistic usage scenarios.

#![cfg(feature = "sqlite")]

use basilica_common::node_identity::{
    validate_identifier, IdentityDisplay, IdentityPersistence, NodeId, NodeIdentity,
    SqliteIdentityStore,
};
use std::collections::HashSet;
use tempfile::TempDir;
use tokio::time::Duration;

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
async fn test_e2e_node_lifecycle() {
    let (store, _temp_dir) = create_test_database().await;

    // Step 1: Create a new node identity
    let node = store.get_or_create().await.expect("Should create new node");

    println!("Created node: {}", node.full_display());

    // Verify the identity format
    assert_eq!(node.uuid().get_version(), Some(uuid::Version::Random));

    // Step 2: Look up by UUID
    let found_by_uuid = store
        .find_by_identifier(&node.uuid().to_string())
        .await
        .expect("Lookup should succeed")
        .expect("Should find node");

    assert_eq!(found_by_uuid.uuid(), node.uuid());
    assert_eq!(found_by_uuid.created_at(), node.created_at());

    // Step 3: Test display formatting
    use basilica_common::node_identity::NodeIdentityDisplay;
    let display = NodeIdentityDisplay::new(&*node);

    let compact = display.format_compact();
    assert!(compact.len() == 8); // Short UUID

    let verbose = display.format_verbose();
    assert!(verbose.contains(&node.uuid().to_string()));

    let json = display.format_json().expect("JSON formatting should work");
    let parsed: serde_json::Value =
        serde_json::from_str(&json).expect("Should parse as valid JSON");
    assert_eq!(parsed["uuid"], node.uuid().to_string());
}

#[tokio::test]
async fn test_e2e_uuid_uniqueness() {
    // Test UUID uniqueness by creating many nodes with different seeds
    let mut nodes = Vec::new();
    let mut uuids = HashSet::new();

    // Create many node IDs directly to test UUID uniqueness
    for i in 0..100 {
        let node = NodeId::new(&format!("test-seed-{}", i)).expect("Should create node");

        // Verify UUID uniqueness
        assert!(
            uuids.insert(*node.uuid()),
            "UUID collision detected at iteration {}: {}",
            i,
            node.uuid()
        );

        nodes.push(node);
    }

    println!(
        "Created {} unique nodes with no UUID collisions",
        nodes.len()
    );

    // Test that the store persists nodes properly
    let (store, _temp_dir) = create_test_database().await;

    // Get the single node identity for this store
    let store_node = store.get_or_create().await.expect("Should create node");

    // Verify it's retrievable
    let found = store
        .find_by_identifier(&store_node.uuid().to_string())
        .await
        .expect("Lookup should succeed")
        .expect("Should find node");

    assert_eq!(found.uuid(), store_node.uuid());
}

#[tokio::test]
async fn test_e2e_persistence_and_caching() {
    let (store, _temp_dir) = create_test_database().await;

    // Create a node
    let node = store.get_or_create().await.expect("Should create node");

    let uuid = node.uuid().to_string();

    // Multiple rapid lookups should use cache
    let start = std::time::Instant::now();
    for _ in 0..100 {
        let found = store
            .find_by_identifier(&uuid)
            .await
            .expect("Lookup should succeed")
            .expect("Should find node");

        assert_eq!(found.uuid().to_string(), uuid);
    }
    let elapsed = start.elapsed();

    println!("100 lookups took {elapsed:?} (should be fast due to caching)");
    assert!(elapsed < Duration::from_millis(100), "Lookups too slow");

    // Verify data persists across store instances
    drop(store);

    // Create new store with same database
    let db_url = format!("sqlite:///{}", _temp_dir.path().join("test.db").display());
    let new_store = SqliteIdentityStore::new(&db_url)
        .await
        .expect("Should create new store");

    // Should find the same node
    let found = new_store
        .find_by_identifier(&uuid)
        .await
        .expect("Lookup should succeed")
        .expect("Should find node");

    assert_eq!(found.uuid().to_string(), uuid);
}

#[tokio::test]
async fn test_e2e_get_or_create_idempotency() {
    let (store, _temp_dir) = create_test_database().await;

    // Call get_or_create multiple times
    let node1 = store.get_or_create().await.expect("Should create node");
    let node2 = store.get_or_create().await.expect("Should get same node");
    let node3 = store.get_or_create().await.expect("Should get same node");

    // All should be the same node
    assert_eq!(node1.uuid(), node2.uuid());
    assert_eq!(node2.uuid(), node3.uuid());
    assert_eq!(node1.created_at(), node2.created_at());
}

#[tokio::test]
async fn test_e2e_deterministic_generation() {
    // Same seed should generate same UUID
    let seed = "test-seed-123";

    let node1 = NodeId::new(seed).expect("Should create node");
    let node2 = NodeId::new(seed).expect("Should create node");

    assert_eq!(node1.uuid(), node2.uuid());
    assert_eq!(node1.created_at(), node2.created_at());
}

#[tokio::test]
async fn test_e2e_validation() {
    // Test valid UUID
    let uuid = uuid::Uuid::new_v4().to_string();
    assert!(validate_identifier(&uuid).is_ok());

    // Test invalid identifiers
    assert!(validate_identifier("").is_err());
    assert!(validate_identifier("ab").is_err()); // Too short
    assert!(validate_identifier("xyz").is_ok()); // Valid 3+ character string
}

#[tokio::test]
async fn test_e2e_cache_stats() {
    let (store, _temp_dir) = create_test_database().await;

    // Initially empty
    let stats = store.cache_stats().await;
    assert_eq!(stats.uuid_entries, 0);

    // Create a node
    let node = store.get_or_create().await.expect("Should create node");

    // Cache should have one entry
    let stats = store.cache_stats().await;
    assert_eq!(stats.uuid_entries, 1);

    // Clear cache
    store.clear_cache().await;
    let stats = store.cache_stats().await;
    assert_eq!(stats.uuid_entries, 0);

    // Finding should repopulate cache
    let _ = store
        .find_by_identifier(&node.uuid().to_string())
        .await
        .expect("Should find");
    let stats = store.cache_stats().await;
    assert_eq!(stats.uuid_entries, 1);
}
