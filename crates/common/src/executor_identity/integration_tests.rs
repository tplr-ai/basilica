//! Comprehensive integration tests for the executor identity system
//!
//! This module tests end-to-end scenarios from uid.md including:
//! - Complete identity lifecycle
//! - Cross-component integration
//! - Performance benchmarks
//! - Edge cases and error conditions

#[cfg(test)]
mod tests {
    use super::super::*;
    use anyhow::Result;
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;
    use std::time::{Duration, Instant};
    use uuid::Uuid;

    // Helper to create a test identity store
    #[cfg(feature = "sqlite")]
    async fn create_test_store() -> Result<SqliteIdentityStore> {
        SqliteIdentityStore::new("sqlite::memory:").await
    }

    // Test scenario from uid.md: Basic identity creation and lookup
    #[tokio::test]
    #[cfg(feature = "sqlite")]
    async fn test_end_to_end_identity_lifecycle() {
        let store = create_test_store().await.expect("Should create store");

        // Step 1: Create an identity
        let identity1 = store.get_or_create().await.expect("Should create identity");
        println!("Created identity: {}", identity1.full_display());

        // Verify format
        assert!(is_valid_huid(identity1.huid()));
        assert_eq!(identity1.uuid().get_version(), Some(uuid::Version::Random));

        // Step 2: Retrieve same identity (should be cached)
        let identity2 = store.get_or_create().await.expect("Should get existing");
        assert_eq!(identity1.uuid(), identity2.uuid());
        assert_eq!(identity1.huid(), identity2.huid());

        // Step 3: Find by UUID
        let found = store
            .find_by_identifier(&identity1.uuid().to_string())
            .await
            .expect("Should find")
            .expect("Should exist");
        assert_eq!(found.uuid(), identity1.uuid());

        // Step 4: Find by HUID prefix
        let prefix = &identity1.huid()[..5];
        let found = store
            .find_by_identifier(prefix)
            .await
            .expect("Should find")
            .expect("Should exist");
        assert_eq!(found.uuid(), identity1.uuid());

        // Step 5: Find by full HUID
        let found = store
            .find_by_identifier(identity1.huid())
            .await
            .expect("Should find")
            .expect("Should exist");
        assert_eq!(found.uuid(), identity1.uuid());
    }

    // Test scenario: Multiple executor handling
    #[tokio::test]
    async fn test_multiple_executor_management() {
        let executors = (0..10)
            .map(|_| ExecutorId::new().expect("Should create"))
            .collect::<Vec<_>>();

        // Test uniqueness
        let mut uuids = HashSet::new();
        let mut huids = HashSet::new();

        for executor in &executors {
            assert!(uuids.insert(*executor.uuid()), "UUID collision detected");
            assert!(
                huids.insert(executor.huid().to_string()),
                "HUID collision detected"
            );
        }

        // Test matching with prefixes
        for executor in &executors {
            // Each should match its own prefix
            let prefix = &executor.huid()[..4];
            assert!(executor.matches(prefix));

            // UUID prefix matching
            let uuid_prefix = &executor.uuid().to_string()[..8];
            assert!(executor.matches(uuid_prefix));
        }
    }

    // Test scenario: Collision detection and handling
    #[tokio::test]
    async fn test_huid_collision_simulation() {
        // Create many executors to increase collision probability
        let mut executors = Vec::new();
        let mut huid_counts: HashMap<String, usize> = HashMap::new();

        // Generate 1000 executors
        for _ in 0..1000 {
            let executor = ExecutorId::new().expect("Should create");

            // Track first 2 components of HUID (adjective-noun)
            let parts: Vec<&str> = executor.huid().split('-').collect();
            if parts.len() >= 2 {
                let prefix = format!("{}-{}", parts[0], parts[1]);
                *huid_counts.entry(prefix).or_insert(0) += 1;
            }

            executors.push(executor);
        }

        // Check collision statistics
        let max_duplicates = huid_counts.values().max().copied().unwrap_or(0);
        let duplicate_prefixes = huid_counts.values().filter(|&&c| c > 1).count();

        println!("Generated {} executors", executors.len());
        println!("Max duplicates for any adjective-noun pair: {max_duplicates}");
        println!("Number of duplicate adjective-noun pairs: {duplicate_prefixes}");

        // Even with duplicates in adjective-noun, full HUIDs should be unique
        let unique_huids: HashSet<_> = executors.iter().map(|e| e.huid()).collect();
        assert_eq!(
            unique_huids.len(),
            executors.len(),
            "HUID collision detected!"
        );
    }

    // Test scenario: Performance benchmark for 1000+ executors
    #[tokio::test]
    async fn test_performance_benchmark() {
        let start = Instant::now();

        // Create 1000 executors
        let mut creation_times = Vec::new();
        let executors: Vec<ExecutorId> = (0..1000)
            .map(|_| {
                let create_start = Instant::now();
                let executor = ExecutorId::new().expect("Should create");
                creation_times.push(create_start.elapsed());
                executor
            })
            .collect();

        let total_creation_time = start.elapsed();

        // Calculate statistics
        let avg_creation_time =
            creation_times.iter().sum::<Duration>() / creation_times.len() as u32;
        let max_creation_time = creation_times
            .iter()
            .max()
            .copied()
            .unwrap_or(Duration::ZERO);

        println!(
            "Created {} executors in {:?}",
            executors.len(),
            total_creation_time
        );
        println!("Average creation time: {avg_creation_time:?}");
        println!("Max creation time: {max_creation_time:?}");

        // Performance assertions
        assert!(
            avg_creation_time < Duration::from_millis(1),
            "Creation too slow"
        );
        assert!(
            total_creation_time < Duration::from_secs(1),
            "Total time too slow"
        );

        // Test matching performance
        let match_start = Instant::now();
        let test_queries = vec![
            executors[0].huid()[..5].to_string(),
            executors[500].uuid().to_string()[..8].to_string(),
            executors[999].huid().to_string(),
        ];

        for query in &test_queries {
            let matches: Vec<_> = executors.iter().filter(|e| e.matches(query)).collect();
            assert!(!matches.is_empty());
        }

        let match_time = match_start.elapsed();
        println!(
            "Matching {} queries against {} executors took {:?}",
            test_queries.len(),
            executors.len(),
            match_time
        );
        assert!(match_time < Duration::from_millis(10), "Matching too slow");
    }

    // Test scenario: Full workflow from uid.md examples
    #[tokio::test]
    #[cfg(feature = "sqlite")]
    async fn test_uid_md_example_workflow() {
        let store = create_test_store().await.expect("Should create store");

        // Example 1: Creating and using executor identities
        let executor = store.get_or_create().await.expect("Should create");
        println!("New executor: {}", executor.huid());
        println!("Full details: {}", executor.full_display());

        // Example 2: Looking up by HUID prefix
        let prefix = "swift"; // Assuming HUID starts with "swift"
        if executor.huid().starts_with(prefix) {
            let found = store
                .find_by_identifier(prefix)
                .await
                .expect("Should complete search");
            assert!(found.is_some());
        }

        // Example 3: Display formatting
        let display = ExecutorIdentityDisplay::new(&*executor);
        let _compact = display.format_compact();
        let _verbose = display.format_verbose();
        let json = display.format_json().expect("Should format JSON");

        // Verify JSON format
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("Should parse JSON");
        assert!(parsed["uuid"].is_string());
        assert!(parsed["huid"].is_string());
        assert!(parsed["created_at"].is_number());
    }

    // Test edge cases and error conditions
    #[tokio::test]
    async fn test_edge_cases_and_errors() {
        // Test 1: Invalid HUID format validation
        let invalid_huids = vec![
            "",                   // Empty
            "invalid",            // No structure
            "Swift-Falcon-A3F2",  // Capital letters
            "swift_falcon_a3f2",  // Underscores
            "swift-falcon",       // Missing hex
            "swift-falcon-g3f2",  // Invalid hex
            "swift-falcon-a3f21", // Too long hex
            "swift-falcon-a3f",   // Too short hex
        ];

        for invalid in &invalid_huids {
            assert!(!is_valid_huid(invalid), "{invalid} should be invalid");

            // Try to create from parts
            let result = ExecutorId::from_parts(
                Uuid::new_v4(),
                invalid.to_string(),
                std::time::SystemTime::now(),
            );
            assert!(result.is_err(), "{invalid} should fail validation");
        }

        // Test 2: Query validation
        let validation_tests = vec![
            ("", false),            // Empty query
            ("ab", false),          // Too short
            ("abc", true),          // Minimum length
            ("swift-falcon", true), // Valid HUID prefix
            ("123e4567", true),     // Valid UUID prefix
        ];

        for (query, should_be_valid) in validation_tests {
            let result = validate_identifier(query);
            assert_eq!(
                result.is_ok(),
                should_be_valid,
                "Query '{query}' validation mismatch"
            );
        }

        // Test 3: UUID parsing errors
        let invalid_uuids = vec![
            "not-a-uuid",
            "12345",
            "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
        ];

        for invalid in invalid_uuids {
            let result = ExecutorId::from_uuid_string(invalid);
            assert!(result.is_err(), "{invalid} should fail UUID parsing");
        }
    }

    // Test concurrent operations
    #[tokio::test]
    #[cfg(feature = "sqlite")]
    async fn test_concurrent_operations() {
        let store = Arc::new(create_test_store().await.expect("Should create store"));

        // Create initial identity
        let initial = store.get_or_create().await.expect("Should create");

        // Spawn multiple concurrent tasks
        let mut handles = Vec::new();

        // Task 1: Repeated lookups
        let store1 = store.clone();
        let uuid = initial.uuid().to_string();
        handles.push(tokio::spawn(async move {
            for _ in 0..100 {
                let _found = store1.find_by_identifier(&uuid).await;
            }
        }));

        // Task 2: Cache operations
        let store2 = store.clone();
        handles.push(tokio::spawn(async move {
            for i in 0..50 {
                if i % 10 == 0 {
                    store2.clear_cache().await;
                }
                let _stats = store2.cache_stats().await;
            }
        }));

        // Task 3: New identity creation attempts
        let store3 = store.clone();
        handles.push(tokio::spawn(async move {
            for _ in 0..50 {
                let _existing = store3.get_or_create().await;
            }
        }));

        // Wait for all tasks
        for handle in handles {
            handle.await.expect("Task should complete");
        }

        // Verify consistency
        let final_identity = store.get_or_create().await.expect("Should get");
        assert_eq!(final_identity.uuid(), initial.uuid());
    }

    // Test migration scenarios
    #[tokio::test]
    #[cfg(feature = "sqlite")]
    async fn test_legacy_migration_workflow() {
        let store = create_test_store().await.expect("Should create store");

        // Simulate legacy IDs
        let legacy_ids = vec!["old-executor-1", "legacy-node-abc123", "worker-id-xyz789"];

        let mut migrated_map = HashMap::new();

        // Migrate each legacy ID
        for legacy_id in &legacy_ids {
            let migrated = store
                .migrate_legacy_id(legacy_id)
                .await
                .expect("Should migrate");

            println!("Migrated {} to {}", legacy_id, migrated.full_display());
            migrated_map.insert(legacy_id.to_string(), migrated);
        }

        // Verify idempotency - migrating again returns same identity
        for legacy_id in &legacy_ids {
            let migrated_again = store
                .migrate_legacy_id(legacy_id)
                .await
                .expect("Should return existing");

            let original = &migrated_map[*legacy_id];
            assert_eq!(migrated_again.uuid(), original.uuid());
            assert_eq!(migrated_again.huid(), original.huid());
        }

        // Verify mappings are persisted
        let all_mappings = store
            .get_legacy_mappings()
            .await
            .expect("Should get mappings");
        assert_eq!(all_mappings.len(), legacy_ids.len());

        for legacy_id in &legacy_ids {
            assert!(all_mappings.contains_key(*legacy_id));
        }
    }

    // Test word provider edge cases
    #[test]
    fn test_word_provider_boundaries() {
        let provider = StaticWordProvider::new();

        // Test boundary conditions
        let adj_count = provider.adjective_count();
        let noun_count = provider.noun_count();

        // Valid boundaries
        assert!(provider.get_adjective(0).is_some());
        assert!(provider.get_adjective(adj_count - 1).is_some());
        assert!(provider.get_noun(0).is_some());
        assert!(provider.get_noun(noun_count - 1).is_some());

        // Invalid boundaries
        assert!(provider.get_adjective(adj_count).is_none());
        assert!(provider.get_noun(noun_count).is_none());
        assert!(provider.get_adjective(usize::MAX).is_none());
        assert!(provider.get_noun(usize::MAX).is_none());

        // Verify minimum counts
        assert!(adj_count >= MIN_ADJECTIVE_COUNT);
        assert!(noun_count >= MIN_NOUN_COUNT);
    }

    // Test display formatting edge cases
    #[test]
    fn test_display_formatting_edge_cases() {
        // Create executor with known values
        let uuid = Uuid::nil();
        let huid = "test-case-0000".to_string();
        let created_at = std::time::UNIX_EPOCH;

        let executor =
            ExecutorId::from_parts(uuid, huid.clone(), created_at).expect("Should create");

        // Test all display formats
        let display = executor.display();

        let compact = display.format_compact();
        assert_eq!(compact, "test-case-0000");

        let verbose = display.format_verbose();
        assert!(verbose.contains("HUID: test-case-0000"));
        assert!(verbose.contains("UUID: 00000000-0000-0000-0000-000000000000"));

        let json = display.format_json().expect("Should format");
        assert!(json.contains("\"huid\": \"test-case-0000\""));
        assert!(json.contains("\"created_at\": 0"));
    }

    // Load test for HUID generation performance
    #[test]
    fn test_huid_generation_load() {
        let provider = StaticWordProvider::new();
        let mut generation_times = Vec::new();

        // Generate many HUIDs
        for _ in 0..10000 {
            let start = Instant::now();
            let _executor = ExecutorId::new_with_provider(&provider).expect("Should create");
            generation_times.push(start.elapsed());
        }

        // Analyze performance
        let total_time: Duration = generation_times.iter().sum();
        let avg_time = total_time / generation_times.len() as u32;
        let max_time = generation_times
            .iter()
            .max()
            .copied()
            .unwrap_or(Duration::ZERO);

        println!("HUID generation performance:");
        println!(
            "  Total: {:?} for {} generations",
            total_time,
            generation_times.len()
        );
        println!("  Average: {avg_time:?}");
        println!("  Max: {max_time:?}");

        // Performance assertions
        assert!(
            avg_time < Duration::from_micros(100),
            "Average generation too slow"
        );
        assert!(
            max_time < Duration::from_millis(1),
            "Max generation too slow"
        );
    }

    // Test matching algorithm edge cases
    #[test]
    fn test_matching_algorithm_edge_cases() {
        let executors = [
            create_test_executor("swift-falcon-a3f2"),
            create_test_executor("swift-eagle-a3f3"),
            create_test_executor("brave-falcon-b4c5"),
        ];

        let executor_refs: Vec<&dyn ExecutorIdentity> = executors
            .iter()
            .map(|e| e as &dyn ExecutorIdentity)
            .collect();

        // Test ambiguous prefixes
        let swift_matches = match_executors(executor_refs.iter().copied(), "swift");
        assert_eq!(swift_matches.len(), 2);

        // Test unique prefix suggestion
        let target = &executors[0];
        let others = vec![&executors[1] as &dyn ExecutorIdentity, &executors[2]];
        let unambiguous = suggest_unambiguous_prefix(target, others.into_iter());

        // Should suggest "swift-f" to distinguish from "swift-eagle"
        assert!(unambiguous.starts_with("swift-f"));
        assert!(target.huid().starts_with(&unambiguous));

        // Verify it's actually unambiguous
        let matches_with_suggestion = executors
            .iter()
            .filter(|e| e.huid().starts_with(&unambiguous))
            .count();
        assert_eq!(matches_with_suggestion, 1);
    }

    // Helper to create test executor with specific HUID
    fn create_test_executor(huid: &str) -> ExecutorId {
        ExecutorId::from_parts(
            Uuid::new_v4(),
            huid.to_string(),
            std::time::SystemTime::now(),
        )
        .expect("Should create")
    }
}
