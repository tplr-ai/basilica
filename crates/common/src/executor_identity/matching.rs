//! Matching algorithms for executor identity lookups
//!
//! This module provides functions to match executor identities by UUID or HUID,
//! supporting both exact matches and prefix-based searches.

use crate::executor_identity::{
    constants::MIN_HUID_PREFIX_LENGTH,
    validation::{validate_identifier, IdentifierType},
    ExecutorIdentity,
};
use uuid::Uuid;

/// Result of a matching operation
///
/// Note: This contains references to avoid cloning trait objects
pub struct MatchResult<'a> {
    /// The matched executor identity
    pub executor: &'a dyn ExecutorIdentity,
    /// The type of match that occurred
    pub match_type: MatchType,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
}

/// Types of matches that can occur
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MatchType {
    /// Exact UUID match
    ExactUuid,
    /// UUID prefix match
    UuidPrefix,
    /// Exact HUID match
    ExactHuid,
    /// HUID prefix match
    HuidPrefix,
}

impl MatchType {
    /// Returns true if this is an exact match (not a prefix match)
    pub fn is_exact(&self) -> bool {
        matches!(self, MatchType::ExactUuid | MatchType::ExactHuid)
    }

    /// Returns true if this is a prefix match
    pub fn is_prefix(&self) -> bool {
        matches!(self, MatchType::UuidPrefix | MatchType::HuidPrefix)
    }
}

/// Matches a single executor identity against a query
///
/// # Arguments
/// * `executor` - The executor identity to test
/// * `query` - The search query (UUID, HUID, or prefix)
///
/// # Returns
/// * `Some(MatchResult)` if the executor matches the query
/// * `None` if the executor does not match
pub fn match_executor<'a>(
    executor: &'a dyn ExecutorIdentity,
    query: &str,
) -> Option<MatchResult<'a>> {
    // Validate the query
    let identifier_type = match validate_identifier(query) {
        Ok(id_type) => id_type,
        Err(_) => return None,
    };

    match identifier_type {
        IdentifierType::FullUuid(query_uuid) => {
            // Exact UUID match
            if executor.uuid() == &query_uuid {
                Some(MatchResult {
                    executor,
                    match_type: MatchType::ExactUuid,
                    confidence: 1.0,
                })
            } else {
                None
            }
        }
        IdentifierType::UuidPrefix(prefix) => {
            // UUID prefix match
            let uuid_str = executor.uuid().to_string();
            if uuid_str.starts_with(&prefix) {
                // Calculate confidence based on prefix length vs full UUID length
                let confidence = prefix.len() as f32 / 36.0; // UUID string is 36 chars
                Some(MatchResult {
                    executor,
                    match_type: MatchType::UuidPrefix,
                    confidence,
                })
            } else {
                None
            }
        }
        IdentifierType::FullHuid(huid) => {
            // Exact HUID match
            if executor.huid() == huid {
                Some(MatchResult {
                    executor,
                    match_type: MatchType::ExactHuid,
                    confidence: 1.0,
                })
            } else {
                None
            }
        }
        IdentifierType::HuidPrefix(prefix) => {
            // HUID prefix match
            if executor.huid().starts_with(&prefix) {
                // Calculate confidence based on prefix length vs HUID length
                let confidence = prefix.len() as f32 / executor.huid().len() as f32;
                Some(MatchResult {
                    executor,
                    match_type: MatchType::HuidPrefix,
                    confidence,
                })
            } else {
                None
            }
        }
    }
}

/// Matches multiple executors against a query
///
/// # Arguments
/// * `executors` - Iterator of executor identities to search
/// * `query` - The search query
///
/// # Returns
/// A vector of match results, sorted by confidence (highest first)
pub fn match_executors<'a, I>(executors: I, query: &str) -> Vec<MatchResult<'a>>
where
    I: Iterator<Item = &'a dyn ExecutorIdentity>,
{
    let mut results: Vec<MatchResult<'a>> = executors
        .filter_map(|executor| match_executor(executor, query))
        .collect();

    // Sort by confidence (highest first), then by match type (exact before prefix)
    results.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(
                || match (a.match_type.is_exact(), b.match_type.is_exact()) {
                    (true, false) => std::cmp::Ordering::Less,
                    (false, true) => std::cmp::Ordering::Greater,
                    _ => std::cmp::Ordering::Equal,
                },
            )
    });

    results
}

/// Finds the best match for a query among multiple executors
///
/// Returns the match with the highest confidence, preferring exact matches
pub fn find_best_match<'a, I>(executors: I, query: &str) -> Option<MatchResult<'a>>
where
    I: Iterator<Item = &'a dyn ExecutorIdentity>,
{
    match_executors(executors, query).into_iter().next()
}

/// Counts how many executors match a given prefix
///
/// Useful for determining if a prefix is ambiguous
pub fn count_prefix_matches<'a, I>(executors: I, prefix: &str) -> usize
where
    I: Iterator<Item = &'a dyn ExecutorIdentity>,
{
    executors
        .filter(|executor| executor.matches(prefix))
        .count()
}

/// Suggests a minimum unambiguous prefix for an executor
///
/// Given an executor and a list of other executors, finds the shortest
/// prefix that uniquely identifies this executor
pub fn suggest_unambiguous_prefix<'a, I>(target: &dyn ExecutorIdentity, others: I) -> String
where
    I: Iterator<Item = &'a dyn ExecutorIdentity>,
{
    let target_huid = target.huid();
    let target_uuid = target.uuid().to_string();

    // Collect all other HUIDs and UUIDs
    let other_identifiers: Vec<(String, String)> = others
        .filter(|e| e.uuid() != target.uuid())
        .map(|e| (e.huid().to_string(), e.uuid().to_string()))
        .collect();

    // Try progressively longer HUID prefixes
    for len in MIN_HUID_PREFIX_LENGTH..=target_huid.len() {
        let prefix = &target_huid[..len];
        let is_ambiguous = other_identifiers
            .iter()
            .any(|(huid, _)| huid.starts_with(prefix));

        if !is_ambiguous {
            return prefix.to_string();
        }
    }

    // If HUID is not unique, try UUID prefix
    for len in MIN_HUID_PREFIX_LENGTH..=8 {
        let prefix = &target_uuid[..len];
        let is_ambiguous = other_identifiers
            .iter()
            .any(|(_, uuid)| uuid.starts_with(prefix));

        if !is_ambiguous {
            return prefix.to_string();
        }
    }

    // Fallback to full UUID if nothing else works
    target_uuid
}

// Helper struct for testing - implements ExecutorIdentity
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct MockExecutor {
    uuid: Uuid,
    huid: String,
    created_at: std::time::SystemTime,
}

impl ExecutorIdentity for MockExecutor {
    fn uuid(&self) -> &Uuid {
        &self.uuid
    }

    fn huid(&self) -> &str {
        &self.huid
    }

    fn created_at(&self) -> std::time::SystemTime {
        self.created_at
    }

    fn matches(&self, query: &str) -> bool {
        if query.len() < MIN_HUID_PREFIX_LENGTH {
            return false;
        }
        self.uuid.to_string().starts_with(query) || self.huid.starts_with(query)
    }

    fn full_display(&self) -> String {
        format!("{} ({})", self.huid, self.uuid)
    }

    fn short_uuid(&self) -> String {
        self.uuid.to_string()[..8].to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::SystemTime;

    fn create_mock_executor(uuid: Uuid, huid: &str) -> MockExecutor {
        MockExecutor {
            uuid,
            huid: huid.to_string(),
            created_at: SystemTime::now(),
        }
    }

    #[test]
    fn test_match_executor_exact_uuid() {
        let uuid = Uuid::new_v4();
        let executor = create_mock_executor(uuid, "swift-falcon-a3f2");

        let result = match_executor(&executor, &uuid.to_string()).unwrap();
        assert_eq!(result.match_type, MatchType::ExactUuid);
        assert_eq!(result.confidence, 1.0);
    }

    #[test]
    fn test_match_executor_uuid_prefix() {
        let uuid = Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap();
        let executor = create_mock_executor(uuid, "swift-falcon-a3f2");

        let result = match_executor(&executor, "550e8400").unwrap();
        assert_eq!(result.match_type, MatchType::UuidPrefix);
        assert!(result.confidence > 0.0 && result.confidence < 1.0);
    }

    #[test]
    fn test_match_executor_exact_huid() {
        let executor = create_mock_executor(Uuid::new_v4(), "swift-falcon-a3f2");

        let result = match_executor(&executor, "swift-falcon-a3f2").unwrap();
        assert_eq!(result.match_type, MatchType::ExactHuid);
        assert_eq!(result.confidence, 1.0);
    }

    #[test]
    fn test_match_executor_huid_prefix() {
        let executor = create_mock_executor(Uuid::new_v4(), "swift-falcon-a3f2");

        let result = match_executor(&executor, "swift").unwrap();
        assert_eq!(result.match_type, MatchType::HuidPrefix);
        assert!(result.confidence > 0.0 && result.confidence < 1.0);

        let result2 = match_executor(&executor, "swift-falcon").unwrap();
        assert!(result2.confidence > result.confidence);
    }

    #[test]
    fn test_match_executor_no_match() {
        let executor = create_mock_executor(Uuid::new_v4(), "swift-falcon-a3f2");

        assert!(match_executor(&executor, "brave").is_none());
        assert!(match_executor(&executor, "12345678").is_none());
    }

    #[test]
    fn test_match_executors_multiple() {
        let executors = [
            create_mock_executor(Uuid::new_v4(), "swift-falcon-a3f2"),
            create_mock_executor(Uuid::new_v4(), "swift-eagle-b4c5"),
            create_mock_executor(Uuid::new_v4(), "brave-lion-d6e7"),
        ];

        let executor_refs: Vec<&dyn ExecutorIdentity> = executors
            .iter()
            .map(|e| e as &dyn ExecutorIdentity)
            .collect();

        let results = match_executors(executor_refs.into_iter(), "swift");
        assert_eq!(results.len(), 2);
        assert!(results
            .iter()
            .all(|r| r.match_type == MatchType::HuidPrefix));
    }

    #[test]
    fn test_find_best_match() {
        let uuid1 = Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap();
        let executors = [
            create_mock_executor(uuid1, "swift-falcon-a3f2"),
            create_mock_executor(Uuid::new_v4(), "swift-eagle-b4c5"),
            create_mock_executor(Uuid::new_v4(), "brave-lion-d6e7"),
        ];

        let executor_refs: Vec<&dyn ExecutorIdentity> = executors
            .iter()
            .map(|e| e as &dyn ExecutorIdentity)
            .collect();

        // Exact UUID match should win
        let best = find_best_match(executor_refs.iter().copied(), &uuid1.to_string()).unwrap();
        assert_eq!(best.match_type, MatchType::ExactUuid);

        // Exact HUID match should win
        let best = find_best_match(executor_refs.iter().copied(), "brave-lion-d6e7").unwrap();
        assert_eq!(best.match_type, MatchType::ExactHuid);

        // Longer prefix should have higher confidence
        let swift_results = match_executors(executor_refs.iter().copied(), "swift");
        assert_eq!(swift_results.len(), 2);
    }

    #[test]
    fn test_count_prefix_matches() {
        let executors = [
            create_mock_executor(Uuid::new_v4(), "swift-falcon-a3f2"),
            create_mock_executor(Uuid::new_v4(), "swift-eagle-b4c5"),
            create_mock_executor(Uuid::new_v4(), "brave-lion-d6e7"),
        ];

        let executor_refs: Vec<&dyn ExecutorIdentity> = executors
            .iter()
            .map(|e| e as &dyn ExecutorIdentity)
            .collect();

        assert_eq!(
            count_prefix_matches(executor_refs.iter().copied(), "swift"),
            2
        );
        assert_eq!(
            count_prefix_matches(executor_refs.iter().copied(), "brave"),
            1
        );
        assert_eq!(
            count_prefix_matches(executor_refs.iter().copied(), "unknown"),
            0
        );
    }

    #[test]
    fn test_suggest_unambiguous_prefix() {
        let target = create_mock_executor(Uuid::new_v4(), "swift-falcon-a3f2");
        let others = [
            create_mock_executor(Uuid::new_v4(), "swift-eagle-b4c5"),
            create_mock_executor(Uuid::new_v4(), "brave-lion-d6e7"),
        ];

        let other_refs: Vec<&dyn ExecutorIdentity> =
            others.iter().map(|e| e as &dyn ExecutorIdentity).collect();

        let prefix = suggest_unambiguous_prefix(&target, other_refs.into_iter());
        assert!(prefix.starts_with("swift-f"));
        assert!(target.huid().starts_with(&prefix));
    }

    #[test]
    fn test_match_type_methods() {
        assert!(MatchType::ExactUuid.is_exact());
        assert!(MatchType::ExactHuid.is_exact());
        assert!(!MatchType::UuidPrefix.is_exact());
        assert!(!MatchType::HuidPrefix.is_exact());

        assert!(!MatchType::ExactUuid.is_prefix());
        assert!(!MatchType::ExactHuid.is_prefix());
        assert!(MatchType::UuidPrefix.is_prefix());
        assert!(MatchType::HuidPrefix.is_prefix());
    }

    #[test]
    fn test_confidence_calculation() {
        let executor = create_mock_executor(Uuid::new_v4(), "swift-falcon-a3f2");

        // UUID prefix confidence
        let uuid_str = executor.uuid().to_string();
        let result = match_executor(&executor, &uuid_str[..8]).unwrap();
        assert_eq!(result.confidence, 8.0 / 36.0);

        // HUID prefix confidence
        let result = match_executor(&executor, "swift").unwrap();
        assert_eq!(result.confidence, 5.0 / executor.huid().len() as f32);
    }
}
