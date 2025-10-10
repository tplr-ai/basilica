//! Matching algorithms for node identity lookups
//!
//! This module provides functions to match node identities by UUID or HUID,
//! supporting both exact matches and prefix-based searches.

use crate::node_identity::{
    validation::{validate_identifier, IdentifierType},
    NodeIdentity,
};
use uuid::Uuid;

/// Result of a matching operation
///
/// Note: This contains references to avoid cloning trait objects
pub struct MatchResult<'a> {
    /// The matched node identity
    pub node: &'a dyn NodeIdentity,
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

/// Matches a single node identity against a query
///
/// # Arguments
/// * `node` - The node identity to test
/// * `query` - The search query (UUID, HUID, or prefix)
///
/// # Returns
/// * `Some(MatchResult)` if the node matches the query
/// * `None` if the node does not match
pub fn match_node<'a>(node: &'a dyn NodeIdentity, query: &str) -> Option<MatchResult<'a>> {
    // Validate the query
    let identifier_type = match validate_identifier(query) {
        Ok(id_type) => id_type,
        Err(_) => return None,
    };

    match identifier_type {
        IdentifierType::FullUuid(query_uuid) => {
            // Exact UUID match
            if node.uuid() == &query_uuid {
                Some(MatchResult {
                    node,
                    match_type: MatchType::ExactUuid,
                    confidence: 1.0,
                })
            } else {
                None
            }
        }
        IdentifierType::UuidPrefix(prefix) => {
            // UUID prefix match
            let uuid_str = node.uuid().to_string();
            if uuid_str.starts_with(&prefix) {
                // Calculate confidence based on prefix length vs full UUID length
                let confidence = prefix.len() as f32 / 36.0; // UUID string is 36 chars
                Some(MatchResult {
                    node,
                    match_type: MatchType::UuidPrefix,
                    confidence,
                })
            } else {
                None
            }
        }
        IdentifierType::FullHuid(_) | IdentifierType::HuidPrefix(_) => {
            // HUID support removed, these cases should not occur
            None
        }
    }
}

/// Matches multiple nodes against a query
///
/// # Arguments
/// * `nodes` - Iterator of node identities to search
/// * `query` - The search query
///
/// # Returns
/// A vector of match results, sorted by confidence (highest first)
pub fn match_nodes<'a, I>(nodes: I, query: &str) -> Vec<MatchResult<'a>>
where
    I: Iterator<Item = &'a dyn NodeIdentity>,
{
    let mut results: Vec<MatchResult<'a>> =
        nodes.filter_map(|node| match_node(node, query)).collect();

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

/// Finds the best match for a query among multiple nodes
///
/// Returns the match with the highest confidence, preferring exact matches
pub fn find_best_match<'a, I>(nodes: I, query: &str) -> Option<MatchResult<'a>>
where
    I: Iterator<Item = &'a dyn NodeIdentity>,
{
    match_nodes(nodes, query).into_iter().next()
}

/// Counts how many nodes match a given prefix
///
/// Useful for determining if a prefix is ambiguous
pub fn count_prefix_matches<'a, I>(nodes: I, prefix: &str) -> usize
where
    I: Iterator<Item = &'a dyn NodeIdentity>,
{
    nodes.filter(|node| node.matches(prefix)).count()
}

/// Suggests a minimum unambiguous prefix for an node
///
/// Given an node and a list of other nodes, finds the shortest
/// prefix that uniquely identifies this node
pub fn suggest_unambiguous_prefix<'a, I>(target: &dyn NodeIdentity, others: I) -> String
where
    I: Iterator<Item = &'a dyn NodeIdentity>,
{
    let target_uuid = target.uuid().to_string();

    // Collect all other UUIDs
    let other_uuids: Vec<String> = others
        .filter(|e| e.uuid() != target.uuid())
        .map(|e| e.uuid().to_string())
        .collect();

    // Try progressively longer UUID prefixes (from 3 chars up to 8)
    for len in 3..=8 {
        let prefix = &target_uuid[..len];
        let is_ambiguous = other_uuids.iter().any(|uuid| uuid.starts_with(prefix));

        if !is_ambiguous {
            return prefix.to_string();
        }
    }

    // Fallback to full UUID if nothing else works
    target_uuid
}

// Helper struct for testing - implements NodeIdentity
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct MockNode {
    uuid: Uuid,
    created_at: std::time::SystemTime,
}

impl NodeIdentity for MockNode {
    fn uuid(&self) -> &Uuid {
        &self.uuid
    }

    fn created_at(&self) -> std::time::SystemTime {
        self.created_at
    }

    fn matches(&self, query: &str) -> bool {
        self.uuid.to_string().starts_with(query)
    }

    fn full_display(&self) -> String {
        self.uuid.to_string()
    }

    fn short_uuid(&self) -> String {
        self.uuid.to_string()[..8].to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::SystemTime;

    fn create_mock_node(uuid: Uuid) -> MockNode {
        MockNode {
            uuid,
            created_at: SystemTime::now(),
        }
    }

    #[test]
    fn test_match_node_exact_uuid() {
        let uuid = Uuid::new_v4();
        let node = create_mock_node(uuid);

        let result = match_node(&node, &uuid.to_string()).unwrap();
        assert_eq!(result.match_type, MatchType::ExactUuid);
        assert_eq!(result.confidence, 1.0);
    }

    #[test]
    fn test_match_node_uuid_prefix() {
        let uuid = Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap();
        let node = create_mock_node(uuid);

        let result = match_node(&node, "550e8400").unwrap();
        assert_eq!(result.match_type, MatchType::UuidPrefix);
        assert!(result.confidence > 0.0 && result.confidence < 1.0);
    }

    #[test]
    fn test_match_node_no_match() {
        let node = create_mock_node(Uuid::new_v4());

        assert!(match_node(&node, "brave").is_none());
        assert!(match_node(&node, "ffffffff").is_none());
    }

    #[test]
    fn test_match_nodes_multiple() {
        let uuid1 = Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap();
        let uuid2 = Uuid::parse_str("550e8401-e29b-41d4-a716-446655440000").unwrap();
        let uuid3 = Uuid::parse_str("660e8400-e29b-41d4-a716-446655440000").unwrap();

        let nodes = [
            create_mock_node(uuid1),
            create_mock_node(uuid2),
            create_mock_node(uuid3),
        ];

        let node_refs: Vec<&dyn NodeIdentity> =
            nodes.iter().map(|e| e as &dyn NodeIdentity).collect();

        let results = match_nodes(node_refs.into_iter(), "550e");
        assert_eq!(results.len(), 2);
        assert!(results
            .iter()
            .all(|r| r.match_type == MatchType::UuidPrefix));
    }

    #[test]
    fn test_find_best_match() {
        let uuid1 = Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap();
        let uuid2 = Uuid::parse_str("550e8401-e29b-41d4-a716-446655440000").unwrap();
        let uuid3 = Uuid::parse_str("660e8400-e29b-41d4-a716-446655440000").unwrap();

        let nodes = [
            create_mock_node(uuid1),
            create_mock_node(uuid2),
            create_mock_node(uuid3),
        ];

        let node_refs: Vec<&dyn NodeIdentity> =
            nodes.iter().map(|e| e as &dyn NodeIdentity).collect();

        // Exact UUID match should win
        let best = find_best_match(node_refs.iter().copied(), &uuid1.to_string()).unwrap();
        assert_eq!(best.match_type, MatchType::ExactUuid);

        // Longer prefix should have higher confidence
        let results = match_nodes(node_refs.iter().copied(), "550e");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_count_prefix_matches() {
        let uuid1 = Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap();
        let uuid2 = Uuid::parse_str("550e8401-e29b-41d4-a716-446655440000").unwrap();
        let uuid3 = Uuid::parse_str("660e8400-e29b-41d4-a716-446655440000").unwrap();

        let nodes = [
            create_mock_node(uuid1),
            create_mock_node(uuid2),
            create_mock_node(uuid3),
        ];

        let node_refs: Vec<&dyn NodeIdentity> =
            nodes.iter().map(|e| e as &dyn NodeIdentity).collect();

        assert_eq!(count_prefix_matches(node_refs.iter().copied(), "550e"), 2);
        assert_eq!(count_prefix_matches(node_refs.iter().copied(), "660e"), 1);
        assert_eq!(count_prefix_matches(node_refs.iter().copied(), "aaaa"), 0);
    }

    #[test]
    fn test_suggest_unambiguous_prefix() {
        let target_uuid = Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap();
        let other_uuid1 = Uuid::parse_str("550e8401-e29b-41d4-a716-446655440000").unwrap();
        let other_uuid2 = Uuid::parse_str("660e8400-e29b-41d4-a716-446655440000").unwrap();

        let target = create_mock_node(target_uuid);
        let others = [create_mock_node(other_uuid1), create_mock_node(other_uuid2)];

        let other_refs: Vec<&dyn NodeIdentity> =
            others.iter().map(|e| e as &dyn NodeIdentity).collect();

        let prefix = suggest_unambiguous_prefix(&target, other_refs.into_iter());
        assert!(target_uuid.to_string().starts_with(&prefix));
        assert!(prefix.len() >= 3 && prefix.len() <= 8);
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
        let node = create_mock_node(Uuid::new_v4());

        // UUID prefix confidence
        let uuid_str = node.uuid().to_string();
        let result = match_node(&node, &uuid_str[..8]).unwrap();
        assert_eq!(result.confidence, 8.0 / 36.0);

        // Longer prefix should have higher confidence
        let result_short = match_node(&node, &uuid_str[..4]).unwrap();
        let result_long = match_node(&node, &uuid_str[..12]).unwrap();
        assert!(result_long.confidence > result_short.confidence);
    }
}
