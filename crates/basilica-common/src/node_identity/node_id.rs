//! Core NodeId implementation with UUID + HUID support
//!
//! This module provides the main NodeId struct that combines:
//! - UUID v4 for guaranteed uniqueness
//! - HUID (Human-Unique Identifier) for user-friendly interaction
//!
//! The HUID format is: adjective-noun-4hex (e.g., "swift-falcon-a3f2")

use anyhow::Result;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::SystemTime;
use uuid::Uuid;

use crate::node_identity::NodeIdentity;

/// Main node identifier using UUID only
#[derive(Debug, Clone)]
pub struct NodeId {
    /// UUID v4 for guaranteed uniqueness
    pub uuid: Uuid,
    /// Creation timestamp
    pub created_at: SystemTime,
}

impl NodeId {
    /// Creates a new NodeId with a seeded RNG for deterministic generation
    ///
    /// # Arguments
    /// * `seed` - String seed to use for RNG generation
    pub fn new(seed: &str) -> Result<Self> {
        // Create seeded RNG from the seed string
        let mut rng = StdRng::seed_from_u64(Self::hash_seed_to_u64(seed));

        // Generate UUID using seeded RNG
        let uuid = Self::generate_uuid_with_rng(&mut rng);

        // Set nanosecond value to zero for consistent timestamps
        let now = SystemTime::now();
        let duration_since_epoch = now.duration_since(SystemTime::UNIX_EPOCH).unwrap();
        let created_at =
            SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(duration_since_epoch.as_secs());

        Ok(Self { uuid, created_at })
    }

    /// Creates a NodeId from existing UUID value
    ///
    /// This is useful when reconstructing from persistent storage
    ///
    /// # Arguments
    /// * `uuid` - The UUID to use
    /// * `created_at` - The creation timestamp
    pub fn from_parts(uuid: Uuid, created_at: SystemTime) -> Result<Self> {
        Ok(Self { uuid, created_at })
    }

    /// Generates a UUID using the provided RNG
    fn generate_uuid_with_rng(rng: &mut StdRng) -> Uuid {
        let mut bytes = [0u8; 16];
        rng.fill(&mut bytes);

        // Set version (4) and variant bits according to RFC 4122
        bytes[6] = (bytes[6] & 0x0f) | 0x40; // Version 4
        bytes[8] = (bytes[8] & 0x3f) | 0x80; // Variant 10

        Uuid::from_bytes(bytes)
    }

    /// Hashes a seed string to a u64 for RNG seeding
    fn hash_seed_to_u64(seed: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        hasher.finish()
    }
}

impl NodeIdentity for NodeId {
    fn uuid(&self) -> &Uuid {
        &self.uuid
    }

    fn created_at(&self) -> SystemTime {
        self.created_at
    }

    fn matches(&self, query: &str) -> bool {
        // Check if query matches UUID prefix
        self.uuid.to_string().starts_with(query)
    }

    fn full_display(&self) -> String {
        self.uuid.to_string()
    }

    fn short_uuid(&self) -> String {
        self.uuid.to_string()[..8].to_string()
    }
}

// Implement Display to return UUID
impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.uuid)
    }
}

// Implement PartialEq based on UUID (the unique identifier)
impl PartialEq for NodeId {
    fn eq(&self, other: &Self) -> bool {
        self.uuid == other.uuid
    }
}

impl Eq for NodeId {}

// Implement Hash based on UUID
impl std::hash::Hash for NodeId {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.uuid.hash(state);
    }
}

// Implement FromStr to allow parsing from strings
impl std::str::FromStr for NodeId {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Try to parse as UUID first
        if let Ok(uuid) = Uuid::parse_str(s) {
            return Ok(NodeId {
                uuid,
                created_at: SystemTime::now(),
            });
        }

        // Otherwise, treat the string as a seed to generate a new NodeId
        NodeId::new(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_id_creation() {
        let seed = "test-seed-123";
        let id = NodeId::new(seed).expect("Should create NodeId");

        // Verify UUID is populated
        assert!(!id.uuid().to_string().is_empty());

        // Verify creation time is recent
        let elapsed = SystemTime::now()
            .duration_since(id.created_at())
            .expect("Time should not go backwards");
        assert!(elapsed.as_secs() < 1); // Should be created within the last second
    }

    #[test]
    fn test_node_id_uniqueness() {
        let seed1 = "test-seed-123";
        let seed2 = "test-seed-456";
        let id1 = NodeId::new(seed1).expect("Should create first NodeId");
        let id2 = NodeId::new(seed2).expect("Should create second NodeId");

        // UUIDs should be different for different seeds
        assert_ne!(id1.uuid(), id2.uuid());
    }

    #[test]
    fn test_node_id_with_seed() {
        let seed = "test-seed-123";

        // Create two NodeIds with the same seed
        let id1 = NodeId::new(seed).expect("Should create first seeded NodeId");
        let id2 = NodeId::new(seed).expect("Should create second seeded NodeId");

        // With the same seed, they should be identical
        assert_eq!(id1.uuid(), id2.uuid());
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_node_id_with_different_seeds() {
        let seed1 = "test-seed-123";
        let seed2 = "test-seed-456";

        // Create NodeIds with different seeds
        let id1 = NodeId::new(seed1).expect("Should create first seeded NodeId");
        let id2 = NodeId::new(seed2).expect("Should create second seeded NodeId");

        // With different seeds, they should be different
        assert_ne!(id1.uuid(), id2.uuid());
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_seeded_generation() {
        let seed1 = "test-seed-abc";
        let seed2 = "test-seed-xyz";

        // Create one with first seed
        let seeded_id1 = NodeId::new(seed1).expect("Should create first seeded NodeId");

        // Create one with second seed
        let seeded_id2 = NodeId::new(seed2).expect("Should create second seeded NodeId");

        // They should be different for different seeds
        assert_ne!(seeded_id1.uuid(), seeded_id2.uuid());
        assert_ne!(seeded_id1, seeded_id2);
    }

    #[test]
    fn test_node_id_matching() {
        let seed = "test-seed-123";
        let id = NodeId::new(seed).expect("Should create NodeId");

        // Test UUID prefix matching
        let uuid_str = id.uuid().to_string();
        let uuid_prefix = &uuid_str[..8];
        assert!(id.matches(uuid_prefix));

        // Test non-matching query
        assert!(!id.matches("nonexistent"));
    }

    #[test]
    fn test_node_id_display() {
        let seed = "test-seed-123";
        let id = NodeId::new(seed).expect("Should create NodeId");

        // Test full_display format
        let display = id.full_display();
        assert_eq!(display, id.uuid().to_string());

        // Test short_uuid
        let short = id.short_uuid();
        assert_eq!(short.len(), 8);
        assert!(id.uuid().to_string().starts_with(&short));
    }

    #[test]
    fn test_from_parts() {
        let uuid = Uuid::new_v4();
        let created_at = SystemTime::now();

        let id = NodeId::from_parts(uuid, created_at).expect("Should create from parts");

        assert_eq!(id.uuid(), &uuid);
        assert_eq!(id.created_at(), created_at);
    }

    #[test]
    fn test_equality() {
        let seed = "test-seed-123";
        let id1 = NodeId::new(seed).unwrap();
        let id2 = NodeId::new(seed).unwrap();

        // Same UUID = equal
        assert_eq!(id1, id2);

        // Different UUIDs = not equal
        let id3 = NodeId::new("test-seed-456").unwrap();
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_hash() {
        use std::collections::HashSet;

        let seed1 = "test-seed-123";
        let seed2 = "test-seed-456";
        let id1 = NodeId::new(seed1).unwrap();
        let id2 = NodeId::new(seed2).unwrap();

        let mut set = HashSet::new();
        set.insert(id1.clone());
        set.insert(id2.clone());

        // Both should be in the set
        assert!(set.contains(&id1));
        assert!(set.contains(&id2));
        assert_eq!(set.len(), 2);

        // Adding the same ID again shouldn't increase the size
        set.insert(id1.clone());
        assert_eq!(set.len(), 2);
    }
}
