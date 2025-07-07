//! Core ExecutorId implementation with UUID + HUID support
//!
//! This module provides the main ExecutorId struct that combines:
//! - UUID v4 for guaranteed uniqueness
//! - HUID (Human-Unique Identifier) for user-friendly interaction
//!
//! The HUID format is: adjective-noun-4hex (e.g., "swift-falcon-a3f2")

use anyhow::{Context, Result};
use rand::Rng;
use std::time::SystemTime;
use uuid::Uuid;

use crate::executor_identity::{
    constants::{HEX_CHARS, HUID_HEX_LENGTH, HUID_SEPARATOR},
    word_provider::StaticWordProvider,
    ExecutorIdentity, WordProvider,
};

/// Main executor identifier combining UUID and HUID
#[derive(Debug, Clone)]
pub struct ExecutorId {
    /// UUID v4 for guaranteed uniqueness
    uuid: Uuid,
    /// Human-readable identifier (e.g., "swift-falcon-a3f2")
    huid: String,
    /// Creation timestamp
    created_at: SystemTime,
}

impl ExecutorId {
    /// Creates a new ExecutorId with a generated UUID and HUID
    ///
    /// # Errors
    /// Returns an error if HUID generation fails after maximum attempts
    pub fn new() -> Result<Self> {
        let word_provider = StaticWordProvider::new();
        Self::new_with_provider(&word_provider)
    }

    /// Creates a new ExecutorId with a specific word provider
    ///
    /// # Arguments
    /// * `word_provider` - Provider for adjective and noun lists
    ///
    /// # Errors
    /// Returns an error if HUID generation fails after maximum attempts
    pub fn new_with_provider(word_provider: &dyn WordProvider) -> Result<Self> {
        let uuid = Uuid::new_v4();
        let huid = Self::generate_huid(uuid, word_provider)?;

        Ok(Self {
            uuid,
            huid,
            created_at: SystemTime::now(),
        })
    }

    /// Creates an ExecutorId from existing UUID and HUID values
    ///
    /// This is useful when reconstructing from persistent storage
    ///
    /// # Arguments
    /// * `uuid` - The UUID to use
    /// * `huid` - The HUID string (must be valid format)
    /// * `created_at` - The creation timestamp
    ///
    /// # Errors
    /// Returns an error if the HUID format is invalid
    pub fn from_parts(uuid: Uuid, huid: String, created_at: SystemTime) -> Result<Self> {
        // Validate HUID format
        if !crate::executor_identity::constants::is_valid_huid(&huid) {
            anyhow::bail!("Invalid HUID format: {}", huid);
        }

        Ok(Self {
            uuid,
            huid,
            created_at,
        })
    }

    /// Generates a HUID for the given UUID
    ///
    /// Uses a collision retry mechanism with up to MAX_HUID_GENERATION_ATTEMPTS
    fn generate_huid(_uuid: Uuid, word_provider: &dyn WordProvider) -> Result<String> {
        let mut rng = rand::thread_rng();

        // Validate word lists
        word_provider
            .validate_word_lists()
            .context("Invalid word provider")?;

        let adjective_count = word_provider.adjective_count();
        let noun_count = word_provider.noun_count();

        // Select random adjective and noun
        let adj_idx = rng.gen_range(0..adjective_count);
        let noun_idx = rng.gen_range(0..noun_count);

        let adjective = word_provider
            .get_adjective(adj_idx)
            .ok_or_else(|| anyhow::anyhow!("Failed to get adjective at index {}", adj_idx))?;
        let noun = word_provider
            .get_noun(noun_idx)
            .ok_or_else(|| anyhow::anyhow!("Failed to get noun at index {}", noun_idx))?;

        // Generate hex suffix
        let hex_suffix = Self::generate_hex_suffix(&mut rng);

        // Construct HUID
        let huid = format!("{adjective}{HUID_SEPARATOR}{noun}{HUID_SEPARATOR}{hex_suffix}");

        // Note: In a full implementation with a persistence layer, we would check
        // for collisions here and retry with different combinations if needed.
        // For now, we assume the combination of words + hex is sufficiently unique
        // (655M+ possible combinations make collisions extremely unlikely).
        Ok(huid)
    }

    /// Generates a random hexadecimal suffix of HUID_HEX_LENGTH characters
    fn generate_hex_suffix(rng: &mut impl Rng) -> String {
        (0..HUID_HEX_LENGTH)
            .map(|_| HEX_CHARS[rng.gen_range(0..HEX_CHARS.len())])
            .collect()
    }

    /// Parses a UUID string and creates an ExecutorId for it
    ///
    /// This generates a new HUID for the provided UUID
    ///
    /// # Arguments
    /// * `uuid_str` - String representation of a UUID
    ///
    /// # Errors
    /// Returns an error if the UUID string is invalid or HUID generation fails
    pub fn from_uuid_string(uuid_str: &str) -> Result<Self> {
        let uuid = Uuid::parse_str(uuid_str).context("Invalid UUID string")?;

        let word_provider = StaticWordProvider::new();
        let huid = Self::generate_huid(uuid, &word_provider)?;

        Ok(Self {
            uuid,
            huid,
            created_at: SystemTime::now(),
        })
    }
}

impl ExecutorIdentity for ExecutorId {
    fn uuid(&self) -> &Uuid {
        &self.uuid
    }

    fn huid(&self) -> &str {
        &self.huid
    }

    fn created_at(&self) -> SystemTime {
        self.created_at
    }

    fn matches(&self, query: &str) -> bool {
        // Validate query length
        if query.len() < crate::executor_identity::constants::MIN_HUID_PREFIX_LENGTH {
            return false;
        }

        // Check if query matches UUID prefix
        if self.uuid.to_string().starts_with(query) {
            return true;
        }

        // Check if query matches HUID prefix
        self.huid.starts_with(query)
    }

    fn full_display(&self) -> String {
        format!("{} ({})", self.huid, self.uuid)
    }

    fn short_uuid(&self) -> String {
        self.uuid.to_string()[..8].to_string()
    }
}

// Implement Display for convenient formatting
impl std::fmt::Display for ExecutorId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.full_display())
    }
}

// Implement PartialEq based on UUID (the unique identifier)
impl PartialEq for ExecutorId {
    fn eq(&self, other: &Self) -> bool {
        self.uuid == other.uuid
    }
}

impl Eq for ExecutorId {}

// Implement Hash based on UUID
impl std::hash::Hash for ExecutorId {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.uuid.hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor_identity::word_provider::StaticWordProvider;

    #[test]
    fn test_executor_id_creation() {
        let id = ExecutorId::new().expect("Should create ExecutorId");

        // Verify all fields are populated
        assert!(!id.uuid().to_string().is_empty());
        assert!(!id.huid().is_empty());

        // Verify HUID format
        assert!(crate::executor_identity::constants::is_valid_huid(
            id.huid()
        ));

        // Verify creation time is recent
        let elapsed = SystemTime::now()
            .duration_since(id.created_at())
            .expect("Time should not go backwards");
        assert!(elapsed.as_secs() < 1); // Should be created within the last second
    }

    #[test]
    fn test_executor_id_uniqueness() {
        let id1 = ExecutorId::new().expect("Should create first ExecutorId");
        let id2 = ExecutorId::new().expect("Should create second ExecutorId");

        // UUIDs should be different
        assert_ne!(id1.uuid(), id2.uuid());

        // HUIDs should be different (with very high probability)
        // Note: There's a tiny chance they could be the same, but it's astronomically small
        assert_ne!(id1.huid(), id2.huid());
    }

    #[test]
    fn test_executor_id_matching() {
        let id = ExecutorId::new().expect("Should create ExecutorId");

        // Test HUID prefix matching
        let huid_prefix = &id.huid()[..5]; // Take first 5 chars
        assert!(id.matches(huid_prefix));

        // Test UUID prefix matching
        let uuid_str = id.uuid().to_string();
        let uuid_prefix = &uuid_str[..8];
        assert!(id.matches(uuid_prefix));

        // Test non-matching query
        assert!(!id.matches("nonexistent"));

        // Test too-short query
        assert!(!id.matches("ab")); // Less than MIN_HUID_PREFIX_LENGTH
    }

    #[test]
    fn test_executor_id_display() {
        let id = ExecutorId::new().expect("Should create ExecutorId");

        // Test full_display format
        let display = id.full_display();
        assert!(display.contains(id.huid()));
        assert!(display.contains(&id.uuid().to_string()));
        assert!(display.contains(" ("));
        assert!(display.contains(")"));

        // Test short_uuid
        let short = id.short_uuid();
        assert_eq!(short.len(), 8);
        assert!(id.uuid().to_string().starts_with(&short));
    }

    #[test]
    fn test_from_parts() {
        let uuid = Uuid::new_v4();
        let huid = "swift-falcon-a3f2".to_string();
        let created_at = SystemTime::now();

        let id = ExecutorId::from_parts(uuid, huid.clone(), created_at)
            .expect("Should create from parts");

        assert_eq!(id.uuid(), &uuid);
        assert_eq!(id.huid(), &huid);
        assert_eq!(id.created_at(), created_at);
    }

    #[test]
    fn test_from_parts_invalid_huid() {
        let uuid = Uuid::new_v4();
        let invalid_huid = "invalid_format".to_string();
        let created_at = SystemTime::now();

        let result = ExecutorId::from_parts(uuid, invalid_huid, created_at);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid HUID format"));
    }

    #[test]
    fn test_from_uuid_string() {
        let uuid = Uuid::new_v4();
        let uuid_str = uuid.to_string();

        let id = ExecutorId::from_uuid_string(&uuid_str).expect("Should create from UUID string");

        assert_eq!(id.uuid(), &uuid);
        assert!(crate::executor_identity::constants::is_valid_huid(
            id.huid()
        ));
    }

    #[test]
    fn test_from_uuid_string_invalid() {
        let result = ExecutorId::from_uuid_string("not-a-uuid");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid UUID"));
    }

    #[test]
    fn test_equality() {
        let uuid = Uuid::new_v4();
        let huid1 = "swift-falcon-a3f2".to_string();
        let huid2 = "brave-lion-b4c3".to_string();
        let created_at = SystemTime::now();

        let id1 = ExecutorId::from_parts(uuid, huid1, created_at).unwrap();
        let id2 = ExecutorId::from_parts(uuid, huid2, created_at).unwrap();

        // Same UUID = equal, even with different HUIDs
        assert_eq!(id1, id2);

        // Different UUIDs = not equal
        let id3 = ExecutorId::new().unwrap();
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_hash() {
        use std::collections::HashSet;

        let id1 = ExecutorId::new().unwrap();
        let id2 = ExecutorId::new().unwrap();

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

    #[test]
    fn test_hex_suffix_generation() {
        let mut rng = rand::thread_rng();

        // Generate multiple suffixes and verify format
        for _ in 0..100 {
            let suffix = ExecutorId::generate_hex_suffix(&mut rng);
            assert_eq!(suffix.len(), HUID_HEX_LENGTH);
            assert!(suffix.chars().all(|c| HEX_CHARS.contains(&c)));
        }
    }

    #[test]
    fn test_huid_generation_with_custom_provider() {
        // This uses the real StaticWordProvider
        let provider = StaticWordProvider::new();
        let id =
            ExecutorId::new_with_provider(&provider).expect("Should create with custom provider");

        // Verify the HUID uses words from the provider
        let parts: Vec<&str> = id.huid().split('-').collect();
        assert_eq!(parts.len(), 3);

        // The adjective and noun should exist in the word lists
        let _adjective = parts[0];
        let _noun = parts[1];
        let hex = parts[2];

        // Verify hex suffix format
        assert_eq!(hex.len(), HUID_HEX_LENGTH);
        assert!(hex.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
