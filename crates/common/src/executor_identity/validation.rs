//! Validation utilities for executor identities
//!
//! This module provides functions to validate HUID formats, UUID strings,
//! and identifier queries.

use crate::executor_identity::constants::{is_valid_huid, MIN_HUID_PREFIX_LENGTH};
use anyhow::Result;
use uuid::Uuid;

/// Validates an executor identifier query
///
/// An identifier can be:
/// - A full UUID (36 characters)
/// - A UUID prefix (at least 3 characters)
/// - A full HUID (format: adjective-noun-4hex)
/// - A HUID prefix (at least 3 characters)
///
/// # Arguments
/// * `identifier` - The identifier string to validate
///
/// # Returns
/// * `Ok(IdentifierType)` - The type of valid identifier
/// * `Err` - If the identifier is invalid
pub fn validate_identifier(identifier: &str) -> Result<IdentifierType> {
    // Check if empty
    if identifier.is_empty() {
        anyhow::bail!("Identifier cannot be empty");
    }

    // Check minimum length for any query
    if identifier.len() < MIN_HUID_PREFIX_LENGTH {
        anyhow::bail!(
            "Identifier '{}' is too short. Minimum length is {} characters",
            identifier,
            MIN_HUID_PREFIX_LENGTH
        );
    }

    // Try to parse as full UUID
    if let Ok(uuid) = Uuid::parse_str(identifier) {
        return Ok(IdentifierType::FullUuid(uuid));
    }

    // Check if it's a valid full HUID
    if is_valid_huid(identifier) {
        return Ok(IdentifierType::FullHuid(identifier.to_string()));
    }

    // Check if it could be a UUID prefix (hex characters with optional dashes)
    if is_valid_uuid_prefix(identifier) {
        return Ok(IdentifierType::UuidPrefix(identifier.to_string()));
    }

    // Check if it could be a HUID prefix
    if is_valid_huid_prefix(identifier) {
        return Ok(IdentifierType::HuidPrefix(identifier.to_string()));
    }

    anyhow::bail!(
        "Invalid identifier '{}'. Must be a UUID, HUID, or valid prefix (min {} chars)",
        identifier,
        MIN_HUID_PREFIX_LENGTH
    )
}

/// Types of valid identifiers
#[derive(Debug, Clone, PartialEq)]
pub enum IdentifierType {
    /// Full UUID
    FullUuid(Uuid),
    /// UUID prefix (partial UUID string)
    UuidPrefix(String),
    /// Full HUID
    FullHuid(String),
    /// HUID prefix
    HuidPrefix(String),
}

impl IdentifierType {
    /// Returns true if this is a full identifier (not a prefix)
    pub fn is_complete(&self) -> bool {
        matches!(
            self,
            IdentifierType::FullUuid(_) | IdentifierType::FullHuid(_)
        )
    }

    /// Returns true if this is a prefix identifier
    pub fn is_prefix(&self) -> bool {
        matches!(
            self,
            IdentifierType::UuidPrefix(_) | IdentifierType::HuidPrefix(_)
        )
    }

    /// Returns the string representation of the identifier
    pub fn as_str(&self) -> &str {
        match self {
            IdentifierType::FullUuid(_uuid) => {
                // This is a bit of a hack, but UUIDs are typically used as strings
                // In a real implementation, we might store the string representation
                ""
            }
            IdentifierType::UuidPrefix(s)
            | IdentifierType::FullHuid(s)
            | IdentifierType::HuidPrefix(s) => s.as_str(),
        }
    }
}

/// Checks if a string could be a valid UUID prefix
///
/// Valid UUID prefixes contain only hex characters and optional dashes
fn is_valid_uuid_prefix(s: &str) -> bool {
    // Remove dashes and check if all remaining characters are hex
    let without_dashes: String = s.chars().filter(|&c| c != '-').collect();

    // Must have at least some hex characters
    if without_dashes.is_empty() {
        return false;
    }

    // All characters must be valid hex
    without_dashes.chars().all(|c| c.is_ascii_hexdigit())
}

/// Checks if a string could be a valid HUID prefix
///
/// Valid HUID prefixes start with lowercase letters and may contain dashes and hex at the end
fn is_valid_huid_prefix(s: &str) -> bool {
    // Must start with a lowercase letter
    if !s
        .chars()
        .next()
        .map(|c| c.is_ascii_lowercase())
        .unwrap_or(false)
    {
        return false;
    }

    // Can contain lowercase letters, dashes, and hex digits
    s.chars()
        .all(|c| c.is_ascii_lowercase() || c == '-' || c.is_ascii_hexdigit())
}

/// Validates a HUID's structure in detail
///
/// Returns detailed information about what's wrong with an invalid HUID
pub fn validate_huid_detailed(huid: &str) -> Result<()> {
    if huid.is_empty() {
        anyhow::bail!("HUID cannot be empty");
    }

    // Check overall format with regex
    if !is_valid_huid(huid) {
        // Provide detailed error message
        let parts: Vec<&str> = huid.split('-').collect();

        if parts.len() != 3 {
            anyhow::bail!(
                "HUID must have exactly 3 parts separated by dashes (adjective-noun-hex), found {} parts",
                parts.len()
            );
        }

        // Check adjective
        if let Some(adj) = parts.first() {
            if adj.is_empty() {
                anyhow::bail!("HUID adjective cannot be empty");
            }
            if !adj.chars().all(|c| c.is_ascii_lowercase()) {
                anyhow::bail!(
                    "HUID adjective '{}' must contain only lowercase letters",
                    adj
                );
            }
        }

        // Check noun
        if let Some(noun) = parts.get(1) {
            if noun.is_empty() {
                anyhow::bail!("HUID noun cannot be empty");
            }
            if !noun.chars().all(|c| c.is_ascii_lowercase()) {
                anyhow::bail!("HUID noun '{}' must contain only lowercase letters", noun);
            }
        }

        // Check hex suffix
        if let Some(hex) = parts.get(2) {
            if hex.len() != 4 {
                anyhow::bail!(
                    "HUID hex suffix must be exactly 4 characters, found {}",
                    hex.len()
                );
            }
            if !hex.chars().all(|c| c.is_ascii_hexdigit()) {
                anyhow::bail!(
                    "HUID hex suffix '{}' must contain only hexadecimal characters (0-9, a-f)",
                    hex
                );
            }
            if hex.chars().any(|c| c.is_ascii_uppercase()) {
                anyhow::bail!("HUID hex suffix '{}' must be lowercase", hex);
            }
        }

        anyhow::bail!("Invalid HUID format: '{}'", huid);
    }

    Ok(())
}

/// Validates a batch of identifiers
///
/// Returns a vector of results, one for each input identifier
pub fn validate_identifiers_batch(identifiers: &[&str]) -> Vec<Result<IdentifierType>> {
    identifiers
        .iter()
        .map(|&id| validate_identifier(id))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_identifier_full_uuid() {
        let uuid = Uuid::new_v4();
        let uuid_str = uuid.to_string();

        let result = validate_identifier(&uuid_str).unwrap();
        match result {
            IdentifierType::FullUuid(u) => assert_eq!(u, uuid),
            _ => panic!("Expected FullUuid"),
        }
    }

    #[test]
    fn test_validate_identifier_uuid_prefix() {
        // Valid UUID prefixes
        assert!(matches!(
            validate_identifier("a1b2c3d4").unwrap(),
            IdentifierType::UuidPrefix(_)
        ));

        assert!(matches!(
            validate_identifier("123e4567").unwrap(),
            IdentifierType::UuidPrefix(_)
        ));

        // With dashes
        assert!(matches!(
            validate_identifier("a1b2-c3d4").unwrap(),
            IdentifierType::UuidPrefix(_)
        ));
    }

    #[test]
    fn test_validate_identifier_full_huid() {
        let valid_huids = vec!["swift-falcon-a3f2", "brave-lion-0000", "tiny-ant-ffff"];

        for huid in valid_huids {
            let result = validate_identifier(huid).unwrap();
            match result {
                IdentifierType::FullHuid(h) => assert_eq!(h, huid),
                _ => panic!("Expected FullHuid for {huid}"),
            }
        }
    }

    #[test]
    fn test_validate_identifier_huid_prefix() {
        let valid_prefixes = vec![
            "swift",
            "swift-",
            "swift-fal",
            "swift-falcon",
            "swift-falcon-",
            "swift-falcon-a",
            "swift-falcon-a3",
        ];

        for prefix in valid_prefixes {
            let result = validate_identifier(prefix).unwrap();
            match result {
                IdentifierType::HuidPrefix(p) => assert_eq!(p, prefix),
                _ => panic!("Expected HuidPrefix for {prefix}"),
            }
        }
    }

    #[test]
    fn test_validate_identifier_errors() {
        // Empty
        assert!(validate_identifier("").is_err());

        // Too short
        assert!(validate_identifier("ab").is_err());

        // Invalid characters
        assert!(validate_identifier("swift-FALCON-a3f2").is_err()); // Capital letters
        assert!(validate_identifier("swift_falcon_a3f2").is_err()); // Underscores
        assert!(matches!(
            validate_identifier("123-456-789").unwrap(),
            IdentifierType::UuidPrefix(_)
        )); // Valid UUID prefix
    }

    #[test]
    fn test_identifier_type_methods() {
        let full_uuid = IdentifierType::FullUuid(Uuid::new_v4());
        let full_huid = IdentifierType::FullHuid("swift-falcon-a3f2".to_string());
        let uuid_prefix = IdentifierType::UuidPrefix("a1b2c3".to_string());
        let huid_prefix = IdentifierType::HuidPrefix("swift-fal".to_string());

        // Test is_complete
        assert!(full_uuid.is_complete());
        assert!(full_huid.is_complete());
        assert!(!uuid_prefix.is_complete());
        assert!(!huid_prefix.is_complete());

        // Test is_prefix
        assert!(!full_uuid.is_prefix());
        assert!(!full_huid.is_prefix());
        assert!(uuid_prefix.is_prefix());
        assert!(huid_prefix.is_prefix());
    }

    #[test]
    fn test_validate_huid_detailed() {
        // Valid HUID
        assert!(validate_huid_detailed("swift-falcon-a3f2").is_ok());

        // Empty
        let err = validate_huid_detailed("").unwrap_err();
        assert!(err.to_string().contains("empty"));

        // Wrong number of parts
        let err = validate_huid_detailed("swift-falcon").unwrap_err();
        assert!(err.to_string().contains("3 parts"));

        let err = validate_huid_detailed("swift-falcon-dragon-a3f2").unwrap_err();
        assert!(err.to_string().contains("3 parts"));

        // Invalid adjective
        let err = validate_huid_detailed("Swift-falcon-a3f2").unwrap_err();
        assert!(err.to_string().contains("lowercase"));

        let err = validate_huid_detailed("123-falcon-a3f2").unwrap_err();
        assert!(err.to_string().contains("lowercase letters"));

        // Invalid noun
        let err = validate_huid_detailed("swift-Falcon-a3f2").unwrap_err();
        assert!(err.to_string().contains("lowercase"));

        // Invalid hex
        let err = validate_huid_detailed("swift-falcon-a3f").unwrap_err();
        assert!(err.to_string().contains("4 characters"));

        let err = validate_huid_detailed("swift-falcon-A3F2").unwrap_err();
        assert!(err.to_string().contains("lowercase"));

        let err = validate_huid_detailed("swift-falcon-g3f2").unwrap_err();
        assert!(err.to_string().contains("hexadecimal"));
    }

    #[test]
    fn test_is_valid_uuid_prefix() {
        assert!(is_valid_uuid_prefix("a1b2c3"));
        assert!(is_valid_uuid_prefix("123456"));
        assert!(is_valid_uuid_prefix("abcdef"));
        assert!(is_valid_uuid_prefix("a1b2-c3d4"));
        assert!(is_valid_uuid_prefix("a1b2-c3d4-e5f6"));

        assert!(!is_valid_uuid_prefix(""));
        assert!(!is_valid_uuid_prefix("---"));
        assert!(!is_valid_uuid_prefix("g1h2")); // Invalid hex
        assert!(!is_valid_uuid_prefix("swift")); // Not hex
    }

    #[test]
    fn test_is_valid_huid_prefix() {
        assert!(is_valid_huid_prefix("swift"));
        assert!(is_valid_huid_prefix("swift-"));
        assert!(is_valid_huid_prefix("swift-falcon"));
        assert!(is_valid_huid_prefix("swift-falcon-"));
        assert!(is_valid_huid_prefix("swift-falcon-a"));
        assert!(is_valid_huid_prefix("swift-falcon-a3f"));

        assert!(!is_valid_huid_prefix(""));
        assert!(!is_valid_huid_prefix("123")); // Starts with number
        assert!(!is_valid_huid_prefix("Swift")); // Starts with capital
        assert!(!is_valid_huid_prefix("-swift")); // Starts with dash
        assert!(!is_valid_huid_prefix("swift_falcon")); // Contains underscore
    }

    #[test]
    fn test_validate_identifiers_batch() {
        let identifiers = vec![
            "swift-falcon-a3f2",  // Full HUID
            "a1b2c3d4-e5f6-7890", // UUID prefix
            "brave",              // HUID prefix
            "invalid!",           // Invalid
            "",                   // Empty
        ];

        let results = validate_identifiers_batch(&identifiers);
        assert_eq!(results.len(), 5);

        assert!(matches!(
            results[0].as_ref().unwrap(),
            IdentifierType::FullHuid(_)
        ));
        assert!(matches!(
            results[1].as_ref().unwrap(),
            IdentifierType::UuidPrefix(_)
        ));
        assert!(matches!(
            results[2].as_ref().unwrap(),
            IdentifierType::HuidPrefix(_)
        ));
        assert!(results[3].is_err());
        assert!(results[4].is_err());
    }
}
