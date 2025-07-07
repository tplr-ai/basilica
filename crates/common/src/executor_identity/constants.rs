//! Constants and configuration for the executor identity system
//!
//! This module defines validation patterns, limits, and other configuration
//! values used throughout the executor identity implementation.

use once_cell::sync::Lazy;
use regex::Regex;

/// Minimum length required for HUID prefix matching
///
/// This ensures that prefix searches are specific enough to avoid
/// excessive false positives while remaining user-friendly.
pub const MIN_HUID_PREFIX_LENGTH: usize = 3;

/// Maximum number of attempts allowed when generating a unique HUID
///
/// This prevents infinite loops in the unlikely event that all
/// reasonable HUID combinations are exhausted.
pub const MAX_HUID_GENERATION_ATTEMPTS: u32 = 10;

/// Length of the hexadecimal suffix in HUIDs
///
/// HUIDs end with exactly 4 hexadecimal characters (0-9, a-f)
/// providing 65,536 additional variations per word combination.
pub const HUID_HEX_LENGTH: usize = 4;

/// Regular expression pattern for validating HUID format
///
/// Valid HUID format: `adjective-noun-hex`
/// - adjective: lowercase letters only
/// - noun: lowercase letters only  
/// - hex: exactly 4 hexadecimal characters
///
/// Example: "swift-falcon-a3f2"
pub static HUID_PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^[a-z]+-[a-z]+-[0-9a-f]{4}$").expect("Invalid HUID regex pattern"));

/// Minimum number of adjectives required in word lists
///
/// This ensures sufficient variety in generated HUIDs
pub const MIN_ADJECTIVE_COUNT: usize = 50;

/// Minimum number of nouns required in word lists
///
/// This ensures sufficient variety in generated HUIDs
pub const MIN_NOUN_COUNT: usize = 50;

/// Expected minimum total word combinations (adjectives × nouns)
///
/// With 50+ adjectives and 50+ nouns, we get at least 2,500 base combinations,
/// which when combined with 65,536 hex variations gives us 163,840,000+ unique HUIDs
pub const MIN_WORD_COMBINATIONS: usize = 2_500;

/// Maximum length for individual words in HUIDs
///
/// This keeps HUIDs reasonably short and readable
pub const MAX_WORD_LENGTH: usize = 12;

/// Separator character used in HUIDs
pub const HUID_SEPARATOR: char = '-';

/// Default format for displaying full executor identity
pub const DEFAULT_DISPLAY_FORMAT: &str = "{huid} ({uuid})";

/// Environment variable name for custom word list path
pub const CUSTOM_WORDLIST_ENV: &str = "EXECUTOR_WORDLIST_PATH";

/// File extension for word list files
pub const WORDLIST_FILE_EXT: &str = ".words";

/// Character set for generating hexadecimal suffixes
pub const HEX_CHARS: &[char] = &[
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
];

/// Validates a potential HUID string
///
/// Returns `true` if the string matches the expected HUID format
pub fn is_valid_huid(huid: &str) -> bool {
    HUID_PATTERN.is_match(huid)
}

/// Validates a prefix for matching operations
///
/// Returns `Ok(())` if the prefix meets minimum length requirements,
/// otherwise returns an error describing the issue
pub fn validate_prefix(prefix: &str) -> Result<(), String> {
    if prefix.is_empty() {
        return Err("Prefix cannot be empty".to_string());
    }

    if prefix.len() < MIN_HUID_PREFIX_LENGTH {
        return Err(format!(
            "Prefix '{prefix}' is too short. Minimum length is {MIN_HUID_PREFIX_LENGTH} characters"
        ));
    }

    Ok(())
}

/// Calculates the total number of unique HUIDs possible with given word counts
pub fn calculate_total_huids(adjective_count: usize, noun_count: usize) -> u64 {
    let word_combinations = adjective_count as u64 * noun_count as u64;
    let hex_variations = 16u64.pow(HUID_HEX_LENGTH as u32);
    word_combinations * hex_variations
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_huid_pattern_validation() {
        // Valid HUIDs
        assert!(is_valid_huid("swift-falcon-a3f2"));
        assert!(is_valid_huid("brave-lion-0000"));
        assert!(is_valid_huid("tiny-ant-ffff"));
        assert!(is_valid_huid("mysterious-dragon-1a2b"));

        // Invalid HUIDs
        assert!(!is_valid_huid("Swift-Falcon-a3f2")); // Capital letters
        assert!(!is_valid_huid("swift_falcon_a3f2")); // Wrong separator
        assert!(!is_valid_huid("swift-falcon-a3f")); // Too few hex digits
        assert!(!is_valid_huid("swift-falcon-a3f2g")); // Invalid hex character
        assert!(!is_valid_huid("swift-falcon-A3F2")); // Capital hex
        assert!(!is_valid_huid("swiftfalcon-a3f2")); // Missing separator
        assert!(!is_valid_huid("swift-falcon-dragon-a3f2")); // Too many parts
        assert!(!is_valid_huid("123-falcon-a3f2")); // Numbers in adjective
        assert!(!is_valid_huid("swift-456-a3f2")); // Numbers in noun
        assert!(!is_valid_huid("")); // Empty string
    }

    #[test]
    fn test_prefix_validation() {
        // Valid prefixes
        assert!(validate_prefix("swi").is_ok());
        assert!(validate_prefix("swift").is_ok());
        assert!(validate_prefix("swift-falcon").is_ok());
        assert!(validate_prefix("abc").is_ok());

        // Invalid prefixes
        assert!(validate_prefix("").is_err());
        assert!(validate_prefix("a").is_err());
        assert!(validate_prefix("ab").is_err());

        // Check error messages
        let err = validate_prefix("ab").unwrap_err();
        assert!(err.contains("too short"));
        assert!(err.contains("3 characters"));
    }

    #[test]
    fn test_constants_values() {
        assert_eq!(MIN_HUID_PREFIX_LENGTH, 3);
        assert_eq!(MAX_HUID_GENERATION_ATTEMPTS, 10);
        assert_eq!(HUID_HEX_LENGTH, 4);
        assert_eq!(MIN_ADJECTIVE_COUNT, 50);
        assert_eq!(MIN_NOUN_COUNT, 50);
        assert_eq!(MIN_WORD_COMBINATIONS, 2_500);
        assert_eq!(MAX_WORD_LENGTH, 12);
        assert_eq!(HUID_SEPARATOR, '-');
        assert_eq!(HEX_CHARS.len(), 16);
    }

    #[test]
    fn test_hex_chars() {
        // Verify all hex characters are present
        let hex_str: String = HEX_CHARS.iter().collect();
        assert_eq!(hex_str, "0123456789abcdef");

        // Verify no duplicates
        let mut chars = HEX_CHARS.to_vec();
        chars.sort();
        let original_len = chars.len();
        chars.dedup();
        assert_eq!(chars.len(), original_len);
    }

    #[test]
    fn test_calculate_total_huids() {
        // Test with minimum counts
        let min_total = calculate_total_huids(MIN_ADJECTIVE_COUNT, MIN_NOUN_COUNT);
        assert_eq!(min_total, 50 * 50 * 65536);
        assert_eq!(min_total, 163_840_000);

        // Test with larger counts
        let large_total = calculate_total_huids(100, 100);
        assert_eq!(large_total, 100 * 100 * 65536);
        assert_eq!(large_total, 655_360_000);

        // Test edge cases
        assert_eq!(calculate_total_huids(0, 100), 0);
        assert_eq!(calculate_total_huids(100, 0), 0);
        assert_eq!(calculate_total_huids(1, 1), 65536);
    }

    #[test]
    fn test_regex_pattern_details() {
        let pattern = &*HUID_PATTERN;

        // Test boundary conditions
        assert!(pattern.is_match("a-b-0000"));
        assert!(pattern.is_match("verylongadjective-verylongnoun-ffff"));

        // Test that pattern correctly identifies parts
        if let Some(captures) = pattern.captures("swift-falcon-a3f2") {
            assert_eq!(&captures[0], "swift-falcon-a3f2");
        }
    }
}
