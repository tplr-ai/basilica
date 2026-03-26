//! Region-to-country mapping utilities for Citadel providers
//!
//! Citadel (datacenter) providers use region codes (e.g., "EU-WEST-1", "US-EAST") while
//! The Bourse uses country names. This module provides utilities to map
//! between region codes and country codes for filtering purposes.

/// Map Citadel provider region codes to ISO country codes
///
/// Returns the two-letter ISO country code for known region patterns.
/// Returns `None` for unrecognized regions.
pub fn region_to_country(region: &str) -> Option<&'static str> {
    let region_upper = region.to_uppercase();

    // US regions
    if region_upper.starts_with("US-")
        || region_upper.contains("US-EAST")
        || region_upper.contains("US-WEST")
        || region_upper.contains("VIRGINIA")
        || region_upper.contains("OREGON")
        || region_upper.contains("OHIO")
        || region_upper.contains("CALIFORNIA")
    {
        return Some("US");
    }

    // EU regions with specific country mappings
    if region_upper.starts_with("EU-") || region_upper.starts_with("EUR-") {
        if region_upper.contains("IRELAND") || region_upper.contains("DUBLIN") {
            return Some("IE");
        }
        if region_upper.contains("FRANKFURT") || region_upper.contains("GERMANY") {
            return Some("DE");
        }
        if region_upper.contains("LONDON") || region_upper.contains("UK") {
            return Some("GB");
        }
        if region_upper.contains("PARIS") || region_upper.contains("FRANCE") {
            return Some("FR");
        }
        if region_upper.contains("STOCKHOLM") || region_upper.contains("SWEDEN") {
            return Some("SE");
        }
        if region_upper.contains("MILAN") || region_upper.contains("ITALY") {
            return Some("IT");
        }
        if region_upper.contains("SPAIN") || region_upper.contains("MADRID") {
            return Some("ES");
        }
        if region_upper.contains("NETHERLANDS") || region_upper.contains("AMSTERDAM") {
            return Some("NL");
        }
        // Generic EU fallback
        return Some("EU");
    }

    // Asia-Pacific regions
    if region_upper.starts_with("AP-") || region_upper.starts_with("APAC-") {
        if region_upper.contains("TOKYO") || region_upper.contains("JAPAN") {
            return Some("JP");
        }
        if region_upper.contains("SINGAPORE") {
            return Some("SG");
        }
        if region_upper.contains("SYDNEY") || region_upper.contains("AUSTRALIA") {
            return Some("AU");
        }
        if region_upper.contains("MUMBAI") || region_upper.contains("INDIA") {
            return Some("IN");
        }
        if region_upper.contains("SEOUL") || region_upper.contains("KOREA") {
            return Some("KR");
        }
        if region_upper.contains("HONG") {
            return Some("HK");
        }
    }

    // Canada
    if region_upper.contains("CANADA") || region_upper.starts_with("CA-") {
        return Some("CA");
    }

    // South America
    if region_upper.contains("SAO") || region_upper.contains("BRAZIL") {
        return Some("BR");
    }

    // Middle East
    if region_upper.contains("BAHRAIN") {
        return Some("BH");
    }
    if region_upper.contains("UAE") || region_upper.contains("DUBAI") {
        return Some("AE");
    }

    // Africa
    if region_upper.contains("CAPE") || region_upper.contains("SOUTH AFRICA") {
        return Some("ZA");
    }

    // Norway
    if region_upper.contains("NORWAY") || region_upper.starts_with("NO-") {
        return Some("NO");
    }

    None
}

/// Extract country code from various location formats
///
/// Handles multiple formats:
/// - Region codes: "US-EAST-1" → "US"
/// - Full location: "San Francisco/California/US" → "US"
/// - Direct country: "US" → "US"
pub fn extract_country_code(location: &str) -> Option<&'static str> {
    // First try region_to_country for region codes
    if let Some(country) = region_to_country(location) {
        return Some(country);
    }

    // Try parsing as LocationProfile (City/Region/Country format)
    // Format is typically: "City/State/Country" or just "Country"
    let parts: Vec<&str> = location.split('/').collect();
    if let Some(last_part) = parts.last() {
        let trimmed = last_part.trim();
        // Map common country names/codes to ISO codes
        return map_country_string(trimmed);
    }

    None
}

/// Map country name or code strings to standard ISO codes
fn map_country_string(country: &str) -> Option<&'static str> {
    let upper = country.to_uppercase();
    match upper.as_str() {
        // Direct ISO codes
        "US" | "USA" | "UNITED STATES" => Some("US"),
        "CA" | "CANADA" => Some("CA"),
        "GB" | "UK" | "UNITED KINGDOM" => Some("GB"),
        "DE" | "GERMANY" => Some("DE"),
        "FR" | "FRANCE" => Some("FR"),
        "IT" | "ITALY" => Some("IT"),
        "ES" | "SPAIN" => Some("ES"),
        "NL" | "NETHERLANDS" => Some("NL"),
        "SE" | "SWEDEN" => Some("SE"),
        "IE" | "IRELAND" => Some("IE"),
        "JP" | "JAPAN" => Some("JP"),
        "SG" | "SINGAPORE" => Some("SG"),
        "AU" | "AUSTRALIA" => Some("AU"),
        "IN" | "INDIA" => Some("IN"),
        "KR" | "KOREA" | "SOUTH KOREA" => Some("KR"),
        "HK" | "HONG KONG" => Some("HK"),
        "BR" | "BRAZIL" => Some("BR"),
        "BH" | "BAHRAIN" => Some("BH"),
        "AE" | "UAE" | "UNITED ARAB EMIRATES" => Some("AE"),
        "ZA" | "SOUTH AFRICA" => Some("ZA"),
        "NO" | "NORWAY" => Some("NO"),
        "EU" | "EUROPE" => Some("EU"),
        _ => None,
    }
}

/// Check if a region matches a country filter (case-insensitive)
///
/// Performs matching in two ways:
/// 1. Direct substring match (e.g., "US" matches "US-EAST-1")
/// 2. Mapped country code match (e.g., "US" matches "Virginia" via region_to_country)
pub fn region_matches_country(region: &str, country_filter: &str) -> bool {
    let filter_upper = country_filter.to_uppercase();
    let region_upper = region.to_uppercase();

    // Direct region name match (e.g., filter "US" matches region "US-EAST-1")
    if region_upper.contains(&filter_upper) {
        return true;
    }

    // Check mapped country code
    if let Some(country_code) = region_to_country(region) {
        if country_code.to_uppercase() == filter_upper {
            return true;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_region_to_country_us() {
        assert_eq!(region_to_country("US-EAST-1"), Some("US"));
        assert_eq!(region_to_country("us-west-2"), Some("US"));
        assert_eq!(region_to_country("Virginia"), Some("US"));
    }

    #[test]
    fn test_region_to_country_eu() {
        assert_eq!(region_to_country("EU-WEST-1"), Some("EU"));
        assert_eq!(region_to_country("EU-Frankfurt"), Some("DE"));
        assert_eq!(region_to_country("eu-london"), Some("GB"));
    }

    #[test]
    fn test_region_to_country_apac() {
        assert_eq!(region_to_country("AP-TOKYO-1"), Some("JP"));
        assert_eq!(region_to_country("ap-singapore"), Some("SG"));
        assert_eq!(region_to_country("AP-Sydney"), Some("AU"));
    }

    #[test]
    fn test_region_matches_country() {
        // Direct match
        assert!(region_matches_country("US-EAST-1", "US"));
        assert!(region_matches_country("us-east-1", "us"));

        // Mapped match
        assert!(region_matches_country("Virginia", "US"));
        assert!(region_matches_country("EU-Frankfurt", "DE"));
        assert!(region_matches_country("AP-Tokyo", "JP"));

        // No match
        assert!(!region_matches_country("US-EAST-1", "EU"));
        assert!(!region_matches_country("EU-WEST-1", "US"));
    }

    #[test]
    fn test_unknown_region() {
        assert_eq!(region_to_country("unknown-region"), None);
        // Unknown regions should still work with direct matching
        assert!(region_matches_country("custom-us-region", "US"));
    }

    #[test]
    fn test_extract_country_code_region_codes() {
        // Region codes should be handled via region_to_country
        assert_eq!(extract_country_code("US-EAST-1"), Some("US"));
        assert_eq!(extract_country_code("EU-WEST-1"), Some("EU"));
        assert_eq!(extract_country_code("EU-Frankfurt"), Some("DE"));
        assert_eq!(extract_country_code("AP-TOKYO-1"), Some("JP"));
    }

    #[test]
    fn test_extract_country_code_location_profiles() {
        // City/Region/Country format (The Bourse)
        assert_eq!(
            extract_country_code("San Francisco/California/US"),
            Some("US")
        );
        assert_eq!(extract_country_code("London/UK"), Some("GB"));
        assert_eq!(extract_country_code("Tokyo/Japan"), Some("JP"));
    }

    #[test]
    fn test_extract_country_code_direct() {
        // Direct country codes or names
        assert_eq!(extract_country_code("US"), Some("US"));
        assert_eq!(extract_country_code("Germany"), Some("DE"));
        assert_eq!(extract_country_code("Japan"), Some("JP"));
    }

    #[test]
    fn test_extract_country_code_unknown() {
        assert_eq!(extract_country_code("unknown-location"), None);
        assert_eq!(extract_country_code("Some City/Some State/Unknown"), None);
    }

    #[test]
    fn test_extract_country_code_norway() {
        assert_eq!(extract_country_code("NORWAY"), Some("NO"));
        assert_eq!(extract_country_code("NO-1"), Some("NO"));
        assert_eq!(extract_country_code("Norway"), Some("NO"));
    }
}
