//! Common types used across Basilica components

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;
use thiserror::Error;

/// Error type for API key name validation
#[derive(Debug, Error)]
pub enum ApiKeyNameError {
    #[error("API key name cannot be empty")]
    Empty,
    #[error("API key name too long (max 100 characters)")]
    TooLong,
    #[error("API key name contains invalid characters. Only alphanumeric characters, hyphens, and underscores are allowed")]
    InvalidCharacters,
}

/// A validated API key name
///
/// API key names must:
/// - Be between 1 and 100 characters long
/// - Only contain alphanumeric characters (a-z, A-Z, 0-9), hyphens (-), and underscores (_)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ApiKeyName(String);

impl ApiKeyName {
    /// Create a new validated API key name
    pub fn new(name: impl Into<String>) -> Result<Self, ApiKeyNameError> {
        let name = name.into();
        Self::validate(&name)?;
        Ok(Self(name))
    }

    /// Validate an API key name
    fn validate(name: &str) -> Result<(), ApiKeyNameError> {
        if name.is_empty() {
            return Err(ApiKeyNameError::Empty);
        }

        if name.len() > 100 {
            return Err(ApiKeyNameError::TooLong);
        }

        // Check each character is alphanumeric, hyphen, or underscore
        if !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            return Err(ApiKeyNameError::InvalidCharacters);
        }

        Ok(())
    }

    /// Get the inner string value
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consume self and return the inner string
    pub fn into_inner(self) -> String {
        self.0
    }
}

impl fmt::Display for ApiKeyName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for ApiKeyName {
    type Err = ApiKeyNameError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::new(s)
    }
}

impl TryFrom<String> for ApiKeyName {
    type Error = ApiKeyNameError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<ApiKeyName> for String {
    fn from(name: ApiKeyName) -> Self {
        name.0
    }
}

impl AsRef<str> for ApiKeyName {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

/// Represents a geographic location profile with city, region, and country components
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocationProfile {
    #[serde(rename = "location_city", skip_serializing_if = "Option::is_none")]
    pub city: Option<String>,
    #[serde(rename = "location_region", skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    #[serde(rename = "location_country", skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
}

impl LocationProfile {
    /// Create a new LocationProfile
    pub fn new(city: Option<String>, region: Option<String>, country: Option<String>) -> Self {
        Self {
            city,
            region,
            country,
        }
    }

    /// Create a LocationProfile with all components as None
    pub fn unknown() -> Self {
        Self {
            city: None,
            region: None,
            country: None,
        }
    }
}

impl FromStr for LocationProfile {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.split('/').collect();

        let city = parts.first().and_then(|c| {
            if c.is_empty() || *c == "Unknown" {
                None
            } else {
                Some(c.to_string())
            }
        });

        let region = parts.get(1).and_then(|r| {
            if r.is_empty() || *r == "Unknown" {
                None
            } else {
                Some(r.to_string())
            }
        });

        let country = parts.get(2).and_then(|c| {
            if c.is_empty() || *c == "Unknown" {
                None
            } else {
                Some(c.to_string())
            }
        });

        Ok(Self {
            city,
            region,
            country,
        })
    }
}

impl fmt::Display for LocationProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}/{}/{}",
            self.city.as_deref().unwrap_or("Unknown"),
            self.region.as_deref().unwrap_or("Unknown"),
            self.country.as_deref().unwrap_or("Unknown")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_location_profile_from_str() {
        // Full location
        let location = LocationProfile::from_str("San Francisco/California/US").unwrap();
        assert_eq!(location.city, Some("San Francisco".to_string()));
        assert_eq!(location.region, Some("California".to_string()));
        assert_eq!(location.country, Some("US".to_string()));

        // Partial location with Unknown
        let location = LocationProfile::from_str("Unknown/California/US").unwrap();
        assert_eq!(location.city, None);
        assert_eq!(location.region, Some("California".to_string()));
        assert_eq!(location.country, Some("US".to_string()));

        // All Unknown
        let location = LocationProfile::from_str("Unknown/Unknown/Unknown").unwrap();
        assert_eq!(location.city, None);
        assert_eq!(location.region, None);
        assert_eq!(location.country, None);

        // Missing parts
        let location = LocationProfile::from_str("San Francisco").unwrap();
        assert_eq!(location.city, Some("San Francisco".to_string()));
        assert_eq!(location.region, None);
        assert_eq!(location.country, None);

        // Empty string
        let location = LocationProfile::from_str("").unwrap();
        assert_eq!(location.city, None);
        assert_eq!(location.region, None);
        assert_eq!(location.country, None);
    }

    #[test]
    fn test_location_profile_display() {
        let location = LocationProfile::new(
            Some("San Francisco".to_string()),
            Some("California".to_string()),
            Some("US".to_string()),
        );
        assert_eq!(location.to_string(), "San Francisco/California/US");

        let location =
            LocationProfile::new(None, Some("California".to_string()), Some("US".to_string()));
        assert_eq!(location.to_string(), "Unknown/California/US");

        let location = LocationProfile::unknown();
        assert_eq!(location.to_string(), "Unknown/Unknown/Unknown");

        let location =
            LocationProfile::new(Some("Tokyo".to_string()), None, Some("JP".to_string()));
        assert_eq!(location.to_string(), "Tokyo/Unknown/JP");
    }

    #[test]
    fn test_location_profile_roundtrip() {
        let original = "San Francisco/California/US";
        let location = LocationProfile::from_str(original).unwrap();
        assert_eq!(location.to_string(), original);

        let original_with_unknown = "Unknown/California/US";
        let location = LocationProfile::from_str(original_with_unknown).unwrap();
        assert_eq!(location.to_string(), original_with_unknown);
    }

    #[test]
    fn test_api_key_name_valid() {
        // Valid names
        assert!(ApiKeyName::new("test").is_ok());
        assert!(ApiKeyName::new("test-key").is_ok());
        assert!(ApiKeyName::new("test_key").is_ok());
        assert!(ApiKeyName::new("test123").is_ok());
        assert!(ApiKeyName::new("TEST_KEY_123").is_ok());
        assert!(ApiKeyName::new("my-api-key_2024").is_ok());

        // Maximum length (100 chars)
        let max_name = "a".repeat(100);
        assert!(ApiKeyName::new(&max_name).is_ok());
    }

    #[test]
    fn test_api_key_name_invalid() {
        // Empty name
        let result = ApiKeyName::new("");
        assert!(matches!(result, Err(ApiKeyNameError::Empty)));

        // Too long (>100 chars)
        let long_name = "a".repeat(101);
        let result = ApiKeyName::new(&long_name);
        assert!(matches!(result, Err(ApiKeyNameError::TooLong)));

        // Invalid characters
        assert!(matches!(
            ApiKeyName::new("test key"),
            Err(ApiKeyNameError::InvalidCharacters)
        ));
        assert!(matches!(
            ApiKeyName::new("test@key"),
            Err(ApiKeyNameError::InvalidCharacters)
        ));
        assert!(matches!(
            ApiKeyName::new("test.key"),
            Err(ApiKeyNameError::InvalidCharacters)
        ));
        assert!(matches!(
            ApiKeyName::new("test/key"),
            Err(ApiKeyNameError::InvalidCharacters)
        ));
        assert!(matches!(
            ApiKeyName::new("test#key"),
            Err(ApiKeyNameError::InvalidCharacters)
        ));
    }

    #[test]
    fn test_api_key_name_conversions() {
        let name = ApiKeyName::new("test-key").unwrap();

        // Display
        assert_eq!(format!("{}", name), "test-key");

        // AsRef<str>
        assert_eq!(name.as_ref(), "test-key");

        // as_str()
        assert_eq!(name.as_str(), "test-key");

        // Into<String>
        let cloned = name.clone();
        let string: String = cloned.into();
        assert_eq!(string, "test-key");

        // into_inner()
        let cloned = name.clone();
        assert_eq!(cloned.into_inner(), "test-key");

        // FromStr
        let parsed: ApiKeyName = "another-key".parse().unwrap();
        assert_eq!(parsed.as_str(), "another-key");

        // TryFrom<String>
        let from_string = ApiKeyName::try_from("from-string".to_string()).unwrap();
        assert_eq!(from_string.as_str(), "from-string");
    }

    #[test]
    fn test_api_key_name_serialization() {
        use serde_json;

        let name = ApiKeyName::new("test-key").unwrap();

        // Serialize
        let serialized = serde_json::to_string(&name).unwrap();
        assert_eq!(serialized, "\"test-key\"");

        // Deserialize valid
        let deserialized: ApiKeyName = serde_json::from_str("\"valid-key\"").unwrap();
        assert_eq!(deserialized.as_str(), "valid-key");

        // Deserialize invalid should fail
        let result: Result<ApiKeyName, _> = serde_json::from_str("\"invalid key\"");
        assert!(result.is_err());
    }
}
