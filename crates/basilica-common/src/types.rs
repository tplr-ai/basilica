//! Common types used across Basilica components

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
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

/// GPU category for network-wide GPU classification and scoring
///
/// This enum represents the GPU models that are officially supported and scored
/// by the Basilica validator network. Any GPU that doesn't match one of these
/// categories is classified as "Other" for general compute purposes.
///
/// Serializes as a plain string (e.g., `"A100"`, `"RTX6000"`).
/// Deserializes from both plain strings and legacy tagged format (`{"Other":"RTX6000"}`).
#[derive(Debug, Clone, PartialEq, Hash, Eq)]
pub enum GpuCategory {
    /// NVIDIA A100 - High-end training & inference
    A100,
    /// NVIDIA H100 - Flagship AI training & inference
    H100,
    /// NVIDIA H200 - High-memory AI training & inference
    H200,
    /// NVIDIA B200 - Next-gen AI acceleration
    B200,
    /// NVIDIA B300 - Ultra-class AI acceleration
    B300,
    /// Other GPU models - General GPU compute
    Other(String),
}

impl Serialize for GpuCategory {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.as_str())
    }
}

impl<'de> Deserialize<'de> for GpuCategory {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de;

        struct GpuCategoryVisitor;

        impl<'de> de::Visitor<'de> for GpuCategoryVisitor {
            type Value = GpuCategory;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a GPU type string or legacy tagged object")
            }

            fn visit_str<E: de::Error>(self, value: &str) -> Result<GpuCategory, E> {
                Ok(GpuCategory::from_str(value).unwrap())
            }

            fn visit_map<A: de::MapAccess<'de>>(self, mut map: A) -> Result<GpuCategory, A::Error> {
                // Handle legacy format: {"Other": "RTX6000"}
                if let Some(key) = map.next_key::<String>()? {
                    let value: String = map.next_value()?;
                    if key == "Other" {
                        return Ok(GpuCategory::from_str(&value).unwrap());
                    }
                    // Unknown key - treat the key itself as a GPU type
                    return Ok(GpuCategory::from_str(&key).unwrap());
                }
                Err(de::Error::custom("empty map for GpuCategory"))
            }
        }

        deserializer.deserialize_any(GpuCategoryVisitor)
    }
}

impl GpuCategory {
    /// Get the list of supported GPU model names as strings
    ///
    /// Returns the GPU models that the validator network officially supports
    /// for scoring and pricing. Use this list when querying external pricing APIs
    /// to ensure you're only fetching prices for GPUs the network can handle.
    ///
    /// # Example
    /// ```
    /// use basilica_common::types::GpuCategory;
    ///
    /// let supported = GpuCategory::supported_models();
    /// assert_eq!(supported, vec!["A100", "H100", "H200", "B200", "B300"]);
    /// ```
    pub fn supported_models() -> Vec<String> {
        vec![
            "A100".to_string(),
            "H100".to_string(),
            "H200".to_string(),
            "B200".to_string(),
            "B300".to_string(),
        ]
    }

    /// Get the use case description for this GPU category
    pub fn description(&self) -> &'static str {
        match self {
            GpuCategory::A100 => "High-end training & inference",
            GpuCategory::H100 => "Flagship AI training & inference",
            GpuCategory::H200 => "High-memory AI training & inference",
            GpuCategory::B200 => "Next-gen AI acceleration",
            GpuCategory::B300 => "Ultra-class AI acceleration",
            GpuCategory::Other(_) => "General GPU compute",
        }
    }

    /// Get the display string for this GPU category (e.g., "A100", "H100", "RTX6000")
    pub fn as_str(&self) -> String {
        match self {
            GpuCategory::A100 => "A100".to_string(),
            GpuCategory::H100 => "H100".to_string(),
            GpuCategory::H200 => "H200".to_string(),
            GpuCategory::B200 => "B200".to_string(),
            GpuCategory::B300 => "B300".to_string(),
            GpuCategory::Other(name) => name.to_uppercase(),
        }
    }
}

impl fmt::Display for GpuCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl FromStr for GpuCategory {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let model = s.to_uppercase();

        // Remove common prefixes and clean up
        let cleaned = model
            .replace("NVIDIA", "")
            .replace("GEFORCE", "")
            .replace("TESLA", "")
            .trim()
            .to_string();

        // Check for known GPU models
        if cleaned.contains("A100") {
            Ok(GpuCategory::A100)
        } else if cleaned.contains("H100") {
            Ok(GpuCategory::H100)
        } else if cleaned.contains("H200") {
            Ok(GpuCategory::H200)
        } else if cleaned.contains("B300") {
            Ok(GpuCategory::B300)
        } else if cleaned.contains("B200") {
            Ok(GpuCategory::B200)
        } else {
            Ok(GpuCategory::Other(s.to_string()))
        }
    }
}

#[cfg(test)]
mod gpu_category_serde_tests {
    use super::*;

    #[test]
    fn serialize_known_variants() {
        assert_eq!(
            serde_json::to_string(&GpuCategory::A100).unwrap(),
            "\"A100\""
        );
        assert_eq!(
            serde_json::to_string(&GpuCategory::H100).unwrap(),
            "\"H100\""
        );
        assert_eq!(
            serde_json::to_string(&GpuCategory::B200).unwrap(),
            "\"B200\""
        );
        assert_eq!(
            serde_json::to_string(&GpuCategory::B300).unwrap(),
            "\"B300\""
        );
    }

    #[test]
    fn serialize_other_variant() {
        let rtx = GpuCategory::Other("RTX6000".to_string());
        assert_eq!(serde_json::to_string(&rtx).unwrap(), "\"RTX6000\"");
    }

    #[test]
    fn deserialize_known_variants_from_string() {
        assert_eq!(
            serde_json::from_str::<GpuCategory>("\"A100\"").unwrap(),
            GpuCategory::A100
        );
        assert_eq!(
            serde_json::from_str::<GpuCategory>("\"H100\"").unwrap(),
            GpuCategory::H100
        );
        assert_eq!(
            serde_json::from_str::<GpuCategory>("\"B200\"").unwrap(),
            GpuCategory::B200
        );
        assert_eq!(
            serde_json::from_str::<GpuCategory>("\"B300\"").unwrap(),
            GpuCategory::B300
        );
    }

    #[test]
    fn deserialize_other_from_string() {
        let rtx: GpuCategory = serde_json::from_str("\"RTX6000\"").unwrap();
        assert_eq!(rtx, GpuCategory::Other("RTX6000".to_string()));
    }

    #[test]
    fn deserialize_legacy_tagged_format() {
        // Backward compat: old serde externally-tagged format
        let rtx: GpuCategory = serde_json::from_str("{\"Other\":\"RTX6000\"}").unwrap();
        assert_eq!(rtx, GpuCategory::Other("RTX6000".to_string()));
    }

    #[test]
    fn deserialize_legacy_tagged_known_gpu() {
        // If legacy format wraps a known GPU name, parse it
        let h100: GpuCategory = serde_json::from_str("{\"Other\":\"H100\"}").unwrap();
        assert_eq!(h100, GpuCategory::H100);
    }

    #[test]
    fn roundtrip_in_struct() {
        #[derive(Debug, Serialize, Deserialize, PartialEq)]
        struct Offering {
            gpu_type: GpuCategory,
            count: u32,
        }

        let original = Offering {
            gpu_type: GpuCategory::Other("L40S".to_string()),
            count: 4,
        };
        let json = serde_json::to_string(&original).unwrap();
        assert!(json.contains("\"gpu_type\":\"L40S\""));

        let deserialized: Offering = serde_json::from_str(&json).unwrap();
        assert_eq!(original, deserialized);
    }

    #[test]
    fn roundtrip_legacy_format_in_struct() {
        // Simulates what old API server sends
        let json = r#"{"gpu_type":{"Other":"RTX6000"},"count":2}"#;

        #[derive(Debug, Deserialize, PartialEq)]
        struct Offering {
            gpu_type: GpuCategory,
            count: u32,
        }

        let parsed: Offering = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.gpu_type, GpuCategory::Other("RTX6000".to_string()));
        assert_eq!(parsed.count, 2);
    }
}

/// Compute category for marketplace differentiation
///
/// Distinguishes between datacenter providers (The Citadel) and miner-provided GPUs (The Bourse).
/// Used for routing compute requests to appropriate infrastructure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputeCategory {
    /// The Citadel - Datacenter providers (aggregator service)
    /// Examples: Verda, Hyperstack, Lambda Labs, HydraHost
    SecureCloud,
    /// The Bourse - Miner-provided GPUs (validator-mediated)
    /// Bittensor subnet miners providing compute resources
    CommunityCloud,
}

impl ComputeCategory {
    /// Get the display name for this compute category
    pub fn as_str(&self) -> &'static str {
        match self {
            ComputeCategory::SecureCloud => "secure-cloud",
            ComputeCategory::CommunityCloud => "community-cloud",
        }
    }

    /// Get a human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            ComputeCategory::SecureCloud => "The Citadel - Datacenter providers",
            ComputeCategory::CommunityCloud => "The Bourse - Miner-provided GPUs",
        }
    }
}

impl fmt::Display for ComputeCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl FromStr for ComputeCategory {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "secure-cloud" | "secure_cloud" | "secure" => Ok(ComputeCategory::SecureCloud),
            "community-cloud" | "community_cloud" | "community" => {
                Ok(ComputeCategory::CommunityCloud)
            }
            _ => Err(format!(
                "Invalid compute category: {}. Expected 'secure-cloud' or 'community-cloud'",
                s
            )),
        }
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

    #[test]
    fn test_compute_category_serialization() {
        use serde_json;

        // Serialize SecureCloud
        let secure = ComputeCategory::SecureCloud;
        let serialized = serde_json::to_string(&secure).unwrap();
        assert_eq!(serialized, "\"secure_cloud\"");

        // Serialize CommunityCloud
        let community = ComputeCategory::CommunityCloud;
        let serialized = serde_json::to_string(&community).unwrap();
        assert_eq!(serialized, "\"community_cloud\"");

        // Deserialize secure_cloud
        let deserialized: ComputeCategory = serde_json::from_str("\"secure_cloud\"").unwrap();
        assert_eq!(deserialized, ComputeCategory::SecureCloud);

        // Deserialize community_cloud
        let deserialized: ComputeCategory = serde_json::from_str("\"community_cloud\"").unwrap();
        assert_eq!(deserialized, ComputeCategory::CommunityCloud);
    }

    #[test]
    fn test_compute_category_from_str() {
        // Test various input formats for SecureCloud
        assert_eq!(
            "secure-cloud".parse::<ComputeCategory>().unwrap(),
            ComputeCategory::SecureCloud
        );
        assert_eq!(
            "secure_cloud".parse::<ComputeCategory>().unwrap(),
            ComputeCategory::SecureCloud
        );
        assert_eq!(
            "secure".parse::<ComputeCategory>().unwrap(),
            ComputeCategory::SecureCloud
        );
        assert_eq!(
            "SECURE-CLOUD".parse::<ComputeCategory>().unwrap(),
            ComputeCategory::SecureCloud
        );

        // Test various input formats for CommunityCloud
        assert_eq!(
            "community-cloud".parse::<ComputeCategory>().unwrap(),
            ComputeCategory::CommunityCloud
        );
        assert_eq!(
            "community_cloud".parse::<ComputeCategory>().unwrap(),
            ComputeCategory::CommunityCloud
        );
        assert_eq!(
            "community".parse::<ComputeCategory>().unwrap(),
            ComputeCategory::CommunityCloud
        );
        assert_eq!(
            "COMMUNITY-CLOUD".parse::<ComputeCategory>().unwrap(),
            ComputeCategory::CommunityCloud
        );

        // Test invalid input
        let result = "invalid".parse::<ComputeCategory>();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid compute category"));
    }

    #[test]
    fn test_compute_category_display() {
        assert_eq!(ComputeCategory::SecureCloud.to_string(), "secure-cloud");
        assert_eq!(
            ComputeCategory::CommunityCloud.to_string(),
            "community-cloud"
        );

        assert_eq!(ComputeCategory::SecureCloud.as_str(), "secure-cloud");
        assert_eq!(ComputeCategory::CommunityCloud.as_str(), "community-cloud");

        assert_eq!(
            ComputeCategory::SecureCloud.description(),
            "The Citadel - Datacenter providers"
        );
        assert_eq!(
            ComputeCategory::CommunityCloud.description(),
            "The Bourse - Miner-provided GPUs"
        );
    }
}

/// Cloud provider identifier for GPU offerings
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CloudProvider {
    Hyperstack,
    Lambda,
    HydraHost,
    /// Verda cloud provider (replacement for DataCrunch)
    Verda,
    /// Mass Compute - VM-based GPU rental provider (polling-based, no webhooks)
    MassCompute,
    /// The Priory - VIP managed machines (not a real cloud provider, but uses Deployment model)
    Vip,
}

impl CloudProvider {
    pub fn as_str(&self) -> &'static str {
        match self {
            CloudProvider::Hyperstack => "hyperstack",
            CloudProvider::Lambda => "lambda",
            CloudProvider::HydraHost => "hydrahost",
            CloudProvider::Verda => "verda",
            CloudProvider::MassCompute => "masscompute",
            CloudProvider::Vip => "vip",
        }
    }
}

impl fmt::Display for CloudProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl FromStr for CloudProvider {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "datacrunch" => Ok(CloudProvider::Verda), // DataCrunch replaced by Verda
            "hyperstack" => Ok(CloudProvider::Hyperstack),
            "lambda" => Ok(CloudProvider::Lambda),
            "hydrahost" => Ok(CloudProvider::HydraHost),
            "verda" => Ok(CloudProvider::Verda),
            "masscompute" | "mass_compute" | "mass-compute" => Ok(CloudProvider::MassCompute),
            "vip" => Ok(CloudProvider::Vip),
            _ => Err(format!("Unknown provider: {}", s)),
        }
    }
}

/// Unified GPU offering structure for marketplace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuOffering {
    pub id: String,
    pub provider: CloudProvider,
    pub gpu_type: GpuCategory,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_memory_gb_per_gpu: Option<u32>,
    pub gpu_count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interconnect: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub storage: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deployment_type: Option<String>,
    pub system_memory_gb: u32,
    pub vcpu_count: u32,
    pub region: String,
    #[serde(with = "rust_decimal::serde::str")]
    pub hourly_rate_per_gpu: Decimal,
    pub availability: bool,
    #[serde(default)]
    pub is_spot: bool,
    pub fetched_at: DateTime<Utc>,
    #[serde(skip)]
    pub raw_metadata: serde_json::Value,
}

#[cfg(test)]
mod cloud_provider_tests {
    use super::*;

    #[test]
    fn test_cloud_provider_as_str() {
        assert_eq!(CloudProvider::Hyperstack.as_str(), "hyperstack");
        assert_eq!(CloudProvider::Lambda.as_str(), "lambda");
        assert_eq!(CloudProvider::HydraHost.as_str(), "hydrahost");
        assert_eq!(CloudProvider::Verda.as_str(), "verda");
        assert_eq!(CloudProvider::MassCompute.as_str(), "masscompute");
        assert_eq!(CloudProvider::Vip.as_str(), "vip");
    }

    #[test]
    fn test_cloud_provider_display() {
        assert_eq!(CloudProvider::MassCompute.to_string(), "masscompute");
        assert_eq!(CloudProvider::Hyperstack.to_string(), "hyperstack");
    }

    #[test]
    fn test_cloud_provider_from_str() {
        assert_eq!(
            "masscompute".parse::<CloudProvider>().unwrap(),
            CloudProvider::MassCompute
        );
        assert_eq!(
            "mass_compute".parse::<CloudProvider>().unwrap(),
            CloudProvider::MassCompute
        );
        assert_eq!(
            "mass-compute".parse::<CloudProvider>().unwrap(),
            CloudProvider::MassCompute
        );
        assert_eq!(
            "MassCompute".parse::<CloudProvider>().unwrap(),
            CloudProvider::MassCompute
        );
        assert_eq!(
            "MASSCOMPUTE".parse::<CloudProvider>().unwrap(),
            CloudProvider::MassCompute
        );
    }

    #[test]
    fn test_cloud_provider_from_str_existing() {
        assert_eq!(
            "hyperstack".parse::<CloudProvider>().unwrap(),
            CloudProvider::Hyperstack
        );
        assert_eq!(
            "lambda".parse::<CloudProvider>().unwrap(),
            CloudProvider::Lambda
        );
        assert_eq!(
            "hydrahost".parse::<CloudProvider>().unwrap(),
            CloudProvider::HydraHost
        );
        assert_eq!(
            "verda".parse::<CloudProvider>().unwrap(),
            CloudProvider::Verda
        );
        assert_eq!(
            "datacrunch".parse::<CloudProvider>().unwrap(),
            CloudProvider::Verda
        );
        assert_eq!("vip".parse::<CloudProvider>().unwrap(), CloudProvider::Vip);
    }

    #[test]
    fn test_cloud_provider_from_str_invalid() {
        assert!("unknown".parse::<CloudProvider>().is_err());
        assert!("".parse::<CloudProvider>().is_err());
    }

    #[test]
    fn test_cloud_provider_serde_roundtrip() {
        let provider = CloudProvider::MassCompute;
        let json = serde_json::to_string(&provider).unwrap();
        assert_eq!(json, "\"masscompute\"");

        let deserialized: CloudProvider = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, CloudProvider::MassCompute);
    }

    #[test]
    fn test_cloud_provider_serde_all_variants() {
        let variants = vec![
            (CloudProvider::Hyperstack, "\"hyperstack\""),
            (CloudProvider::Lambda, "\"lambda\""),
            (CloudProvider::HydraHost, "\"hydrahost\""),
            (CloudProvider::Verda, "\"verda\""),
            (CloudProvider::MassCompute, "\"masscompute\""),
            (CloudProvider::Vip, "\"vip\""),
        ];

        for (variant, expected_json) in variants {
            let json = serde_json::to_string(&variant).unwrap();
            assert_eq!(
                json, expected_json,
                "Serialization failed for {:?}",
                variant
            );

            let deserialized: CloudProvider = serde_json::from_str(&json).unwrap();
            assert_eq!(
                deserialized, variant,
                "Deserialization failed for {}",
                expected_json
            );
        }
    }

    #[test]
    fn test_gpu_offering_with_masscompute_provider() {
        let json = r#"{
            "id": "masscompute-8x-h100-us-east",
            "provider": "masscompute",
            "gpu_type": "H100",
            "gpu_memory_gb_per_gpu": 80,
            "gpu_count": 8,
            "interconnect": "NVLink",
            "system_memory_gb": 1800,
            "vcpu_count": 208,
            "region": "US-EAST",
            "hourly_rate_per_gpu": "2.49",
            "availability": true,
            "is_spot": false,
            "fetched_at": "2025-01-01T00:00:00Z"
        }"#;

        let offering: GpuOffering = serde_json::from_str(json).unwrap();
        assert_eq!(offering.provider, CloudProvider::MassCompute);
        assert_eq!(offering.gpu_type, GpuCategory::H100);
        assert_eq!(offering.gpu_count, 8);
        assert_eq!(offering.region, "US-EAST");
    }
}
