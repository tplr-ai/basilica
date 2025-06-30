//! Identity types for Basilca
//!
//! This module defines the core identity types used throughout the system:
//! - `Hotkey`: Bittensor hotkey in SS58 format
//! - `ExecutorId`: Unique identifier for executor agents
//! - `ValidatorUid`: Bittensor validator UID
//! - `MinerUid`: Bittensor miner UID
//! - `JobId`: Unique identifier for computational jobs
//!
//! All types implement the standard traits needed for hashing, comparison,
//! serialization, and display formatting.

use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

/// Bittensor hotkey identifier in SS58 format
///
/// # Implementation Notes
/// - Must validate SS58 format on construction
/// - Used for cryptographic operations and authentication
/// - Should be treated as a public key identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Hotkey(String);

impl Hotkey {
    /// Create a new Hotkey from a string
    ///
    /// # Arguments
    /// * `hotkey` - SS58 formatted string
    ///
    /// # Returns
    /// * `Result<Hotkey, String>` - Ok if valid SS58 format, Err with description otherwise
    ///
    /// # Implementation
    /// Validates SS58 format using sp-core::crypto::Ss58Codec for full compatibility
    /// with Substrate/Bittensor ecosystem. Checks format, checksum, and address type.
    pub fn new(hotkey: String) -> Result<Self, String> {
        use sp_core::crypto::Ss58Codec;
        use std::str::FromStr;

        // Basic validation
        if hotkey.is_empty() {
            return Err("Hotkey cannot be empty".to_string());
        }

        // Length check for SS58 addresses (typical range is 47-48 characters)
        if hotkey.len() < 47 || hotkey.len() > 48 {
            return Err(format!(
                "Invalid hotkey length: expected 47-48 characters, got {}",
                hotkey.len()
            ));
        }

        // Validate SS58 format using sp-core
        // This validates format, checksum, and ensures it's a valid SS58 address
        match sp_core::sr25519::Public::from_ss58check(&hotkey) {
            Ok(_) => {
                // Additional validation: ensure it can be converted to AccountId
                // This provides compatibility with crabtensor expectations
                match crabtensor::AccountId::from_str(&hotkey) {
                    Ok(_) => Ok(Hotkey(hotkey)),
                    Err(_) => Err(format!(
                        "Hotkey is valid SS58 but incompatible with Bittensor network: {hotkey}"
                    )),
                }
            }
            Err(_) => {
                // Try with AccountId32 format for broader compatibility
                match sp_core::crypto::AccountId32::from_ss58check(&hotkey) {
                    Ok(_) => {
                        // Verify crabtensor compatibility
                        match crabtensor::AccountId::from_str(&hotkey) {
                            Ok(_) => Ok(Hotkey(hotkey)),
                            Err(_) => Err(format!(
                                "Hotkey is valid SS58 but incompatible with Bittensor network: {hotkey}"
                            )),
                        }
                    }
                    Err(_) => Err(format!(
                        "Invalid SS58 format: checksum validation failed for {hotkey}"
                    )),
                }
            }
        }
    }

    /// Get the inner string representation
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Convert to owned String
    pub fn into_string(self) -> String {
        self.0
    }
}

impl fmt::Display for Hotkey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::str::FromStr for Hotkey {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::new(s.to_string())
    }
}

// Conversion methods to/from crabtensor::AccountId
impl Hotkey {
    /// Convert Hotkey to crabtensor::AccountId
    ///
    /// # Returns
    /// * `Result<crabtensor::AccountId, String>` - Ok if valid SS58, Err otherwise
    ///
    /// # Implementation Notes
    /// Uses crabtensor's built-in SS58 parsing to convert the hotkey string
    /// to an AccountId that can be used with the crabtensor API.
    pub fn to_account_id(&self) -> Result<crabtensor::AccountId, String> {
        use std::str::FromStr;
        crabtensor::AccountId::from_str(&self.0)
            .map_err(|e| format!("Failed to parse hotkey as AccountId: {e}"))
    }

    /// Create Hotkey from crabtensor::AccountId
    ///
    /// # Arguments
    /// * `account_id` - The AccountId to convert
    ///
    /// # Returns
    /// * `Self` - Hotkey containing the SS58 representation
    pub fn from_account_id(account_id: &crabtensor::AccountId) -> Self {
        Hotkey(account_id.to_string())
    }
}

// Note: From<crabtensor::AccountId> for Hotkey conflicts with blanket From<T> for T
// Use explicit from_account_id() method instead

// Note: TryFrom<&Hotkey> for crabtensor::AccountId conflicts with blanket TryFrom
// Use explicit to_account_id() method instead

/// Unique identifier for executor agents
///
/// # Implementation Notes
/// - Uses UUID v4 for global uniqueness
/// - Generated when executor agent starts up
/// - Persisted in executor's local database
/// - Used for tracking executor state across restarts
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExecutorId(Uuid);

impl ExecutorId {
    /// Generate a new random ExecutorId
    pub fn new() -> Self {
        ExecutorId(Uuid::new_v4())
    }

    /// Create ExecutorId from existing UUID
    pub fn from_uuid(uuid: Uuid) -> Self {
        ExecutorId(uuid)
    }

    /// Get the inner UUID
    pub fn as_uuid(&self) -> &Uuid {
        &self.0
    }

    /// Convert to UUID
    pub fn into_uuid(self) -> Uuid {
        self.0
    }
}

impl Default for ExecutorId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ExecutorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::str::FromStr for ExecutorId {
    type Err = uuid::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(ExecutorId(Uuid::from_str(s)?))
    }
}

/// Bittensor validator unique identifier
///
/// # Implementation Notes
/// - u16 as per Bittensor protocol specifications
/// - Range typically 0-4095 depending on subnet configuration
/// - Used in metagraph operations and weight setting
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ValidatorUid(u16);

impl ValidatorUid {
    /// Create a new ValidatorUid
    ///
    /// # Arguments
    /// * `uid` - The validator UID (0-4095 typically)
    pub fn new(uid: u16) -> Self {
        ValidatorUid(uid)
    }

    /// Get the inner u16 value
    pub fn as_u16(&self) -> u16 {
        self.0
    }

    /// Convert to u16
    pub fn into_u16(self) -> u16 {
        self.0
    }
}

impl fmt::Display for ValidatorUid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<u16> for ValidatorUid {
    fn from(uid: u16) -> Self {
        ValidatorUid(uid)
    }
}

impl From<ValidatorUid> for u16 {
    fn from(uid: ValidatorUid) -> u16 {
        uid.0
    }
}

/// Bittensor miner unique identifier
///
/// # Implementation Notes
/// - u16 as per Bittensor protocol specifications
/// - Range typically 0-4095 depending on subnet configuration
/// - Used in metagraph operations and scoring
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MinerUid(u16);

impl MinerUid {
    /// Create a new MinerUid
    ///
    /// # Arguments
    /// * `uid` - The miner UID (0-4095 typically)
    pub fn new(uid: u16) -> Self {
        MinerUid(uid)
    }

    /// Get the inner u16 value
    pub fn as_u16(&self) -> u16 {
        self.0
    }

    /// Convert to u16
    pub fn into_u16(self) -> u16 {
        self.0
    }
}

impl fmt::Display for MinerUid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<u16> for MinerUid {
    fn from(uid: u16) -> Self {
        MinerUid(uid)
    }
}

impl From<MinerUid> for u16 {
    fn from(uid: MinerUid) -> u16 {
        uid.0
    }
}

/// Unique identifier for computational jobs
///
/// # Implementation Notes
/// - Uses UUID v4 for global uniqueness
/// - Generated when job is created
/// - Used for tracking job state across executors and miners
/// - Persisted in verification logs and job databases
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct JobId(Uuid);

impl JobId {
    /// Generate a new random JobId
    pub fn new() -> Self {
        JobId(Uuid::new_v4())
    }

    /// Create JobId from existing UUID
    pub fn from_uuid(uuid: Uuid) -> Self {
        JobId(uuid)
    }

    /// Get the inner UUID
    pub fn as_uuid(&self) -> &Uuid {
        &self.0
    }

    /// Convert to UUID
    pub fn into_uuid(self) -> Uuid {
        self.0
    }
}

impl Default for JobId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for JobId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::str::FromStr for JobId {
    type Err = uuid::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(JobId(Uuid::from_str(s)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hotkey_creation_valid_addresses() {
        // Test known valid SS58 addresses
        let valid_hotkeys = vec![
            "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            "5DAAnrj7VHTznn2AWBemMuyBwZWs6FNFjdyVXUeYum3PTXFy",
            "5HGjWAeFDfFCWPsjFQdVV2Msvz2XtMktvgocEZcCj68kUMaw",
        ];

        for hotkey_str in valid_hotkeys {
            let hotkey = Hotkey::new(hotkey_str.to_string());
            assert!(
                hotkey.is_ok(),
                "Failed to create hotkey from valid SS58 address: {hotkey_str}"
            );

            // Verify round-trip conversion
            let hotkey = hotkey.unwrap();
            assert_eq!(hotkey.as_str(), hotkey_str);

            // Verify it can be converted to AccountId
            assert!(
                hotkey.to_account_id().is_ok(),
                "Valid hotkey should convert to AccountId: {hotkey_str}"
            );
        }
    }

    #[test]
    fn test_hotkey_creation_invalid_addresses() {
        // Test various invalid formats
        let long_address = format!("5{}", "a".repeat(50));
        let invalid_hotkeys = vec![
            ("", "Empty hotkey should be rejected"),
            ("invalid", "Too short address should be rejected"),
            (
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQ",
                "Too short valid-looking address should be rejected",
            ),
            (
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQYZ",
                "Too long address should be rejected",
            ),
            (
                "1GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                "Invalid SS58 prefix should be rejected",
            ),
            (
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQZ",
                "Invalid checksum should be rejected",
            ),
            (
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQ0",
                "Invalid base58 character '0' should be rejected",
            ),
            (
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQO",
                "Invalid base58 character 'O' should be rejected",
            ),
            (
                "not_an_address_at_all",
                "Completely invalid format should be rejected",
            ),
            (
                long_address.as_str(),
                "Address with wrong length should be rejected",
            ),
        ];

        for (invalid_hotkey, reason) in invalid_hotkeys {
            let result = Hotkey::new(invalid_hotkey.to_string());
            assert!(result.is_err(), "{reason}: {invalid_hotkey}");
        }
    }

    #[test]
    fn test_hotkey_error_messages() {
        // Test specific error message content
        let empty_result = Hotkey::new("".to_string());
        assert!(empty_result.is_err());
        assert!(empty_result.unwrap_err().contains("cannot be empty"));

        let short_result = Hotkey::new("short".to_string());
        assert!(short_result.is_err());
        assert!(short_result.unwrap_err().contains("Invalid hotkey length"));

        let invalid_checksum_result =
            Hotkey::new("5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQZ".to_string());
        assert!(invalid_checksum_result.is_err());
        assert!(invalid_checksum_result
            .unwrap_err()
            .contains("checksum validation failed"));
    }

    #[test]
    fn test_hotkey_edge_cases() {
        // Test boundary conditions for length
        let min_length = "a".repeat(47);
        let max_length = "a".repeat(48);
        let too_short = "a".repeat(46);
        let too_long = "a".repeat(49);

        // These should fail because they're not valid SS58, but they pass length check
        assert!(Hotkey::new(too_short).is_err());
        assert!(Hotkey::new(too_long).is_err());
        assert!(Hotkey::new(min_length).is_err()); // Invalid SS58 format
        assert!(Hotkey::new(max_length).is_err()); // Invalid SS58 format

        // Test whitespace handling
        let hotkey_with_spaces = " 5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY ";
        assert!(Hotkey::new(hotkey_with_spaces.to_string()).is_err());

        // Test newline handling
        let hotkey_with_newline = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY\n";
        assert!(Hotkey::new(hotkey_with_newline.to_string()).is_err());
    }

    #[test]
    fn test_executor_id() {
        let id1 = ExecutorId::new();
        let id2 = ExecutorId::new();
        assert_ne!(id1, id2); // Should be unique

        let uuid = uuid::Uuid::new_v4();
        let id3 = ExecutorId::from_uuid(uuid);
        assert_eq!(id3.as_uuid(), &uuid);
    }

    #[test]
    fn test_validator_uid() {
        let uid = ValidatorUid::new(42);
        assert_eq!(uid.as_u16(), 42);
        assert_eq!(uid.to_string(), "42");

        let uid_from_u16: ValidatorUid = 100u16.into();
        assert_eq!(uid_from_u16.as_u16(), 100);
    }

    #[test]
    fn test_miner_uid() {
        let uid = MinerUid::new(123);
        assert_eq!(uid.as_u16(), 123);
        assert_eq!(uid.to_string(), "123");

        let uid_from_u16: MinerUid = 456u16.into();
        assert_eq!(uid_from_u16.as_u16(), 456);
    }

    #[test]
    fn test_job_id() {
        let id1 = JobId::new();
        let id2 = JobId::new();
        assert_ne!(id1, id2); // Should be unique

        let uuid = uuid::Uuid::new_v4();
        let id3 = JobId::from_uuid(uuid);
        assert_eq!(id3.as_uuid(), &uuid);
    }

    #[test]
    fn test_serialization() {
        let executor_id = ExecutorId::new();
        let serialized = serde_json::to_string(&executor_id).unwrap();
        let deserialized: ExecutorId = serde_json::from_str(&serialized).unwrap();
        assert_eq!(executor_id, deserialized);
    }

    #[test]
    fn test_job_id_serialization() {
        let job_id = JobId::new();
        let serialized = serde_json::to_string(&job_id).unwrap();
        let deserialized: JobId = serde_json::from_str(&serialized).unwrap();
        assert_eq!(job_id, deserialized);
    }

    #[test]
    fn test_hotkey_account_id_conversion() {
        // Test valid SS58 address
        let valid_hotkey = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY";
        let hotkey = Hotkey::new(valid_hotkey.to_string()).unwrap();

        // Test conversion to AccountId and back
        if let Ok(account_id) = hotkey.to_account_id() {
            let converted_hotkey = Hotkey::from_account_id(&account_id);
            assert_eq!(hotkey.as_str(), converted_hotkey.as_str());

            // Test roundtrip conversion
            let account_id2 = converted_hotkey.to_account_id().unwrap();
            assert_eq!(account_id, account_id2);
        }
    }
}
