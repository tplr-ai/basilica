//! Identity types for Basilica
//!
//! This module defines the core identity types used throughout the system:
//! - `Hotkey`: Bittensor hotkey in SS58 format (re-exported from bittensor crate)
//! - `NodeId`: Unique identifier for compute nodes (from node_identity module)
//! - `ValidatorUid`: Bittensor validator UID
//! - `MinerUid`: Bittensor miner UID
//! - `JobId`: Unique identifier for computational jobs
//!
//! All types implement the standard traits needed for hashing, comparison,
//! serialization, and display formatting.

use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

// Import NodeId from node_identity module
pub use crate::node_identity::NodeId;

// Re-export Hotkey and AccountId from the bittensor crate as the canonical source
pub use bittensor::AccountId;
pub use bittensor::Hotkey;

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
/// - Used for tracking job state across nodes and miners
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

            // Verify it can be converted to AccountId (via bittensor)
            assert!(
                hotkey.to_account_id().is_ok(),
                "Valid hotkey should convert to AccountId: {hotkey_str}"
            );
        }
    }

    #[test]
    fn test_hotkey_creation_invalid_addresses() {
        // Test various invalid formats
        let invalid_hotkeys = vec![
            ("", "Empty hotkey should be rejected"),
            ("invalid", "Too short address should be rejected"),
            (
                "not_an_address_at_all",
                "Completely invalid format should be rejected",
            ),
        ];

        for (invalid_hotkey, reason) in invalid_hotkeys {
            let result = Hotkey::new(invalid_hotkey.to_string());
            assert!(result.is_err(), "{reason}: {invalid_hotkey}");
        }
    }

    #[test]
    fn test_node_id() {
        // NodeId from node_identity module requires a seed
        let id1 = NodeId::new("test-seed-1").unwrap();
        let id2 = NodeId::new("test-seed-2").unwrap();
        // Different seeds produce different IDs
        assert_ne!(id1.uuid, id2.uuid);

        // Same seed produces same ID (deterministic)
        let id3 = NodeId::new("test-seed-1").unwrap();
        assert_eq!(id1.uuid, id3.uuid);
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

        // Test conversion to AccountId and back (using bittensor types)
        let account_id = hotkey.to_account_id().unwrap();
        let converted_hotkey = Hotkey::from_account_id(&account_id);
        assert_eq!(hotkey.as_str(), converted_hotkey.as_str());
    }
}
