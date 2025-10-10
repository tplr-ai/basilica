//! Shared authentication test utilities and mock types
//!
//! This module provides common mock implementations and helper functions
//! for testing authentication functionality across integration tests.

use anyhow::Result;
use basilica_node::miner_auth;
use basilica_miner::node_auth;
use basilica_protocol::common::MinerAuthentication;
use chrono::{Duration, Utc};

/// Mock Bittensor service for testing
#[derive(Clone)]
pub struct MockBittensorService {
    account_id: String,
}

impl MockBittensorService {
    pub fn new(account_id: &str) -> Self {
        Self {
            account_id: account_id.to_string(),
        }
    }

    pub fn get_account_id(&self) -> &str {
        &self.account_id
    }

    pub fn sign_data(&self, data: &[u8]) -> Result<String> {
        // Create a deterministic "signature" for testing
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        self.account_id.hash(&mut hasher);
        let hash = hasher.finish();

        // Return hex encoded "signature" like the real service
        Ok(hex::encode(hash.to_be_bytes()))
    }
}

/// Mock node auth service that wraps the functionality for testing
#[derive(Clone)]
pub struct MockNodeAuthService {
    miner_hotkey: String,
    bittensor_service: MockBittensorService,
}

impl MockNodeAuthService {
    pub fn new(miner_hotkey: &str) -> Self {
        Self {
            miner_hotkey: miner_hotkey.to_string(),
            bittensor_service: MockBittensorService::new(miner_hotkey),
        }
    }

    pub fn create_auth(&self, request_data: &[u8]) -> Result<MinerAuthentication> {
        let timestamp_ms = Utc::now().timestamp_millis() as u64;
        let nonce = uuid::Uuid::new_v4().to_string();
        let request_id = uuid::Uuid::new_v4().to_string();

        // Create canonical data to sign
        let canonical_data = node_auth::create_canonical_data(
            &self.miner_hotkey,
            timestamp_ms,
            &nonce,
            &request_id,
            request_data,
        );

        // Sign the canonical data
        let signature = self
            .bittensor_service
            .sign_data(canonical_data.as_bytes())?;

        // Convert signature from hex string to bytes (matching the expected format)
        let signature_bytes = hex::decode(&signature)
            .map_err(|e| anyhow::anyhow!("Failed to decode signature: {}", e))?;

        Ok(MinerAuthentication {
            miner_hotkey: self.miner_hotkey.clone(),
            timestamp_ms,
            nonce: nonce.into_bytes(),
            signature: signature_bytes,
            request_id: request_id.into_bytes(),
        })
    }

    pub fn get_miner_hotkey(&self) -> &str {
        &self.miner_hotkey
    }
}

/// Test helper to create a valid authentication using mock service
pub fn create_valid_auth(miner_hotkey: &str, request_data: &[u8]) -> Result<MinerAuthentication> {
    let auth_service = MockNodeAuthService::new(miner_hotkey);
    auth_service.create_auth(request_data)
}

/// Test helper to create miner auth service with default config (signature verification disabled)
pub fn create_miner_auth_service(expected_miner_hotkey: &str) -> miner_auth::MinerAuthService {
    create_miner_auth_service_with_config(expected_miner_hotkey, Duration::minutes(5), false)
}

/// Test helper to create miner auth service with custom configuration
pub fn create_miner_auth_service_with_config(
    expected_miner_hotkey: &str,
    max_request_age: Duration,
    verify_signatures: bool,
) -> miner_auth::MinerAuthService {
    let config = miner_auth::MinerAuthConfig {
        managing_miner_hotkey: basilica_common::identity::Hotkey::new(
            expected_miner_hotkey.to_string(),
        )
        .unwrap(),
        max_request_age,
        verify_signatures,
    };
    miner_auth::MinerAuthService::new(config)
}

/// Common test hotkeys for consistency across tests
pub mod test_hotkeys {
    pub const MINER_HOTKEY_1: &str = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY";
    pub const MINER_HOTKEY_2: &str = "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM";
    pub const VALIDATOR_HOTKEY_1: &str = "5DAAnrj7VHTznn2AWBemMuyBwZWs6FNFjdyVXUeYum3PTXFy";
    pub const VALIDATOR_HOTKEY_2: &str = "5HGjWAeFDfFCWPsjFQdVV2Msvz2XtMktvgocEZcCj68kUMaw";
}

/// Helper to create a test authentication with custom parameters
pub fn create_test_auth(
    miner_hotkey: &str,
    timestamp_ms: Option<u64>,
    nonce: Option<Vec<u8>>,
    request_id: Option<Vec<u8>>,
) -> MinerAuthentication {
    MinerAuthentication {
        miner_hotkey: miner_hotkey.to_string(),
        timestamp_ms: timestamp_ms.unwrap_or_else(|| Utc::now().timestamp_millis() as u64),
        nonce: nonce.unwrap_or_else(|| uuid::Uuid::new_v4().to_string().into_bytes()),
        signature: b"test_signature".to_vec(),
        request_id: request_id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string().into_bytes()),
    }
}

/// Helper to create a test authentication with corrupted signature
pub fn create_test_auth_with_bad_signature(miner_hotkey: &str) -> MinerAuthentication {
    MinerAuthentication {
        miner_hotkey: miner_hotkey.to_string(),
        timestamp_ms: Utc::now().timestamp_millis() as u64,
        nonce: uuid::Uuid::new_v4().to_string().into_bytes(),
        signature: b"invalid_signature".to_vec(),
        request_id: uuid::Uuid::new_v4().to_string().into_bytes(),
    }
}

/// Helper to verify authentication result patterns
pub fn assert_auth_error_contains(result: &Result<(), anyhow::Error>, expected_message: &str) {
    assert!(result.is_err(), "Expected authentication to fail");
    let error_msg = result.as_ref().unwrap_err().to_string();
    assert!(
        error_msg.contains(expected_message),
        "Expected error message to contain '{expected_message}', got: {error_msg}"
    );
}

/// Helper to verify gRPC status error patterns
pub fn assert_grpc_error_contains(result: &Result<(), tonic::Status>, expected_message: &str) {
    assert!(result.is_err(), "Expected gRPC call to fail");
    let error_msg = result.as_ref().unwrap_err().message();
    assert!(
        error_msg.contains(expected_message),
        "Expected error message to contain '{expected_message}', got: {error_msg}"
    );
}

/// Creates an authenticated request by serializing the request without auth field,
/// creating authentication based on that data, and then adding auth to the request.
/// This prevents the circular dependency issue where auth field is included in signature.
pub fn create_authenticated_request<T>(request: T, miner_hotkey: &str) -> Result<T>
where
    T: prost::Message + node_auth::AuthenticatedRequest + Clone,
{
    // Serialize request without auth field for signature calculation
    let request_bytes = request.encode_to_vec();

    // Create authentication based on request data without auth
    let auth = create_valid_auth(miner_hotkey, &request_bytes)?;

    // Add auth to request and return
    Ok(request.with_auth(auth))
}

/// Creates an authenticated request with expired timestamp for testing
pub fn create_expired_authenticated_request<T>(
    request: T,
    miner_hotkey: &str,
    hours_ago: i64,
) -> Result<T>
where
    T: prost::Message + node_auth::AuthenticatedRequest + Clone,
{
    let _request_bytes = request.encode_to_vec();

    // Create expired auth
    let expired_timestamp = (Utc::now() - Duration::hours(hours_ago)).timestamp_millis() as u64;
    let auth = create_test_auth(miner_hotkey, Some(expired_timestamp), None, None);

    Ok(request.with_auth(auth))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_bittensor_service_deterministic() {
        let service = MockBittensorService::new("test_account");
        let data = b"test data";

        let sig1 = service.sign_data(data).unwrap();
        let sig2 = service.sign_data(data).unwrap();

        assert_eq!(sig1, sig2, "Signatures should be deterministic");
    }

    #[test]
    fn test_mock_bittensor_service_different_data() {
        let service = MockBittensorService::new("test_account");
        let data1 = b"test data 1";
        let data2 = b"test data 2";

        let sig1 = service.sign_data(data1).unwrap();
        let sig2 = service.sign_data(data2).unwrap();

        assert_ne!(
            sig1, sig2,
            "Different data should produce different signatures"
        );
    }

    #[test]
    fn test_mock_node_auth_service() {
        let service = MockNodeAuthService::new("test_miner");
        let request_data = b"test request";

        let auth = service.create_auth(request_data).unwrap();

        assert_eq!(auth.miner_hotkey, "test_miner");
        assert!(!auth.nonce.is_empty());
        assert!(!auth.signature.is_empty());
        assert!(!auth.request_id.is_empty());
        assert!(auth.timestamp_ms > 0);
    }

    #[test]
    fn test_create_test_auth_with_defaults() {
        let auth = create_test_auth("test_miner", None, None, None);

        assert_eq!(auth.miner_hotkey, "test_miner");
        assert!(!auth.nonce.is_empty());
        assert_eq!(auth.signature, b"test_signature");
        assert!(!auth.request_id.is_empty());
        assert!(auth.timestamp_ms > 0);
    }

    #[test]
    fn test_create_test_auth_with_custom_values() {
        let custom_nonce = "550e8400-e29b-41d4-a716-446655440000"
            .to_string()
            .into_bytes(); // Valid UUID
        let custom_timestamp = 1234567890u64;
        let custom_request_id = b"custom_request_id".to_vec();

        let auth = create_test_auth(
            "test_miner",
            Some(custom_timestamp),
            Some(custom_nonce.clone()),
            Some(custom_request_id.clone()),
        );

        assert_eq!(auth.miner_hotkey, "test_miner");
        assert_eq!(auth.nonce, custom_nonce);
        assert_eq!(auth.timestamp_ms, custom_timestamp);
        assert_eq!(auth.request_id, custom_request_id);
    }

    #[test]
    fn test_test_hotkeys_constants() {
        // Ensure they're all different
        let hotkeys = [
            test_hotkeys::MINER_HOTKEY_1,
            test_hotkeys::MINER_HOTKEY_2,
            test_hotkeys::VALIDATOR_HOTKEY_1,
            test_hotkeys::VALIDATOR_HOTKEY_2,
        ];

        for (i, hotkey1) in hotkeys.iter().enumerate() {
            for (j, hotkey2) in hotkeys.iter().enumerate() {
                if i != j {
                    assert_ne!(hotkey1, hotkey2, "Hotkeys should be unique");
                }
            }
        }
    }
}
