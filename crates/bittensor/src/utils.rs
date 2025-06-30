//! # Bittensor Utilities
//!
//! Helper functions for converting between common types and crabtensor types.

use crate::error::BittensorError;
use common::identity::Hotkey;
use crabtensor::AccountId;
use std::str::FromStr;

/// Convert a Hotkey to a crabtensor AccountId
pub fn hotkey_to_account_id(hotkey: &Hotkey) -> Result<AccountId, BittensorError> {
    AccountId::from_str(hotkey.as_str()).map_err(|_| BittensorError::InvalidHotkey {
        hotkey: hotkey.as_str().to_string(),
    })
}

/// Convert a crabtensor AccountId to a Hotkey
pub fn account_id_to_hotkey(account_id: &AccountId) -> Result<Hotkey, BittensorError> {
    Hotkey::from_str(&account_id.to_string()).map_err(|_| BittensorError::InvalidHotkey {
        hotkey: account_id.to_string(),
    })
}

// Re-export weight-related types from crabtensor
pub use crabtensor::weights::{normalize_weights, set_weights_payload, NormalizedWeight};

/// Convert stake from TAO to RAO (1 TAO = 10^9 RAO)
pub fn tao_to_rao(tao: f64) -> u64 {
    (tao * 1_000_000_000.0) as u64
}

/// Convert stake from RAO to TAO (1 TAO = 10^9 RAO)
pub fn rao_to_tao(rao: u64) -> f64 {
    rao as f64 / 1_000_000_000.0
}

/// Verify Bittensor signature using crabtensor
pub fn verify_bittensor_signature(
    hotkey: &Hotkey,
    signature_hex: &str,
    data: &[u8],
) -> Result<(), BittensorError> {
    // Validate inputs
    if signature_hex.is_empty() {
        return Err(BittensorError::InvalidHotkey {
            hotkey: "Empty signature".to_string(),
        });
    }

    if data.is_empty() {
        return Err(BittensorError::AuthError {
            message: "Empty data".to_string(),
        });
    }

    // Decode signature from hex
    let signature_bytes = hex::decode(signature_hex).map_err(|e| BittensorError::AuthError {
        message: format!("Invalid hex signature format: {e}"),
    })?;

    // Convert to AccountId
    let account_id = hotkey_to_account_id(hotkey)?;

    // Convert signature bytes to the expected type for crabtensor
    if signature_bytes.len() != 64 {
        return Err(BittensorError::AuthError {
            message: format!(
                "Invalid signature length: expected 64 bytes, got {}",
                signature_bytes.len()
            ),
        });
    }

    // Convert Vec<u8> to fixed-size array for signature
    let mut signature_array = [0u8; 64];
    signature_array.copy_from_slice(&signature_bytes);

    // Use sp_core types that crabtensor expects
    let signature = subxt::ext::sp_core::sr25519::Signature::from_raw(signature_array);

    // Use crabtensor to verify the signature
    let is_valid = crabtensor::sign::verify_signature(&account_id, &signature, data);

    if is_valid {
        Ok(())
    } else {
        Err(BittensorError::AuthError {
            message: "Signature verification failed".to_string(),
        })
    }
}

/// Create a signature using crabtensor signer
pub fn create_signature(signer: &crabtensor::wallet::Signer, data: &[u8]) -> String {
    let signature = crabtensor::sign::sign_message(signer, data);
    hex::encode(signature.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: normalize_weights tests are removed since we're using crabtensor's implementation directly

    #[test]
    fn test_tao_rao_conversion() {
        assert_eq!(tao_to_rao(1.0), 1_000_000_000);
        assert_eq!(rao_to_tao(1_000_000_000), 1.0);
        assert_eq!(tao_to_rao(0.5), 500_000_000);
        assert_eq!(rao_to_tao(500_000_000), 0.5);
    }

    #[test]
    fn test_hotkey_account_id_conversion() {
        // Test with a valid hotkey format
        let hotkey_str = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY";
        let hotkey = Hotkey::new(hotkey_str.to_string()).unwrap();

        // Test conversion to AccountId and back
        if let Ok(account_id) = hotkey_to_account_id(&hotkey) {
            let converted_hotkey = account_id_to_hotkey(&account_id).unwrap();
            assert_eq!(hotkey.as_str(), converted_hotkey.as_str());
        }
    }

    #[test]
    fn test_signature_verification_inputs() {
        let hotkey =
            Hotkey::new("5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string()).unwrap();

        // Test empty signature
        let result = verify_bittensor_signature(&hotkey, "", b"data");
        assert!(result.is_err());

        // Test empty data
        let result = verify_bittensor_signature(&hotkey, "deadbeef".repeat(16).as_str(), b"");
        assert!(result.is_err());

        // Test invalid hex
        let result = verify_bittensor_signature(&hotkey, "invalid_hex_!", b"data");
        assert!(result.is_err());
    }
}
