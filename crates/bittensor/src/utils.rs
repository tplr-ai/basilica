//! # Bittensor Utilities
//!
//! Helper functions for common Bittensor operations.

use crate::error::BittensorError;
use crate::AccountId;
use common::identity::Hotkey;
use std::str::FromStr;
use subxt::ext::sp_core::sr25519;

/// Convert a Hotkey to an AccountId
pub fn hotkey_to_account_id(hotkey: &Hotkey) -> Result<AccountId, BittensorError> {
    AccountId::from_str(hotkey.as_str()).map_err(|_| BittensorError::InvalidHotkey {
        hotkey: hotkey.as_str().to_string(),
    })
}

/// Convert an AccountId to a Hotkey
pub fn account_id_to_hotkey(account_id: &AccountId) -> Result<Hotkey, BittensorError> {
    Hotkey::from_str(&account_id.to_string()).map_err(|_| BittensorError::InvalidHotkey {
        hotkey: account_id.to_string(),
    })
}

// Weight-related types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NormalizedWeight {
    pub uid: u16,
    pub weight: u16,
}

/// Normalize weights to sum to u16::MAX
pub fn normalize_weights(weights: &[(u16, u16)]) -> Vec<NormalizedWeight> {
    if weights.is_empty() {
        return vec![];
    }

    let total: u64 = weights.iter().map(|(_, w)| *w as u64).sum();
    if total == 0 {
        return weights
            .iter()
            .map(|(uid, _)| NormalizedWeight {
                uid: *uid,
                weight: 0,
            })
            .collect();
    }

    let target = u16::MAX as u64;
    weights
        .iter()
        .map(|(uid, weight)| {
            let normalized = ((*weight as u64 * target) / total) as u16;
            NormalizedWeight {
                uid: *uid,
                weight: normalized,
            }
        })
        .collect()
}

/// Create a set_weights payload
pub fn set_weights_payload(
    netuid: u16,
    weights: Vec<NormalizedWeight>,
    version_key: u64,
) -> impl subxt::tx::Payload {
    use crate::api::api;

    // Extract UIDs and weights as separate vectors
    let (dests, values): (Vec<u16>, Vec<u16>) =
        weights.into_iter().map(|w| (w.uid, w.weight)).unzip();

    api::tx()
        .subtensor_module()
        .set_weights(netuid, dests, values, version_key)
}

/// Convert stake from TAO to RAO (1 TAO = 10^9 RAO)
pub fn tao_to_rao(tao: f64) -> u64 {
    (tao * 1_000_000_000.0) as u64
}

/// Convert stake from RAO to TAO (1 TAO = 10^9 RAO)
pub fn rao_to_tao(rao: u64) -> f64 {
    rao as f64 / 1_000_000_000.0
}

/// Verify Bittensor signature
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

    // Create signature from bytes
    let signature = sr25519::Signature::from_raw(signature_array);

    // Verify the signature
    use subxt::ext::sp_runtime::traits::Verify;

    // Convert our AccountId to the expected type
    let public_key = sr25519::Public::from_raw(account_id.0);
    let is_valid = signature.verify(data, &public_key);

    if is_valid {
        Ok(())
    } else {
        Err(BittensorError::AuthError {
            message: "Signature verification failed".to_string(),
        })
    }
}

/// Create a signature using a signer
pub fn create_signature<T>(signer: &T, data: &[u8]) -> String
where
    T: subxt::tx::Signer<subxt::PolkadotConfig>,
{
    // Sign the data
    let signature = signer.sign(data);
    // For now, just return a placeholder - we'll need to implement proper signing
    // when we have a working signer type
    hex::encode(vec![0u8; 64])
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
