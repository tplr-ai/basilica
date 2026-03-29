//! Wallet-based request signing for autonomous agents.
//!
//! This module provides the [`RequestSigner`] trait and a concrete [`Sr25519Signer`]
//! implementation. An agent provides something that can sign bytes and report its
//! address -- we never see the private key.

use basilica_common::crypto::{
    request_signing::build_canonical_message,
    wallet::{sign_with_sr25519, sr25519_pair_from_mnemonic},
};
use bittensor::crypto::sr25519;
use std::time::{SystemTime, UNIX_EPOCH};

/// Trait for wallet-based request signing.
///
/// The agent provides an implementation -- we never see the private key.
pub trait RequestSigner: Send + Sync {
    /// The SS58 address of the signing key
    fn address(&self) -> &str;

    /// Sign arbitrary bytes, returning hex-encoded sr25519 signature
    fn sign(&self, message: &[u8]) -> std::result::Result<String, Box<dyn std::error::Error + Send + Sync>>;
}

/// Concrete signer using an sr25519 keypair (for agents that have one in-process).
///
/// This is ONE implementation of [`RequestSigner`]. Agents can provide their own.
pub struct Sr25519Signer {
    pair: sr25519::Pair,
    address: String,
}

impl Sr25519Signer {
    /// Create from an sr25519 keypair and its SS58 address
    pub fn new(pair: sr25519::Pair, address: String) -> Self {
        Self { pair, address }
    }

    /// Create from a BIP39 mnemonic phrase
    pub fn from_mnemonic(
        mnemonic: &str,
        ss58_prefix: u16,
    ) -> std::result::Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let pair = sr25519_pair_from_mnemonic(mnemonic)?;
        let wallet = basilica_common::crypto::wallet::generate_sr25519_wallet_from_mnemonic(
            mnemonic,
            ss58_prefix,
        )?;
        Ok(Self::new(pair, wallet.address))
    }
}

impl RequestSigner for Sr25519Signer {
    fn address(&self) -> &str {
        &self.address
    }

    fn sign(&self, message: &[u8]) -> std::result::Result<String, Box<dyn std::error::Error + Send + Sync>> {
        Ok(sign_with_sr25519(&self.pair, message))
    }
}

/// Headers produced by signing a request
pub struct WalletHeaders {
    /// SS58 address
    pub address: String,
    /// Hex-encoded sr25519 signature
    pub signature: String,
    /// Unix timestamp string
    pub timestamp: String,
}

/// Sign a request, returning the three headers to attach.
///
/// This is used by the SDK client internals to sign outgoing requests.
pub fn sign_request(
    signer: &dyn RequestSigner,
    method: &str,
    path: &str,
    body: &[u8],
) -> std::result::Result<WalletHeaders, Box<dyn std::error::Error + Send + Sync>> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .to_string();
    let message = build_canonical_message(method, path, body, &timestamp);
    let signature = signer.sign(message.as_bytes())?;

    Ok(WalletHeaders {
        address: signer.address().to_string(),
        signature,
        timestamp,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use basilica_common::crypto::{
        verify_bittensor_signature,
        wallet::generate_sr25519_wallet,
    };
    use basilica_common::identity::Hotkey;

    const TEST_MNEMONIC: &str =
        "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about";

    #[test]
    fn test_sr25519_signer_from_mnemonic_valid() {
        let signer = Sr25519Signer::from_mnemonic(TEST_MNEMONIC, 42).unwrap();
        assert!(!signer.address().is_empty());

        // Same mnemonic should produce same address
        let signer2 = Sr25519Signer::from_mnemonic(TEST_MNEMONIC, 42).unwrap();
        assert_eq!(signer.address(), signer2.address());
    }

    #[test]
    fn test_sr25519_signer_from_mnemonic_invalid() {
        let result = Sr25519Signer::from_mnemonic("invalid mnemonic words", 42);
        assert!(result.is_err());
    }

    #[test]
    fn test_sign_request_produces_verifiable_signature() {
        let wallet = generate_sr25519_wallet(42).unwrap();
        let signer = Sr25519Signer::from_mnemonic(&wallet.mnemonic, 42).unwrap();

        let method = "GET";
        let path = "/test";
        let body = b"";

        let headers = sign_request(&signer, method, path, body).unwrap();

        // Reconstruct the canonical message the same way the backend would
        let canonical_msg = build_canonical_message(method, path, body, &headers.timestamp);

        // Verify with verify_bittensor_signature (same as backend uses)
        let hotkey = Hotkey::new(headers.address.clone()).unwrap();
        let result = verify_bittensor_signature(&hotkey, &headers.signature, canonical_msg.as_bytes());
        assert!(result.is_ok(), "Signature should be verifiable by backend: {:?}", result);
    }

    #[test]
    fn test_sign_request_different_bodies_different_signatures() {
        let wallet = generate_sr25519_wallet(42).unwrap();
        let signer = Sr25519Signer::from_mnemonic(&wallet.mnemonic, 42).unwrap();

        let headers1 = sign_request(&signer, "POST", "/test", b"body1").unwrap();
        let headers2 = sign_request(&signer, "POST", "/test", b"body2").unwrap();

        // Signatures should differ (different canonical messages)
        // Note: sr25519 includes randomness, so signatures always differ,
        // but we verify the canonical messages are different
        let msg1 = build_canonical_message("POST", "/test", b"body1", &headers1.timestamp);
        let msg2 = build_canonical_message("POST", "/test", b"body2", &headers2.timestamp);
        assert_ne!(msg1, msg2);
    }

    #[test]
    fn test_sign_request_different_methods_different_messages() {
        let msg1 = build_canonical_message("GET", "/test", b"", "12345");
        let msg2 = build_canonical_message("POST", "/test", b"", "12345");
        assert_ne!(msg1, msg2);
    }

    #[test]
    fn test_sign_request_different_paths_different_messages() {
        let msg1 = build_canonical_message("GET", "/path1", b"", "12345");
        let msg2 = build_canonical_message("GET", "/path2", b"", "12345");
        assert_ne!(msg1, msg2);
    }

    #[test]
    fn test_sign_request_timestamp_is_reasonable() {
        let wallet = generate_sr25519_wallet(42).unwrap();
        let signer = Sr25519Signer::from_mnemonic(&wallet.mnemonic, 42).unwrap();

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let headers = sign_request(&signer, "GET", "/test", b"").unwrap();
        let ts: u64 = headers.timestamp.parse().unwrap();

        // Timestamp should be within 5 seconds of now
        assert!(ts >= now - 5 && ts <= now + 5);
    }

    #[test]
    fn test_custom_request_signer() {
        struct MockSigner;

        impl RequestSigner for MockSigner {
            fn address(&self) -> &str {
                "5MockAddress"
            }

            fn sign(&self, _message: &[u8]) -> std::result::Result<String, Box<dyn std::error::Error + Send + Sync>> {
                Ok("mock_signature".to_string())
            }
        }

        let signer = MockSigner;
        let headers = sign_request(&signer, "GET", "/test", b"").unwrap();

        assert_eq!(headers.address, "5MockAddress");
        assert_eq!(headers.signature, "mock_signature");
        assert!(!headers.timestamp.is_empty());
    }
}
