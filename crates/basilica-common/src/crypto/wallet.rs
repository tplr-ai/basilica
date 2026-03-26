use crate::error::CryptoError;
use bittensor::crypto::{sr25519, Pair, Ss58AddressFormat, Ss58Codec};

/// SR25519 wallet information
#[derive(Debug, Clone)]
pub struct Sr25519Wallet {
    /// SS58 encoded address
    pub address: String,
    /// Account ID as hex string (public key)
    pub account_hex: String,
    /// Public key as hex string
    pub public_hex: String,
    /// Mnemonic seed phrase
    pub mnemonic: String,
}

/// Generate a new SR25519 wallet with the given SS58 prefix
///
/// # Arguments
/// * `ss58_prefix` - The SS58 prefix for address encoding (e.g., 42 for generic Substrate)
///
/// # Returns
/// * `Ok(Sr25519Wallet)` - Generated wallet information
/// * `Err(CryptoError)` - If wallet generation fails
///
/// # Example
/// ```rust
/// use basilica_common::crypto::wallet::generate_sr25519_wallet;
///
/// let wallet = generate_sr25519_wallet(42).unwrap();
/// assert!(wallet.address.len() > 0);
/// assert!(wallet.mnemonic.split_whitespace().count() >= 12);
/// ```
///
/// # Security Notes
/// - The mnemonic phrase should be stored securely and never logged
/// - This function generates a new random wallet each time
/// - Use appropriate SS58 prefix for your chain (42 for generic, 0 for Polkadot, etc.)
pub fn generate_sr25519_wallet(ss58_prefix: u16) -> Result<Sr25519Wallet, CryptoError> {
    let (pair, mnemonic, _) = sr25519::Pair::generate_with_phrase(None);

    let public_hex = hex::encode(pair.public().0);

    // Account hex is the same as public hex for SR25519
    let account_hex = public_hex.clone();

    // Generate SS58 address with the specified prefix
    let address = pair
        .public()
        .to_ss58check_with_version(Ss58AddressFormat::custom(ss58_prefix));

    Ok(Sr25519Wallet {
        address,
        account_hex,
        public_hex,
        mnemonic,
    })
}

/// Generate SR25519 wallet with a specific mnemonic phrase
///
/// # Arguments
/// * `mnemonic` - BIP39 mnemonic phrase
/// * `ss58_prefix` - The SS58 prefix for address encoding
///
/// # Returns
/// * `Ok(Sr25519Wallet)` - Wallet derived from mnemonic
/// * `Err(CryptoError)` - If mnemonic is invalid or derivation fails
///
/// # Example
/// ```rust,no_run
/// use basilica_common::crypto::wallet::generate_sr25519_wallet_from_mnemonic;
///
/// let mnemonic = "...";
/// let wallet = generate_sr25519_wallet_from_mnemonic(mnemonic, 42).unwrap();
/// assert!(wallet.address.len() > 0);
/// ```
pub fn generate_sr25519_wallet_from_mnemonic(
    mnemonic: &str,
    ss58_prefix: u16,
) -> Result<Sr25519Wallet, CryptoError> {
    // Create keypair from mnemonic
    let (pair, _) = sr25519::Pair::from_phrase(mnemonic, None).map_err(|e| {
        CryptoError::KeyDerivationFailed {
            details: format!("Invalid mnemonic: {}", e),
        }
    })?;

    // Get public key as hex
    let public_hex = hex::encode(pair.public().0);
    let account_hex = public_hex.clone();

    // Generate SS58 address
    let address = pair
        .public()
        .to_ss58check_with_version(Ss58AddressFormat::custom(ss58_prefix));

    Ok(Sr25519Wallet {
        address,
        account_hex,
        public_hex,
        mnemonic: mnemonic.to_string(),
    })
}

/// Create SR25519 keypair from mnemonic for signing operations
///
/// # Arguments
/// * `mnemonic` - BIP39 mnemonic phrase
///
/// # Returns
/// * `Ok(sr25519::Pair)` - Keypair for signing
/// * `Err(CryptoError)` - If mnemonic is invalid
///
/// # Security Notes
/// - The returned keypair contains the private key
/// - Should be zeroized after use when possible
/// - Never log or persist the keypair
pub fn sr25519_pair_from_mnemonic(mnemonic: &str) -> Result<sr25519::Pair, CryptoError> {
    let (pair, _) = sr25519::Pair::from_phrase(mnemonic, None).map_err(|e| {
        CryptoError::KeyDerivationFailed {
            details: format!("Invalid mnemonic: {}", e),
        }
    })?;
    Ok(pair)
}

/// Sign data with SR25519 keypair
///
/// # Arguments
/// * `pair` - SR25519 keypair
/// * `data` - Data to sign
///
/// # Returns
/// * Signature as hex string
pub fn sign_with_sr25519(pair: &sr25519::Pair, data: &[u8]) -> String {
    let signature = pair.sign(data);
    hex::encode(signature.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_wallet() {
        let wallet = generate_sr25519_wallet(42).unwrap();

        // Check all fields are populated
        assert!(!wallet.address.is_empty());
        assert!(!wallet.account_hex.is_empty());
        assert!(!wallet.public_hex.is_empty());
        assert!(!wallet.mnemonic.is_empty());

        // Check hex format
        assert_eq!(wallet.public_hex.len(), 64); // 32 bytes * 2
        assert!(wallet.public_hex.chars().all(|c| c.is_ascii_hexdigit()));

        // Check mnemonic has expected word count
        let word_count = wallet.mnemonic.split_whitespace().count();
        assert!(word_count >= 12);

        // Generate another wallet - should be different
        let wallet2 = generate_sr25519_wallet(42).unwrap();
        assert_ne!(wallet.address, wallet2.address);
        assert_ne!(wallet.public_hex, wallet2.public_hex);
        assert_ne!(wallet.mnemonic, wallet2.mnemonic);
    }

    #[test]
    fn test_wallet_from_mnemonic() {
        let mnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about";

        let wallet1 = generate_sr25519_wallet_from_mnemonic(mnemonic, 42).unwrap();
        let wallet2 = generate_sr25519_wallet_from_mnemonic(mnemonic, 42).unwrap();

        // Same mnemonic should produce same wallet
        assert_eq!(wallet1.address, wallet2.address);
        assert_eq!(wallet1.public_hex, wallet2.public_hex);
        assert_eq!(wallet1.account_hex, wallet2.account_hex);

        // Different prefix should produce different address
        let wallet3 = generate_sr25519_wallet_from_mnemonic(mnemonic, 0).unwrap();
        assert_ne!(wallet1.address, wallet3.address);
        // But same public key
        assert_eq!(wallet1.public_hex, wallet3.public_hex);
    }

    #[test]
    fn test_invalid_mnemonic() {
        let result = generate_sr25519_wallet_from_mnemonic("invalid mnemonic", 42);
        assert!(result.is_err());
    }

    #[test]
    fn test_signing() {
        let mnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about";
        let pair = sr25519_pair_from_mnemonic(mnemonic).unwrap();

        let data = b"test message";
        let signature = sign_with_sr25519(&pair, data);

        // Check signature format
        assert_eq!(signature.len(), 128); // 64 bytes * 2
        assert!(signature.chars().all(|c| c.is_ascii_hexdigit()));

        // SR25519 signatures include randomness, so they won't be identical
        // But we can verify the signature is valid
        let signature_bytes = hex::decode(&signature).unwrap();
        assert_eq!(signature_bytes.len(), 64);

        // Different data should produce different signature
        let signature2 = sign_with_sr25519(&pair, b"different message");
        assert_ne!(signature, signature2);

        // Verify signature works with the public key
        use bittensor::crypto::Pair as _;
        let sig_array: [u8; 64] = signature_bytes.try_into().unwrap();
        let sig = bittensor::crypto::sr25519::Signature::from_raw(sig_array);
        assert!(bittensor::crypto::sr25519::Pair::verify(
            &sig,
            data,
            &pair.public()
        ));
    }

    #[test]
    fn test_different_ss58_prefixes() {
        let wallet_generic = generate_sr25519_wallet(42).unwrap();
        let wallet_polkadot = generate_sr25519_wallet(0).unwrap();
        let wallet_kusama = generate_sr25519_wallet(2).unwrap();

        // All should have valid addresses
        assert!(!wallet_generic.address.is_empty());
        assert!(!wallet_polkadot.address.is_empty());
        assert!(!wallet_kusama.address.is_empty());

        // Different wallets should have different keys
        assert_ne!(wallet_generic.public_hex, wallet_polkadot.public_hex);
        assert_ne!(wallet_generic.public_hex, wallet_kusama.public_hex);
    }
}
