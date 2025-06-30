//! Integration tests for crypto module

use common::crypto::{
    argon2_derive_key, generate_ephemeral_ed25519_keypair, generate_p256_keypair_formatted,
    pbkdf2_derive_key, KdfParams,
};

#[test]
fn test_ed25519_keypair_no_placeholder() {
    // Generate multiple keypairs and ensure they're different
    let (priv1, pub1) = generate_ephemeral_ed25519_keypair();
    let (priv2, pub2) = generate_ephemeral_ed25519_keypair();

    // Keys should be properly formatted
    assert!(priv1.contains("-----BEGIN PRIVATE KEY-----"));
    assert!(priv1.contains("-----END PRIVATE KEY-----"));
    assert!(pub1.starts_with("ssh-ed25519"));
    assert!(pub1.contains("basilica-ephemeral-key"));

    // Keys should be different (no placeholders)
    assert_ne!(priv1, priv2);
    assert_ne!(pub1, pub2);
}

#[test]
fn test_p256_keypair_generation_works() {
    let result = generate_p256_keypair_formatted();
    assert!(result.is_ok());

    let (priv_pem, pub_pem, pub_hex) = result.unwrap();

    // Check format
    assert!(priv_pem.contains("-----BEGIN PRIVATE KEY-----"));
    assert!(pub_pem.contains("-----BEGIN PUBLIC KEY-----"));
    assert_eq!(pub_hex.len(), 66); // 33 bytes compressed * 2 hex chars

    // Generate another pair - should be different
    let (priv2, _, hex2) = generate_p256_keypair_formatted().unwrap();
    assert_ne!(priv_pem, priv2);
    assert_ne!(pub_hex, hex2);
}

#[test]
fn test_pbkdf2_replaces_blake3() {
    let password = "test_password";
    let salt = b"test_salt_16byte";

    // Use PBKDF2 with specific parameters
    let params = KdfParams::pbkdf2_default()
        .with_salt(salt.to_vec())
        .with_iterations(1000);

    let key1 = pbkdf2_derive_key(password, &params).unwrap();
    let key2 = pbkdf2_derive_key(password, &params).unwrap();

    // Same inputs should produce same output
    assert_eq!(key1, key2);
    assert_eq!(key1.len(), 32);

    // Different password should produce different key
    let key3 = pbkdf2_derive_key("different_password", &params).unwrap();
    assert_ne!(key1, key3);
}

#[test]
fn test_argon2_key_derivation() {
    let password = "secure_password";
    let params = KdfParams::argon2_default()
        .with_salt(b"unique_salt_here".to_vec())
        .with_iterations(1)
        .with_memory_cost(1024) // 1 MiB for testing
        .with_parallelism(1);

    let key = argon2_derive_key(password, &params).unwrap();
    assert_eq!(key.len(), 32);

    // Different parameters should produce different keys
    let params2 = params.clone().with_memory_cost(2048);
    let key2 = argon2_derive_key(password, &params2).unwrap();
    assert_ne!(key, key2);
}

#[test]
fn test_crypto_module_exports_all_functions() {
    // This test ensures all functions are properly exported
    use common::crypto::{
        generate_random_key,
        // Core functions
        hash_blake3,
        secure_compare,
    };

    // Basic sanity checks
    let _ = hash_blake3(b"test");
    let _ = generate_random_key(32);
    assert!(secure_compare(b"test", b"test"));
}

#[test]
fn test_no_placeholder_implementations() {
    use common::crypto::{
        derive_key_simple, generate_ed25519_keypair, generate_p256_keypair, generate_random_key,
    };

    // Test that ED25519 key generation is not using placeholder implementation
    let (private_pem1, public_ssh1) = generate_ed25519_keypair();
    let (private_pem2, public_ssh2) = generate_ed25519_keypair();

    // Keys should be different each time (not placeholder)
    assert_ne!(private_pem1, private_pem2);
    assert_ne!(public_ssh1, public_ssh2);

    // Check that keys don't contain placeholder text
    assert!(!private_pem1.contains("MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQg"));
    assert!(!public_ssh1.contains("Placeholder"));

    // Test that KDF is not using Blake3 as a placeholder
    let salt1 = generate_random_key(16);
    let salt2 = generate_random_key(16);
    let password = "test_password";

    let key1 = derive_key_simple(password, &salt1);
    let key2 = derive_key_simple(password, &salt2);

    // Keys should be different with different salts
    assert_ne!(key1, key2);

    // Test P256 key generation
    let p256_keypair1 = generate_p256_keypair().unwrap();
    let p256_keypair2 = generate_p256_keypair().unwrap();

    // P256 keys should be different each time
    assert_ne!(
        p256_keypair1.private_key().to_pem().unwrap(),
        p256_keypair2.private_key().to_pem().unwrap()
    );
    assert_ne!(
        p256_keypair1.public_key().to_hex(),
        p256_keypair2.public_key().to_hex()
    );
}
