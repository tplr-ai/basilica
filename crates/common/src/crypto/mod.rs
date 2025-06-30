//! Cryptographic utilities module for Basilica
//!
//! This module provides comprehensive cryptographic operations including:
//! - Ed25519 key generation and management
//! - P256 (secp256r1) ECDSA operations
//! - Key derivation functions (PBKDF2, Argon2)
//! - Hashing and signature verification

// Core cryptographic functions
mod core;

// Public submodules
pub mod ed25519;
pub mod kdf;
pub mod keys;
pub mod p256;

// Re-export core hashing and symmetric encryption from core module
pub use core::{
    decrypt_aes_gcm, derive_key_from_gpu_info, derive_key_simple, encrypt_aes_gcm,
    generate_ephemeral_ed25519_keypair, generate_random_key, hash_blake3, hash_blake3_string,
    secure_compare, symmetric_decrypt, symmetric_encrypt, verify_bittensor_signature,
    verify_signature, verify_signature_bittensor, AES_KEY_SIZE, AES_NONCE_SIZE, BLAKE3_DIGEST_SIZE,
};

// Re-export commonly used types and functions
pub use ed25519::{Ed25519KeyPair, Ed25519PrivateKey, Ed25519PublicKey};
pub use kdf::{argon2_derive_key, pbkdf2_derive_key, KdfParams};
pub use keys::{generate_ed25519_keypair, generate_p256_keypair, generate_p256_keypair_formatted};
pub use p256::{verify_p256_signature, P256KeyPair, P256PrivateKey, P256PublicKey, P256Signature};
