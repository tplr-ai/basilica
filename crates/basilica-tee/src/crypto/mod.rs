//! Cryptographic utilities for TEE operations.
//!
//! This module provides cryptographic primitives used across the TEE crate,
//! including certificate hashing for TDX report data binding.

mod cert_hasher;

pub use cert_hasher::{CertHasher, OpenSslCertHasher};
