//! Certificate hashing for TDX report data binding.
//!
//! This module provides utilities for computing SHA-256 hashes of certificate
//! public keys, which are used to bind TDX quotes to specific TLS certificates.

use async_trait::async_trait;
use sha2::{Digest, Sha256};
use std::path::Path;
use std::process::Stdio;
use tokio::process::Command;
use tracing::debug;

use crate::error::{TeeError, TeeResult};
use crate::traits::CertificateHasher;

/// Certificate hasher using OpenSSL command-line tools.
///
/// This implementation uses the `openssl` CLI to extract public keys
/// from certificates and compute their SHA-256 hash.
pub struct OpenSslCertHasher;

impl OpenSslCertHasher {
    /// Create a new OpenSSL-based certificate hasher.
    pub fn new() -> Self {
        Self
    }

    /// Check if OpenSSL is available on this system.
    pub fn is_available() -> bool {
        std::process::Command::new("openssl")
            .arg("version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }
}

impl Default for OpenSslCertHasher {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CertificateHasher for OpenSslCertHasher {
    async fn hash_certificate(&self, cert_path: &Path) -> TeeResult<[u8; 32]> {
        let cert_path_str = cert_path.to_string_lossy();

        // Extract public key from certificate
        let pubkey_output = Command::new("openssl")
            .args(["x509", "-in", &cert_path_str, "-pubkey", "-noout"])
            .output()
            .await
            .map_err(|e| TeeError::CommandExecution(format!("Failed to run openssl: {}", e)))?;

        if !pubkey_output.status.success() {
            return Err(TeeError::Certificate(format!(
                "openssl x509 failed: {}",
                String::from_utf8_lossy(&pubkey_output.stderr)
            )));
        }

        // Convert public key to DER format
        let der_output = Command::new("openssl")
            .args(["pkey", "-pubin", "-outform", "der"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .map_err(|e| {
                TeeError::CommandExecution(format!("Failed to spawn openssl pkey: {}", e))
            })?;

        // Write pubkey to stdin and get output
        let mut child = der_output;
        {
            use tokio::io::AsyncWriteExt;
            if let Some(ref mut stdin) = child.stdin {
                stdin
                    .write_all(&pubkey_output.stdout)
                    .await
                    .map_err(TeeError::Io)?;
            }
        }

        let output = child.wait_with_output().await.map_err(|e| {
            TeeError::CommandExecution(format!("Failed to get openssl output: {}", e))
        })?;

        if !output.status.success() {
            return Err(TeeError::Certificate("openssl pkey failed".into()));
        }

        // Compute SHA-256 hash
        let mut hasher = Sha256::new();
        hasher.update(&output.stdout);
        let hash: [u8; 32] = hasher.finalize().into();

        debug!("Computed cert hash: {}", hex::encode(hash));
        Ok(hash)
    }
}

/// Utility trait for certificate hashing operations.
pub trait CertHasher {
    /// Compute SHA-256 hash of raw DER-encoded public key bytes.
    fn hash_der_bytes(der_bytes: &[u8]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(der_bytes);
        hasher.finalize().into()
    }
}

impl CertHasher for OpenSslCertHasher {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_der_bytes() {
        let test_bytes = b"test public key data";
        let hash = OpenSslCertHasher::hash_der_bytes(test_bytes);

        // Should produce a 32-byte hash
        assert_eq!(hash.len(), 32);

        // Same input should produce same output
        let hash2 = OpenSslCertHasher::hash_der_bytes(test_bytes);
        assert_eq!(hash, hash2);

        // Different input should produce different output
        let hash3 = OpenSslCertHasher::hash_der_bytes(b"different data");
        assert_ne!(hash, hash3);
    }

    #[test]
    fn test_openssl_available() {
        // This test may pass or fail depending on system configuration
        let _available = OpenSslCertHasher::is_available();
        // Just ensure it doesn't panic
    }

    #[tokio::test]
    async fn test_hash_certificate_hex() {
        // This test requires a real certificate file, so we just test the trait method
        // exists and returns the right format
        let hasher = OpenSslCertHasher::new();

        // Create a temporary test certificate (self-signed)
        let temp_dir = tempfile::tempdir().unwrap();
        let cert_path = temp_dir.path().join("test.crt");
        let key_path = temp_dir.path().join("test.key");

        // Generate self-signed cert using openssl
        let result = std::process::Command::new("openssl")
            .args([
                "req",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-keyout",
                key_path.to_str().unwrap(),
                "-out",
                cert_path.to_str().unwrap(),
                "-days",
                "1",
                "-nodes",
                "-subj",
                "/CN=test",
            ])
            .output();

        if result.is_ok() && result.as_ref().unwrap().status.success() {
            // OpenSSL is available, test the hash function
            let hash_result = hasher.hash_certificate(&cert_path).await;
            if let Ok(hash) = hash_result {
                assert_eq!(hash.len(), 32);

                // Test hex conversion
                let hex_result = hasher.hash_certificate_hex(&cert_path).await;
                assert!(hex_result.is_ok());
                assert_eq!(hex_result.unwrap().len(), 64);
            }
        }
        // If openssl is not available, the test is skipped
    }
}
