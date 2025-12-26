//! TDX Quote Provider
//!
//! Generates TDX quotes by invoking the tdx-quote-generator CLI tool.

use crate::config::TdxConfig;
use crate::error::{TeeError, TeeResult};
use sha2::{Digest, Sha256};
use std::path::Path;
use std::process::Stdio;
use tempfile::NamedTempFile;
use tokio::process::Command;
use tracing::{debug, error, info};

/// Async TDX quote provider with cert hash binding.
///
/// Generates TDX quotes by invoking the tdx-quote-generator CLI tool.
/// The quote includes both a nonce and the hash of the server certificate
/// for binding.
pub struct TdxQuoteProvider {
    /// Path to the quote generator binary
    quote_generator_path: String,
    /// Path to the server certificate
    server_cert_path: Option<String>,
}

impl TdxQuoteProvider {
    /// Create a new TdxQuoteProvider with default paths
    pub fn new() -> Self {
        Self {
            quote_generator_path: "/usr/bin/tdx-quote-generator".to_string(),
            server_cert_path: None,
        }
    }

    /// Create a new TdxQuoteProvider from config
    pub fn from_config(config: &TdxConfig) -> Self {
        Self {
            quote_generator_path: config.quote_generator_path.to_string_lossy().to_string(),
            server_cert_path: config
                .server_cert_path
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
        }
    }

    /// Create a new TdxQuoteProvider with custom paths
    pub fn with_paths(quote_generator: &str, server_cert: Option<&str>) -> Self {
        Self {
            quote_generator_path: quote_generator.to_string(),
            server_cert_path: server_cert.map(|s| s.to_string()),
        }
    }

    /// Check if the quote generator binary exists
    pub fn is_available(&self) -> bool {
        Path::new(&self.quote_generator_path).exists()
    }

    /// Compute SHA-256 hash of the server certificate's public key.
    ///
    /// This binds the quote to the specific certificate being used.
    /// Returns 64-character hex string (SHA-256 hash).
    async fn get_cert_hash(&self) -> TeeResult<String> {
        let cert_path = self.server_cert_path.as_ref().ok_or_else(|| {
            TeeError::Certificate("Server certificate path not configured".into())
        })?;

        // Extract public key from certificate
        let pubkey_output = Command::new("openssl")
            .args(["x509", "-in", cert_path, "-pubkey", "-noout"])
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
        let cert_hash = hex::encode(hasher.finalize());

        debug!("Computed cert hash: {}", cert_hash);
        Ok(cert_hash)
    }

    /// Generate a TDX quote with nonce and optional certificate hash in report data.
    ///
    /// # Arguments
    /// * `nonce` - 64-character hex string (32 bytes)
    ///
    /// # Returns
    /// Raw quote bytes
    pub async fn get_quote(&self, nonce: &str) -> TeeResult<Vec<u8>> {
        // Get certificate hash if configured
        let cert_hash = if self.server_cert_path.is_some() {
            self.get_cert_hash().await.ok()
        } else {
            None
        };

        // Combine nonce and cert hash for report data
        // TDX report data is 64 bytes (128 hex chars)
        let report_data = if let Some(hash) = cert_hash {
            // nonce (64 hex chars) + cert_hash (64 hex chars) = 128 hex chars
            let mut data = format!("{}{}", nonce, hash);
            data.truncate(128);
            data
        } else {
            // Pad nonce to 128 hex chars
            let mut data = nonce.to_string();
            data.truncate(128);
            while data.len() < 128 {
                data.push('0');
            }
            data
        };

        debug!(
            "Report data length: {} chars (nonce: {} chars)",
            report_data.len(),
            nonce.len()
        );

        // Create temp file for output
        let output_file = NamedTempFile::new().map_err(TeeError::Io)?;
        let output_path = output_file.path().to_string_lossy().to_string();

        // Run quote generator
        let result = Command::new(&self.quote_generator_path)
            .args([
                "--report-data",
                &report_data,
                "--hex",
                "--output",
                &output_path,
            ])
            .output()
            .await
            .map_err(|e| TeeError::TdxQuoteGeneration(format!("Failed to execute: {}", e)))?;

        if result.status.success() {
            info!(
                "Successfully generated quote with nonce.\n{}",
                String::from_utf8_lossy(&result.stdout)
            );

            // Read quote from file
            let quote_content = tokio::fs::read(output_file.path()).await.map_err(|e| {
                TeeError::TdxQuoteGeneration(format!("Failed to read quote file: {}", e))
            })?;

            Ok(quote_content)
        } else {
            let stderr = String::from_utf8_lossy(&result.stderr);
            error!("Failed to generate quote: {}", stderr);
            Err(TeeError::TdxQuoteGeneration(format!(
                "Quote generation failed: {}",
                stderr
            )))
        }
    }

    /// Generate a TDX quote with a random nonce
    pub async fn get_quote_with_random_nonce(&self) -> TeeResult<(Vec<u8>, [u8; 32])> {
        let nonce: [u8; 32] = rand::random();
        let nonce_hex = hex::encode(nonce);
        let quote = self.get_quote(&nonce_hex).await?;
        Ok((quote, nonce))
    }
}

impl Default for TdxQuoteProvider {
    fn default() -> Self {
        Self::new()
    }
}

/// Mock TDX quote provider for testing
#[cfg(test)]
pub struct MockTdxQuoteProvider {
    /// Pre-configured quote to return
    quote: Vec<u8>,
    /// Whether to simulate failure
    should_fail: bool,
}

#[cfg(test)]
impl MockTdxQuoteProvider {
    pub fn new(quote: Vec<u8>) -> Self {
        Self {
            quote,
            should_fail: false,
        }
    }

    pub fn failing() -> Self {
        Self {
            quote: vec![],
            should_fail: true,
        }
    }

    pub async fn get_quote(&self, _nonce: &str) -> TeeResult<Vec<u8>> {
        if self.should_fail {
            Err(TeeError::TdxQuoteGeneration("Mock failure".into()))
        } else {
            Ok(self.quote.clone())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_new() {
        let provider = TdxQuoteProvider::new();
        assert_eq!(
            provider.quote_generator_path,
            "/usr/bin/tdx-quote-generator"
        );
        assert!(provider.server_cert_path.is_none());
    }

    #[test]
    fn test_provider_with_paths() {
        let provider = TdxQuoteProvider::with_paths("/custom/path", Some("/cert/path"));
        assert_eq!(provider.quote_generator_path, "/custom/path");
        assert_eq!(provider.server_cert_path.as_deref(), Some("/cert/path"));
    }

    #[test]
    fn test_provider_from_config() {
        let config = TdxConfig::default();
        let provider = TdxQuoteProvider::from_config(&config);
        assert_eq!(
            provider.quote_generator_path,
            config.quote_generator_path.to_string_lossy()
        );
    }

    #[test]
    fn test_is_available_missing_binary() {
        let provider = TdxQuoteProvider::with_paths("/nonexistent/binary", None);
        assert!(!provider.is_available());
    }

    #[tokio::test]
    async fn test_mock_provider_success() {
        let test_quote = vec![1, 2, 3, 4, 5];
        let provider = MockTdxQuoteProvider::new(test_quote.clone());

        let result = provider.get_quote("test_nonce").await.unwrap();
        assert_eq!(result, test_quote);
    }

    #[tokio::test]
    async fn test_mock_provider_failure() {
        let provider = MockTdxQuoteProvider::failing();

        let result = provider.get_quote("test_nonce").await;
        assert!(result.is_err());
    }
}
