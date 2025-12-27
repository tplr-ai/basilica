//! TDX Quote Provider
//!
//! Generates TDX quotes by invoking the tdx-quote-generator CLI tool.

use async_trait::async_trait;
use std::path::Path;
use std::sync::Arc;
use tempfile::NamedTempFile;
use tokio::process::Command;
use tracing::{debug, error, info};

use crate::config::TdxConfig;
use crate::crypto::OpenSslCertHasher;
use crate::error::{TeeError, TeeResult};
use crate::traits::{CertificateHasher, QuoteProvider};

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
    /// Certificate hasher for computing cert hash
    cert_hasher: Arc<dyn CertificateHasher>,
}

impl TdxQuoteProvider {
    /// Create a new TdxQuoteProvider with default paths
    pub fn new() -> Self {
        Self {
            quote_generator_path: "/usr/bin/tdx-quote-generator".to_string(),
            server_cert_path: None,
            cert_hasher: Arc::new(OpenSslCertHasher::new()),
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
            cert_hasher: Arc::new(OpenSslCertHasher::new()),
        }
    }

    /// Create a new TdxQuoteProvider with custom paths
    pub fn with_paths(quote_generator: &str, server_cert: Option<&str>) -> Self {
        Self {
            quote_generator_path: quote_generator.to_string(),
            server_cert_path: server_cert.map(|s| s.to_string()),
            cert_hasher: Arc::new(OpenSslCertHasher::new()),
        }
    }

    /// Create a new TdxQuoteProvider with a custom certificate hasher
    pub fn with_cert_hasher(
        quote_generator: &str,
        server_cert: Option<&str>,
        cert_hasher: Arc<dyn CertificateHasher>,
    ) -> Self {
        Self {
            quote_generator_path: quote_generator.to_string(),
            server_cert_path: server_cert.map(|s| s.to_string()),
            cert_hasher,
        }
    }

    /// Get the certificate hash using the configured hasher.
    async fn get_cert_hash(&self) -> TeeResult<String> {
        let cert_path = self.server_cert_path.as_ref().ok_or_else(|| {
            TeeError::Certificate("Server certificate path not configured".into())
        })?;

        self.cert_hasher
            .hash_certificate_hex(Path::new(cert_path))
            .await
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

    /// Check if the quote generator binary exists
    pub fn is_available(&self) -> bool {
        Path::new(&self.quote_generator_path).exists()
    }
}

impl Default for TdxQuoteProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl QuoteProvider for TdxQuoteProvider {
    async fn generate_quote(&self, report_data: &[u8]) -> TeeResult<Vec<u8>> {
        let report_data_hex = hex::encode(report_data);
        self.get_quote(&report_data_hex).await
    }

    async fn generate_quote_with_nonce(&self, nonce_hex: &str) -> TeeResult<Vec<u8>> {
        self.get_quote(nonce_hex).await
    }

    fn is_available(&self) -> bool {
        TdxQuoteProvider::is_available(self)
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
