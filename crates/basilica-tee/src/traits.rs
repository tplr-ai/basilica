//! Core trait abstractions for TEE operations.
//!
//! This module defines the fundamental traits that enable:
//! - Dependency injection for providers and verifiers
//! - Easy mocking for testing
//! - Extension points for new TEE types
//! - Feature-gated implementations without code duplication

use async_trait::async_trait;

use crate::error::TeeResult;
use crate::types::{GpuAttestationEvidence, GpuCcVerificationResult, TdxVerificationResult};

/// Trait for generating TDX quotes.
///
/// Implementations of this trait handle the specifics of generating
/// attestation quotes from the TDX hardware.
#[async_trait]
pub trait QuoteProvider: Send + Sync {
    /// Generate a quote with the given report data.
    ///
    /// # Arguments
    /// * `report_data` - Data to include in the quote (typically nonce + cert hash)
    ///
    /// # Returns
    /// Raw quote bytes on success
    async fn generate_quote(&self, report_data: &[u8]) -> TeeResult<Vec<u8>>;

    /// Generate a quote with a hex-encoded nonce string.
    ///
    /// # Arguments
    /// * `nonce_hex` - Hex-encoded nonce string
    ///
    /// # Returns
    /// Raw quote bytes on success
    async fn generate_quote_with_nonce(&self, nonce_hex: &str) -> TeeResult<Vec<u8>>;

    /// Check if the quote provider is available on this system.
    fn is_available(&self) -> bool;
}

/// Trait for generating GPU attestation evidence.
///
/// Implementations handle gathering attestation evidence from
/// NVIDIA GPUs with Confidential Computing support.
#[async_trait]
pub trait EvidenceProvider: Send + Sync {
    /// Generate attestation evidence for GPUs.
    ///
    /// # Arguments
    /// * `name` - Node/host name to include in evidence
    /// * `nonce` - Nonce for freshness (hex string)
    /// * `gpu_ids` - Optional filter for specific GPU IDs
    ///
    /// # Returns
    /// JSON string containing evidence on success
    async fn generate_evidence(
        &self,
        name: &str,
        nonce: &str,
        gpu_ids: Option<&[String]>,
    ) -> TeeResult<String>;

    /// Check if the evidence provider is available on this system.
    fn is_available(&self) -> bool;
}

/// Trait for verifying TDX quotes.
///
/// Implementations can perform local or remote verification
/// of TDX attestation quotes.
#[async_trait]
pub trait TdxVerifier: Send + Sync {
    /// Verify a TDX quote.
    ///
    /// # Arguments
    /// * `quote_bytes` - Raw quote bytes to verify
    /// * `expected_nonce` - Optional expected nonce for verification
    ///
    /// # Returns
    /// Verification result with details about each check
    async fn verify(
        &self,
        quote_bytes: &[u8],
        expected_nonce: Option<&[u8]>,
    ) -> TeeResult<TdxVerificationResult>;
}

/// Trait for verifying GPU attestation evidence.
///
/// Implementations can perform local or remote verification
/// of GPU CC attestation evidence.
#[async_trait]
pub trait GpuVerifier: Send + Sync {
    /// Verify GPU attestation evidence.
    ///
    /// # Arguments
    /// * `evidence` - The attestation evidence to verify
    /// * `expected_nonce` - Optional expected nonce for verification
    ///
    /// # Returns
    /// Verification result with details about each check
    async fn verify(
        &self,
        evidence: &GpuAttestationEvidence,
        expected_nonce: Option<&str>,
    ) -> TeeResult<GpuCcVerificationResult>;

    /// Verify multiple GPU attestation evidence entries.
    ///
    /// # Arguments
    /// * `evidence_list` - List of evidence entries to verify
    /// * `expected_nonce` - Optional expected nonce for verification
    ///
    /// # Returns
    /// List of verification results
    async fn verify_all(
        &self,
        evidence_list: &[GpuAttestationEvidence],
        expected_nonce: Option<&str>,
    ) -> TeeResult<Vec<GpuCcVerificationResult>> {
        let mut results = Vec::with_capacity(evidence_list.len());
        for evidence in evidence_list {
            results.push(self.verify(evidence, expected_nonce).await?);
        }
        Ok(results)
    }
}

/// Trait for parsing GPU attestation evidence from various formats.
pub trait EvidenceParser: Send + Sync {
    /// Parse evidence from a JSON string.
    ///
    /// # Arguments
    /// * `json` - JSON string containing evidence
    ///
    /// # Returns
    /// List of parsed evidence entries
    fn parse(&self, json: &str) -> TeeResult<Vec<GpuAttestationEvidence>>;
}

/// Trait for computing certificate hashes for TDX report data binding.
#[async_trait]
pub trait CertificateHasher: Send + Sync {
    /// Compute hash of a certificate's public key.
    ///
    /// # Arguments
    /// * `cert_path` - Path to the certificate file
    ///
    /// # Returns
    /// SHA-256 hash of the public key (32 bytes)
    async fn hash_certificate(&self, cert_path: &std::path::Path) -> TeeResult<[u8; 32]>;

    /// Compute hash and return as hex string.
    ///
    /// # Arguments
    /// * `cert_path` - Path to the certificate file
    ///
    /// # Returns
    /// Hex-encoded SHA-256 hash (64 characters)
    async fn hash_certificate_hex(&self, cert_path: &std::path::Path) -> TeeResult<String> {
        let hash = self.hash_certificate(cert_path).await?;
        Ok(hex::encode(hash))
    }
}

#[cfg(test)]
pub mod mocks {
    //! Mock implementations for testing.

    use super::*;
    use crate::error::TeeError;

    /// Mock quote provider for testing.
    pub struct MockQuoteProvider {
        quote: Vec<u8>,
        available: bool,
        should_fail: bool,
    }

    impl MockQuoteProvider {
        pub fn new(quote: Vec<u8>) -> Self {
            Self {
                quote,
                available: true,
                should_fail: false,
            }
        }

        pub fn unavailable() -> Self {
            Self {
                quote: vec![],
                available: false,
                should_fail: false,
            }
        }

        pub fn failing() -> Self {
            Self {
                quote: vec![],
                available: true,
                should_fail: true,
            }
        }
    }

    #[async_trait]
    impl QuoteProvider for MockQuoteProvider {
        async fn generate_quote(&self, _report_data: &[u8]) -> TeeResult<Vec<u8>> {
            if self.should_fail {
                Err(TeeError::TdxQuoteGeneration("Mock failure".into()))
            } else {
                Ok(self.quote.clone())
            }
        }

        async fn generate_quote_with_nonce(&self, _nonce_hex: &str) -> TeeResult<Vec<u8>> {
            if self.should_fail {
                Err(TeeError::TdxQuoteGeneration("Mock failure".into()))
            } else {
                Ok(self.quote.clone())
            }
        }

        fn is_available(&self) -> bool {
            self.available
        }
    }

    /// Mock evidence provider for testing.
    pub struct MockEvidenceProvider {
        evidence: String,
        available: bool,
        should_fail: bool,
    }

    impl MockEvidenceProvider {
        pub fn new(evidence: String) -> Self {
            Self {
                evidence,
                available: true,
                should_fail: false,
            }
        }

        pub fn with_sample() -> Self {
            let evidence = serde_json::json!([{
                "gpu_uuid": "GPU-mock-123",
                "attestation_report": "deadbeef",
                "signature": "cafebabe",
                "nonce": "test_nonce",
                "gpu_model": "Mock H100",
                "driver_version": "555.0"
            }])
            .to_string();
            Self::new(evidence)
        }

        pub fn failing() -> Self {
            Self {
                evidence: String::new(),
                available: true,
                should_fail: true,
            }
        }
    }

    #[async_trait]
    impl EvidenceProvider for MockEvidenceProvider {
        async fn generate_evidence(
            &self,
            _name: &str,
            _nonce: &str,
            _gpu_ids: Option<&[String]>,
        ) -> TeeResult<String> {
            if self.should_fail {
                Err(TeeError::GpuAttestation("Mock failure".into()))
            } else {
                Ok(self.evidence.clone())
            }
        }

        fn is_available(&self) -> bool {
            self.available
        }
    }

    /// Mock TDX verifier for testing.
    pub struct MockTdxVerifier {
        result: TdxVerificationResult,
        should_fail: bool,
    }

    impl MockTdxVerifier {
        pub fn passing() -> Self {
            Self {
                result: TdxVerificationResult {
                    quote_valid: true,
                    mrtd_matches: true,
                    rtmr_matches: vec![true, true, true, true],
                    report_data_matches: true,
                    raw_quote: vec![],
                    mrtd_hex: "00".repeat(48),
                    verified_at: chrono::Utc::now(),
                },
                should_fail: false,
            }
        }

        pub fn failing() -> Self {
            Self {
                result: TdxVerificationResult {
                    quote_valid: false,
                    mrtd_matches: false,
                    rtmr_matches: vec![false, false, false, false],
                    report_data_matches: false,
                    raw_quote: vec![],
                    mrtd_hex: String::new(),
                    verified_at: chrono::Utc::now(),
                },
                should_fail: false,
            }
        }

        pub fn error() -> Self {
            Self {
                result: TdxVerificationResult {
                    quote_valid: false,
                    mrtd_matches: false,
                    rtmr_matches: vec![],
                    report_data_matches: false,
                    raw_quote: vec![],
                    mrtd_hex: String::new(),
                    verified_at: chrono::Utc::now(),
                },
                should_fail: true,
            }
        }
    }

    #[async_trait]
    impl TdxVerifier for MockTdxVerifier {
        async fn verify(
            &self,
            _quote_bytes: &[u8],
            _expected_nonce: Option<&[u8]>,
        ) -> TeeResult<TdxVerificationResult> {
            if self.should_fail {
                Err(TeeError::TdxQuoteVerification("Mock failure".into()))
            } else {
                Ok(self.result.clone())
            }
        }
    }

    /// Mock GPU verifier for testing.
    pub struct MockGpuVerifier {
        result: GpuCcVerificationResult,
        should_fail: bool,
    }

    impl MockGpuVerifier {
        pub fn passing() -> Self {
            Self {
                result: GpuCcVerificationResult {
                    cc_mode_enabled: true,
                    attestation_valid: true,
                    gpu_uuid: "GPU-mock-123".to_string(),
                    nonce_verified: true,
                    gpu_model: "Mock H100".to_string(),
                    driver_version: "555.0".to_string(),
                    verified_at: chrono::Utc::now(),
                },
                should_fail: false,
            }
        }

        pub fn failing() -> Self {
            Self {
                result: GpuCcVerificationResult {
                    cc_mode_enabled: false,
                    attestation_valid: false,
                    gpu_uuid: String::new(),
                    nonce_verified: false,
                    gpu_model: String::new(),
                    driver_version: String::new(),
                    verified_at: chrono::Utc::now(),
                },
                should_fail: false,
            }
        }
    }

    #[async_trait]
    impl GpuVerifier for MockGpuVerifier {
        async fn verify(
            &self,
            _evidence: &GpuAttestationEvidence,
            _expected_nonce: Option<&str>,
        ) -> TeeResult<GpuCcVerificationResult> {
            if self.should_fail {
                Err(TeeError::GpuAttestation("Mock failure".into()))
            } else {
                Ok(self.result.clone())
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[tokio::test]
        async fn test_mock_quote_provider() {
            let provider = MockQuoteProvider::new(vec![1, 2, 3]);
            assert!(provider.is_available());

            let quote = provider.generate_quote(&[]).await.unwrap();
            assert_eq!(quote, vec![1, 2, 3]);
        }

        #[tokio::test]
        async fn test_mock_quote_provider_failure() {
            let provider = MockQuoteProvider::failing();
            let result = provider.generate_quote(&[]).await;
            assert!(result.is_err());
        }

        #[tokio::test]
        async fn test_mock_evidence_provider() {
            let provider = MockEvidenceProvider::with_sample();
            assert!(provider.is_available());

            let evidence = provider
                .generate_evidence("test", "nonce", None)
                .await
                .unwrap();
            assert!(evidence.contains("GPU-mock-123"));
        }

        #[tokio::test]
        async fn test_mock_tdx_verifier() {
            let verifier = MockTdxVerifier::passing();
            let result = verifier.verify(&[], None).await.unwrap();
            assert!(result.quote_valid);
            assert!(result.mrtd_matches);
        }

        #[tokio::test]
        async fn test_mock_gpu_verifier() {
            let verifier = MockGpuVerifier::passing();
            let evidence = GpuAttestationEvidence {
                gpu_uuid: "GPU-123".to_string(),
                attestation_report: String::new(),
                signature: String::new(),
                cert_chain: vec![],
                nonce: String::new(),
                gpu_model: String::new(),
                driver_version: String::new(),
                timestamp: chrono::Utc::now(),
            };
            let result = verifier.verify(&evidence, None).await.unwrap();
            assert!(result.cc_mode_enabled);
            assert!(result.attestation_valid);
        }
    }
}
