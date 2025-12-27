//! TEE Service Layer
//!
//! Provides a unified interface for TEE attestation operations,
//! orchestrating providers and verifiers through dependency injection.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::config::TeeConfig;
use crate::error::{TeeError, TeeResult};
use crate::gpu::{JsonEvidenceParser, LocalGpuVerifier, NvEvidenceProvider};
use crate::tdx::{TdxQuoteProvider, TdxQuoteVerifier};
use crate::traits::{EvidenceParser, EvidenceProvider, GpuVerifier, QuoteProvider, TdxVerifier};
use crate::types::{
    ExpectedMeasurements, GpuAttestationEvidence, GpuCcVerificationResult, TdxVerificationResult,
    TeeVerificationResult,
};

/// Combined attestation result from TDX and GPU.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeeAttestationResult {
    /// Base64-encoded TDX quote
    pub tdx_quote: Option<String>,
    /// GPU attestation evidence (JSON)
    pub gpu_evidence: Option<String>,
    /// Nonce used for attestation
    pub nonce_hex: String,
    /// Hostname of the attesting node
    pub hostname: String,
}

/// TEE Service configuration.
#[derive(Debug, Clone)]
pub struct TeeServiceConfig {
    /// Enable TDX attestation
    pub enable_tdx: bool,
    /// Enable GPU CC attestation
    pub enable_gpu: bool,
    /// Hostname for attestation
    pub hostname: String,
}

impl Default for TeeServiceConfig {
    fn default() -> Self {
        Self {
            enable_tdx: true,
            enable_gpu: true,
            hostname: "unknown".to_string(),
        }
    }
}

/// TEE Service
///
/// Central orchestration layer for TEE attestation operations.
/// Uses dependency injection for providers and verifiers, enabling
/// easy testing and extension.
pub struct TeeService {
    /// Configuration
    config: TeeServiceConfig,
    /// TDX quote provider
    tdx_provider: Arc<dyn QuoteProvider>,
    /// GPU evidence provider
    gpu_provider: Arc<dyn EvidenceProvider>,
    /// TDX quote verifier
    tdx_verifier: Arc<dyn TdxVerifier>,
    /// GPU evidence verifier
    gpu_verifier: Arc<dyn GpuVerifier>,
    /// Evidence parser
    evidence_parser: Arc<dyn EvidenceParser>,
}

impl TeeService {
    /// Create a new TEE service with default providers.
    pub fn new(hostname: String) -> TeeResult<Self> {
        Self::with_config(TeeServiceConfig {
            hostname,
            ..Default::default()
        })
    }

    /// Create a new TEE service with the given configuration.
    pub fn with_config(config: TeeServiceConfig) -> TeeResult<Self> {
        let tdx_provider = Arc::new(TdxQuoteProvider::new());
        let gpu_provider = Arc::new(NvEvidenceProvider::new());
        let tdx_verifier = Arc::new(TdxQuoteVerifier::default());
        let gpu_verifier = Arc::new(LocalGpuVerifier::new());
        let evidence_parser = Arc::new(JsonEvidenceParser::new());

        Ok(Self {
            config,
            tdx_provider,
            gpu_provider,
            tdx_verifier,
            gpu_verifier,
            evidence_parser,
        })
    }

    /// Create a new TEE service from TeeConfig.
    pub fn from_tee_config(tee_config: &TeeConfig, hostname: String) -> TeeResult<Self> {
        let tdx_provider = Arc::new(TdxQuoteProvider::from_config(&tee_config.tdx));
        let gpu_provider = Arc::new(NvEvidenceProvider::from_config(&tee_config.gpu));
        let tdx_verifier = Arc::new(TdxQuoteVerifier::from_config(&tee_config.tdx));
        let gpu_verifier = Arc::new(LocalGpuVerifier::new());
        let evidence_parser = Arc::new(JsonEvidenceParser::new());

        Ok(Self {
            config: TeeServiceConfig {
                enable_tdx: tee_config.enabled,
                enable_gpu: tee_config.enabled,
                hostname,
            },
            tdx_provider,
            gpu_provider,
            tdx_verifier,
            gpu_verifier,
            evidence_parser,
        })
    }

    /// Create a service builder for custom configuration.
    pub fn builder(hostname: String) -> TeeServiceBuilder {
        TeeServiceBuilder::new(hostname)
    }

    /// Generate attestation for the current TEE environment.
    ///
    /// # Arguments
    /// * `nonce_hex` - Hex-encoded nonce for freshness
    /// * `gpu_ids` - Optional list of GPU IDs to include
    ///
    /// # Returns
    /// Combined attestation result from TDX and GPU
    pub async fn attest(
        &self,
        nonce_hex: &str,
        gpu_ids: Option<&[String]>,
    ) -> TeeResult<TeeAttestationResult> {
        info!("[TeeService] Generating attestation with nonce");

        let mut result = TeeAttestationResult {
            tdx_quote: None,
            gpu_evidence: None,
            nonce_hex: nonce_hex.to_string(),
            hostname: self.config.hostname.clone(),
        };

        // Generate TDX quote if enabled and available
        if self.config.enable_tdx && self.tdx_provider.is_available() {
            debug!("[TeeService] Generating TDX quote");
            match self.tdx_provider.generate_quote_with_nonce(nonce_hex).await {
                Ok(quote) => {
                    use base64::Engine;
                    result.tdx_quote =
                        Some(base64::engine::general_purpose::STANDARD.encode(&quote));
                    info!("[TeeService] TDX quote generated successfully");
                }
                Err(e) => {
                    warn!("[TeeService] Failed to generate TDX quote: {}", e);
                }
            }
        } else if self.config.enable_tdx {
            debug!("[TeeService] TDX provider not available, skipping");
        }

        // Generate GPU evidence if enabled and available
        if self.config.enable_gpu && self.gpu_provider.is_available() {
            debug!("[TeeService] Generating GPU evidence");
            match self
                .gpu_provider
                .generate_evidence(&self.config.hostname, nonce_hex, gpu_ids)
                .await
            {
                Ok(evidence) => {
                    result.gpu_evidence = Some(evidence);
                    info!("[TeeService] GPU evidence generated successfully");
                }
                Err(e) => {
                    warn!("[TeeService] Failed to generate GPU evidence: {}", e);
                }
            }
        } else if self.config.enable_gpu {
            debug!("[TeeService] GPU provider not available, skipping");
        }

        Ok(result)
    }

    /// Verify attestation result.
    ///
    /// # Arguments
    /// * `attestation` - The attestation result to verify
    /// * `expected_nonce` - Optional expected nonce (uses attestation nonce if not provided)
    ///
    /// # Returns
    /// Combined verification result
    pub async fn verify(
        &self,
        attestation: &TeeAttestationResult,
        expected_nonce: Option<&[u8]>,
    ) -> TeeResult<TeeVerificationResult> {
        info!("[TeeService] Verifying attestation");

        let nonce = expected_nonce.unwrap_or(attestation.nonce_hex.as_bytes());

        let mut tdx_result: Option<TdxVerificationResult> = None;
        let mut gpu_result: Option<GpuCcVerificationResult> = None;

        // Verify TDX quote if present
        if let Some(ref quote_b64) = attestation.tdx_quote {
            debug!("[TeeService] Verifying TDX quote");
            use base64::Engine;
            let quote_bytes = base64::engine::general_purpose::STANDARD
                .decode(quote_b64)
                .map_err(|e| TeeError::TdxQuoteParsing(format!("Failed to decode quote: {}", e)))?;

            match self.tdx_verifier.verify(&quote_bytes, Some(nonce)).await {
                Ok(result) => {
                    tdx_result = Some(result);
                    info!("[TeeService] TDX verification complete");
                }
                Err(e) => {
                    warn!("[TeeService] TDX verification failed: {}", e);
                    return Err(e);
                }
            }
        }

        // Verify GPU evidence if present
        if let Some(ref evidence_json) = attestation.gpu_evidence {
            debug!("[TeeService] Verifying GPU evidence");
            let evidence_list = self.evidence_parser.parse(evidence_json)?;

            if let Some(first_evidence) = evidence_list.first() {
                match self
                    .gpu_verifier
                    .verify(first_evidence, Some(&attestation.nonce_hex))
                    .await
                {
                    Ok(result) => {
                        gpu_result = Some(result);
                        info!("[TeeService] GPU verification complete");
                    }
                    Err(e) => {
                        warn!("[TeeService] GPU verification failed: {}", e);
                        return Err(e);
                    }
                }
            }
        }

        // Determine overall result
        let tee_verified = match (&tdx_result, &gpu_result) {
            (Some(tdx), Some(gpu)) => {
                tdx.quote_valid && tdx.mrtd_matches && gpu.cc_mode_enabled && gpu.attestation_valid
            }
            (Some(tdx), None) => tdx.quote_valid && tdx.mrtd_matches,
            (None, Some(gpu)) => gpu.cc_mode_enabled && gpu.attestation_valid,
            (None, None) => false,
        };

        Ok(TeeVerificationResult {
            tdx: tdx_result,
            gpu_cc: gpu_result,
            tee_verified,
        })
    }

    /// Check if any TEE capabilities are available.
    pub fn is_available(&self) -> bool {
        self.tdx_provider.is_available() || self.gpu_provider.is_available()
    }

    /// Check if TDX is available.
    pub fn is_tdx_available(&self) -> bool {
        self.tdx_provider.is_available()
    }

    /// Check if GPU CC is available.
    pub fn is_gpu_available(&self) -> bool {
        self.gpu_provider.is_available()
    }

    /// Get the hostname.
    pub fn hostname(&self) -> &str {
        &self.config.hostname
    }

    /// Parse GPU evidence from JSON.
    pub fn parse_evidence(&self, json: &str) -> TeeResult<Vec<GpuAttestationEvidence>> {
        self.evidence_parser.parse(json)
    }
}

/// Builder for TeeService with custom providers.
pub struct TeeServiceBuilder {
    config: TeeServiceConfig,
    tdx_provider: Option<Arc<dyn QuoteProvider>>,
    gpu_provider: Option<Arc<dyn EvidenceProvider>>,
    tdx_verifier: Option<Arc<dyn TdxVerifier>>,
    gpu_verifier: Option<Arc<dyn GpuVerifier>>,
    evidence_parser: Option<Arc<dyn EvidenceParser>>,
}

impl TeeServiceBuilder {
    /// Create a new builder with the given hostname.
    pub fn new(hostname: String) -> Self {
        Self {
            config: TeeServiceConfig {
                hostname,
                ..Default::default()
            },
            tdx_provider: None,
            gpu_provider: None,
            tdx_verifier: None,
            gpu_verifier: None,
            evidence_parser: None,
        }
    }

    /// Set whether TDX is enabled.
    pub fn enable_tdx(mut self, enable: bool) -> Self {
        self.config.enable_tdx = enable;
        self
    }

    /// Set whether GPU CC is enabled.
    pub fn enable_gpu(mut self, enable: bool) -> Self {
        self.config.enable_gpu = enable;
        self
    }

    /// Set a custom TDX quote provider.
    pub fn with_tdx_provider(mut self, provider: Arc<dyn QuoteProvider>) -> Self {
        self.tdx_provider = Some(provider);
        self
    }

    /// Set a custom GPU evidence provider.
    pub fn with_gpu_provider(mut self, provider: Arc<dyn EvidenceProvider>) -> Self {
        self.gpu_provider = Some(provider);
        self
    }

    /// Set a custom TDX verifier.
    pub fn with_tdx_verifier(mut self, verifier: Arc<dyn TdxVerifier>) -> Self {
        self.tdx_verifier = Some(verifier);
        self
    }

    /// Set a custom GPU verifier.
    pub fn with_gpu_verifier(mut self, verifier: Arc<dyn GpuVerifier>) -> Self {
        self.gpu_verifier = Some(verifier);
        self
    }

    /// Set a custom evidence parser.
    pub fn with_evidence_parser(mut self, parser: Arc<dyn EvidenceParser>) -> Self {
        self.evidence_parser = Some(parser);
        self
    }

    /// Set expected measurements for TDX verification.
    pub fn with_expected_measurements(self, _measurements: ExpectedMeasurements) -> Self {
        // TODO: Allow configuring expected measurements on the verifier
        self
    }

    /// Build the TeeService.
    pub fn build(self) -> TeeResult<TeeService> {
        Ok(TeeService {
            config: self.config,
            tdx_provider: self
                .tdx_provider
                .unwrap_or_else(|| Arc::new(TdxQuoteProvider::new())),
            gpu_provider: self
                .gpu_provider
                .unwrap_or_else(|| Arc::new(NvEvidenceProvider::new())),
            tdx_verifier: self
                .tdx_verifier
                .unwrap_or_else(|| Arc::new(TdxQuoteVerifier::default())),
            gpu_verifier: self
                .gpu_verifier
                .unwrap_or_else(|| Arc::new(LocalGpuVerifier::new())),
            evidence_parser: self
                .evidence_parser
                .unwrap_or_else(|| Arc::new(JsonEvidenceParser::new())),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::mocks::{MockEvidenceProvider, MockQuoteProvider, MockTdxVerifier};

    #[test]
    fn test_service_builder() {
        let service = TeeService::builder("test-host".to_string())
            .enable_tdx(true)
            .enable_gpu(false)
            .build()
            .unwrap();

        assert_eq!(service.hostname(), "test-host");
        assert!(service.config.enable_tdx);
        assert!(!service.config.enable_gpu);
    }

    #[test]
    fn test_service_with_mock_providers() {
        let mock_quote = MockQuoteProvider::new(vec![1, 2, 3]);
        let mock_evidence = MockEvidenceProvider::with_sample();

        let service = TeeService::builder("test".to_string())
            .with_tdx_provider(Arc::new(mock_quote))
            .with_gpu_provider(Arc::new(mock_evidence))
            .build()
            .unwrap();

        assert!(service.is_tdx_available());
        assert!(service.is_gpu_available());
    }

    #[tokio::test]
    async fn test_attest_with_mocks() {
        let mock_quote = MockQuoteProvider::new(vec![1, 2, 3, 4]);
        let mock_evidence = MockEvidenceProvider::with_sample();

        let service = TeeService::builder("test-node".to_string())
            .with_tdx_provider(Arc::new(mock_quote))
            .with_gpu_provider(Arc::new(mock_evidence))
            .build()
            .unwrap();

        let result = service.attest("deadbeef", None).await.unwrap();

        assert!(result.tdx_quote.is_some());
        assert!(result.gpu_evidence.is_some());
        assert_eq!(result.nonce_hex, "deadbeef");
        assert_eq!(result.hostname, "test-node");
    }

    #[tokio::test]
    async fn test_verify_with_mocks() {
        let mock_verifier = MockTdxVerifier::passing();

        let service = TeeService::builder("test".to_string())
            .with_tdx_verifier(Arc::new(mock_verifier))
            .build()
            .unwrap();

        // Create a mock attestation result
        use base64::Engine;
        let attestation = TeeAttestationResult {
            tdx_quote: Some(base64::engine::general_purpose::STANDARD.encode([1, 2, 3])),
            gpu_evidence: None,
            nonce_hex: "test".to_string(),
            hostname: "test".to_string(),
        };

        let result = service.verify(&attestation, None).await.unwrap();

        assert!(result.tdx.is_some());
        assert!(result.tdx.as_ref().unwrap().quote_valid);
    }

    #[test]
    fn test_attestation_result_serialization() {
        let result = TeeAttestationResult {
            tdx_quote: Some("quote".to_string()),
            gpu_evidence: Some("{}".to_string()),
            nonce_hex: "nonce".to_string(),
            hostname: "host".to_string(),
        };

        let json = serde_json::to_string(&result).unwrap();
        let parsed: TeeAttestationResult = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.nonce_hex, "nonce");
        assert_eq!(parsed.hostname, "host");
    }
}
