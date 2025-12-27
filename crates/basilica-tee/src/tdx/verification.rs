//! TDX Quote Verification
//!
//! Provides verification of TDX quotes against expected measurements.
//! Note: Full cryptographic verification requires Intel's Quote Verification Library (QVL).

use async_trait::async_trait;
use tracing::{debug, info, warn};

use crate::error::{TeeError, TeeResult};
use crate::tdx::TdxQuoteV4;
use crate::traits::TdxVerifier;
use crate::types::{ExpectedMeasurements, TdxVerificationResult};

/// TDX Quote Verifier
///
/// Verifies TDX quotes against expected measurements.
///
/// Note: This implementation provides measurement comparison but does not
/// perform full cryptographic signature verification. For production use,
/// integrate with Intel's Quote Verification Library (QVL) or a remote
/// attestation service.
pub struct TdxQuoteVerifier {
    /// Expected measurements to verify against
    expected: ExpectedMeasurements,
}

impl TdxQuoteVerifier {
    /// Create a new verifier with expected measurements
    pub fn new(expected: ExpectedMeasurements) -> Self {
        Self { expected }
    }

    /// Create a verifier from TDX config
    pub fn from_config(config: &crate::config::TdxConfig) -> Self {
        Self {
            expected: ExpectedMeasurements::from_config(config),
        }
    }

    /// Verify a TDX quote
    ///
    /// # Arguments
    /// * `quote_bytes` - Raw quote bytes
    /// * `expected_nonce` - Expected nonce that should be in report data
    ///
    /// # Returns
    /// Verification result with details about each check
    pub fn verify(
        &self,
        quote_bytes: &[u8],
        expected_nonce: Option<&[u8]>,
    ) -> TeeResult<TdxVerificationResult> {
        info!("[TDX] Verifying quote ({} bytes)", quote_bytes.len());

        // Parse the quote
        let quote = TdxQuoteV4::parse(quote_bytes)?;

        // Verify quote signature (stub - returns true)
        // TODO: Implement actual signature verification using Intel QVL
        let quote_valid = self.verify_signature(&quote)?;

        // Verify MRTD
        let mrtd_matches = self.expected.matches_mrtd(quote.mrtd());
        if !mrtd_matches {
            warn!("[TDX] MRTD mismatch: got {}", quote.mrtd_hex());
        }

        // Verify RTMRs
        let rtmrs = quote.rtmrs();
        let rtmr_matches: Vec<bool> = (0..4)
            .map(|i| {
                let matches = self.expected.matches_rtmr(i, &rtmrs[i]);
                if !matches {
                    warn!("[TDX] RTMR[{}] mismatch: got {}", i, hex::encode(rtmrs[i]));
                }
                matches
            })
            .collect();

        // Verify nonce in report data
        let report_data_matches = if let Some(nonce) = expected_nonce {
            let matches = quote.verify_nonce(nonce);
            if !matches {
                warn!("[TDX] Nonce mismatch in report data");
            }
            matches
        } else {
            debug!("[TDX] No nonce provided, skipping nonce verification");
            true
        };

        let result = TdxVerificationResult {
            quote_valid,
            mrtd_matches,
            rtmr_matches,
            report_data_matches,
            raw_quote: quote_bytes.to_vec(),
            mrtd_hex: quote.mrtd_hex(),
            verified_at: chrono::Utc::now(),
        };

        if result.quote_valid && result.mrtd_matches && result.report_data_matches {
            info!("[TDX] Quote verification passed");
        } else {
            warn!(
                "[TDX] Quote verification failed: quote_valid={}, mrtd_matches={}, report_data_matches={}",
                result.quote_valid, result.mrtd_matches, result.report_data_matches
            );
        }

        Ok(result)
    }

    /// Verify the quote signature
    ///
    /// Note: This is a stub implementation. In production, this should use
    /// Intel's Quote Verification Library (QVL) or call a remote attestation
    /// service to verify the signature chain.
    fn verify_signature(&self, _quote: &TdxQuoteV4) -> TeeResult<bool> {
        // TODO: Implement actual signature verification
        // Options:
        // 1. Use Intel's SGX SDK with QVL
        // 2. Call Intel's remote attestation service
        // 3. Use a third-party verification library

        debug!("[TDX] Signature verification stub - returning true");
        debug!("[TDX] NOTE: Implement actual verification for production use");

        Ok(true)
    }

    /// Verify quote using a remote attestation service
    ///
    /// # Arguments
    /// * `quote_bytes` - Raw quote bytes
    /// * `attestation_url` - URL of the attestation service
    ///
    /// Note: This is a stub implementation.
    #[allow(dead_code)]
    async fn verify_remote(&self, _quote_bytes: &[u8], _attestation_url: &str) -> TeeResult<bool> {
        // TODO: Implement remote verification
        // This would POST the quote to an attestation service (like Intel's)
        // and receive a verification result

        Err(TeeError::TdxQuoteVerification(
            "Remote verification not implemented".into(),
        ))
    }
}

impl Default for TdxQuoteVerifier {
    fn default() -> Self {
        Self::new(ExpectedMeasurements::default())
    }
}

#[async_trait]
impl TdxVerifier for TdxQuoteVerifier {
    async fn verify(
        &self,
        quote_bytes: &[u8],
        expected_nonce: Option<&[u8]>,
    ) -> TeeResult<TdxVerificationResult> {
        // Call the synchronous verify method
        TdxQuoteVerifier::verify(self, quote_bytes, expected_nonce)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tdx::quote::create_test_quote;

    /// Helper to create a test quote with specific measurements
    fn create_test_quote_bytes(mrtd: [u8; 48], rtmr0: [u8; 48], nonce: &[u8]) -> Vec<u8> {
        let mut report_data = [0u8; 64];
        let len = nonce.len().min(32);
        report_data[..len].copy_from_slice(&nonce[..len]);
        create_test_quote(mrtd, rtmr0, report_data)
    }

    #[test]
    fn test_verify_matching_measurements() {
        let mrtd = [0xAAu8; 48];
        let rtmr0 = [0xBBu8; 48];
        let nonce = b"test_nonce_12345";

        let quote_bytes = create_test_quote_bytes(mrtd, rtmr0, nonce);

        let expected = ExpectedMeasurements {
            mrtd: Some(mrtd),
            rtmr0: Some(rtmr0),
            ..Default::default()
        };

        let verifier = TdxQuoteVerifier::new(expected);
        let result = verifier.verify(&quote_bytes, Some(nonce)).unwrap();

        assert!(result.quote_valid);
        assert!(result.mrtd_matches);
        assert!(result.rtmr_matches[0]);
        assert!(result.report_data_matches);
    }

    #[test]
    fn test_verify_mrtd_mismatch() {
        let mrtd = [0xAAu8; 48];
        let quote_bytes = create_test_quote_bytes(mrtd, [0u8; 48], b"nonce");

        let expected = ExpectedMeasurements {
            mrtd: Some([0xBBu8; 48]), // Different from quote
            ..Default::default()
        };

        let verifier = TdxQuoteVerifier::new(expected);
        let result = verifier.verify(&quote_bytes, None).unwrap();

        assert!(result.quote_valid); // Signature stub returns true
        assert!(!result.mrtd_matches);
    }

    #[test]
    fn test_verify_nonce_mismatch() {
        let quote_bytes = create_test_quote_bytes([0u8; 48], [0u8; 48], b"correct_nonce");

        let verifier = TdxQuoteVerifier::default();
        let result = verifier.verify(&quote_bytes, Some(b"wrong_nonce")).unwrap();

        assert!(!result.report_data_matches);
    }

    #[test]
    fn test_verify_no_expected_measurements() {
        // When no expected measurements are configured, should match any
        let quote_bytes = create_test_quote_bytes([0xFFu8; 48], [0xFFu8; 48], b"nonce");

        let verifier = TdxQuoteVerifier::default();
        let result = verifier.verify(&quote_bytes, Some(b"nonce")).unwrap();

        assert!(result.mrtd_matches);
        assert!(result.rtmr_matches.iter().all(|&m| m));
    }

    #[test]
    fn test_verify_invalid_quote() {
        let invalid_bytes = vec![0u8; 100]; // Too short

        let verifier = TdxQuoteVerifier::default();
        let result = verifier.verify(&invalid_bytes, None);

        assert!(result.is_err());
    }

    #[test]
    fn test_verifier_from_config() {
        let config = crate::config::TdxConfig {
            expected_mrtd: Some("aa".repeat(48)),
            ..Default::default()
        };

        let verifier = TdxQuoteVerifier::from_config(&config);

        assert!(verifier.expected.mrtd.is_some());
        assert_eq!(verifier.expected.mrtd.unwrap(), [0xAAu8; 48]);
    }

    #[test]
    fn test_verification_result_contains_mrtd_hex() {
        let mrtd = [0xDEu8; 48];
        let quote_bytes = create_test_quote_bytes(mrtd, [0u8; 48], b"");

        let verifier = TdxQuoteVerifier::default();
        let result = verifier.verify(&quote_bytes, None).unwrap();

        assert_eq!(result.mrtd_hex, "de".repeat(48));
    }
}
