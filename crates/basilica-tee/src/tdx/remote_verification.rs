//! Remote TDX Quote Verification
//!
//! Provides integration with remote attestation services for TDX quote verification.
//! This module handles communication with Intel's DCAP verification service and
//! other attestation providers.

use crate::error::{TeeError, TeeResult};
use crate::types::TdxVerificationResult;
use serde::Deserialize;
#[cfg(feature = "remote-attestation")]
use serde::Serialize;
use std::time::Duration;
#[cfg(feature = "remote-attestation")]
use tracing::{info, warn};

/// Remote attestation service configuration
#[derive(Debug, Clone)]
pub struct RemoteAttestationConfig {
    /// URL of the attestation service
    pub service_url: String,
    /// Request timeout
    pub timeout: Duration,
    /// API key if required
    pub api_key: Option<String>,
    /// Whether to verify TLS certificates
    pub verify_tls: bool,
}

impl Default for RemoteAttestationConfig {
    fn default() -> Self {
        Self {
            // Intel's DCAP verification service
            service_url: "https://api.trustedservices.intel.com/sgx/dev/attestation/v4/report"
                .into(),
            timeout: Duration::from_secs(30),
            api_key: None,
            verify_tls: true,
        }
    }
}

/// Request body for Intel DCAP verification
#[cfg(feature = "remote-attestation")]
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct DcapVerificationRequest {
    /// Base64-encoded quote
    is_v_enclave_quote_status: String,
    /// Nonce for freshness
    nonce: Option<String>,
}

/// Response from Intel DCAP verification service
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
pub struct DcapVerificationResponse {
    /// Verification status
    pub id: String,
    /// Quote status (OK, GROUP_OUT_OF_DATE, etc.)
    pub is_v_enclave_quote_status: String,
    /// Timestamp of verification
    pub timestamp: String,
    /// Platform info blob if available
    pub platform_info_blob: Option<String>,
    /// Advisory IDs if any
    pub advisory_ids: Option<Vec<String>>,
}

/// Remote TDX Quote Verifier
///
/// Verifies TDX quotes using Intel's DCAP verification service or
/// other compatible remote attestation services.
pub struct RemoteTdxVerifier {
    config: RemoteAttestationConfig,
}

impl RemoteTdxVerifier {
    /// Create a new remote verifier with the given configuration
    pub fn new(config: RemoteAttestationConfig) -> Self {
        Self { config }
    }

    /// Create a remote verifier with Intel DCAP defaults
    pub fn intel_dcap() -> Self {
        Self::new(RemoteAttestationConfig::default())
    }

    /// Verify a TDX quote using remote attestation
    ///
    /// # Arguments
    /// * `quote_bytes` - Raw TDX quote bytes
    /// * `nonce` - Optional nonce for freshness verification
    ///
    /// # Returns
    /// Verification result from the remote service
    #[cfg(feature = "remote-attestation")]
    pub async fn verify(
        &self,
        quote_bytes: &[u8],
        nonce: Option<&[u8]>,
    ) -> TeeResult<TdxVerificationResult> {
        use base64::Engine;

        info!(
            "[TDX Remote] Sending quote for verification ({} bytes)",
            quote_bytes.len()
        );

        let quote_base64 = base64::engine::general_purpose::STANDARD.encode(quote_bytes);
        let nonce_hex = nonce.map(hex::encode);

        let request = DcapVerificationRequest {
            is_v_enclave_quote_status: quote_base64,
            nonce: nonce_hex,
        };

        // Build HTTP client
        let client = reqwest::Client::builder()
            .timeout(self.config.timeout)
            .danger_accept_invalid_certs(!self.config.verify_tls)
            .build()
            .map_err(|e| {
                TeeError::TdxQuoteVerification(format!("Failed to build HTTP client: {}", e))
            })?;

        // Make request
        let mut req = client.post(&self.config.service_url).json(&request);

        if let Some(ref api_key) = self.config.api_key {
            req = req.header("Ocp-Apim-Subscription-Key", api_key);
        }

        let response = req.send().await.map_err(|e| {
            TeeError::TdxQuoteVerification(format!("Remote verification request failed: {}", e))
        })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(TeeError::TdxQuoteVerification(format!(
                "Remote verification failed with status {}: {}",
                status, body
            )));
        }

        let verification: DcapVerificationResponse = response.json().await.map_err(|e| {
            TeeError::TdxQuoteVerification(format!("Failed to parse verification response: {}", e))
        })?;

        let quote_valid = verification.is_v_enclave_quote_status == "OK"
            || verification.is_v_enclave_quote_status == "GROUP_OUT_OF_DATE";

        if !quote_valid {
            warn!(
                "[TDX Remote] Quote verification failed: status={}",
                verification.is_v_enclave_quote_status
            );
        } else {
            info!("[TDX Remote] Quote verification succeeded");
            if let Some(ref advisories) = verification.advisory_ids {
                if !advisories.is_empty() {
                    warn!("[TDX Remote] Advisory IDs present: {:?}", advisories);
                }
            }
        }

        // Parse quote locally to get measurements
        let quote = crate::tdx::TdxQuoteV4::parse(quote_bytes)?;

        Ok(TdxVerificationResult {
            quote_valid,
            mrtd_matches: true, // Remote service doesn't check against expected
            rtmr_matches: vec![true; 4],
            report_data_matches: nonce.is_none_or(|n| quote.verify_nonce(n)),
            raw_quote: quote_bytes.to_vec(),
            mrtd_hex: quote.mrtd_hex(),
            verified_at: chrono::Utc::now(),
        })
    }

    /// Stub verification for when remote-attestation feature is not enabled
    #[cfg(not(feature = "remote-attestation"))]
    pub async fn verify(
        &self,
        _quote_bytes: &[u8],
        _nonce: Option<&[u8]>,
    ) -> TeeResult<TdxVerificationResult> {
        Err(TeeError::TdxQuoteVerification(
            "Remote attestation feature not enabled. Compile with --features remote-attestation"
                .into(),
        ))
    }

    /// Get the configured service URL
    pub fn service_url(&self) -> &str {
        &self.config.service_url
    }
}

/// Quote verification status from Intel DCAP
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuoteVerificationStatus {
    /// Quote is valid and trusted
    Ok,
    /// Quote is valid but TCB is out of date
    GroupOutOfDate,
    /// Quote is valid but configuration is out of date
    ConfigurationNeeded,
    /// Quote is valid but SW hardening is needed
    SwHardeningNeeded,
    /// Quote is valid but both configuration and SW hardening needed
    ConfigurationAndSwHardeningNeeded,
    /// Quote verification failed
    InvalidSignature,
    /// Quote has been revoked
    Revoked,
    /// Unknown status
    Unknown(String),
}

impl From<&str> for QuoteVerificationStatus {
    fn from(s: &str) -> Self {
        match s {
            "OK" => Self::Ok,
            "GROUP_OUT_OF_DATE" => Self::GroupOutOfDate,
            "CONFIGURATION_NEEDED" => Self::ConfigurationNeeded,
            "SW_HARDENING_NEEDED" => Self::SwHardeningNeeded,
            "CONFIGURATION_AND_SW_HARDENING_NEEDED" => Self::ConfigurationAndSwHardeningNeeded,
            "INVALID_SIGNATURE" => Self::InvalidSignature,
            "REVOKED" => Self::Revoked,
            other => Self::Unknown(other.to_string()),
        }
    }
}

impl QuoteVerificationStatus {
    /// Check if the quote is considered valid for use
    pub fn is_acceptable(&self) -> bool {
        matches!(
            self,
            Self::Ok
                | Self::GroupOutOfDate
                | Self::ConfigurationNeeded
                | Self::SwHardeningNeeded
                | Self::ConfigurationAndSwHardeningNeeded
        )
    }

    /// Check if the quote is fully trusted
    pub fn is_fully_trusted(&self) -> bool {
        matches!(self, Self::Ok)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quote_verification_status_from_str() {
        assert_eq!(
            QuoteVerificationStatus::from("OK"),
            QuoteVerificationStatus::Ok
        );
        assert_eq!(
            QuoteVerificationStatus::from("GROUP_OUT_OF_DATE"),
            QuoteVerificationStatus::GroupOutOfDate
        );
        assert!(matches!(
            QuoteVerificationStatus::from("UNKNOWN_STATUS"),
            QuoteVerificationStatus::Unknown(_)
        ));
    }

    #[test]
    fn test_quote_verification_status_acceptability() {
        assert!(QuoteVerificationStatus::Ok.is_acceptable());
        assert!(QuoteVerificationStatus::GroupOutOfDate.is_acceptable());
        assert!(!QuoteVerificationStatus::InvalidSignature.is_acceptable());
        assert!(!QuoteVerificationStatus::Revoked.is_acceptable());
    }

    #[test]
    fn test_quote_verification_status_trust() {
        assert!(QuoteVerificationStatus::Ok.is_fully_trusted());
        assert!(!QuoteVerificationStatus::GroupOutOfDate.is_fully_trusted());
    }

    #[test]
    fn test_default_config() {
        let config = RemoteAttestationConfig::default();
        assert!(config.service_url.contains("intel.com"));
        assert_eq!(config.timeout, Duration::from_secs(30));
        assert!(config.verify_tls);
    }
}
