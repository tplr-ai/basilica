//! # Request Verification Module
//!
//! Provides request signing and verification for secure communication between validators and miners.

use anyhow::{anyhow, Result};
use blake3::Hasher;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, warn};

use common::identity::Hotkey;

/// Request signature data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestSignature {
    /// The validator's hotkey that signed the request
    pub validator_hotkey: String,
    /// The signature of the request
    pub signature: String,
    /// Timestamp when the request was signed
    pub timestamp: DateTime<Utc>,
    /// Nonce to prevent replay attacks
    pub nonce: String,
    /// The HTTP method of the request
    pub method: String,
    /// The URI path of the request
    pub path: String,
    /// Optional request body hash
    pub body_hash: Option<String>,
}

/// Request verification service
#[derive(Clone)]
pub struct RequestVerificationService {
    /// Maximum age of a valid request (default: 5 minutes)
    max_request_age: Duration,
    /// Used nonces to prevent replay attacks
    used_nonces: Arc<RwLock<HashMap<String, DateTime<Utc>>>>,
    /// Allowed validators (empty means all are allowed)
    allowed_validators: Vec<Hotkey>,
    /// Whether to verify signatures
    verify_signatures: bool,
}

impl RequestVerificationService {
    /// Create a new request verification service
    pub fn new(
        max_request_age: Duration,
        allowed_validators: Vec<Hotkey>,
        verify_signatures: bool,
    ) -> Self {
        Self {
            max_request_age,
            used_nonces: Arc::new(RwLock::new(HashMap::new())),
            allowed_validators,
            verify_signatures,
        }
    }

    /// Generate a canonical request string for signing
    pub fn canonical_request(
        method: &str,
        path: &str,
        timestamp: &DateTime<Utc>,
        nonce: &str,
        body: Option<&[u8]>,
    ) -> String {
        let mut hasher = Hasher::new();

        // Hash the body if present
        let body_hash = if let Some(body_data) = body {
            hasher.update(body_data);
            Some(hasher.finalize().to_hex().to_string())
        } else {
            None
        };

        // Create canonical string
        let mut canonical = String::new();
        canonical.push_str(method);
        canonical.push('\n');
        canonical.push_str(path);
        canonical.push('\n');
        canonical.push_str(&timestamp.to_rfc3339());
        canonical.push('\n');
        canonical.push_str(nonce);

        if let Some(hash) = body_hash {
            canonical.push('\n');
            canonical.push_str(&hash);
        }

        canonical
    }

    /// Verify a request signature
    pub async fn verify_request(
        &self,
        signature_data: &RequestSignature,
        body: Option<&[u8]>,
    ) -> Result<()> {
        // Check timestamp age
        let now = Utc::now();
        let request_age = now - signature_data.timestamp;

        if request_age > self.max_request_age {
            return Err(anyhow!("Request too old: {:?}", request_age));
        }

        if request_age < Duration::zero() {
            // Allow small clock skew (up to 1 minute in the future)
            if request_age.abs() > Duration::minutes(1) {
                return Err(anyhow!("Request timestamp is in the future"));
            }
        }

        // Check nonce for replay attack prevention
        let mut used_nonces = self.used_nonces.write().await;

        if used_nonces.contains_key(&signature_data.nonce) {
            return Err(anyhow!("Nonce already used"));
        }

        // Store nonce with expiration
        used_nonces.insert(signature_data.nonce.clone(), now);

        // Clean up old nonces
        let cutoff = now - self.max_request_age - Duration::hours(1);
        used_nonces.retain(|_, timestamp| *timestamp > cutoff);
        drop(used_nonces);

        // Verify validator is allowed
        if !self.allowed_validators.is_empty() {
            let validator_hotkey =
                Hotkey::new(signature_data.validator_hotkey.clone()).map_err(|e| anyhow!(e))?;

            if !self.allowed_validators.contains(&validator_hotkey) {
                return Err(anyhow!("Validator not in allowlist"));
            }
        }

        // Verify signature if enabled
        if self.verify_signatures {
            let validator_hotkey =
                Hotkey::new(signature_data.validator_hotkey.clone()).map_err(|e| anyhow!(e))?;

            // Generate canonical request for verification
            let canonical = Self::canonical_request(
                &signature_data.method,
                &signature_data.path,
                &signature_data.timestamp,
                &signature_data.nonce,
                body,
            );

            // Verify body hash if present
            if let Some(body_data) = body {
                if let Some(expected_hash) = &signature_data.body_hash {
                    let mut hasher = Hasher::new();
                    hasher.update(body_data);
                    let actual_hash = hasher.finalize().to_hex().to_string();

                    if actual_hash != *expected_hash {
                        return Err(anyhow!("Body hash mismatch"));
                    }
                }
            }

            // Verify signature using bittensor
            if let Err(e) = bittensor::utils::verify_bittensor_signature(
                &validator_hotkey,
                &signature_data.signature,
                canonical.as_bytes(),
            ) {
                warn!(
                    "Signature verification failed for validator {}: {}",
                    signature_data.validator_hotkey, e
                );
                return Err(anyhow!("Invalid signature"));
            }
        }

        debug!(
            "Request verified successfully from validator: {}",
            signature_data.validator_hotkey
        );

        Ok(())
    }

    /// Clean up expired nonces
    pub async fn cleanup_expired_nonces(&self) -> Result<()> {
        let mut used_nonces = self.used_nonces.write().await;
        let now = Utc::now();
        let cutoff = now - self.max_request_age - Duration::hours(1);

        let initial_count = used_nonces.len();
        used_nonces.retain(|_, timestamp| *timestamp > cutoff);
        let removed = initial_count - used_nonces.len();

        if removed > 0 {
            debug!("Cleaned up {} expired nonces", removed);
        }

        Ok(())
    }

    /// Create a request signature for testing
    #[cfg(test)]
    pub fn create_test_signature(
        validator_hotkey: &str,
        method: &str,
        path: &str,
        body: Option<&[u8]>,
    ) -> RequestSignature {
        let timestamp = Utc::now();
        let nonce = uuid::Uuid::new_v4().to_string();

        let mut hasher = Hasher::new();
        let body_hash = if let Some(body_data) = body {
            hasher.update(body_data);
            Some(hasher.finalize().to_hex().to_string())
        } else {
            None
        };

        RequestSignature {
            validator_hotkey: validator_hotkey.to_string(),
            signature: "test_signature".to_string(),
            timestamp,
            nonce,
            method: method.to_string(),
            path: path.to_string(),
            body_hash,
        }
    }
}

/// gRPC interceptor for request verification
pub struct RequestVerificationInterceptor {
    verification_service: Arc<RequestVerificationService>,
}

impl RequestVerificationInterceptor {
    pub fn new(verification_service: Arc<RequestVerificationService>) -> Self {
        Self {
            verification_service,
        }
    }

    /// Verify a gRPC request
    pub async fn verify_request<T>(
        &self,
        request: &tonic::Request<T>,
    ) -> Result<(), tonic::Status> {
        // Extract signature headers
        let headers = request.metadata();

        let validator_hotkey = headers
            .get("x-validator-hotkey")
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| tonic::Status::unauthenticated("Missing validator hotkey header"))?;

        let signature = headers
            .get("x-request-signature")
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| tonic::Status::unauthenticated("Missing request signature header"))?;

        let timestamp_str = headers
            .get("x-request-timestamp")
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| tonic::Status::unauthenticated("Missing request timestamp header"))?;

        let nonce = headers
            .get("x-request-nonce")
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| tonic::Status::unauthenticated("Missing request nonce header"))?;

        let timestamp = DateTime::parse_from_rfc3339(timestamp_str)
            .map_err(|_| tonic::Status::invalid_argument("Invalid timestamp format"))?
            .with_timezone(&Utc);

        let signature_data = RequestSignature {
            validator_hotkey: validator_hotkey.to_string(),
            signature: signature.to_string(),
            timestamp,
            nonce: nonce.to_string(),
            method: "POST".to_string(), // gRPC always uses POST
            path: "/grpc".to_string(),
            body_hash: None, // gRPC bodies are handled differently
        };

        // Verify the request
        self.verification_service
            .verify_request(&signature_data, None)
            .await
            .map_err(|e| {
                debug!("Request verification failed: {}", e);
                tonic::Status::unauthenticated(format!("Request verification failed: {e}"))
            })?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_request_verification() {
        let service = RequestVerificationService::new(
            Duration::minutes(5),
            vec![],
            false, // Disable signature verification for test
        );

        let signature = RequestSignature {
            validator_hotkey: "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string(),
            signature: "test_signature".to_string(),
            timestamp: Utc::now(),
            nonce: uuid::Uuid::new_v4().to_string(),
            method: "POST".to_string(),
            path: "/api/test".to_string(),
            body_hash: None,
        };

        // Should succeed
        assert!(service.verify_request(&signature, None).await.is_ok());

        // Should fail with duplicate nonce
        assert!(service.verify_request(&signature, None).await.is_err());
    }

    #[tokio::test]
    async fn test_request_age_validation() {
        let service = RequestVerificationService::new(Duration::minutes(5), vec![], false);

        // Old request
        let old_signature = RequestSignature {
            validator_hotkey: "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string(),
            signature: "test_signature".to_string(),
            timestamp: Utc::now() - Duration::minutes(10),
            nonce: uuid::Uuid::new_v4().to_string(),
            method: "POST".to_string(),
            path: "/api/test".to_string(),
            body_hash: None,
        };

        // Should fail
        assert!(service.verify_request(&old_signature, None).await.is_err());

        // Future request (within allowed skew)
        let future_signature = RequestSignature {
            validator_hotkey: "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string(),
            signature: "test_signature".to_string(),
            timestamp: Utc::now() + Duration::seconds(30),
            nonce: uuid::Uuid::new_v4().to_string(),
            method: "POST".to_string(),
            path: "/api/test".to_string(),
            body_hash: None,
        };

        // Should succeed (within clock skew tolerance)
        assert!(service
            .verify_request(&future_signature, None)
            .await
            .is_ok());
    }

    #[tokio::test]
    async fn test_canonical_request_generation() {
        let timestamp = Utc::now();
        let body = b"test body content";

        let canonical = RequestVerificationService::canonical_request(
            "POST",
            "/api/test",
            &timestamp,
            "test-nonce",
            Some(body),
        );

        assert!(canonical.contains("POST"));
        assert!(canonical.contains("/api/test"));
        assert!(canonical.contains(&timestamp.to_rfc3339()));
        assert!(canonical.contains("test-nonce"));

        // Should contain body hash
        let mut hasher = Hasher::new();
        hasher.update(body);
        let expected_hash = hasher.finalize().to_hex().to_string();
        assert!(canonical.contains(&expected_hash));
    }

    #[tokio::test]
    async fn test_allowlist_validation() {
        let allowed_hotkey =
            Hotkey::new("5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string()).unwrap();
        let service =
            RequestVerificationService::new(Duration::minutes(5), vec![allowed_hotkey], false);

        // Allowed validator
        let allowed_signature = RequestSignature {
            validator_hotkey: "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string(),
            signature: "test_signature".to_string(),
            timestamp: Utc::now(),
            nonce: uuid::Uuid::new_v4().to_string(),
            method: "POST".to_string(),
            path: "/api/test".to_string(),
            body_hash: None,
        };

        // Should succeed
        assert!(service
            .verify_request(&allowed_signature, None)
            .await
            .is_ok());

        // Disallowed validator
        let disallowed_signature = RequestSignature {
            validator_hotkey: "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM".to_string(),
            signature: "test_signature".to_string(),
            timestamp: Utc::now(),
            nonce: uuid::Uuid::new_v4().to_string(),
            method: "POST".to_string(),
            path: "/api/test".to_string(),
            body_hash: None,
        };

        // Should fail
        assert!(service
            .verify_request(&disallowed_signature, None)
            .await
            .is_err());
    }
}
