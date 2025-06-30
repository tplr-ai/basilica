//! # Authentication Module
//!
//! Provides JWT-based authentication for validator sessions with proper security controls.

use anyhow::{anyhow, Result};
use chrono::{Duration, Utc};
use jsonwebtoken::{decode, encode, Algorithm, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, warn};

use common::identity::Hotkey;

/// JWT claims for validator authentication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorClaims {
    /// Subject - validator hotkey
    pub sub: String,
    /// Issuer - miner identifier
    pub iss: String,
    /// Audience - expected to be this miner's ID
    pub aud: String,
    /// Expiration time (Unix timestamp)
    pub exp: i64,
    /// Issued at (Unix timestamp)
    pub iat: i64,
    /// Not before (Unix timestamp)
    pub nbf: i64,
    /// Unique token ID
    pub jti: String,
    /// Session ID for tracking
    pub session_id: String,
    /// Validator permissions
    pub permissions: Vec<String>,
}

/// Session information for active validators
#[derive(Debug, Clone)]
pub struct ValidatorSession {
    pub hotkey: Hotkey,
    pub session_id: String,
    pub token_id: String,
    pub created_at: chrono::DateTime<Utc>,
    pub expires_at: chrono::DateTime<Utc>,
    pub last_activity: chrono::DateTime<Utc>,
    pub ip_address: Option<IpAddr>,
    pub permissions: Vec<String>,
}

/// Revoked token information
#[derive(Debug, Clone)]
struct RevokedToken {
    token_id: String,
    revoked_at: chrono::DateTime<Utc>,
    expires_at: chrono::DateTime<Utc>,
}

/// JWT authentication service
#[derive(Clone)]
pub struct JwtAuthService {
    encoding_key: EncodingKey,
    decoding_key: DecodingKey,
    algorithm: Algorithm,
    issuer: String,
    audience: String,
    token_expiration: Duration,
    sessions: Arc<RwLock<HashMap<String, ValidatorSession>>>,
    revoked_tokens: Arc<RwLock<HashMap<String, RevokedToken>>>,
}

impl std::fmt::Debug for JwtAuthService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JwtAuthService")
            .field("algorithm", &self.algorithm)
            .field("issuer", &self.issuer)
            .field("audience", &self.audience)
            .field("token_expiration", &self.token_expiration)
            .field("sessions_count", &"<hidden>")
            .field("revoked_tokens_count", &"<hidden>")
            .finish()
    }
}

impl JwtAuthService {
    /// Create a new JWT authentication service
    pub fn new(
        jwt_secret: &str,
        issuer: String,
        audience: String,
        token_expiration: Duration,
    ) -> Result<Self> {
        if jwt_secret.len() < 32 {
            return Err(anyhow!(
                "JWT secret must be at least 32 characters for security"
            ));
        }

        Ok(Self {
            encoding_key: EncodingKey::from_secret(jwt_secret.as_bytes()),
            decoding_key: DecodingKey::from_secret(jwt_secret.as_bytes()),
            algorithm: Algorithm::HS256,
            issuer,
            audience,
            token_expiration,
            sessions: Arc::new(RwLock::new(HashMap::new())),
            revoked_tokens: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Generate a new JWT token for a validator
    pub async fn generate_token(
        &self,
        validator_hotkey: &Hotkey,
        session_id: &str,
        permissions: Vec<String>,
        ip_address: Option<IpAddr>,
    ) -> Result<String> {
        let now = Utc::now();
        let expires_at = now + self.token_expiration;
        let token_id = uuid::Uuid::new_v4().to_string();

        let claims = ValidatorClaims {
            sub: validator_hotkey.to_string(),
            iss: self.issuer.clone(),
            aud: self.audience.clone(),
            exp: expires_at.timestamp(),
            iat: now.timestamp(),
            nbf: now.timestamp(),
            jti: token_id.clone(),
            session_id: session_id.to_string(),
            permissions: permissions.clone(),
        };

        let header = Header::new(self.algorithm);
        let token = encode(&header, &claims, &self.encoding_key)?;

        // Store session information
        let session = ValidatorSession {
            hotkey: validator_hotkey.clone(),
            session_id: session_id.to_string(),
            token_id: token_id.clone(),
            created_at: now,
            expires_at,
            last_activity: now,
            ip_address,
            permissions,
        };

        let mut sessions = self.sessions.write().await;
        sessions.insert(session_id.to_string(), session);

        debug!(
            "Generated JWT token for validator {} with session {}",
            validator_hotkey, session_id
        );

        Ok(token)
    }

    /// Validate a JWT token and return the claims
    pub async fn validate_token(&self, token: &str) -> Result<ValidatorClaims> {
        // Decode and validate the token
        let mut validation = Validation::new(self.algorithm);
        validation.set_audience(&[&self.audience]);
        validation.set_issuer(&[&self.issuer]);
        validation.validate_exp = true;
        validation.validate_nbf = true;
        validation.leeway = 0; // No leeway for clock skew

        let token_data = decode::<ValidatorClaims>(token, &self.decoding_key, &validation)?;
        let claims = token_data.claims;

        // Check if token is revoked
        let revoked = self.revoked_tokens.read().await;
        if revoked.contains_key(&claims.jti) {
            return Err(anyhow!("Token has been revoked"));
        }

        // Update last activity for the session
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.get_mut(&claims.session_id) {
            session.last_activity = Utc::now();
        } else {
            return Err(anyhow!("Session not found"));
        }

        Ok(claims)
    }

    /// Revoke a specific token with expiration time
    pub async fn revoke_token(
        &self,
        token_id: &str,
        expires_at: chrono::DateTime<Utc>,
    ) -> Result<()> {
        let mut revoked = self.revoked_tokens.write().await;
        let revoked_token = RevokedToken {
            token_id: token_id.to_string(),
            revoked_at: Utc::now(),
            expires_at,
        };
        revoked.insert(token_id.to_string(), revoked_token);

        debug!("Revoked token: {}", token_id);
        Ok(())
    }

    /// Revoke all tokens for a session
    pub async fn revoke_session(&self, session_id: &str) -> Result<()> {
        let mut sessions = self.sessions.write().await;

        if let Some(session) = sessions.remove(session_id) {
            let mut revoked = self.revoked_tokens.write().await;
            let revoked_token = RevokedToken {
                token_id: session.token_id.clone(),
                revoked_at: Utc::now(),
                expires_at: session.expires_at,
            };
            revoked.insert(session.token_id, revoked_token);

            debug!("Revoked session: {}", session_id);
            Ok(())
        } else {
            Err(anyhow!("Session not found"))
        }
    }

    /// Clean up expired sessions and tokens
    pub async fn cleanup_expired(&self) -> Result<()> {
        let now = Utc::now();
        let mut sessions = self.sessions.write().await;
        let mut revoked = self.revoked_tokens.write().await;

        // Remove expired sessions
        let expired_sessions: Vec<String> = sessions
            .iter()
            .filter(|(_, session)| session.expires_at < now)
            .map(|(id, _)| id.clone())
            .collect();

        let expired_count = expired_sessions.len();
        for session_id in expired_sessions {
            if let Some(session) = sessions.remove(&session_id) {
                let revoked_token = RevokedToken {
                    token_id: session.token_id.clone(),
                    revoked_at: now,
                    expires_at: session.expires_at,
                };
                revoked.insert(session.token_id, revoked_token);
            }
        }

        // Clean up old revoked tokens (keep for 24 hours after expiration)
        let initial_revoked_count = revoked.len();
        let mut oldest_revoked_at = None;
        revoked.retain(|token_id, token| {
            // Keep tokens that haven't been expired for more than 24 hours
            let should_keep = token.expires_at + Duration::hours(24) > now;
            if !should_keep {
                debug!(
                    "Removing revoked token {} (revoked at: {}, expired at: {})",
                    token_id, token.revoked_at, token.expires_at
                );
            } else if oldest_revoked_at.is_none() || token.revoked_at < oldest_revoked_at.unwrap() {
                oldest_revoked_at = Some(token.revoked_at);
            }
            should_keep
        });
        let cleaned_revoked = initial_revoked_count - revoked.len();

        if expired_count > 0 || cleaned_revoked > 0 {
            debug!(
                "Cleaned up {} expired sessions and {} old revoked tokens",
                expired_count, cleaned_revoked
            );
        }

        if let Some(oldest) = oldest_revoked_at {
            debug!(
                "Oldest revoked token is from {} ({} ago)",
                oldest,
                now - oldest
            );
        }

        Ok(())
    }

    /// Get active session information
    pub async fn get_session(&self, session_id: &str) -> Result<ValidatorSession> {
        let sessions = self.sessions.read().await;
        sessions
            .get(session_id)
            .cloned()
            .ok_or_else(|| anyhow!("Session not found"))
    }

    /// List all active sessions
    pub async fn list_active_sessions(&self) -> Vec<ValidatorSession> {
        let sessions = self.sessions.read().await;
        sessions.values().cloned().collect()
    }

    /// Update session IP address (for tracking)
    pub async fn update_session_ip(&self, session_id: &str, ip_address: IpAddr) -> Result<()> {
        let mut sessions = self.sessions.write().await;

        if let Some(session) = sessions.get_mut(session_id) {
            let old_ip = session.ip_address;
            session.ip_address = Some(ip_address);

            if old_ip != Some(ip_address) {
                warn!(
                    "Session {} IP address changed from {:?} to {}",
                    session_id, old_ip, ip_address
                );
            }

            Ok(())
        } else {
            Err(anyhow!("Session not found"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_jwt_token_generation_and_validation() {
        let secret = "test_secret_key_that_is_long_enough_for_security";
        let auth_service = JwtAuthService::new(
            secret,
            "test-miner".to_string(),
            "test-miner".to_string(),
            Duration::hours(1),
        )
        .unwrap();

        let hotkey =
            Hotkey::new("5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string()).unwrap();
        let session_id = "test-session-123";
        let permissions = vec!["executor.list".to_string(), "executor.access".to_string()];

        // Generate token
        let token = auth_service
            .generate_token(&hotkey, session_id, permissions.clone(), None)
            .await
            .unwrap();

        // Validate token
        let claims = auth_service.validate_token(&token).await.unwrap();

        assert_eq!(claims.sub, hotkey.to_string());
        assert_eq!(claims.session_id, session_id);
        assert_eq!(claims.permissions, permissions);
        assert_eq!(claims.iss, "test-miner");
        assert_eq!(claims.aud, "test-miner");
    }

    #[tokio::test]
    async fn test_token_revocation() {
        let secret = "test_secret_key_that_is_long_enough_for_security";
        let auth_service = JwtAuthService::new(
            secret,
            "test-miner".to_string(),
            "test-miner".to_string(),
            Duration::hours(1),
        )
        .unwrap();

        let hotkey =
            Hotkey::new("5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string()).unwrap();
        let session_id = "test-session-456";

        // Generate token
        let token = auth_service
            .generate_token(&hotkey, session_id, vec![], None)
            .await
            .unwrap();

        // Validate token succeeds
        let claims = auth_service.validate_token(&token).await.unwrap();

        // Revoke token with expiration time
        let expires_at = Utc::now() + Duration::hours(1);
        auth_service
            .revoke_token(&claims.jti, expires_at)
            .await
            .unwrap();

        // Validation should now fail
        let result = auth_service.validate_token(&token).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("revoked"));
    }

    #[tokio::test]
    async fn test_session_management() {
        let secret = "test_secret_key_that_is_long_enough_for_security";
        let auth_service = JwtAuthService::new(
            secret,
            "test-miner".to_string(),
            "test-miner".to_string(),
            Duration::hours(1),
        )
        .unwrap();

        let hotkey =
            Hotkey::new("5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string()).unwrap();
        let session_id = "test-session-789";

        // Generate token
        let _token = auth_service
            .generate_token(&hotkey, session_id, vec![], None)
            .await
            .unwrap();

        // Get session
        let session = auth_service.get_session(session_id).await.unwrap();
        assert_eq!(session.hotkey, hotkey);
        assert_eq!(session.session_id, session_id);

        // List active sessions
        let sessions = auth_service.list_active_sessions().await;
        assert_eq!(sessions.len(), 1);

        // Revoke session
        auth_service.revoke_session(session_id).await.unwrap();

        // Session should no longer exist
        let result = auth_service.get_session(session_id).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_expired_token_validation() {
        let secret = "test_secret_key_that_is_long_enough_for_security";
        let auth_service = JwtAuthService::new(
            secret,
            "test-miner".to_string(),
            "test-miner".to_string(),
            Duration::seconds(1), // Very short expiration
        )
        .unwrap();

        let hotkey =
            Hotkey::new("5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string()).unwrap();
        let session_id = "expired-session";

        // Generate token (already expired)
        let token = auth_service
            .generate_token(&hotkey, session_id, vec![], None)
            .await
            .unwrap();

        // Sleep to ensure token is expired
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;

        // Validation should fail
        let result = auth_service.validate_token(&token).await;
        assert!(result.is_err());
    }
}
