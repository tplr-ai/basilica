//! API Key (Personal Access Token) management module
//!
//! This module provides functionality for generating, validating, and managing
//! API keys for the Basilica API.

use argon2::{
    password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString},
    Argon2,
};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use rand::{rngs::StdRng, RngCore, SeedableRng};
use serde::{Deserialize, Serialize};
use sqlx::{FromRow, PgPool};
use std::fmt;
use std::str::FromStr;
use tracing::debug;

/// API key database row
#[derive(Debug, FromRow, Serialize, Deserialize)]
pub struct ApiKey {
    pub user_id: String,
    pub kid: String,
    pub name: String,
    pub hash: String,
    pub scopes: Vec<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_used_at: Option<chrono::DateTime<chrono::Utc>>,
}

/// API key verification result
#[derive(Debug)]
pub struct VerifiedApiKey {
    pub user_id: String,
    pub scopes: Vec<String>,
}

/// Error type for API key operations
#[derive(Debug, thiserror::Error)]
pub enum ApiKeyError {
    #[error("Invalid token format")]
    InvalidFormat,

    #[error("API key not found")]
    NotFound,

    #[error("Invalid API key secret")]
    InvalidSecret,

    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("Hashing error: {0}")]
    Hashing(String),
}

/// Complete API key data generated when creating a new key.
/// Contains all components including the salt needed for hashing.
#[derive(Clone, PartialEq)]
pub struct GeneratedApiKey {
    /// Key ID - 16 bytes (displayed as 32 hex chars in database)
    pub kid_bytes: [u8; 16],
    /// Secret - 32 bytes for cryptographic strength
    pub secret_bytes: [u8; 32],
    /// Salt for Argon2 hashing - 16 bytes
    pub salt: [u8; 16],
}

impl GeneratedApiKey {
    /// Generate a new API key with random values
    pub fn generate() -> Self {
        let mut rng = StdRng::from_entropy();
        let mut kid_bytes = [0u8; 16];
        let mut secret_bytes = [0u8; 32];
        let mut salt = [0u8; 16];

        rng.fill_bytes(&mut kid_bytes);
        rng.fill_bytes(&mut secret_bytes);
        rng.fill_bytes(&mut salt);

        Self {
            kid_bytes,
            secret_bytes,
            salt,
        }
    }

    /// Get the kid as a hex string (for database storage)
    pub fn kid_hex(&self) -> String {
        hex::encode(self.kid_bytes)
    }

    /// Get the secret as bytes (for hashing)
    pub fn secret(&self) -> &[u8] {
        &self.secret_bytes
    }

    /// Generate the hash in PHC string format for database storage.
    /// This computes the Argon2 hash using the stored salt
    pub fn hash_string(&self) -> Result<String, ApiKeyError> {
        let argon2 = Argon2::default();
        let salt_string =
            SaltString::encode_b64(&self.salt).map_err(|e| ApiKeyError::Hashing(e.to_string()))?;

        // Hash the secret with the stored salt
        let password_hash = argon2
            .hash_password(&self.secret_bytes, &salt_string)
            .map_err(|e| ApiKeyError::Hashing(e.to_string()))?;

        // Return the PHC string format
        Ok(password_hash.to_string())
    }

    /// Convert to an ApiKeyToken (drops the salt)
    pub fn into_token(self) -> ApiKeyToken {
        ApiKeyToken {
            kid_bytes: self.kid_bytes,
            secret_bytes: self.secret_bytes,
        }
    }
}

impl fmt::Debug for GeneratedApiKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Show truncated kid for correlation, redact all sensitive data
        let kid_hex = hex::encode(&self.kid_bytes[..4]);
        f.debug_struct("GeneratedApiKey")
            .field("kid_bytes", &format!("{}...", kid_hex))
            .field("secret_bytes", &"<redacted>")
            .field("salt", &"<redacted>")
            .finish()
    }
}

/// API key token representation without salt.
/// This is what users work with - contains only the kid and secret.
/// The salt is stored in the database as part of the hash.
#[derive(Clone, PartialEq)]
pub struct ApiKeyToken {
    /// Key ID - 16 bytes (displayed as 32 hex chars in database)
    pub kid_bytes: [u8; 16],
    /// Secret - 32 bytes for cryptographic strength
    pub secret_bytes: [u8; 32],
}

impl fmt::Debug for ApiKeyToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Show truncated kid for correlation, redact secret
        let kid_hex = hex::encode(&self.kid_bytes[..4]);
        f.debug_struct("ApiKeyToken")
            .field("kid_bytes", &format!("{}...", kid_hex))
            .field("secret_bytes", &"<redacted>")
            .finish()
    }
}

impl ApiKeyToken {
    /// Get the kid as a hex string (for database storage)
    pub fn kid_hex(&self) -> String {
        hex::encode(self.kid_bytes)
    }

    /// Get the secret as bytes (for hashing/verification)
    pub fn secret(&self) -> &[u8] {
        &self.secret_bytes
    }
}

impl fmt::Display for ApiKeyToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Concatenate kid and secret bytes
        let mut combined = Vec::with_capacity(48);
        combined.extend_from_slice(&self.kid_bytes);
        combined.extend_from_slice(&self.secret_bytes);

        // Encode as base64url
        let encoded = URL_SAFE_NO_PAD.encode(&combined);
        write!(f, "basilica_{}", encoded)
    }
}

impl FromStr for ApiKeyToken {
    type Err = ApiKeyError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Check prefix
        if !s.starts_with("basilica_") {
            return Err(ApiKeyError::InvalidFormat);
        }

        // Strip prefix and decode
        let encoded = &s[9..]; // Skip "basilica_"
        let decoded = URL_SAFE_NO_PAD
            .decode(encoded)
            .map_err(|_| ApiKeyError::InvalidFormat)?;

        // Validate length (16 bytes kid + 32 bytes secret = 48 bytes)
        if decoded.len() != 48 {
            return Err(ApiKeyError::InvalidFormat);
        }

        // Split into kid and secret
        let mut kid_bytes = [0u8; 16];
        let mut secret_bytes = [0u8; 32];
        kid_bytes.copy_from_slice(&decoded[0..16]);
        secret_bytes.copy_from_slice(&decoded[16..48]);

        // Note: When parsing from string, we don't have the salt.
        // The salt is stored in the database as part of the hash.
        Ok(Self {
            kid_bytes,
            secret_bytes,
        })
    }
}

/// Insert a new API key into the database
pub async fn insert_api_key(
    pool: &PgPool,
    user_id: &str,
    name: &str,
    kid: &str,
    hash: &str,
    scopes: &[String],
) -> Result<ApiKey, ApiKeyError> {
    let key = sqlx::query_as::<_, ApiKey>(
        r#"
        INSERT INTO api_keys (user_id, kid, name, hash, scopes)
        VALUES ($1, $2, $3, $4, $5)
        RETURNING *
        "#,
    )
    .bind(user_id)
    .bind(kid)
    .bind(name)
    .bind(hash)
    .bind(scopes)
    .fetch_one(pool)
    .await?;

    Ok(key)
}

/// Get an API key by kid
pub async fn get_api_key_by_kid(pool: &PgPool, kid: &str) -> Result<Option<ApiKey>, ApiKeyError> {
    let key = sqlx::query_as::<_, ApiKey>(
        r#"
        SELECT * FROM api_keys
        WHERE kid = $1
        "#,
    )
    .bind(kid)
    .fetch_optional(pool)
    .await?;

    Ok(key)
}

/// List all API keys for a user
pub async fn list_user_api_keys(pool: &PgPool, user_id: &str) -> Result<Vec<ApiKey>, ApiKeyError> {
    let keys = sqlx::query_as::<_, ApiKey>(
        r#"
        SELECT * FROM api_keys
        WHERE user_id = $1
        ORDER BY created_at DESC
        "#,
    )
    .bind(user_id)
    .fetch_all(pool)
    .await?;

    Ok(keys)
}

/// Delete an API key by user_id and name
pub async fn delete_api_key_by_name(
    pool: &PgPool,
    user_id: &str,
    name: &str,
) -> Result<bool, ApiKeyError> {
    let result = sqlx::query(
        r#"
        DELETE FROM api_keys
        WHERE user_id = $1 AND name = $2
        "#,
    )
    .bind(user_id)
    .bind(name)
    .execute(pool)
    .await?;

    Ok(result.rows_affected() > 0)
}

/// Update the last_used_at timestamp for an API key
pub async fn update_last_used(pool: &PgPool, kid: &str) -> Result<(), ApiKeyError> {
    sqlx::query(
        r#"
        UPDATE api_keys
        SET last_used_at = now()
        WHERE kid = $1
        "#,
    )
    .bind(kid)
    .execute(pool)
    .await?;

    Ok(())
}

/// Verify an API key token
pub async fn verify_api_key(pool: &PgPool, token: &str) -> Result<VerifiedApiKey, ApiKeyError> {
    // Parse the token using ApiKeyToken
    let api_token = ApiKeyToken::from_str(token)?;

    // Get kid as hex for database lookup
    let kid = api_token.kid_hex();

    // Fetch the key from database
    let key = get_api_key_by_kid(pool, &kid)
        .await?
        .ok_or(ApiKeyError::NotFound)?;

    // Verify the secret bytes against the hash
    let argon2 = Argon2::default();
    let parsed_hash =
        PasswordHash::new(&key.hash).map_err(|e| ApiKeyError::Hashing(e.to_string()))?;

    argon2
        .verify_password(api_token.secret(), &parsed_hash)
        .map_err(|_| ApiKeyError::InvalidSecret)?;

    // Update last_used_at asynchronously (don't wait for it)
    let pool_clone = pool.clone();
    let kid_clone = kid.clone();
    tokio::spawn(async move {
        if let Err(e) = update_last_used(&pool_clone, &kid_clone).await {
            debug!("Failed to update last_used_at for API key: {}", e);
        }
    });

    Ok(VerifiedApiKey {
        user_id: key.user_id,
        scopes: key.scopes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parse a display token into its components
    fn parse_display_token(token: &str) -> Result<(String, String), ApiKeyError> {
        // Parse using ApiKeyToken
        let api_token = ApiKeyToken::from_str(token)?;

        // Get kid as hex string for database lookup
        let kid = api_token.kid_hex();

        // Get secret as base64url encoded string
        let secret = URL_SAFE_NO_PAD.encode(api_token.secret_bytes);

        Ok((kid, secret))
    }

    #[test]
    fn test_api_key_token_roundtrip() {
        // Generate a new key
        let generated = GeneratedApiKey::generate();

        // Convert to token
        let token = generated.into_token();

        // Convert to string
        let token_str = token.to_string();

        // Parse it back
        let parsed = ApiKeyToken::from_str(&token_str).unwrap();

        // Kid and secret should match
        assert_eq!(token.kid_bytes, parsed.kid_bytes);
        assert_eq!(token.secret_bytes, parsed.secret_bytes);
    }

    #[test]
    fn test_api_key_token_format() {
        let generated = GeneratedApiKey::generate();
        let token = generated.into_token();
        let token_str = token.to_string();

        // Check format
        assert!(token_str.starts_with("basilica_"));

        // Check length - should be around 64 chars (9 prefix + ~64 base64 for 48 bytes)
        // Base64 encoding of 48 bytes = ceil(48 * 4/3) = 64 chars without padding
        assert!(token_str.len() >= 60 && token_str.len() <= 75);

        // The token body can contain underscores since we use base64url encoding
        // which uses - and _ as special characters. The key point is that we parse
        // by taking everything after "basilica_" as one block, not by splitting on _
        let after_prefix = &token_str[9..];

        // Should be valid base64url characters only
        assert!(
            after_prefix
                .chars()
                .all(|c| { c.is_ascii_alphanumeric() || c == '-' || c == '_' }),
            "Token body should only contain base64url characters"
        );
    }

    #[test]
    fn test_api_key_token_from_str_errors() {
        // Wrong prefix
        assert!(ApiKeyToken::from_str("wrong_abc123").is_err());

        // Too short
        assert!(ApiKeyToken::from_str("basilica_short").is_err());

        // Invalid base64
        assert!(ApiKeyToken::from_str("basilica_!!!invalid!!!").is_err());

        // Empty
        assert!(ApiKeyToken::from_str("").is_err());

        // Just prefix
        assert!(ApiKeyToken::from_str("basilica_").is_err());
    }

    #[test]
    fn test_generated_api_key() {
        let generated = GeneratedApiKey::generate();
        let kid = generated.kid_hex();
        let hash = generated.hash_string().unwrap();
        let token = generated.into_token();
        let display_token = token.to_string();

        // Check kid is 32 hex chars (16 bytes)
        assert_eq!(kid.len(), 32);
        assert!(kid.chars().all(|c| c.is_ascii_hexdigit()));

        // Check display token format
        assert!(display_token.starts_with("basilica_"));

        // Token should be parseable
        let parsed = ApiKeyToken::from_str(&display_token).unwrap();
        assert_eq!(parsed.kid_hex(), kid);

        // Check hash is not empty and is valid PHC format
        assert!(!hash.is_empty());
        assert!(hash.starts_with("$argon2"));
    }

    #[test]
    fn test_parse_display_token() {
        // Generate a real token
        let generated = GeneratedApiKey::generate();
        let api_token = generated.into_token();
        let token_str = api_token.to_string();

        // Parse it
        let (kid, _secret) = parse_display_token(&token_str).unwrap();

        // Kid should match
        assert_eq!(kid, api_token.kid_hex());

        // Invalid tokens should error
        assert!(parse_display_token("invalid_token").is_err());
        assert!(parse_display_token("basilica_tooshort").is_err());
    }

    #[test]
    fn test_api_key_no_delimiter_conflicts() {
        // Generate many tokens to ensure no delimiter issues
        for _ in 0..100 {
            let generated = GeneratedApiKey::generate();
            let token = generated.into_token();
            let token_str = token.to_string();

            // Should parse successfully every time
            let parsed = ApiKeyToken::from_str(&token_str).unwrap();
            // Kid and secret should match
            assert_eq!(token.kid_bytes, parsed.kid_bytes);
            assert_eq!(token.secret_bytes, parsed.secret_bytes);

            // parse_display_token should also work
            let (kid, _) = parse_display_token(&token_str).unwrap();
            assert_eq!(kid, token.kid_hex());
        }
    }

    #[test]
    fn test_argon2_hash_verification_with_bytes() {
        let generated = GeneratedApiKey::generate();
        let token = generated.into_token();
        let secret_bytes = token.secret();

        // Hash the secret bytes
        let argon2 = Argon2::default();
        let mut rng = StdRng::from_entropy();
        let salt = SaltString::generate(&mut rng);
        let hash = argon2
            .hash_password(secret_bytes, &salt)
            .unwrap()
            .to_string();

        // Verify correct secret
        let parsed_hash = PasswordHash::new(&hash).unwrap();
        assert!(argon2.verify_password(secret_bytes, &parsed_hash).is_ok());

        // Verify incorrect secret
        assert!(argon2
            .verify_password(b"wrong_secret", &parsed_hash)
            .is_err());
    }

    #[test]
    fn test_debug_impl_redacts_secrets() {
        // Test that Debug implementations don't expose sensitive data
        let generated = GeneratedApiKey::generate();
        let debug_str = format!("{:?}", generated);

        // Should NOT contain the actual secret bytes
        assert!(!debug_str.contains(&format!("{:?}", generated.secret_bytes)));
        // Should contain redacted marker
        assert!(debug_str.contains("<redacted>"));
        // Should have truncated kid
        assert!(debug_str.contains("..."));

        // Test ApiKeyToken Debug too
        let token = generated.into_token();
        let token_debug = format!("{:?}", token);

        // Should NOT contain the actual secret bytes
        assert!(!token_debug.contains(&format!("{:?}", token.secret_bytes)));
        // Should contain redacted marker
        assert!(token_debug.contains("<redacted>"));
        // Should have truncated kid
        assert!(token_debug.contains("..."));
    }

    #[test]
    fn test_generated_api_key_hash_deterministic() {
        let generated = GeneratedApiKey::generate();
        let secret_bytes = generated.secret_bytes;

        // Generate hash multiple times - should be the same
        let hash1 = generated.hash_string().unwrap();
        let hash2 = generated.hash_string().unwrap();
        assert_eq!(hash1, hash2);

        // Hash can verify the secret
        let argon2 = Argon2::default();
        let parsed_hash = PasswordHash::new(&hash1).unwrap();
        assert!(argon2.verify_password(&secret_bytes, &parsed_hash).is_ok());
    }
}
