//! Canonical request signing utilities shared between SDK (client) and backend (server).
//!
//! Both sides MUST use these same functions to construct the canonical message
//! that gets signed/verified. This ensures client-generated signatures are
//! verifiable by the server.

use super::core::hash_blake3_string;

/// Normalize a URL path for canonical message construction.
///
/// Both client and server MUST use this same function.
///
/// Rules:
/// - Strip trailing slash (except for "/")
/// - Keep query string as-is
/// - Keep URL encoding as-is
///
/// # Examples
/// ```
/// use basilica_common::crypto::request_signing::canonical_path;
///
/// assert_eq!(canonical_path("/health/"), "/health");
/// assert_eq!(canonical_path("/"), "/");
/// assert_eq!(canonical_path("/billing/balance?foo=bar"), "/billing/balance?foo=bar");
/// assert_eq!(canonical_path("/deployments/my%20app"), "/deployments/my%20app");
/// ```
pub fn canonical_path(path: &str) -> &str {
    if path.len() > 1 && path.ends_with('/') {
        &path[..path.len() - 1]
    } else {
        path
    }
}

/// Build the canonical message string that gets signed.
///
/// Both client and server MUST use this same function.
///
/// Format: `"<METHOD>:<CANONICAL_PATH>:<BODY_BLAKE3>:<TIMESTAMP>"`
///
/// # Arguments
/// * `method` - HTTP method (e.g., "GET", "POST")
/// * `path` - URL path with optional query string (will be canonicalized)
/// * `body` - Request body bytes (empty slice for bodyless requests)
/// * `timestamp` - Unix timestamp as string
///
/// # Examples
/// ```
/// use basilica_common::crypto::request_signing::build_canonical_message;
///
/// let msg = build_canonical_message("GET", "/health", b"", "1700000000");
/// assert!(msg.starts_with("GET:/health:"));
/// assert!(msg.ends_with(":1700000000"));
/// ```
pub fn build_canonical_message(method: &str, path: &str, body: &[u8], timestamp: &str) -> String {
    let path = canonical_path(path);
    let body_hash = hash_blake3_string(body);
    format!("{method}:{path}:{body_hash}:{timestamp}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canonical_path_strips_trailing_slash() {
        assert_eq!(canonical_path("/health/"), "/health");
        assert_eq!(canonical_path("/api/v1/test/"), "/api/v1/test");
    }

    #[test]
    fn test_canonical_path_preserves_root() {
        assert_eq!(canonical_path("/"), "/");
    }

    #[test]
    fn test_canonical_path_preserves_no_trailing_slash() {
        assert_eq!(canonical_path("/health"), "/health");
        assert_eq!(canonical_path("/billing/balance"), "/billing/balance");
    }

    #[test]
    fn test_canonical_path_preserves_query_string() {
        assert_eq!(
            canonical_path("/billing/balance?foo=bar"),
            "/billing/balance?foo=bar"
        );
        assert_eq!(
            canonical_path("/api?a=1&b=2"),
            "/api?a=1&b=2"
        );
    }

    #[test]
    fn test_canonical_path_preserves_url_encoding() {
        assert_eq!(
            canonical_path("/deployments/my%20app"),
            "/deployments/my%20app"
        );
    }

    #[test]
    fn test_build_canonical_message_format() {
        let msg = build_canonical_message("GET", "/billing/balance", b"", "1700000000");
        let parts: Vec<&str> = msg.splitn(4, ':').collect();
        assert_eq!(parts.len(), 4);
        assert_eq!(parts[0], "GET");
        assert_eq!(parts[1], "/billing/balance");
        // parts[2] is the blake3 hash of empty bytes
        assert_eq!(parts[2].len(), 64); // blake3 hex hash
        assert_eq!(parts[3], "1700000000");
    }

    #[test]
    fn test_build_canonical_message_with_body() {
        let body = br#"{"amount": 100}"#;
        let msg1 = build_canonical_message("POST", "/transfer", body, "1700000000");
        let msg2 = build_canonical_message("POST", "/transfer", b"", "1700000000");
        // Different body should produce different message
        assert_ne!(msg1, msg2);
    }

    #[test]
    fn test_build_canonical_message_strips_trailing_slash() {
        let msg1 = build_canonical_message("GET", "/health/", b"", "1700000000");
        let msg2 = build_canonical_message("GET", "/health", b"", "1700000000");
        assert_eq!(msg1, msg2);
    }

    #[test]
    fn test_build_canonical_message_different_methods() {
        let msg1 = build_canonical_message("GET", "/test", b"", "1700000000");
        let msg2 = build_canonical_message("POST", "/test", b"", "1700000000");
        assert_ne!(msg1, msg2);
    }

    #[test]
    fn test_build_canonical_message_different_paths() {
        let msg1 = build_canonical_message("GET", "/path1", b"", "1700000000");
        let msg2 = build_canonical_message("GET", "/path2", b"", "1700000000");
        assert_ne!(msg1, msg2);
    }

    #[test]
    fn test_build_canonical_message_different_timestamps() {
        let msg1 = build_canonical_message("GET", "/test", b"", "1700000000");
        let msg2 = build_canonical_message("GET", "/test", b"", "1700000001");
        assert_ne!(msg1, msg2);
    }

    #[test]
    fn test_empty_body_hash_consistency() {
        let msg1 = build_canonical_message("GET", "/test", b"", "1700000000");
        let msg2 = build_canonical_message("DELETE", "/other", b"", "9999999999");
        // The body hash portion should be the same since both have empty bodies
        let hash1 = msg1.splitn(4, ':').nth(2).unwrap();
        let hash2 = msg2.splitn(4, ':').nth(2).unwrap();
        assert_eq!(hash1, hash2);
    }
}
