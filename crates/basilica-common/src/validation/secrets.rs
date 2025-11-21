use crate::error::ValidationError;
use std::collections::BTreeSet;

/// Required keys for storage secrets
pub const REQUIRED_STORAGE_SECRET_KEYS: &[&str] = &[
    "STORAGE_ACCESS_KEY_ID",
    "STORAGE_SECRET_ACCESS_KEY",
    "STORAGE_BUCKET",
    "STORAGE_ENDPOINT",
    "STORAGE_BACKEND",
];

/// Allowed secret name prefixes (security policy)
pub const ALLOWED_SECRET_PREFIXES: &[&str] = &[
    "basilica-r2-credentials",
    "basilica-s3-credentials",
    "basilica-gcs-credentials",
    "user-storage-",
];

/// Validates a secret name against security policy
///
/// Secret names must match approved patterns to prevent
/// privilege escalation by referencing system secrets
pub fn validate_secret_name(secret_name: &str) -> Result<(), ValidationError> {
    if secret_name.is_empty() {
        return Err(ValidationError::InvalidFormat {
            field: "secret_name".to_string(),
            value: "(empty)".to_string(),
        });
    }

    if secret_name.len() > 253 {
        return Err(ValidationError::OutOfRange {
            field: "secret_name".to_string(),
            value: secret_name.len().to_string(),
            min: "1".to_string(),
            max: "253".to_string(),
        });
    }

    let is_allowed = ALLOWED_SECRET_PREFIXES
        .iter()
        .any(|prefix| secret_name.starts_with(prefix));

    if !is_allowed {
        return Err(ValidationError::ConstraintViolation {
            field: "secret_name".to_string(),
            constraint: format!("must start with: {}", ALLOWED_SECRET_PREFIXES.join(", ")),
        });
    }

    Ok(())
}

/// Validates that a secret contains all required keys for storage
pub fn validate_secret_keys(secret_keys: &[String]) -> Result<(), ValidationError> {
    let key_set: BTreeSet<&str> = secret_keys.iter().map(|s| s.as_str()).collect();

    let mut missing_keys = Vec::new();
    for required_key in REQUIRED_STORAGE_SECRET_KEYS {
        if !key_set.contains(required_key) {
            missing_keys.push(*required_key);
        }
    }

    if !missing_keys.is_empty() {
        return Err(ValidationError::MissingField {
            field: format!("missing: {}", missing_keys.join(", ")),
        });
    }

    Ok(())
}

/// Validates storage backend value
pub fn validate_storage_backend(backend: &str) -> Result<(), ValidationError> {
    let valid_backends = ["r2", "s3", "gcs"];

    if !valid_backends.contains(&backend.to_lowercase().as_str()) {
        return Err(ValidationError::InvalidEnum {
            enum_name: "storage_backend".to_string(),
            value: backend.to_string(),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_secret_name_valid() {
        assert!(validate_secret_name("basilica-r2-credentials").is_ok());
        assert!(validate_secret_name("basilica-s3-credentials").is_ok());
        assert!(validate_secret_name("basilica-gcs-credentials").is_ok());
        assert!(validate_secret_name("user-storage-custom").is_ok());
        assert!(validate_secret_name("user-storage-123").is_ok());
    }

    #[test]
    fn test_validate_secret_name_invalid_prefix() {
        assert!(validate_secret_name("kube-root-token").is_err());
        assert!(validate_secret_name("default-token-abc").is_err());
        assert!(validate_secret_name("custom-secret").is_err());
    }

    #[test]
    fn test_validate_secret_name_empty() {
        assert!(validate_secret_name("").is_err());
    }

    #[test]
    fn test_validate_secret_name_too_long() {
        let long_name = "a".repeat(254);
        assert!(validate_secret_name(&long_name).is_err());
    }

    #[test]
    fn test_validate_secret_keys_valid() {
        let keys = vec![
            "STORAGE_ACCESS_KEY_ID".to_string(),
            "STORAGE_SECRET_ACCESS_KEY".to_string(),
            "STORAGE_BUCKET".to_string(),
            "STORAGE_ENDPOINT".to_string(),
            "STORAGE_BACKEND".to_string(),
        ];
        assert!(validate_secret_keys(&keys).is_ok());
    }

    #[test]
    fn test_validate_secret_keys_with_extra() {
        let keys = vec![
            "STORAGE_ACCESS_KEY_ID".to_string(),
            "STORAGE_SECRET_ACCESS_KEY".to_string(),
            "STORAGE_BUCKET".to_string(),
            "STORAGE_ENDPOINT".to_string(),
            "STORAGE_BACKEND".to_string(),
            "EXTRA_KEY".to_string(),
        ];
        assert!(validate_secret_keys(&keys).is_ok());
    }

    #[test]
    fn test_validate_secret_keys_missing() {
        let keys = vec![
            "STORAGE_ACCESS_KEY_ID".to_string(),
            "STORAGE_SECRET_ACCESS_KEY".to_string(),
        ];
        assert!(validate_secret_keys(&keys).is_err());
    }

    #[test]
    fn test_validate_secret_keys_empty() {
        let keys: Vec<String> = vec![];
        assert!(validate_secret_keys(&keys).is_err());
    }

    #[test]
    fn test_validate_storage_backend_valid() {
        assert!(validate_storage_backend("r2").is_ok());
        assert!(validate_storage_backend("s3").is_ok());
        assert!(validate_storage_backend("gcs").is_ok());
        assert!(validate_storage_backend("R2").is_ok());
        assert!(validate_storage_backend("S3").is_ok());
    }

    #[test]
    fn test_validate_storage_backend_invalid() {
        assert!(validate_storage_backend("azure").is_err());
        assert!(validate_storage_backend("minio").is_err());
        assert!(validate_storage_backend("").is_err());
    }
}
