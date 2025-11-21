use crate::error::ValidationError;

/// Builds a storage key with namespace, experiment ID, and path
///
/// This is the single source of truth for storage key generation,
/// following the DRY principle. Format: namespace/experiment_id/path
///
/// # Arguments
/// * `namespace` - The Kubernetes namespace (e.g., "u-user-123")
/// * `experiment_id` - The experiment or instance ID (UUID)
/// * `path` - The file path (relative, without leading slash)
///
/// # Returns
/// Storage key in format: "namespace/experiment_id/path"
pub fn build_storage_key(
    namespace: &str,
    experiment_id: &str,
    path: &str,
) -> Result<String, ValidationError> {
    if namespace.is_empty() {
        return Err(ValidationError::InvalidFormat {
            field: "namespace".to_string(),
            value: "(empty)".to_string(),
        });
    }

    if experiment_id.is_empty() {
        return Err(ValidationError::InvalidFormat {
            field: "experiment_id".to_string(),
            value: "(empty)".to_string(),
        });
    }

    if path.is_empty() {
        return Err(ValidationError::InvalidFormat {
            field: "path".to_string(),
            value: "(empty)".to_string(),
        });
    }

    let normalized_path = path.trim_start_matches('/');

    if normalized_path.is_empty() {
        return Err(ValidationError::InvalidFormat {
            field: "path".to_string(),
            value: "just '/'".to_string(),
        });
    }

    Ok(format!(
        "{}/{}/{}",
        namespace, experiment_id, normalized_path
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_storage_key_valid() {
        let key = build_storage_key(
            "u-user-123",
            "9966a80f-d1a3-4332-bd17-2eb7d75ac0b9",
            "data/file.txt",
        )
        .unwrap();
        assert_eq!(
            key,
            "u-user-123/9966a80f-d1a3-4332-bd17-2eb7d75ac0b9/data/file.txt"
        );
    }

    #[test]
    fn test_build_storage_key_with_leading_slash() {
        let key = build_storage_key(
            "u-user-123",
            "9966a80f-d1a3-4332-bd17-2eb7d75ac0b9",
            "/data/file.txt",
        )
        .unwrap();
        assert_eq!(
            key,
            "u-user-123/9966a80f-d1a3-4332-bd17-2eb7d75ac0b9/data/file.txt"
        );
    }

    #[test]
    fn test_build_storage_key_empty_namespace() {
        assert!(build_storage_key("", "exp-123", "file.txt").is_err());
    }

    #[test]
    fn test_build_storage_key_empty_experiment() {
        assert!(build_storage_key("u-user-123", "", "file.txt").is_err());
    }

    #[test]
    fn test_build_storage_key_empty_path() {
        assert!(build_storage_key("u-user-123", "exp-123", "").is_err());
    }

    #[test]
    fn test_build_storage_key_just_slash() {
        assert!(build_storage_key("u-user-123", "exp-123", "/").is_err());
    }

    #[test]
    fn test_build_storage_key_nested_path() {
        let key = build_storage_key("u-user-123", "exp-456", "a/b/c/d/file.txt").unwrap();
        assert_eq!(key, "u-user-123/exp-456/a/b/c/d/file.txt");
    }
}
