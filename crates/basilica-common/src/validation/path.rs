use crate::error::ValidationError;
use std::path::Path;

const MAX_PATH_LENGTH: usize = 4096;
const MAX_COMPONENT_LENGTH: usize = 255;

/// Validates a storage path for security vulnerabilities
///
/// Checks for:
/// - Path traversal attacks (../, ..\, etc.)
/// - Absolute paths
/// - Invalid characters
/// - Length limits
/// - Null bytes
pub fn validate_storage_path(path: &str) -> Result<(), ValidationError> {
    if path.is_empty() {
        return Err(ValidationError::InvalidFormat {
            field: "path".to_string(),
            value: "(empty)".to_string(),
        });
    }

    if path.len() > MAX_PATH_LENGTH {
        return Err(ValidationError::OutOfRange {
            field: "path".to_string(),
            value: path.len().to_string(),
            min: "1".to_string(),
            max: MAX_PATH_LENGTH.to_string(),
        });
    }

    if path.contains('\0') {
        return Err(ValidationError::InvalidFormat {
            field: "path".to_string(),
            value: "contains null byte".to_string(),
        });
    }

    if path.starts_with('/') {
        return Err(ValidationError::InvalidFormat {
            field: "path".to_string(),
            value: format!("absolute path: {}", path),
        });
    }

    if path.contains("/./") || path.starts_with("./") || path.ends_with("/.") {
        return Err(ValidationError::InvalidFormat {
            field: "path".to_string(),
            value: "contains '.' (current directory)".to_string(),
        });
    }

    let normalized_path = Path::new(path);
    for component in normalized_path.components() {
        let component_str = component.as_os_str().to_string_lossy();

        if component_str == ".." {
            return Err(ValidationError::InvalidFormat {
                field: "path".to_string(),
                value: "contains '..' (path traversal)".to_string(),
            });
        }

        if component_str == "." {
            return Err(ValidationError::InvalidFormat {
                field: "path".to_string(),
                value: "contains '.' (current directory)".to_string(),
            });
        }

        if component_str.len() > MAX_COMPONENT_LENGTH {
            return Err(ValidationError::OutOfRange {
                field: "path_component".to_string(),
                value: component_str.len().to_string(),
                min: "1".to_string(),
                max: MAX_COMPONENT_LENGTH.to_string(),
            });
        }
    }

    if path.contains("..") {
        return Err(ValidationError::InvalidFormat {
            field: "path".to_string(),
            value: "contains '..' pattern (path traversal)".to_string(),
        });
    }

    if path.contains("//") {
        return Err(ValidationError::InvalidFormat {
            field: "path".to_string(),
            value: "contains consecutive slashes".to_string(),
        });
    }

    Ok(())
}

/// Validates a storage path is within a specific namespace prefix
///
/// Ensures the path starts with the expected namespace to prevent
/// cross-namespace access
pub fn validate_namespaced_path(path: &str, namespace: &str) -> Result<(), ValidationError> {
    validate_storage_path(path)?;

    if namespace.is_empty() {
        return Err(ValidationError::InvalidFormat {
            field: "namespace".to_string(),
            value: "(empty)".to_string(),
        });
    }

    let expected_prefix = format!("{}/", namespace);
    if !path.starts_with(&expected_prefix) && path != namespace {
        return Err(ValidationError::ConstraintViolation {
            field: "path".to_string(),
            constraint: format!("must start with namespace '{}', got '{}'", namespace, path),
        });
    }

    Ok(())
}

/// Validates a mount path for Kubernetes containers
///
/// Mount paths must:
/// - Start with /
/// - Not contain path traversal
/// - Not be system directories
pub fn validate_mount_path(mount_path: &str) -> Result<(), ValidationError> {
    if mount_path.is_empty() {
        return Err(ValidationError::InvalidFormat {
            field: "mount_path".to_string(),
            value: "(empty)".to_string(),
        });
    }

    if !mount_path.starts_with('/') {
        return Err(ValidationError::InvalidFormat {
            field: "mount_path".to_string(),
            value: format!("relative path: {}", mount_path),
        });
    }

    if mount_path.contains("..") {
        return Err(ValidationError::InvalidFormat {
            field: "mount_path".to_string(),
            value: "contains '..' (path traversal)".to_string(),
        });
    }

    if mount_path.contains('\0') {
        return Err(ValidationError::InvalidFormat {
            field: "mount_path".to_string(),
            value: "contains null byte".to_string(),
        });
    }

    let dangerous_paths = [
        "/", "/bin", "/boot", "/dev", "/etc", "/lib", "/lib64", "/proc", "/root", "/sbin", "/sys",
        "/usr", "/var",
    ];

    for dangerous in &dangerous_paths {
        if mount_path == *dangerous || mount_path.starts_with(&format!("{}/", dangerous)) {
            return Err(ValidationError::ConstraintViolation {
                field: "mount_path".to_string(),
                constraint: format!(
                    "system directory '{}' not allowed, use /data or /mnt",
                    dangerous
                ),
            });
        }
    }

    Ok(())
}

/// Sanitizes a path component by removing or replacing invalid characters
///
/// Uses allowlist approach: only allows alphanumeric, dash, underscore, and dot
pub fn sanitize_path_component(component: &str) -> Result<String, ValidationError> {
    if component.is_empty() {
        return Err(ValidationError::InvalidFormat {
            field: "path_component".to_string(),
            value: "(empty)".to_string(),
        });
    }

    if component == "." || component == ".." {
        return Err(ValidationError::InvalidFormat {
            field: "path_component".to_string(),
            value: component.to_string(),
        });
    }

    let sanitized: String = component
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '-'
            }
        })
        .collect();

    if sanitized.len() > MAX_COMPONENT_LENGTH {
        return Err(ValidationError::OutOfRange {
            field: "path_component".to_string(),
            value: sanitized.len().to_string(),
            min: "1".to_string(),
            max: MAX_COMPONENT_LENGTH.to_string(),
        });
    }

    Ok(sanitized)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_storage_path_valid() {
        assert!(validate_storage_path("file.txt").is_ok());
        assert!(validate_storage_path("dir/file.txt").is_ok());
        assert!(validate_storage_path("a/b/c/file.txt").is_ok());
        assert!(validate_storage_path("data-2024/model.bin").is_ok());
    }

    #[test]
    fn test_validate_storage_path_traversal() {
        assert!(validate_storage_path("../etc/passwd").is_err());
        assert!(validate_storage_path("dir/../../../etc/passwd").is_err());
        assert!(validate_storage_path("..\\windows\\system32").is_err());
        assert!(validate_storage_path("data/../../sensitive").is_err());
    }

    #[test]
    fn test_validate_storage_path_absolute() {
        assert!(validate_storage_path("/etc/passwd").is_err());
        assert!(validate_storage_path("/root/.ssh/id_rsa").is_err());
    }

    #[test]
    fn test_validate_storage_path_null_byte() {
        assert!(validate_storage_path("file\0.txt").is_err());
    }

    #[test]
    fn test_validate_storage_path_empty() {
        assert!(validate_storage_path("").is_err());
    }

    #[test]
    fn test_validate_storage_path_consecutive_slashes() {
        assert!(validate_storage_path("dir//file.txt").is_err());
    }

    #[test]
    fn test_validate_storage_path_current_dir() {
        assert!(validate_storage_path("./file.txt").is_err());
        assert!(validate_storage_path("dir/./file.txt").is_err());
    }

    #[test]
    fn test_validate_namespaced_path_valid() {
        assert!(validate_namespaced_path("u-user-123/data/file.txt", "u-user-123").is_ok());
        assert!(validate_namespaced_path("u-user-123", "u-user-123").is_ok());
    }

    #[test]
    fn test_validate_namespaced_path_wrong_namespace() {
        assert!(validate_namespaced_path("u-user-456/data/file.txt", "u-user-123").is_err());
        assert!(validate_namespaced_path("data/file.txt", "u-user-123").is_err());
    }

    #[test]
    fn test_validate_mount_path_valid() {
        assert!(validate_mount_path("/data").is_ok());
        assert!(validate_mount_path("/mnt/storage").is_ok());
        assert!(validate_mount_path("/app/data").is_ok());
    }

    #[test]
    fn test_validate_mount_path_system_dirs() {
        assert!(validate_mount_path("/").is_err());
        assert!(validate_mount_path("/etc").is_err());
        assert!(validate_mount_path("/etc/config").is_err());
        assert!(validate_mount_path("/bin").is_err());
        assert!(validate_mount_path("/usr").is_err());
        assert!(validate_mount_path("/root").is_err());
    }

    #[test]
    fn test_validate_mount_path_traversal() {
        assert!(validate_mount_path("/data/../etc").is_err());
    }

    #[test]
    fn test_validate_mount_path_relative() {
        assert!(validate_mount_path("data").is_err());
        assert!(validate_mount_path("./data").is_err());
    }

    #[test]
    fn test_sanitize_path_component_valid() {
        assert_eq!(sanitize_path_component("file.txt").unwrap(), "file.txt");
        assert_eq!(
            sanitize_path_component("model-v1.0_final").unwrap(),
            "model-v1.0_final"
        );
    }

    #[test]
    fn test_sanitize_path_component_special_chars() {
        assert_eq!(
            sanitize_path_component("file name.txt").unwrap(),
            "file-name.txt"
        );
        assert_eq!(
            sanitize_path_component("file@2024.txt").unwrap(),
            "file-2024.txt"
        );
    }

    #[test]
    fn test_sanitize_path_component_invalid() {
        assert!(sanitize_path_component("").is_err());
        assert!(sanitize_path_component(".").is_err());
        assert!(sanitize_path_component("..").is_err());
    }
}
