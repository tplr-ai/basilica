//! GPU utility functions.
//!
//! This module provides shared utilities for working with NVIDIA GPUs,
//! including GPU ID normalization and formatting.

/// GPU ID prefix used by NVIDIA tools.
pub const GPU_ID_PREFIX: &str = "GPU-";

/// Sanitize GPU ID by removing the 'GPU-' prefix and all hyphens.
///
/// This is useful for comparing GPU IDs that may or may not have
/// the prefix and may have varying hyphen formats.
///
/// # Examples
///
/// ```
/// use basilica_tee::gpu::utils::sanitize_gpu_id;
///
/// assert_eq!(sanitize_gpu_id("GPU-abc-def-123"), "abcdef123");
/// assert_eq!(sanitize_gpu_id("gpu-xyz"), "xyz");
/// assert_eq!(sanitize_gpu_id("plain"), "plain");
/// ```
pub fn sanitize_gpu_id(gpu_id: &str) -> String {
    gpu_id
        .replace("GPU-", "")
        .replace("gpu-", "")
        .replace('-', "")
}

/// Normalize GPU ID to include the 'GPU-' prefix if not present.
///
/// # Examples
///
/// ```
/// use basilica_tee::gpu::utils::normalize_gpu_id;
///
/// assert_eq!(normalize_gpu_id("GPU-abc123"), "GPU-abc123");
/// assert_eq!(normalize_gpu_id("abc123"), "GPU-abc123");
/// ```
pub fn normalize_gpu_id(gpu_id: &str) -> String {
    if gpu_id.starts_with(GPU_ID_PREFIX) {
        gpu_id.to_string()
    } else {
        format!("{}{}", GPU_ID_PREFIX, gpu_id)
    }
}

/// Check if two GPU IDs refer to the same GPU.
///
/// This comparison ignores the 'GPU-' prefix and hyphens.
///
/// # Examples
///
/// ```
/// use basilica_tee::gpu::utils::gpu_ids_match;
///
/// assert!(gpu_ids_match("GPU-abc-def", "abcdef"));
/// assert!(gpu_ids_match("GPU-123", "GPU-123"));
/// assert!(!gpu_ids_match("GPU-123", "GPU-456"));
/// ```
pub fn gpu_ids_match(id1: &str, id2: &str) -> bool {
    sanitize_gpu_id(id1) == sanitize_gpu_id(id2)
}

/// Check if a GPU ID is in a list of target IDs.
///
/// This comparison is lenient, allowing for different formats.
pub fn gpu_id_in_list(gpu_id: &str, target_ids: &[String]) -> bool {
    let sanitized = sanitize_gpu_id(gpu_id);
    target_ids
        .iter()
        .any(|target| sanitize_gpu_id(target) == sanitized)
}

/// Check if a GPU ID contains any of the target IDs.
///
/// This is a more lenient match that checks for substring containment.
pub fn gpu_id_contains_any(gpu_id: &str, target_ids: &[String]) -> bool {
    target_ids.iter().any(|target| {
        let normalized_target = normalize_gpu_id(target);
        gpu_id.contains(&normalized_target) || normalized_target.contains(gpu_id)
    })
}

/// Format a list of GPU IDs for display.
pub fn format_gpu_ids(gpu_ids: &[String]) -> String {
    if gpu_ids.is_empty() {
        "none".to_string()
    } else {
        gpu_ids.join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_gpu_id() {
        assert_eq!(sanitize_gpu_id("GPU-abc-def-123"), "abcdef123");
        assert_eq!(sanitize_gpu_id("gpu-xyz"), "xyz");
        assert_eq!(sanitize_gpu_id("no-prefix"), "noprefix");
        assert_eq!(sanitize_gpu_id("plain"), "plain");
        assert_eq!(sanitize_gpu_id("GPU-"), "");
        assert_eq!(sanitize_gpu_id(""), "");
    }

    #[test]
    fn test_normalize_gpu_id() {
        assert_eq!(normalize_gpu_id("GPU-abc123"), "GPU-abc123");
        assert_eq!(normalize_gpu_id("abc123"), "GPU-abc123");
        assert_eq!(normalize_gpu_id("GPU-"), "GPU-");
    }

    #[test]
    fn test_gpu_ids_match() {
        assert!(gpu_ids_match("GPU-abc-def", "abcdef"));
        assert!(gpu_ids_match("GPU-123", "GPU-123"));
        assert!(gpu_ids_match("gpu-abc", "GPU-abc"));
        assert!(!gpu_ids_match("GPU-123", "GPU-456"));
        assert!(!gpu_ids_match("abc", "def"));
    }

    #[test]
    fn test_gpu_id_in_list() {
        let targets = vec!["GPU-123".to_string(), "GPU-456".to_string()];

        assert!(gpu_id_in_list("GPU-123", &targets));
        assert!(gpu_id_in_list("123", &targets));
        assert!(gpu_id_in_list("GPU-456", &targets));
        assert!(!gpu_id_in_list("GPU-789", &targets));
    }

    #[test]
    fn test_gpu_id_contains_any() {
        let targets = vec!["GPU-abc".to_string(), "xyz".to_string()];

        assert!(gpu_id_contains_any("GPU-abc123", &targets));
        assert!(gpu_id_contains_any("GPU-xyz", &targets));
        assert!(!gpu_id_contains_any("GPU-def", &targets));
    }

    #[test]
    fn test_format_gpu_ids() {
        assert_eq!(format_gpu_ids(&[]), "none");
        assert_eq!(format_gpu_ids(&["GPU-123".to_string()]), "GPU-123");
        assert_eq!(
            format_gpu_ids(&["GPU-123".to_string(), "GPU-456".to_string()]),
            "GPU-123, GPU-456"
        );
    }
}
