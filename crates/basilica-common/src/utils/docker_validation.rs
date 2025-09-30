//! Docker image reference validation utilities
//!
//! Provides security validation for Docker image references to prevent
//! command injection attacks when image names are used in shell commands.

use anyhow::{anyhow, Result};
use oci_client::Reference;

/// Validate Docker image reference
///
/// Validates that a Docker image reference is safe to use in shell commands
/// by ensuring it matches the expected format and doesn't contain shell metacharacters.
///
/// # Valid formats
/// - `nginx` (simple name)
/// - `nginx:latest` (with tag)
/// - `nginx@sha256:abc123...` (with digest)
/// - `registry.com/namespace/image:tag` (full registry path)
/// - `registry.com:5000/namespace/image:tag@sha256:abc123` (with port and digest)
pub fn validate_docker_image(image: &str) -> Result<()> {
    parse_docker_image(image)?;

    Ok(())
}

/// Parse Docker image and return reference
pub fn parse_docker_image(image: &str) -> Result<Reference> {
    if image.is_empty() {
        return Err(anyhow!("Docker image reference cannot be empty"));
    }

    if image.len() > 500 {
        return Err(anyhow!(
            "Docker image reference is too long (max 500 characters)"
        ));
    }

    image
        .parse::<Reference>()
        .map_err(|e| {
            anyhow!(
                "Invalid Docker image reference '{}'. Must match format: [registry[:port]/]namespace/name[:tag][@digest]. Error: {}",
                image,
                e
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_docker_images() {
        // Test valid Docker image references
        assert!(validate_docker_image("nginx").is_ok());
        assert!(validate_docker_image("nginx:latest").is_ok());
        assert!(validate_docker_image("nginx:1.21.3").is_ok());
        assert!(validate_docker_image("library/nginx").is_ok());
        assert!(validate_docker_image("docker.io/library/nginx").is_ok());
        assert!(validate_docker_image("registry.example.com/namespace/image:tag").is_ok());
        assert!(validate_docker_image("registry.example.com:5000/namespace/image:tag").is_ok());
        assert!(validate_docker_image("nvidia/cuda:12.8.0-runtime-ubuntu22.04").is_ok());
        assert!(validate_docker_image("gcr.io/project-id/image:tag").is_ok());
        assert!(validate_docker_image(
            "image@sha256:abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
        )
        .is_ok());
    }

    #[test]
    fn test_invalid_docker_images() {
        // Test empty string
        assert!(validate_docker_image("").is_err());

        // Test shell injection attempts
        assert!(validate_docker_image("nginx; rm -rf /").is_err());
        assert!(validate_docker_image("nginx | cat /etc/passwd").is_err());
        assert!(validate_docker_image("nginx && echo hacked").is_err());
        assert!(validate_docker_image("nginx`whoami`").is_err());
        assert!(validate_docker_image("nginx$(whoami)").is_err());
        assert!(validate_docker_image("nginx > /tmp/out").is_err());
        assert!(validate_docker_image("nginx < /etc/passwd").is_err());
        assert!(validate_docker_image("nginx\\nrm -rf /").is_err());
        assert!(validate_docker_image("nginx\nrm -rf /").is_err());
        assert!(validate_docker_image("nginx\trm -rf /").is_err());
        assert!(validate_docker_image("nginx'test'").is_err());
        assert!(validate_docker_image("nginx\"test\"").is_err());

        // Test excessively long strings
        let long_string = "a".repeat(501);
        assert!(validate_docker_image(&long_string).is_err());
    }
}
