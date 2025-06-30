//! Unit tests for validation session

use executor::validation_session::{
    AccessControlConfig, HotkeyVerificationConfig, RateLimitConfig, ValidationSessionService,
    ValidatorAccess, ValidatorConfig, ValidatorId, ValidatorRole,
};
use std::collections::HashMap;
use std::time::SystemTime;

// Helper function to create test service
fn create_test_service() -> ValidationSessionService {
    let config = ValidatorConfig::default();
    ValidationSessionService::new(config).unwrap()
}

#[tokio::test]
async fn test_validation_session_service_creation() {
    let service = create_test_service();

    // Service should be created successfully
    let access_list = service.list_active_access().await.unwrap();
    assert_eq!(access_list.len(), 0);
}

#[tokio::test]
#[ignore = "Requires root access to create system users"]
async fn test_grant_access() {
    let service = create_test_service();

    let validator_id = ValidatorId::new("test_validator".to_string());
    let ssh_public_key = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQC... test@example.com";

    let result = service
        .grant_ssh_access(&validator_id, ssh_public_key)
        .await;
    assert!(result.is_ok());

    // Check access was granted
    let access_list = service.list_active_access().await.unwrap();
    assert_eq!(access_list.len(), 1);
    assert_eq!(access_list[0].validator_id.hotkey, "test_validator");
}

#[tokio::test]
#[ignore = "Requires root access to create system users"]
async fn test_revoke_access() {
    let service = create_test_service();

    let validator_id = ValidatorId::new("test_validator".to_string());
    let ssh_public_key = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQC... test@example.com";

    // Grant access first
    service
        .grant_ssh_access(&validator_id, ssh_public_key)
        .await
        .unwrap();

    // Revoke access
    let result = service.revoke_ssh_access(&validator_id).await;
    assert!(result.is_ok());

    // Check access was revoked
    let access_list = service.list_active_access().await.unwrap();
    assert_eq!(access_list.len(), 0);
}

#[tokio::test]
#[ignore = "Requires root access to create system users"]
async fn test_check_access() {
    let service = create_test_service();

    let validator_id = ValidatorId::new("test_validator".to_string());
    let ssh_public_key = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQC... test@example.com";

    // Initially no access
    assert!(!service.has_ssh_access(&validator_id).await);

    // Grant access
    service
        .grant_ssh_access(&validator_id, ssh_public_key)
        .await
        .unwrap();

    // Now should have access
    assert!(service.has_ssh_access(&validator_id).await);
}

#[tokio::test]
async fn test_cleanup_expired_access() {
    let service = create_test_service();

    // This test would require mocking time or setting up expired access
    // For now, just test that cleanup runs without error
    let result = service.cleanup_access().await;
    assert!(result.is_ok());
}

#[test]
fn test_validator_id() {
    let validator_id = ValidatorId::new("test_hotkey".to_string());
    assert_eq!(validator_id.hotkey, "test_hotkey");
    assert!(validator_id.label.is_none());

    let validator_id_with_label =
        ValidatorId::with_label("test_hotkey".to_string(), "Test Validator".to_string());
    assert_eq!(validator_id_with_label.hotkey, "test_hotkey");
    assert_eq!(
        validator_id_with_label.label,
        Some("Test Validator".to_string())
    );
}

#[test]
fn test_validator_id_display() {
    let validator_id = ValidatorId::new("test_hotkey".to_string());
    assert_eq!(format!("{validator_id}"), "test_hotkey");

    let validator_id_with_label =
        ValidatorId::with_label("test_hotkey".to_string(), "Test Validator".to_string());
    assert_eq!(
        format!("{validator_id_with_label}"),
        "test_hotkey(Test Validator)"
    );
}

#[test]
fn test_validator_access() {
    let validator_id = ValidatorId::new("test_validator".to_string());
    let access = ValidatorAccess::new(validator_id.clone(), "ssh-rsa AAAAB3...");

    assert_eq!(access.validator_id.hotkey, "test_validator");
    assert!(access.ssh_key.public_key.starts_with("ssh-rsa"));
    assert!(access.granted_at <= SystemTime::now());
    assert!(access.expires_at > SystemTime::now());
}

#[test]
fn test_validator_access_expiry() {
    let validator_id = ValidatorId::new("test_validator".to_string());
    let access = ValidatorAccess::new(validator_id, "ssh-rsa AAAAB3...");

    // Should not be expired immediately after creation
    assert!(!access.is_expired(SystemTime::now()));

    // Should have time until expiry
    let time_until_expiry = access.time_until_expiry(SystemTime::now());
    assert!(time_until_expiry.is_some());
    assert!(time_until_expiry.unwrap().as_secs() > 0);
}

#[test]
fn test_access_control_config() {
    let mut config = AccessControlConfig::default();

    // Initially empty
    assert_eq!(config.ip_whitelist.len(), 0);
    assert_eq!(config.required_permissions.len(), 0);

    // Add IP whitelist
    config.ip_whitelist.push("192.168.1.0/24".to_string());
    config.ip_whitelist.push("10.0.0.0/8".to_string());
    assert_eq!(config.ip_whitelist.len(), 2);

    // Add required permissions
    config.required_permissions.insert(
        "execute".to_string(),
        vec!["admin".to_string(), "validator".to_string()],
    );
    assert_eq!(config.required_permissions.len(), 1);
}

#[test]
fn test_validator_config() {
    let config = ValidatorConfig::default();

    assert!(config.enabled);
    assert!(!config.strict_ssh_restrictions);
    assert_eq!(config.access_config.ip_whitelist.len(), 0);
    assert_eq!(config.access_config.required_permissions.len(), 0);
}

#[test]
fn test_validator_config_custom() {
    let config = ValidatorConfig {
        enabled: true,
        strict_ssh_restrictions: true,
        access_config: AccessControlConfig {
            ip_whitelist: vec!["10.0.0.0/8".to_string()],
            required_permissions: {
                let mut perms = HashMap::new();
                perms.insert("ssh".to_string(), vec!["validator".to_string()]);
                perms
            },
            hotkey_verification: HotkeyVerificationConfig {
                enabled: true,
                challenge_timeout_seconds: 60,
                max_signature_attempts: 3,
                cleanup_interval_seconds: 300,
            },
            rate_limits: RateLimitConfig {
                ssh_requests_per_minute: 15,
                api_requests_per_minute: 75,
                burst_allowance: 8,
                rate_limit_window_seconds: 60,
            },
            role_assignments: {
                let mut roles = HashMap::new();
                roles.insert("test_validator".to_string(), ValidatorRole::Standard);
                roles
            },
        },
    };

    assert!(config.enabled);
    assert!(config.strict_ssh_restrictions);
    assert_eq!(config.access_config.ip_whitelist.len(), 1);
    assert_eq!(config.access_config.required_permissions.len(), 1);
}

#[tokio::test]
#[ignore = "Requires root access to create system users"]
async fn test_concurrent_access_operations() {
    let service = create_test_service();

    // Spawn multiple concurrent grant operations
    let mut handles = vec![];

    for i in 0..5 {
        let service_clone = service.clone();
        let handle = tokio::spawn(async move {
            let validator_id = ValidatorId::new(format!("validator_{i}"));
            let ssh_key = format!("ssh-rsa AAAAB3... validator_{i}@example.com");
            service_clone
                .grant_ssh_access(&validator_id, &ssh_key)
                .await
        });
        handles.push(handle);
    }

    // All grants should succeed
    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result.is_ok());
    }

    // Should have 5 active access entries
    let access_list = service.list_active_access().await.unwrap();
    assert_eq!(access_list.len(), 5);
}

#[tokio::test]
#[ignore = "Requires root access to create system users"]
async fn test_duplicate_grant() {
    let service = create_test_service();

    let validator_id = ValidatorId::new("test_validator".to_string());
    let ssh_key = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQC...";

    // First grant should succeed
    let result1 = service.grant_ssh_access(&validator_id, ssh_key).await;
    assert!(result1.is_ok());

    // Second grant with same validator should update the entry
    let result2 = service.grant_ssh_access(&validator_id, ssh_key).await;
    assert!(result2.is_ok());

    // Should still have only 1 entry
    let access_list = service.list_active_access().await.unwrap();
    assert_eq!(access_list.len(), 1);
}

#[test]
fn test_ssh_access() {
    let validator_id = ValidatorId::new("test_validator".to_string());
    let access = ValidatorAccess::new(validator_id, "ssh-rsa AAAAB3...");

    // Should have SSH access by default
    assert!(access.has_ssh_access());
}
