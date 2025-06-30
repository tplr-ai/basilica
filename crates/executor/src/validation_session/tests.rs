//! Comprehensive tests for enhanced validator access control
//!
//! Tests cover hotkey verification, role-based access control, rate limiting,
//! and audit logging functionality.

use crate::validation_session::{
    hotkey_verifier::HotkeyVerificationConfig, AccessControlConfig, RateLimitConfig, RequestType,
    ValidatorAccessControl, ValidatorId, ValidatorRole,
};
use common::crypto::P256KeyPair;
use std::collections::HashMap;
use tokio::time::{sleep, Duration};

fn create_test_config() -> AccessControlConfig {
    let mut role_assignments = HashMap::new();
    role_assignments.insert("admin_validator".to_string(), ValidatorRole::Admin);
    role_assignments.insert("owner_validator".to_string(), ValidatorRole::Owner);

    AccessControlConfig {
        ip_whitelist: vec!["127.0.0.1".to_string(), "192.168.1.0/24".to_string()],
        required_permissions: HashMap::new(),
        hotkey_verification: HotkeyVerificationConfig {
            enabled: true,
            challenge_timeout_seconds: 60,
            max_signature_attempts: 3,
            cleanup_interval_seconds: 300,
        },
        rate_limits: RateLimitConfig {
            ssh_requests_per_minute: 10,
            api_requests_per_minute: 50,
            burst_allowance: 5,
            rate_limit_window_seconds: 60,
        },
        role_assignments,
    }
}

#[tokio::test]
async fn test_hotkey_challenge_generation() {
    let config = create_test_config();
    let access_control = ValidatorAccessControl::new(config);
    let validator_id = ValidatorId::new("test_validator".to_string());

    // Generate challenge
    let challenge = access_control
        .generate_hotkey_challenge(&validator_id)
        .await
        .expect("Failed to generate challenge");

    assert_eq!(challenge.validator_hotkey, "test_validator");
    assert_eq!(challenge.challenge_data.len(), 32);
    assert!(!challenge.is_expired());
}

#[tokio::test]
async fn test_hotkey_signature_verification() {
    let config = create_test_config();
    let access_control = ValidatorAccessControl::new(config);
    let validator_id = ValidatorId::new("test_validator".to_string());

    // Generate test key pair
    let key_pair = P256KeyPair::generate();
    let public_key = key_pair.public_key();

    // Generate challenge
    let challenge = access_control
        .generate_hotkey_challenge(&validator_id)
        .await
        .expect("Failed to generate challenge");

    // Sign the challenge
    let signature = key_pair.private_key().sign(&challenge.challenge_data);

    // Verify the signature
    let result = access_control
        .verify_hotkey_signature(
            &challenge.challenge_id,
            &signature.to_bytes(),
            &public_key.to_compressed_bytes(),
        )
        .await
        .expect("Failed to verify signature");

    assert!(result, "Signature verification should succeed");
}

#[tokio::test]
async fn test_role_based_authorization() {
    let config = create_test_config();
    let access_control = ValidatorAccessControl::new(config);

    // Test basic validator
    let basic_validator = ValidatorId::new("basic_validator".to_string());
    access_control
        .grant_access_with_verification(
            &basic_validator,
            "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPlaceholder test-key",
            ValidatorRole::Basic,
            false,
        )
        .await
        .expect("Failed to grant basic access");

    // Basic validator should have ssh_access and system_info
    assert!(access_control
        .authorize_operation(&basic_validator, "ssh_access")
        .await
        .unwrap());
    assert!(access_control
        .authorize_operation(&basic_validator, "system_info")
        .await
        .unwrap());
    assert!(!access_control
        .authorize_operation(&basic_validator, "container_list")
        .await
        .unwrap());

    // Test admin validator
    let admin_validator = ValidatorId::new("admin_validator".to_string());
    access_control
        .grant_access_with_verification(
            &admin_validator,
            "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPlaceholder admin-key",
            ValidatorRole::Admin,
            true, // hotkey verified
        )
        .await
        .expect("Failed to grant admin access");

    // Admin validator should have container_exec permission
    assert!(access_control
        .authorize_operation(&admin_validator, "ssh_access")
        .await
        .unwrap());
    assert!(access_control
        .authorize_operation(&admin_validator, "container_list")
        .await
        .unwrap());
    assert!(access_control
        .authorize_operation(&admin_validator, "container_exec")
        .await
        .unwrap());
    assert!(!access_control
        .authorize_operation(&admin_validator, "config_change")
        .await
        .unwrap());

    // Test owner validator
    let owner_validator = ValidatorId::new("owner_validator".to_string());
    access_control
        .grant_access_with_verification(
            &owner_validator,
            "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPlaceholder owner-key",
            ValidatorRole::Owner,
            true,
        )
        .await
        .expect("Failed to grant owner access");

    // Owner should have all permissions
    assert!(access_control
        .authorize_operation(&owner_validator, "config_change")
        .await
        .unwrap());
    assert!(access_control
        .authorize_operation(&owner_validator, "service_control")
        .await
        .unwrap());
}

#[tokio::test]
async fn test_hotkey_verification_requirement() {
    let config = create_test_config();
    let access_control = ValidatorAccessControl::new(config);

    // Grant access without hotkey verification
    let validator_id = ValidatorId::new("unverified_admin".to_string());
    access_control
        .grant_access_with_verification(
            &validator_id,
            "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPlaceholder test-key",
            ValidatorRole::Admin,
            false, // NOT hotkey verified
        )
        .await
        .expect("Failed to grant access");

    // Should be denied for sensitive operations
    assert!(!access_control
        .authorize_operation(&validator_id, "container_exec")
        .await
        .unwrap());
    assert!(!access_control
        .authorize_operation(&validator_id, "config_change")
        .await
        .unwrap());

    // But allowed for non-sensitive operations
    assert!(access_control
        .authorize_operation(&validator_id, "ssh_access")
        .await
        .unwrap());
    assert!(access_control
        .authorize_operation(&validator_id, "system_info")
        .await
        .unwrap());
}

#[tokio::test]
async fn test_rate_limiting() {
    let mut config = create_test_config();
    config.rate_limits.ssh_requests_per_minute = 1; // Very low limit - only 1 per minute
    config.rate_limits.burst_allowance = 1; // Allow 1 burst
    config.ip_whitelist.clear(); // Remove IP whitelist for this test

    let access_control = ValidatorAccessControl::new(config);
    let validator_id = ValidatorId::new("rate_limited_validator".to_string());

    // Grant access first
    access_control
        .grant_access_with_verification(
            &validator_id,
            "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPlaceholder test-key",
            ValidatorRole::Basic,
            false,
        )
        .await
        .expect("Failed to grant access");

    // Should allow first request
    let result1 = access_control
        .authenticate_request(&validator_id, None, RequestType::Ssh)
        .await
        .unwrap();
    assert!(result1, "First request should be allowed");

    // Should allow second request
    let result2 = access_control
        .authenticate_request(&validator_id, None, RequestType::Ssh)
        .await
        .unwrap();
    assert!(result2, "Second request should be allowed");

    // Should deny third request (rate limit exceeded)
    let result3 = access_control
        .authenticate_request(&validator_id, None, RequestType::Ssh)
        .await
        .unwrap();
    assert!(!result3, "Third request should be denied due to rate limit");

    // Reset rate limits
    access_control.reset_rate_limits(&validator_id).await;

    // Should allow requests again
    assert!(access_control
        .authenticate_request(&validator_id, None, RequestType::Ssh)
        .await
        .unwrap());
}

#[tokio::test]
async fn test_ip_whitelist() {
    let config = create_test_config();
    let access_control = ValidatorAccessControl::new(config);
    let validator_id = ValidatorId::new("test_validator".to_string());

    // Grant access first
    access_control
        .grant_access_with_verification(
            &validator_id,
            "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPlaceholder test-key",
            ValidatorRole::Basic,
            false,
        )
        .await
        .expect("Failed to grant access");

    // Allowed IP
    let allowed_ip = Some("127.0.0.1".parse().unwrap());
    assert!(access_control
        .authenticate_request(&validator_id, allowed_ip, RequestType::Api)
        .await
        .unwrap());

    // Blocked IP (not in whitelist)
    let blocked_ip = Some("10.0.0.1".parse().unwrap());
    assert!(!access_control
        .authenticate_request(&validator_id, blocked_ip, RequestType::Api)
        .await
        .unwrap());
}

#[tokio::test]
async fn test_cleanup_expired_data() {
    let mut config = create_test_config();
    config.hotkey_verification.challenge_timeout_seconds = 1; // Very short timeout

    let access_control = ValidatorAccessControl::new(config);
    let validator_id = ValidatorId::new("test_validator".to_string());

    // Generate challenge
    let _challenge = access_control
        .generate_hotkey_challenge(&validator_id)
        .await
        .expect("Failed to generate challenge");

    // Wait for expiration
    sleep(Duration::from_secs(2)).await;

    // Clean up expired data
    let (_cleaned_sessions, cleaned_challenges) = access_control
        .cleanup_expired_data()
        .await
        .expect("Failed to cleanup expired data");

    assert!(
        cleaned_challenges > 0,
        "Should have cleaned up expired challenges"
    );
}

#[tokio::test]
async fn test_role_permission_mapping() {
    // Test ValidatorRole permission methods
    assert!(ValidatorRole::Basic.has_permission("ssh_access"));
    assert!(ValidatorRole::Basic.has_permission("system_info"));
    assert!(!ValidatorRole::Basic.has_permission("container_list"));

    assert!(ValidatorRole::Standard.has_permission("container_list"));
    assert!(!ValidatorRole::Standard.has_permission("container_exec"));

    assert!(ValidatorRole::Admin.has_permission("container_exec"));
    assert!(!ValidatorRole::Admin.has_permission("config_change"));

    assert!(ValidatorRole::Owner.has_permission("config_change"));
    assert!(ValidatorRole::Owner.has_permission("service_control"));
    assert!(ValidatorRole::Owner.has_permission("any_permission")); // Owner has all
}

#[tokio::test]
async fn test_session_statistics() {
    let config = create_test_config();
    let access_control = ValidatorAccessControl::new(config);

    // Grant access to multiple validators
    for i in 0..3 {
        let validator_id = ValidatorId::new(format!("test_validator_{i}"));
        access_control
            .grant_access_with_verification(
                &validator_id,
                &format!("ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPlaceholder key-{i}"),
                ValidatorRole::Basic,
                false,
            )
            .await
            .expect("Failed to grant access");
    }

    let stats = access_control.get_session_stats().await;
    assert_eq!(stats.total_sessions, 3);
    assert_eq!(stats.active_sessions, 3);
    assert_eq!(stats.ssh_sessions, 3);
}

#[tokio::test]
async fn test_signature_verification_failure() {
    let config = create_test_config();
    let access_control = ValidatorAccessControl::new(config);
    let validator_id = ValidatorId::new("test_validator".to_string());

    // Generate challenge
    let challenge = access_control
        .generate_hotkey_challenge(&validator_id)
        .await
        .expect("Failed to generate challenge");

    // Try to verify with invalid signature
    let invalid_signature = vec![0u8; 64];
    let invalid_public_key = vec![0u8; 33];

    let result = access_control
        .verify_hotkey_signature(
            &challenge.challenge_id,
            &invalid_signature,
            &invalid_public_key,
        )
        .await
        .expect("Failed to attempt verification");

    assert!(!result, "Invalid signature should fail verification");
}

#[tokio::test]
async fn test_challenge_expiration() {
    let mut config = create_test_config();
    config.hotkey_verification.challenge_timeout_seconds = 1; // 1 second timeout

    let access_control = ValidatorAccessControl::new(config);
    let validator_id = ValidatorId::new("test_validator".to_string());

    // Generate challenge
    let challenge = access_control
        .generate_hotkey_challenge(&validator_id)
        .await
        .expect("Failed to generate challenge");

    // Wait for expiration
    sleep(Duration::from_secs(2)).await;

    // Try to verify expired challenge
    let result = access_control
        .verify_hotkey_signature(&challenge.challenge_id, &[0u8; 64], &[0u8; 33])
        .await
        .expect("Failed to attempt verification");

    assert!(!result, "Expired challenge should fail verification");
}
