//! Unit tests for validation session

use chrono::Utc;
use executor::validation_session::{
    AccessManager, AccessType, ValidationSession, ValidationSessionService, ValidatorAccessInfo,
    ValidatorConfig, ValidatorHealthStatus, ValidatorId,
};
use std::collections::HashMap;
use std::time::Duration;
use tempfile::TempDir;

// Legacy service tests for backward compatibility
#[test]
fn test_validation_session_service_creation() {
    let config = ValidatorConfig::default();
    let service = ValidationSessionService::new(config);

    assert!(service.is_ok());
}

#[tokio::test]
async fn test_add_validator_key() {
    let config = ValidatorConfig::default();
    let service = ValidationSessionService::new(config).unwrap();

    let validator_hotkey = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY";
    let ssh_key = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQC...";

    let result = service.add_validator_key(validator_hotkey, ssh_key).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_remove_validator_key() {
    let config = ValidatorConfig::default();
    let service = ValidationSessionService::new(config).unwrap();

    let validator_hotkey = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY";
    let ssh_key = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQC...";

    // Add key first
    service
        .add_validator_key(validator_hotkey, ssh_key)
        .await
        .unwrap();

    // Remove key
    let result = service.remove_validator_key(validator_hotkey).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_list_active_validators() {
    let config = ValidatorConfig::default();
    let service = ValidationSessionService::new(config).unwrap();

    // Initially empty
    let validators = service.list_active_validators().await.unwrap();
    assert_eq!(validators.len(), 0);

    // Add a validator
    let validator_hotkey = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY";
    let ssh_key = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQC...";
    service
        .add_validator_key(validator_hotkey, ssh_key)
        .await
        .unwrap();

    // Should have one validator
    let validators = service.list_active_validators().await.unwrap();
    assert_eq!(validators.len(), 1);
    assert_eq!(validators[0], validator_hotkey);
}

#[tokio::test]
async fn test_cleanup_access() {
    let config = ValidatorConfig::default();
    let service = ValidationSessionService::new(config).unwrap();

    // Cleanup should work even with no sessions
    let cleaned = service.cleanup_access().await.unwrap();
    assert_eq!(cleaned, 0);
}

#[tokio::test]
async fn test_validator_config_defaults() {
    let config = ValidatorConfig::default();

    assert!(config.enabled);
    assert_eq!(config.max_concurrent_sessions, 10);
    assert_eq!(config.session_timeout, Duration::from_secs(3600));
    assert_eq!(config.ssh_port, 22);
    assert!(config.require_auth_token);
    assert!(config.allowed_ssh_keys.is_empty());
}

// New comprehensive tests for ValidationSession

fn create_test_session() -> (ValidationSession, TempDir) {
    let temp_dir = TempDir::new().unwrap();
    let config = ValidatorConfig {
        ssh_authorized_keys_path: temp_dir.path().join("authorized_keys"),
        max_concurrent_validators: 10,
        access_timeout_seconds: 3600,
        cleanup_interval_seconds: 300,
        enabled: true,
        max_concurrent_sessions: 10,
        session_timeout: Duration::from_secs(3600),
        ssh_port: 22,
        require_auth_token: true,
        allowed_ssh_keys: vec![],
    };
    let session = ValidationSession::new(config);
    (session, temp_dir)
}

#[tokio::test]
async fn test_validation_session_creation() {
    let (session, _temp_dir) = create_test_session();
    assert_eq!(session.config.max_concurrent_validators, 10);
    assert_eq!(session.config.access_timeout_seconds, 3600);
}

#[tokio::test]
async fn test_grant_ssh_access() {
    let (session, _temp_dir) = create_test_session();
    let validator = ValidatorId::new("test_validator_1".to_string());
    let ssh_key = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQC... test@example.com";

    let result = session.grant_ssh_access(&validator, ssh_key).await;
    assert!(result.is_ok());

    // Verify access was granted
    let active = session.list_active_access().await.unwrap();
    assert_eq!(active.len(), 1);
    assert_eq!(active[0].validator_id, validator);
}

#[tokio::test]
async fn test_grant_ssh_access_duplicate() {
    let (session, _temp_dir) = create_test_session();
    let validator = ValidatorId::new("test_validator_2".to_string());
    let ssh_key = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQC... test@example.com";

    // Grant access first time
    let result1 = session.grant_ssh_access(&validator, ssh_key).await;
    assert!(result1.is_ok());

    // Try to grant again
    let result2 = session.grant_ssh_access(&validator, ssh_key).await;
    assert!(result2.is_err());
    assert!(result2
        .unwrap_err()
        .to_string()
        .contains("already has active access"));
}

#[tokio::test]
async fn test_revoke_ssh_access() {
    let (session, _temp_dir) = create_test_session();
    let validator = ValidatorId::new("test_validator_3".to_string());
    let ssh_key = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQC... test@example.com";

    // Grant access first
    session.grant_ssh_access(&validator, ssh_key).await.unwrap();

    // Then revoke
    let result = session.revoke_ssh_access(&validator).await;
    assert!(result.is_ok());

    // Verify access was revoked
    let active = session.list_active_access().await.unwrap();
    assert_eq!(active.len(), 0);
}

#[tokio::test]
async fn test_revoke_nonexistent_access() {
    let (session, _temp_dir) = create_test_session();
    let validator = ValidatorId::new("test_validator_4".to_string());

    let result = session.revoke_ssh_access(&validator).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("No active access"));
}

#[tokio::test]
async fn test_check_validator_health() {
    let (session, _temp_dir) = create_test_session();
    let validator = ValidatorId::new("test_validator_5".to_string());
    let ssh_key = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQC... test@example.com";

    // Grant access first
    session.grant_ssh_access(&validator, ssh_key).await.unwrap();

    // Check health
    let result = session.check_validator_health(&validator).await;
    assert!(result.is_ok());

    let health = result.unwrap();
    assert_eq!(health.status, "healthy");
    assert!(health.access_granted);
}

#[tokio::test]
async fn test_check_health_no_access() {
    let (session, _temp_dir) = create_test_session();
    let validator = ValidatorId::new("test_validator_6".to_string());

    let result = session.check_validator_health(&validator).await;
    assert!(result.is_ok());

    let health = result.unwrap();
    assert_eq!(health.status, "no_access");
    assert!(!health.access_granted);
}

#[tokio::test]
async fn test_list_active_access() {
    let (session, _temp_dir) = create_test_session();

    // Grant access to multiple validators
    for i in 0..3 {
        let validator = ValidatorId::new(format!("test_validator_{}", i + 10));
        let ssh_key = format!("ssh-rsa AAAAB3NzaC1yc2E... test{}@example.com", i);
        session
            .grant_ssh_access(&validator, &ssh_key)
            .await
            .unwrap();
    }

    let active = session.list_active_access().await.unwrap();
    assert_eq!(active.len(), 3);

    // Verify all validators are in the list
    let validator_ids: Vec<String> = active.iter().map(|a| a.validator_id.0.clone()).collect();
    assert!(validator_ids.contains(&"test_validator_10".to_string()));
    assert!(validator_ids.contains(&"test_validator_11".to_string()));
    assert!(validator_ids.contains(&"test_validator_12".to_string()));
}

#[tokio::test]
async fn test_cleanup_expired_access() {
    let (session, _temp_dir) = create_test_session();

    // Create a custom config with short timeout
    let mut session_with_timeout = session;
    session_with_timeout.config.access_timeout_seconds = 1; // 1 second timeout

    let validator = ValidatorId::new("test_validator_20".to_string());
    let ssh_key = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQC... test@example.com";

    // Grant access
    session_with_timeout
        .grant_ssh_access(&validator, ssh_key)
        .await
        .unwrap();

    // Wait for timeout
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

    // Run cleanup
    let cleaned = session_with_timeout.cleanup_expired_access().await.unwrap();
    assert_eq!(cleaned, 1);

    // Verify access was removed
    let active = session_with_timeout.list_active_access().await.unwrap();
    assert_eq!(active.len(), 0);
}

#[tokio::test]
async fn test_concurrent_access_limit() {
    let (mut session, _temp_dir) = create_test_session();
    session.config.max_concurrent_validators = 2;

    // Grant access to max validators
    for i in 0..2 {
        let validator = ValidatorId::new(format!("test_validator_{}", i + 30));
        let ssh_key = format!("ssh-rsa AAAAB3NzaC1yc2E... test{}@example.com", i);
        session
            .grant_ssh_access(&validator, &ssh_key)
            .await
            .unwrap();
    }

    // Try to grant access to one more
    let validator = ValidatorId::new("test_validator_32".to_string());
    let ssh_key = "ssh-rsa AAAAB3NzaC1yc2E... test32@example.com";

    let result = session.grant_ssh_access(&validator, ssh_key).await;
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("Maximum concurrent validators"));
}

#[tokio::test]
async fn test_authorized_keys_file_update() {
    let (session, temp_dir) = create_test_session();
    let validator = ValidatorId::new("test_validator_40".to_string());
    let ssh_key = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQC... test@example.com";

    // Grant access
    session.grant_ssh_access(&validator, ssh_key).await.unwrap();

    // Check that authorized_keys file was created and contains the key
    let auth_keys_path = temp_dir.path().join("authorized_keys");
    assert!(auth_keys_path.exists());

    let contents = tokio::fs::read_to_string(&auth_keys_path).await.unwrap();
    assert!(contents.contains(ssh_key));
    assert!(contents.contains(&format!("# Validator: {}", validator.0)));
}

#[tokio::test]
async fn test_validator_id_validation() {
    let valid_id = ValidatorId::new("5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string());
    assert!(valid_id.is_valid());

    let empty_id = ValidatorId::new("".to_string());
    assert!(!empty_id.is_valid());
}

#[tokio::test]
async fn test_validator_access_info() {
    let info = ValidatorAccessInfo {
        validator_id: ValidatorId::new("test_validator".to_string()),
        ssh_public_key: "ssh-rsa AAAAB3NzaC1yc2E...".to_string(),
        granted_at: Utc::now(),
        expires_at: Utc::now() + chrono::Duration::hours(1),
        access_type: AccessType::Ssh,
        metadata: HashMap::new(),
    };

    assert!(!info.is_expired());
    assert!(info.is_valid());
}

#[tokio::test]
async fn test_validator_health_status() {
    let health = ValidatorHealthStatus {
        validator_id: ValidatorId::new("test_validator".to_string()),
        status: "healthy".to_string(),
        access_granted: true,
        last_activity: Some(Utc::now()),
        connection_active: true,
        error_message: None,
    };

    assert!(health.is_healthy());
}

// Access Manager specific tests
#[tokio::test]
async fn test_access_manager_add() {
    let manager = AccessManager::new();
    let validator = ValidatorId::new("test_validator".to_string());
    let access_info = ValidatorAccessInfo {
        validator_id: validator.clone(),
        ssh_public_key: "ssh-rsa...".to_string(),
        granted_at: Utc::now(),
        expires_at: Utc::now() + chrono::Duration::hours(1),
        access_type: AccessType::Ssh,
        metadata: HashMap::new(),
    };

    let result = manager.add(access_info.clone()).await;
    assert!(result.is_ok());

    let info = manager.get(&validator).await.unwrap();
    assert_eq!(info.validator_id, validator);
}

#[tokio::test]
async fn test_access_manager_remove() {
    let manager = AccessManager::new();
    let validator = ValidatorId::new("test_validator".to_string());
    let access_info = ValidatorAccessInfo {
        validator_id: validator.clone(),
        ssh_public_key: "ssh-rsa...".to_string(),
        granted_at: Utc::now(),
        expires_at: Utc::now() + chrono::Duration::hours(1),
        access_type: AccessType::Ssh,
        metadata: HashMap::new(),
    };

    manager.add(access_info).await.unwrap();
    let removed = manager.remove(&validator).await;
    assert!(removed.is_some());

    let info = manager.get(&validator).await;
    assert!(info.is_none());
}

#[tokio::test]
async fn test_access_manager_list() {
    let manager = AccessManager::new();

    for i in 0..3 {
        let validator = ValidatorId::new(format!("test_validator_{}", i));
        let access_info = ValidatorAccessInfo {
            validator_id: validator,
            ssh_public_key: format!("ssh-rsa... test{}@example.com", i),
            granted_at: Utc::now(),
            expires_at: Utc::now() + chrono::Duration::hours(1),
            access_type: AccessType::Ssh,
            metadata: HashMap::new(),
        };
        manager.add(access_info).await.unwrap();
    }

    let list = manager.list().await;
    assert_eq!(list.len(), 3);
}

#[tokio::test]
async fn test_access_manager_cleanup_expired() {
    let manager = AccessManager::new();

    // Add expired access
    let validator = ValidatorId::new("expired_validator".to_string());
    let expired_info = ValidatorAccessInfo {
        validator_id: validator,
        ssh_public_key: "ssh-rsa...".to_string(),
        granted_at: Utc::now() - chrono::Duration::hours(2),
        expires_at: Utc::now() - chrono::Duration::hours(1),
        access_type: AccessType::Ssh,
        metadata: HashMap::new(),
    };
    manager.add(expired_info).await.unwrap();

    // Add valid access
    let validator2 = ValidatorId::new("valid_validator".to_string());
    let valid_info = ValidatorAccessInfo {
        validator_id: validator2,
        ssh_public_key: "ssh-rsa...".to_string(),
        granted_at: Utc::now(),
        expires_at: Utc::now() + chrono::Duration::hours(1),
        access_type: AccessType::Ssh,
        metadata: HashMap::new(),
    };
    manager.add(valid_info).await.unwrap();

    let expired = manager.cleanup_expired().await;
    assert_eq!(expired.len(), 1);

    let remaining = manager.list().await;
    assert_eq!(remaining.len(), 1);
    assert_eq!(remaining[0].validator_id.0, "valid_validator");
}
