//! Structured logging functions

use crate::journal::types::SecuritySeverity;
use std::collections::HashMap;
use tracing::{error, info, warn};

/// Log validator access events
pub fn log_validator_access_granted(
    validator_id: &str,
    access_type: &str,
    duration_secs: u64,
    metadata: HashMap<String, String>,
) {
    info!(
        validator_id = %validator_id,
        access_type = %access_type,
        duration_secs = duration_secs,
        ?metadata,
        "Validator access granted"
    );
}

/// Log validator access revoked
pub fn log_validator_access_revoked(
    validator_id: &str,
    reason: &str,
    metadata: HashMap<String, String>,
) {
    warn!(
        validator_id = %validator_id,
        reason = %reason,
        ?metadata,
        "Validator access revoked"
    );
}

/// Log security violations
pub fn log_security_violation(
    validator_id: Option<&str>,
    violation_type: &str,
    description: &str,
    source_ip: Option<&str>,
    severity: SecuritySeverity,
    metadata: HashMap<String, String>,
) {
    match severity {
        SecuritySeverity::Low => info!(
            validator_id = validator_id,
            violation_type = %violation_type,
            description = %description,
            source_ip = source_ip,
            severity = %severity,
            ?metadata,
            "Security violation detected"
        ),
        SecuritySeverity::Medium => warn!(
            validator_id = validator_id,
            violation_type = %violation_type,
            description = %description,
            source_ip = source_ip,
            severity = %severity,
            ?metadata,
            "Security violation detected"
        ),
        SecuritySeverity::High | SecuritySeverity::Critical => error!(
            validator_id = validator_id,
            violation_type = %violation_type,
            description = %description,
            source_ip = source_ip,
            severity = %severity,
            ?metadata,
            "Security violation detected"
        ),
    }
}

/// Log SSH key operations
pub fn log_ssh_key_operation(
    key_id: &str,
    operation: &str,
    username: &str,
    success: bool,
    metadata: HashMap<String, String>,
) {
    if success {
        info!(
            key_id = %key_id,
            operation = %operation,
            username = %username,
            ?metadata,
            "SSH key operation completed"
        );
    } else {
        error!(
            key_id = %key_id,
            operation = %operation,
            username = %username,
            ?metadata,
            "SSH key operation failed"
        );
    }
}

/// Log system cleanup operations
pub fn log_cleanup_operation(
    cleanup_type: &str,
    items_cleaned: u32,
    metadata: HashMap<String, String>,
) {
    info!(
        cleanup_type = %cleanup_type,
        items_cleaned = items_cleaned,
        ?metadata,
        "Cleanup operation completed"
    );
}
