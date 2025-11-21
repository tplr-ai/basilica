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

/// Log storage file operations
pub fn log_storage_operation(
    namespace: &str,
    operation: &str,
    path: &str,
    bytes: u64,
    success: bool,
    error_message: Option<&str>,
    metadata: HashMap<String, String>,
) {
    if success {
        info!(
            namespace = %namespace,
            operation = %operation,
            path = %path,
            bytes = bytes,
            ?metadata,
            "Storage operation completed"
        );
    } else {
        error!(
            namespace = %namespace,
            operation = %operation,
            path = %path,
            bytes = bytes,
            error = error_message,
            ?metadata,
            "Storage operation failed"
        );
    }
}

/// Log storage synchronization to object storage backend
pub fn log_storage_sync(
    namespace: &str,
    path: &str,
    bytes: u64,
    success: bool,
    duration_ms: u64,
    error_message: Option<&str>,
    metadata: HashMap<String, String>,
) {
    if success {
        info!(
            namespace = %namespace,
            path = %path,
            bytes = bytes,
            duration_ms = duration_ms,
            ?metadata,
            "Storage sync completed"
        );
    } else {
        warn!(
            namespace = %namespace,
            path = %path,
            bytes = bytes,
            duration_ms = duration_ms,
            error = error_message,
            ?metadata,
            "Storage sync failed"
        );
    }
}

/// Log storage quota violations
pub fn log_storage_quota_exceeded(
    namespace: &str,
    quota_type: &str,
    current: u64,
    limit: u64,
    operation: &str,
    metadata: HashMap<String, String>,
) {
    warn!(
        namespace = %namespace,
        quota_type = %quota_type,
        current = current,
        limit = limit,
        operation = %operation,
        ?metadata,
        "Storage quota exceeded"
    );
}

/// Log storage path validation failures (security)
pub fn log_storage_path_validation_failure(
    namespace: &str,
    path: &str,
    reason: &str,
    severity: SecuritySeverity,
    metadata: HashMap<String, String>,
) {
    match severity {
        SecuritySeverity::Low | SecuritySeverity::Medium => warn!(
            namespace = %namespace,
            path = %path,
            reason = %reason,
            severity = %severity,
            ?metadata,
            "Storage path validation failed"
        ),
        SecuritySeverity::High | SecuritySeverity::Critical => error!(
            namespace = %namespace,
            path = %path,
            reason = %reason,
            severity = %severity,
            ?metadata,
            "Storage path validation failed - potential security violation"
        ),
    }
}

/// Log storage secret validation failures
pub fn log_storage_secret_validation_failure(
    namespace: &str,
    secret_name: &str,
    reason: &str,
    metadata: HashMap<String, String>,
) {
    error!(
        namespace = %namespace,
        secret_name = %secret_name,
        reason = %reason,
        ?metadata,
        "Storage secret validation failed"
    );
}

/// Log storage rate limit exceeded
pub fn log_storage_rate_limit_exceeded(
    namespace: &str,
    operation_type: &str,
    current_rate: u32,
    limit: u32,
    metadata: HashMap<String, String>,
) {
    warn!(
        namespace = %namespace,
        operation_type = %operation_type,
        current_rate = current_rate,
        limit = limit,
        ?metadata,
        "Storage rate limit exceeded"
    );
}
