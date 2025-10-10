use chrono::Utc;
use uuid::Uuid;

use crate::journal::events::{SecuritySeverity, VerificationEvent};
use basilica_common::journal::*;

/// Validator-specific journal logger for verification events
pub struct VerificationLogger {
    validator_hotkey: String,
}

impl VerificationLogger {
    pub fn new(validator_hotkey: String) -> Self {
        Self { validator_hotkey }
    }

    /// Log verification started event
    pub async fn log_verification_started(
        &self,
        verification_id: Uuid,
        node_id: &str,
        verification_type: &str,
    ) {
        let event = VerificationEvent::VerificationStarted {
            verification_id,
            node_id: node_id.to_string(),
            validator_hotkey: self.validator_hotkey.clone(),
            verification_type: verification_type.to_string(),
            timestamp: Utc::now(),
        };

        tracing::info!(
            event_type = "VALIDATOR_VERIFICATION_STARTED",
            verification_id = %verification_id,
            node_id = %node_id,
            validator_hotkey = %self.validator_hotkey,
            verification_type = %verification_type,
            event_data = ?serde_json::to_string(&event).unwrap_or_default(),
            "Verification started"
        );
    }

    /// Log verification completed event
    pub async fn log_verification_completed(
        &self,
        verification_id: Uuid,
        node_id: &str,
        success: bool,
        score: f64,
        duration_ms: i64,
    ) {
        let event = VerificationEvent::VerificationCompleted {
            verification_id,
            node_id: node_id.to_string(),
            success,
            score,
            duration_ms,
            timestamp: Utc::now(),
        };

        let status = if success { "SUCCESS" } else { "FAILURE" };

        if success {
            tracing::info!(
                event_type = "VALIDATOR_VERIFICATION_COMPLETED",
                verification_id = %verification_id,
                node_id = %node_id,
                validator_hotkey = %self.validator_hotkey,
                status = %status,
                score = %score,
                duration_ms = %duration_ms,
                event_data = ?serde_json::to_string(&event).unwrap_or_default(),
                "Verification completed successfully"
            );
        } else {
            tracing::warn!(
                event_type = "VALIDATOR_VERIFICATION_COMPLETED",
                verification_id = %verification_id,
                node_id = %node_id,
                validator_hotkey = %self.validator_hotkey,
                status = %status,
                score = %score,
                duration_ms = %duration_ms,
                event_data = ?serde_json::to_string(&event).unwrap_or_default(),
                "Verification failed"
            );
        }
    }

    /// Log challenge issued event
    pub async fn log_challenge_issued(
        &self,
        challenge_id: Uuid,
        node_id: &str,
        challenge_type: &str,
        difficulty_level: i32,
    ) {
        let event = VerificationEvent::ChallengeIssued {
            challenge_id,
            node_id: node_id.to_string(),
            challenge_type: challenge_type.to_string(),
            difficulty_level,
            timestamp: Utc::now(),
        };

        tracing::info!(
            event_type = "VALIDATOR_CHALLENGE_ISSUED",
            challenge_id = %challenge_id,
            node_id = %node_id,
            validator_hotkey = %self.validator_hotkey,
            challenge_type = %challenge_type,
            difficulty_level = %difficulty_level,
            event_data = ?serde_json::to_string(&event).unwrap_or_default(),
            "Challenge issued to node"
        );
    }

    /// Log challenge completed event
    pub async fn log_challenge_completed(
        &self,
        challenge_id: Uuid,
        node_id: &str,
        success: bool,
        score: f64,
        execution_time_ms: Option<i64>,
    ) {
        let event = VerificationEvent::ChallengeCompleted {
            challenge_id,
            node_id: node_id.to_string(),
            success,
            score,
            execution_time_ms,
            timestamp: Utc::now(),
        };

        let status = if success { "SUCCESS" } else { "FAILURE" };
        let execution_time_str = execution_time_ms.map_or("N/A".to_string(), |t| t.to_string());

        if success {
            tracing::info!(
                event_type = "VALIDATOR_CHALLENGE_COMPLETED",
                challenge_id = %challenge_id,
                node_id = %node_id,
                validator_hotkey = %self.validator_hotkey,
                status = %status,
                score = %score,
                execution_time_ms = %execution_time_str,
                event_data = ?serde_json::to_string(&event).unwrap_or_default(),
                "Challenge completed successfully"
            );
        } else {
            tracing::warn!(
                event_type = "VALIDATOR_CHALLENGE_COMPLETED",
                challenge_id = %challenge_id,
                node_id = %node_id,
                validator_hotkey = %self.validator_hotkey,
                status = %status,
                score = %score,
                execution_time_ms = %execution_time_str,
                event_data = ?serde_json::to_string(&event).unwrap_or_default(),
                "Challenge failed"
            );
        }
    }

    /// Log environment validation event
    pub async fn log_environment_validated(
        &self,
        validation_id: Uuid,
        node_id: &str,
        overall_score: f64,
        issues_count: usize,
        warnings_count: usize,
    ) {
        let event = VerificationEvent::EnvironmentValidated {
            validation_id,
            node_id: node_id.to_string(),
            overall_score,
            issues_count,
            warnings_count,
            timestamp: Utc::now(),
        };

        // Environment validation logging

        if overall_score >= 0.7 && issues_count == 0 {
            tracing::info!(
                event_type = "VALIDATOR_ENVIRONMENT_VALIDATED",
                validation_id = %validation_id,
                node_id = %node_id,
                validator_hotkey = %self.validator_hotkey,
                overall_score = %overall_score,
                issues_count = %issues_count,
                warnings_count = %warnings_count,
                event_data = ?serde_json::to_string(&event).unwrap_or_default(),
                "Environment validation passed"
            );
        } else {
            tracing::warn!(
                event_type = "VALIDATOR_ENVIRONMENT_VALIDATED",
                validation_id = %validation_id,
                node_id = %node_id,
                validator_hotkey = %self.validator_hotkey,
                overall_score = %overall_score,
                issues_count = %issues_count,
                warnings_count = %warnings_count,
                event_data = ?serde_json::to_string(&event).unwrap_or_default(),
                "Environment validation failed or has issues"
            );
        }
    }

    /// Log node connection failure
    pub async fn log_node_connection_failed(&self, node_id: &str, error_message: &str) {
        let event = VerificationEvent::NodeConnectionFailed {
            node_id: node_id.to_string(),
            error_message: error_message.to_string(),
            timestamp: Utc::now(),
        };

        tracing::error!(
            event_type = "VALIDATOR_NODE_CONNECTION_FAILED",
            node_id = %node_id,
            validator_hotkey = %self.validator_hotkey,
            error = %error_message,
            event_data = ?serde_json::to_string(&event).unwrap_or_default(),
            "Failed to connect to node"
        );
    }

    /// Log security violation
    pub async fn log_security_violation(
        &self,
        node_id: &str,
        violation_type: &str,
        severity: SecuritySeverity,
        details: &str,
    ) {
        let event = VerificationEvent::SecurityViolation {
            node_id: node_id.to_string(),
            violation_type: violation_type.to_string(),
            severity: severity.clone(),
            details: details.to_string(),
            timestamp: Utc::now(),
        };

        let severity_str = match severity {
            SecuritySeverity::Low => "LOW",
            SecuritySeverity::Medium => "MEDIUM",
            SecuritySeverity::High => "HIGH",
            SecuritySeverity::Critical => "CRITICAL",
        };

        // Convert internal SecuritySeverity to basilica_common::journal::SecuritySeverity
        let common_severity = match severity {
            SecuritySeverity::Low => basilica_common::journal::SecuritySeverity::Low,
            SecuritySeverity::Medium => basilica_common::journal::SecuritySeverity::Medium,
            SecuritySeverity::High => basilica_common::journal::SecuritySeverity::High,
            SecuritySeverity::Critical => basilica_common::journal::SecuritySeverity::Critical,
        };

        log_security_violation(
            Some(node_id),
            violation_type,
            details,
            None, // source_ip
            common_severity,
            std::collections::HashMap::new(),
        );

        match severity {
            SecuritySeverity::Critical | SecuritySeverity::High => {
                tracing::error!(
                    event_type = "VALIDATOR_SECURITY_VIOLATION",
                    node_id = %node_id,
                    violation_type = %violation_type,
                    severity = %severity_str,
                    details = %details,
                    event_data = ?serde_json::to_string(&event).unwrap_or_default(),
                    "Security violation detected"
                );
            }
            SecuritySeverity::Medium => {
                tracing::warn!(
                    event_type = "VALIDATOR_SECURITY_VIOLATION",
                    node_id = %node_id,
                    violation_type = %violation_type,
                    severity = %severity_str,
                    details = %details,
                    event_data = ?serde_json::to_string(&event).unwrap_or_default(),
                    "Security violation detected"
                );
            }
            SecuritySeverity::Low => {
                tracing::info!(
                    event_type = "VALIDATOR_SECURITY_VIOLATION",
                    node_id = %node_id,
                    violation_type = %violation_type,
                    severity = %severity_str,
                    details = %details,
                    event_data = ?serde_json::to_string(&event).unwrap_or_default(),
                    "Security violation detected"
                );
            }
        }
    }

    /// Query verification events from journal
    pub async fn query_verification_events(
        &self,
        node_id: Option<&str>,
        event_type: Option<&str>,
        since: Option<chrono::DateTime<Utc>>,
        limit: Option<usize>,
    ) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let mut filters = vec![
            format!("_SYSTEMD_UNIT=basilica-validator.service"),
            format!("VALIDATOR_HOTKEY={}", self.validator_hotkey),
        ];

        if let Some(node_id) = node_id {
            filters.push(format!("NODE_ID={node_id}"));
        }

        if let Some(event_type) = event_type {
            filters.push(format!("MESSAGE={}", event_type.to_uppercase()));
        }

        // Use the common journal query functionality
        let since_str = since.map(|dt| dt.to_rfc3339());
        query_logs(Some(&self.validator_hotkey), since_str.as_deref(), limit)
    }

    /// Get verification statistics from journal
    pub async fn get_verification_stats(
        &self,
        node_id: Option<&str>,
        days: u32,
    ) -> Result<VerificationStats, Box<dyn std::error::Error>> {
        let since = Utc::now() - chrono::Duration::days(days as i64);
        let entries = self
            .query_verification_events(node_id, None, Some(since), None)
            .await?;

        let mut stats = VerificationStats::default();

        for entry in entries {
            if entry.contains("VALIDATOR_VERIFICATION_COMPLETED") {
                stats.total_verifications += 1;
                if entry.contains("STATUS=SUCCESS") {
                    stats.successful_verifications += 1;
                }
            } else if entry.contains("VALIDATOR_CHALLENGE_COMPLETED") {
                stats.total_challenges += 1;
                if entry.contains("STATUS=SUCCESS") {
                    stats.successful_challenges += 1;
                }
            } else if entry.contains("VALIDATOR_NODE_CONNECTION_FAILED") {
                stats.connection_failures += 1;
            } else if entry.contains("SECURITY_VIOLATION") {
                stats.security_violations += 1;
            }
        }

        Ok(stats)
    }
}

#[derive(Debug, Default)]
pub struct VerificationStats {
    pub total_verifications: u64,
    pub successful_verifications: u64,
    pub total_challenges: u64,
    pub successful_challenges: u64,
    pub connection_failures: u64,
    pub security_violations: u64,
}

impl VerificationStats {
    pub fn verification_success_rate(&self) -> f64 {
        if self.total_verifications == 0 {
            0.0
        } else {
            self.successful_verifications as f64 / self.total_verifications as f64
        }
    }

    pub fn challenge_success_rate(&self) -> f64 {
        if self.total_challenges == 0 {
            0.0
        } else {
            self.successful_challenges as f64 / self.total_challenges as f64
        }
    }
}
