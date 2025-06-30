use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Validator-specific verification events for journal logging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerificationEvent {
    VerificationStarted {
        verification_id: Uuid,
        executor_id: String,
        validator_hotkey: String,
        verification_type: String,
        timestamp: DateTime<Utc>,
    },
    VerificationCompleted {
        verification_id: Uuid,
        executor_id: String,
        success: bool,
        score: f64,
        duration_ms: i64,
        timestamp: DateTime<Utc>,
    },
    ChallengeIssued {
        challenge_id: Uuid,
        executor_id: String,
        challenge_type: String,
        difficulty_level: i32,
        timestamp: DateTime<Utc>,
    },
    ChallengeCompleted {
        challenge_id: Uuid,
        executor_id: String,
        success: bool,
        score: f64,
        execution_time_ms: Option<i64>,
        timestamp: DateTime<Utc>,
    },
    EnvironmentValidated {
        validation_id: Uuid,
        executor_id: String,
        overall_score: f64,
        issues_count: usize,
        warnings_count: usize,
        timestamp: DateTime<Utc>,
    },
    ExecutorConnectionFailed {
        executor_id: String,
        error_message: String,
        timestamp: DateTime<Utc>,
    },
    SecurityViolation {
        executor_id: String,
        violation_type: String,
        severity: SecuritySeverity,
        details: String,
        timestamp: DateTime<Utc>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecuritySeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl VerificationEvent {
    pub fn timestamp(&self) -> DateTime<Utc> {
        match self {
            Self::VerificationStarted { timestamp, .. } => *timestamp,
            Self::VerificationCompleted { timestamp, .. } => *timestamp,
            Self::ChallengeIssued { timestamp, .. } => *timestamp,
            Self::ChallengeCompleted { timestamp, .. } => *timestamp,
            Self::EnvironmentValidated { timestamp, .. } => *timestamp,
            Self::ExecutorConnectionFailed { timestamp, .. } => *timestamp,
            Self::SecurityViolation { timestamp, .. } => *timestamp,
        }
    }

    pub fn executor_id(&self) -> &str {
        match self {
            Self::VerificationStarted { executor_id, .. } => executor_id,
            Self::VerificationCompleted { executor_id, .. } => executor_id,
            Self::ChallengeIssued { executor_id, .. } => executor_id,
            Self::ChallengeCompleted { executor_id, .. } => executor_id,
            Self::EnvironmentValidated { executor_id, .. } => executor_id,
            Self::ExecutorConnectionFailed { executor_id, .. } => executor_id,
            Self::SecurityViolation { executor_id, .. } => executor_id,
        }
    }

    pub fn event_type(&self) -> &'static str {
        match self {
            Self::VerificationStarted { .. } => "verification_started",
            Self::VerificationCompleted { .. } => "verification_completed",
            Self::ChallengeIssued { .. } => "challenge_issued",
            Self::ChallengeCompleted { .. } => "challenge_completed",
            Self::EnvironmentValidated { .. } => "environment_validated",
            Self::ExecutorConnectionFailed { .. } => "executor_connection_failed",
            Self::SecurityViolation { .. } => "security_violation",
        }
    }

    pub fn is_security_event(&self) -> bool {
        matches!(self, Self::SecurityViolation { .. })
    }

    pub fn is_error_event(&self) -> bool {
        matches!(
            self,
            Self::ExecutorConnectionFailed { .. }
                | Self::SecurityViolation { .. }
                | Self::VerificationCompleted { success: false, .. }
                | Self::ChallengeCompleted { success: false, .. }
        )
    }
}
