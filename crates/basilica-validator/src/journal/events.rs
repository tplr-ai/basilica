use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Validator-specific verification events for journal logging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerificationEvent {
    VerificationStarted {
        verification_id: Uuid,
        node_id: String,
        validator_hotkey: String,
        verification_type: String,
        timestamp: DateTime<Utc>,
    },
    VerificationCompleted {
        verification_id: Uuid,
        node_id: String,
        success: bool,
        score: f64,
        duration_ms: i64,
        timestamp: DateTime<Utc>,
    },
    ChallengeIssued {
        challenge_id: Uuid,
        node_id: String,
        challenge_type: String,
        difficulty_level: i32,
        timestamp: DateTime<Utc>,
    },
    ChallengeCompleted {
        challenge_id: Uuid,
        node_id: String,
        success: bool,
        score: f64,
        execution_time_ms: Option<i64>,
        timestamp: DateTime<Utc>,
    },
    EnvironmentValidated {
        validation_id: Uuid,
        node_id: String,
        overall_score: f64,
        issues_count: usize,
        warnings_count: usize,
        timestamp: DateTime<Utc>,
    },
    NodeConnectionFailed {
        node_id: String,
        error_message: String,
        timestamp: DateTime<Utc>,
    },
    SecurityViolation {
        node_id: String,
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
            Self::NodeConnectionFailed { timestamp, .. } => *timestamp,
            Self::SecurityViolation { timestamp, .. } => *timestamp,
        }
    }

    pub fn node_id(&self) -> &str {
        match self {
            Self::VerificationStarted { node_id, .. } => node_id,
            Self::VerificationCompleted { node_id, .. } => node_id,
            Self::ChallengeIssued { node_id, .. } => node_id,
            Self::ChallengeCompleted { node_id, .. } => node_id,
            Self::EnvironmentValidated { node_id, .. } => node_id,
            Self::NodeConnectionFailed { node_id, .. } => node_id,
            Self::SecurityViolation { node_id, .. } => node_id,
        }
    }

    pub fn event_type(&self) -> &'static str {
        match self {
            Self::VerificationStarted { .. } => "verification_started",
            Self::VerificationCompleted { .. } => "verification_completed",
            Self::ChallengeIssued { .. } => "challenge_issued",
            Self::ChallengeCompleted { .. } => "challenge_completed",
            Self::EnvironmentValidated { .. } => "environment_validated",
            Self::NodeConnectionFailed { .. } => "node_connection_failed",
            Self::SecurityViolation { .. } => "security_violation",
        }
    }

    pub fn is_security_event(&self) -> bool {
        matches!(self, Self::SecurityViolation { .. })
    }

    pub fn is_error_event(&self) -> bool {
        matches!(
            self,
            Self::NodeConnectionFailed { .. }
                | Self::SecurityViolation { .. }
                | Self::VerificationCompleted { success: false, .. }
                | Self::ChallengeCompleted { success: false, .. }
        )
    }
}
