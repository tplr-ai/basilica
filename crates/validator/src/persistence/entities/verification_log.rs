use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

/// Verification log entry for tracking validator operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationLog {
    pub id: Uuid,
    pub executor_id: String,
    pub validator_hotkey: String,
    pub verification_type: String,
    pub timestamp: DateTime<Utc>,
    pub score: f64,
    pub success: bool,
    pub details: Value,
    pub duration_ms: i64,
    pub error_message: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl VerificationLog {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        executor_id: String,
        validator_hotkey: String,
        verification_type: String,
        score: f64,
        success: bool,
        details: Value,
        duration_ms: i64,
        error_message: Option<String>,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            executor_id,
            validator_hotkey,
            verification_type,
            timestamp: now,
            score,
            success,
            details,
            duration_ms,
            error_message,
            created_at: now,
            updated_at: now,
        }
    }

    pub fn is_successful(&self) -> bool {
        self.success
    }

    pub fn has_error(&self) -> bool {
        self.error_message.is_some()
    }
}

/// Statistics for executor verification history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutorVerificationStats {
    pub executor_id: String,
    pub total_verifications: u64,
    pub successful_verifications: u64,
    pub average_score: Option<f64>,
    pub average_duration_ms: Option<f64>,
    pub first_verification: Option<DateTime<Utc>>,
    pub last_verification: Option<DateTime<Utc>>,
}

impl ExecutorVerificationStats {
    pub fn success_rate(&self) -> f64 {
        if self.total_verifications == 0 {
            0.0
        } else {
            self.successful_verifications as f64 / self.total_verifications as f64
        }
    }
}
