use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

/// Challenge result entry for detailed tracking of verification challenges
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChallengeResult {
    pub id: Uuid,
    pub executor_id: String,
    pub challenge_type: String,
    pub challenge_parameters: Value,
    pub solution_data: Option<Value>,
    pub success: bool,
    pub score: f64,
    pub execution_time_ms: Option<i64>,
    pub verification_time_ms: i64,
    pub issued_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub difficulty_level: i32,
    pub expected_ops: Option<i64>,
    pub timeout_seconds: Option<i32>,
    pub error_message: Option<String>,
    pub error_code: Option<String>,
}

impl ChallengeResult {
    pub fn new(
        executor_id: String,
        challenge_type: String,
        challenge_parameters: Value,
        difficulty_level: i32,
        expected_ops: Option<i64>,
        timeout_seconds: Option<i32>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            executor_id,
            challenge_type,
            challenge_parameters,
            solution_data: None,
            success: false,
            score: 0.0,
            execution_time_ms: None,
            verification_time_ms: 0,
            issued_at: Utc::now(),
            completed_at: None,
            difficulty_level,
            expected_ops,
            timeout_seconds,
            error_message: None,
            error_code: None,
        }
    }

    pub fn complete_with_success(
        &mut self,
        solution_data: Value,
        score: f64,
        execution_time_ms: i64,
        verification_time_ms: i64,
    ) {
        self.solution_data = Some(solution_data);
        self.success = true;
        self.score = score;
        self.execution_time_ms = Some(execution_time_ms);
        self.verification_time_ms = verification_time_ms;
        self.completed_at = Some(Utc::now());
    }

    pub fn complete_with_failure(
        &mut self,
        error_message: String,
        error_code: Option<String>,
        verification_time_ms: i64,
    ) {
        self.success = false;
        self.score = 0.0;
        self.error_message = Some(error_message);
        self.error_code = error_code;
        self.verification_time_ms = verification_time_ms;
        self.completed_at = Some(Utc::now());
    }

    pub fn is_timed_out(&self) -> bool {
        if let Some(timeout_seconds) = self.timeout_seconds {
            let elapsed_ms = self.issued_at.timestamp_millis() - Utc::now().timestamp_millis();
            elapsed_ms > (timeout_seconds as i64 * 1000)
        } else {
            false
        }
    }
}
