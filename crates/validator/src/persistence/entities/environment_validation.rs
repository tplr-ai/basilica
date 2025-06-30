use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

/// Environment validation result for tracking executor environment health
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentValidation {
    pub id: Uuid,
    pub executor_id: String,
    pub docker_score: f64,
    pub gpu_score: f64,
    pub security_score: f64,
    pub performance_score: f64,
    pub overall_score: f64,
    pub issues: Value,
    pub warnings: Value,
    pub environment_data: Value,
    pub validation_duration_ms: Option<i64>,
    pub created_at: DateTime<Utc>,
}

impl EnvironmentValidation {
    pub fn new(executor_id: String) -> Self {
        Self {
            id: Uuid::new_v4(),
            executor_id,
            docker_score: 0.0,
            gpu_score: 0.0,
            security_score: 0.0,
            performance_score: 0.0,
            overall_score: 0.0,
            issues: Value::Array(vec![]),
            warnings: Value::Array(vec![]),
            environment_data: Value::Object(serde_json::Map::new()),
            validation_duration_ms: None,
            created_at: Utc::now(),
        }
    }

    pub fn calculate_overall_score(&mut self) {
        let scores = [
            self.docker_score,
            self.gpu_score,
            self.security_score,
            self.performance_score,
        ];

        let valid_scores: Vec<f64> = scores.iter().filter(|&&s| s > 0.0).copied().collect();

        if !valid_scores.is_empty() {
            self.overall_score = valid_scores.iter().sum::<f64>() / valid_scores.len() as f64;
        }
    }

    pub fn has_issues(&self) -> bool {
        matches!(self.issues, Value::Array(ref arr) if !arr.is_empty())
    }

    pub fn has_warnings(&self) -> bool {
        matches!(self.warnings, Value::Array(ref arr) if !arr.is_empty())
    }

    pub fn is_passing(&self) -> bool {
        self.overall_score >= 0.7 && !self.has_issues()
    }

    pub fn add_issue(&mut self, issue: &str) {
        if let Value::Array(ref mut issues) = self.issues {
            issues.push(Value::String(issue.to_string()));
        }
    }

    pub fn add_warning(&mut self, warning: &str) {
        if let Value::Array(ref mut warnings) = self.warnings {
            warnings.push(Value::String(warning.to_string()));
        }
    }
}
