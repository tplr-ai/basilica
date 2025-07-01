use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

/// GPU rental record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rental {
    pub id: Uuid,
    pub executor_id: String,
    pub customer_public_key: String,
    pub docker_image: String,
    pub env_vars: Option<Value>,
    pub gpu_requirements: Value,
    pub ssh_access_info: Value,
    pub max_duration_hours: u32,
    pub cost_per_hour: f64,
    pub status: RentalStatus,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub terminated_at: Option<DateTime<Utc>>,
    pub termination_reason: Option<String>,
    pub total_cost: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RentalStatus {
    Pending,
    Active,
    Terminated,
    Failed,
}

impl Rental {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        executor_id: String,
        customer_public_key: String,
        docker_image: String,
        env_vars: Option<Value>,
        gpu_requirements: Value,
        ssh_access_info: Value,
        max_duration_hours: u32,
        cost_per_hour: f64,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            executor_id,
            customer_public_key,
            docker_image,
            env_vars,
            gpu_requirements,
            ssh_access_info,
            max_duration_hours,
            cost_per_hour,
            status: RentalStatus::Pending,
            created_at: now,
            updated_at: now,
            started_at: None,
            terminated_at: None,
            termination_reason: None,
            total_cost: None,
        }
    }

    pub fn activate(&mut self) {
        self.status = RentalStatus::Active;
        self.started_at = Some(Utc::now());
        self.updated_at = Utc::now();
    }

    pub fn terminate(&mut self, reason: Option<String>, total_cost: f64) {
        self.status = RentalStatus::Terminated;
        self.terminated_at = Some(Utc::now());
        self.termination_reason = reason;
        self.total_cost = Some(total_cost);
        self.updated_at = Utc::now();
    }

    pub fn fail(&mut self, reason: String) {
        self.status = RentalStatus::Failed;
        self.termination_reason = Some(reason);
        self.updated_at = Utc::now();
    }

    pub fn is_active(&self) -> bool {
        matches!(self.status, RentalStatus::Active)
    }

    pub fn is_terminated(&self) -> bool {
        matches!(self.status, RentalStatus::Terminated)
    }

    pub fn duration_hours(&self) -> Option<f64> {
        if let (Some(started), Some(terminated)) = (self.started_at, self.terminated_at) {
            let duration = terminated.signed_duration_since(started);
            Some(duration.num_milliseconds() as f64 / 3_600_000.0)
        } else if let Some(started) = self.started_at {
            let duration = Utc::now().signed_duration_since(started);
            Some(duration.num_milliseconds() as f64 / 3_600_000.0)
        } else {
            None
        }
    }

    pub fn current_cost(&self) -> f64 {
        if let Some(duration) = self.duration_hours() {
            duration * self.cost_per_hour
        } else {
            0.0
        }
    }
}
