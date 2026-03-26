use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// Type of misbehaviour that can trigger a ban
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MisbehaviourType {
    /// Deployment failed (covers bid-won-deployment-failed and bad-rental)
    DeploymentFailed,
    /// Rental halted unexpectedly
    HaltedRental,
    /// Rental interrupted (machine yanked / interruptible)
    InterruptedRental,
    /// Provided malicious or incorrect results
    MaliciousResult,
    /// Full validation GPU facts don't match miner-declared bid metadata
    GpuDeclarationMismatch,
}

impl MisbehaviourType {
    /// Convert to database string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::DeploymentFailed => "deployment_failed",
            Self::HaltedRental => "halted_rental",
            Self::InterruptedRental => "interrupted_rental",
            Self::MaliciousResult => "malicious_result",
            Self::GpuDeclarationMismatch => "gpu_declaration_mismatch",
        }
    }
}

impl FromStr for MisbehaviourType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "deployment_failed"
            | "bad_rental"
            | "bid_won_deployment_failed"
            | "rejected_rental" => Ok(Self::DeploymentFailed),
            "halted_rental" => Ok(Self::HaltedRental),
            "interrupted_rental" => Ok(Self::InterruptedRental),
            "malicious_result" => Ok(Self::MaliciousResult),
            "gpu_declaration_mismatch" => Ok(Self::GpuDeclarationMismatch),
            _ => Err(format!("Unknown misbehaviour type: {}", s)),
        }
    }
}

impl fmt::Display for MisbehaviourType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_misbehaviour_types_roundtrip() {
        let types = vec![
            MisbehaviourType::DeploymentFailed,
            MisbehaviourType::InterruptedRental,
            MisbehaviourType::GpuDeclarationMismatch,
        ];
        for t in types {
            let as_str = t.as_str();
            let parsed = MisbehaviourType::from_str(as_str).unwrap();
            assert_eq!(t, parsed);
        }
    }
}

/// Log entry for executor misbehaviour
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MisbehaviourLog {
    /// Miner UID
    pub miner_uid: u16,
    /// Node ID that misbehaved
    pub node_id: String,
    /// When the misbehaviour was recorded
    pub recorded_at: DateTime<Utc>,
    /// Executor endpoint
    pub endpoint_executor: String,
    /// Type of misbehaviour
    pub type_of_misbehaviour: MisbehaviourType,
    /// JSON details of the misbehaviour
    pub details: String,
    /// When the record was created
    pub created_at: DateTime<Utc>,
    /// When the record was last updated
    pub updated_at: DateTime<Utc>,
}

impl MisbehaviourLog {
    /// Create a new misbehaviour log entry
    pub fn new(
        miner_uid: u16,
        node_id: String,
        endpoint_executor: String,
        type_of_misbehaviour: MisbehaviourType,
        details: String,
    ) -> Self {
        let now = Utc::now();
        Self {
            miner_uid,
            node_id,
            recorded_at: now,
            endpoint_executor,
            type_of_misbehaviour,
            details,
            created_at: now,
            updated_at: now,
        }
    }
}
