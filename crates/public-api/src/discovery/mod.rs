//! Validator discovery module

mod health_monitor;
mod validator_discovery;

pub use health_monitor::HealthMonitor;
pub use validator_discovery::{ValidatorDiscovery, ValidatorInfo};
