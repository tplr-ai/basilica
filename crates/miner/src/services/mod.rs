//! # Miner Services
//!
//! Background services for the miner

pub mod assignment_manager;
pub mod stake_monitor;

pub use assignment_manager::{AssignmentManager, AssignmentSuggester};
pub use stake_monitor::StakeMonitor;