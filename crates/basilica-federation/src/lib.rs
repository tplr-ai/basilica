//! # Basilica Federation
//!
//! Multi-cluster federation system for managing multiple K3s clusters
//! with geographic distribution and high availability.
//!
//! ## Features
//!
//! - **Multi-cluster API Gateway**: Unified API for accessing resources across clusters
//! - **Cross-cluster Service Discovery**: Automatic service discovery across federated clusters
//! - **Federated Resource Management**: Manage resources across multiple clusters
//! - **Cluster Health Aggregation**: Aggregate health status from all clusters
//! - **Cross-cluster Load Balancing**: Intelligent load balancing across clusters

pub mod api;
pub mod config;
pub mod discovery;
pub mod error;
pub mod health;
pub mod load_balancer;
pub mod resource_manager;
mod utils;

pub use config::FederationConfig;
pub use error::{FederationError, Result};

/// Version of the basilica-federation crate
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

