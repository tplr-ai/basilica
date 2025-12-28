//! # basilica-autoscaler
//!
//! Basilica GPU node autoscaler: dynamic scaling of K3s GPU nodes.
//!
//! This crate provides automatic scaling of GPU nodes in K3s clusters based on demand.
//! It monitors workload requirements and dynamically provisions or removes GPU nodes
//! to optimize cost and availability.
//!
//! ## Overview
//!
//! `basilica-autoscaler` enables:
//!
//! - **Demand-Based Scaling**: Scale nodes based on pending workload requirements
//! - **GPU-Aware**: Understands GPU types and allocates appropriate nodes
//! - **Cost Optimization**: Minimize costs by scaling down idle resources
//! - **K3s Integration**: Native integration with K3s clusters
//! - **Warm Pools**: Pre-provisioned nodes for fast scaling
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use basilica_autoscaler::{AutoscalerConfig, KubeClient};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = AutoscalerConfig::from_env()?;
//!     let client = KubeClient::new().await?;
//!     
//!     // Start the autoscaler controller
//!     client.run(config).await?;
//!     
//!     Ok(())
//! }
//! ```
//!
//! ## Related Crates
//!
//! - [`basilica-operator`](https://crates.io/crates/basilica-operator) - K8s operator
//! - [`basilica-common`](https://crates.io/crates/basilica-common) - Core types

pub mod api;
pub mod config;
pub mod controllers;
pub mod crd;
pub mod error;
pub mod health;
pub mod leader_election;
pub mod metrics;
pub mod offering_matcher;
pub mod provisioner;
pub mod runtime;
pub mod warm_pool;

pub use config::AutoscalerConfig;
pub use error::{AutoscalerError, Result};

// Re-export commonly used types
pub use controllers::{AutoscalerK8sClient, KubeClient};
pub use offering_matcher::{
    GpuRequirements, MaybeOfferingSelector, OfferingConstraints, OfferingMatcher,
    OfferingMatcherConfig, OfferingSelector, PendingGpuPod,
};
pub use provisioner::{NodeProvisioner, SshProvisioner};
