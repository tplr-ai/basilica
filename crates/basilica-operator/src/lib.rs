//! # basilica-operator
//!
//! Basilica Kubernetes operator: CRDs and controllers for GPU workload orchestration.
//!
//! This crate provides a Kubernetes operator that manages GPU workloads on K3s clusters.
//! It defines Custom Resource Definitions (CRDs) for declarative workload management
//! and handles the full lifecycle of GPU rentals.
//!
//! ## Overview
//!
//! `basilica-operator` provides:
//!
//! - **Custom Resources**: `UserDeployment`, `GpuRental`, `GpuNode` CRDs
//! - **Lifecycle Management**: Full workload lifecycle from creation to cleanup
//! - **Node Onboarding**: Automatic GPU node discovery and registration
//! - **Resource Scheduling**: GPU-aware scheduling with affinity rules
//! - **Health Monitoring**: Continuous health checks and auto-recovery
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use basilica_operator::runtime::OperatorRuntime;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let runtime = OperatorRuntime::new().await?;
//!     runtime.run().await?;
//!     Ok(())
//! }
//! ```
//!
//! ## Custom Resources
//!
//! ### UserDeployment
//!
//! Represents a user's GPU workload deployment.
//!
//! ### GpuRental
//!
//! Represents an active GPU rental session.
//!
//! ### GpuNode
//!
//! Represents a registered GPU node in the cluster.
//!
//! ## Related Crates
//!
//! - [`basilica-autoscaler`](https://crates.io/crates/basilica-autoscaler) - Node autoscaling
//! - [`basilica-api`](https://crates.io/crates/basilica-api) - API gateway
//! - [`basilica-common`](https://crates.io/crates/basilica-common) - Core types

pub mod admission;
pub mod billing;
pub mod config;
pub mod controllers;
pub mod crd;
pub mod k8s_client;
pub mod labels;
pub mod metrics;
pub mod metrics_provider;
pub mod runtime;
pub mod security;
