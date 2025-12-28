//! # basilica-storage
//!
//! Storage daemon for Basilica GPU workloads with R2/S3 and FUSE support.
//!
//! This crate provides persistent storage for GPU workloads running on Basilica.
//! It supports object storage (S3/R2 compatible) and can mount storage as a FUSE
//! filesystem for transparent access.
//!
//! ## Overview
//!
//! `basilica-storage` provides:
//!
//! - **Object Storage**: S3/R2 compatible storage backend
//! - **FUSE Filesystem**: Mount object storage as a local filesystem
//! - **Namespace Isolation**: Per-user/per-workload storage isolation
//! - **Quota Management**: Storage quotas and rate limiting
//! - **Kubernetes Integration**: Auto-credential fetching from K8s secrets
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use basilica_storage::{StorageConfig, S3Backend};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = StorageConfig::from_env()?;
//!     let backend = S3Backend::new(&config).await?;
//!     
//!     // Upload a file
//!     backend.put_object("models/checkpoint.pt", &data).await?;
//!     
//!     // Download a file
//!     let data = backend.get_object("models/checkpoint.pt").await?;
//!     
//!     Ok(())
//! }
//! ```
//!
//! ## Feature Flags
//!
//! - `fuse` - Enable FUSE filesystem support (default, requires `libfuse3-dev`)
//!
//! ## Architecture
//!
//! ### FUSE Filesystem (Production)
//! - Transparent file I/O backed by object storage
//! - In-memory caching with background sync
//! - Continuous protection (syncs every 1 second)
//! - Zero code changes for users
//! - Supports mmap for numpy/PyTorch
//!
//! ### DaemonSet Mode (Multi-Tenant)
//! - Single daemon per node manages mounts for all namespaces
//! - Reads credentials from Kubernetes secrets in user namespaces
//! - Namespace-scoped isolation with RBAC
//!
//! ### Snapshot Manager (Legacy/Testing)
//! - Manual snapshot-on-pause approach
//! - For testing and backwards compatibility
//! - Use FUSE for production workloads
//!
//! ## Related Crates
//!
//! - [`basilica-operator`](https://crates.io/crates/basilica-operator) - K8s operator
//! - [`basilica-common`](https://crates.io/crates/basilica-common) - Core types

pub mod backend;
pub mod config;
pub mod credentials;
#[cfg(feature = "fuse")]
pub mod daemon;
pub mod error;
#[cfg(feature = "fuse")]
pub mod fuse;
#[cfg(feature = "fuse")]
pub mod http;
pub mod metrics;
pub mod quota;
pub mod snapshot;

pub use backend::{S3Backend, StorageBackend};
pub use config::StorageConfig;
pub use credentials::{
    CredentialError, CredentialProvider, KubernetesCredentialProvider, StorageCredentials,
};
#[cfg(feature = "fuse")]
pub use daemon::{
    MountError, MountInfo, MountManager, MountStatus, NamespaceMetrics, NamespaceMetricsSnapshot,
    NamespaceWatcher, PerNamespaceMetricsStore, WatcherError, DEFAULT_BASE_PATH,
};
pub use error::{Result, StorageError};
#[cfg(feature = "fuse")]
pub use fuse::{BasilicaFS, DirtyPageTracker, PageCache, SyncWorker};
#[cfg(feature = "fuse")]
pub use http::DaemonHttpServer;
#[cfg(feature = "fuse")]
pub use http::HttpServer;
pub use metrics::StorageMetrics;
pub use quota::{QuotaError, QuotaUsage, StorageQuota};
pub use snapshot::{SnapshotManager, SnapshotMetadata};
