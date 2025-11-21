//! Basilica Storage Layer
//!
//! Provides persistent storage for stateful jobs using object storage backends (S3, R2, GCS).
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
//! ### Snapshot Manager (Legacy/Testing)
//! - Manual snapshot-on-pause approach
//! - For testing and backwards compatibility
//! - Use FUSE for production workloads

pub mod backend;
pub mod config;
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
pub use error::{Result, StorageError};
#[cfg(feature = "fuse")]
pub use fuse::{BasilicaFS, DirtyPageTracker, PageCache, SyncWorker};
#[cfg(feature = "fuse")]
pub use http::HttpServer;
pub use metrics::StorageMetrics;
pub use quota::{QuotaError, QuotaUsage, StorageQuota};
pub use snapshot::{SnapshotManager, SnapshotMetadata};
