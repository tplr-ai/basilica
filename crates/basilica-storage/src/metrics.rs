use std::sync::atomic::AtomicU64;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct StorageMetrics {
    pub reads: Arc<AtomicU64>,
    pub writes: Arc<AtomicU64>,
    pub bytes_read: Arc<AtomicU64>,
    pub bytes_written: Arc<AtomicU64>,
    pub cache_hits: Arc<AtomicU64>,
    pub cache_misses: Arc<AtomicU64>,
    pub quota_exceeded: Arc<AtomicU64>,
}

impl StorageMetrics {
    pub fn new() -> Self {
        Self {
            reads: Arc::new(AtomicU64::new(0)),
            writes: Arc::new(AtomicU64::new(0)),
            bytes_read: Arc::new(AtomicU64::new(0)),
            bytes_written: Arc::new(AtomicU64::new(0)),
            cache_hits: Arc::new(AtomicU64::new(0)),
            cache_misses: Arc::new(AtomicU64::new(0)),
            quota_exceeded: Arc::new(AtomicU64::new(0)),
        }
    }
}

impl Default for StorageMetrics {
    fn default() -> Self {
        Self::new()
    }
}
