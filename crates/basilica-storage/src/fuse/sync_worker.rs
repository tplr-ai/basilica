//! Background worker that syncs dirty pages to object storage

use crate::backend::StorageBackend;
use crate::fuse::cache::PageCache;
use crate::fuse::dirty_tracker::{DirtyPageTracker, DirtyRegion};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

const STORAGE_OP_TIMEOUT_SECS: u64 = 30;

/// Background worker for syncing dirty pages
pub struct SyncWorker {
    /// Experiment ID (used as prefix in object storage)
    experiment_id: String,

    /// Storage backend
    storage: Arc<dyn StorageBackend>,

    /// Page cache
    cache: Arc<RwLock<PageCache>>,

    /// Dirty page tracker
    dirty_tracker: Arc<DirtyPageTracker>,

    /// Sync interval
    sync_interval: Duration,

    /// Whether the worker is running
    running: Arc<tokio::sync::RwLock<bool>>,
}

impl SyncWorker {
    /// Create a new sync worker
    pub fn new(
        experiment_id: String,
        storage: Arc<dyn StorageBackend>,
        cache: Arc<RwLock<PageCache>>,
        dirty_tracker: Arc<DirtyPageTracker>,
        sync_interval_ms: u64,
    ) -> Self {
        Self {
            experiment_id,
            storage,
            cache,
            dirty_tracker,
            sync_interval: Duration::from_millis(sync_interval_ms),
            running: Arc::new(tokio::sync::RwLock::new(false)),
        }
    }

    /// Start the sync worker
    pub async fn start(&self) -> tokio::task::JoinHandle<()> {
        let experiment_id = self.experiment_id.clone();
        let storage = self.storage.clone();
        let cache = self.cache.clone();
        let dirty_tracker = self.dirty_tracker.clone();
        let sync_interval = self.sync_interval;
        let running = self.running.clone();

        *running.write().await = true;

        tokio::spawn(async move {
            info!("Starting sync worker for experiment {}", experiment_id);

            while *running.read().await {
                tokio::time::sleep(sync_interval).await;

                // Get all dirty regions
                let dirty_regions = dirty_tracker.get_dirty_regions().await;

                if dirty_regions.is_empty() {
                    debug!("No dirty pages to sync");
                    continue;
                }

                info!("Syncing {} dirty regions", dirty_regions.len());

                // Sync each region
                for region in dirty_regions {
                    if let Err(e) =
                        Self::sync_region(&experiment_id, &storage, &cache, &dirty_tracker, &region)
                            .await
                    {
                        error!(
                            "Failed to sync region {} @ {}: {}",
                            region.path, region.offset, e
                        );
                    }
                }
            }

            info!("Sync worker stopped");
        })
    }

    /// Stop the sync worker
    pub async fn stop(&self) {
        *self.running.write().await = false;
    }

    /// Sync a single region to object storage
    async fn sync_region(
        experiment_id: &str,
        storage: &Arc<dyn StorageBackend>,
        cache: &Arc<RwLock<PageCache>>,
        dirty_tracker: &Arc<DirtyPageTracker>,
        region: &DirtyRegion,
    ) -> Result<(), String> {
        debug!(
            "Syncing region: {} @ {} len {}",
            region.path, region.offset, region.length
        );

        // Read data from cache
        let cache = cache.read().await;
        let data = cache
            .read(&region.path, region.offset, region.length)
            .await
            .ok_or_else(|| format!("Data not in cache: {}", region.path))?;

        drop(cache); // Release cache lock before I/O

        // Generate storage key
        let key = region.storage_key(experiment_id);

        // Upload to object storage with timeout to prevent blocking forever
        let timeout = Duration::from_secs(STORAGE_OP_TIMEOUT_SECS);
        let put_result = tokio::time::timeout(timeout, storage.put(&key, data)).await;

        match put_result {
            Ok(Ok(_)) => {
                debug!("Successfully synced: {} @ {}", region.path, region.offset);
                dirty_tracker.mark_clean(region).await;
                Ok(())
            }
            Ok(Err(e)) => {
                warn!("Failed to sync {} @ {}: {}", region.path, region.offset, e);
                Err(format!("Storage error: {}", e))
            }
            Err(_) => {
                warn!(
                    "Timeout syncing {} @ {} after {}s",
                    region.path, region.offset, STORAGE_OP_TIMEOUT_SECS
                );
                Err(format!(
                    "Storage timeout after {}s",
                    STORAGE_OP_TIMEOUT_SECS
                ))
            }
        }
    }

    /// Force sync all dirty pages (for graceful shutdown)
    pub async fn flush_all(&self) -> Result<(), String> {
        info!(
            "Flushing all dirty pages for experiment {}",
            self.experiment_id
        );

        let dirty_regions = self.dirty_tracker.get_dirty_regions().await;

        info!("Flushing {} dirty regions", dirty_regions.len());

        for region in dirty_regions {
            Self::sync_region(
                &self.experiment_id,
                &self.storage,
                &self.cache,
                &self.dirty_tracker,
                &region,
            )
            .await?;
        }

        info!("All dirty pages flushed");
        Ok(())
    }

    /// Get sync statistics
    pub async fn get_stats(&self) -> SyncStats {
        SyncStats {
            dirty_count: self.dirty_tracker.dirty_count().await,
            sync_interval_ms: self.sync_interval.as_millis() as u64,
            running: *self.running.read().await,
        }
    }
}

/// Sync worker statistics
#[derive(Debug, Clone)]
pub struct SyncStats {
    /// Number of dirty regions
    pub dirty_count: usize,
    /// Sync interval in milliseconds
    pub sync_interval_ms: u64,
    /// Whether the worker is running
    pub running: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::StorageBackend;
    use crate::error::Result;
    use async_trait::async_trait;
    use bytes::Bytes;
    use std::collections::HashMap;
    use tokio::sync::RwLock as TokioRwLock;

    /// Mock storage backend for testing
    struct MockStorage {
        data: Arc<TokioRwLock<HashMap<String, Bytes>>>,
    }

    impl MockStorage {
        fn new() -> Self {
            Self {
                data: Arc::new(TokioRwLock::new(HashMap::new())),
            }
        }

        async fn get_data(&self, key: &str) -> Option<Bytes> {
            self.data.read().await.get(key).cloned()
        }
    }

    #[async_trait]
    impl StorageBackend for MockStorage {
        async fn put(&self, key: &str, data: Bytes) -> Result<()> {
            self.data.write().await.insert(key.to_string(), data);
            Ok(())
        }

        async fn get(&self, key: &str) -> Result<Bytes> {
            self.data
                .read()
                .await
                .get(key)
                .cloned()
                .ok_or_else(|| crate::error::StorageError::SnapshotNotFound(key.to_string()))
        }

        async fn exists(&self, key: &str) -> Result<bool> {
            Ok(self.data.read().await.contains_key(key))
        }

        async fn delete(&self, key: &str) -> Result<()> {
            self.data.write().await.remove(key);
            Ok(())
        }

        async fn list(&self, _prefix: &str) -> Result<Vec<String>> {
            Ok(self.data.read().await.keys().cloned().collect())
        }
    }

    #[tokio::test]
    async fn test_sync_worker() {
        let storage = Arc::new(MockStorage::new());
        let cache = Arc::new(RwLock::new(PageCache::new(10)));
        let dirty_tracker = Arc::new(DirtyPageTracker::new());

        // Write some data to cache
        let data = b"Hello, World!";
        cache
            .write()
            .await
            .write("/test.txt", 0, data)
            .await
            .unwrap();

        // Mark as dirty
        dirty_tracker.mark_dirty("/test.txt", 0, data.len()).await;

        // Create and start sync worker
        let worker = SyncWorker::new(
            "exp-123".to_string(),
            storage.clone(),
            cache.clone(),
            dirty_tracker.clone(),
            100, // 100ms sync interval
        );

        let handle = worker.start().await;

        // Wait for sync
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Stop worker
        worker.stop().await;
        handle.abort();

        // Verify data was synced
        let synced_data = storage.get_data("exp-123/test.txt").await;
        assert!(synced_data.is_some());
        assert_eq!(&synced_data.unwrap()[..], data);

        // Verify region was marked clean
        let dirty_count = dirty_tracker.dirty_count().await;
        assert_eq!(dirty_count, 0);
    }

    #[tokio::test]
    async fn test_flush_all() {
        let storage = Arc::new(MockStorage::new());
        let cache = Arc::new(RwLock::new(PageCache::new(10)));
        let dirty_tracker = Arc::new(DirtyPageTracker::new());

        // Write multiple files
        for i in 0..5 {
            let path = format!("/file-{}.txt", i);
            let data = format!("content-{}", i);

            cache
                .write()
                .await
                .write(&path, 0, data.as_bytes())
                .await
                .unwrap();

            dirty_tracker.mark_dirty(&path, 0, data.len()).await;
        }

        // Create worker (don't start)
        let worker = SyncWorker::new(
            "exp-123".to_string(),
            storage.clone(),
            cache.clone(),
            dirty_tracker.clone(),
            1000,
        );

        // Flush all
        worker.flush_all().await.unwrap();

        // Verify all files were synced
        for i in 0..5 {
            let key = format!("exp-123/file-{}.txt", i);
            let data = storage.get_data(&key).await;
            assert!(data.is_some(), "File {} not synced", i);
        }

        // Verify all regions marked clean
        let dirty_count = dirty_tracker.dirty_count().await;
        assert_eq!(dirty_count, 0);
    }
}
