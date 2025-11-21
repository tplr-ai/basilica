use crate::fuse::BasilicaFS;
use crate::metrics::StorageMetrics;
use axum::{extract::State, routing::get, Json, Router};
use parking_lot::Mutex;
use serde_json::json;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tokio::net::TcpListener;
use tracing::info;

pub struct HttpServer {
    metrics: Arc<StorageMetrics>,
    fs: Arc<Mutex<BasilicaFS>>,
}

impl HttpServer {
    pub fn new(metrics: Arc<StorageMetrics>, fs: Arc<Mutex<BasilicaFS>>) -> Self {
        Self { metrics, fs }
    }

    pub async fn serve(self, addr: &str) -> Result<(), Box<dyn std::error::Error>> {
        let app = Router::new()
            .route("/health", get(health_handler))
            .route("/ready", get(ready_handler))
            .route("/metrics", get(metrics_handler))
            .with_state(Arc::new(self));

        let listener = TcpListener::bind(addr).await?;
        info!("HTTP server listening on {}", addr);

        axum::serve(listener, app).await?;
        Ok(())
    }
}

async fn health_handler() -> Json<serde_json::Value> {
    Json(json!({
        "status": "healthy",
        "timestamp": chrono::Utc::now().to_rfc3339(),
    }))
}

async fn ready_handler(State(server): State<Arc<HttpServer>>) -> Json<serde_json::Value> {
    let is_ready = {
        let fs = server.fs.lock();
        let inodes = fs.inodes.read();
        inodes.contains_key(&1)
    };

    if is_ready {
        Json(json!({
            "status": "ready",
            "timestamp": chrono::Utc::now().to_rfc3339(),
        }))
    } else {
        Json(json!({
            "status": "not_ready",
            "reason": "FUSE filesystem not initialized",
            "timestamp": chrono::Utc::now().to_rfc3339(),
        }))
    }
}

async fn metrics_handler(State(server): State<Arc<HttpServer>>) -> String {
    let m = &server.metrics;
    let used_bytes = server.fs.lock().used_bytes.load(Ordering::Relaxed);

    format!(
        "# HELP storage_reads_total Total read operations\n\
         # TYPE storage_reads_total counter\n\
         storage_reads_total {}\n\
         \n\
         # HELP storage_writes_total Total write operations\n\
         # TYPE storage_writes_total counter\n\
         storage_writes_total {}\n\
         \n\
         # HELP storage_bytes_read_total Total bytes read\n\
         # TYPE storage_bytes_read_total counter\n\
         storage_bytes_read_total {}\n\
         \n\
         # HELP storage_bytes_written_total Total bytes written\n\
         # TYPE storage_bytes_written_total counter\n\
         storage_bytes_written_total {}\n\
         \n\
         # HELP storage_cache_hits_total Cache hit count\n\
         # TYPE storage_cache_hits_total counter\n\
         storage_cache_hits_total {}\n\
         \n\
         # HELP storage_cache_misses_total Cache miss count\n\
         # TYPE storage_cache_misses_total counter\n\
         storage_cache_misses_total {}\n\
         \n\
         # HELP storage_quota_exceeded_total Quota exceeded errors\n\
         # TYPE storage_quota_exceeded_total counter\n\
         storage_quota_exceeded_total {}\n\
         \n\
         # HELP storage_used_bytes Current storage usage\n\
         # TYPE storage_used_bytes gauge\n\
         storage_used_bytes {}\n",
        m.reads.load(Ordering::Relaxed),
        m.writes.load(Ordering::Relaxed),
        m.bytes_read.load(Ordering::Relaxed),
        m.bytes_written.load(Ordering::Relaxed),
        m.cache_hits.load(Ordering::Relaxed),
        m.cache_misses.load(Ordering::Relaxed),
        m.quota_exceeded.load(Ordering::Relaxed),
        used_bytes,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::StorageBackend;
    use crate::error::Result;
    use crate::fuse::BasilicaFS;
    use async_trait::async_trait;
    use bytes::Bytes;
    use std::collections::HashMap;
    use tokio::sync::RwLock as TokioRwLock;

    struct MockStorage {
        data: Arc<TokioRwLock<HashMap<String, Bytes>>>,
    }

    impl MockStorage {
        fn new() -> Self {
            Self {
                data: Arc::new(TokioRwLock::new(HashMap::new())),
            }
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
    async fn test_health_endpoint() {
        let json_response = health_handler().await;
        let value = json_response.0;

        assert_eq!(value["status"], "healthy");
        assert!(value["timestamp"].is_string());
    }

    #[tokio::test]
    async fn test_ready_endpoint() {
        let storage = Arc::new(MockStorage::new());
        let metrics = Arc::new(StorageMetrics::new());
        let fs = BasilicaFS::new(
            "u-test".to_string(),
            "exp-test".to_string(),
            storage,
            1000,
            10,
            1024 * 1024 * 1024,
            metrics.clone(),
        );

        let fs_mutex = Arc::new(Mutex::new(fs));
        let server = Arc::new(HttpServer::new(metrics, fs_mutex));

        let state = axum::extract::State(server);
        let json_response = ready_handler(state).await;
        let value = json_response.0;

        assert_eq!(value["status"], "ready");
        assert!(value["timestamp"].is_string());
    }

    #[tokio::test]
    async fn test_metrics_endpoint() {
        let storage = Arc::new(MockStorage::new());
        let metrics = Arc::new(StorageMetrics::new());

        metrics.reads.store(100, Ordering::Relaxed);
        metrics.writes.store(50, Ordering::Relaxed);
        metrics.bytes_read.store(1024, Ordering::Relaxed);
        metrics.bytes_written.store(512, Ordering::Relaxed);
        metrics.cache_hits.store(75, Ordering::Relaxed);
        metrics.cache_misses.store(25, Ordering::Relaxed);

        let fs = BasilicaFS::new(
            "u-test".to_string(),
            "exp-test".to_string(),
            storage,
            1000,
            10,
            1024 * 1024 * 1024,
            metrics.clone(),
        );

        let fs_mutex = Arc::new(Mutex::new(fs));
        let server = Arc::new(HttpServer::new(metrics, fs_mutex));

        let state = axum::extract::State(server);
        let text = metrics_handler(state).await;

        assert!(text.contains("storage_reads_total 100"));
        assert!(text.contains("storage_writes_total 50"));
        assert!(text.contains("storage_bytes_read_total 1024"));
        assert!(text.contains("storage_bytes_written_total 512"));
        assert!(text.contains("storage_cache_hits_total 75"));
        assert!(text.contains("storage_cache_misses_total 25"));
    }
}
