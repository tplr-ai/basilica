use governor::{
    clock::DefaultClock,
    state::{direct::NotKeyed, InMemoryState},
    Quota, RateLimiter,
};
use std::num::NonZeroU32;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum QuotaError {
    #[error("Storage size quota exceeded: current={current} bytes, limit={limit} bytes")]
    SizeExceeded { current: u64, limit: u64 },

    #[error("File count quota exceeded: current={current}, limit={limit}")]
    FileCountExceeded { current: u64, limit: u64 },

    #[error("Operation rate limit exceeded: {operations_per_second} ops/sec")]
    RateLimitExceeded { operations_per_second: u32 },
}

/// Storage quota configuration and enforcement
///
/// Tracks and enforces limits on:
/// - Total storage size (bytes)
/// - File count
/// - Operations per second (rate limiting)
#[derive(Clone)]
pub struct StorageQuota {
    max_size_bytes: u64,
    max_files: u64,
    max_operations_per_second: u32,

    current_size: Arc<AtomicU64>,
    current_files: Arc<AtomicU64>,
    rate_limiter: Arc<RateLimiter<NotKeyed, InMemoryState, DefaultClock>>,
}

impl StorageQuota {
    /// Create a new storage quota with specified limits
    pub fn new(max_size_bytes: u64, max_files: u64, max_operations_per_second: u32) -> Self {
        let quota = Quota::per_second(
            NonZeroU32::new(max_operations_per_second).unwrap_or(NonZeroU32::new(100).unwrap()),
        );
        let rate_limiter = Arc::new(RateLimiter::direct(quota));

        Self {
            max_size_bytes,
            max_files,
            max_operations_per_second,
            current_size: Arc::new(AtomicU64::new(0)),
            current_files: Arc::new(AtomicU64::new(0)),
            rate_limiter,
        }
    }

    /// Check rate limit before operation
    pub fn check_rate_limit(&self) -> Result<(), QuotaError> {
        match self.rate_limiter.check() {
            Ok(_) => Ok(()),
            Err(_) => Err(QuotaError::RateLimitExceeded {
                operations_per_second: self.max_operations_per_second,
            }),
        }
    }

    /// Check if adding bytes would exceed size quota
    pub fn check_size_quota(&self, bytes: u64) -> Result<(), QuotaError> {
        let current = self.current_size.load(Ordering::Relaxed);
        let new_total = current.saturating_add(bytes);

        if new_total > self.max_size_bytes {
            return Err(QuotaError::SizeExceeded {
                current: new_total,
                limit: self.max_size_bytes,
            });
        }

        Ok(())
    }

    /// Check if adding a file would exceed file count quota
    pub fn check_file_quota(&self) -> Result<(), QuotaError> {
        let current = self.current_files.load(Ordering::Relaxed);

        if current >= self.max_files {
            return Err(QuotaError::FileCountExceeded {
                current,
                limit: self.max_files,
            });
        }

        Ok(())
    }

    /// Reserve space atomically (call after check_size_quota succeeds)
    pub fn reserve_space(&self, bytes: u64) {
        self.current_size.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Release space atomically
    pub fn release_space(&self, bytes: u64) {
        self.current_size.fetch_sub(bytes, Ordering::Relaxed);
    }

    /// Increment file count atomically
    pub fn increment_file_count(&self) {
        self.current_files.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement file count atomically
    pub fn decrement_file_count(&self) {
        self.current_files.fetch_sub(1, Ordering::Relaxed);
    }

    /// Get current usage statistics
    pub fn get_usage(&self) -> QuotaUsage {
        QuotaUsage {
            current_size_bytes: self.current_size.load(Ordering::Relaxed),
            max_size_bytes: self.max_size_bytes,
            current_files: self.current_files.load(Ordering::Relaxed),
            max_files: self.max_files,
            max_operations_per_second: self.max_operations_per_second,
        }
    }
}

/// Current quota usage statistics
#[derive(Debug, Clone)]
pub struct QuotaUsage {
    pub current_size_bytes: u64,
    pub max_size_bytes: u64,
    pub current_files: u64,
    pub max_files: u64,
    pub max_operations_per_second: u32,
}

impl QuotaUsage {
    /// Calculate size usage percentage
    pub fn size_percentage(&self) -> f64 {
        if self.max_size_bytes == 0 {
            return 0.0;
        }
        (self.current_size_bytes as f64 / self.max_size_bytes as f64) * 100.0
    }

    /// Calculate file count usage percentage
    pub fn file_percentage(&self) -> f64 {
        if self.max_files == 0 {
            return 0.0;
        }
        (self.current_files as f64 / self.max_files as f64) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_quota_creation() {
        let quota = StorageQuota::new(1024 * 1024 * 1024, 10000, 100);
        let usage = quota.get_usage();

        assert_eq!(usage.max_size_bytes, 1024 * 1024 * 1024);
        assert_eq!(usage.max_files, 10000);
        assert_eq!(usage.current_size_bytes, 0);
        assert_eq!(usage.current_files, 0);
    }

    #[test]
    fn test_size_quota_check() {
        let quota = StorageQuota::new(1000, 10, 100);

        assert!(quota.check_size_quota(500).is_ok());
        assert!(quota.check_size_quota(1000).is_ok());
        assert!(quota.check_size_quota(1001).is_err());
    }

    #[test]
    fn test_size_quota_enforcement() {
        let quota = StorageQuota::new(1000, 10, 100);

        quota.reserve_space(500);
        assert_eq!(quota.get_usage().current_size_bytes, 500);

        assert!(quota.check_size_quota(500).is_ok());
        assert!(quota.check_size_quota(501).is_err());

        quota.release_space(200);
        assert_eq!(quota.get_usage().current_size_bytes, 300);

        assert!(quota.check_size_quota(700).is_ok());
    }

    #[test]
    fn test_file_quota_check() {
        let quota = StorageQuota::new(1000000, 5, 100);

        assert!(quota.check_file_quota().is_ok());

        quota.increment_file_count();
        quota.increment_file_count();
        quota.increment_file_count();
        quota.increment_file_count();
        quota.increment_file_count();

        assert_eq!(quota.get_usage().current_files, 5);
        assert!(quota.check_file_quota().is_err());

        quota.decrement_file_count();
        assert!(quota.check_file_quota().is_ok());
    }

    #[tokio::test]
    async fn test_rate_limiting() {
        let quota = StorageQuota::new(1000000, 10000, 2);

        assert!(quota.check_rate_limit().is_ok());
        assert!(quota.check_rate_limit().is_ok());

        let result = quota.check_rate_limit();
        assert!(result.is_err());

        tokio::time::sleep(Duration::from_millis(600)).await;

        assert!(quota.check_rate_limit().is_ok());
    }

    #[test]
    fn test_usage_percentages() {
        let quota = StorageQuota::new(1000, 100, 100);

        quota.reserve_space(500);
        quota.increment_file_count();
        quota.increment_file_count();
        quota.increment_file_count();

        let usage = quota.get_usage();
        assert_eq!(usage.size_percentage(), 50.0);
        assert_eq!(usage.file_percentage(), 3.0);
    }

    #[test]
    fn test_concurrent_space_reservation() {
        use std::thread;

        let quota = Arc::new(StorageQuota::new(10000, 1000, 1000));
        let mut handles = vec![];

        for _ in 0..10 {
            let quota_clone = quota.clone();
            let handle = thread::spawn(move || {
                for _ in 0..100 {
                    quota_clone.reserve_space(1);
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(quota.get_usage().current_size_bytes, 1000);
    }
}
