//! Tracks dirty pages that need to be synced to object storage

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// A dirty region that needs syncing
#[derive(Debug, Clone)]
pub struct DirtyRegion {
    /// File path
    pub path: String,
    /// Start offset in file
    pub offset: u64,
    /// Length of dirty region
    pub length: usize,
    /// When this region was marked dirty
    pub marked_at: std::time::Instant,
}

// Implement PartialEq to compare regions without marked_at
impl PartialEq for DirtyRegion {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path && self.offset == other.offset && self.length == other.length
    }
}

impl Eq for DirtyRegion {}

impl std::hash::Hash for DirtyRegion {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.path.hash(state);
        self.offset.hash(state);
        self.length.hash(state);
    }
}

impl DirtyRegion {
    /// Get the object storage key for this region
    pub fn storage_key(&self, namespace: &str, experiment_id: &str) -> String {
        let path = self.path.trim_start_matches('/');
        format!("{}/{}/{}", namespace, experiment_id, path)
    }

    /// Get the end offset of this region
    pub fn end_offset(&self) -> u64 {
        self.offset + self.length as u64
    }

    /// Check if this region overlaps with another
    pub fn overlaps(&self, other: &DirtyRegion) -> bool {
        self.path == other.path
            && self.offset < other.end_offset()
            && other.offset < self.end_offset()
    }

    /// Merge with another overlapping region
    pub fn merge(&self, other: &DirtyRegion) -> DirtyRegion {
        assert!(self.overlaps(other), "Regions must overlap to merge");

        let start = self.offset.min(other.offset);
        let end = self.end_offset().max(other.end_offset());

        DirtyRegion {
            path: self.path.clone(),
            offset: start,
            length: (end - start) as usize,
            marked_at: self.marked_at.min(other.marked_at),
        }
    }
}

/// Tracks dirty pages for background syncing
pub struct DirtyPageTracker {
    /// All dirty regions, indexed by file path
    dirty_regions: Arc<RwLock<HashMap<String, Vec<DirtyRegion>>>>,

    /// Memory-mapped regions (tracked separately for msync)
    mmapped_regions: Arc<RwLock<HashMap<String, Vec<DirtyRegion>>>>,
}

impl DirtyPageTracker {
    /// Create a new dirty page tracker
    pub fn new() -> Self {
        Self {
            dirty_regions: Arc::new(RwLock::new(HashMap::new())),
            mmapped_regions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Mark a region as dirty
    pub async fn mark_dirty(&self, path: &str, offset: u64, length: usize) {
        let mut regions = self.dirty_regions.write().await;

        let new_region = DirtyRegion {
            path: path.to_string(),
            offset,
            length,
            marked_at: std::time::Instant::now(),
        };

        let file_regions = regions.entry(path.to_string()).or_insert_with(Vec::new);

        // Try to merge with existing overlapping regions
        let mut merged = new_region;
        file_regions.retain(|existing| {
            if merged.overlaps(existing) {
                merged = merged.merge(existing);
                false // Remove the old region
            } else {
                true // Keep non-overlapping regions
            }
        });

        file_regions.push(merged);

        tracing::debug!("Marked dirty: {} @ {} len {}", path, offset, length);
    }

    /// Track a memory-mapped region
    pub async fn track_mmap(&self, path: &str, offset: u64, length: usize) {
        let mut regions = self.mmapped_regions.write().await;

        let region = DirtyRegion {
            path: path.to_string(),
            offset,
            length,
            marked_at: std::time::Instant::now(),
        };

        regions
            .entry(path.to_string())
            .or_insert_with(Vec::new)
            .push(region);

        tracing::debug!("Tracking mmap: {} @ {} len {}", path, offset, length);
    }

    /// Get all dirty regions (consumes them)
    pub async fn get_dirty_regions(&self) -> Vec<DirtyRegion> {
        let regions = self.dirty_regions.read().await;
        let mut all_regions = Vec::new();

        for (_path, file_regions) in regions.iter() {
            all_regions.extend(file_regions.clone());
        }

        // Also check mmapped regions (assume all dirty)
        let mmapped = self.mmapped_regions.read().await;
        for (_path, file_regions) in mmapped.iter() {
            all_regions.extend(file_regions.clone());
        }

        all_regions
    }

    /// Mark a region as clean (synced to storage)
    pub async fn mark_clean(&self, region: &DirtyRegion) {
        let mut regions = self.dirty_regions.write().await;

        if let Some(file_regions) = regions.get_mut(&region.path) {
            file_regions.retain(|r| r != region);

            if file_regions.is_empty() {
                regions.remove(&region.path);
            }
        }

        tracing::debug!(
            "Marked clean: {} @ {} len {}",
            region.path,
            region.offset,
            region.length
        );
    }

    /// Get the number of dirty regions
    pub async fn dirty_count(&self) -> usize {
        let regions = self.dirty_regions.read().await;
        regions.values().map(|v| v.len()).sum()
    }

    /// Clear all dirty tracking (for testing)
    pub async fn clear(&self) {
        let mut regions = self.dirty_regions.write().await;
        regions.clear();

        let mut mmapped = self.mmapped_regions.write().await;
        mmapped.clear();
    }
}

impl Default for DirtyPageTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mark_dirty() {
        let tracker = DirtyPageTracker::new();

        tracker.mark_dirty("/test.txt", 0, 100).await;
        tracker.mark_dirty("/test.txt", 200, 100).await;

        let regions = tracker.get_dirty_regions().await;
        assert_eq!(regions.len(), 2);
    }

    #[tokio::test]
    async fn test_merge_overlapping_regions() {
        let tracker = DirtyPageTracker::new();

        // These should merge into one region
        tracker.mark_dirty("/test.txt", 0, 100).await;
        tracker.mark_dirty("/test.txt", 50, 100).await; // Overlaps

        let regions = tracker.get_dirty_regions().await;
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].offset, 0);
        assert_eq!(regions[0].length, 150);
    }

    #[tokio::test]
    async fn test_mark_clean() {
        let tracker = DirtyPageTracker::new();

        tracker.mark_dirty("/test.txt", 0, 100).await;

        let regions = tracker.get_dirty_regions().await;
        assert_eq!(regions.len(), 1);

        tracker.mark_clean(&regions[0]).await;

        let regions = tracker.get_dirty_regions().await;
        assert_eq!(regions.len(), 0);
    }

    #[test]
    fn test_region_overlaps() {
        let r1 = DirtyRegion {
            path: "/test.txt".to_string(),
            offset: 0,
            length: 100,
            marked_at: std::time::Instant::now(),
        };

        let r2 = DirtyRegion {
            path: "/test.txt".to_string(),
            offset: 50,
            length: 100,
            marked_at: std::time::Instant::now(),
        };

        assert!(r1.overlaps(&r2));
        assert!(r2.overlaps(&r1));
    }

    #[test]
    fn test_region_merge() {
        let r1 = DirtyRegion {
            path: "/test.txt".to_string(),
            offset: 0,
            length: 100,
            marked_at: std::time::Instant::now(),
        };

        let r2 = DirtyRegion {
            path: "/test.txt".to_string(),
            offset: 50,
            length: 100,
            marked_at: std::time::Instant::now(),
        };

        let merged = r1.merge(&r2);
        assert_eq!(merged.offset, 0);
        assert_eq!(merged.length, 150);
    }
}
