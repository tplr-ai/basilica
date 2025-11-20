//! In-memory page cache for FUSE filesystem
//!
//! Provides fast read/write access while data syncs to object storage in background.

use bytes::Bytes;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Size of a cache page (64KB - good balance for large files)
pub const PAGE_SIZE: usize = 64 * 1024;

/// A single page of cached data
#[derive(Debug, Clone)]
pub struct Page {
    /// Page data
    pub data: Bytes,
    /// Whether this page has been modified
    pub dirty: bool,
    /// Last access timestamp
    pub last_access: std::time::Instant,
}

/// Cache entry for a single file
#[derive(Debug)]
pub struct FileCache {
    /// File path
    pub path: String,
    /// File size (may be larger than cached data)
    pub size: u64,
    /// Cached pages, keyed by page offset
    pub pages: BTreeMap<u64, Page>,
    /// File metadata
    pub metadata: FileMetadata,
}

/// File metadata
#[derive(Debug, Clone)]
pub struct FileMetadata {
    /// File mode (permissions)
    pub mode: u32,
    /// Last modified time
    pub mtime: std::time::SystemTime,
    /// Creation time
    pub ctime: std::time::SystemTime,
}

impl Default for FileMetadata {
    fn default() -> Self {
        let now = std::time::SystemTime::now();
        Self {
            mode: 0o644, // rw-r--r--
            mtime: now,
            ctime: now,
        }
    }
}

/// In-memory page cache for fast I/O
pub struct PageCache {
    /// All cached files
    files: HashMap<String, Arc<RwLock<FileCache>>>,
    /// Maximum cache size in bytes
    max_size: usize,
    /// Current cache size in bytes
    current_size: usize,
}

impl PageCache {
    /// Create a new page cache with the given maximum size
    pub fn new(max_size_mb: usize) -> Self {
        Self {
            files: HashMap::new(),
            max_size: max_size_mb * 1024 * 1024,
            current_size: 0,
        }
    }

    /// Get or create a file cache entry
    pub fn get_or_create_file(&mut self, path: &str) -> Arc<RwLock<FileCache>> {
        self.files
            .entry(path.to_string())
            .or_insert_with(|| {
                Arc::new(RwLock::new(FileCache {
                    path: path.to_string(),
                    size: 0,
                    pages: BTreeMap::new(),
                    metadata: FileMetadata::default(),
                }))
            })
            .clone()
    }

    /// Read data from cache
    pub async fn read(&self, path: &str, offset: u64, size: usize) -> Option<Bytes> {
        let file = self.files.get(path)?;
        let file = file.read().await;

        // Calculate which pages we need
        let start_page = offset / PAGE_SIZE as u64;
        let end_page = (offset + size as u64).div_ceil(PAGE_SIZE as u64);

        let mut result = Vec::new();
        let mut current_offset = offset;

        for page_idx in start_page..end_page {
            let page_offset = page_idx * PAGE_SIZE as u64;

            if let Some(page) = file.pages.get(&page_offset) {
                // Calculate slice within this page
                let page_start = if current_offset > page_offset {
                    (current_offset - page_offset) as usize
                } else {
                    0
                };

                let remaining = size - result.len();
                let page_end = (page_start + remaining).min(page.data.len());

                result.extend_from_slice(&page.data[page_start..page_end]);
                current_offset = page_offset + page_end as u64;

                if result.len() >= size {
                    break;
                }
            } else {
                // Page not in cache
                return None;
            }
        }

        if result.len() == size {
            Some(Bytes::from(result))
        } else {
            None
        }
    }

    /// Write data to cache
    pub async fn write(&mut self, path: &str, offset: u64, data: &[u8]) -> Result<(), String> {
        // Ensure we don't exceed cache size
        if self.current_size + data.len() > self.max_size {
            self.evict_pages(data.len()).await;
        }

        let file = self.get_or_create_file(path);
        let mut file = file.write().await;

        // Calculate which pages we need to write
        let start_page = offset / PAGE_SIZE as u64;
        let end_page = (offset + data.len() as u64).div_ceil(PAGE_SIZE as u64);

        let mut data_offset = 0;

        for page_idx in start_page..end_page {
            let page_offset = page_idx * PAGE_SIZE as u64;

            // Calculate slice within this page
            let page_start = if offset > page_offset {
                (offset - page_offset) as usize
            } else {
                0
            };

            let remaining = data.len() - data_offset;
            let page_end = (page_start + remaining).min(PAGE_SIZE);
            let chunk_size = page_end - page_start;

            // Check if page already exists to track size correctly
            let page_exists = file.pages.contains_key(&page_offset);

            // Get or create page
            let page = file.pages.entry(page_offset).or_insert_with(|| {
                let new_page = Page {
                    data: Bytes::from(vec![0u8; PAGE_SIZE]),
                    dirty: false,
                    last_access: std::time::Instant::now(),
                };
                if !page_exists {
                    self.current_size += PAGE_SIZE;
                }
                new_page
            });

            // Copy data into page
            let mut page_data = page.data.to_vec();
            page_data[page_start..page_end]
                .copy_from_slice(&data[data_offset..data_offset + chunk_size]);

            page.data = Bytes::from(page_data);
            page.dirty = true;
            page.last_access = std::time::Instant::now();

            data_offset += chunk_size;
        }

        // Update file size
        file.size = file.size.max(offset + data.len() as u64);
        file.metadata.mtime = std::time::SystemTime::now();

        Ok(())
    }

    /// Insert a page from object storage (for lazy loading)
    pub async fn insert_page(&mut self, path: &str, offset: u64, data: Bytes) {
        let file = self.get_or_create_file(path);
        let mut file = file.write().await;

        let page_offset = (offset / PAGE_SIZE as u64) * PAGE_SIZE as u64;
        let data_len = data.len();

        if let Some(old_page) = file.pages.insert(
            page_offset,
            Page {
                data,
                dirty: false,
                last_access: std::time::Instant::now(),
            },
        ) {
            self.current_size = self.current_size.saturating_sub(old_page.data.len());
        }

        self.current_size += data_len;
    }

    /// Get file metadata
    pub async fn get_metadata(&self, path: &str) -> Option<FileMetadata> {
        let file = self.files.get(path)?;
        let file = file.read().await;
        Some(file.metadata.clone())
    }

    /// Get file size
    pub async fn get_size(&self, path: &str) -> Option<u64> {
        let file = self.files.get(path)?;
        let file = file.read().await;
        Some(file.size)
    }

    /// List all files in cache
    pub fn list_files(&self) -> Vec<String> {
        self.files.keys().cloned().collect()
    }

    /// Truncate a file to the specified size
    pub async fn truncate(&mut self, path: &str, new_size: u64) -> Result<(), String> {
        let file = self.files.get(path).ok_or("File not found")?;
        let mut file = file.write().await;

        let old_size = file.size;
        file.size = new_size;
        file.metadata.mtime = std::time::SystemTime::now();

        if new_size < old_size {
            let new_last_page = new_size / PAGE_SIZE as u64;
            let pages_to_remove: Vec<u64> = file
                .pages
                .keys()
                .filter(|&&offset| offset / PAGE_SIZE as u64 > new_last_page)
                .copied()
                .collect();

            for offset in pages_to_remove {
                if let Some(page) = file.pages.remove(&offset) {
                    self.current_size = self.current_size.saturating_sub(page.data.len());
                }
            }

            if new_size % PAGE_SIZE as u64 != 0 {
                let last_page_offset = new_last_page * PAGE_SIZE as u64;
                if let Some(page) = file.pages.get_mut(&last_page_offset) {
                    let new_page_size = (new_size % PAGE_SIZE as u64) as usize;
                    let mut page_data = page.data.to_vec();
                    page_data.truncate(new_page_size);
                    page_data.resize(PAGE_SIZE, 0);
                    page.data = Bytes::from(page_data);
                    page.dirty = true;
                }
            }
        }

        Ok(())
    }

    /// Remove a file from cache
    pub async fn remove_file(&mut self, path: &str) -> Option<()> {
        if let Some(file_lock) = self.files.remove(path) {
            let file = file_lock.read().await;
            for page in file.pages.values() {
                self.current_size = self.current_size.saturating_sub(page.data.len());
            }
            Some(())
        } else {
            None
        }
    }

    /// Evict least recently used pages to make room
    async fn evict_pages(&mut self, needed: usize) {
        // Simple LRU eviction
        let mut pages_to_evict = Vec::new();

        for (path, file_lock) in &self.files {
            let file = file_lock.read().await;
            for (offset, page) in &file.pages {
                if !page.dirty {
                    pages_to_evict.push((path.clone(), *offset, page.last_access));
                }
            }
        }

        // Sort by last access time
        pages_to_evict.sort_by_key(|(_, _, last_access)| *last_access);

        // Evict oldest pages
        let mut freed = 0;
        for (path, offset, _) in pages_to_evict {
            if freed >= needed {
                break;
            }

            if let Some(file_lock) = self.files.get(&path) {
                let mut file = file_lock.write().await;
                if let Some(page) = file.pages.remove(&offset) {
                    freed += page.data.len();
                    self.current_size = self.current_size.saturating_sub(page.data.len());
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cache_write_read() {
        let mut cache = PageCache::new(10); // 10MB cache

        let data = b"Hello, World!";
        cache.write("/test.txt", 0, data).await.unwrap();

        let result = cache.read("/test.txt", 0, data.len()).await.unwrap();
        assert_eq!(&result[..], data);
    }

    #[tokio::test]
    async fn test_cache_large_file() {
        let mut cache = PageCache::new(10);

        // Write 128KB across multiple pages
        let data = vec![42u8; 128 * 1024];
        cache.write("/large.bin", 0, &data).await.unwrap();

        // Read it back
        let result = cache.read("/large.bin", 0, data.len()).await.unwrap();
        assert_eq!(result.len(), data.len());
        assert_eq!(&result[..], &data[..]);
    }

    #[tokio::test]
    async fn test_cache_offset_write() {
        let mut cache = PageCache::new(10);

        // Write at offset
        let data1 = b"Hello";
        let data2 = b"World";

        cache.write("/test.txt", 0, data1).await.unwrap();
        cache.write("/test.txt", 6, data2).await.unwrap();

        // Read entire file
        let result = cache.read("/test.txt", 0, 11).await.unwrap();
        assert_eq!(&result[..5], b"Hello");
        assert_eq!(&result[6..11], b"World");
    }
}
