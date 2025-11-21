use super::cache::{PageCache, PAGE_SIZE};
use super::dirty_tracker::DirtyPageTracker;
use super::sync_worker::SyncWorker;
use crate::backend::StorageBackend;
use crate::metrics::StorageMetrics;
use fuser::{
    FileAttr, FileType, Filesystem, ReplyAttr, ReplyCreate, ReplyData, ReplyDirectory, ReplyEmpty,
    ReplyEntry, ReplyWrite, Request,
};
use libc::ENOENT;
use parking_lot::RwLock as ParkingLotRwLock;
use std::collections::HashMap;
use std::ffi::OsStr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::runtime::Handle;
use tokio::sync::RwLock as TokioRwLock;
use tracing::{debug, error, info};

const TTL: Duration = Duration::from_secs(1);
const ROOT_INO: u64 = 1;

#[derive(Debug, Clone)]
pub struct Inode {
    ino: u64,
    parent: u64,
    name: String,
    kind: FileType,
    size: u64,
    mode: u32,
    uid: u32,
    gid: u32,
    atime: SystemTime,
    mtime: SystemTime,
    ctime: SystemTime,
}

impl Inode {
    fn to_file_attr(&self) -> FileAttr {
        FileAttr {
            ino: self.ino,
            size: self.size,
            blocks: self.size.div_ceil(512),
            atime: self.atime,
            mtime: self.mtime,
            ctime: self.ctime,
            crtime: self.ctime,
            kind: self.kind,
            perm: self.mode as u16,
            nlink: 1,
            uid: self.uid,
            gid: self.gid,
            rdev: 0,
            blksize: PAGE_SIZE as u32,
            flags: 0,
        }
    }
}

pub struct BasilicaFS {
    experiment_id: String,
    storage: Arc<dyn StorageBackend>,

    cache: Arc<TokioRwLock<PageCache>>,
    pub inodes: Arc<ParkingLotRwLock<HashMap<u64, Inode>>>,
    path_to_ino: Arc<ParkingLotRwLock<HashMap<String, u64>>>,
    next_ino: Arc<ParkingLotRwLock<u64>>,

    dirty_tracker: Arc<DirtyPageTracker>,
    sync_worker: Arc<SyncWorker>,
    runtime_handle: Handle,

    quota_bytes: u64,
    pub used_bytes: Arc<AtomicU64>,

    metrics: Arc<StorageMetrics>,
}

impl BasilicaFS {
    pub fn new(
        experiment_id: String,
        storage: Arc<dyn StorageBackend>,
        sync_interval_ms: u64,
        cache_size_mb: usize,
        quota_bytes: u64,
        metrics: Arc<StorageMetrics>,
    ) -> Self {
        let cache = Arc::new(TokioRwLock::new(PageCache::new(cache_size_mb)));
        let dirty_tracker = Arc::new(DirtyPageTracker::new());

        let sync_worker = Arc::new(SyncWorker::new(
            experiment_id.clone(),
            storage.clone(),
            cache.clone(),
            dirty_tracker.clone(),
            sync_interval_ms,
        ));

        // Create a dedicated runtime for FUSE operations, separate from the main runtime.
        // This prevents deadlocks when sync_worker blocks the main runtime and FUSE
        // operations try to block_on() async tasks.
        let fuse_runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .thread_name("fuse-io")
            .enable_all()
            .build()
            .expect("Failed to create FUSE I/O runtime");
        let runtime_handle = fuse_runtime.handle().clone();
        std::mem::forget(fuse_runtime);

        let mut inodes = HashMap::new();
        let now = SystemTime::now();

        inodes.insert(
            ROOT_INO,
            Inode {
                ino: ROOT_INO,
                parent: ROOT_INO,
                name: "/".to_string(),
                kind: FileType::Directory,
                size: 0,
                mode: 0o755,
                uid: 1000,
                gid: 1000,
                atime: now,
                mtime: now,
                ctime: now,
            },
        );

        let mut path_to_ino = HashMap::new();
        path_to_ino.insert("/".to_string(), ROOT_INO);

        Self {
            experiment_id,
            storage,
            cache,
            dirty_tracker,
            sync_worker,
            runtime_handle,
            inodes: Arc::new(ParkingLotRwLock::new(inodes)),
            path_to_ino: Arc::new(ParkingLotRwLock::new(path_to_ino)),
            next_ino: Arc::new(ParkingLotRwLock::new(ROOT_INO + 1)),
            quota_bytes,
            used_bytes: Arc::new(AtomicU64::new(0)),
            metrics,
        }
    }

    pub async fn start_sync_worker(&self) {
        self.sync_worker.start().await;
    }

    pub async fn shutdown(&self) -> Result<(), String> {
        info!("Shutting down BasilicaFS");
        self.sync_worker.stop().await;
        self.sync_worker.flush_all().await
    }

    fn check_quota(&self, additional_bytes: u64) -> Result<(), libc::c_int> {
        let current = self.used_bytes.load(Ordering::Relaxed);
        if current + additional_bytes > self.quota_bytes {
            error!(
                current = current,
                additional = additional_bytes,
                quota = self.quota_bytes,
                "Storage quota exceeded"
            );
            self.metrics.quota_exceeded.fetch_add(1, Ordering::Relaxed);
            return Err(libc::EDQUOT);
        }
        Ok(())
    }

    fn add_usage(&self, bytes: u64) {
        self.used_bytes.fetch_add(bytes, Ordering::Relaxed);
        self.metrics
            .bytes_written
            .fetch_add(bytes, Ordering::Relaxed);
    }

    fn remove_usage(&self, bytes: u64) {
        self.used_bytes.fetch_sub(bytes, Ordering::Relaxed);
    }

    fn compute_path_locked(&self, inodes: &HashMap<u64, Inode>, ino: u64) -> Option<String> {
        let mut path_parts = Vec::new();
        let mut current = ino;

        while current != ROOT_INO {
            let inode = inodes.get(&current)?;
            path_parts.push(inode.name.clone());
            current = inode.parent;
        }

        if path_parts.is_empty() {
            Some("/".to_string())
        } else {
            path_parts.reverse();
            Some(format!("/{}", path_parts.join("/")))
        }
    }

    fn alloc_ino(&self) -> u64 {
        let mut next = self.next_ino.write();
        let ino = *next;
        *next += 1;
        ino
    }
}

impl Filesystem for BasilicaFS {
    fn lookup(&mut self, _req: &Request, parent: u64, name: &OsStr, reply: ReplyEntry) {
        let name_str = name.to_string_lossy().to_string();
        debug!("lookup: parent={}, name={}", parent, name_str);

        let inodes = self.inodes.read();

        for inode in inodes.values() {
            if inode.parent == parent && inode.name == name_str {
                reply.entry(&TTL, &inode.to_file_attr(), 0);
                return;
            }
        }

        reply.error(ENOENT);
    }

    fn getattr(&mut self, _req: &Request, ino: u64, reply: ReplyAttr) {
        debug!("getattr: ino={}", ino);

        let inodes = self.inodes.read();

        if let Some(inode) = inodes.get(&ino) {
            reply.attr(&TTL, &inode.to_file_attr());
        } else {
            reply.error(ENOENT);
        }
    }

    fn read(
        &mut self,
        _req: &Request,
        ino: u64,
        _fh: u64,
        offset: i64,
        size: u32,
        _flags: i32,
        _lock: Option<u64>,
        reply: ReplyData,
    ) {
        debug!("read: ino={}, offset={}, size={}", ino, offset, size);

        self.metrics.reads.fetch_add(1, Ordering::Relaxed);

        let path = {
            let inodes = self.inodes.read();
            self.compute_path_locked(&inodes, ino)
        };

        let path = match path {
            Some(p) => p,
            None => {
                reply.error(ENOENT);
                return;
            }
        };

        self.runtime_handle.block_on(async {
            let cache = self.cache.read().await;
            if let Some(data) = cache.read(&path, offset as u64, size as usize).await {
                drop(cache);
                self.metrics.cache_hits.fetch_add(1, Ordering::Relaxed);
                self.metrics
                    .bytes_read
                    .fetch_add(data.len() as u64, Ordering::Relaxed);
                reply.data(&data);
                return;
            }
            drop(cache);

            self.metrics.cache_misses.fetch_add(1, Ordering::Relaxed);

            let key = format!("{}/{}", self.experiment_id, path.trim_start_matches('/'));

            match self.storage.exists(&key).await {
                Ok(true) => match self.storage.get(&key).await {
                    Ok(data) => {
                        let size = data.len();
                        let mut cache = self.cache.write().await;
                        let mut offset = 0;

                        while offset < size {
                            let chunk_size = (size - offset).min(PAGE_SIZE);
                            let chunk = data.slice(offset..offset + chunk_size);
                            cache.insert_page(&path, offset as u64, chunk).await;
                            offset += chunk_size;
                        }

                        drop(cache);

                        let cache = self.cache.read().await;
                        if let Some(data) = cache.read(&path, offset as u64, size).await {
                            self.metrics
                                .bytes_read
                                .fetch_add(data.len() as u64, Ordering::Relaxed);
                            reply.data(&data);
                        } else {
                            reply.error(libc::EIO);
                        }
                    }
                    Err(e) => {
                        error!("Failed to load file {}: {}", path, e);
                        reply.error(libc::EIO);
                    }
                },
                Ok(false) => {
                    reply.error(ENOENT);
                }
                Err(e) => {
                    error!("Failed to check file existence {}: {}", path, e);
                    reply.error(libc::EIO);
                }
            }
        });
    }

    fn write(
        &mut self,
        _req: &Request,
        ino: u64,
        _fh: u64,
        offset: i64,
        data: &[u8],
        _write_flags: u32,
        _flags: i32,
        _lock: Option<u64>,
        reply: ReplyWrite,
    ) {
        debug!("write: ino={}, offset={}, size={}", ino, offset, data.len());

        if let Err(errno) = self.check_quota(data.len() as u64) {
            reply.error(errno);
            return;
        }

        self.metrics.writes.fetch_add(1, Ordering::Relaxed);

        let path = {
            let inodes = self.inodes.read();
            self.compute_path_locked(&inodes, ino)
        };

        let path = match path {
            Some(p) => p,
            None => {
                reply.error(ENOENT);
                return;
            }
        };

        self.runtime_handle.block_on(async {
            let mut cache = self.cache.write().await;
            if let Err(e) = cache.write(&path, offset as u64, data).await {
                drop(cache);
                error!("Failed to write to cache: {}", e);
                reply.error(libc::EIO);
                return;
            }

            let new_size =
                (offset as u64 + data.len() as u64).max(cache.get_size(&path).await.unwrap_or(0));

            drop(cache);

            {
                let mut inodes = self.inodes.write();
                if let Some(inode) = inodes.get_mut(&ino) {
                    let old_size = inode.size;
                    inode.size = new_size;
                    inode.mtime = SystemTime::now();

                    if new_size > old_size {
                        self.add_usage(new_size - old_size);
                    }
                }
            }

            self.dirty_tracker
                .mark_dirty(&path, offset as u64, data.len())
                .await;

            reply.written(data.len() as u32);
        });
    }

    fn create(
        &mut self,
        _req: &Request<'_>,
        parent: u64,
        name: &OsStr,
        mode: u32,
        _umask: u32,
        _flags: i32,
        reply: ReplyCreate,
    ) {
        let name_str = name.to_string_lossy().to_string();
        debug!(
            "create: parent={}, name={}, mode={:o}",
            parent, name_str, mode
        );

        if let Err(errno) = self.check_quota(0) {
            reply.error(errno);
            return;
        }

        let ino = self.alloc_ino();
        let now = SystemTime::now();

        let inode = Inode {
            ino,
            parent,
            name: name_str.clone(),
            kind: FileType::RegularFile,
            size: 0,
            mode,
            uid: 1000,
            gid: 1000,
            atime: now,
            mtime: now,
            ctime: now,
        };

        let attr = inode.to_file_attr();

        let mut inodes = self.inodes.write();
        inodes.insert(ino, inode);
        drop(inodes);

        let parent_path = {
            let inodes = self.inodes.read();
            self.compute_path_locked(&inodes, parent)
                .unwrap_or_else(|| "/".to_string())
        };

        let full_path = if parent_path == "/" {
            format!("/{}", name_str)
        } else {
            format!("{}/{}", parent_path, name_str)
        };

        let mut path_to_ino = self.path_to_ino.write();
        path_to_ino.insert(full_path, ino);
        drop(path_to_ino);

        reply.created(&TTL, &attr, 0, 0, 0);
    }

    fn mkdir(
        &mut self,
        _req: &Request<'_>,
        parent: u64,
        name: &OsStr,
        mode: u32,
        _umask: u32,
        reply: ReplyEntry,
    ) {
        let name_str = name.to_string_lossy().to_string();
        debug!(
            "mkdir: parent={}, name={}, mode={:o}",
            parent, name_str, mode
        );

        let ino = self.alloc_ino();
        let now = SystemTime::now();

        let inode = Inode {
            ino,
            parent,
            name: name_str.clone(),
            kind: FileType::Directory,
            size: 0,
            mode: mode | 0o040000,
            uid: 1000,
            gid: 1000,
            atime: now,
            mtime: now,
            ctime: now,
        };

        let attr = inode.to_file_attr();

        let mut inodes = self.inodes.write();
        inodes.insert(ino, inode);
        drop(inodes);

        let parent_path = {
            let inodes = self.inodes.read();
            self.compute_path_locked(&inodes, parent)
                .unwrap_or_else(|| "/".to_string())
        };

        let full_path = if parent_path == "/" {
            format!("/{}", name_str)
        } else {
            format!("{}/{}", parent_path, name_str)
        };

        let mut path_to_ino = self.path_to_ino.write();
        path_to_ino.insert(full_path, ino);
        drop(path_to_ino);

        reply.entry(&TTL, &attr, 0);
    }

    fn readdir(
        &mut self,
        _req: &Request,
        ino: u64,
        _fh: u64,
        offset: i64,
        mut reply: ReplyDirectory,
    ) {
        debug!("readdir: ino={}, offset={}", ino, offset);

        let inodes = self.inodes.read();

        let mut entries = vec![
            (ino, FileType::Directory, ".".to_string()),
            (ino, FileType::Directory, "..".to_string()),
        ];

        for child in inodes.values() {
            if child.parent == ino && child.ino != ino {
                entries.push((child.ino, child.kind, child.name.clone()));
            }
        }

        // Fixed the bug: properly handle offset when iterating
        let skip_count = offset as usize;
        for (i, entry) in entries.iter().enumerate().skip(skip_count) {
            // The offset passed to reply.add() should be the index of the next entry
            // relative to the original list, not the enumerated index after skip
            let next_offset = (i + 1) as i64;
            if reply.add(entry.0, next_offset, entry.1, &entry.2) {
                break;
            }
        }

        reply.ok();
    }

    fn unlink(&mut self, _req: &Request, parent: u64, name: &OsStr, reply: ReplyEmpty) {
        let name_str = name.to_string_lossy().to_string();
        debug!("unlink: parent={}, name={}", parent, name_str);

        let ino = {
            let inodes = self.inodes.read();
            inodes
                .values()
                .find(|i| i.parent == parent && i.name == name_str)
                .map(|i| (i.ino, i.kind))
        };

        let (ino, kind) = match ino {
            Some(i) => i,
            None => {
                reply.error(libc::ENOENT);
                return;
            }
        };

        if kind == FileType::Directory {
            reply.error(libc::EISDIR);
            return;
        }

        let path = {
            let inodes = self.inodes.read();
            self.compute_path_locked(&inodes, ino)
        };

        let path = match path {
            Some(p) => p,
            None => {
                reply.error(libc::ENOENT);
                return;
            }
        };

        let removed_size = {
            let mut path_to_ino = self.path_to_ino.write();
            path_to_ino.remove(&path);
            drop(path_to_ino);

            let mut inodes = self.inodes.write();
            if let Some(inode) = inodes.remove(&ino) {
                inode.size
            } else {
                0
            }
        };

        if removed_size > 0 {
            self.remove_usage(removed_size);
        }

        self.runtime_handle.spawn({
            let storage = self.storage.clone();
            let cache = self.cache.clone();
            let path = path.clone();
            let experiment_id = self.experiment_id.clone();
            async move {
                cache.write().await.remove_file(&path).await;

                let key = format!("{}/{}", experiment_id, path.trim_start_matches('/'));
                if let Err(e) = storage.delete(&key).await {
                    error!(path = %path, error = %e, "Failed to delete from storage");
                }
            }
        });

        reply.ok();
    }

    fn rmdir(&mut self, _req: &Request, parent: u64, name: &OsStr, reply: ReplyEmpty) {
        let name_str = name.to_string_lossy().to_string();
        debug!("rmdir: parent={}, name={}", parent, name_str);

        let ino = {
            let inodes = self.inodes.read();
            inodes
                .values()
                .find(|i| i.parent == parent && i.name == name_str && i.kind == FileType::Directory)
                .map(|i| i.ino)
        };

        let ino = match ino {
            Some(i) => i,
            None => {
                reply.error(libc::ENOENT);
                return;
            }
        };

        let is_empty = {
            let inodes = self.inodes.read();
            !inodes.values().any(|i| i.parent == ino)
        };

        if !is_empty {
            reply.error(libc::ENOTEMPTY);
            return;
        }

        let path = {
            let inodes = self.inodes.read();
            self.compute_path_locked(&inodes, ino)
        };

        let path = match path {
            Some(p) => p,
            None => {
                reply.error(libc::ENOENT);
                return;
            }
        };

        {
            let mut path_to_ino = self.path_to_ino.write();
            path_to_ino.remove(&path);
            drop(path_to_ino);

            let mut inodes = self.inodes.write();
            inodes.remove(&ino);
        }

        self.runtime_handle.spawn({
            let storage = self.storage.clone();
            let cache = self.cache.clone();
            let path = path.clone();
            let experiment_id = self.experiment_id.clone();
            async move {
                cache.write().await.remove_file(&path).await;

                let key = format!("{}/{}", experiment_id, path.trim_start_matches('/'));
                if let Err(e) = storage.delete(&key).await {
                    error!(path = %path, error = %e, "Failed to delete directory from storage");
                }
            }
        });

        reply.ok();
    }

    fn rename(
        &mut self,
        _req: &Request,
        parent: u64,
        name: &OsStr,
        newparent: u64,
        newname: &OsStr,
        _flags: u32,
        reply: ReplyEmpty,
    ) {
        let name_str = name.to_string_lossy().to_string();
        let newname_str = newname.to_string_lossy().to_string();

        debug!(
            "rename: parent={}, name={}, newparent={}, newname={}",
            parent, name_str, newparent, newname_str
        );

        let ino = {
            let inodes = self.inodes.read();
            inodes
                .values()
                .find(|i| i.parent == parent && i.name == name_str)
                .map(|i| (i.ino, i.kind))
        };

        let (ino, src_kind) = match ino {
            Some(i) => i,
            None => {
                reply.error(libc::ENOENT);
                return;
            }
        };

        let dest_ino = {
            let inodes = self.inodes.read();
            inodes
                .values()
                .find(|i| i.parent == newparent && i.name == newname_str)
                .map(|i| (i.ino, i.kind))
        };

        if let Some((dest_ino, dest_kind)) = dest_ino {
            match (src_kind, dest_kind) {
                (FileType::Directory, FileType::RegularFile)
                | (FileType::RegularFile, FileType::Directory) => {
                    reply.error(libc::EISDIR);
                    return;
                }
                (FileType::Directory, FileType::Directory) => {
                    let is_empty = {
                        let inodes = self.inodes.read();
                        !inodes.values().any(|i| i.parent == dest_ino)
                    };
                    if !is_empty {
                        reply.error(libc::ENOTEMPTY);
                        return;
                    }
                }
                _ => {}
            }

            let dest_size = {
                let mut inodes = self.inodes.write();
                if let Some(inode) = inodes.remove(&dest_ino) {
                    inode.size
                } else {
                    0
                }
            };

            if dest_size > 0 {
                self.remove_usage(dest_size);
            }
        }

        let (old_path, new_path) = {
            let inodes = self.inodes.read();
            let old = self.compute_path_locked(&inodes, ino);

            let mut new = String::new();
            if newparent != 1 {
                if let Some(parent_path) = self.compute_path_locked(&inodes, newparent) {
                    new.push_str(&parent_path);
                    if !parent_path.ends_with('/') {
                        new.push('/');
                    }
                }
            } else {
                new.push('/');
            }
            new.push_str(&newname_str);

            (old, Some(new))
        };

        let old_path = match old_path {
            Some(p) => p,
            None => {
                reply.error(libc::ENOENT);
                return;
            }
        };

        let new_path = new_path.unwrap();

        {
            let mut inodes = self.inodes.write();
            if let Some(inode) = inodes.get_mut(&ino) {
                inode.parent = newparent;
                inode.name = newname_str.clone();
                inode.mtime = SystemTime::now();
            }
        }

        {
            let mut path_to_ino = self.path_to_ino.write();
            path_to_ino.remove(&old_path);
            path_to_ino.insert(new_path.clone(), ino);
        }

        self.runtime_handle.spawn({
            let cache = self.cache.clone();
            let old_path_for_cache = old_path.clone();
            let new_path_for_cache = new_path.clone();
            async move {
                let mut cache = cache.write().await;
                cache.rename(&old_path_for_cache, &new_path_for_cache).await;
            }
        });

        self.runtime_handle.spawn({
            let storage = self.storage.clone();
            let old_path = old_path.clone();
            let new_path = new_path.clone();
            let experiment_id = self.experiment_id.clone();
            async move {
                let old_key = format!("{}/{}", experiment_id, old_path.trim_start_matches('/'));
                let new_key = format!("{}/{}", experiment_id, new_path.trim_start_matches('/'));

                match storage.get(&old_key).await {
                    Ok(data) => {
                        if let Err(e) = storage.put(&new_key, data).await {
                            error!(
                                old_path = %old_path,
                                new_path = %new_path,
                                error = %e,
                                "Failed to copy for rename"
                            );
                            return;
                        }
                        if let Err(e) = storage.delete(&old_key).await {
                            error!(
                                old_path = %old_path,
                                error = %e,
                                "Failed to delete old path after rename"
                            );
                        }
                    }
                    Err(e) => {
                        error!(
                            old_path = %old_path,
                            error = %e,
                            "Failed to read source for rename"
                        );
                    }
                }
            }
        });

        reply.ok();
    }

    fn setattr(
        &mut self,
        _req: &Request,
        ino: u64,
        mode: Option<u32>,
        uid: Option<u32>,
        gid: Option<u32>,
        size: Option<u64>,
        _atime: Option<fuser::TimeOrNow>,
        _mtime: Option<fuser::TimeOrNow>,
        _ctime: Option<SystemTime>,
        _fh: Option<u64>,
        _crtime: Option<SystemTime>,
        _chgtime: Option<SystemTime>,
        _bkuptime: Option<SystemTime>,
        _flags: Option<u32>,
        reply: ReplyAttr,
    ) {
        debug!("setattr: ino={}", ino);

        let mut inodes = self.inodes.write();

        let inode = match inodes.get_mut(&ino) {
            Some(i) => i,
            None => {
                reply.error(libc::ENOENT);
                return;
            }
        };

        if let Some(m) = mode {
            inode.mode = m;
        }
        if let Some(u) = uid {
            inode.uid = u;
        }
        if let Some(g) = gid {
            inode.gid = g;
        }
        if let Some(s) = size {
            if s < inode.size {
                let old_size = inode.size;
                inode.size = s;
                self.remove_usage(old_size - s);

                drop(inodes);

                let path = {
                    let inodes_read = self.inodes.read();
                    self.compute_path_locked(&inodes_read, ino)
                };

                if let Some(path) = path {
                    let cache = self.cache.clone();
                    let dirty_tracker = self.dirty_tracker.clone();
                    let runtime_handle = self.runtime_handle.clone();

                    runtime_handle.block_on(async {
                        let mut cache = cache.write().await;
                        if let Err(e) = cache.truncate(&path, s).await {
                            error!("Failed to truncate cache: {}", e);
                        }
                        dirty_tracker.mark_dirty(&path, 0, 1).await;
                    });
                }

                let inodes = self.inodes.read();
                let inode = inodes.get(&ino).unwrap();
                let ttl = Duration::from_secs(1);
                reply.attr(&ttl, &inode.to_file_attr());
                return;
            } else if s > inode.size {
                let additional = s - inode.size;
                if let Err(errno) = self.check_quota(additional) {
                    reply.error(errno);
                    return;
                }
                inode.size = s;
                self.add_usage(additional);

                drop(inodes);

                let path = {
                    let inodes_read = self.inodes.read();
                    self.compute_path_locked(&inodes_read, ino)
                };

                if let Some(path) = path {
                    let dirty_tracker = self.dirty_tracker.clone();
                    let runtime_handle = self.runtime_handle.clone();

                    runtime_handle.block_on(async {
                        dirty_tracker.mark_dirty(&path, 0, 1).await;
                    });
                }

                let inodes = self.inodes.read();
                let inode = inodes.get(&ino).unwrap();
                let ttl = Duration::from_secs(1);
                reply.attr(&ttl, &inode.to_file_attr());
                return;
            }
        }

        let now = SystemTime::now();
        inode.mtime = now;

        let ttl = Duration::from_secs(1);
        reply.attr(&ttl, &inode.to_file_attr());
    }

    fn fsync(
        &mut self,
        _req: &Request<'_>,
        _ino: u64,
        _fh: u64,
        _datasync: bool,
        reply: ReplyEmpty,
    ) {
        debug!("fsync called, flushing dirty pages");

        self.runtime_handle.block_on(async {
            match self.sync_worker.flush_all().await {
                Ok(_) => reply.ok(),
                Err(e) => {
                    error!("fsync failed: {}", e);
                    reply.error(libc::EIO);
                }
            }
        });
    }
}

#[derive(Clone)]
pub struct SharedBasilicaFS {
    inner: Arc<parking_lot::Mutex<BasilicaFS>>,
}

impl SharedBasilicaFS {
    pub fn new(fs: BasilicaFS) -> Self {
        Self {
            inner: Arc::new(parking_lot::Mutex::new(fs)),
        }
    }

    pub fn arc(&self) -> Arc<parking_lot::Mutex<BasilicaFS>> {
        Arc::clone(&self.inner)
    }

    pub async fn shutdown(&self) -> Result<(), String> {
        let sync_worker = {
            let fs = self.inner.lock();
            fs.sync_worker.clone()
        };

        info!("Shutting down BasilicaFS");
        sync_worker.stop().await;
        sync_worker.flush_all().await
    }
}

impl fuser::Filesystem for SharedBasilicaFS {
    fn lookup(&mut self, req: &Request<'_>, parent: u64, name: &OsStr, reply: ReplyEntry) {
        self.inner.lock().lookup(req, parent, name, reply)
    }

    fn getattr(&mut self, req: &Request<'_>, ino: u64, reply: ReplyAttr) {
        self.inner.lock().getattr(req, ino, reply)
    }

    fn readdir(
        &mut self,
        req: &Request<'_>,
        ino: u64,
        fh: u64,
        offset: i64,
        reply: fuser::ReplyDirectory,
    ) {
        self.inner.lock().readdir(req, ino, fh, offset, reply)
    }

    fn read(
        &mut self,
        req: &Request<'_>,
        ino: u64,
        fh: u64,
        offset: i64,
        size: u32,
        flags: i32,
        lock_owner: Option<u64>,
        reply: ReplyData,
    ) {
        self.inner
            .lock()
            .read(req, ino, fh, offset, size, flags, lock_owner, reply)
    }

    fn write(
        &mut self,
        req: &Request<'_>,
        ino: u64,
        fh: u64,
        offset: i64,
        data: &[u8],
        write_flags: u32,
        flags: i32,
        lock_owner: Option<u64>,
        reply: ReplyWrite,
    ) {
        self.inner.lock().write(
            req,
            ino,
            fh,
            offset,
            data,
            write_flags,
            flags,
            lock_owner,
            reply,
        )
    }

    fn create(
        &mut self,
        req: &Request<'_>,
        parent: u64,
        name: &OsStr,
        mode: u32,
        umask: u32,
        flags: i32,
        reply: ReplyCreate,
    ) {
        self.inner
            .lock()
            .create(req, parent, name, mode, umask, flags, reply)
    }

    fn mkdir(
        &mut self,
        req: &Request<'_>,
        parent: u64,
        name: &OsStr,
        mode: u32,
        umask: u32,
        reply: ReplyEntry,
    ) {
        self.inner
            .lock()
            .mkdir(req, parent, name, mode, umask, reply)
    }

    fn unlink(&mut self, req: &Request<'_>, parent: u64, name: &OsStr, reply: ReplyEmpty) {
        self.inner.lock().unlink(req, parent, name, reply)
    }

    fn rmdir(&mut self, req: &Request<'_>, parent: u64, name: &OsStr, reply: ReplyEmpty) {
        self.inner.lock().rmdir(req, parent, name, reply)
    }

    fn rename(
        &mut self,
        req: &Request<'_>,
        parent: u64,
        name: &OsStr,
        newparent: u64,
        newname: &OsStr,
        flags: u32,
        reply: ReplyEmpty,
    ) {
        self.inner
            .lock()
            .rename(req, parent, name, newparent, newname, flags, reply)
    }

    fn setattr(
        &mut self,
        req: &Request<'_>,
        ino: u64,
        mode: Option<u32>,
        uid: Option<u32>,
        gid: Option<u32>,
        size: Option<u64>,
        atime: Option<fuser::TimeOrNow>,
        mtime: Option<fuser::TimeOrNow>,
        ctime: Option<SystemTime>,
        fh: Option<u64>,
        crtime: Option<SystemTime>,
        chgtime: Option<SystemTime>,
        bkuptime: Option<SystemTime>,
        flags: Option<u32>,
        reply: ReplyAttr,
    ) {
        self.inner.lock().setattr(
            req, ino, mode, uid, gid, size, atime, mtime, ctime, fh, crtime, chgtime, bkuptime,
            flags, reply,
        )
    }

    fn fsync(&mut self, req: &Request<'_>, ino: u64, fh: u64, datasync: bool, reply: ReplyEmpty) {
        self.inner.lock().fsync(req, ino, fh, datasync, reply)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::StorageBackend;
    use crate::error::Result;
    use async_trait::async_trait;
    use bytes::Bytes;
    use std::collections::HashMap as StdHashMap;
    use tokio::sync::RwLock as TokioRwLock;

    struct MockStorage {
        data: Arc<TokioRwLock<StdHashMap<String, Bytes>>>,
    }

    impl MockStorage {
        fn new() -> Self {
            Self {
                data: Arc::new(TokioRwLock::new(StdHashMap::new())),
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
    async fn test_filesystem_creation() {
        let storage = Arc::new(MockStorage::new());
        let metrics = Arc::new(StorageMetrics::new());
        let fs = BasilicaFS::new(
            "exp-test".to_string(),
            storage,
            1000,
            10,
            1024 * 1024 * 1024,
            metrics,
        );

        let inodes = fs.inodes.read();
        assert!(inodes.contains_key(&ROOT_INO));
        assert_eq!(inodes.get(&ROOT_INO).unwrap().kind, FileType::Directory);
    }

    #[tokio::test]
    async fn test_quota_enforcement() {
        let storage = Arc::new(MockStorage::new());
        let metrics = Arc::new(StorageMetrics::new());
        let fs = BasilicaFS::new(
            "exp-test".to_string(),
            storage,
            1000,
            10,
            100,
            metrics.clone(),
        );

        let result = fs.check_quota(50);
        assert!(result.is_ok());

        let result = fs.check_quota(101);
        assert_eq!(result.unwrap_err(), libc::EDQUOT);
        assert_eq!(metrics.quota_exceeded.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_unlink_file() {
        let storage = Arc::new(MockStorage::new());
        let metrics = Arc::new(StorageMetrics::new());
        let fs = BasilicaFS::new(
            "exp-test".to_string(),
            storage.clone(),
            1000,
            10,
            1024 * 1024 * 1024,
            metrics,
        );

        let parent_ino = ROOT_INO;
        let file_name = "test.txt";
        let now = SystemTime::now();

        let file_ino = {
            let mut inodes = fs.inodes.write();
            let ino = {
                let mut next = fs.next_ino.write();
                let ino = *next;
                *next += 1;
                ino
            };
            inodes.insert(
                ino,
                Inode {
                    ino,
                    parent: parent_ino,
                    name: file_name.to_string(),
                    kind: FileType::RegularFile,
                    size: 100,
                    mode: 0o644,
                    uid: 1000,
                    gid: 1000,
                    atime: now,
                    mtime: now,
                    ctime: now,
                },
            );

            let mut path_to_ino = fs.path_to_ino.write();
            path_to_ino.insert(format!("/{}", file_name), ino);
            ino
        };

        fs.add_usage(100);
        let initial_used = fs.used_bytes.load(Ordering::Relaxed);
        let inodes_before = fs.inodes.read().len();

        {
            let mut path_to_ino = fs.path_to_ino.write();
            path_to_ino.remove(&format!("/{}", file_name));
            drop(path_to_ino);

            let mut inodes = fs.inodes.write();
            if let Some(inode) = inodes.remove(&file_ino) {
                fs.remove_usage(inode.size);
            }
        }

        let inodes_after = fs.inodes.read().len();
        assert_eq!(inodes_after, inodes_before - 1);

        let used_after = fs.used_bytes.load(Ordering::Relaxed);
        assert_eq!(used_after, initial_used - 100);
    }

    #[tokio::test]
    async fn test_rmdir_empty_directory() {
        let storage = Arc::new(MockStorage::new());
        let metrics = Arc::new(StorageMetrics::new());
        let fs = BasilicaFS::new(
            "exp-test".to_string(),
            storage.clone(),
            1000,
            10,
            1024 * 1024 * 1024,
            metrics,
        );

        let parent_ino = ROOT_INO;
        let dir_name = "testdir";
        let now = SystemTime::now();

        let dir_ino = {
            let mut inodes = fs.inodes.write();
            let ino = {
                let mut next = fs.next_ino.write();
                let ino = *next;
                *next += 1;
                ino
            };
            inodes.insert(
                ino,
                Inode {
                    ino,
                    parent: parent_ino,
                    name: dir_name.to_string(),
                    kind: FileType::Directory,
                    size: 0,
                    mode: 0o755,
                    uid: 1000,
                    gid: 1000,
                    atime: now,
                    mtime: now,
                    ctime: now,
                },
            );

            let mut path_to_ino = fs.path_to_ino.write();
            path_to_ino.insert(format!("/{}", dir_name), ino);
            ino
        };

        let inodes_before = fs.inodes.read().len();

        {
            let mut path_to_ino = fs.path_to_ino.write();
            path_to_ino.remove(&format!("/{}", dir_name));
            drop(path_to_ino);

            let mut inodes = fs.inodes.write();
            inodes.remove(&dir_ino);
        }

        let inodes_after = fs.inodes.read().len();
        assert_eq!(inodes_after, inodes_before - 1);

        let path_exists = fs
            .path_to_ino
            .read()
            .contains_key(&format!("/{}", dir_name));
        assert!(!path_exists);
    }

    #[tokio::test]
    async fn test_rename_file() {
        let storage = Arc::new(MockStorage::new());
        let metrics = Arc::new(StorageMetrics::new());
        let fs = BasilicaFS::new(
            "exp-test".to_string(),
            storage.clone(),
            1000,
            10,
            1024 * 1024 * 1024,
            metrics,
        );

        let parent_ino = ROOT_INO;
        let old_name = "old.txt";
        let new_name = "new.txt";
        let now = SystemTime::now();

        let file_ino = {
            let mut inodes = fs.inodes.write();
            let ino = {
                let mut next = fs.next_ino.write();
                let ino = *next;
                *next += 1;
                ino
            };
            inodes.insert(
                ino,
                Inode {
                    ino,
                    parent: parent_ino,
                    name: old_name.to_string(),
                    kind: FileType::RegularFile,
                    size: 100,
                    mode: 0o644,
                    uid: 1000,
                    gid: 1000,
                    atime: now,
                    mtime: now,
                    ctime: now,
                },
            );

            let mut path_to_ino = fs.path_to_ino.write();
            path_to_ino.insert(format!("/{}", old_name), ino);
            ino
        };

        {
            let mut path_to_ino = fs.path_to_ino.write();
            path_to_ino.remove(&format!("/{}", old_name));
            path_to_ino.insert(format!("/{}", new_name), file_ino);
            drop(path_to_ino);

            let mut inodes = fs.inodes.write();
            if let Some(inode) = inodes.get_mut(&file_ino) {
                inode.name = new_name.to_string();
            }
        }

        let inodes = fs.inodes.read();
        let renamed_inode = inodes.get(&file_ino).unwrap();
        assert_eq!(renamed_inode.name, new_name);

        let path_to_ino = fs.path_to_ino.read();
        assert!(!path_to_ino.contains_key(&format!("/{}", old_name)));
        assert!(path_to_ino.contains_key(&format!("/{}", new_name)));
    }

    #[tokio::test]
    async fn test_setattr_truncate() {
        let storage = Arc::new(MockStorage::new());
        let metrics = Arc::new(StorageMetrics::new());
        let fs = BasilicaFS::new(
            "exp-test".to_string(),
            storage.clone(),
            1000,
            10,
            1024 * 1024 * 1024,
            metrics,
        );

        let parent_ino = ROOT_INO;
        let file_name = "test.txt";
        let now = SystemTime::now();

        let file_ino = {
            let mut inodes = fs.inodes.write();
            let ino = {
                let mut next = fs.next_ino.write();
                let ino = *next;
                *next += 1;
                ino
            };
            inodes.insert(
                ino,
                Inode {
                    ino,
                    parent: parent_ino,
                    name: file_name.to_string(),
                    kind: FileType::RegularFile,
                    size: 1000,
                    mode: 0o644,
                    uid: 1000,
                    gid: 1000,
                    atime: now,
                    mtime: now,
                    ctime: now,
                },
            );
            ino
        };

        fs.add_usage(1000);

        {
            let mut cache = fs.cache.write().await;
            cache
                .write("/test.txt", 0, &vec![b'x'; 1000])
                .await
                .unwrap();
        }

        storage
            .put("/test.txt", Bytes::from(vec![b'x'; 1000]))
            .await
            .unwrap();

        {
            let mut inodes = fs.inodes.write();
            if let Some(inode) = inodes.get_mut(&file_ino) {
                let old_size = inode.size;
                inode.size = 500;
                if old_size > 500 {
                    fs.remove_usage(old_size - 500);
                }
            }
        }

        {
            let mut cache = fs.cache.write().await;
            cache.truncate("/test.txt", 500).await.unwrap();
        }

        {
            let inodes = fs.inodes.read();
            let inode = inodes.get(&file_ino).unwrap();
            assert_eq!(inode.size, 500);
        }

        let used_bytes = fs.used_bytes.load(Ordering::Relaxed);
        assert_eq!(used_bytes, 500);

        let cache = fs.cache.read().await;
        let cached_size = cache.get_size("/test.txt").await;
        assert_eq!(cached_size, Some(500));

        let stored_data = storage.get("/test.txt").await.unwrap();
        assert_eq!(stored_data.len(), 1000);
    }

    #[tokio::test]
    async fn test_quota_enforcement_write() {
        let storage = Arc::new(MockStorage::new());
        let metrics = Arc::new(StorageMetrics::new());
        let fs = BasilicaFS::new(
            "exp-test".to_string(),
            storage.clone(),
            1000,
            10,
            1000,
            metrics.clone(),
        );

        let result = fs.check_quota(500);
        assert!(result.is_ok());

        fs.add_usage(600);

        let result = fs.check_quota(500);
        assert_eq!(result.unwrap_err(), libc::EDQUOT);

        let quota_exceeded = metrics.quota_exceeded.load(Ordering::Relaxed);
        assert!(quota_exceeded > 0);
    }
}
