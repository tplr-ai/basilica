# basilica-storage

Storage daemon for Basilica GPU workloads with R2/S3 and FUSE support.

[![Crates.io](https://img.shields.io/crates/v/basilica-storage.svg)](https://crates.io/crates/basilica-storage)
[![Documentation](https://docs.rs/basilica-storage/badge.svg)](https://docs.rs/basilica-storage)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Documentation](https://docs.rs/basilica-storage) | [Repository](https://github.com/one-covenant/basilica) | [Website](https://basilica.ai)

## Overview

`basilica-storage` provides persistent storage for GPU workloads running on Basilica. It supports object storage (S3/R2 compatible) and can mount storage as a FUSE filesystem for transparent access.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
basilica-storage = "0.1"
```

## Features

- **Object Storage**: S3/R2 compatible storage backend
- **FUSE Filesystem**: Mount object storage as a local filesystem
- **Namespace Isolation**: Per-user/per-workload storage isolation
- **Quota Management**: Storage quotas and rate limiting
- **Kubernetes Integration**: Auto-credential fetching from K8s secrets

## Feature Flags

- `fuse` - Enable FUSE filesystem support (default, requires `libfuse3-dev`)

## Architecture

```
┌─────────────────┐
│   GPU Workload  │
│                 │
│  /data (mount)  │
└────────┬────────┘
         │ FUSE
         ▼
┌─────────────────────────────┐
│    Storage Daemon           │
│                             │
│  ┌─────────────────────┐   │
│  │   FUSE Handler      │   │
│  └──────────┬──────────┘   │
│             │               │
│  ┌──────────▼──────────┐   │
│  │   Object Cache      │   │
│  └──────────┬──────────┘   │
│             │               │
│  ┌──────────▼──────────┐   │
│  │   S3/R2 Client      │   │
│  └─────────────────────┘   │
└─────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Object Storage │
│   (S3 / R2)     │
└─────────────────┘
```

## Example

```rust
use basilica_storage::{StorageClient, StorageConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure storage client
    let config = StorageConfig {
        endpoint: "https://your-bucket.r2.cloudflarestorage.com".to_string(),
        access_key_id: std::env::var("R2_ACCESS_KEY")?,
        secret_access_key: std::env::var("R2_SECRET_KEY")?,
        bucket: "basilica-data".to_string(),
    };
    
    let client = StorageClient::new(config).await?;
    
    // Upload a file
    client.put_object("models/checkpoint.pt", &data).await?;
    
    // Download a file
    let data = client.get_object("models/checkpoint.pt").await?;
    
    Ok(())
}
```

## FUSE Daemon

Run the storage daemon to mount object storage:

```bash
# Mount storage to /mnt/basilica
basilica-storage-daemon \
  --mount-point /mnt/basilica \
  --namespace user-123 \
  --bucket basilica-data
```

## Configuration

```toml
[storage]
endpoint = "https://your-bucket.r2.cloudflarestorage.com"
bucket = "basilica-data"
region = "auto"

[cache]
enabled = true
max_size_mb = 1024
eviction_policy = "lru"

[fuse]
mount_point = "/mnt/basilica"
allow_other = true
auto_unmount = true

[quota]
max_size_gb = 100
rate_limit_mbps = 100
```

## Kubernetes Integration

The daemon can fetch credentials from Kubernetes secrets:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: basilica-storage-credentials
  namespace: basilica-system
type: Opaque
data:
  access_key_id: <base64>
  secret_access_key: <base64>
```

## Related Crates

- [`basilica-operator`](https://crates.io/crates/basilica-operator) - K8s operator
- [`basilica-common`](https://crates.io/crates/basilica-common) - Core types

## System Requirements

For FUSE support:
- Linux with FUSE 3.x
- `libfuse3-dev` package installed

## License

MIT License - see [LICENSE](LICENSE) for details.

