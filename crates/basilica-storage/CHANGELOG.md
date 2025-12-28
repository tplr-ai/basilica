# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial crates.io release preparation
- Comprehensive documentation and examples

## [0.1.0] - 2025-01-01

### Added
- Storage daemon for GPU workloads
- S3/R2 compatible object storage backend
- FUSE filesystem for transparent access
- Per-namespace storage isolation
- Quota management and enforcement
- Rate limiting for bandwidth
- Local caching layer
- Kubernetes credential integration
- CLI for storage operations

### Features
- `fuse` - Enable FUSE filesystem support (default, requires libfuse3-dev)

[Unreleased]: https://github.com/one-covenant/basilica/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/one-covenant/basilica/releases/tag/v0.1.0

