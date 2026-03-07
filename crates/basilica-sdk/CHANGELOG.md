# Changelog

All notable changes to the basilica-sdk package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.22.0] - 2026-02-23

### Changed
- Version bump for release alignment

## [0.20.2] - 2026-02-15

### Fixed
- CI formatting compliance for `GpuCategory` serde tests

## [0.20.1] - 2026-02-15

### Fixed
- `GpuCategory` now serializes as a plain string (e.g., `"RTX6000"`) instead of tagged enum format (`{"Other":"RTX6000"}`)
- Backward-compatible deserialization accepts both plain strings and legacy tagged format

## [0.20.0] - 2026-02-15

### Added
- `WebSocketConfig` type with `enabled` flag and `idle_timeout_seconds` (60-3600 range, default 1800s)
- `websocket` field on `CreateDeploymentRequest`, `DeploymentResponse`, and `DeploymentSummary`
- `Default` implementation for `WebSocketConfig` (enabled=true, 1800s idle timeout)

## [0.19.0] - 2026-02-12

### Added
- `enroll_metadata()` method for toggling public metadata enrollment (authenticated POST)
- `get_enrollment_status()` method for checking enrollment state (authenticated GET)
- `get_public_deployment_metadata()` method for unauthenticated public metadata lookup
- `public_metadata` field on `CreateDeploymentRequest`, `DeploymentResponse`, and `DeploymentSummary`
- `EnrollMetadataResponse` and `PublicDeploymentMetadataResponse` types
- `get_public()` helper for requests that skip authentication headers

## [0.17.0] - 2026-02-04
### Changed
- Replaced `DataCrunch` provider with `Verda` in CloudProvider enum
- Updated documentation to reflect Verda as replacement for DataCrunch

## [0.16.0] - 2026-02-02
### Added
- Health check support for AFINE deployments
