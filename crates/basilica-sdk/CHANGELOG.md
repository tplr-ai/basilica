# Changelog

All notable changes to the basilica-sdk package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Secure-cloud GPU, CPU, and volume methods now call the V2 API paths under
  `/v2/secure-cloud/*`.
- `create_volume()` prints a temporary stderr warning when callers pass a
  legacy secure-cloud provider tag such as `hyperstack` or `verda`; V2 volume
  requests should use public availability-zone values.
- `create_distributed_deployment()` prints the same temporary stderr warning
  when `providerFilter.include` or `providerFilter.exclude` contains legacy
  secure-cloud provider tags.

## [0.31.2] - 2026-06-09

### Changed
- `GpuOffering.provider` is now a free-form `String` instead of the
  `CloudProvider` enum, and the `CloudProvider` enum has been removed. The API
  wire boundary emits availability-zone root codenames (e.g. `cyan`, `plum`,
  `opal`) in `provider`, with the region in the separate `region` field. Keeping
  this a `String` means new providers/AZs deserialize cleanly without an SDK
  release; the previous enum rejected unknown values with `unknown variant ...`,
  breaking every GPU listing call against a newer API.

## [0.29.0] - 2026-05-04

### Added
- Distributed-training endpoints — `create_distributed_deployment`
  and `scale_distributed_deployment` on `BasilicaClient`, with
  `CreateDistributedDeploymentRequest` and `DistributedSpec` on the
  write path and `DistributedStatus` on the read path.
- `image` and `distributed` fields on `DeploymentResponse`, mirroring
  the API wire shape for distributed deployments.
- `friendly_name` field on `DeploymentResponse`, `DeploymentSummary`,
  and `PublicDeploymentMetadataResponse`, exposing the user-supplied
  display name alongside the UUID `instance_name`.

## [0.28.0] - 2026-04-27

### Added
- Denvr Data cloud provider support (`CloudProvider::Denvr`).
  `GpuOffering` responses from `/secure-cloud/gpu-prices` that include
  Denvr offerings now deserialize successfully instead of failing
  the entire response.
- Card payment APIs on the payments client: create checkout session,
  paginated listing of card purchases, status filtering, and receipt
  + invoice metadata fields on each session.

### Security
- Bumped rustls-webpki to 0.103.13 to address RUSTSEC-2026-0104
  (reachable panic in CRL parsing).

## [0.27.0] - 2026-04-20

### Added
- Shadeform cloud provider support (`CloudProvider::Shadeform`).
  `GpuOffering` responses from `/secure-cloud/gpu-prices` that include
  Shadeform offerings now deserialize successfully instead of failing
  the entire response.

## [0.26.0] - 2026-04-18

### Added
- `name` field on `RentalResponse`, `RentalStatusWithSshResponse`,
  `ApiRentalListItem`, `HistoricalRentalItem`, and the Secure Cloud
  rental response and list-item variants.
- Optional `name` field on `StartRentalApiRequest`,
  `StartSecureCloudRentalRequest`, and `StartCpuRentalRequest`.
- `stop_rental` and `get_rental_status` accept either a rental name or a
  rental ID as the target.

### Changed
- `stop_rental` now returns `()` to match the backend's HTTP 204 No
  Content response.
- `RentalStatusWithSshResponse::from_validator_response()` now accepts the
  rental name as a parameter rather than defaulting to a placeholder
  that callers had to overwrite.

## [0.25.0] - 2026-03-05

### Added
- Mass Compute cloud provider support (CloudProvider::MassCompute)

## [0.24.0] - 2026-02-26

### Added
- restart_deployment method for triggering Kubernetes rolling restarts

## [0.23.0] - 2026-02-25

### Added
- GPU flavour preferences: GpuPriceQuery with region filter, GpuRequirementsSpec with interconnect/geo/spot/infiniband fields

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
