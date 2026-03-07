# Changelog

All notable changes to the basilica-cli package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.25.0] - 2026-03-05

### Added
- Mass Compute cloud provider support (CloudProvider::MassCompute)

## [0.24.0] - 2026-02-26

### Added
- `deploy restart` subcommand for triggering rolling restarts

## [0.23.0] - 2026-02-25

### Added
- GPU flavour preferences: --interconnect, --region, --spot, --exclude-spot flags on gpu list, gpu up, and deploy commands

## [0.22.0] - 2026-02-23

### Changed
- Version bump for release alignment

## [0.21.0] - 2026-02-17

### Changed
- Refactor the UX for Bourse compute

## [0.20.2] - 2026-02-15

### Fixed
- CI formatting compliance for `GpuCategory` serde tests

## [0.20.1] - 2026-02-15

### Fixed
- GPU types other than A100/H100/B200 no longer display as "OTHER" in listings

## [0.20.0] - 2026-02-15

### Added
- `--websocket` flag for `deploy` command to enable WebSocket support for long-lived connections
- `--ws-idle-timeout` flag for configuring WebSocket idle timeout (60-3600 seconds, default 1800)
- WebSocket enabled by default for `basilica summon openclaw` template
- Accept all GPU types through secure cloud pipeline (`basilica up rtx6000`)

## [0.19.0] - 2026-02-12

### Added
- `basilica deploy enroll-metadata` subcommand for toggling public metadata enrollment
- `basilica deploy metadata` subcommand for unauthenticated public metadata lookup
- `--public-metadata` flag for `deploy` command to enable enrollment at creation time
- "Verified" column in deployment list table showing enrollment status
- Public metadata line in deployment detail view

## [0.18.4] - 2026-02-11

### Fixed
- `bs upgrade` no longer fails when run from a working directory different from the binary's location
- Deployment list table now uses `Style::modern()` borders, properly cased column headers, formatted timestamps, and full URLs to match `ls`/`ps` table styling

## [0.18.3] - 2026-02-11

### Fixed
- Update notifications no longer trigger incorrectly during CI releases or when already on the latest version

## [0.18.2] - 2026-02-11

### Added
- `basilica summon tau` deployment template

## [0.18.1] - 2026-02-10

### Changed
- Community cloud rentals now use GPU-based bid matching instead of targeting specific node IDs.

### Fixed
- Credits formatting edge case where values could display with inconsistent decimal precision.

## [0.18.0] - 2026-02-08

### Fixed
- Fixed `bs upgrade` breaking the `bs` shorthand

## [0.17.0] - 2026-02-04

### Changed
- Replaced `DataCrunch` provider with `Verda` in CloudProvider enum

## [0.16.0] - 2026-02-02

### Added
- `basilica summon openclaw` deployment template with provider presets, model autodetect, and improved readiness output.

## [0.15.0]

### Added
- Share token management for private deployments (`share-token regenerate`, `share-token status`, `share-token revoke`)
- `--private` flag for deploy command (deployments are public by default)
- Access column (Public/Token) in deployments list table
- `--show-token` flag for deploy status command
- Spot instance indicator in GPU listings, rental selectors, and status views

## [0.14.0]

### Added
- Topology spread options for deploy command (`--spread-mode`, `--max-skew`, `--topology-key`)

## [0.13.0]

### Changed
- Renamed product offerings in user-facing text (API and SDK unchanged):
  - "Secure Cloud" → "The Citadel"
  - "Community Cloud" → "The Bourse"
  - "VIP" → "The Priory"
  - "Deployments" → "Summons"
- Added `summon` as visible alias for `deploy` command
- CLI `--compute` option now uses `citadel` and `bourse` as primary values (old names remain as aliases for backward compatibility)

### Fixed
- Display of logs in `deploy logs`

## [0.12.0]

### Added
- Volume management commands for persistent storage across rentals
  - `volumes create` - Create new volumes with configurable size and region
  - `volumes list` - List all volumes with status and attachment info
  - `volumes delete` - Delete volumes that are not attached
  - `volumes attach` - Attach a volume to an active rental
  - `volumes detach` - Detach a volume from a rental
- Show IP address in interactive rental selection for use to be able to differentiate between rentals

## [0.11.0]

### Added
- GPU interconnect type display in offering selection prompt (e.g., `8x H100 (SXM5)` instead of just `8x H100`)
- CPU-only machine rental support across all commands (`ls`, `up`, `ps`, `status`, `ssh`)

### Fixed
- Price-max filter now correctly applies to total community node cost

### Removed
- SSH access is now always enabled; removed no-ssh flag for simplified rental flow

## [0.10.1]

### Added
- Support for `vip` rentals

## [0.10.0]

### Added
- `basilica deploy vllm` command for deploying vLLM inference servers
  - Configurable tensor parallelism, dtype, and quantization options
  - Automatic GPU requirement detection based on model names
  - HuggingFace model cache storage configuration
- `basilica deploy sglang` command for deploying SGLang inference servers
  - Configurable context length and memory fraction settings
  - Same model-based GPU detection as vLLM
- SSH key ID is now displayed in `ssh-keys list` output for easier key management

### Changed
- Default GPU for model sizing recommendations updated from RTX A4000 (16GB) to A100 (40GB)
- GPU recommendations now use canonical A100/H100 model names

## [0.9.0]

### Added
- `bs` command alias as a shorthand for `basilica` (automatically created during `upgrade`)

### Changed
- SSH key discovery is now automatic - removed `ssh.key_path` and `ssh.private_key_path` config options
- SSH public key is stored directly on rentals, allowing SSH access even after deleting the original key from your account

### Fixed
- Improved SSH retry logic and error messages with clearer retry guidance

## [0.8.0]

### Fixed
- Fixed macOS build image in CI pipeline

## [0.7.0]

### Added
- New `deploy` command for Python script deployment to Basilica cloud
  - Subcommands: `create`, `delete`, `list`, `logs`, `status`
  - Comprehensive option groups: naming, resources, GPU, storage, health checks, networking, lifecycle
  - Source file packaging with automatic validation
- FIDO/U2F security key support for SSH authentication

### Fixed
- Deployment tracking now uses API-returned instance_name for accurate status monitoring

## [0.6.0]

### Added
- Secure cloud integration: rent GPUs from secure cloud providers alongside community cloud
- New `restart` command to restart rental containers
- Automatic SSH private key detection for rentals
- SSH keys are automatically registered during `basilica up`

### Changed
- Simplified `balance` command output to show single balance value instead of separate "available" and "total" fields

### Fixed
- Fixed duplicate "Fetching available GPUs..." spinner in `basilica ls`
- Added validation for community-cloud-only options (`--container-image`, `--ports`, `--env`) in `up` command

### Removed
- Removed `--compact` and `--detailed` view flags for now
- Removed `validator` and `miner` subcommands

## [0.6.0-alpha.2]

### Changed
- Simplified `balance` command output to show single balance value instead of separate "available" and "total" fields

### Fixed
- Fixed duplicate "Fetching available GPUs..." spinner in `basilica ls`

## [0.6.0-alpha.1]

### Added
- Secure cloud integration: rent GPUs from secure cloud providers alongside community cloud
- New `restart` command to restart rental containers
- Automatic SSH private key detection for rentals
- SSH keys are automatically registered during `basilica up`

## [0.5.5]

### Added
- New `upgrade` command for automatic CLI updates

### Fixed
- Fixed issue where rental price was not displayed correctly in the `basilica ps` command.

## [0.5.4]

### Changed
- Removed reserved balance from `basilica balance` command output to align with pay-as-you-go billing model
- Rental pricing changed from fixed to dynamic model.

## [0.5.3]

### Fixed
- Use musl instead of gnu build to avoid issues with glibc version in older distributions.

## [0.5.2]

### Added
- `basilica ps --history` now lists completed rentals with per-rental totals and an overall spend summary to simplify billing reviews.
- (Debug builds) `basilica packages` exposes the raw billing package feed for troubleshooting pricing mismatches.

### Changed
- `basilica ls`/`ps` consume live billing package data so hourly USD pricing shows up directly in the tables, including recalculated totals for multi-GPU nodes.
- Balance and rental cost displays now share the same credit formatter, keeping dollar figures aligned to two decimal places in every view.
- Table output for rentals highlights hourly rate, accumulated cost, and durations inline, removing the need to cross-reference separate commands.

### Removed
- Deprecated `basilica price` and `basilica usage` subcommands; their workflows are now part of the enhanced `ls`/`ps` experience.

## [0.5.1]

### Added
- Account balance management via `basilica balance` to inspect available and reserved compute credits
- TAO deposit funding with per-user wallet addresses and automatic credit conversion after 12 confirmations
  - Full deposit history available through `basilica fund list`
- GPU pricing calculator with `basilica price` showing real-time hourly rates across all GPU types
  - Adds affordability estimates based on your current balance to project run-time hours
- Usage tracking and cost monitoring via `basilica usage` for active and completed rentals
  - Displays resource consumption metrics and accumulated costs across rentals

- Account Balance Management - Check your available and reserved compute credits with the basilica balance command

- TAO Deposit Funding - Fund your account by depositing TAO to a unique (per user) wallet, with automatic credit conversion after 12 confirmations and full deposit history tracking via `basilica fund list`

- GPU Pricing Calculator - View real-time hourly rates for all GPU types with `basilica price`, including affordability calculations that show how many hours you can run based on your current balance.

- Usage Tracking & Cost Monitoring - Track rental costs and resource consumption with `basilica usage`, displaying detailed metrics and accumulated costs for all active and completed rentals

## [0.5.0] - skipped release

## [0.4.1]

### Added
- Port mappings display in rental listings (`ps` command)
  - New "Ports (Host → Container)" column showing mapped ports (e.g., "8080→80")
  - Full port list displayed when using `--detailed` flag
- Port mappings in `status` command output with clear "Host → Container" format
- Enhanced SSH connection instructions in `status` output with both CLI and standard SSH command examples

### Fixed
- Removed duplicate port entries in rental display

## [0.4.0]

### Changed
- All CLI output messages now use "node" terminology instead of "executor"
  - List output shows "available nodes" instead of "available executors"
  - Error messages refer to "nodes" instead of "executors"
- The `--detailed` flag help text now references "node IDs" instead of "executor IDs"

### Removed
- Removed `basilica executor` command (previously `basilica-executor` binary) as this component is no longer part of the architecture

## [0.3.5]

### Fixed
- Fixed `--gpu-count` flag to properly filter GPU configurations in all selection modes (default, compact, and detailed)
  - Interactive selector now shows only nodes/configurations with the exact GPU count specified
  - Previously showed all nodes with the minimum count or more

## [0.3.4]

### Added
- Three-tier display system for `ls`, `up`, and `ps` commands with flexible output control:
  - `--detailed` flag shows internal IDs (node IDs in `ls`/`up`, rental IDs in `ps`) for debugging
  - `--compact` flag provides minimal grouped display for cleaner overview
  - Default mode shows essential information without internal IDs
- Enhanced interactive selector with improved GPU information display in detailed mode

### Changed
- Renamed `--gpu-min` to `--gpu-count` in the `up` command for clarity
- GPU selection now uses exact count matching instead of minimum matching when provisioning instances
  - `basilica up a100 --gpu-count 2` now gets nodes with exactly 2 GPUs, not "at least 2"

### Fixed
- Compact mode GPU selections (e.g., selecting "1x A100") now correctly filter for nodes with exactly that GPU count, not nodes with more GPUs

## [0.3.3]

### Added
- New `--all` flag for `down` command to stop all active rentals at once

## [0.3.2]

### Changed
- Fixed some default values, this shouldn't have any affect on users

## [0.3.1]

### Added
- New `tokens` command for API token management:
  - `tokens create` - Create a new API token with optional name and scopes
  - `tokens list` - List all API tokens
  - `tokens revoke` - Revoke the current API token
  - Basilica API now supports authentication via API tokens generated via this in addition to JWT

### Removed
- Deprecated `export-token` command (replaced by the new `tokens` subcommands)

## [0.3.0]

### Added
- Country-based filtering with `--country` flag for both `ls` and `up` commands (e.g., `--country US`)
- Hardware profile display with CPU model, cores, and RAM in node details
- Enhanced rental list (`rentals` command) with CPU specs, RAM, and location information in detailed view
- Network speed information display for rentals and nodes

### Changed
- GPU type filtering now uses direct parameter instead of `--gpu-type` flag (e.g., `basilica ls h100` instead of `basilica ls --gpu-type h100`)
- Simplified GPU display by removing memory information - now shows count and type only (e.g., "2x H100" instead of "2x H100 (80GB)")
- Node list now groups by country with full country names instead of codes
- Improved rental list display with compact (default) and detailed (`--detailed`) views
- Location display now uses country names from basilica-api's country mapping for better readability

### Fixed
- Performance improvement: eliminated N+1 database queries when listing rentals

## [0.2.0]

### Added
- Use registered callback ports for OAuth flow instead of dynamic port allocation
- Add coloring to clap help page with clap v3 styles for better readability
- Automatic authentication prompts when commands require auth (no manual login needed)
- GPU requirements-based selection - specify GPU needs and auto-select matching nodes
- `export-token` command for exporting authentication tokens in various formats (env, json, shell) for automation

### Changed
- Simplified token storage from keyring to file-based system
- Enhanced GPU node display with grouped selection mode, compact view by default (use `--detailed` flag for full GPU names), and improved table formatting
- Unified GPU node targeting - accept either node UUID or GPU category (h100, h200, b200) as target parameter, removing separate --gpu-type option
- Migrated from basilica-api to basilica-sdk for all API interactions
- Refactored authentication to use oauth2 crate for improved token refresh mechanism
- Restructured authentication flow between SDK and CLI for better separation of concerns

### Fixed
- Consistent GPU count prefixes in all displays (e.g., "2x H100")
- Better expired/invalid token handling with clear user guidance
- Improved token refresh reliability with oauth2 crate integration

## [0.1.1] - Previous Release

Initial release
