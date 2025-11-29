# Changelog

All notable changes to the basilica-cli package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
