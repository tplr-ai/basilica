//! Shared GitHub release fetching and version management utilities
//!
//! This module provides common functionality for checking CLI updates,
//! fetching releases from GitHub, and managing version compatibility.

use color_eyre::eyre::{eyre, Result};
use semver::Version;
use serde::Deserialize;
use std::io::IsTerminal;

/// Minimum supported version for auto-updates (first release with new CI binary format)
pub const MIN_SUPPORTED_VERSION: &str = "0.5.5";

/// GitHub repository configuration
pub struct GitHubConfig {
    pub owner: &'static str,
    pub repo: &'static str,
}

impl GitHubConfig {
    /// Default configuration for Basilica repository
    pub const fn basilica() -> Self {
        Self {
            owner: "one-covenant",
            repo: "basilica",
        }
    }
}

/// Information about a CLI release
#[derive(Debug, Clone)]
pub struct ReleaseInfo {
    /// Semantic version of the release
    pub version: Version,
    /// Full Git tag name (e.g., "basilica-cli-v0.5.5")
    pub tag: String,
}

/// Extract version from a CLI release tag
///
/// Handles tags in format "basilica-cli-vX.Y.Z" or "basilica-cli-X.Y.Z"
/// Returns None if tag doesn't match expected format or version is invalid
///
/// # Examples
/// ```
/// use basilica_cli::github_releases::extract_version_from_tag;
///
/// assert!(extract_version_from_tag("basilica-cli-v0.5.5").is_some());
/// assert!(extract_version_from_tag("basilica-cli-0.5.5").is_some());
/// assert!(extract_version_from_tag("invalid").is_none());
/// ```
pub fn extract_version_from_tag(tag: &str) -> Option<Version> {
    let version_str = tag
        .trim_start_matches("basilica-cli-v")
        .trim_start_matches("basilica-cli-")
        .trim_start_matches('v');

    Version::parse(version_str).ok()
}

/// Format a version string into a CLI release tag
///
/// Handles version strings with or without 'v' prefix
///
/// # Examples
/// ```
/// use basilica_cli::github_releases::format_cli_tag;
///
/// assert_eq!(format_cli_tag("0.5.5"), "basilica-cli-v0.5.5");
/// assert_eq!(format_cli_tag("v0.5.5"), "basilica-cli-v0.5.5");
/// ```
pub fn format_cli_tag(version: &str) -> String {
    let clean_version = version.trim_start_matches('v');
    format!("basilica-cli-v{}", clean_version)
}

/// Check if a version is supported for auto-updates
///
/// Returns true if the version is >= MIN_SUPPORTED_VERSION
pub fn is_version_supported(version: &Version) -> bool {
    let min_version = match Version::parse(MIN_SUPPORTED_VERSION) {
        Ok(v) => v,
        Err(_) => return false,
    };

    version >= &min_version
}

/// Check if update checks should be performed
///
/// Returns false if:
/// - BASILICA_NO_UPDATE_CHECK environment variable is set
/// - Not running in a TTY (to avoid polluting scripts/CI output)
pub fn should_check_for_updates() -> bool {
    // Skip if explicitly disabled
    if std::env::var("BASILICA_NO_UPDATE_CHECK").is_ok() {
        return false;
    }

    // Skip if not running in a TTY
    if !std::io::stdout().is_terminal() {
        return false;
    }

    true
}

/// A GitHub release as returned by the GitHub API.
///
/// We fetch releases directly instead of using `self_update::backends::github::ReleaseList`
/// because `self_update`'s `Release` struct drops the `prerelease` and `draft` fields during
/// JSON parsing, making it impossible to filter out GitHub pre-releases that have a clean
/// semver tag (e.g., `0.18.2` marked as pre-release while CI builds assets).
#[derive(Debug, Deserialize)]
struct GitHubRelease {
    tag_name: String,
    prerelease: bool,
    draft: bool,
}

/// Fetch all releases from GitHub
fn fetch_cli_releases() -> Result<Vec<GitHubRelease>> {
    let config = GitHubConfig::basilica();
    let url = format!(
        "https://api.github.com/repos/{}/{}/releases?per_page=100",
        config.owner, config.repo
    );

    let client = reqwest::blocking::Client::new();
    let response = client
        .get(&url)
        .header("Accept", "application/vnd.github+json")
        .header("User-Agent", "basilica-cli")
        .send()
        .map_err(|e| eyre!("Failed to fetch releases from GitHub: {}", e))?;

    if !response.status().is_success() {
        return Err(eyre!("GitHub API returned status {}", response.status()));
    }

    response
        .json::<Vec<GitHubRelease>>()
        .map_err(|e| eyre!("Failed to parse GitHub releases: {}", e))
}

/// Find the latest compatible CLI release
///
/// Filters releases to find the latest version that:
/// - Matches the "basilica-cli-v*" tag pattern
/// - Is >= MIN_SUPPORTED_VERSION
/// - Is newer than current_version (if check_newer is true)
///
/// Returns None if no compatible release is found
pub fn find_latest_cli_release(
    current_version: &str,
    check_newer: bool,
) -> Result<Option<ReleaseInfo>> {
    let releases = fetch_cli_releases()?;

    let current = if check_newer {
        Some(Version::parse(current_version).map_err(|e| eyre!("Invalid current version: {}", e))?)
    } else {
        None
    };

    // Filter and find the latest compatible release
    let latest = releases
        .iter()
        .filter_map(|r| {
            // Skip GitHub pre-releases and drafts (e.g., CI still building assets)
            if r.prerelease || r.draft {
                return None;
            }

            // Must match CLI release tag pattern
            if !r.tag_name.starts_with("basilica-cli-v") {
                return None;
            }

            // Extract and parse version
            let version = extract_version_from_tag(&r.tag_name)?;

            // Skip semver prerelease versions (rc, beta, alpha, etc.)
            if !version.pre.is_empty() {
                return None;
            }

            // Must be supported version
            if !is_version_supported(&version) {
                return None;
            }

            // If checking for newer versions, must be > current
            if let Some(ref cur) = current {
                if version <= *cur {
                    return None;
                }
            }

            Some(ReleaseInfo {
                version,
                tag: r.tag_name.clone(),
            })
        })
        .max_by(|a, b| a.version.cmp(&b.version));

    Ok(latest)
}

/// Get the latest CLI release version string (for background checks)
///
/// This is a convenience wrapper around find_latest_cli_release that returns
/// just the version string, suitable for caching.
pub fn fetch_latest_version_string(current_version: &str) -> Result<String> {
    match find_latest_cli_release(current_version, true)? {
        Some(release) => Ok(release.version.to_string()),
        None => Err(eyre!("No newer CLI releases found")),
    }
}
