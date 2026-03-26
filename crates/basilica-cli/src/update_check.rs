//! Background update checker for the Basilica CLI
//!
//! Checks for new versions once per day and displays a notification if available.
//! Inspired by Deno's upgrade notification system.

use crate::github_releases::{fetch_latest_version_string, should_check_for_updates};
use chrono::{DateTime, Duration, Utc};
use console::style;
use etcetera::{choose_base_strategy, BaseStrategy};
use self_update::cargo_crate_version;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

const UPDATE_CHECK_FILE: &str = "update_check.json";
const CHECK_INTERVAL_HOURS: i64 = 24;

#[derive(Debug, Serialize, Deserialize)]
struct UpdateCheckCache {
    last_check: DateTime<Utc>,
    latest_version: Option<String>,
    last_prompt: Option<DateTime<Utc>>,
}

/// Check the cache and show update notification if appropriate.
/// This should be called at CLI startup, before executing the command.
pub fn check_cache_and_show_notification() {
    // Skip if update checks are disabled
    if !should_check_for_updates() {
        return;
    }

    let cache_path = match get_cache_path() {
        Ok(path) => path,
        Err(_) => return, // Silently fail if we can't determine cache path
    };

    let mut cache = match load_cache(&cache_path) {
        Ok(Some(c)) => c,
        _ => return, // No cache or error reading it
    };

    // Check if we have a newer version available
    if let Some(latest_version) = &cache.latest_version {
        let current_version = cargo_crate_version!();

        // Only show if the latest version is strictly newer than current
        if let (Ok(latest), Ok(current)) = (
            semver::Version::parse(latest_version),
            semver::Version::parse(current_version),
        ) {
            if latest > current {
                // Show the notification
                eprintln!(
                    "{} {} → {}",
                    style("A new version of Basilica is available:").yellow(),
                    style(current_version).cyan(),
                    style(latest_version).green().bold()
                );
                eprintln!(
                    "{} {}",
                    style("Run").dim(),
                    style("basilica upgrade").cyan().bold()
                );
                eprintln!(); // Empty line for spacing

                // Update last_prompt time
                cache.last_prompt = Some(Utc::now());
                let _ = save_cache(&cache_path, &cache);
            } else {
                // Cached version is stale (user upgraded past it), update cache
                cache.latest_version = Some(current_version.to_string());
                let _ = save_cache(&cache_path, &cache);
            }
        }
    }
}

/// Check for updates and display a notification if a new version is available.
/// This runs at most once per day and respects the BASILICA_NO_UPDATE_CHECK env var.
pub fn check_and_notify_update() {
    // Skip if update checks are disabled
    if !should_check_for_updates() {
        return;
    }

    // Check if we should run the update check (once per day)
    let cache_path = match get_cache_path() {
        Ok(path) => path,
        Err(_) => return, // Silently fail if we can't determine cache path
    };

    let should_check = match load_cache(&cache_path) {
        Ok(Some(cache)) => {
            let now = Utc::now();
            let elapsed = now.signed_duration_since(cache.last_check);
            elapsed > Duration::hours(CHECK_INTERVAL_HOURS)
        }
        Ok(None) => true, // No cache file, first run
        Err(_) => true,   // Error reading cache, try checking
    };

    if !should_check {
        // Already checked recently, skip background check
        return;
    }

    let cache_path_for_task = cache_path;

    // self_update uses reqwest's blocking client, which manages its own Tokio runtime.
    // Running it on the async executor causes Tokio to drop a blocking runtime while in
    // async context, leading to "Cannot drop a runtime in a context where blocking is not allowed".
    tokio::task::spawn_blocking(move || perform_update_check(cache_path_for_task));
}

fn perform_update_check(cache_path: PathBuf) {
    let current_version = cargo_crate_version!();

    // Preserve last_prompt from existing cache
    let existing_last_prompt = load_cache(&cache_path)
        .ok()
        .flatten()
        .and_then(|c| c.last_prompt);

    let cache = match fetch_latest_version_string(current_version) {
        Ok(latest_version) => UpdateCheckCache {
            last_check: Utc::now(),
            latest_version: Some(latest_version),
            last_prompt: existing_last_prompt,
        },
        Err(_) => UpdateCheckCache {
            last_check: Utc::now(),
            latest_version: Some(current_version.to_string()),
            last_prompt: existing_last_prompt,
        },
    };

    let _ = save_cache(&cache_path, &cache);
}

/// Get the path to the update check cache file
fn get_cache_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let strategy = choose_base_strategy()?;
    let cache_dir = strategy.cache_dir().join("basilica");

    // Ensure directory exists
    fs::create_dir_all(&cache_dir)?;

    Ok(cache_dir.join(UPDATE_CHECK_FILE))
}

/// Load the update check cache
fn load_cache(path: &PathBuf) -> Result<Option<UpdateCheckCache>, Box<dyn std::error::Error>> {
    if !path.exists() {
        return Ok(None);
    }

    let content = fs::read_to_string(path)?;
    let cache: UpdateCheckCache = serde_json::from_str(&content)?;
    Ok(Some(cache))
}

/// Save the update check cache
fn save_cache(path: &PathBuf, cache: &UpdateCheckCache) -> Result<(), Box<dyn std::error::Error>> {
    let content = serde_json::to_string_pretty(cache)?;
    fs::write(path, content)?;
    Ok(())
}
