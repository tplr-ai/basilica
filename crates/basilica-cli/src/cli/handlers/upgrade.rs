//! CLI upgrade handler using self_update crate

use crate::error::CliError;
use crate::github_releases::{
    find_latest_cli_release, format_cli_tag, is_version_supported, GitHubConfig,
    MIN_SUPPORTED_VERSION,
};
use color_eyre::eyre::eyre;
use console::style;
use self_update::cargo_crate_version;
use semver::Version;

/// Handle the upgrade command
/// Note: This function uses blocking operations from self_update crate
pub fn handle_upgrade(version: Option<String>, dry_run: bool) -> Result<(), CliError> {
    let current_version = cargo_crate_version!();

    // Validate version if specified
    if let Some(ref ver) = version {
        let target_version = ver.trim_start_matches('v');

        // Parse and validate the requested version
        let requested_version = Version::parse(target_version).map_err(|e| {
            CliError::Internal(eyre!("Invalid version format '{}': {}", target_version, e))
        })?;

        if !is_version_supported(&requested_version) {
            return Err(CliError::Internal(eyre!(
                "Version {} is not supported for auto-updates. Minimum supported version is {}",
                target_version,
                MIN_SUPPORTED_VERSION
            )));
        }
    }

    // Handle dry-run mode: check for updates without installing
    if dry_run {
        return handle_dry_run(current_version);
    }

    println!("Current version: {}", style(current_version).cyan());
    println!("Checking for updates...");

    // Ensure alias symlink exists even when already up to date
    ensure_alias_symlink();

    // Determine target tag
    let target_tag = if let Some(ref ver) = version {
        // User specified a version - use it directly
        format_cli_tag(ver)
    } else {
        // Find latest release
        match find_latest_cli_release(current_version, true).map_err(CliError::Internal)? {
            Some(release) => release.tag,
            None => {
                println!("{}", style("Already up to date!").green());
                println!("Current version: {}", style(current_version).cyan());
                return Ok(());
            }
        }
    };

    // Configure and execute the update
    let config = GitHubConfig::basilica();
    let mut update_builder = self_update::backends::github::Update::configure();

    update_builder
        .repo_owner(config.owner)
        .repo_name(config.repo)
        .bin_name("basilica")
        .current_version(current_version)
        .show_download_progress(true)
        .show_output(false)
        .no_confirm(true)
        .target_version_tag(&target_tag);

    // Build and execute the update
    let status = update_builder
        .build()
        .map_err(|e| CliError::Internal(eyre!("Failed to configure updater: {}", e)))?
        .update()
        .map_err(|e| {
            // Provide helpful error messages for common failures
            let error_msg = format!("{}", e);
            if error_msg.contains("permission") || error_msg.contains("Permission") {
                CliError::Internal(eyre!(
                    "Failed to replace binary: {}. You may need elevated permissions.\n\
                     Try running: sudo -E basilica upgrade",
                    e
                ))
            } else if error_msg.contains("not found") || error_msg.contains("404") {
                CliError::Internal(eyre!(
                    "Release not found. Please check that the version exists.\n\
                     View available releases: https://github.com/{}/{}/releases",
                    config.owner,
                    config.repo
                ))
            } else if error_msg.contains("target") || error_msg.contains("asset") {
                CliError::Internal(eyre!(
                    "No binary available for your platform.\n\
                     Supported platforms: Linux (x86_64, aarch64), macOS (x86_64, aarch64)\n\
                     Error: {}",
                    e
                ))
            } else {
                CliError::Internal(eyre!("Update failed: {}", e))
            }
        })?;

    // Display results
    match status {
        self_update::Status::UpToDate(v) => {
            println!("{}", style("Already up to date!").green());
            println!("Current version: {}", style(v).cyan());
        }
        self_update::Status::Updated(v) => {
            println!(
                "\n{} Updated to version {}",
                style("✓").green().bold(),
                style(v).green().bold()
            );

            println!(
                "\nRun {} or {} to verify the new version",
                style("basilica --version").cyan(),
                style("bs --version").cyan()
            );
        }
    }

    Ok(())
}

/// Ensure the 'bs' alias symlink exists alongside the main binary
#[cfg(unix)]
fn ensure_alias_symlink() {
    use std::os::unix::fs::symlink;

    let Ok(current_exe) = std::env::current_exe() else {
        return;
    };
    let Some(parent) = current_exe.parent() else {
        return;
    };
    let Some(binary_name) = current_exe.file_name() else {
        return;
    };

    let alias_path = parent.join("bs");

    // Remove existing symlink/file if present (ignore errors)
    let _ = std::fs::remove_file(&alias_path);

    // Create new symlink (relative, so it survives directory moves)
    match symlink(binary_name, &alias_path) {
        Ok(_) => println!("Created 'bs' alias"),
        Err(e) => eprintln!("Failed to create 'bs' alias: {}", e),
    }
}

#[cfg(not(unix))]
fn ensure_alias_symlink() {
    // Symlinks on Windows require special permissions
    // Skip for now - Windows not currently supported
}

/// Handle dry-run mode: check for updates without installing
fn handle_dry_run(current_version: &str) -> Result<(), CliError> {
    println!("Current version: {}", style(current_version).cyan());
    println!("Checking for updates...");

    // Use shared logic to find latest release
    match find_latest_cli_release(current_version, true).map_err(CliError::Internal)? {
        Some(release) => {
            println!(
                "Latest version available: {}",
                style(&release.version).green()
            );
            println!(
                "\nRun {} to upgrade",
                style("basilica upgrade").cyan().bold()
            );
        }
        None => {
            println!("{}", style("Already up to date!").green());
        }
    }

    Ok(())
}
