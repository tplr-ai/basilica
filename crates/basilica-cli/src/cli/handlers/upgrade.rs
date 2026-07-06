//! CLI upgrade handler using self_update crate

use crate::error::CliError;
use crate::github_releases::{
    extract_version_from_tag, find_latest_cli_release, format_cli_tag, is_version_supported,
    GitHubConfig, MIN_SUPPORTED_VERSION,
};
use color_eyre::eyre::{eyre, Result as EyreResult};
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use self_update::cargo_crate_version;
use self_update::update::{Release, ReleaseAsset};
use semver::Version;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

/// Short alias name for the CLI binary
const CLI_ALIAS: &str = "bs";
const BINARY_NAME: &str = "basilica";

/// Resolve the actual binary path by following symlinks.
///
/// `std::env::current_exe()` on macOS does NOT resolve symlinks — when invoked
/// as `bs` (a symlink to `basilica`), it returns `.../bin/bs`. This helper
/// canonicalizes the path so we always operate on the real `basilica` binary.
fn resolve_binary_path() -> Option<std::path::PathBuf> {
    let current_exe = std::env::current_exe().ok()?;
    Some(std::fs::canonicalize(&current_exe).unwrap_or(current_exe))
}

fn parse_sha256_file(contents: &str) -> Option<String> {
    let digest = contents.split_whitespace().next()?.to_lowercase();

    if digest.len() == 64 && digest.chars().all(|c| c.is_ascii_hexdigit()) {
        Some(digest)
    } else {
        None
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn sha256_file_hex(path: &Path) -> EyreResult<String> {
    let bytes = std::fs::read(path)?;
    Ok(sha256_hex(&bytes))
}

fn expected_asset_name(version: &str, target: &str) -> String {
    format!("{BINARY_NAME}-{version}-{target}.tar.gz")
}

fn download_url_to_path(url: &str, path: &Path, show_progress: bool) -> EyreResult<()> {
    let client = reqwest::blocking::Client::new();
    let mut response = client
        .get(url)
        .header("Accept", "application/octet-stream")
        .header("User-Agent", "basilica-cli")
        .send()
        .map_err(|e| eyre!("Failed to download release archive: {}", e))?;

    if !response.status().is_success() {
        return Err(eyre!(
            "Download request failed with status: {}",
            response.status()
        ));
    }

    let content_length = response.content_length();
    let mut progress = if show_progress {
        content_length.map(|len| {
            let progress = ProgressBar::new(len);
            if let Ok(style) = ProgressStyle::with_template(
                "[{elapsed_precise}] [{bar:40}] {bytes}/{total_bytes} ({eta}) {msg}",
            ) {
                progress.set_style(style.progress_chars("=>-"));
            }
            progress
        })
    } else {
        None
    };

    let mut output = File::create(path)?;
    let mut buffer = [0; 8192];

    loop {
        let bytes_read = response.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }

        output.write_all(&buffer[..bytes_read])?;

        if let Some(progress) = &mut progress {
            progress.inc(bytes_read as u64);
        }
    }

    if let Some(progress) = progress {
        progress.finish_with_message("Done");
    }

    Ok(())
}

fn download_url_to_string(url: &str) -> EyreResult<String> {
    let response = reqwest::blocking::Client::new()
        .get(url)
        .header("Accept", "application/octet-stream")
        .header("User-Agent", "basilica-cli")
        .send()
        .map_err(|e| eyre!("Failed to download checksum file: {}", e))?;

    if !response.status().is_success() {
        return Err(eyre!(
            "Download request failed with status: {}",
            response.status()
        ));
    }

    response
        .text()
        .map_err(|e| eyre!("Failed to read checksum file: {}", e))
}

fn find_exact_asset(release: &Release, asset_name: &str) -> EyreResult<ReleaseAsset> {
    release
        .assets
        .iter()
        .find(|asset| asset.name == asset_name)
        .cloned()
        .ok_or_else(|| eyre!("No release asset found named `{}`", asset_name))
}

fn fetch_release(
    config: &GitHubConfig,
    current_version: &str,
    target_tag: &str,
) -> EyreResult<Release> {
    let updater = self_update::backends::github::Update::configure()
        .repo_owner(config.owner)
        .repo_name(config.repo)
        .bin_name(BINARY_NAME)
        .current_version(current_version)
        .no_confirm(true)
        .target_version_tag(target_tag)
        .build()
        .map_err(|e| eyre!("Failed to configure updater: {}", e))?;

    updater
        .get_release_version(target_tag)
        .map_err(|e| eyre!("{}", e))
}

fn verify_archive_checksum(archive_path: &Path, checksum_contents: &str) -> EyreResult<()> {
    let expected = parse_sha256_file(checksum_contents)
        .ok_or_else(|| eyre!("Release checksum file did not contain a valid SHA-256 digest"))?;
    let actual = sha256_file_hex(archive_path)?;

    if actual.eq_ignore_ascii_case(&expected) {
        Ok(())
    } else {
        Err(eyre!(
            "Checksum verification failed: expected {}, got {}. The downloaded archive was discarded and no files were changed.",
            expected,
            actual
        ))
    }
}

fn map_update_error(error: impl std::fmt::Display, config: &GitHubConfig) -> CliError {
    let error_msg = format!("{}", error);
    if error_msg.contains("permission") || error_msg.contains("Permission") {
        CliError::Internal(eyre!(
            "Failed to replace binary: {}. You may need elevated permissions.\n\
             Try running: sudo -E basilica upgrade",
            error
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
            error
        ))
    } else {
        CliError::Internal(eyre!("Update failed: {}", error))
    }
}

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

    let config = GitHubConfig::basilica();
    let resolved_exe = resolve_binary_path()
        .ok_or_else(|| CliError::Internal(eyre!("Failed to determine executable path")))?;
    let bin_dir = resolved_exe
        .parent()
        .ok_or_else(|| CliError::Internal(eyre!("Failed to determine binary directory")))?
        .to_path_buf();
    let target = self_update::get_target();
    let target_version = extract_version_from_tag(&target_tag)
        .ok_or_else(|| CliError::Internal(eyre!("Invalid release tag '{}'", target_tag)))?;
    let target_version = target_version.to_string();
    let archive_name = expected_asset_name(&target_version, target);
    let checksum_name = format!("{archive_name}.sha256");

    println!("Downloading release assets for {}", style(target).cyan());

    let release = fetch_release(&config, current_version, &target_tag)
        .map_err(|e| map_update_error(e, &config))?;
    let archive_asset =
        find_exact_asset(&release, &archive_name).map_err(|e| map_update_error(e, &config))?;
    let checksum_asset =
        find_exact_asset(&release, &checksum_name).map_err(|e| map_update_error(e, &config))?;

    let temp_dir = self_update::TempDir::new()
        .map_err(|e| CliError::Internal(eyre!("Failed to create temporary directory: {}", e)))?;
    let archive_path = temp_dir.path().join(&archive_name);

    download_url_to_path(&archive_asset.download_url, &archive_path, true)
        .map_err(|e| map_update_error(e, &config))?;
    let checksum_contents = download_url_to_string(&checksum_asset.download_url)
        .map_err(|e| map_update_error(e, &config))?;

    verify_archive_checksum(&archive_path, &checksum_contents).map_err(CliError::Internal)?;

    println!("Extracting archive...");
    self_update::Extract::from_source(&archive_path)
        .archive(self_update::ArchiveKind::Tar(Some(
            self_update::Compression::Gz,
        )))
        .extract_file(temp_dir.path(), BINARY_NAME)
        .map_err(|e| map_update_error(e, &config))?;

    let new_exe = temp_dir.path().join(BINARY_NAME);

    // Change CWD to the binary's directory so that self_replace's
    // read_link() → relative path resolves correctly
    std::env::set_current_dir(&bin_dir).map_err(|e| {
        CliError::Internal(eyre!(
            "failed to set CWD to binary's parent directory: {}",
            e
        ))
    })?;

    // The running binary is not touched until every fallible preparation step
    // above has completed: download, checksum verification, and extraction.
    self_update::self_replace::self_replace(&new_exe).map_err(|e| map_update_error(e, &config))?;
    let status = self_update::Status::Updated(target_version);

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

    let Some(resolved) = resolve_binary_path() else {
        return;
    };
    let Some(parent) = resolved.parent() else {
        return;
    };
    let Some(binary_name) = resolved.file_name() else {
        return;
    };

    // If the resolved binary name is already the alias, skip to avoid a self-referencing symlink
    if binary_name == CLI_ALIAS {
        return;
    }

    let alias_path = parent.join(CLI_ALIAS);

    // Remove existing symlink/file if present (ignore errors)
    let _ = std::fs::remove_file(&alias_path);

    // Create new symlink (relative, so it survives directory moves)
    match symlink(binary_name, &alias_path) {
        Ok(_) => println!("Created '{}' alias", CLI_ALIAS),
        Err(e) => eprintln!("Failed to create '{}' alias: {}", CLI_ALIAS, e),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_sha256_file_accepts_shasum_format() {
        let checksum = "BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD  basilica-0.1.0-x86_64-unknown-linux-gnu.tar.gz\n";

        assert_eq!(
            parse_sha256_file(checksum),
            Some("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".to_string())
        );
    }

    #[test]
    fn parse_sha256_file_accepts_sha256sum_format() {
        let checksum = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad *basilica-0.1.0-x86_64-unknown-linux-gnu.tar.gz\n";

        assert_eq!(
            parse_sha256_file(checksum),
            Some("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".to_string())
        );
    }

    #[test]
    fn parse_sha256_file_rejects_garbage() {
        assert_eq!(parse_sha256_file("not-a-checksum archive.tar.gz"), None);
        assert_eq!(parse_sha256_file("abcd"), None);
        assert_eq!(
            parse_sha256_file(
                "zz7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad archive"
            ),
            None
        );
    }

    #[test]
    fn sha256_hex_hashes_known_bytes() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn expected_asset_name_matches_release_workflow_pattern() {
        assert_eq!(
            expected_asset_name("0.5.5", "aarch64-apple-darwin"),
            "basilica-0.5.5-aarch64-apple-darwin.tar.gz"
        );
    }
}
