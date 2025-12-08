//! SSH key management handlers for the Basilica CLI

use crate::error::CliError;
use crate::output::{compress_path, print_success};
use crate::ssh::find_local_public_key_path;
use basilica_sdk::BasilicaClient;
use console::style;
use dialoguer::{theme::ColorfulTheme, Confirm, Input};
use etcetera::{choose_base_strategy, BaseStrategy};
use ssh_key::PublicKey;
use std::fs;
use std::path::PathBuf;

/// Generate a new SSH key for Basilica
async fn generate_ssh_key() -> Result<PathBuf, CliError> {
    let strategy = choose_base_strategy().map_err(|e| {
        CliError::Internal(color_eyre::eyre::eyre!(
            "Failed to determine home directory: {}",
            e
        ))
    })?;
    let home = strategy.home_dir();
    let ssh_dir = home.join(".ssh");
    let private_key_path = ssh_dir.join("basilica_ed25519");
    let public_key_path = ssh_dir.join("basilica_ed25519.pub");

    // Check if key already exists (check both public and private key)
    if public_key_path.exists() {
        println!(
            "{}",
            style(format!(
                "SSH key already exists: {}",
                public_key_path.display()
            ))
            .cyan()
        );
        return Ok(public_key_path);
    }

    // If private key exists but public key doesn't, we can't proceed
    if private_key_path.exists() {
        return Err(CliError::Internal(color_eyre::eyre::eyre!(
            "Cannot generate key: {} exists but {} does not",
            private_key_path.display(),
            public_key_path.display()
        )));
    }

    // Create ~/.ssh if it doesn't exist
    if !ssh_dir.exists() {
        fs::create_dir_all(&ssh_dir).map_err(|e| {
            CliError::Internal(color_eyre::eyre::eyre!(
                "Failed to create ~/.ssh directory: {}",
                e
            ))
        })?;
        // Set permissions to 700 on Unix
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&ssh_dir, fs::Permissions::from_mode(0o700)).map_err(|e| {
                CliError::Internal(color_eyre::eyre::eyre!(
                    "Failed to set ~/.ssh permissions: {}",
                    e
                ))
            })?;
        }
    }

    println!("{}", style("Generating a new ed25519 key...").cyan());

    let output = tokio::process::Command::new("ssh-keygen")
        .args([
            "-t",
            "ed25519",
            "-f",
            private_key_path.to_str().unwrap(),
            "-N",
            "", // Empty passphrase
            "-C",
            "basilica",
        ])
        .output()
        .await
        .map_err(|e| {
            CliError::Internal(color_eyre::eyre::eyre!("Failed to run ssh-keygen: {}", e))
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(CliError::Internal(color_eyre::eyre::eyre!(
            "ssh-keygen failed: {}",
            stderr
        )));
    }

    println!(
        "{}",
        style(format!("Generated SSH key: {}", public_key_path.display())).green()
    );

    Ok(public_key_path)
}

/// Find all SSH public keys in ~/.ssh directory
pub fn find_ssh_public_keys() -> Vec<PathBuf> {
    let strategy = match choose_base_strategy() {
        Ok(s) => s,
        Err(_) => return vec![],
    };
    let home = strategy.home_dir();
    let ssh_dir = home.join(".ssh");

    if !ssh_dir.exists() {
        return vec![];
    }

    // Read all .pub files in ~/.ssh directory
    let mut keys = vec![];
    if let Ok(entries) = std::fs::read_dir(&ssh_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("pub") {
                keys.push(path);
            }
        }
    }

    // Sort alphabetically by filename
    keys.sort_by(|a, b| {
        let a_name = a.file_name().and_then(|n| n.to_str()).unwrap_or("");
        let b_name = b.file_name().and_then(|n| n.to_str()).unwrap_or("");
        a_name.cmp(b_name)
    });

    keys
}

/// Validate SSH public key format
pub fn validate_ssh_public_key(content: &str) -> Result<(), String> {
    let trimmed = content.trim();

    if trimmed.is_empty() {
        return Err("SSH public key is empty".to_string());
    }

    // Check if it looks like a private key (common mistake)
    if trimmed.contains("PRIVATE KEY") {
        return Err(
            "This appears to be a private key. Please use the public key file (*.pub)".to_string(),
        );
    }

    // Check if it starts with a known key type
    if PublicKey::from_openssh(trimmed).is_err() {
        return Err("Invalid SSH public key format".to_string());
    }

    Ok(())
}

/// Result of SSH key selection containing path and content
pub struct SelectedSshKey {
    pub path: PathBuf,
    pub content: String,
}

/// Discover SSH public keys in ~/.ssh and let user select one interactively
///
/// Returns the selected key's path and validated content.
/// Always shows a selector with existing keys plus an option to generate a new key.
pub async fn select_and_read_ssh_key() -> Result<SelectedSshKey, CliError> {
    use dialoguer::Select;

    // Find all SSH public keys in ~/.ssh
    let keys = find_ssh_public_keys();

    let key_path = if keys.is_empty() {
        // No keys found, generate one automatically
        generate_ssh_key().await?
    } else {
        // Build options from existing keys
        let options: Vec<String> = keys
            .iter()
            .map(|path| {
                path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string()
            })
            .collect();

        // Run interactive selection in a blocking context
        let selection = tokio::task::spawn_blocking(move || {
            let theme = ColorfulTheme::default();
            Select::with_theme(&theme)
                .with_prompt("Select a key to register")
                .items(&options)
                .default(0)
                .interact()
        })
        .await
        .map_err(|e| CliError::Internal(color_eyre::eyre::eyre!("Task join error: {}", e)))?
        .map_err(|e| CliError::Internal(e.into()))?;

        keys[selection].clone()
    };

    // Read and validate SSH public key
    let content = fs::read_to_string(&key_path).map_err(|e| {
        CliError::Internal(color_eyre::eyre::eyre!(
            "Failed to read SSH key file: {}",
            e
        ))
    })?;

    if let Err(e) = validate_ssh_public_key(&content) {
        return Err(CliError::Internal(color_eyre::eyre::eyre!(
            "Invalid SSH public key: {}",
            e
        )));
    }

    Ok(SelectedSshKey {
        path: key_path,
        content: content.trim().to_string(),
    })
}

/// Handle adding a new SSH key
pub async fn handle_add_ssh_key(
    client: &BasilicaClient,
    name: Option<String>,
    file: Option<PathBuf>,
) -> Result<(), CliError> {
    // Step 1: Get SSH public key file path
    let key_path = match file {
        Some(path) => {
            if !path.exists() {
                return Err(CliError::Internal(color_eyre::eyre::eyre!(
                    "SSH key file not found: {}",
                    path.display()
                )));
            }
            path
        }
        None => {
            // Find all SSH public keys in ~/.ssh
            let keys = find_ssh_public_keys();

            if keys.is_empty() {
                // No keys found, generate one automatically
                generate_ssh_key().await?
            } else {
                // Show interactive selection with existing keys
                use dialoguer::Select;

                let options: Vec<String> = keys
                    .iter()
                    .map(|path| {
                        path.file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("unknown")
                            .to_string()
                    })
                    .collect();

                // Run interactive selection in a blocking context
                let selection = tokio::task::spawn_blocking(move || {
                    let theme = ColorfulTheme::default();
                    Select::with_theme(&theme)
                        .with_prompt("Select an SSH key to register")
                        .items(&options)
                        .default(0)
                        .interact()
                })
                .await
                .map_err(|e| CliError::Internal(color_eyre::eyre::eyre!("Task join error: {}", e)))?
                .map_err(|e| CliError::Internal(e.into()))?;

                let path = &keys[selection];
                println!(
                    "{}",
                    style(format!("Selected SSH public key: {}", path.display())).cyan()
                );
                path.clone()
            }
        }
    };

    // Step 2: Read and validate SSH public key
    let public_key = fs::read_to_string(&key_path).map_err(|e| {
        CliError::Internal(color_eyre::eyre::eyre!(
            "Failed to read SSH key file: {}",
            e
        ))
    })?;

    if let Err(e) = validate_ssh_public_key(&public_key) {
        return Err(CliError::Internal(color_eyre::eyre::eyre!(
            "Invalid SSH public key: {}",
            e
        )));
    }

    // Step 3: Get name interactively if not provided
    let name = match name {
        Some(n) => {
            if n.trim().is_empty() {
                return Err(CliError::Internal(color_eyre::eyre::eyre!(
                    "SSH key name cannot be empty"
                )));
            }
            if n.len() > 100 {
                return Err(CliError::Internal(color_eyre::eyre::eyre!(
                    "SSH key name must be 100 characters or less"
                )));
            }
            n
        }
        None => {
            let input: String = tokio::task::spawn_blocking(move || {
                let theme = ColorfulTheme::default();
                Input::with_theme(&theme)
                    .with_prompt("Enter a name for this SSH key")
                    .default("default".to_string())
                    .interact_text()
            })
            .await
            .map_err(|e| CliError::Internal(color_eyre::eyre::eyre!("Task join error: {}", e)))?
            .map_err(|e| CliError::Internal(e.into()))?;

            if input.trim().is_empty() {
                return Err(CliError::Internal(color_eyre::eyre::eyre!(
                    "SSH key name cannot be empty"
                )));
            }
            input
        }
    };

    // Step 4: Check if user already has an SSH key (warn about replacement)
    match client.get_ssh_key().await {
        Ok(Some(existing)) => {
            println!(
                "{}",
                style(format!(
                    "⚠️  You already have an SSH key registered: '{}'",
                    existing.name
                ))
                .yellow()
            );
            println!(
                "{}",
                style("Note: Only one SSH key is allowed per user.").yellow()
            );

            let confirmed = tokio::task::spawn_blocking(move || {
                let theme = ColorfulTheme::default();
                Confirm::with_theme(&theme)
                    .with_prompt("Do you want to replace it with the new key?")
                    .default(false)
                    .interact()
            })
            .await
            .map_err(|e| CliError::Internal(color_eyre::eyre::eyre!("Task join error: {}", e)))?
            .map_err(|e| CliError::Internal(e.into()))?;

            if !confirmed {
                println!("Operation cancelled.");
                return Ok(());
            }

            // Delete existing key first
            client.delete_ssh_key().await.map_err(CliError::Api)?;
            println!("{}", style("Existing SSH key deleted.").dim());
        }
        Ok(None) => {
            // No existing key, proceed
        }
        Err(e) => {
            return Err(CliError::Api(e));
        }
    }

    // Step 5: Register the new SSH key
    let response = client
        .register_ssh_key(&name, public_key.trim())
        .await
        .map_err(CliError::Api)?;

    // Display success - keep it simple, users don't need the details
    let _ = response; // suppress unused warning
    print_success("SSH key registered successfully!");

    Ok(())
}

/// Handle listing SSH keys
pub async fn handle_list_ssh_keys(client: &BasilicaClient) -> Result<(), CliError> {
    match client.get_ssh_key().await.map_err(CliError::Api)? {
        Some(key) => {
            println!("{}", style("Registered SSH Key:").bold());
            println!();
            println!("  Name:       {}", style(&key.name).cyan());
            println!(
                "  Created:    {}",
                key.created_at.format("%Y-%m-%d %H:%M:%S")
            );

            // Show local path if the key exists on this machine
            if let Some(local_path) = find_local_public_key_path(&key.public_key) {
                println!(
                    "  Local Path: {}",
                    style(compress_path(&local_path)).green()
                );
            } else {
                println!("  Local Path: {}", style("Not found locally").yellow());
            }

            println!();
            println!(
                "{}",
                style("Note: Only one SSH key is allowed per user.").dim()
            );
        }
        None => {
            println!("No SSH key registered.");
            println!();
            println!("Add one with: {} ssh-keys add", style("basilica").cyan());
        }
    }

    Ok(())
}

/// Handle deleting SSH key
pub async fn handle_delete_ssh_key(
    client: &BasilicaClient,
    skip_confirm: bool,
) -> Result<(), CliError> {
    // Check if SSH key exists
    let existing = match client.get_ssh_key().await.map_err(CliError::Api)? {
        Some(key) => key,
        None => {
            println!("No SSH key registered.");
            return Ok(());
        }
    };

    // Confirm deletion if not skipped
    if !skip_confirm {
        let key_name = existing.name.clone();
        let confirmed = tokio::task::spawn_blocking(move || {
            let theme = ColorfulTheme::default();
            Confirm::with_theme(&theme)
                .with_prompt(format!(
                    "Are you sure you want to delete SSH key '{}'?",
                    key_name
                ))
                .default(false)
                .interact()
        })
        .await
        .map_err(|e| CliError::Internal(color_eyre::eyre::eyre!("Task join error: {}", e)))?
        .map_err(|e| CliError::Internal(e.into()))?;

        if !confirmed {
            println!("Deletion cancelled.");
            return Ok(());
        }
    }

    // Delete the SSH key
    client.delete_ssh_key().await.map_err(CliError::Api)?;

    println!(
        "{}",
        style(format!(
            "✅ SSH key '{}' deleted successfully.",
            existing.name
        ))
        .green()
    );
    println!();
    println!(
        "{}",
        style("Note: This key has been removed from all cloud providers.").dim()
    );

    Ok(())
}
