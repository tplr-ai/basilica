//! SSH key matching utilities to find private keys based on public keys

use color_eyre::eyre::{eyre, Result};
use color_eyre::Section;
use std::fs;
use std::path::PathBuf;

/// Find the private key file that corresponds to a given public key
///
/// This function searches ~/.ssh/ for .pub files and compares their content
/// with the registered public key. When a match is found, it returns the
/// path to the corresponding private key (same filename without .pub extension).
///
/// # Arguments
/// * `registered_public_key` - The public key string registered with the API
///
/// # Returns
/// Path to the matching private key file
///
/// # Errors
/// - No matching public key found in ~/.ssh/
/// - Multiple matching keys found (shouldn't happen)
/// - Matching public key found but private key doesn't exist
pub fn find_private_key_for_public_key(registered_public_key: &str) -> Result<PathBuf> {
    let home = std::env::var("HOME")
        .map_err(|_| eyre!("Could not determine home directory from HOME environment variable"))?;
    let ssh_dir = PathBuf::from(home).join(".ssh");

    if !ssh_dir.exists() {
        return Err(eyre!("SSH directory not found at {}", ssh_dir.display())
            .note("Your registered SSH key is not available on this machine")
            .suggestion("Run 'basilica ssh-keys add' to register a key from this machine"));
    }

    // Parse the registered public key (format: "key-type key-data [optional-comment]")
    let registered_key_parts = parse_public_key(registered_public_key)?;

    // Search for all .pub files in ~/.ssh/
    let mut matches = Vec::new();

    let entries = fs::read_dir(&ssh_dir)
        .map_err(|e| eyre!("Failed to read SSH directory {}: {}", ssh_dir.display(), e))?;

    for entry in entries {
        let entry = entry.map_err(|e| eyre!("Failed to read directory entry: {}", e))?;
        let path = entry.path();

        // Only process .pub files
        if !path.is_file() || path.extension().and_then(|s| s.to_str()) != Some("pub") {
            continue;
        }

        // Read and parse the public key file
        let content = match fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => continue, // Skip files we can't read
        };

        let file_key_parts = match parse_public_key(&content) {
            Ok(parts) => parts,
            Err(_) => continue, // Skip invalid key files
        };

        // Compare key type and key data (ignore comments)
        if file_key_parts.key_type == registered_key_parts.key_type
            && file_key_parts.key_data == registered_key_parts.key_data
        {
            matches.push(path);
        }
    }

    // Handle results
    match matches.len() {
        0 => Err(eyre!(
            "Could not find private key matching your registered public key in ~/.ssh/"
        )
        .note("The SSH key you registered with Basilica is not available on this machine")
        .suggestion("Either copy your SSH key to this machine or register a new key with 'basilica ssh-keys add'")),

        1 => {
            let pub_key_path = &matches[0];
            // Derive private key path by removing .pub extension
            let private_key_path = pub_key_path.with_extension("");

            // Verify private key exists
            if !private_key_path.exists() {
                return Err(eyre!(
                    "Found matching public key at {} but private key is missing",
                    pub_key_path.display()
                )
                .suggestion("Ensure both the public and private key files exist"));
            }

            Ok(private_key_path)
        }

        _ => Err(eyre!(
            "Found multiple private keys matching your registered public key (unusual)"
        )
        .note("This shouldn't happen with standard SSH key setup")
        .suggestion("Please report this issue with the list of matching keys")),
    }
}

/// Find the local public key file that corresponds to a given public key string
///
/// This function searches ~/.ssh/ for .pub files and compares their content
/// with the provided public key. When a match is found, it returns the path
/// to the public key file.
///
/// # Arguments
/// * `registered_public_key` - The public key string to search for
///
/// # Returns
/// Some(PathBuf) to the matching public key file, or None if:
/// - No ~/.ssh directory exists
/// - No matching .pub file found
/// - The public key is malformed
pub fn find_local_public_key_path(registered_public_key: &str) -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let ssh_dir = PathBuf::from(home).join(".ssh");

    if !ssh_dir.exists() {
        return None;
    }

    let registered_key_parts = parse_public_key(registered_public_key).ok()?;

    let entries = fs::read_dir(&ssh_dir).ok()?;

    for entry in entries.flatten() {
        let path = entry.path();

        if !path.is_file() || path.extension().and_then(|s| s.to_str()) != Some("pub") {
            continue;
        }

        let content = match fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let file_key_parts = match parse_public_key(&content) {
            Ok(parts) => parts,
            Err(_) => continue,
        };

        if file_key_parts.key_type == registered_key_parts.key_type
            && file_key_parts.key_data == registered_key_parts.key_data
        {
            return Some(path);
        }
    }

    None
}

/// Parsed components of an SSH public key
#[derive(Debug, PartialEq)]
struct PublicKeyParts {
    key_type: String,
    key_data: String,
}

/// Parse an SSH public key into its components
///
/// SSH public keys have format: "key-type key-data [optional-comment]"
/// This function extracts just the key-type and key-data parts, ignoring comments.
fn parse_public_key(public_key: &str) -> Result<PublicKeyParts> {
    let trimmed = public_key.trim();
    let parts: Vec<&str> = trimmed.split_whitespace().collect();

    if parts.len() < 2 {
        return Err(eyre!("Invalid public key format"));
    }

    Ok(PublicKeyParts {
        key_type: parts[0].to_string(),
        key_data: parts[1].to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_public_key_with_comment() {
        let key = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIGxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx user@hostname";
        let parsed = parse_public_key(key).unwrap();
        assert_eq!(parsed.key_type, "ssh-ed25519");
        assert_eq!(
            parsed.key_data,
            "AAAAC3NzaC1lZDI1NTE5AAAAIGxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        );
    }

    #[test]
    fn test_parse_public_key_without_comment() {
        let key = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQC1xxxxxxxxxx";
        let parsed = parse_public_key(key).unwrap();
        assert_eq!(parsed.key_type, "ssh-rsa");
        assert_eq!(
            parsed.key_data,
            "AAAAB3NzaC1yc2EAAAADAQABAAABgQC1xxxxxxxxxx"
        );
    }

    #[test]
    fn test_parse_public_key_invalid() {
        let key = "invalid-key";
        assert!(parse_public_key(key).is_err());
    }

    #[test]
    fn test_keys_match_ignoring_comment() {
        let key1 = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIGxxxxxxx comment1";
        let key2 = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIGxxxxxxx comment2";

        let parsed1 = parse_public_key(key1).unwrap();
        let parsed2 = parse_public_key(key2).unwrap();

        assert_eq!(parsed1, parsed2);
    }
}
