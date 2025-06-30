use anyhow::{Context, Result};
use p256::ecdsa::VerifyingKey;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;

// Include the embedded keys generated at build time
// Re-export EMBEDDED_VALIDATOR_KEY as public for debugging
pub use self::embedded_keys::EMBEDDED_VALIDATOR_KEY;

mod embedded_keys {
    include!(concat!(env!("OUT_DIR"), "/embedded_keys.rs"));
}

pub fn get_current_binary_path() -> Result<std::path::PathBuf> {
    std::env::current_exe().context("Failed to get current executable path")
}

pub fn calculate_binary_checksum() -> Result<String> {
    let current_exe = std::env::current_exe().context("Failed to get current executable path")?;

    let mut file = File::open(&current_exe)
        .context("Failed to open current executable for checksum calculation")?;

    let mut hasher = Sha256::new();
    let mut buffer = [0; 8192];

    loop {
        let bytes_read = file
            .read(&mut buffer)
            .context("Failed to read executable file")?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    let hash = hasher.finalize();
    Ok(hex::encode(hash))
}

pub fn extract_embedded_key() -> Result<VerifyingKey> {
    let public_key = VerifyingKey::from_sec1_bytes(embedded_keys::EMBEDDED_VALIDATOR_KEY)
        .with_context(|| {
            format!(
                "Failed to parse embedded public key. Key length: {} bytes, first byte: 0x{:02x}",
                embedded_keys::EMBEDDED_VALIDATOR_KEY.len(),
                embedded_keys::EMBEDDED_VALIDATOR_KEY.first().unwrap_or(&0)
            )
        })?;

    tracing::info!(
        "Using embedded validator key (compressed): {}",
        hex::encode(embedded_keys::EMBEDDED_VALIDATOR_KEY)
    );
    Ok(public_key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_embedded_key() {
        let key = extract_embedded_key();
        assert!(key.is_ok());
    }
}
