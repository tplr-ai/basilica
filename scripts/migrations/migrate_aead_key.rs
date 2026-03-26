//! AEAD Key Migration Script
//!
//! This script migrates encrypted mnemonics from one AEAD key to another.
//!
//! USAGE:
//!   1. Set environment variables:
//!      - DATABASE_URL: PostgreSQL connection string
//!      - OLD_AEAD_KEY_HEX: The current encryption key (likely all zeros)
//!      - NEW_AEAD_KEY_HEX: The new encryption key from Secrets Manager
//!      - DRY_RUN: Set to "false" to actually perform migration (default: true)
//!
//!   2. Run: cargo run --bin migrate-aead-key
//!
//! SAFETY:
//!   - Always run with DRY_RUN=true first to verify
//!   - Back up the database before running
//!   - Run during a maintenance window with payments service stopped

use anyhow::{Context, Result};
use sqlx::{postgres::PgPoolOptions, Row};
use std::env;

// Inline AEAD implementation to avoid circular dependencies
mod aead {
    use aes_gcm::{
        aead::{Aead as AeadTrait, KeyInit},
        Aes256Gcm, Nonce,
    };
    use anyhow::{anyhow, Result};
    use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
    use rand::RngCore;

    pub struct Aead {
        cipher: Aes256Gcm,
    }

    impl Aead {
        pub fn new(key_hex: &str) -> Result<Self> {
            let key_bytes = hex::decode(key_hex)
                .map_err(|e| anyhow!("Invalid hex key: {}", e))?;

            if key_bytes.len() != 32 {
                return Err(anyhow!("Key must be 32 bytes (256 bits), got {}", key_bytes.len()));
            }

            let cipher = Aes256Gcm::new_from_slice(&key_bytes)
                .map_err(|e| anyhow!("Failed to create cipher: {}", e))?;

            Ok(Self { cipher })
        }

        pub fn decrypt(&self, ciphertext_b64: &str) -> Result<String> {
            let parts: Vec<&str> = ciphertext_b64.split(':').collect();
            if parts.len() != 2 {
                return Err(anyhow!("Invalid ciphertext format, expected 'nonce:ciphertext'"));
            }

            let nonce_bytes = BASE64.decode(parts[0])
                .map_err(|e| anyhow!("Invalid nonce base64: {}", e))?;
            let ciphertext = BASE64.decode(parts[1])
                .map_err(|e| anyhow!("Invalid ciphertext base64: {}", e))?;

            if nonce_bytes.len() != 12 {
                return Err(anyhow!("Nonce must be 12 bytes, got {}", nonce_bytes.len()));
            }

            let nonce = Nonce::from_slice(&nonce_bytes);
            let plaintext = self.cipher
                .decrypt(nonce, ciphertext.as_ref())
                .map_err(|_| anyhow!("Decryption failed - wrong key or corrupted data"))?;

            String::from_utf8(plaintext)
                .map_err(|e| anyhow!("Decrypted data is not valid UTF-8: {}", e))
        }

        pub fn encrypt(&self, plaintext: &str) -> Result<String> {
            let mut nonce_bytes = [0u8; 12];
            rand::thread_rng().fill_bytes(&mut nonce_bytes);
            let nonce = Nonce::from_slice(&nonce_bytes);

            let ciphertext = self.cipher
                .encrypt(nonce, plaintext.as_bytes())
                .map_err(|e| anyhow!("Encryption failed: {}", e))?;

            let nonce_b64 = BASE64.encode(&nonce_bytes);
            let ciphertext_b64 = BASE64.encode(&ciphertext);

            Ok(format!("{}:{}", nonce_b64, ciphertext_b64))
        }
    }
}

#[derive(Debug)]
#[allow(dead_code)]
struct DepositAccount {
    id: i64,
    account_hex: String,
    address_ss58: String,
    encrypted_mnemonic: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Load configuration from environment
    let database_url = env::var("DATABASE_URL")
        .context("DATABASE_URL environment variable required")?;

    let old_key_hex = env::var("OLD_AEAD_KEY_HEX")
        .unwrap_or_else(|_| "0000000000000000000000000000000000000000000000000000000000000000".to_string());

    let new_key_hex = env::var("NEW_AEAD_KEY_HEX")
        .context("NEW_AEAD_KEY_HEX environment variable required")?;

    let dry_run = env::var("DRY_RUN")
        .map(|v| v.to_lowercase() != "false")
        .unwrap_or(true);

    println!("=== AEAD Key Migration Script ===");
    println!("Dry run: {}", dry_run);
    println!("Old key (first 8 chars): {}...", &old_key_hex[..8]);
    println!("New key (first 8 chars): {}...", &new_key_hex[..8]);
    println!();

    if old_key_hex == new_key_hex {
        println!("ERROR: Old and new keys are identical. Nothing to migrate.");
        return Ok(());
    }

    // Validate keys
    let old_aead = aead::Aead::new(&old_key_hex)
        .context("Failed to create AEAD with old key")?;
    let new_aead = aead::Aead::new(&new_key_hex)
        .context("Failed to create AEAD with new key")?;

    println!("Both keys validated successfully.");

    // Connect to database
    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&database_url)
        .await
        .context("Failed to connect to database")?;

    println!("Connected to database.");

    // Fetch all deposit accounts
    let accounts: Vec<DepositAccount> = sqlx::query(
        r#"SELECT id, account_hex, address_ss58, encrypted_mnemonic
           FROM deposit_accounts
           ORDER BY id"#
    )
    .fetch_all(&pool)
    .await
    .context("Failed to fetch deposit accounts")?
    .into_iter()
    .map(|row| DepositAccount {
        id: row.get("id"),
        account_hex: row.get("account_hex"),
        address_ss58: row.get("address_ss58"),
        encrypted_mnemonic: row.get("encrypted_mnemonic"),
    })
    .collect();

    println!("Found {} deposit accounts to migrate.", accounts.len());
    println!();

    let mut success_count = 0;
    let mut failure_count = 0;
    let mut already_migrated = 0;

    for account in &accounts {
        let account_preview = &account.address_ss58[..10];

        // Try decrypting with old key first
        match old_aead.decrypt(&account.encrypted_mnemonic) {
            Ok(mnemonic) => {
                // Validate mnemonic (basic check - should be 12 or 24 words)
                let word_count = mnemonic.split_whitespace().count();
                if word_count != 12 && word_count != 24 {
                    println!("[WARN] Account {}: Decrypted but invalid word count ({})",
                             account_preview, word_count);
                    failure_count += 1;
                    continue;
                }

                // Re-encrypt with new key
                let new_encrypted = new_aead.encrypt(&mnemonic)
                    .context("Failed to re-encrypt mnemonic")?;

                if dry_run {
                    println!("[DRY RUN] Account {}: Would migrate (mnemonic: {} words)",
                             account_preview, word_count);
                } else {
                    // Update database
                    sqlx::query(
                        "UPDATE deposit_accounts SET encrypted_mnemonic = $1 WHERE id = $2"
                    )
                    .bind(&new_encrypted)
                    .bind(account.id)
                    .execute(&pool)
                    .await
                    .context("Failed to update account")?;

                    println!("[MIGRATED] Account {}: Successfully re-encrypted", account_preview);
                }
                success_count += 1;
            }
            Err(_) => {
                // Try decrypting with new key - maybe already migrated?
                match new_aead.decrypt(&account.encrypted_mnemonic) {
                    Ok(_) => {
                        println!("[SKIP] Account {}: Already encrypted with new key", account_preview);
                        already_migrated += 1;
                    }
                    Err(_) => {
                        println!("[ERROR] Account {}: Cannot decrypt with either key!", account_preview);
                        failure_count += 1;
                    }
                }
            }
        }
    }

    println!();
    println!("=== Migration Summary ===");
    println!("Total accounts: {}", accounts.len());
    println!("Successfully migrated: {}", success_count);
    println!("Already migrated: {}", already_migrated);
    println!("Failed: {}", failure_count);

    if dry_run {
        println!();
        println!("This was a DRY RUN. No changes were made.");
        println!("To perform actual migration, run with DRY_RUN=false");
    }

    if failure_count > 0 {
        println!();
        println!("WARNING: {} accounts failed to migrate!", failure_count);
        println!("These accounts may have corrupted data or were encrypted with a different key.");
    }

    Ok(())
}
