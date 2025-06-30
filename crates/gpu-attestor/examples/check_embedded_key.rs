//! Check if embedded key is present in the binary

fn main() {
    // Try to access the embedded key directly
    match std::panic::catch_unwind(|| {
        // This will panic if EMBEDDED_VALIDATOR_KEY is not available
        gpu_attestor::integrity::EMBEDDED_VALIDATOR_KEY
    }) {
        Ok(key) => {
            println!("Embedded key found!");
            println!("  Length: {} bytes", key.len());
            println!("  Hex: {}", hex::encode(key));
            if !key.is_empty() {
                println!("  First byte: 0x{:02x}", key[0]);
            }

            // Try to parse it
            match p256::ecdsa::VerifyingKey::from_sec1_bytes(key) {
                Ok(_) => println!("  ✓ Valid P256 public key"),
                Err(e) => println!("  ✗ Invalid P256 key: {e}"),
            }
        }
        Err(_) => {
            println!("ERROR: EMBEDDED_VALIDATOR_KEY not found!");
            println!("This means the build process didn't embed a key.");
        }
    }
}
