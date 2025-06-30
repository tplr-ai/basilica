//! Test embedded key parsing

fn main() {
    // Simulate what would be embedded
    let key_hex = "02b161da8b6da0666755637c39b056c407f15335971d70c38ed65470d2ccac9d8b";
    let key_bytes = hex::decode(key_hex).expect("valid hex");

    println!("Testing key parsing:");
    println!("  Hex: {key_hex}");
    println!("  Length: {} bytes", key_bytes.len());
    println!("  First byte: 0x{:02x}", key_bytes[0]);

    // Test parsing
    match p256::ecdsa::VerifyingKey::from_sec1_bytes(&key_bytes) {
        Ok(key) => {
            println!("✓ Successfully parsed as P256 public key");
            let point = key.to_encoded_point(true);
            println!("  Compressed: {}", hex::encode(point.as_bytes()));
        }
        Err(e) => {
            println!("✗ Failed to parse: {e}");
        }
    }
}
