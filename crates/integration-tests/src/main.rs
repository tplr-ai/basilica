//! Integration Tests Runner
//!
//! This binary can be used to run specific integration tests with proper setup

use anyhow::Result;

fn main() -> Result<()> {
    println!("Basilica Integration Tests");
    println!("==========================");
    println!();
    println!("Available tests:");
    println!("  - Authentication E2E: cargo test --test auth_e2e_tests -- --nocapture");
    println!("  - Authentication gRPC: cargo test --test grpc_auth_integration -- --nocapture");
    println!("  - Authentication Security: cargo test --test auth_security_tests -- --nocapture");
    println!("  - GPU PoW End-to-End: cargo test gpu_pow_e2e --test gpu_pow_e2e -- --nocapture");
    println!(
        "  - Miner-Node Flow: cargo test miner_node_flow --test miner_node_flow -- --nocapture"
    );
    println!();
    println!("To run all integration tests:");
    println!("  cargo test -- --nocapture");
    println!();
    println!("To run authentication tests only:");
    println!("  cargo test auth -- --nocapture");
    println!();
    println!("Note: GPU tests require a CUDA-capable GPU and the gpu-attestor binary built.");
    println!("      Set GPU_ATTESTOR_PATH environment variable if needed.");

    Ok(())
}
