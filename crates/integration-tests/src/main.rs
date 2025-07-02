//! Integration Tests Runner
//!
//! This binary can be used to run specific integration tests with proper setup

use anyhow::Result;

fn main() -> Result<()> {
    println!("Basilisk Integration Tests");
    println!("=========================");
    println!();
    println!("Available tests:");
    println!("  - GPU PoW End-to-End: cargo test gpu_pow_e2e --test gpu_pow_e2e -- --nocapture");
    println!("  - Miner-Executor Flow: cargo test miner_executor_flow --test miner_executor_flow -- --nocapture");
    println!();
    println!("To run all integration tests:");
    println!("  cargo test -- --nocapture");
    println!();
    println!("Note: GPU tests require a CUDA-capable GPU and the gpu-attestor binary built.");
    println!("      Set GPU_ATTESTOR_PATH environment variable if needed.");
    
    Ok(())
}
