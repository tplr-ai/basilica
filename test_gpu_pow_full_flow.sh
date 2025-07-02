#!/bin/bash
# Test the full GPU PoW flow with real validator components

set -e

echo "=== GPU PoW Full Flow Test ==="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Paths
GPU_ATTESTOR="./target/debug/gpu-attestor"
WORK_DIR=$(pwd)

# Step 1: Build validator test binary
echo -e "${YELLOW}Step 1: Building validator test binary...${NC}"
cd crates/validator
cargo build --bin validator_gpu_pow_test 2>/dev/null || {
    # Create a simple test binary if it doesn't exist
    cat > src/bin/validator_gpu_pow_test.rs << 'EOF'
use anyhow::Result;
use validator::validation::{
    challenge_generator::ChallengeGenerator,
    gpu_validator::GpuValidator,
};
use protocol::basilca::common::v1::ChallengeResult;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    
    println!("=== Validator GPU PoW Test ===");
    
    // Initialize validator GPU
    let gpu_attestor_path = std::env::var("GPU_ATTESTOR_PATH")
        .unwrap_or_else(|_| "../../target/debug/gpu-attestor".to_string());
    
    let mut gpu_validator = GpuValidator::new(gpu_attestor_path.clone());
    gpu_validator.initialize().await?;
    
    let validator_gpu = gpu_validator.get_gpu_model()
        .ok_or_else(|| anyhow::anyhow!("No GPU detected on validator"))?;
    
    println!("Validator GPU: {}", validator_gpu);
    
    // Generate challenge
    let generator = ChallengeGenerator::new();
    let challenge = generator.generate_challenge(
        validator_gpu,
        80, // Assume 80GB for H100
        Some("test_full_flow_123".to_string()),
    )?;
    
    println!("\nGenerated challenge:");
    println!("  Seed: {}", challenge.gpu_pow_seed);
    println!("  Matrix dim: {}x{}", challenge.matrix_dim, challenge.matrix_dim);
    println!("  Num matrices: {}", challenge.num_matrices);
    println!("  Matrix A index: {}", challenge.matrix_a_index);
    println!("  Matrix B index: {}", challenge.matrix_b_index);
    
    // Simulate miner execution
    println!("\nSimulating miner execution...");
    let challenge_json = serde_json::to_string(&challenge)?;
    let challenge_b64 = base64::encode(&challenge_json);
    
    let output = std::process::Command::new(&gpu_attestor_path)
        .arg("--challenge")
        .arg(&challenge_b64)
        .output()?;
    
    if !output.status.success() {
        anyhow::bail!("Miner execution failed: {}", String::from_utf8_lossy(&output.stderr));
    }
    
    let result: ChallengeResult = serde_json::from_slice(&output.stdout)?;
    
    println!("\nMiner result:");
    println!("  Checksum: {}", hex::encode(&result.result_checksum));
    println!("  Execution time: {} ms", result.execution_time_ms);
    println!("  GPU model: {}", result.gpu_model);
    println!("  VRAM allocated: {} MB", result.vram_allocated_mb);
    
    // Verify result
    println!("\nVerifying challenge result...");
    let is_valid = gpu_validator.verify_challenge(&challenge, &result).await?;
    
    if is_valid {
        println!("\n✓ Challenge verification PASSED!");
        Ok(())
    } else {
        anyhow::bail!("Challenge verification FAILED!")
    }
}
EOF
    
    cargo build --bin validator_gpu_pow_test
}
cd "$WORK_DIR"

# Step 2: Check GPU availability
echo -e "\n${YELLOW}Step 2: Checking GPU availability...${NC}"
$GPU_ATTESTOR --info > gpu_info.json 2>&1 || {
    echo -e "${RED}Failed to detect GPU${NC}"
    exit 1
}

if [ -f gpu_info.json ]; then
    GPU_NAME=$(jq -r '.gpus[0].name' gpu_info.json 2>/dev/null || echo "Unknown")
    echo -e "${GREEN}✓ GPU detected: $GPU_NAME${NC}"
else
    echo -e "${RED}No GPU information available${NC}"
    exit 1
fi

# Step 3: Run the full flow test
echo -e "\n${YELLOW}Step 3: Running full GPU PoW flow test...${NC}"
GPU_ATTESTOR_PATH="$GPU_ATTESTOR" ./target/debug/validator_gpu_pow_test

echo -e "\n${GREEN}=== GPU PoW Full Flow Test Complete ===${NC}"

# Cleanup
rm -f gpu_info.json