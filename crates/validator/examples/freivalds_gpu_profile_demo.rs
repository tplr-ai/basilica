//! Demonstration of GPU profile-aware Freivalds validation
//!
//! This example shows how validators can automatically detect executor GPU
//! capabilities and adapt challenge parameters accordingly.

use anyhow::Result;
use common::ssh::SshConnectionDetails;
use std::path::PathBuf;
use std::time::Duration;
use validator::validation::freivalds_validator::{FreivaldsGpuValidator, FreivaldsValidatorConfig};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== Freivalds GPU Profile-Aware Validation Demo ===\n");

    // Configuration
    let config = FreivaldsValidatorConfig {
        gpu_attestor_path: PathBuf::from("./crates/gpu-attestor"),
        temp_dir: PathBuf::from("/tmp/freivalds_demo"),
        ssh_timeout: Duration::from_secs(300),
        max_matrix_size: 8192,
        num_spot_checks: 20,
    };

    // Example SSH connection (would come from executor registration in practice)
    let connection = SshConnectionDetails {
        host: "executor.example.com".to_string(),
        username: "ubuntu".to_string(),
        port: 22,
        private_key_path: PathBuf::from("/home/validator/.ssh/id_rsa"),
        timeout: Duration::from_secs(30),
    };

    // Create validator
    let validator = FreivaldsGpuValidator::new(config)?;

    // Session ID for tracking
    let session_id = format!(
        "demo_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_millis()
    );

    println!("📊 Step 1: Traditional static challenge generation");
    println!("─────────────────────────────────────────────────");

    // Generate traditional static challenge
    let static_challenge = validator.generate_challenge(
        format!("static_{session_id}"),
        2048, // Fixed matrix size
        4,    // Assumed GPU count
    );

    println!("Static Challenge Parameters:");
    println!(
        "  • Matrix size: {}×{}",
        static_challenge.n, static_challenge.n
    );
    println!("  • Expected GPUs: {}", static_challenge.expected_gpu_count);
    println!(
        "  • Computation timeout: {}ms",
        static_challenge.computation_timeout_ms
    );
    println!(
        "  • Protocol timeout: {}ms",
        static_challenge.protocol_timeout_ms
    );
    println!("  • Based on: H100 benchmark averages");

    println!("\n🤖 Step 2: GPU profile-aware challenge generation");
    println!("─────────────────────────────────────────────────");

    // Simulate GPU profile query (in real scenario, this would SSH to executor)
    println!("Querying executor GPU profile via SSH...");

    // For demo purposes, show what would happen with different GPU configurations
    let gpu_configs = vec![
        ("8× H100 PCIe (DataCenter)", 8, 350.2, 4096),
        ("4× RTX 4090 (Professional)", 4, 330.4, 2048),
        ("2× RTX 3080 (Consumer)", 2, 60.0, 1024),
        ("1× GTX 1660 (Entry)", 1, 12.0, 512),
    ];

    for (config_name, gpu_count, compute_power, optimal_size) in gpu_configs {
        println!("\n🖥️  Configuration: {config_name}");
        println!("  • Total compute: {compute_power:.1} TFLOPS");
        println!("  • Optimal matrix size: {optimal_size}");

        // Calculate adaptive timeouts
        let base_time = match optimal_size {
            512 => 30,
            1024 => 120,
            2048 => 600,
            4096 => 4500,
            _ => 1000,
        };

        let parallel_efficiency = 0.75;
        let adjusted_time = (base_time as f32 / (gpu_count as f32 * parallel_efficiency)) as u32;
        let safety_factor = if compute_power > 300.0 { 1.5 } else { 2.0 };
        let computation_timeout = (adjusted_time as f32 * safety_factor) as u32;
        let protocol_timeout = computation_timeout + 150;

        println!("  • Adaptive computation timeout: {computation_timeout}ms");
        println!("  • Adaptive protocol timeout: {protocol_timeout}ms");
        println!(
            "  • Efficiency gain: {:.1}%",
            ((static_challenge.computation_timeout_ms as f32 - computation_timeout as f32)
                / static_challenge.computation_timeout_ms as f32)
                * 100.0
        );
    }

    println!("\n✨ Step 3: Integrated execution flow");
    println!("─────────────────────────────────────────────");

    println!("In production, execute_challenge_with_profiling() would:");
    println!("1. Upload gpu-attestor binary to executor");
    println!("2. Run --detect-gpus-json to get GPU profile");
    println!("3. Calculate optimal matrix size and timeouts");
    println!("4. Generate adapted Freivalds challenge");
    println!("5. Execute challenge with optimal parameters");
    println!("6. Return commitment for verification");

    println!("\n📈 Benefits of GPU-aware validation:");
    println!("─────────────────────────────────────────────");
    println!("• Automatic adaptation to executor hardware");
    println!("• Optimal resource utilization");
    println!("• Reduced false timeouts on slower GPUs");
    println!("• Faster validation on high-end GPUs");
    println!("• Support for heterogeneous GPU farms");
    println!("• Future-proof for new GPU models");

    println!("\n🔐 Security considerations:");
    println!("─────────────────────────────────────────────");
    println!("• GPU profile is queried via validator's own binary");
    println!("• Profile data is cached to prevent DoS");
    println!("• Timeouts have safety factors to prevent gaming");
    println!("• Matrix size is capped by validator configuration");

    Ok(())
}
