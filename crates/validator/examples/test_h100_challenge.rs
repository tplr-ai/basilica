//! Test H100 challenge generation

use anyhow::Result;
use validator::validation::challenge_generator::ChallengeGenerator;

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("Testing H100 Challenge Generation\n");

    let generator = ChallengeGenerator::new();

    // Generate challenge for H100 80GB
    let challenge =
        generator.generate_challenge("NVIDIA H100", 80, Some("test_nonce".to_string()))?;

    // Calculate memory usage
    let matrix_size_bytes = (challenge.matrix_dim as u64 * challenge.matrix_dim as u64 * 8) as f64;
    let total_memory = challenge.num_matrices as f64 * matrix_size_bytes;
    let memory_gb = total_memory / (1024.0 * 1024.0 * 1024.0);

    println!("H100 Challenge Parameters:");
    println!("- GPU: NVIDIA H100");
    println!("- VRAM: 80GB");
    println!(
        "- Matrix dimension: {}x{}",
        challenge.matrix_dim, challenge.matrix_dim
    );
    println!("- Number of matrices: {}", challenge.num_matrices);
    println!("- Memory usage: {memory_gb:.2}GB");
    println!(
        "- Memory per matrix: {:.2}MB",
        matrix_size_bytes / (1024.0 * 1024.0)
    );
    println!(
        "- Expected duration: {}s",
        challenge.expected_duration_seconds
    );

    println!("\nAnalysis:");
    if challenge.matrix_dim == 1024 {
        println!("✓ Using optimized 1024x1024 matrices for H100");
    } else {
        println!(
            "✗ Not using optimized matrix size! Got {}x{}",
            challenge.matrix_dim, challenge.matrix_dim
        );
    }

    if memory_gb > 65.0 && memory_gb < 75.0 {
        println!(
            "✓ Memory usage is appropriate for H100 ({}% of 80GB)",
            (memory_gb / 80.0 * 100.0) as i32
        );
    } else {
        println!("✗ Memory usage is not optimal: {memory_gb:.2}GB");
    }

    if challenge.num_matrices > 8000 && challenge.num_matrices < 10000 {
        println!("✓ Matrix count is in expected range for H100");
    } else {
        println!(
            "✗ Matrix count {} is outside expected range",
            challenge.num_matrices
        );
    }

    Ok(())
}
