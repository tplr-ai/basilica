#[cfg(test)]
mod freivalds_integration_tests {
    use common::ssh::SshConnectionDetails;
    use std::path::PathBuf;
    use std::time::Duration;
    use validator::validation::freivalds_validator::{
        FreivaldsGpuValidator, FreivaldsValidatorConfig,
    };

    #[tokio::test]
    #[ignore] // Run with: cargo test --ignored freivalds_integration_test
    async fn test_freivalds_validation_flow() {
        // Configure validator
        let config = FreivaldsValidatorConfig {
            gpu_attestor_path: PathBuf::from("../../crates/gpu-attestor"),
            temp_dir: PathBuf::from("/tmp/freivalds_test"),
            ssh_timeout: Duration::from_secs(300),
            max_matrix_size: 4096,
            num_spot_checks: 10,
        };

        let validator = FreivaldsGpuValidator::new(config).expect("Failed to create validator");

        // Generate challenge
        let challenge = validator.generate_challenge(
            "test_session_123".to_string(),
            1024, // 1024x1024 matrices
            2,    // Expect 2 GPUs
        );

        println!("Generated Freivalds challenge:");
        println!("  Session ID: {}", challenge.session_id);
        println!("  Matrix size: {}x{}", challenge.n, challenge.n);
        println!("  Expected GPUs: {}", challenge.expected_gpu_count);

        // In a real scenario, you would have SSH connection details
        let connection = SshConnectionDetails {
            host: "executor.example.com".to_string(),
            username: "validator_user".to_string(),
            port: 22,
            private_key_path: PathBuf::from("/path/to/key"),
            timeout: Duration::from_secs(30),
        };

        // Execute challenge (would fail without real SSH connection)
        match validator.execute_challenge(&connection, &challenge).await {
            Ok(commitment) => {
                println!("\nReceived commitment:");
                println!("  Merkle root: {}", hex::encode(&commitment.merkle_root));
                println!("  Row count: {}", commitment.row_count);

                if let Some(metadata) = &commitment.metadata {
                    println!("  Execution time: {}ms", metadata.execution_time_ms);
                    println!("  GPUs used: {}", metadata.gpus.len());
                }

                // Verify the result
                match validator
                    .verify_result(&connection, &challenge, &commitment)
                    .await
                {
                    Ok(result) => {
                        println!("\nVerification result:");
                        println!("  Verified: {}", result.verified);
                        println!("  Freivalds valid: {}", result.freivalds_valid);
                        println!(
                            "  Spot checks: {}/{}",
                            result.spot_checks_passed, result.spot_checks_performed
                        );

                        if let Some(metrics) = &result.metrics {
                            println!(
                                "  Computation saved: {:.1}%",
                                metrics.computation_saved_percent
                            );
                        }
                    }
                    Err(e) => {
                        eprintln!("Verification failed: {e}");
                    }
                }
            }
            Err(e) => {
                eprintln!("Challenge execution failed: {e}");
            }
        }
    }

    #[test]
    fn test_challenge_generation() {
        let config = FreivaldsValidatorConfig::default();
        let validator = FreivaldsGpuValidator::new(config).unwrap();

        let challenge = validator.generate_challenge("test_session".to_string(), 2048, 4);

        assert_eq!(challenge.session_id, "test_session");
        assert_eq!(challenge.n, 2048);
        assert_eq!(challenge.expected_gpu_count, 4);
        assert_eq!(challenge.master_seed.len(), 16);
    }
}
