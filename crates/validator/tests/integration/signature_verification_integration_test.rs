use anyhow::Result;
use common::crypto::{generate_p256_keypair, sign_p256, verify_p256_signature};
use p256::{
    ecdsa::{signature::Signer, SigningKey, VerifyingKey},
    elliptic_curve::rand_core::OsRng,
    pkcs8::{DecodePrivateKey, EncodePrivateKey, EncodePublicKey, LineEnding},
    PublicKey,
};
use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;
use validator::validation::{
    AttestationData, AttestationIntegration, AttestationReport, GpuInfo, HardwareInfo,
    SignatureVerifier, VdfResult,
};

#[tokio::test]
async fn test_p256_key_generation_and_serialization() -> Result<()> {
    let temp_dir = TempDir::new()?;
    
    // Generate P256 key pair
    let signing_key = SigningKey::random(&mut OsRng);
    let verifying_key = signing_key.verifying_key();
    
    // Export to PEM
    let private_pem = signing_key
        .to_pkcs8_pem(LineEnding::LF)?
        .to_string();
    
    let public_pem = verifying_key
        .to_public_key_pem(LineEnding::LF)?;
    
    // Save to files
    let private_path = temp_dir.path().join("private_key.pem");
    let public_path = temp_dir.path().join("public_key.pem");
    
    fs::write(&private_path, &private_pem)?;
    fs::write(&public_path, &public_pem)?;
    
    // Load back and verify
    let loaded_private = SigningKey::from_pkcs8_pem(&private_pem)?;
    let loaded_public = VerifyingKey::from_public_key_pem(&public_pem)?;
    
    assert_eq!(
        loaded_private.verifying_key().to_encoded_point(true),
        signing_key.verifying_key().to_encoded_point(true),
        "Private key should round-trip correctly"
    );
    
    assert_eq!(
        loaded_public.to_encoded_point(true),
        verifying_key.to_encoded_point(true),
        "Public key should round-trip correctly"
    );
    
    Ok(())
}

#[tokio::test]
async fn test_attestation_signature_verification() -> Result<()> {
    // Generate test keys
    let signing_key = SigningKey::random(&mut OsRng);
    let verifying_key = signing_key.verifying_key();
    
    // Create test attestation data
    let attestation_data = AttestationData {
        executor_id: "test-executor-123".to_string(),
        timestamp: chrono::Utc::now(),
        hardware_info: HardwareInfo {
            cpu_count: 16,
            cpu_model: "AMD EPYC 7763".to_string(),
            memory_gb: 128,
            disk_gb: 2000,
            architecture: "x86_64".to_string(),
        },
        gpu_info: vec![
            GpuInfo {
                index: 0,
                name: "NVIDIA RTX 4090".to_string(),
                uuid: "GPU-12345678-1234-1234-1234-123456789012".to_string(),
                memory_mb: 24576,
                compute_capability: "8.9".to_string(),
                driver_version: "535.129.03".to_string(),
            },
            GpuInfo {
                index: 1,
                name: "NVIDIA RTX 4090".to_string(),
                uuid: "GPU-87654321-4321-4321-4321-210987654321".to_string(),
                memory_mb: 24576,
                compute_capability: "8.9".to_string(),
                driver_version: "535.129.03".to_string(),
            },
        ],
        os_info: serde_json::json!({
            "name": "Ubuntu",
            "version": "22.04.3 LTS",
            "kernel": "5.15.0-88-generic"
        }),
        docker_info: Some(serde_json::json!({
            "version": "24.0.7",
            "runtime": "nvidia",
            "containers_running": 5
        })),
        network_info: serde_json::json!({
            "public_ip": "203.0.113.42",
            "bandwidth_mbps": 10000,
            "latency_ms": 2.5
        }),
        vdf_result: Some(VdfResult {
            algorithm: "simple".to_string(),
            difficulty: 1000,
            input: "test-input-12345".to_string(),
            output: "computed-output-67890".to_string(),
            iterations: 1000,
            duration_ms: 125,
        }),
        integrity_hash: "1234567890abcdef".to_string(),
    };
    
    // Serialize attestation data
    let attestation_json = serde_json::to_string(&attestation_data)?;
    let attestation_bytes = attestation_json.as_bytes();
    
    // Sign the attestation
    let signature: p256::ecdsa::Signature = signing_key.sign(attestation_bytes);
    let signature_bytes = signature.to_bytes();
    
    // Create attestation report
    let report = AttestationReport {
        attestation_data: attestation_data.clone(),
        signature: signature_bytes.to_vec(),
        public_key: verifying_key.to_encoded_point(true).as_bytes().to_vec(),
    };
    
    // Create verifier
    let verifier = SignatureVerifier::new(verifying_key);
    
    // Verify signature
    let is_valid = verifier.verify_attestation(&report)?;
    assert!(is_valid, "Signature should be valid");
    
    // Test with tampered data
    let mut tampered_report = report.clone();
    tampered_report.attestation_data.executor_id = "tampered-executor".to_string();
    
    let tampered_result = verifier.verify_attestation(&tampered_report);
    assert!(tampered_result.is_err() || !tampered_result.unwrap(), 
        "Tampered signature should be invalid");
    
    // Test with wrong signature
    let mut wrong_sig_report = report.clone();
    wrong_sig_report.signature[0] ^= 0xFF; // Flip bits
    
    let wrong_sig_result = verifier.verify_attestation(&wrong_sig_report);
    assert!(wrong_sig_result.is_err() || !wrong_sig_result.unwrap(),
        "Wrong signature should be invalid");
    
    Ok(())
}

#[tokio::test]
async fn test_attestation_integration_parsing() -> Result<()> {
    let temp_dir = TempDir::new()?;
    
    // Generate keys
    let signing_key = SigningKey::random(&mut OsRng);
    let verifying_key = signing_key.verifying_key();
    
    // Create test attestation files
    let attestation_data = AttestationData {
        executor_id: "integration-test-executor".to_string(),
        timestamp: chrono::Utc::now(),
        hardware_info: HardwareInfo {
            cpu_count: 32,
            cpu_model: "Intel Xeon Platinum 8375C".to_string(),
            memory_gb: 256,
            disk_gb: 4000,
            architecture: "x86_64".to_string(),
        },
        gpu_info: vec![
            GpuInfo {
                index: 0,
                name: "NVIDIA A100-SXM4-80GB".to_string(),
                uuid: "GPU-a100-0000-0000-0000-000000000000".to_string(),
                memory_mb: 81920,
                compute_capability: "8.0".to_string(),
                driver_version: "535.129.03".to_string(),
            },
        ],
        os_info: serde_json::json!({
            "name": "Ubuntu",
            "version": "22.04.3 LTS",
            "kernel": "5.15.0-88-generic"
        }),
        docker_info: Some(serde_json::json!({
            "version": "24.0.7",
            "runtime": "nvidia"
        })),
        network_info: serde_json::json!({
            "public_ip": "198.51.100.42",
            "bandwidth_mbps": 25000
        }),
        vdf_result: Some(VdfResult {
            algorithm: "advanced".to_string(),
            difficulty: 5000,
            input: "integration-test-input".to_string(),
            output: "integration-test-output".to_string(),
            iterations: 5000,
            duration_ms: 523,
        }),
        integrity_hash: "fedcba0987654321".to_string(),
    };
    
    // Write attestation JSON
    let json_path = temp_dir.path().join("attestation.json");
    let json_content = serde_json::to_string_pretty(&attestation_data)?;
    fs::write(&json_path, &json_content)?;
    
    // Sign the attestation
    let signature: p256::ecdsa::Signature = signing_key.sign(json_content.as_bytes());
    
    // Write signature
    let sig_path = temp_dir.path().join("attestation.sig");
    fs::write(&sig_path, signature.to_bytes())?;
    
    // Write public key
    let pub_path = temp_dir.path().join("attestation.pub");
    let pub_key_bytes = verifying_key.to_encoded_point(true).as_bytes();
    fs::write(&pub_path, pub_key_bytes)?;
    
    // Parse attestation files
    let integration = AttestationIntegration::new(verifying_key);
    let parsed_report = integration.parse_attestation_files(
        &json_path,
        &sig_path,
        &pub_path
    ).await?;
    
    // Verify parsed data
    assert_eq!(parsed_report.attestation_data.executor_id, "integration-test-executor");
    assert_eq!(parsed_report.attestation_data.hardware_info.cpu_count, 32);
    assert_eq!(parsed_report.attestation_data.gpu_info.len(), 1);
    assert_eq!(parsed_report.attestation_data.gpu_info[0].name, "NVIDIA A100-SXM4-80GB");
    
    // Verify signature is valid
    let verifier = SignatureVerifier::new(verifying_key);
    let is_valid = verifier.verify_attestation(&parsed_report)?;
    assert!(is_valid, "Parsed attestation signature should be valid");
    
    Ok(())
}

#[tokio::test]
async fn test_binary_integrity_verification() -> Result<()> {
    let temp_dir = TempDir::new()?;
    
    // Create test binary
    let binary_path = temp_dir.path().join("gpu-attestor");
    let binary_content = b"This is a test GPU attestor binary content";
    fs::write(&binary_path, binary_content)?;
    
    // Calculate expected hash
    let expected_hash = blake3::hash(binary_content);
    let expected_hash_hex = expected_hash.to_hex().to_string();
    
    // Create integrity verifier
    let integrity_verifier = validator::validation::IntegrityVerifier::new(vec![
        expected_hash_hex.clone(),
        "another-valid-hash-12345".to_string(), // Additional valid hash
    ]);
    
    // Verify correct binary
    let is_valid = integrity_verifier.verify_binary(&binary_path)?;
    assert!(is_valid, "Binary with correct hash should be valid");
    
    // Test with modified binary
    let modified_path = temp_dir.path().join("gpu-attestor-modified");
    fs::write(&modified_path, b"This is modified content")?;
    
    let is_invalid = integrity_verifier.verify_binary(&modified_path)?;
    assert!(!is_invalid, "Modified binary should be invalid");
    
    // Test with non-existent file
    let missing_path = temp_dir.path().join("non-existent");
    let missing_result = integrity_verifier.verify_binary(&missing_path);
    assert!(missing_result.is_err(), "Non-existent file should error");
    
    Ok(())
}

#[tokio::test]
async fn test_key_rotation_scenario() -> Result<()> {
    // Generate old and new key pairs
    let old_signing_key = SigningKey::random(&mut OsRng);
    let old_verifying_key = old_signing_key.verifying_key();
    
    let new_signing_key = SigningKey::random(&mut OsRng);
    let new_verifying_key = new_signing_key.verifying_key();
    
    // Create attestation data
    let attestation_data = AttestationData {
        executor_id: "key-rotation-test".to_string(),
        timestamp: chrono::Utc::now(),
        hardware_info: HardwareInfo {
            cpu_count: 8,
            cpu_model: "Test CPU".to_string(),
            memory_gb: 32,
            disk_gb: 500,
            architecture: "x86_64".to_string(),
        },
        gpu_info: vec![],
        os_info: serde_json::json!({"name": "TestOS"}),
        docker_info: None,
        network_info: serde_json::json!({"test": true}),
        vdf_result: None,
        integrity_hash: "test-hash".to_string(),
    };
    
    let attestation_json = serde_json::to_string(&attestation_data)?;
    
    // Sign with old key
    let old_signature: p256::ecdsa::Signature = old_signing_key.sign(attestation_json.as_bytes());
    
    // Sign with new key
    let new_signature: p256::ecdsa::Signature = new_signing_key.sign(attestation_json.as_bytes());
    
    // Create reports
    let old_report = AttestationReport {
        attestation_data: attestation_data.clone(),
        signature: old_signature.to_bytes().to_vec(),
        public_key: old_verifying_key.to_encoded_point(true).as_bytes().to_vec(),
    };
    
    let new_report = AttestationReport {
        attestation_data: attestation_data.clone(),
        signature: new_signature.to_bytes().to_vec(),
        public_key: new_verifying_key.to_encoded_point(true).as_bytes().to_vec(),
    };
    
    // Verify with old key
    let old_verifier = SignatureVerifier::new(old_verifying_key);
    assert!(old_verifier.verify_attestation(&old_report)?, "Old key should verify old signature");
    assert!(old_verifier.verify_attestation(&new_report).is_err() || 
           !old_verifier.verify_attestation(&new_report).unwrap(),
           "Old key should not verify new signature");
    
    // Verify with new key
    let new_verifier = SignatureVerifier::new(new_verifying_key);
    assert!(new_verifier.verify_attestation(&new_report)?, "New key should verify new signature");
    assert!(new_verifier.verify_attestation(&old_report).is_err() ||
           !new_verifier.verify_attestation(&old_report).unwrap(),
           "New key should not verify old signature");
    
    Ok(())
}

#[tokio::test]
async fn test_concurrent_signature_verifications() -> Result<()> {
    let signing_key = SigningKey::random(&mut OsRng);
    let verifying_key = signing_key.verifying_key();
    
    // Create multiple attestations
    let mut handles = vec![];
    
    for i in 0..10 {
        let signing_key_clone = signing_key.clone();
        let verifying_key_clone = verifying_key.clone();
        
        let handle = tokio::spawn(async move {
            let attestation_data = AttestationData {
                executor_id: format!("concurrent-executor-{}", i),
                timestamp: chrono::Utc::now(),
                hardware_info: HardwareInfo {
                    cpu_count: 16,
                    cpu_model: "Test CPU".to_string(),
                    memory_gb: 64,
                    disk_gb: 1000,
                    architecture: "x86_64".to_string(),
                },
                gpu_info: vec![],
                os_info: serde_json::json!({"test": i}),
                docker_info: None,
                network_info: serde_json::json!({"concurrent": true}),
                vdf_result: None,
                integrity_hash: format!("hash-{}", i),
            };
            
            let json = serde_json::to_string(&attestation_data).unwrap();
            let signature: p256::ecdsa::Signature = signing_key_clone.sign(json.as_bytes());
            
            let report = AttestationReport {
                attestation_data,
                signature: signature.to_bytes().to_vec(),
                public_key: verifying_key_clone.to_encoded_point(true).as_bytes().to_vec(),
            };
            
            let verifier = SignatureVerifier::new(verifying_key_clone);
            verifier.verify_attestation(&report)
        });
        
        handles.push(handle);
    }
    
    // Wait for all verifications
    let results = futures::future::try_join_all(handles).await?;
    
    // All should succeed
    for (i, result) in results.iter().enumerate() {
        match result {
            Ok(is_valid) => assert!(*is_valid, "Concurrent verification {} should be valid", i),
            Err(e) => panic!("Concurrent verification {} failed: {}", i, e),
        }
    }
    
    Ok(())
}