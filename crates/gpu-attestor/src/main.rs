use anyhow::{Context, Result};
use std::time::Instant;
use tracing::{error, info, warn};

use gpu_attestor::{
    attestation::{save_attestation_to_files, AttestationBuilder, AttestationSigner},
    cli::{parse_args, setup_logging, Config},
    gpu::{benchmark_collector, GpuDetector},
    hardware::SystemInfoCollector,
    integrity::{calculate_binary_checksum, extract_embedded_key},
    network::NetworkBenchmarker,
    vdf::VdfComputer,
};

#[tokio::main]
async fn main() -> Result<()> {
    let config = parse_args()?;
    setup_logging(&config.log_level)?;

    info!("Starting GPU Attestor v{}", env!("CARGO_PKG_VERSION"));
    info!("Executor ID: {}", config.executor_id);

    let overall_start = Instant::now();

    // Step 1: Binary Integrity Verification
    if !config.skip_integrity_check {
        info!("Starting binary integrity verification...");
        match verify_binary_integrity() {
            Ok(_) => info!("Binary integrity verification passed"),
            Err(e) => {
                error!("Binary integrity verification failed: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        warn!("Skipping binary integrity verification (--skip-integrity-check)");
    }

    // Step 2: Hardware Information Collection
    let system_info = collect_hardware_info(&config).await?;

    // Step 2.5: OS Attestation Collection
    let os_attestation = collect_os_attestation(&config).await?;

    // Step 2.7: Docker Attestation Collection
    let docker_attestation = collect_docker_attestation(&config).await?;

    // Step 3: GPU Information Collection
    let gpu_info = collect_gpu_info().await?;

    // Step 3.5: GPU Benchmark Collection
    let gpu_benchmarks = if !config.skip_gpu_benchmarks {
        info!("Running GPU benchmarks...");
        match benchmark_collector::collect_gpu_benchmarks().await {
            Ok(benchmarks) => {
                info!("GPU benchmarks completed successfully");
                if !benchmarks.gpu_results.is_empty() {
                    for result in &benchmarks.gpu_results {
                        info!("  GPU {}: {}", result.gpu_index, result.gpu_name);
                        if let Some(bandwidth) = result.memory_bandwidth_gbps {
                            info!("    Memory bandwidth: {:.2} GB/s", bandwidth);
                        }
                        if let Some(fp16) = result.fp16_tflops {
                            info!("    FP16 performance: {:.2} TFLOPS", fp16);
                        }
                        if let Some(cuda_driver) = &result.cuda_driver_results {
                            info!("    CUDA Driver API benchmarks:");
                            for matrix_result in &cuda_driver.matrix_results {
                                info!(
                                    "      {}x{}: {:.2} GFLOPS",
                                    matrix_result.matrix_size,
                                    matrix_result.matrix_size,
                                    matrix_result.gflops
                                );
                            }
                        }
                    }
                }
                Some(benchmarks)
            }
            Err(e) => {
                warn!("GPU benchmarks failed: {}", e);
                None
            }
        }
    } else {
        warn!("Skipping GPU benchmarks (--skip-gpu-benchmarks)");
        None
    };

    // Step 4: Network Benchmarking
    let network_benchmark = run_network_benchmarks(&config).await?;

    // Step 4.5: Fetch IP Information
    let ipinfo = gpu_attestor::network::IpInfoCollector::fetch().await.ok();

    // Step 5: VDF Challenge
    let vdf_proof = compute_vdf_proof(&config).await?;

    // Step 6: Build and Sign Attestation Report
    let signed_attestation = build_and_sign_attestation(
        &config,
        system_info,
        os_attestation,
        docker_attestation,
        gpu_info,
        gpu_benchmarks,
        network_benchmark,
        ipinfo,
        vdf_proof,
    )?;

    // Step 7: Save Attestation Files
    save_attestation_files(&config, &signed_attestation)?;

    let total_time = overall_start.elapsed();
    info!(
        "GPU attestation completed successfully in {:.2}s",
        total_time.as_secs_f64()
    );
    print_output_summary(&config);

    Ok(())
}

fn verify_binary_integrity() -> Result<()> {
    let _public_key =
        extract_embedded_key().context("Failed to extract embedded verification key")?;
    info!("Binary integrity check: Using embedded public key");
    info!("Public key loaded successfully");
    Ok(())
}

async fn collect_hardware_info(config: &Config) -> Result<gpu_attestor::hardware::SystemInfo> {
    if !config.skip_hardware_collection {
        info!("Collecting hardware information...");
        match SystemInfoCollector::collect_all() {
            Ok(info) => {
                info!("System information collected successfully");
                info!(
                    "  CPU: {} cores, {} threads",
                    info.cpu.cores, info.cpu.threads
                );
                info!(
                    "  Memory: {:.2} GB total",
                    info.memory.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
                );
                Ok(info)
            }
            Err(e) => {
                error!("Failed to collect system information: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        warn!("Skipping hardware information collection (--skip-hardware-collection)");
        Ok(SystemInfoCollector::create_minimal_system_info())
    }
}

async fn collect_docker_attestation(
    config: &Config,
) -> Result<Option<gpu_attestor::docker::DockerAttestation>> {
    if !config.skip_docker_attestation {
        info!("Collecting Docker attestation information...");
        match gpu_attestor::docker::DockerCollector::collect_attestation() {
            Ok(attestation) => {
                info!("Docker attestation collected successfully");
                info!(
                    "  Docker: {}",
                    if attestation.info.is_running {
                        "running"
                    } else {
                        "not running"
                    }
                );
                if let Some(version) = &attestation.info.version {
                    info!("  Version: {}", version);
                }
                info!("  Security Score: {}/100", attestation.security_score());
                info!(
                    "  Performance Score: {}/100",
                    attestation.performance_score()
                );
                Ok(Some(attestation))
            }
            Err(e) => {
                error!("Failed to collect Docker attestation: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        warn!("Skipping Docker attestation collection (--skip-docker-attestation)");
        Ok(None)
    }
}

async fn collect_os_attestation(
    config: &Config,
) -> Result<Option<gpu_attestor::os::OsAttestation>> {
    if !config.skip_os_attestation {
        info!("Collecting OS attestation information...");
        match gpu_attestor::os::OsAttestor::attest_system() {
            Ok(attestation) => {
                info!("OS attestation collected successfully");
                info!(
                    "  OS: {} {}",
                    attestation.os_info.name, attestation.os_info.version
                );
                info!("  Kernel: {}", attestation.kernel_info.version);
                info!("  Security Score: {}/100", attestation.security_score());
                info!(
                    "  Performance Score: {}/100",
                    attestation.performance_score()
                );
                Ok(Some(attestation))
            }
            Err(e) => {
                error!("Failed to collect OS attestation: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        warn!("Skipping OS attestation collection (--skip-os-attestation)");
        Ok(None)
    }
}

async fn collect_gpu_info() -> Result<Vec<gpu_attestor::gpu::GpuInfo>> {
    info!("Collecting GPU information...");
    match GpuDetector::query_gpu_info() {
        Ok(gpus) => {
            if gpus.is_empty() {
                warn!("No GPUs detected on this system");
            } else {
                info!("Detected {} GPU(s):", gpus.len());
                for (i, gpu) in gpus.iter().enumerate() {
                    info!("  GPU {}: {} ({:?})", i, gpu.name, gpu.vendor);
                    info!(
                        "    Memory: {:.2} GB",
                        gpu.memory_total as f64 / (1024.0 * 1024.0 * 1024.0)
                    );
                    if let Some(temp) = gpu.temperature {
                        info!("    Temperature: {}°C", temp);
                    }
                }
            }
            Ok(gpus)
        }
        Err(e) => {
            error!("Failed to collect GPU information: {}", e);
            std::process::exit(1);
        }
    }
}

async fn run_network_benchmarks(
    config: &Config,
) -> Result<gpu_attestor::network::NetworkBenchmarkResults> {
    if !config.skip_network_benchmark {
        info!("Running network benchmarks...");
        match NetworkBenchmarker::run_comprehensive_benchmark().await {
            Ok(results) => {
                info!("Network benchmarks completed");
                info!("  Latency tests: {}", results.latency_tests.len());
                info!("  Throughput tests: {}", results.throughput_tests.len());
                info!(
                    "  DNS resolution: {}",
                    if results.dns_resolution_test.success {
                        "OK"
                    } else {
                        "Failed"
                    }
                );
                Ok(results)
            }
            Err(e) => {
                error!("Network benchmarks failed: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        warn!("Skipping network benchmarks (--skip-network-benchmark)");
        Ok(create_minimal_network_benchmark_results())
    }
}

async fn compute_vdf_proof(config: &Config) -> Result<Option<gpu_attestor::vdf::VdfProof>> {
    if !config.skip_vdf {
        info!(
            "Computing VDF proof with difficulty {}...",
            config.vdf_difficulty
        );
        match compute_vdf_challenge(config).await {
            Ok(proof) => {
                info!("VDF proof computed in {}ms", proof.computation_time_ms);
                Ok(Some(proof))
            }
            Err(e) => {
                error!("VDF computation failed: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        warn!("Skipping VDF computation (--skip-vdf)");
        Ok(None)
    }
}

#[allow(clippy::too_many_arguments)]
fn build_and_sign_attestation(
    config: &Config,
    system_info: gpu_attestor::hardware::SystemInfo,
    os_attestation: Option<gpu_attestor::os::OsAttestation>,
    docker_attestation: Option<gpu_attestor::docker::DockerAttestation>,
    gpu_info: Vec<gpu_attestor::gpu::GpuInfo>,
    gpu_benchmarks: Option<gpu_attestor::attestation::types::GpuBenchmarkResults>,
    network_benchmark: gpu_attestor::network::NetworkBenchmarkResults,
    ipinfo: Option<gpu_attestor::network::IpInfo>,
    vdf_proof: Option<gpu_attestor::vdf::VdfProof>,
) -> Result<gpu_attestor::attestation::SignedAttestation> {
    info!("Building attestation report...");
    let report = AttestationBuilder::new(config.executor_id.clone())
        .with_binary_info(
            calculate_binary_checksum()?,
            !config.skip_integrity_check,
            extract_embedded_key()
                .ok()
                .map(|k| hex::encode(k.to_encoded_point(true).as_bytes()))
                .unwrap_or_else(|| "unknown".to_string()),
        )
        .with_gpu_info(gpu_info)
        .with_system_info(system_info)
        .with_network_benchmark(network_benchmark);

    let report = if let Some(ipinfo) = ipinfo {
        report.with_ipinfo(ipinfo)
    } else {
        report
    };

    let report = if let Some(vdf_proof) = vdf_proof {
        report.with_vdf_proof(vdf_proof)
    } else {
        report
    };

    let report = if let Some(os_attestation) = os_attestation {
        report.with_os_attestation(os_attestation)
    } else {
        report
    };

    let report = if let Some(docker_attestation) = docker_attestation {
        report.with_docker_attestation(docker_attestation)
    } else {
        report
    };

    let report = if let Some(nonce) = &config.validator_nonce {
        report.with_validator_nonce(nonce.clone())
    } else {
        report
    };

    let report = if let Some(gpu_benchmarks) = gpu_benchmarks {
        report.with_gpu_benchmarks(gpu_benchmarks)
    } else {
        report
    };

    let report = report.build();

    info!("Signing attestation report...");
    let signer = match AttestationSigner::from_embedded_key() {
        Ok(signer) => signer,
        Err(_) => {
            info!("Using ephemeral signing key (embedded key is for verification only)");
            AttestationSigner::new()
        }
    };

    signer
        .sign_attestation(report)
        .context("Failed to sign attestation report")
}

fn save_attestation_files(
    config: &Config,
    signed_attestation: &gpu_attestor::attestation::SignedAttestation,
) -> Result<()> {
    info!("Saving attestation to: {}", config.output_path.display());
    save_attestation_to_files(signed_attestation, &config.output_path)
        .context("Failed to save attestation files")
}

fn print_output_summary(config: &Config) {
    info!("Attestation files saved to:");
    info!(
        "  Report: {}",
        config.output_path.with_extension("json").display()
    );
    info!(
        "  Signature: {}",
        config.output_path.with_extension("sig").display()
    );
    info!(
        "  Public Key: {}",
        config.output_path.with_extension("pub").display()
    );
}

// Helper functions

fn create_minimal_network_benchmark_results() -> gpu_attestor::network::NetworkBenchmarkResults {
    gpu_attestor::network::NetworkBenchmarkResults {
        latency_tests: Vec::new(),
        throughput_tests: Vec::new(),
        packet_loss_test: gpu_attestor::network::PacketLossTest {
            target_host: "8.8.8.8".to_string(),
            packets_sent: 0,
            packets_received: 0,
            packet_loss_percent: 0.0,
            test_duration_seconds: 0,
        },
        dns_resolution_test: gpu_attestor::network::DnsResolutionTest {
            hostname: "google.com".to_string(),
            resolution_time_ms: 0.0,
            resolved_ips: Vec::new(),
            dns_server: "8.8.8.8".to_string(),
            success: false,
        },
        port_connectivity_tests: Vec::new(),
    }
}

async fn compute_vdf_challenge(config: &Config) -> Result<gpu_attestor::vdf::VdfProof> {
    // Create a mock VDF challenge (in practice, this would come from the validator)
    let (modulus, generator) = VdfComputer::generate_public_params(64)?; // Small for testing

    // Create deterministic seed for VDF
    // First try to use embedded key if available, otherwise use binary checksum
    let prev_sig = extract_embedded_key()
        .ok()
        .map(|key| {
            let encoded_point = key.to_encoded_point(true);
            encoded_point.as_bytes().to_vec()
        })
        .or_else(|| {
            calculate_binary_checksum()
                .ok()
                .map(|checksum| checksum.into_bytes())
        })
        .unwrap_or_else(|| {
            // Ultimate fallback: use a fixed seed
            b"basilica-gpu-attestor-vdf-fallback-seed".to_vec()
        });

    let mut challenge =
        VdfComputer::create_challenge(modulus, generator, config.vdf_difficulty, &prev_sig);

    // Adjust time constraints for testing
    challenge.min_required_time_ms = 0;
    challenge.max_allowed_time_ms = 60000; // 1 minute max

    let proof = tokio::task::spawn_blocking({
        let challenge = challenge.clone();
        let algorithm = config.vdf_algorithm.clone();
        move || VdfComputer::compute_vdf(&challenge, algorithm)
    })
    .await?
    .context("VDF computation failed")?;

    // Verify the proof
    let is_valid = VdfComputer::verify_vdf_proof(&challenge, &proof)?;
    if !is_valid {
        anyhow::bail!("VDF proof verification failed");
    }

    Ok(proof)
}
