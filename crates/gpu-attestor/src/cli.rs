use anyhow::{Context, Result};
use clap::{Arg, Command};
use std::path::PathBuf;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::vdf::VdfAlgorithm;

#[derive(Debug)]
pub struct Config {
    pub executor_id: String,
    pub output_path: PathBuf,
    pub skip_integrity_check: bool,
    pub skip_network_benchmark: bool,
    pub skip_vdf: bool,
    pub skip_hardware_collection: bool,
    pub skip_os_attestation: bool,
    pub skip_docker_attestation: bool,
    pub skip_gpu_benchmarks: bool,
    pub vdf_difficulty: u64,
    pub vdf_algorithm: VdfAlgorithm,
    pub log_level: String,
    pub validator_nonce: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            executor_id: gethostname::gethostname().to_string_lossy().to_string(),
            output_path: PathBuf::from("./attestation"),
            skip_integrity_check: false,
            skip_network_benchmark: false,
            skip_vdf: false,
            skip_hardware_collection: false,
            skip_os_attestation: false,
            skip_docker_attestation: false,
            skip_gpu_benchmarks: false,
            vdf_difficulty: 1000, // Low default for testing
            vdf_algorithm: VdfAlgorithm::SimpleSequential,
            log_level: "info".to_string(),
            validator_nonce: None,
        }
    }
}

pub fn parse_args() -> Result<Config> {
    let matches = Command::new("gpu-attestor")
        .version(env!("CARGO_PKG_VERSION"))
        .about("Secure GPU hardware attestation for the Basilica network")
        .arg(
            Arg::new("executor-id")
                .long("executor-id")
                .help("Unique identifier for this executor")
                .value_name("ID"),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .help("Output path for attestation files (without extension)")
                .value_name("PATH")
                .default_value("./attestation"),
        )
        .arg(
            Arg::new("skip-integrity-check")
                .long("skip-integrity-check")
                .help("Skip binary integrity verification (for testing)")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("skip-network-benchmark")
                .long("skip-network-benchmark")
                .help("Skip network benchmarking")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("skip-vdf")
                .long("skip-vdf")
                .help("Skip VDF computation")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("skip-hardware-collection")
                .long("skip-hardware-collection")
                .help("Skip detailed hardware information collection")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("skip-os-attestation")
                .long("skip-os-attestation")
                .help("Skip OS attestation collection")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("skip-docker-attestation")
                .long("skip-docker-attestation")
                .help("Skip Docker attestation collection")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("skip-gpu-benchmarks")
                .long("skip-gpu-benchmarks")
                .help("Skip GPU benchmark collection")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("vdf-difficulty")
                .long("vdf-difficulty")
                .help("VDF computation difficulty (number of iterations)")
                .value_name("NUMBER")
                .default_value("1000"),
        )
        .arg(
            Arg::new("vdf-algorithm")
                .long("vdf-algorithm")
                .help("VDF algorithm to use")
                .value_name("ALGORITHM")
                .value_parser(["wesolowski", "pietrzak", "simple"])
                .default_value("simple"),
        )
        .arg(
            Arg::new("debug")
                .long("debug")
                .help("Enable debug logging")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("log-level")
                .long("log-level")
                .help("Logging level")
                .value_name("LEVEL")
                .value_parser(["error", "warn", "info", "debug", "trace"])
                .default_value("info"),
        )
        .arg(
            Arg::new("nonce")
                .long("nonce")
                .help("Validator-provided nonce for replay protection")
                .value_name("NONCE"),
        )
        .get_matches();

    let mut config = Config::default();

    if let Some(executor_id) = matches.get_one::<String>("executor-id") {
        config.executor_id = executor_id.clone();
    }

    if let Some(output) = matches.get_one::<String>("output") {
        config.output_path = PathBuf::from(output);
    }

    config.skip_integrity_check = matches.get_flag("skip-integrity-check");
    config.skip_network_benchmark = matches.get_flag("skip-network-benchmark");
    config.skip_docker_attestation = matches.get_flag("skip-docker-attestation");
    config.skip_vdf = matches.get_flag("skip-vdf");
    config.skip_hardware_collection = matches.get_flag("skip-hardware-collection");
    config.skip_os_attestation = matches.get_flag("skip-os-attestation");
    config.skip_gpu_benchmarks = matches.get_flag("skip-gpu-benchmarks");

    if let Some(difficulty) = matches.get_one::<String>("vdf-difficulty") {
        config.vdf_difficulty = difficulty.parse().context("Invalid VDF difficulty value")?;
    }

    if let Some(algorithm) = matches.get_one::<String>("vdf-algorithm") {
        config.vdf_algorithm = match algorithm.as_str() {
            "wesolowski" => VdfAlgorithm::Wesolowski,
            "pietrzak" => VdfAlgorithm::Pietrzak,
            "simple" => VdfAlgorithm::SimpleSequential,
            _ => return Err(anyhow::anyhow!("Invalid VDF algorithm: {}", algorithm)),
        };
    }

    if matches.get_flag("debug") {
        config.log_level = "debug".to_string();
    } else if let Some(log_level) = matches.get_one::<String>("log-level") {
        config.log_level = log_level.clone();
    }

    if let Some(nonce) = matches.get_one::<String>("nonce") {
        config.validator_nonce = Some(nonce.clone());
    }

    Ok(config)
}

pub fn setup_logging(level: &str) -> Result<()> {
    let level_filter = match level {
        "error" => tracing::Level::ERROR,
        "warn" => tracing::Level::WARN,
        "info" => tracing::Level::INFO,
        "debug" => tracing::Level::DEBUG,
        "trace" => tracing::Level::TRACE,
        _ => return Err(anyhow::anyhow!("Invalid log level: {}", level)),
    };

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer()
                .with_level(true)
                .with_target(false)
                .with_ansi(true),
        )
        .with(tracing_subscriber::filter::LevelFilter::from_level(
            level_filter,
        ))
        .init();

    Ok(())
}

// Helper function to get hostname
mod gethostname {
    use std::ffi::OsString;

    pub fn gethostname() -> OsString {
        std::env::var_os("HOSTNAME")
            .or_else(|| std::env::var_os("COMPUTERNAME"))
            .unwrap_or_else(|| "unknown-host".into())
    }
}
