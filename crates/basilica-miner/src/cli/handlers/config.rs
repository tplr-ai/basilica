//! # Configuration Management Commands
//!
//! Handles configuration validation and display operations for the miner configuration system.

use crate::config::MinerConfig;
use alloy::signers::local::PrivateKeySigner;
use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};
use tracing::{error, info};

use basilica_common::config::ConfigValidation;

/// Configuration operation types
#[derive(Debug, Clone)]
pub enum ConfigOperation {
    Validate { path: Option<String> },
    Show { show_sensitive: bool },
}

/// Configuration validation result
#[derive(Debug)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub suggestions: Vec<String>,
}

/// Handle configuration management commands
pub async fn handle_config_command(
    operation: ConfigOperation,
    current_config: &MinerConfig,
) -> Result<()> {
    match operation {
        ConfigOperation::Validate { path } => validate_config(path, current_config).await,
        ConfigOperation::Show { show_sensitive } => {
            show_config(current_config, show_sensitive).await
        }
    }
}

/// Validate configuration file
async fn validate_config(config_path: Option<String>, current_config: &MinerConfig) -> Result<()> {
    let path = config_path.as_deref().unwrap_or("miner.toml");

    info!("Validating configuration file: {}", path);
    println!("Validating configuration: {path}");

    // Load configuration to validate
    let config_to_validate = if let Some(path) = config_path {
        if !Path::new(&path).exists() {
            return Err(anyhow!("Configuration file not found: {}", path));
        }

        match MinerConfig::load_from_file(&PathBuf::from(&path)) {
            Ok(config) => config,
            Err(e) => {
                error!("Failed to load configuration: {}", e);
                println!("ERROR: Configuration loading failed: {e}");
                return Err(e);
            }
        }
    } else {
        current_config.clone()
    };

    // Perform comprehensive validation
    let validation_result = perform_comprehensive_validation(&config_to_validate).await?;

    // Display results
    display_validation_results(&validation_result);

    if !validation_result.is_valid {
        return Err(anyhow!("Configuration validation failed"));
    }

    println!("Configuration validation passed");
    Ok(())
}

/// Show current configuration
async fn show_config(config: &MinerConfig, show_sensitive: bool) -> Result<()> {
    println!("📋 Current Miner Configuration");
    println!("==============================");

    // Create a display version of the config
    let mut config_display = config.clone();

    if !show_sensitive {
        // Mask sensitive fields
        mask_sensitive_fields(&mut config_display);
        println!("INFO: Sensitive fields are masked. Use --show-sensitive to display them.");
    }

    // Convert to TOML for display
    let toml_content = toml::to_string_pretty(&config_display)
        .map_err(|e| anyhow!("Failed to serialize configuration: {}", e))?;

    println!("\n{toml_content}");

    // Show derived/computed values
    println!("\n=== Derived Configuration ===");
    println!("Database Type: SQLite");
    println!("Metrics Enabled: {}", config.metrics.enabled);
    println!("Node Count: {}", config.node_management.nodes.len());

    // Show validation status
    let validation_result = perform_comprehensive_validation(config).await?;
    println!("\n=== Validation Status ===");
    if validation_result.is_valid {
        println!("Configuration is valid");
    } else {
        println!(
            "ERROR: Configuration has {} errors",
            validation_result.errors.len()
        );
        for error in &validation_result.errors {
            println!("   Error: {error}");
        }
    }

    if !validation_result.warnings.is_empty() {
        println!(
            "WARNING: {} warnings found:",
            validation_result.warnings.len()
        );
        for warning in &validation_result.warnings {
            println!("   Warning: {warning}");
        }
    }

    Ok(())
}

/// Perform comprehensive configuration validation
async fn perform_comprehensive_validation(config: &MinerConfig) -> Result<ValidationResult> {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();
    let mut suggestions = Vec::new();

    // Basic validation using the built-in validate method
    if let Err(e) = config.validate() {
        errors.push(format!("Configuration validation failed: {e}"));
    }

    // Database configuration validation
    validate_database_config(&config.database, &mut errors, &mut warnings);

    // Bittensor configuration validation
    validate_bittensor_config(
        &config.bittensor,
        &mut errors,
        &mut warnings,
        &mut suggestions,
    );

    // Node management validation
    validate_node_config(
        &config.node_management,
        &mut errors,
        &mut warnings,
        &mut suggestions,
    );

    // Security configuration validation
    validate_security_config(
        &config.security,
        &mut errors,
        &mut warnings,
        &mut suggestions,
    );

    Ok(ValidationResult {
        is_valid: errors.is_empty(),
        errors,
        warnings,
        suggestions,
    })
}

/// Validate database configuration
fn validate_database_config(
    config: &basilica_common::config::DatabaseConfig,
    errors: &mut Vec<String>,
    warnings: &mut Vec<String>,
) {
    if !config.url.starts_with("sqlite:") {
        errors.push("Only SQLite databases are currently supported".to_string());
    }

    if config.max_connections < config.min_connections {
        errors.push("max_connections must be >= min_connections".to_string());
    }

    if config.max_connections > 50 {
        warnings.push("High max_connections value may impact performance".to_string());
    }
}

/// Validate Bittensor configuration
fn validate_bittensor_config(
    config: &crate::config::MinerBittensorConfig,
    errors: &mut Vec<String>,
    warnings: &mut Vec<String>,
    suggestions: &mut Vec<String>,
) {
    if config.common.wallet_name.is_empty() {
        errors.push("wallet_name cannot be empty".to_string());
    }

    if config.common.hotkey_name.is_empty() {
        errors.push("hotkey_name cannot be empty".to_string());
    }

    // UID will be auto-discovered from chain during startup

    if config.external_ip.is_none() {
        suggestions.push("Consider setting external_ip for production deployments".to_string());
    }

    if config.axon_port == config.common.netuid {
        warnings.push("axon_port same as netuid, potential conflict".to_string());
    }
}

/// Validate node management configuration
fn validate_node_config(
    config: &crate::config::NodeManagementConfig,
    _errors: &mut [String],
    warnings: &mut Vec<String>,
    suggestions: &mut Vec<String>,
) {
    if config.nodes.is_empty() {
        warnings.push("No nodes configured".to_string());
        suggestions.push("Add node configurations or enable remote deployment".to_string());
    }

    if config.health_check_interval.as_secs() < 30 {
        warnings.push("Very frequent health checks may impact performance".to_string());
    }

    if config.max_retry_attempts > 10 {
        warnings.push("High retry attempts may cause long delays".to_string());
    }
}

/// Validate security configuration
fn validate_security_config(
    config: &crate::config::SecurityConfig,
    errors: &mut Vec<String>,
    warnings: &mut Vec<String>,
    suggestions: &mut Vec<String>,
) {
    if !config.verify_signatures {
        warnings.push("Signature verification is disabled".to_string());
        suggestions.push("Enable signature verification for production".to_string());
    }

    if config.private_key_file.is_some() {
        match config.get_private_key() {
            Ok(private_key) => {
                validate_private_key_config(&private_key, errors, warnings, suggestions);
            }
            Err(e) => {
                errors.push(format!("Failed to get private key: {e}"));
            }
        }
    } else {
        suggestions.push(
            "Consider setting private_key_file for collateral contract operations".to_string(),
        );
    }
}

/// Validate private key configuration
fn validate_private_key_config(
    private_key: &str,
    errors: &mut Vec<String>,
    _warnings: &mut [String],
    _suggestions: &mut [String],
) {
    if private_key.is_empty() {
        errors.push("private_key cannot be empty".to_string());
        return;
    }

    let private_key = private_key.trim_start_matches("0x");
    if private_key.len() != 64 {
        errors.push("private_key must be exactly 64 characters long".to_string());
        return;
    }

    // Validate that the private key can be parsed and get the corresponding address
    if private_key.parse::<PrivateKeySigner>().is_err() {
        errors.push("Invalid private key format".to_string());
    }
}

/// Display validation results
fn display_validation_results(result: &ValidationResult) {
    if result.is_valid {
        println!("Configuration validation passed");
    } else {
        println!(
            "ERROR: Configuration validation failed with {} errors",
            result.errors.len()
        );
    }

    if !result.errors.is_empty() {
        println!("\n🚨 Errors:");
        for error in &result.errors {
            println!("   • {error}");
        }
    }

    if !result.warnings.is_empty() {
        println!("\nWARNINGS:");
        for warning in &result.warnings {
            println!("   • {warning}");
        }
    }

    if !result.suggestions.is_empty() {
        println!("\n💡 Suggestions:");
        for suggestion in &result.suggestions {
            println!("   • {suggestion}");
        }
    }
}

/// Mask sensitive configuration fields
fn mask_sensitive_fields(config: &mut MinerConfig) {
    // Mask database connection details if they contain passwords
    if config.database.url.contains("password") {
        config.database.url = config
            .database
            .url
            .split('?')
            .next()
            .unwrap_or("****MASKED****")
            .to_string();
    }
}
