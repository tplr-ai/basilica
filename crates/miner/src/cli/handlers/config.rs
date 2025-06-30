//! # Configuration Management Commands
//!
//! Handles configuration validation, reloading, display, and management
//! operations for the miner configuration system.

use anyhow::{anyhow, Result};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{error, info};

use crate::config::MinerConfig;
use common::config::ConfigValidation;

/// Configuration operation types
#[derive(Debug, Clone)]
pub enum ConfigOperation {
    Validate { path: Option<String> },
    Show { show_sensitive: bool },
    Reload,
    Diff { other_path: String },
    Export { format: ConfigFormat, path: String },
}

/// Configuration export formats
#[derive(Debug, Clone)]
pub enum ConfigFormat {
    Toml,
    Json,
    Yaml,
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
        ConfigOperation::Reload => reload_config(current_config).await,
        ConfigOperation::Diff { other_path } => diff_config(current_config, &other_path).await,
        ConfigOperation::Export { format, path } => {
            export_config(current_config, format, &path).await
        }
    }
}

/// Validate configuration file
async fn validate_config(config_path: Option<String>, current_config: &MinerConfig) -> Result<()> {
    let path = config_path.as_deref().unwrap_or("miner.toml");

    info!("Validating configuration file: {}", path);
    println!("🔍 Validating configuration: {path}");

    // Load configuration to validate
    let config_to_validate = if let Some(path) = config_path {
        if !Path::new(&path).exists() {
            return Err(anyhow!("Configuration file not found: {}", path));
        }

        match MinerConfig::load_from_file(&PathBuf::from(&path)) {
            Ok(config) => config,
            Err(e) => {
                error!("Failed to load configuration: {}", e);
                println!("❌ Configuration loading failed: {e}");
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

    println!("✅ Configuration validation passed");
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
        println!("ℹ️  Sensitive fields are masked. Use --show-sensitive to display them.");
    }

    // Convert to TOML for display
    let toml_content = toml::to_string_pretty(&config_display)
        .map_err(|e| anyhow!("Failed to serialize configuration: {}", e))?;

    println!("\n{toml_content}");

    // Show derived/computed values
    println!("\n=== Derived Configuration ===");
    println!("Database Type: SQLite");
    println!(
        "Server Address: {}:{}",
        config.server.host, config.server.port
    );
    println!("Metrics Enabled: {}", config.metrics.enabled);
    println!(
        "Executor Count: {}",
        config.executor_management.executors.len()
    );
    println!(
        "Remote Deployment: {}",
        config.remote_executor_deployment.is_some()
    );

    // Show validation status
    let validation_result = perform_comprehensive_validation(config).await?;
    println!("\n=== Validation Status ===");
    if validation_result.is_valid {
        println!("✅ Configuration is valid");
    } else {
        println!(
            "❌ Configuration has {} errors",
            validation_result.errors.len()
        );
        for error in &validation_result.errors {
            println!("   Error: {error}");
        }
    }

    if !validation_result.warnings.is_empty() {
        println!("⚠️  {} warnings found:", validation_result.warnings.len());
        for warning in &validation_result.warnings {
            println!("   Warning: {warning}");
        }
    }

    Ok(())
}

/// Reload configuration (simulate)
async fn reload_config(current_config: &MinerConfig) -> Result<()> {
    info!("Simulating configuration reload");
    println!("🔄 Testing configuration reload...");

    // In a real implementation, this would signal the running service
    // For now, we validate that the current config file can be reloaded

    // Try to load the configuration again
    let reloaded_config = MinerConfig::load()?;

    // Validate the reloaded configuration
    let validation_result = perform_comprehensive_validation(&reloaded_config).await?;

    if !validation_result.is_valid {
        println!("❌ Configuration reload would fail due to validation errors:");
        for error in &validation_result.errors {
            println!("   Error: {error}");
        }
        return Err(anyhow!("Configuration reload validation failed"));
    }

    // Check for differences that would require restart
    let requires_restart = check_restart_required(current_config, &reloaded_config);

    if requires_restart {
        println!("⚠️  Configuration changes detected that require service restart:");
        print_config_differences(current_config, &reloaded_config);
    } else {
        println!("✅ Configuration can be reloaded without restart");
    }

    println!("ℹ️  Note: This is a simulation. Actual reload requires service integration.");
    Ok(())
}

/// Compare configurations and show differences
async fn diff_config(current_config: &MinerConfig, other_path: &str) -> Result<()> {
    println!("📊 Comparing configurations...");
    println!("Current config vs: {other_path}");

    if !Path::new(other_path).exists() {
        return Err(anyhow!("Comparison file not found: {}", other_path));
    }

    // Load the other configuration
    let other_config = MinerConfig::load_from_file(&PathBuf::from(other_path))?;

    // Perform detailed comparison
    let differences = compare_configurations(current_config, &other_config)?;

    if differences.is_empty() {
        println!("✅ Configurations are identical");
        return Ok(());
    }

    println!("\n=== Configuration Differences ===");
    for (key, (current_val, other_val)) in differences {
        println!("🔸 {key}");
        println!("   Current: {}", format_config_value(&current_val));
        println!("   Other:   {}", format_config_value(&other_val));
        println!();
    }

    Ok(())
}

/// Export configuration in different formats
async fn export_config(
    config: &MinerConfig,
    format: ConfigFormat,
    output_path: &str,
) -> Result<()> {
    info!(
        "Exporting configuration to: {} (format: {:?})",
        output_path, format
    );
    println!("📤 Exporting configuration to: {output_path} ({format:?})");

    // Create a clean version without sensitive data
    let mut export_config = config.clone();
    mask_sensitive_fields(&mut export_config);

    let exported_content = match format {
        ConfigFormat::Toml => toml::to_string_pretty(&export_config)
            .map_err(|e| anyhow!("Failed to serialize to TOML: {}", e))?,
        ConfigFormat::Json => serde_json::to_string_pretty(&export_config)
            .map_err(|e| anyhow!("Failed to serialize to JSON: {}", e))?,
        ConfigFormat::Yaml => {
            // For now, fallback to JSON since serde_yaml might not be available
            serde_json::to_string_pretty(&export_config)
                .map_err(|e| anyhow!("Failed to serialize to YAML (using JSON): {}", e))?
        }
    };

    // Write to file
    fs::write(output_path, exported_content)
        .map_err(|e| anyhow!("Failed to write configuration file: {}", e))?;

    println!("✅ Configuration exported successfully");
    println!("   Size: {} bytes", fs::metadata(output_path)?.len());
    println!("   Note: Sensitive fields have been masked for security");

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

    // Server configuration validation
    validate_server_config(&config.server, &mut errors, &mut warnings);

    // Bittensor configuration validation
    validate_bittensor_config(
        &config.bittensor,
        &mut errors,
        &mut warnings,
        &mut suggestions,
    );

    // Executor management validation
    validate_executor_config(
        &config.executor_management,
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

    // Remote deployment validation (if configured)
    if let Some(ref deployment) = config.remote_executor_deployment {
        validate_remote_deployment_config(deployment, &mut errors, &mut warnings);
    }

    Ok(ValidationResult {
        is_valid: errors.is_empty(),
        errors,
        warnings,
        suggestions,
    })
}

/// Validate database configuration
fn validate_database_config(
    config: &common::config::DatabaseConfig,
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

/// Validate server configuration
fn validate_server_config(
    config: &common::config::ServerConfig,
    _errors: &mut [String],
    warnings: &mut Vec<String>,
) {
    if config.port < 1024 && config.host != "127.0.0.1" && config.host != "localhost" {
        warnings.push("Using privileged port (<1024) requires elevated permissions".to_string());
    }

    if config.host == "0.0.0.0" {
        warnings.push("Binding to 0.0.0.0 exposes service to all network interfaces".to_string());
    }

    if config.max_connections > 10000 {
        warnings.push("Very high max_connections may cause resource exhaustion".to_string());
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

    if config.uid.as_u16() == 0 {
        warnings.push("UID is 0, will be auto-discovered from chain".to_string());
    }

    if config.external_ip.is_none() {
        suggestions.push("Consider setting external_ip for production deployments".to_string());
    }

    if config.axon_port == config.common.netuid {
        warnings.push("axon_port same as netuid, potential conflict".to_string());
    }
}

/// Validate executor management configuration
fn validate_executor_config(
    config: &crate::config::ExecutorManagementConfig,
    _errors: &mut [String],
    warnings: &mut Vec<String>,
    suggestions: &mut Vec<String>,
) {
    if config.executors.is_empty() {
        warnings.push("No executors configured".to_string());
        suggestions.push("Add executor configurations or enable remote deployment".to_string());
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

    if config.jwt_secret == "change-me-in-production" {
        errors.push("Default JWT secret must be changed for production".to_string());
    }

    if config.jwt_secret.len() < 32 {
        warnings.push("JWT secret should be at least 32 characters long".to_string());
    }

    if config.token_expiration.as_secs() > 86400 {
        warnings.push("Long token expiration may pose security risks".to_string());
    }
}

/// Validate remote deployment configuration
fn validate_remote_deployment_config(
    config: &crate::config::RemoteExecutorDeploymentConfig,
    errors: &mut Vec<String>,
    warnings: &mut Vec<String>,
) {
    if config.remote_machines.is_empty() {
        warnings.push("Remote deployment configured but no machines specified".to_string());
    }

    for machine in &config.remote_machines {
        if machine.ssh.host.is_empty() {
            errors.push(format!("SSH host empty for machine: {}", machine.id));
        }

        if machine.ssh.username.is_empty() {
            errors.push(format!("SSH username empty for machine: {}", machine.id));
        }

        if machine.ssh.private_key_path.to_string_lossy().is_empty() {
            warnings.push(format!("No SSH key configured for machine: {}", machine.id));
        }
    }
}

/// Display validation results
fn display_validation_results(result: &ValidationResult) {
    if result.is_valid {
        println!("✅ Configuration validation passed");
    } else {
        println!(
            "❌ Configuration validation failed with {} errors",
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
        println!("\n⚠️  Warnings:");
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
    // Mask JWT secret
    if !config.security.jwt_secret.is_empty() {
        config.security.jwt_secret = "****MASKED****".to_string();
    }

    // Mask SSH private key paths
    if let Some(ref mut deployment) = config.remote_executor_deployment {
        for machine in &mut deployment.remote_machines {
            if !machine.ssh.private_key_path.to_string_lossy().is_empty() {
                machine.ssh.private_key_path = PathBuf::from("****MASKED****");
            }
        }
    }

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

/// Check if configuration changes require restart
fn check_restart_required(current: &MinerConfig, new: &MinerConfig) -> bool {
    // Changes that require restart
    current.server.port != new.server.port
        || current.server.host != new.server.host
        || current.database.url != new.database.url
        || current.bittensor.common.netuid != new.bittensor.common.netuid
        || current.bittensor.axon_port != new.bittensor.axon_port
}

/// Print configuration differences that require restart
fn print_config_differences(current: &MinerConfig, new: &MinerConfig) {
    if current.server.port != new.server.port {
        println!(
            "   • Server port: {} → {}",
            current.server.port, new.server.port
        );
    }
    if current.server.host != new.server.host {
        println!(
            "   • Server host: {} → {}",
            current.server.host, new.server.host
        );
    }
    if current.database.url != new.database.url {
        println!("   • Database URL changed");
    }
}

/// Compare two configurations and return differences
fn compare_configurations(
    current: &MinerConfig,
    other: &MinerConfig,
) -> Result<HashMap<String, (Value, Value)>> {
    let current_json = serde_json::to_value(current)?;
    let other_json = serde_json::to_value(other)?;

    let mut differences = HashMap::new();
    compare_json_values("", &current_json, &other_json, &mut differences);

    Ok(differences)
}

/// Recursively compare JSON values
fn compare_json_values(
    prefix: &str,
    current: &Value,
    other: &Value,
    differences: &mut HashMap<String, (Value, Value)>,
) {
    match (current, other) {
        (Value::Object(current_obj), Value::Object(other_obj)) => {
            // Compare all keys from both objects
            let mut all_keys = std::collections::HashSet::new();
            all_keys.extend(current_obj.keys());
            all_keys.extend(other_obj.keys());

            for key in all_keys {
                let new_prefix = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };

                match (current_obj.get(key), other_obj.get(key)) {
                    (Some(c), Some(o)) => compare_json_values(&new_prefix, c, o, differences),
                    (Some(c), None) => {
                        differences.insert(new_prefix, (c.clone(), Value::Null));
                    }
                    (None, Some(o)) => {
                        differences.insert(new_prefix, (Value::Null, o.clone()));
                    }
                    (None, None) => {}
                }
            }
        }
        _ => {
            if current != other {
                differences.insert(prefix.to_string(), (current.clone(), other.clone()));
            }
        }
    }
}

/// Format a configuration value for display
fn format_config_value(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::String(s) => format!("\"{s}\""),
        _ => value.to_string(),
    }
}
