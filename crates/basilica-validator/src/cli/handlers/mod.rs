use std::path::PathBuf;

use crate::config::ValidatorConfig;
use anyhow::Result;
use basilica_common::config::ConfigValidation;

pub mod rental;
pub mod service;

pub struct HandlerUtils;

impl HandlerUtils {
    pub fn load_config(config_path: PathBuf) -> Result<ValidatorConfig> {
        if !config_path.exists() {
            return Err(anyhow::anyhow!(
                "Configuration file not found: {}",
                config_path.display()
            ));
        }

        tracing::info!("Loading configuration from: {}", config_path.display());
        let config = ValidatorConfig::load_from_file(config_path.as_path())?;
        tracing::info!(
            "Configuration loaded: burn_uid={}, burn_percentage={:.2}%, weight_interval_blocks={}, netuid={}, network={}",
            config.emission.burn_uid,
            config.emission.burn_percentage,
            config.emission.weight_set_interval_blocks,
            config.bittensor.common.netuid,
            config.bittensor.common.network
        );
        Ok(config)
    }

    pub fn validate_config(config: &ValidatorConfig) -> Result<()> {
        config
            .validate()
            .map_err(|e| anyhow::anyhow!("Configuration validation failed: {}", e))?;

        let warnings = config.warnings();
        if !warnings.is_empty() {
            for warning in warnings {
                Self::print_warning(&format!("Configuration warning: {warning}"));
            }
        }

        Ok(())
    }

    pub fn print_success(message: &str) {
        println!("[SUCCESS] {message}");
    }

    pub fn print_error(message: &str) {
        eprintln!("[ERROR] {message}");
    }

    pub fn print_info(message: &str) {
        println!("[INFO] {message}");
    }

    pub fn print_warning(message: &str) {
        println!("[WARNING] {message}");
    }
}
