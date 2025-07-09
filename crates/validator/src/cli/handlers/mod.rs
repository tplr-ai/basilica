use crate::cli::commands::Command;
use crate::config::ValidatorConfig;
use anyhow::Result;
use common::config::ConfigValidation;

pub mod database;
pub mod scores;
pub mod service;
pub mod weights;

pub struct CommandHandler;

impl CommandHandler {
    pub fn new() -> Self {
        Self
    }

    pub async fn execute(&self, command: Command) -> Result<()> {
        self.execute_with_context(command, false).await
    }

    pub async fn execute_with_context(&self, command: Command, local_test: bool) -> Result<()> {
        match command {
            Command::Start { config } => service::handle_start(config, local_test).await,
            Command::Stop => service::handle_stop().await,
            Command::Status => service::handle_status().await,
            Command::GenConfig { output } => service::handle_gen_config(output).await,

            // Validation commands removed with HardwareValidator
            Command::Connect { .. } => {
                Err(anyhow::anyhow!("Hardware validation commands have been removed. Use the verification engine API instead."))
            }

            Command::Verify { .. } => {
                Err(anyhow::anyhow!("Hardware validation commands have been removed. Use the verification engine API instead."))
            }

            // Legacy verification command (deprecated)
            #[allow(deprecated)]
            Command::VerifyLegacy { .. } => {
                Err(anyhow::anyhow!("Legacy validation commands have been removed. Use the verification engine API instead."))
            }

            Command::Weights { action } => weights::handle_weights(action).await,
            Command::Scores { action } => scores::handle_scores(action).await,
            Command::Database { action } => database::handle_database(action).await,
        }
    }
}

impl Default for CommandHandler {
    fn default() -> Self {
        Self::new()
    }
}

pub struct HandlerUtils;

impl HandlerUtils {
    pub fn load_config(config_path: Option<&str>) -> Result<ValidatorConfig> {
        match config_path {
            Some(path) if std::path::Path::new(path).exists() => {
                ValidatorConfig::load_from_file(std::path::Path::new(path))
            }
            _ => ValidatorConfig::load(),
        }
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

    pub fn format_json<T: serde::Serialize>(data: &T) -> Result<String> {
        Ok(serde_json::to_string_pretty(data)?)
    }

    pub fn format_table<T: std::fmt::Display>(headers: &[&str], rows: &[Vec<T>]) -> String {
        let mut output = String::new();

        output.push_str(&format!("{:<20}", headers.join(" | ")));
        output.push('\n');
        output.push_str(&"-".repeat(headers.len() * 20));
        output.push('\n');

        for row in rows {
            let row_str: Vec<String> = row.iter().map(|cell| format!("{cell:<20}")).collect();
            output.push_str(&row_str.join(" | "));
            output.push('\n');
        }

        output
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
