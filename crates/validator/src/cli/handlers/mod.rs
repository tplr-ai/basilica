use self::validation::{SshConnectArgs, SshVerifyArgs};
use crate::cli::commands::Command;
use crate::config::ValidatorConfig;
use anyhow::Result;
use common::config::ConfigValidation;

pub mod database;
pub mod scores;
pub mod service;
pub mod validation;
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

            // New SSH validation commands
            Command::Connect {
                host,
                username,
                port,
                private_key,
                timeout,
                executor_id,
            } => {
                if host.is_none() && username.is_none() && executor_id.is_none() {
                    validation::print_usage_examples();
                    return Ok(());
                }

                let args = SshConnectArgs {
                    host: host.unwrap_or_else(|| {
                        eprintln!("Error: --host is required when not using --executor-id");
                        std::process::exit(1);
                    }),
                    username: username.unwrap_or_else(|| {
                        eprintln!("Error: --username is required when not using --executor-id");
                        std::process::exit(1);
                    }),
                    port,
                    private_key,
                    timeout,
                    executor_id,
                };
                validation::handle_ssh_connect(args).await
            }

            Command::Verify {
                host,
                username,
                port,
                private_key,
                timeout,
                executor_id,
                miner_uid,
                gpu_attestor_path,
                remote_work_dir,
                execution_timeout,
                skip_cleanup,
                verbose,
            } => {
                if host.is_none()
                    && username.is_none()
                    && executor_id.is_none()
                    && miner_uid.is_none()
                {
                    validation::print_usage_examples();
                    return Ok(());
                }

                let ssh_args = if host.is_some() || username.is_some() {
                    Some(SshConnectArgs {
                        host: host.unwrap_or_else(|| {
                            eprintln!("Error: --host is required when not using --executor-id or --miner-uid");
                            std::process::exit(1);
                        }),
                        username: username.unwrap_or_else(|| {
                            eprintln!("Error: --username is required when not using --executor-id or --miner-uid");
                            std::process::exit(1);
                        }),
                        port,
                        private_key,
                        timeout,
                        executor_id: None,
                    })
                } else {
                    None
                };

                let args = SshVerifyArgs {
                    ssh_args,
                    executor_id,
                    miner_uid,
                    gpu_attestor_path,
                    remote_work_dir,
                    execution_timeout,
                    skip_cleanup,
                    verbose,
                };
                validation::handle_ssh_verify(args).await
            }

            // Legacy verification command (deprecated)
            #[allow(deprecated)]
            Command::VerifyLegacy {
                miner_uid,
                executor_id,
                all,
            } => {
                HandlerUtils::print_warning("WARNING: This verification method is deprecated!");
                HandlerUtils::print_info("Please use the new validation commands:");
                HandlerUtils::print_info("   - validator connect --host <HOST> --username <USER>");
                HandlerUtils::print_info("   - validator verify --host <HOST> --username <USER>");
                if let Some(id) = executor_id {
                    HandlerUtils::print_info(&format!("   - validator verify --executor-id {id}"));
                }
                if let Some(uid) = miner_uid {
                    HandlerUtils::print_info(&format!("   - validator verify --miner-uid {uid}"));
                }
                if all {
                    HandlerUtils::print_info(
                        "   - Use --miner-uid with multiple UIDs for bulk verification",
                    );
                }
                Err(anyhow::anyhow!(
                    "Legacy command deprecated. Use 'validator verify' instead."
                ))
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
