//! CLI argument parsing and application configuration
//!
//! Provides structured argument parsing with separation of concerns:
//! - Server configuration arguments
//! - Operational mode selection
//! - Command routing

use super::Commands;
use anyhow::Result;
use clap::Parser;
use std::net::SocketAddr;

/// Main application arguments
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct ExecutorArgs {
    /// Configuration file path
    #[arg(short, long, default_value = "executor.toml")]
    pub config: String,

    /// Log level (trace, debug, info, warn, error)
    #[arg(short, long, default_value = "info")]
    pub log_level: String,

    /// Enable prometheus metrics endpoint
    #[arg(long)]
    pub metrics: bool,

    /// Metrics server address
    #[arg(long, default_value = "0.0.0.0:9090")]
    pub metrics_addr: SocketAddr,

    /// Generate sample configuration file
    #[arg(long)]
    pub gen_config: bool,

    /// Start server mode instead of CLI mode
    #[arg(long)]
    pub server: bool,

    /// CLI command (default mode)
    #[command(subcommand)]
    pub command: Option<Commands>,
}

impl ExecutorArgs {
    /// Parse arguments from command line
    pub fn parse_args() -> Self {
        Self::parse()
    }

    /// Get the application mode based on arguments
    pub fn get_mode(&self) -> ApplicationMode {
        if self.gen_config {
            ApplicationMode::GenerateConfig
        } else if self.server {
            ApplicationMode::Server
        } else if self.command.is_some() {
            ApplicationMode::Cli
        } else {
            ApplicationMode::Help
        }
    }

    /// Get server configuration from arguments
    pub fn get_server_config(&self) -> ServerConfig {
        ServerConfig {
            log_level: self.log_level.clone(),
            config_path: self.config.clone(),
            metrics_enabled: self.metrics,
            metrics_addr: self.metrics_addr,
        }
    }

    /// Get CLI configuration from arguments
    pub fn get_cli_config(&self) -> CliConfig {
        CliConfig {
            config_path: self.config.clone(),
            command: self.command.clone(),
        }
    }

    /// Get config generation configuration
    pub fn get_config_gen_config(&self) -> ConfigGenConfig {
        ConfigGenConfig {
            output_path: self.config.clone(),
        }
    }
}

/// Application operating mode
#[derive(Debug, Clone, PartialEq)]
pub enum ApplicationMode {
    /// Generate configuration file and exit
    GenerateConfig,
    /// Run as server daemon
    Server,
    /// Execute CLI command and exit
    Cli,
    /// Show help and exit
    Help,
}

/// Server mode configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub log_level: String,
    pub config_path: String,
    pub metrics_enabled: bool,
    pub metrics_addr: SocketAddr,
}

/// CLI mode configuration
#[derive(Debug, Clone)]
pub struct CliConfig {
    pub config_path: String,
    pub command: Option<Commands>,
}

/// Configuration generation settings
#[derive(Debug, Clone)]
pub struct ConfigGenConfig {
    pub output_path: String,
}

/// Application configuration resolver
pub struct AppConfigResolver;

impl AppConfigResolver {
    /// Resolve application configuration from arguments
    pub fn resolve(args: &ExecutorArgs) -> Result<AppConfig> {
        let mode = args.get_mode();

        match mode {
            ApplicationMode::GenerateConfig => {
                Ok(AppConfig::ConfigGeneration(args.get_config_gen_config()))
            }
            ApplicationMode::Server => Ok(AppConfig::Server(args.get_server_config())),
            ApplicationMode::Cli => Ok(AppConfig::Cli(args.get_cli_config())),
            ApplicationMode::Help => Ok(AppConfig::Help),
        }
    }
}

/// Unified application configuration
#[derive(Debug, Clone)]
pub enum AppConfig {
    /// Configuration file generation mode
    ConfigGeneration(ConfigGenConfig),
    /// Server daemon mode
    Server(ServerConfig),
    /// CLI command execution mode
    Cli(CliConfig),
    /// Help display mode
    Help,
}

impl AppConfig {
    /// Check if this configuration requires config file loading
    pub fn requires_config_loading(&self) -> bool {
        matches!(self, AppConfig::Server(_) | AppConfig::Cli(_))
    }

    /// Check if this configuration enables metrics
    pub fn metrics_enabled(&self) -> bool {
        match self {
            AppConfig::Server(config) => config.metrics_enabled,
            _ => false,
        }
    }

    /// Get metrics address if applicable
    pub fn metrics_addr(&self) -> Option<SocketAddr> {
        match self {
            AppConfig::Server(config) if config.metrics_enabled => Some(config.metrics_addr),
            _ => None,
        }
    }

    /// Get log level if applicable
    pub fn log_level(&self) -> Option<&str> {
        match self {
            AppConfig::Server(config) => Some(&config.log_level),
            _ => None,
        }
    }

    /// Get config file path if applicable
    pub fn config_path(&self) -> Option<&str> {
        match self {
            AppConfig::Server(config) => Some(&config.config_path),
            AppConfig::Cli(config) => Some(&config.config_path),
            AppConfig::ConfigGeneration(config) => Some(&config.output_path),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_application_mode_detection() {
        let args = ExecutorArgs {
            config: "test.toml".to_string(),
            log_level: "debug".to_string(),
            metrics: false,
            metrics_addr: "127.0.0.1:9090".parse().unwrap(),
            gen_config: true,
            server: false,
            command: None,
        };

        assert_eq!(args.get_mode(), ApplicationMode::GenerateConfig);

        let args = ExecutorArgs {
            gen_config: false,
            server: true,
            command: None,
            ..args
        };

        assert_eq!(args.get_mode(), ApplicationMode::Server);
    }

    #[test]
    fn test_config_resolution() {
        let args = ExecutorArgs {
            config: "test.toml".to_string(),
            log_level: "debug".to_string(),
            metrics: true,
            metrics_addr: "127.0.0.1:9090".parse().unwrap(),
            gen_config: false,
            server: true,
            command: None,
        };

        let config = AppConfigResolver::resolve(&args).unwrap();
        assert!(matches!(config, AppConfig::Server(_)));
        assert!(config.metrics_enabled());
        assert_eq!(config.log_level(), Some("debug"));
    }
}
