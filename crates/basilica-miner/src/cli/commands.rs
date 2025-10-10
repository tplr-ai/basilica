use clap::Subcommand;

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Service management commands
    Service {
        #[command(subcommand)]
        service_cmd: ServiceCommand,
    },
    /// Database management commands
    Database {
        #[command(subcommand)]
        database_cmd: DatabaseCommand,
    },
    /// Configuration management commands
    Config {
        #[command(subcommand)]
        config_cmd: ConfigCommand,
    },
    /// Show miner status and statistics
    Status,
    /// Run database migrations
    Migrate,
}

/// Service management subcommands
#[derive(Subcommand, Debug)]
pub enum ServiceCommand {
    /// Start the miner service
    Start,

    /// Stop the miner service
    Stop,

    /// Restart the miner service
    Restart,

    /// Show service status
    Status,

    /// Reload service configuration
    Reload,
}

/// Database management subcommands
#[derive(Subcommand, Debug)]
pub enum DatabaseCommand {
    /// Backup the database
    Backup {
        /// Backup file path
        path: String,
    },

    /// Restore database from backup
    Restore {
        /// Backup file path to restore from
        path: String,
    },

    /// Show database statistics
    Stats,

    /// Vacuum database to reclaim space
    Vacuum,

    /// Check database integrity
    Integrity,
}

/// Configuration management subcommands
#[derive(Subcommand, Debug)]
pub enum ConfigCommand {
    /// Validate configuration file
    Validate {
        /// Configuration file path to validate (default: current config)
        #[arg(short, long)]
        path: Option<String>,
    },

    /// Show current configuration
    Show {
        /// Show sensitive fields (default: masked)
        #[arg(long)]
        show_sensitive: bool,
    },

    /// Reload configuration (test only)
    Reload,

    /// Compare configurations
    Diff {
        /// Path to configuration file to compare with
        other_path: String,
    },

    /// Export configuration in different formats
    Export {
        /// Export format (toml, json, yaml)
        #[arg(short, long, default_value = "toml")]
        format: String,
        /// Output file path
        path: String,
    },
}
