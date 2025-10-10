use basilica_sdk::types::RentalState;
use clap::{Subcommand, ValueHint};
use std::path::PathBuf;

use crate::handlers::gpu_rental::TargetType;
use basilica_validator::gpu::categorization::GpuCategory;

/// Main CLI commands
#[derive(Subcommand, Debug, Clone)]
pub enum Commands {
    /// List available GPU resources
    #[command(alias = "list")]
    Ls {
        /// Filter by GPU category (e.g., 'h100', 'h200', 'b200') (optional)
        gpu_type: Option<GpuCategory>,

        #[command(flatten)]
        filters: ListFilters,
    },

    /// Provision and start GPU instances
    #[command(alias = "start")]
    Up {
        /// Target node ID (UUID) or GPU category (e.g., 'h100', 'h200', 'b200') (optional)
        target: Option<TargetType>,

        #[command(flatten)]
        options: UpOptions,
    },

    /// List active rentals and their status
    Ps {
        #[command(flatten)]
        filters: PsFilters,
    },

    /// Check instance status
    Status {
        /// Rental UUID (optional)
        target: Option<String>,
    },

    /// View instance logs
    Logs {
        /// Rental UUID (optional)
        target: Option<String>,

        #[command(flatten)]
        options: LogsOptions,
    },

    /// Terminate instance
    #[command(alias = "stop")]
    Down {
        /// Rental UUID to terminate (optional)
        target: Option<String>,

        /// Stop all active rentals
        #[arg(long, conflicts_with = "target")]
        all: bool,
    },

    /// Execute commands on instances
    Exec {
        /// Command to execute
        command: String,

        /// Rental UUID (optional)
        #[arg(long)]
        target: Option<String>,
    },

    /// SSH into instances
    #[command(alias = "connect")]
    Ssh {
        /// Rental UUID (optional)
        target: Option<String>,

        #[command(flatten)]
        options: SshOptions,
    },

    /// Copy files to/from instances
    Cp {
        /// Source path (local or remote)
        #[arg(value_hint = ValueHint::AnyPath)]
        source: String,

        /// Destination path (local or remote)
        #[arg(value_hint = ValueHint::AnyPath)]
        destination: String,
    },

    /// Run validator (delegates to basilica-validator)
    #[command(disable_help_flag = true)]
    Validator {
        /// Arguments to pass to basilica-validator
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },

    /// Run miner (delegates to basilica-miner)
    #[command(disable_help_flag = true)]
    Miner {
        /// Arguments to pass to basilica-miner
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },

    /// Log in to Basilica
    Login {
        /// Use device authorization flow (for WSL, SSH, containers)
        #[arg(long)]
        device_code: bool,
    },

    /// Log out of Basilica
    Logout,

    /// Test authentication token
    #[cfg(debug_assertions)]
    TestAuth {
        /// Test against Basilica API instead of Auth0
        #[arg(long)]
        api: bool,
    },

    /// Tokens management commands
    Tokens {
        #[command(subcommand)]
        action: TokenAction,
    },
}

/// Token management actions
#[derive(Subcommand, Debug, Clone)]
pub enum TokenAction {
    /// Create a new API key
    Create {
        /// Name for the API key (will prompt if not provided)
        name: Option<String>,
    },

    /// List all API keys
    List,

    /// Revoke an API key
    Revoke {
        /// Name of the API key to revoke (will prompt if not provided)
        name: Option<String>,

        /// Skip confirmation prompt
        #[arg(long, short = 'y')]
        yes: bool,
    },
}

impl Commands {
    /// Check if this command requires authentication
    pub fn requires_auth(&self) -> bool {
        match self {
            // GPU rental commands require authentication
            Commands::Ls { .. }
            | Commands::Up { .. }
            | Commands::Ps { .. }
            | Commands::Status { .. }
            | Commands::Logs { .. }
            | Commands::Down { .. }
            | Commands::Exec { .. }
            | Commands::Ssh { .. }
            | Commands::Cp { .. }
            | Commands::Tokens { .. } => true,

            // Authentication and delegation commands don't require auth
            Commands::Login { .. }
            | Commands::Logout
            | Commands::Validator { .. }
            | Commands::Miner { .. } => false,

            // Test auth command requires authentication
            #[cfg(debug_assertions)]
            Commands::TestAuth { .. } => true,
        }
    }
}

/// Filters for listing GPUs
#[derive(clap::Args, Debug, Clone)]
pub struct ListFilters {
    /// Minimum GPU count
    #[arg(long)]
    pub gpu_min: Option<u32>,

    /// Maximum GPU count
    #[arg(long)]
    pub gpu_max: Option<u32>,

    /// Maximum price per hour
    #[arg(long)]
    pub price_max: Option<f64>,

    /// Minimum memory in GB
    #[arg(long)]
    pub memory_min: Option<u32>,

    /// Filter by country code (e.g., US, UK, DE)
    #[arg(long)]
    pub country: Option<String>,

    /// Use compact view (group by country and GPU type)
    #[arg(long)]
    pub compact: bool,

    /// Use detailed view (shows node IDs)
    #[arg(long)]
    pub detailed: bool,
}

/// Options for provisioning instances
#[derive(clap::Args, Debug, Clone)]
pub struct UpOptions {
    /// Exact GPU count required
    #[arg(long)]
    pub gpu_count: Option<u32>,

    /// Docker image to run
    #[arg(long)]
    pub image: Option<String>,

    /// Environment variables (KEY=VALUE)
    #[arg(long)]
    pub env: Vec<String>,

    /// Instance name
    #[arg(long)]
    pub name: Option<String>,

    /// SSH public key file path
    #[arg(long, value_hint = ValueHint::FilePath)]
    pub ssh_key: Option<PathBuf>,

    /// Port mappings (host:container)
    #[arg(long)]
    pub ports: Vec<String>,

    /// CPU cores
    #[arg(long)]
    pub cpu_cores: Option<f64>,

    /// Memory in MB
    #[arg(long)]
    pub memory_mb: Option<i64>,

    /// Storage in MB
    #[arg(long)]
    pub storage_mb: Option<i64>,

    /// Command to run
    #[arg(long)]
    pub command: Vec<String>,

    /// Filter by country code (e.g., US, UK, DE)
    #[arg(long)]
    pub country: Option<String>,

    /// Disable SSH access (faster startup)
    #[arg(long)]
    pub no_ssh: bool,

    /// Create rental in detached mode (don't auto-connect via SSH)
    #[arg(short = 'd', long)]
    pub detach: bool,

    /// Use compact view (group nodes by GPU type)
    #[arg(long)]
    pub compact: bool,

    /// Use detailed view (shows node IDs during selection)
    #[arg(long)]
    pub detailed: bool,
}

/// Filters for listing active rentals
#[derive(clap::Args, Debug, Clone)]
pub struct PsFilters {
    /// Filter by status (defaults to 'active' if not specified)
    #[arg(long, value_enum)]
    pub status: Option<RentalState>,

    /// Filter by GPU type
    #[arg(long)]
    pub gpu_type: Option<String>,

    /// Minimum GPU count
    #[arg(long)]
    pub min_gpu_count: Option<u32>,

    /// Use compact view (minimal columns)
    #[arg(long)]
    pub compact: bool,

    /// Use detailed view (shows rental and node IDs)
    #[arg(long)]
    pub detailed: bool,
}

/// Options for viewing logs
#[derive(clap::Args, Debug, Clone)]
pub struct LogsOptions {
    /// Follow logs in real-time
    #[arg(short, long)]
    pub follow: bool,

    /// Number of lines to tail
    #[arg(long)]
    pub tail: Option<u32>,
}

/// Options for SSH connections
#[derive(clap::Args, Debug, Clone)]
pub struct SshOptions {
    /// Local port forwarding (local_port:remote_host:remote_port)
    #[arg(short = 'L', long)]
    pub local_forward: Vec<String>,

    /// Remote port forwarding (remote_port:local_host:local_port)
    #[arg(short = 'R', long)]
    pub remote_forward: Vec<String>,
}
