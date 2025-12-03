use basilica_common::types::GpuCategory;
use basilica_sdk::types::RentalState;
use clap::{Subcommand, ValueEnum, ValueHint};
use std::path::PathBuf;

use crate::handlers::gpu_rental::GpuTarget;

/// CLI wrapper for ComputeCategory to implement ValueEnum
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum ComputeCategoryArg {
    #[value(name = "secure-cloud", alias = "secure", alias = "secure_cloud")]
    SecureCloud,
    #[value(
        name = "community-cloud",
        alias = "community",
        alias = "community_cloud"
    )]
    CommunityCloud,
}

impl From<ComputeCategoryArg> for basilica_common::types::ComputeCategory {
    fn from(arg: ComputeCategoryArg) -> Self {
        match arg {
            ComputeCategoryArg::SecureCloud => basilica_common::types::ComputeCategory::SecureCloud,
            ComputeCategoryArg::CommunityCloud => {
                basilica_common::types::ComputeCategory::CommunityCloud
            }
        }
    }
}

/// Main CLI commands
#[derive(Subcommand, Debug, Clone)]
pub enum Commands {
    /// List available GPU resources
    #[command(alias = "list")]
    Ls {
        /// Filter by GPU category (e.g., 'h100', 'h200', 'b200') (optional)
        gpu_type: Option<GpuCategory>,

        /// Compute source: 'secure-cloud' or 'community-cloud'
        #[arg(long, value_name = "TYPE")]
        compute: Option<ComputeCategoryArg>,

        #[command(flatten)]
        filters: ListFilters,
    },

    /// Provision and start GPU instances
    #[command(alias = "start")]
    Up {
        /// GPU category to filter by (e.g., 'h100', 'a100', 'b200') (optional)
        target: Option<GpuTarget>,

        /// Compute source: 'secure-cloud' or 'community-cloud'
        #[arg(long, value_name = "TYPE")]
        compute: Option<ComputeCategoryArg>,

        #[command(flatten)]
        options: UpOptions,
    },

    /// List active rentals and their status
    Ps {
        /// Compute source: 'secure-cloud' or 'community-cloud'
        #[arg(long, value_name = "TYPE")]
        compute: Option<ComputeCategoryArg>,

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

        /// Compute source: 'secure-cloud' or 'community-cloud'
        #[arg(long, value_name = "TYPE")]
        compute: Option<ComputeCategoryArg>,

        /// Stop all active rentals
        #[arg(long, conflicts_with = "target")]
        all: bool,
    },

    /// Restart instance container
    Restart {
        /// Rental UUID to restart (optional)
        target: Option<String>,
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

    /// SSH keys management commands
    SshKeys {
        #[command(subcommand)]
        action: SshKeyAction,
    },

    /// Fund your account with Bittensor TAO
    Fund {
        #[command(subcommand)]
        action: Option<FundAction>,

        /// Output as JSON
        #[arg(long, global = true)]
        json: bool,
    },

    /// Check your account balance
    Balance {
        /// Output as JSON
        #[arg(long, global = true)]
        json: bool,
    },

    /// Upgrade the Basilica CLI to a newer version
    Upgrade {
        /// Specific version to upgrade to (e.g., "0.5.4")
        #[arg(long)]
        version: Option<String>,

        /// Check for updates without installing
        #[arg(long)]
        dry_run: bool,
    },
}

/// Fund management actions
#[derive(Subcommand, Debug, Clone)]
pub enum FundAction {
    /// List deposit history
    List {
        /// Limit number of results (default: 50)
        #[arg(long, default_value = "50")]
        limit: u32,

        /// Offset for pagination (default: 0)
        #[arg(long, default_value = "0")]
        offset: u32,
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

/// SSH key management actions
#[derive(Subcommand, Debug, Clone)]
pub enum SshKeyAction {
    /// Add a new SSH key
    Add {
        /// Name for the SSH key (will prompt if not provided)
        #[arg(short, long)]
        name: Option<String>,

        /// Path to SSH public key file (default: auto-detect from ~/.ssh/)
        #[arg(short, long, value_hint = ValueHint::FilePath)]
        file: Option<PathBuf>,
    },

    /// List registered SSH keys
    List,

    /// Delete the registered SSH key
    Delete {
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
            | Commands::Restart { .. }
            | Commands::Exec { .. }
            | Commands::Ssh { .. }
            | Commands::Cp { .. }
            | Commands::Tokens { .. }
            | Commands::SshKeys { .. }
            | Commands::Fund { .. }
            | Commands::Balance { .. } => true,

            // Authentication commands don't require auth
            Commands::Login { .. } | Commands::Logout | Commands::Upgrade { .. } => false,

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
}

/// Options for provisioning instances
#[derive(clap::Args, Debug, Clone)]
pub struct UpOptions {
    /// Exact GPU count required
    #[arg(long)]
    pub gpu_count: Option<u32>,

    /// Docker image to run (community cloud only)
    #[arg(long)]
    pub image: Option<String>,

    /// Environment variables (KEY=VALUE) (community cloud only)
    #[arg(long)]
    pub env: Vec<String>,

    /// Instance name
    #[arg(long)]
    pub name: Option<String>,

    /// SSH public key file path
    #[arg(long, value_hint = ValueHint::FilePath)]
    pub ssh_key: Option<PathBuf>,

    /// Port mappings (host:container) (community cloud only)
    #[arg(long)]
    pub ports: Vec<String>,

    /// CPU cores (community cloud only)
    #[arg(long)]
    pub cpu_cores: Option<f64>,

    /// Memory in MB (community cloud only)
    #[arg(long)]
    pub memory_mb: Option<i64>,

    /// Storage in MB (community cloud only)
    #[arg(long)]
    pub storage_mb: Option<i64>,

    /// Command to run (community cloud only)
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

    /// Show rental history instead of active rentals
    #[arg(long)]
    pub history: bool,
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
