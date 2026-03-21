use clap::Subcommand;
use std::path::PathBuf;

#[derive(Subcommand, Debug, Clone)]
pub enum Command {
    Start,

    Stop,

    Status,

    GenConfig {
        #[arg(short, long, default_value = "validator.toml")]
        output: PathBuf,
    },

    /// Container rental commands
    Rental {
        #[command(subcommand)]
        action: RentalAction,

        /// API URL override (default: from config)
        #[arg(long, global = true)]
        api_url: Option<String>,
    },
}

#[derive(Subcommand, Debug, Clone)]
#[allow(dead_code, unused_imports, clippy::large_enum_variant)]
pub enum RentalAction {
    /// Start a new container rental
    Start {
        /// GPU category (e.g., "H100", "A100", "B200")
        #[arg(long)]
        gpu_category: String,

        /// Number of GPUs required
        #[arg(long, default_value = "1")]
        gpu_count: u32,

        /// Minimum GPU memory in GB (optional)
        #[arg(long)]
        min_memory_gb: Option<u32>,

        /// Maximum hourly rate per GPU in cents
        #[arg(long)]
        max_hourly_rate_cents: u32,

        /// Docker image to deploy (e.g., ubuntu:22.04, nginx:alpine)
        #[arg(long)]
        image: String,

        /// Port mappings (format: host:container:protocol)
        #[arg(long)]
        ports: Vec<String>,

        /// Environment variables (format: KEY=VALUE)
        #[arg(long)]
        env: Vec<String>,

        /// End-user's SSH public key (e.g., "ssh-rsa AAAA...")
        #[arg(long)]
        ssh_public_key: String,

        /// Command to run in container
        #[arg(long, num_args = 0..)]
        command: Vec<String>,

        /// CPU cores
        #[arg(long)]
        cpu_cores: Option<f64>,

        /// Memory in MB
        #[arg(long)]
        memory_mb: Option<i64>,

        /// Storage size in MB (default: 102400 MB / 100 GB)
        #[arg(long)]
        storage_mb: Option<i64>,
    },

    /// Get rental status
    Status {
        /// Rental ID
        #[arg(long)]
        id: String,
    },

    /// Stream rental logs
    Logs {
        /// Rental ID
        #[arg(long)]
        id: String,

        /// Follow logs
        #[arg(long)]
        follow: bool,

        /// Number of lines to tail
        #[arg(long)]
        tail: Option<u32>,
    },

    /// Stop a rental
    Stop {
        /// Rental ID
        #[arg(long)]
        id: String,

        /// Force stop
        #[arg(long)]
        force: bool,
    },

    /// List available nodes for rental
    Ls {
        /// Filter by minimum GPU memory in GB
        #[arg(long)]
        memory_min: Option<u32>,

        /// Filter by GPU type (e.g., A100, RTX4090)
        #[arg(long)]
        gpu_type: Option<String>,

        /// Filter by minimum GPU count
        #[arg(long)]
        gpu_min: Option<u32>,
    },

    /// List active rentals
    Ps {
        /// Filter by state (active, stopped, all)
        #[arg(long, default_value = "all")]
        state: String,
    },
}
