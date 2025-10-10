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

    /// Test SSH connection to node machines
    Connect {
        /// SSH hostname or IP address
        #[arg(long)]
        host: Option<String>,

        /// SSH username
        #[arg(long)]
        username: Option<String>,

        /// SSH port (default: 22)
        #[arg(long)]
        port: Option<u16>,

        /// Path to SSH private key
        #[arg(long)]
        private_key: Option<PathBuf>,

        /// Connection timeout in seconds (default: 30)
        #[arg(long)]
        timeout: Option<u64>,

        /// Node ID to connect to (alternative to host/username)
        #[arg(long)]
        node_id: Option<String>,
    },

    /// Verify node hardware via SSH validation protocol
    Verify {
        /// SSH hostname or IP address
        #[arg(long)]
        host: Option<String>,

        /// SSH username
        #[arg(long)]
        username: Option<String>,

        /// SSH port (default: 22)
        #[arg(long)]
        port: Option<u16>,

        /// Path to SSH private key
        #[arg(long)]
        private_key: Option<PathBuf>,

        /// Connection timeout in seconds (default: 30)
        #[arg(long)]
        timeout: Option<u64>,

        /// Node ID to verify
        #[arg(short, long)]
        node_id: Option<String>,

        /// Miner UID to verify all nodes
        #[arg(short, long)]
        miner_uid: Option<u16>,

        /// Path to gpu-attestor binary
        #[arg(long)]
        gpu_attestor_path: Option<PathBuf>,

        /// Remote working directory (default: /tmp/basilica_validation)
        #[arg(long)]
        remote_work_dir: Option<String>,

        /// Execution timeout in seconds (default: 300)
        #[arg(long)]
        execution_timeout: Option<u64>,

        /// Skip cleanup after verification
        #[arg(long)]
        skip_cleanup: bool,

        /// Enable verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Legacy verification command (deprecated)
    #[deprecated(note = "Use 'verify' command instead")]
    VerifyLegacy {
        #[arg(short, long)]
        miner_uid: Option<u16>,

        #[arg(short, long)]
        node_id: Option<String>,

        #[arg(long)]
        all: bool,
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
        /// Node ID
        #[arg(long)]
        node: String,

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

        /// GPU count
        #[arg(long)]
        gpu_count: Option<u32>,

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
