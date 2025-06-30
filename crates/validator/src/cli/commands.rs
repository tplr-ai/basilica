use clap::Subcommand;
use std::path::PathBuf;

#[derive(Subcommand, Debug, Clone)]
pub enum Command {
    Start {
        #[arg(long)]
        config: Option<PathBuf>,
    },

    Stop,

    Status,

    GenConfig {
        #[arg(short, long, default_value = "validator.toml")]
        output: PathBuf,
    },

    /// Test SSH connection to executor machines
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

        /// Executor ID to connect to (alternative to host/username)
        #[arg(long)]
        executor_id: Option<String>,
    },

    /// Verify executor hardware via SSH validation protocol
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

        /// Executor ID to verify
        #[arg(short, long)]
        executor_id: Option<String>,

        /// Miner UID to verify all executors
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
        executor_id: Option<String>,

        #[arg(long)]
        all: bool,
    },

    Weights {
        #[command(subcommand)]
        action: WeightAction,
    },

    Scores {
        #[command(subcommand)]
        action: ScoreAction,
    },

    Database {
        #[command(subcommand)]
        action: DatabaseAction,
    },
}

#[derive(Subcommand, Debug, Clone)]
pub enum WeightAction {
    Set {
        #[arg(long)]
        force: bool,
    },

    Show,

    History {
        #[arg(short, long, default_value = "10")]
        limit: u32,
    },
}

#[derive(Subcommand, Debug, Clone)]
pub enum ScoreAction {
    Show {
        #[arg(short, long)]
        miner_uid: Option<u16>,
    },

    Update {
        #[arg(short, long)]
        miner_uid: u16,

        #[arg(short, long)]
        score: f64,
    },

    Clear {
        #[arg(short, long)]
        miner_uid: Option<u16>,

        #[arg(long)]
        all: bool,
    },
}

#[derive(Subcommand, Debug, Clone)]
pub enum DatabaseAction {
    Migrate,

    Reset {
        #[arg(long)]
        confirm: bool,
    },

    Status,

    Cleanup {
        #[arg(long, default_value = "30")]
        days: u32,
    },
}
