use basilica_common::types::GpuCategory;
use basilica_sdk::types::RentalState;
use clap::{Subcommand, ValueEnum, ValueHint};
use std::path::PathBuf;

use crate::handlers::gpu_rental::GpuTarget;

/// CLI wrapper for ComputeCategory to implement ValueEnum
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum ComputeCategoryArg {
    /// The Citadel - Datacenter providers
    #[value(
        name = "citadel",
        alias = "secure-cloud",
        alias = "secure",
        alias = "secure_cloud"
    )]
    SecureCloud,
    /// The Bourse - Miner-provided GPUs
    #[value(
        name = "bourse",
        alias = "community-cloud",
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

        /// Compute source: 'citadel' (The Citadel) or 'bourse' (The Bourse)
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

        /// Compute source: 'citadel' (The Citadel) or 'bourse' (The Bourse)
        #[arg(long, value_name = "TYPE")]
        compute: Option<ComputeCategoryArg>,

        #[command(flatten)]
        options: UpOptions,
    },

    /// List active rentals and their status
    Ps {
        /// Compute source: 'citadel' (The Citadel) or 'bourse' (The Bourse)
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

        /// Compute source: 'citadel' (The Citadel) or 'bourse' (The Bourse)
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
    },

    /// Check your account balance
    Balance,

    /// Upgrade the Basilica CLI to a newer version
    Upgrade {
        /// Specific version to upgrade to (e.g., "0.5.4")
        #[arg(long)]
        version: Option<String>,

        /// Check for updates without installing
        #[arg(long)]
        dry_run: bool,
    },

    /// Deploy applications to Basilica
    #[command(name = "deploy", visible_alias = "summon", alias = "d")]
    Deploy(Box<DeployCommand>),

    /// Volume management commands
    Volumes {
        #[command(subcommand)]
        action: VolumeAction,
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

/// Volume management actions
#[derive(Subcommand, Debug, Clone)]
pub enum VolumeAction {
    /// Create a new volume
    Create {
        /// Volume name (unique per user, will prompt if not provided)
        #[arg(short, long)]
        name: Option<String>,

        /// Volume size in GB (1-10240, will prompt if not provided)
        #[arg(short, long)]
        size: Option<u32>,

        /// Cloud provider (e.g., hyperstack, will prompt if not provided)
        #[arg(short, long)]
        provider: Option<String>,

        /// Region (e.g., US-1, CANADA-1, NORWAY-1, will prompt if not provided)
        #[arg(short, long)]
        region: Option<String>,

        /// Optional description
        #[arg(short, long)]
        description: Option<String>,
    },

    /// List all volumes
    #[command(alias = "ls")]
    List,

    /// Delete a volume (must be detached first)
    #[command(alias = "rm")]
    Delete {
        /// Volume ID or name (will prompt if not provided)
        volume: Option<String>,

        /// Skip confirmation prompt
        #[arg(long, short = 'y')]
        yes: bool,
    },

    /// Attach a volume to a rental
    Attach {
        /// Volume ID or name (will prompt if not provided)
        volume: Option<String>,

        /// Rental ID to attach to (will prompt if not provided)
        #[arg(long)]
        rental: Option<String>,
    },

    /// Detach a volume from its current rental
    Detach {
        /// Volume ID or name (will prompt if not provided)
        volume: Option<String>,

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
            | Commands::Volumes { .. }
            | Commands::Fund { .. }
            | Commands::Balance => true,

            // Deploy commands: most require auth, except Metadata (public endpoint)
            Commands::Deploy(cmd) => !matches!(cmd.action, Some(DeployAction::Metadata { .. })),

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

    /// Filter by interconnect type (e.g., SXM, PCIe)
    #[arg(long)]
    pub interconnect: Option<String>,

    /// Filter by region (geo codes: US, CA, EU, APAC or region substrings)
    #[arg(long)]
    pub region: Option<String>,

    /// Show only spot offerings
    #[arg(long, conflicts_with = "exclude_spot")]
    pub spot: bool,

    /// Exclude spot offerings
    #[arg(long, conflicts_with = "spot")]
    pub exclude_spot: bool,
}

/// Options for provisioning instances
#[derive(clap::Args, Debug, Clone)]
pub struct UpOptions {
    /// Exact GPU count required
    #[arg(long)]
    pub gpu_count: Option<u32>,

    /// Maximum USD per GPU-hour for community cloud rentals
    #[arg(long)]
    pub max_hourly_rate: Option<f64>,

    /// Docker image to run (Bourse only)
    #[arg(long)]
    pub image: Option<String>,

    /// Environment variables (KEY=VALUE) (Bourse only)
    #[arg(long)]
    pub env: Vec<String>,

    /// Instance name
    #[arg(long)]
    pub name: Option<String>,

    /// SSH public key file path
    #[arg(long, value_hint = ValueHint::FilePath)]
    pub ssh_key: Option<PathBuf>,

    /// Port mappings (host:container) (Bourse only)
    #[arg(long)]
    pub ports: Vec<String>,

    /// CPU cores (Bourse only)
    #[arg(long)]
    pub cpu_cores: Option<f64>,

    /// Memory in MB (Bourse only)
    #[arg(long)]
    pub memory_mb: Option<i64>,

    /// Storage in MB (Bourse only)
    #[arg(long)]
    pub storage_mb: Option<i64>,

    /// Command to run (Bourse only)
    #[arg(long)]
    pub command: Vec<String>,

    /// Filter by country code (e.g., US, UK, DE)
    #[arg(long)]
    pub country: Option<String>,

    /// Create rental in detached mode (don't auto-connect via SSH)
    #[arg(short = 'd', long)]
    pub detach: bool,

    /// Filter by interconnect type (e.g., SXM, PCIe)
    #[arg(long)]
    pub interconnect: Option<String>,

    /// Filter by region (geo codes: US, CA, EU, APAC or region substrings)
    #[arg(long)]
    pub region: Option<String>,

    /// Prefer spot offerings
    #[arg(long, conflicts_with = "exclude_spot")]
    pub spot: bool,

    /// Exclude spot offerings
    #[arg(long, conflicts_with = "spot")]
    pub exclude_spot: bool,
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

// ============================================================================
// Deploy Command Definitions
// ============================================================================

/// Deploy command with subcommands and options
#[derive(clap::Parser, Debug, Clone)]
pub struct DeployCommand {
    #[command(subcommand)]
    pub action: Option<DeployAction>,

    /// Source file or Docker image to deploy
    #[arg(value_name = "SOURCE")]
    pub source: Option<String>,

    #[command(flatten)]
    pub naming: NamingOptions,

    #[command(flatten)]
    pub resources: ResourceOptions,

    #[command(flatten)]
    pub gpu: GpuOptions,

    #[command(flatten)]
    pub storage: StorageOptions,

    #[command(flatten)]
    pub topology_spread: TopologySpreadOptions,

    #[command(flatten)]
    pub websocket: WebSocketOptions,

    #[command(flatten)]
    pub health: HealthCheckOptions,

    #[command(flatten)]
    pub networking: NetworkingOptions,

    #[command(flatten)]
    pub lifecycle: LifecycleOptions,

    /// Output as JSON
    #[arg(long, global = true)]
    pub json: bool,

    /// Show detailed deployment phases during progress
    #[arg(long, global = true)]
    pub show_phases: bool,
}

/// Naming and identification options
#[derive(clap::Args, Debug, Clone, Default)]
pub struct NamingOptions {
    /// Deployment name (auto-generated if not specified)
    #[arg(short, long)]
    pub name: Option<String>,

    /// Docker image (default: python:3.11-slim for .py files)
    #[arg(long)]
    pub image: Option<String>,

    /// Number of replicas (default: 1)
    #[arg(long, default_value = "1")]
    pub replicas: u32,
}

/// Resource allocation options (limits and requests)
#[derive(clap::Args, Debug, Clone)]
pub struct ResourceOptions {
    /// CPU limit (e.g., "500m", "2")
    #[arg(long, default_value = "500m")]
    pub cpu: String,

    /// Memory limit (e.g., "512Mi", "2Gi")
    #[arg(long, default_value = "512Mi")]
    pub memory: String,

    /// CPU request (defaults to cpu limit)
    #[arg(long)]
    pub cpu_request: Option<String>,

    /// Memory request (defaults to memory limit)
    #[arg(long)]
    pub memory_request: Option<String>,
}

impl Default for ResourceOptions {
    fn default() -> Self {
        Self {
            cpu: "500m".to_string(),
            memory: "512Mi".to_string(),
            cpu_request: None,
            memory_request: None,
        }
    }
}

/// GPU configuration options
#[derive(clap::Args, Debug, Clone, Default)]
pub struct GpuOptions {
    /// Number of GPUs (1-8)
    #[arg(long)]
    pub gpu: Option<u32>,

    /// GPU model requirements (e.g., "A100", "H100")
    #[arg(long)]
    pub gpu_model: Vec<String>,

    /// Minimum CUDA version (e.g., "12.0")
    #[arg(long)]
    pub cuda_version: Option<String>,

    /// Minimum GPU memory in GB
    #[arg(long)]
    pub gpu_memory_gb: Option<u32>,

    /// GPU vendor (nvidia, amd)
    #[arg(long)]
    pub gpu_vendor: Option<String>,

    /// Node selector labels (format: key=value, can be specified multiple times)
    #[arg(long, value_name = "KEY=VALUE")]
    pub node_selector: Vec<String>,

    /// Preferred node affinity labels (soft constraint, format: key=value)
    #[arg(long, value_name = "KEY=VALUE")]
    pub prefer_node: Vec<String>,

    /// Required node affinity labels (hard constraint, format: key=value)
    #[arg(long, value_name = "KEY=VALUE")]
    pub require_node: Vec<String>,

    /// GPU interconnect type (e.g., SXM, PCIe)
    #[arg(long)]
    pub interconnect: Option<String>,

    /// Region preference (geo codes: US, CA, EU, APAC)
    #[arg(long)]
    pub region: Option<String>,

    /// Prefer spot instances (preferredDuringScheduling)
    #[arg(long, conflicts_with = "exclude_spot")]
    pub spot: bool,

    /// Exclude spot instances (NotIn scheduling)
    #[arg(long, conflicts_with = "spot")]
    pub exclude_spot: bool,
}

/// Storage configuration options
#[derive(clap::Args, Debug, Clone)]
pub struct StorageOptions {
    /// Enable persistent storage
    #[arg(long)]
    pub storage: bool,

    /// Storage mount path (default: /data)
    #[arg(long, default_value = "/data")]
    pub storage_path: String,

    /// Storage cache size in MB (default: 2048)
    #[arg(long, default_value = "2048")]
    pub storage_cache_mb: usize,

    /// Storage sync interval in ms (default: 1000)
    #[arg(long, default_value = "1000")]
    pub storage_sync_ms: u64,
}

impl Default for StorageOptions {
    fn default() -> Self {
        Self {
            storage: false,
            storage_path: "/data".to_string(),
            storage_cache_mb: 2048,
            storage_sync_ms: 1000,
        }
    }
}

/// CLI argument for spread mode
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum SpreadModeArg {
    /// Best-effort spreading (ScheduleAnyway)
    #[default]
    Preferred,
    /// Strict spreading (DoNotSchedule)
    Required,
    /// One pod per node (pod anti-affinity)
    #[value(name = "unique-nodes", alias = "unique_nodes")]
    UniqueNodes,
}

/// Topology spread configuration options
#[derive(clap::Args, Debug, Clone)]
pub struct TopologySpreadOptions {
    /// Pod spreading mode: preferred, required, or unique-nodes
    /// - preferred: Best-effort spreading (default)
    /// - required: Strict spreading, pods fail to schedule if constraints unsatisfied
    /// - unique-nodes: One pod per node guaranteed (for unique IP requirements)
    #[arg(long, value_name = "MODE")]
    pub spread_mode: Option<SpreadModeArg>,

    /// Shorthand for --spread-mode unique-nodes (one pod per node)
    #[arg(long, conflicts_with = "spread_mode")]
    pub unique_nodes: bool,

    /// Maximum skew for pod spreading (1-10, default: 1)
    /// Only applies to preferred and required modes
    #[arg(long, default_value = "1")]
    pub max_skew: i32,

    /// Topology key for spreading
    /// - kubernetes.io/hostname (default, node-level)
    /// - topology.kubernetes.io/zone (zone-level)
    /// - topology.kubernetes.io/region (region-level)
    #[arg(long, default_value = "kubernetes.io/hostname")]
    pub topology_key: String,
}

impl Default for TopologySpreadOptions {
    fn default() -> Self {
        Self {
            spread_mode: None,
            unique_nodes: false,
            max_skew: 1,
            topology_key: "kubernetes.io/hostname".to_string(),
        }
    }
}

/// WebSocket configuration options
#[derive(clap::Args, Debug, Clone)]
pub struct WebSocketOptions {
    /// Enable WebSocket support for long-lived connections
    #[arg(long)]
    pub websocket: bool,

    /// WebSocket idle timeout in seconds (60-3600, default: 1800)
    #[arg(long, default_value = "1800")]
    pub ws_idle_timeout: u32,
}

impl Default for WebSocketOptions {
    fn default() -> Self {
        Self {
            websocket: false,
            ws_idle_timeout: 1800,
        }
    }
}

/// Health check configuration options
#[derive(clap::Args, Debug, Clone, Default)]
pub struct HealthCheckOptions {
    /// HTTP path for liveness probe
    #[arg(long)]
    pub liveness_path: Option<String>,

    /// HTTP path for readiness probe
    #[arg(long)]
    pub readiness_path: Option<String>,

    /// HTTP path for startup probe (delays liveness/readiness until app is ready)
    #[arg(long)]
    pub startup_path: Option<String>,

    /// Shorthand for all probes (same path)
    #[arg(long)]
    pub health_path: Option<String>,

    /// Port for health probes (defaults to primary container port)
    #[arg(long)]
    pub health_port: Option<u16>,

    /// Initial delay before liveness/readiness probes start (seconds)
    #[arg(long, default_value = "30")]
    pub health_initial_delay: u32,

    /// Probe interval (seconds)
    #[arg(long, default_value = "10")]
    pub health_period: u32,

    /// Probe timeout (seconds)
    #[arg(long, default_value = "5")]
    pub health_timeout: u32,

    /// Failure threshold before restart
    #[arg(long, default_value = "3")]
    pub health_failure_threshold: u32,

    /// Startup probe failure threshold (higher for slow-starting apps)
    #[arg(long, default_value = "30")]
    pub startup_failure_threshold: u32,
}

/// Networking options
#[derive(clap::Args, Debug, Clone)]
pub struct NetworkingOptions {
    /// Container ports (format: PORT[:NAME], e.g., 8000:http, 9090:metrics)
    #[arg(short, long, value_name = "PORT[:NAME]", default_value = "8000")]
    pub port: Vec<String>,

    /// Make deployment private (requires share token for access).
    /// By default, deployments are public.
    #[arg(long)]
    pub private: bool,

    /// Environment variables (KEY=VALUE)
    #[arg(short, long, value_name = "KEY=VALUE")]
    pub env: Vec<String>,

    /// Additional pip packages to install
    #[arg(long, num_args = 1..)]
    pub pip: Vec<String>,

    /// Enroll deployment in public metadata for validator verification
    #[arg(long)]
    pub public_metadata: bool,
}

impl NetworkingOptions {
    /// Resolve whether deployment should be public.
    /// Default is public unless --private is specified.
    pub fn is_public(&self) -> bool {
        !self.private
    }
}

impl Default for NetworkingOptions {
    fn default() -> Self {
        Self {
            port: vec!["8000".to_string()],
            private: false,
            env: vec![],
            pip: vec![],
            public_metadata: false,
        }
    }
}

/// Lifecycle options
#[derive(clap::Args, Debug, Clone)]
pub struct LifecycleOptions {
    /// Time-to-live in seconds (60-604800, auto-delete after expiry)
    #[arg(long)]
    pub ttl: Option<u32>,

    /// Deployment timeout in seconds
    #[arg(long, default_value = "300")]
    pub timeout: u32,

    /// Don't wait for deployment to be ready
    #[arg(long)]
    pub detach: bool,

    /// Termination grace period in seconds
    #[arg(long, default_value = "30")]
    pub grace_period: u32,

    /// Skip GPU resource correlation validation (use with caution)
    #[arg(long)]
    pub skip_gpu_validation: bool,
}

impl Default for LifecycleOptions {
    fn default() -> Self {
        Self {
            ttl: None,
            timeout: 300,
            detach: false,
            grace_period: 30,
            skip_gpu_validation: false,
        }
    }
}

/// Deploy subcommands
#[derive(Subcommand, Debug, Clone)]
pub enum DeployAction {
    /// List all deployments
    #[command(name = "ls", visible_alias = "list")]
    List,

    /// Get deployment status
    #[command(name = "status", visible_alias = "get")]
    Status {
        /// Deployment name (interactive selection if omitted)
        name: Option<String>,
        /// Show share token status for private deployments
        #[arg(long)]
        show_token: bool,
    },

    /// Stream deployment logs
    #[command(name = "logs")]
    Logs {
        /// Deployment name (interactive selection if omitted)
        name: Option<String>,
        /// Follow log output
        #[arg(short, long)]
        follow: bool,
        /// Number of lines to show from end
        #[arg(long)]
        tail: Option<u32>,
    },

    /// Delete a deployment
    #[command(name = "delete", visible_alias = "rm")]
    Delete {
        /// Deployment name (interactive selection if omitted)
        name: Option<String>,
        /// Skip confirmation
        #[arg(short, long)]
        yes: bool,
    },

    /// Scale deployment replicas
    #[command(name = "scale")]
    Scale {
        /// Deployment name (interactive selection if omitted)
        name: Option<String>,
        /// Number of replicas
        #[arg(long)]
        replicas: u32,
    },

    /// Restart a deployment (rolling restart)
    #[command(name = "restart")]
    Restart {
        /// Deployment name (interactive selection if omitted)
        name: Option<String>,
    },

    /// Manage share tokens for private deployments
    #[command(name = "share-token")]
    ShareToken {
        #[command(subcommand)]
        action: ShareTokenAction,
    },

    /// Manage public metadata enrollment for validator verification
    #[command(name = "enroll-metadata")]
    EnrollMetadata {
        /// Deployment name (interactive selection if omitted)
        name: Option<String>,
        /// Enable public metadata enrollment
        #[arg(long)]
        enable: bool,
        /// Disable public metadata enrollment
        #[arg(long, conflicts_with = "enable")]
        disable: bool,
    },

    /// View public metadata for a deployment (no authentication required)
    #[command(name = "metadata")]
    Metadata {
        /// Instance name to look up
        name: String,
    },

    /// Deploy vLLM inference server
    #[command(name = "vllm")]
    Vllm {
        /// HuggingFace model ID (default: Qwen/Qwen3-0.6B)
        model: Option<String>,

        #[command(flatten)]
        common: TemplateCommonOptions,

        #[command(flatten)]
        vllm: VllmOptions,
    },

    /// Deploy SGLang inference server
    #[command(name = "sglang")]
    Sglang {
        /// HuggingFace model ID (default: Qwen/Qwen2.5-0.5B-Instruct)
        model: Option<String>,

        #[command(flatten)]
        common: TemplateCommonOptions,

        #[command(flatten)]
        sglang: SglangOptions,
    },

    /// Deploy OpenClaw gateway
    #[command(name = "openclaw")]
    Openclaw {
        #[command(flatten)]
        common: TemplateCommonOptions,

        #[command(flatten)]
        openclaw: OpenclawOptions,
    },

    /// Deploy Tau agent
    #[command(name = "tau")]
    Tau {
        #[command(flatten)]
        common: TemplateCommonOptions,

        #[command(flatten)]
        tau: TauOptions,
    },
}

/// Share token management actions
#[derive(Subcommand, Debug, Clone)]
pub enum ShareTokenAction {
    /// Regenerate share token (creates new token, invalidates previous)
    #[command(name = "regenerate", visible_alias = "create")]
    Regenerate {
        /// Deployment name (interactive selection if omitted)
        name: Option<String>,
    },

    /// Check if share token exists for a deployment
    #[command(name = "status")]
    Status {
        /// Deployment name (interactive selection if omitted)
        name: Option<String>,
    },

    /// Revoke share token (deployment becomes inaccessible via share URL)
    #[command(name = "revoke", visible_alias = "delete")]
    Revoke {
        /// Deployment name (interactive selection if omitted)
        name: Option<String>,

        /// Skip confirmation prompt
        #[arg(short = 'y', long)]
        yes: bool,
    },
}

/// Common options for deployment templates (vLLM, SGLang, etc.)
#[derive(clap::Args, Debug, Clone, Default)]
pub struct TemplateCommonOptions {
    /// Deployment name (auto-generated if not specified)
    #[arg(short, long)]
    pub name: Option<String>,

    /// Number of GPUs (auto-detected if not specified)
    #[arg(long)]
    pub gpu: Option<u32>,

    /// GPU model requirements (e.g., "A100", "H100")
    #[arg(long)]
    pub gpu_model: Vec<String>,

    /// Memory allocation (default: 16Gi)
    #[arg(long, default_value = "16Gi")]
    pub memory: String,

    /// Disable persistent storage cache
    #[arg(long)]
    pub no_storage: bool,

    /// Time-to-live in seconds
    #[arg(long)]
    pub ttl: Option<u32>,

    /// Deployment timeout in seconds
    #[arg(long, default_value = "600")]
    pub timeout: u32,

    /// Environment variables (KEY=VALUE)
    #[arg(short, long, value_name = "KEY=VALUE")]
    pub env: Vec<String>,

    /// Don't wait for deployment to be ready
    #[arg(long)]
    pub detach: bool,

    /// Output as JSON
    #[arg(long)]
    pub json: bool,
}

/// vLLM-specific deployment options
#[derive(clap::Args, Debug, Clone, Default)]
pub struct VllmOptions {
    /// Tensor parallel size (number of GPUs for parallelism)
    #[arg(long)]
    pub tensor_parallel_size: Option<u32>,

    /// Maximum model length (sequence length)
    #[arg(long)]
    pub max_model_len: Option<u32>,

    /// Model dtype (auto, float16, bfloat16, float32)
    #[arg(long)]
    pub dtype: Option<String>,

    /// Quantization method (awq, gptq, squeezellm, fp8)
    #[arg(long)]
    pub quantization: Option<String>,

    /// OpenAI API model name
    #[arg(long)]
    pub served_model_name: Option<String>,

    /// API key for authentication
    #[arg(long)]
    pub api_key: Option<String>,

    /// GPU memory utilization (0.0-1.0)
    #[arg(long)]
    pub gpu_memory_utilization: Option<f32>,

    /// Disable CUDA graphs (use eager mode)
    #[arg(long)]
    pub enforce_eager: bool,

    /// Trust remote code from HuggingFace
    #[arg(long)]
    pub trust_remote_code: bool,
}

/// SGLang-specific deployment options
#[derive(clap::Args, Debug, Clone, Default)]
pub struct SglangOptions {
    /// Tensor parallel size (number of GPUs for parallelism)
    #[arg(long)]
    pub tensor_parallel_size: Option<u32>,

    /// Maximum context length
    #[arg(long)]
    pub context_length: Option<u32>,

    /// Quantization method
    #[arg(long)]
    pub quantization: Option<String>,

    /// Static memory fraction (0.0-1.0)
    #[arg(long)]
    pub mem_fraction_static: Option<f32>,

    /// Trust remote code from HuggingFace
    #[arg(long)]
    pub trust_remote_code: bool,
}

/// OpenClaw-specific deployment options
#[derive(clap::Args, Debug, Clone)]
pub struct OpenclawOptions {
    /// Provider preset (openai, anthropic)
    #[arg(long, value_name = "PROVIDER", default_value = "openai")]
    pub provider: OpenclawProvider,

    /// Backend URL for OpenClaw (OpenAI-compatible API base URL)
    #[arg(long, value_name = "URL")]
    pub backend_url: Option<String>,

    /// Model ID to use (e.g., Qwen/Qwen2.5-7B-Instruct)
    #[arg(long, value_name = "MODEL_ID")]
    pub model_id: Option<String>,

    /// Provider ID (default: openai)
    #[arg(long, value_name = "PROVIDER_ID")]
    pub provider_id: Option<String>,

    /// Provider API type (default: openai-completions)
    #[arg(long, value_name = "API")]
    pub provider_api: Option<String>,

    /// Context window size (default: 32768)
    #[arg(long, value_name = "TOKENS")]
    pub context_window: Option<u32>,

    /// Max tokens (default: 8192)
    #[arg(long, value_name = "TOKENS")]
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum OpenclawProvider {
    Openai,
    Anthropic,
}
/// Tau-specific deployment options
#[derive(clap::Args, Debug, Clone)]
pub struct TauOptions {
    /// Telegram bot token (from `@BotFather`)
    #[arg(
        long,
        value_name = "TOKEN",
        env = "TAU_BOT_TOKEN",
        hide_env_values = true
    )]
    pub bot_token: Option<String>,

    /// Chutes API token for Tau's LLM + voice backend
    #[arg(
        long,
        value_name = "TOKEN",
        env = "CHUTES_API_TOKEN",
        hide_env_values = true
    )]
    pub chutes_api_token: Option<String>,

    /// Chat model override for Tau (maps to TAU_CHAT_MODEL)
    #[arg(long, value_name = "MODEL", env = "TAU_CHAT_MODEL")]
    pub chat_model: Option<String>,
}
