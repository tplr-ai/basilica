//! CLI command definitions
//!
//! Defines all available CLI commands for executor capabilities.

use super::{CliCommand, CliContext};
use anyhow::Result;
use async_trait::async_trait;
use clap::Subcommand;

#[derive(Subcommand, Debug, Clone)]
pub enum ValidatorCommands {
    Grant {
        #[arg(long)]
        hotkey: String,
        #[arg(short, long, default_value = "ssh")]
        access_type: String,
        #[arg(short, long, default_value = "24")]
        duration: u64,
        #[arg(short = 'k', long)]
        ssh_public_key: Option<String>,
    },
    Revoke {
        #[arg(long)]
        hotkey: String,
    },
    List,
    Logs {
        #[arg(long)]
        hotkey: Option<String>,
        #[arg(short, long, default_value = "50")]
        limit: u32,
    },
}

#[derive(Subcommand, Debug, Clone)]
pub enum SystemCommands {
    Status,
    Profile,
    Resources,
    Monitor {
        #[arg(short, long, default_value = "5")]
        interval: u64,
    },
}

#[derive(Subcommand, Debug, Clone)]
pub enum ContainerCommands {
    Create {
        #[arg(short, long)]
        image: String,
        #[arg(short, long)]
        command: String,
        #[arg(long)]
        memory: Option<u64>,
        #[arg(long)]
        cpu: Option<f64>,
        /// Stream output to stdout/stderr in real-time
        #[arg(short, long)]
        stream: bool,
    },
    Exec {
        container_id: String,
        command: String,
        #[arg(short, long)]
        timeout: Option<u64>,
    },
    Remove {
        container_id: String,
        #[arg(short, long)]
        force: bool,
    },
    List,
    Logs {
        container_id: String,
        #[arg(short, long)]
        follow: bool,
        #[arg(short, long)]
        tail: Option<i32>,
    },
    Status {
        container_id: String,
    },
    Cleanup,
}

#[derive(Subcommand, Debug, Clone)]
pub enum ResourceCommands {
    Show,
    SetLimits {
        #[arg(long)]
        cpu: Option<f32>,
        #[arg(long)]
        memory: Option<f32>,
        #[arg(long)]
        gpu_memory: Option<f32>,
    },
    Stats {
        #[arg(short, long, default_value = "24")]
        hours: u32,
    },
    Optimize,
}

#[derive(Subcommand, Debug, Clone)]
pub enum NetworkCommands {
    Status,
    Test {
        #[arg(short, long)]
        host: Option<String>,
    },
    Grpc,
    Metrics,
    SpeedTest,
}

#[derive(Subcommand, Debug, Clone)]
pub enum ServiceCommands {
    Start,
    Stop,
    Restart,
    Status,
    Health,
    Reload,
    Logs {
        #[arg(short, long, default_value = "100")]
        lines: u32,
        #[arg(short, long)]
        follow: bool,
    },
}

#[async_trait]
impl CliCommand for ValidatorCommands {
    async fn execute(&self, context: &CliContext) -> Result<()> {
        crate::cli::handlers::validator::handle_validator_command(self, context).await
    }
}

#[async_trait]
impl CliCommand for SystemCommands {
    async fn execute(&self, context: &CliContext) -> Result<()> {
        crate::cli::handlers::system::handle_system_command(self, context).await
    }
}

#[async_trait]
impl CliCommand for ContainerCommands {
    async fn execute(&self, context: &CliContext) -> Result<()> {
        crate::cli::handlers::container::handle_container_command(self, context).await
    }
}

#[async_trait]
impl CliCommand for ResourceCommands {
    async fn execute(&self, context: &CliContext) -> Result<()> {
        crate::cli::handlers::resource::handle_resource_command(self, context).await
    }
}

#[async_trait]
impl CliCommand for NetworkCommands {
    async fn execute(&self, context: &CliContext) -> Result<()> {
        crate::cli::handlers::network::handle_network_command(self, context).await
    }
}

#[async_trait]
impl CliCommand for ServiceCommands {
    async fn execute(&self, context: &CliContext) -> Result<()> {
        crate::cli::handlers::service::handle_service_command(self, context).await
    }
}
