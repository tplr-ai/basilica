//! Container log streaming functionality

use super::types::{ContainerLogEntry, LogLevel};
use anyhow::Result;
use bollard::{container::LogsOptions, Docker};
use futures_util::stream::StreamExt;
use tracing::info;

#[derive(Debug, Clone)]
pub struct LogStreamer {
    docker: Docker,
}

impl LogStreamer {
    pub fn new(docker: Docker) -> Self {
        Self { docker }
    }

    pub async fn stream_logs(
        &self,
        container_id: &str,
        follow: bool,
        tail_lines: Option<i32>,
    ) -> Result<impl futures_util::Stream<Item = ContainerLogEntry>> {
        info!(
            "Streaming logs for container: {} (follow: {}, tail: {:?})",
            container_id, follow, tail_lines
        );

        let logs_options = LogsOptions {
            follow,
            stdout: true,
            stderr: true,
            tail: tail_lines
                .map(|n| n.to_string())
                .unwrap_or_else(|| "all".to_string()),
            timestamps: true,
            ..Default::default()
        };

        let logs_stream = self.docker.logs(container_id, Some(logs_options));

        let container_id = container_id.to_string();
        let stream = logs_stream.map(move |log_result| match log_result {
            Ok(log) => {
                let message = String::from_utf8_lossy(&log.into_bytes()).to_string();
                ContainerLogEntry {
                    timestamp: chrono::Utc::now().timestamp(),
                    level: LogLevel::Info,
                    message,
                    container_id: container_id.clone(),
                }
            }
            Err(e) => ContainerLogEntry {
                timestamp: chrono::Utc::now().timestamp(),
                level: LogLevel::Error,
                message: format!("Log stream error: {e}"),
                container_id: container_id.clone(),
            },
        });

        Ok(stream)
    }
}
