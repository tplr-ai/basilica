//! Network monitoring functionality

use super::types::{NetworkInfo, NetworkInterface};
use anyhow::Result;
use sysinfo::Networks;

/// Network monitoring handler
#[derive(Debug)]
pub struct NetworkMonitor {
    networks: Networks,
}

impl NetworkMonitor {
    /// Create new network monitor
    pub fn new() -> Self {
        let networks = Networks::new_with_refreshed_list();
        Self { networks }
    }

    /// Get network information
    pub async fn get_network_info(&self) -> Result<NetworkInfo> {
        let mut interfaces = Vec::new();
        let mut total_sent = 0;
        let mut total_received = 0;

        for (interface_name, network) in &self.networks {
            let interface_info = NetworkInterface {
                name: interface_name.clone(),
                bytes_sent: network.total_transmitted(),
                bytes_received: network.total_received(),
                packets_sent: network.total_packets_transmitted(),
                packets_received: network.total_packets_received(),
                errors_sent: network.total_errors_on_transmitted(),
                errors_received: network.total_errors_on_received(),
                is_up: true,
            };

            total_sent += interface_info.bytes_sent;
            total_received += interface_info.bytes_received;
            interfaces.push(interface_info);
        }

        Ok(NetworkInfo {
            interfaces,
            total_bytes_sent: total_sent,
            total_bytes_received: total_received,
        })
    }

    /// Refresh network data
    pub fn refresh(&mut self) {
        self.networks.refresh();
    }
}

impl Default for NetworkMonitor {
    fn default() -> Self {
        Self::new()
    }
}
