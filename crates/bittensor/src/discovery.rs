//! # Neuron Discovery
//!
//! Provides functionality to discover neurons (miners and validators) from the Bittensor metagraph,
//! including their registration status and axon endpoints.

use anyhow::Result;
use std::collections::HashMap;
use std::net::SocketAddr;
use tracing::{debug, info, warn};

use crate::{AccountId, Metagraph};

/// Information about a discovered neuron
#[derive(Debug, Clone)]
pub struct NeuronInfo {
    /// The neuron's unique identifier
    pub uid: u16,
    /// The neuron's hotkey (SS58 address)
    pub hotkey: String,
    /// The neuron's coldkey (SS58 address)
    pub coldkey: String,
    /// The neuron's stake amount
    pub stake: u64,
    /// Whether this neuron is a validator
    pub is_validator: bool,
    /// The neuron's axon endpoint (if published)
    pub axon_info: Option<AxonInfo>,
}

/// Axon endpoint information
#[derive(Debug, Clone)]
pub struct AxonInfo {
    /// IP address
    pub ip: String,
    /// Port number
    pub port: u16,
    /// Protocol version
    pub version: u32,
    /// Full socket address
    pub socket_addr: SocketAddr,
}

/// Discovers neurons from the metagraph
pub struct NeuronDiscovery<'a> {
    metagraph: &'a Metagraph<AccountId>,
}

impl<'a> NeuronDiscovery<'a> {
    /// Create a new neuron discovery instance
    pub fn new(metagraph: &'a Metagraph<AccountId>) -> Self {
        Self { metagraph }
    }

    /// Get all neurons with their information
    pub fn get_all_neurons(&self) -> Result<Vec<NeuronInfo>> {
        let mut neurons = Vec::new();

        // Iterate through all UIDs based on hotkeys length
        for (idx, hotkey) in self.metagraph.hotkeys.iter().enumerate() {
            let uid = idx as u16;

            if let Some(axon_info) = self.extract_axon_info(uid) {
                debug!(
                    "Found neuron UID {} with axon endpoint {}:{}",
                    uid, axon_info.ip, axon_info.port
                );
            }

            // Get corresponding data from parallel arrays
            let coldkey = self
                .metagraph
                .coldkeys
                .get(idx)
                .map(|c| c.to_string())
                .unwrap_or_default();

            let stake = self
                .metagraph
                .total_stake
                .get(idx)
                .map(|s| s.0)
                .unwrap_or(0);

            let is_validator = self
                .metagraph
                .validator_permit
                .get(idx)
                .copied()
                .unwrap_or(false);

            neurons.push(NeuronInfo {
                uid,
                hotkey: hotkey.to_string(),
                coldkey,
                stake,
                is_validator,
                axon_info: self.extract_axon_info(uid),
            });
        }

        info!("Discovered {} neurons from metagraph", neurons.len());
        Ok(neurons)
    }

    /// Get only validators (neurons with validator permit)
    pub fn get_validators(&self) -> Result<Vec<NeuronInfo>> {
        let all_neurons = self.get_all_neurons()?;
        let validators: Vec<NeuronInfo> =
            all_neurons.into_iter().filter(|n| n.is_validator).collect();

        info!("Found {} validators in metagraph", validators.len());
        Ok(validators)
    }

    /// Get only miners (neurons without validator permit)
    pub fn get_miners(&self) -> Result<Vec<NeuronInfo>> {
        let all_neurons = self.get_all_neurons()?;
        let miners: Vec<NeuronInfo> = all_neurons
            .into_iter()
            .filter(|n| !n.is_validator)
            .collect();

        info!("Found {} miners in metagraph", miners.len());
        Ok(miners)
    }

    /// Get neurons with published axon endpoints
    pub fn get_neurons_with_axons(&self) -> Result<Vec<NeuronInfo>> {
        let all_neurons = self.get_all_neurons()?;
        let with_axons: Vec<NeuronInfo> = all_neurons
            .into_iter()
            .filter(|n| n.axon_info.is_some())
            .collect();

        info!("Found {} neurons with axon endpoints", with_axons.len());
        Ok(with_axons)
    }

    /// Find a specific neuron by hotkey
    pub fn find_neuron_by_hotkey(&self, hotkey: &str) -> Option<NeuronInfo> {
        for (idx, h) in self.metagraph.hotkeys.iter().enumerate() {
            if h.to_string() == hotkey {
                let uid = idx as u16;
                let coldkey = self
                    .metagraph
                    .coldkeys
                    .get(idx)
                    .map(|c| c.to_string())
                    .unwrap_or_default();

                let stake = self
                    .metagraph
                    .total_stake
                    .get(idx)
                    .map(|s| s.0)
                    .unwrap_or(0);

                let is_validator = self
                    .metagraph
                    .validator_permit
                    .get(idx)
                    .copied()
                    .unwrap_or(false);

                return Some(NeuronInfo {
                    uid,
                    hotkey: h.to_string(),
                    coldkey,
                    stake,
                    is_validator,
                    axon_info: self.extract_axon_info(uid),
                });
            }
        }
        None
    }

    /// Find a specific neuron by UID
    pub fn find_neuron_by_uid(&self, uid: u16) -> Option<NeuronInfo> {
        let idx = uid as usize;
        if idx >= self.metagraph.hotkeys.len() {
            return None;
        }

        let hotkey = self.metagraph.hotkeys.get(idx)?;
        let coldkey = self
            .metagraph
            .coldkeys
            .get(idx)
            .map(|c| c.to_string())
            .unwrap_or_default();

        let stake = self
            .metagraph
            .total_stake
            .get(idx)
            .map(|s| s.0)
            .unwrap_or(0);

        let is_validator = self
            .metagraph
            .validator_permit
            .get(idx)
            .copied()
            .unwrap_or(false);

        Some(NeuronInfo {
            uid,
            hotkey: hotkey.to_string(),
            coldkey,
            stake,
            is_validator,
            axon_info: self.extract_axon_info(uid),
        })
    }

    /// Check if a hotkey is registered in the metagraph
    pub fn is_hotkey_registered(&self, hotkey: &str) -> bool {
        self.find_neuron_by_hotkey(hotkey).is_some()
    }

    /// Get neurons filtered by minimum stake
    pub fn get_neurons_by_min_stake(&self, min_stake: u64) -> Result<Vec<NeuronInfo>> {
        let all_neurons = self.get_all_neurons()?;
        let filtered: Vec<NeuronInfo> = all_neurons
            .into_iter()
            .filter(|n| n.stake >= min_stake)
            .collect();

        info!(
            "Found {} neurons with stake >= {}",
            filtered.len(),
            min_stake
        );
        Ok(filtered)
    }

    /// Extract axon information for a specific UID
    pub fn extract_axon_info(&self, uid: u16) -> Option<AxonInfo> {
        self.metagraph.axons.get(uid as usize).and_then(|axon| {
            // Check if axon has valid IP and port
            if axon.ip == 0 || axon.port == 0 {
                return None;
            }

            // Convert IP to string format based on IP type
            let ip_str = if axon.ip_type == 4 {
                // IPv4 addresses are stored in the lower 32 bits of the u128
                let ipv4_bits = axon.ip as u32;
                let ip_bytes = ipv4_bits.to_be_bytes();
                format!(
                    "{}.{}.{}.{}",
                    ip_bytes[0], ip_bytes[1], ip_bytes[2], ip_bytes[3]
                )
            } else {
                // IPv6 handling - full u128
                format!("{:x}", axon.ip)
            };

            // Validate IP address
            if ip_str == "0.0.0.0" || ip_str == "127.0.0.1" {
                debug!("Skipping invalid axon IP {} for UID {}", ip_str, uid);
                return None;
            }

            // Create socket address
            match format!("{}:{}", ip_str, axon.port).parse::<SocketAddr>() {
                Ok(socket_addr) => Some(AxonInfo {
                    ip: ip_str,
                    port: axon.port,
                    version: axon.version,
                    socket_addr,
                }),
                Err(e) => {
                    warn!(
                        "Failed to parse socket address for UID {}: {}:{} - {}",
                        uid, ip_str, axon.port, e
                    );
                    None
                }
            }
        })
    }
}

/// Create a mapping of hotkeys to UIDs for quick lookup
pub fn create_hotkey_to_uid_map(metagraph: &Metagraph<AccountId>) -> HashMap<String, u16> {
    let mut map = HashMap::new();
    for (idx, hotkey) in metagraph.hotkeys.iter().enumerate() {
        map.insert(hotkey.to_string(), idx as u16);
    }
    map
}

/// Create a mapping of UIDs to axon endpoints
pub fn create_uid_to_axon_map(metagraph: &Metagraph<AccountId>) -> HashMap<u16, SocketAddr> {
    let discovery = NeuronDiscovery::new(metagraph);
    let mut map = HashMap::new();

    for idx in 0..metagraph.hotkeys.len() {
        let uid = idx as u16;
        if let Some(axon_info) = discovery.extract_axon_info(uid) {
            map.insert(uid, axon_info.socket_addr);
        }
    }
    map
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_ip_conversion() {
        // Test IP address conversion from u32 to string
        let ip_u32: u32 = 0xC0A80101; // 192.168.1.1
        let ip_bytes = ip_u32.to_be_bytes();
        let ip_str = format!(
            "{}.{}.{}.{}",
            ip_bytes[0], ip_bytes[1], ip_bytes[2], ip_bytes[3]
        );
        assert_eq!(ip_str, "192.168.1.1");
    }
}
