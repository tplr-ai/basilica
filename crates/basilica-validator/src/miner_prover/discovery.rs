//! # Miner Discovery
//!
//! Responsible for discovering and selecting miners from the Bittensor metagraph.
//! Implements Single Responsibility Principle by focusing only on miner discovery.

use super::types::MinerInfo;
use crate::metrics::ValidatorMetrics;
use anyhow::Result;
use basilica_common::identity::{Hotkey, MinerUid};
use bittensor::{AxonInfo, Service as BittensorService};
use std::sync::Arc;
use tracing::{debug, info, warn};

#[derive(Clone)]
pub struct MinerDiscovery {
    bittensor_service: Arc<BittensorService>,
    netuid: u16,
    metrics: Option<Arc<ValidatorMetrics>>,
}

impl MinerDiscovery {
    pub fn new(bittensor_service: Arc<BittensorService>, netuid: u16) -> Self {
        Self {
            bittensor_service,
            netuid,
            metrics: None,
        }
    }

    pub fn with_metrics(mut self, metrics: Arc<ValidatorMetrics>) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// Get list of miners that need verification using metagraph
    pub async fn get_miners_for_verification(&self) -> Result<Vec<MinerInfo>> {
        info!("Fetching ALL miners from metagraph for verification (no filtering)");

        let metagraph = self.bittensor_service.get_metagraph(self.netuid).await?;

        if metagraph.hotkeys.is_empty() {
            // Update metrics for zero discovered miners
            if let Some(metrics) = &self.metrics {
                metrics.prometheus().set_discovered_miners_total(0);
            }

            info!("No miners found in metagraph for netuid {}", self.netuid);
            return Ok(Vec::new());
        }

        let miners = self.extract_miners_from_metagraph(&metagraph)?;

        // Update discovered miners metrics
        if let Some(metrics) = &self.metrics {
            metrics
                .prometheus()
                .set_discovered_miners_total(miners.len() as u64);
        }

        info!(
            "Selected ALL {} miners for verification from {} total neurons",
            miners.len(),
            metagraph.hotkeys.len()
        );

        Ok(miners)
    }

    fn extract_miners_from_metagraph(
        &self,
        metagraph: &bittensor::Metagraph,
    ) -> Result<Vec<MinerInfo>> {
        let mut miners = Vec::new();

        for (uid, hotkey_account) in metagraph.hotkeys.iter().enumerate() {
            let uid = uid as u16;

            let is_validator = metagraph
                .validator_permit
                .get(uid as usize)
                .copied()
                .unwrap_or(false);

            if is_validator {
                continue;
            }

            let total_stake = metagraph
                .total_stake
                .get(uid as usize)
                .map(|s| s.0)
                .unwrap_or(0);
            let stake_tao = bittensor::rao_to_tao(total_stake);

            let endpoint = self.extract_endpoint(metagraph, uid as usize)?;
            if endpoint.is_none() {
                warn!("Miner {} has no axon info, skipping", uid);
                continue;
            }

            let miner = MinerInfo {
                uid: MinerUid::new(uid),
                hotkey: Hotkey::from_account_id(hotkey_account),
                endpoint: endpoint.unwrap(),
                is_validator,
                stake_tao,
                last_verified: None,
                verification_score: 0.0,
            };

            miners.push(miner);
        }

        Ok(miners)
    }

    fn extract_endpoint(
        &self,
        metagraph: &bittensor::Metagraph,
        uid: usize,
    ) -> Result<Option<String>> {
        Ok(metagraph.axons.get(uid).and_then(|axon| {
            // Validate that the IP is not zero (unset)
            if axon.ip == 0 {
                debug!("Miner {} has zero IP address, skipping", uid);
                return None;
            }

            // For IPv4, validate that we have a non-zero IP in the lower 32 bits
            if axon.ip_type == 4 {
                let ipv4_bits = axon.ip as u32;
                if ipv4_bits == 0 {
                    debug!(
                        "Miner {} has zero IPv4 address in lower 32 bits, skipping",
                        uid
                    );
                    return None;
                }
            }

            // Validate that the port is reasonable
            if axon.port == 0 {
                debug!("Miner {} has invalid port {}, skipping", uid, axon.port);
                return None;
            }

            let endpoint = self.axon_info_to_endpoint(axon);
            debug!(
                "Miner {} endpoint: {} (IP: {}, port: {})",
                uid, endpoint, axon.ip, axon.port
            );
            Some(endpoint)
        }))
    }

    /// Convert AxonInfo to endpoint string
    pub fn axon_info_to_endpoint(&self, axon: &AxonInfo) -> String {
        // Convert IP from u128 to string
        let ip = self.u128_to_ip(axon.ip, axon.ip_type);
        format!("http://{}:{}", ip, axon.port)
    }

    /// Convert u128 IP representation to string
    fn u128_to_ip(&self, ip: u128, ip_type: u8) -> String {
        if ip == 0 {
            warn!("Attempting to convert zero IP address to string");
            return "0.0.0.0".to_string();
        }

        if ip_type == 4 {
            // IPv4 - Extract from lower 32 bits (this is how miners actually store IPs)
            let ipv4_bits = ip as u32;
            let ip_bytes = ipv4_bits.to_be_bytes();
            format!(
                "{}.{}.{}.{}",
                ip_bytes[0], ip_bytes[1], ip_bytes[2], ip_bytes[3]
            )
        } else {
            // IPv6
            let bytes = ip.to_be_bytes();
            let segments: Vec<u16> = (0..8)
                .map(|i| u16::from_be_bytes([bytes[i * 2], bytes[i * 2 + 1]]))
                .collect();

            // Use standard IPv6 representation with :: compression for zeros
            let ipv6_str = segments
                .iter()
                .map(|&s| format!("{s:x}"))
                .collect::<Vec<_>>()
                .join(":");

            // Handle IPv6 zero compression (simplified)
            if ipv6_str == "0:0:0:0:0:0:0:0" {
                "::".to_string()
            } else {
                ipv6_str
            }
        }
    }
}
