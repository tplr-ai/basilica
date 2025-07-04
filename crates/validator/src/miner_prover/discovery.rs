//! # Miner Discovery
//!
//! Responsible for discovering and selecting miners from the Bittensor metagraph.
//! Implements Single Responsibility Principle by focusing only on miner discovery.

use super::types::MinerInfo;
use crate::config::VerificationConfig;
use anyhow::Result;
use bittensor::{AccountId, AxonInfo, Service as BittensorService};
use common::identity::{Hotkey, MinerUid};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Cached miner endpoint information
#[derive(Clone, Debug)]
struct CachedMinerEndpoint {
    axon_endpoint: String,
    cached_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Clone)]
pub struct MinerDiscovery {
    bittensor_service: Arc<BittensorService>,
    config: VerificationConfig,
    /// Cache of miner endpoints with TTL
    endpoint_cache: Arc<RwLock<HashMap<MinerUid, CachedMinerEndpoint>>>,
    cache_ttl: Duration,
}

impl MinerDiscovery {
    pub fn new(bittensor_service: Arc<BittensorService>, config: VerificationConfig) -> Self {
        Self {
            bittensor_service,
            config,
            endpoint_cache: Arc::new(RwLock::new(HashMap::new())),
            cache_ttl: Duration::from_secs(300), // 5 minutes default
        }
    }

    /// Create with custom cache TTL
    pub fn with_cache_ttl(mut self, ttl: Duration) -> Self {
        self.cache_ttl = ttl;
        self
    }

    /// Get list of miners that need verification using metagraph
    pub async fn get_miners_for_verification(&self) -> Result<Vec<MinerInfo>> {
        info!("Fetching ALL miners from metagraph for verification (no filtering)");

        let metagraph = self
            .bittensor_service
            .get_metagraph(self.config.netuid)
            .await?;

        if metagraph.hotkeys.is_empty() {
            info!(
                "No miners found in metagraph for netuid {}",
                self.config.netuid
            );
            return Ok(Vec::new());
        }

        let miners = self.extract_miners_from_metagraph(&metagraph)?;

        info!(
            "Selected ALL {} miners for verification from {} total neurons",
            miners.len(),
            metagraph.hotkeys.len()
        );

        Ok(miners)
    }

    fn extract_miners_from_metagraph(
        &self,
        metagraph: &bittensor::Metagraph<AccountId>,
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
        metagraph: &bittensor::Metagraph<AccountId>,
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
            if axon.port == 0 || axon.port > 65535 {
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

    /// Get miner axon info from metagraph by UID
    pub async fn get_miner_axon_info(&self, uid: MinerUid) -> Result<Option<AxonInfo>> {
        debug!("Fetching axon info for miner UID {}", uid.as_u16());

        // Check cache first
        {
            let cache = self.endpoint_cache.read().await;
            if let Some(cached) = cache.get(&uid) {
                let age = chrono::Utc::now()
                    .signed_duration_since(cached.cached_at)
                    .to_std()
                    .unwrap_or(Duration::MAX);
                if age < self.cache_ttl {
                    debug!("Using cached endpoint for miner {}", uid.as_u16());
                    // We still need to fetch fresh axon info, but we have cached endpoint
                }
            }
        }

        let metagraph = self
            .bittensor_service
            .get_metagraph(self.config.netuid)
            .await?;

        let axon_info = metagraph.axons.get(uid.as_u16() as usize).cloned();

        if let Some(ref axon) = axon_info {
            // Update cache with fresh axon endpoint
            let axon_endpoint = self.axon_info_to_endpoint(axon);
            let mut cache = self.endpoint_cache.write().await;
            cache.insert(
                uid,
                CachedMinerEndpoint {
                    axon_endpoint,
                    cached_at: chrono::Utc::now(),
                },
            );
        }

        Ok(axon_info)
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

    /// Get cached miner endpoints
    pub async fn get_cached_endpoints(&self) -> HashMap<MinerUid, String> {
        let cache = self.endpoint_cache.read().await;
        let now = chrono::Utc::now();

        cache
            .iter()
            .filter_map(|(uid, cached)| {
                let age = now
                    .signed_duration_since(cached.cached_at)
                    .to_std()
                    .unwrap_or(Duration::MAX);
                if age < self.cache_ttl {
                    Some((*uid, cached.axon_endpoint.clone()))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Clear the endpoint cache
    pub async fn clear_cache(&self) {
        let mut cache = self.endpoint_cache.write().await;
        cache.clear();
        info!("Cleared miner endpoint cache");
    }
}
