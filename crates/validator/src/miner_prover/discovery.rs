//! # Miner Discovery
//!
//! Responsible for discovering and selecting miners from the Bittensor metagraph.
//! Implements Single Responsibility Principle by focusing only on miner discovery.

use super::types::MinerInfo;
use crate::config::VerificationConfig;
use anyhow::Result;
use bittensor::{AccountId, Service as BittensorService};
use common::identity::{Hotkey, MinerUid};
use std::sync::Arc;
use tracing::{debug, info, warn};

#[derive(Clone)]
pub struct MinerDiscovery {
    bittensor_service: Arc<BittensorService>,
    config: VerificationConfig,
}

impl MinerDiscovery {
    pub fn new(bittensor_service: Arc<BittensorService>, config: VerificationConfig) -> Self {
        Self {
            bittensor_service,
            config,
        }
    }

    /// Get list of miners that need verification using metagraph
    pub async fn get_miners_for_verification(&self) -> Result<Vec<MinerInfo>> {
        info!("Fetching miners from metagraph for verification");

        let metagraph = self
            .bittensor_service
            .get_metagraph(self.config.netuid)
            .await?;

        if metagraph.hotkeys.is_empty() {
            return self.get_placeholder_miners();
        }

        let mut miners = self.extract_miners_from_metagraph(&metagraph)?;
        self.prioritize_by_stake(&mut miners, &metagraph);
        miners.truncate(self.config.max_miners_per_round);

        info!(
            "Selected {} miners for verification from {} total neurons",
            miners.len(),
            metagraph.hotkeys.len()
        );

        Ok(miners)
    }

    fn get_placeholder_miners(&self) -> Result<Vec<MinerInfo>> {
        warn!("No neurons in metagraph - using placeholder data for testing");

        let placeholder_miners = vec![
            MinerInfo {
                uid: MinerUid::new(0),
                hotkey: Hotkey::new("5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string())
                    .unwrap_or_else(|_| Hotkey::new("placeholder".to_string()).unwrap()),
                endpoint: "http://placeholder-miner-0.example.com:8091".to_string(),
                is_validator: false,
                stake_tao: 10.0,
                last_verified: None,
                verification_score: 0.5,
            },
            MinerInfo {
                uid: MinerUid::new(1),
                hotkey: Hotkey::new("5Fbe6qGVfGPFGUCjn1AV7xfHVExrJNy1VR9CzjCPJoHPPuT4".to_string())
                    .unwrap_or_else(|_| Hotkey::new("placeholder".to_string()).unwrap()),
                endpoint: "http://placeholder-miner-1.example.com:8092".to_string(),
                is_validator: false,
                stake_tao: 5.0,
                last_verified: None,
                verification_score: 0.3,
            },
        ];

        info!(
            "Using {} placeholder miners for testing",
            placeholder_miners.len()
        );
        Ok(placeholder_miners)
    }

    fn extract_miners_from_metagraph(
        &self,
        metagraph: &bittensor::Metagraph<AccountId>,
    ) -> Result<Vec<MinerInfo>> {
        let mut miners = Vec::new();
        let now = chrono::Utc::now();

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

            if stake_tao < self.config.min_stake_threshold {
                debug!(
                    "Skipping miner {} due to low stake: {:.2} TAO",
                    uid, stake_tao
                );
                continue;
            }

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

            let min_interval =
                chrono::Duration::from_std(self.config.min_verification_interval).unwrap();
            if miner.needs_verification(min_interval, now) {
                miners.push(miner);
            }
        }

        Ok(miners)
    }

    fn extract_endpoint(
        &self,
        metagraph: &bittensor::Metagraph<AccountId>,
        uid: usize,
    ) -> Result<Option<String>> {
        Ok(metagraph
            .axons
            .get(uid)
            .map(|axon| format!("http://{}:{}", axon.ip, axon.port)))
    }

    fn prioritize_by_stake(
        &self,
        miners: &mut Vec<MinerInfo>,
        metagraph: &bittensor::Metagraph<AccountId>,
    ) {
        miners.sort_by(|a, b| {
            let stake_a = metagraph
                .total_stake
                .get(a.uid.as_u16() as usize)
                .map(|s| s.0)
                .unwrap_or(0);
            let stake_b = metagraph
                .total_stake
                .get(b.uid.as_u16() as usize)
                .map(|s| s.0)
                .unwrap_or(0);
            stake_b.cmp(&stake_a)
        });
    }
}
