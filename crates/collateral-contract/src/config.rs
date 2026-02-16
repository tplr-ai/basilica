#![allow(dead_code)]
use alloy_primitives::{address, Address};
use clap::ValueEnum;
use std::str::FromStr;
// Deployed Collateral contract address in product environment, will be updated after deployment
pub const COLLATERAL_ADDRESS: Address = address!("0x0000000000000000000000000000000000000000");
pub const PROXY_ADDRESS: Address = address!("0x0000000000000000000000000000000000000001");
pub const CHAIN_ID: u64 = 964;
pub const RPC_URL: &str = "https://lite.chain.opentensor.ai:443";

// Test environment
pub const TEST_CHAIN_ID: u64 = 945;
pub const TEST_RPC_URL: &str = "https://test.chain.opentensor.ai";

// Local network configuration
pub const LOCAL_CHAIN_ID: u64 = 42;
pub const LOCAL_RPC_URL: &str = "http://localhost:9944";
pub const LOCAL_WS_URL: &str = "ws://localhost:9944";

/// Maximum number of blocks to scan in a single iteration when scanning for collateral events
pub const MAX_BLOCKS_PER_SCAN: u64 = 1000;

/// Block number at which the collateral contract was deployed. Used as starting point for event scanning.
pub const CONTRACT_DEPLOYED_BLOCK_NUMBER: u64 = 0;

pub const DEFAULT_CONTRACT_ADDRESS: Address =
    address!("0x0000000000000000000000000000000000000002");

#[derive(Debug, Clone, ValueEnum, Default)]
pub enum Network {
    /// Mainnet (default)
    #[default]
    Mainnet,
    /// Testnet
    Testnet,
    /// Local development network
    Local,
}

#[derive(Debug, Clone)]
pub struct CollateralNetworkConfig {
    pub chain_id: u64,
    pub rpc_url: String,
    pub contract_address: Address,
}

impl Default for CollateralNetworkConfig {
    fn default() -> Self {
        Self::from_network(&Network::Mainnet, None)
            .expect("Failed to create default network config")
    }
}

impl CollateralNetworkConfig {
    pub fn from_network(
        network: &Network,
        contract_address: Option<String>,
    ) -> anyhow::Result<Self> {
        let parsed_addr: Option<Address> = match contract_address {
            Some(s) => Some(
                Address::from_str(&s)
                    .map_err(|_| anyhow::anyhow!(format!("Invalid contract address: {s}")))?,
            ),
            None => None,
        };
        match network {
            Network::Mainnet => Ok(CollateralNetworkConfig {
                chain_id: CHAIN_ID,
                rpc_url: RPC_URL.to_string(),
                contract_address: parsed_addr.unwrap_or(COLLATERAL_ADDRESS),
            }),
            Network::Testnet => Ok(CollateralNetworkConfig {
                chain_id: TEST_CHAIN_ID,
                rpc_url: TEST_RPC_URL.to_string(),
                contract_address: parsed_addr.unwrap_or(DEFAULT_CONTRACT_ADDRESS),
            }),
            Network::Local => Ok(CollateralNetworkConfig {
                chain_id: LOCAL_CHAIN_ID,
                rpc_url: LOCAL_RPC_URL.to_string(),
                contract_address: parsed_addr.unwrap_or(DEFAULT_CONTRACT_ADDRESS),
            }),
        }
    }
}
