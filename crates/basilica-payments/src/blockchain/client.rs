use crate::error::{PaymentsError, Result};
use anyhow::Context;
use bittensor::connect::{ConnectionPool, ConnectionPoolBuilder};
use sp_core::sr25519;
use std::sync::Arc;
use subxt::{dynamic::At, OnlineClient, PolkadotConfig};
use tracing::{debug, info};

pub struct BlockchainClient {
    pool: Arc<ConnectionPool>,
    endpoint: String,
}

impl BlockchainClient {
    pub async fn new(endpoint: &str) -> Result<Self> {
        info!("Initializing blockchain client connection to {}", endpoint);

        let pool = ConnectionPoolBuilder::new(vec![endpoint.to_string()])
            .max_connections(1)
            .build();

        pool.initialize()
            .await
            .context("Failed to initialize blockchain connection pool")
            .map_err(|e| PaymentsError::Blockchain(e.to_string()))?;

        info!("Successfully connected to blockchain at {}", endpoint);

        Ok(Self {
            pool: Arc::new(pool),
            endpoint: endpoint.to_string(),
        })
    }

    async fn get_client(&self) -> Result<Arc<OnlineClient<PolkadotConfig>>> {
        self.pool
            .get_healthy_client()
            .await
            .context("Failed to get healthy blockchain client")
            .map_err(|e| PaymentsError::Blockchain(e.to_string()))
    }

    pub async fn get_balance(&self, account_hex: &str) -> Result<u128> {
        let account_bytes = hex::decode(account_hex)
            .context("Invalid account hex")
            .map_err(|e| PaymentsError::Blockchain(e.to_string()))?;

        if account_bytes.len() != 32 {
            return Err(PaymentsError::Blockchain(format!(
                "Invalid account ID length: expected 32 bytes, got {}",
                account_bytes.len()
            )));
        }

        let mut account_id = [0u8; 32];
        account_id.copy_from_slice(&account_bytes);

        let account = subxt::utils::AccountId32(account_id);

        let storage_query = subxt::dynamic::storage(
            "System",
            "Account",
            vec![subxt::dynamic::Value::from_bytes(&account)],
        );

        let client = self.get_client().await?;

        let result = client
            .storage()
            .at_latest()
            .await
            .context("Failed to query storage")
            .map_err(|e| PaymentsError::Blockchain(e.to_string()))?
            .fetch(&storage_query)
            .await
            .context("Failed to fetch account data")
            .map_err(|e| PaymentsError::Blockchain(e.to_string()))?;

        let account_preview = account_hex.chars().take(8).collect::<String>();

        let Some(account_info) = result else {
            debug!(
                "Account {} not found on chain, returning 0 balance",
                account_preview
            );
            return Ok(0);
        };

        let value = account_info
            .to_value()
            .context("Failed to decode account info")
            .map_err(|e| PaymentsError::Blockchain(e.to_string()))?;

        // Extract balance from AccountInfo structure: { data: { free: u128, ... }, ... }
        if let Some(data_field) = value.at("data") {
            if let Some(free_balance) = data_field.at("free").and_then(|v| v.as_u128()) {
                debug!("Balance for {}: {} plancks", account_preview, free_balance);
                return Ok(free_balance);
            }
        }

        // Fallback: try string parsing if structure parsing fails
        let value_str = format!("{:?}", value);
        debug!(
            "Could not parse AccountInfo structure for {}, trying string fallback",
            account_preview
        );

        if let Some(start) = value_str.find("free: ") {
            let rest = &value_str[start + 6..];
            if let Some(end) = rest
                .find(',')
                .or_else(|| rest.find(' ').or_else(|| rest.find('}')))
            {
                let balance_str = &rest[..end];
                if let Ok(balance) = balance_str.trim().parse::<u128>() {
                    debug!(
                        "Balance for {} (via fallback): {} plancks",
                        account_preview, balance
                    );
                    return Ok(balance);
                }
            }
        }

        Err(PaymentsError::Blockchain(format!(
            "Failed to parse balance for account {}. AccountInfo structure could not be decoded. \
             This may indicate a chain metadata mismatch.",
            account_preview
        )))
    }

    pub async fn transfer(
        &self,
        keypair: &sr25519::Pair,
        to_address_ss58: &str,
        amount_plancks: u128,
    ) -> Result<TransferReceipt> {
        use sp_core::crypto::Ss58Codec;

        let dest_account = sp_core::sr25519::Public::from_ss58check(to_address_ss58)
            .map_err(|e| PaymentsError::Blockchain(format!("Invalid SS58 address: {}", e)))?;

        let dest = subxt::utils::AccountId32(dest_account.0);

        let amount_u64 = u64::try_from(amount_plancks).map_err(|_| {
            PaymentsError::Blockchain(format!("Amount {} exceeds u64::MAX", amount_plancks))
        })?;

        let dest_multi = subxt::dynamic::Value::unnamed_variant(
            "Id",
            vec![subxt::dynamic::Value::from_bytes(&dest)],
        );

        let transfer_tx = subxt::dynamic::tx(
            "Balances",
            "transfer_keep_alive",
            vec![dest_multi, subxt::dynamic::Value::u128(amount_u64 as u128)],
        );

        use sp_core::Pair as PairTrait;
        let raw = keypair.to_raw_vec();
        let secret_bytes: [u8; 32] = raw[..32].try_into().map_err(|_| {
            PaymentsError::Blockchain("Failed to extract keypair secret".to_string())
        })?;
        let signer = subxt_signer::sr25519::Keypair::from_secret_key(secret_bytes)
            .map_err(|e| PaymentsError::Blockchain(format!("Failed to create signer: {}", e)))?;

        let client = self.get_client().await?;

        let progress = client
            .tx()
            .sign_and_submit_then_watch_default(&transfer_tx, &signer)
            .await
            .context("Failed to submit transaction")
            .map_err(|e| PaymentsError::Blockchain(e.to_string()))?;

        let tx_hash = format!("0x{}", hex::encode(progress.extrinsic_hash()));
        info!("Transaction submitted: {}", tx_hash);

        // Wait for finalization first to get block info
        let tx_in_block = progress
            .wait_for_finalized()
            .await
            .context("Failed to wait for block finalization")
            .map_err(|e| PaymentsError::Blockchain(e.to_string()))?;

        let block_hash = format!("0x{}", hex::encode(tx_in_block.block_hash()));
        let block_number = client
            .blocks()
            .at(tx_in_block.block_hash())
            .await
            .map(|b| b.number() as i64)
            .ok();

        // Verify transaction success by waiting for events
        let _events = tx_in_block
            .wait_for_success()
            .await
            .context("Transaction failed during finalization")
            .map_err(|e| PaymentsError::Blockchain(e.to_string()))?;

        info!(
            "Transaction finalized: block_hash={}, block_number={:?}",
            block_hash, block_number
        );

        Ok(TransferReceipt {
            tx_hash,
            block_hash,
            block_number,
            status: TransferStatus::Finalized,
        })
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }
}

#[derive(Debug, Clone)]
pub struct TransferReceipt {
    pub tx_hash: String,
    pub block_hash: String,
    pub block_number: Option<i64>,
    pub status: TransferStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferStatus {
    InBlock,
    Finalized,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transfer_status() {
        let status = TransferStatus::InBlock;
        assert_eq!(status, TransferStatus::InBlock);
        assert_ne!(status, TransferStatus::Finalized);
    }

    #[test]
    fn test_invalid_account_hex() {
        assert!(hex::decode("invalid_hex").is_err());
    }

    #[test]
    fn test_account_hex_length() {
        let too_short = "abcd";
        let bytes = hex::decode(too_short).unwrap();
        assert_ne!(bytes.len(), 32);
    }
}
