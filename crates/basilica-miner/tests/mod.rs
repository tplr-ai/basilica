mod unit;

#[cfg(test)]
mod tests {
    use alloy::signers::local::PrivateKeySigner;
    use alloy::signers::Signer;
    use alloy_primitives::{address, FixedBytes, U256};
    use alloy_provider::ProviderBuilder;
    use alloy_sol_types::SolEvent;
    use basilica_miner::config::MinerConfig;
    use collateral_contract::CollateralUpgradeable::{self, CollateralUpgradeableInstance};
    use rand::Rng;

    const TESTNET_URL: &str = "https://test.finney.opentensor.ai";

    /// 0xa8b2b82247e3f2b49ee8858b088405e35755c096 deployed in testnet
    /// minCollateralIncrease is 1, decisionTimeout is 1, trustee is 0xABCaD56aa87f3718C8892B48cB443c017Cd632BB
    ///
    /// 0x119ecacb1322cd9d581d550b52e199ec97a33e2e deployed in testnet
    /// decisionTimeout is 20 for the reclaim deny testing
    /// ~/.basilca/private_key is default for private key file path
    async fn get_contract(
    ) -> anyhow::Result<CollateralUpgradeableInstance<impl alloy_provider::Provider>> {
        let config = MinerConfig::load()?;
        let rpc_url = TESTNET_URL;
        let private_key = config.security.get_private_key()?;
        let proxy_contract_address = address!("0xa8b2b82247e3f2b49ee8858b088405e35755c096");

        let mut signer: PrivateKeySigner = private_key.parse().unwrap();
        signer.set_chain_id(Some(945));

        let provider = ProviderBuilder::new()
            .wallet(signer)
            .connect(rpc_url)
            .await?;

        let contract = CollateralUpgradeable::new(proxy_contract_address, provider);
        Ok(contract)
    }

    #[tokio::test]
    #[ignore]
    // cargo test --package basilica-miner --test mod -- tests::test_deposit --exact --nocapture
    async fn test_deposit() -> anyhow::Result<()> {
        let contract = get_contract().await?;
        println!("trustee: {:?}", contract.TRUSTEE().call().await.unwrap());
        println!("netuid: {:?}", contract.NETUID().call().await.unwrap());
        println!(
            "decision_timeout: {:?}",
            contract.DECISION_TIMEOUT().call().await.unwrap()
        );
        println!(
            "min_collateral_increase: {:?}",
            contract.MIN_COLLATERAL_INCREASE().call().await.unwrap()
        );

        let node_id: u128 = rand::thread_rng().gen_range(0..10000000000);
        let hotkey: [u8; 32] = [1u8; 32];
        let amount = U256::from(10);
        let alpha_hotkey: [u8; 32] = [2u8; 32];
        let deposit_tx = contract
            .deposit(
                FixedBytes::from_slice(&hotkey),
                FixedBytes::from_slice(&node_id.to_be_bytes()),
                FixedBytes::from_slice(&alpha_hotkey),
                amount,
            )
            .value(U256::ZERO);
        let deposit_tx_receipt = deposit_tx.send().await?.get_receipt().await?;

        let mut deposit_found = false;
        deposit_tx_receipt.logs().iter().for_each(|log| {
            if let Ok(event) = CollateralUpgradeable::Deposit::decode_log(&log.inner) {
                assert!(FixedBytes::from(node_id) == event.nodeId);
                deposit_found = true;
            }
        });
        assert!(deposit_found);

        let collaterals = contract
            .alphaCollaterals(
                FixedBytes::from_slice(&hotkey),
                FixedBytes::from_slice(&node_id.to_be_bytes()),
            )
            .call()
            .await
            .unwrap();

        assert_eq!(collaterals, amount);

        Ok(())
    }

    #[tokio::test]
    #[ignore]
    // cargo test --package basilica-miner --test mod -- tests::test_reclaim_finalize --exact --nocapture
    async fn test_reclaim_finalize() -> anyhow::Result<()> {
        let contract = get_contract().await?;

        // Deposit first
        let node_id: u128 = rand::thread_rng().gen_range(0..10000000000);
        let hotkey: [u8; 32] = [1u8; 32];
        let amount = U256::from(10);
        let alpha_hotkey: [u8; 32] = [2u8; 32];
        let alpha_coldkey: [u8; 32] = [3u8; 32];
        let deposit_tx = contract
            .deposit(
                FixedBytes::from_slice(&hotkey),
                FixedBytes::from_slice(&node_id.to_be_bytes()),
                FixedBytes::from_slice(&alpha_hotkey),
                amount,
            )
            .value(U256::ZERO);
        let _deposit_tx_receipt = deposit_tx.send().await?.get_receipt().await?;

        // Start reclaim process
        let url = "example.com";
        let url_checksum = 123_u128;
        let reclaim_tx = contract.reclaimCollateral(
            FixedBytes::from_slice(&hotkey),
            FixedBytes::from_slice(&node_id.to_be_bytes()),
            FixedBytes::from_slice(&alpha_coldkey),
            url.to_owned(),
            FixedBytes::from_slice(&url_checksum.to_be_bytes()),
        );
        let reclaim_receipt = reclaim_tx.send().await?.get_receipt().await?;

        let mut reclaim_id = U256::from(0);
        reclaim_receipt.logs().iter().for_each(|log| {
            if let Ok(event) = CollateralUpgradeable::ReclaimProcessStarted::decode_log(&log.inner)
            {
                reclaim_id = event.reclaimRequestId;
            }
        });

        tokio::time::sleep(std::time::Duration::from_secs(30)).await;

        // Test denyReclaimRequest (trustee only)
        let finalize_tx = contract.finalizeReclaim(reclaim_id);
        let finalize_receipt = finalize_tx.send().await?.get_receipt().await?;
        let mut finalize_found = false;
        // Check for Denied event
        finalize_receipt.logs().iter().for_each(|log| {
            if let Ok(event) = CollateralUpgradeable::Reclaimed::decode_log(&log.inner) {
                assert_eq!(event.reclaimRequestId, reclaim_id);
                finalize_found = true;
            }
        });
        assert!(finalize_found);
        Ok(())
    }

    #[tokio::test]
    #[ignore]
    // cargo test --package basilica-miner --test mod -- tests::test_reclaim_deny --exact --nocapture
    // to test deny, need set the decision timeout to a bigger number
    async fn test_reclaim_deny() -> anyhow::Result<()> {
        let contract = get_contract().await?;

        let node_id: u128 = rand::thread_rng().gen_range(0..10000000000);
        let hotkey: [u8; 32] = [1u8; 32];
        let amount = U256::from(10);
        let alpha_hotkey: [u8; 32] = [2u8; 32];
        let alpha_coldkey: [u8; 32] = [3u8; 32];
        let deposit_tx = contract
            .deposit(
                FixedBytes::from_slice(&hotkey),
                FixedBytes::from_slice(&node_id.to_be_bytes()),
                FixedBytes::from_slice(&alpha_hotkey),
                amount,
            )
            .value(U256::ZERO);
        let _deposit_tx_receipt = deposit_tx.send().await?.get_receipt().await?;

        // Start reclaim process
        let url = "example.com";
        let url_checksum = 123_u128;
        let reclaim_tx = contract.reclaimCollateral(
            FixedBytes::from_slice(&hotkey),
            FixedBytes::from_slice(&node_id.to_be_bytes()),
            FixedBytes::from_slice(&alpha_coldkey),
            url.to_owned(),
            FixedBytes::from_slice(&url_checksum.to_be_bytes()),
        );
        let reclaim_receipt = reclaim_tx.send().await?.get_receipt().await?;

        let mut reclaim_id = U256::from(0);
        reclaim_receipt.logs().iter().for_each(|log| {
            if let Ok(event) = CollateralUpgradeable::ReclaimProcessStarted::decode_log(&log.inner)
            {
                reclaim_id = event.reclaimRequestId;
            }
        });

        // Test denyReclaimRequest (trustee only)
        let deny_tx = contract.denyReclaimRequest(
            reclaim_id,
            url.to_owned(),
            FixedBytes::from_slice(&url_checksum.to_be_bytes()),
        );
        let deny_receipt = deny_tx.send().await?.get_receipt().await?;

        // Check for Denied event
        let mut denied_found = false;
        deny_receipt.logs().iter().for_each(|log| {
            if let Ok(event) = CollateralUpgradeable::Denied::decode_log(&log.inner) {
                assert_eq!(event.reclaimRequestId, reclaim_id);
                denied_found = true;
            }
        });
        assert!(denied_found);
        Ok(())
    }

    #[tokio::test]
    #[ignore]
    // cargo test --package basilica-miner --test mod -- tests::test_slash --exact --nocapture
    async fn test_slash() -> anyhow::Result<()> {
        let contract = get_contract().await?;

        let node_id: u128 = rand::thread_rng().gen_range(0..10000000000);
        let hotkey: [u8; 32] = [1u8; 32];
        let amount = U256::from(10);
        let alpha_hotkey: [u8; 32] = [2u8; 32];
        let deposit_tx = contract
            .deposit(
                FixedBytes::from_slice(&hotkey),
                FixedBytes::from_slice(&node_id.to_be_bytes()),
                FixedBytes::from_slice(&alpha_hotkey),
                amount,
            )
            .value(U256::ZERO);
        let _deposit_tx_receipt = deposit_tx.send().await?.get_receipt().await?;

        // Start reclaim process
        let url = "example.com";
        let url_checksum = 123_u128;
        let slash_tx = contract.slashCollateral(
            FixedBytes::from_slice(&hotkey),
            FixedBytes::from_slice(&node_id.to_be_bytes()),
            U256::ZERO,
            amount,
            url.to_owned(),
            FixedBytes::from_slice(&url_checksum.to_be_bytes()),
        );
        let slash_receipt = slash_tx.send().await?.get_receipt().await?;

        slash_receipt.logs().iter().for_each(|log| {
            if let Ok(event) = CollateralUpgradeable::Slashed::decode_log(&log.inner) {
                assert_eq!(event.nodeId, FixedBytes::from(node_id));
            }
        });
        Ok(())
    }
}
