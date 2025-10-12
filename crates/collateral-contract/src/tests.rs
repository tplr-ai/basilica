// The unit tests are for testing against local network
// Just can be executed if local subtensor node is running
use super::CollateralUpgradeable::CollateralUpgradeableInstance;
use super::*;
use alloy_primitives::Bytes;
use alloy_sol_types::{sol, SolCall};
use bittensor::api::api::{self as bittensorapi};
use subxt::utils::H160;

use sp_io::hashing::blake2_256;

use proxy::Proxy;
// use sp_core::crypto::Ss58Codec;
use std::{str::FromStr, time::Duration};
use subxt::{
    utils::{AccountId32, MultiAddress},
    OnlineClient, PolkadotConfig,
};
use subxt_signer::sr25519::dev;

use config::{LOCAL_CHAIN_ID, LOCAL_RPC_URL, LOCAL_WS_URL, TEST_CHAIN_ID, TEST_RPC_URL};

// function to initialize the contract
sol! {
    function initialize(uint16 netuid, address trustee, uint256 minCollateralIncrease, uint64 decisionTimeout, address admin, bytes32 alphaHotkey);
}

#[tokio::test]
// only test in local network, the testnet will reject such quick transactions
// to test against local network, must get the metadata for local network
// ./scripts/generate-metadata.sh local
// export BITTENSOR_NETWORK=local
// cargo test --package collateral-contract --lib -- tests::test_collateral_deploy_local --exact --show-output
// export OPEN_EVM_PRIVATE_KEY=5fb92d6e98884f76de468fa3f6278f8807c48bebc13595d45af5bdc4da702133
// LOCAL_CHAIN_ID=31337
// #[ignore]
async fn test_collateral_deploy_local() {
    // parameters for miner to deposit collateral
    let hotkey = FixedBytes::from_slice(&[1u8; 32]);
    let node_id = FixedBytes::from_slice(&1u128.to_be_bytes());
    let miner_alpha_hotkey = [3u8; 32];

    // as contract hotkey
    let alpha_hotkey = [2u8; 32];

    let client = OnlineClient::<PolkadotConfig>::from_url(LOCAL_WS_URL)
        .await
        .unwrap();

    // set up local subtensor network
    disable_whitelist(&client).await.unwrap();
    let netuid = create_subnet(&client, &miner_alpha_hotkey).await.unwrap();
    add_stake(&client, netuid, &miner_alpha_hotkey)
        .await
        .unwrap();

    // get predefined evm account alithe signer, the account is miner's ethereum wallet
    let alithe_private_key = std::env::var("OPEN_EVM_PRIVATE_KEY").unwrap_or_else(|_| {
        "5fb92d6e98884f76de468fa3f6278f8807c48bebc13595d45af5bdc4da702133".to_string()
    });

    let mut signer: PrivateKeySigner = alithe_private_key.parse().unwrap();
    let signer_public_key = convert_h160_to_public_key(signer.address().as_slice()).unwrap();
    println!("signer: {:?}", signer.address());

    signer.set_chain_id(Some(LOCAL_CHAIN_ID));

    let provider = ProviderBuilder::new()
        .wallet(signer.clone())
        .connect(LOCAL_RPC_URL)
        .await
        .unwrap();

    println!(
        "signer balance: {:?}",
        provider.get_balance(signer.address()).await.unwrap()
    );

    // deploy contract, get the proxy contract address and contract public key
    let (proxied, contract_public_key) = deploy_contract(netuid, &signer, &provider, alpha_hotkey)
        .await
        .unwrap();
    // force set balance to the contract to avoid insufficient balance error for extrinsic call
    force_set_balance(&client, &AccountId32::from(contract_public_key))
        .await
        .unwrap();
    // burn register neuron to set the contract hotkey and coldkey relations
    burn_register(&client, &proxied).await.unwrap();

    // transfer stake from alith to miner
    transfer_stake(&client, netuid, &miner_alpha_hotkey, &signer_public_key)
        .await
        .unwrap();

    // Test deposit
    let amount = U256::from(2_000_000_000_000_000_000u128); // 2 TAO
    let alpha_amount = U256::from(1_000_000_000u128);

    let tao_before_deposit = proxied.collaterals(hotkey, node_id).call().await.unwrap();
    let alpha_before_deposit = proxied
        .alphaCollaterals(hotkey, node_id)
        .call()
        .await
        .unwrap();

    println!("tao before deposit: {:?}", tao_before_deposit);
    println!("alpha before deposit: {:?}", alpha_before_deposit);

    let tx = proxied
        .deposit(
            hotkey,
            node_id,
            FixedBytes::from_slice(&miner_alpha_hotkey),
            alpha_amount,
        )
        .value(amount);
    let tx = tx.send().await.unwrap();
    let _ = tx.get_receipt().await.unwrap();

    let tao_after_deposit = proxied.collaterals(hotkey, node_id).call().await.unwrap();
    let alpha_after_deposit = proxied
        .alphaCollaterals(hotkey, node_id)
        .call()
        .await
        .unwrap();

    println!("tao after deposit: {:?}", tao_after_deposit);
    println!("alpha after deposit: {:?}", alpha_after_deposit);

    let miner = proxied.nodeToMiner(hotkey, node_id).call().await.unwrap();
    println!("miner: {:?}", miner);

    let tx = proxied.reclaimCollateral(
        hotkey,
        node_id,
        FixedBytes::from_slice(&signer_public_key),
        "https://www.tplr.ai/".to_string(),
        FixedBytes::from_slice(&[
            26u8, 9u8, 245u8, 1u8, 14u8, 0u8, 17u8, 59u8, 89u8, 65u8, 234u8, 75u8, 0u8, 12u8, 12u8,
            13u8,
        ]),
    );
    let pending_tx = tx.send().await.unwrap();
    let receipt = pending_tx.get_receipt().await.unwrap();

    let mut reclaim_id = U256::from(0);

    for log in receipt.inner.logs() {
        if let Ok(event) = CollateralUpgradeable::ReclaimProcessStarted::decode_log(&log.inner) {
            reclaim_id = event.reclaimRequestId;
            println!(
                "======================== ReclaimProcessStarted ================================"
            );
            println!("reclaim_id: {:?}", reclaim_id);
            println!("hotkey: {:?}", event.hotkey);
            println!("nodeId: {:?}", event.nodeId);
            println!("miner: {:?}", event.miner);
            println!("amount: {:?}", event.amount);
            println!("alphaColdkey: {:?}", event.alphaColdkey);
            println!("alphaAmount: {:?}", event.alphaAmount);
            println!("========================================================");
        }
    }

    println!("reclaim_id: {:?}", reclaim_id);

    let reclaim = proxied.reclaims(reclaim_id).call().await.unwrap();
    println!("reclaim: {:?}", reclaim.denyTimeout);
    assert_eq!(
        reclaim.alphaColdkey,
        FixedBytes::from_slice(&signer_public_key)
    );

    // sleep for 3 seconds to wait for the reclaim to be finalized
    tokio::time::sleep(Duration::from_secs(3)).await;

    let tx = proxied.finalizeReclaim(reclaim_id);
    let pending_tx = tx.send().await.unwrap();
    let receipt = pending_tx.get_receipt().await.unwrap();

    for log in receipt.inner.logs() {
        if let Ok(event) = CollateralUpgradeable::Reclaimed::decode_log(&log.inner) {
            assert_eq!(event.reclaimRequestId, reclaim_id);
            // println!("reclaimed: {:?}", event.reclaimRequestId);
            // println!("hotkey: {:?}", event.hotkey);
            // println!("nodeId: {:?}", event.nodeId);
            // println!("miner: {:?}", event.miner);
            println!("amount: {:?}", event.amount);
            // println!("alphaColdkey: {:?}", event.alphaColdkey);
            println!("alphaAmount: {:?}", event.alphaAmount);
            // println!("denyTimeout: {:?}", event.denyTimeout);
        }
    }

    let tao_after_reclaim = proxied.collaterals(hotkey, node_id).call().await.unwrap();
    let alpha_after_reclaim = proxied
        .alphaCollaterals(hotkey, node_id)
        .call()
        .await
        .unwrap();

    assert_eq!(tao_after_reclaim, 0);
    assert_eq!(alpha_after_reclaim, 0);
}

#[tokio::test]
#[ignore]
async fn test_deploy_upgradable_collateral() {
    let alithe_private_key = std::env::var("OPEN_EVM_PRIVATE_KEY").unwrap_or_else(|_| {
        "5fb92d6e98884f76de468fa3f6278f8807c48bebc13595d45af5bdc4da702133".to_string()
    });

    let mut signer: PrivateKeySigner = alithe_private_key.trim().parse().unwrap();
    signer.set_chain_id(Some(TEST_CHAIN_ID));

    let provider = ProviderBuilder::new()
        .wallet(signer)
        .connect(TEST_RPC_URL)
        .await
        .unwrap();

    let contract = CollateralUpgradeable::deploy(provider.clone())
        .await
        .unwrap();

    // Convert EVM address to Substrate SS58 address with "evm:" prefix
    let contract_address = contract.address();
    let mut account_bytes = [0u8; 32];
    account_bytes[0..4].copy_from_slice(b"evm:");
    account_bytes[4..24].copy_from_slice(contract_address.as_slice());
    // Last 8 bytes remain zeros
    let account_id = AccountId32::from(account_bytes);
    let contract_ss58 = account_id.to_string();

    println!("Deployed contract at: {:?}", contract.address());
    println!("Contract SS58 address: {}", contract_ss58);
}

/// Convert EVM address to Substrate AccountId32 by hashing "evm:" prefix + address
pub fn convert_h160_to_public_key(eth_address: &[u8]) -> Result<[u8; 32], anyhow::Error> {
    if eth_address.len() != 20 {
        return Err(anyhow::Error::msg(
            "Invalid address length, expected 20 bytes",
        ));
    }

    let mut data = [0u8; 24];
    data[0..4].copy_from_slice(b"evm:");
    data[4..24].copy_from_slice(&eth_address[..]);
    let result = blake2_256(&data);

    Ok(result)
}

#[test]
fn test_convert_h160_to_public_key() {
    let eth_address = H160::from_str("0x0000000000000000000000000000000000000001").unwrap();
    let public_key = convert_h160_to_public_key(eth_address.0.as_slice()).unwrap();
    let hex_public_key = hex::encode(public_key);
    assert_eq!(
        hex_public_key,
        "c16e18ed0b2d27d6fd0dc832a699af7d3adf762cd6d51d780abc460476ad1c58"
    );
}

#[allow(dead_code)]
async fn force_set_balance(
    client: &OnlineClient<PolkadotConfig>,
    account_id: &AccountId32,
) -> Result<(), anyhow::Error> {
    // Connect to local node
    let signer = dev::alice();
    let amount = 1_000_000_000_000u64; // 1000 TAO

    let inner_call =
        bittensorapi::runtime_types::pallet_balances::pallet::Call::force_set_balance {
            who: MultiAddress::Id(account_id.clone()),
            new_free: amount,
        };

    let runtime_call =
        bittensorapi::runtime_types::node_subtensor_runtime::RuntimeCall::Balances(inner_call);

    let call = bittensorapi::tx().sudo().sudo(runtime_call);

    client
        .tx()
        .sign_and_submit_then_watch_default(&call, &signer)
        .await?;

    tokio::time::sleep(Duration::from_secs(3)).await;

    let balance_query = bittensorapi::storage()
        .system()
        .account(AccountId32::from(account_id.clone()));

    let balance = client
        .storage()
        .at_latest()
        .await?
        .fetch(&balance_query)
        .await?
        .unwrap();

    assert_eq!(balance.data.free, amount);

    Ok(())
}

#[allow(dead_code)]
async fn disable_whitelist(client: &OnlineClient<PolkadotConfig>) -> Result<(), anyhow::Error> {
    let storage_query = bittensorapi::storage().evm().disable_whitelist_check();

    let result = client
        .storage()
        .at_latest()
        .await?
        .fetch(&storage_query)
        .await?;

    // already whitelisted
    if let Some(true) = result {
        return Ok(());
    }

    // Create signer from Alice's dev account
    let signer = dev::alice();

    let inner_call =
        bittensorapi::runtime_types::pallet_evm::pallet::Call::disable_whitelist { disabled: true };

    let runtime_call =
        bittensorapi::runtime_types::node_subtensor_runtime::RuntimeCall::EVM(inner_call);

    let call = bittensorapi::tx().sudo().sudo(runtime_call);

    client
        .tx()
        .sign_and_submit_then_watch_default(&call, &signer)
        .await?
        .wait_for_finalized_success()
        .await?;

    let storage_query = bittensorapi::storage().evm().disable_whitelist_check();

    let result = client
        .storage()
        .at_latest()
        .await?
        .fetch(&storage_query)
        .await?;

    assert_eq!(result, Some(true));

    Ok(())
}

// register a new subnet with alice and bob
#[allow(dead_code)]
async fn create_subnet(
    client: &OnlineClient<PolkadotConfig>,
    hotkey: &[u8; 32],
) -> Result<u16, anyhow::Error> {
    // Create signer from Alice's dev account
    let signer = dev::alice();

    let total_networks_query = bittensorapi::storage().subtensor_module().total_networks();
    let total_networks = client
        .storage()
        .at_latest()
        .await?
        .fetch(&total_networks_query)
        .await?
        .unwrap();

    let call = bittensorapi::tx()
        .subtensor_module()
        .register_network(AccountId32::from(hotkey.clone()));
    client
        .tx()
        .sign_and_submit_then_watch_default(&call, &signer)
        .await?
        .wait_for_finalized_success()
        .await?;

    let new_total_networks = client
        .storage()
        .at_latest()
        .await?
        .fetch(&total_networks_query)
        .await?
        .unwrap();

    assert_eq!(new_total_networks, total_networks + 1);

    // wait for 10 blocks for fast blocks
    tokio::time::sleep(Duration::from_secs(6)).await;

    // start the subnet then we can stake to it
    let call = bittensorapi::tx()
        .subtensor_module()
        .start_call(total_networks);

    client
        .tx()
        .sign_and_submit_then_watch_default(&call, &signer)
        .await?
        .wait_for_finalized_success()
        .await?;

    let started_query = bittensorapi::storage()
        .subtensor_module()
        .subtoken_enabled(total_networks);

    let started = client
        .storage()
        .at_latest()
        .await?
        .fetch(&started_query)
        .await?
        .unwrap();
    // confirm the subnet is started
    assert_eq!(started, true);

    Ok(total_networks)
}

// add stake to the subnet with alice
#[allow(dead_code)]
async fn add_stake(
    client: &OnlineClient<PolkadotConfig>,
    netuid: u16,
    hotkey: &[u8; 32],
) -> Result<(), anyhow::Error> {
    // Create signer from Alice's dev account
    let signer = dev::alice();
    let amount = 1_000_000_000_000u64;

    let call = bittensorapi::tx().subtensor_module().add_stake(
        AccountId32::from(hotkey.clone()),
        netuid,
        amount,
    );

    client
        .tx()
        .sign_and_submit_then_watch_default(&call, &signer)
        .await?
        .wait_for_finalized_success()
        .await?;

    let stake = bittensorapi::storage().subtensor_module().alpha(
        AccountId32::from(hotkey.clone()),
        AccountId32::from(signer.public_key()),
        netuid,
    );

    let stake = client
        .storage()
        .at_latest()
        .await?
        .fetch(&stake)
        .await?
        .unwrap();
    println!("stake after add stake call: {:?}", stake.bits);

    assert_eq!(stake.bits > 0, true);

    Ok(())
}

#[allow(dead_code)]
async fn transfer_stake(
    client: &OnlineClient<PolkadotConfig>,
    netuid: u16,
    hotkey: &[u8; 32],
    coldkey: &[u8; 32],
) -> Result<(), anyhow::Error> {
    let signer = dev::alice();
    let amount = 1_000_000_000_000u64;

    let call = bittensorapi::tx().subtensor_module().transfer_stake(
        AccountId32::from(coldkey.clone()),
        AccountId32::from(hotkey.clone()),
        netuid,
        netuid,
        amount,
    );

    client
        .tx()
        .sign_and_submit_then_watch_default(&call, &signer)
        .await?
        .wait_for_finalized_success()
        .await?;

    let stake = bittensorapi::storage().subtensor_module().alpha(
        AccountId32::from(hotkey.clone()),
        AccountId32::from(coldkey.clone()),
        netuid,
    );

    let stake = client
        .storage()
        .at_latest()
        .await?
        .fetch(&stake)
        .await?
        .unwrap();
    println!("stake after transfer: {:?}", stake.bits);
    assert_eq!(stake.bits > 0, true);
    Ok(())
}

#[allow(dead_code)]
async fn burn_register(
    client: &OnlineClient<PolkadotConfig>,
    contract: &CollateralUpgradeableInstance<impl Provider>,
) -> Result<(), anyhow::Error> {
    let tx = contract.burnRegister();
    let tx = tx.send().await.unwrap();
    let _ = tx.get_receipt().await.unwrap();

    let hotkey = contract.CONTRACT_HOTKEY().call().await.unwrap();
    let coldkey = contract.CONTRACT_COLDKEY().call().await.unwrap();
    println!("hotkey: {:?}", hotkey);
    println!("coldkey: {:?}", coldkey);

    // wait for the evm transaction to be included in the block
    tokio::time::sleep(Duration::from_secs(3)).await;

    let storage_query = bittensorapi::storage()
        .subtensor_module()
        .owner(AccountId32::from(hotkey.0));

    let result = client
        .storage()
        .at_latest()
        .await?
        .fetch(&storage_query)
        .await?
        .unwrap();

    assert_eq!(result, AccountId32::from(coldkey.0));
    println!("burn register success");

    Ok(())
}

#[allow(dead_code)]
async fn deploy_contract<P: Provider + Clone>(
    netuid: u16,
    signer: &PrivateKeySigner,
    provider: &P,
    alpha_hotkey: [u8; 32],
) -> Result<(CollateralUpgradeableInstance<P>, [u8; 32]), anyhow::Error> {
    let trustee = signer.address(); // for slash collateral role
    let min_collateral_increase = U256::from(1_000_000u128);
    let decision_timeout = 1u64; // 1 hour
    let admin = signer.address(); // for upgrader role

    let contract = CollateralUpgradeable::deploy(provider.clone()).await?;

    // Convert EVM address to Substrate account ID
    let contract_address = contract.address();
    let contract_public_key = convert_h160_to_public_key(contract_address.as_slice())?;

    println!("Deployed contract at: {}", contract.address());
    println!(
        "Contract public key: 0x{}",
        hex::encode(contract_public_key)
    );

    let data: Bytes = Bytes::from(
        initializeCall {
            netuid,
            trustee,
            minCollateralIncrease: min_collateral_increase,
            decisionTimeout: decision_timeout,
            admin,
            alphaHotkey: FixedBytes::from_slice(&alpha_hotkey),
        }
        .abi_encode(),
    );

    let proxy = Proxy::deploy(provider.clone(), *contract.address(), data).await?;
    let proxy_public_key = convert_h160_to_public_key(proxy.address().as_slice())?; // used as contract coldkey

    println!("Deployed proxy at: {}", proxy.address());
    println!("Proxy public key: 0x{}", hex::encode(proxy_public_key));

    let proxied = CollateralUpgradeable::new(*proxy.address(), provider.clone());
    println!("Proxied contract at: {}", proxied.address());

    let tx = proxied.setContractColdkey(FixedBytes::from_slice(&proxy_public_key));
    let tx = tx.send().await.unwrap();
    let _ = tx.get_receipt().await.unwrap();

    // Test get methods
    assert_eq!(
        proxied.CONTRACT_COLDKEY().call().await.unwrap(),
        FixedBytes::from_slice(&proxy_public_key)
    );

    let netuid_result = proxied.NETUID().call().await?;
    assert_eq!(netuid_result, netuid);

    let trustee_result = proxied.TRUSTEE().call().await?;
    assert_eq!(trustee_result, trustee);

    let min_collateral_increase_result = proxied.MIN_COLLATERAL_INCREASE().call().await?;
    assert_eq!(min_collateral_increase_result, min_collateral_increase);

    let decision_timeout_result = proxied.DECISION_TIMEOUT().call().await?;
    assert_eq!(decision_timeout_result, decision_timeout);

    Ok((proxied, proxy_public_key))
}
