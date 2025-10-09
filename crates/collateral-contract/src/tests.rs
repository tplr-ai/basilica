// The unit tests are for testing against local network
// Just can be executed if local subtensor node is running
use super::CollateralUpgradeable::CollateralUpgradeableInstance;
use super::*;
use alloy::hex::FromHex;
use alloy_primitives::Bytes;
use alloy_sol_types::{sol, SolCall};
use bittensor::api::api::{self as bittensorapi};
use blake2::{Blake2b512, Digest};
use proxy::Proxy;
// use sp_core::crypto::Ss58Codec;
use std::time::Duration;
use subxt::{utils::AccountId32, OnlineClient, PolkadotConfig};
use subxt_signer::sr25519::dev;

use config::{LOCAL_CHAIN_ID, LOCAL_RPC_URL, LOCAL_WS_URL, TEST_CHAIN_ID, TEST_RPC_URL};

// function to initialize the contract
sol! {
    function initialize(uint16 netuid, address trustee, uint256 minCollateralIncrease, uint64 decisionTimeout, address admin, bytes32 alphaHotkey);
}

/// Convert EVM address to Substrate AccountId32 by hashing "evm:" prefix + address
pub fn convert_h160_to_public_key(eth_address: &[u8]) -> Result<[u8; 32], anyhow::Error> {
    if eth_address.len() != 20 {
        return Err(anyhow::Error::msg(
            "Invalid address length, expected 20 bytes",
        ));
    }

    let prefix = b"evm:";

    // Concatenate prefix and Ethereum address
    let mut hasher = Blake2b512::new();
    hasher.update(prefix);
    hasher.update(eth_address);

    // Get first 32 bytes of Blake2b-512 hash
    let result = hasher.finalize();
    let mut output = [0u8; 32];
    output.copy_from_slice(&result[..32]);

    Ok(output)
}

#[allow(dead_code)]
async fn disable_whitelist() -> Result<(), anyhow::Error> {
    // Connect to local node
    let client = OnlineClient::<PolkadotConfig>::from_url(LOCAL_WS_URL).await?;

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
        .await?;

    tokio::time::sleep(Duration::from_secs(3)).await;

    let storage_query = bittensorapi::storage().evm().disable_whitelist_check();

    let result = client
        .storage()
        .at_latest()
        .await?
        .fetch(&storage_query)
        .await?;

    println!("Value: {result:?}");
    assert_eq!(result, Some(true));

    Ok(())
}

// register a new subnet with alice and bob
#[allow(dead_code)]
async fn create_subnet() -> Result<u16, anyhow::Error> {
    // Connect to local node
    let client = OnlineClient::<PolkadotConfig>::from_url(LOCAL_WS_URL).await?;

    // Create signer from Alice's dev account
    let signer = dev::alice();

    let hotkey = dev::bob().public_key();

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
        .register_network(AccountId32::from(hotkey));

    client
        .tx()
        .sign_and_submit_then_watch_default(&call, &signer)
        .await?;

    tokio::time::sleep(Duration::from_secs(3)).await;

    let new_total_networks = client
        .storage()
        .at_latest()
        .await?
        .fetch(&total_networks_query)
        .await?
        .unwrap();

    assert_eq!(new_total_networks, total_networks + 1);

    Ok(total_networks)
}

// add stake to the subnet with alice and bob
#[allow(dead_code)]
async fn add_stake(netuid: u16) -> Result<(), anyhow::Error> {
    // Connect to local node
    let client = OnlineClient::<PolkadotConfig>::from_url(LOCAL_WS_URL).await?;

    // Create signer from Alice's dev account
    let signer = dev::alice();
    let hotkey = dev::bob().public_key();
    let amount = 1_000_000_000_000u64;

    let call =
        bittensorapi::tx()
            .subtensor_module()
            .add_stake(AccountId32::from(hotkey), netuid, amount);

    client
        .tx()
        .sign_and_submit_then_watch_default(&call, &signer)
        .await?;

    Ok(())
}

// #[allow(dead_code)]
// async fn transfer_stake_to_contract(
//     netuid: u16,
//     cold_key: [u8; 32],
//     hotkey: [u8; 32],
//     amount: u64,
// ) -> Result<(), anyhow::Error> {
//     // Connect to local node
//     let client = OnlineClient::<PolkadotConfig>::from_url(LOCAL_WS_URL).await?;

//     // Create signer from Alice's dev account
//     let signer = dev::alice();

//     let call = bittensorapi::tx().subtensor_module().transfer_stake(
//         AccountId32::from(cold_key),
//         AccountId32::from(hotkey),
//         netuid,
//         netuid,
//         amount,
//     );

//     client
//         .tx()
//         .sign_and_submit_then_watch_default(&call, &signer)
//         .await?;

//     Ok(())
// }

#[allow(dead_code)]
async fn deploy_contract<P: Provider + Clone>(
    netuid: u16,
    signer: &PrivateKeySigner,
    provider: &P,
) -> Result<(CollateralUpgradeableInstance<P>, [u8; 32]), anyhow::Error> {
    let trustee = signer.address();
    let min_collateral_increase = U256::from(1_000_000_000u128);
    let decision_timeout = 3600u64; // 1 hour
    let admin = signer.address();
    let alpha_hotkey = [2u8; 32];

    let contract = CollateralUpgradeable::deploy(provider.clone()).await?;

    // Convert EVM address to Substrate account ID
    let contract_address = contract.address();
    let contract_public_key = convert_h160_to_public_key(contract_address.as_slice())?;

    println!("Deployed contract at: {}", contract.address());
    println!(
        "Contract public key: 0x{}",
        hex::encode(contract_public_key)
    );
    tokio::time::sleep(Duration::from_secs(3)).await;

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

    println!("Deployed proxy at: {}", proxy.address());

    let proxied = CollateralUpgradeable::new(*proxy.address(), provider.clone());

    // Test get methods
    let netuid_result = proxied.NETUID().call().await?;
    assert_eq!(netuid_result, netuid);

    let trustee_result = proxied.TRUSTEE().call().await?;
    assert_eq!(trustee_result, trustee);

    let min_collateral_increase_result = proxied.MIN_COLLATERAL_INCREASE().call().await?;
    assert_eq!(min_collateral_increase_result, min_collateral_increase);

    let decision_timeout_result = proxied.DECISION_TIMEOUT().call().await?;
    assert_eq!(decision_timeout_result, decision_timeout);

    Ok((proxied, contract_public_key))
}

#[tokio::test]
// only test in local network, the testnet will reject such quick transactions
// to test against local network, must get the metadata for local network
// ./scripts/generate-metadata.sh local
// export BITTENSOR_NETWORK=local
// cargo test --package collateral-contract --lib -- tests::test_collateral_deploy_local --exact --show-output
// export OPEN_EVM_PRIVATE_KEY=5fb92d6e98884f76de468fa3f6278f8807c48bebc13595d45af5bdc4da702133
// LOCAL_RPC_URL=127.0.0.1:8545
// LOCAL_CHAIN_ID=31337
// #[ignore]
async fn test_collateral_deploy_local() {
    // set up local subtensor network
    disable_whitelist().await.unwrap();
    let netuid = create_subnet().await.unwrap();
    add_stake(netuid).await.unwrap();

    // get predefined evm account alithe signer
    // let alithe_private_key = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80";
    let alithe_private_key = std::env::var("OPEN_EVM_PRIVATE_KEY").unwrap_or_else(|_| {
        "5fb92d6e98884f76de468fa3f6278f8807c48bebc13595d45af5bdc4da702133".to_string()
    });

    let mut signer: PrivateKeySigner = alithe_private_key.parse().unwrap();
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

    let (proxied, contract_public_key) = deploy_contract(netuid, &signer, &provider).await.unwrap();

    // Test deposit
    let hotkey = FixedBytes::from_slice(&[1u8; 32]);
    let executor_id = FixedBytes::from_slice(&1u128.to_be_bytes());
    let amount = U256::from(2_000_000_000_000_000_000u128); // 2 TAO
    let alpha_hotkey = FixedBytes::from_slice(&[2u8; 32]);
    let alpha_amount = U256::from(1_000_000_000u128);

    // set coldkey as the contract mapped substrate address
    let tx = proxied.setContractColdkey(FixedBytes::from_slice(&contract_public_key));
    let tx = tx.send().await.unwrap();
    let receipt = tx.get_receipt().await.unwrap();
    println!("Set contract coldkey receipt: {:?}", receipt);
    assert_eq!(
        proxied.CONTRACT_COLDKEY().call().await.unwrap(),
        FixedBytes::from_slice(&contract_public_key)
    );

    let tao_before_deposit = proxied
        .collaterals(hotkey, executor_id)
        .call()
        .await
        .unwrap();
    let alpha_before_deposit = proxied
        .alphaCollaterals(
            FixedBytes::from_slice(&hotkey),
            FixedBytes::from_slice(&executor_id),
        )
        .call()
        .await
        .unwrap();

    println!("tao before deposit: {:?}", tao_before_deposit);
    println!("alpha before deposit: {:?}", alpha_before_deposit);

    let tx = proxied
        .deposit(hotkey, executor_id, alpha_hotkey, alpha_amount)
        .value(amount);
    let tx = tx.send().await.unwrap();
    let receipt = tx.get_receipt().await.unwrap();
    println!("Deposit receipt: {:?}", receipt);

    let executor_to_miner_result = proxied
        .executorToMiner(hotkey, executor_id)
        .call()
        .await
        .unwrap();
    assert_eq!(executor_to_miner_result, signer.address());

    let collaterals_result = proxied
        .collaterals(hotkey, executor_id)
        .call()
        .await
        .unwrap();
    assert_eq!(collaterals_result, amount);
}

#[tokio::test]
#[ignore]
async fn test_deploy_upgradable_collateral_in_testnet() {
    let private_key = std::env::var("OPEN_EVM_PRIVATE_KEY").unwrap();

    let mut signer: PrivateKeySigner = private_key.trim().parse().unwrap();
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

#[tokio::test]
#[ignore]
async fn test_deploy_proxy_in_testnet() {
    let contract_address = Address::from_hex("0x4894035ccc55143c791ef85e31bc225b7918eb68").unwrap();
    let private_key = std::env::var("OPEN_EVM_PRIVATE_KEY").unwrap();

    let mut signer: PrivateKeySigner = private_key.trim().parse().unwrap();
    signer.set_chain_id(Some(TEST_CHAIN_ID));

    let provider = ProviderBuilder::new()
        .wallet(signer.clone())
        .connect(TEST_RPC_URL)
        .await
        .unwrap();

    let netuid = 1;
    let trustee = signer.address();
    let min_collateral_increase = U256::from(1);
    let decision_timeout = 1; // 1 hour
    let admin = signer.address();
    let alpha_hotkey = [2u8; 32];

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

    let contract = Proxy::deploy(provider.clone(), contract_address, data)
        .await
        .unwrap();

    println!("Deployed proxy at: {:?}", contract.address());
}
