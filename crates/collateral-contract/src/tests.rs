// The unit tests are for testing against local network
// Just can be executed if local subtensor node is running
use super::*;
use crate::config::LOCAL_CHAIN_ID;
use alloy::hex::FromHex;
use alloy_primitives::Bytes;
use alloy_sol_types::{sol, SolCall};
use bittensor::api::api::{self as bittensorapi};
use proxy::Proxy;
use subxt::{OnlineClient, PolkadotConfig};
use subxt_signer::sr25519::dev;

use config::{LOCAL_RPC_URL, LOCAL_WS_URL, TEST_CHAIN_ID, TEST_RPC_URL};

// function to initialize the contract
sol! {
    function initialize(uint16 netuid, address trustee, uint256 minCollateralIncrease, uint64 decisionTimeout, address admin);
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

#[tokio::test]
// only test in local network, the testnet will reject such quick transactions
// to test against local network, must get the metadata for local network
// ./scripts/generate-metadata.sh local
// export BITTENSOR_NETWORK=local
// cargo test --package collateral --lib -- test::test_collateral_deploy --exact --show-output --ignored
#[ignore]
async fn test_collateral_deploy() {
    disable_whitelist().await.unwrap();

    // get predefined evm account alithe signer
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

    let netuid = 1;
    let trustee = signer.address();
    let min_collateral_increase = U256::from(1_000_000_000_000_000_000u128); // 1 TAO
    let decision_timeout = 3600u64; // 1 hour
    let admin = signer.address();

    let contract = CollateralUpgradeable::deploy(provider.clone())
        .await
        .unwrap();

    println!("Deployed contract at: {:?}", contract.address());

    let data: Bytes = Bytes::from(
        initializeCall {
            netuid,
            trustee,
            minCollateralIncrease: min_collateral_increase,
            decisionTimeout: decision_timeout,
            admin,
        }
        .abi_encode(),
    );

    let proxy = Proxy::deploy(provider.clone(), *contract.address(), data)
        .await
        .unwrap();

    // Test deposit
    let hotkey = [1u8; 32];
    let node_id = 1u128;
    let alpha_hotkey = [2u8; 32];
    let alpha_amount = U256::from(2_000_000_000_000_000_000u128); // 2 Alpha

    // Call through proxy address
    let proxied = CollateralUpgradeable::new(*proxy.address(), provider.clone());

    let tx = proxied
        .deposit(
            FixedBytes::from_slice(&hotkey),
            FixedBytes::from_slice(&node_id.to_be_bytes()),
            FixedBytes::from_slice(&alpha_hotkey),
            alpha_amount,
        )
        .value(U256::ZERO);
    let tx = tx.send().await.unwrap();
    let receipt = tx.get_receipt().await.unwrap();
    println!("Deposit receipt: {:?}", receipt);

    // Test get methods
    let netuid_result = proxied.NETUID().call().await.unwrap();
    assert_eq!(netuid_result, netuid);

    let trustee_result = proxied.TRUSTEE().call().await.unwrap();
    assert_eq!(trustee_result, trustee);

    let min_collateral_increase_result = proxied.MIN_COLLATERAL_INCREASE().call().await.unwrap();
    assert_eq!(min_collateral_increase_result, min_collateral_increase);

    let decision_timeout_result = proxied.DECISION_TIMEOUT().call().await.unwrap();
    assert_eq!(decision_timeout_result, decision_timeout);

    let node_to_miner_result = proxied
        .nodeToMiner(
            FixedBytes::from_slice(&hotkey),
            FixedBytes::from_slice(&node_id.to_be_bytes()),
        )
        .call()
        .await
        .unwrap();
    assert_eq!(node_to_miner_result, signer.address());

    let collaterals_result = proxied
        .collaterals(
            FixedBytes::from_slice(&hotkey),
            FixedBytes::from_slice(&node_id.to_be_bytes()),
        )
        .call()
        .await
        .unwrap();
    assert_eq!(collaterals_result, U256::ZERO);

    let alpha_collaterals_result = proxied
        .alphaCollaterals(
            FixedBytes::from_slice(&hotkey),
            FixedBytes::from_slice(&node_id.to_be_bytes()),
        )
        .call()
        .await
        .unwrap();
    assert_eq!(alpha_collaterals_result, alpha_amount);
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

    println!("Deployed contract at: {:?}", contract.address());
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

    let data: Bytes = Bytes::from(
        initializeCall {
            netuid,
            trustee,
            minCollateralIncrease: min_collateral_increase,
            decisionTimeout: decision_timeout,
            admin,
        }
        .abi_encode(),
    );

    let contract = Proxy::deploy(provider.clone(), contract_address, data)
        .await
        .unwrap();

    println!("Deployed proxy at: {:?}", contract.address());
}
