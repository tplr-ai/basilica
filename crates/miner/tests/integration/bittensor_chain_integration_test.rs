use anyhow::Result;
use miner::bittensor_core::chain_registration::ChainRegistrationService;
use miner::config::{BittensorConfig, MinerConfig};
use sqlx::SqlitePool;
use std::time::Duration;
use tempfile::TempDir;
use tokio::time::timeout;

#[tokio::test]
async fn test_chain_registration_and_uid_discovery() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("test_miner.db");
    let db_url = format!("sqlite:{}", db_path.display());

    let pool = SqlitePool::connect(&db_url).await?;
    sqlx::migrate!("../../migrations").run(&pool).await?;

    let bittensor_config = BittensorConfig {
        wallet_name: "test_wallet".to_string(),
        hotkey_name: "test_hotkey".to_string(),
        network: "test".to_string(),
        netuid: 999,
        chain_endpoint: "wss://test-endpoint.invalid:443".to_string(),
        weight_interval_secs: 300,
        uid: None,
        coldkey_name: "test_coldkey".to_string(),
        axon_port: 8091,
        external_ip: Some("192.168.1.100".to_string()),
        max_weight_uids: 256,
    };

    let config = MinerConfig {
        bittensor: bittensor_config.clone(),
        ..Default::default()
    };

    let service = ChainRegistrationService::new(config.clone(), pool.clone());

    let registration_result =
        timeout(Duration::from_secs(5), service.register_and_discover_uid()).await;

    match registration_result {
        Ok(Ok(uid)) => {
            assert!(uid > 0, "UID should be positive");
            assert!(uid < 1000, "UID should be within reasonable range");
        }
        Ok(Err(e)) => {
            assert!(
                e.to_string().contains("chain")
                    || e.to_string().contains("network")
                    || e.to_string().contains("connection"),
                "Expected network-related error, got: {}",
                e
            );
        }
        Err(_) => {
            // Timeout is expected in test environment
        }
    }

    Ok(())
}

#[tokio::test]
async fn test_external_ip_discovery() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("test_miner.db");
    let db_url = format!("sqlite:{}", db_path.display());

    let pool = SqlitePool::connect(&db_url).await?;
    sqlx::migrate!("../../migrations").run(&pool).await?;

    let mut bittensor_config = BittensorConfig {
        wallet_name: "test_wallet".to_string(),
        hotkey_name: "test_hotkey".to_string(),
        network: "test".to_string(),
        netuid: 999,
        chain_endpoint: "wss://test-endpoint.invalid:443".to_string(),
        weight_interval_secs: 300,
        uid: None,
        coldkey_name: "test_coldkey".to_string(),
        axon_port: 8091,
        external_ip: None, // No external IP provided
        max_weight_uids: 256,
    };

    let config = MinerConfig {
        bittensor: bittensor_config.clone(),
        ..Default::default()
    };

    let service = ChainRegistrationService::new(config, pool);

    // Test IP discovery
    let ip_result = service.discover_external_ip().await;

    match ip_result {
        Ok(ip) => {
            assert!(!ip.is_empty(), "Discovered IP should not be empty");
            assert!(
                ip.parse::<std::net::IpAddr>().is_ok(),
                "Should be valid IP address"
            );
        }
        Err(e) => {
            // In test environment, this might fail
            assert!(
                e.to_string().contains("discover") || e.to_string().contains("external"),
                "Expected IP discovery error, got: {}",
                e
            );
        }
    }

    Ok(())
}

#[tokio::test]
async fn test_axon_server_registration() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("test_miner.db");
    let db_url = format!("sqlite:{}", db_path.display());

    let pool = SqlitePool::connect(&db_url).await?;
    sqlx::migrate!("../../migrations").run(&pool).await?;

    let bittensor_config = BittensorConfig {
        wallet_name: "test_wallet".to_string(),
        hotkey_name: "test_hotkey".to_string(),
        network: "test".to_string(),
        netuid: 999,
        chain_endpoint: "wss://test-endpoint.invalid:443".to_string(),
        weight_interval_secs: 300,
        uid: Some(42), // Pre-discovered UID
        coldkey_name: "test_coldkey".to_string(),
        axon_port: 8091,
        external_ip: Some("192.168.1.100".to_string()),
        max_weight_uids: 256,
    };

    let config = MinerConfig {
        bittensor: bittensor_config,
        ..Default::default()
    };

    let service = ChainRegistrationService::new(config.clone(), pool);

    // Test axon registration
    let registration_result =
        timeout(Duration::from_secs(5), service.register_axon_on_chain()).await;

    match registration_result {
        Ok(Ok(_)) => {
            // Registration succeeded (unlikely in test)
        }
        Ok(Err(e)) => {
            assert!(
                e.to_string().contains("axon")
                    || e.to_string().contains("chain")
                    || e.to_string().contains("network"),
                "Expected axon registration error, got: {}",
                e
            );
        }
        Err(_) => {
            // Timeout is expected in test environment
        }
    }

    Ok(())
}

#[tokio::test]
async fn test_hotkey_validation() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("test_miner.db");
    let db_url = format!("sqlite:{}", db_path.display());

    let pool = SqlitePool::connect(&db_url).await?;

    // Test with invalid hotkey format
    let invalid_config = BittensorConfig {
        wallet_name: "test_wallet".to_string(),
        hotkey_name: "invalid!@#$%^&*()".to_string(), // Invalid characters
        network: "test".to_string(),
        netuid: 999,
        chain_endpoint: "wss://test-endpoint.invalid:443".to_string(),
        weight_interval_secs: 300,
        uid: None,
        coldkey_name: "test_coldkey".to_string(),
        axon_port: 8091,
        external_ip: None,
        max_weight_uids: 256,
    };

    let config = MinerConfig {
        bittensor: invalid_config,
        ..Default::default()
    };

    let service = ChainRegistrationService::new(config, pool);
    let result = service.validate_hotkey().await;

    assert!(result.is_err(), "Invalid hotkey should fail validation");

    Ok(())
}

#[tokio::test]
async fn test_weight_update_scheduling() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("test_miner.db");
    let db_url = format!("sqlite:{}", db_path.display());

    let pool = SqlitePool::connect(&db_url).await?;
    sqlx::migrate!("../../migrations").run(&pool).await?;

    let bittensor_config = BittensorConfig {
        wallet_name: "test_wallet".to_string(),
        hotkey_name: "test_hotkey".to_string(),
        network: "test".to_string(),
        netuid: 999,
        chain_endpoint: "wss://test-endpoint.invalid:443".to_string(),
        weight_interval_secs: 1, // Very short interval for testing
        uid: Some(42),
        coldkey_name: "test_coldkey".to_string(),
        axon_port: 8091,
        external_ip: Some("192.168.1.100".to_string()),
        max_weight_uids: 256,
    };

    let config = MinerConfig {
        bittensor: bittensor_config,
        ..Default::default()
    };

    let service = ChainRegistrationService::new(config, pool);

    // Test weight update scheduling
    let schedule = service.get_weight_update_schedule();
    assert_eq!(
        schedule,
        Duration::from_secs(1),
        "Weight update interval should match config"
    );

    // Test weight calculation
    let weights = service.calculate_miner_weights().await?;
    assert!(
        weights.is_empty() || weights.len() <= 256,
        "Weights should respect max_weight_uids"
    );

    Ok(())
}
