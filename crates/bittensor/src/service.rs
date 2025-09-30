//! # Bittensor Service
//!
//! Central service for all Bittensor chain interactions using crabtensor.

use crate::connect::{CircuitBreaker, HealthChecker, RetryConfig, RetryExecutor};
use crate::connect::{ConnectionManager, ConnectionPool, ConnectionPoolBuilder};
use crate::error::BittensorError;
use anyhow::Result;
use basilica_common::config::BittensorConfig;
// Import our own utilities
use crate::utils::{set_weights_payload, NormalizedWeight};
use crate::AccountId;

// Wallet utilities - we'll implement these
use std::path::PathBuf;

// Always use our own generated API module
use crate::api::api;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, info, warn};

// Use subxt directly with our generated API
use subxt::PolkadotConfig;

// Type alias for our chain client removed; Service uses pooled clients via connect::pool

// Type alias for Signer
type Signer = subxt::tx::PairSigner<PolkadotConfig, subxt::ext::sp_core::sr25519::Pair>;

// Wallet helper functions
fn home_hotkey_location(wallet_name: &str, hotkey_name: &str) -> Option<PathBuf> {
    home::home_dir().map(|home| {
        home.join(".bittensor")
            .join("wallets")
            .join(wallet_name)
            .join("hotkeys")
            .join(hotkey_name)
    })
}

fn load_key_seed(path: &PathBuf) -> Result<String, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)?;

    // Try to parse as JSON first (new format)
    if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(&content) {
        if let Some(secret_phrase) = json_value.get("secretPhrase").and_then(|v| v.as_str()) {
            return Ok(secret_phrase.to_string());
        }
        // Fall back to secretSeed if secretPhrase is not available
        if let Some(secret_seed) = json_value.get("secretSeed").and_then(|v| v.as_str()) {
            return Ok(secret_seed.to_string());
        }
        return Err("JSON wallet file missing secretPhrase or secretSeed".into());
    }

    // If not JSON, assume it's a raw seed phrase (old format)
    Ok(content.trim().to_string())
}

fn signer_from_seed(seed: &str) -> Result<Signer, Box<dyn std::error::Error>> {
    use subxt::ext::sp_core::Pair;

    // Try to create pair from string (could be mnemonic or hex seed)
    let pair = if let Some(stripped) = seed.strip_prefix("0x") {
        // It's a hex seed
        let seed_bytes = hex::decode(stripped)?;
        if seed_bytes.len() != 32 {
            return Err("Invalid seed length".into());
        }
        let mut seed_array = [0u8; 32];
        seed_array.copy_from_slice(&seed_bytes);
        subxt::ext::sp_core::sr25519::Pair::from_seed(&seed_array)
    } else {
        // It's a mnemonic phrase
        subxt::ext::sp_core::sr25519::Pair::from_string(seed, None)?
    };

    Ok(subxt::tx::PairSigner::new(pair))
}

// Import the metagraph types
use crate::{Metagraph, SelectiveMetagraph};

/// Central service for Bittensor chain interactions with connection pooling and retry mechanisms
pub struct Service {
    config: BittensorConfig,
    connection_pool: Arc<ConnectionPool>,
    connection_manager: Arc<ConnectionManager>,
    signer: Signer,
    retry_executor: RetryExecutor,
    circuit_breaker: Arc<tokio::sync::Mutex<CircuitBreaker>>,
    health_monitor_handle: Option<tokio::task::JoinHandle<()>>,
}

impl Service {
    /// Creates a new Service instance with the provided configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The Bittensor configuration containing network and wallet settings
    ///
    /// # Returns
    ///
    /// * `Result<Self, BittensorError>` - A Result containing either the initialized Service or an error
    ///
    /// # Errors
    ///
    /// * `NetworkError` - If connection to the client network fails
    /// * `WalletError` - If wallet key loading or signer creation fails
    pub async fn new(config: BittensorConfig) -> Result<Self, BittensorError> {
        info!(
            "Initializing enhanced Bittensor service for network: {}",
            config.network
        );

        // Build connection pool with configuration
        let pool = Arc::new(
            ConnectionPoolBuilder::new(config.get_chain_endpoints())
                .max_connections(config.connection_pool_size.unwrap_or(3))
                .retry_config(RetryConfig::network())
                .build(),
        );

        // Initialize pool with retry logic
        let retry_executor = RetryExecutor::new().with_timeout(Duration::from_secs(120));

        retry_executor
            .execute_with_config(
                || async {
                    pool.initialize()
                        .await
                        .map_err(|e| BittensorError::NetworkError {
                            message: format!("Pool initialization failed: {}", e),
                        })
                },
                RetryConfig::network(),
            )
            .await?;

        info!(
            "Connection pool initialized with {} healthy connections",
            pool.healthy_connection_count().await
        );

        // Create connection manager for state tracking
        let connection_manager = Arc::new(ConnectionManager::new(config.clone()));

        // Start health monitoring
        let health_checker = Arc::new(
            HealthChecker::new()
                .with_interval(
                    config
                        .health_check_interval
                        .unwrap_or(Duration::from_secs(60)),
                )
                .with_timeout(Duration::from_secs(5))
                .with_failure_threshold(3),
        );

        let monitor_handle = health_checker.start_monitoring(Arc::clone(&pool));

        // Load wallet signer
        let hotkey_path = home_hotkey_location(&config.wallet_name, &config.hotkey_name)
            .ok_or_else(|| BittensorError::WalletError {
                message: "Failed to find home directory".to_string(),
            })?;

        info!(
            "Loading hotkey from path: {:?} (wallet: {}, hotkey: {})",
            hotkey_path, config.wallet_name, config.hotkey_name
        );

        let seed = load_key_seed(&hotkey_path).map_err(|e| BittensorError::WalletError {
            message: format!("Failed to load hotkey from {hotkey_path:?}: {e}"),
        })?;

        let signer = signer_from_seed(&seed).map_err(|e| BittensorError::WalletError {
            message: format!("Failed to create signer from seed: {e}"),
        })?;

        // Initialize circuit breaker for cascade failure prevention
        let circuit_breaker = Arc::new(tokio::sync::Mutex::new(CircuitBreaker::new(
            config.circuit_breaker_threshold.unwrap_or(5),
            config
                .circuit_breaker_recovery
                .unwrap_or(Duration::from_secs(60)),
        )));

        let service = Self {
            config,
            connection_pool: pool,
            connection_manager,
            signer,
            retry_executor,
            circuit_breaker,
            health_monitor_handle: Some(monitor_handle),
        };

        info!("Enhanced Bittensor service initialized with connection pooling");
        Ok(service)
    }

    /// Serves an axon on the Bittensor network with retry logic.
    ///
    /// # Arguments
    ///
    /// * `netuid` - The subnet UID to serve the axon on
    /// * `axon_addr` - The socket address where the axon will be served
    ///
    /// # Returns
    ///
    /// * `Result<(), BittensorError>` - A Result indicating success or failure
    ///
    /// # Errors
    ///
    /// * `TxSubmissionError` - If the serve_axon transaction fails to submit
    /// * `MaxRetriesExceeded` - If all retry attempts are exhausted
    pub async fn serve_axon(
        &self,
        netuid: u16,
        axon_addr: SocketAddr,
    ) -> Result<(), BittensorError> {
        info!(
            "Serving axon for netuid {} at {} with retry logic",
            netuid, axon_addr
        );

        let operation = || {
            // Create serve_axon payload using our generated API
            let (ip, ip_type): (u128, u8) = match axon_addr.ip() {
                std::net::IpAddr::V4(ipv4) => (u32::from(ipv4) as u128, 4),
                std::net::IpAddr::V6(ipv6) => (u128::from(ipv6), 6),
            };
            let port = axon_addr.port();
            let protocol = 0; // TCP = 0, UDP = 1

            let payload = api::tx().subtensor_module().serve_axon(
                netuid, 0, // version (u32)
                ip, port, ip_type, protocol, 0, // placeholder1
                0, // placeholder2
            );

            let signer = &self.signer;

            async move {
                // Get a healthy client from the pool
                let client = self
                    .connection_pool
                    .get_healthy_client()
                    .await
                    .map_err(|e| BittensorError::NetworkError {
                        message: format!("Failed to get healthy client: {}", e),
                    })?;

                client
                    .tx()
                    .sign_and_submit_then_watch_default(&payload, signer)
                    .await
                    .map_err(|e| {
                        let err_msg = e.to_string();
                        let err_lower = err_msg.to_lowercase();

                        if err_lower.contains("timeout") {
                            BittensorError::TxTimeoutError {
                                message: format!("serve_axon transaction timeout: {err_msg}"),
                                timeout: Duration::from_secs(60),
                            }
                        } else if err_lower.contains("fee") || err_lower.contains("balance") {
                            BittensorError::InsufficientTxFees {
                                required: 0,
                                available: 0,
                            }
                        } else if err_lower.contains("nonce") {
                            BittensorError::InvalidNonce {
                                expected: 0,
                                actual: 0,
                            }
                        } else {
                            BittensorError::TxSubmissionError {
                                message: format!("Failed to submit serve_axon: {err_msg}"),
                            }
                        }
                    })?;
                Ok(())
            }
        };

        self.retry_executor.execute(operation).await?;
        info!("Axon served successfully");
        Ok(())
    }

    /// Sets weights for neurons in the subnet with retry logic.
    ///
    /// # Arguments
    ///
    /// * `netuid` - The subnet UID to set weights for
    /// * `weights` - Vector of (uid, weight) pairs representing neuron weights
    ///
    /// # Returns
    ///
    /// * `Result<(), BittensorError>` - A Result indicating success or failure
    ///
    /// # Errors
    ///
    /// * `TxSubmissionError` - If the set_weights transaction fails to submit
    /// * `InvalidWeights` - If the weight vector is invalid
    /// * `MaxRetriesExceeded` - If all retry attempts are exhausted
    pub async fn set_weights(
        &self,
        netuid: u16,
        weights: Vec<(u16, u16)>,
    ) -> Result<(), BittensorError> {
        info!(
            "Setting weights for netuid {} with {} weights using retry logic",
            netuid,
            weights.len()
        );

        // Validate weights before attempting transaction
        if weights.is_empty() {
            return Err(BittensorError::InvalidWeights {
                reason: "Weight vector cannot be empty".to_string(),
            });
        }

        // Check for duplicate UIDs
        let mut seen_uids = std::collections::HashSet::new();
        for (uid, _) in &weights {
            if !seen_uids.insert(*uid) {
                return Err(BittensorError::InvalidWeights {
                    reason: format!("Duplicate UID found: {uid}"),
                });
            }
        }

        let operation = || {
            // Convert to NormalizedWeight format
            let normalized_weights: Vec<NormalizedWeight> = weights
                .iter()
                .map(|(uid, weight)| NormalizedWeight {
                    uid: *uid,
                    weight: *weight,
                })
                .collect();

            // Create set_weights payload with version_key = 0
            let payload = set_weights_payload(netuid, normalized_weights, 0);
            let signer = &self.signer;

            async move {
                // Get a healthy client from the pool
                let client = self
                    .connection_pool
                    .get_healthy_client()
                    .await
                    .map_err(|e| BittensorError::NetworkError {
                        message: format!("Failed to get healthy client: {}", e),
                    })?;

                client
                    .tx()
                    .sign_and_submit_then_watch_default(&payload, signer)
                    .await
                    .map_err(|e| {
                        let err_msg = e.to_string();
                        let err_lower = err_msg.to_lowercase();

                        if err_lower.contains("timeout") {
                            BittensorError::TxTimeoutError {
                                message: format!("set_weights transaction timeout: {err_msg}"),
                                timeout: Duration::from_secs(120),
                            }
                        } else if err_lower.contains("weight") || err_lower.contains("invalid") {
                            BittensorError::WeightSettingFailed {
                                netuid,
                                reason: format!("Weight validation failed: {err_msg}"),
                            }
                        } else if err_lower.contains("fee") || err_lower.contains("balance") {
                            BittensorError::InsufficientTxFees {
                                required: 0,
                                available: 0,
                            }
                        } else if err_lower.contains("nonce") {
                            BittensorError::InvalidNonce {
                                expected: 0,
                                actual: 0,
                            }
                        } else {
                            BittensorError::TxSubmissionError {
                                message: format!("Failed to submit set_weights: {err_msg}"),
                            }
                        }
                    })?;
                Ok(())
            }
        };

        self.retry_executor.execute(operation).await?;
        info!("Weights set successfully for netuid {}", netuid);
        Ok(())
    }

    /// Gets neuron information for a specific UID in the subnet.
    ///
    /// # Arguments
    ///
    /// * `netuid` - The subnet UID
    /// * `uid` - The neuron UID to get information for
    ///
    /// # Returns
    ///
    /// * `Result<Option<NeuronInfo<AccountId>>, BittensorError>` - A Result containing either the neuron info or an error
    ///
    /// # Errors
    ///
    /// * `RpcError` - If the RPC call fails
    pub async fn get_neuron(
        &self,
        netuid: u16,
        uid: u16,
    ) -> Result<
        Option<api::runtime_types::pallet_subtensor::rpc_info::neuron_info::NeuronInfo<AccountId>>,
        BittensorError,
    > {
        debug!("Getting neuron info for UID: {} on netuid: {}", uid, netuid);

        // Get a healthy client from the pool
        let client = self
            .connection_pool
            .get_healthy_client()
            .await
            .map_err(|e| BittensorError::NetworkError {
                message: format!("Failed to get healthy client: {}", e),
            })?;

        let runtime_api =
            client
                .runtime_api()
                .at_latest()
                .await
                .map_err(|e| BittensorError::RpcError {
                    message: format!("Failed to get runtime API: {e}"),
                })?;

        let neuron_info = runtime_api
            .call(
                api::runtime_apis::neuron_info_runtime_api::NeuronInfoRuntimeApi
                    .get_neuron(netuid, uid),
            )
            .await
            .map_err(|e| BittensorError::RpcError {
                message: format!("Failed to call get_neuron: {e}"),
            })?;

        Ok(neuron_info)
    }

    /// Gets the complete metagraph for a subnet with circuit breaker protection.
    ///
    /// # Arguments
    ///
    /// * `netuid` - The subnet UID to get the metagraph for
    ///
    /// # Returns
    ///
    /// * `Result<Metagraph<AccountId>, BittensorError>` - A Result containing either the metagraph or an error
    ///
    /// # Errors
    ///
    /// * `RpcError` - If the RPC call fails
    /// * `SubnetNotFound` - If the subnet doesn't exist
    /// * `ServiceUnavailable` - If circuit breaker is open
    pub async fn get_metagraph(&self, netuid: u16) -> Result<Metagraph<AccountId>, BittensorError> {
        info!(
            "Fetching metagraph for netuid: {} with circuit breaker protection",
            netuid
        );

        let operation = || {
            async move {
                // Get a healthy client from the pool
                let client = self
                    .connection_pool
                    .get_healthy_client()
                    .await
                    .map_err(|e| BittensorError::NetworkError {
                        message: format!("Failed to get healthy client: {}", e),
                    })?;

                let runtime_api = client.runtime_api().at_latest().await.map_err(|e| {
                    let err_msg = e.to_string();
                    let err_lower = err_msg.to_lowercase();

                    if err_lower.contains("timeout") {
                        BittensorError::RpcTimeoutError {
                            message: format!("Runtime API timeout: {err_msg}"),
                            timeout: Duration::from_secs(30),
                        }
                    } else if err_lower.contains("connection") {
                        BittensorError::RpcConnectionError {
                            message: format!("Runtime API connection failed: {err_msg}"),
                        }
                    } else {
                        BittensorError::RpcMethodError {
                            method: "runtime_api".to_string(),
                            message: err_msg,
                        }
                    }
                })?;

                let metagraph = runtime_api
                    .call(
                        api::runtime_apis::subnet_info_runtime_api::SubnetInfoRuntimeApi
                            .get_metagraph(netuid),
                    )
                    .await
                    .map_err(|e| {
                        let err_msg = e.to_string();
                        if err_msg.to_lowercase().contains("timeout") {
                            BittensorError::RpcTimeoutError {
                                message: format!("get_metagraph call timeout: {err_msg}"),
                                timeout: Duration::from_secs(30),
                            }
                        } else {
                            BittensorError::RpcMethodError {
                                method: "get_metagraph".to_string(),
                                message: err_msg,
                            }
                        }
                    })?
                    .ok_or(BittensorError::SubnetNotFound { netuid })?;

                Ok(metagraph)
            }
        };

        // Use circuit breaker for RPC calls
        // Clone the circuit breaker to avoid holding the lock across await
        let mut circuit_breaker = {
            let cb = self.circuit_breaker.lock().await;
            cb.clone()
        };
        let result = circuit_breaker.execute(operation).await;

        // Update the original circuit breaker with the new state
        {
            let mut original_cb = self.circuit_breaker.lock().await;
            *original_cb = circuit_breaker;
        }

        match &result {
            Ok(_) => info!("Metagraph fetched successfully for netuid: {}", netuid),
            Err(e) => warn!("Failed to fetch metagraph for netuid {}: {}", netuid, e),
        }

        result
    }

    /// Retrieves a selective metagraph for a specific subnet, containing only the requested fields.
    ///
    /// # Arguments
    ///
    /// * `netuid` - The subnet UID to get the selective metagraph for
    /// * `fields` - Vector of field indices to include in the selective metagraph
    ///
    /// # Returns
    ///
    /// * `Result<SelectiveMetagraph<AccountId>, BittensorError>` - A Result containing either the selective metagraph or an error
    ///
    /// # Errors
    ///
    /// * `RpcError` - If connection to the runtime API fails
    /// * `RpcError` - If the selective metagraph call fails
    /// * `RpcError` - If no selective metagraph is found for the subnet
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use bittensor::Service;
    /// # use basilica_common::config::BittensorConfig;
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let config = BittensorConfig::default();
    /// # let service = Service::new(config).await?;
    /// let fields = vec![0, 1, 2]; // Include first three fields
    /// let selective_metagraph = service.get_selective_metagraph(1, fields).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_selective_metagraph(
        &self,
        netuid: u16,
        fields: Vec<u16>,
    ) -> Result<SelectiveMetagraph<AccountId>, BittensorError> {
        info!(
            "Fetching selective metagraph for netuid: {} with {} fields",
            netuid,
            fields.len()
        );

        // Get a healthy client from the pool
        let client = self
            .connection_pool
            .get_healthy_client()
            .await
            .map_err(|e| BittensorError::NetworkError {
                message: format!("Failed to get healthy client: {}", e),
            })?;

        let runtime_api =
            client
                .runtime_api()
                .at_latest()
                .await
                .map_err(|e| BittensorError::RpcError {
                    message: format!("Failed to get runtime API: {e}"),
                })?;

        let selective_metagraph = runtime_api
            .call(
                api::runtime_apis::subnet_info_runtime_api::SubnetInfoRuntimeApi
                    .get_selective_metagraph(netuid, fields),
            )
            .await
            .map_err(|e| BittensorError::RpcError {
                message: format!("Failed to call get_selective_metagraph: {e}"),
            })?
            .ok_or_else(|| BittensorError::RpcError {
                message: format!("Selective metagraph not found for subnet {netuid}"),
            })?;

        Ok(selective_metagraph)
    }

    /// Retrieves the current block number from the Bittensor network.
    ///
    /// # Returns
    ///
    /// * `Result<u64, BittensorError>` - A Result containing either the current block number or an error
    ///
    /// # Errors
    ///
    /// * `RpcError` - If connection to the latest block fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use bittensor::Service;
    /// # use basilica_common::config::BittensorConfig;
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let config = BittensorConfig::default();
    /// # let service = Service::new(config).await?;
    /// let block_number = service.get_block_number().await?;
    /// println!("Current block: {}", block_number);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_block_number(&self) -> Result<u64, BittensorError> {
        // Get a healthy client from the pool
        let client = self
            .connection_pool
            .get_healthy_client()
            .await
            .map_err(|e| BittensorError::NetworkError {
                message: format!("Failed to get healthy client: {}", e),
            })?;

        let latest_block =
            client
                .blocks()
                .at_latest()
                .await
                .map_err(|e| BittensorError::RpcError {
                    message: format!("Failed to get latest block: {e}"),
                })?;

        Ok(latest_block.number().into())
    }

    /// Returns the account ID associated with the service's signer.
    ///
    /// # Returns
    ///
    /// * `&AccountId` - Reference to the signer's account ID
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use bittensor::Service;
    /// # use basilica_common::config::BittensorConfig;
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let config = BittensorConfig::default();
    /// # let service = Service::new(config).await?;
    /// let account_id = service.get_account_id();
    /// println!("Account ID: {}", account_id);
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_account_id(&self) -> &AccountId {
        self.signer.account_id()
    }

    /// Get current block number (alias for get_block_number)
    pub async fn get_current_block(&self) -> Result<u64, BittensorError> {
        self.get_block_number().await
    }

    /// Submit an extrinsic (transaction) to the chain
    pub async fn submit_extrinsic<T>(&self, payload: T) -> Result<(), BittensorError>
    where
        T: subxt::tx::Payload,
    {
        // Get a healthy client from the pool
        let client = self
            .connection_pool
            .get_healthy_client()
            .await
            .map_err(|e| BittensorError::NetworkError {
                message: format!("Failed to get healthy client: {}", e),
            })?;

        let tx_result = client
            .tx()
            .sign_and_submit_default(&payload, &self.signer)
            .await
            .map_err(|e| BittensorError::TxSubmissionError {
                message: format!("Failed to submit extrinsic: {e}"),
            })?;

        info!("Transaction submitted with hash: {:?}", tx_result);
        Ok(())
    }

    /// Returns the configured network name for this service instance.
    ///
    /// # Returns
    ///
    /// * `&str` - Reference to the network name
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use bittensor::Service;
    /// # use basilica_common::config::BittensorConfig;
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let config = BittensorConfig::default();
    /// # let service = Service::new(config).await?;
    /// let network = service.get_network();
    /// println!("Connected to network: {}", network);
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_network(&self) -> &str {
        &self.config.network
    }

    /// Returns the configured subnet UID for this service instance.
    ///
    /// # Returns
    ///
    /// * `u16` - The subnet UID
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use bittensor::Service;
    /// # use basilica_common::config::BittensorConfig;
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let config = BittensorConfig::default();
    /// # let service = Service::new(config).await?;
    /// let netuid = service.get_netuid();
    /// println!("Subnet UID: {}", netuid);
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_netuid(&self) -> u16 {
        self.config.netuid
    }

    /// Sign data with the service's signer (hotkey)
    ///
    /// This method signs arbitrary data with the validator/miner's hotkey.
    /// The signature can be verified using `verify_bittensor_signature`.
    ///
    /// # Arguments
    ///
    /// * `data` - The data to sign
    ///
    /// # Returns
    ///
    /// * `Result<String, BittensorError>` - Hex-encoded signature string
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use bittensor::Service;
    /// # use basilica_common::config::BittensorConfig;
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let config = BittensorConfig::default();
    /// # let service = Service::new(config).await?;
    /// let nonce = "test-nonce-123";
    /// let signature = service.sign_data(nonce.as_bytes())?;
    /// println!("Signature: {}", signature);
    /// # Ok(())
    /// # }
    /// ```
    pub fn sign_data(&self, data: &[u8]) -> Result<String, BittensorError> {
        use subxt::tx::Signer as SignerTrait;

        // Sign the data with our signer
        let signature = self.signer.sign(data);

        // For sr25519, we need to extract the signature bytes
        // The MultiSignature contains the actual signature data
        match signature {
            subxt::utils::MultiSignature::Sr25519(sig) => Ok(hex::encode(sig)),
            _ => Err(BittensorError::AuthError {
                message: "Unexpected signature type - expected Sr25519".to_string(),
            }),
        }
    }
}

impl Service {
    /// Gets retry statistics for monitoring
    pub async fn get_retry_stats(&self) -> RetryStats {
        RetryStats {
            circuit_breaker_state: {
                let cb = self.circuit_breaker.lock().await;
                format!("{cb:?}")
            },
        }
    }

    /// Resets the circuit breaker state (for recovery operations)
    pub async fn reset_circuit_breaker(&self) {
        let mut cb = self.circuit_breaker.lock().await;
        *cb = CircuitBreaker::new(5, Duration::from_secs(60));
        info!("Circuit breaker reset");
    }

    /// Get connection pool metrics for monitoring
    pub async fn connection_metrics(&self) -> ConnectionPoolMetrics {
        ConnectionPoolMetrics {
            total_connections: self.connection_pool.total_connections().await,
            healthy_connections: self.connection_pool.healthy_connection_count().await,
            connection_state: self.connection_manager.get_state().await.status_message(),
            metrics: self.connection_manager.metrics(),
        }
    }

    /// Force reconnection of all connections
    pub async fn force_reconnect(&self) -> Result<(), BittensorError> {
        warn!("Forcing reconnection of all connections");
        self.connection_pool.refresh_connections().await?;
        self.connection_manager.force_reconnect().await?;
        Ok(())
    }

    /// Gracefully shutdown the service
    pub async fn shutdown(mut self) {
        info!("Shutting down enhanced Bittensor service");

        // Stop health monitoring
        if let Some(handle) = self.health_monitor_handle.take() {
            handle.abort();
        }

        // Connection pool will be dropped automatically
    }
}

/// Statistics for retry mechanisms
#[derive(Debug, Clone)]
pub struct RetryStats {
    pub circuit_breaker_state: String,
}

/// Connection pool metrics for monitoring
#[derive(Debug, Clone)]
pub struct ConnectionPoolMetrics {
    pub total_connections: usize,
    pub healthy_connections: usize,
    pub connection_state: String,
    pub metrics: crate::connect::ConnectionMetricsSnapshot,
}

// Implement Drop to ensure cleanup
impl Drop for Service {
    fn drop(&mut self) {
        if let Some(handle) = self.health_monitor_handle.take() {
            handle.abort();
        }
    }
}
