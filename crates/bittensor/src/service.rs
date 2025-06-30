//! # Bittensor Service
//!
//! Central service for all Bittensor chain interactions using crabtensor.

use crate::error::BittensorError;
use crate::retry::{CircuitBreaker, RetryExecutor};
use anyhow::Result;
use common::config::BittensorConfig;
use crabtensor::{
    axon::{serve_axon_payload, AxonProtocol},
    subtensor::{from_url, Subtensor, SubtensorUrl},
    wallet::{home_hotkey_location, load_key_seed, signer_from_seed, Signer},
    weights::{set_weights_payload, NormalizedWeight},
    AccountId,
};

// Use our own API module for testnet compatibility
#[cfg(feature = "testnet")]
use crate::api;

// Use crabtensor's API for mainnet (default)
#[cfg(not(feature = "testnet"))]
use crabtensor::api;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tracing::{debug, info, warn};

// Import the metagraph types
use crate::{Metagraph, SelectiveMetagraph};

/// Central service for Bittensor chain interactions with retry mechanisms
pub struct Service {
    config: BittensorConfig,
    subtensor: Subtensor,
    signer: Signer,
    retry_executor: RetryExecutor,
    circuit_breaker: Arc<Mutex<CircuitBreaker>>,
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
    /// * `NetworkError` - If connection to the subtensor network fails
    /// * `WalletError` - If wallet key loading or signer creation fails
    pub async fn new(config: BittensorConfig) -> Result<Self, BittensorError> {
        info!(
            "Initializing Bittensor service for network: {}",
            config.network
        );

        // Create subtensor connection
        let subtensor_url =
            Self::get_subtensor_url(&config.network, config.chain_endpoint.as_deref());

        info!("Subtensor URL: {}", subtensor_url);

        // Create subtensor connection - handle insecure ws:// URLs for local development
        let subtensor = if subtensor_url.starts_with("ws://")
            && !subtensor_url.starts_with("wss://")
        {
            warn!(
                "Using insecure WebSocket connection for local development: {}",
                subtensor_url
            );

            // For subxt 0.38, we need to create an RpcClient that allows insecure URLs
            use subxt::backend::legacy::LegacyBackend;
            use subxt::backend::rpc::RpcClient;

            let rpc_client = RpcClient::from_insecure_url(&subtensor_url)
                .await
                .map_err(|e| BittensorError::NetworkError {
                    message: format!("Failed to create RPC client: {e}"),
                })?;

            let backend = LegacyBackend::builder().build(rpc_client);
            Subtensor::from_backend(Arc::new(backend))
                .await
                .map_err(|e| BittensorError::NetworkError {
                    message: format!("Failed to create subtensor from backend: {e}"),
                })?
        } else {
            info!("Using secure connection for: {}", subtensor_url);
            from_url(&subtensor_url)
                .await
                .map_err(|e| BittensorError::NetworkError {
                    message: format!("Failed to connect to subtensor: {e}"),
                })?
        };

        // Load wallet signer
        let hotkey_path = home_hotkey_location(&config.wallet_name, &config.hotkey_name)
            .ok_or_else(|| BittensorError::WalletError {
                message: "Failed to find home directory".to_string(),
            })?;

        let seed = load_key_seed(&hotkey_path).map_err(|e| BittensorError::WalletError {
            message: format!("Failed to load hotkey seed: {e}"),
        })?;

        let signer = signer_from_seed(&seed).map_err(|e| BittensorError::WalletError {
            message: format!("Failed to create signer from seed: {e}"),
        })?;

        let service = Self {
            config,
            subtensor,
            signer,
            retry_executor: RetryExecutor::new().with_timeout(Duration::from_secs(300)),
            circuit_breaker: Arc::new(Mutex::new(CircuitBreaker::new(
                5,                       // failure threshold
                Duration::from_secs(60), // recovery timeout
            ))),
        };

        info!("Bittensor service initialized successfully with retry mechanisms");
        Ok(service)
    }

    /// Gets the subtensor URL for the specified network.
    ///
    /// # Arguments
    ///
    /// * `network` - The network name (e.g., "finney", "test", "local", "archive")
    /// * `chain_endpoint` - Optional custom chain endpoint URL
    ///
    /// # Returns
    ///
    /// * `String` - The appropriate subtensor URL for the network
    fn get_subtensor_url(network: &str, chain_endpoint: Option<&str>) -> String {
        if let Some(endpoint) = chain_endpoint {
            return endpoint.to_string();
        }

        match network {
            "finney" => SubtensorUrl::Finney.as_ref().to_string(),
            "test" => SubtensorUrl::Test.as_ref().to_string(),
            "local" => SubtensorUrl::Local.as_ref().to_string(),
            "archive" => SubtensorUrl::Archive.as_ref().to_string(),
            _ => {
                warn!("Unknown network '{}', using finney endpoint", network);
                SubtensorUrl::Finney.as_ref().to_string()
            }
        }
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
            let payload = serve_axon_payload(netuid, axon_addr, AxonProtocol::Tcp);
            let subtensor = &self.subtensor;
            let signer = &self.signer;

            async move {
                subtensor
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
            let subtensor = &self.subtensor;
            let signer = &self.signer;

            async move {
                subtensor
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

        let runtime_api = self
            .subtensor
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
            let subtensor = &self.subtensor;
            async move {
                let runtime_api = subtensor.runtime_api().at_latest().await.map_err(|e| {
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
            let cb = self.circuit_breaker.lock().unwrap();
            cb.clone()
        };
        let result = circuit_breaker.execute(operation).await;

        // Update the original circuit breaker with the new state
        {
            let mut original_cb = self.circuit_breaker.lock().unwrap();
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
    /// # use common::config::BittensorConfig;
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

        let runtime_api = self
            .subtensor
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
    /// # use common::config::BittensorConfig;
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
        let latest_block =
            self.subtensor
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
    /// # use common::config::BittensorConfig;
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
        let tx_result = self
            .subtensor
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
    /// # use common::config::BittensorConfig;
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
    /// # use common::config::BittensorConfig;
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
}

impl Service {
    /// Gets retry statistics for monitoring
    pub fn get_retry_stats(&self) -> RetryStats {
        RetryStats {
            circuit_breaker_state: {
                let cb = self.circuit_breaker.lock().unwrap();
                format!("{cb:?}")
            },
        }
    }

    /// Resets the circuit breaker state (for recovery operations)
    pub fn reset_circuit_breaker(&self) {
        let mut cb = self.circuit_breaker.lock().unwrap();
        *cb = CircuitBreaker::new(5, Duration::from_secs(60));
        info!("Circuit breaker reset");
    }
}

/// Statistics for retry mechanisms
#[derive(Debug, Clone)]
pub struct RetryStats {
    pub circuit_breaker_state: String,
}
