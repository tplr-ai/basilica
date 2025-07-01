//! # Bittensor Error Types
//!
//! Comprehensive error handling for Bittensor chain interactions with detailed
//! error categorization and retry support.

use std::time::Duration;
use thiserror::Error;

/// Errors that can occur during Bittensor operations
#[derive(Error, Debug, Clone)]
pub enum BittensorError {
    // === Transaction Errors ===
    #[error("Transaction submission failed: {message}")]
    TxSubmissionError { message: String },

    #[error("Transaction timeout after {timeout:?}: {message}")]
    TxTimeoutError { message: String, timeout: Duration },

    #[error("Transaction fees insufficient: required {required}, available {available}")]
    InsufficientTxFees { required: u64, available: u64 },

    #[error("Transaction nonce invalid: expected {expected}, got {actual}")]
    InvalidNonce { expected: u64, actual: u64 },

    #[error("Transaction finalization failed: {reason}")]
    TxFinalizationError { reason: String },

    #[error("Transaction dropped from pool: {reason}")]
    TxDroppedError { reason: String },

    // === RPC and Network Errors ===
    #[error("RPC connection error: {message}")]
    RpcConnectionError { message: String },

    #[error("RPC method error: {method} - {message}")]
    RpcMethodError { method: String, message: String },

    #[error("RPC timeout after {timeout:?}: {message}")]
    RpcTimeoutError { message: String, timeout: Duration },

    #[error("Network connectivity issue: {message}")]
    NetworkConnectivityError { message: String },

    #[error("Chain synchronization error: {message}")]
    ChainSyncError { message: String },

    #[error("Websocket connection error: {message}")]
    WebsocketError { message: String },

    // === Chain State Errors ===
    #[error("Chain metadata error: {message}")]
    MetadataError { message: String },

    #[error("Runtime version mismatch: expected {expected}, got {actual}")]
    RuntimeVersionMismatch { expected: String, actual: String },

    #[error("Storage query failed: {key} - {message}")]
    StorageQueryError { key: String, message: String },

    #[error("Block hash not found: {hash}")]
    BlockNotFound { hash: String },

    #[error("Invalid block number: {number}")]
    InvalidBlockNumber { number: u64 },

    // === Wallet and Authentication Errors ===
    #[error("Wallet loading error: {message}")]
    WalletLoadingError { message: String },

    #[error("Key derivation error: {message}")]
    KeyDerivationError { message: String },

    #[error("Signature verification failed: {message}")]
    SignatureError { message: String },

    #[error("Invalid hotkey format: {hotkey}")]
    InvalidHotkey { hotkey: String },

    #[error("Hotkey not registered on subnet {netuid}: {hotkey}")]
    HotkeyNotRegistered { hotkey: String, netuid: u16 },

    // === Neuron and Subnet Errors ===
    #[error("Neuron not found: uid {uid} on subnet {netuid}")]
    NeuronNotFound { uid: u16, netuid: u16 },

    #[error("Subnet not found: {netuid}")]
    SubnetNotFound { netuid: u16 },

    #[error("Insufficient stake: {available} TAO < {required} TAO")]
    InsufficientStake { available: u64, required: u64 },

    #[error("Weight setting failed on subnet {netuid}: {reason}")]
    WeightSettingFailed { netuid: u16, reason: String },

    #[error("Invalid weight vector: {reason}")]
    InvalidWeights { reason: String },

    #[error("Registration failed on subnet {netuid}: {reason}")]
    RegistrationFailed { netuid: u16, reason: String },

    // === Operational Errors ===
    #[error("Serialization error: {message}")]
    SerializationError { message: String },

    #[error("Configuration error: {field} - {message}")]
    ConfigError { field: String, message: String },

    #[error("Operation timeout after {timeout:?}: {operation}")]
    OperationTimeout {
        operation: String,
        timeout: Duration,
    },

    #[error("Rate limit exceeded: {message}")]
    RateLimitExceeded { message: String },

    #[error("Service unavailable: {message}")]
    ServiceUnavailable { message: String },

    // === Retry and Recovery Errors ===
    #[error("Maximum retry attempts exceeded: {attempts} attempts failed")]
    MaxRetriesExceeded { attempts: u32 },

    #[error("Backoff timeout reached: operation abandoned after {duration:?}")]
    BackoffTimeoutReached { duration: Duration },

    #[error("Non-retryable error: {message}")]
    NonRetryable { message: String },

    // === Legacy Error Types (for backwards compatibility) ===
    #[error("RPC error: {message}")]
    RpcError { message: String },

    #[error("Network error: {message}")]
    NetworkError { message: String },

    #[error("Chain error: {message}")]
    ChainError { message: String },

    #[error("Wallet error: {message}")]
    WalletError { message: String },

    #[error("Timeout error: {message}")]
    TimeoutError { message: String },

    #[error("Authentication error: {message}")]
    AuthError { message: String },

    #[error("Insufficient balance: {available} < {required}")]
    InsufficientBalance { available: u64, required: u64 },
}

/// Classification of errors for retry logic
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    /// Transient errors that can be retried with exponential backoff
    Transient,
    /// Rate limiting errors that require specific backoff strategies
    RateLimit,
    /// Authentication/authorization errors that may be retryable
    Auth,
    /// Configuration or input validation errors (not retryable)
    Config,
    /// Network connectivity issues (retryable with longer backoff)
    Network,
    /// Permanent errors that should not be retried
    Permanent,
}

/// Retry configuration for different error categories
#[derive(Debug, Clone)]
pub struct RetryConfig {
    pub max_attempts: u32,
    pub initial_delay: Duration,
    pub max_delay: Duration,
    pub backoff_multiplier: f64,
    pub jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            jitter: true,
        }
    }
}

impl RetryConfig {
    /// Configuration for transient errors
    pub fn transient() -> Self {
        Self {
            max_attempts: 5,
            initial_delay: Duration::from_millis(200),
            max_delay: Duration::from_secs(10),
            backoff_multiplier: 1.5,
            jitter: true,
        }
    }

    /// Configuration for rate limit errors
    pub fn rate_limit() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(60),
            backoff_multiplier: 2.0,
            jitter: false,
        }
    }

    /// Configuration for network errors
    pub fn network() -> Self {
        Self {
            max_attempts: 4,
            initial_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            jitter: true,
        }
    }

    /// Configuration for authentication errors
    pub fn auth() -> Self {
        Self {
            max_attempts: 2,
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(5),
            backoff_multiplier: 1.0,
            jitter: false,
        }
    }
}

impl From<anyhow::Error> for BittensorError {
    fn from(err: anyhow::Error) -> Self {
        BittensorError::ChainError {
            message: err.to_string(),
        }
    }
}

// Enhanced conversions from subxt errors with detailed error mapping
impl From<subxt::Error> for BittensorError {
    fn from(err: subxt::Error) -> Self {
        let err_str = err.to_string().to_lowercase();

        match err {
            subxt::Error::Rpc(rpc_err) => {
                let rpc_msg = rpc_err.to_string();
                let rpc_lower = rpc_msg.to_lowercase();

                if rpc_lower.contains("timeout") {
                    BittensorError::RpcTimeoutError {
                        message: rpc_msg,
                        timeout: Duration::from_secs(30), // Default timeout
                    }
                } else if rpc_lower.contains("connection") || rpc_lower.contains("network") {
                    BittensorError::RpcConnectionError { message: rpc_msg }
                } else if rpc_lower.contains("rate") || rpc_lower.contains("limit") {
                    BittensorError::RateLimitExceeded { message: rpc_msg }
                } else {
                    BittensorError::RpcMethodError {
                        method: "unknown".to_string(),
                        message: rpc_msg,
                    }
                }
            }
            subxt::Error::Metadata(meta_err) => {
                let meta_msg = meta_err.to_string();
                if meta_msg.to_lowercase().contains("version") {
                    BittensorError::RuntimeVersionMismatch {
                        expected: "unknown".to_string(),
                        actual: "unknown".to_string(),
                    }
                } else {
                    BittensorError::MetadataError { message: meta_msg }
                }
            }
            subxt::Error::Codec(codec_err) => BittensorError::SerializationError {
                message: codec_err.to_string(),
            },
            subxt::Error::Transaction(tx_err) => {
                let tx_msg = tx_err.to_string();
                let tx_lower = tx_msg.to_lowercase();

                if tx_lower.contains("timeout") {
                    BittensorError::TxTimeoutError {
                        message: tx_msg,
                        timeout: Duration::from_secs(60),
                    }
                } else if tx_lower.contains("fee") || tx_lower.contains("balance") {
                    BittensorError::InsufficientTxFees {
                        required: 0,
                        available: 0,
                    }
                } else if tx_lower.contains("nonce") {
                    BittensorError::InvalidNonce {
                        expected: 0,
                        actual: 0,
                    }
                } else if tx_lower.contains("dropped") || tx_lower.contains("pool") {
                    BittensorError::TxDroppedError { reason: tx_msg }
                } else if tx_lower.contains("finalization") || tx_lower.contains("finalized") {
                    BittensorError::TxFinalizationError { reason: tx_msg }
                } else {
                    BittensorError::TxSubmissionError { message: tx_msg }
                }
            }
            subxt::Error::Block(block_err) => {
                let block_msg = format!("Block error: {block_err}");
                if err_str.contains("not found") {
                    BittensorError::BlockNotFound {
                        hash: "unknown".to_string(),
                    }
                } else {
                    BittensorError::ChainError { message: block_msg }
                }
            }
            subxt::Error::Runtime(runtime_err) => {
                let runtime_msg = format!("Runtime error: {runtime_err}");
                if err_str.contains("version") {
                    BittensorError::RuntimeVersionMismatch {
                        expected: "unknown".to_string(),
                        actual: "unknown".to_string(),
                    }
                } else {
                    BittensorError::ChainError {
                        message: runtime_msg,
                    }
                }
            }
            subxt::Error::Other(other_err) => {
                if err_str.contains("websocket") || err_str.contains("ws") {
                    BittensorError::WebsocketError { message: other_err }
                } else if err_str.contains("network") || err_str.contains("connection") {
                    BittensorError::NetworkConnectivityError { message: other_err }
                } else {
                    BittensorError::ChainError { message: other_err }
                }
            }
            _ => {
                if err_str.contains("timeout") {
                    BittensorError::OperationTimeout {
                        operation: "subxt_operation".to_string(),
                        timeout: Duration::from_secs(30),
                    }
                } else if err_str.contains("network") || err_str.contains("connection") {
                    BittensorError::NetworkConnectivityError {
                        message: err.to_string(),
                    }
                } else {
                    BittensorError::ChainError {
                        message: err.to_string(),
                    }
                }
            }
        }
    }
}

// Enhanced conversions from wallet errors
impl From<std::io::Error> for BittensorError {
    fn from(err: std::io::Error) -> Self {
        let err_msg = err.to_string();
        let err_lower = err_msg.to_lowercase();

        if err_lower.contains("file") || err_lower.contains("path") || err_lower.contains("io") {
            BittensorError::WalletLoadingError {
                message: format!("Wallet file access failed: {err}"),
            }
        } else if err_lower.contains("key") || err_lower.contains("derivation") {
            BittensorError::KeyDerivationError {
                message: format!("Key derivation failed: {err}"),
            }
        } else if err_lower.contains("format") || err_lower.contains("invalid") {
            BittensorError::InvalidHotkey {
                hotkey: "unknown".to_string(),
            }
        } else {
            BittensorError::WalletLoadingError {
                message: format!("Account loading failed: {err}"),
            }
        }
    }
}

// Enhanced conversions from sp_core errors (used by crabtensor for keys)
impl From<subxt::ext::sp_core::crypto::SecretStringError> for BittensorError {
    fn from(err: subxt::ext::sp_core::crypto::SecretStringError) -> Self {
        BittensorError::KeyDerivationError {
            message: format!("Key derivation failed: {err}"),
        }
    }
}

// Remove duplicate - already implemented above

impl BittensorError {
    /// Gets the error category for retry logic
    pub fn category(&self) -> ErrorCategory {
        match self {
            // Transient errors - can be retried
            BittensorError::RpcConnectionError { .. }
            | BittensorError::RpcTimeoutError { .. }
            | BittensorError::TxTimeoutError { .. }
            | BittensorError::WebsocketError { .. }
            | BittensorError::ChainSyncError { .. }
            | BittensorError::ServiceUnavailable { .. }
            | BittensorError::OperationTimeout { .. }
            | BittensorError::TxDroppedError { .. } => ErrorCategory::Transient,

            // Network errors - retryable with longer backoff
            BittensorError::NetworkConnectivityError { .. }
            | BittensorError::NetworkError { .. } => ErrorCategory::Network,

            // Rate limiting errors
            BittensorError::RateLimitExceeded { .. } => ErrorCategory::RateLimit,

            // Authentication errors - may be retryable
            BittensorError::SignatureError { .. }
            | BittensorError::AuthError { .. }
            | BittensorError::HotkeyNotRegistered { .. } => ErrorCategory::Auth,

            // Configuration errors - not retryable
            BittensorError::ConfigError { .. }
            | BittensorError::InvalidHotkey { .. }
            | BittensorError::InvalidWeights { .. }
            | BittensorError::InvalidNonce { .. }
            | BittensorError::RuntimeVersionMismatch { .. }
            | BittensorError::SerializationError { .. } => ErrorCategory::Config,

            // Permanent errors - not retryable
            BittensorError::NeuronNotFound { .. }
            | BittensorError::SubnetNotFound { .. }
            | BittensorError::InsufficientStake { .. }
            | BittensorError::InsufficientTxFees { .. }
            | BittensorError::InsufficientBalance { .. }
            | BittensorError::NonRetryable { .. }
            | BittensorError::MaxRetriesExceeded { .. }
            | BittensorError::BackoffTimeoutReached { .. }
            | BittensorError::BlockNotFound { .. }
            | BittensorError::InvalidBlockNumber { .. } => ErrorCategory::Permanent,

            // Legacy errors - categorize based on content
            BittensorError::RpcError { message }
            | BittensorError::ChainError { message }
            | BittensorError::TimeoutError { message } => {
                if message.to_lowercase().contains("timeout")
                    || message.to_lowercase().contains("connection")
                {
                    ErrorCategory::Transient
                } else {
                    ErrorCategory::Permanent
                }
            }

            BittensorError::WalletError { message } => {
                if message.to_lowercase().contains("loading")
                    || message.to_lowercase().contains("file")
                {
                    ErrorCategory::Config
                } else {
                    ErrorCategory::Auth
                }
            }

            // Default categorization for remaining errors
            BittensorError::TxSubmissionError { .. }
            | BittensorError::TxFinalizationError { .. }
            | BittensorError::RpcMethodError { .. }
            | BittensorError::MetadataError { .. }
            | BittensorError::StorageQueryError { .. }
            | BittensorError::WalletLoadingError { .. }
            | BittensorError::KeyDerivationError { .. }
            | BittensorError::WeightSettingFailed { .. }
            | BittensorError::RegistrationFailed { .. } => ErrorCategory::Transient,
        }
    }

    /// Gets the appropriate retry configuration for this error
    pub fn retry_config(&self) -> Option<RetryConfig> {
        match self.category() {
            ErrorCategory::Transient => Some(RetryConfig::transient()),
            ErrorCategory::RateLimit => Some(RetryConfig::rate_limit()),
            ErrorCategory::Network => Some(RetryConfig::network()),
            ErrorCategory::Auth => Some(RetryConfig::auth()),
            ErrorCategory::Config | ErrorCategory::Permanent => None,
        }
    }

    /// Checks if this error is retryable
    pub fn is_retryable(&self) -> bool {
        !matches!(
            self.category(),
            ErrorCategory::Config | ErrorCategory::Permanent
        )
    }

    /// Creates a retry exhausted error
    pub fn max_retries_exceeded(attempts: u32) -> Self {
        BittensorError::MaxRetriesExceeded { attempts }
    }

    /// Creates a backoff timeout error
    pub fn backoff_timeout(duration: Duration) -> Self {
        BittensorError::BackoffTimeoutReached { duration }
    }
}
