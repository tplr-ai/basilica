//! # Protocol
//!
//! gRPC protocol definitions and message types for Basilca communication.
//! This crate provides typed interfaces for all inter-service communication.
//!
//! ## Services
//!
//! ### MinerDiscovery
//! Service for Validator ↔ Miner coordination. Allows validators to:
//! - Authenticate with miners using Bittensor signatures
//! - Discover available nodes with resource information
//! - Get node access credentials
//!
//! ### ValidatorExternalApi
//! Service for external → Validator communication. Allows external services to:
//! - List available capacity
//! - Rent GPU capacity with container specifications
//! - Manage rentals
//! - Stream logs
//!
//! ## Usage
//!
//! The protocol crate provides generated gRPC service definitions and message types.
//! Generated code is emitted to `OUT_DIR` at build time and included here via
//! `tonic::include_proto!`.
//!
//! ### Client Example
//!
//! ```rust,ignore
//! use basilica_protocol::miner_discovery::miner_discovery_client::MinerDiscoveryClient;
//! use basilica_protocol::miner_discovery::DiscoverNodesRequest;
//! use tonic::Request;
//!
//! let mut client = MinerDiscoveryClient::connect("http://[::1]:50051").await?;
//! let request = Request::new(DiscoverNodesRequest {
//!     validator_hotkey: "validator-key".to_string(),
//!     signature: "signature".to_string(),
//!     validator_public_key: "ssh-rsa ...".to_string(),
//!     timestamp: Some(current_timestamp()),
//!     target_miner_hotkey: "miner-key".to_string(),
//! });
//! let response = client.discover_nodes(request).await?;
//! ```

// Create proper module hierarchy for generated protobuf code.
// `tonic::include_proto!("foo.bar.v1")` expands to
// `include!(concat!(env!("OUT_DIR"), "/foo.bar.v1.rs"))` so the generated
// files live in the cargo OUT_DIR rather than the source tree.
pub mod basilca {
    pub mod common {
        pub mod v1 {
            tonic::include_proto!("basilca.common.v1");
        }
    }

    pub mod rental {
        pub mod v1 {
            tonic::include_proto!("basilica.rental.v1");
        }
    }

    pub mod miner {
        pub mod v1 {
            tonic::include_proto!("basilca.miner.v1");
        }
    }

    pub mod validator {
        pub mod v1 {
            tonic::include_proto!("basilca.validator.v1");
        }
    }

    pub mod billing {
        pub mod v1 {
            tonic::include_proto!("basilica.billing.v1");
        }
    }
}

// Structured re-exports for better organization
pub mod common {
    //! Common types and data structures used across all services
    pub use crate::basilca::common::v1::*;
}

pub mod miner_discovery {
    //! Miner discovery service for Validator ↔ Miner coordination
    //!
    //! This service supports steps 3-4 of the interaction flow:
    //! - Bittensor signature-based validator authentication
    //! - Node discovery and access
    pub use crate::basilca::miner::v1::*;
}

pub mod rental {
    //! Rental service for managing GPU rentals
    pub use crate::basilca::rental::v1::*;
}

pub mod validator_api {
    //! External API service for services to interact with validators
    //!
    //! Public interface for capacity rental:
    //! - Discover available GPU capacity across the network
    //! - Rent GPU capacity with container specifications
    //! - Manage rental lifecycle (terminate, status, logs)
    pub use crate::basilca::validator::v1::*;
}

pub mod billing {
    //! Billing service for credit management and rental tracking
    //!
    //! Provides comprehensive billing functionality:
    //! - Credit balance management and reservations
    //! - Rental lifecycle tracking with usage metrics
    //! - Real-time telemetry ingestion and aggregation
    //! - Metered per-token usage record ingestion with per-record settlement results
    //! - Billing packages and rules engine
    pub use crate::basilca::billing::v1::*;
}

// Re-export common types at crate root for convenience
pub use basilica_common::*;

// Utility functions for working with protocol types
pub mod utils {
    use super::common::*;

    /// Convert NodeId to string representation for protobuf
    pub fn node_id_to_string(id: &str) -> String {
        id.to_string()
    }

    /// Convert string to NodeId (with validation)
    pub fn string_to_node_id(s: &str) -> Result<String, String> {
        if s.is_empty() {
            Err("Node ID cannot be empty".to_string())
        } else {
            Ok(s.to_string())
        }
    }

    /// Validate GPU specification
    pub fn validate_gpu_spec(gpu: &GpuSpec) -> Result<(), String> {
        if gpu.model.is_empty() {
            return Err("GPU model cannot be empty".to_string());
        }
        if gpu.memory_mb == 0 {
            return Err("GPU memory must be greater than 0".to_string());
        }
        Ok(())
    }

    /// Validate container specification
    pub fn validate_container_spec(spec: &ContainerSpec) -> Result<(), String> {
        if spec.image.is_empty() {
            return Err("Container image cannot be empty".to_string());
        }
        Ok(())
    }

    /// Create a timestamp from current time
    pub fn current_timestamp() -> Timestamp {
        Timestamp {
            value: Some(prost_types::Timestamp::from(std::time::SystemTime::now())),
        }
    }

    /// Convert ResourceUsageStats to human-readable format
    pub fn format_resource_usage(stats: &ResourceUsageStats) -> String {
        format!(
            "CPU: {:.1}%, Memory: {} MB, GPU: {:?}%",
            stats.cpu_percent, stats.memory_mb, stats.gpu_utilization
        )
    }
}

// Error handling for protocol operations
pub mod errors {
    use thiserror::Error;

    /// Protocol-specific errors
    #[derive(Error, Debug)]
    pub enum ProtocolError {
        #[error("gRPC communication failed: {0}")]
        GrpcError(#[from] tonic::Status),

        #[error("Invalid message format: {0}")]
        InvalidMessage(String),

        #[error("Authentication failed: {0}")]
        AuthenticationFailed(String),

        #[error("Protocol version mismatch: expected {expected}, got {actual}")]
        VersionMismatch { expected: String, actual: String },

        #[error("Validation error: {0}")]
        ValidationError(String),

        #[error("Resource not found: {0}")]
        ResourceNotFound(String),

        #[error("Resource conflict: {0}")]
        ResourceConflict(String),

        #[error("Timeout error: {0}")]
        Timeout(String),
    }

    impl From<ProtocolError> for tonic::Status {
        fn from(err: ProtocolError) -> Self {
            match err {
                ProtocolError::GrpcError(status) => status,
                ProtocolError::InvalidMessage(msg) => tonic::Status::invalid_argument(msg),
                ProtocolError::AuthenticationFailed(msg) => tonic::Status::unauthenticated(msg),
                ProtocolError::ValidationError(msg) => tonic::Status::invalid_argument(msg),
                ProtocolError::ResourceNotFound(msg) => tonic::Status::not_found(msg),
                ProtocolError::ResourceConflict(msg) => tonic::Status::already_exists(msg),
                ProtocolError::Timeout(msg) => tonic::Status::deadline_exceeded(msg),
                _ => tonic::Status::internal(err.to_string()),
            }
        }
    }
}

/// Version of the protocol definitions
pub const PROTOCOL_VERSION: &str = "1.0.0";

// Implementation notes for generated types:
// The protobuf-generated types may have different field names than expected.
// Generated code lives in cargo's OUT_DIR; inspect with `cargo expand` or by
// running `cargo build` and reading `target/.../build/.../out/*.rs`.
// Key differences from common expectations:
// - HealthCheckResponse uses 'status' as an i32 enum value, not a struct field
// - SystemProfileResponse uses encrypted_profile instead of direct machine_info
// - Some message types may be nested under different modules than expected

// Implementation notes for generated types:
// The protobuf-generated types may have different field names than expected.
// Generated code lives in cargo's OUT_DIR; inspect with `cargo expand` or by
// running `cargo build` and reading `target/.../build/.../out/*.rs`.
// Key differences from common expectations:
// - HealthCheckResponse uses 'status' as an i32 enum value, not a struct field
// - SystemProfileResponse uses encrypted_profile instead of direct machine_info
// - Some message types may be nested under different modules than expected

// Helper types for common gRPC patterns
pub mod helpers {
    use tonic::{Request, Status};
    use tracing::instrument;

    /// Extract metadata value from gRPC request
    #[allow(clippy::result_large_err)]
    pub fn extract_metadata(
        request: &Request<impl std::fmt::Debug>,
        key: &str,
    ) -> Result<String, Status> {
        request
            .metadata()
            .get(key)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string())
            .ok_or_else(|| Status::invalid_argument(format!("Missing required metadata: {key}")))
    }

    /// Verify protocol version compatibility
    #[allow(clippy::result_large_err)]
    pub fn verify_protocol_version(client_version: &str) -> Result<(), Status> {
        if client_version != crate::PROTOCOL_VERSION {
            return Err(Status::failed_precondition(format!(
                "Protocol version mismatch: client={}, server={}",
                client_version,
                crate::PROTOCOL_VERSION
            )));
        }
        Ok(())
    }

    /// Helper for creating authenticated requests
    #[instrument(skip(request))]
    pub fn add_auth_metadata<T>(
        mut request: Request<T>,
        hotkey: &str,
        signature: &str,
    ) -> Request<T> {
        request
            .metadata_mut()
            .insert("x-hotkey", hotkey.parse().expect("Invalid hotkey format"));
        request.metadata_mut().insert(
            "x-signature",
            signature.parse().expect("Invalid signature format"),
        );
        request
    }

    /// Create a basic gRPC TLS config using system roots
    pub fn create_tls_config(
    ) -> Result<tonic::transport::ClientTlsConfig, Box<dyn std::error::Error>> {
        Ok(tonic::transport::ClientTlsConfig::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_utility_functions() {
        let id = "test-node-123";
        let id_str = utils::node_id_to_string(id);
        let parsed_id = utils::string_to_node_id(&id_str).unwrap();

        assert_eq!(id, parsed_id);
    }

    #[test]
    fn test_gpu_spec_validation() {
        let mut gpu = common::GpuSpec {
            model: "RTX 4090".to_string(),
            memory_mb: 24000,
            uuid: "GPU-12345".to_string(),
            driver_version: "535.86.05".to_string(),
            cuda_version: "12.2".to_string(),
            utilization_percent: 0.0,
            memory_utilization_percent: 0.0,
            temperature_celsius: 45.0,
            power_watts: 350.0,
            core_clock_mhz: 2205,
            memory_clock_mhz: 10501,
            compute_capability: "8.9".to_string(),
        };

        assert!(utils::validate_gpu_spec(&gpu).is_ok());

        gpu.model = String::new();
        assert!(utils::validate_gpu_spec(&gpu).is_err());
    }

    #[test]
    fn test_container_spec_validation() {
        let mut spec = common::ContainerSpec {
            image: "nvidia/cuda:12.2-runtime-ubuntu20.04".to_string(),
            environment: std::collections::HashMap::new(),
            port_mappings: std::collections::HashMap::new(),
            volume_mounts: std::collections::HashMap::new(),
            resource_limits: None,
            command: vec![],
            working_directory: "/app".to_string(),
            user: "root".to_string(),
            gpu_requirements: vec!["nvidia".to_string()],
            network_mode: "bridge".to_string(),
        };

        assert!(utils::validate_container_spec(&spec).is_ok());

        spec.image = String::new();
        assert!(utils::validate_container_spec(&spec).is_err());
    }

    #[test]
    fn test_error_conversion() {
        let proto_err =
            errors::ProtocolError::AuthenticationFailed("Invalid signature".to_string());
        let status: tonic::Status = proto_err.into();

        assert_eq!(status.code(), tonic::Code::Unauthenticated);
        assert!(status.message().contains("Invalid signature"));
    }

    #[test]
    fn test_timestamp_creation() {
        let ts = utils::current_timestamp();
        assert!(ts.value.is_some());
    }

    #[test]
    fn test_helpers_extract_metadata() {
        use tonic::Request;
        let mut request = Request::new(());
        request
            .metadata_mut()
            .insert("test-key", "test-value".parse().unwrap());

        let result = helpers::extract_metadata(&request, "test-key");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "test-value");

        let missing = helpers::extract_metadata(&request, "missing-key");
        assert!(missing.is_err());
    }

    #[test]
    fn test_helpers_verify_protocol_version() {
        let result = helpers::verify_protocol_version(PROTOCOL_VERSION);
        assert!(result.is_ok());

        let mismatch = helpers::verify_protocol_version("0.9.0");
        assert!(mismatch.is_err());
    }

    #[test]
    fn test_helpers_add_auth_metadata() {
        use tonic::Request;
        let request = Request::new(());
        let authed = helpers::add_auth_metadata(request, "test-hotkey", "test-signature");

        assert_eq!(
            authed.metadata().get("x-hotkey").unwrap().to_str().unwrap(),
            "test-hotkey"
        );
        assert_eq!(
            authed
                .metadata()
                .get("x-signature")
                .unwrap()
                .to_str()
                .unwrap(),
            "test-signature"
        );
    }

    #[test]
    fn test_usage_record_proto_round_trip() {
        use prost::Message;

        let record = billing::UsageRecord {
            request_id: "018e4c2f-9f2a-7b3c-8d4e-5f6a7b8c9d0e".to_string(),
            tenant_id: "auth0|user_12345".to_string(),
            model: "llama-3.1-70b-instruct".to_string(),
            prompt_tokens: 1024,
            completion_tokens: 256,
            cached_tokens: 512,
            fanout: 3,
            price: Some(billing::TokenPrice {
                input_per_million: "0.15000000".to_string(),
                output_per_million: "0.60000000".to_string(),
                cached_input_per_million: "0.01500000".to_string(),
            }),
        };

        let bytes = record.encode_to_vec();
        let decoded = billing::UsageRecord::decode(&bytes[..]).expect("decode UsageRecord");

        assert_eq!(decoded, record);
        assert_eq!(decoded.request_id, "018e4c2f-9f2a-7b3c-8d4e-5f6a7b8c9d0e");
        assert_eq!(decoded.tenant_id, "auth0|user_12345");
        assert_eq!(decoded.model, "llama-3.1-70b-instruct");
        assert_eq!(decoded.prompt_tokens, 1024);
        assert_eq!(decoded.completion_tokens, 256);
        assert_eq!(decoded.cached_tokens, 512);
        assert_eq!(decoded.fanout, 3);
        let price = decoded.price.expect("price snapshot survives round trip");
        assert_eq!(price.input_per_million, "0.15000000");
        assert_eq!(price.output_per_million, "0.60000000");
        assert_eq!(price.cached_input_per_million, "0.01500000");
    }

    #[test]
    fn test_ingest_usage_records_response_proto_round_trip() {
        use prost::Message;

        use billing::record_result::Status;

        let response = billing::IngestUsageRecordsResponse {
            results: vec![
                billing::RecordResult {
                    request_id: "req-settled".to_string(),
                    status: Status::Settled as i32,
                    reason: String::new(),
                    remainder_credits: String::new(),
                },
                billing::RecordResult {
                    request_id: "req-arrears".to_string(),
                    status: Status::Arrears as i32,
                    reason: "balance exhausted; remainder booked to arrears".to_string(),
                    remainder_credits: "0.75".to_string(),
                },
                billing::RecordResult {
                    request_id: "req-retry".to_string(),
                    status: Status::Retry as i32,
                    reason: "transient storage error".to_string(),
                    remainder_credits: String::new(),
                },
                billing::RecordResult {
                    request_id: "req-rejected".to_string(),
                    status: Status::Rejected as i32,
                    reason: "token counts outside [0, registry_cap]".to_string(),
                    remainder_credits: String::new(),
                },
            ],
        };

        let bytes = response.encode_to_vec();
        let decoded =
            billing::IngestUsageRecordsResponse::decode(&bytes[..]).expect("decode response");

        assert_eq!(decoded, response);
        assert_eq!(decoded.results.len(), 4);
        assert_eq!(decoded.results[0].request_id, "req-settled");
        assert_eq!(decoded.results[0].status(), Status::Settled);
        assert_eq!(decoded.results[0].reason, "");
        // SETTLED carries no remainder: empty stays empty across the wire.
        assert_eq!(decoded.results[0].remainder_credits, "");
        assert_eq!(decoded.results[1].request_id, "req-arrears");
        assert_eq!(decoded.results[1].status(), Status::Arrears);
        assert_eq!(
            decoded.results[1].reason,
            "balance exhausted; remainder booked to arrears"
        );
        // ARREARS: the structured remainder survives encode/decode exactly.
        assert_eq!(decoded.results[1].remainder_credits, "0.75");
        assert_eq!(decoded.results[2].status(), Status::Retry);
        assert_eq!(decoded.results[3].status(), Status::Rejected);
        assert_eq!(
            decoded.results[3].reason,
            "token counts outside [0, registry_cap]"
        );
    }

    #[test]
    fn test_billing_service_stubs_include_ingest_usage_records() {
        use billing::billing_service_client::BillingServiceClient;
        use billing::billing_service_server::BillingService;
        use billing::record_result::Status;

        // Compile-time assertions: the bodies of these never-called functions are
        // still type-checked, so the build fails if the generated stubs lose the RPC.

        // Server: the generated BillingService trait must declare ingest_usage_records.
        #[allow(dead_code)]
        fn assert_server_trait_method<T: BillingService>() {
            let _ = T::ingest_usage_records;
        }

        // Client: the generated BillingServiceClient must expose ingest_usage_records.
        #[allow(dead_code)]
        fn assert_client_method(client: &mut BillingServiceClient<tonic::transport::Channel>) {
            std::mem::drop(client.ingest_usage_records(tonic::Request::new(
                billing::IngestUsageRecordsRequest { records: vec![] },
            )));
        }

        // Enum wire values are pinned by the proto contract.
        assert_eq!(Status::Settled as i32, 0);
        assert_eq!(Status::Arrears as i32, 1);
        assert_eq!(Status::Retry as i32, 2);
        assert_eq!(Status::Rejected as i32, 3);
    }
}
