//! # Validator Communications
//!
//! gRPC server for handling validator requests with direct node access.
//! Primary responsibilities:
//! - Authenticate validators using Bittensor signatures
//! - Provide node connection details to authorized validators

use anyhow::{Context, Result};
use rand::Rng;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::RwLock;

use tonic::{transport::Server, Request, Response, Status};
use tonic_health::server::health_reporter;
use tracing::{debug, error, info, warn};

use basilica_common::identity::Hotkey;
use basilica_protocol::miner_discovery::{
    miner_discovery_server::{MinerDiscovery, MinerDiscoveryServer},
    DiscoverNodesRequest, ListNodeConnectionDetailsResponse, MinerAuthResponse,
    ValidatorAuthRequest,
};

use crate::config::{SecurityConfig, ValidatorCommsConfig};
use crate::node_manager::NodeManager;
use crate::validator_discovery::ValidatorDiscovery;

/// Validator communications server
#[derive(Clone)]
pub struct ValidatorCommsServer {
    config: ValidatorCommsConfig,
    security_config: SecurityConfig,
    node_manager: Arc<NodeManager>,
    validator_discovery: Arc<ValidatorDiscovery>,
    authenticated_validators: Arc<RwLock<HashMap<String, String>>>,
    bittensor_service: Arc<bittensor::Service>,
}

impl ValidatorCommsServer {
    /// Create new validator communications server
    #[allow(clippy::too_many_arguments)]
    pub async fn new(
        config: ValidatorCommsConfig,
        security_config: SecurityConfig,
        node_manager: Arc<NodeManager>,
        validator_discovery: Arc<ValidatorDiscovery>,
        bittensor_service: Arc<bittensor::Service>,
    ) -> Result<Self> {
        Ok(Self {
            config,
            security_config,
            node_manager,
            validator_discovery,
            authenticated_validators: Arc::new(RwLock::new(HashMap::new())),
            bittensor_service,
        })
    }

    /// Start the gRPC server
    pub async fn start(&self) -> Result<()> {
        let addr: SocketAddr = format!("{}:{}", self.config.host, self.config.port)
            .parse()
            .context("Failed to parse server address")?;

        let (mut health_reporter, health_service) = health_reporter();
        health_reporter
            .set_serving::<MinerDiscoveryServer<MinerDiscoveryService>>()
            .await;

        // Create the discovery service
        let discovery_service = MinerDiscoveryService {
            server: self.clone(),
            bittensor_service: self.bittensor_service.clone(),
        };

        info!("Starting validator communications server on {}", addr);

        Server::builder()
            .add_service(health_service)
            .add_service(MinerDiscoveryServer::new(discovery_service))
            .serve(addr)
            .await
            .context("Failed to start gRPC server")?;

        Ok(())
    }

    /// Get the gRPC server address
    pub fn address(&self) -> String {
        format!("{}:{}", self.config.host, self.config.port)
    }

    /// Check if a validator is authorized
    async fn is_validator_authorized(&self, validator_hotkey: &str) -> bool {
        match self.validator_discovery.get_active_validators().await {
            Ok(validators) => validators.iter().any(|v| v.hotkey == validator_hotkey),
            Err(e) => {
                error!(error = %e, "Failed to fetch active validators");
                false
            }
        }
    }
}

/// Generate a secure session token
fn generate_session_token() -> String {
    const TOKEN_LENGTH: usize = 32;
    let mut rng = rand::thread_rng();
    let token: Vec<u8> = (0..TOKEN_LENGTH).map(|_| rng.gen()).collect();
    hex::encode(token)
}

/// gRPC service implementation for MinerDiscovery
#[derive(Clone)]
pub struct MinerDiscoveryService {
    server: ValidatorCommsServer,
    bittensor_service: Arc<bittensor::Service>,
}

#[tonic::async_trait]
impl MinerDiscovery for MinerDiscoveryService {
    /// Authenticate a validator using Bittensor signature
    async fn authenticate_validator(
        &self,
        request: Request<ValidatorAuthRequest>,
    ) -> Result<Response<MinerAuthResponse>, Status> {
        let auth_request = request.into_inner();

        if auth_request.target_miner_hotkey.trim().is_empty() {
            return Err(Status::invalid_argument("target_miner_hotkey is required"));
        }

        debug!(
            "Received authentication request from validator: {} for target miner: {}",
            auth_request.validator_hotkey, auth_request.target_miner_hotkey
        );

        // Verify target miner hotkey matches ours
        let our_hotkey = self.bittensor_service.get_account_id().to_string();
        if auth_request.target_miner_hotkey != our_hotkey {
            warn!(
                "Authentication request intended for different miner. Target: {}, Our hotkey: {}",
                auth_request.target_miner_hotkey, our_hotkey
            );
            return Err(Status::permission_denied(
                "Authentication request not intended for this miner",
            ));
        }
        debug!("Target miner hotkey matches our hotkey");

        // Verify the signature if enabled
        let validator_hotkey = Hotkey::new(auth_request.validator_hotkey.clone())
            .map_err(|e| Status::invalid_argument(format!("Invalid hotkey: {e}")))?;

        if self.server.security_config.verify_signatures {
            // Extract timestamp if provided
            let timestamp_secs = auth_request
                .timestamp
                .as_ref()
                .and_then(|t| t.value.as_ref())
                .map(|pt| pt.seconds)
                .unwrap_or(0);

            // Canonical payload with prefix and timestamp
            const AUTH_PREFIX: &str = "BASILICA_AUTH_V1";
            let canonical_payload = format!(
                "{}:{}:{}:{}",
                AUTH_PREFIX, auth_request.nonce, auth_request.target_miner_hotkey, timestamp_secs
            );

            // Verify signature
            use basilica_common::crypto::verify_bittensor_signature;
            match verify_bittensor_signature(
                &validator_hotkey,
                &auth_request.signature,
                canonical_payload.as_bytes(),
            ) {
                Ok(()) => {
                    debug!("Signature verification successful");
                }
                Err(e) => {
                    warn!("Signature verification failed for validator: {}", e);
                    return Err(Status::unauthenticated("Invalid signature"));
                }
            }

            // Check timestamp freshness (5 minutes)
            if timestamp_secs > 0 {
                let current_time = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs() as i64;
                let request_time = timestamp_secs;
                let time_diff = (current_time - request_time).abs();

                if time_diff > 300 {
                    warn!(
                        "Authentication request timestamp too old: {} seconds",
                        time_diff
                    );
                    return Err(Status::unauthenticated("Request timestamp too old"));
                }
            }
        }

        // Check if validator is authorized
        if !self
            .server
            .is_validator_authorized(&auth_request.validator_hotkey)
            .await
        {
            warn!(
                "Validator {} is not authorized",
                auth_request.validator_hotkey
            );
            return Err(Status::permission_denied("Validator not authorized"));
        }

        // Store authenticated validator
        let mut validators = self.server.authenticated_validators.write().await;
        validators.insert(
            auth_request.validator_hotkey.clone(),
            auth_request.nonce.clone(),
        );

        info!(
            "Successfully authenticated validator: {}",
            auth_request.validator_hotkey
        );

        // Generate session token for validator
        let session_token = generate_session_token();

        // Sign the response with miner's hotkey
        // Generate a fresh nonce for the response (security best practice)
        let response_nonce = uuid::Uuid::new_v4().to_string();
        let miner_hotkey = self.bittensor_service.get_account_id().to_string();

        // Create canonical response payload for signing
        let canonical_response = format!(
            "MINER_AUTH_RESPONSE:{}:{}:{}",
            auth_request.validator_hotkey, response_nonce, session_token
        );

        // Sign with miner's hotkey
        let (miner_hotkey, miner_signature, response_nonce) = match self
            .bittensor_service
            .sign_data(canonical_response.as_bytes())
        {
            Ok(sig) => (miner_hotkey, sig, response_nonce),
            Err(e) => {
                warn!("Failed to sign response: {}", e);
                (String::new(), String::new(), String::new())
            }
        };

        Ok(Response::new(MinerAuthResponse {
            authenticated: true,
            session_token,
            expires_at: Some(basilica_protocol::basilca::common::v1::Timestamp {
                value: Some(prost_types::Timestamp {
                    seconds: (std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs()
                        + 3600) as i64,
                    nanos: 0,
                }),
            }),
            error: None,
            miner_hotkey,
            miner_signature,
            response_nonce,
        }))
    }

    /// Discover available nodes for validator
    async fn discover_nodes(
        &self,
        request: Request<DiscoverNodesRequest>,
    ) -> Result<Response<ListNodeConnectionDetailsResponse>, Status> {
        let discover_request = request.into_inner();

        // Verify the validator is authenticated
        let validators = self.server.authenticated_validators.read().await;
        if !validators.contains_key(&discover_request.validator_hotkey) {
            return Err(Status::unauthenticated(
                "Validator must authenticate before discovering nodes",
            ));
        }

        // Verify the validator is providing an SSH public key
        if discover_request.validator_public_key.is_empty() {
            return Err(Status::invalid_argument(
                "Validator must provide SSH public key",
            ));
        }

        debug!(
            "Validator {} discovering nodes with SSH key",
            discover_request.validator_hotkey
        );

        // Get the currently assigned validator from NodeManager (single source of truth)
        let assigned_validator = self.server.node_manager.get_assigned_validator().await;

        // Check if the requesting validator is the assigned validator
        if assigned_validator.as_deref() != Some(&discover_request.validator_hotkey) {
            info!(
                "Validator {} is not the assigned validator; returning empty node list",
                discover_request.validator_hotkey
            );
            return Ok(Response::new(ListNodeConnectionDetailsResponse {
                nodes: vec![],
            }));
        }

        // Deploy SSH keys to all nodes (exclusive access, removes old validator keys)
        if let Err(e) = self
            .server
            .node_manager
            .deploy_validator_keys(
                &discover_request.validator_hotkey,
                &discover_request.validator_public_key,
            )
            .await
        {
            error!("Failed to deploy validator SSH keys: {}", e);
            return Err(Status::internal(format!("Failed to deploy SSH keys: {e}")));
        }

        // Get all nodes and return them to the validator
        let nodes = match self.server.node_manager.list_nodes().await {
            Ok(nodes) => nodes,
            Err(e) => {
                error!("Failed to list nodes: {}", e);
                return Err(Status::internal(format!("Failed to list nodes: {e}")));
            }
        };

        // Convert to protocol format
        let node_details: Vec<basilica_protocol::miner_discovery::NodeConnectionDetails> = nodes
            .into_iter()
            .map(|registered_node| {
                basilica_protocol::miner_discovery::NodeConnectionDetails {
                    node_id: registered_node.node_id,
                    host: registered_node.config.host,
                    port: registered_node.config.port.to_string(),
                    username: registered_node.config.username,
                    additional_opts: registered_node.config.additional_opts.unwrap_or_default(),
                    gpu_spec: None, // Validators discover GPU specs via SSH
                    status: "available".to_string(),
                }
            })
            .collect();

        info!(
            "Returning {} nodes to validator {}",
            node_details.len(),
            discover_request.validator_hotkey
        );

        Ok(Response::new(ListNodeConnectionDetailsResponse {
            nodes: node_details,
        }))
    }
}
