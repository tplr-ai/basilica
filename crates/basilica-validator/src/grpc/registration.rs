//! MinerRegistration gRPC service implementation.
//!
//! This service handles the miner→validator registration flow:
//! - RegisterBid: One-time registration of nodes with SSH details + pricing
//! - UpdateBid: Update bid price for existing registered node
//! - RemoveBid: Explicitly remove node(s) from availability
//! - HealthCheck: Lightweight periodic heartbeat to keep registrations active

use std::sync::Arc;

use anyhow::Result;
use basilica_common::config::GrpcServerConfig;
use basilica_common::crypto::verify_signature_bittensor;
use basilica_common::identity::Hotkey;
use basilica_common::types::GpuCategory;
use basilica_protocol::miner_discovery::{
    miner_registration_server::{MinerRegistration, MinerRegistrationServer},
    HealthCheckRequest, HealthCheckResponse, RegisterBidRequest, RegisterBidResponse,
    RemoveBidRequest, RemoveBidResponse, UpdateBidRequest, UpdateBidResponse,
};
use chrono::{TimeZone, Utc};
use tonic::{transport::Server, Request, Response, Status};
use tonic_health::server::health_reporter;
use tracing::{info, warn};
use uuid::Uuid;

use crate::basilica_api::BasilicaApiClient;
use crate::collateral::{CollateralManager, CollateralState, CollateralStatus};
use crate::config::bidding::BiddingConfig;
use crate::persistence::SimplePersistence;

/// Convert internal CollateralStatus to proto CollateralStatus
fn status_to_proto(
    status: CollateralStatus,
) -> basilica_protocol::miner_discovery::CollateralStatus {
    basilica_protocol::miner_discovery::CollateralStatus {
        current_alpha: status.current_alpha.to_f64().unwrap_or_default(),
        current_usd_value: status.current_usd_value.to_f64().unwrap_or_default(),
        minimum_usd_required: status.minimum_usd_required.to_f64().unwrap_or_default(),
        status: status.status,
        grace_period_remaining: status
            .grace_period_remaining
            .map(format_duration)
            .unwrap_or_default(),
        action_required: status.action_required.unwrap_or_default(),
        alpha_usd_price: status
            .alpha_usd_price
            .and_then(|price| price.to_f64())
            .unwrap_or_default(),
        price_stale: status.price_stale,
    }
}

fn format_duration(duration: chrono::Duration) -> String {
    let total_seconds = duration.num_seconds().max(0);
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    if hours > 0 {
        format!("{}h {}m", hours, minutes)
    } else {
        format!("{}m", minutes)
    }
}

fn status_rank(state: &CollateralState) -> u8 {
    match state {
        CollateralState::Excluded { .. } => 4,
        CollateralState::Undercollateralized { .. } => 3,
        CollateralState::Warning { .. } => 2,
        CollateralState::Sufficient { .. } => 1,
        CollateralState::Unknown { .. } => 0,
    }
}

fn status_rank_from_status(status: &str) -> u8 {
    match status {
        "excluded" => 4,
        "undercollateralized" => 3,
        "warning" => 2,
        "sufficient" => 1,
        _ => 0,
    }
}

fn select_worst_status(
    current: Option<CollateralStatus>,
    state: CollateralState,
    next: CollateralStatus,
) -> CollateralStatus {
    if let Some(existing) = current {
        let existing_rank = status_rank_from_status(&existing.status);
        let next_rank = status_rank(&state);
        if next_rank > existing_rank {
            next
        } else {
            existing
        }
    } else {
        next
    }
}

use rust_decimal::prelude::ToPrimitive;

#[derive(Clone)]
pub struct RegistrationService {
    persistence: Arc<SimplePersistence>,
    bidding_config: BiddingConfig,
    collateral_manager: Option<Arc<CollateralManager>>,
    validator_ssh_public_key: String,
    api_client: Option<Arc<BasilicaApiClient>>,
}

#[allow(clippy::result_large_err)]
impl RegistrationService {
    pub fn new(
        persistence: Arc<SimplePersistence>,
        bidding_config: BiddingConfig,
        collateral_manager: Option<Arc<CollateralManager>>,
        validator_ssh_public_key: String,
        api_client: Option<Arc<BasilicaApiClient>>,
    ) -> Self {
        Self {
            persistence,
            bidding_config,
            collateral_manager,
            validator_ssh_public_key,
            api_client,
        }
    }

    /// Verify timestamp is within allowed window
    fn ensure_timestamp_freshness(&self, timestamp: i64) -> Result<(), Status> {
        let submitted_at = self.parse_timestamp(timestamp)?;
        let now = Utc::now();
        let max_skew_secs = self.bidding_config.rpc_timestamp_tolerance_secs as i64;
        if (now - submitted_at).num_seconds().abs() > max_skew_secs {
            return Err(Status::invalid_argument("timestamp outside allowed window"));
        }
        Ok(())
    }

    fn parse_timestamp(&self, timestamp: i64) -> Result<chrono::DateTime<Utc>, Status> {
        if timestamp <= 0 {
            return Err(Status::invalid_argument("timestamp must be positive"));
        }
        let dt = if timestamp > 1_000_000_000_000 {
            let secs = timestamp / 1000;
            Utc.timestamp_opt(secs, 0)
        } else {
            Utc.timestamp_opt(timestamp, 0)
        }
        .single()
        .ok_or_else(|| Status::invalid_argument("invalid timestamp"))?;
        Ok(dt)
    }

    /// Resolve miner hotkey to miner_id (miner_UID format)
    async fn resolve_miner_id(&self, miner_hotkey: &str) -> Result<String, Status> {
        let miner_id = self
            .persistence
            .check_miner_by_hotkey(miner_hotkey)
            .await
            .map_err(|e| Status::internal(format!("database error: {e}")))?
            .ok_or_else(|| Status::not_found("unknown miner_hotkey"))?;
        Ok(miner_id)
    }

    /// Build the message to verify for RegisterBid signature
    fn build_register_bid_message(&self, req: &RegisterBidRequest) -> String {
        format!("{}|{}", req.miner_hotkey.trim(), req.timestamp)
    }

    /// Build the message to verify for UpdateBid signature
    fn build_update_bid_message(&self, req: &UpdateBidRequest) -> String {
        format!(
            "{}|{}|{}|{}",
            req.miner_hotkey.trim(),
            req.host.trim(),
            req.hourly_rate_cents,
            req.timestamp,
        )
    }

    /// Build the message to verify for RemoveBid signature
    fn build_remove_bid_message(&self, req: &RemoveBidRequest) -> String {
        let hosts_str = req.hosts.join(",");
        format!(
            "{}|{}|{}",
            req.miner_hotkey.trim(),
            hosts_str,
            req.timestamp,
        )
    }

    /// Build the message to verify for HealthCheck signature
    fn build_health_check_message(&self, req: &HealthCheckRequest) -> String {
        let hosts_str = req.hosts.join(",");
        format!(
            "{}|{}|{}",
            req.miner_hotkey.trim(),
            hosts_str,
            req.timestamp
        )
    }

    /// Verify a Bittensor signature
    fn verify_signature(
        &self,
        hotkey: &str,
        signature: &[u8],
        message: &str,
    ) -> Result<(), Status> {
        let hotkey = Hotkey::new(hotkey.to_string())
            .map_err(|e| Status::invalid_argument(format!("invalid hotkey: {e}")))?;
        verify_signature_bittensor(&hotkey, signature, message.as_bytes()).map_err(|e| {
            Status::permission_denied(format!("signature verification failed: {e}"))
        })?;
        Ok(())
    }
}

#[tonic::async_trait]
impl MinerRegistration for RegistrationService {
    async fn register_bid(
        &self,
        request: Request<RegisterBidRequest>,
    ) -> Result<Response<RegisterBidResponse>, Status> {
        let req = request.into_inner();

        // Validate required fields
        if req.miner_hotkey.trim().is_empty() {
            return Err(Status::invalid_argument("miner_hotkey is required"));
        }
        if req.signature.is_empty() {
            return Err(Status::invalid_argument("signature is required"));
        }

        // Verify timestamp freshness
        self.ensure_timestamp_freshness(req.timestamp)?;

        // Verify signature
        let message = self.build_register_bid_message(&req);
        self.verify_signature(&req.miner_hotkey, &req.signature, &message)?;

        // Resolve miner ID
        let miner_id = self.resolve_miner_id(&req.miner_hotkey).await?;

        // Generate registration ID
        let registration_id = format!("reg-{}", Uuid::new_v4());

        // Fetch baseline prices for bid floor enforcement (best-effort)
        let baseline_prices = if let Some(ref api_client) = self.api_client {
            match api_client.get_baseline_prices().await {
                Ok(prices) => Some(prices),
                Err(e) => {
                    warn!(
                        miner_hotkey = %req.miner_hotkey,
                        error = %e,
                        "Failed to fetch baseline prices for bid floor check - allowing bids"
                    );
                    None
                }
            }
        } else {
            None
        };

        // Upsert each node
        let mut worst_collateral_status: Option<CollateralStatus> = None;
        let mut active_node_ids: Vec<String> = Vec::new();
        for node in &req.nodes {
            // Validate node fields
            if node.host.trim().is_empty() {
                return Err(Status::invalid_argument("host is required"));
            }
            if node.port == 0 {
                return Err(Status::invalid_argument("port must be greater than 0"));
            }
            if node.username.trim().is_empty() {
                return Err(Status::invalid_argument("username is required"));
            }
            if node.gpu_category.trim().is_empty() {
                return Err(Status::invalid_argument("gpu_category is required"));
            }
            // Validate gpu_category is a known GPU type
            let gpu_cat: GpuCategory = node.gpu_category.parse().unwrap(); // Infallible
            if matches!(&gpu_cat, GpuCategory::Other(_)) {
                return Err(Status::invalid_argument(format!(
                    "GPU type '{}' is not supported",
                    node.gpu_category
                )));
            }
            if node.gpu_count == 0 {
                return Err(Status::invalid_argument("gpu_count must be greater than 0"));
            }
            if node.hourly_rate_cents == 0 {
                return Err(Status::invalid_argument(
                    "hourly_rate_cents must be greater than 0",
                ));
            }

            // Enforce bid floor: bid must be >= min_bid_floor_fraction * baseline price
            if let Some(ref prices) = baseline_prices {
                let category_key = gpu_cat.to_string();
                if let Some(&baseline_dollars) = prices.get(&category_key) {
                    let floor_cents =
                        (baseline_dollars * 100.0 * self.bidding_config.min_bid_floor_fraction)
                            .round() as u32;
                    if floor_cents > 0 && node.hourly_rate_cents < floor_cents {
                        return Err(Status::invalid_argument(format!(
                            "hourly_rate_cents {} is below minimum floor {} ({:.0}% of baseline ${:.2}/hr) for {}",
                            node.hourly_rate_cents,
                            floor_cents,
                            self.bidding_config.min_bid_floor_fraction * 100.0,
                            baseline_dollars,
                            category_key,
                        )));
                    }
                }
            }

            // Compute node_id server-side from host (deterministic, not trusting miner)
            let node_id = basilica_common::node_identity::NodeId::new(&node.host)
                .map_err(|e| Status::internal(format!("failed to compute node_id: {e}")))?
                .uuid
                .to_string();
            active_node_ids.push(node_id.clone());

            // Upsert node (node_id computed server-side from host)
            self.persistence
                .upsert_registered_node(
                    &miner_id,
                    &node.host,
                    node.port,
                    &node.username,
                    &node.gpu_category,
                    node.gpu_count,
                    node.hourly_rate_cents,
                )
                .await
                .map_err(|e| Status::internal(format!("failed to register node: {e}")))?;

            // Get collateral status for this node
            if let Some(ref manager) = self.collateral_manager {
                let (state, status) = manager
                    .get_collateral_status(
                        &req.miner_hotkey,
                        &node_id,
                        &node.gpu_category,
                        node.gpu_count,
                    )
                    .await
                    .map_err(|e| Status::internal(format!("collateral status error: {e}")))?;
                worst_collateral_status =
                    Some(select_worst_status(worst_collateral_status, state, status));
            }
        }

        // Deactivate bids for nodes NOT in this RegisterBid request
        self.persistence
            .deactivate_missing_bids(&miner_id, &active_node_ids)
            .await
            .map_err(|e| Status::internal(format!("failed to deactivate missing bids: {e}")))?;

        info!(
            miner_hotkey = %req.miner_hotkey,
            miner_id = %miner_id,
            registration_id = %registration_id,
            node_count = req.nodes.len(),
            "Accepted miner registration via RegisterBid"
        );

        // Return validator's SSH public key for miner to deploy
        let validator_ssh_public_key = self.validator_ssh_public_key.clone();

        Ok(Response::new(RegisterBidResponse {
            accepted: true,
            registration_id,
            validator_ssh_public_key,
            health_check_interval_secs: self.bidding_config.health_check_interval_secs as u32,
            error_message: String::new(),
            collateral_status: worst_collateral_status.map(status_to_proto),
        }))
    }

    async fn update_bid(
        &self,
        request: Request<UpdateBidRequest>,
    ) -> Result<Response<UpdateBidResponse>, Status> {
        let req = request.into_inner();

        // Validate required fields
        if req.miner_hotkey.trim().is_empty() {
            return Err(Status::invalid_argument("miner_hotkey is required"));
        }
        if req.host.trim().is_empty() {
            return Err(Status::invalid_argument("host is required"));
        }
        if req.hourly_rate_cents == 0 {
            return Err(Status::invalid_argument(
                "hourly_rate_cents must be greater than 0",
            ));
        }
        if req.signature.is_empty() {
            return Err(Status::invalid_argument("signature is required"));
        }

        // Verify timestamp freshness
        self.ensure_timestamp_freshness(req.timestamp)?;

        // Verify signature
        let message = self.build_update_bid_message(&req);
        self.verify_signature(&req.miner_hotkey, &req.signature, &message)?;

        // Resolve miner ID
        let miner_id = self.resolve_miner_id(&req.miner_hotkey).await?;

        // Compute node_id server-side from host
        let node_id = basilica_common::node_identity::NodeId::new(&req.host)
            .map_err(|e| Status::internal(format!("failed to compute node_id: {e}")))?
            .uuid
            .to_string();

        // Load stored node metadata to enforce bid floor for UpdateBid.
        let node_metadata = self
            .persistence
            .get_node_bid_metadata(&miner_id, &node_id)
            .await
            .map_err(|e| Status::internal(format!("failed to fetch node metadata: {e}")))?
            .ok_or_else(|| Status::not_found("node not found"))?;

        let gpu_category = node_metadata.gpu_category.trim();
        if gpu_category.is_empty() {
            return Err(Status::failed_precondition(
                "node missing gpu_category; re-register via RegisterBid",
            ));
        }

        let gpu_cat: GpuCategory = gpu_category.parse().unwrap(); // Infallible
        if matches!(&gpu_cat, GpuCategory::Other(_)) {
            return Err(Status::failed_precondition(format!(
                "unsupported GPU type '{}' for node; re-register via RegisterBid",
                gpu_category
            )));
        }

        // Fetch baseline prices for bid floor enforcement (best-effort)
        let baseline_prices = if let Some(ref api_client) = self.api_client {
            match api_client.get_baseline_prices().await {
                Ok(prices) => Some(prices),
                Err(e) => {
                    warn!(
                        miner_hotkey = %req.miner_hotkey,
                        error = %e,
                        "Failed to fetch baseline prices for UpdateBid floor check - allowing update"
                    );
                    None
                }
            }
        } else {
            None
        };

        // Enforce bid floor: bid must be >= min_bid_floor_fraction * baseline price
        if let Some(ref prices) = baseline_prices {
            let category_key = gpu_cat.to_string();
            if let Some(&baseline_dollars) = prices.get(&category_key) {
                let floor_cents =
                    (baseline_dollars * 100.0 * self.bidding_config.min_bid_floor_fraction).round()
                        as u32;
                if floor_cents > 0 && req.hourly_rate_cents < floor_cents {
                    return Err(Status::invalid_argument(format!(
                        "hourly_rate_cents {} is below minimum floor {} ({:.0}% of baseline ${:.2}/hr) for {}",
                        req.hourly_rate_cents,
                        floor_cents,
                        self.bidding_config.min_bid_floor_fraction * 100.0,
                        baseline_dollars,
                        category_key,
                    )));
                }
            }
        }

        // Update node price
        let updated = self
            .persistence
            .update_node_hourly_rate(&miner_id, &node_id, req.hourly_rate_cents)
            .await
            .map_err(|e| Status::internal(format!("failed to update node: {e}")))?;

        if !updated {
            return Err(Status::not_found("node not found"));
        }

        info!(
            miner_hotkey = %req.miner_hotkey,
            host = %req.host,
            hourly_rate_cents = req.hourly_rate_cents,
            "Updated node price via UpdateBid"
        );

        Ok(Response::new(UpdateBidResponse {
            accepted: true,
            error_message: String::new(),
        }))
    }

    async fn remove_bid(
        &self,
        request: Request<RemoveBidRequest>,
    ) -> Result<Response<RemoveBidResponse>, Status> {
        let req = request.into_inner();

        // Validate required fields
        if req.miner_hotkey.trim().is_empty() {
            return Err(Status::invalid_argument("miner_hotkey is required"));
        }
        if req.signature.is_empty() {
            return Err(Status::invalid_argument("signature is required"));
        }

        // Verify timestamp freshness
        self.ensure_timestamp_freshness(req.timestamp)?;

        // Verify signature
        let message = self.build_remove_bid_message(&req);
        self.verify_signature(&req.miner_hotkey, &req.signature, &message)?;

        // Resolve miner ID
        let miner_id = self.resolve_miner_id(&req.miner_hotkey).await?;

        // Compute node_ids server-side from hosts
        let node_ids: Vec<String> = req
            .hosts
            .iter()
            .map(|host| {
                basilica_common::node_identity::NodeId::new(host).map(|id| id.uuid.to_string())
            })
            .collect::<Result<_, _>>()
            .map_err(|e| Status::internal(format!("failed to compute node_ids: {e}")))?;

        // Remove nodes
        let removed = self
            .persistence
            .remove_registered_nodes(&miner_id, &node_ids)
            .await
            .map_err(|e| Status::internal(format!("failed to remove nodes: {e}")))?;

        info!(
            miner_hotkey = %req.miner_hotkey,
            nodes_removed = removed,
            "Removed nodes via RemoveBid"
        );

        Ok(Response::new(RemoveBidResponse {
            accepted: true,
            nodes_removed: removed,
            error_message: String::new(),
        }))
    }

    async fn health_check(
        &self,
        request: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        let req = request.into_inner();

        // Validate required fields
        if req.miner_hotkey.trim().is_empty() {
            return Err(Status::invalid_argument("miner_hotkey is required"));
        }
        if req.signature.is_empty() {
            return Err(Status::invalid_argument("signature is required"));
        }

        // Verify timestamp freshness (use shorter window for health checks)
        self.ensure_timestamp_freshness(req.timestamp)?;

        // Verify signature
        let message = self.build_health_check_message(&req);
        self.verify_signature(&req.miner_hotkey, &req.signature, &message)?;

        // Resolve miner ID
        let miner_id = self.resolve_miner_id(&req.miner_hotkey).await?;

        // Compute node_ids server-side from hosts
        let node_ids: Vec<String> = req
            .hosts
            .iter()
            .map(|host| {
                basilica_common::node_identity::NodeId::new(host).map(|id| id.uuid.to_string())
            })
            .collect::<Result<_, _>>()
            .map_err(|e| Status::internal(format!("failed to compute node_ids: {e}")))?;

        // Update health check timestamp
        let updated = self
            .persistence
            .update_nodes_health_check(&miner_id, &node_ids)
            .await
            .map_err(|e| Status::internal(format!("failed to update health check: {e}")))?;

        Ok(Response::new(HealthCheckResponse {
            accepted: true,
            nodes_active: updated,
            error_message: String::new(),
        }))
    }
}

/// Start the MinerRegistration gRPC server
pub async fn start_registration_server(
    config: GrpcServerConfig,
    persistence: Arc<SimplePersistence>,
    bidding_config: BiddingConfig,
    collateral_manager: Option<Arc<CollateralManager>>,
    validator_ssh_public_key: String,
    api_client: Option<Arc<BasilicaApiClient>>,
) -> Result<()> {
    let service = RegistrationService::new(
        persistence,
        bidding_config,
        collateral_manager,
        validator_ssh_public_key,
        api_client,
    );
    let addr = config
        .listen_address
        .parse()
        .map_err(|e| anyhow::anyhow!("invalid gRPC listen address: {}", e))?;

    let (mut health_reporter, health_service) = health_reporter();
    health_reporter
        .set_serving::<MinerRegistrationServer<RegistrationService>>()
        .await;

    info!(address = %config.listen_address, "Starting miner registration gRPC server");

    Server::builder()
        .add_service(health_service)
        .add_service(MinerRegistrationServer::new(service))
        .serve(addr)
        .await
        .map_err(|e| anyhow::anyhow!("registration gRPC server failed: {}", e))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn create_test_service() -> RegistrationService {
        let persistence = SimplePersistence::for_testing().await.unwrap();
        RegistrationService::new(
            Arc::new(persistence),
            BiddingConfig::default(),
            None,
            "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITestPublicKey test@validator".to_string(),
            None,
        )
    }

    #[tokio::test]
    async fn test_build_register_bid_message() {
        let service = create_test_service().await;

        let req = RegisterBidRequest {
            miner_hotkey: "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string(),
            nodes: vec![],
            timestamp: 1234567890,
            signature: vec![],
        };

        let message = service.build_register_bid_message(&req);
        assert_eq!(
            message,
            "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY|1234567890"
        );
    }

    #[tokio::test]
    async fn test_build_update_bid_message() {
        let service = create_test_service().await;

        let req = UpdateBidRequest {
            miner_hotkey: "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string(),
            hourly_rate_cents: 250,
            timestamp: 1234567890,
            signature: vec![],
            host: "192.168.1.1".to_string(),
        };

        let message = service.build_update_bid_message(&req);
        assert_eq!(
            message,
            "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY|192.168.1.1|250|1234567890"
        );
    }

    #[tokio::test]
    async fn test_build_remove_bid_message() {
        let service = create_test_service().await;

        let req = RemoveBidRequest {
            miner_hotkey: "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string(),
            timestamp: 1234567890,
            signature: vec![],
            hosts: vec!["192.168.1.1".to_string(), "192.168.1.2".to_string()],
        };

        let message = service.build_remove_bid_message(&req);
        assert_eq!(
            message,
            "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY|192.168.1.1,192.168.1.2|1234567890"
        );
    }

    #[tokio::test]
    async fn test_build_health_check_message() {
        let service = create_test_service().await;

        // Test with empty hosts (all nodes)
        let req = HealthCheckRequest {
            miner_hotkey: "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string(),
            timestamp: 1234567890,
            signature: vec![],
            hosts: vec![],
        };

        let message = service.build_health_check_message(&req);
        assert_eq!(
            message,
            "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY||1234567890"
        );

        // Test with specific hosts
        let req = HealthCheckRequest {
            miner_hotkey: "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".to_string(),
            timestamp: 1234567890,
            signature: vec![],
            hosts: vec!["192.168.1.1".to_string(), "192.168.1.2".to_string()],
        };

        let message = service.build_health_check_message(&req);
        assert_eq!(
            message,
            "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY|192.168.1.1,192.168.1.2|1234567890"
        );
    }
}
