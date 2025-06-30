//! Simplified Validator Access gRPC Service

use super::types::{
    GrpcResult, SharedExecutorState, ValidatorAccessInfo, ValidatorAccessRequest,
    ValidatorAccessResponse, ValidatorHealthRequest, ValidatorHealthResponse, ValidatorListRequest,
    ValidatorListResponse, ValidatorRevokeRequest, ValidatorRevokeResponse, ValidatorServiceTrait,
};
use crate::validation_session::{RequestType, ValidatorId, ValidatorRole};
use anyhow::Result;
use tonic::{Request, Response, Status};
use tracing::{error, info, warn};

/// Simplified validator access gRPC handler
pub struct ValidatorAccessService {
    state: SharedExecutorState,
}

impl ValidatorAccessService {
    /// Create new validator access service
    pub fn new(state: SharedExecutorState) -> Self {
        Self { state }
    }

    /// Handle validator access request with enhanced security
    pub async fn handle_access_request(
        &self,
        request: Request<ValidatorAccessRequest>,
    ) -> Result<Response<ValidatorAccessResponse>, Status> {
        let req = request.into_inner();
        info!(
            "Handling enhanced access request for validator: {}",
            req.hotkey
        );

        let state = &*self.state;

        // Check if validator service is enabled
        let validation_service = match &state.validation_service {
            Some(service) => service,
            None => {
                warn!("Validation service is not enabled");
                return Err(Status::unavailable("Validation service not enabled"));
            }
        };

        // Create validator ID
        let validator_id = ValidatorId::new(req.hotkey.clone());

        // Check rate limits first
        let access_control = validation_service.access_control();
        let client_ip = None; // TODO: Extract from request metadata

        if !access_control
            .authenticate_request(&validator_id, client_ip, RequestType::Api)
            .await
            .map_err(|e| {
                error!("Authentication failed: {}", e);
                Status::unauthenticated("Authentication failed")
            })?
        {
            return Err(Status::permission_denied("Authentication failed"));
        }

        // Handle hotkey verification flow
        if access_control.config.hotkey_verification.enabled {
            // If signature is provided, verify it
            if let (Some(signature_hex), Some(public_key_hex)) =
                (req.signature.as_ref(), req.public_key.as_ref())
            {
                // Decode hex signature and public key
                let signature_bytes = hex::decode(signature_hex)
                    .map_err(|_| Status::invalid_argument("Invalid signature format"))?;
                let public_key_bytes = hex::decode(public_key_hex)
                    .map_err(|_| Status::invalid_argument("Invalid public key format"))?;

                // Verify signature against challenge
                let challenge_id = req.challenge_id.ok_or_else(|| {
                    Status::invalid_argument("Challenge ID required for signature verification")
                })?;

                let verification_result = access_control
                    .verify_hotkey_signature(&challenge_id, &signature_bytes, &public_key_bytes)
                    .await
                    .map_err(|e| {
                        error!("Signature verification failed: {}", e);
                        Status::unauthenticated("Signature verification failed")
                    })?;

                if !verification_result {
                    return Err(Status::permission_denied("Invalid signature"));
                }

                // Grant access with verified status
                let role = self.determine_validator_role(&validator_id).await;
                let ssh_public_key = req
                    .ssh_public_key
                    .ok_or_else(|| Status::invalid_argument("SSH public key required"))?;

                access_control
                    .grant_access_with_verification(
                        &validator_id,
                        &ssh_public_key,
                        role,
                        true, // hotkey verified
                    )
                    .await
                    .map_err(|e| {
                        error!("Failed to grant verified access: {}", e);
                        Status::internal("Failed to grant access")
                    })?;

                // Also grant SSH access
                validation_service
                    .grant_ssh_access(&validator_id, &ssh_public_key)
                    .await
                    .map_err(|e| {
                        error!("Failed to grant SSH access: {}", e);
                        Status::internal("Failed to grant SSH access")
                    })?;

                let response = ValidatorAccessResponse {
                    success: true,
                    message: "Access granted with hotkey verification".to_string(),
                    ssh_public_key: Some(ssh_public_key),
                    ssh_username: Some(format!("validator_{}", validator_id.hotkey)),
                    access_token: None,
                    expires_at: 0,
                };

                Ok(Response::new(response))
            } else {
                // Generate challenge for hotkey verification
                let challenge = access_control
                    .generate_hotkey_challenge(&validator_id)
                    .await
                    .map_err(|e| {
                        error!("Failed to generate challenge: {}", e);
                        Status::internal("Failed to generate challenge")
                    })?;

                let response = ValidatorAccessResponse {
                    success: false,
                    message: format!(
                        "Hotkey verification required. Challenge: {}",
                        challenge.challenge_id
                    ),
                    ssh_public_key: None,
                    ssh_username: None,
                    access_token: Some(hex::encode(&challenge.challenge_data)),
                    expires_at: challenge
                        .expires_at
                        .duration_since(std::time::SystemTime::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs(),
                };

                Ok(Response::new(response))
            }
        } else {
            // Simplified flow without hotkey verification
            let role = self.determine_validator_role(&validator_id).await;
            let ssh_public_key = req
                .ssh_public_key
                .ok_or_else(|| Status::invalid_argument("SSH public key required"))?;

            // Grant access without verification
            access_control
                .grant_access_with_verification(
                    &validator_id,
                    &ssh_public_key,
                    role,
                    false, // hotkey not verified
                )
                .await
                .map_err(|e| {
                    error!("Failed to grant access: {}", e);
                    Status::internal("Failed to grant access")
                })?;

            // Also grant SSH access
            validation_service
                .grant_ssh_access(&validator_id, &ssh_public_key)
                .await
                .map_err(|e| {
                    error!("Failed to grant SSH access: {}", e);
                    Status::internal("Failed to grant SSH access")
                })?;

            let response = ValidatorAccessResponse {
                success: true,
                message: "SSH access granted successfully".to_string(),
                ssh_public_key: Some(ssh_public_key),
                ssh_username: Some(format!("validator_{}", validator_id.hotkey)),
                access_token: None,
                expires_at: 0,
            };

            Ok(Response::new(response))
        }
    }

    /// Determine the role for a validator based on configuration
    async fn determine_validator_role(&self, validator_id: &ValidatorId) -> ValidatorRole {
        let state = &*self.state;

        if let Some(validation_service) = &state.validation_service {
            // Check role assignments in config
            let access_control = validation_service.access_control();
            if let Some(role) = access_control
                .config
                .role_assignments
                .get(&validator_id.hotkey)
            {
                return role.clone();
            }
        }

        // Default to basic role
        ValidatorRole::Basic
    }

    /// Handle validator access revocation
    pub async fn handle_revoke_request(
        &self,
        request: Request<ValidatorRevokeRequest>,
    ) -> Result<Response<ValidatorRevokeResponse>, Status> {
        let req = request.into_inner();
        info!("Handling revoke request for validator: {}", req.hotkey);

        let state = &*self.state;

        // Check if validator service is enabled
        let validation_service = match &state.validation_service {
            Some(service) => service,
            None => {
                warn!("Validation service is not enabled");
                return Err(Status::unavailable("Validation service not enabled"));
            }
        };

        let validator_id = ValidatorId::new(req.hotkey.clone());

        // Simply revoke SSH access
        match validation_service.revoke_ssh_access(&validator_id).await {
            Ok(()) => {
                let response = ValidatorRevokeResponse {
                    success: true,
                    message: "SSH access revoked successfully".to_string(),
                };
                Ok(Response::new(response))
            }
            Err(e) => {
                error!("Failed to revoke SSH access: {}", e);
                Err(Status::internal("Failed to revoke access"))
            }
        }
    }

    /// Handle list active access request
    pub async fn handle_list_request(
        &self,
        _request: Request<ValidatorListRequest>,
    ) -> Result<Response<ValidatorListResponse>, Status> {
        let state = &*self.state;

        // Check if validator service is enabled
        let validation_service = match &state.validation_service {
            Some(service) => service,
            None => {
                warn!("Validation service is not enabled");
                return Err(Status::unavailable("Validation service not enabled"));
            }
        };

        // Get active access list
        match validation_service.list_active_access().await {
            Ok(access_list) => {
                let access_entries: Vec<ValidatorAccessInfo> = access_list
                    .into_iter()
                    .map(|access| ValidatorAccessInfo {
                        hotkey: access.validator_id.hotkey,
                        access_type: "ssh".to_string(), // Simplified - always SSH
                        granted_at: access
                            .granted_at
                            .duration_since(std::time::SystemTime::UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                        expires_at: 0, // Not enforced by us
                        has_ssh_access: true,
                        has_token_access: false,
                    })
                    .collect();

                let response = ValidatorListResponse {
                    success: true,
                    message: format!("Found {} active SSH access entries", access_entries.len()),
                    access_entries,
                };

                Ok(Response::new(response))
            }
            Err(e) => {
                error!("Failed to list active access: {}", e);
                Err(Status::internal("Failed to list active access"))
            }
        }
    }

    /// Handle health check request
    pub async fn handle_health_check(
        &self,
        _request: Request<ValidatorHealthRequest>,
    ) -> Result<Response<ValidatorHealthResponse>, Status> {
        let state = &*self.state;

        // Check if validator service is enabled
        let validation_service = match &state.validation_service {
            Some(service) => service,
            None => {
                return Ok(Response::new(ValidatorHealthResponse {
                    healthy: false,
                    message: "Validation service not enabled".to_string(),
                    active_sessions: 0,
                    total_sessions: 0,
                    ssh_sessions: 0,
                    token_sessions: 0,
                    log_entries: 0,
                    uptime_seconds: 0,
                }));
            }
        };

        // Get basic session count
        let active_access = validation_service
            .list_active_access()
            .await
            .unwrap_or_default();
        let session_count = active_access.len() as u32;

        let response = ValidatorHealthResponse {
            healthy: true,
            message: "Validation service is healthy".to_string(),
            active_sessions: session_count,
            total_sessions: session_count,
            ssh_sessions: session_count, // All sessions are SSH now
            token_sessions: 0,           // No token sessions in simplified approach
            log_entries: 0,              // Simplified logging
            uptime_seconds: 0,           // Not tracked
        };

        Ok(Response::new(response))
    }

    /// Simple provision SSH access (backward compatibility)
    pub async fn provision_ssh_access(
        &self,
        validator_hotkey: &str,
        ssh_public_key: String,
    ) -> GrpcResult<()> {
        info!("Provisioning SSH access for hotkey: {}", validator_hotkey);

        let state = &*self.state;

        // Check if validator service is enabled
        let validation_service = match &state.validation_service {
            Some(service) => service,
            None => {
                warn!("Validation service is not enabled");
                return Err(anyhow::anyhow!("Validation service not enabled"));
            }
        };

        let validator_id = ValidatorId::new(validator_hotkey.to_string());

        match validation_service
            .grant_ssh_access(&validator_id, &ssh_public_key)
            .await
        {
            Ok(()) => {
                info!(
                    "Successfully provisioned SSH access for {}",
                    validator_hotkey
                );
                Ok(())
            }
            Err(e) => {
                warn!("Failed to provision SSH access: {}", e);
                Err(anyhow::anyhow!("Failed to provision access: {}", e))
            }
        }
    }

    /// Simple revoke SSH access (backward compatibility)
    pub async fn revoke_ssh_access(&self, validator_hotkey: &str) -> GrpcResult<()> {
        info!("Revoking SSH access for hotkey: {}", validator_hotkey);

        let state = &*self.state;

        // Check if validator service is enabled
        let validation_service = match &state.validation_service {
            Some(service) => service,
            None => {
                warn!("Validation service is not enabled");
                return Err(anyhow::anyhow!("Validation service not enabled"));
            }
        };

        let validator_id = ValidatorId::new(validator_hotkey.to_string());

        match validation_service.revoke_ssh_access(&validator_id).await {
            Ok(()) => {
                info!("Successfully revoked SSH access for {}", validator_hotkey);
                Ok(())
            }
            Err(e) => {
                warn!("Failed to revoke SSH access: {}", e);
                Err(anyhow::anyhow!("Failed to revoke access: {}", e))
            }
        }
    }

    /// Simple list active SSH access (backward compatibility)
    pub async fn list_ssh_access(&self) -> GrpcResult<Vec<String>> {
        info!("Listing active SSH access");

        let state = &*self.state;

        // Check if validator service is enabled
        let validation_service = match &state.validation_service {
            Some(service) => service,
            None => {
                warn!("Validation service is not enabled");
                return Ok(vec![]);
            }
        };

        match validation_service.list_active_access().await {
            Ok(access_list) => {
                let hotkeys: Vec<String> = access_list
                    .into_iter()
                    .map(|access| access.validator_id.hotkey)
                    .collect();
                Ok(hotkeys)
            }
            Err(e) => {
                warn!("Failed to list SSH access: {}", e);
                Err(anyhow::anyhow!("Failed to list access: {}", e))
            }
        }
    }
}

#[tonic::async_trait]
impl ValidatorServiceTrait for ValidatorAccessService {
    async fn request_access(
        &self,
        request: Request<ValidatorAccessRequest>,
    ) -> Result<Response<ValidatorAccessResponse>, Status> {
        self.handle_access_request(request).await
    }

    async fn revoke_access(
        &self,
        request: Request<ValidatorRevokeRequest>,
    ) -> Result<Response<ValidatorRevokeResponse>, Status> {
        self.handle_revoke_request(request).await
    }

    async fn list_access(
        &self,
        request: Request<ValidatorListRequest>,
    ) -> Result<Response<ValidatorListResponse>, Status> {
        self.handle_list_request(request).await
    }

    async fn health_check(
        &self,
        request: Request<ValidatorHealthRequest>,
    ) -> Result<Response<ValidatorHealthResponse>, Status> {
        self.handle_health_check(request).await
    }
}
