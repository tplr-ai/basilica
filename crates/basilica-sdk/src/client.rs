//! HTTP client for the Basilica API
//!
//! This module provides a type-safe client for interacting with the Basilica API.
//! It supports both authenticated and unauthenticated requests.
//!
//! # Authentication
//!
//! The client uses Auth0 JWT Bearer token authentication:
//!
//! ## Auth0 JWT Authentication
//! - Uses `Authorization: Bearer {token}` header with Auth0-issued JWT tokens
//! - Thread-safe token management with async/await support
//! - Secure authentication via Auth0 identity provider
//!
//! # Usage Examples
//!
//! ```rust,no_run
//! use basilica_sdk::{BasilicaClient, ClientBuilder};
//! use std::sync::Arc;
//!
//! # async fn example() -> basilica_sdk::Result<()> {
//! // Direct token authentication with refresh support
//! let client = ClientBuilder::default()
//!     .base_url("https://api.basilica.ai")
//!     .with_tokens("access_token", "refresh_token")
//!     .build()?;
//!
//! // Or use file-based authentication (reads from ~/.local/share/basilica/)
//! let client = ClientBuilder::default()
//!     .base_url("https://api.basilica.ai")
//!     .with_file_auth()
//!     .build()?;
//!
//! # Ok(())
//! # }
//! ```

use crate::{
    auth::TokenManager,
    error::{ApiError, ErrorResponse, Result},
    jobs::{
        CreateJobRequest, CreateJobResponse, DeleteJobResponse, JobLogsResponse, JobStatusResponse,
        ReadFileRequest, ReadFileResponse, ResumeJobResponse, SuspendJobResponse,
    },
    types::{
        ApiKeyInfo, ApiKeyResponse, ApiListRentalsResponse, BalanceResponse, CreateApiKeyRequest,
        CreateDeploymentRequest, CreateDepositAccountResponse, DeleteDeploymentResponse,
        DeleteShareTokenResponse, DeploymentEventsResponse, DeploymentListResponse,
        DeploymentResponse, DepositAccountResponse, EnrollMetadataRequest, EnrollMetadataResponse,
        HealthCheckResponse, HistoricalRentalsResponse, ListAvailableNodesQuery, ListDepositsQuery,
        ListDepositsResponse, ListRentalsQuery, PublicDeploymentMetadataResponse,
        RegenerateShareTokenResponse, RegisterSshKeyRequest, RentalStatusWithSshResponse,
        RentalUsageResponse, ScaleDeploymentRequest, ShareTokenStatusResponse, SshKeyResponse,
        UsageHistoryResponse, WaitOptions, WaitResult,
    },
    StartRentalApiRequest,
};

/// Default API URL when not specified
pub const DEFAULT_API_URL: &str = "https://api.basilica.ai";

/// Default timeout in seconds for API requests
pub const DEFAULT_TIMEOUT_SECS: u64 = 1200;
use basilica_common::ApiKeyName;
use basilica_validator::api::types::ListAvailableNodesResponse;
use basilica_validator::rental::RentalResponse;
use reqwest::{RequestBuilder, Response, StatusCode};
use serde::{de::DeserializeOwned, Serialize};
use std::sync::Arc;
use std::time::Duration;

/// HTTP client for interacting with the Basilica API
#[derive(Debug)]
pub struct BasilicaClient {
    http_client: reqwest::Client,
    base_url: String,
    token_manager: Arc<TokenManager>,
}

impl BasilicaClient {
    /// Create a new client (private - use ClientBuilder instead)
    fn new(
        base_url: impl Into<String>,
        timeout: Duration,
        token_manager: Arc<TokenManager>,
    ) -> Result<Self> {
        let http_client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(ApiError::HttpClient)?;

        Ok(Self {
            http_client,
            base_url: base_url.into(),
            token_manager,
        })
    }

    // ===== Rentals =====

    /// Get rental status
    pub async fn get_rental_status(&self, rental_id: &str) -> Result<RentalStatusWithSshResponse> {
        let path = format!("/rentals/{rental_id}");
        self.get(&path).await
    }

    /// Start a new rental
    pub async fn start_rental(&self, request: StartRentalApiRequest) -> Result<RentalResponse> {
        self.post("/rentals", &request).await
    }

    /// Stop a rental
    pub async fn stop_rental(&self, rental_id: &str) -> Result<()> {
        let path = format!("/rentals/{rental_id}");
        let response: Response = self.delete_empty(&path).await?;
        if response.status().is_success() {
            Ok(())
        } else {
            let err = self
                .handle_error_response::<serde_json::Value>(response)
                .await
                .err()
                .unwrap_or(ApiError::Internal {
                    message: "Unknown error".into(),
                });
            Err(err)
        }
    }

    /// Restart a rental's container
    pub async fn restart_rental(
        &self,
        rental_id: &str,
    ) -> Result<basilica_validator::rental::RentalRestartResponse> {
        let path = format!("/rentals/{rental_id}/restart");
        self.post(&path, &serde_json::json!({})).await
    }

    /// Get rental logs
    pub async fn get_rental_logs(
        &self,
        rental_id: &str,
        follow: bool,
        tail: Option<u32>,
    ) -> Result<reqwest::Response> {
        let url = format!("{}/rentals/{}/logs", self.base_url, rental_id);
        let mut request = self.http_client.get(&url);

        let mut params: Vec<(&str, String)> = vec![];
        if follow {
            params.push(("follow", "true".to_string()));
        }
        if let Some(tail_lines) = tail {
            params.push(("tail", tail_lines.to_string()));
        }

        if !params.is_empty() {
            request = request.query(&params);
        }

        let request = self.apply_auth(request).await?;
        request.send().await.map_err(ApiError::HttpClient)
    }

    /// List rentals
    pub async fn list_rentals(
        &self,
        query: Option<ListRentalsQuery>,
    ) -> Result<ApiListRentalsResponse> {
        let url = format!("{}/rentals", self.base_url);
        let mut request = self.http_client.get(&url);

        if let Some(q) = &query {
            request = request.query(&q);
        }

        let request = self.apply_auth(request).await?;
        let response = request.send().await.map_err(ApiError::HttpClient)?;
        self.handle_response(response).await
    }

    /// List historical (completed/failed) rentals
    pub async fn list_rental_history(
        &self,
        limit: Option<u32>,
    ) -> Result<HistoricalRentalsResponse> {
        let url = format!("{}/rentals/history", self.base_url);
        let mut request = self.http_client.get(&url);

        if let Some(limit) = limit {
            request = request.query(&[("limit", limit)]);
        }

        let request = self.apply_auth(request).await?;
        let response = request.send().await.map_err(ApiError::HttpClient)?;
        self.handle_response(response).await
    }

    /// List available nodes for rental
    pub async fn list_available_nodes(
        &self,
        query: Option<ListAvailableNodesQuery>,
    ) -> Result<ListAvailableNodesResponse> {
        let url = format!("{}/nodes", self.base_url);
        let mut request = self.http_client.get(&url);

        if let Some(q) = &query {
            request = request.query(&q);
        }

        let request = self.apply_auth(request).await?;
        let response = request.send().await.map_err(ApiError::HttpClient)?;
        self.handle_response(response).await
    }

    // ===== Health & Discovery =====

    /// Health check
    pub async fn health_check(&self) -> Result<HealthCheckResponse> {
        self.get("/health").await
    }

    // ===== API Key Management =====

    /// Create a new API key (requires JWT authentication)
    /// The API key will inherit scopes from the current JWT token
    pub async fn create_api_key(&self, name: &str) -> Result<ApiKeyResponse> {
        // Validate the name early to provide better error messages
        ApiKeyName::new(name).map_err(|e| ApiError::InvalidRequest {
            message: format!("Invalid API key name: {}", e),
        })?;

        let request = CreateApiKeyRequest {
            name: name.to_string(),
            scopes: None, // Will inherit from JWT
        };
        self.post("/api-keys", &request).await
    }

    /// Get current API key info (requires JWT authentication)
    /// Returns the first (and only) key if it exists
    pub async fn get_api_key(&self) -> Result<Option<ApiKeyInfo>> {
        let keys: Vec<ApiKeyInfo> = self.get("/api-keys").await?;
        Ok(keys.into_iter().next())
    }

    /// List all API keys for the authenticated user (requires JWT authentication)
    pub async fn list_api_keys(&self) -> Result<Vec<ApiKeyInfo>> {
        self.get("/api-keys").await
    }

    /// Delete a specific API key by name (requires JWT authentication)
    pub async fn revoke_api_key(&self, name: &str) -> Result<()> {
        let encoded_name = urlencoding::encode(name);
        let response = self
            .delete_empty(&format!("/api-keys/{}", encoded_name))
            .await?;
        if response.status().is_success() {
            Ok(())
        } else {
            self.handle_error_response(response).await
        }
    }

    // ===== SSH Key Management =====

    /// Register a new SSH key for the authenticated user (requires JWT authentication)
    /// Only one SSH key per user is allowed.
    pub async fn register_ssh_key(&self, name: &str, public_key: &str) -> Result<SshKeyResponse> {
        let request = RegisterSshKeyRequest {
            name: name.to_string(),
            public_key: public_key.to_string(),
        };
        self.post("/ssh-keys", &request).await
    }

    /// Get the authenticated user's SSH key (requires JWT authentication)
    /// Returns None if no SSH key is registered.
    pub async fn get_ssh_key(&self) -> Result<Option<SshKeyResponse>> {
        self.get("/ssh-keys").await
    }

    /// Delete the authenticated user's SSH key (requires JWT authentication)
    /// Also removes the key from all cloud providers.
    pub async fn delete_ssh_key(&self) -> Result<()> {
        let response = self.delete_empty("/ssh-keys").await?;
        if response.status().is_success() {
            Ok(())
        } else {
            self.handle_error_response(response).await
        }
    }

    // ===== Jobs API =====

    /// Create a new job
    pub async fn create_job(&self, request: CreateJobRequest) -> Result<CreateJobResponse> {
        self.post("/v2/jobs", &request).await
    }

    /// Get job status
    pub async fn get_job_status(&self, job_id: &str) -> Result<JobStatusResponse> {
        let path = format!("/v2/jobs/{}", job_id);
        self.get(&path).await
    }

    /// Delete a job
    pub async fn delete_job(&self, job_id: &str) -> Result<DeleteJobResponse> {
        let path = format!("/v2/jobs/{}", job_id);
        let response = self.delete_empty(&path).await?;
        if response.status().is_success() {
            response.json().await.map_err(ApiError::HttpClient)
        } else {
            self.handle_error_response(response).await
        }
    }

    /// Get job logs
    pub async fn get_job_logs(&self, job_id: &str) -> Result<JobLogsResponse> {
        let path = format!("/v2/jobs/{}/logs", job_id);
        self.get(&path).await
    }

    /// Read a file from a job's container
    pub async fn read_job_file(&self, job_id: &str, file_path: &str) -> Result<ReadFileResponse> {
        let path = format!("/v2/jobs/{}/files", job_id);
        let request = ReadFileRequest {
            file_path: file_path.to_string(),
        };
        self.post(&path, &request).await
    }

    /// Suspend a job (pause execution)
    pub async fn suspend_job(&self, job_id: &str) -> Result<SuspendJobResponse> {
        let path = format!("/v2/jobs/{}/suspend", job_id);
        self.post(&path, &serde_json::json!({})).await
    }

    /// Resume a suspended job
    pub async fn resume_job(&self, job_id: &str) -> Result<ResumeJobResponse> {
        let path = format!("/v2/jobs/{}/resume", job_id);
        self.post(&path, &serde_json::json!({})).await
    }

    // ===== Payment Management =====

    /// Get deposit account for the authenticated user
    pub async fn get_deposit_account(&self) -> Result<DepositAccountResponse> {
        self.get("/payments/deposit-account").await
    }

    /// Create a deposit account for the authenticated user
    pub async fn create_deposit_account(&self) -> Result<CreateDepositAccountResponse> {
        self.post("/payments/deposit-account", &serde_json::json!({}))
            .await
    }

    /// List deposits for the authenticated user
    pub async fn list_deposits(
        &self,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> Result<ListDepositsResponse> {
        let query = ListDepositsQuery {
            limit: limit.unwrap_or(50),
            offset: offset.unwrap_or(0),
        };

        let url = format!("{}/payments/deposits", self.base_url);
        let mut request = self.http_client.get(&url);
        request = request.query(&query);

        let request = self.apply_auth(request).await?;
        let response = request.send().await.map_err(ApiError::HttpClient)?;
        self.handle_response(response).await
    }

    // ===== Billing Management =====

    /// Get balance for the authenticated user
    pub async fn get_balance(&self) -> Result<BalanceResponse> {
        self.get("/billing/balance").await
    }

    /// List usage history for authenticated user
    pub async fn list_usage_history(
        &self,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> Result<UsageHistoryResponse> {
        let mut params = Vec::new();
        if let Some(limit) = limit {
            params.push(("limit", limit.to_string()));
        }
        if let Some(offset) = offset {
            params.push(("offset", offset.to_string()));
        }

        let url = format!("{}/billing/usage", self.base_url);
        let mut request = self.http_client.get(&url);

        if !params.is_empty() {
            request = request.query(&params);
        }

        let request = self.apply_auth(request).await?;
        let response = request.send().await.map_err(ApiError::HttpClient)?;
        self.handle_response(response).await
    }

    /// Get detailed usage for a specific rental
    pub async fn get_rental_usage(&self, rental_id: &str) -> Result<RentalUsageResponse> {
        let path = format!("/billing/usage/{}", rental_id);
        self.get(&path).await
    }

    // ===== Secure Cloud (GPU Aggregator) =====

    /// List secure cloud GPU offerings from datacenter providers
    /// Returns GPUs available from providers like Verda, Hyperstack, Lambda Labs, etc.
    pub async fn list_secure_cloud_gpus(&self) -> Result<Vec<crate::types::GpuOffering>> {
        self.list_secure_cloud_gpus_filtered(&Default::default())
            .await
    }

    /// List secure cloud GPU offerings with flavour filters
    ///
    /// Accepts a `GpuPriceQuery` to filter by interconnect, geo, spot preferences.
    pub async fn list_secure_cloud_gpus_filtered(
        &self,
        query: &crate::types::GpuPriceQuery,
    ) -> Result<Vec<crate::types::GpuOffering>> {
        let mut url = String::from("/secure-cloud/gpu-prices?available_only=true");
        if let Some(ref interconnect) = query.interconnect {
            url.push_str(&format!("&interconnect={}", interconnect));
        }
        if let Some(ref region) = query.region {
            url.push_str(&format!("&region={}", region));
        }
        if let Some(true) = query.spot_only {
            url.push_str("&spot_only=true");
        }
        if let Some(true) = query.exclude_spot {
            url.push_str("&exclude_spot=true");
        }
        let response: crate::types::ListSecureCloudGpusResponse = self.get(&url).await?;
        Ok(response.nodes)
    }

    /// List secure cloud rentals for the authenticated user
    ///
    /// Returns all secure cloud (datacenter) rentals including their GPU details,
    /// status, IP addresses, and cost information.
    pub async fn list_secure_cloud_rentals(
        &self,
    ) -> Result<crate::types::ListSecureCloudRentalsResponse> {
        self.get("/secure-cloud/rentals").await
    }

    /// Start a secure cloud rental
    ///
    /// Deploys a GPU instance via datacenter provider (Verda, Hyperstack, etc.)
    /// and registers it with the billing service for incremental charging.
    pub async fn start_secure_cloud_rental(
        &self,
        request: crate::types::StartSecureCloudRentalRequest,
    ) -> Result<crate::types::SecureCloudRentalResponse> {
        self.post("/secure-cloud/rentals/start", &request).await
    }

    /// Stop a secure cloud rental
    ///
    /// Terminates the provider instance, finalizes billing, and returns the total cost.
    pub async fn stop_secure_cloud_rental(
        &self,
        rental_id: &str,
    ) -> Result<crate::types::StopSecureCloudRentalResponse> {
        let path = format!("/secure-cloud/rentals/{}/stop", rental_id);
        self.post(&path, &serde_json::json!({})).await
    }

    // ===== Volume Management =====

    /// List all volumes for the authenticated user
    ///
    /// Returns all volumes including their status, size, and cost information.
    pub async fn list_volumes(&self) -> Result<crate::types::ListVolumesResponse> {
        self.get("/secure-cloud/volumes").await
    }

    /// Create a new volume
    ///
    /// Creates a block storage volume that can be attached to secure cloud rentals.
    /// Volume names are unique per user (case-insensitive).
    ///
    /// # Arguments
    ///
    /// * `request` - Volume creation parameters including name, size, provider, and region
    pub async fn create_volume(
        &self,
        request: crate::types::CreateVolumeRequest,
    ) -> Result<crate::types::VolumeResponse> {
        self.post("/secure-cloud/volumes", &request).await
    }

    /// Delete a volume
    ///
    /// Permanently deletes a volume. The volume must be detached before deletion.
    ///
    /// # Arguments
    ///
    /// * `volume_id` - The volume ID to delete
    pub async fn delete_volume(&self, volume_id: &str) -> Result<()> {
        let path = format!("/secure-cloud/volumes/{}", volume_id);
        let response = self.delete_empty(&path).await?;
        if response.status().is_success() {
            Ok(())
        } else {
            self.handle_error_response(response).await
        }
    }

    /// Attach a volume to a rental
    ///
    /// Attaches a volume to a running secure cloud rental.
    /// The volume and rental must be in the same provider and region.
    ///
    /// # Arguments
    ///
    /// * `volume_id` - The volume ID to attach
    /// * `request` - Attachment parameters including the rental ID
    pub async fn attach_volume(
        &self,
        volume_id: &str,
        request: crate::types::AttachVolumeRequest,
    ) -> Result<crate::types::VolumeOperationResponse> {
        let path = format!("/secure-cloud/volumes/{}/attach", volume_id);
        self.post(&path, &request).await
    }

    /// Detach a volume from its current rental
    ///
    /// Detaches a volume from its currently attached rental.
    ///
    /// # Arguments
    ///
    /// * `volume_id` - The volume ID to detach
    pub async fn detach_volume(
        &self,
        volume_id: &str,
    ) -> Result<crate::types::VolumeOperationResponse> {
        let path = format!("/secure-cloud/volumes/{}/detach", volume_id);
        self.post(&path, &serde_json::json!({})).await
    }

    // ===== CPU-Only Secure Cloud =====

    /// List CPU-only offerings from secure cloud providers
    ///
    /// Returns CPU-only instances (no GPU) from providers like Hyperstack.
    /// These have flat hourly rates rather than per-GPU pricing.
    pub async fn list_cpu_offerings(&self) -> Result<Vec<crate::types::CpuOffering>> {
        let response: crate::types::ListCpuOfferingsResponse = self
            .get("/secure-cloud/cpu-prices?available_only=true")
            .await?;
        Ok(response.nodes)
    }

    /// List CPU-only rentals for the authenticated user
    ///
    /// Returns all CPU-only secure cloud rentals including their status,
    /// IP addresses, and cost information.
    pub async fn list_cpu_rentals(&self) -> Result<crate::types::ListSecureCloudRentalsResponse> {
        self.get("/secure-cloud/cpu-rentals").await
    }

    /// Start a CPU-only rental
    ///
    /// Deploys a CPU-only instance via datacenter provider and registers
    /// it with the billing service for incremental charging.
    pub async fn start_cpu_rental(
        &self,
        request: crate::types::StartSecureCloudRentalRequest,
    ) -> Result<crate::types::SecureCloudRentalResponse> {
        self.post("/secure-cloud/cpu-rentals/start", &request).await
    }

    /// Stop a CPU-only rental
    ///
    /// Terminates the CPU instance, finalizes billing, and returns the total cost.
    pub async fn stop_cpu_rental(
        &self,
        rental_id: &str,
    ) -> Result<crate::types::StopSecureCloudRentalResponse> {
        let path = format!("/secure-cloud/cpu-rentals/{}/stop", rental_id);
        self.post(&path, &serde_json::json!({})).await
    }

    // ===== SSH Key Management =====

    /// Get the authenticated user's registered SSH key
    ///
    /// Returns None if no SSH key is registered yet.
    pub async fn get_user_ssh_key(&self) -> Result<Option<crate::types::SshKeyResponse>> {
        match self
            .get::<Option<crate::types::SshKeyResponse>>("/ssh-keys")
            .await
        {
            Ok(Some(key)) => Ok(Some(key)),
            Ok(None) => Ok(None),
            Err(e) => Err(e),
        }
    }

    // ===== Deployment Management Methods =====

    /// Create a new deployment
    ///
    /// Deploys a container to the K3s cluster with the specified configuration.
    /// Deployments are idempotent by `instance_name` - calling this multiple times
    /// with the same instance_name will return the existing deployment if it's active.
    ///
    /// # Arguments
    ///
    /// * `request` - The deployment configuration
    ///
    /// # Returns
    ///
    /// Returns the created or existing deployment information including the public URL
    ///
    /// # Errors
    ///
    /// * `ApiError::InvalidRequest` - Invalid instance name, image, port, or resources
    /// * `ApiError::Authorization` - Insufficient permissions
    /// * `ApiError::ServiceUnavailable` - K8s cluster unavailable
    ///
    /// # Example
    ///
    /// ```no_run
    /// use basilica_sdk::{ClientBuilder, CreateDeploymentRequest};
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = ClientBuilder::default()
    ///     .with_api_key("your-api-key")
    ///     .build()?;
    ///
    /// let request = CreateDeploymentRequest {
    ///     instance_name: "my-nginx".to_string(),
    ///     image: "nginx:latest".to_string(),
    ///     replicas: 2,
    ///     port: 80,
    ///     command: None,
    ///     args: None,
    ///     env: None,
    ///     resources: None,
    ///     ttl_seconds: None,
    ///     public: true,
    ///     storage: None,
    ///     health_check: None,
    ///     enable_billing: true,
    ///     queue_name: None,
    ///     suspended: false,
    ///     priority: None,
    /// };
    ///
    /// let deployment = client.create_deployment(request).await?;
    /// println!("Deployment URL: {}", deployment.url);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn create_deployment(
        &self,
        request: CreateDeploymentRequest,
    ) -> Result<DeploymentResponse> {
        self.post("/deployments", &request).await
    }

    /// Get deployment status by instance name
    ///
    /// Retrieves the current status of a deployment including replica counts,
    /// pod information, and the public URL.
    ///
    /// # Arguments
    ///
    /// * `instance_name` - The unique instance name of the deployment
    ///
    /// # Returns
    ///
    /// Returns the deployment status with current replica counts and pod details
    ///
    /// # Errors
    ///
    /// * `ApiError::NotFound` - Deployment doesn't exist or doesn't belong to user
    /// * `ApiError::Authorization` - Insufficient permissions
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use basilica_sdk::ClientBuilder;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let client = ClientBuilder::default().with_api_key("key").build()?;
    /// let deployment = client.get_deployment("my-nginx").await?;
    /// println!("State: {}, Ready: {}/{}",
    ///     deployment.state,
    ///     deployment.replicas.ready,
    ///     deployment.replicas.desired
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_deployment(&self, instance_name: &str) -> Result<DeploymentResponse> {
        let path = format!("/deployments/{}", instance_name);
        self.get(&path).await
    }

    /// Delete a deployment
    ///
    /// Deletes the deployment and all associated K8s resources (pods, services, etc.).
    /// This also removes the routing configuration from Envoy.
    ///
    /// # Arguments
    ///
    /// * `instance_name` - The unique instance name of the deployment to delete
    ///
    /// # Returns
    ///
    /// Returns confirmation of deletion initiation
    ///
    /// # Errors
    ///
    /// * `ApiError::NotFound` - Deployment doesn't exist or doesn't belong to user
    /// * `ApiError::Authorization` - Insufficient permissions
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use basilica_sdk::ClientBuilder;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let client = ClientBuilder::default().with_api_key("key").build()?;
    /// let result = client.delete_deployment("my-nginx").await?;
    /// println!("Deletion initiated: {}", result.message);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn delete_deployment(&self, instance_name: &str) -> Result<DeleteDeploymentResponse> {
        let path = format!("/deployments/{}", instance_name);
        self.delete(&path).await
    }

    /// List all deployments for the authenticated user
    ///
    /// Returns a summary of all active deployments including their state and URLs.
    ///
    /// # Returns
    ///
    /// Returns a list of deployment summaries with total count
    ///
    /// # Errors
    ///
    /// * `ApiError::Authorization` - Insufficient permissions
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use basilica_sdk::ClientBuilder;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let client = ClientBuilder::default().with_api_key("key").build()?;
    /// let deployments = client.list_deployments().await?;
    /// println!("Total deployments: {}", deployments.total);
    /// for deployment in deployments.deployments {
    ///     println!("{}: {} ({})", deployment.instance_name, deployment.state, deployment.url);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn list_deployments(&self) -> Result<DeploymentListResponse> {
        self.get("/deployments").await
    }

    /// Get deployment logs
    ///
    /// Fetches logs from a deployment's pods.
    ///
    /// # Arguments
    ///
    /// * `instance_name` - The deployment instance name
    /// * `follow` - Whether to stream logs continuously
    /// * `tail` - Optional number of lines to return from the end
    ///
    /// # Returns
    ///
    /// Returns a Response that can be used to read logs
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use basilica_sdk::ClientBuilder;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let client = ClientBuilder::default().with_api_key("key").build()?;
    /// let logs = client.get_deployment_logs("my-app", false, Some(100)).await?;
    /// let body = logs.text().await?;
    /// println!("Logs: {}", body);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_deployment_logs(
        &self,
        instance_name: &str,
        follow: bool,
        tail: Option<u32>,
    ) -> Result<reqwest::Response> {
        let url = format!("{}/deployments/{}/logs", self.base_url, instance_name);

        let mut params = Vec::new();
        if follow {
            params.push(("follow", follow.to_string()));
        }
        if let Some(t) = tail {
            params.push(("tail", t.to_string()));
        }

        let mut request = self.http_client.get(&url);
        if !params.is_empty() {
            request = request.query(&params);
        }
        let request = self.apply_auth(request).await?;

        request.send().await.map_err(ApiError::HttpClient)
    }

    /// Get deployment events
    ///
    /// Fetches Kubernetes events related to a deployment's pods.
    /// Useful for debugging failed or unhealthy deployments.
    ///
    /// # Arguments
    ///
    /// * `instance_name` - The deployment instance name
    /// * `limit` - Optional maximum number of events to return
    ///
    /// # Returns
    ///
    /// Returns deployment events sorted by timestamp (newest first)
    pub async fn get_deployment_events(
        &self,
        instance_name: &str,
        limit: Option<u32>,
    ) -> Result<DeploymentEventsResponse> {
        let url = format!("{}/deployments/{}/events", self.base_url, instance_name);
        let mut request = self.http_client.get(&url);

        if let Some(l) = limit {
            request = request.query(&[("limit", l)]);
        }

        let request = self.apply_auth(request).await?;
        let response = request.send().await.map_err(ApiError::HttpClient)?;
        self.handle_response(response).await
    }

    /// Scale a deployment to a specific number of replicas
    ///
    /// Updates the desired replica count for a deployment.
    ///
    /// # Arguments
    ///
    /// * `instance_name` - The deployment instance name
    /// * `replicas` - The desired number of replicas
    ///
    /// # Returns
    ///
    /// Returns the updated deployment state
    ///
    /// # Errors
    ///
    /// * `ApiError::NotFound` - Deployment doesn't exist
    /// * `ApiError::InvalidRequest` - Invalid replica count
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use basilica_sdk::ClientBuilder;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let client = ClientBuilder::default().with_api_key("key").build()?;
    /// let result = client.scale_deployment("my-app", 3).await?;
    /// println!("Scaled to {} replicas", result.replicas.desired);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn scale_deployment(
        &self,
        instance_name: &str,
        replicas: u32,
    ) -> Result<DeploymentResponse> {
        let path = format!("/deployments/{}/scale", instance_name);
        let request = ScaleDeploymentRequest { replicas };
        self.post(&path, &request).await
    }

    /// Wait for a deployment to become ready
    ///
    /// Polls the deployment status until it reaches a ready state, fails, or times out.
    /// Optionally calls a progress callback on each status update.
    ///
    /// # Arguments
    ///
    /// * `instance_name` - The deployment instance name
    /// * `options` - Wait configuration (timeout, poll interval)
    /// * `on_progress` - Optional callback invoked on each status poll with (phase, status)
    ///
    /// # Returns
    ///
    /// Returns `WaitResult::Ready` with the final deployment status when ready,
    /// `WaitResult::Failed` if the deployment enters a failed state,
    /// or `WaitResult::Timeout` if the timeout is exceeded.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use basilica_sdk::{ClientBuilder, WaitOptions, WaitResult};
    ///
    /// let client = ClientBuilder::default()
    ///     .with_api_key("key")
    ///     .build()?;
    ///
    /// // Wait with default options (300s timeout)
    /// let result = client.wait_for_ready("my-app", WaitOptions::default(), None::<fn(Option<&str>, &_)>).await?;
    ///
    /// // Wait with progress callback
    /// let result = client.wait_for_ready(
    ///     "my-app",
    ///     WaitOptions::with_timeout(120),
    ///     Some(|phase, status| {
    ///         println!("Phase: {} ({}/{})",
    ///             phase.unwrap_or("unknown"),
    ///             status.replicas.ready,
    ///             status.replicas.desired
    ///         );
    ///     }),
    /// ).await?;
    ///
    /// match result {
    ///     WaitResult::Ready(deployment) => println!("Ready at {}", deployment.url),
    ///     WaitResult::Failed { reason } => eprintln!("Failed: {}", reason),
    ///     WaitResult::Timeout { last_state, .. } => eprintln!("Timeout in state: {}", last_state),
    /// }
    /// ```
    pub async fn wait_for_ready<F>(
        &self,
        instance_name: &str,
        options: WaitOptions,
        on_progress: Option<F>,
    ) -> Result<WaitResult>
    where
        F: Fn(Option<&str>, &DeploymentResponse),
    {
        use std::time::{Duration, Instant};

        let start = Instant::now();
        let timeout = Duration::from_secs(options.timeout_secs);
        let poll_interval = Duration::from_secs(options.poll_interval_secs);
        let mut last_phase: Option<String> = None;
        let mut last_state: String = "Unknown".to_string();

        loop {
            if start.elapsed() > timeout {
                return Ok(WaitResult::Timeout {
                    last_state,
                    last_phase,
                });
            }

            let status = self.get_deployment(instance_name).await?;

            // Track state for timeout reporting
            last_state = status.state.clone();

            // Call progress callback if phase changed or on first poll
            if let Some(ref callback) = on_progress {
                let current_phase = status.phase.as_deref();
                if last_phase.as_deref() != current_phase {
                    callback(current_phase, &status);
                }
            }

            // Always track phase for timeout reporting
            last_phase = status.phase.clone();

            // Check for ready state (Active or Running with sufficient replicas)
            if matches!(status.state.as_str(), "Active" | "Running")
                && status.replicas.ready >= status.replicas.desired
            {
                return Ok(WaitResult::Ready(Box::new(status)));
            }

            // Check for failed state
            if status.state == "Failed" {
                let reason = status
                    .message
                    .unwrap_or_else(|| "Unknown error".to_string());
                return Ok(WaitResult::Failed { reason });
            }

            // Check for terminating state
            if status.state == "Terminating" || status.phase.as_deref() == Some("terminating") {
                return Ok(WaitResult::Failed {
                    reason: "Deployment is being terminated".to_string(),
                });
            }

            // Dynamic sleep based on phase
            let sleep_duration = match status.phase.as_deref() {
                Some("scheduling") | Some("pulling") => Duration::from_secs(10),
                Some("storage_sync") => Duration::from_secs(3),
                _ => poll_interval,
            };

            tokio::time::sleep(sleep_duration).await;
        }
    }

    /// Wait for a deployment to become ready with default progress output
    ///
    /// This is a convenience method that waits for a deployment and prints
    /// progress updates to stdout, similar to the CLI behavior.
    ///
    /// # Arguments
    ///
    /// * `instance_name` - The deployment instance name
    /// * `timeout_secs` - Maximum time to wait in seconds
    ///
    /// # Returns
    ///
    /// Returns the ready deployment or an error
    ///
    /// # Example
    ///
    /// ```ignore
    /// use basilica_sdk::ClientBuilder;
    ///
    /// let client = ClientBuilder::default()
    ///     .with_api_key("key")
    ///     .build()?;
    ///
    /// let deployment = client.wait_for_ready_with_output("my-app", 120).await?;
    /// println!("Deployment ready at {}", deployment.url);
    /// ```
    pub async fn wait_for_ready_with_output(
        &self,
        instance_name: &str,
        timeout_secs: u64,
    ) -> Result<DeploymentResponse> {
        let options = WaitOptions::with_timeout(timeout_secs);

        let result = self
            .wait_for_ready(
                instance_name,
                options,
                Some(|phase: Option<&str>, status: &DeploymentResponse| {
                    let phase_msg = format_phase_message(phase.unwrap_or("pending"));
                    let replica_info =
                        format!("{}/{}", status.replicas.ready, status.replicas.desired);
                    println!(
                        "[{}] {} (replicas: {})",
                        instance_name, phase_msg, replica_info
                    );

                    // Show progress for storage sync
                    if phase == Some("storage_sync") {
                        if let Some(ref progress) = status.progress {
                            if let Some(pct) = progress.percentage {
                                println!("  Storage sync: {:.1}%", pct);
                            }
                        }
                    }
                }),
            )
            .await?;

        match result {
            WaitResult::Ready(deployment) => {
                println!("[{}] Deployment ready!", instance_name);
                Ok(*deployment)
            }
            WaitResult::Failed { reason } => Err(ApiError::Internal {
                message: format!("Deployment failed: {}", reason),
            }),
            WaitResult::Timeout {
                last_state,
                last_phase,
            } => Err(ApiError::Internal {
                message: format!(
                    "Timeout waiting for deployment (state: {}, phase: {})",
                    last_state,
                    last_phase.unwrap_or_else(|| "unknown".to_string())
                ),
            }),
        }
    }

    // ============================================================================
    // Share Token Management
    // ============================================================================

    /// Regenerate share token for a private deployment.
    ///
    /// Creates a new token and invalidates any previous token. The raw token
    /// is only returned once and cannot be retrieved later.
    ///
    /// # Arguments
    /// * `instance_name` - The deployment instance name
    ///
    /// # Errors
    /// * `ApiError::BadRequest` - Deployment is public (public deployments don't need tokens)
    /// * `ApiError::NotFound` - Deployment not found or not owned by user
    /// * `ApiError::Authentication` - Invalid or missing authentication
    pub async fn regenerate_share_token(
        &self,
        instance_name: &str,
    ) -> Result<RegenerateShareTokenResponse> {
        let path = format!(
            "/deployments/{}/share-token",
            urlencoding::encode(instance_name)
        );
        self.post_empty(&path).await
    }

    /// Check if a share token exists for a deployment.
    ///
    /// # Arguments
    /// * `instance_name` - The deployment instance name
    ///
    /// # Errors
    /// * `ApiError::BadRequest` - Deployment is public
    /// * `ApiError::NotFound` - Deployment not found or not owned by user
    /// * `ApiError::Authentication` - Invalid or missing authentication
    pub async fn get_share_token_status(
        &self,
        instance_name: &str,
    ) -> Result<ShareTokenStatusResponse> {
        let path = format!(
            "/deployments/{}/share-token",
            urlencoding::encode(instance_name)
        );
        self.get(&path).await
    }

    /// Revoke (delete) the share token for a deployment.
    ///
    /// After revocation, the deployment will not be accessible via the share URL
    /// until a new token is generated with `regenerate_share_token`.
    ///
    /// # Arguments
    /// * `instance_name` - The deployment instance name
    ///
    /// # Errors
    /// * `ApiError::BadRequest` - Deployment is public
    /// * `ApiError::NotFound` - Deployment not found or not owned by user
    /// * `ApiError::Authentication` - Invalid or missing authentication
    pub async fn delete_share_token(
        &self,
        instance_name: &str,
    ) -> Result<DeleteShareTokenResponse> {
        let path = format!(
            "/deployments/{}/share-token",
            urlencoding::encode(instance_name)
        );
        self.delete(&path).await
    }

    // ============================================================================
    // Public Deployment Metadata
    // ============================================================================

    /// Enroll or unenroll a deployment in public metadata exposure.
    ///
    /// When enrolled, non-sensitive deployment metadata becomes publicly queryable
    /// via the unauthenticated `/public/deployments/:name/metadata` endpoint.
    ///
    /// # Arguments
    /// * `instance_name` - The deployment instance name
    /// * `enabled` - Whether to enable or disable metadata enrollment
    ///
    /// # Errors
    /// * `ApiError::NotFound` - Deployment not found or not owned by user
    /// * `ApiError::Conflict` - Another deployment with same instance name already enrolled
    pub async fn enroll_metadata(
        &self,
        instance_name: &str,
        enabled: bool,
    ) -> Result<EnrollMetadataResponse> {
        let path = format!(
            "/deployments/{}/enroll-metadata",
            urlencoding::encode(instance_name)
        );
        let body = EnrollMetadataRequest { enabled };
        self.post(&path, &body).await
    }

    /// Check the current metadata enrollment status of a deployment.
    ///
    /// # Arguments
    /// * `instance_name` - The deployment instance name
    ///
    /// # Errors
    /// * `ApiError::NotFound` - Deployment not found or not owned by user
    pub async fn get_enrollment_status(
        &self,
        instance_name: &str,
    ) -> Result<EnrollMetadataResponse> {
        let path = format!(
            "/deployments/{}/enroll-metadata",
            urlencoding::encode(instance_name)
        );
        self.get(&path).await
    }

    /// Fetch public metadata for a deployment (no authentication required).
    ///
    /// Returns non-sensitive metadata for deployments that have opted-in
    /// via `enroll_metadata`.
    ///
    /// # Arguments
    /// * `instance_name` - The deployment instance name
    ///
    /// # Errors
    /// * `ApiError::NotFound` - Deployment not found or metadata not enrolled
    pub async fn get_public_deployment_metadata(
        &self,
        instance_name: &str,
    ) -> Result<PublicDeploymentMetadataResponse> {
        let path = format!(
            "/public/deployments/{}/metadata",
            urlencoding::encode(instance_name)
        );
        self.get_public(&path).await
    }

    // ===== Private Helper Methods =====

    /// Apply authentication to request
    /// Uses TokenManager for automatic token refresh
    async fn apply_auth(&self, request: RequestBuilder) -> Result<RequestBuilder> {
        let token =
            self.token_manager
                .get_access_token()
                .await
                .map_err(|e| ApiError::Internal {
                    message: format!("Failed to get access token: {}", e),
                })?;
        Ok(request.header("Authorization", format!("Bearer {}", token)))
    }

    /// Generic GET request
    async fn get<T: DeserializeOwned>(&self, path: &str) -> Result<T> {
        let url = format!("{}{}", self.base_url, path);
        let request = self.http_client.get(&url);
        let request = self.apply_auth(request).await?;

        let response = request.send().await.map_err(ApiError::HttpClient)?;
        self.handle_response(response).await
    }

    /// Unauthenticated GET request for public endpoints
    async fn get_public<T: DeserializeOwned>(&self, path: &str) -> Result<T> {
        let url = format!("{}{}", self.base_url, path);
        let request = self.http_client.get(&url);
        let response = request.send().await.map_err(ApiError::HttpClient)?;
        self.handle_response(response).await
    }

    /// Generic POST request
    async fn post<B: Serialize, T: DeserializeOwned>(&self, path: &str, body: &B) -> Result<T> {
        let url = format!("{}{}", self.base_url, path);
        let request = self.http_client.post(&url).json(body);
        let request = self.apply_auth(request).await?;

        let response = request.send().await.map_err(ApiError::HttpClient)?;
        self.handle_response(response).await
    }

    /// Generic POST request without body
    async fn post_empty<T: DeserializeOwned>(&self, path: &str) -> Result<T> {
        let url = format!("{}{}", self.base_url, path);
        let request = self.http_client.post(&url);
        let request = self.apply_auth(request).await?;

        let response = request.send().await.map_err(ApiError::HttpClient)?;
        self.handle_response(response).await
    }

    /// Generic DELETE request without body
    async fn delete_empty(&self, path: &str) -> Result<Response> {
        let url = format!("{}{}", self.base_url, path);
        let request = self.http_client.delete(&url);
        let request = self.apply_auth(request).await?;

        let response = request.send().await.map_err(ApiError::HttpClient)?;
        Ok(response)
    }

    /// Generic DELETE request with typed response
    async fn delete<T: DeserializeOwned>(&self, path: &str) -> Result<T> {
        let url = format!("{}{}", self.base_url, path);
        let request = self.http_client.delete(&url);
        let request = self.apply_auth(request).await?;

        let response = request.send().await.map_err(ApiError::HttpClient)?;
        self.handle_response(response).await
    }

    /// Handle successful response
    async fn handle_response<T: DeserializeOwned>(&self, response: Response) -> Result<T> {
        if response.status().is_success() {
            response.json().await.map_err(ApiError::HttpClient)
        } else {
            self.handle_error_response(response).await
        }
    }

    /// Handle error response
    async fn handle_error_response<T>(&self, response: Response) -> Result<T> {
        let status = response.status();
        let error_text = response.text().await.unwrap_or_default();

        // Try to parse error response
        if let Ok(error_response) = serde_json::from_str::<ErrorResponse>(&error_text) {
            match status {
                StatusCode::UNAUTHORIZED => {
                    // Distinguish between missing auth and expired/invalid auth based on error code
                    match error_response.error.code.as_str() {
                        "BASILICA_API_AUTH_MISSING" => Err(ApiError::MissingAuthentication {
                            message: error_response.error.message,
                        }),
                        _ => Err(ApiError::Authentication {
                            message: error_response.error.message,
                        }),
                    }
                }
                StatusCode::FORBIDDEN => Err(ApiError::Authorization {
                    message: error_response.error.message,
                }),
                StatusCode::TOO_MANY_REQUESTS => Err(ApiError::RateLimitExceeded),
                StatusCode::NOT_FOUND => Err(ApiError::NotFound {
                    resource: error_response.error.message,
                }),
                StatusCode::BAD_REQUEST => Err(ApiError::BadRequest {
                    message: error_response.error.message,
                }),
                StatusCode::CONFLICT => Err(ApiError::Conflict {
                    message: error_response.error.message,
                }),
                _ => Err(ApiError::Internal {
                    message: error_response.error.message,
                }),
            }
        } else {
            // Fallback if we can't parse the error
            match status {
                StatusCode::UNAUTHORIZED => Err(ApiError::Authentication {
                    message: "Authentication failed".into(),
                }),
                StatusCode::FORBIDDEN => Err(ApiError::Authorization {
                    message: "Access forbidden".into(),
                }),
                StatusCode::TOO_MANY_REQUESTS => Err(ApiError::RateLimitExceeded),
                StatusCode::NOT_FOUND => Err(ApiError::NotFound {
                    resource: "Resource not found".into(),
                }),
                StatusCode::BAD_REQUEST => Err(ApiError::BadRequest {
                    message: error_text,
                }),
                StatusCode::CONFLICT => Err(ApiError::Conflict {
                    message: error_text,
                }),
                _ => Err(ApiError::Internal {
                    message: format!("Request failed with status {status}: {error_text}"),
                }),
            }
        }
    }
}

/// Format human-readable phase message for progress output
fn format_phase_message(phase: &str) -> &'static str {
    match phase {
        "pending" => "Waiting for scheduler...",
        "scheduling" => "Finding suitable node...",
        "pulling" => "Pulling container image...",
        "initializing" => "Running init containers...",
        "storage_sync" => "Syncing storage volume...",
        "starting" => "Starting application...",
        "health_check" => "Running health checks...",
        "ready" => "Deployment ready!",
        "degraded" => "Deployment degraded",
        "failed" => "Deployment failed",
        "terminating" => "Terminating...",
        _ => "Processing...",
    }
}

/// Builder for constructing a BasilicaClient with custom configuration
#[derive(Default)]
pub struct ClientBuilder {
    base_url: Option<String>,
    access_token: Option<String>,
    refresh_token: Option<String>,
    timeout: Option<Duration>,
    connect_timeout: Option<Duration>,
    pool_max_idle_per_host: Option<usize>,
    use_file_auth: bool,
    api_key: Option<String>,
}

impl ClientBuilder {
    /// Create a new builder with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the base URL for the API
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Set tokens for direct authentication (both tokens required)
    pub fn with_tokens(
        mut self,
        access_token: impl Into<String>,
        refresh_token: impl Into<String>,
    ) -> Self {
        self.access_token = Some(access_token.into());
        self.refresh_token = Some(refresh_token.into());
        self.use_file_auth = false;
        self
    }

    /// Use file-based authentication (reads tokens from ~/.local/share/basilica/)
    pub fn with_file_auth(mut self) -> Self {
        self.use_file_auth = true;
        self.access_token = None;
        self.refresh_token = None;
        self
    }

    /// Set the request timeout
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set the connection timeout
    pub fn connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = Some(timeout);
        self
    }

    /// Set the maximum idle connections per host
    pub fn pool_max_idle_per_host(mut self, max: usize) -> Self {
        self.pool_max_idle_per_host = Some(max);
        self
    }

    /// Use API key for authentication (from provided string)
    pub fn with_api_key(mut self, api_key: &str) -> Self {
        self.api_key = Some(api_key.to_string());
        self
    }

    /// Build the client with automatic authentication detection
    /// This will automatically find and use CLI tokens if available
    pub async fn build_auto(self) -> Result<BasilicaClient> {
        let base_url = self.base_url.unwrap_or_else(|| DEFAULT_API_URL.to_string());

        // Always try file-based auth for auto mode
        let token_manager = TokenManager::new_file_based().map_err(|e| ApiError::Internal {
            message: format!("Failed to create file-based token manager: {}", e),
        })?;

        let timeout = self
            .timeout
            .unwrap_or(Duration::from_secs(DEFAULT_TIMEOUT_SECS));

        BasilicaClient::new(base_url, timeout, Arc::new(token_manager))
    }

    /// Build the client
    pub fn build(self) -> Result<BasilicaClient> {
        let base_url = self.base_url.unwrap_or_else(|| DEFAULT_API_URL.to_string());

        // Create token manager based on auth configuration
        let token_manager = if let Some(api_key) = self.api_key {
            // API key takes precedence
            TokenManager::new_api_key(api_key)
        } else if self.use_file_auth {
            // File-based auth (also checks for BASILICA_API_KEY env var)
            TokenManager::new_file_based().map_err(|e| ApiError::Internal {
                message: format!("Failed to create file-based token manager: {}", e),
            })?
        } else if let (Some(access_token), Some(refresh_token)) =
            (self.access_token, self.refresh_token)
        {
            TokenManager::new_direct(access_token, refresh_token)
        } else {
            return Err(ApiError::InvalidRequest {
                message: "Either use with_tokens() with both access and refresh tokens, with_file_auth(), or with_api_key()"
                    .into(),
            });
        };

        let timeout = self
            .timeout
            .unwrap_or(Duration::from_secs(DEFAULT_TIMEOUT_SECS));

        BasilicaClient::new(base_url, timeout, Arc::new(token_manager))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{body_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn test_health_check() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/health"))
            .and(header("Authorization", "Bearer test-token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2024-01-01T00:00:00Z",
                "healthy_validators": 10,
                "total_validators": 10,
            })))
            .mount(&mock_server)
            .await;

        let client = ClientBuilder::default()
            .base_url(mock_server.uri())
            .with_tokens("test-token", "refresh-token")
            .build()
            .unwrap();
        let health = client.health_check().await.unwrap();

        assert_eq!(health.status, "healthy");
        assert_eq!(health.version, "1.0.0");
    }

    #[tokio::test]
    async fn test_token_auth_with_refresh() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/health"))
            .and(header("Authorization", "Bearer test-token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2024-01-01T00:00:00Z",
                "healthy_validators": 10,
                "total_validators": 10,
            })))
            .mount(&mock_server)
            .await;

        let client = ClientBuilder::default()
            .base_url(mock_server.uri())
            .with_tokens("test-token", "refresh-token")
            .build()
            .unwrap();

        let result = client.health_check().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_error_handling() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/health"))
            .and(header("Authorization", "Bearer test-token"))
            .respond_with(ResponseTemplate::new(401).set_body_json(json!({
                "error": {
                    "code": "BASILICA_API_AUTH_MISSING",
                    "message": "Authentication required",
                    "timestamp": "2024-01-01T00:00:00Z",
                    "retryable": false,
                }
            })))
            .mount(&mock_server)
            .await;

        let client = ClientBuilder::default()
            .base_url(mock_server.uri())
            .with_tokens("test-token", "refresh-token")
            .build()
            .unwrap();
        let result = client.health_check().await;

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ApiError::MissingAuthentication { .. }
        ));
    }

    #[test]
    fn test_builder_requires_auth() {
        let result = ClientBuilder::default().build();
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ApiError::InvalidRequest { .. }
        ));
    }

    #[test]
    fn test_builder_with_all_options() {
        let client = ClientBuilder::default()
            .base_url("https://api.basilica.ai")
            .with_tokens("test-token", "refresh-token")
            .timeout(Duration::from_secs(60))
            .connect_timeout(Duration::from_secs(10))
            .pool_max_idle_per_host(100)
            .build();

        assert!(client.is_ok());
    }

    #[tokio::test]
    async fn test_create_job() {
        use crate::jobs::{CreateJobRequest, JobGpuRequirements, JobResources};

        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v2/jobs"))
            .and(header("Authorization", "Bearer test-token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "job_id": "job-123",
            })))
            .mount(&mock_server)
            .await;

        let client = ClientBuilder::default()
            .base_url(mock_server.uri())
            .with_tokens("test-token", "refresh-token")
            .build()
            .unwrap();

        let request = CreateJobRequest {
            image: "pytorch/pytorch:latest".to_string(),
            command: vec![],
            args: vec![],
            env: vec![],
            resources: JobResources {
                cpu: "4".to_string(),
                memory: "8Gi".to_string(),
                gpus: JobGpuRequirements {
                    count: 1,
                    model: vec!["H100".to_string()],
                },
            },
            ttl_seconds: 3600,
            name: Some("training-job".to_string()),
            namespace: None,
            ports: vec![],
            storage: None,
        };

        let response = client.create_job(request).await.unwrap();
        assert_eq!(response.job_id, "job-123");
    }

    #[tokio::test]
    async fn test_get_job_status() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/v2/jobs/job-123"))
            .and(header("Authorization", "Bearer test-token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "job_id": "job-123",
                "status": {
                    "phase": "Running",
                    "message": "Job is running",
                    "endpoints": ["http://example.com:8080"]
                }
            })))
            .mount(&mock_server)
            .await;

        let client = ClientBuilder::default()
            .base_url(mock_server.uri())
            .with_tokens("test-token", "refresh-token")
            .build()
            .unwrap();

        let response = client.get_job_status("job-123").await.unwrap();
        assert_eq!(response.job_id, "job-123");
        assert_eq!(response.status.phase, "Running");
        assert_eq!(response.status.endpoints.len(), 1);
    }

    #[tokio::test]
    async fn test_suspend_resume_job() {
        let mock_server = MockServer::start().await;

        // Mock suspend endpoint
        Mock::given(method("POST"))
            .and(path("/v2/jobs/job-123/suspend"))
            .and(header("Authorization", "Bearer test-token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "job_id": "job-123",
            })))
            .mount(&mock_server)
            .await;

        // Mock resume endpoint
        Mock::given(method("POST"))
            .and(path("/v2/jobs/job-123/resume"))
            .and(header("Authorization", "Bearer test-token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "job_id": "job-123",
            })))
            .mount(&mock_server)
            .await;

        let client = ClientBuilder::default()
            .base_url(mock_server.uri())
            .with_tokens("test-token", "refresh-token")
            .build()
            .unwrap();

        // Test suspend
        let suspend_response = client.suspend_job("job-123").await.unwrap();
        assert_eq!(suspend_response.job_id, "job-123");

        // Test resume
        let resume_response = client.resume_job("job-123").await.unwrap();
        assert_eq!(resume_response.job_id, "job-123");
    }

    #[tokio::test]
    async fn test_read_job_file() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v2/jobs/job-123/files"))
            .and(header("Authorization", "Bearer test-token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "content": "File contents here",
            })))
            .mount(&mock_server)
            .await;

        let client = ClientBuilder::default()
            .base_url(mock_server.uri())
            .with_tokens("test-token", "refresh-token")
            .build()
            .unwrap();

        let response = client
            .read_job_file("job-123", "/path/to/file.txt")
            .await
            .unwrap();
        assert_eq!(response.content, "File contents here");
    }

    #[tokio::test]
    async fn test_create_deployment() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/deployments"))
            .and(header("Authorization", "Bearer test-token"))
            .respond_with(ResponseTemplate::new(201).set_body_json(json!({
                "instanceName": "my-nginx",
                "userId": "user123",
                "namespace": "u-user123",
                "state": "Pending",
                "url": "http://3.21.154.119:8080/deployments/my-nginx/",
                "replicas": {
                    "desired": 2,
                    "ready": 0
                },
                "createdAt": "2025-10-31T10:00:00Z"
            })))
            .mount(&mock_server)
            .await;

        let client = ClientBuilder::default()
            .base_url(mock_server.uri())
            .with_tokens("test-token", "refresh-token")
            .build()
            .unwrap();

        let request = CreateDeploymentRequest {
            instance_name: "my-nginx".to_string(),
            image: "nginx:latest".to_string(),
            replicas: 2,
            port: 80,
            command: None,
            args: None,
            env: None,
            resources: None,
            ttl_seconds: None,
            public: true,
            storage: None,
            health_check: None,
            enable_billing: true,
            queue_name: None,
            suspended: false,
            priority: None,
            topology_spread: None,
            websocket: None,
            public_metadata: false,
        };

        let response = client.create_deployment(request).await.unwrap();

        assert_eq!(response.instance_name, "my-nginx");
        assert_eq!(response.user_id, "user123");
        assert_eq!(response.namespace, "u-user123");
        assert_eq!(response.state, "Pending");
        assert_eq!(
            response.url,
            "http://3.21.154.119:8080/deployments/my-nginx/"
        );
        assert_eq!(response.replicas.desired, 2);
        assert_eq!(response.replicas.ready, 0);
        assert_eq!(response.created_at, "2025-10-31T10:00:00Z");
    }

    #[tokio::test]
    async fn test_create_deployment_with_resources() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/deployments"))
            .and(body_json(json!({
                "instanceName": "my-app",
                "image": "myapp:v1",
                "replicas": 1,
                "port": 8080,
                "env": {
                    "ENV_VAR": "value"
                },
                "resources": {
                    "cpu": "1000m",
                    "memory": "1Gi"
                },
                "public": true,
                "enableBilling": true,
                "suspended": false,
                "publicMetadata": false
            })))
            .respond_with(ResponseTemplate::new(201).set_body_json(json!({
                "instanceName": "my-app",
                "userId": "user123",
                "namespace": "u-user123",
                "state": "Pending",
                "url": "http://3.21.154.119:8080/deployments/my-app/",
                "replicas": {"desired": 1, "ready": 0},
                "createdAt": "2025-10-31T10:00:00Z"
            })))
            .mount(&mock_server)
            .await;

        let client = ClientBuilder::default()
            .base_url(mock_server.uri())
            .with_tokens("test-token", "refresh-token")
            .build()
            .unwrap();

        let request = CreateDeploymentRequest {
            instance_name: "my-app".to_string(),
            image: "myapp:v1".to_string(),
            replicas: 1,
            port: 8080,
            command: None,
            args: None,
            env: Some(std::collections::HashMap::from([(
                "ENV_VAR".to_string(),
                "value".to_string(),
            )])),
            resources: Some(crate::types::ResourceRequirements {
                cpu: "1000m".to_string(),
                memory: "1Gi".to_string(),
                cpu_request: None,
                memory_request: None,
                gpus: None,
            }),
            ttl_seconds: None,
            public: true,
            storage: None,
            health_check: None,
            enable_billing: true,
            queue_name: None,
            suspended: false,
            priority: None,
            topology_spread: None,
            websocket: None,
            public_metadata: false,
        };

        let response = client.create_deployment(request).await.unwrap();
        assert_eq!(response.instance_name, "my-app");
    }

    #[tokio::test]
    async fn test_create_deployment_idempotent() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/deployments"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "instanceName": "existing-app",
                "userId": "user123",
                "namespace": "u-user123",
                "state": "Active",
                "url": "http://3.21.154.119:8080/deployments/existing-app/",
                "replicas": {"desired": 2, "ready": 2},
                "createdAt": "2025-10-31T09:00:00Z",
                "updatedAt": "2025-10-31T10:00:00Z"
            })))
            .mount(&mock_server)
            .await;

        let client = ClientBuilder::default()
            .base_url(mock_server.uri())
            .with_tokens("test-token", "refresh-token")
            .build()
            .unwrap();

        let request = CreateDeploymentRequest {
            instance_name: "existing-app".to_string(),
            image: "nginx:latest".to_string(),
            replicas: 2,
            port: 80,
            command: None,
            args: None,
            env: None,
            resources: None,
            ttl_seconds: None,
            public: true,
            storage: None,
            health_check: None,
            enable_billing: true,
            queue_name: None,
            suspended: false,
            priority: None,
            topology_spread: None,
            websocket: None,
            public_metadata: false,
        };

        let response = client.create_deployment(request).await.unwrap();
        assert_eq!(response.state, "Active");
        assert_eq!(response.replicas.ready, 2);
        assert!(response.updated_at.is_some());
    }

    #[tokio::test]
    async fn test_get_deployment() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/deployments/my-nginx"))
            .and(header("Authorization", "Bearer test-token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "instanceName": "my-nginx",
                "userId": "user123",
                "namespace": "u-user123",
                "state": "Active",
                "url": "http://3.21.154.119:8080/deployments/my-nginx/",
                "replicas": {"desired": 2, "ready": 2},
                "createdAt": "2025-10-31T09:00:00Z",
                "updatedAt": "2025-10-31T10:00:00Z",
                "pods": [
                    {
                        "name": "my-nginx-5d8f7b9c-abc12",
                        "status": "Running",
                        "node": "ip-172-31-18-204"
                    },
                    {
                        "name": "my-nginx-5d8f7b9c-def34",
                        "status": "Running",
                        "node": "ip-172-31-18-204"
                    }
                ]
            })))
            .mount(&mock_server)
            .await;

        let client = ClientBuilder::default()
            .base_url(mock_server.uri())
            .with_tokens("test-token", "refresh-token")
            .build()
            .unwrap();

        let response = client.get_deployment("my-nginx").await.unwrap();

        assert_eq!(response.instance_name, "my-nginx");
        assert_eq!(response.state, "Active");
        assert_eq!(response.replicas.desired, 2);
        assert_eq!(response.replicas.ready, 2);
        assert!(response.pods.is_some());
        let pods = response.pods.unwrap();
        assert_eq!(pods.len(), 2);
        assert_eq!(pods[0].name, "my-nginx-5d8f7b9c-abc12");
        assert_eq!(pods[0].status, "Running");
        assert_eq!(pods[0].node, Some("ip-172-31-18-204".to_string()));
    }

    #[tokio::test]
    async fn test_get_deployment_not_found() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/deployments/nonexistent"))
            .respond_with(ResponseTemplate::new(404).set_body_json(json!({
                "error": "NOT_FOUND",
                "message": "Deployment not found"
            })))
            .mount(&mock_server)
            .await;

        let client = ClientBuilder::default()
            .base_url(mock_server.uri())
            .with_tokens("test-token", "refresh-token")
            .build()
            .unwrap();

        let result = client.get_deployment("nonexistent").await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ApiError::NotFound { .. }));
    }

    #[tokio::test]
    async fn test_delete_deployment() {
        let mock_server = MockServer::start().await;

        Mock::given(method("DELETE"))
            .and(path("/deployments/my-nginx"))
            .and(header("Authorization", "Bearer test-token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "instanceName": "my-nginx",
                "state": "Terminating",
                "message": "Deployment deletion initiated"
            })))
            .mount(&mock_server)
            .await;

        let client = ClientBuilder::default()
            .base_url(mock_server.uri())
            .with_tokens("test-token", "refresh-token")
            .build()
            .unwrap();

        let response = client.delete_deployment("my-nginx").await.unwrap();

        assert_eq!(response.instance_name, "my-nginx");
        assert_eq!(response.state, "Terminating");
        assert_eq!(response.message, "Deployment deletion initiated");
    }

    #[tokio::test]
    async fn test_list_deployments() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/deployments"))
            .and(header("Authorization", "Bearer test-token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "deployments": [
                    {
                        "instanceName": "my-nginx",
                        "state": "Active",
                        "url": "http://3.21.154.119:8080/deployments/my-nginx/",
                        "replicas": {"desired": 2, "ready": 2},
                        "createdAt": "2025-10-31T09:00:00Z"
                    },
                    {
                        "instanceName": "my-postgres",
                        "state": "Pending",
                        "url": "http://3.21.154.119:8080/deployments/my-postgres/",
                        "replicas": {"desired": 1, "ready": 0},
                        "createdAt": "2025-10-31T10:00:00Z"
                    }
                ],
                "total": 2
            })))
            .mount(&mock_server)
            .await;

        let client = ClientBuilder::default()
            .base_url(mock_server.uri())
            .with_tokens("test-token", "refresh-token")
            .build()
            .unwrap();

        let response = client.list_deployments().await.unwrap();

        assert_eq!(response.total, 2);
        assert_eq!(response.deployments.len(), 2);
        assert_eq!(response.deployments[0].instance_name, "my-nginx");
        assert_eq!(response.deployments[0].state, "Active");
        assert_eq!(response.deployments[1].instance_name, "my-postgres");
        assert_eq!(response.deployments[1].state, "Pending");
    }

    #[tokio::test]
    async fn test_list_deployments_empty() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/deployments"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "deployments": [],
                "total": 0
            })))
            .mount(&mock_server)
            .await;

        let client = ClientBuilder::default()
            .base_url(mock_server.uri())
            .with_tokens("test-token", "refresh-token")
            .build()
            .unwrap();

        let response = client.list_deployments().await.unwrap();

        assert_eq!(response.total, 0);
        assert_eq!(response.deployments.len(), 0);
    }

    #[tokio::test]
    async fn test_create_deployment_validation_error() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/deployments"))
            .respond_with(ResponseTemplate::new(400).set_body_json(json!({
                "error": "INVALID_REQUEST",
                "message": "instance_name must be DNS-safe"
            })))
            .mount(&mock_server)
            .await;

        let client = ClientBuilder::default()
            .base_url(mock_server.uri())
            .with_tokens("test-token", "refresh-token")
            .build()
            .unwrap();

        let request = CreateDeploymentRequest {
            instance_name: "Invalid_Name!".to_string(),
            image: "nginx:latest".to_string(),
            replicas: 2,
            port: 80,
            command: None,
            args: None,
            env: None,
            resources: None,
            ttl_seconds: None,
            public: true,
            storage: None,
            health_check: None,
            enable_billing: true,
            queue_name: None,
            suspended: false,
            priority: None,
            topology_spread: None,
            websocket: None,
            public_metadata: false,
        };

        let result = client.create_deployment(request).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            ApiError::InvalidRequest { message } => {
                assert!(message.contains("DNS-safe"));
            }
            ApiError::BadRequest { message } => {
                assert!(message.contains("DNS-safe"));
            }
            e => panic!("Expected InvalidRequest or BadRequest error, got: {:?}", e),
        }
    }

    #[tokio::test]
    async fn test_enroll_metadata_enable() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/deployments/my-app/enroll-metadata"))
            .and(header("Authorization", "Bearer test-token"))
            .and(body_json(json!({"enabled": true})))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"publicMetadata": true})))
            .mount(&mock_server)
            .await;

        let client = ClientBuilder::default()
            .base_url(mock_server.uri())
            .with_tokens("test-token", "refresh-token")
            .build()
            .unwrap();

        let response = client.enroll_metadata("my-app", true).await.unwrap();
        assert!(response.public_metadata);
    }

    #[tokio::test]
    async fn test_enroll_metadata_disable() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/deployments/my-app/enroll-metadata"))
            .and(header("Authorization", "Bearer test-token"))
            .and(body_json(json!({"enabled": false})))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!({"publicMetadata": false})),
            )
            .mount(&mock_server)
            .await;

        let client = ClientBuilder::default()
            .base_url(mock_server.uri())
            .with_tokens("test-token", "refresh-token")
            .build()
            .unwrap();

        let response = client.enroll_metadata("my-app", false).await.unwrap();
        assert!(!response.public_metadata);
    }

    #[tokio::test]
    async fn test_get_enrollment_status() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/deployments/my-app/enroll-metadata"))
            .and(header("Authorization", "Bearer test-token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"publicMetadata": true})))
            .mount(&mock_server)
            .await;

        let client = ClientBuilder::default()
            .base_url(mock_server.uri())
            .with_tokens("test-token", "refresh-token")
            .build()
            .unwrap();

        let response = client.get_enrollment_status("my-app").await.unwrap();
        assert!(response.public_metadata);
    }

    #[tokio::test]
    async fn test_get_public_deployment_metadata() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/public/deployments/my-app/metadata"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "instanceName": "my-app",
                "image": "nginx",
                "imageTag": "latest",
                "id": "dep-123",
                "uptimeSeconds": 3600,
                "replicas": {"desired": 2, "ready": 2},
                "state": "Active"
            })))
            .mount(&mock_server)
            .await;

        let client = ClientBuilder::default()
            .base_url(mock_server.uri())
            .with_tokens("test-token", "refresh-token")
            .build()
            .unwrap();

        let response = client
            .get_public_deployment_metadata("my-app")
            .await
            .unwrap();
        assert_eq!(response.instance_name, "my-app");
        assert_eq!(response.image, "nginx");
        assert_eq!(response.image_tag, "latest");
        assert_eq!(response.id, "dep-123");
        assert_eq!(response.uptime_seconds, 3600);
        assert_eq!(response.replicas.desired, 2);
        assert_eq!(response.replicas.ready, 2);
        assert_eq!(response.state, "Active");
    }

    #[tokio::test]
    async fn test_enroll_metadata_conflict() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/deployments/my-app/enroll-metadata"))
            .respond_with(ResponseTemplate::new(409).set_body_json(json!({
                "error": {
                    "code": "BASILICA_API_CONFLICT",
                    "message": "Another deployment with instance name 'my-app' is already enrolled",
                    "timestamp": "2024-01-01T00:00:00Z",
                    "retryable": false
                }
            })))
            .mount(&mock_server)
            .await;

        let client = ClientBuilder::default()
            .base_url(mock_server.uri())
            .with_tokens("test-token", "refresh-token")
            .build()
            .unwrap();

        let result = client.enroll_metadata("my-app", true).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ApiError::Conflict { .. }));
    }

    #[tokio::test]
    async fn test_create_deployment_with_public_metadata() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/deployments"))
            .and(header("Authorization", "Bearer test-token"))
            .respond_with(ResponseTemplate::new(201).set_body_json(json!({
                "instanceName": "my-app",
                "userId": "user123",
                "namespace": "u-user123",
                "state": "Pending",
                "url": "http://example.com/deployments/my-app/",
                "replicas": {"desired": 1, "ready": 0},
                "createdAt": "2025-01-01T00:00:00Z",
                "publicMetadata": true
            })))
            .mount(&mock_server)
            .await;

        let client = ClientBuilder::default()
            .base_url(mock_server.uri())
            .with_tokens("test-token", "refresh-token")
            .build()
            .unwrap();

        let request = CreateDeploymentRequest {
            instance_name: "my-app".to_string(),
            image: "nginx:latest".to_string(),
            replicas: 1,
            port: 80,
            command: None,
            args: None,
            env: None,
            resources: None,
            ttl_seconds: None,
            public: true,
            storage: None,
            health_check: None,
            enable_billing: true,
            queue_name: None,
            suspended: false,
            priority: None,
            topology_spread: None,
            websocket: None,
            public_metadata: true,
        };

        let json_body = serde_json::to_string(&request).unwrap();
        assert!(json_body.contains("\"publicMetadata\":true"));

        let response = client.create_deployment(request).await.unwrap();
        assert!(response.public_metadata);
    }
}
