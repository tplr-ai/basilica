//! Training Session API routes.
//!
//! Provides REST API endpoints for managing training session lifecycle:
//! - Create session (creates TrainingSession CRD)
//! - Get session status
//! - Delete session
//! - List sessions
//!
//! Also provides proxy endpoints for training operations that forward requests
//! to the training pod via the K8s Service:
//! - Create internal session
//! - Forward-backward pass
//! - Optimizer step
//! - Sample generation
//! - Checkpoint save/load

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Extension, Json,
};
use kube::{
    api::{Api, DeleteParams, ListParams, PostParams},
    core::{ApiResource, DynamicObject, GroupVersionKind},
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::time::Instant;
use tracing::{debug, error, info, warn};

use crate::api::middleware::AuthContext;
use crate::apimetrics;
use crate::error::{ApiError, Result};
use crate::server::AppState;

// === Constants ===

const TRAINING_SESSION_GROUP: &str = "basilica.ai";
const TRAINING_SESSION_VERSION: &str = "v1";
const TRAINING_SESSION_KIND: &str = "TrainingSession";

// === Request/Response Types ===

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LoraConfigRequest {
    #[serde(default = "default_rank")]
    pub rank: u32,
    #[serde(default = "default_alpha")]
    pub alpha: u32,
    #[serde(default = "default_dropout")]
    pub dropout: f32,
    #[serde(default)]
    pub target_modules: Option<Vec<String>>,
    /// Apply LoRA to MLP layers (gate_proj, up_proj, down_proj)
    #[serde(default = "default_true")]
    pub train_mlp: bool,
    /// Apply LoRA to attention layers (q_proj, k_proj, v_proj, o_proj)
    #[serde(default = "default_true")]
    pub train_attn: bool,
    /// Apply LoRA to unembedding layer (lm_head)
    #[serde(default = "default_true")]
    pub train_unembed: bool,
}

fn default_rank() -> u32 {
    32
}
fn default_alpha() -> u32 {
    64
}
fn default_dropout() -> f32 {
    0.05
}
fn default_true() -> bool {
    true
}

impl Default for LoraConfigRequest {
    fn default() -> Self {
        Self {
            rank: default_rank(),
            alpha: default_alpha(),
            dropout: default_dropout(),
            target_modules: None,
            train_mlp: default_true(),
            train_attn: default_true(),
            train_unembed: default_true(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OptimizerConfigRequest {
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f64,
    #[serde(default = "default_weight_decay")]
    pub weight_decay: f64,
    #[serde(default)]
    pub grad_clip: Option<f64>,
}

fn default_learning_rate() -> f64 {
    1e-4
}
fn default_weight_decay() -> f64 {
    0.01
}

impl Default for OptimizerConfigRequest {
    fn default() -> Self {
        Self {
            learning_rate: default_learning_rate(),
            weight_decay: default_weight_decay(),
            grad_clip: Some(1.0),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CheckpointStorageRequest {
    pub backend: String,
    pub bucket: String,
    pub path: String,
    #[serde(default)]
    pub credentials_secret: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GpuResourcesRequest {
    #[serde(default = "default_gpu_count")]
    pub count: u32,
    #[serde(default)]
    pub model: Vec<String>,
    #[serde(default)]
    pub min_memory_gb: Option<u32>,
}

fn default_gpu_count() -> u32 {
    1
}

impl Default for GpuResourcesRequest {
    fn default() -> Self {
        Self {
            count: default_gpu_count(),
            model: vec![],
            min_memory_gb: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateSessionRequest {
    pub base_model: String,
    pub checkpoint_storage: CheckpointStorageRequest,
    #[serde(default)]
    pub lora_config: Option<LoraConfigRequest>,
    #[serde(default)]
    pub optimizer_config: Option<OptimizerConfigRequest>,
    #[serde(default)]
    pub gpu_resources: Option<GpuResourcesRequest>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default = "default_ttl")]
    pub ttl_seconds: u64,
}

fn default_ttl() -> u64 {
    86400
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateSessionResponse {
    pub session_id: String,
    pub status: String,
    pub endpoint: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionStatusResponse {
    pub session_id: String,
    pub phase: String,
    pub base_model: String,
    pub steps_completed: u64,
    pub tokens_processed: u64,
    pub endpoint: Option<String>,
    pub last_checkpoint: Option<String>,
    pub error: Option<String>,
}

/// Server capabilities response.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CapabilitiesResponse {
    /// Available models for training
    pub models: Vec<String>,
    /// Maximum tokens per batch
    pub max_batch_tokens: u32,
    /// Maximum sequence length
    pub max_sequence_length: u32,
    /// Available GPU types
    pub gpu_types: Vec<String>,
    /// LoRA rank options
    pub lora_ranks: Vec<u32>,
}

/// Training run summary.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingRunResponse {
    pub run_id: String,
    pub session_id: String,
    pub user_id: String,
    pub base_model: String,
    pub status: String,
    pub steps_completed: u64,
    pub tokens_processed: u64,
    pub checkpoints: Vec<String>,
    pub created_at: String,
    pub updated_at: String,
}

/// List training runs response.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ListTrainingRunsResponse {
    pub runs: Vec<TrainingRunResponse>,
    pub total: u64,
    pub offset: u64,
    pub limit: u64,
}

/// Checkpoint info response.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CheckpointResponse {
    pub checkpoint_id: String,
    pub run_id: String,
    pub path: String,
    pub step: u64,
    pub size_bytes: u64,
    pub created_at: String,
    pub is_published: bool,
    pub public_url: Option<String>,
}

/// List checkpoints response.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ListCheckpointsResponse {
    pub checkpoints: Vec<CheckpointResponse>,
    pub total: u64,
}

/// List training runs query params.
#[derive(Debug, Clone, Deserialize)]
pub struct ListTrainingRunsQuery {
    #[serde(default = "default_limit")]
    pub limit: u64,
    #[serde(default)]
    pub offset: u64,
}

fn default_limit() -> u64 {
    20
}

/// List checkpoints query params.
#[derive(Debug, Clone, Deserialize)]
pub struct ListCheckpointsQuery {
    pub run_id: Option<String>,
    #[serde(default = "default_checkpoint_limit")]
    pub limit: u64,
}

fn default_checkpoint_limit() -> u64 {
    100
}

/// Checkpoint path query param.
#[derive(Debug, Clone, Deserialize)]
pub struct CheckpointPathQuery {
    pub path: String,
}

/// Publish checkpoint request.
#[derive(Debug, Clone, Deserialize)]
pub struct PublishCheckpointRequest {
    pub path: String,
}

/// Publish checkpoint response.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PublishCheckpointResponse {
    pub public_url: String,
}

/// Download URL response.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadUrlResponse {
    pub url: String,
    pub expires_in_seconds: u64,
}

// === Helper Functions ===

fn user_namespace(user_id: &str) -> String {
    // Sanitize user_id for K8s namespace (lowercase, alphanumeric + hyphens)
    let sanitized: String = user_id
        .chars()
        .map(|c| if c.is_alphanumeric() { c.to_ascii_lowercase() } else { '-' })
        .collect();
    format!("u-{}", sanitized)
}

fn get_training_session_api(client: &kube::Client, namespace: &str) -> Api<DynamicObject> {
    let gvk = GroupVersionKind::gvk(TRAINING_SESSION_GROUP, TRAINING_SESSION_VERSION, TRAINING_SESSION_KIND);
    let ar = ApiResource::from_gvk(&gvk);
    Api::namespaced_with(client.clone(), namespace, &ar)
}

fn build_training_session_crd(
    session_id: &str,
    user_id: &str,
    req: &CreateSessionRequest,
) -> serde_json::Value {
    let lora = req.lora_config.clone().unwrap_or_default();
    let optimizer = req.optimizer_config.clone().unwrap_or_default();
    let gpu = req.gpu_resources.clone().unwrap_or_default();

    // Build target_modules from flags if not explicitly provided
    let target_modules = lora.target_modules.unwrap_or_else(|| {
        let mut modules = Vec::new();
        if lora.train_attn {
            modules.extend(["q_proj", "k_proj", "v_proj", "o_proj"].map(String::from));
        }
        if lora.train_mlp {
            modules.extend(["gate_proj", "up_proj", "down_proj"].map(String::from));
        }
        if lora.train_unembed {
            modules.push("lm_head".to_string());
        }
        // Fallback to attention if nothing selected
        if modules.is_empty() {
            modules.extend(["q_proj", "k_proj", "v_proj", "o_proj"].map(String::from));
        }
        modules
    });

    json!({
        "apiVersion": format!("{}/{}", TRAINING_SESSION_GROUP, TRAINING_SESSION_VERSION),
        "kind": TRAINING_SESSION_KIND,
        "metadata": {
            "name": session_id,
            "labels": {
                "app": "basilica-training",
                "user": user_id
            }
        },
        "spec": {
            "userId": user_id,
            "baseModel": req.base_model,
            "loraConfig": {
                "rank": lora.rank,
                "alpha": lora.alpha,
                "dropout": lora.dropout,
                "targetModules": target_modules,
                "trainMlp": lora.train_mlp,
                "trainAttn": lora.train_attn,
                "trainUnembed": lora.train_unembed
            },
            "optimizerConfig": {
                "learningRate": optimizer.learning_rate,
                "weightDecay": optimizer.weight_decay,
                "gradClip": optimizer.grad_clip
            },
            "checkpointStorage": {
                "backend": req.checkpoint_storage.backend,
                "bucket": req.checkpoint_storage.bucket,
                "path": req.checkpoint_storage.path,
                "credentialsSecret": req.checkpoint_storage.credentials_secret
            },
            "gpuResources": {
                "count": gpu.count,
                "model": gpu.model,
                "minMemoryGb": gpu.min_memory_gb
            },
            "ttlSeconds": req.ttl_seconds,
            "seed": req.seed,
            "enableBilling": true
        }
    })
}

// === Session Management Handlers ===

/// Create a new training session.
///
/// Creates a TrainingSession CRD which the operator will reconcile to create
/// the training pod, service, and HTTPRoute for Envoy Gateway routing.
pub async fn create_session(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Json(req): Json<CreateSessionRequest>,
) -> Result<Json<CreateSessionResponse>> {
    let start = Instant::now();

    info!(
        user_id = %auth.user_id,
        model = %req.base_model,
        "Creating training session"
    );

    let k8s_client = state.k8s.as_ref().ok_or(ApiError::ServiceUnavailable)?;
    let kube_client = k8s_client.kube_client();

    let namespace = user_namespace(&auth.user_id);

    // Generate a unique session ID
    let session_id = format!(
        "ts-{}",
        uuid::Uuid::new_v4()
            .to_string()
            .split('-')
            .next()
            .unwrap()
    );

    // Ensure namespace exists
    if let Err(e) = k8s_client.create_namespace(&namespace).await {
        // Ignore "already exists" errors
        if !e.to_string().contains("409") && !e.to_string().contains("AlreadyExists") {
            error!(error = %e, namespace = %namespace, "Failed to create namespace");
            apimetrics::record_request("training.create", "POST", start, false);
            return Err(ApiError::Internal {
                message: format!("Failed to create namespace: {}", e),
            });
        }
    }

    // Build and create the TrainingSession CRD
    let crd_json = build_training_session_crd(&session_id, &auth.user_id, &req);
    let crd: DynamicObject = serde_json::from_value(crd_json).map_err(|e| {
        error!(error = %e, "Failed to build TrainingSession CRD");
        ApiError::Internal {
            message: format!("Failed to build CRD: {}", e),
        }
    })?;

    let api = get_training_session_api(&kube_client, &namespace);

    match api.create(&PostParams::default(), &crd).await {
        Ok(_) => {
            info!(
                session_id = %session_id,
                namespace = %namespace,
                "Created TrainingSession CRD"
            );
        }
        Err(kube::Error::Api(ae)) if ae.code == 409 => {
            warn!(
                session_id = %session_id,
                "TrainingSession already exists"
            );
            apimetrics::record_request("training.create", "POST", start, false);
            return Err(ApiError::Conflict {
                message: format!("Session {} already exists", session_id),
            });
        }
        Err(e) => {
            error!(error = %e, "Failed to create TrainingSession CRD");
            apimetrics::record_request("training.create", "POST", start, false);
            return Err(ApiError::Internal {
                message: format!("Failed to create session: {}", e),
            });
        }
    }

    // The endpoint where SDK will call training operations (via Envoy Gateway)
    let endpoint = format!("https://api.basilica.ai/sessions/{}/", session_id);

    apimetrics::record_request("training.create", "POST", start, true);

    Ok(Json(CreateSessionResponse {
        session_id,
        status: "pending".to_string(),
        endpoint,
    }))
}

/// Get training session status.
pub async fn get_session(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Path(session_id): Path<String>,
) -> Result<Json<SessionStatusResponse>> {
    let start = Instant::now();

    let k8s_client = state.k8s.as_ref().ok_or(ApiError::ServiceUnavailable)?;
    let kube_client = k8s_client.kube_client();

    let namespace = user_namespace(&auth.user_id);
    let api = get_training_session_api(&kube_client, &namespace);

    let session = api.get(&session_id).await.map_err(|e| {
        if e.to_string().contains("404") || e.to_string().contains("NotFound") {
            ApiError::NotFound {
                message: format!("Session {} not found", session_id),
            }
        } else {
            error!(error = %e, session_id = %session_id, "Failed to get session");
            ApiError::Internal {
                message: format!("Failed to get session: {}", e),
            }
        }
    })?;

    // Extract status from the CRD
    let status = session.data.get("status").cloned().unwrap_or_else(|| json!({}));
    let spec = session.data.get("spec").cloned().unwrap_or_else(|| json!({}));

    let phase = status
        .get("phase")
        .and_then(|v| v.as_str())
        .unwrap_or("pending")
        .to_string();

    let base_model = spec
        .get("baseModel")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    let steps_completed = status
        .get("stepsCompleted")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let tokens_processed = status
        .get("tokensProcessed")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let endpoint = status
        .get("endpoint")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let last_checkpoint = status
        .get("lastCheckpoint")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let error = status
        .get("error")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    apimetrics::record_request("training.get", "GET", start, true);

    Ok(Json(SessionStatusResponse {
        session_id,
        phase,
        base_model,
        steps_completed,
        tokens_processed,
        endpoint,
        last_checkpoint,
        error,
    }))
}

/// Delete a training session.
///
/// Deletes the TrainingSession CRD. The operator will handle cleanup of
/// pods, services, and HTTPRoutes via owner references.
pub async fn delete_session(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Path(session_id): Path<String>,
) -> Result<StatusCode> {
    let start = Instant::now();

    info!(
        user_id = %auth.user_id,
        session_id = %session_id,
        "Deleting training session"
    );

    let k8s_client = state.k8s.as_ref().ok_or(ApiError::ServiceUnavailable)?;
    let kube_client = k8s_client.kube_client();

    let namespace = user_namespace(&auth.user_id);
    let api = get_training_session_api(&kube_client, &namespace);

    match api.delete(&session_id, &DeleteParams::default()).await {
        Ok(_) => {
            info!(
                session_id = %session_id,
                namespace = %namespace,
                "Deleted TrainingSession CRD"
            );
        }
        Err(kube::Error::Api(ae)) if ae.code == 404 => {
            warn!(
                session_id = %session_id,
                "TrainingSession not found, may already be deleted"
            );
        }
        Err(e) => {
            error!(error = %e, "Failed to delete TrainingSession");
            apimetrics::record_request("training.delete", "DELETE", start, false);
            return Err(ApiError::Internal {
                message: format!("Failed to delete session: {}", e),
            });
        }
    }

    apimetrics::record_request("training.delete", "DELETE", start, true);
    Ok(StatusCode::NO_CONTENT)
}

/// List all training sessions for the authenticated user.
pub async fn list_sessions(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
) -> Result<Json<Vec<SessionStatusResponse>>> {
    let start = Instant::now();

    let k8s_client = state.k8s.as_ref().ok_or(ApiError::ServiceUnavailable)?;
    let kube_client = k8s_client.kube_client();

    let namespace = user_namespace(&auth.user_id);
    let api = get_training_session_api(&kube_client, &namespace);

    // List all TrainingSession CRDs in the user's namespace
    let sessions = api
        .list(&ListParams::default().labels(&format!("user={}", auth.user_id)))
        .await
        .map_err(|e| {
            // If namespace doesn't exist, return empty list
            if e.to_string().contains("404") || e.to_string().contains("NotFound") {
                return ApiError::Internal {
                    message: "empty".to_string(), // Sentinel for empty list
                };
            }
            error!(error = %e, "Failed to list sessions");
            ApiError::Internal {
                message: format!("Failed to list sessions: {}", e),
            }
        });

    let sessions = match sessions {
        Ok(list) => list,
        Err(e) if e.to_string().contains("empty") => {
            apimetrics::record_request("training.list", "GET", start, true);
            return Ok(Json(vec![]));
        }
        Err(e) => {
            apimetrics::record_request("training.list", "GET", start, false);
            return Err(e);
        }
    };

    let mut responses = Vec::new();

    for session in sessions {
        let session_id = session
            .metadata
            .name
            .clone()
            .unwrap_or_else(|| "unknown".to_string());

        let status = session.data.get("status").cloned().unwrap_or_else(|| json!({}));
        let spec = session.data.get("spec").cloned().unwrap_or_else(|| json!({}));

        let phase = status
            .get("phase")
            .and_then(|v| v.as_str())
            .unwrap_or("pending")
            .to_string();

        let base_model = spec
            .get("baseModel")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        let steps_completed = status
            .get("stepsCompleted")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        let tokens_processed = status
            .get("tokensProcessed")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        let endpoint = status
            .get("endpoint")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let last_checkpoint = status
            .get("lastCheckpoint")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let error = status
            .get("error")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        responses.push(SessionStatusResponse {
            session_id,
            phase,
            base_model,
            steps_completed,
            tokens_processed,
            endpoint,
            last_checkpoint,
            error,
        });
    }

    apimetrics::record_request("training.list", "GET", start, true);
    Ok(Json(responses))
}

// === Training Operation Proxy Handlers ===
//
// These handlers proxy training operations to the training pod via the K8s Service.
// This allows the SDK to route all traffic through the API instead of requiring
// direct access to the Envoy Gateway.

/// Proxy a request to the training service via the K8s API server proxy.
///
/// This uses the K8s API server's built-in service proxy feature, which allows
/// reaching services from outside the cluster without needing cluster-local DNS.
/// The proxy URL format is: /api/v1/namespaces/{namespace}/services/{service}:{port}/proxy/{path}
async fn proxy_to_training_service(
    kube_client: &kube::Client,
    namespace: &str,
    session_id: &str,
    path: &str,
    method: http::Method,
    body: Option<serde_json::Value>,
) -> Result<axum::response::Response> {
    let service_name = format!("training-{}", session_id);

    // Build the request
    let mut request_builder = http::Request::builder()
        .method(method.clone())
        .uri(format!(
            "/api/v1/namespaces/{}/services/{}:8000/proxy{}",
            namespace, service_name, path
        ));

    // Add content-type header for POST requests with body
    if body.is_some() {
        request_builder = request_builder.header("content-type", "application/json");
    }

    let request = if let Some(json_body) = body {
        let body_bytes = serde_json::to_vec(&json_body).map_err(|e| ApiError::Internal {
            message: format!("Failed to serialize request body: {}", e),
        })?;
        request_builder
            .body(body_bytes)
            .map_err(|e| ApiError::Internal {
                message: format!("Failed to build request: {}", e),
            })?
    } else {
        request_builder
            .body(vec![])
            .map_err(|e| ApiError::Internal {
                message: format!("Failed to build request: {}", e),
            })?
    };

    debug!(
        service = %service_name,
        namespace = %namespace,
        path = %path,
        method = %method,
        "Proxying request to training service via K8s API"
    );

    // Execute the request through the kube client
    let response = kube_client
        .request::<serde_json::Value>(request)
        .await
        .map_err(|e| {
            error!(
                error = %e,
                service = %service_name,
                namespace = %namespace,
                path = %path,
                "Failed to proxy request to training service"
            );
            ApiError::ServiceUnavailable
        })?;

    // Convert to axum response
    let body_bytes = serde_json::to_vec(&response).map_err(|e| ApiError::Internal {
        message: format!("Failed to serialize response: {}", e),
    })?;

    axum::response::Response::builder()
        .status(200)
        .header("content-type", "application/json")
        .body(axum::body::Body::from(body_bytes))
        .map_err(|e| ApiError::Internal {
            message: format!("Failed to build response: {}", e),
        })
}

/// Create a training session in the training pod.
/// POST /sessions/{session_id}/internal
pub async fn create_internal_session(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Path(session_id): Path<String>,
    Json(body): Json<serde_json::Value>,
) -> Result<axum::response::Response> {
    let start = Instant::now();

    let k8s_client = state.k8s.as_ref().ok_or(ApiError::ServiceUnavailable)?;
    let kube_client = k8s_client.kube_client();
    let namespace = user_namespace(&auth.user_id);

    let result = proxy_to_training_service(
        &kube_client,
        &namespace,
        &session_id,
        "/sessions",
        http::Method::POST,
        Some(body),
    )
    .await;

    apimetrics::record_request("training.proxy.create_internal", "POST", start, result.is_ok());
    result
}

/// Get internal session status.
/// GET /sessions/{session_id}/internal/{internal_session_id}
pub async fn get_internal_session(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Path((session_id, internal_session_id)): Path<(String, String)>,
) -> Result<axum::response::Response> {
    let start = Instant::now();

    let k8s_client = state.k8s.as_ref().ok_or(ApiError::ServiceUnavailable)?;
    let kube_client = k8s_client.kube_client();
    let namespace = user_namespace(&auth.user_id);

    let result = proxy_to_training_service(
        &kube_client,
        &namespace,
        &session_id,
        &format!("/sessions/{}", internal_session_id),
        http::Method::GET,
        None,
    )
    .await;

    apimetrics::record_request("training.proxy.get_internal", "GET", start, result.is_ok());
    result
}

/// Forward-backward pass.
/// POST /sessions/{session_id}/internal/{internal_session_id}/forward_backward
pub async fn forward_backward(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Path((session_id, internal_session_id)): Path<(String, String)>,
    Json(body): Json<serde_json::Value>,
) -> Result<axum::response::Response> {
    let start = Instant::now();

    let k8s_client = state.k8s.as_ref().ok_or(ApiError::ServiceUnavailable)?;
    let kube_client = k8s_client.kube_client();
    let namespace = user_namespace(&auth.user_id);

    let result = proxy_to_training_service(
        &kube_client,
        &namespace,
        &session_id,
        &format!("/sessions/{}/forward_backward", internal_session_id),
        http::Method::POST,
        Some(body),
    )
    .await;

    apimetrics::record_request("training.proxy.forward_backward", "POST", start, result.is_ok());
    result
}

/// Optimizer step.
/// POST /sessions/{session_id}/internal/{internal_session_id}/optim_step
pub async fn optim_step(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Path((session_id, internal_session_id)): Path<(String, String)>,
) -> Result<axum::response::Response> {
    let start = Instant::now();

    let k8s_client = state.k8s.as_ref().ok_or(ApiError::ServiceUnavailable)?;
    let kube_client = k8s_client.kube_client();
    let namespace = user_namespace(&auth.user_id);

    let result = proxy_to_training_service(
        &kube_client,
        &namespace,
        &session_id,
        &format!("/sessions/{}/optim_step", internal_session_id),
        http::Method::POST,
        None,
    )
    .await;

    apimetrics::record_request("training.proxy.optim_step", "POST", start, result.is_ok());
    result
}

/// Generate text sample.
/// POST /sessions/{session_id}/internal/{internal_session_id}/sample
pub async fn sample(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Path((session_id, internal_session_id)): Path<(String, String)>,
    Json(body): Json<serde_json::Value>,
) -> Result<axum::response::Response> {
    let start = Instant::now();

    let k8s_client = state.k8s.as_ref().ok_or(ApiError::ServiceUnavailable)?;
    let kube_client = k8s_client.kube_client();
    let namespace = user_namespace(&auth.user_id);

    let result = proxy_to_training_service(
        &kube_client,
        &namespace,
        &session_id,
        &format!("/sessions/{}/sample", internal_session_id),
        http::Method::POST,
        Some(body),
    )
    .await;

    apimetrics::record_request("training.proxy.sample", "POST", start, result.is_ok());
    result
}

/// Save checkpoint.
/// POST /sessions/{session_id}/internal/{internal_session_id}/save
pub async fn save_checkpoint(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Path((session_id, internal_session_id)): Path<(String, String)>,
    Json(body): Json<serde_json::Value>,
) -> Result<axum::response::Response> {
    let start = Instant::now();

    let k8s_client = state.k8s.as_ref().ok_or(ApiError::ServiceUnavailable)?;
    let kube_client = k8s_client.kube_client();
    let namespace = user_namespace(&auth.user_id);

    let result = proxy_to_training_service(
        &kube_client,
        &namespace,
        &session_id,
        &format!("/sessions/{}/save", internal_session_id),
        http::Method::POST,
        Some(body),
    )
    .await;

    apimetrics::record_request("training.proxy.save", "POST", start, result.is_ok());
    result
}

/// Load checkpoint.
/// POST /sessions/{session_id}/internal/{internal_session_id}/load
pub async fn load_checkpoint(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Path((session_id, internal_session_id)): Path<(String, String)>,
    Json(body): Json<serde_json::Value>,
) -> Result<axum::response::Response> {
    let start = Instant::now();

    let k8s_client = state.k8s.as_ref().ok_or(ApiError::ServiceUnavailable)?;
    let kube_client = k8s_client.kube_client();
    let namespace = user_namespace(&auth.user_id);

    let result = proxy_to_training_service(
        &kube_client,
        &namespace,
        &session_id,
        &format!("/sessions/{}/load", internal_session_id),
        http::Method::POST,
        Some(body),
    )
    .await;

    apimetrics::record_request("training.proxy.load", "POST", start, result.is_ok());
    result
}

/// Forward pass (inference only, no gradients).
/// POST /sessions/{session_id}/internal/{internal_session_id}/forward
pub async fn forward(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Path((session_id, internal_session_id)): Path<(String, String)>,
    Json(body): Json<serde_json::Value>,
) -> Result<axum::response::Response> {
    let start = Instant::now();

    let k8s_client = state.k8s.as_ref().ok_or(ApiError::ServiceUnavailable)?;
    let kube_client = k8s_client.kube_client();
    let namespace = user_namespace(&auth.user_id);

    let result = proxy_to_training_service(
        &kube_client,
        &namespace,
        &session_id,
        &format!("/sessions/{}/forward", internal_session_id),
        http::Method::POST,
        Some(body),
    )
    .await;

    apimetrics::record_request("training.proxy.forward", "POST", start, result.is_ok());
    result
}

/// Compute log probabilities for a token sequence.
/// POST /sessions/{session_id}/internal/{internal_session_id}/compute_logprobs
pub async fn compute_logprobs(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Path((session_id, internal_session_id)): Path<(String, String)>,
    Json(body): Json<serde_json::Value>,
) -> Result<axum::response::Response> {
    let start = Instant::now();

    let k8s_client = state.k8s.as_ref().ok_or(ApiError::ServiceUnavailable)?;
    let kube_client = k8s_client.kube_client();
    let namespace = user_namespace(&auth.user_id);

    let result = proxy_to_training_service(
        &kube_client,
        &namespace,
        &session_id,
        &format!("/sessions/{}/compute_logprobs", internal_session_id),
        http::Method::POST,
        Some(body),
    )
    .await;

    apimetrics::record_request("training.proxy.compute_logprobs", "POST", start, result.is_ok());
    result
}

/// Save weights formatted for use with SamplingClient.
/// POST /sessions/{session_id}/internal/{internal_session_id}/save_for_sampler
pub async fn save_for_sampler(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Path((session_id, internal_session_id)): Path<(String, String)>,
    Json(body): Json<serde_json::Value>,
) -> Result<axum::response::Response> {
    let start = Instant::now();

    let k8s_client = state.k8s.as_ref().ok_or(ApiError::ServiceUnavailable)?;
    let kube_client = k8s_client.kube_client();
    let namespace = user_namespace(&auth.user_id);

    let result = proxy_to_training_service(
        &kube_client,
        &namespace,
        &session_id,
        &format!("/sessions/{}/save_for_sampler", internal_session_id),
        http::Method::POST,
        Some(body),
    )
    .await;

    apimetrics::record_request("training.proxy.save_for_sampler", "POST", start, result.is_ok());
    result
}

// === Phase 3: Capabilities & Training Run Management ===

/// Get server capabilities.
/// GET /capabilities
pub async fn get_capabilities() -> Json<CapabilitiesResponse> {
    Json(CapabilitiesResponse {
        models: vec![
            "facebook/opt-125m".to_string(),
            "facebook/opt-350m".to_string(),
            "facebook/opt-1.3b".to_string(),
            "meta-llama/Llama-2-7b-hf".to_string(),
            "meta-llama/Llama-2-13b-hf".to_string(),
            "mistralai/Mistral-7B-v0.1".to_string(),
        ],
        max_batch_tokens: 4096,
        max_sequence_length: 2048,
        gpu_types: vec![
            "A100".to_string(),
            "H100".to_string(),
            "A10G".to_string(),
            "L4".to_string(),
        ],
        lora_ranks: vec![4, 8, 16, 32, 64, 128],
    })
}

/// List training runs for the authenticated user.
/// GET /training_runs
pub async fn list_training_runs(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Query(params): Query<ListTrainingRunsQuery>,
) -> Result<Json<ListTrainingRunsResponse>> {
    let start = Instant::now();

    let k8s_client = state.k8s.as_ref().ok_or(ApiError::ServiceUnavailable)?;
    let kube_client = k8s_client.kube_client();
    let namespace = user_namespace(&auth.user_id);
    let api = get_training_session_api(&kube_client, &namespace);

    // List all TrainingSession CRDs in the user's namespace
    let sessions = match api.list(&ListParams::default().labels(&format!("user={}", auth.user_id))).await {
        Ok(list) => list,
        Err(e) => {
            if e.to_string().contains("404") || e.to_string().contains("NotFound") {
                apimetrics::record_request("training.runs.list", "GET", start, true);
                return Ok(Json(ListTrainingRunsResponse {
                    runs: vec![],
                    total: 0,
                    offset: params.offset,
                    limit: params.limit,
                }));
            }
            error!(error = %e, "Failed to list training runs");
            apimetrics::record_request("training.runs.list", "GET", start, false);
            return Err(ApiError::Internal {
                message: format!("Failed to list training runs: {}", e),
            });
        }
    };

    let total = sessions.items.len() as u64;
    let runs: Vec<TrainingRunResponse> = sessions
        .items
        .into_iter()
        .skip(params.offset as usize)
        .take(params.limit as usize)
        .map(|session| {
            let session_id = session.metadata.name.clone().unwrap_or_default();
            let status = session.data.get("status").cloned().unwrap_or_else(|| json!({}));
            let spec = session.data.get("spec").cloned().unwrap_or_else(|| json!({}));

            TrainingRunResponse {
                run_id: session_id.clone(),
                session_id: session_id.clone(),
                user_id: auth.user_id.clone(),
                base_model: spec.get("baseModel").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                status: status.get("phase").and_then(|v| v.as_str()).unwrap_or("pending").to_string(),
                steps_completed: status.get("stepsCompleted").and_then(|v| v.as_u64()).unwrap_or(0),
                tokens_processed: status.get("tokensProcessed").and_then(|v| v.as_u64()).unwrap_or(0),
                checkpoints: status.get("checkpoints").and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                    .unwrap_or_default(),
                created_at: session.metadata.creation_timestamp
                    .map(|t| t.0.to_rfc3339())
                    .unwrap_or_default(),
                updated_at: status.get("lastUpdated").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            }
        })
        .collect();

    apimetrics::record_request("training.runs.list", "GET", start, true);
    Ok(Json(ListTrainingRunsResponse {
        runs,
        total,
        offset: params.offset,
        limit: params.limit,
    }))
}

/// Get a specific training run.
/// GET /training_runs/:run_id
pub async fn get_training_run(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Path(run_id): Path<String>,
) -> Result<Json<TrainingRunResponse>> {
    let start = Instant::now();

    let k8s_client = state.k8s.as_ref().ok_or(ApiError::ServiceUnavailable)?;
    let kube_client = k8s_client.kube_client();
    let namespace = user_namespace(&auth.user_id);
    let api = get_training_session_api(&kube_client, &namespace);

    let session = api.get(&run_id).await.map_err(|e| {
        if e.to_string().contains("404") || e.to_string().contains("NotFound") {
            ApiError::NotFound {
                message: format!("Training run {} not found", run_id),
            }
        } else {
            error!(error = %e, run_id = %run_id, "Failed to get training run");
            ApiError::Internal {
                message: format!("Failed to get training run: {}", e),
            }
        }
    })?;

    let status = session.data.get("status").cloned().unwrap_or_else(|| json!({}));
    let spec = session.data.get("spec").cloned().unwrap_or_else(|| json!({}));

    apimetrics::record_request("training.runs.get", "GET", start, true);
    Ok(Json(TrainingRunResponse {
        run_id: run_id.clone(),
        session_id: run_id,
        user_id: auth.user_id.clone(),
        base_model: spec.get("baseModel").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        status: status.get("phase").and_then(|v| v.as_str()).unwrap_or("pending").to_string(),
        steps_completed: status.get("stepsCompleted").and_then(|v| v.as_u64()).unwrap_or(0),
        tokens_processed: status.get("tokensProcessed").and_then(|v| v.as_u64()).unwrap_or(0),
        checkpoints: status.get("checkpoints").and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default(),
        created_at: session.metadata.creation_timestamp
            .map(|t| t.0.to_rfc3339())
            .unwrap_or_default(),
        updated_at: status.get("lastUpdated").and_then(|v| v.as_str()).unwrap_or("").to_string(),
    }))
}

/// Get training run by checkpoint path.
/// GET /training_runs/by_path?path=...
pub async fn get_training_run_by_path(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Query(params): Query<CheckpointPathQuery>,
) -> Result<Json<TrainingRunResponse>> {
    let start = Instant::now();

    // Parse session ID from path (format: user_id/session_id/checkpoint_name)
    let parts: Vec<&str> = params.path.split('/').collect();
    if parts.len() < 2 {
        apimetrics::record_request("training.runs.by_path", "GET", start, false);
        return Err(ApiError::BadRequest {
            message: "Invalid checkpoint path format".to_string(),
        });
    }

    let session_id = parts[1].to_string();

    let k8s_client = state.k8s.as_ref().ok_or(ApiError::ServiceUnavailable)?;
    let kube_client = k8s_client.kube_client();
    let namespace = user_namespace(&auth.user_id);
    let api = get_training_session_api(&kube_client, &namespace);

    let session = api.get(&session_id).await.map_err(|e| {
        if e.to_string().contains("404") || e.to_string().contains("NotFound") {
            ApiError::NotFound {
                message: format!("Training run for path {} not found", params.path),
            }
        } else {
            error!(error = %e, path = %params.path, "Failed to get training run by path");
            ApiError::Internal {
                message: format!("Failed to get training run: {}", e),
            }
        }
    })?;

    let status = session.data.get("status").cloned().unwrap_or_else(|| json!({}));
    let spec = session.data.get("spec").cloned().unwrap_or_else(|| json!({}));

    apimetrics::record_request("training.runs.by_path", "GET", start, true);
    Ok(Json(TrainingRunResponse {
        run_id: session_id.clone(),
        session_id,
        user_id: auth.user_id.clone(),
        base_model: spec.get("baseModel").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        status: status.get("phase").and_then(|v| v.as_str()).unwrap_or("pending").to_string(),
        steps_completed: status.get("stepsCompleted").and_then(|v| v.as_u64()).unwrap_or(0),
        tokens_processed: status.get("tokensProcessed").and_then(|v| v.as_u64()).unwrap_or(0),
        checkpoints: status.get("checkpoints").and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default(),
        created_at: session.metadata.creation_timestamp
            .map(|t| t.0.to_rfc3339())
            .unwrap_or_default(),
        updated_at: status.get("lastUpdated").and_then(|v| v.as_str()).unwrap_or("").to_string(),
    }))
}

// === Phase 3: Checkpoint Management ===

/// List checkpoints.
/// GET /checkpoints?run_id=...&limit=...
pub async fn list_checkpoints(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Query(params): Query<ListCheckpointsQuery>,
) -> Result<Json<ListCheckpointsResponse>> {
    let start = Instant::now();

    let k8s_client = state.k8s.as_ref().ok_or(ApiError::ServiceUnavailable)?;
    let kube_client = k8s_client.kube_client();
    let namespace = user_namespace(&auth.user_id);
    let api = get_training_session_api(&kube_client, &namespace);

    // Get all sessions or filter by run_id
    let sessions = if let Some(ref run_id) = params.run_id {
        match api.get(run_id).await {
            Ok(s) => vec![s],
            Err(e) => {
                if e.to_string().contains("404") || e.to_string().contains("NotFound") {
                    vec![]
                } else {
                    error!(error = %e, run_id = %run_id, "Failed to get session for checkpoints");
                    apimetrics::record_request("training.checkpoints.list", "GET", start, false);
                    return Err(ApiError::Internal {
                        message: format!("Failed to list checkpoints: {}", e),
                    });
                }
            }
        }
    } else {
        match api.list(&ListParams::default().labels(&format!("user={}", auth.user_id))).await {
            Ok(list) => list.items,
            Err(e) => {
                if e.to_string().contains("404") || e.to_string().contains("NotFound") {
                    vec![]
                } else {
                    error!(error = %e, "Failed to list sessions for checkpoints");
                    apimetrics::record_request("training.checkpoints.list", "GET", start, false);
                    return Err(ApiError::Internal {
                        message: format!("Failed to list checkpoints: {}", e),
                    });
                }
            }
        }
    };

    // Extract checkpoints from all sessions
    let mut checkpoints = Vec::new();
    for session in sessions {
        let session_id = session.metadata.name.clone().unwrap_or_default();
        let status = session.data.get("status").cloned().unwrap_or_else(|| json!({}));

        if let Some(cp_array) = status.get("checkpointDetails").and_then(|v| v.as_array()) {
            for cp in cp_array {
                checkpoints.push(CheckpointResponse {
                    checkpoint_id: cp.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    run_id: session_id.clone(),
                    path: cp.get("path").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    step: cp.get("step").and_then(|v| v.as_u64()).unwrap_or(0),
                    size_bytes: cp.get("sizeBytes").and_then(|v| v.as_u64()).unwrap_or(0),
                    created_at: cp.get("createdAt").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    is_published: cp.get("isPublished").and_then(|v| v.as_bool()).unwrap_or(false),
                    public_url: cp.get("publicUrl").and_then(|v| v.as_str()).map(String::from),
                });
            }
        }
    }

    let total = checkpoints.len() as u64;
    checkpoints.truncate(params.limit as usize);

    apimetrics::record_request("training.checkpoints.list", "GET", start, true);
    Ok(Json(ListCheckpointsResponse { checkpoints, total }))
}

/// Get checkpoint info by path.
/// GET /checkpoints/info?path=...
pub async fn get_checkpoint_info(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Query(params): Query<CheckpointPathQuery>,
) -> Result<Json<CheckpointResponse>> {
    let start = Instant::now();

    // Parse session ID from path
    let parts: Vec<&str> = params.path.split('/').collect();
    if parts.len() < 3 {
        apimetrics::record_request("training.checkpoints.info", "GET", start, false);
        return Err(ApiError::BadRequest {
            message: "Invalid checkpoint path format".to_string(),
        });
    }

    let session_id = parts[1].to_string();
    let checkpoint_name = parts[2..].join("/");

    let k8s_client = state.k8s.as_ref().ok_or(ApiError::ServiceUnavailable)?;
    let kube_client = k8s_client.kube_client();
    let namespace = user_namespace(&auth.user_id);
    let api = get_training_session_api(&kube_client, &namespace);

    let session = api.get(&session_id).await.map_err(|e| {
        if e.to_string().contains("404") || e.to_string().contains("NotFound") {
            ApiError::NotFound {
                message: format!("Checkpoint at path {} not found", params.path),
            }
        } else {
            error!(error = %e, path = %params.path, "Failed to get checkpoint info");
            ApiError::Internal {
                message: format!("Failed to get checkpoint: {}", e),
            }
        }
    })?;

    let status = session.data.get("status").cloned().unwrap_or_else(|| json!({}));

    // Find the checkpoint in the session's checkpoint list
    if let Some(cp_array) = status.get("checkpointDetails").and_then(|v| v.as_array()) {
        for cp in cp_array {
            let cp_path = cp.get("path").and_then(|v| v.as_str()).unwrap_or("");
            if cp_path.contains(&checkpoint_name) || cp_path == params.path {
                apimetrics::record_request("training.checkpoints.info", "GET", start, true);
                return Ok(Json(CheckpointResponse {
                    checkpoint_id: cp.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    run_id: session_id,
                    path: cp_path.to_string(),
                    step: cp.get("step").and_then(|v| v.as_u64()).unwrap_or(0),
                    size_bytes: cp.get("sizeBytes").and_then(|v| v.as_u64()).unwrap_or(0),
                    created_at: cp.get("createdAt").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    is_published: cp.get("isPublished").and_then(|v| v.as_bool()).unwrap_or(false),
                    public_url: cp.get("publicUrl").and_then(|v| v.as_str()).map(String::from),
                }));
            }
        }
    }

    apimetrics::record_request("training.checkpoints.info", "GET", start, false);
    Err(ApiError::NotFound {
        message: format!("Checkpoint at path {} not found", params.path),
    })
}

/// Get checkpoint download URL.
/// GET /checkpoints/:checkpoint_id/download_url
pub async fn get_checkpoint_download_url(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Path(checkpoint_id): Path<String>,
) -> Result<Json<DownloadUrlResponse>> {
    let start = Instant::now();

    // In a real implementation, this would generate a signed URL from the storage backend
    // For now, we return a placeholder URL
    info!(
        user_id = %auth.user_id,
        checkpoint_id = %checkpoint_id,
        "Generating download URL for checkpoint"
    );

    // Check that we have storage config
    let _k8s_client = state.k8s.as_ref().ok_or(ApiError::ServiceUnavailable)?;

    // TODO: Generate actual signed URL from R2/S3
    // For now, return a placeholder that indicates the checkpoint path
    let url = format!(
        "https://storage.basilica.ai/checkpoints/{}/{}?token=placeholder",
        auth.user_id, checkpoint_id
    );

    apimetrics::record_request("training.checkpoints.download_url", "GET", start, true);
    Ok(Json(DownloadUrlResponse {
        url,
        expires_in_seconds: 3600,
    }))
}

/// Delete a checkpoint.
/// DELETE /checkpoints/:checkpoint_id
pub async fn delete_checkpoint(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Path(checkpoint_id): Path<String>,
) -> Result<StatusCode> {
    let start = Instant::now();

    info!(
        user_id = %auth.user_id,
        checkpoint_id = %checkpoint_id,
        "Deleting checkpoint"
    );

    // Check that we have K8s access
    let _k8s_client = state.k8s.as_ref().ok_or(ApiError::ServiceUnavailable)?;

    // TODO: Actually delete the checkpoint from storage
    // This would involve:
    // 1. Finding the session that owns this checkpoint
    // 2. Updating the session's checkpoint list
    // 3. Deleting the checkpoint files from R2/S3

    warn!(
        checkpoint_id = %checkpoint_id,
        "Checkpoint deletion not fully implemented - returning success"
    );

    apimetrics::record_request("training.checkpoints.delete", "DELETE", start, true);
    Ok(StatusCode::NO_CONTENT)
}

/// Publish a checkpoint (make it publicly accessible).
/// POST /checkpoints/publish
pub async fn publish_checkpoint(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Json(req): Json<PublishCheckpointRequest>,
) -> Result<Json<PublishCheckpointResponse>> {
    let start = Instant::now();

    info!(
        user_id = %auth.user_id,
        path = %req.path,
        "Publishing checkpoint"
    );

    let _k8s_client = state.k8s.as_ref().ok_or(ApiError::ServiceUnavailable)?;

    // TODO: Actually publish the checkpoint
    // This would involve:
    // 1. Making the storage object public or generating a permanent URL
    // 2. Updating the checkpoint metadata

    let public_url = format!(
        "https://public.basilica.ai/checkpoints/{}",
        req.path.replace('/', "-")
    );

    apimetrics::record_request("training.checkpoints.publish", "POST", start, true);
    Ok(Json(PublishCheckpointResponse { public_url }))
}

/// Unpublish a checkpoint (make it private again).
/// POST /checkpoints/unpublish
pub async fn unpublish_checkpoint(
    State(state): State<AppState>,
    Extension(auth): Extension<AuthContext>,
    Json(req): Json<PublishCheckpointRequest>,
) -> Result<StatusCode> {
    let start = Instant::now();

    info!(
        user_id = %auth.user_id,
        path = %req.path,
        "Unpublishing checkpoint"
    );

    let _k8s_client = state.k8s.as_ref().ok_or(ApiError::ServiceUnavailable)?;

    // TODO: Actually unpublish the checkpoint
    // This would involve revoking public access to the storage object

    apimetrics::record_request("training.checkpoints.unpublish", "POST", start, true);
    Ok(StatusCode::NO_CONTENT)
}
