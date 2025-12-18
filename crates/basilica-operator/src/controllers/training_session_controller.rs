//! Controller for TrainingSession CRD.
//!
//! This controller manages the lifecycle of training sessions, creating
//! pods and services for the training workloads.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use k8s_openapi::api::core::v1::{
    Container, ContainerPort, EnvVar, Pod, PodSpec, ResourceRequirements, Service, ServicePort,
    ServiceSpec,
};
use k8s_openapi::apimachinery::pkg::api::resource::Quantity;
use k8s_openapi::apimachinery::pkg::apis::meta::v1::{ObjectMeta, OwnerReference};
use k8s_openapi::apimachinery::pkg::util::intstr::IntOrString;
use kube::core::DynamicObject;
use serde::Deserialize;

use crate::crd::training_session::{
    TrainingSession, TrainingSessionPhase, TrainingSessionStatus,
};
use crate::k8s_client::K8sClient;
use anyhow::Result;
use tracing::{debug, error, info, warn};

/// Default gateway name for Envoy Gateway
const DEFAULT_GATEWAY_NAME: &str = "basilica-gateway";
/// Default gateway namespace
const DEFAULT_GATEWAY_NAMESPACE: &str = "envoy-gateway-system";

const TRAINING_SERVICE_PORT: i32 = 8000;

/// Response from training service /sessions endpoint (list)
#[derive(Debug, Deserialize)]
struct TrainingSessionListResponse {
    sessions: Vec<String>,
}

/// Response from training service /sessions/{id} endpoint
#[derive(Debug, Deserialize)]
struct TrainingSessionStatusResponse {
    session_id: String,
    #[allow(dead_code)]
    base_model: String,
    step_count: u64,
    tokens_processed: u64,
    #[allow(dead_code)]
    lora_rank: u32,
    #[allow(dead_code)]
    learning_rate: f64,
}

/// Controller for managing TrainingSession resources.
#[derive(Clone)]
pub struct TrainingSessionController<C: K8sClient> {
    pub client: C,
    pub http_client: Arc<reqwest::Client>,
}

impl<C: K8sClient> TrainingSessionController<C> {
    /// Create a new TrainingSessionController.
    pub fn new(client: C) -> Self {
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .expect("Failed to create HTTP client");
        Self {
            client,
            http_client: Arc::new(http_client),
        }
    }

    /// Fetch training status from the training service.
    /// Returns (steps_completed, tokens_processed) if successful.
    async fn fetch_training_status(
        &self,
        namespace: &str,
        name: &str,
    ) -> Option<(u64, u64)> {
        let svc_name = format!("training-{}", name);
        let internal_endpoint = format!(
            "http://{}.{}.svc:{}",
            svc_name, namespace, TRAINING_SERVICE_PORT
        );

        // First, list sessions to find active ones
        let list_url = format!("{}/sessions", internal_endpoint);
        let list_response = match self.http_client.get(&list_url).send().await {
            Ok(resp) => resp,
            Err(e) => {
                debug!(error = %e, "failed to connect to training service");
                return None;
            }
        };

        if !list_response.status().is_success() {
            debug!(status = %list_response.status(), "training service returned error");
            return None;
        }

        let sessions: TrainingSessionListResponse = match list_response.json().await {
            Ok(s) => s,
            Err(e) => {
                debug!(error = %e, "failed to parse sessions list");
                return None;
            }
        };

        if sessions.sessions.is_empty() {
            debug!("no active sessions in training service");
            return None;
        }

        // Get status of the first session (typically there's only one per pod)
        let session_id = &sessions.sessions[0];
        let status_url = format!("{}/sessions/{}", internal_endpoint, session_id);

        let status_response = match self.http_client.get(&status_url).send().await {
            Ok(resp) => resp,
            Err(e) => {
                debug!(error = %e, "failed to get session status");
                return None;
            }
        };

        if !status_response.status().is_success() {
            debug!(status = %status_response.status(), "failed to get session status");
            return None;
        }

        let status: TrainingSessionStatusResponse = match status_response.json().await {
            Ok(s) => s,
            Err(e) => {
                debug!(error = %e, "failed to parse session status");
                return None;
            }
        };

        info!(
            session = %session_id,
            steps = status.step_count,
            tokens = status.tokens_processed,
            "fetched training status"
        );

        Some((status.step_count, status.tokens_processed))
    }

    /// Reconcile a TrainingSession resource.
    pub async fn reconcile(&self, ns: &str, session: &TrainingSession) -> Result<()> {
        let name = session
            .metadata
            .name
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("TrainingSession missing name"))?;

        info!(name = %name, namespace = %ns, "reconciling training session");

        let current_status = session
            .status
            .clone()
            .unwrap_or_else(TrainingSessionStatus::new);
        let phase = &current_status.phase;

        // Handle deletion
        if session.metadata.deletion_timestamp.is_some() {
            self.handle_deletion(session, current_status).await?;
            return Ok(());
        }

        let new_status = match phase {
            TrainingSessionPhase::Pending => {
                self.handle_pending(session, current_status.clone(), ns, name)
                    .await?
            }
            TrainingSessionPhase::Scheduling => {
                self.handle_scheduling(session, current_status.clone(), ns, name)
                    .await?
            }
            TrainingSessionPhase::Initializing => {
                self.handle_initializing(session, current_status.clone(), ns, name)
                    .await?
            }
            TrainingSessionPhase::LoadingModel => {
                self.handle_loading_model(session, current_status.clone(), ns, name)
                    .await?
            }
            TrainingSessionPhase::Ready => {
                self.handle_ready(session, current_status.clone(), ns, name)
                    .await?
            }
            TrainingSessionPhase::Suspended => {
                self.handle_suspended(session, current_status.clone(), ns, name)
                    .await?
            }
            TrainingSessionPhase::Failed | TrainingSessionPhase::Terminated => {
                // Terminal states - no action needed
                current_status.clone()
            }
        };

        // Update status if changed
        if new_status.phase != current_status.phase
            || new_status.steps_completed != current_status.steps_completed
            || new_status.tokens_processed != current_status.tokens_processed
            || new_status.error != current_status.error
        {
            self.client
                .update_training_session_status(ns, name, new_status)
                .await?;
        }

        Ok(())
    }

    /// Handle deletion of a training session.
    async fn handle_deletion(
        &self,
        _session: &TrainingSession,
        _status: TrainingSessionStatus,
    ) -> Result<TrainingSessionStatus> {
        // Pod and service will be garbage collected via owner references
        info!("training session being deleted");
        Ok(TrainingSessionStatus::new().with_phase(TrainingSessionPhase::Terminated))
    }

    /// Handle pending state - create pod and service.
    async fn handle_pending(
        &self,
        session: &TrainingSession,
        status: TrainingSessionStatus,
        namespace: &str,
        name: &str,
    ) -> Result<TrainingSessionStatus> {
        let pod_name = format!("training-{}", name);
        let svc_name = format!("training-{}", name);

        // Check if pod already exists
        let existing_pod = self.client.get_pod(namespace, &pod_name).await;
        if existing_pod.is_err() {
            // Create the pod
            let pod = build_training_pod(session, namespace, name)?;
            match self.client.create_pod(namespace, &pod).await {
                Ok(_) => info!(pod = %pod_name, "created training pod"),
                Err(e) => {
                    error!(error = %e, "failed to create training pod");
                    return Ok(status.with_error(format!("Failed to create pod: {}", e)));
                }
            }
        }

        // Check if service already exists
        let existing_svc = self.client.get_service(namespace, &svc_name).await;
        if existing_svc.is_err() {
            // Create the service
            let svc = build_training_service(session, namespace, name)?;
            match self.client.create_service(namespace, &svc).await {
                Ok(_) => info!(service = %svc_name, "created training service"),
                Err(e) => {
                    error!(error = %e, "failed to create training service");
                    return Ok(status.with_error(format!("Failed to create service: {}", e)));
                }
            }
        }

        // Create NetworkPolicy to allow operator access (for status polling)
        let np_name = format!("allow-operator-training-{}", name);
        let existing_np = self.client.get_network_policy(namespace, &np_name).await;
        if existing_np.is_err() {
            let np = build_operator_access_network_policy(session, namespace, name)?;
            match self.client.create_network_policy(namespace, &np).await {
                Ok(_) => info!(policy = %np_name, "created operator access network policy"),
                Err(e) => {
                    // Non-fatal - operator can still function, just won't get status updates
                    warn!(error = %e, "failed to create network policy for operator access");
                }
            }
        }

        // Transition to scheduling
        Ok(status
            .with_phase(TrainingSessionPhase::Scheduling)
            .with_pod_name(pod_name))
    }

    /// Handle scheduling state - wait for pod to be scheduled.
    async fn handle_scheduling(
        &self,
        _session: &TrainingSession,
        status: TrainingSessionStatus,
        namespace: &str,
        name: &str,
    ) -> Result<TrainingSessionStatus> {
        let pod_name = format!("training-{}", name);

        match self.client.get_pod(namespace, &pod_name).await {
            Ok(pod) => {
                let pod_phase = pod
                    .status
                    .as_ref()
                    .and_then(|s| s.phase.clone())
                    .unwrap_or_default();

                match pod_phase.as_str() {
                    "Running" => {
                        info!(pod = %pod_name, "pod is running, transitioning to initializing");
                        Ok(status.with_phase(TrainingSessionPhase::Initializing))
                    }
                    "Pending" => {
                        debug!(pod = %pod_name, "pod still pending");
                        Ok(status)
                    }
                    "Failed" => {
                        let message = pod
                            .status
                            .as_ref()
                            .and_then(|s| s.message.clone())
                            .unwrap_or_else(|| "Pod failed".into());
                        Ok(status.with_error(message))
                    }
                    _ => {
                        debug!(pod = %pod_name, phase = %pod_phase, "pod in unknown phase");
                        Ok(status)
                    }
                }
            }
            Err(e) => {
                warn!(error = %e, "pod not found, reverting to pending");
                Ok(status.with_phase(TrainingSessionPhase::Pending))
            }
        }
    }

    /// Handle initializing state - wait for service to be ready and create HTTPRoute.
    async fn handle_initializing(
        &self,
        session: &TrainingSession,
        status: TrainingSessionStatus,
        namespace: &str,
        name: &str,
    ) -> Result<TrainingSessionStatus> {
        let svc_name = format!("training-{}", name);

        // Verify service exists
        match self.client.get_service(namespace, &svc_name).await {
            Ok(_) => {
                // Create HTTPRoute for Envoy Gateway to route external traffic
                let route_name = format!("training-route-{}", name);
                let http_route = build_training_http_route(
                    name,
                    namespace,
                    &svc_name,
                    TRAINING_SERVICE_PORT as u16,
                )?;

                match self.client.create_http_route(namespace, &http_route).await {
                    Ok(_) => {
                        info!(route = %route_name, "created HTTPRoute for training session");
                    }
                    Err(e) => {
                        // HTTPRoute may already exist (409 conflict) - that's fine
                        if !e.to_string().contains("409") && !e.to_string().contains("AlreadyExists")
                        {
                            warn!(error = %e, "failed to create HTTPRoute, will retry");
                            return Ok(status);
                        }
                        debug!(route = %route_name, "HTTPRoute already exists");
                    }
                }

                // External endpoint via Envoy Gateway
                let external_endpoint = format!("https://api.basilica.ai/sessions/{}/", name);
                // Internal endpoint for health checks
                let internal_endpoint =
                    format!("http://{}.{}.svc:{}", svc_name, namespace, TRAINING_SERVICE_PORT);

                info!(
                    external = %external_endpoint,
                    internal = %internal_endpoint,
                    "training service ready with HTTPRoute"
                );

                // Transition to loading model
                Ok(status
                    .with_phase(TrainingSessionPhase::LoadingModel)
                    .with_endpoint(external_endpoint))
            }
            Err(e) => {
                warn!(error = %e, "service not found");
                // Recreate service
                let svc = build_training_service(session, namespace, name)?;
                self.client.create_service(namespace, &svc).await?;
                Ok(status)
            }
        }
    }

    /// Handle loading model state - wait for health check.
    async fn handle_loading_model(
        &self,
        _session: &TrainingSession,
        status: TrainingSessionStatus,
        _namespace: &str,
        name: &str,
    ) -> Result<TrainingSessionStatus> {
        // In a full implementation, we would do an HTTP health check here
        // For MVP, we transition directly to Ready after a brief wait
        // The actual health check would be: GET {endpoint}/health

        info!(session = %name, "model loading complete, transitioning to ready");
        let mut new_status = status.with_phase(TrainingSessionPhase::Ready);
        new_status.start_time = Some(chrono::Utc::now().to_rfc3339());
        Ok(new_status)
    }

    /// Handle ready state - monitor health, TTL, and poll training status.
    async fn handle_ready(
        &self,
        session: &TrainingSession,
        status: TrainingSessionStatus,
        namespace: &str,
        name: &str,
    ) -> Result<TrainingSessionStatus> {
        let pod_name = format!("training-{}", name);

        // Check pod health
        match self.client.get_pod(namespace, &pod_name).await {
            Ok(pod) => {
                let pod_phase = pod
                    .status
                    .as_ref()
                    .and_then(|s| s.phase.clone())
                    .unwrap_or_default();

                if pod_phase != "Running" {
                    warn!(pod = %pod_name, phase = %pod_phase, "pod no longer running");
                    return Ok(status.with_error(format!("Pod entered {} state", pod_phase)));
                }
            }
            Err(e) => {
                error!(error = %e, "pod not found");
                return Ok(status.with_error("Pod not found".into()));
            }
        }

        // Check TTL
        if let Some(start_time) = &status.start_time {
            if let Ok(start) = chrono::DateTime::parse_from_rfc3339(start_time) {
                let elapsed = chrono::Utc::now().signed_duration_since(start);
                if elapsed.num_seconds() as u64 > session.spec.ttl_seconds {
                    info!(session = %name, "session TTL expired, terminating");
                    // Delete the pod
                    let _ = self.client.delete_pod(namespace, &pod_name).await;
                    return Ok(status.with_phase(TrainingSessionPhase::Terminated));
                }
            }
        }

        // Poll training service for status updates
        let mut updated_status = status.clone();
        if let Some((steps, tokens)) = self.fetch_training_status(namespace, name).await {
            // Only update if there's actual progress
            if steps != status.steps_completed || tokens != status.tokens_processed {
                info!(
                    session = %name,
                    steps = steps,
                    tokens = tokens,
                    "updating training progress"
                );
                updated_status = updated_status.with_steps(steps, tokens);
            }
        }

        // Update last activity timestamp
        updated_status.last_updated = chrono::Utc::now().to_rfc3339();

        Ok(updated_status)
    }

    /// Handle suspended state.
    async fn handle_suspended(
        &self,
        _session: &TrainingSession,
        status: TrainingSessionStatus,
        _namespace: &str,
        _name: &str,
    ) -> Result<TrainingSessionStatus> {
        // In suspended state, we keep the pod but could scale down resources
        // For MVP, we just stay in this state
        Ok(status)
    }
}

/// Build the training pod spec.
fn build_training_pod(session: &TrainingSession, namespace: &str, name: &str) -> Result<Pod> {
    let spec = &session.spec;
    let pod_name = format!("training-{}", name);

    // Allow TRAINING_IMAGE env var to override the default image (for local dev)
    let image = std::env::var("TRAINING_IMAGE").unwrap_or_else(|_| spec.image.clone());

    // Build resource requirements
    let mut limits = BTreeMap::new();
    let mut requests = BTreeMap::new();

    // GPU resources (only if count > 0)
    if spec.gpu_resources.count > 0 {
        limits.insert(
            "nvidia.com/gpu".to_string(),
            Quantity(spec.gpu_resources.count.to_string()),
        );
        requests.insert(
            "nvidia.com/gpu".to_string(),
            Quantity(spec.gpu_resources.count.to_string()),
        );
        // CPU and memory defaults for GPU training workloads
        limits.insert("cpu".to_string(), Quantity("16".into()));
        limits.insert("memory".to_string(), Quantity("64Gi".into()));
        requests.insert("cpu".to_string(), Quantity("8".into()));
        requests.insert("memory".to_string(), Quantity("32Gi".into()));
    } else {
        // CPU-only mode (for local development/testing)
        limits.insert("cpu".to_string(), Quantity("2".into()));
        limits.insert("memory".to_string(), Quantity("4Gi".into()));
        requests.insert("cpu".to_string(), Quantity("1".into()));
        requests.insert("memory".to_string(), Quantity("2Gi".into()));
    }

    let resources = ResourceRequirements {
        limits: Some(limits),
        requests: Some(requests),
        claims: None,
    };

    // Environment variables
    let mut env_vars = vec![
        EnvVar {
            name: "MODEL_CACHE_DIR".to_string(),
            value: Some("/models".to_string()),
            ..Default::default()
        },
        EnvVar {
            name: "CHECKPOINT_DIR".to_string(),
            value: Some("/checkpoints".to_string()),
            ..Default::default()
        },
        EnvVar {
            name: "BASE_MODEL".to_string(),
            value: Some(spec.base_model.clone()),
            ..Default::default()
        },
        EnvVar {
            name: "LORA_RANK".to_string(),
            value: Some(spec.lora_config.rank.to_string()),
            ..Default::default()
        },
        EnvVar {
            name: "LORA_ALPHA".to_string(),
            value: Some(spec.lora_config.alpha.to_string()),
            ..Default::default()
        },
        EnvVar {
            name: "LEARNING_RATE".to_string(),
            value: Some(spec.optimizer_config.learning_rate.to_string()),
            ..Default::default()
        },
    ];

    // Add seed if specified
    if let Some(seed) = spec.seed {
        env_vars.push(EnvVar {
            name: "SEED".to_string(),
            value: Some(seed.to_string()),
            ..Default::default()
        });
    }

    // Build labels
    let mut labels = BTreeMap::new();
    labels.insert("app".to_string(), "basilica-training".to_string());
    labels.insert("session".to_string(), name.to_string());
    labels.insert("user".to_string(), spec.user_id.clone());

    // Build owner reference
    let owner_ref = OwnerReference {
        api_version: "basilica.ai/v1".to_string(),
        kind: "TrainingSession".to_string(),
        name: name.to_string(),
        uid: session.metadata.uid.clone().unwrap_or_default(),
        controller: Some(true),
        block_owner_deletion: Some(true),
    };

    let pod = Pod {
        metadata: ObjectMeta {
            name: Some(pod_name),
            namespace: Some(namespace.to_string()),
            labels: Some(labels),
            owner_references: Some(vec![owner_ref]),
            ..Default::default()
        },
        spec: Some(PodSpec {
            containers: vec![Container {
                name: "training".to_string(),
                image: Some(image.clone()),
                // Use Never for local images (when TRAINING_IMAGE env is set)
                image_pull_policy: Some(
                    if std::env::var("TRAINING_IMAGE").is_ok() {
                        "Never".to_string()
                    } else {
                        "IfNotPresent".to_string()
                    }
                ),
                ports: Some(vec![ContainerPort {
                    container_port: TRAINING_SERVICE_PORT,
                    name: Some("http".to_string()),
                    ..Default::default()
                }]),
                resources: Some(resources),
                env: Some(env_vars),
                ..Default::default()
            }],
            restart_policy: Some("Never".to_string()),
            ..Default::default()
        }),
        ..Default::default()
    };

    Ok(pod)
}

/// Build the training service spec.
fn build_training_service(session: &TrainingSession, namespace: &str, name: &str) -> Result<Service> {
    let svc_name = format!("training-{}", name);

    // Build labels and selector
    let mut labels = BTreeMap::new();
    labels.insert("app".to_string(), "basilica-training".to_string());
    labels.insert("session".to_string(), name.to_string());

    let mut selector = BTreeMap::new();
    selector.insert("session".to_string(), name.to_string());

    // Build owner reference
    let owner_ref = OwnerReference {
        api_version: "basilica.ai/v1".to_string(),
        kind: "TrainingSession".to_string(),
        name: name.to_string(),
        uid: session.metadata.uid.clone().unwrap_or_default(),
        controller: Some(true),
        block_owner_deletion: Some(true),
    };

    let service = Service {
        metadata: ObjectMeta {
            name: Some(svc_name),
            namespace: Some(namespace.to_string()),
            labels: Some(labels),
            owner_references: Some(vec![owner_ref]),
            ..Default::default()
        },
        spec: Some(ServiceSpec {
            selector: Some(selector),
            ports: Some(vec![ServicePort {
                port: TRAINING_SERVICE_PORT,
                target_port: Some(IntOrString::Int(TRAINING_SERVICE_PORT)),
                name: Some("http".to_string()),
                ..Default::default()
            }]),
            ..Default::default()
        }),
        ..Default::default()
    };

    Ok(service)
}

/// Build HTTPRoute for Envoy Gateway to route traffic to training session.
///
/// Routes `/sessions/{session_id}/*` to the training service pod.
/// This enables SDK to call training operations directly through Envoy Gateway.
fn build_training_http_route(
    session_name: &str,
    namespace: &str,
    backend_service: &str,
    port: u16,
) -> Result<DynamicObject> {
    let route_name = format!("training-route-{}", session_name);
    let path_prefix = format!("/sessions/{}/", session_name);

    let route_json = serde_json::json!({
        "apiVersion": "gateway.networking.k8s.io/v1",
        "kind": "HTTPRoute",
        "metadata": {
            "name": route_name,
            "namespace": namespace,
            "labels": {
                "app": "basilica-training",
                "session": session_name
            }
        },
        "spec": {
            "parentRefs": [{
                "name": DEFAULT_GATEWAY_NAME,
                "namespace": DEFAULT_GATEWAY_NAMESPACE
            }],
            "rules": [{
                "matches": [{
                    "path": {
                        "type": "PathPrefix",
                        "value": path_prefix
                    }
                }],
                "filters": [{
                    "type": "URLRewrite",
                    "urlRewrite": {
                        "path": {
                            "type": "ReplacePrefixMatch",
                            "replacePrefixMatch": "/"
                        }
                    }
                }],
                "backendRefs": [{
                    "name": backend_service,
                    "namespace": namespace,
                    "port": port
                }]
            }]
        }
    });

    let route: DynamicObject = serde_json::from_value(route_json)?;
    Ok(route)
}

/// Build a NetworkPolicy to allow the operator to access the training pod.
///
/// This enables the operator (running in basilica-system namespace) to poll
/// the training service for status updates (steps, tokens processed).
fn build_operator_access_network_policy(
    session: &TrainingSession,
    namespace: &str,
    name: &str,
) -> Result<k8s_openapi::api::networking::v1::NetworkPolicy> {
    use k8s_openapi::api::networking::v1::{
        NetworkPolicy, NetworkPolicyIngressRule, NetworkPolicyPeer, NetworkPolicyPort,
        NetworkPolicySpec,
    };
    use k8s_openapi::apimachinery::pkg::apis::meta::v1::LabelSelector;

    let policy_name = format!("allow-operator-training-{}", name);

    // Create owner reference for garbage collection
    let owner_ref = OwnerReference {
        api_version: "basilica.ai/v1".into(),
        kind: "TrainingSession".into(),
        name: name.to_string(),
        uid: session.metadata.uid.clone().unwrap_or_default(),
        controller: Some(true),
        block_owner_deletion: Some(true),
    };

    let np = NetworkPolicy {
        metadata: ObjectMeta {
            name: Some(policy_name),
            namespace: Some(namespace.to_string()),
            owner_references: Some(vec![owner_ref]),
            labels: Some(BTreeMap::from([
                ("app".to_string(), "basilica-training".to_string()),
                ("session".to_string(), name.to_string()),
            ])),
            ..Default::default()
        },
        spec: Some(NetworkPolicySpec {
            pod_selector: LabelSelector {
                match_labels: Some(BTreeMap::from([
                    ("app".to_string(), "basilica-training".to_string()),
                    ("session".to_string(), name.to_string()),
                ])),
                ..Default::default()
            },
            policy_types: Some(vec!["Ingress".to_string()]),
            ingress: Some(vec![NetworkPolicyIngressRule {
                from: Some(vec![NetworkPolicyPeer {
                    namespace_selector: Some(LabelSelector {
                        match_labels: Some(BTreeMap::from([(
                            "kubernetes.io/metadata.name".to_string(),
                            "basilica-system".to_string(),
                        )])),
                        ..Default::default()
                    }),
                    ..Default::default()
                }]),
                ports: Some(vec![NetworkPolicyPort {
                    protocol: Some("TCP".to_string()),
                    port: Some(IntOrString::Int(TRAINING_SERVICE_PORT)),
                    ..Default::default()
                }]),
            }]),
            egress: None,
        }),
    };

    Ok(np)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crd::training_session::{
        CheckpointStorage, GpuResources, LoraConfig, OptimizerConfig, TrainingSessionSpec,
    };
    use crate::crd::user_deployment::StorageBackend;

    fn make_test_session() -> TrainingSession {
        let spec = TrainingSessionSpec {
            user_id: "test-user".into(),
            base_model: "meta-llama/Llama-3.1-8B".into(),
            lora_config: LoraConfig::default(),
            optimizer_config: OptimizerConfig::default(),
            checkpoint_storage: CheckpointStorage {
                backend: StorageBackend::R2,
                bucket: "test-bucket".into(),
                path: "checkpoints".into(),
                credentials_secret: None,
                region: None,
                endpoint: None,
            },
            gpu_resources: GpuResources {
                count: 1,
                model: vec!["H100".into()],
                min_memory_gb: None,
            },
            image: "basilica/training:latest".into(),
            ttl_seconds: 3600,
            seed: None,
            enable_billing: true,
        };

        TrainingSession {
            metadata: ObjectMeta {
                name: Some("test-session".into()),
                namespace: Some("default".into()),
                uid: Some("test-uid-123".into()),
                ..Default::default()
            },
            spec,
            status: None,
        }
    }

    #[test]
    fn test_build_training_pod() {
        let session = make_test_session();
        let pod = build_training_pod(&session, "default", "test-session").unwrap();

        assert_eq!(pod.metadata.name, Some("training-test-session".into()));
        assert_eq!(pod.metadata.namespace, Some("default".into()));

        let spec = pod.spec.unwrap();
        assert_eq!(spec.containers.len(), 1);

        let container = &spec.containers[0];
        assert_eq!(container.name, "training");
        assert_eq!(container.image, Some("basilica/training:latest".into()));

        // Check GPU resources
        let resources = container.resources.as_ref().unwrap();
        let limits = resources.limits.as_ref().unwrap();
        assert_eq!(limits.get("nvidia.com/gpu"), Some(&Quantity("1".into())));
    }

    #[test]
    fn test_build_training_service() {
        let session = make_test_session();
        let svc = build_training_service(&session, "default", "test-session").unwrap();

        assert_eq!(svc.metadata.name, Some("training-test-session".into()));

        let spec = svc.spec.unwrap();
        let ports = spec.ports.unwrap();
        assert_eq!(ports.len(), 1);
        assert_eq!(ports[0].port, TRAINING_SERVICE_PORT);

        let selector = spec.selector.unwrap();
        assert_eq!(selector.get("session"), Some(&"test-session".to_string()));
    }
}
