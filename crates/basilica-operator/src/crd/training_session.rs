//! TrainingSession CRD for managing GPU training workloads.

use kube::CustomResource;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::user_deployment::StorageBackend;

/// LoRA configuration for the training session.
#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct LoraConfig {
    /// LoRA rank (default: 32)
    #[serde(default = "default_rank")]
    #[schemars(range(min = 1, max = 256))]
    pub rank: u32,

    /// LoRA alpha scaling factor (default: 64)
    #[serde(default = "default_alpha")]
    #[schemars(range(min = 1, max = 512))]
    pub alpha: u32,

    /// Dropout rate (default: 0.05)
    #[serde(default = "default_dropout")]
    #[schemars(range(min = 0.0, max = 0.5))]
    pub dropout: f32,

    /// Target modules for LoRA
    #[serde(default = "default_target_modules")]
    pub target_modules: Vec<String>,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: default_rank(),
            alpha: default_alpha(),
            dropout: default_dropout(),
            target_modules: default_target_modules(),
        }
    }
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
fn default_target_modules() -> Vec<String> {
    vec![
        "q_proj".into(),
        "k_proj".into(),
        "v_proj".into(),
        "o_proj".into(),
    ]
}

/// Optimizer configuration.
#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct OptimizerConfig {
    /// Learning rate (default: 1e-4)
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f64,

    /// Weight decay (default: 0.01)
    #[serde(default = "default_weight_decay")]
    pub weight_decay: f64,

    /// Gradient clipping (default: 1.0)
    #[serde(default = "default_grad_clip")]
    pub grad_clip: Option<f64>,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            learning_rate: default_learning_rate(),
            weight_decay: default_weight_decay(),
            grad_clip: default_grad_clip(),
        }
    }
}

fn default_learning_rate() -> f64 {
    1e-4
}
fn default_weight_decay() -> f64 {
    0.01
}
fn default_grad_clip() -> Option<f64> {
    Some(1.0)
}

/// Checkpoint storage configuration.
#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct CheckpointStorage {
    /// Storage backend: "r2", "s3", "gcs"
    pub backend: StorageBackend,

    /// Bucket name
    pub bucket: String,

    /// Path prefix within bucket
    pub path: String,

    /// Credentials secret name
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credentials_secret: Option<String>,

    /// Region for the storage bucket
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,

    /// Custom endpoint URL
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
}

/// GPU resource requirements.
#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct GpuResources {
    /// Number of GPUs (default: 1)
    #[serde(default = "default_gpu_count")]
    #[schemars(range(min = 1, max = 8))]
    pub count: u32,

    /// GPU model filter (e.g., ["A100", "H100"])
    #[serde(default)]
    pub model: Vec<String>,

    /// Minimum GPU memory in GB
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 8, max = 256))]
    pub min_memory_gb: Option<u32>,
}

impl Default for GpuResources {
    fn default() -> Self {
        Self {
            count: default_gpu_count(),
            model: Vec::new(),
            min_memory_gb: None,
        }
    }
}

fn default_gpu_count() -> u32 {
    1
}

/// TrainingSession spec.
#[derive(CustomResource, Serialize, Deserialize, Clone, Debug, JsonSchema)]
#[kube(
    group = "basilica.ai",
    version = "v1",
    kind = "TrainingSession",
    namespaced
)]
#[kube(status = "TrainingSessionStatus")]
#[kube(printcolumn = r#"{"name":"Phase", "type":"string", "jsonPath":".status.phase"}"#)]
#[kube(printcolumn = r#"{"name":"Steps", "type":"integer", "jsonPath":".status.stepsCompleted"}"#)]
#[kube(printcolumn = r#"{"name":"Model", "type":"string", "jsonPath":".spec.baseModel"}"#)]
#[kube(printcolumn = r#"{"name":"Age", "type":"date", "jsonPath":".metadata.creationTimestamp"}"#)]
#[serde(rename_all = "camelCase")]
pub struct TrainingSessionSpec {
    /// User ID owning this session
    pub user_id: String,

    /// Base model to fine-tune (HuggingFace model ID)
    pub base_model: String,

    /// LoRA configuration
    #[serde(default)]
    pub lora_config: LoraConfig,

    /// Optimizer configuration
    #[serde(default)]
    pub optimizer_config: OptimizerConfig,

    /// Checkpoint storage configuration
    pub checkpoint_storage: CheckpointStorage,

    /// GPU resource requirements
    #[serde(default)]
    pub gpu_resources: GpuResources,

    /// Training service image
    #[serde(default = "default_image")]
    pub image: String,

    /// Session TTL in seconds (default: 86400 = 24 hours)
    #[serde(default = "default_ttl")]
    pub ttl_seconds: u64,

    /// Random seed for reproducibility
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,

    /// Enable billing for this session
    #[serde(default = "default_enable_billing")]
    pub enable_billing: bool,
}

fn default_image() -> String {
    "basilica/training:latest".into()
}
fn default_ttl() -> u64 {
    86400
}
fn default_enable_billing() -> bool {
    true
}

/// Training session phase.
#[derive(Clone, Debug, Default, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TrainingSessionPhase {
    #[default]
    Pending,
    Scheduling,
    Initializing,
    LoadingModel,
    Ready,
    Suspended,
    Failed,
    Terminated,
}

impl TrainingSessionPhase {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Scheduling => "scheduling",
            Self::Initializing => "initializing",
            Self::LoadingModel => "loading_model",
            Self::Ready => "ready",
            Self::Suspended => "suspended",
            Self::Failed => "failed",
            Self::Terminated => "terminated",
        }
    }

    pub fn requeue_interval(&self) -> std::time::Duration {
        use std::time::Duration;
        match self {
            Self::Pending | Self::Scheduling => Duration::from_secs(5),
            Self::Initializing | Self::LoadingModel => Duration::from_secs(10),
            Self::Ready => Duration::from_secs(60),
            Self::Suspended => Duration::from_secs(120),
            Self::Failed | Self::Terminated => Duration::from_secs(300),
        }
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Failed | Self::Terminated)
    }
}

/// TrainingSession status.
#[derive(Clone, Debug, Default, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct TrainingSessionStatus {
    /// Current phase
    #[serde(default)]
    pub phase: TrainingSessionPhase,

    /// Training steps completed
    #[serde(default)]
    pub steps_completed: u64,

    /// Tokens processed
    #[serde(default)]
    pub tokens_processed: u64,

    /// Last checkpoint name
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_checkpoint: Option<String>,

    /// Last checkpoint path in storage
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_checkpoint_path: Option<String>,

    /// Pod name running the training service
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pod_name: Option<String>,

    /// Service endpoint for API access
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,

    /// Last activity timestamp (RFC 3339)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_activity: Option<String>,

    /// Session start time
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_time: Option<String>,

    /// Error message if failed
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,

    /// Last updated timestamp
    #[serde(default)]
    pub last_updated: String,
}

impl TrainingSessionStatus {
    pub fn new() -> Self {
        Self {
            phase: TrainingSessionPhase::Pending,
            last_updated: k8s_openapi::chrono::Utc::now().to_rfc3339(),
            ..Default::default()
        }
    }

    pub fn with_phase(mut self, phase: TrainingSessionPhase) -> Self {
        self.phase = phase;
        self.last_updated = k8s_openapi::chrono::Utc::now().to_rfc3339();
        self
    }

    pub fn with_pod_name(mut self, pod_name: String) -> Self {
        self.pod_name = Some(pod_name);
        self
    }

    pub fn with_endpoint(mut self, endpoint: String) -> Self {
        self.endpoint = Some(endpoint);
        self
    }

    pub fn with_error(mut self, error: String) -> Self {
        self.error = Some(error);
        self.phase = TrainingSessionPhase::Failed;
        self.last_updated = k8s_openapi::chrono::Utc::now().to_rfc3339();
        self
    }

    pub fn with_steps(mut self, steps: u64, tokens: u64) -> Self {
        self.steps_completed = steps;
        self.tokens_processed = tokens;
        self.last_activity = Some(k8s_openapi::chrono::Utc::now().to_rfc3339());
        self
    }

    pub fn with_checkpoint(mut self, name: String, path: String) -> Self {
        self.last_checkpoint = Some(name);
        self.last_checkpoint_path = Some(path);
        self
    }

    pub fn is_ready(&self) -> bool {
        self.phase == TrainingSessionPhase::Ready
    }

    pub fn is_failed(&self) -> bool {
        self.phase == TrainingSessionPhase::Failed
    }

    pub fn is_terminated(&self) -> bool {
        self.phase == TrainingSessionPhase::Terminated
    }
}

impl TrainingSessionSpec {
    pub fn new(
        user_id: String,
        base_model: String,
        checkpoint_storage: CheckpointStorage,
    ) -> Self {
        Self {
            user_id,
            base_model,
            checkpoint_storage,
            lora_config: LoraConfig::default(),
            optimizer_config: OptimizerConfig::default(),
            gpu_resources: GpuResources::default(),
            image: default_image(),
            ttl_seconds: default_ttl(),
            seed: None,
            enable_billing: default_enable_billing(),
        }
    }

    pub fn with_lora_config(mut self, config: LoraConfig) -> Self {
        self.lora_config = config;
        self
    }

    pub fn with_optimizer_config(mut self, config: OptimizerConfig) -> Self {
        self.optimizer_config = config;
        self
    }

    pub fn with_gpu_resources(mut self, resources: GpuResources) -> Self {
        self.gpu_resources = resources;
        self
    }

    pub fn with_image(mut self, image: String) -> Self {
        self.image = image;
        self
    }

    pub fn with_ttl(mut self, ttl_seconds: u64) -> Self {
        self.ttl_seconds = ttl_seconds;
        self
    }

    pub fn with_seed(mut self, seed: i64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn disable_billing(mut self) -> Self {
        self.enable_billing = false;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spec_builder() {
        let storage = CheckpointStorage {
            backend: StorageBackend::R2,
            bucket: "my-bucket".into(),
            path: "checkpoints/user123".into(),
            credentials_secret: Some("r2-creds".into()),
            region: None,
            endpoint: None,
        };

        let spec = TrainingSessionSpec::new(
            "user123".into(),
            "meta-llama/Llama-3.1-8B-Instruct".into(),
            storage,
        )
        .with_lora_config(LoraConfig {
            rank: 64,
            alpha: 128,
            ..Default::default()
        })
        .with_gpu_resources(GpuResources {
            count: 1,
            model: vec!["H100".into()],
            min_memory_gb: Some(80),
        })
        .with_ttl(3600);

        assert_eq!(spec.user_id, "user123");
        assert_eq!(spec.base_model, "meta-llama/Llama-3.1-8B-Instruct");
        assert_eq!(spec.lora_config.rank, 64);
        assert_eq!(spec.lora_config.alpha, 128);
        assert_eq!(spec.gpu_resources.count, 1);
        assert_eq!(spec.ttl_seconds, 3600);
    }

    #[test]
    fn test_status_builder() {
        let status = TrainingSessionStatus::new()
            .with_phase(TrainingSessionPhase::Ready)
            .with_pod_name("training-abc123".into())
            .with_endpoint("http://training-abc123.default.svc:8000".into());

        assert!(status.is_ready());
        assert!(!status.is_failed());
        assert_eq!(status.pod_name, Some("training-abc123".into()));
    }

    #[test]
    fn test_phase_methods() {
        assert!(TrainingSessionPhase::Failed.is_terminal());
        assert!(TrainingSessionPhase::Terminated.is_terminal());
        assert!(!TrainingSessionPhase::Ready.is_terminal());
        assert!(!TrainingSessionPhase::Pending.is_terminal());

        assert_eq!(TrainingSessionPhase::Ready.as_str(), "ready");
        assert_eq!(TrainingSessionPhase::LoadingModel.as_str(), "loading_model");
    }

    #[test]
    fn test_default_values() {
        let lora = LoraConfig::default();
        assert_eq!(lora.rank, 32);
        assert_eq!(lora.alpha, 64);
        assert!((lora.dropout - 0.05).abs() < f32::EPSILON);

        let optimizer = OptimizerConfig::default();
        assert!((optimizer.learning_rate - 1e-4).abs() < f64::EPSILON);
        assert_eq!(optimizer.grad_clip, Some(1.0));

        let gpu = GpuResources::default();
        assert_eq!(gpu.count, 1);
    }
}
