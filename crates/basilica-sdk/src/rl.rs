//! RL Training API client: GRPO post-training on the Basilica platform.
//!
//! Warm GPU clusters, GRPO jobs with custom rewards / datasets / in-cluster
//! LLM-judge, and the declarative manifest surface. The request DTOs here
//! mirror `basilica-api`'s `/rl/*` route DTOs field-for-field (the server
//! uses `deny_unknown_fields`, so the wire shape is the contract); keeping
//! them as shared serde types is what lets the compiler — not a runtime
//! test — catch drift from the server.

use serde::{Deserialize, Serialize};

// Client methods live in `client.rs` (the crate convention: DTOs here, the
// `impl BasilicaClient` beside the private get/post transport helpers).

// ---------------------------------------------------------------------------
// Cluster request DTOs
// ---------------------------------------------------------------------------

/// A trainer/rollout fleet's per-pod GPU shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RlGpuRequest {
    /// GPU model name, e.g. `H100`, `H200`.
    pub model: String,
    /// GPUs per pod. The trainer total (`count * replicas`) must divide the
    /// GRPO train batch — admission rejects shapes that cannot (valid: 1/2/4/8).
    pub count: u32,
    /// Optional minimum VRAM per GPU in GB. >=140 marks an H200-class fleet,
    /// which the >=16B recipe requires.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_memory_gb: Option<u32>,
}

/// One fleet (trainer or rollout).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RlFleetRequest {
    /// Pod replicas (v0 admits exactly 1 for the trainer).
    pub replicas: u32,
    /// Per-pod GPU shape.
    pub gpu: RlGpuRequest,
}

/// Create a warm RL cluster (`POST /rl/clusters`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateRlClusterRequest {
    /// Optional cluster name (DNS-1035; generated when omitted).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Base model to pin, e.g. `Qwen/Qwen2.5-7B-Instruct`. Must be on the
    /// platform allowlist; >=16B models require H200-class trainers.
    pub base_model: String,
    /// Trainer fleet (FSDP).
    pub trainer: RlFleetRequest,
    /// Rollout fleet (vLLM).
    pub rollout: RlFleetRequest,
    /// Optional idle-TTL (e.g. `30m`) after which an idle cluster reaps itself.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub idle_ttl: Option<String>,
    /// Forward-compat catch-all: fields this SDK version doesn't know are
    /// preserved verbatim on the wire (the `body=` escape hatch depends on
    /// this — server-side schema additions must never be silently dropped).
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Job request DTOs
// ---------------------------------------------------------------------------

/// In-cluster LLM-judge opt-in (WS-G v0.5). The judge is a platform-owned
/// vLLM pod serving an allowlisted open model; the reward reaches it via
/// `ctx["judge"](prompt, ...) -> str`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RlJudgeRequest {
    /// Judge model id; omit for the platform default (must be on the judge
    /// allowlist).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

/// Custom reward: a user Python scoring function run in the isolated,
/// credential-free, zero-egress executor pod.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RlRewardRequest {
    /// Reward ref, `user:<dns-1035-name>`.
    #[serde(rename = "ref")]
    pub reward_ref: String,
    /// Reward source: stdlib Python defining
    /// `reward(prompt, completion, **ctx) -> float` (<=64 KiB).
    pub source: String,
    /// Optional in-cluster LLM-judge.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub judge: Option<RlJudgeRequest>,
}

/// A public Hugging Face dataset source + column mapping.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RlHfDatasetSource {
    /// HF dataset id, `org/name` (public datasets only).
    pub repo: String,
    /// Optional HF config (subset) name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<String>,
    /// Split to load, e.g. `train`.
    pub split: String,
    /// Column holding the prompt text.
    pub prompt_column: String,
    /// Column handed to the reward as `ground_truth`.
    pub answer_column: String,
}

/// Custom dataset: platform code fetches and renders it (dataset-as-data;
/// no user code touches the data path).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RlDatasetRequest {
    /// Dataset ref, `user:<dns-1035-name>`.
    #[serde(rename = "ref")]
    pub dataset_ref: String,
    /// The public HF source + column mapping.
    pub hf: RlHfDatasetSource,
}

/// Create a GRPO training job on a Ready cluster (`POST /rl/jobs`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateRlJobRequest {
    /// Name of the warm cluster to bind to.
    pub cluster_ref: String,
    /// Optional job name (generated when omitted).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Algorithm, e.g. `grpo`.
    pub algorithm: String,
    /// Custom reward; omit for the builtin reward.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reward: Option<RlRewardRequest>,
    /// Custom dataset; omit for the builtin GSM8K dataset.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset: Option<RlDatasetRequest>,
    /// Training steps (<=3000 on custom datasets).
    pub max_steps: u32,
    /// Optional learning-rate override (string, e.g. `3.0e-6`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lr: Option<String>,
    /// Forward-compat catch-all: fields this SDK version doesn't know are
    /// preserved verbatim on the wire (the `body=` escape hatch depends on
    /// this — server-side schema additions must never be silently dropped).
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// Declarative manifest: one document that renders a cluster and/or a job
/// (`POST /rl/manifest`). Freeform to mirror the server's document surface.
pub type RlManifestRequest = serde_json::Value;

// ---------------------------------------------------------------------------
// Response DTOs
// ---------------------------------------------------------------------------

/// Response after creating a cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateRlClusterResponse {
    /// The cluster's name (its `clusterRef`).
    pub name: String,
    /// Unique identifier.
    pub uid: String,
    /// Lifecycle phase at creation (`Provisioning`).
    pub phase: String,
}

/// Cluster status (`GET /rl/clusters/{name}`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RlClusterStatusResponse {
    /// `Provisioning` | `Warming` | `Ready` | `Degraded` | `Terminating`.
    pub phase: String,
    /// Whether every fleet pod verified the pinned base model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_loaded: Option<bool>,
    /// The bound job's name, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_job_name: Option<String>,
}

/// Response after creating a job.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateRlJobResponse {
    /// The job's name (its identifier for status).
    pub name: String,
    /// Unique identifier.
    pub uid: String,
    /// Lifecycle phase at creation (`Pending`).
    pub phase: String,
    /// sha256 of the admitted reward source (absent for builtin-reward jobs).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reward_sha256: Option<String>,
}

/// Latest training metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RlJobMetrics {
    /// Latest training loss.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loss: Option<f64>,
    /// Latest mean reward.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reward_mean: Option<f64>,
    /// Latest KL divergence.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kl: Option<f64>,
}

/// Job status (`GET /rl/jobs/{name}`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RlJobStatusResponse {
    /// `Pending` | `Binding` | `Running` | `Succeeded` | `Failed` | `TimedOut`.
    pub phase: String,
    /// Best-effort latest training step.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub step: Option<i64>,
    /// Latest training metrics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics: Option<RlJobMetrics>,
    /// Object-store location of the trained model (null until bound).
    #[serde(rename = "artifactURI", skip_serializing_if = "Option::is_none")]
    pub artifact_uri: Option<String>,
}

/// Manifest submission result.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RlManifestResponse {
    /// The created cluster, when the manifest carried a `cluster` block.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cluster: Option<CreateRlClusterResponse>,
    /// The created job, when the manifest carried a `job` block.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job: Option<CreateRlJobResponse>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // The SDK request DTOs must serialize to the exact wire shape the server's
    // deny_unknown_fields DTOs accept. These pin the field renames and casing
    // that the server contract depends on.
    #[test]
    fn job_request_wire_shape() {
        let req = CreateRlJobRequest {
            cluster_ref: "my-pool".into(),
            name: None,
            algorithm: "grpo".into(),
            reward: Some(RlRewardRequest {
                reward_ref: "user:my-reward".into(),
                source: "def reward(p, c, **k):\n    return 1.0\n".into(),
                judge: Some(RlJudgeRequest { model: None }),
            }),
            dataset: Some(RlDatasetRequest {
                dataset_ref: "user:my-data".into(),
                hf: RlHfDatasetSource {
                    repo: "openai/gsm8k".into(),
                    config: Some("main".into()),
                    split: "train".into(),
                    prompt_column: "question".into(),
                    answer_column: "answer".into(),
                },
            }),
            max_steps: 50,
            lr: None,
            extra: Default::default(),
        };
        let v = serde_json::to_value(&req).unwrap();
        assert_eq!(v["clusterRef"], "my-pool");
        assert_eq!(v["maxSteps"], 50);
        // the serde renames: nested identity fields are `ref`
        assert_eq!(v["reward"]["ref"], "user:my-reward");
        assert_eq!(v["dataset"]["ref"], "user:my-data");
        assert_eq!(v["dataset"]["hf"]["promptColumn"], "question");
        // judge with no model serializes as an empty object (not null)
        assert!(v["reward"]["judge"].is_object());
        assert_eq!(v["reward"]["judge"].as_object().unwrap().len(), 0);
        // None optionals are ABSENT (deny_unknown_fields tolerates absence,
        // but a null in a non-Option server slot would 400)
        assert!(v.get("name").is_none());
        assert!(v.get("lr").is_none());
    }

    #[test]
    fn cluster_request_wire_shape() {
        let req = CreateRlClusterRequest {
            name: Some("my-pool".into()),
            base_model: "Qwen/Qwen2.5-7B-Instruct".into(),
            trainer: RlFleetRequest {
                replicas: 1,
                gpu: RlGpuRequest {
                    model: "H100".into(),
                    count: 4,
                    min_memory_gb: None,
                },
            },
            rollout: RlFleetRequest {
                replicas: 1,
                gpu: RlGpuRequest {
                    model: "H100".into(),
                    count: 4,
                    min_memory_gb: None,
                },
            },
            idle_ttl: Some("30m".into()),
            extra: Default::default(),
        };
        let v = serde_json::to_value(&req).unwrap();
        assert_eq!(v["baseModel"], "Qwen/Qwen2.5-7B-Instruct");
        assert_eq!(v["trainer"]["gpu"]["count"], 4);
        assert_eq!(v["idleTtl"], "30m");
        assert!(v["trainer"]["gpu"].get("minMemoryGb").is_none());
    }

    #[test]
    fn job_status_parses_artifact_uri_rename() {
        let body = r#"{"phase":"Succeeded","step":50,"artifactURI":"s3://x/uid"}"#;
        let s: RlJobStatusResponse = serde_json::from_str(body).unwrap();
        assert_eq!(s.phase, "Succeeded");
        assert_eq!(s.artifact_uri.as_deref(), Some("s3://x/uid"));
        assert_eq!(s.step, Some(50));
    }
}
