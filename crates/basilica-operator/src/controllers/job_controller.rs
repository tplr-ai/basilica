use k8s_openapi::api::batch::v1::{Job, JobSpec};
use k8s_openapi::api::core::v1::{
    Affinity, Capabilities, Container, EmptyDirVolumeSource, EnvFromSource, EnvVar,
    HostPathVolumeSource, PodSecurityContext, PodSpec, PodTemplateSpec, ResourceRequirements,
    SeccompProfile, SecretEnvSource, SecurityContext, Toleration, Volume, VolumeMount,
};
use k8s_openapi::api::core::v1::{
    NodeAffinity, NodeSelector, NodeSelectorRequirement, NodeSelectorTerm,
};
use k8s_openapi::apimachinery::pkg::api::resource::Quantity;
use k8s_openapi::apimachinery::pkg::apis::meta::v1::ObjectMeta;

use crate::billing::{BillingClient, RuntimeMetrics};
use crate::controllers::storage_utils;
use crate::crd::basilica_job::{
    BasilicaJob, BasilicaJobSpec, BasilicaJobStatus, GpuSpec as JobGpuSpec,
    Resources as JobResources,
};
use crate::k8s_client::K8sClient;
use crate::metrics as opmetrics;
use crate::metrics_provider::{NoopRuntimeMetricsProvider, RuntimeMetricsProvider};
use anyhow::Result;
use k8s_openapi::api::core::v1::PodStatus;
use std::time::Instant;

fn to_quantity(s: &str) -> Quantity {
    Quantity(s.to_string())
}

fn build_resources(res: &JobResources) -> ResourceRequirements {
    use std::collections::BTreeMap;
    let mut limits = BTreeMap::new();
    let mut requests = BTreeMap::new();

    limits.insert("cpu".to_string(), to_quantity(&res.cpu));
    limits.insert("memory".to_string(), to_quantity(&res.memory));
    requests.insert("cpu".to_string(), to_quantity(&res.cpu));
    requests.insert("memory".to_string(), to_quantity(&res.memory));

    if res.gpus.count > 0 {
        let gpuq = Quantity(res.gpus.count.to_string());
        limits.insert("nvidia.com/gpu".to_string(), gpuq.clone());
        requests.insert("nvidia.com/gpu".to_string(), gpuq);
    }

    ResourceRequirements {
        limits: Some(limits),
        requests: Some(requests),
        claims: None,
    }
}

fn build_env(env: &[crate::crd::basilica_job::EnvVar]) -> Vec<EnvVar> {
    env.iter()
        .map(|e| EnvVar {
            name: e.name.clone(),
            value: Some(e.value.clone()),
            ..Default::default()
        })
        .collect()
}

fn build_tolerations() -> Vec<Toleration> {
    vec![Toleration {
        key: Some("basilica.ai/workloads-only".into()),
        operator: Some("Equal".into()),
        value: Some("true".into()),
        effect: Some("NoSchedule".into()),
        ..Default::default()
    }]
}

fn build_node_affinity(gpu: &JobGpuSpec) -> Option<Affinity> {
    let mut match_expressions = Vec::new();

    // CRITICAL: Always require miner nodes (never schedule on control plane)
    match_expressions.push(NodeSelectorRequirement {
        key: "basilica.ai/node-role".into(),
        operator: "In".into(),
        values: Some(vec!["miner".into()]),
    });

    // CRITICAL: Always require validated nodes
    match_expressions.push(NodeSelectorRequirement {
        key: "basilica.ai/validated".into(),
        operator: "In".into(),
        values: Some(vec!["true".into()]),
    });

    // CRITICAL: Jobs must schedule on nodes in the "jobs" group
    match_expressions.push(NodeSelectorRequirement {
        key: "basilica.ai/node-group".into(),
        operator: "In".into(),
        values: Some(vec!["jobs".into()]),
    });

    // Add GPU model requirement if specified
    if !gpu.model.is_empty() {
        match_expressions.push(NodeSelectorRequirement {
            key: "basilica.ai/gpu-model".into(),
            operator: "In".into(),
            values: Some(gpu.model.clone()),
        });
    }

    let term = NodeSelectorTerm {
        match_expressions: Some(match_expressions),
        match_fields: None,
    };
    let ns = NodeSelector {
        node_selector_terms: vec![term],
    };
    Some(Affinity {
        node_affinity: Some(NodeAffinity {
            required_during_scheduling_ignored_during_execution: Some(ns),
            ..Default::default()
        }),
        ..Default::default()
    })
}

fn build_security_contexts() -> (Option<PodSecurityContext>, Option<SecurityContext>) {
    let pod_sc = Some(PodSecurityContext {
        run_as_non_root: Some(true),
        run_as_user: Some(1000), // Run as non-root user ID 1000
        seccomp_profile: Some(SeccompProfile {
            type_: "RuntimeDefault".into(),
            localhost_profile: None,
        }),
        ..Default::default()
    });
    let container_sc = Some(SecurityContext {
        allow_privilege_escalation: Some(false),
        read_only_root_filesystem: Some(true),
        capabilities: Some(Capabilities {
            drop: Some(vec!["ALL".into()]),
            ..Default::default()
        }),
        seccomp_profile: Some(SeccompProfile {
            type_: "RuntimeDefault".into(),
            localhost_profile: None,
        }),
        ..Default::default()
    });
    (pod_sc, container_sc)
}

fn build_fuse_wait_init_container(mount_path: &str) -> Container {
    Container {
        name: "wait-for-fuse".to_string(),
        image: Some("busybox:1.36".to_string()),
        command: Some(vec![
            "sh".to_string(),
            "-c".to_string(),
            format!(
                r#"
            echo "Waiting for FUSE mount at {}..."
            for i in $(seq 1 60); do
                if mountpoint -q {}; then
                    echo "FUSE mount ready"
                    exit 0
                fi
                echo "Waiting... ($i/60)"
                sleep 1
            done
            echo "ERROR: FUSE mount not ready after 60s"
            exit 1
            "#,
                mount_path, mount_path
            ),
        ]),
        volume_mounts: Some(vec![VolumeMount {
            name: "basilica-storage".to_string(),
            mount_path: mount_path.to_string(),
            mount_propagation: Some("HostToContainer".to_string()),
            ..Default::default()
        }]),
        resources: Some(ResourceRequirements {
            requests: Some(
                vec![
                    ("memory".to_string(), to_quantity("64Mi")),
                    ("cpu".to_string(), to_quantity("50m")),
                ]
                .into_iter()
                .collect(),
            ),
            limits: Some(
                vec![
                    ("memory".to_string(), to_quantity("128Mi")),
                    ("cpu".to_string(), to_quantity("100m")),
                ]
                .into_iter()
                .collect(),
            ),
            claims: None,
        }),
        ..Default::default()
    }
}

// Phase 3: Added volume support for FUSE mounts
pub fn render_job(name: &str, spec: &BasilicaJobSpec) -> Job {
    let (pod_sc, container_sc) = build_security_contexts();

    // Track persistent storage config for volume mounts
    let storage_mount_path = spec
        .storage
        .as_ref()
        .and_then(|s| s.persistent.as_ref())
        .filter(|p| p.enabled)
        .map(|p| {
            if p.mount_path.is_empty() {
                "/data".to_string()
            } else {
                p.mount_path.clone()
            }
        });

    // Main container with optional storage volume mount
    // If storage is enabled, wrap user command to wait for FUSE mount
    let (container_command, container_args) = if let Some(ref mount_path) = storage_mount_path {
        let user_cmd = if !spec.command.is_empty() {
            spec.command.join(" ")
        } else {
            // If no command provided, just wait for FUSE and then sleep
            "sleep infinity".to_string()
        };
        let user_args = if !spec.args.is_empty() {
            format!(" {}", spec.args.join(" "))
        } else {
            String::new()
        };

        // Inject wait-for-fuse logic before user command
        let wrapped_cmd = format!(
            "echo 'Waiting for FUSE mount...'; \
             while [ ! -f {}/.fuse_ready ]; do sleep 0.1; done; \
             echo 'FUSE ready, starting workload...'; \
             {}{}",
            mount_path, user_cmd, user_args
        );

        (
            Some(vec!["sh".into(), "-c".into()]),
            Some(vec![wrapped_cmd]),
        )
    } else {
        // No storage, use original command
        (
            if spec.command.is_empty() {
                None
            } else {
                Some(spec.command.clone())
            },
            if spec.args.is_empty() {
                None
            } else {
                Some(spec.args.clone())
            },
        )
    };

    let mut main_container = Container {
        name: name.to_string(),
        image: Some(spec.image.clone()),
        command: container_command,
        args: container_args,
        env: Some(build_env(&spec.env)),
        resources: Some(build_resources(&spec.resources)),
        security_context: container_sc,
        ..Default::default()
    };

    // Add volume mount to main container if storage is enabled
    if let Some(ref mount_path) = storage_mount_path {
        main_container.volume_mounts = Some(vec![VolumeMount {
            name: "basilica-storage".into(),
            mount_path: mount_path.clone(),
            ..Default::default()
        }]);
    }

    let mut containers = vec![main_container];

    // Optional persistent storage sidecar (FUSE)
    if let Some(storage) = &spec.storage {
        if let Some(persistent) = &storage.persistent {
            if persistent.enabled {
                let mut env = Vec::new();

                // Experiment ID (use job name as unique identifier)
                env.push(EnvVar {
                    name: "EXPERIMENT_ID".into(),
                    value: Some(name.to_string()),
                    ..Default::default()
                });

                // Mount point
                let mount_path = if persistent.mount_path.is_empty() {
                    "/data".to_string()
                } else {
                    persistent.mount_path.clone()
                };
                env.push(EnvVar {
                    name: "MOUNT_POINT".into(),
                    value: Some(mount_path.clone()),
                    ..Default::default()
                });

                // Sync interval
                env.push(EnvVar {
                    name: "SYNC_INTERVAL_MS".into(),
                    value: Some(persistent.sync_interval_ms.unwrap_or(1000).to_string()),
                    ..Default::default()
                });

                // Cache size
                env.push(EnvVar {
                    name: "CACHE_SIZE_MB".into(),
                    value: Some(persistent.cache_size_mb.unwrap_or(2048).to_string()),
                    ..Default::default()
                });

                // Enable debug logging for FUSE daemon
                env.push(EnvVar {
                    name: "RUST_LOG".into(),
                    value: Some("basilica_storage=debug".into()),
                    ..Default::default()
                });

                // Credentials from secret
                // Use validator-provided centralized R2 credentials by default
                // Falls back to user-provided secret for backward compatibility
                // The secret provides: STORAGE_BACKEND, STORAGE_BUCKET, STORAGE_ENDPOINT,
                // STORAGE_ACCESS_KEY_ID, STORAGE_SECRET_ACCESS_KEY
                let storage_secret = persistent
                    .credentials_secret
                    .as_ref()
                    .cloned()
                    .unwrap_or_else(|| "basilica-r2-credentials".to_string());

                let env_from = Some(vec![EnvFromSource {
                    secret_ref: Some(SecretEnvSource {
                        name: Some(storage_secret),
                        optional: Some(false),
                    }),
                    ..Default::default()
                }]);

                // Phase 3.5: Storage daemon with AWS SDK and fixed account ID extraction
                let cache_size_mb = persistent.cache_size_mb.unwrap_or(2048) as u32;
                let (startup_probe, liveness_probe, readiness_probe) =
                    storage_utils::build_fuse_health_probes();

                let storage_sidecar = Container {
                    name: format!("basilica-storage-{}", name),
                    image: Some("ghcr.io/one-covenant/basilica/storage-daemon:k3_sdk_fix".into()),
                    command: Some(vec!["/usr/local/bin/basilica-storage-daemon".into()]),
                    args: None,
                    env: Some(env),
                    env_from,
                    volume_mounts: Some(vec![
                        VolumeMount {
                            name: "fuse-device".into(),
                            mount_path: "/dev/fuse".into(),
                            ..Default::default()
                        },
                        VolumeMount {
                            name: "basilica-storage".into(),
                            mount_path: mount_path.clone(),
                            mount_propagation: Some("HostToContainer".into()),
                            ..Default::default()
                        },
                        VolumeMount {
                            name: "tmp".into(),
                            mount_path: "/tmp".into(),
                            ..Default::default()
                        },
                    ]),
                    security_context: Some(storage_utils::build_fuse_security_context()),
                    resources: Some(
                        storage_utils::build_fuse_sidecar_resources(cache_size_mb, true)
                            .expect("Valid cache size"),
                    ),
                    lifecycle: Some(storage_utils::build_fuse_lifecycle_hook(120)),
                    startup_probe,
                    liveness_probe,
                    readiness_probe,
                    ..Default::default()
                };
                containers.push(storage_sidecar);
            }
        }
    }

    // Optional artifact sidecar
    if let Some(art) = &spec.artifacts {
        if art.enabled {
            let mut env = Vec::new();
            env.push(EnvVar {
                name: "DESTINATION".into(),
                value: Some(art.destination.clone()),
                ..Default::default()
            });
            env.push(EnvVar {
                name: "FROM_PATH".into(),
                value: Some(art.from_path.clone()),
                ..Default::default()
            });
            env.push(EnvVar {
                name: "PROVIDER".into(),
                value: Some(if art.provider.is_empty() {
                    "s3".into()
                } else {
                    art.provider.clone()
                }),
                ..Default::default()
            });
            let env_from = art.credentials_secret.as_ref().map(|name| {
                vec![EnvFromSource {
                    secret_ref: Some(SecretEnvSource {
                        name: Some(name.clone()),
                        optional: Some(false),
                    }),
                    ..Default::default()
                }]
            });
            let sidecar = Container {
                name: format!("artifact-uploader-{}", name),
                image: Some("basilica/artifact-uploader:latest".into()),
                command: Some(vec!["/uploader".into()]),
                env: Some(env),
                env_from,
                security_context: Some(SecurityContext {
                    allow_privilege_escalation: Some(false),
                    read_only_root_filesystem: Some(true),
                    capabilities: Some(Capabilities {
                        drop: Some(vec!["ALL".into()]),
                        ..Default::default()
                    }),
                    seccomp_profile: Some(SeccompProfile {
                        type_: "RuntimeDefault".into(),
                        localhost_profile: None,
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            };
            containers.push(sidecar);
        }
    }

    // Add volumes for FUSE storage if enabled
    let volumes = if storage_mount_path.is_some() {
        Some(vec![
            Volume {
                name: "fuse-device".into(),
                host_path: Some(HostPathVolumeSource {
                    path: "/dev/fuse".into(),
                    type_: Some("CharDevice".into()),
                }),
                ..Default::default()
            },
            Volume {
                name: "basilica-storage".into(),
                empty_dir: Some(EmptyDirVolumeSource::default()),
                ..Default::default()
            },
            Volume {
                name: "tmp".into(),
                empty_dir: Some(EmptyDirVolumeSource::default()),
                ..Default::default()
            },
        ])
    } else {
        None
    };

    let init_containers = storage_mount_path
        .as_ref()
        .map(|mount_path| vec![build_fuse_wait_init_container(mount_path)]);

    let pod_spec = PodSpec {
        containers,
        init_containers,
        volumes,
        restart_policy: Some("Never".into()),
        termination_grace_period_seconds: if storage_mount_path.is_some() {
            Some(120)
        } else {
            None
        },
        tolerations: Some(build_tolerations()),
        security_context: pod_sc,
        affinity: build_node_affinity(&spec.resources.gpus),
        ..Default::default()
    };

    let mut labels_map: std::collections::BTreeMap<String, String> = vec![
        ("basilica.ai/type".to_string(), "job".to_string()),
        ("basilica.ai/job".to_string(), name.to_string()),
    ]
    .into_iter()
    .collect();
    let gpu_bound = (spec.resources.gpus.count > 0).to_string();
    labels_map.insert("basilica.ai/gpu-bound".to_string(), gpu_bound);
    let template = PodTemplateSpec {
        metadata: Some(ObjectMeta {
            labels: Some(labels_map.clone()),
            ..Default::default()
        }),
        spec: Some(pod_spec),
    };

    let active_deadline_seconds = if spec.ttl_seconds > 0 {
        Some(spec.ttl_seconds as i64)
    } else {
        None
    };

    Job {
        metadata: ObjectMeta {
            name: Some(name.to_string()),
            labels: Some(labels_map),
            ..Default::default()
        },
        spec: Some(JobSpec {
            template,
            backoff_limit: Some(0),
            active_deadline_seconds,
            ..Default::default()
        }),
        status: None,
    }
}

#[derive(Clone)]
pub struct JobController<C: K8sClient> {
    pub client: C,
    pub billing: std::sync::Arc<dyn BillingClient + Send + Sync>,
    pub metrics_provider: std::sync::Arc<dyn RuntimeMetricsProvider + Send + Sync>,
}

impl<C: K8sClient> JobController<C> {
    pub fn new(client: C) -> Self {
        Self {
            client,
            billing: std::sync::Arc::new(crate::billing::MockBillingClient::default()),
            metrics_provider: std::sync::Arc::new(NoopRuntimeMetricsProvider),
        }
    }
    pub fn new_with_billing(
        client: C,
        billing: std::sync::Arc<dyn BillingClient + Send + Sync>,
    ) -> Self {
        Self {
            client,
            billing,
            metrics_provider: std::sync::Arc::new(NoopRuntimeMetricsProvider),
        }
    }
    pub fn with_metrics_provider(
        mut self,
        provider: std::sync::Arc<dyn RuntimeMetricsProvider + Send + Sync>,
    ) -> Self {
        self.metrics_provider = provider;
        self
    }

    pub async fn reconcile(&self, ns: &str, cr: &BasilicaJob) -> Result<()> {
        let start = Instant::now();
        let name = cr.metadata.name.clone().unwrap_or_default();
        let spec = cr.spec.clone();
        // Enforce BasilicaQueue concurrency (if configured)
        if let Ok(queues) = self.client.list_basilica_queues(ns).await {
            if let Some(q) = queues.first() {
                // Count running job pods in namespace
                let pods = self
                    .client
                    .list_pods_with_label(ns, "basilica.ai/type", "job")
                    .await?;
                let running = pods
                    .iter()
                    .filter(|p| {
                        p.status.as_ref().and_then(|s| s.phase.as_deref()) == Some("Running")
                    })
                    .count() as u32;
                if running >= q.spec.concurrency {
                    // Mark queued and exit
                    let status = crate::crd::basilica_job::BasilicaJobStatus {
                        phase: Some("Queued".into()),
                        pod_name: None,
                        start_time: None,
                        completion_time: None,
                    };
                    self.client
                        .update_basilica_job_status(ns, &name, status)
                        .await?;
                    return Ok(());
                }
            }
        }
        // Observe previous status (if any) to record transitions
        let prev = self
            .client
            .get_basilica_job(ns, &name)
            .await
            .ok()
            .and_then(|bj| bj.status.and_then(|s| s.phase))
            .unwrap_or_else(|| "Unknown".into());

        // Ensure Job exists
        let created = if self.client.get_job(ns, &name).await.is_err() {
            let job = render_job(&name, &spec);
            self.client.create_job(ns, &job).await?;
            true
        } else {
            false
        };

        // Derive status from Pods with our label
        let pods = self
            .client
            .list_pods_with_label(ns, "basilica.ai/job", &name)
            .await?;

        let (phase, pod_name) = compute_phase_from_pods(&pods);
        let mut status = BasilicaJobStatus {
            phase: Some(phase.clone()),
            pod_name,
            start_time: None,
            completion_time: None,
        };
        // Set start_time on first Running and completion_time on terminal states
        if phase == "Running" && prev != "Running" {
            status.start_time = Some(k8s_openapi::chrono::Utc::now().to_rfc3339());
        }
        if phase == "Succeeded" || phase == "Failed" {
            status.completion_time = Some(k8s_openapi::chrono::Utc::now().to_rfc3339());
        }
        let to = status.phase.clone().unwrap_or_else(|| "Unknown".into());
        self.client
            .update_basilica_job_status(ns, &name, status.clone())
            .await?;
        // Emit job event (best-effort)
        let current = self.client.get_basilica_job(ns, &name).await?;
        let rm: Option<RuntimeMetrics> = if let Some(pod) = status.pod_name.as_deref() {
            self.metrics_provider.fetch_pod_metrics(ns, pod).await
        } else {
            None
        };
        let _ = self
            .billing
            .emit_job_event(
                &current,
                current
                    .status
                    .as_ref()
                    .unwrap_or(&BasilicaJobStatus::default()),
                rm.as_ref(),
            )
            .await;
        opmetrics::record_job_reconcile(ns, &name, created, &prev, &to, start);
        let prev_active = prev == "Running";
        let new_active = to == "Running";
        opmetrics::record_job_active_change(ns, prev_active, new_active);
        // Outcome counters
        opmetrics::record_job_outcome(ns, &to);
        Ok(())
    }
}

fn compute_phase_from_pods(pods: &[k8s_openapi::api::core::v1::Pod]) -> (String, Option<String>) {
    // Prefer running over pending; succeeded/failed if any pod indicates so.
    let mut running: Option<String> = None;
    let mut succeeded: Option<String> = None;
    let mut failed: Option<String> = None;
    let mut pending: Option<String> = None;

    for p in pods {
        let name = p.metadata.name.clone();
        if let Some(PodStatus {
            phase: Some(ph), ..
        }) = &p.status
        {
            match ph.as_str() {
                "Running" => running = name,
                "Succeeded" => succeeded = name,
                "Failed" => failed = name,
                "Pending" => pending = name,
                _ => {}
            }
        }
    }

    if let Some(n) = running {
        return ("Running".into(), Some(n));
    }
    if let Some(n) = succeeded {
        return ("Succeeded".into(), Some(n));
    }
    if let Some(n) = failed {
        return ("Failed".into(), Some(n));
    }
    if let Some(n) = pending {
        return ("Pending".into(), Some(n));
    }
    ("Pending".into(), None)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_spec() -> BasilicaJobSpec {
        BasilicaJobSpec {
            image: "nvidia/cuda:12.0-base".into(),
            command: vec!["python".into()],
            args: vec!["main.py".into()],
            env: vec![crate::crd::basilica_job::EnvVar {
                name: "FOO".into(),
                value: "bar".into(),
            }],
            resources: crate::crd::basilica_job::Resources {
                cpu: "4".into(),
                memory: "16Gi".into(),
                gpus: JobGpuSpec {
                    count: 1,
                    model: vec!["A100".into()],
                },
            },
            storage: None,
            artifacts: None,
            ttl_seconds: 3600,
            priority: "normal".into(),
        }
    }

    #[test]
    fn render_includes_resources_and_security() {
        let spec = sample_spec();
        let job = render_job("job-abc", &spec);
        let tmpl = job.spec.unwrap().template;
        let pod = tmpl.spec.unwrap();
        assert_eq!(pod.restart_policy.unwrap(), "Never");
        assert!(pod
            .security_context
            .as_ref()
            .unwrap()
            .run_as_non_root
            .unwrap());
        let c = &pod.containers[0];
        let res = c.resources.as_ref().unwrap();
        let limits = res.limits.as_ref().unwrap();
        assert_eq!(limits.get("cpu").unwrap().0, "4");
        assert_eq!(limits.get("memory").unwrap().0, "16Gi");
        assert_eq!(limits.get("nvidia.com/gpu").unwrap().0, "1");
        let sc = c.security_context.as_ref().unwrap();
        assert_eq!(sc.allow_privilege_escalation, Some(false));
        assert_eq!(sc.read_only_root_filesystem, Some(true));
        assert!(sc
            .capabilities
            .as_ref()
            .unwrap()
            .drop
            .as_ref()
            .unwrap()
            .contains(&"ALL".into()));
        assert_eq!(sc.seccomp_profile.as_ref().unwrap().type_, "RuntimeDefault");
        assert_eq!(
            pod.security_context
                .as_ref()
                .unwrap()
                .seccomp_profile
                .as_ref()
                .unwrap()
                .type_,
            "RuntimeDefault"
        );
    }

    #[test]
    fn render_includes_affinity_tolerations_and_ttl() {
        let spec = sample_spec();
        let job = render_job("job-abc", &spec);
        let jobspec = job.spec.as_ref().unwrap();
        assert_eq!(jobspec.active_deadline_seconds, Some(3600));
        let pod = jobspec.template.spec.as_ref().unwrap();
        // Tolerations
        let t = pod.tolerations.as_ref().unwrap();
        assert!(t
            .iter()
            .any(|x| x.key.as_deref() == Some("basilica.ai/workloads-only")));
        // Affinity
        let aff = pod.affinity.as_ref().unwrap();
        let node_aff = aff.node_affinity.as_ref().unwrap();
        let match_exprs = node_aff
            .required_during_scheduling_ignored_during_execution
            .as_ref()
            .unwrap()
            .node_selector_terms[0]
            .match_expressions
            .as_ref()
            .unwrap();

        // Verify GPU model requirement exists
        let gpu_req = match_exprs
            .iter()
            .find(|r| r.key == "basilica.ai/gpu-model")
            .expect("GPU model requirement should exist");
        assert_eq!(gpu_req.operator, "In");
        assert_eq!(gpu_req.values.as_ref().unwrap()[0], "A100");

        // Verify other critical requirements exist
        assert!(match_exprs.iter().any(|r| r.key == "basilica.ai/node-role"));
        assert!(match_exprs.iter().any(|r| r.key == "basilica.ai/validated"));
        assert!(match_exprs
            .iter()
            .any(|r| r.key == "basilica.ai/node-group"));
    }

    #[test]
    fn render_includes_artifact_sidecar_when_enabled() {
        let mut spec = sample_spec();
        spec.artifacts = Some(crate::crd::basilica_job::ArtifactUploadSpec {
            destination: "s3://bucket/prefix".into(),
            from_path: "/outputs".into(),
            provider: "s3".into(),
            credentials_secret: None,
            enabled: true,
        });
        let job = render_job("job-artifacts", &spec);
        let pod = job.spec.unwrap().template.spec.unwrap();
        assert!(pod
            .containers
            .iter()
            .any(|c| c.name.starts_with("artifact-uploader-")));
        let sidecar = pod
            .containers
            .iter()
            .find(|c| c.name.starts_with("artifact-uploader-"))
            .unwrap();
        let envs = sidecar.env.as_ref().unwrap();
        assert!(envs.iter().any(|e| e.name == "DESTINATION"));
        assert!(envs.iter().any(|e| e.name == "FROM_PATH"));
    }

    #[test]
    fn render_includes_storage_sidecar_when_enabled() {
        let mut spec = sample_spec();
        spec.storage = Some(crate::crd::basilica_job::StorageSpec {
            ephemeral: String::new(),
            persistent: Some(crate::crd::basilica_job::PersistentStorageSpec {
                enabled: true,
                backend: String::new(),
                bucket: String::new(),
                region: None,
                endpoint: None,
                credentials_secret: None,
                sync_interval_ms: None,
                cache_size_mb: None,
                mount_path: String::new(),
            }),
        });
        let job = render_job("storage-job", &spec);
        let pod = job.spec.unwrap().template.spec.unwrap();
        assert_eq!(
            pod.containers.len(),
            2,
            "Should have main container + storage sidecar"
        );

        // Verify volumes exist
        assert!(pod.volumes.is_some(), "Should have volumes");
        let volumes = pod.volumes.as_ref().unwrap();
        assert_eq!(
            volumes.len(),
            3,
            "Should have fuse-device, basilica-storage, and tmp volumes"
        );
        assert!(volumes.iter().any(|v| v.name == "fuse-device"));
        assert!(volumes.iter().any(|v| v.name == "basilica-storage"));
        assert!(volumes.iter().any(|v| v.name == "tmp"));

        // Verify main container has volume mount
        let main_container = &pod.containers[0];
        assert!(main_container.volume_mounts.is_some());
        let main_mounts = main_container.volume_mounts.as_ref().unwrap();
        assert!(main_mounts
            .iter()
            .any(|m| m.name == "basilica-storage" && m.mount_path == "/data"));

        // Verify storage sidecar
        let sidecar = pod
            .containers
            .iter()
            .find(|c| c.name.starts_with("basilica-storage-"))
            .expect("Storage sidecar should exist");
        assert_eq!(
            sidecar.image.as_ref().unwrap(),
            "ghcr.io/one-covenant/basilica/storage-daemon:k3_sdk_fix"
        );

        // Verify env vars
        let envs = sidecar.env.as_ref().unwrap();
        assert!(envs.iter().any(|e| e.name == "EXPERIMENT_ID"));
        assert!(envs.iter().any(|e| e.name == "MOUNT_POINT"));

        // Verify sidecar has volume mounts
        assert!(sidecar.volume_mounts.is_some());
        let sidecar_mounts = sidecar.volume_mounts.as_ref().unwrap();
        assert_eq!(
            sidecar_mounts.len(),
            3,
            "Should have fuse-device, basilica-storage, and tmp mounts"
        );

        let storage_mount = sidecar_mounts
            .iter()
            .find(|m| m.name == "basilica-storage")
            .expect("Should have storage mount");
        assert_eq!(storage_mount.mount_path, "/data");
        assert_eq!(
            storage_mount.mount_propagation.as_ref().unwrap(),
            "HostToContainer",
            "Should use HostToContainer propagation for security"
        );

        // Verify security context uses CAP_SYS_ADMIN (not privileged mode)
        let sc = sidecar
            .security_context
            .as_ref()
            .expect("Security context missing");
        assert_eq!(sc.run_as_user, Some(0), "Should run as root for FUSE");
        assert_eq!(sc.run_as_non_root, Some(false));
        assert_eq!(
            sc.privileged,
            Some(false),
            "Should NOT be privileged (use CAP_SYS_ADMIN instead)"
        );
        assert_eq!(
            sc.allow_privilege_escalation,
            Some(true),
            "Required by K8s with CAP_SYS_ADMIN"
        );
        assert_eq!(
            sc.read_only_root_filesystem,
            Some(true),
            "Should have read-only root fs"
        );
        let caps = sc.capabilities.as_ref().expect("Capabilities missing");
        assert_eq!(
            caps.drop,
            Some(vec!["ALL".into()]),
            "Should drop ALL capabilities first"
        );
        assert_eq!(
            caps.add,
            Some(vec!["SYS_ADMIN".into()]),
            "Should add CAP_SYS_ADMIN for FUSE mount/umount"
        );
        let seccomp = sc
            .seccomp_profile
            .as_ref()
            .expect("Seccomp profile missing");
        assert_eq!(
            seccomp.type_, "RuntimeDefault",
            "Should use RuntimeDefault (now actually enforced!)"
        );

        // Verify resource limits
        let resources = sidecar.resources.as_ref().expect("Resources missing");
        let limits = resources.limits.as_ref().expect("Limits missing");
        let requests = resources.requests.as_ref().expect("Requests missing");
        assert_eq!(limits.get("cpu").unwrap().0, "500m");
        assert_eq!(
            limits.get("memory").unwrap().0,
            "1Gi",
            "Memory should be 1Gi for Job (2x UserDeployment)"
        );
        assert_eq!(
            limits.get("ephemeral-storage").unwrap().0,
            "4096Mi",
            "Should have 2x cache size"
        );
        assert_eq!(requests.get("cpu").unwrap().0, "500m");
        assert_eq!(
            requests.get("memory").unwrap().0,
            "1Gi",
            "Memory should be 1Gi for Job"
        );

        // Verify lifecycle hook with timeout
        let lifecycle = sidecar.lifecycle.as_ref().expect("Lifecycle hook missing");
        let pre_stop = lifecycle.pre_stop.as_ref().expect("PreStop hook missing");
        let exec = pre_stop.exec.as_ref().expect("PreStop exec missing");
        let command = exec.command.as_ref().expect("PreStop command missing");
        assert_eq!(command.len(), 3);
        assert_eq!(command[0], "sh");
        assert_eq!(command[1], "-c");
        assert!(
            command[2].contains("timeout 120"),
            "Should have 120s timeout"
        );
        assert!(command[2].contains("kill -TERM 1"));

        // Verify termination grace period
        assert_eq!(
            pod.termination_grace_period_seconds,
            Some(120),
            "Should have 120s grace period for storage flush"
        );

        // Verify health probes
        let startup_probe = sidecar
            .startup_probe
            .as_ref()
            .expect("Startup probe missing");
        let startup_http = startup_probe
            .http_get
            .as_ref()
            .expect("Startup HTTP missing");
        assert_eq!(startup_http.path.as_ref().unwrap(), "/ready");
        assert_eq!(startup_probe.initial_delay_seconds, Some(10));
        assert_eq!(startup_probe.period_seconds, Some(5));
        assert_eq!(startup_probe.failure_threshold, Some(30));

        let liveness_probe = sidecar
            .liveness_probe
            .as_ref()
            .expect("Liveness probe missing");
        let liveness_http = liveness_probe
            .http_get
            .as_ref()
            .expect("Liveness HTTP missing");
        assert_eq!(liveness_http.path.as_ref().unwrap(), "/health");
        assert_eq!(liveness_probe.period_seconds, Some(10));
        assert_eq!(liveness_probe.failure_threshold, Some(3));

        let readiness_probe = sidecar
            .readiness_probe
            .as_ref()
            .expect("Readiness probe missing");
        let readiness_http = readiness_probe
            .http_get
            .as_ref()
            .expect("Readiness HTTP missing");
        assert_eq!(readiness_http.path.as_ref().unwrap(), "/ready");
        assert_eq!(readiness_probe.period_seconds, Some(5));
        assert_eq!(readiness_probe.failure_threshold, Some(1));

        // Verify envFrom for secret
        assert!(sidecar.env_from.is_some());
        let env_from = sidecar.env_from.as_ref().unwrap();
        assert_eq!(env_from.len(), 1);
        assert_eq!(
            env_from[0]
                .secret_ref
                .as_ref()
                .unwrap()
                .name
                .as_ref()
                .unwrap(),
            "basilica-r2-credentials"
        );
    }

    use crate::k8s_client::MockK8sClient;
    use k8s_openapi::api::core::v1::PodStatus;

    #[tokio::test]
    async fn reconcile_creates_job_and_updates_status() {
        let _ = metrics_exporter_prometheus::PrometheusBuilder::new().install_recorder();
        let client = MockK8sClient::default();
        let controller = super::JobController::new(client.clone());

        let spec = sample_spec();
        let bj = BasilicaJob::new("bj1", spec);

        // First register CR in mock and reconcile: creates Job, status pending
        controller
            .client
            .create_basilica_job("ns", &bj)
            .await
            .unwrap();
        controller.reconcile("ns", &bj).await.unwrap();
        // Create a running pod labeled for this job
        let pod = k8s_openapi::api::core::v1::Pod {
            metadata: ObjectMeta {
                name: Some("pod1".into()),
                labels: Some(
                    vec![("basilica.ai/job".into(), "bj1".into())]
                        .into_iter()
                        .collect(),
                ),
                ..Default::default()
            },
            status: Some(PodStatus {
                phase: Some("Running".into()),
                ..Default::default()
            }),
            ..Default::default()
        };
        controller.client.create_pod("ns", &pod).await.unwrap();

        // Second reconcile: sees running pod, updates status
        controller.reconcile("ns", &bj).await.unwrap();
        let updated = controller
            .client
            .get_basilica_job("ns", "bj1")
            .await
            .unwrap();
        assert_eq!(updated.status.unwrap().phase.unwrap(), "Running");
        // Exercise metrics path (no-op if already installed)
        let _ = metrics_exporter_prometheus::PrometheusBuilder::new().install_recorder();
    }
}
