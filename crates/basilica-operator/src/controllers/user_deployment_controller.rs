use k8s_openapi::api::apps::v1::{Deployment, DeploymentSpec};
use k8s_openapi::api::core::v1::{
    Affinity, NodeAffinity, NodeSelector, NodeSelectorRequirement, NodeSelectorTerm,
};
use k8s_openapi::api::core::v1::{
    Capabilities, Container, EmptyDirVolumeSource, EnvVar, EnvVarSource, HTTPGetAction,
    HostPathVolumeSource, PodSecurityContext, PodSpec, PodTemplateSpec, Probe,
    ResourceRequirements, SecretKeySelector, SecurityContext, Service, ServicePort, ServiceSpec,
    TCPSocketAction, Toleration, Volume, VolumeMount,
};
use k8s_openapi::api::networking::v1::{
    NetworkPolicy, NetworkPolicyIngressRule, NetworkPolicyPeer, NetworkPolicyPort,
    NetworkPolicySpec,
};
use k8s_openapi::apimachinery::pkg::api::resource::Quantity;
use k8s_openapi::apimachinery::pkg::apis::meta::v1::{LabelSelector, ObjectMeta};
use k8s_openapi::apimachinery::pkg::util::intstr::IntOrString;
use std::collections::BTreeMap;
use std::sync::Arc;

use crate::controllers::storage_utils;
use crate::crd::user_deployment::{EnvVar as CrdEnvVar, UserDeployment, UserDeploymentStatus};
use crate::k8s_client::K8sClient;
use anyhow::Result;
use tracing::debug;

fn deployment_needs_update(current: &Deployment, desired: &Deployment) -> bool {
    let current_spec = match &current.spec {
        Some(s) => s,
        None => return true,
    };
    let desired_spec = match &desired.spec {
        Some(s) => s,
        None => return false,
    };

    if current_spec.replicas != desired_spec.replicas {
        return true;
    }

    let current_template = &current_spec.template;
    let desired_template = &desired_spec.template;

    let current_pod_spec = match &current_template.spec {
        Some(s) => s,
        None => return true,
    };
    let desired_pod_spec = match &desired_template.spec {
        Some(s) => s,
        None => return false,
    };

    if current_pod_spec.containers.len() != desired_pod_spec.containers.len() {
        return true;
    }

    for (current_c, desired_c) in current_pod_spec
        .containers
        .iter()
        .zip(desired_pod_spec.containers.iter())
    {
        if current_c.image != desired_c.image
            || current_c.command != desired_c.command
            || current_c.args != desired_c.args
        {
            return true;
        }
    }

    false
}

fn service_needs_update(current: &Service, desired: &Service) -> bool {
    let current_spec = match &current.spec {
        Some(s) => s,
        None => return true,
    };
    let desired_spec = match &desired.spec {
        Some(s) => s,
        None => return false,
    };

    current_spec.ports != desired_spec.ports || current_spec.selector != desired_spec.selector
}

fn network_policy_needs_update(current: &NetworkPolicy, desired: &NetworkPolicy) -> bool {
    let current_spec = match &current.spec {
        Some(s) => s,
        None => return true,
    };
    let desired_spec = match &desired.spec {
        Some(s) => s,
        None => return false,
    };

    current_spec.ingress != desired_spec.ingress || current_spec.egress != desired_spec.egress
}

fn to_quantity(s: &str) -> Quantity {
    Quantity(s.to_string())
}

fn sanitize_user_id(user_id: &str) -> String {
    let mut out = String::new();
    for ch in user_id.chars() {
        if ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '-' {
            out.push(ch);
        } else if ch.is_ascii_uppercase() {
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push('-');
        }
        if out.len() >= 60 {
            break;
        }
    }
    if out.ends_with('-') {
        out.pop();
    }
    out
}

fn make_service_name(instance_name: &str) -> String {
    format!("s-{}", instance_name)
}

fn build_resources(
    cpu: &str,
    memory: &str,
    gpu: Option<&crate::crd::user_deployment::GpuSpec>,
) -> ResourceRequirements {
    build_resources_with_storage(cpu, memory, gpu, None)
}

fn build_resources_with_storage(
    cpu: &str,
    memory: &str,
    gpu: Option<&crate::crd::user_deployment::GpuSpec>,
    ephemeral_storage: Option<&str>,
) -> ResourceRequirements {
    let mut limits = BTreeMap::new();
    let mut requests = BTreeMap::new();
    limits.insert("cpu".to_string(), to_quantity(cpu));
    limits.insert("memory".to_string(), to_quantity(memory));
    requests.insert("cpu".to_string(), to_quantity(cpu));
    requests.insert("memory".to_string(), to_quantity(memory));

    if let Some(storage) = ephemeral_storage {
        limits.insert("ephemeral-storage".to_string(), to_quantity(storage));
        requests.insert("ephemeral-storage".to_string(), to_quantity(storage));
    }

    if let Some(gpu_spec) = gpu {
        let gpu_count = to_quantity(&gpu_spec.count.to_string());
        limits.insert("nvidia.com/gpu".to_string(), gpu_count.clone());
        requests.insert("nvidia.com/gpu".to_string(), gpu_count);
    }

    ResourceRequirements {
        limits: Some(limits),
        requests: Some(requests),
        claims: None,
    }
}

fn build_env(env: &[CrdEnvVar]) -> Vec<k8s_openapi::api::core::v1::EnvVar> {
    env.iter()
        .map(|e| k8s_openapi::api::core::v1::EnvVar {
            name: e.name.clone(),
            value: Some(e.value.clone()),
            ..Default::default()
        })
        .collect()
}

fn build_node_selector() -> BTreeMap<String, String> {
    let mut selector = BTreeMap::new();
    selector.insert("basilica.ai/workloads-only".to_string(), "true".to_string());
    selector
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

fn build_writable_volumes() -> (
    Vec<k8s_openapi::api::core::v1::Volume>,
    Vec<k8s_openapi::api::core::v1::VolumeMount>,
) {
    (vec![], vec![])
}

fn build_security_contexts() -> (Option<PodSecurityContext>, Option<SecurityContext>) {
    let pod_sc = Some(PodSecurityContext {
        fs_group: Some(1000),
        seccomp_profile: Some(k8s_openapi::api::core::v1::SeccompProfile {
            type_: "RuntimeDefault".into(),
            localhost_profile: None,
        }),
        ..Default::default()
    });
    let container_sc = Some(SecurityContext {
        allow_privilege_escalation: Some(false),
        capabilities: Some(Capabilities {
            drop: Some(vec!["ALL".into()]),
            add: Some(vec!["SETUID".into(), "SETGID".into(), "CHOWN".into()]),
        }),
        seccomp_profile: Some(k8s_openapi::api::core::v1::SeccompProfile {
            type_: "RuntimeDefault".into(),
            localhost_profile: None,
        }),
        ..Default::default()
    });
    (pod_sc, container_sc)
}

fn build_health_probes(
    port: u32,
    health_check: &Option<crate::crd::user_deployment::HealthCheckConfig>,
) -> (Option<Probe>, Option<Probe>) {
    match health_check {
        Some(config) => {
            let liveness_probe = config.liveness.as_ref().map(|probe_cfg| Probe {
                http_get: Some(HTTPGetAction {
                    path: Some(probe_cfg.path.clone()),
                    port: IntOrString::Int(port as i32),
                    ..Default::default()
                }),
                initial_delay_seconds: Some(probe_cfg.initial_delay_seconds as i32),
                period_seconds: Some(probe_cfg.period_seconds as i32),
                timeout_seconds: Some(probe_cfg.timeout_seconds as i32),
                failure_threshold: Some(probe_cfg.failure_threshold as i32),
                ..Default::default()
            });

            let readiness_probe = config.readiness.as_ref().map(|probe_cfg| Probe {
                http_get: Some(HTTPGetAction {
                    path: Some(probe_cfg.path.clone()),
                    port: IntOrString::Int(port as i32),
                    ..Default::default()
                }),
                initial_delay_seconds: Some(probe_cfg.initial_delay_seconds as i32),
                period_seconds: Some(probe_cfg.period_seconds as i32),
                timeout_seconds: Some(probe_cfg.timeout_seconds as i32),
                failure_threshold: Some(probe_cfg.failure_threshold as i32),
                ..Default::default()
            });

            (liveness_probe, readiness_probe)
        }
        None => {
            let liveness_probe = Some(Probe {
                tcp_socket: Some(TCPSocketAction {
                    port: IntOrString::Int(port as i32),
                    ..Default::default()
                }),
                initial_delay_seconds: Some(30),
                period_seconds: Some(10),
                timeout_seconds: Some(5),
                failure_threshold: Some(3),
                ..Default::default()
            });

            let readiness_probe = Some(Probe {
                tcp_socket: Some(TCPSocketAction {
                    port: IntOrString::Int(port as i32),
                    ..Default::default()
                }),
                initial_delay_seconds: Some(10),
                period_seconds: Some(5),
                timeout_seconds: Some(3),
                failure_threshold: Some(2),
                ..Default::default()
            });

            (liveness_probe, readiness_probe)
        }
    }
}

fn build_storage_volumes(
    instance_name: &str,
    _storage: &crate::crd::user_deployment::PersistentStorageSpec,
) -> Vec<Volume> {
    vec![
        Volume {
            name: "fuse-device".to_string(),
            host_path: Some(HostPathVolumeSource {
                path: "/dev/fuse".to_string(),
                type_: Some("CharDevice".to_string()),
            }),
            ..Default::default()
        },
        Volume {
            name: "basilica-storage".to_string(),
            host_path: Some(HostPathVolumeSource {
                path: format!("/var/lib/basilica/fuse/{}", instance_name),
                type_: Some("DirectoryOrCreate".to_string()),
            }),
            ..Default::default()
        },
        Volume {
            name: "tmp".to_string(),
            empty_dir: Some(EmptyDirVolumeSource::default()),
            ..Default::default()
        },
    ]
}

fn build_fuse_sidecar(
    instance_name: &str,
    storage: &crate::crd::user_deployment::PersistentStorageSpec,
) -> Container {
    let backend_str = match storage.backend {
        crate::crd::user_deployment::StorageBackend::R2 => "r2",
        crate::crd::user_deployment::StorageBackend::S3 => "s3",
        crate::crd::user_deployment::StorageBackend::GCS => "gcs",
    };

    let mut env = vec![
        EnvVar {
            name: "STORAGE_BACKEND".to_string(),
            value: Some(backend_str.to_string()),
            ..Default::default()
        },
        EnvVar {
            name: "STORAGE_BUCKET".to_string(),
            value: Some(storage.bucket.clone()),
            ..Default::default()
        },
        EnvVar {
            name: "MOUNT_PATH".to_string(),
            value: Some(storage.mount_path.clone()),
            ..Default::default()
        },
        EnvVar {
            name: "SYNC_INTERVAL_MS".to_string(),
            value: Some(storage.sync_interval_ms.to_string()),
            ..Default::default()
        },
        EnvVar {
            name: "CACHE_SIZE_MB".to_string(),
            value: Some(storage.cache_size_mb.to_string()),
            ..Default::default()
        },
    ];

    if let Some(ref region) = storage.region {
        env.push(EnvVar {
            name: "STORAGE_REGION".to_string(),
            value: Some(region.clone()),
            ..Default::default()
        });
    }

    if let Some(ref endpoint) = storage.endpoint {
        env.push(EnvVar {
            name: "STORAGE_ENDPOINT".to_string(),
            value: Some(endpoint.clone()),
            ..Default::default()
        });
    }

    if let Some(ref secret_name) = storage.credentials_secret {
        env.push(EnvVar {
            name: "STORAGE_ACCESS_KEY_ID".to_string(),
            value_from: Some(EnvVarSource {
                secret_key_ref: Some(SecretKeySelector {
                    name: Some(secret_name.clone()),
                    key: "STORAGE_ACCESS_KEY_ID".to_string(),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        });
        env.push(EnvVar {
            name: "STORAGE_SECRET_ACCESS_KEY".to_string(),
            value_from: Some(EnvVarSource {
                secret_key_ref: Some(SecretKeySelector {
                    name: Some(secret_name.clone()),
                    key: "STORAGE_SECRET_ACCESS_KEY".to_string(),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        });
    }

    let cache_size_mb = storage.cache_size_mb as u32;
    let (startup_probe, liveness_probe, readiness_probe) =
        storage_utils::build_fuse_health_probes();

    let args = vec![
        "--experiment-id".to_string(),
        instance_name.to_string(),
        "--bucket".to_string(),
        storage.bucket.clone(),
        "--backend".to_string(),
        format!("{:?}", storage.backend).to_lowercase(),
        "--sync-interval-ms".to_string(),
        storage.sync_interval_ms.to_string(),
        "--cache-size-mb".to_string(),
        storage.cache_size_mb.to_string(),
        "--mount-point".to_string(),
        storage.mount_path.clone(),
    ];

    Container {
        name: "fuse-storage".to_string(),
        image: Some("ghcr.io/one-covenant/basilica-storage-daemon:latest".to_string()),
        command: Some(vec!["/usr/local/bin/basilica-storage-daemon".to_string()]),
        args: Some(args),
        env: Some(env),
        volume_mounts: Some(vec![
            VolumeMount {
                name: "fuse-device".to_string(),
                mount_path: "/dev/fuse".to_string(),
                ..Default::default()
            },
            VolumeMount {
                name: "basilica-storage".to_string(),
                mount_path: storage.mount_path.clone(),
                mount_propagation: Some("Bidirectional".to_string()),
                ..Default::default()
            },
            VolumeMount {
                name: "tmp".to_string(),
                mount_path: "/tmp".to_string(),
                ..Default::default()
            },
        ]),
        security_context: Some(storage_utils::build_fuse_security_context()),
        resources: Some(
            storage_utils::build_fuse_sidecar_resources(cache_size_mb, false)
                .expect("Valid cache size"),
        ),
        lifecycle: Some(storage_utils::build_fuse_lifecycle_hook(120)),
        startup_probe,
        liveness_probe,
        readiness_probe,
        ..Default::default()
    }
}

fn build_node_affinity(gpu: &crate::crd::user_deployment::GpuSpec) -> Option<Affinity> {
    let mut match_expressions = Vec::new();

    match_expressions.push(NodeSelectorRequirement {
        key: "basilica.ai/gpu-model".to_string(),
        operator: "In".to_string(),
        values: Some(gpu.model.clone()),
    });

    if gpu.min_cuda_version.is_some() {
        match_expressions.push(NodeSelectorRequirement {
            key: "basilica.ai/cuda-version".to_string(),
            operator: "Exists".to_string(),
            values: None,
        });
    }

    if let Some(min_vram) = gpu.min_gpu_memory_gb {
        let acceptable_memory: Vec<String> = (min_vram..=256).map(|n| n.to_string()).collect();
        match_expressions.push(NodeSelectorRequirement {
            key: "basilica.ai/gpu-memory-gb".to_string(),
            operator: "In".to_string(),
            values: Some(acceptable_memory),
        });
    }

    let acceptable_counts: Vec<String> = (gpu.count..=8).map(|n| n.to_string()).collect();
    match_expressions.push(NodeSelectorRequirement {
        key: "basilica.ai/gpu-count".to_string(),
        operator: "In".to_string(),
        values: Some(acceptable_counts),
    });

    Some(Affinity {
        node_affinity: Some(NodeAffinity {
            required_during_scheduling_ignored_during_execution: Some(NodeSelector {
                node_selector_terms: vec![NodeSelectorTerm {
                    match_expressions: Some(match_expressions),
                    ..Default::default()
                }],
            }),
            ..Default::default()
        }),
        ..Default::default()
    })
}

pub fn render_deployment(
    instance_name: &str,
    namespace: &str,
    spec: &crate::crd::user_deployment::UserDeploymentSpec,
) -> Deployment {
    let (pod_sc, container_sc) = build_security_contexts();
    let (mut volumes, mut volume_mounts) = build_writable_volumes();
    let (liveness_probe, readiness_probe) = build_health_probes(spec.port, &spec.health_check);

    let storage_config = spec
        .storage
        .as_ref()
        .and_then(|s| s.persistent.as_ref())
        .filter(|p| p.enabled);

    if let Some(storage) = storage_config {
        volumes.extend(build_storage_volumes(instance_name, storage));
        volume_mounts.push(VolumeMount {
            name: "basilica-storage".to_string(),
            mount_path: storage.mount_path.clone(),
            mount_propagation: Some("HostToContainer".to_string()),
            ..Default::default()
        });
    }

    let gpu_config = spec.resources.as_ref().and_then(|r| r.gpus.as_ref());

    let node_affinity = gpu_config.and_then(build_node_affinity);

    let mut labels = BTreeMap::new();
    labels.insert("app".to_string(), instance_name.to_string());
    labels.insert(
        "basilica.ai/type".to_string(),
        "user-deployment".to_string(),
    );
    labels.insert(
        "basilica.ai/instance".to_string(),
        instance_name.to_string(),
    );
    labels.insert(
        "basilica.ai/user-id".to_string(),
        sanitize_user_id(&spec.user_id),
    );

    let resources = if let Some(ref res) = spec.resources {
        build_resources(&res.cpu, &res.memory, res.gpus.as_ref())
    } else {
        build_resources("100m", "128Mi", None)
    };

    let (container_command, container_args) = if let Some(storage) = storage_config {
        storage_utils::wrap_command_with_fuse_wait(
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
            &storage.mount_path,
        )
        .expect("shell escape should not fail for valid UTF-8 strings")
    } else {
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

    let container = Container {
        name: instance_name.to_string(),
        image: Some(spec.image.clone()),
        command: container_command,
        args: container_args,
        env: Some(build_env(&spec.env)),
        ports: Some(vec![k8s_openapi::api::core::v1::ContainerPort {
            container_port: spec.port as i32,
            protocol: Some("TCP".into()),
            ..Default::default()
        }]),
        resources: Some(resources),
        security_context: container_sc,
        volume_mounts: Some(volume_mounts),
        liveness_probe,
        readiness_probe,
        ..Default::default()
    };

    let mut containers = vec![container];
    let mut pod_annotations = BTreeMap::new();

    if let Some(storage) = storage_config {
        containers.push(build_fuse_sidecar(instance_name, storage));
        pod_annotations.insert(
            "container.apparmor.security.beta.kubernetes.io/fuse-storage".to_string(),
            "unconfined".to_string(),
        );
    }

    let pod_template = PodTemplateSpec {
        metadata: Some(ObjectMeta {
            labels: Some(labels.clone()),
            annotations: if pod_annotations.is_empty() {
                None
            } else {
                Some(pod_annotations)
            },
            ..Default::default()
        }),
        spec: Some(PodSpec {
            containers,
            security_context: pod_sc,
            termination_grace_period_seconds: Some(120),
            node_selector: Some(build_node_selector()),
            tolerations: Some(build_tolerations()),
            affinity: node_affinity,
            restart_policy: Some("Always".into()),
            automount_service_account_token: Some(false),
            volumes: Some(volumes),
            ..Default::default()
        }),
    };

    let replicas = if spec.suspended {
        0
    } else {
        spec.replicas as i32
    };

    Deployment {
        metadata: ObjectMeta {
            name: Some(format!("{}-deployment", instance_name)),
            namespace: Some(namespace.to_string()),
            labels: Some(labels.clone()),
            ..Default::default()
        },
        spec: Some(DeploymentSpec {
            replicas: Some(replicas),
            selector: LabelSelector {
                match_labels: Some({
                    let mut sel = BTreeMap::new();
                    sel.insert("app".to_string(), instance_name.to_string());
                    sel
                }),
                ..Default::default()
            },
            template: pod_template,
            ..Default::default()
        }),
        ..Default::default()
    }
}

pub fn render_service(instance_name: &str, namespace: &str, port: u32) -> Service {
    let mut labels = BTreeMap::new();
    labels.insert("app".to_string(), instance_name.to_string());
    labels.insert(
        "basilica.ai/type".to_string(),
        "user-deployment".to_string(),
    );

    let mut selector = BTreeMap::new();
    selector.insert("app".to_string(), instance_name.to_string());

    let service_name = make_service_name(instance_name);

    Service {
        metadata: ObjectMeta {
            name: Some(service_name),
            namespace: Some(namespace.to_string()),
            labels: Some(labels),
            ..Default::default()
        },
        spec: Some(ServiceSpec {
            type_: Some("ClusterIP".into()),
            selector: Some(selector),
            ports: Some(vec![ServicePort {
                port: port as i32,
                target_port: Some(IntOrString::Int(port as i32)),
                protocol: Some("TCP".into()),
                ..Default::default()
            }]),
            ..Default::default()
        }),
        ..Default::default()
    }
}

pub fn render_network_policy(instance_name: &str, namespace: &str, port: u32) -> NetworkPolicy {
    let mut labels = BTreeMap::new();
    labels.insert("app".to_string(), instance_name.to_string());
    labels.insert(
        "basilica.ai/type".to_string(),
        "user-deployment".to_string(),
    );

    let mut pod_selector_labels = BTreeMap::new();
    pod_selector_labels.insert("app".to_string(), instance_name.to_string());

    let mut envoy_namespace_labels = BTreeMap::new();
    envoy_namespace_labels.insert(
        "kubernetes.io/metadata.name".to_string(),
        "envoy-gateway-system".to_string(),
    );

    let mut envoy_pod_labels = BTreeMap::new();
    envoy_pod_labels.insert(
        "gateway.envoyproxy.io/owning-gateway-name".to_string(),
        "basilica-gateway".to_string(),
    );

    NetworkPolicy {
        metadata: ObjectMeta {
            name: Some(format!("{}-netpol", instance_name)),
            namespace: Some(namespace.to_string()),
            labels: Some(labels),
            ..Default::default()
        },
        spec: Some(NetworkPolicySpec {
            pod_selector: LabelSelector {
                match_labels: Some(pod_selector_labels),
                ..Default::default()
            },
            policy_types: Some(vec!["Ingress".into()]),
            ingress: Some(vec![NetworkPolicyIngressRule {
                from: Some(vec![NetworkPolicyPeer {
                    namespace_selector: Some(LabelSelector {
                        match_labels: Some(envoy_namespace_labels),
                        ..Default::default()
                    }),
                    pod_selector: Some(LabelSelector {
                        match_labels: Some(envoy_pod_labels),
                        ..Default::default()
                    }),
                    ..Default::default()
                }]),
                ports: Some(vec![NetworkPolicyPort {
                    port: Some(IntOrString::Int(port as i32)),
                    protocol: Some("TCP".into()),
                    ..Default::default()
                }]),
            }]),
            egress: None,
        }),
    }
}

#[derive(Clone)]
pub struct UserDeploymentController {
    client: Arc<dyn K8sClient>,
    public_ip: String,
    public_port: u16,
}

impl UserDeploymentController {
    pub fn new(client: Arc<dyn K8sClient>, public_ip: String, public_port: u16) -> Self {
        Self {
            client,
            public_ip,
            public_port,
        }
    }

    pub async fn reconcile(&self, ns: &str, cr: &UserDeployment) -> Result<()> {
        let name = cr
            .metadata
            .name
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("UserDeployment missing metadata.name"))?;
        let spec = &cr.spec;
        let instance_name = &spec.instance_name;

        let deployment_name = format!("{}-deployment", instance_name);
        let service_name = make_service_name(instance_name);
        let netpol_name = format!("{}-netpol", instance_name);

        let desired_deployment = render_deployment(instance_name, ns, spec);
        let current_deployment = self.client.get_deployment(ns, &deployment_name).await.ok();
        if current_deployment
            .as_ref()
            .map(|c| deployment_needs_update(c, &desired_deployment))
            .unwrap_or(true)
        {
            debug!("Deployment {} needs update, patching", deployment_name);
            self.client
                .patch_deployment(ns, &deployment_name, &desired_deployment)
                .await?;
        }

        let desired_service = render_service(instance_name, ns, spec.port);
        let current_service = self.client.get_service(ns, &service_name).await.ok();
        if current_service
            .as_ref()
            .map(|c| service_needs_update(c, &desired_service))
            .unwrap_or(true)
        {
            debug!("Service {} needs update, patching", service_name);
            self.client
                .patch_service(ns, &service_name, &desired_service)
                .await?;
        }

        let desired_netpol = render_network_policy(instance_name, ns, spec.port);
        let current_netpol = self.client.get_network_policy(ns, &netpol_name).await.ok();
        if current_netpol
            .as_ref()
            .map(|c| network_policy_needs_update(c, &desired_netpol))
            .unwrap_or(true)
        {
            debug!("NetworkPolicy {} needs update, patching", netpol_name);
            self.client
                .patch_network_policy(ns, &netpol_name, &desired_netpol)
                .await?;
        }

        let pods = self
            .client
            .list_pods_with_label(ns, "app", instance_name)
            .await?;

        let (state, replicas_ready) = compute_state_from_pods(&pods, spec.replicas);

        let endpoint = format!("{}.{}:{}", service_name, ns, spec.port);
        let public_url = format!(
            "http://{}:{}{}/",
            self.public_ip, self.public_port, spec.path_prefix
        );

        let mut status = UserDeploymentStatus::new()
            .with_state(&state)
            .with_deployment_name(deployment_name)
            .with_service_name(service_name)
            .with_replicas(spec.replicas, replicas_ready)
            .with_endpoint(endpoint)
            .with_public_url(public_url);

        if state == "Active" {
            if cr.status.as_ref().map(|s| s.state.as_str()).unwrap_or("") != "Active" {
                status.start_time = Some(k8s_openapi::chrono::Utc::now().to_rfc3339());
            } else if let Some(existing_status) = &cr.status {
                status.start_time = existing_status.start_time.clone();
            }
        }

        status.last_updated = k8s_openapi::chrono::Utc::now().to_rfc3339();

        self.client
            .update_user_deployment_status(ns, name, status)
            .await?;

        Ok(())
    }
}

fn compute_state_from_pods(
    pods: &[k8s_openapi::api::core::v1::Pod],
    desired_replicas: u32,
) -> (String, u32) {
    if pods.is_empty() {
        return ("Pending".to_string(), 0);
    }

    let mut running_count = 0;
    let mut failed_count = 0;
    let mut _pending_count = 0;

    for pod in pods {
        if let Some(status) = &pod.status {
            match status.phase.as_deref() {
                Some("Running") => {
                    let ready = status
                        .conditions
                        .as_ref()
                        .and_then(|conditions| {
                            conditions
                                .iter()
                                .find(|c| c.type_ == "Ready" && c.status == "True")
                        })
                        .is_some();
                    if ready {
                        running_count += 1;
                    } else {
                        _pending_count += 1;
                    }
                }
                Some("Failed") => failed_count += 1,
                Some("Succeeded") => {}
                _ => _pending_count += 1,
            }
        } else {
            _pending_count += 1;
        }
    }

    if failed_count > 0 {
        ("Failed".to_string(), running_count)
    } else if running_count == desired_replicas {
        ("Active".to_string(), running_count)
    } else {
        ("Pending".to_string(), running_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crd::user_deployment::{ResourceRequirements, UserDeploymentSpec};

    #[test]
    fn test_render_deployment() {
        let spec = UserDeploymentSpec::new(
            "user123".to_string(),
            "my-app".to_string(),
            "nginx:latest".to_string(),
            2,
            80,
            "/deployments/my-app".to_string(),
        )
        .with_resources(ResourceRequirements {
            cpu: "500m".to_string(),
            memory: "512Mi".to_string(),
            gpus: None,
        });

        let deployment = render_deployment("my-app", "u-user123", &spec);

        assert_eq!(
            deployment.metadata.name,
            Some("my-app-deployment".to_string())
        );
        assert_eq!(deployment.metadata.namespace, Some("u-user123".to_string()));

        let spec = deployment.spec.unwrap();
        assert_eq!(spec.replicas, Some(2));

        let template = spec.template;
        let pod_spec = template.spec.unwrap();

        assert_eq!(pod_spec.automount_service_account_token, Some(false));

        assert!(pod_spec.node_selector.is_some());
        let node_selector = pod_spec.node_selector.unwrap();
        assert_eq!(
            node_selector.get("basilica.ai/workloads-only"),
            Some(&"true".to_string())
        );

        assert!(pod_spec.security_context.is_some());
        let pod_sc = pod_spec.security_context.unwrap();
        assert_eq!(pod_sc.fs_group, Some(1000));

        assert_eq!(pod_spec.containers.len(), 1);
        let container = &pod_spec.containers[0];
        assert_eq!(container.name, "my-app");
        assert_eq!(container.image, Some("nginx:latest".to_string()));

        let container_sc = container.security_context.as_ref().unwrap();
        assert_eq!(container_sc.allow_privilege_escalation, Some(false));
        assert!(container_sc.capabilities.is_some());
        let caps = container_sc.capabilities.as_ref().unwrap();
        assert_eq!(caps.drop, Some(vec!["ALL".to_string()]));
    }

    #[test]
    fn test_render_service() {
        let service = render_service("my-app", "u-user123", 80);

        assert_eq!(service.metadata.name, Some("s-my-app".to_string()));
        assert_eq!(service.metadata.namespace, Some("u-user123".to_string()));

        let spec = service.spec.unwrap();
        assert_eq!(spec.type_, Some("ClusterIP".to_string()));

        let selector = spec.selector.unwrap();
        assert_eq!(selector.get("app"), Some(&"my-app".to_string()));

        let ports = spec.ports.unwrap();
        assert_eq!(ports.len(), 1);
        assert_eq!(ports[0].port, 80);
        assert_eq!(ports[0].target_port, Some(IntOrString::Int(80)));
    }

    #[test]
    fn test_render_network_policy() {
        let netpol = render_network_policy("my-app", "u-user123", 80);

        assert_eq!(netpol.metadata.name, Some("my-app-netpol".to_string()));
        assert_eq!(netpol.metadata.namespace, Some("u-user123".to_string()));

        let spec = netpol.spec.unwrap();
        assert_eq!(spec.policy_types, Some(vec!["Ingress".to_string()]));

        let pod_selector = spec.pod_selector;
        assert_eq!(
            pod_selector.match_labels.unwrap().get("app"),
            Some(&"my-app".to_string())
        );

        let ingress_rules = spec.ingress.unwrap();
        assert_eq!(ingress_rules.len(), 1);

        let rule = &ingress_rules[0];
        assert!(rule.from.is_some());
        let from_peers = rule.from.as_ref().unwrap();
        assert_eq!(from_peers.len(), 1);

        let peer = &from_peers[0];
        assert!(peer.namespace_selector.is_some());
        assert!(peer.pod_selector.is_some());

        let ports = rule.ports.as_ref().unwrap();
        assert_eq!(ports.len(), 1);
        assert_eq!(ports[0].port, Some(IntOrString::Int(80)));
        assert_eq!(ports[0].protocol, Some("TCP".to_string()));

        assert!(spec.egress.is_none());
    }

    #[test]
    fn test_compute_state_from_pods_empty() {
        let pods = vec![];
        let (state, ready) = compute_state_from_pods(&pods, 2);
        assert_eq!(state, "Pending");
        assert_eq!(ready, 0);
    }

    #[test]
    fn test_compute_state_from_pods_all_ready() {
        use k8s_openapi::api::core::v1::{Pod, PodCondition, PodStatus};

        let pod1 = Pod {
            status: Some(PodStatus {
                phase: Some("Running".to_string()),
                conditions: Some(vec![PodCondition {
                    type_: "Ready".to_string(),
                    status: "True".to_string(),
                    ..Default::default()
                }]),
                ..Default::default()
            }),
            ..Default::default()
        };

        let pod2 = pod1.clone();

        let (state, ready) = compute_state_from_pods(&[pod1, pod2], 2);
        assert_eq!(state, "Active");
        assert_eq!(ready, 2);
    }

    #[test]
    fn test_compute_state_from_pods_partial_ready() {
        use k8s_openapi::api::core::v1::{Pod, PodCondition, PodStatus};

        let ready_pod = Pod {
            status: Some(PodStatus {
                phase: Some("Running".to_string()),
                conditions: Some(vec![PodCondition {
                    type_: "Ready".to_string(),
                    status: "True".to_string(),
                    ..Default::default()
                }]),
                ..Default::default()
            }),
            ..Default::default()
        };

        let pending_pod = Pod {
            status: Some(PodStatus {
                phase: Some("Pending".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        };

        let (state, ready) = compute_state_from_pods(&[ready_pod, pending_pod], 2);
        assert_eq!(state, "Pending");
        assert_eq!(ready, 1);
    }

    #[test]
    fn test_compute_state_from_pods_with_failure() {
        use k8s_openapi::api::core::v1::{Pod, PodCondition, PodStatus};

        let ready_pod = Pod {
            status: Some(PodStatus {
                phase: Some("Running".to_string()),
                conditions: Some(vec![PodCondition {
                    type_: "Ready".to_string(),
                    status: "True".to_string(),
                    ..Default::default()
                }]),
                ..Default::default()
            }),
            ..Default::default()
        };

        let failed_pod = Pod {
            status: Some(PodStatus {
                phase: Some("Failed".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        };

        let (state, ready) = compute_state_from_pods(&[ready_pod, failed_pod], 2);
        assert_eq!(state, "Failed");
        assert_eq!(ready, 1);
    }

    #[test]
    fn test_security_contexts() {
        let (pod_sc, container_sc) = build_security_contexts();

        let pod_sc = pod_sc.unwrap();
        assert_eq!(pod_sc.fs_group, Some(1000));
        assert!(pod_sc.seccomp_profile.is_some());

        let container_sc = container_sc.unwrap();
        assert_eq!(container_sc.allow_privilege_escalation, Some(false));
        assert!(container_sc.capabilities.is_some());
        let caps = container_sc.capabilities.unwrap();
        assert_eq!(caps.drop, Some(vec!["ALL".to_string()]));
    }

    #[test]
    fn test_node_selector() {
        let selector = build_node_selector();
        assert_eq!(selector.len(), 1);
        assert_eq!(
            selector.get("basilica.ai/workloads-only"),
            Some(&"true".to_string())
        );
    }

    #[test]
    fn test_tolerations() {
        let tolerations = build_tolerations();
        assert_eq!(tolerations.len(), 1);
        assert_eq!(
            tolerations[0].key.as_deref(),
            Some("basilica.ai/workloads-only")
        );
        assert_eq!(tolerations[0].operator.as_deref(), Some("Equal"));
        assert_eq!(tolerations[0].value.as_deref(), Some("true"));
        assert_eq!(tolerations[0].effect.as_deref(), Some("NoSchedule"));
    }

    #[test]
    fn test_make_service_name() {
        assert_eq!(make_service_name("my-app"), "s-my-app");
        assert_eq!(
            make_service_name("30b9d5fe-3285-43dd-847d-2a02736ef23a"),
            "s-30b9d5fe-3285-43dd-847d-2a02736ef23a"
        );
        assert_eq!(make_service_name("abc123"), "s-abc123");
    }

    #[test]
    fn test_render_service_with_numeric_instance_name() {
        let service = render_service("30b9d5fe-3285-43dd-847d-2a02736ef23a", "u-user123", 80);

        let name = service.metadata.name.unwrap();
        assert!(name.starts_with("s-"));
        assert_eq!(name, "s-30b9d5fe-3285-43dd-847d-2a02736ef23a");

        let first_char = name.chars().next().unwrap();
        assert!(first_char.is_ascii_alphabetic());
    }

    #[test]
    fn test_render_deployment_with_gpu_node_affinity() {
        use crate::crd::user_deployment::GpuSpec;

        let spec = UserDeploymentSpec::new(
            "user123".to_string(),
            "gpu-app".to_string(),
            "pytorch:latest".to_string(),
            1,
            8080,
            "/deployments/gpu-app".to_string(),
        )
        .with_resources(ResourceRequirements {
            cpu: "4000m".to_string(),
            memory: "16Gi".to_string(),
            gpus: Some(GpuSpec {
                count: 2,
                model: vec!["A100".to_string(), "H100".to_string()],
                min_cuda_version: Some("12.2".to_string()),
                min_gpu_memory_gb: Some(40),
            }),
        });

        let deployment = render_deployment("gpu-app", "u-user123", &spec);
        let pod_spec = deployment.spec.unwrap().template.spec.unwrap();

        assert!(pod_spec.affinity.is_some());
        let affinity = pod_spec.affinity.unwrap();
        assert!(affinity.node_affinity.is_some());

        let node_affinity = affinity.node_affinity.unwrap();
        assert!(node_affinity
            .required_during_scheduling_ignored_during_execution
            .is_some());

        let node_selector = node_affinity
            .required_during_scheduling_ignored_during_execution
            .unwrap();
        let terms = node_selector.node_selector_terms;
        assert_eq!(terms.len(), 1);

        let expressions = &terms[0].match_expressions;
        assert!(expressions.is_some());
        let exprs = expressions.as_ref().unwrap();

        let gpu_model_expr = exprs
            .iter()
            .find(|e| e.key == "basilica.ai/gpu-model")
            .unwrap();
        assert_eq!(gpu_model_expr.operator, "In");
        assert_eq!(
            gpu_model_expr.values.as_ref().unwrap(),
            &vec!["A100".to_string(), "H100".to_string()]
        );

        let cuda_expr = exprs
            .iter()
            .find(|e| e.key == "basilica.ai/cuda-version")
            .unwrap();
        assert_eq!(cuda_expr.operator, "Exists");
        assert!(cuda_expr.values.is_none());

        let gpu_memory_expr = exprs
            .iter()
            .find(|e| e.key == "basilica.ai/gpu-memory-gb")
            .unwrap();
        assert_eq!(gpu_memory_expr.operator, "In");
        let memory_values = gpu_memory_expr.values.as_ref().unwrap();
        assert!(memory_values.contains(&"40".to_string()));
        assert!(memory_values.contains(&"256".to_string()));
        assert_eq!(memory_values.len(), 217);

        let gpu_count_expr = exprs
            .iter()
            .find(|e| e.key == "basilica.ai/gpu-count")
            .unwrap();
        assert_eq!(gpu_count_expr.operator, "In");
        let count_values = gpu_count_expr.values.as_ref().unwrap();
        assert_eq!(
            count_values,
            &vec![
                "2".to_string(),
                "3".to_string(),
                "4".to_string(),
                "5".to_string(),
                "6".to_string(),
                "7".to_string(),
                "8".to_string()
            ]
        );

        let container = &pod_spec.containers[0];
        let resources = container.resources.as_ref().unwrap();
        let limits = resources.limits.as_ref().unwrap();
        assert_eq!(limits.get("nvidia.com/gpu").unwrap().0, "2");
    }

    #[test]
    fn test_render_deployment_with_fuse_storage_sidecar() {
        use crate::crd::user_deployment::{PersistentStorageSpec, StorageBackend, StorageSpec};

        let spec = UserDeploymentSpec::new(
            "user123".to_string(),
            "storage-app".to_string(),
            "nginx:latest".to_string(),
            1,
            80,
            "/deployments/storage-app".to_string(),
        )
        .with_storage(StorageSpec {
            ephemeral: None,
            persistent: Some(PersistentStorageSpec {
                enabled: true,
                backend: StorageBackend::R2,
                bucket: "my-bucket".to_string(),
                region: Some("us-west-2".to_string()),
                endpoint: Some("https://r2.example.com".to_string()),
                credentials_secret: Some("r2-creds".to_string()),
                sync_interval_ms: 1000,
                cache_size_mb: 2048,
                mount_path: "/data".to_string(),
            }),
        });

        let deployment = render_deployment("storage-app", "u-user123", &spec);
        let pod_spec = deployment.spec.unwrap().template.spec.unwrap();

        assert_eq!(pod_spec.containers.len(), 2);

        let main_container = &pod_spec.containers[0];
        assert_eq!(main_container.name, "storage-app");
        let main_mounts = main_container.volume_mounts.as_ref().unwrap();
        let data_mount = main_mounts
            .iter()
            .find(|m| m.name == "basilica-storage")
            .unwrap();
        assert_eq!(data_mount.mount_path, "/data");
        assert_eq!(
            data_mount.mount_propagation,
            Some("HostToContainer".to_string())
        );

        let fuse_container = &pod_spec.containers[1];
        assert_eq!(fuse_container.name, "fuse-storage");
        assert_eq!(
            fuse_container.image,
            Some("ghcr.io/one-covenant/basilica-storage-daemon:latest".to_string())
        );

        let fuse_sc = fuse_container.security_context.as_ref().unwrap();
        assert_eq!(fuse_sc.run_as_user, Some(0), "Should run as root for FUSE");
        assert_eq!(fuse_sc.run_as_non_root, Some(false));
        assert_eq!(
            fuse_sc.privileged,
            Some(true),
            "Must use privileged mode to access /dev/fuse from hostPath"
        );
        assert_eq!(
            fuse_sc.allow_privilege_escalation,
            Some(true),
            "Required with privileged mode"
        );
        assert_eq!(fuse_sc.read_only_root_filesystem, Some(true));

        let fuse_env = fuse_container.env.as_ref().unwrap();
        assert!(fuse_env
            .iter()
            .any(|e| e.name == "STORAGE_BACKEND" && e.value.as_deref() == Some("r2")));
        assert!(fuse_env
            .iter()
            .any(|e| e.name == "STORAGE_BUCKET" && e.value.as_deref() == Some("my-bucket")));
        assert!(fuse_env
            .iter()
            .any(|e| e.name == "MOUNT_PATH" && e.value.as_deref() == Some("/data")));

        let fuse_mounts = fuse_container.volume_mounts.as_ref().unwrap();
        let fuse_storage_mount = fuse_mounts
            .iter()
            .find(|m| m.name == "basilica-storage")
            .unwrap();
        assert_eq!(
            fuse_storage_mount.mount_propagation,
            Some("Bidirectional".to_string())
        );

        assert!(
            pod_spec.init_containers.is_none(),
            "Init containers should not be present"
        );

        let main_command = main_container.command.as_ref().unwrap();
        assert_eq!(main_command[0], "/bin/sh");
        assert_eq!(main_command[1], "-c");

        let main_args = main_container.args.as_ref().unwrap();
        assert_eq!(main_args.len(), 1);
        let wrapped_script = &main_args[0];
        assert!(
            wrapped_script.contains("Waiting for FUSE mount at /data"),
            "Script should contain FUSE wait logic"
        );
        assert!(
            wrapped_script.contains(".fuse_ready"),
            "Script should check for .fuse_ready marker"
        );

        let volumes = pod_spec.volumes.as_ref().unwrap();
        assert!(volumes.iter().any(|v| v.name == "basilica-storage"));
        assert!(volumes.iter().any(|v| v.name == "fuse-device"));

        let fuse_lifecycle = fuse_container
            .lifecycle
            .as_ref()
            .expect("Lifecycle hook missing");
        let pre_stop = fuse_lifecycle
            .pre_stop
            .as_ref()
            .expect("PreStop hook missing");
        let exec = pre_stop.exec.as_ref().expect("PreStop exec missing");
        let command = exec.command.as_ref().expect("PreStop command missing");
        assert_eq!(command.len(), 3);
        assert_eq!(command[0], "sh");
        assert_eq!(command[1], "-c");
        assert!(command[2].contains("timeout 120"));
        assert!(command[2].contains("kill -TERM 1"));
        assert!(command[2].contains("while kill -0 1"));

        assert_eq!(
            pod_spec.termination_grace_period_seconds,
            Some(120),
            "terminationGracePeriodSeconds should be 120 for storage flush"
        );

        let fuse_resources = fuse_container
            .resources
            .as_ref()
            .expect("Resources missing");
        let limits = fuse_resources.limits.as_ref().expect("Limits missing");
        let requests = fuse_resources.requests.as_ref().expect("Requests missing");
        assert_eq!(
            limits.get("memory").unwrap().0,
            "512Mi",
            "Memory should be 512Mi for UserDeployment"
        );
        assert_eq!(requests.get("memory").unwrap().0, "512Mi");
        assert_eq!(limits.get("cpu").unwrap().0, "500m");
        assert_eq!(requests.get("cpu").unwrap().0, "500m");
        assert_eq!(
            limits.get("ephemeral-storage").unwrap().0,
            "4096Mi",
            "Should be 2x cache size"
        );

        let startup_probe = fuse_container
            .startup_probe
            .as_ref()
            .expect("Startup probe missing");
        let startup_http = startup_probe
            .http_get
            .as_ref()
            .expect("Startup HTTP missing");
        assert_eq!(startup_http.path.as_ref().unwrap(), "/ready");
        assert_eq!(startup_probe.initial_delay_seconds, Some(2));
        assert_eq!(startup_probe.period_seconds, Some(2));
        assert_eq!(startup_probe.failure_threshold, Some(15));

        let liveness_probe = fuse_container
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

        let readiness_probe = fuse_container
            .readiness_probe
            .as_ref()
            .expect("Readiness probe missing");
        let readiness_http = readiness_probe
            .http_get
            .as_ref()
            .expect("Readiness HTTP missing");
        assert_eq!(readiness_http.path.as_ref().unwrap(), "/ready");
        assert_eq!(readiness_probe.period_seconds, Some(3));
        assert_eq!(readiness_probe.failure_threshold, Some(1));
    }

    #[test]
    fn test_render_deployment_suspended() {
        let spec = UserDeploymentSpec::new(
            "user123".to_string(),
            "suspended-app".to_string(),
            "nginx:latest".to_string(),
            3,
            80,
            "/deployments/suspended-app".to_string(),
        )
        .suspended();

        let deployment = render_deployment("suspended-app", "u-user123", &spec);
        let deployment_spec = deployment.spec.unwrap();

        assert_eq!(deployment_spec.replicas, Some(0));
    }

    #[test]
    fn test_render_deployment_not_suspended() {
        let spec = UserDeploymentSpec::new(
            "user123".to_string(),
            "active-app".to_string(),
            "nginx:latest".to_string(),
            3,
            80,
            "/deployments/active-app".to_string(),
        );

        let deployment = render_deployment("active-app", "u-user123", &spec);
        let deployment_spec = deployment.spec.unwrap();

        assert_eq!(deployment_spec.replicas, Some(3));
    }

    #[test]
    fn test_render_deployment_gpu_without_optional_fields() {
        use crate::crd::user_deployment::GpuSpec;

        let spec = UserDeploymentSpec::new(
            "user123".to_string(),
            "minimal-gpu-app".to_string(),
            "tensorflow:latest".to_string(),
            1,
            8080,
            "/deployments/minimal-gpu-app".to_string(),
        )
        .with_resources(ResourceRequirements {
            cpu: "2000m".to_string(),
            memory: "8Gi".to_string(),
            gpus: Some(GpuSpec {
                count: 1,
                model: vec!["V100".to_string()],
                min_cuda_version: None,
                min_gpu_memory_gb: None,
            }),
        });

        let deployment = render_deployment("minimal-gpu-app", "u-user123", &spec);
        let pod_spec = deployment.spec.unwrap().template.spec.unwrap();

        assert!(pod_spec.affinity.is_some());
        let affinity = pod_spec.affinity.unwrap();
        let node_affinity = affinity.node_affinity.unwrap();
        let node_selector = node_affinity
            .required_during_scheduling_ignored_during_execution
            .unwrap();
        let terms = node_selector.node_selector_terms;
        let expressions = terms[0].match_expressions.as_ref().unwrap();

        let gpu_model_expr = expressions
            .iter()
            .find(|e| e.key == "basilica.ai/gpu-model")
            .unwrap();
        assert_eq!(
            gpu_model_expr.values.as_ref().unwrap(),
            &vec!["V100".to_string()]
        );

        let cuda_expr = expressions
            .iter()
            .find(|e| e.key == "basilica.ai/cuda-version");
        assert!(cuda_expr.is_none());

        let gpu_memory_expr = expressions
            .iter()
            .find(|e| e.key == "basilica.ai/gpu-memory-gb");
        assert!(gpu_memory_expr.is_none());

        let gpu_count_expr = expressions
            .iter()
            .find(|e| e.key == "basilica.ai/gpu-count")
            .unwrap();
        assert_eq!(gpu_count_expr.operator, "In");
        let count_values = gpu_count_expr.values.as_ref().unwrap();
        assert_eq!(
            count_values,
            &vec![
                "1".to_string(),
                "2".to_string(),
                "3".to_string(),
                "4".to_string(),
                "5".to_string(),
                "6".to_string(),
                "7".to_string(),
                "8".to_string()
            ]
        );

        let container = &pod_spec.containers[0];
        let resources = container.resources.as_ref().unwrap();
        let limits = resources.limits.as_ref().unwrap();
        assert_eq!(limits.get("nvidia.com/gpu").unwrap().0, "1");
    }
}
