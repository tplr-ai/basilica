use serde_json::{json, Value};

use basilica_common::rental::RentalSpec;

/// Convert a RentalSpec into a Basilica GpuRental CustomResource JSON document.
pub fn rental_spec_to_gpurental_cr(name: &str, namespace: &str, spec: &RentalSpec) -> Value {
    let ports: Vec<Value> = spec
        .container
        .ports
        .iter()
        .map(|p| {
            json!({
                "containerPort": p.container_port,
                "protocol": p.protocol,
            })
        })
        .collect();
    let network = json!({
        "ingress": spec.network.ingress.iter().map(|i| json!({"port": i.port, "exposure": i.exposure})).collect::<Vec<_>>(),
        "egressPolicy": spec.network.egress_policy,
        "allowedEgress": spec.network.allowed_egress,
        "publicIpRequired": spec.network.public_ip_required,
        "bandwidthMbps": spec.network.bandwidth_mbps,
    });

    let volumes: Vec<Value> = spec
        .container
        .volumes
        .iter()
        .map(|v| {
            json!({
                "hostPath": v.host_path,
                "containerPath": v.container_path,
                "readOnly": v.read_only,
            })
        })
        .collect();

    let container = json!({
        "image": spec.container.image,
        "env": spec.container.env,
        "command": spec.container.command,
        "ports": ports,
        "volumes": volumes,
        "resources": {
            "cpu": spec.container.resources.cpu,
            "memory": spec.container.resources.memory,
            "gpus": {
                "count": spec.container.resources.gpus.count,
                "model": spec.container.resources.gpus.model,
            }
        }
    });

    let cr = json!({
        "apiVersion": "basilica.ai/v1",
        "kind": "GpuRental",
        "metadata": { "name": name, "namespace": namespace },
        "spec": {
            "container": container,
            "duration": { "hours": spec.duration.hours, "autoExtend": spec.duration.auto_extend, "maxExtensions": spec.duration.max_extensions },
            "accessType": match spec.access_type {
                basilica_common::rental::AccessType::Ssh => "Ssh",
                basilica_common::rental::AccessType::Jupyter => "Jupyter",
                basilica_common::rental::AccessType::Vscode => "Vscode",
                basilica_common::rental::AccessType::Custom => "Custom",
            },
            "network": network,
            "storage": spec.storage.as_ref().map(|s| json!({
                "persistentVolumeGb": s.persistent_volume_gb,
                "storageClass": s.storage_class,
                "mountPath": s.mount_path,
            })),
            "ttlSeconds": spec.ttl_seconds,
            "tenancy": spec.tenancy.as_ref().map(|(user_id, project_id)| json!({"userId": user_id, "projectId": project_id})),
        }
    });
    cr
}

#[cfg(test)]
mod tests {
    use super::*;
    use basilica_common::compute::{GpuSpec, Resources};
    use basilica_common::rental::{
        AccessType, IngressRule, RentalContainer, RentalDuration, RentalNetwork, RentalSpec,
        VolumeMount,
    };

    fn sample_rental() -> RentalSpec {
        RentalSpec {
            container: RentalContainer {
                image: "nvidia/cuda:12.2-base".into(),
                env: vec![("A".into(), "1".into())],
                command: vec!["bash".into(), "-lc".into(), "echo hi".into()],
                ports: vec![],
                volumes: vec![VolumeMount {
                    host_path: None,
                    container_path: "/data".into(),
                    read_only: false,
                }],
                resources: Resources {
                    cpu: "4".into(),
                    memory: "16Gi".into(),
                    gpus: GpuSpec {
                        count: 1,
                        model: vec!["A100".into()],
                    },
                },
            },
            duration: RentalDuration {
                hours: 24,
                auto_extend: true,
                max_extensions: 2,
            },
            access_type: AccessType::Ssh,
            network: RentalNetwork {
                ingress: vec![IngressRule {
                    port: 8080,
                    exposure: "NodePort".into(),
                }],
                egress_policy: "restricted".into(),
                allowed_egress: vec!["10.0.0.0/8".into()],
                public_ip_required: false,
                bandwidth_mbps: Some(100),
            },
            storage: None,
            ssh: None,
            jupyter: None,
            environment: None,
            miner_selector: None,
            billing: None,
            ttl_seconds: 0,
            tenancy: Some(("user1".into(), "proj1".into())),
        }
    }

    #[test]
    fn conversion_writes_expected_cr_fields() {
        let spec = sample_rental();
        let cr = rental_spec_to_gpurental_cr("r1", "ns1", &spec);

        assert_eq!(cr["apiVersion"], "basilica.ai/v1");
        assert_eq!(cr["kind"], "GpuRental");
        assert_eq!(cr["metadata"]["name"], "r1");
        assert_eq!(cr["metadata"]["namespace"], "ns1");
        assert_eq!(cr["spec"]["container"]["image"], "nvidia/cuda:12.2-base");
        assert_eq!(cr["spec"]["duration"]["hours"], 24);
        assert_eq!(cr["spec"]["accessType"], "Ssh");
        assert_eq!(cr["spec"]["tenancy"]["userId"], "user1");
        assert_eq!(cr["spec"]["tenancy"]["projectId"], "proj1");
    }
}
