use std::collections::BTreeMap;

use kube::{
    api::{Patch, PatchParams, PostParams},
    core::{ApiResource, DynamicObject, GroupVersionKind},
    Api, Client,
};

use crate::node_profile::NodeProfileSpec;

#[async_trait::async_trait]
pub trait NodeProfilePublisher: Send + Sync {
    async fn upsert_node_profile(&self, ns: &str, obj: &DynamicObject) -> anyhow::Result<()>;
    async fn apply_node_labels(
        &self,
        node_name: &str,
        labels: &BTreeMap<String, String>,
    ) -> anyhow::Result<()>;
    async fn set_node_profile_health(
        &self,
        ns: &str,
        name: &str,
        health: &str,
    ) -> anyhow::Result<()>;
}

pub struct K8sNodeProfilePublisher {
    client: Client,
}

impl K8sNodeProfilePublisher {
    pub async fn try_default() -> anyhow::Result<Self> {
        let client = Client::try_default().await?;
        Ok(Self { client })
    }

    fn cr_api(&self, ns: &str) -> Api<DynamicObject> {
        let gvk = GroupVersionKind::gvk("basilica.ai", "v1", "BasilicaNodeProfile");
        let ar = ApiResource::from_gvk(&gvk);
        Api::namespaced_with(self.client.clone(), ns, &ar)
    }

    pub fn build_node_profile_cr(
        name: &str,
        ns: &str,
        spec: &NodeProfileSpec,
        kube_node_name: Option<&str>,
        last_validated: Option<&str>,
        health: Option<&str>,
        failure_reasons: Option<&[String]>,
    ) -> anyhow::Result<DynamicObject> {
        let val = serde_json::json!({
            "apiVersion": "basilica.ai/v1",
            "kind": "BasilicaNodeProfile",
            "metadata": {"name": name, "namespace": ns},
            "spec": {
                "provider": spec.provider,
                "region": spec.region,
                "gpu": {"model": spec.gpu.model, "count": spec.gpu.count, "memoryGb": spec.gpu.memory_gb},
                "cpu": {"model": spec.cpu.model, "cores": spec.cpu.cores},
                "memoryGb": spec.memory_gb,
                "storageGb": spec.storage_gb,
                "networkGbps": spec.network_gbps,
            },
            "status": {
                "kubeNodeName": kube_node_name,
                "lastValidated": last_validated,
                "health": health,
                "failureReasons": failure_reasons,
            }
        });
        let obj: DynamicObject = serde_json::from_value(val)?;
        Ok(obj)
    }

    pub async fn upsert_node_profile(&self, ns: &str, obj: &DynamicObject) -> anyhow::Result<()> {
        let api = self.cr_api(ns);
        let name = obj.metadata.name.clone().unwrap_or_default();
        // Try create; if exists, patch status
        match api.create(&PostParams::default(), obj).await {
            Ok(_) => Ok(()),
            Err(kube::Error::Api(ae)) if ae.code == 409 => {
                // Patch status only
                let status = obj
                    .data
                    .get("status")
                    .cloned()
                    .unwrap_or(serde_json::json!({}));
                let patch = serde_json::json!({"status": status});
                let _ = api
                    .patch_status(&name, &PatchParams::default(), &Patch::Merge(&patch))
                    .await?;
                Ok(())
            }
            Err(e) => Err(e.into()),
        }
    }

    pub async fn apply_node_labels(
        &self,
        node_name: &str,
        labels: &BTreeMap<String, String>,
    ) -> anyhow::Result<()> {
        let nodes: Api<k8s_openapi::api::core::v1::Node> = Api::all(self.client.clone());
        let patch = Self::build_label_merge_patch(labels);
        let _ = nodes
            .patch(node_name, &PatchParams::default(), &Patch::Merge(&patch))
            .await?;
        Ok(())
    }

    pub async fn set_node_profile_health(
        &self,
        ns: &str,
        name: &str,
        health: &str,
    ) -> anyhow::Result<()> {
        let api = self.cr_api(ns);
        let patch = serde_json::json!({"status": {"health": health}});
        api.patch_status(name, &PatchParams::default(), &Patch::Merge(&patch))
            .await?;
        Ok(())
    }

    fn build_label_merge_patch(labels: &BTreeMap<String, String>) -> serde_json::Value {
        serde_json::json!({
            "metadata": {
                "labels": labels
            }
        })
    }
}

#[async_trait::async_trait]
impl NodeProfilePublisher for K8sNodeProfilePublisher {
    async fn upsert_node_profile(&self, ns: &str, obj: &DynamicObject) -> anyhow::Result<()> {
        Self::upsert_node_profile(self, ns, obj).await
    }

    async fn apply_node_labels(
        &self,
        node_name: &str,
        labels: &BTreeMap<String, String>,
    ) -> anyhow::Result<()> {
        Self::apply_node_labels(self, node_name, labels).await
    }

    async fn set_node_profile_health(
        &self,
        ns: &str,
        name: &str,
        health: &str,
    ) -> anyhow::Result<()> {
        Self::set_node_profile_health(self, ns, name, health).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node_profile::{NodeCpu, NodeGpu};

    #[test]
    fn builds_dynamic_cr_with_spec_and_status() {
        let spec = NodeProfileSpec {
            provider: "onprem".into(),
            region: "us-east-1".into(),
            gpu: NodeGpu {
                model: "A100".into(),
                count: 4,
                memory_gb: 80,
            },
            cpu: NodeCpu {
                model: "AMD EPYC".into(),
                cores: 64,
            },
            memory_gb: 512,
            storage_gb: 2000,
            network_gbps: 10,
        };
        let cr = K8sNodeProfilePublisher::build_node_profile_cr(
            "node-123",
            "ns",
            &spec,
            Some("kube-node-1"),
            Some("2024-10-04T00:00:00Z"),
            Some("Valid"),
            None,
        )
        .unwrap();
        assert_eq!(cr.metadata.name.as_deref(), Some("node-123"));
        let specv = cr.data.get("spec").unwrap();
        assert_eq!(
            specv.get("provider").and_then(|v| v.as_str()).unwrap(),
            "onprem"
        );
        let statusv = cr.data.get("status").unwrap();
        assert_eq!(
            statusv
                .get("kubeNodeName")
                .and_then(|v| v.as_str())
                .unwrap(),
            "kube-node-1"
        );
        assert_eq!(
            statusv.get("health").and_then(|v| v.as_str()).unwrap(),
            "Valid"
        );
    }

    #[test]
    fn builds_label_merge_patch() {
        let labels = BTreeMap::from([
            ("basilica.ai/validated".into(), "true".into()),
            ("basilica.ai/gpu-model".into(), "A100".into()),
        ]);
        let patch = K8sNodeProfilePublisher::build_label_merge_patch(&labels);
        assert_eq!(
            patch
                .get("metadata")
                .unwrap()
                .get("labels")
                .unwrap()
                .get("basilica.ai/validated")
                .and_then(|v| v.as_str())
                .unwrap(),
            "true"
        );
    }
}
