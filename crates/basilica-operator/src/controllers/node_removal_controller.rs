use crate::crd::basilica_node_profile::BasilicaNodeProfile;
use crate::k8s_client::K8sClient;
use anyhow::Result;
use k8s_openapi::api::core::v1::Pod;

#[derive(Clone)]
pub struct NodeRemovalController<C: K8sClient> {
    pub client: C,
}

impl<C: K8sClient> NodeRemovalController<C> {
    pub fn new(client: C) -> Self {
        Self { client }
    }

    /// Reconcile a BasilicaNodeProfile and remove the corresponding Node when invalid.
    pub async fn reconcile(&self, obj: &BasilicaNodeProfile) -> Result<()> {
        let status = match &obj.status {
            Some(s) => s,
            None => return Ok(()),
        };
        // Expect an explicit health indicator and kube node name
        let health = status.health.as_deref().unwrap_or("").to_ascii_lowercase();
        if health != "invalid" {
            return Ok(());
        }
        let node_name = match status.kube_node_name.as_deref() {
            Some(n) if !n.is_empty() => n,
            _ => return Ok(()),
        };

        // 1) Cordon (best-effort)
        let _ = self.client.cordon_node(node_name).await;
        // 2) Drain: attempt eviction (PDB-aware). Best-effort; do not force delete.
        let pods: Vec<Pod> = self
            .client
            .list_pods_on_node(node_name)
            .await
            .unwrap_or_default();
        for p in &pods {
            if let (Some(ns), Some(name)) = (p.metadata.namespace.clone(), p.metadata.name.clone())
            {
                let _ = self.client.evict_pod(&ns, &name, Some(30)).await;
            }
        }
        // 3) Delete Node only when no pods remain
        let remaining: Vec<Pod> = self
            .client
            .list_pods_on_node(node_name)
            .await
            .unwrap_or_default();
        if remaining.is_empty() {
            let _ = self.client.delete_node(node_name).await;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::k8s_client::MockK8sClient;
    use k8s_openapi::apimachinery::pkg::apis::meta::v1::ObjectMeta;

    #[tokio::test]
    async fn invalid_profile_triggers_node_removal_and_pod_evictions() {
        let client = MockK8sClient::default();
        // Add a node and two pods scheduled on it in different namespaces
        client.add_node("gpu-node-1").await;
        let pod1 = Pod {
            metadata: ObjectMeta {
                name: Some("p1".into()),
                namespace: Some("ns1".into()),
                ..Default::default()
            },
            spec: Some(k8s_openapi::api::core::v1::PodSpec {
                node_name: Some("gpu-node-1".into()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let pod2 = Pod {
            metadata: ObjectMeta {
                name: Some("p2".into()),
                namespace: Some("ns2".into()),
                ..Default::default()
            },
            spec: Some(k8s_openapi::api::core::v1::PodSpec {
                node_name: Some("gpu-node-1".into()),
                ..Default::default()
            }),
            ..Default::default()
        };
        client.create_pod("ns1", &pod1).await.unwrap();
        client.create_pod("ns2", &pod2).await.unwrap();

        // Build a NodeProfile with health=Invalid and kube_node_name set
        let mut np = BasilicaNodeProfile::new(
            "np1",
            crate::crd::basilica_node_profile::BasilicaNodeProfileSpec {
                provider: "onprem".into(),
                region: "us-east-1".into(),
                gpu: crate::crd::basilica_node_profile::NodeGpu {
                    model: "A100".into(),
                    count: 1,
                    memory_gb: 80,
                    cc_capable: false,
                },
                cpu: crate::crd::basilica_node_profile::NodeCpu {
                    model: "AMD EPYC".into(),
                    cores: 64,
                },
                memory_gb: 128,
                storage_gb: 1000,
                network_gbps: 10,
                tee: None,
            },
        );
        np.status = Some(
            crate::crd::basilica_node_profile::BasilicaNodeProfileStatus {
                last_validated: None,
                kube_node_name: Some("gpu-node-1".into()),
                health: Some("Invalid".into()),
                tee_verified: false,
                tee_error: None,
            },
        );

        let ctrl = NodeRemovalController::new(client.clone());
        ctrl.reconcile(&np).await.unwrap();

        // Pods should be gone
        assert!(client.get_pod("ns1", "p1").await.is_err());
        assert!(client.get_pod("ns2", "p2").await.is_err());
        // Node should be removed
        assert!(client.get_node("gpu-node-1").await.is_err());
    }

    #[tokio::test]
    async fn eviction_blocked_keeps_node_until_cleared() {
        let client = MockK8sClient::default();
        client.add_node("gpu-node-2").await;
        let pod1 = Pod {
            metadata: ObjectMeta {
                name: Some("pa".into()),
                namespace: Some("nsA".into()),
                ..Default::default()
            },
            spec: Some(k8s_openapi::api::core::v1::PodSpec {
                node_name: Some("gpu-node-2".into()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let pod2 = Pod {
            metadata: ObjectMeta {
                name: Some("pb".into()),
                namespace: Some("nsB".into()),
                ..Default::default()
            },
            spec: Some(k8s_openapi::api::core::v1::PodSpec {
                node_name: Some("gpu-node-2".into()),
                ..Default::default()
            }),
            ..Default::default()
        };
        client.create_pod("nsA", &pod1).await.unwrap();
        client.create_pod("nsB", &pod2).await.unwrap();
        // Block eviction for pb
        client.set_evict_block("nsB", "pb", true).await;

        let mut np = BasilicaNodeProfile::new(
            "np2",
            crate::crd::basilica_node_profile::BasilicaNodeProfileSpec {
                provider: "onprem".into(),
                region: "us-east-1".into(),
                gpu: crate::crd::basilica_node_profile::NodeGpu {
                    model: "A100".into(),
                    count: 1,
                    memory_gb: 80,
                    cc_capable: false,
                },
                cpu: crate::crd::basilica_node_profile::NodeCpu {
                    model: "AMD EPYC".into(),
                    cores: 64,
                },
                memory_gb: 128,
                storage_gb: 1000,
                network_gbps: 10,
                tee: None,
            },
        );
        np.status = Some(
            crate::crd::basilica_node_profile::BasilicaNodeProfileStatus {
                last_validated: None,
                kube_node_name: Some("gpu-node-2".into()),
                health: Some("Invalid".into()),
                tee_verified: false,
                tee_error: None,
            },
        );

        let ctrl = NodeRemovalController::new(client.clone());
        ctrl.reconcile(&np).await.unwrap();
        // One pod remains; node not deleted
        assert!(client.get_pod("nsA", "pa").await.is_err());
        assert!(client.get_pod("nsB", "pb").await.is_ok());
        assert!(client.get_node("gpu-node-2").await.is_ok());

        // Unblock and reconcile again
        client.set_evict_block("nsB", "pb", false).await;
        ctrl.reconcile(&np).await.unwrap();
        assert!(client.get_pod("nsB", "pb").await.is_err());
        assert!(client.get_node("gpu-node-2").await.is_err());
    }
}
