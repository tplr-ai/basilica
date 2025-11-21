use anyhow::{anyhow, Result};
use async_trait::async_trait;
use k8s_openapi::api::apps::v1::Deployment;
use k8s_openapi::api::batch::v1::Job;
use k8s_openapi::api::core::v1::{Node, PersistentVolumeClaim, Pod, Secret, Service};
use k8s_openapi::api::networking::v1::NetworkPolicy;
use kube::ResourceExt;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::crd::basilica_job::BasilicaJob;
use crate::crd::basilica_node_profile::BasilicaNodeProfile;
use crate::crd::basilica_queue::BasilicaQueue;
use crate::crd::gpu_rental::GpuRental;
use crate::crd::user_deployment::UserDeployment;

#[async_trait]
pub trait K8sClient: Send + Sync {
    // CRDs
    async fn create_basilica_job(&self, ns: &str, obj: &BasilicaJob) -> Result<BasilicaJob>;
    async fn get_basilica_job(&self, ns: &str, name: &str) -> Result<BasilicaJob>;
    async fn delete_basilica_job(&self, ns: &str, name: &str) -> Result<()>;
    async fn update_basilica_job_status(
        &self,
        ns: &str,
        name: &str,
        status: crate::crd::basilica_job::BasilicaJobStatus,
    ) -> Result<()>;

    async fn create_gpu_rental(&self, ns: &str, obj: &GpuRental) -> Result<GpuRental>;
    async fn get_gpu_rental(&self, ns: &str, name: &str) -> Result<GpuRental>;
    async fn delete_gpu_rental(&self, ns: &str, name: &str) -> Result<()>;
    async fn update_gpu_rental_status(
        &self,
        ns: &str,
        name: &str,
        status: crate::crd::gpu_rental::GpuRentalStatus,
    ) -> Result<()>;

    async fn create_user_deployment(
        &self,
        ns: &str,
        obj: &UserDeployment,
    ) -> Result<UserDeployment>;
    async fn get_user_deployment(&self, ns: &str, name: &str) -> Result<UserDeployment>;
    async fn delete_user_deployment(&self, ns: &str, name: &str) -> Result<()>;
    async fn update_user_deployment_status(
        &self,
        ns: &str,
        name: &str,
        status: crate::crd::user_deployment::UserDeploymentStatus,
    ) -> Result<()>;

    async fn create_node_profile(
        &self,
        ns: &str,
        obj: &BasilicaNodeProfile,
    ) -> Result<BasilicaNodeProfile>;
    async fn get_node_profile(&self, ns: &str, name: &str) -> Result<BasilicaNodeProfile>;

    // Core
    async fn create_pod(&self, ns: &str, pod: &Pod) -> Result<Pod>;
    async fn get_pod(&self, ns: &str, name: &str) -> Result<Pod>;
    async fn delete_pod(&self, ns: &str, name: &str) -> Result<()>;
    async fn list_pods_with_label(&self, ns: &str, key: &str, value: &str) -> Result<Vec<Pod>>;

    async fn create_service(&self, ns: &str, svc: &Service) -> Result<Service>;
    async fn get_service(&self, ns: &str, name: &str) -> Result<Service>;
    async fn delete_service(&self, ns: &str, name: &str) -> Result<()>;
    async fn list_services_with_label(
        &self,
        ns: &str,
        key: &str,
        value: &str,
    ) -> Result<Vec<Service>>;

    async fn create_network_policy(&self, ns: &str, np: &NetworkPolicy) -> Result<NetworkPolicy>;
    async fn get_network_policy(&self, ns: &str, name: &str) -> Result<NetworkPolicy>;
    async fn create_pvc(
        &self,
        ns: &str,
        pvc: &PersistentVolumeClaim,
    ) -> Result<PersistentVolumeClaim>;
    async fn create_secret(&self, ns: &str, secret: &Secret) -> Result<Secret>;

    async fn create_job(&self, ns: &str, job: &Job) -> Result<Job>;
    async fn get_job(&self, ns: &str, name: &str) -> Result<Job>;

    async fn create_deployment(&self, ns: &str, dep: &Deployment) -> Result<Deployment>;
    async fn get_deployment(&self, ns: &str, name: &str) -> Result<Deployment>;
    async fn patch_deployment(&self, ns: &str, name: &str, dep: &Deployment) -> Result<Deployment>;
    async fn patch_service(&self, ns: &str, name: &str, svc: &Service) -> Result<Service>;
    async fn patch_network_policy(
        &self,
        ns: &str,
        name: &str,
        np: &NetworkPolicy,
    ) -> Result<NetworkPolicy>;

    // Queues
    async fn create_basilica_queue(&self, ns: &str, obj: &BasilicaQueue) -> Result<BasilicaQueue>;
    async fn list_basilica_queues(&self, ns: &str) -> Result<Vec<BasilicaQueue>>;

    // Node management (cordon/drain/delete)
    async fn cordon_node(&self, name: &str) -> Result<()>;
    async fn list_pods_on_node(&self, node_name: &str) -> Result<Vec<Pod>>;
    async fn delete_node(&self, name: &str) -> Result<()>;
    async fn remove_node_taint(&self, name: &str, taint_key: &str) -> Result<()>;
    /// Attempt to evict a pod using the Eviction subresource. Returns Ok(true)
    /// when accepted, Ok(false) when blocked (e.g., by PDB), or Err on errors.
    async fn evict_pod(&self, ns: &str, name: &str, grace_seconds: Option<i64>) -> Result<bool>;

    /// Create a Gateway API HTTPRoute (dynamic object)
    async fn create_http_route(
        &self,
        ns: &str,
        obj: &kube::core::DynamicObject,
    ) -> Result<kube::core::DynamicObject>;
}

/// In-memory mock client implementing a subset of the Kubernetes API for tests.
#[derive(Default, Clone)]
pub struct MockK8sClient {
    // namespace -> (name -> obj)
    jobs: Arc<RwLock<HashMap<String, HashMap<String, Job>>>>,
    pods: Arc<RwLock<HashMap<String, HashMap<String, Pod>>>>,
    services: Arc<RwLock<HashMap<String, HashMap<String, Service>>>>,
    network_policies: Arc<RwLock<HashMap<String, HashMap<String, NetworkPolicy>>>>,
    pvcs: Arc<RwLock<HashMap<String, HashMap<String, PersistentVolumeClaim>>>>,
    secrets: Arc<RwLock<HashMap<String, HashMap<String, Secret>>>>,
    rent_crds: Arc<RwLock<HashMap<String, HashMap<String, GpuRental>>>>,
    job_crds: Arc<RwLock<HashMap<String, HashMap<String, BasilicaJob>>>>,
    queue_crds: Arc<RwLock<HashMap<String, HashMap<String, BasilicaQueue>>>>,
    user_deployment_crds: Arc<RwLock<HashMap<String, HashMap<String, UserDeployment>>>>,
    node_profile_crds: Arc<RwLock<HashMap<String, HashMap<String, BasilicaNodeProfile>>>>,
    deployments: Arc<RwLock<HashMap<String, HashMap<String, Deployment>>>>,
    nodes: Arc<RwLock<HashMap<String, Node>>>,
    evict_block: Arc<RwLock<std::collections::HashSet<String>>>,
    http_routes: Arc<RwLock<HashMap<String, HashMap<String, kube::core::DynamicObject>>>>,
}

fn key(ns: &str) -> String {
    ns.to_string()
}

#[async_trait]
impl K8sClient for MockK8sClient {
    async fn create_basilica_job(&self, ns: &str, obj: &BasilicaJob) -> Result<BasilicaJob> {
        let name = obj.name_any();
        if name.is_empty() {
            return Err(anyhow!("BasilicaJob missing metadata.name"));
        }
        let mut map = self.job_crds.write().await;
        map.entry(key(ns))
            .or_default()
            .insert(name.clone(), obj.clone());
        Ok(obj.clone())
    }

    async fn get_basilica_job(&self, ns: &str, name: &str) -> Result<BasilicaJob> {
        let map = self.job_crds.read().await;
        map.get(ns)
            .and_then(|m| m.get(name))
            .cloned()
            .ok_or_else(|| anyhow!("BasilicaJob not found: {}/{}", ns, name))
    }

    async fn delete_basilica_job(&self, ns: &str, name: &str) -> Result<()> {
        let mut map = self.job_crds.write().await;
        map.get_mut(ns).and_then(|m| m.remove(name));
        Ok(())
    }

    async fn update_basilica_job_status(
        &self,
        ns: &str,
        name: &str,
        status: crate::crd::basilica_job::BasilicaJobStatus,
    ) -> Result<()> {
        let mut map = self.job_crds.write().await;
        let ns_map = map
            .get_mut(ns)
            .ok_or_else(|| anyhow!("namespace not found: {}", ns))?;
        let bj = ns_map
            .get_mut(name)
            .ok_or_else(|| anyhow!("BasilicaJob not found: {}/{}", ns, name))?;
        bj.status = Some(status);
        Ok(())
    }

    async fn create_gpu_rental(&self, ns: &str, obj: &GpuRental) -> Result<GpuRental> {
        let name = obj.name_any();
        if name.is_empty() {
            return Err(anyhow!("GpuRental missing metadata.name"));
        }
        let mut map = self.rent_crds.write().await;
        map.entry(key(ns))
            .or_default()
            .insert(name.clone(), obj.clone());
        Ok(obj.clone())
    }

    async fn get_gpu_rental(&self, ns: &str, name: &str) -> Result<GpuRental> {
        let map = self.rent_crds.read().await;
        map.get(ns)
            .and_then(|m| m.get(name))
            .cloned()
            .ok_or_else(|| anyhow!("GpuRental not found: {}/{}", ns, name))
    }

    async fn delete_gpu_rental(&self, ns: &str, name: &str) -> Result<()> {
        let mut map = self.rent_crds.write().await;
        map.get_mut(ns).and_then(|m| m.remove(name));
        Ok(())
    }

    async fn update_gpu_rental_status(
        &self,
        ns: &str,
        name: &str,
        status: crate::crd::gpu_rental::GpuRentalStatus,
    ) -> Result<()> {
        let mut map = self.rent_crds.write().await;
        let ns_map = map
            .get_mut(ns)
            .ok_or_else(|| anyhow!("namespace not found: {}", ns))?;
        let gr = ns_map
            .get_mut(name)
            .ok_or_else(|| anyhow!("GpuRental not found: {}/{}", ns, name))?;
        gr.status = Some(status);
        Ok(())
    }

    async fn create_user_deployment(
        &self,
        ns: &str,
        obj: &UserDeployment,
    ) -> Result<UserDeployment> {
        let name = obj.name_any();
        if name.is_empty() {
            return Err(anyhow!("UserDeployment missing metadata.name"));
        }
        let mut map = self.user_deployment_crds.write().await;
        map.entry(key(ns))
            .or_default()
            .insert(name.clone(), obj.clone());
        Ok(obj.clone())
    }

    async fn get_user_deployment(&self, ns: &str, name: &str) -> Result<UserDeployment> {
        let map = self.user_deployment_crds.read().await;
        map.get(ns)
            .and_then(|m| m.get(name))
            .cloned()
            .ok_or_else(|| anyhow!("UserDeployment not found: {}/{}", ns, name))
    }

    async fn delete_user_deployment(&self, ns: &str, name: &str) -> Result<()> {
        let mut map = self.user_deployment_crds.write().await;
        map.get_mut(ns).and_then(|m| m.remove(name));
        Ok(())
    }

    async fn update_user_deployment_status(
        &self,
        ns: &str,
        name: &str,
        status: crate::crd::user_deployment::UserDeploymentStatus,
    ) -> Result<()> {
        let mut map = self.user_deployment_crds.write().await;
        let ns_map = map
            .get_mut(ns)
            .ok_or_else(|| anyhow!("namespace not found: {}", ns))?;
        let ud = ns_map
            .get_mut(name)
            .ok_or_else(|| anyhow!("UserDeployment not found: {}/{}", ns, name))?;
        ud.status = Some(status);
        Ok(())
    }

    async fn create_node_profile(
        &self,
        ns: &str,
        obj: &BasilicaNodeProfile,
    ) -> Result<BasilicaNodeProfile> {
        let name = obj.name_any();
        if name.is_empty() {
            return Err(anyhow!("BasilicaNodeProfile missing metadata.name"));
        }
        let mut map = self.node_profile_crds.write().await;
        map.entry(key(ns))
            .or_default()
            .insert(name.clone(), obj.clone());
        Ok(obj.clone())
    }

    async fn get_node_profile(&self, ns: &str, name: &str) -> Result<BasilicaNodeProfile> {
        let map = self.node_profile_crds.read().await;
        map.get(ns)
            .and_then(|m| m.get(name))
            .cloned()
            .ok_or_else(|| anyhow!("BasilicaNodeProfile not found: {}/{}", ns, name))
    }

    async fn create_pod(&self, ns: &str, pod: &Pod) -> Result<Pod> {
        let name = pod.name_any();
        if name.is_empty() {
            return Err(anyhow!("Pod missing metadata.name"));
        }
        let mut map = self.pods.write().await;
        map.entry(key(ns))
            .or_default()
            .insert(name.clone(), pod.clone());
        Ok(pod.clone())
    }

    async fn get_pod(&self, ns: &str, name: &str) -> Result<Pod> {
        let map = self.pods.read().await;
        map.get(ns)
            .and_then(|m| m.get(name))
            .cloned()
            .ok_or_else(|| anyhow!("Pod not found: {}/{}", ns, name))
    }

    async fn delete_pod(&self, ns: &str, name: &str) -> Result<()> {
        let mut map = self.pods.write().await;
        map.get_mut(ns).and_then(|m| m.remove(name));
        Ok(())
    }

    async fn list_pods_with_label(&self, ns: &str, key: &str, value: &str) -> Result<Vec<Pod>> {
        let map = self.pods.read().await;
        let list = map
            .get(ns)
            .map(|m| {
                m.values()
                    .filter(|p| {
                        p.labels()
                            .get(key)
                            .map(|v| v.as_str() == value)
                            .unwrap_or(false)
                    })
                    .cloned()
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        Ok(list)
    }

    async fn create_service(&self, ns: &str, svc: &Service) -> Result<Service> {
        let name = svc.metadata.name.clone().unwrap_or_default();
        if name.is_empty() {
            return Err(anyhow!("Service missing metadata.name"));
        }
        let mut map = self.services.write().await;
        map.entry(key(ns))
            .or_default()
            .insert(name.clone(), svc.clone());
        Ok(svc.clone())
    }

    async fn get_service(&self, ns: &str, name: &str) -> Result<Service> {
        let map = self.services.read().await;
        map.get(ns)
            .and_then(|m| m.get(name))
            .cloned()
            .ok_or_else(|| anyhow!("Service not found: {}/{}", ns, name))
    }

    async fn delete_service(&self, ns: &str, name: &str) -> Result<()> {
        let mut map = self.services.write().await;
        map.get_mut(ns).and_then(|m| m.remove(name));
        Ok(())
    }

    async fn list_services_with_label(
        &self,
        ns: &str,
        key: &str,
        value: &str,
    ) -> Result<Vec<Service>> {
        let map = self.services.read().await;
        let list = map
            .get(ns)
            .map(|m| {
                m.values()
                    .filter(|s| {
                        s.metadata
                            .labels
                            .as_ref()
                            .and_then(|l| l.get(key))
                            .map(|v| v == value)
                            .unwrap_or(false)
                    })
                    .cloned()
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        Ok(list)
    }

    async fn create_network_policy(&self, ns: &str, np: &NetworkPolicy) -> Result<NetworkPolicy> {
        let name = np.metadata.name.clone().unwrap_or_default();
        if name.is_empty() {
            return Err(anyhow!("NetworkPolicy missing metadata.name"));
        }
        let mut map = self.network_policies.write().await;
        map.entry(key(ns))
            .or_default()
            .insert(name.clone(), np.clone());
        Ok(np.clone())
    }

    async fn get_network_policy(&self, ns: &str, name: &str) -> Result<NetworkPolicy> {
        let map = self.network_policies.read().await;
        map.get(&key(ns))
            .and_then(|m| m.get(name))
            .cloned()
            .ok_or_else(|| anyhow!("NetworkPolicy {} not found in namespace {}", name, ns))
    }

    async fn create_pvc(
        &self,
        ns: &str,
        pvc: &PersistentVolumeClaim,
    ) -> Result<PersistentVolumeClaim> {
        let name = pvc.metadata.name.clone().unwrap_or_default();
        if name.is_empty() {
            return Err(anyhow!("PVC missing metadata.name"));
        }
        let mut map = self.pvcs.write().await;
        map.entry(key(ns))
            .or_default()
            .insert(name.clone(), pvc.clone());
        Ok(pvc.clone())
    }

    async fn create_secret(&self, ns: &str, secret: &Secret) -> Result<Secret> {
        let name = secret.metadata.name.clone().unwrap_or_default();
        if name.is_empty() {
            return Err(anyhow!("Secret missing metadata.name"));
        }
        let mut map = self.secrets.write().await;
        map.entry(key(ns))
            .or_default()
            .insert(name.clone(), secret.clone());
        Ok(secret.clone())
    }

    async fn create_job(&self, ns: &str, job: &Job) -> Result<Job> {
        let name = job.metadata.name.clone().unwrap_or_default();
        if name.is_empty() {
            return Err(anyhow!("Job missing metadata.name"));
        }
        let mut map = self.jobs.write().await;
        map.entry(key(ns))
            .or_default()
            .insert(name.clone(), job.clone());
        Ok(job.clone())
    }

    async fn get_job(&self, ns: &str, name: &str) -> Result<Job> {
        let map = self.jobs.read().await;
        map.get(ns)
            .and_then(|m| m.get(name))
            .cloned()
            .ok_or_else(|| anyhow!("Job not found: {}/{}", ns, name))
    }

    async fn create_deployment(&self, ns: &str, dep: &Deployment) -> Result<Deployment> {
        let name = dep.metadata.name.clone().unwrap_or_default();
        if name.is_empty() {
            return Err(anyhow!("Deployment missing metadata.name"));
        }
        let mut map = self.deployments.write().await;
        map.entry(key(ns))
            .or_default()
            .insert(name.clone(), dep.clone());
        Ok(dep.clone())
    }

    async fn get_deployment(&self, ns: &str, name: &str) -> Result<Deployment> {
        let map = self.deployments.read().await;
        map.get(ns)
            .and_then(|m| m.get(name))
            .cloned()
            .ok_or_else(|| anyhow!("Deployment not found: {}/{}", ns, name))
    }

    async fn patch_deployment(
        &self,
        ns: &str,
        _name: &str,
        dep: &Deployment,
    ) -> Result<Deployment> {
        let name = dep.name_any();
        let mut map = self.deployments.write().await;
        map.entry(ns.to_string())
            .or_default()
            .insert(name.clone(), dep.clone());
        Ok(dep.clone())
    }

    async fn patch_service(&self, ns: &str, _name: &str, svc: &Service) -> Result<Service> {
        let name = svc.name_any();
        let mut map = self.services.write().await;
        map.entry(ns.to_string())
            .or_default()
            .insert(name.clone(), svc.clone());
        Ok(svc.clone())
    }

    async fn patch_network_policy(
        &self,
        ns: &str,
        _name: &str,
        np: &NetworkPolicy,
    ) -> Result<NetworkPolicy> {
        let name = np.name_any();
        let mut map = self.network_policies.write().await;
        map.entry(ns.to_string())
            .or_default()
            .insert(name.clone(), np.clone());
        Ok(np.clone())
    }

    async fn create_basilica_queue(&self, ns: &str, obj: &BasilicaQueue) -> Result<BasilicaQueue> {
        let name = obj.name_any();
        if name.is_empty() {
            return Err(anyhow!("BasilicaQueue missing metadata.name"));
        }
        let mut map = self.queue_crds.write().await;
        map.entry(key(ns))
            .or_default()
            .insert(name.clone(), obj.clone());
        Ok(obj.clone())
    }

    async fn list_basilica_queues(&self, ns: &str) -> Result<Vec<BasilicaQueue>> {
        let map = self.queue_crds.read().await;
        Ok(map
            .get(ns)
            .map(|m| m.values().cloned().collect())
            .unwrap_or_default())
    }

    async fn cordon_node(&self, name: &str) -> Result<()> {
        let mut nodes = self.nodes.write().await;
        if let Some(node) = nodes.get_mut(name) {
            let mut spec = node.spec.clone().unwrap_or_default();
            spec.unschedulable = Some(true);
            node.spec = Some(spec);
        }
        Ok(())
    }

    async fn list_pods_on_node(&self, node_name: &str) -> Result<Vec<Pod>> {
        let pods = self.pods.read().await;
        let mut out = Vec::new();
        for (_ns, map) in pods.iter() {
            for p in map.values() {
                if let Some(spec) = &p.spec {
                    if spec.node_name.as_deref() == Some(node_name) {
                        out.push(p.clone());
                    }
                }
            }
        }
        Ok(out)
    }

    async fn delete_node(&self, name: &str) -> Result<()> {
        let mut nodes = self.nodes.write().await;
        nodes.remove(name);
        Ok(())
    }

    async fn remove_node_taint(&self, name: &str, taint_key: &str) -> Result<()> {
        let mut nodes = self.nodes.write().await;
        if let Some(node) = nodes.get_mut(name) {
            if let Some(spec) = &mut node.spec {
                if let Some(taints) = &mut spec.taints {
                    taints.retain(|t| t.key != taint_key);
                }
            }
        }
        Ok(())
    }

    async fn evict_pod(&self, ns: &str, name: &str, _grace_seconds: Option<i64>) -> Result<bool> {
        let key = format!("{}/{}", ns, name);
        if self.evict_block.read().await.contains(&key) {
            return Ok(false);
        }
        let mut pods = self.pods.write().await;
        if let Some(nsmap) = pods.get_mut(ns) {
            nsmap.remove(name);
        }
        Ok(true)
    }

    async fn create_http_route(
        &self,
        ns: &str,
        obj: &kube::core::DynamicObject,
    ) -> Result<kube::core::DynamicObject> {
        let name = obj.metadata.name.clone().unwrap_or_default();
        if name.is_empty() {
            return Err(anyhow!("HTTPRoute missing metadata.name"));
        }
        let mut map = self.http_routes.write().await;
        map.entry(key(ns))
            .or_default()
            .insert(name.clone(), obj.clone());
        Ok(obj.clone())
    }
}

impl MockK8sClient {
    /// Test helpers to manipulate Nodes in the mock
    pub async fn add_node(&self, name: &str) {
        use k8s_openapi::apimachinery::pkg::apis::meta::v1::ObjectMeta;
        let node = Node {
            metadata: ObjectMeta {
                name: Some(name.to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        let mut nodes = self.nodes.write().await;
        nodes.insert(name.to_string(), node);
    }

    pub async fn get_node(&self, name: &str) -> Result<Node> {
        let nodes = self.nodes.read().await;
        nodes
            .get(name)
            .cloned()
            .ok_or_else(|| anyhow!("Node not found: {}", name))
    }

    /// Configure a pod eviction block (namespace/name). When blocked, evict_pod returns Ok(false).
    pub async fn set_evict_block(&self, ns: &str, name: &str, block: bool) {
        let key = format!("{}/{}", ns, name);
        let mut blk = self.evict_block.write().await;
        if block {
            blk.insert(key);
        } else {
            blk.remove(&key);
        }
    }

    /// List stored HTTPRoutes in the mock (test helper)
    pub async fn list_http_routes(&self, ns: &str) -> Vec<kube::core::DynamicObject> {
        let map = self.http_routes.read().await;
        map.get(ns)
            .map(|m| m.values().cloned().collect())
            .unwrap_or_default()
    }
}

/// Real Kubernetes client backed by kube
#[derive(Clone)]
pub struct KubeClient {
    pub client: kube::Client,
}

impl KubeClient {
    pub async fn try_default() -> Result<Self> {
        let client = kube::Client::try_default().await?;
        Ok(Self { client })
    }

    fn api<T>(&self, ns: &str) -> kube::Api<T>
    where
        T: kube::Resource<DynamicType = (), Scope = kube::core::NamespaceResourceScope>
            + Clone
            + serde::de::DeserializeOwned
            + serde::Serialize
            + 'static,
    {
        kube::Api::namespaced(self.client.clone(), ns)
    }
}

#[async_trait]
impl K8sClient for KubeClient {
    async fn create_basilica_job(
        &self,
        ns: &str,
        obj: &crate::crd::basilica_job::BasilicaJob,
    ) -> Result<crate::crd::basilica_job::BasilicaJob> {
        use kube::api::PostParams;
        let api: kube::Api<crate::crd::basilica_job::BasilicaJob> = self.api(ns);
        match api.create(&PostParams::default(), obj).await {
            Ok(o) => Ok(o),
            Err(kube::Error::Api(ae)) if ae.code == 409 => Ok(obj.clone()),
            Err(e) => Err(anyhow!(e)),
        }
    }

    async fn get_basilica_job(
        &self,
        ns: &str,
        name: &str,
    ) -> Result<crate::crd::basilica_job::BasilicaJob> {
        let api: kube::Api<crate::crd::basilica_job::BasilicaJob> = self.api(ns);
        api.get(name).await.map_err(|e| anyhow!(e))
    }

    async fn delete_basilica_job(&self, ns: &str, name: &str) -> Result<()> {
        use kube::api::DeleteParams;
        let api: kube::Api<crate::crd::basilica_job::BasilicaJob> = self.api(ns);
        api.delete(name, &DeleteParams::default())
            .await
            .map(|_| ())
            .map_err(|e| anyhow!(e))
    }

    async fn update_basilica_job_status(
        &self,
        ns: &str,
        name: &str,
        status: crate::crd::basilica_job::BasilicaJobStatus,
    ) -> Result<()> {
        use kube::api::{Patch, PatchParams};
        let api: kube::Api<crate::crd::basilica_job::BasilicaJob> = self.api(ns);
        let patch = serde_json::json!({"status": status});
        api.patch_status(name, &PatchParams::default(), &Patch::Merge(&patch))
            .await
            .map(|_| ())
            .map_err(|e| anyhow!(e))
    }

    async fn create_gpu_rental(
        &self,
        ns: &str,
        obj: &crate::crd::gpu_rental::GpuRental,
    ) -> Result<crate::crd::gpu_rental::GpuRental> {
        use kube::api::PostParams;
        let api: kube::Api<crate::crd::gpu_rental::GpuRental> = self.api(ns);
        match api.create(&PostParams::default(), obj).await {
            Ok(o) => Ok(o),
            Err(kube::Error::Api(ae)) if ae.code == 409 => Ok(obj.clone()),
            Err(e) => Err(anyhow!(e)),
        }
    }

    async fn get_gpu_rental(
        &self,
        ns: &str,
        name: &str,
    ) -> Result<crate::crd::gpu_rental::GpuRental> {
        let api: kube::Api<crate::crd::gpu_rental::GpuRental> = self.api(ns);
        api.get(name).await.map_err(|e| anyhow!(e))
    }

    async fn delete_gpu_rental(&self, ns: &str, name: &str) -> Result<()> {
        use kube::api::DeleteParams;
        let api: kube::Api<crate::crd::gpu_rental::GpuRental> = self.api(ns);
        api.delete(name, &DeleteParams::default())
            .await
            .map(|_| ())
            .map_err(|e| anyhow!(e))
    }

    async fn update_gpu_rental_status(
        &self,
        ns: &str,
        name: &str,
        status: crate::crd::gpu_rental::GpuRentalStatus,
    ) -> Result<()> {
        use kube::api::{Patch, PatchParams};
        let api: kube::Api<crate::crd::gpu_rental::GpuRental> = self.api(ns);
        let patch = serde_json::json!({"status": status});
        api.patch_status(name, &PatchParams::default(), &Patch::Merge(&patch))
            .await
            .map(|_| ())
            .map_err(|e| anyhow!(e))
    }

    async fn create_user_deployment(
        &self,
        ns: &str,
        obj: &UserDeployment,
    ) -> Result<UserDeployment> {
        use kube::api::PostParams;
        let api: kube::Api<UserDeployment> = self.api(ns);
        match api.create(&PostParams::default(), obj).await {
            Ok(o) => Ok(o),
            Err(kube::Error::Api(ae)) if ae.code == 409 => Ok(obj.clone()),
            Err(e) => Err(anyhow!(e)),
        }
    }

    async fn get_user_deployment(&self, ns: &str, name: &str) -> Result<UserDeployment> {
        let api: kube::Api<UserDeployment> = self.api(ns);
        api.get(name).await.map_err(|e| anyhow!(e))
    }

    async fn delete_user_deployment(&self, ns: &str, name: &str) -> Result<()> {
        use kube::api::DeleteParams;
        let api: kube::Api<UserDeployment> = self.api(ns);
        api.delete(name, &DeleteParams::default())
            .await
            .map(|_| ())
            .map_err(|e| anyhow!(e))
    }

    async fn update_user_deployment_status(
        &self,
        ns: &str,
        name: &str,
        status: crate::crd::user_deployment::UserDeploymentStatus,
    ) -> Result<()> {
        use kube::api::{Patch, PatchParams};
        let api: kube::Api<UserDeployment> = self.api(ns);
        let patch = serde_json::json!({"status": status});
        api.patch_status(name, &PatchParams::default(), &Patch::Merge(&patch))
            .await
            .map(|_| ())
            .map_err(|e| anyhow!(e))
    }

    async fn create_node_profile(
        &self,
        ns: &str,
        obj: &BasilicaNodeProfile,
    ) -> Result<BasilicaNodeProfile> {
        use kube::api::PostParams;
        let api: kube::Api<BasilicaNodeProfile> = self.api(ns);
        match api.create(&PostParams::default(), obj).await {
            Ok(o) => Ok(o),
            Err(kube::Error::Api(ae)) if ae.code == 409 => Ok(obj.clone()),
            Err(e) => Err(anyhow!(e)),
        }
    }

    async fn get_node_profile(&self, ns: &str, name: &str) -> Result<BasilicaNodeProfile> {
        let api: kube::Api<BasilicaNodeProfile> = self.api(ns);
        api.get(name).await.map_err(|e| anyhow!(e))
    }

    async fn create_pod(&self, ns: &str, pod: &Pod) -> Result<Pod> {
        use kube::api::PostParams;
        let api: kube::Api<Pod> = self.api(ns);
        match api.create(&PostParams::default(), pod).await {
            Ok(o) => Ok(o),
            Err(kube::Error::Api(ae)) if ae.code == 409 => Ok(pod.clone()),
            Err(e) => Err(anyhow!(e)),
        }
    }

    async fn get_pod(&self, ns: &str, name: &str) -> Result<Pod> {
        let api: kube::Api<Pod> = self.api(ns);
        api.get(name).await.map_err(|e| anyhow!(e))
    }

    async fn delete_pod(&self, ns: &str, name: &str) -> Result<()> {
        use kube::api::DeleteParams;
        let api: kube::Api<Pod> = self.api(ns);
        api.delete(name, &DeleteParams::default())
            .await
            .map(|_| ())
            .map_err(|e| anyhow!(e))
    }

    async fn list_pods_with_label(&self, ns: &str, key: &str, value: &str) -> Result<Vec<Pod>> {
        use kube::api::ListParams;
        let api: kube::Api<Pod> = self.api(ns);
        let lp = ListParams::default().labels(&format!("{}={}", key, value));
        let list = api.list(&lp).await?;
        Ok(list.items)
    }

    async fn create_service(&self, ns: &str, svc: &Service) -> Result<Service> {
        use kube::api::PostParams;
        let api: kube::Api<Service> = self.api(ns);
        match api.create(&PostParams::default(), svc).await {
            Ok(o) => Ok(o),
            Err(kube::Error::Api(ae)) if ae.code == 409 => Ok(svc.clone()),
            Err(e) => Err(anyhow!(e)),
        }
    }

    async fn get_service(&self, ns: &str, name: &str) -> Result<Service> {
        let api: kube::Api<Service> = self.api(ns);
        api.get(name).await.map_err(|e| anyhow!(e))
    }

    async fn delete_service(&self, ns: &str, name: &str) -> Result<()> {
        use kube::api::DeleteParams;
        let api: kube::Api<Service> = self.api(ns);
        api.delete(name, &DeleteParams::default())
            .await
            .map(|_| ())
            .map_err(|e| anyhow!(e))
    }

    async fn list_services_with_label(
        &self,
        ns: &str,
        key: &str,
        value: &str,
    ) -> Result<Vec<Service>> {
        use kube::api::ListParams;
        let api: kube::Api<Service> = self.api(ns);
        let lp = ListParams::default().labels(&format!("{}={}", key, value));
        let list = api.list(&lp).await?;
        Ok(list.items)
    }

    async fn create_network_policy(&self, ns: &str, np: &NetworkPolicy) -> Result<NetworkPolicy> {
        use kube::api::PostParams;
        let api: kube::Api<NetworkPolicy> = self.api(ns);
        match api.create(&PostParams::default(), np).await {
            Ok(o) => Ok(o),
            Err(kube::Error::Api(ae)) if ae.code == 409 => Ok(np.clone()),
            Err(e) => Err(anyhow!(e)),
        }
    }

    async fn get_network_policy(&self, ns: &str, name: &str) -> Result<NetworkPolicy> {
        let api: kube::Api<NetworkPolicy> = self.api(ns);
        api.get(name).await.map_err(|e| anyhow!(e))
    }

    async fn create_pvc(
        &self,
        ns: &str,
        pvc: &PersistentVolumeClaim,
    ) -> Result<PersistentVolumeClaim> {
        use kube::api::PostParams;
        let api: kube::Api<PersistentVolumeClaim> = self.api(ns);
        match api.create(&PostParams::default(), pvc).await {
            Ok(o) => Ok(o),
            Err(kube::Error::Api(ae)) if ae.code == 409 => Ok(pvc.clone()),
            Err(e) => Err(anyhow!(e)),
        }
    }

    async fn create_secret(&self, ns: &str, secret: &Secret) -> Result<Secret> {
        use kube::api::PostParams;
        let api: kube::Api<Secret> = self.api(ns);
        match api.create(&PostParams::default(), secret).await {
            Ok(o) => Ok(o),
            Err(kube::Error::Api(ae)) if ae.code == 409 => Ok(secret.clone()),
            Err(e) => Err(anyhow!(e)),
        }
    }

    async fn create_job(&self, ns: &str, job: &Job) -> Result<Job> {
        use kube::api::PostParams;
        let api: kube::Api<Job> = self.api(ns);
        match api.create(&PostParams::default(), job).await {
            Ok(o) => Ok(o),
            Err(kube::Error::Api(ae)) if ae.code == 409 => Ok(job.clone()),
            Err(e) => Err(anyhow!(e)),
        }
    }

    async fn get_job(&self, ns: &str, name: &str) -> Result<Job> {
        let api: kube::Api<Job> = self.api(ns);
        api.get(name).await.map_err(|e| anyhow!(e))
    }

    async fn create_deployment(&self, ns: &str, dep: &Deployment) -> Result<Deployment> {
        use kube::api::PostParams;
        let api: kube::Api<Deployment> = self.api(ns);
        match api.create(&PostParams::default(), dep).await {
            Ok(o) => Ok(o),
            Err(kube::Error::Api(ae)) if ae.code == 409 => Ok(dep.clone()),
            Err(e) => Err(anyhow!(e)),
        }
    }

    async fn get_deployment(&self, ns: &str, name: &str) -> Result<Deployment> {
        let api: kube::Api<Deployment> = self.api(ns);
        api.get(name).await.map_err(|e| anyhow!(e))
    }

    async fn patch_deployment(&self, ns: &str, name: &str, dep: &Deployment) -> Result<Deployment> {
        use kube::api::{Patch, PatchParams};
        let api: kube::Api<Deployment> = self.api(ns);
        let dep_json = serde_json::to_value(dep)?;
        let patch_params = PatchParams::apply("basilica-operator");
        match api
            .patch(name, &patch_params, &Patch::Apply(&dep_json))
            .await
        {
            Ok(result) => Ok(result),
            Err(kube::Error::Api(ae)) if ae.code == 409 => {
                let force_params = patch_params.force();
                api.patch(name, &force_params, &Patch::Apply(&dep_json))
                    .await
                    .map_err(|e| anyhow!(e))
            }
            Err(e) => Err(anyhow!(e)),
        }
    }

    async fn patch_service(&self, ns: &str, name: &str, svc: &Service) -> Result<Service> {
        use kube::api::{Patch, PatchParams};
        let api: kube::Api<Service> = self.api(ns);
        let svc_json = serde_json::to_value(svc)?;
        let patch_params = PatchParams::apply("basilica-operator");
        match api
            .patch(name, &patch_params, &Patch::Apply(&svc_json))
            .await
        {
            Ok(result) => Ok(result),
            Err(kube::Error::Api(ae)) if ae.code == 409 => {
                let force_params = patch_params.force();
                api.patch(name, &force_params, &Patch::Apply(&svc_json))
                    .await
                    .map_err(|e| anyhow!(e))
            }
            Err(e) => Err(anyhow!(e)),
        }
    }

    async fn patch_network_policy(
        &self,
        ns: &str,
        name: &str,
        np: &NetworkPolicy,
    ) -> Result<NetworkPolicy> {
        use kube::api::{Patch, PatchParams};
        let api: kube::Api<NetworkPolicy> = self.api(ns);
        let np_json = serde_json::to_value(np)?;
        let patch_params = PatchParams::apply("basilica-operator");
        match api
            .patch(name, &patch_params, &Patch::Apply(&np_json))
            .await
        {
            Ok(result) => Ok(result),
            Err(kube::Error::Api(ae)) if ae.code == 409 => {
                let force_params = patch_params.force();
                api.patch(name, &force_params, &Patch::Apply(&np_json))
                    .await
                    .map_err(|e| anyhow!(e))
            }
            Err(e) => Err(anyhow!(e)),
        }
    }

    async fn create_basilica_queue(&self, ns: &str, obj: &BasilicaQueue) -> Result<BasilicaQueue> {
        use kube::api::PostParams;
        let api: kube::Api<BasilicaQueue> = self.api(ns);
        match api.create(&PostParams::default(), obj).await {
            Ok(o) => Ok(o),
            Err(kube::Error::Api(ae)) if ae.code == 409 => Ok(obj.clone()),
            Err(e) => Err(anyhow!(e)),
        }
    }

    async fn list_basilica_queues(&self, ns: &str) -> Result<Vec<BasilicaQueue>> {
        use kube::api::ListParams;
        let api: kube::Api<BasilicaQueue> = self.api(ns);
        let list = api.list(&ListParams::default()).await?;
        Ok(list.items)
    }

    async fn cordon_node(&self, name: &str) -> Result<()> {
        use kube::api::{Api, Patch, PatchParams};
        let api: Api<Node> = Api::all(self.client.clone());
        let patch = serde_json::json!({"spec": {"unschedulable": true}});
        api.patch(name, &PatchParams::default(), &Patch::Merge(&patch))
            .await
            .map(|_| ())
            .map_err(|e| anyhow!(e))
    }

    async fn create_http_route(
        &self,
        ns: &str,
        obj: &kube::core::DynamicObject,
    ) -> Result<kube::core::DynamicObject> {
        use kube::api::PostParams;
        let gvk = kube::core::GroupVersionKind::gvk("gateway.networking.k8s.io", "v1", "HTTPRoute");
        let ar = kube::core::ApiResource::from_gvk(&gvk);
        let api: kube::Api<kube::core::DynamicObject> =
            kube::Api::namespaced_with(self.client.clone(), ns, &ar);
        match api.create(&PostParams::default(), obj).await {
            Ok(o) => Ok(o),
            Err(kube::Error::Api(ae)) if ae.code == 409 => Ok(obj.clone()),
            Err(e) => Err(anyhow!(e)),
        }
    }

    async fn list_pods_on_node(&self, node_name: &str) -> Result<Vec<Pod>> {
        use kube::api::{Api, ListParams};
        let pods: Api<Pod> = Api::all(self.client.clone());
        let listed = pods.list(&ListParams::default()).await?;
        Ok(listed
            .items
            .into_iter()
            .filter(|p| {
                p.spec
                    .as_ref()
                    .and_then(|s| s.node_name.as_ref())
                    .map(|n| n == node_name)
                    .unwrap_or(false)
            })
            .collect())
    }

    async fn delete_node(&self, name: &str) -> Result<()> {
        use kube::api::{Api, DeleteParams};
        let api: Api<Node> = Api::all(self.client.clone());
        api.delete(name, &DeleteParams::default())
            .await
            .map(|_| ())
            .map_err(|e| anyhow!(e))
    }

    async fn remove_node_taint(&self, name: &str, taint_key: &str) -> Result<()> {
        use kube::api::{Api, Patch, PatchParams};
        let api: Api<Node> = Api::all(self.client.clone());

        let node = api.get(name).await.map_err(|e| anyhow!(e))?;

        let current_taints: Vec<serde_json::Value> = node
            .spec
            .and_then(|s| s.taints)
            .unwrap_or_default()
            .into_iter()
            .filter(|t| t.key != taint_key)
            .map(|t| {
                serde_json::json!({
                    "key": t.key,
                    "value": t.value,
                    "effect": t.effect,
                })
            })
            .collect();

        let patch = serde_json::json!({
            "spec": {
                "taints": current_taints
            }
        });

        api.patch(name, &PatchParams::default(), &Patch::Merge(&patch))
            .await
            .map(|_| ())
            .map_err(|e| anyhow!(e))
    }

    async fn evict_pod(&self, ns: &str, name: &str, grace_seconds: Option<i64>) -> Result<bool> {
        use k8s_openapi::apimachinery::pkg::apis::meta::v1::{DeleteOptions, ObjectMeta};
        use kube::api::{Api, PostParams};
        let pods: Api<Pod> = Api::namespaced(self.client.clone(), ns);
        let ev = k8s_openapi::api::policy::v1::Eviction {
            metadata: ObjectMeta {
                name: Some(name.to_string()),
                ..Default::default()
            },
            delete_options: Some(DeleteOptions {
                grace_period_seconds: grace_seconds,
                ..Default::default()
            }),
        };
        let body = match serde_json::to_vec(&ev) {
            Ok(v) => v,
            Err(e) => return Err(anyhow!(e)),
        };
        match pods
            .create_subresource::<k8s_openapi::apimachinery::pkg::apis::meta::v1::Status>(
                "eviction",
                name,
                &PostParams::default(),
                body,
            )
            .await
        {
            Ok(_) => Ok(true),
            Err(kube::Error::Api(ae)) if ae.code == 429 => Ok(false),
            Err(e) => Err(anyhow!(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use k8s_openapi::apimachinery::pkg::apis::meta::v1::ObjectMeta;

    fn pod_with_labels(name: &str, labels: &[(&str, &str)]) -> Pod {
        let mut meta = ObjectMeta {
            name: Some(name.to_string()),
            ..Default::default()
        };
        if !labels.is_empty() {
            meta.labels = Some(
                labels
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect(),
            );
        }
        Pod {
            metadata: meta,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn mock_client_crd_and_core_crud() {
        let client = MockK8sClient::default();

        // CRDs
        let bj = BasilicaJob::new(
            "job1",
            crate::crd::basilica_job::BasilicaJobSpec {
                image: "img".into(),
                command: vec![],
                args: vec![],
                env: vec![],
                resources: crate::crd::basilica_job::Resources {
                    cpu: "1".into(),
                    memory: "512Mi".into(),
                    gpus: crate::crd::basilica_job::GpuSpec {
                        count: 0,
                        model: vec![],
                    },
                },
                storage: None,
                artifacts: None,
                ttl_seconds: 0,
                priority: "normal".into(),
            },
        );
        client.create_basilica_job("ns", &bj).await.unwrap();
        let bj_get = client.get_basilica_job("ns", "job1").await.unwrap();
        assert_eq!(bj_get.name_any(), "job1");
        client.delete_basilica_job("ns", "job1").await.unwrap();
        assert!(client.get_basilica_job("ns", "job1").await.is_err());

        let gr = GpuRental::new(
            "rent1",
            crate::crd::gpu_rental::GpuRentalSpec {
                container: crate::crd::gpu_rental::RentalContainer {
                    image: "img".into(),
                    env: vec![],
                    command: vec![],
                    ports: vec![],
                    volumes: vec![],
                    resources: crate::crd::gpu_rental::Resources {
                        cpu: "1".into(),
                        memory: "1024Mi".into(),
                        gpus: crate::crd::gpu_rental::GpuSpec {
                            count: 1,
                            model: vec!["A100".into()],
                        },
                    },
                },
                duration: crate::crd::gpu_rental::RentalDuration {
                    hours: 24,
                    auto_extend: false,
                    max_extensions: 0,
                },
                access_type: crate::crd::gpu_rental::AccessType::Ssh,
                network: Default::default(),
                storage: None,
                artifacts: None,
                ssh: None,
                jupyter_access: None,
                environment: None,
                miner_selector: None,
                billing: None,
                ttl_seconds: 0,
                tenancy: None,
                exclusive: false,
            },
        );
        client.create_gpu_rental("ns", &gr).await.unwrap();
        let gr_get = client.get_gpu_rental("ns", "rent1").await.unwrap();
        assert_eq!(gr_get.name_any(), "rent1");
        client.delete_gpu_rental("ns", "rent1").await.unwrap();
        assert!(client.get_gpu_rental("ns", "rent1").await.is_err());

        // Core
        let p1 = pod_with_labels("p1", &[("a", "1"), ("b", "2")]);
        let p2 = pod_with_labels("p2", &[("a", "1"), ("b", "3")]);
        client.create_pod("ns", &p1).await.unwrap();
        client.create_pod("ns", &p2).await.unwrap();
        let got = client.get_pod("ns", "p1").await.unwrap();
        assert_eq!(got.name_any(), "p1");
        let list = client.list_pods_with_label("ns", "b", "2").await.unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].name_any(), "p1");
        client.delete_pod("ns", "p2").await.unwrap();
        assert!(client.get_pod("ns", "p2").await.is_err());

        // Queues
        let q = crate::crd::basilica_queue::BasilicaQueue::new(
            "q1",
            crate::crd::basilica_queue::BasilicaQueueSpec {
                concurrency: 1,
                gpu_limits: None,
            },
        );
        client.create_basilica_queue("ns", &q).await.unwrap();
        let qs = client.list_basilica_queues("ns").await.unwrap();
        assert_eq!(qs.len(), 1);
    }
}
