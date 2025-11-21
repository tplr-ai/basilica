use async_trait::async_trait;
use k8s_openapi::api::core::v1::Namespace;
use k8s_openapi::apimachinery::pkg::apis::meta::v1::ObjectMeta;
use kube::api::{Api, DeleteParams, ListParams, LogParams, Patch, PatchParams, PostParams};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use super::helpers;
use super::r#trait::ApiK8sClient;
use super::types::*;
use crate::error::{ApiError, Result};

#[derive(Clone)]
pub struct K8sClient {
    client: kube::Client,
}

impl K8sClient {
    pub async fn try_default() -> Result<Self> {
        let client = helpers::create_client()
            .await
            .map_err(|e| ApiError::Internal {
                message: format!("k8s client init failed: {e}"),
            })?;
        Ok(Self { client })
    }

    pub fn client(&self) -> &kube::Client {
        &self.client
    }

    fn cr_api(
        &self,
        ns: &str,
        group: &str,
        version: &str,
        kind: &str,
    ) -> kube::Api<kube::core::DynamicObject> {
        use kube::core::{ApiResource, GroupVersionKind};
        let gvk = GroupVersionKind::gvk(group, version, kind);
        let ar = ApiResource::from_gvk(&gvk);
        kube::Api::namespaced_with(self.client.clone(), ns, &ar)
    }

    async fn get_pod_by_label(
        &self,
        ns: &str,
        key: &str,
        value: &str,
    ) -> Result<Option<k8s_openapi::api::core::v1::Pod>> {
        let pods: Api<k8s_openapi::api::core::v1::Pod> = Api::namespaced(self.client.clone(), ns);
        let lp = ListParams::default().labels(&format!("{}={}", key, value));
        let list = pods.list(&lp).await.map_err(|e| ApiError::Internal {
            message: format!("list pods failed: {e}"),
        })?;
        Ok(list.items.into_iter().next())
    }
}

#[async_trait]
impl ApiK8sClient for K8sClient {
    fn kube_client(&self) -> kube::Client {
        self.client.clone()
    }

    async fn create_job(&self, ns: &str, name: &str, spec: JobSpecDto) -> Result<String> {
        let api = self.cr_api(ns, "basilica.ai", "v1", "BasilicaJob");

        let ports_json: Vec<Value> = spec
            .ports
            .iter()
            .map(|p| json!({"containerPort": p.container_port, "protocol": p.protocol}))
            .collect();

        let mut spec_json = json!({
            "image": spec.image,
            "command": spec.command,
            "args": spec.args,
            "env": spec.env,
            "resources": {
                "cpu": spec.resources.cpu,
                "memory": spec.resources.memory,
                "gpus": {
                    "count": spec.resources.gpus.count,
                    "model": spec.resources.gpus.model
                }
            },
            "ttlSeconds": spec.ttl_seconds,
            "priority": "normal",
            "ports": ports_json,
        });

        if let Some(storage) = spec.storage {
            spec_json["storage"] = json!({
                "backend": storage.backend,
                "bucket": storage.bucket,
                "prefix": storage.prefix,
                "credentials": storage.credentials,
            });
        }

        let obj = json!({
            "apiVersion": "basilica.ai/v1",
            "kind": "BasilicaJob",
            "metadata": {"name": name, "namespace": ns},
            "spec": spec_json
        });

        let dynobj: kube::core::DynamicObject =
            serde_json::from_value(obj).map_err(|e| ApiError::Internal {
                message: format!("serde dynobj: {e}"),
            })?;

        api.create(&PostParams::default(), &dynobj)
            .await
            .map_err(|e| ApiError::Internal {
                message: format!("create BasilicaJob failed: {e}"),
            })?;

        Ok(name.to_string())
    }

    async fn get_job_status(&self, ns: &str, name: &str) -> Result<JobStatusDto> {
        let api = self.cr_api(ns, "basilica.ai", "v1", "BasilicaJob");
        let obj = api.get(name).await.map_err(|e| ApiError::NotFound {
            message: format!("job not found: {e}"),
        })?;

        let val: Value = serde_json::to_value(&obj).map_err(|e| ApiError::Internal {
            message: format!("to_value: {e}"),
        })?;

        let phase = val
            .get("status")
            .and_then(|s| s.get("phase"))
            .and_then(|v| v.as_str())
            .unwrap_or("Pending")
            .to_string();

        let pod_name = val
            .get("status")
            .and_then(|s| s.get("podName"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let endpoints = helpers::parse_status_endpoints(&val);

        Ok(JobStatusDto {
            phase,
            pod_name,
            endpoints,
        })
    }

    async fn delete_job(&self, ns: &str, name: &str) -> Result<()> {
        let api = self.cr_api(ns, "basilica.ai", "v1", "BasilicaJob");
        api.delete(name, &DeleteParams::default())
            .await
            .map_err(|e| ApiError::Internal {
                message: format!("delete job failed: {e}"),
            })?;
        Ok(())
    }

    async fn get_job_logs(&self, ns: &str, name: &str) -> Result<String> {
        if let Some(pod) = self.get_pod_by_label(ns, "basilica.ai/job", name).await? {
            let pods: Api<k8s_openapi::api::core::v1::Pod> =
                Api::namespaced(self.client.clone(), ns);
            let lp = LogParams {
                container: None,
                follow: false,
                ..Default::default()
            };
            let pod_name = pod.metadata.name.unwrap_or_default();
            let logs = pods
                .logs(&pod_name, &lp)
                .await
                .map_err(|e| ApiError::Internal {
                    message: format!("get logs failed: {e}"),
                })?;
            Ok(logs)
        } else {
            Err(ApiError::NotFound {
                message: "pod not found".into(),
            })
        }
    }

    async fn exec_job(
        &self,
        ns: &str,
        job_name: &str,
        command: Vec<String>,
        stdin: Option<String>,
        tty: bool,
    ) -> Result<ExecResultDto> {
        let pod = self
            .get_pod_by_label(ns, "basilica.ai/job", job_name)
            .await?
            .ok_or_else(|| ApiError::NotFound {
                message: "pod not found".into(),
            })?;

        let pod_name = pod.metadata.name.unwrap_or_default();
        let pods: Api<k8s_openapi::api::core::v1::Pod> = Api::namespaced(self.client.clone(), ns);

        let container_name = pod
            .spec
            .as_ref()
            .and_then(|spec| spec.containers.first().map(|c| c.name.clone()));

        let params = kube::api::AttachParams {
            stdout: true,
            stderr: true,
            stdin: stdin.is_some(),
            tty,
            container: container_name,
            ..Default::default()
        };

        let args: Vec<&str> = command.iter().map(|s| s.as_str()).collect();
        let mut attached =
            pods.exec(&pod_name, args, &params)
                .await
                .map_err(|e| ApiError::Internal {
                    message: format!("exec failed: {e}"),
                })?;

        if let (Some(input), Some(mut sin)) = (stdin, attached.stdin()) {
            let _ = sin.write_all(input.as_bytes()).await;
            let _ = sin.shutdown().await;
        }

        let mut stdout_buf = Vec::new();
        if let Some(mut out) = attached.stdout() {
            out.read_to_end(&mut stdout_buf)
                .await
                .map_err(|e| ApiError::Internal {
                    message: format!("read stdout failed: {e}"),
                })?;
        }

        let mut stderr_buf = Vec::new();
        if let Some(mut err) = attached.stderr() {
            err.read_to_end(&mut stderr_buf)
                .await
                .map_err(|e| ApiError::Internal {
                    message: format!("read stderr failed: {e}"),
                })?;
        }

        let joined = attached.join().await;
        let exit_code = if joined.is_ok() { 0 } else { 1 };

        Ok(ExecResultDto {
            stdout: String::from_utf8_lossy(&stdout_buf).into_owned(),
            stderr: String::from_utf8_lossy(&stderr_buf).into_owned(),
            exit_code,
        })
    }

    async fn suspend_job(&self, ns: &str, name: &str) -> Result<()> {
        let api = self.cr_api(ns, "basilica.ai", "v1", "BasilicaJob");
        let patch = json!({
            "spec": {
                "suspended": true
            }
        });

        api.patch(name, &PatchParams::default(), &Patch::Merge(&patch))
            .await
            .map_err(|e| ApiError::Internal {
                message: format!("suspend job failed: {e}"),
            })?;
        Ok(())
    }

    async fn resume_job(&self, ns: &str, name: &str) -> Result<()> {
        let api = self.cr_api(ns, "basilica.ai", "v1", "BasilicaJob");
        let patch = json!({
            "spec": {
                "suspended": false
            }
        });

        api.patch(name, &PatchParams::default(), &Patch::Merge(&patch))
            .await
            .map_err(|e| ApiError::Internal {
                message: format!("resume job failed: {e}"),
            })?;
        Ok(())
    }

    async fn create_rental(&self, ns: &str, name: &str, spec: RentalSpecDto) -> Result<String> {
        let api = self.cr_api(ns, "basilica.ai", "v1", "GpuRental");

        let env_objs: Vec<Value> = spec
            .container_env
            .iter()
            .map(|(k, v)| json!({"name": k, "value": v}))
            .collect();

        let ports_json: Vec<Value> = spec
            .container_ports
            .iter()
            .map(|p| json!({"containerPort": p.container_port, "protocol": p.protocol}))
            .collect();

        let obj = json!({
            "apiVersion": "basilica.ai/v1",
            "kind": "GpuRental",
            "metadata": {
                "name": name,
                "namespace": ns,
                "labels": spec.labels.clone().unwrap_or_default(),
                "annotations": spec.annotations.clone().unwrap_or_default(),
            },
            "spec": {
                "container": {
                    "image": spec.container_image,
                    "env": env_objs,
                    "command": spec.container_command,
                    "ports": ports_json,
                    "volumes": [],
                    "resources": {
                        "cpu": spec.resources.cpu,
                        "memory": spec.resources.memory,
                        "gpus": {
                            "count": spec.resources.gpus.count,
                            "model": spec.resources.gpus.model
                        }
                    },
                },
                "duration": {"hours": 0, "autoExtend": false, "maxExtensions": 0},
                "accessType": "ssh",
                "network": {
                    "ingress": spec.network_ingress,
                    "egressPolicy": "restricted",
                    "allowedEgress": [],
                    "publicIpRequired": false
                },
                "ssh": spec.ssh,
                "ttlSeconds": 0
            }
        });

        let dynobj: kube::core::DynamicObject =
            serde_json::from_value(obj).map_err(|e| ApiError::Internal {
                message: format!("serde dynobj: {e}"),
            })?;

        api.create(&PostParams::default(), &dynobj)
            .await
            .map_err(|e| ApiError::Internal {
                message: format!("create GpuRental failed: {e}"),
            })?;

        Ok(name.to_string())
    }

    async fn get_rental_status(&self, ns: &str, name: &str) -> Result<RentalStatusDto> {
        let api = self.cr_api(ns, "basilica.ai", "v1", "GpuRental");
        let obj = api.get(name).await.map_err(|e| ApiError::NotFound {
            message: format!("rental not found: {e}"),
        })?;

        let val: Value = serde_json::to_value(&obj).map_err(|e| ApiError::Internal {
            message: format!("to_value: {e}"),
        })?;

        let state = val
            .get("status")
            .and_then(|s| s.get("state"))
            .and_then(|v| v.as_str())
            .unwrap_or("Provisioning")
            .to_string();

        let pod_name = val
            .get("status")
            .and_then(|s| s.get("podName"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let endpoints = helpers::parse_status_endpoints(&val);

        Ok(RentalStatusDto {
            state,
            pod_name,
            endpoints,
        })
    }

    async fn delete_rental(&self, ns: &str, name: &str) -> Result<()> {
        let api = self.cr_api(ns, "basilica.ai", "v1", "GpuRental");
        api.delete(name, &DeleteParams::default())
            .await
            .map_err(|e| ApiError::Internal {
                message: format!("delete rental failed: {e}"),
            })?;
        Ok(())
    }

    async fn get_rental_logs(
        &self,
        ns: &str,
        name: &str,
        tail: Option<u32>,
        since_seconds: Option<u32>,
    ) -> Result<String> {
        if let Some(pod) = self
            .get_pod_by_label(ns, "basilica.ai/rental", name)
            .await?
        {
            let pods: Api<k8s_openapi::api::core::v1::Pod> =
                Api::namespaced(self.client.clone(), ns);
            let lp = LogParams {
                container: None,
                follow: false,
                tail_lines: tail.map(|x| x as i64),
                since_seconds: since_seconds.map(|x| x as i64),
                ..Default::default()
            };
            let pod_name = pod.metadata.name.unwrap_or_default();
            let logs = pods
                .logs(&pod_name, &lp)
                .await
                .map_err(|e| ApiError::Internal {
                    message: format!("get logs failed: {e}"),
                })?;
            Ok(logs)
        } else {
            Err(ApiError::NotFound {
                message: "pod not found".into(),
            })
        }
    }

    async fn exec_rental(
        &self,
        ns: &str,
        name: &str,
        command: Vec<String>,
        stdin: Option<String>,
        tty: bool,
    ) -> Result<ExecResultDto> {
        let pod = self
            .get_pod_by_label(ns, "basilica.ai/rental", name)
            .await?
            .ok_or_else(|| ApiError::NotFound {
                message: "pod not found".into(),
            })?;

        let pod_name = pod.metadata.name.clone().unwrap_or_default();
        let pods: Api<k8s_openapi::api::core::v1::Pod> = Api::namespaced(self.client.clone(), ns);

        let container_name = pod
            .spec
            .as_ref()
            .and_then(|spec| spec.containers.first().map(|c| c.name.clone()));

        let params = kube::api::AttachParams {
            stdout: true,
            stderr: true,
            stdin: stdin.is_some(),
            tty,
            container: container_name,
            ..Default::default()
        };

        let args: Vec<&str> = command.iter().map(|s| s.as_str()).collect();
        let mut attached =
            pods.exec(&pod_name, args, &params)
                .await
                .map_err(|e| ApiError::Internal {
                    message: format!("exec failed: {e}"),
                })?;

        if let (Some(input), Some(mut sin)) = (stdin, attached.stdin()) {
            let _ = sin.write_all(input.as_bytes()).await;
            let _ = sin.shutdown().await;
        }

        let mut stdout_buf = Vec::new();
        if let Some(mut out) = attached.stdout() {
            out.read_to_end(&mut stdout_buf)
                .await
                .map_err(|e| ApiError::Internal {
                    message: format!("read stdout failed: {e}"),
                })?;
        }

        let mut stderr_buf = Vec::new();
        if let Some(mut err) = attached.stderr() {
            err.read_to_end(&mut stderr_buf)
                .await
                .map_err(|e| ApiError::Internal {
                    message: format!("read stderr failed: {e}"),
                })?;
        }

        let joined = attached.join().await;
        let mut exit_code = if joined.is_ok() { 0 } else { 1 };

        if exit_code != 0 {
            if let Ok(p) = pods.get(&pod_name).await {
                if let Some(status) = &p.status {
                    if let Some(cstatuses) = &status.container_statuses {
                        let prefer = params.container.clone();
                        let chosen = if let Some(pref_name) = prefer {
                            cstatuses
                                .iter()
                                .find(|cs| cs.name == pref_name)
                                .or_else(|| cstatuses.first())
                        } else {
                            cstatuses.first()
                        };

                        if let Some(cs) = chosen {
                            if let Some(state) = &cs.state {
                                if let Some(term) = &state.terminated {
                                    exit_code = term.exit_code;
                                }
                            }
                            if exit_code != 0 {
                                if let Some(last) = &cs.last_state {
                                    if let Some(term) = &last.terminated {
                                        exit_code = term.exit_code;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(ExecResultDto {
            stdout: String::from_utf8_lossy(&stdout_buf).into_owned(),
            stderr: String::from_utf8_lossy(&stderr_buf).into_owned(),
            exit_code,
        })
    }

    async fn extend_rental(
        &self,
        ns: &str,
        name: &str,
        _additional_hours: u32,
    ) -> Result<RentalStatusDto> {
        self.get_rental_status(ns, name).await
    }

    async fn list_rentals(&self, ns: &str) -> Result<Vec<RentalListItemDto>> {
        let api = self.cr_api(ns, "basilica.ai", "v1", "GpuRental");
        let list = api
            .list(&ListParams::default())
            .await
            .map_err(|e| ApiError::Internal {
                message: format!("list rentals failed: {e}"),
            })?;

        let mut out = Vec::new();
        for item in list.items {
            let name = item.metadata.name.clone().unwrap_or_default();
            let val: Value = serde_json::to_value(&item).map_err(|e| ApiError::Internal {
                message: format!("to_value: {e}"),
            })?;

            let state = val
                .get("status")
                .and_then(|s| s.get("state"))
                .and_then(|v| v.as_str())
                .unwrap_or("Provisioning")
                .to_string();

            let pod_name = val
                .get("status")
                .and_then(|s| s.get("podName"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());

            let endpoints = helpers::parse_status_endpoints(&val);

            out.push(RentalListItemDto {
                rental_id: name,
                status: RentalStatusDto {
                    state,
                    pod_name,
                    endpoints,
                },
            });
        }

        Ok(out)
    }

    async fn create_namespace(&self, name: &str) -> Result<()> {
        let api: Api<Namespace> = Api::all(self.client.clone());
        let ns = Namespace {
            metadata: ObjectMeta {
                name: Some(name.to_string()),
                ..Default::default()
            },
            ..Default::default()
        };

        match api.create(&PostParams::default(), &ns).await {
            Ok(_) => {
                if name.starts_with("u-") {
                    helpers::create_reference_grant_for_namespace(&self.client, name).await?;
                    helpers::copy_default_storage_secret(&self.client, name).await?;
                }
                Ok(())
            }
            Err(kube::Error::Api(ae)) if ae.code == 409 => {
                if name.starts_with("u-") {
                    helpers::copy_default_storage_secret(&self.client, name).await?;
                }
                Ok(())
            }
            Err(e) => Err(ApiError::Internal {
                message: format!("create namespace failed: {e}"),
            }),
        }
    }

    async fn get_namespace(&self, name: &str) -> Result<()> {
        let api: Api<Namespace> = Api::all(self.client.clone());
        api.get(name).await.map_err(|e| ApiError::Internal {
            message: format!("get namespace failed: {e}"),
        })?;
        Ok(())
    }

    async fn get_configmap(
        &self,
        ns: &str,
        name: &str,
    ) -> Result<std::collections::BTreeMap<String, String>> {
        let api: Api<k8s_openapi::api::core::v1::ConfigMap> =
            Api::namespaced(self.client.clone(), ns);
        let cm = api.get(name).await.map_err(|e| ApiError::Internal {
            message: format!("get configmap failed: {e}"),
        })?;
        Ok(cm.data.unwrap_or_default())
    }

    async fn patch_configmap(
        &self,
        ns: &str,
        name: &str,
        data: std::collections::BTreeMap<String, String>,
    ) -> Result<()> {
        let api: Api<k8s_openapi::api::core::v1::ConfigMap> =
            Api::namespaced(self.client.clone(), ns);
        let patch = json!({"data": data});
        api.patch(name, &PatchParams::default(), &Patch::Merge(&patch))
            .await
            .map_err(|e| ApiError::Internal {
                message: format!("patch configmap failed: {e}"),
            })?;
        Ok(())
    }

    async fn restart_deployment(&self, ns: &str, name: &str) -> Result<()> {
        let api: Api<k8s_openapi::api::apps::v1::Deployment> =
            Api::namespaced(self.client.clone(), ns);
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            .to_string();

        let patch = json!({
            "spec": {
                "template": {
                    "metadata": {
                        "annotations": {
                            "kubectl.kubernetes.io/restartedAt": timestamp
                        }
                    }
                }
            }
        });

        api.patch(name, &PatchParams::default(), &Patch::Strategic(&patch))
            .await
            .map_err(|e| ApiError::Internal {
                message: format!("restart deployment failed: {e}"),
            })?;
        Ok(())
    }

    async fn list_pods(&self, ns: &str, label_selector: &str) -> Result<Vec<String>> {
        let api: Api<k8s_openapi::api::core::v1::Pod> = Api::namespaced(self.client.clone(), ns);
        let lp = ListParams::default().labels(label_selector);

        let pods = api.list(&lp).await.map_err(|e| ApiError::Internal {
            message: format!("list pods failed: {e}"),
        })?;

        let pod_names = pods
            .items
            .iter()
            .filter_map(|pod| pod.metadata.name.clone())
            .collect();

        Ok(pod_names)
    }

    async fn create_user_deployment(
        &self,
        ns: &str,
        name: &str,
        user_id: &str,
        instance_name: &str,
        req: &crate::api::routes::deployments::types::CreateDeploymentRequest,
        path_prefix: &str,
    ) -> Result<()> {
        let api = self.cr_api(ns, "basilica.ai", "v1", "UserDeployment");

        if self.user_deployment_exists(ns, name).await? {
            tracing::debug!(
                namespace = ns,
                name = name,
                "UserDeployment already exists, skipping creation"
            );
            return Ok(());
        }

        let env_objs: Vec<Value> = req
            .env
            .iter()
            .map(|(k, v)| json!({"name": k, "value": v}))
            .collect();

        let mut spec = json!({
            "userId": user_id,
            "instanceName": instance_name,
            "image": req.image,
            "replicas": req.replicas,
            "port": req.port,
            "command": req.command,
            "args": req.args,
            "env": env_objs,
            "resources": {
                "cpu": req.resources.as_ref().map(|r| r.cpu.clone()).unwrap_or_else(|| "500m".to_string()),
                "memory": req.resources.as_ref().map(|r| r.memory.clone()).unwrap_or_else(|| "512Mi".to_string()),
            },
            "pathPrefix": path_prefix,
            "ttlSeconds": req.ttl_seconds,
        });

        if let Some(ref health_check) = req.health_check {
            let mut health_check_obj = json!({});

            if let Some(ref liveness) = health_check.liveness {
                health_check_obj["liveness"] = json!({
                    "path": liveness.path,
                    "initialDelaySeconds": liveness.initial_delay_seconds,
                    "periodSeconds": liveness.period_seconds,
                    "timeoutSeconds": liveness.timeout_seconds,
                    "failureThreshold": liveness.failure_threshold,
                });
            }

            if let Some(ref readiness) = health_check.readiness {
                health_check_obj["readiness"] = json!({
                    "path": readiness.path,
                    "initialDelaySeconds": readiness.initial_delay_seconds,
                    "periodSeconds": readiness.period_seconds,
                    "timeoutSeconds": readiness.timeout_seconds,
                    "failureThreshold": readiness.failure_threshold,
                });
            }

            spec["healthCheck"] = health_check_obj;
        }

        if let Some(ref storage) = req.storage {
            if let Some(ref persistent) = storage.persistent {
                let backend_str = match persistent.backend {
                    crate::api::routes::deployments::types::StorageBackend::R2 => "r2",
                    crate::api::routes::deployments::types::StorageBackend::S3 => "s3",
                    crate::api::routes::deployments::types::StorageBackend::GCS => "gcs",
                };
                let mut persistent_obj = json!({
                    "enabled": persistent.enabled,
                    "backend": backend_str,
                    "bucket": persistent.bucket,
                    "syncIntervalMs": persistent.sync_interval_ms,
                    "cacheSizeMb": persistent.cache_size_mb,
                    "mountPath": persistent.mount_path,
                });
                if let Some(ref region) = persistent.region {
                    persistent_obj["region"] = json!(region);
                }
                if let Some(ref endpoint) = persistent.endpoint {
                    persistent_obj["endpoint"] = json!(endpoint);
                }
                if let Some(ref credentials_secret) = persistent.credentials_secret {
                    persistent_obj["credentialsSecret"] = json!(credentials_secret);
                }
                spec["storage"] = json!({
                    "persistent": persistent_obj
                });
            }
        }

        let obj = json!({
            "apiVersion": "basilica.ai/v1",
            "kind": "UserDeployment",
            "metadata": {
                "name": name,
                "namespace": ns,
            },
            "spec": spec,
        });

        let dynobj: kube::core::DynamicObject =
            serde_json::from_value(obj).map_err(|e| ApiError::Internal {
                message: format!("serde dynobj: {e}"),
            })?;

        api.create(&PostParams::default(), &dynobj)
            .await
            .map_err(|e| ApiError::Internal {
                message: format!("create UserDeployment failed: {e}"),
            })?;

        Ok(())
    }

    async fn delete_user_deployment(&self, ns: &str, name: &str) -> Result<()> {
        let api = self.cr_api(ns, "basilica.ai", "v1", "UserDeployment");

        match api.delete(name, &DeleteParams::default()).await {
            Ok(_) => {
                tracing::info!(
                    namespace = ns,
                    name = name,
                    "Successfully deleted UserDeployment CR"
                );
                Ok(())
            }
            Err(kube::Error::Api(err)) if err.code == 404 => {
                tracing::debug!(
                    namespace = ns,
                    name = name,
                    "UserDeployment CR already gone"
                );
                Ok(())
            }
            Err(e) => Err(ApiError::Internal {
                message: format!("delete UserDeployment failed: {e}"),
            }),
        }
    }

    async fn delete_deployment(&self, ns: &str, name: &str) -> Result<()> {
        let api: Api<k8s_openapi::api::apps::v1::Deployment> =
            Api::namespaced(self.client.clone(), ns);

        match api.delete(name, &DeleteParams::default()).await {
            Ok(_) => {
                tracing::info!(
                    namespace = ns,
                    name = name,
                    "Successfully deleted Deployment"
                );
                Ok(())
            }
            Err(kube::Error::Api(err)) if err.code == 404 => {
                tracing::debug!(
                    namespace = ns,
                    name = name,
                    "Deployment not found, already deleted"
                );
                Ok(())
            }
            Err(e) => Err(ApiError::Internal {
                message: format!("delete Deployment failed: {e}"),
            }),
        }
    }

    async fn delete_service(&self, ns: &str, name: &str) -> Result<()> {
        let api: Api<k8s_openapi::api::core::v1::Service> =
            Api::namespaced(self.client.clone(), ns);

        match api.delete(name, &DeleteParams::default()).await {
            Ok(_) => {
                tracing::info!(namespace = ns, name = name, "Successfully deleted Service");
                Ok(())
            }
            Err(kube::Error::Api(err)) if err.code == 404 => {
                tracing::debug!(
                    namespace = ns,
                    name = name,
                    "Service not found, already deleted"
                );
                Ok(())
            }
            Err(e) => Err(ApiError::Internal {
                message: format!("delete Service failed: {e}"),
            }),
        }
    }

    async fn delete_network_policy(&self, ns: &str, name: &str) -> Result<()> {
        let api: Api<k8s_openapi::api::networking::v1::NetworkPolicy> =
            Api::namespaced(self.client.clone(), ns);

        match api.delete(name, &DeleteParams::default()).await {
            Ok(_) => {
                tracing::info!(
                    namespace = ns,
                    name = name,
                    "Successfully deleted NetworkPolicy"
                );
                Ok(())
            }
            Err(kube::Error::Api(err)) if err.code == 404 => {
                tracing::debug!(
                    namespace = ns,
                    name = name,
                    "NetworkPolicy not found, already deleted"
                );
                Ok(())
            }
            Err(e) => Err(ApiError::Internal {
                message: format!("delete NetworkPolicy failed: {e}"),
            }),
        }
    }

    async fn user_deployment_exists(&self, ns: &str, name: &str) -> Result<bool> {
        let api = self.cr_api(ns, "basilica.ai", "v1", "UserDeployment");

        match api.get(name).await {
            Ok(_) => Ok(true),
            Err(kube::Error::Api(err_resp)) if err_resp.code == 404 => Ok(false),
            Err(e) => Err(ApiError::Internal {
                message: format!("check UserDeployment existence failed: {e}"),
            }),
        }
    }

    async fn get_user_deployment_status(&self, ns: &str, name: &str) -> Result<(u32, u32)> {
        let api = self.cr_api(ns, "basilica.ai", "v1", "UserDeployment");

        match api.get(name).await {
            Ok(obj) => {
                tracing::debug!(
                    namespace = ns,
                    name = name,
                    "Retrieved UserDeployment CR, checking status"
                );

                let status = obj.data.get("status");
                if status.is_none() {
                    tracing::warn!(
                        namespace = ns,
                        name = name,
                        "UserDeployment CR has no status field"
                    );
                    return Ok((0, 0));
                }

                let status = status.unwrap();

                let desired = status
                    .get("replicasDesired")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as u32;

                let ready = status
                    .get("replicasReady")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as u32;

                tracing::debug!(
                    namespace = ns,
                    name = name,
                    desired = desired,
                    ready = ready,
                    "Extracted replica counts from UserDeployment status"
                );

                Ok((desired, ready))
            }
            Err(kube::Error::Api(err_resp)) if err_resp.code == 404 => {
                tracing::warn!(
                    namespace = ns,
                    name = name,
                    "UserDeployment CR not found (404)"
                );
                Ok((0, 0))
            }
            Err(e) => {
                tracing::error!(
                    namespace = ns,
                    name = name,
                    error = %e,
                    "Failed to get UserDeployment status"
                );
                Err(ApiError::Internal {
                    message: format!("get UserDeployment status failed: {e}"),
                })
            }
        }
    }

    async fn get_user_deployment_logs(
        &self,
        ns: &str,
        name: &str,
        tail: Option<u32>,
        since_seconds: Option<u32>,
    ) -> Result<String> {
        let api: Api<k8s_openapi::api::core::v1::Pod> = Api::namespaced(self.client.clone(), ns);

        let label_selector = format!("basilica.ai/instance={}", name);
        let pods = api
            .list(&ListParams::default().labels(&label_selector))
            .await
            .map_err(|e| {
                tracing::error!(
                    error = %e,
                    namespace = ns,
                    instance = name,
                    "Failed to list pods for deployment"
                );
                ApiError::Internal {
                    message: format!("Failed to list pods: {}", e),
                }
            })?;

        if pods.items.is_empty() {
            return Ok(String::new());
        }

        let pod = &pods.items[0];
        let pod_name = pod
            .metadata
            .name
            .as_ref()
            .ok_or_else(|| ApiError::Internal {
                message: "Pod has no name".to_string(),
            })?;

        let mut log_params = LogParams::default();
        if let Some(t) = tail {
            log_params.tail_lines = Some(t as i64);
        }
        if let Some(s) = since_seconds {
            log_params.since_seconds = Some(s as i64);
        }

        let logs = api.logs(pod_name, &log_params).await.map_err(|e| {
            tracing::error!(error = %e, namespace = ns, pod = pod_name, "Failed to get pod logs");
            ApiError::Internal {
                message: format!("Failed to get logs: {}", e),
            }
        })?;

        Ok(logs)
    }
}
