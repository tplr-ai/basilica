use crate::error::{ApiError, Result};
use serde_json::Value;

pub fn parse_status_endpoints(val: &Value) -> Vec<String> {
    val.get("status")
        .and_then(|s| s.get("endpoints"))
        .and_then(|e| e.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_str().map(|s| s.to_string()))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

pub async fn client_from_kubeconfig_content(content: &str) -> anyhow::Result<kube::Client> {
    use kube::config::Kubeconfig;

    let kubeconfig: Kubeconfig = serde_yaml::from_str(content)
        .map_err(|e| anyhow::anyhow!("Failed to parse kubeconfig YAML: {}", e))?;

    let config = kube::Config::from_custom_kubeconfig(
        kubeconfig,
        &kube::config::KubeConfigOptions {
            context: None,
            cluster: None,
            user: None,
        },
    )
    .await?;

    kube::Client::try_from(config).map_err(Into::into)
}

pub async fn create_client() -> anyhow::Result<kube::Client> {
    if let Ok(kubeconfig_content) = std::env::var("KUBECONFIG_CONTENT") {
        tracing::info!("Loading K8s config from KUBECONFIG_CONTENT environment variable");
        return client_from_kubeconfig_content(&kubeconfig_content).await;
    }

    if let Ok(kubeconfig_path) = std::env::var("KUBECONFIG") {
        tracing::info!("Loading K8s config from KUBECONFIG: {}", kubeconfig_path);
        let config = kube::Config::from_kubeconfig(&kube::config::KubeConfigOptions {
            context: None,
            cluster: None,
            user: None,
        })
        .await?;
        return Ok(kube::Client::try_from(config)?);
    }

    tracing::info!("Attempting default K8s client initialization");
    kube::Client::try_default().await.map_err(Into::into)
}

pub async fn create_reference_grant_for_namespace(
    client: &kube::Client,
    user_namespace: &str,
) -> Result<()> {
    use kube::{
        api::{Api, PostParams},
        core::{ApiResource, DynamicObject, GroupVersionKind},
    };
    use serde_json::json;

    let gvk = GroupVersionKind::gvk("gateway.networking.k8s.io", "v1beta1", "ReferenceGrant");
    let ar = ApiResource::from_gvk(&gvk);
    let api: Api<DynamicObject> = Api::namespaced_with(client.clone(), "basilica-system", &ar);

    let reference_grant_name = format!("allow-httproutes-{}", user_namespace);
    let reference_grant = DynamicObject::new(&reference_grant_name, &ar).data(json!({
        "apiVersion": "gateway.networking.k8s.io/v1beta1",
        "kind": "ReferenceGrant",
        "metadata": {
            "name": reference_grant_name,
            "namespace": "basilica-system"
        },
        "spec": {
            "from": [{
                "group": "gateway.networking.k8s.io",
                "kind": "HTTPRoute",
                "namespace": user_namespace
            }],
            "to": [{
                "group": "gateway.networking.k8s.io",
                "kind": "Gateway",
                "name": "basilica-gateway"
            }]
        }
    }));

    match api.create(&PostParams::default(), &reference_grant).await {
        Ok(_) => {
            tracing::info!(
                user_namespace = %user_namespace,
                reference_grant_name = %reference_grant_name,
                "ReferenceGrant created for user namespace"
            );
            Ok(())
        }
        Err(kube::Error::Api(ae)) if ae.code == 409 => {
            tracing::debug!(
                user_namespace = %user_namespace,
                "ReferenceGrant already exists for namespace"
            );
            Ok(())
        }
        Err(e) => {
            tracing::error!(
                error = %e,
                user_namespace = %user_namespace,
                "Failed to create ReferenceGrant"
            );
            Err(ApiError::Internal {
                message: format!(
                    "Failed to create ReferenceGrant for namespace {}: {}",
                    user_namespace, e
                ),
            })
        }
    }
}

pub async fn copy_default_storage_secret(
    client: &kube::Client,
    user_namespace: &str,
) -> Result<()> {
    use k8s_openapi::api::core::v1::Secret;
    use kube::api::{Api, PostParams};

    let source_namespace = "basilica-system";
    let secret_name = "basilica-r2-credentials";

    let source_api: Api<Secret> = Api::namespaced(client.clone(), source_namespace);
    let dest_api: Api<Secret> = Api::namespaced(client.clone(), user_namespace);

    match source_api.get(secret_name).await {
        Ok(mut source_secret) => {
            source_secret.metadata.namespace = Some(user_namespace.to_string());
            source_secret.metadata.resource_version = None;
            source_secret.metadata.uid = None;
            source_secret.metadata.creation_timestamp = None;
            source_secret.metadata.owner_references = None;

            match dest_api
                .create(&PostParams::default(), &source_secret)
                .await
            {
                Ok(_) => {
                    tracing::info!(
                        user_namespace = %user_namespace,
                        secret_name = %secret_name,
                        "Default storage secret copied to user namespace"
                    );
                    Ok(())
                }
                Err(kube::Error::Api(ae)) if ae.code == 409 => {
                    tracing::debug!(
                        user_namespace = %user_namespace,
                        "Storage secret already exists in namespace"
                    );
                    Ok(())
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        user_namespace = %user_namespace,
                        "Failed to copy storage secret to user namespace"
                    );
                    Ok(())
                }
            }
        }
        Err(e) => {
            tracing::warn!(
                error = %e,
                source_namespace = %source_namespace,
                secret_name = %secret_name,
                "Default storage secret not found, skipping copy"
            );
            Ok(())
        }
    }
}

pub fn create_temp_kubeconfig(kubeconfig_content: &str) -> anyhow::Result<tempfile::NamedTempFile> {
    use std::io::Write;

    let mut temp_file = tempfile::NamedTempFile::new()?;
    temp_file.write_all(kubeconfig_content.as_bytes())?;
    temp_file.flush()?;
    Ok(temp_file)
}

pub async fn execute_k3s_command_with_kubeconfig(
    args: &[&str],
) -> anyhow::Result<std::process::Output> {
    let mut cmd = tokio::process::Command::new("k3s");
    cmd.args(args);

    let _temp_kubeconfig;
    let _temp_data_dir;

    if let Ok(kubeconfig_content) = std::env::var("KUBECONFIG_CONTENT") {
        _temp_kubeconfig = create_temp_kubeconfig(&kubeconfig_content)?;
        let kubeconfig_path = _temp_kubeconfig
            .path()
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Failed to convert kubeconfig path to string"))?;
        cmd.env("KUBECONFIG", kubeconfig_path);

        _temp_data_dir = tempfile::tempdir()?;
        cmd.arg("--data-dir").arg(_temp_data_dir.path());
    } else if let Ok(kubeconfig_path) = std::env::var("KUBECONFIG") {
        cmd.env("KUBECONFIG", kubeconfig_path);

        _temp_data_dir = tempfile::tempdir()?;
        cmd.arg("--data-dir").arg(_temp_data_dir.path());
    } else {
        return Err(anyhow::anyhow!(
            "Neither KUBECONFIG_CONTENT nor KUBECONFIG is set"
        ));
    }

    cmd.output().await.map_err(Into::into)
}

pub fn get_k3s_server_url() -> anyhow::Result<String> {
    std::env::var("K3S_SERVER_URL")
        .map_err(|_| anyhow::anyhow!("K3S_SERVER_URL environment variable not set"))
}

pub fn validate_node_id(node_id: &str) -> anyhow::Result<()> {
    if node_id.is_empty() {
        return Err(anyhow::anyhow!("node_id cannot be empty"));
    }

    if node_id.len() > 253 {
        return Err(anyhow::anyhow!("node_id cannot exceed 253 characters"));
    }

    if !node_id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '.')
    {
        return Err(anyhow::anyhow!(
            "node_id must be DNS-1123 compliant (alphanumeric, '-', or '.')"
        ));
    }

    if node_id.starts_with('-') || node_id.ends_with('-') {
        return Err(anyhow::anyhow!("node_id cannot start or end with '-'"));
    }

    Ok(())
}

pub struct NodeLabelParams<'a> {
    pub node_id: &'a str,
    pub datacenter_id: &'a str,
    pub gpu_model: &'a str,
    pub gpu_count: u32,
    pub gpu_memory_gb: u32,
    pub driver_version: &'a str,
    pub cuda_version: &'a str,
}

fn sanitize_label_value(value: &str) -> String {
    value
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '-'
            }
        })
        .collect()
}

pub fn build_node_labels(params: NodeLabelParams) -> std::collections::HashMap<String, String> {
    let mut labels = std::collections::HashMap::new();
    labels.insert("basilica.ai/node-type".to_string(), "gpu".to_string());
    labels.insert(
        "basilica.ai/datacenter".to_string(),
        sanitize_label_value(params.datacenter_id),
    );
    labels.insert(
        "basilica.ai/node-id".to_string(),
        params.node_id.to_string(),
    );
    labels.insert(
        "basilica.ai/gpu-model".to_string(),
        sanitize_label_value(params.gpu_model),
    );
    labels.insert(
        "basilica.ai/gpu-count".to_string(),
        params.gpu_count.to_string(),
    );
    labels.insert(
        "basilica.ai/gpu-memory-gb".to_string(),
        params.gpu_memory_gb.to_string(),
    );
    labels.insert(
        "basilica.ai/driver-version".to_string(),
        params.driver_version.to_string(),
    );
    labels.insert(
        "basilica.ai/cuda-version".to_string(),
        params.cuda_version.to_string(),
    );
    labels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_label_value_with_pipe() {
        assert_eq!(sanitize_label_value("github|434149"), "github-434149");
    }

    #[test]
    fn test_sanitize_label_value_valid_chars() {
        assert_eq!(sanitize_label_value("abc-123_xyz.456"), "abc-123_xyz.456");
    }

    #[test]
    fn test_sanitize_label_value_multiple_invalid() {
        assert_eq!(sanitize_label_value("test@value#123"), "test-value-123");
    }

    #[test]
    fn test_sanitize_label_value_spaces() {
        assert_eq!(sanitize_label_value("NVIDIA RTX A4000"), "NVIDIA-RTX-A4000");
    }
}
