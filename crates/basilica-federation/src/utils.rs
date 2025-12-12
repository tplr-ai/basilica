//! Utility functions

use crate::error::{FederationError, Result};
use kube::Client;

/// Create Kubernetes client from kubeconfig
pub async fn create_kube_client(kubeconfig: &str) -> Result<Client> {
    // Try to load from path first
    if std::path::Path::new(kubeconfig).exists() {
        let config = kube::Config::from_kubeconfig(&kube::config::KubeconfigOptions {
            cluster: None,
            user: None,
            context: None,
        })
        .await?;
        return Ok(Client::try_from(config)?);
    }
    
    // Try to parse as kubeconfig content
    let kubeconfig_data: kube::config::Kubeconfig = serde_yaml::from_str(kubeconfig)
        .map_err(|e| FederationError::Config(format!("Invalid kubeconfig: {}", e)))?;
    
    let config = kube::Config::from_custom_kubeconfig(kubeconfig_data, &kube::config::KubeconfigOptions {
        cluster: None,
        user: None,
        context: None,
    })
    .await?;
    
    Ok(Client::try_from(config)?)
}

