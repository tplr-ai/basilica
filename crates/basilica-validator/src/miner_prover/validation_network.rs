//! Network Geolocation Validation Module
//!
//! This module handles network geolocation and proximity information collection
//! for nodes by querying ipinfo.io to gather location and network metadata.

use crate::persistence::SimplePersistence;
use crate::ssh::ValidatorSshClient;
use anyhow::Result;
use basilica_common::ssh::SshConnectionDetails;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{info, warn};

/// Network profile information including geolocation and ISP details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkProfile {
    pub ip_address: Option<String>,
    pub hostname: Option<String>,
    pub city: Option<String>,
    pub region: Option<String>,
    pub country: Option<String>,
    pub location: Option<String>,
    pub organization: Option<String>,
    pub postal_code: Option<String>,
    pub timezone: Option<String>,
    pub test_timestamp: chrono::DateTime<chrono::Utc>,
    pub full_json: String,
}

impl NetworkProfile {
    /// Parse ipinfo.io JSON output
    pub fn from_ipinfo_json(json_str: &str) -> Result<Self> {
        let now = chrono::Utc::now();

        let json_value: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| anyhow::anyhow!("Failed to parse ipinfo JSON: {}", e))?;

        let ip_address = json_value
            .get("ip")
            .and_then(|v| v.as_str())
            .map(String::from);

        let hostname = json_value
            .get("hostname")
            .and_then(|v| v.as_str())
            .map(String::from);

        let city = json_value
            .get("city")
            .and_then(|v| v.as_str())
            .map(String::from);

        let region = json_value
            .get("region")
            .and_then(|v| v.as_str())
            .map(String::from);

        let country = json_value
            .get("country")
            .and_then(|v| v.as_str())
            .map(String::from);

        let location = json_value
            .get("loc")
            .and_then(|v| v.as_str())
            .map(String::from);

        let organization = json_value
            .get("org")
            .and_then(|v| v.as_str())
            .map(String::from);

        let postal_code = json_value
            .get("postal")
            .and_then(|v| v.as_str())
            .map(String::from);

        let timezone = json_value
            .get("timezone")
            .and_then(|v| v.as_str())
            .map(String::from);

        Ok(NetworkProfile {
            ip_address,
            hostname,
            city,
            region,
            country,
            location,
            organization,
            postal_code,
            timezone,
            test_timestamp: now,
            full_json: json_str.to_string(),
        })
    }
}

/// Network profile collector
#[derive(Clone)]
pub struct NetworkProfileCollector {
    ssh_client: Arc<ValidatorSshClient>,
    persistence: Arc<SimplePersistence>,
}

impl NetworkProfileCollector {
    pub fn new(ssh_client: Arc<ValidatorSshClient>, persistence: Arc<SimplePersistence>) -> Self {
        Self {
            ssh_client,
            persistence,
        }
    }

    pub async fn collect(
        &self,
        node_id: &str,
        ssh_details: &SshConnectionDetails,
    ) -> Result<NetworkProfile> {
        info!(
            node_id = node_id,
            "[NETWORK_PROFILE] Starting network geolocation collection"
        );

        self.ssh_client
            .ensure_installed(ssh_details, "curl", "curl")
            .await?;

        let ipinfo_output = self
            .ssh_client
            .execute_command(
                ssh_details,
                "timeout 10 curl -s 'https://ipinfo.io/json'",
                true,
            )
            .await?;

        let network_profile = NetworkProfile::from_ipinfo_json(&ipinfo_output)?;

        let location_info = format!(
            "{}, {}, {}",
            network_profile.city.as_deref().unwrap_or("Unknown"),
            network_profile.region.as_deref().unwrap_or("Unknown"),
            network_profile.country.as_deref().unwrap_or("Unknown")
        );

        info!(
            node_id = node_id,
            ip = network_profile.ip_address.as_deref().unwrap_or("Unknown"),
            location = location_info,
            organization = network_profile.organization.as_deref().unwrap_or("Unknown"),
            "[NETWORK_PROFILE] Successfully collected network profile"
        );

        Ok(network_profile)
    }

    /// Store network profile in database
    pub async fn store(
        &self,
        miner_uid: u16,
        node_id: &str,
        network_profile: &NetworkProfile,
    ) -> Result<()> {
        self.persistence
            .store_node_network_profile(
                miner_uid,
                node_id,
                network_profile.ip_address.clone(),
                network_profile.hostname.clone(),
                network_profile.city.clone(),
                network_profile.region.clone(),
                network_profile.country.clone(),
                network_profile.location.clone(),
                network_profile.organization.clone(),
                network_profile.postal_code.clone(),
                network_profile.timezone.clone(),
                &network_profile.test_timestamp.to_rfc3339(),
                &network_profile.full_json,
            )
            .await?;

        info!(
            miner_uid = miner_uid,
            node_id = node_id,
            "[NETWORK_PROFILE] Stored network profile in database"
        );

        Ok(())
    }

    /// Collect network profile from node and store in database
    pub async fn collect_and_store(
        &self,
        node_id: &str,
        miner_uid: u16,
        ssh_details: &SshConnectionDetails,
    ) -> Result<NetworkProfile> {
        let network_profile = self.collect(node_id, ssh_details).await?;
        self.store(miner_uid, node_id, &network_profile).await?;
        Ok(network_profile)
    }

    /// Collect network profile with error handling (non-critical operation)
    pub async fn collect_with_fallback(
        &self,
        node_id: &str,
        miner_uid: u16,
        ssh_details: &SshConnectionDetails,
    ) -> Option<NetworkProfile> {
        match self.collect(node_id, ssh_details).await {
            Ok(profile) => {
                if let Err(e) = self.store(miner_uid, node_id, &profile).await {
                    warn!(
                        miner_uid = miner_uid,
                        node_id = node_id,
                        error = %e,
                        "[NETWORK_PROFILE] Failed to store network profile (non-critical)"
                    );
                }
                Some(profile)
            }
            Err(e) => {
                warn!(
                    miner_uid = miner_uid,
                    node_id = node_id,
                    error = %e,
                    "[NETWORK_PROFILE] Failed to collect network profile (non-critical)"
                );
                None
            }
        }
    }

    /// Retrieve network profile from database
    pub async fn retrieve(&self, miner_uid: u16, node_id: &str) -> Result<Option<NetworkProfile>> {
        let result = self
            .persistence
            .get_node_network_profile(miner_uid, node_id)
            .await?;

        match result {
            Some((
                full_json,
                ip_address,
                hostname,
                city,
                region,
                country,
                location,
                organization,
                postal_code,
                timezone,
                test_timestamp,
            )) => {
                let test_timestamp = chrono::DateTime::parse_from_rfc3339(&test_timestamp)
                    .map_err(|e| anyhow::anyhow!("Failed to parse timestamp: {}", e))?
                    .with_timezone(&chrono::Utc);

                let profile = NetworkProfile {
                    ip_address,
                    hostname,
                    city,
                    region,
                    country,
                    location,
                    organization,
                    postal_code,
                    timezone,
                    test_timestamp,
                    full_json,
                };

                info!(
                    miner_uid = miner_uid,
                    node_id = node_id,
                    "[NETWORK_PROFILE] Retrieved network profile from database"
                );

                Ok(Some(profile))
            }
            None => {
                info!(
                    miner_uid = miner_uid,
                    node_id = node_id,
                    "[NETWORK_PROFILE] No network profile found in database"
                );
                Ok(None)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_ipinfo_json() {
        let test_json = r#"{
            "ip": "203.0.113.42",
            "hostname": "example-host.provider.net",
            "city": "Frankfurt",
            "region": "Hesse",
            "country": "DE",
            "loc": "50.1109,8.6821",
            "org": "AS12345 Example Provider GmbH",
            "postal": "60329",
            "timezone": "Europe/Berlin",
            "readme": "https://ipinfo.io/missingauth"
        }"#;

        let result = NetworkProfile::from_ipinfo_json(test_json);
        assert!(result.is_ok());

        let profile = result.unwrap();
        assert_eq!(profile.ip_address, Some("203.0.113.42".to_string()));
        assert_eq!(
            profile.hostname,
            Some("example-host.provider.net".to_string())
        );
        assert_eq!(profile.city, Some("Frankfurt".to_string()));
        assert_eq!(profile.region, Some("Hesse".to_string()));
        assert_eq!(profile.country, Some("DE".to_string()));
        assert_eq!(profile.location, Some("50.1109,8.6821".to_string()));
        assert_eq!(
            profile.organization,
            Some("AS12345 Example Provider GmbH".to_string())
        );
        assert_eq!(profile.postal_code, Some("60329".to_string()));
        assert_eq!(profile.timezone, Some("Europe/Berlin".to_string()));
    }

    #[test]
    fn test_parse_ipinfo_json_partial() {
        let test_json = r#"{
            "ip": "8.8.8.8",
            "city": "Mountain View",
            "country": "US"
        }"#;

        let result = NetworkProfile::from_ipinfo_json(test_json);
        assert!(result.is_ok());

        let profile = result.unwrap();
        assert_eq!(profile.ip_address, Some("8.8.8.8".to_string()));
        assert_eq!(profile.city, Some("Mountain View".to_string()));
        assert_eq!(profile.country, Some("US".to_string()));
        assert_eq!(profile.hostname, None);
        assert_eq!(profile.region, None);
        assert_eq!(profile.location, None);
    }

    #[test]
    fn test_parse_ipinfo_json_malformed() {
        let test_json = "not valid json";

        let result = NetworkProfile::from_ipinfo_json(test_json);
        assert!(result.is_err());
    }
}
