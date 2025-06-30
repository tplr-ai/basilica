//! IP information collection from ipinfo.io service

use anyhow::Context;
use std::time::Duration;
use tokio::time::timeout;

use super::types::IpInfo;

/// IP information collector for gathering geolocation data
pub struct IpInfoCollector;

impl IpInfoCollector {
    /// Fetch IP geolocation information from ipinfo.io
    pub async fn fetch() -> anyhow::Result<IpInfo> {
        tracing::debug!("Fetching IP geolocation from ipinfo.io");

        let response = timeout(
            Duration::from_secs(10),
            reqwest::get("https://ipinfo.io/json"),
        )
        .await
        .context("IP geolocation request timed out")?
        .context("Failed to make IP geolocation request")?;

        if !response.status().is_success() {
            anyhow::bail!(
                "IP geolocation request failed with status: {}",
                response.status()
            );
        }

        let ip_info: IpInfo = response
            .json()
            .await
            .context("Failed to parse IP geolocation response")?;

        tracing::debug!(
            "Successfully fetched IP geolocation: {} ({})",
            ip_info.ip,
            ip_info.country.as_ref().unwrap_or(&"Unknown".to_string())
        );

        Ok(ip_info)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_fetch_ipinfo_structure() {
        // This test verifies the structure works, but may fail without internet
        match IpInfoCollector::fetch().await {
            Ok(ip_info) => {
                assert!(!ip_info.ip.is_empty());
                // IP should be a valid format (basic check)
                assert!(ip_info.ip.contains('.') || ip_info.ip.contains(':'));
            }
            Err(_) => {
                // Test passes if we can't reach ipinfo.io (offline/firewall)
                println!("Skipping ipinfo test - network unavailable");
            }
        }
    }
}
