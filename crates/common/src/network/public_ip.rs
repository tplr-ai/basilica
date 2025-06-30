use anyhow::Result;
use serde::Deserialize;

#[derive(Deserialize)]
struct IpInfo {
    ip: String,
}

/// Get the public IP address of this machine
pub async fn get_public_ip() -> String {
    match get_public_ip_result().await {
        Ok(ip) => ip,
        Err(_) => "<unknown-ip>".to_string(),
    }
}

/// Get the public IP address with proper error handling
pub async fn get_public_ip_result() -> Result<String> {
    let response = reqwest::get("https://ipinfo.io/json").await?;
    let ip_info: IpInfo = response.json().await?;
    Ok(ip_info.ip)
}

/// Get public IP with a custom timeout
pub async fn get_public_ip_with_timeout(timeout_secs: u64) -> String {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(timeout_secs))
        .build();

    match client {
        Ok(client) => match client.get("https://ipinfo.io/json").send().await {
            Ok(response) => match response.json::<IpInfo>().await {
                Ok(ip_info) => ip_info.ip,
                Err(_) => "<unknown-ip>".to_string(),
            },
            Err(_) => "<unknown-ip>".to_string(),
        },
        Err(_) => "<unknown-ip>".to_string(),
    }
}
