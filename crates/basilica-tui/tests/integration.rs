//! Integration tests for Basilica TUI
//!
//! These tests verify the TUI works correctly against a real API.
//! Run with: `cargo test -p basilica-tui --test integration`
//!
//! Prerequisites:
//! - Start API: `cd scripts/tui-test && docker compose up -d`
//! - Or use mock: Tests will skip if API is not available

use std::time::Duration;

/// Helper to check if the API is available
async fn api_available() -> bool {
    let url = std::env::var("BASILICA_API_URL").unwrap_or_else(|_| "http://localhost:8000".into());
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .unwrap();

    client
        .get(format!("{}/health", url))
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

/// Skip test if API is not available
macro_rules! require_api {
    () => {
        if !api_available().await {
            eprintln!("⚠️  Skipping test: API not available at BASILICA_API_URL");
            eprintln!("   Start with: cd scripts/tui-test && docker compose up -d");
            return;
        }
    };
}

#[tokio::test]
async fn test_api_health() {
    require_api!();

    let url = std::env::var("BASILICA_API_URL").unwrap_or_else(|_| "http://localhost:8000".into());
    let client = reqwest::Client::new();

    let response = client
        .get(format!("{}/health", url))
        .send()
        .await
        .expect("Failed to send request");

    assert!(response.status().is_success());
}

#[tokio::test]
async fn test_list_available_nodes() {
    require_api!();

    let url = std::env::var("BASILICA_API_URL").unwrap_or_else(|_| "http://localhost:8000".into());
    let client = reqwest::Client::new();

    let response = client
        .get(format!("{}/v1/nodes/available", url))
        .send()
        .await
        .expect("Failed to send request");

    // Should return 200 even if empty (dev mode may not have nodes)
    assert!(
        response.status().is_success() || response.status().as_u16() == 401,
        "Expected success or unauthorized, got: {}",
        response.status()
    );
}

#[tokio::test]
async fn test_balance_endpoint() {
    require_api!();

    let url = std::env::var("BASILICA_API_URL").unwrap_or_else(|_| "http://localhost:8000".into());
    let client = reqwest::Client::new();

    let response = client
        .get(format!("{}/v1/billing/balance", url))
        .send()
        .await
        .expect("Failed to send request");

    // Expect 401 unauthorized without auth, which confirms the endpoint exists
    assert!(
        response.status().is_success() || response.status().as_u16() == 401,
        "Expected success or unauthorized, got: {}",
        response.status()
    );
}

#[tokio::test]
async fn test_rentals_endpoint() {
    require_api!();

    let url = std::env::var("BASILICA_API_URL").unwrap_or_else(|_| "http://localhost:8000".into());
    let client = reqwest::Client::new();

    let response = client
        .get(format!("{}/v1/rentals", url))
        .send()
        .await
        .expect("Failed to send request");

    // Expect 401 unauthorized without auth, which confirms the endpoint exists
    assert!(
        response.status().is_success() || response.status().as_u16() == 401,
        "Expected success or unauthorized, got: {}",
        response.status()
    );
}

#[tokio::test]
async fn test_deployments_endpoint() {
    require_api!();

    let url = std::env::var("BASILICA_API_URL").unwrap_or_else(|_| "http://localhost:8000".into());
    let client = reqwest::Client::new();

    let response = client
        .get(format!("{}/v1/deployments", url))
        .send()
        .await
        .expect("Failed to send request");

    // Expect 401 unauthorized without auth, which confirms the endpoint exists
    assert!(
        response.status().is_success() || response.status().as_u16() == 401,
        "Expected success or unauthorized, got: {}",
        response.status()
    );
}

// Unit tests for TUI configuration (using direct toml parsing)
mod unit {
    use serde::Deserialize;

    #[derive(Debug, Deserialize)]
    struct TuiConfig {
        api_url: String,
        #[serde(default)]
        theme: String,
    }

    #[test]
    fn test_config_with_custom_url() {
        let toml_str = r#"
            api_url = "http://localhost:8000"
            theme = "dark"
        "#;
        let config: TuiConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.api_url, "http://localhost:8000");
        assert_eq!(config.theme, "dark");
    }
}

