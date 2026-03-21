//! HTTP client for interacting with the Validator API
//!
//! This module provides a client implementation for external services
//! to interact with the Validator's REST API endpoints.

use crate::api::types::*;
use anyhow::{Context, Result};
use eventsource_stream::Eventsource;
use futures::StreamExt;
use futures_util::Stream;
use reqwest::{
    header::{HeaderMap, HeaderName, HeaderValue},
    Client,
};
use std::{pin::Pin, time::Duration};

/// HTTP client for the Validator API
#[derive(Clone, Debug)]
pub struct ValidatorClient {
    base_url: String,
    http_client: Client,
}

impl ValidatorClient {
    /// Create a new ValidatorClient instance
    pub fn new(base_url: impl Into<String>, timeout: Duration) -> Result<Self> {
        Self::new_with_api_key(base_url, timeout, None)
    }

    /// Create a new ValidatorClient instance with an optional API key.
    pub fn new_with_api_key(
        base_url: impl Into<String>,
        timeout: Duration,
        api_key: Option<String>,
    ) -> Result<Self> {
        let http_client = Client::builder()
            .timeout(timeout)
            .default_headers(default_headers(api_key.as_deref())?)
            .build()
            .context("Failed to build HTTP client")?;

        Ok(Self {
            base_url: base_url.into(),
            http_client,
        })
    }

    /// Create a new ValidatorClient with a custom HTTP client
    pub fn with_client(base_url: impl Into<String>, http_client: Client) -> Self {
        Self {
            base_url: base_url.into(),
            http_client,
        }
    }

    /// List rentals with optional state filter
    pub async fn list_rentals(
        &self,
        filter: Option<ApiRentalState>,
    ) -> Result<ListRentalsResponse> {
        let url = format!("{}/rentals", self.base_url);

        let mut req = self.http_client.get(&url);
        if let Some(state_filter) = filter {
            // Serialize the enum value as lowercase string for the query parameter
            let state_str = state_filter.to_string();
            req = req.query(&[("state", state_str)]);
        }

        let response = req.send().await.context("Failed to send list request")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_body = response.text().await.unwrap_or_default();
            anyhow::bail!("Failed to list rentals: {} - {}", status, error_body);
        }

        let json = response
            .json()
            .await
            .context("Failed to parse list response")?;

        Ok(json)
    }

    /// Start a new rental
    pub async fn start_rental(&self, request: StartRentalRequest) -> Result<ApiRentalResponse> {
        let url = format!("{}/rentals", self.base_url);

        let response = self
            .http_client
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("Failed to send rental request")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_body = response.text().await.unwrap_or_default();
            anyhow::bail!("Failed to start rental: {} - {}", status, error_body);
        }

        response
            .json()
            .await
            .context("Failed to parse rental response")
    }

    /// Get rental status
    pub async fn get_rental_status(&self, rental_id: &str) -> Result<RentalStatusResponse> {
        let url = format!("{}/rentals/{}", self.base_url, rental_id);

        let response = self
            .http_client
            .get(&url)
            .send()
            .await
            .context("Failed to send status request")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_body = response.text().await.unwrap_or_default();
            anyhow::bail!("Failed to get rental status: {} - {}", status, error_body);
        }

        response
            .json()
            .await
            .context("Failed to parse status response")
    }

    /// Terminate a rental
    pub async fn terminate_rental(
        &self,
        rental_id: &str,
        _request: TerminateRentalRequest, // Maintained for API compatibility
    ) -> Result<()> {
        let url = format!("{}/rentals/{}", self.base_url, rental_id);

        let response = self
            .http_client
            .delete(&url)
            .send()
            .await
            .context("Failed to send termination request")?;

        if response.status() == reqwest::StatusCode::NO_CONTENT {
            Ok(())
        } else {
            let status = response.status();
            let error_body = response.text().await.unwrap_or_default();
            anyhow::bail!("Failed to terminate rental: {} - {}", status, error_body)
        }
    }

    /// Restart a rental's container
    pub async fn restart_rental(&self, rental_id: &str) -> Result<ApiRentalRestartResponse> {
        let url = format!("{}/rentals/{}/restart", self.base_url, rental_id);

        let response = self
            .http_client
            .post(&url)
            .send()
            .await
            .context("Failed to send restart request")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_body = response.text().await.unwrap_or_default();
            anyhow::bail!("Failed to restart rental: {} - {}", status, error_body);
        }

        response
            .json()
            .await
            .context("Failed to parse restart response")
    }

    /// Stream rental logs
    pub async fn stream_rental_logs(
        &self,
        rental_id: &str,
        query: LogQuery,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Event>> + Send>>> {
        let url = format!("{}/rentals/{}/logs", self.base_url, rental_id);

        let response = self
            .http_client
            .get(&url)
            .query(&query)
            .send()
            .await
            .context("Failed to send log request")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_body = response.text().await.unwrap_or_default();
            anyhow::bail!("Failed to stream logs: {} - {}", status, error_body);
        }

        // Use eventsource-stream to parse SSE
        let stream = response
            .bytes_stream()
            .eventsource()
            .filter_map(|result| async move {
                match result {
                    Ok(sse_event) => {
                        // Parse the data field as JSON
                        match serde_json::from_str::<Event>(&sse_event.data) {
                            Ok(event) => Some(Ok(event)),
                            Err(e) => {
                                tracing::error!(
                                    "Failed to parse log event: {}, data: {}",
                                    e,
                                    sse_event.data
                                );
                                None
                            }
                        }
                    }
                    Err(e) => Some(Err(anyhow::anyhow!("SSE stream error: {}", e))),
                }
            });

        Ok(Box::pin(stream))
    }

    /// List available nodes for rental
    pub async fn list_available_nodes(
        &self,
        query: Option<ListAvailableNodesQuery>,
    ) -> Result<ListAvailableNodesResponse> {
        let url = format!("{}/nodes", self.base_url);

        let mut req = self.http_client.get(&url);

        if let Some(query_params) = query {
            req = req.query(&query_params);
        }

        let response = req
            .send()
            .await
            .context("Failed to send available nodes request")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_body = response.text().await.unwrap_or_default();
            anyhow::bail!(
                "Failed to list available nodes: {} - {}",
                status,
                error_body
            );
        }

        let json = response
            .json()
            .await
            .context("Failed to parse available nodes response")?;

        Ok(json)
    }
}

fn default_headers(api_key: Option<&str>) -> Result<HeaderMap> {
    let mut headers = HeaderMap::new();

    if let Some(api_key) = api_key {
        headers.insert(
            HeaderName::from_static("x-api-key"),
            HeaderValue::from_str(api_key).context("Invalid validator API key header value")?,
        );
    }

    Ok(headers)
}

/// Event type for log streaming
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Event {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub stream: String,
    pub message: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::{
        matchers::{header, method, path},
        Mock, MockServer, ResponseTemplate,
    };

    #[test]
    fn test_client_creation() {
        let client = ValidatorClient::new("http://localhost:8080", Duration::from_secs(30));
        assert!(client.is_ok());
    }

    #[test]
    fn test_client_with_custom_client() {
        let http_client = Client::new();
        let client = ValidatorClient::with_client("http://localhost:8080", http_client);
        assert_eq!(client.base_url, "http://localhost:8080");
    }

    #[tokio::test]
    async fn test_client_with_api_key_sends_header() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/nodes"))
            .and(header("x-api-key", "secret-key"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"available_nodes": [], "total_count": 0})),
            )
            .mount(&mock_server)
            .await;

        let client = ValidatorClient::new_with_api_key(
            mock_server.uri(),
            Duration::from_secs(30),
            Some("secret-key".to_string()),
        )
        .unwrap();

        let response = client.list_available_nodes(None).await.unwrap();
        assert_eq!(response.total_count, 0);
    }
}
