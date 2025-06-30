//! Request distribution across validators

use crate::{discovery::ValidatorInfo, error::Result, load_balancer::LoadBalancer, Error};
use futures::future::join_all;
use reqwest::{Client, Request, Response};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, warn};

/// Request distributor for sending requests to multiple validators
pub struct RequestDistributor {
    /// HTTP client
    client: Client,

    /// Load balancer
    load_balancer: Arc<RwLock<LoadBalancer>>,
}

impl RequestDistributor {
    /// Create a new request distributor
    pub fn new(client: Client, load_balancer: Arc<RwLock<LoadBalancer>>) -> Self {
        Self {
            client,
            load_balancer,
        }
    }

    /// Send a request to a single validator
    pub async fn send_to_single(&self, request: Request) -> Result<Response> {
        let validator = self.load_balancer.read().await.select_validator().await?;

        debug!("Sending request to validator {}", validator.uid);

        match self.send_request_to_validator(request, &validator).await {
            Ok(response) => {
                self.load_balancer
                    .read()
                    .await
                    .report_success(validator.uid);
                Ok(response)
            }
            Err(e) => {
                self.load_balancer
                    .read()
                    .await
                    .report_failure(validator.uid);
                Err(e)
            }
        }
    }

    /// Send a request to multiple validators
    pub async fn send_to_multiple(&self, request: Request, count: usize) -> Result<Vec<Response>> {
        let mut validators = Vec::new();
        let lb = self.load_balancer.read().await;

        // Select multiple validators
        for _ in 0..count {
            match lb.select_validator().await {
                Ok(validator) => validators.push(validator),
                Err(e) => {
                    warn!("Failed to select validator: {}", e);
                    if validators.is_empty() {
                        return Err(e);
                    }
                    break;
                }
            }
        }

        drop(lb); // Release the read lock

        // Clone the request for each validator
        let mut tasks = Vec::new();
        for validator in validators {
            let req = self.clone_request(&request)?;
            let client = self.client.clone();
            let load_balancer = self.load_balancer.clone();

            tasks.push(tokio::spawn(async move {
                let result = Self::send_request_to_validator_static(&client, req, &validator).await;

                match &result {
                    Ok(_) => load_balancer.read().await.report_success(validator.uid),
                    Err(_) => load_balancer.read().await.report_failure(validator.uid),
                }

                result
            }));
        }

        // Wait for all requests to complete
        let results = join_all(tasks).await;

        let mut responses = Vec::new();
        for result in results {
            match result {
                Ok(Ok(response)) => responses.push(response),
                Ok(Err(e)) => error!("Request to validator failed: {}", e),
                Err(e) => error!("Task failed: {}", e),
            }
        }

        if responses.is_empty() {
            Err(Error::Aggregation {
                message: "All validator requests failed".to_string(),
            })
        } else {
            Ok(responses)
        }
    }

    /// Send request to a specific validator
    async fn send_request_to_validator(
        &self,
        request: Request,
        validator: &ValidatorInfo,
    ) -> Result<Response> {
        Self::send_request_to_validator_static(&self.client, request, validator).await
    }

    /// Static version for use in spawned tasks
    async fn send_request_to_validator_static(
        client: &Client,
        mut request: Request,
        validator: &ValidatorInfo,
    ) -> Result<Response> {
        // Update the request URL to point to the validator
        let original_path = request.url().path();
        let new_url = format!("{}{}", validator.endpoint, original_path);
        *request.url_mut() = new_url.parse().map_err(|e| Error::InvalidRequest {
            message: format!("Invalid URL: {e}"),
        })?;

        client
            .execute(request)
            .await
            .map_err(|e| Error::ValidatorCommunication {
                message: format!(
                    "Failed to communicate with validator {}: {}",
                    validator.uid, e
                ),
            })
    }

    /// Clone a request (since Request doesn't implement Clone)
    fn clone_request(&self, request: &Request) -> Result<Request> {
        self.client
            .request(request.method().clone(), request.url().as_str())
            .headers(request.headers().clone())
            .build()
            .map_err(|e| Error::Internal {
                message: format!("Failed to clone request: {e}"),
            })
    }
}
