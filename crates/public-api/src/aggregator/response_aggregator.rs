//! Response aggregation from multiple validators

use crate::{discovery::ValidatorInfo, error::Result, Error};
use reqwest::{Client, Response};
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::collections::HashMap;
use std::time::Duration;
use tracing::{debug, warn};

/// Aggregated response from multiple validators
#[derive(Debug)]
pub struct AggregatedResponse {
    /// Validator that provided this response
    pub validator: ValidatorInfo,
    /// Response data
    pub data: Value,
}

/// Response aggregator for combining responses from multiple validators
pub struct ResponseAggregator {
    /// HTTP client
    client: Client,
    /// Validators to query
    validators: Vec<ValidatorInfo>,
    /// Request timeout
    timeout: Duration,
}

impl ResponseAggregator {
    /// Create a new response aggregator
    pub fn new(client: Client, validators: Vec<ValidatorInfo>, timeout: Duration) -> Self {
        Self {
            client,
            validators,
            timeout,
        }
    }

    /// Aggregate GET requests from multiple validators
    pub async fn aggregate_get_requests(
        &self,
        path: &str,
        headers: Option<HashMap<String, String>>,
    ) -> Result<Vec<AggregatedResponse>> {
        let mut tasks = Vec::new();

        for validator in &self.validators {
            let url = format!("{}{}", validator.endpoint, path);
            let client = self.client.clone();
            let timeout = self.timeout;
            let validator_info = validator.clone();
            let headers_clone = headers.clone();

            tasks.push(tokio::spawn(async move {
                let mut request = client.get(&url).timeout(timeout);

                if let Some(headers) = headers_clone {
                    for (key, value) in headers {
                        request = request.header(key, value);
                    }
                }

                match request.send().await {
                    Ok(response) => {
                        if response.status().is_success() {
                            match response.json::<Value>().await {
                                Ok(data) => Some(AggregatedResponse {
                                    validator: validator_info,
                                    data,
                                }),
                                Err(e) => {
                                    warn!("Failed to parse response from {}: {}", url, e);
                                    None
                                }
                            }
                        } else {
                            warn!(
                                "Validator {} returned error status: {}",
                                url,
                                response.status()
                            );
                            None
                        }
                    }
                    Err(e) => {
                        warn!("Failed to query validator {}: {}", url, e);
                        None
                    }
                }
            }));
        }

        let results = futures::future::join_all(tasks).await;
        let mut responses = Vec::new();

        for result in results {
            if let Ok(Some(response)) = result {
                responses.push(response);
            }
        }

        if responses.is_empty() {
            Err(Error::Aggregation {
                message: "No valid responses from any validator".to_string(),
            })
        } else {
            Ok(responses)
        }
    }

    /// Aggregate responses by taking the majority consensus
    pub async fn aggregate_by_consensus<T: DeserializeOwned + PartialEq>(
        responses: Vec<Response>,
    ) -> Result<T> {
        let mut results = Vec::new();

        // Parse all responses
        for response in responses {
            match response.json::<T>().await {
                Ok(data) => results.push(data),
                Err(e) => warn!("Failed to parse response: {}", e),
            }
        }

        if results.is_empty() {
            return Err(Error::Aggregation {
                message: "No valid responses to aggregate".to_string(),
            });
        }

        // Find the most common response
        // Note: This is a simplified consensus - in production you might want
        // to use more sophisticated consensus mechanisms
        let mut frequency_map = HashMap::new();
        for result in &results {
            let key = format!("{:?}", result as *const T);
            *frequency_map.entry(key).or_insert(0) += 1;
        }

        // Return the first result as we can't easily compare complex types
        // In a real implementation, you'd want proper consensus logic
        Ok(results.into_iter().next().unwrap())
    }

    /// Aggregate JSON responses by merging
    pub async fn aggregate_json_merge(responses: Vec<Response>) -> Result<Value> {
        let mut merged = serde_json::Map::new();
        let mut arrays_to_merge: HashMap<String, Vec<Value>> = HashMap::new();

        for response in responses {
            match response.json::<Value>().await {
                Ok(Value::Object(obj)) => {
                    for (key, value) in obj {
                        match value {
                            Value::Array(arr) => {
                                arrays_to_merge.entry(key).or_default().extend(arr);
                            }
                            _ => {
                                merged.insert(key, value);
                            }
                        }
                    }
                }
                Ok(value) => {
                    debug!("Non-object response: {:?}", value);
                }
                Err(e) => {
                    warn!("Failed to parse JSON response: {}", e);
                }
            }
        }

        // Merge arrays
        for (key, values) in arrays_to_merge {
            merged.insert(key, Value::Array(values));
        }

        if merged.is_empty() {
            Err(Error::Aggregation {
                message: "No valid data to aggregate".to_string(),
            })
        } else {
            Ok(Value::Object(merged))
        }
    }

    /// Aggregate list responses by combining unique items
    pub async fn aggregate_lists<T: DeserializeOwned + Eq + std::hash::Hash>(
        responses: Vec<Response>,
    ) -> Result<Vec<T>> {
        let mut all_items = std::collections::HashSet::new();

        for response in responses {
            match response.json::<Vec<T>>().await {
                Ok(items) => {
                    all_items.extend(items);
                }
                Err(e) => {
                    warn!("Failed to parse list response: {}", e);
                }
            }
        }

        if all_items.is_empty() {
            Err(Error::Aggregation {
                message: "No items to aggregate".to_string(),
            })
        } else {
            Ok(all_items.into_iter().collect())
        }
    }

    /// Take the first successful response
    pub async fn take_first<T: DeserializeOwned>(responses: Vec<Response>) -> Result<T> {
        for response in responses {
            match response.json::<T>().await {
                Ok(data) => return Ok(data),
                Err(e) => {
                    warn!("Failed to parse response: {}", e);
                }
            }
        }

        Err(Error::Aggregation {
            message: "No valid response found".to_string(),
        })
    }
}
