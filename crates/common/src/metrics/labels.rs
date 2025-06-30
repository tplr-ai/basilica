//! # Standardized Metric Labels
//!
//! Common label keys and values used across all Basilca metrics for consistency.

/// Component label key for identifying which component generated the metric
pub const COMPONENT_LABEL: &str = "component";

/// Operation label key for identifying the operation being measured
pub const OPERATION_LABEL: &str = "operation";

/// Service label key for identifying which service generated the metric
pub const SERVICE_LABEL: &str = "service";

/// Status label key for indicating success/failure status
pub const STATUS_LABEL: &str = "status";

/// Method label key for HTTP/gRPC method names
pub const METHOD_LABEL: &str = "method";

/// Endpoint label key for identifying API endpoints
pub const ENDPOINT_LABEL: &str = "endpoint";

/// Error type label key for categorizing errors
pub const ERROR_TYPE_LABEL: &str = "error_type";

/// Version label key for component versioning
pub const VERSION_LABEL: &str = "version";

/// Instance label key for identifying specific instances
pub const INSTANCE_LABEL: &str = "instance";

/// Environment label key (dev, staging, production)
pub const ENVIRONMENT_LABEL: &str = "environment";

/// Hotkey label key for Bittensor operations
pub const HOTKEY_LABEL: &str = "hotkey";

/// Netuid label key for Bittensor subnet identification
pub const NETUID_LABEL: &str = "netuid";

/// Executor ID label key
pub const EXECUTOR_ID_LABEL: &str = "executor_id";

/// Validator UID label key
pub const VALIDATOR_UID_LABEL: &str = "validator_uid";

/// Miner UID label key
pub const MINER_UID_LABEL: &str = "miner_uid";

/// Task type label key
pub const TASK_TYPE_LABEL: &str = "task_type";

/// GPU type label key
pub const GPU_TYPE_LABEL: &str = "gpu_type";

/// Common label values
pub mod values {
    // Component values
    pub const COMPONENT_VALIDATOR: &str = "validator";
    pub const COMPONENT_MINER: &str = "miner";
    pub const COMPONENT_EXECUTOR: &str = "executor";
    pub const COMPONENT_COMMON: &str = "common";

    // Status values
    pub const STATUS_SUCCESS: &str = "success";
    pub const STATUS_FAILURE: &str = "failure";
    pub const STATUS_PENDING: &str = "pending";
    pub const STATUS_TIMEOUT: &str = "timeout";

    // Operation values
    pub const OPERATION_CREATE: &str = "create";
    pub const OPERATION_READ: &str = "read";
    pub const OPERATION_UPDATE: &str = "update";
    pub const OPERATION_DELETE: &str = "delete";
    pub const OPERATION_QUERY: &str = "query";
    pub const OPERATION_VERIFICATION: &str = "verification";
    pub const OPERATION_CHALLENGE: &str = "challenge";
    pub const OPERATION_MINING: &str = "mining";
    pub const OPERATION_WEIGHT_SETTING: &str = "weight_setting";

    // Service values
    pub const SERVICE_GRPC: &str = "grpc";
    pub const SERVICE_HTTP: &str = "http";
    pub const SERVICE_DATABASE: &str = "database";
    pub const SERVICE_BITTENSOR: &str = "bittensor";

    // Error type values
    pub const ERROR_NETWORK: &str = "network";
    pub const ERROR_DATABASE: &str = "database";
    pub const ERROR_VALIDATION: &str = "validation";
    pub const ERROR_AUTHENTICATION: &str = "authentication";
    pub const ERROR_AUTHORIZATION: &str = "authorization";
    pub const ERROR_TIMEOUT: &str = "timeout";
    pub const ERROR_RATE_LIMIT: &str = "rate_limit";

    // Task type values
    pub const TASK_SYSTEM_PROFILE: &str = "system_profile";
    pub const TASK_COMPUTATIONAL_CHALLENGE: &str = "computational_challenge";
    pub const TASK_BENCHMARK: &str = "benchmark";
    pub const TASK_CONTAINER_MANAGEMENT: &str = "container_management";
}

/// Helper function to create common label sets
pub fn create_component_labels(component: &str) -> Vec<(&'static str, &str)> {
    vec![(COMPONENT_LABEL, component)]
}

/// Helper function to create operation labels
pub fn create_operation_labels<'a>(
    component: &'a str,
    operation: &'a str,
) -> Vec<(&'static str, &'a str)> {
    vec![(COMPONENT_LABEL, component), (OPERATION_LABEL, operation)]
}

/// Helper function to create service labels
pub fn create_service_labels<'a>(
    component: &'a str,
    service: &'a str,
    operation: &'a str,
) -> Vec<(&'static str, &'a str)> {
    vec![
        (COMPONENT_LABEL, component),
        (SERVICE_LABEL, service),
        (OPERATION_LABEL, operation),
    ]
}

/// Helper function to create status labels
pub fn create_status_labels<'a>(
    component: &'a str,
    operation: &'a str,
    status: &'a str,
) -> Vec<(&'static str, &'a str)> {
    vec![
        (COMPONENT_LABEL, component),
        (OPERATION_LABEL, operation),
        (STATUS_LABEL, status),
    ]
}

/// Helper function to create Bittensor-specific labels
pub fn create_bittensor_labels(
    component: &str,
    operation: &str,
    hotkey: &str,
    netuid: u16,
) -> Vec<(String, String)> {
    vec![
        (COMPONENT_LABEL.to_string(), component.to_string()),
        (OPERATION_LABEL.to_string(), operation.to_string()),
        (HOTKEY_LABEL.to_string(), hotkey.to_string()),
        (NETUID_LABEL.to_string(), netuid.to_string()),
    ]
}

/// Helper function to create executor-specific labels
pub fn create_executor_labels<'a>(
    executor_id: &'a str,
    operation: &'a str,
    status: &'a str,
) -> Vec<(&'static str, &'a str)> {
    vec![
        (COMPONENT_LABEL, values::COMPONENT_EXECUTOR),
        (EXECUTOR_ID_LABEL, executor_id),
        (OPERATION_LABEL, operation),
        (STATUS_LABEL, status),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_label_creation() {
        let component_labels = create_component_labels(values::COMPONENT_VALIDATOR);
        assert_eq!(component_labels.len(), 1);
        assert_eq!(
            component_labels[0],
            (COMPONENT_LABEL, values::COMPONENT_VALIDATOR)
        );

        let operation_labels =
            create_operation_labels(values::COMPONENT_VALIDATOR, values::OPERATION_VERIFICATION);
        assert_eq!(operation_labels.len(), 2);
        assert!(operation_labels.contains(&(COMPONENT_LABEL, values::COMPONENT_VALIDATOR)));
        assert!(operation_labels.contains(&(OPERATION_LABEL, values::OPERATION_VERIFICATION)));
    }

    #[test]
    fn test_bittensor_labels() {
        let labels = create_bittensor_labels(
            values::COMPONENT_VALIDATOR,
            values::OPERATION_VERIFICATION,
            "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            1,
        );
        assert_eq!(labels.len(), 4);
        assert!(labels.contains(&(
            COMPONENT_LABEL.to_string(),
            values::COMPONENT_VALIDATOR.to_string()
        )));
        assert!(labels.contains(&(NETUID_LABEL.to_string(), "1".to_string())));
    }
}
