//! Test utilities for executor

use crate::{ExecutorConfig, ExecutorState};
use std::sync::Arc;

/// Create a test executor state with default configuration
pub async fn create_test_executor_state() -> ExecutorState {
    let config = ExecutorConfig::default();
    ExecutorState::new(config)
        .await
        .expect("Failed to create test executor state")
}

/// Create a test executor state with custom configuration
pub async fn create_test_executor_state_with_config(config: ExecutorConfig) -> ExecutorState {
    ExecutorState::new(config)
        .await
        .expect("Failed to create test executor state with config")
}

/// Create a shared test executor state
pub async fn create_shared_test_executor_state() -> Arc<ExecutorState> {
    Arc::new(create_test_executor_state().await)
}
