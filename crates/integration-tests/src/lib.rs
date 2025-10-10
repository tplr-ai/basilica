//! Integration tests library with shared utilities
//!
//! This crate provides shared test utilities and helpers for integration testing
//! across the Basilica system components.

// pub mod auth_test_utils;  // Commented out - needs refactoring for node-based architecture
pub mod config;

// Re-export commonly used types for convenience
// pub use auth_test_utils::{
//     create_authenticated_request, create_expired_authenticated_request, create_miner_auth_service,
//     create_miner_auth_service_with_config, create_test_auth, create_test_auth_with_bad_signature,
//     create_valid_auth, test_hotkeys, MockBittensorService, MockNodeAuthService,
// };

pub use config::{ServiceAvailability, TestConfig};
