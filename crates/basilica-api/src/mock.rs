//! Mock implementations for dev mode
//!
//! This module provides constants and helpers for running the API
//! without requiring Bittensor/Validator connections. Useful for:
//! - Local development
//! - TUI testing
//! - Integration tests
//!
//! Note: The API routes still work normally - they just return appropriate
//! responses based on database state. The mock is primarily for bypassing
//! the Bittensor metagraph discovery at startup.

/// Mock validator hotkey for dev mode
pub const DEV_VALIDATOR_HOTKEY: &str = "5DevMockValidatorHotkey000000000000000000000000000";

/// Mock validator UID for dev mode
pub const DEV_VALIDATOR_UID: u16 = 999;

/// Mock validator endpoint for dev mode
pub const DEV_VALIDATOR_ENDPOINT: &str = "http://localhost:8080";
