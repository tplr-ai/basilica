//! Executor Management Module
//!
//! This module handles executor connections, registration, and management.

pub mod connection_manager;

pub use connection_manager::{
    ExecutorConnection, ExecutorConnectionConfig, ExecutorConnectionManager, ExecutorInfo,
};
