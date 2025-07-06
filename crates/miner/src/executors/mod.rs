//! Executor Management Module
//!
//! This module handles executor connections, registration, and management.

pub mod connection_manager;
pub mod grpc_client;

pub use connection_manager::{ExecutorConnectionConfig, ExecutorConnectionManager, ExecutorInfo};
pub use grpc_client::{ExecutorGrpcClient, ExecutorGrpcConfig};
