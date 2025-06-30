//! SSH Management Module
//!
//! This module provides SSH key management capabilities following SOLID principles:
//! - Single Responsibility: Each trait handles one aspect of SSH management
//! - Open/Closed: Easy to extend with new SSH providers
//! - Liskov Substitution: All implementations are interchangeable
//! - Interface Segregation: Specific traits for different SSH concerns
//! - Dependency Inversion: Abstractions over concrete implementations

pub mod config;
pub mod connection;
pub mod manager;
pub mod simple;
pub mod traits;
pub mod types;

pub use config::*;
pub use connection::*;
pub use manager::*;
pub use simple::*;
pub use traits::*;
pub use types::*;
