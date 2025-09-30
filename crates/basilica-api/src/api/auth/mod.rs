//! Authentication module for the Basilica API
//!
//! This module provides JWT-based authentication functionality for validating
//! Auth0 tokens and managing user authentication state.

pub mod api_keys;
pub mod jwt_validator;

// Re-export commonly used types and functions
pub use jwt_validator::{
    fetch_jwks, validate_jwt_with_options, verify_audience, verify_issuer, Claims,
};
