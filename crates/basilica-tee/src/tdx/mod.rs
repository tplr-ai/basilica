//! Intel TDX (Trust Domain Extensions) support
//!
//! This module provides functionality for:
//! - TDX quote parsing (v4/v5)
//! - Quote generation via CLI tools
//! - Quote signature verification
//! - Remote attestation service integration

pub mod provider;
pub mod quote;
pub mod remote_verification;
pub mod verification;

pub use provider::TdxQuoteProvider;
pub use quote::{QuoteHeader, TdReport, TdxQuoteV4};
pub use remote_verification::{RemoteAttestationConfig, RemoteTdxVerifier};
pub use verification::TdxQuoteVerifier;
