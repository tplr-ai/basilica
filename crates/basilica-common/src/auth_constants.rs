//! Auth0 configuration constants for Basilica authentication
//!
//! These constants can be overridden at compile-time via environment variables
//! or at runtime via environment variables. The priority order is:
//! 1. Runtime environment variable (highest)
//! 2. Compile-time environment variable
//! 3. Default hardcoded value (lowest)

use once_cell::sync::Lazy;
use std::env;

// Include the generated compile-time constants
include!(concat!(env!("OUT_DIR"), "/build_constants.rs"));

/// Get Auth0 domain, checking runtime env var first, then falling back to compile-time constant
pub fn auth0_domain() -> &'static str {
    static RUNTIME_VALUE: Lazy<Option<String>> =
        Lazy::new(|| env::var("BASILICA_AUTH0_DOMAIN").ok());

    RUNTIME_VALUE.as_deref().unwrap_or(AUTH0_DOMAIN)
}

/// Get Auth0 client ID, checking runtime env var first, then falling back to compile-time constant
pub fn auth0_client_id() -> &'static str {
    static RUNTIME_VALUE: Lazy<Option<String>> =
        Lazy::new(|| env::var("BASILICA_AUTH0_CLIENT_ID").ok());

    RUNTIME_VALUE.as_deref().unwrap_or(AUTH0_CLIENT_ID)
}

/// Get Auth0 audience, checking runtime env var first, then falling back to compile-time constant
pub fn auth0_audience() -> &'static str {
    static RUNTIME_VALUE: Lazy<Option<String>> =
        Lazy::new(|| env::var("BASILICA_AUTH0_AUDIENCE").ok());

    RUNTIME_VALUE.as_deref().unwrap_or(AUTH0_AUDIENCE)
}

/// Get Auth0 issuer URL, checking runtime env var first, then falling back to compile-time constant
pub fn auth0_issuer() -> &'static str {
    static RUNTIME_VALUE: Lazy<Option<String>> =
        Lazy::new(|| env::var("BASILICA_AUTH0_ISSUER").ok());

    RUNTIME_VALUE.as_deref().unwrap_or(AUTH0_ISSUER)
}

/// Default callback ports for OAuth flow
/// These are non-standard ports that should be registered in Auth0's Allowed Callback URLs
/// Using less common ports to avoid conflicts with typical development services
pub const AUTH_CALLBACK_PORTS: &[u16] = &[34521, 45632, 23457, 51234, 38901];

/// Get Auth0 callback ports
pub fn auth0_callback_ports() -> &'static [u16] {
    AUTH_CALLBACK_PORTS
}

const DEV_AUTH0_DOMAIN: &str = "dev-ndynjuhl74mrh162.us.auth0.com";

/// Check if running in development environment
///
/// Returns `true` when the current Auth0 domain matches the compile-time default domain,
/// indicating a development environment. Returns `false` when a different domain is
/// configured via the `BASILICA_AUTH0_DOMAIN` environment variable.
///
/// # Examples
///
/// ```
/// use basilica_common::is_development_environment;
///
/// if is_development_environment() {
///     println!("Running in development mode");
/// } else {
///     println!("Running with custom Auth0 domain");
/// }
/// ```
pub fn is_development_environment() -> bool {
    auth0_domain() == DEV_AUTH0_DOMAIN
}
