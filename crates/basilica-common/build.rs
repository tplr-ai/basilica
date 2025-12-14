//! Build script for basilica-common
//!
//! This script generates compile-time constants that can be overridden
//! by environment variables during the build process.
//! Supports reading from .env files in the project root.

use std::env;
use std::fs;
use std::path::Path;

fn main() {
    // Tell Cargo to rerun this build script if .env files change (only if file exists)
    if Path::new(".env").exists() {
        println!("cargo:rerun-if-changed=.env");
    }

    // Tell Cargo to rerun this build script if these environment variables change
    println!("cargo:rerun-if-env-changed=BASILICA_AUTH0_DOMAIN");
    println!("cargo:rerun-if-env-changed=BASILICA_AUTH0_CLIENT_ID");
    println!("cargo:rerun-if-env-changed=BASILICA_AUTH0_AUDIENCE");
    println!("cargo:rerun-if-env-changed=BASILICA_AUTH0_ISSUER");

    // Try to load .env file from project root (ignore if it doesn't exist)
    let _ = dotenvy::dotenv();

    // Default values (same as current hardcoded values)
    let defaults = [
        ("AUTH0_DOMAIN", "dev-ndynjuhl74mrh162.us.auth0.com"),
        ("AUTH0_CLIENT_ID", "CVwgCKL9MT5txAGRLCUQu89rvKXAwOVB"),
        ("AUTH0_AUDIENCE", "https://api.basilica.ai/"),
        ("AUTH0_ISSUER", "https://dev-ndynjuhl74mrh162.us.auth0.com/"),
    ];

    let out_dir = env::var_os("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("build_constants.rs");

    let mut content = String::new();
    content.push_str("// Generated compile-time constants\n\n");

    for (name, default_value) in &defaults {
        let env_var_name = format!("BASILICA_{}", name);
        let value = env::var(&env_var_name).unwrap_or_else(|_| default_value.to_string());

        content.push_str(&format!("pub const {}: &str = \"{}\";\n", name, value));
    }

    fs::write(&dest_path, content).unwrap();
}
