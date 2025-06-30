//! Operating System attestation and benchmarking module

pub mod attestor;
pub mod benchmarker;
pub mod types;

pub use attestor::OsAttestor;
pub use benchmarker::OsBenchmarker;
pub use types::*;

/// Perform complete OS attestation
pub fn attest_system() -> anyhow::Result<OsAttestation> {
    OsAttestor::attest_system()
}

/// Run OS performance benchmarks
pub fn benchmark_system() -> anyhow::Result<OsPerformanceMetrics> {
    OsBenchmarker::run_benchmarks()
}

/// Quick OS security check
pub fn quick_security_check() -> anyhow::Result<SecurityFeatures> {
    OsAttestor::check_security_features()
}
