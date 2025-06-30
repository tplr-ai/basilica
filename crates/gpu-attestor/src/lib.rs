pub mod attestation;
pub mod cli;
pub mod docker;
pub mod gpu;
pub mod hardware;
pub mod integrity;
pub mod network;
pub mod os;
pub mod validation;
pub mod vdf;

// Re-export commonly used items for convenience
pub use attestation::{
    AttestationBuilder, AttestationSigner, AttestationVerifier, SignedAttestation,
};
pub use docker::{
    DockerAttestation, DockerBenchmarkResults, DockerBenchmarker, DockerCapabilities,
    DockerCollector, DockerInfo, DockerSecurityFeatures,
};
pub use gpu::{detect_primary_vendor, query_all_gpus, GpuDetector, GpuInfo, GpuSummary, GpuVendor};
pub use hardware::{
    check_system_requirements, collect_system_info, get_system_summary, BenchmarkResults,
    BenchmarkRunner, CpuInfo, MemoryInfo, MotherboardInfo, NetworkInfo, StorageInfo, SystemInfo,
    SystemInfoCollector,
};
pub use integrity::extract_embedded_key;
pub use network::{NetworkBenchmarkResults, NetworkBenchmarker};
pub use os::{
    attest_system, benchmark_system, OsAttestation, OsAttestor, OsBenchmarker,
    OsPerformanceMetrics, SecurityFeatures,
};
pub use vdf::{VdfAlgorithm, VdfComputer, VdfProof};
