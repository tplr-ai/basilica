pub use basilica_common::validator_api::{CpuSpec, GpuSpec, NetworkSpeedInfo, NodeDetails};

pub fn unknown_node_details(id: String) -> NodeDetails {
    NodeDetails {
        id,
        gpu_specs: Vec::new(),
        cpu_specs: CpuSpec {
            cores: 0,
            model: "Unknown".to_string(),
            memory_gb: 0,
        },
        location: None,
        network_speed: None,
        hourly_rate_cents: None,
    }
}
