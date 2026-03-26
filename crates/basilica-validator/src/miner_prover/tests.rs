//! Integration tests for dynamic discovery

#[test]
fn test_ip_conversion_logic() {
    // Test IPv4 address conversion from u128
    let ipv4_u128: u128 = 0x00000000000000000000ffffc0a80101; // 192.168.1.1
    let ipv4_bytes = ipv4_u128.to_be_bytes();
    let ipv4_str = format!(
        "{}.{}.{}.{}",
        ipv4_bytes[12], ipv4_bytes[13], ipv4_bytes[14], ipv4_bytes[15]
    );
    assert_eq!(ipv4_str, "192.168.1.1");

    // Test IPv6 address conversion from u128
    let ipv6_u128: u128 = 0x20010db8000000000000000000000001; // 2001:db8::1
    let ipv6_bytes = ipv6_u128.to_be_bytes();
    let segments: Vec<u16> = (0..8)
        .map(|i| u16::from_be_bytes([ipv6_bytes[i * 2], ipv6_bytes[i * 2 + 1]]))
        .collect();
    let ipv6_str = segments
        .iter()
        .map(|&s| format!("{s:x}"))
        .collect::<Vec<_>>()
        .join(":");
    assert_eq!(ipv6_str, "2001:db8:0:0:0:0:0:1");
}

#[tokio::test]
async fn test_dynamic_discovery_config() {
    use crate::config::VerificationConfig;
    use std::time::Duration;

    let config = VerificationConfig {
        verification_interval: Duration::from_secs(3600),
        max_concurrent_verifications: 10,
        max_concurrent_full_validations: 1,
        challenge_timeout: Duration::from_secs(60),
        min_score_threshold: 0.0,
        max_miners_per_round: 10,
        min_verification_interval: Duration::from_secs(3600),
        use_dynamic_discovery: true,
        discovery_timeout: Duration::from_secs(30),
        fallback_to_static: true,
        cache_miner_info_ttl: Duration::from_secs(300),
        grpc_port_offset: Some(42000),
        binary_validation: None,
        docker_validation: crate::config::DockerValidationConfig::default(),
        collateral_event_scan_interval: Duration::from_secs(12),
        node_validation_interval: Duration::from_secs(12 * 3600),
        gpu_assignment_cleanup_ttl: Some(Duration::from_secs(30 * 60)),
        enable_worker_queue: false,
        storage_validation: crate::config::StorageValidationConfig::default(),
        node_groups: crate::config::NodeGroupConfig::default(),
    };

    // Verify configuration
    assert!(config.use_dynamic_discovery);
    assert_eq!(config.discovery_timeout, Duration::from_secs(30));
    assert!(config.fallback_to_static);
    assert_eq!(config.cache_miner_info_ttl, Duration::from_secs(300));
    assert_eq!(config.grpc_port_offset, Some(42000));
}
